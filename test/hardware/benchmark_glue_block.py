# SPDX-License-Identifier: Apache-2.0
"""In-graph device A/B of the fused glue kernels: per kernel, per phase, per bucket.

Run it with ``NEURON_RT_VISIBLE_CORES`` (and the LNC) set by whatever owns the device;
this script does not select cores, and refuses to start without them.

Why not ``benchmark_layer_glue.py``. That benchmark compiles each layer with no
process group, so both row-parallel reductions were the identity and the graph held no
collective. neuronx-cc schedules a graph with an HLO all-reduce differently (the
moe-prefill block ran 5.47 ms without one and 12.30 ms with a one-rank one), and the
served graph has two per layer, 21-77 us each at TP=64. Without them, 0a08ff4's
weight-only glue (fp32 casts and transposes of the projection weights, which depend on
no activation) is fully exposed; in the served graph it can run under the reductions'
waits. ``benchmark_layer_glue.py`` predicted -59 us (KDA + MoE layer) and -78 us
(DSA + MoE) at B=1; a served TP=64 decode profile showed +52 us and -13 us.

What runs. One graph per (case, variant), all from this tree:

* A case is ``family:phase:rows``. ``kda`` is checkpoint layer 4 (linear attention + MoE),
  ``dsa`` checkpoint layer 3 (sparse attention at ctx 1024 + MoE); ``glue_case`` builds
  them at one rank's TP=64 / EP=16 shapes. ``decode:B`` runs ``glue_case.layer_step``
  (the model's own layer forward: the attention site and the FFN site) on B requests'
  decode carriers; ``prefill:T`` runs it on one opening request's T-row chunk
  (``is_prefill=True``; ``kda`` only). ``--stack N`` chains N such layers in one graph.
* Collectives (``--collectives``): ``one-rank`` (default) makes both reductions of every
  layer (the attention output's and the FFN's) HLO all-reduces over a one-rank gloo
  world: ``model_fp8._resolve_tp_group`` answers a stand-in group whose ``all_reduce`` is
  the functional collective (``benchmark_moe_prefill_block.py``'s one-rank mode).
  ``identity`` is ``benchmark_layer_glue.py``'s setting (no collective in the graph).
* A variant is a ``VLLM_NEURON_GLUE_FUSED`` value: ``off`` (``0``, 0a08ff4's routes),
  one kernel alone (its name: every row count and phase), ``all``, ``default`` (``1``),
  or ``name=spec``; ``aa=0`` is a second copy of ``off`` (the A/A control). Each
  variant's graph is traced from its own code object with the switch set, so no graph
  traced under one value serves another; the trace-time dispatch counts of the three
  counted kernels are recorded per variant.

What is measured, per case:

1. Device ms per execution from the runtime system trace (``nc_exec_running``, the two
   physical cores merged). ``--rounds`` rounds; each round interleaves every variant for
   ``--iterations`` calls (after ``--warmup`` calls in round 0). Per variant: the
   per-round medians, their median and spread (max - min). Per variant against ``off``:
   the paired per-round delta. Verdict ``win`` when every round's delta is < 0 and
   |median delta| exceeds both variants' spreads and the A/A floor (|median delta| of
   ``aa``: an identical graph, loaded again, has measured up to 31 us apart);
   ``loss`` likewise with > 0; else ``neutral``.
2. A device profile of ``--profile-iterations`` calls per variant; ``--analyze-only``
   ingests each with ``neuron-explorer`` and buckets it with the serving profile's frozen
   rules (``benchmark_layer_glue.py --bucket``): execution ms from the profile's
   ``ExecutionInfo`` and the per-bucket ms.
3. Each variant's first-call output against ``off``'s on the same operands: rel_l2,
   max|d| (also in bf16 steps at max|ref|), the fraction of elements that differ, and
   ``agreement_ok``: rel_l2 at most :data:`DEVICE_REL_L2_TOL` per layer. With
   ``--save-io`` the operands, weights and outputs are saved, and ``--sim-check`` (CPU,
   no device) compares every variant's device output to the simulator's ``0`` on the
   same operands: at ``test_layer_glue.py``'s layer tolerances
   (``within_layer_tolerance``), and against the device ``off`` output's own distance
   from it (``no_further_than_off``).
4. Each variant's graph: the compile-cache key of every graph its first call compiled or
   loaded (the backend's ``create_cache_hash``: the FX graph, shapes and strides, not the
   kernel bodies) and whether that key was already in the cache root. Variants whose keys
   are equal ran the same graph (``same_graph_as`` in the summary): a variant that selects
   no kernel a case's rows reach is ``off``'s graph, by key.

The compile cache is used (``NEURON_LIBTORCH_CACHE_ROOT``): its key ignores kernel
bodies, so give it a fresh root after a kernel edit.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
import types


def ccop_main(name: str, out_path: str, global_dir: str) -> None:
    """Per execution: its ms and its collectives' spans (runs under the duckdb python)."""
    import glob

    import duckdb

    path = sorted(glob.glob(f"{global_dir}/{name}_*_session_*@latest"))[0]
    c = duckdb.connect()
    execs = c.sql(f"SELECT execution_index, execution_start_ts, execution_end_ts FROM "
                  f"'{path}/ExecutionInfo.parquet' ORDER BY 2").fetchall()
    have_cc = Path(f"{path}/CcOp.parquet").exists()
    rows = []
    for idx, s0, e0 in execs:
        ops = c.sql(f"SELECT start_ts, end_ts, operation FROM '{path}/CcOp.parquet' "
                    f"WHERE start_ts >= {s0} AND end_ts <= {e0} ORDER BY 1").fetchall() \
            if have_cc else []
        rows.append({"execution": idx, "execution_ms": (e0 - s0) / 1e6,
                     "collectives_us": [[(a - s0) / 1e3, (b - s0) / 1e3, op]
                                        for a, b, op in ops]})
    Path(out_path).write_text(json.dumps({"trace_dir": path, "executions": rows}, indent=1))


if __name__ == "__main__" and sys.argv[1:2] == ["--ccops"]:
    ccop_main(*sys.argv[2:5])
    sys.exit(0)

#: The worktree root, ahead of any installed copy (the lease command sets no PYTHONPATH).
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import torch  # noqa: E402

import vllm_neuron  # noqa: E402,F401 -- registers the Neuron compilation backend
from test.hardware import benchmark_layer_glue as micro  # noqa: E402
from test.vllm_neuron.functional.glue import glue_case  # noqa: E402
from test.vllm_neuron.functional.glue import test_layer_glue as layer_glue  # noqa: E402
from vllm_neuron import envs  # noqa: E402
from vllm_neuron.functional import glue  # noqa: E402
from vllm_neuron.functional.glue import kda_output, kda_projections, mhc_pre  # noqa: E402
from vllm_neuron.model.glm5_next import model_fp8 as live  # noqa: E402

DEVICE = "neuron:0"
CASES = ("kda:decode:1", "dsa:decode:1", "kda:decode:64", "dsa:decode:64",
         "kda:prefill:1024")
COLLECTIVES = ("one-rank", "identity")
#: The kernels whose sites a family's layer holds.
FAMILY_KERNELS = {"kda": glue.KERNELS, "dsa": ("mhc_pre", "mhc_post")}
#: The three kernels with dispatch counters (``mhc_post`` is a dtype choice, uncounted).
COUNTED = {"mhc_pre": mhc_pre, "kda_projections": kda_projections,
           "kda_output": kda_output}
#: A variant's device output agrees with ``off``'s device output when their rel_l2 is at
#: most this, per layer: the largest fused-vs-torch-route device rel_l2 that
#: ``benchmark_layer_glue.py`` measured for these kernels (3.622e-3 on a KDA layer to
#: 5.6225e-3 on a DSA layer), rounded up to three significant digits so that the
#: measurement itself agrees. The device pair differs by one bf16
#: rounding step in about half the elements (the device's XLA convert and the kernels'
#: on-chip rounding differ), which a layer then carries, so the simulator's layer
#: tolerances (``test_layer_glue.py``) do not hold for it. They do not hold for the
#: device's own ``0`` against the simulator's ``0`` either (the device's XLA ops and the
#: simulator do not round alike), so ``--sim-check`` holds each device output to the
#: device ``0``'s distance from the simulator's ``0`` as well.
DEVICE_REL_L2_TOL = 5.63e-3
#: ``test_layer_glue.py``'s layer tolerances: rel_l2, max|d| in bf16 steps at max|ref|,
#: fraction of elements changed.
SIM_REL_L2 = layer_glue.LAYER_REL_L2
SIM_MAX_STEPS = layer_glue.LAYER_MAX_STEPS
SIM_MAX_FLIPS = layer_glue.LAYER_MAX_FLIPS

#: ``(cache key, already in the cache root)`` for every graph the backend hashed, in order.
_GRAPH_KEYS: list = []


def _record_graph_keys() -> None:
    """Wrap the backend's cache-key function so each compile records its key."""
    from libtorch_neuronx_lite import envs as libtorch_envs
    from libtorch_neuronx_lite.compile import cache

    if getattr(cache.create_cache_hash, "_glue_block_bench", False):
        return
    original = cache.create_cache_hash

    def recording(*a, **k):
        key = original(*a, **k)
        root = libtorch_envs.get_neuron_compile_cache_dir()
        _GRAPH_KEYS.append((key, os.path.isdir(os.path.join(root, key))))
        return key

    recording._glue_block_bench = True
    cache.create_cache_hash = recording


def cache_entries() -> int:
    """Compiled graphs in the cache root (``NEURON_LIBTORCH_CACHE_ROOT``)."""
    from libtorch_neuronx_lite import envs as libtorch_envs

    root = Path(libtorch_envs.get_neuron_compile_cache_dir())
    return sum(1 for entry in root.iterdir() if entry.is_dir()) if root.is_dir() else 0


def parse_case(text: str) -> dict:
    family, phase, rows = text.split(":")
    if family not in FAMILY_KERNELS or phase not in glue.PHASES or int(rows) < 1:
        raise ValueError(f"case {text!r}: want kda|dsa : prefill|decode : rows")
    if phase == "prefill" and family != "kda":
        raise ValueError(f"case {text!r}: prefill is built for kda layers only")
    return {"family": family, "phase": phase, "rows": int(rows), "tag": text}


#: Kernel options a variant may set after its switch value (``name=spec;KEY=VALUE``);
#: a variant that does not set one runs with the option unset (its default).
VARIANT_OPTIONS = ("VLLM_NEURON_GLUE_KDA_DMA_TRANSPOSE",)


def split_spec(spec: str) -> tuple[str, dict]:
    """``(switch value, {option: value})`` from ``spec[;KEY=VALUE...]``."""
    head, *rest = spec.split(";")
    options = dict(part.split("=", 1) for part in rest)
    unknown = set(options) - set(VARIANT_OPTIONS)
    if unknown:
        raise ValueError(f"variant {spec!r}: unknown option(s) {sorted(unknown)}")
    return head, options


def variant_specs(case: dict, names: list[str] | None) -> dict:
    """``{variant: VLLM_NEURON_GLUE_FUSED value[;option=value...]}`` for ``case``."""
    if not names:
        names = ["off", "aa=0", *FAMILY_KERNELS[case["family"]], "all", "default"]
    out = {}
    for name in names:
        if "=" in name:
            name, spec = name.split("=", 1)
        else:
            spec = {"off": "0", "default": "1", "all": "all"}.get(name, name)
        glue.glue_selection(split_spec(spec)[0])  # refuse a bad spec before compiling
        out[name] = spec
    if "off" not in out:
        out = {"off": "0", **out}
    return out


#: ``(attention site, FFN site)`` chain lengths for the next trace (``--ar-chain``).
_AR_CHAIN = [1, 1]
#: Reductions issued in the current trace; a layer issues the attention's, then the FFN's.
_AR_CALLS = [0]


class _OneRankGroup:
    """``_resolve_tp_group``'s answer in ``one-rank`` mode: the served group's in-place
    ``all_reduce``, as the functional collective over the one-rank world.

    A one-rank all-reduce completes in a few microseconds; the served TP=64 ones take
    21-77 us at B=1 (rank 0's median, peer waits included) and 54-217 us at B=64, and
    work that depends on no reduced value can run under them. ``--ar-chain A F`` issues
    ``A`` (attention site) or ``F`` (FFN site) dependent one-rank all-reduces in a row,
    each the identity on the values, so the wait lasts about as long as the served one
    while the compute engines stay free.
    """

    def all_reduce(self, tensor):
        import torch.distributed as dist
        from torch.distributed._functional_collectives import all_reduce

        chain = _AR_CHAIN[_AR_CALLS[0] % 2]
        _AR_CALLS[0] += 1
        out = all_reduce(tensor, "sum", dist.group.WORLD)
        for _ in range(chain - 1):
            out = all_reduce(out, "sum", dist.group.WORLD)
        tensor.copy_(out)


def init_one_rank_world() -> None:
    """A one-rank gloo world, and a separate port for the runtime's communicator."""
    import socket

    import torch.distributed as dist

    if dist.is_initialized():
        return
    ports = []
    for _ in range(2):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            ports.append(sock.getsockname()[1])
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(ports[0]),
                      NEURON_RT_ROOT_COMM_ID=f"127.0.0.1:{ports[1]}")
    dist.init_process_group("gloo", rank=0, world_size=1)


def build_layers(case: dict, stack: int, device: str = DEVICE) -> list:
    torch.manual_seed(0)
    build = glue_case.kda_layer if case["family"] == "kda" else glue_case.dsa_layer
    base = glue_case.KDA_LAYER if case["family"] == "kda" else glue_case.DSA_LAYER
    return [build(live, seed=base + 10 * i, device=device) for i in range(stack)]


def carriers_for(case: dict, layer_case, device: str = DEVICE) -> dict:
    if case["phase"] == "prefill":
        return glue_case.kda_prefill_carriers(layer_case, case["rows"], device=device)
    if case["family"] == "kda":
        return glue_case.kda_carriers(layer_case, case["rows"], device=device)
    return glue_case.dsa_carriers(layer_case, case["rows"], device=device)


def _fresh(fn, tag: str):
    """``fn`` with its own code object, so dynamo keeps a separate cache for it."""
    code = fn.__code__.replace(co_name=f"{fn.__name__}_{tag.replace('-', '_')}")
    return types.FunctionType(code, fn.__globals__, code.co_name, fn.__defaults__,
                              fn.__closure__)


def build_variant(case: dict, layers: list, spec: str, tag: str, args):
    """``(call, inputs, dispatch)``: the compiled stack under ``spec``, traced now."""
    quant = glue_case.quant_config(live)
    per_layer = [micro._flatten(carriers_for(case, lc)) for lc in layers]
    counts = [len(tensors) for _, tensors, _ in per_layer]
    rank = torch.tensor(0, dtype=torch.int64, device=DEVICE)

    def step(streams, rank, *flat):
        at = 0
        for layer_case, (names, _, statics), n in zip(layers, per_layer, counts):
            carriers = micro._unflatten(names, flat[at:at + n], statics)
            at += n
            streams = glue_case.layer_step(live, layer_case, streams, carriers, rank,
                                           quant)
        return streams

    graph = torch.compile(_fresh(step, tag), backend="neuron_libtorch", fullgraph=True,
                          dynamic=False, options={"compiler_args": args.compiler_args})

    switch, options = split_spec(spec)

    def call(*inputs):
        os.environ[glue.GLUE_FUSED_ENV] = switch
        for key in VARIANT_OPTIONS:
            if key in options:
                os.environ[key] = options[key]
            else:
                os.environ.pop(key, None)
        # The trace reads the counter (dynamo guards on its value) and the compiled
        # graph replays the increments, so every call starts it where the trace did.
        _AR_CALLS[0] = 0
        return graph(*inputs)

    streams = glue_case.streams_input(layers[0].cfg, case["rows"], device=DEVICE)
    inputs = (streams, rank, *(t for _, tensors, _ in per_layer for t in tensors))
    for module in COUNTED.values():
        module.reset_dispatch_counters()
    _AR_CALLS[0] = 0
    keys_from = len(_GRAPH_KEYS)
    pristine = [t.to("cpu").clone() for t in inputs]  # the calls write the state banks
    t0 = time.time()
    first = call(*inputs).to("cpu")
    compile_s = time.time() - t0
    dispatch = {name: list(module.dispatch_counters()) for name, module in COUNTED.items()}
    dispatch["graphs"] = [{"key": key, "cached_before": hit}
                          for key, hit in _GRAPH_KEYS[keys_from:]]
    dispatch["mhc_post_bf16"] = glue.glue_selection(switch).selects("mhc_post",
                                                                    case["rows"])
    return call, inputs, first, dispatch, compile_s, pristine


def _bf16_step(peak: float) -> float:
    """One bf16 step (unit in the last place) at magnitude ``peak``."""
    return 2.0 ** (math.floor(math.log2(max(peak, 2.0 ** -126))) - 7)


def agreement(got: torch.Tensor, want: torch.Tensor, layers: int = 1) -> dict:
    """``got`` (a device output) against ``want`` (``off``'s device output)."""
    a, b = got.float(), want.float()
    rel = float((a - b).norm() / b.norm().clamp_min(1e-30))
    peak = float(b.abs().max())
    return {"rel_l2": rel,
            "max_abs_diff": float((a - b).abs().max()),
            "max_abs_off": peak,
            "max_steps": float((a - b).abs().max()) / _bf16_step(peak),
            "elements_differing": int((a != b).sum()), "elements": int(a.numel()),
            "flips": float((a != b).float().mean()),
            "tolerance_rel_l2": DEVICE_REL_L2_TOL * layers,
            "agreement_ok": rel <= DEVICE_REL_L2_TOL * layers}


def sim_agreement(got: torch.Tensor, want: torch.Tensor) -> dict:
    """``got`` against the simulator's ``0`` output, at ``test_layer_glue.py``'s tolerances."""
    out = agreement(got, want)
    out = {k: out[k] for k in ("rel_l2", "max_abs_diff", "max_steps", "flips")}
    out["within_layer_tolerance"] = (out["rel_l2"] <= SIM_REL_L2
                                     and out["max_steps"] <= SIM_MAX_STEPS
                                     and out["flips"] <= SIM_MAX_FLIPS)
    return out


def verdict(off: list[float], var: list[float], floor: float = 0.0) -> dict:
    """``win`` / ``loss`` when every round moves one way and the median delta clears both
    the spread (either variant's max - min over rounds) and ``floor``, the A/A delta (an
    identical graph compiled and loaded again, variant ``aa``); else ``neutral``."""
    deltas = [v - o for v, o in zip(var, off)]
    spread = max(max(off) - min(off), max(var) - min(var))
    bar = max(spread, floor)
    med = statistics.median(deltas)
    if all(d < 0 for d in deltas) and -med > bar:
        call = "win"
    elif all(d > 0 for d in deltas) and med > bar:
        call = "loss"
    else:
        call = "neutral"
    return {"delta_ms_per_round": deltas, "delta_median_ms": med,
            "delta_min_ms": min(deltas), "delta_max_ms": max(deltas),
            "spread_ms": spread, "aa_floor_ms": floor, "verdict": call}


def case_key(case: dict, collectives: str, args) -> str:
    chain = "" if collectives != "one-rank" else f"_ar{args.ar_chain[0]}x{args.ar_chain[1]}"
    return f"{case['tag'].replace(':', '_')}_{collectives}{chain}_s{args.stack}"


def run_case(case: dict, collectives: str, specs: dict, args) -> dict:
    torch._dynamo.reset()
    resolve = live._resolve_tp_group
    _AR_CHAIN[:] = args.ar_chain
    if collectives == "one-rank":
        live._resolve_tp_group = _OneRankGroup
    try:
        layers = build_layers(case, args.stack)
        graphs, inputs, firsts, dispatch, compile_s, pristine = {}, {}, {}, {}, {}, {}
        for name, spec in specs.items():
            tag = f"{case_key(case, collectives, args)}_{name}"
            call, ins, first, counts, secs, pristine[name] = build_variant(
                case, layers, spec, tag, args)
            graphs[name], inputs[name], firsts[name] = call, ins, first
            dispatch[name], compile_s[name] = counts, secs
            print(json.dumps({"case": case["tag"], "variant": name, "spec": spec,
                              "dispatch": counts, "compile_s": round(secs, 1)}), flush=True)
        rounds = []
        for r in range(args.rounds):
            warm = args.warmup if r == 0 else 1
            saved, args.warmup = args.warmup, warm
            try:
                rounds.append(micro.time_graphs(graphs, inputs, args))
            finally:
                args.warmup = saved
        result = {
            "case": case["tag"], **case, "collectives": collectives, "stack": args.stack,
            "ar_chain": list(args.ar_chain) if collectives == "one-rank" else None,
            "checkpoint_weights": all(bool(lc.checkpoint) for lc in layers),
            "variants": specs, "dispatch": dispatch, "compile_and_first_call_s": compile_s,
            "agreement_vs_off": {n: agreement(firsts[n], firsts["off"], args.stack)
                                 for n in specs},
            "rounds": [{n: rd[n]["device"] for n in specs} for rd in rounds],
            "host_rounds": [{n: rd[n]["host"]["median_ms"] for n in specs}
                            for rd in rounds],
        }
        if args.save_io is not None:
            args.save_io.mkdir(parents=True, exist_ok=True)
            path = args.save_io / f"{case_key(case, collectives, args)}.pt"
            torch.save({"inputs": pristine["off"], "outputs": firsts,
                        "layers": [{k: v.to("cpu") for k, v in lc.layer.state_dict().items()}
                                   for lc in layers]}, path)
            result["io_file"] = str(path)
        if args.profile_dir is not None:
            result["profiles"] = {}
            for name in specs:
                pname = f"{case_key(case, collectives, args)}_{name}"
                directory = micro.profile_graph(pname, graphs[name], inputs[name], args)
                result["profiles"][name] = {"profile_dir": str(directory),
                                            "name": pname}
        return result
    finally:
        live._resolve_tp_group = resolve


def _summary_key(case: dict) -> str:
    chain = case.get("ar_chain")
    mode = case["collectives"] + (f" ar_chain={chain[0]}x{chain[1]}" if chain else "")
    return f"{case['case']} {mode} stack={case['stack']}"


def summarise(cases: list[dict]) -> dict:
    out = {}
    for case in cases:
        key = _summary_key(case)
        medians = {n: [rd[n]["median_ms"] for rd in case["rounds"]] for n in case["variants"]}
        off = medians["off"]
        floor = (abs(statistics.median([a - o for a, o in zip(medians["aa"], off)]))
                 if "aa" in medians else 0.0)
        keys = {n: tuple(g["key"] for g in case["dispatch"][n].get("graphs", []))
                for n in case["variants"]}
        rows = {}
        for name in case["variants"]:
            m = medians[name]
            row = {"spec": case["variants"][name], "round_medians_ms": m,
                   "median_ms": statistics.median(m), "spread_ms": max(m) - min(m),
                   "per_layer_ms": statistics.median(m) / case["stack"]}
            if keys[name]:
                row["graph_keys"] = list(keys[name])
                row["compiled_here"] = [not g["cached_before"]
                                        for g in case["dispatch"][name]["graphs"]]
                row["same_graph_as"] = [n for n in case["variants"]
                                        if n != name and keys[n] == keys[name]]
            ag = case.get("agreement_vs_off", {}).get(name, {})
            if "agreement_ok" in ag:
                row["agreement_rel_l2"] = ag["rel_l2"]
                row["agreement_ok"] = ag["agreement_ok"]
            sim = case.get("agreement_vs_sim0", {}).get(name)
            if sim is not None:
                row["sim0_rel_l2"] = sim["rel_l2"]
                row["sim0_within_layer_tolerance"] = sim["within_layer_tolerance"]
                row["sim0_no_further_than_off"] = sim["no_further_than_off"]
            if name != "off":
                v = verdict(off, m, 0.0 if name == "aa" else floor)
                v["delta_median_ms_per_layer"] = v["delta_median_ms"] / case["stack"]
                row.update(v)
            prof = case.get("profiles", {}).get(name, {})
            if "mean_execution_ms" in prof:
                row["profile_execution_ms"] = prof["mean_execution_ms"]
            rows[name] = row
        out[key] = rows
    return out


def analyse(entry: dict, args) -> dict:
    """Ingest one profile and bucket it with the serving profile's frozen rules."""
    import shutil

    display = f"glue2-{entry['name']}"
    data = args.explorer_data
    global_dir = data / "profiles" / "global"
    for stale in global_dir.glob(f"{display}*"):
        shutil.rmtree(stale, ignore_errors=True)
    ingest = subprocess.run(
        ["neuron-explorer", "view", "-d", entry["profile_dir"], "--display-name", display,
         "--ingest-only", "--data-path", str(data)],
        capture_output=True, text=True, timeout=1500)
    if ingest.returncode != 0:
        raise RuntimeError(f"ingest of {entry['profile_dir']} failed: {ingest.stderr[-2000:]}")
    out_json = Path(entry["profile_dir"]) / "buckets.json"
    run = subprocess.run(
        [str(micro.DUCKDB_PYTHON), str(Path(micro.__file__).resolve()), "--bucket",
         display, str(out_json), str(global_dir)],
        capture_output=True, text=True, timeout=1500)
    if run.returncode != 0:
        raise RuntimeError(f"bucketing of {display} failed: {run.stderr[-2000:]}")
    report = json.loads(out_json.read_text())
    cc_json = Path(entry["profile_dir"]) / "ccops.json"
    run = subprocess.run(
        [str(micro.DUCKDB_PYTHON), str(Path(__file__).resolve()), "--ccops", display,
         str(cc_json), str(global_dir)], capture_output=True, text=True, timeout=600)
    if run.returncode != 0:
        raise RuntimeError(f"collective spans of {display} failed: {run.stderr[-2000:]}")
    spans = json.loads(cc_json.read_text())["executions"]
    return {"profile": display, "trace_dir": report["trace_dir"],
            "collectives_per_execution": spans,
            "fully_traced_executions": report["fully_traced_steps"],
            "mean_execution_ms": report["mean_step_ms"],
            "buckets_ms": dict(sorted(report["buckets_ms"].items(), key=lambda kv: -kv[1])),
            "per_label_cores": report["per_label_cores"]}


def sim_check(args) -> None:
    """``--sim-check``: every saved device output against the CPU simulator's ``0``.

    For each case of ``--output`` that saved its operands (``--save-io``): build the same
    layers on the CPU, load the device's weights (after its load-time preps) into them,
    run the ``0`` routes in the NKI simulator on the device's own operands, and compare
    each variant's device output to that, at ``test_layer_glue.py``'s layer tolerances
    (:data:`SIM_REL_L2`, :data:`SIM_MAX_STEPS`, :data:`SIM_MAX_FLIPS`). ``off``'s own row
    is the control: the device's ``0`` against the simulator's.
    """
    if (os.environ.get("VLLM_NEURON_CPU_MODE") != "1" or os.environ.get("NKI_SIMULATOR") != "1"
            or not os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE")):
        # The simulator asks the runtime for the platform unless the override names it.
        raise ValueError("--sim-check runs in the simulator: set VLLM_NEURON_CPU_MODE=1, "
                         "NKI_SIMULATOR=1 and NEURON_PLATFORM_TARGET_OVERRIDE (the "
                         "platform the device run used, e.g. trn2)")
    report = json.loads(args.output.read_text())
    quant = glue_case.quant_config(live)
    for case in report["cases"]:
        if "io_file" not in case or ("agreement_vs_sim0" in case and not args.reanalyze):
            continue
        saved = torch.load(case["io_file"])
        spec = {k: case[k] for k in ("family", "phase", "rows", "tag")}
        layers = build_layers(spec, case["stack"], device="cpu")
        state_equal = []
        for lc, state in zip(layers, saved["layers"]):
            mine = lc.layer.state_dict()
            state_equal.append(set(mine) == set(state) and all(
                torch.equal(mine[k], v) for k, v in state.items()))
            lc.layer.load_state_dict(state, strict=True)
        per_layer = [micro._flatten(carriers_for(spec, lc, device="cpu")) for lc in layers]
        fresh = [glue_case.streams_input(layers[0].cfg, spec["rows"]),
                 torch.tensor(0, dtype=torch.int64),
                 *(t for _, tensors, _ in per_layer for t in tensors)]
        operands_equal = len(fresh) == len(saved["inputs"]) and all(
            torch.equal(a, b) for a, b in zip(fresh, saved["inputs"]))
        operands = [t.clone() for t in saved["inputs"]]  # the layer writes its state banks
        os.environ[glue.GLUE_FUSED_ENV] = "0"
        streams, rank, at = operands[0], operands[1], 2
        with torch.no_grad():
            for lc, (names, tensors, statics) in zip(layers, per_layer):
                carriers = micro._unflatten(names, operands[at:at + len(tensors)], statics)
                at += len(tensors)
                streams = glue_case.layer_step(live, lc, streams, carriers, rank, quant)
        versus = {name: sim_agreement(out, streams) for name, out in saved["outputs"].items()}
        for row in versus.values():
            # The bar a device output can be held to: as close to the simulator's ``0`` as
            # the device's own ``0`` (its torch routes) is.
            row["no_further_than_off"] = row["rel_l2"] <= versus["off"]["rel_l2"]
        case["agreement_vs_sim0"] = {"operands_equal": operands_equal,
                                     "weights_equal_before_load": state_equal, **versus}
        print(json.dumps({"case": _summary_key(case), "sim0": {
            n: (round(v["rel_l2"], 6), v["within_layer_tolerance"], v["no_further_than_off"])
            for n, v in versus.items()}}), flush=True)
        report["summary"] = summarise(report["cases"])
        args.output.write_text(json.dumps(report, indent=1) + "\n")


def analyse_only(args) -> None:
    report = json.loads(args.output.read_text())
    for case in report["cases"]:
        for name, entry in case.get("profiles", {}).items():
            if "mean_execution_ms" in entry and not args.reanalyze:
                continue
            entry.update(analyse(entry, args))
            print(json.dumps({"case": case["case"], "variant": name,
                              "profile_execution_ms": entry["mean_execution_ms"]}),
                  flush=True)
            args.output.write_text(json.dumps(report, indent=1) + "\n")
    report["summary"] = summarise(report["cases"])
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["summary"], indent=1), flush=True)


def _shapes() -> dict:
    """The model shapes every case is built at (one rank of the served layout)."""
    cfg = glue_case.text_config()
    return {"hidden": int(cfg.hidden_size), "mhc_streams": int(cfg.hc_mult),
            "tp_world": glue_case.TP_WORLD, "ep_degree": glue_case.EP_DEGREE,
            "routed_experts": int(cfg.n_routed_experts)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=list(CASES))
    parser.add_argument("--variants", nargs="*", default=None,
                        help="off, a kernel name, all, default, or name=spec "
                             "(default: off, each kernel the layer holds, all)")
    parser.add_argument("--collectives", nargs="+", default=["one-rank"],
                        choices=COLLECTIVES)
    parser.add_argument("--stack", type=int, default=1, help="layers per graph")
    parser.add_argument("--ar-chain", type=int, nargs=2, default=[1, 1],
                        metavar=("ATTN", "FFN"),
                        help="one-rank all-reduces in a row at each attention / FFN "
                             "reduction (one-rank mode)")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--profile-iterations", type=int, default=3)
    parser.add_argument("--profile-dir", type=Path, default=None,
                        help="where the device profiles go (required unless --no-profile)")
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument("--explorer-data", type=Path, default=None,
                        help="neuron-explorer's data directory (default: "
                             "PROFILE_DIR/explorer-data)")
    parser.add_argument("--analyze-only", action="store_true",
                        help="ingest and bucket the profiles --output names (no device)")
    parser.add_argument("--reanalyze", action="store_true")
    parser.add_argument("--save-io", type=Path, default=None,
                        help="save each case's operands, weights and first outputs here "
                             "(for --sim-check)")
    parser.add_argument("--sim-check", action="store_true",
                        help="compare the saved device outputs to the CPU simulator's 0 "
                             "(VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1; no device)")
    parser.add_argument("--merge", action="store_true",
                        help="keep the cases already in --output that this run does not redo")
    parser.add_argument("--compiler-args", nargs="*",
                        default=(os.environ.get("NEURON_CC_FLAGS", "").split()
                                 or micro.MODEL_COMPILER_ARGS))
    args = parser.parse_args()
    if args.explorer_data is None and args.profile_dir is not None:
        args.explorer_data = args.profile_dir / "explorer-data"
    if args.analyze_only and args.explorer_data is None:
        parser.error("--analyze-only needs --explorer-data or --profile-dir")
    if not (args.no_profile or args.sim_check or args.analyze_only) and args.profile_dir is None:
        parser.error("--profile-dir is required unless --no-profile")
    for name in ("output", "profile_dir", "explorer_data", "save_io"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    if args.sim_check:
        sim_check(args)
        return
    if args.analyze_only:
        analyse_only(args)
        return
    if args.no_profile:
        args.profile_dir = None
    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1":
        raise ValueError("Hardware benchmark cannot run in VLLM_NEURON_CPU_MODE=1")
    if not os.environ.get("NEURON_RT_VISIBLE_CORES"):
        raise ValueError("Set NEURON_RT_VISIBLE_CORES to the cores this benchmark may use; "
                         "it does not select cores itself")
    if not Path(live.__file__).resolve().is_relative_to(ROOT):
        raise RuntimeError(f"model_fp8 imported from {live.__file__}, not {ROOT}")
    cases = [parse_case(c) for c in args.cases]
    plans = [(case, variant_specs(case, args.variants)) for case in cases]
    # The compilers write their intermediates into the working directory.
    os.chdir(tempfile.mkdtemp(prefix="glue-block-bench-cwd-"))
    report = {
        "environment": {k: os.environ.get(k) for k in (
            "NEURON_RT_VISIBLE_CORES", "NEURON_LOGICAL_NC_CONFIG", "NEURON_CC_FLAGS",
            "NEURON_LIBTORCH_CACHE_ROOT")},
        "device": "neuron:0 (one logical core = 2 physical cores at LNC2)",
        "compiler_args": args.compiler_args,
        "default_spec": envs.DEFAULT_GLUE_FUSED_SPEC,
        "shapes": _shapes(),
        "method": __doc__.split("What is measured", 1)[1].split("The compile cache", 1)[0]
        .strip(),
        "rounds": args.rounds, "warmup": args.warmup, "iterations": args.iterations,
        "profile_iterations": args.profile_iterations,
        "cases": [],
    }
    if args.merge and args.output.exists():
        previous = json.loads(args.output.read_text())
        redo = {_summary_key({"case": c["tag"], "collectives": m, "stack": args.stack,
                              "ar_chain": list(args.ar_chain) if m == "one-rank" else None})
                for c in cases for m in args.collectives}
        report["cases"] = [c for c in previous.get("cases", [])
                           if _summary_key(c) not in redo]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _record_graph_keys()
    report["environment"]["cache_entries_at_start"] = cache_entries()
    if "one-rank" in args.collectives:
        init_one_rank_world()
    for case, specs in plans:
        for collectives in args.collectives:
            t0 = time.time()
            result = run_case(case, collectives, specs, args)
            result["wall_s"] = time.time() - t0
            report["cases"].append(result)
            report["summary"] = summarise(report["cases"])
            args.output.write_text(json.dumps(report, indent=1) + "\n")
            key = _summary_key(result)
            print(json.dumps({key: {n: {k: r.get(k) for k in (
                "median_ms", "spread_ms", "delta_median_ms", "verdict")}
                for n, r in report["summary"][key].items()}}), flush=True)


if __name__ == "__main__":
    main()
