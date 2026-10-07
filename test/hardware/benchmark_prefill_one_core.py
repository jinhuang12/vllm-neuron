# SPDX-License-Identifier: Apache-2.0
"""Time the two prefill kernels that ran on one physical core: this tree against 342e93e.

Two subcommands.

``run`` (device; through the lease, which pins the cores and the LNC -- this script
selects none). For each prompt length ``T`` (default ``64 1024``) it builds ``--layers``
(default 8) layers of operands at the per-rank TP=64 prefill shapes and times, for
``before`` (342e93e, ``test/hardware/baselines/prefill_cores_342e93e``) and ``after``
(this tree):

* ``kda`` -- one KDA layer's chunked recurrence as ``Glm5NextKDAAttention`` runs it at
  prefill: one head per rank (64 heads of 128 at TP=64), ``chunk`` 8 (the layer's own
  choice at ``gate_lower_bound`` -5), ``T / 8`` chunks, ``kda_intra_chunk`` then
  ``kda_inter_chunk`` with an entering state. ``intra`` and ``inter`` time the two
  dispatches alone (``inter`` on the CPU oracle's intra outputs).
* ``kpool`` -- ``dsa_kpool_hadamard`` on ``T`` pools of ``index_kpool`` 4 keys of 128
  (``pool_window`` makes every position a candidate, so ``n_pools == T``), bf16.
* ``pool_leg`` -- the prefill pool write as ``Glm5NextDSAIndexer.forward`` runs it from
  ``key`` and ``gate_score``: ``pool_window`` (the window gather and the kernel), the
  write mask, and the ``index_copy_`` into the pool cache. This is the single-layer
  critical-path reading the kernel row alone does not give.
* ``qrot`` -- context only: the prefill query rotation ``dsa_hadamard128`` on
  ``T * 32`` rows, which the same indexer leg runs and this change does not edit.

Every graph returns the last 8 elements of each layer's outputs, so every kernel stays
live and the copy to CPU that synchronises each timed call is a few hundred bytes. The
reported unit is the graph's wall time divided by the layer count; ``empty`` times a
graph that only slices its inputs the same way, the fixed per-call cost the other rows
carry. Before timing, ``before`` and ``after`` run once on the same operands and every
output is compared: ``bitwise`` and the largest absolute difference are recorded.

With ``--profile-dir`` the ``kda`` and ``kpool`` graphs of both trees then run
``--profile-iterations`` times under the runtime profiler (device and system profile).

``cores`` (CPU only; any python with ``duckdb``): ``benchmark_dsa_indexer.py cores``,
the per-kernel, per-pcore instruction and active-time table of an ingested profile.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

DEVICE = "neuron:0"
DEFAULT_TOKENS = (64, 1024)
TAIL = 8
CHUNK = 8
HEAD = 128
POOL = 4
INDEX_HEADS = 32


def _torch():
    import torch
    return torch


def load_baseline(directory: Path):
    """The baseline package's ``load()``: 342e93e's modules, read by ``git show``."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "prefill_cores_baseline_loader", directory / "__init__.py",
        submodule_search_locations=[str(directory)])
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load baseline package {directory}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.load()


def compiled(fn):
    torch = _torch()
    return torch.compile(fn, backend="neuron_libtorch", fullgraph=True, dynamic=False,
                         options={"compiler_args": os.environ.get("NEURON_CC_FLAGS", "")})


def tails(*tensors):
    torch = _torch()
    return torch.stack([t.reshape(-1)[-TAIL:].to(torch.float32) for t in tensors])


def measure(fn, inputs, per: int, warmup: int, iterations: int) -> dict:
    for _ in range(warmup):
        fn(*inputs).to("cpu")
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        fn(*inputs).to("cpu")
        samples.append((time.perf_counter_ns() - started) / 1000.0 / per)
    samples.sort()
    return {"iterations": iterations, "units_per_sample": per,
            "median_us": statistics.median(samples),
            "p90_us": samples[min(len(samples) - 1, int(0.9 * len(samples)))],
            "min_us": samples[0], "max_us": samples[-1]}


# ------------------------------------------------------------------------------------
# operands
# ------------------------------------------------------------------------------------


def kda_operands(tokens: int, seed: int) -> dict:
    """One KDA layer's prefill operands for one head, as the layer hands them over.

    The gate is the clamp's range, ``[-5, 0]``, so a chunk-local cumulative gate stays
    inside ``GATE_CUMSUM_ABS_LIMIT`` at chunk 8 as the layer's chunk choice guarantees.
    """
    torch = _torch()
    gen = torch.Generator().manual_seed(int(seed))
    n = tokens // CHUNK
    shape = (n, CHUNK, HEAD)
    return {
        "q": torch.randn(shape, generator=gen),
        "k": torch.randn(shape, generator=gen),
        "v": torch.randn(shape, generator=gen),
        "beta": torch.rand((n, CHUNK), generator=gen) * 0.9 + 0.05,
        "gk": -torch.rand(shape, generator=gen) * 5.0,
        "state": torch.randn((HEAD, HEAD), generator=gen) * 0.1,
    }


def kpool_operands(tokens: int, seed: int) -> dict:
    torch = _torch()
    gen = torch.Generator().manual_seed(int(seed))
    key = torch.randn((tokens, HEAD), generator=gen).to(torch.bfloat16)
    gate = torch.randn((tokens, HEAD), generator=gen).to(torch.bfloat16)
    pos = torch.arange(tokens)
    window = (pos - (POOL - 1)).clamp_min(0)[:, None] + torch.arange(POOL)[None, :]
    # Pool-granular slots: the last token of each complete pool carries its pool index.
    slots = torch.where((pos % POOL) == POOL - 1, pos // POOL, torch.full_like(pos, -1))
    return {
        "key": key, "gate": gate,
        "slot_k": key[window].contiguous(), "slot_score": gate[window].contiguous(),
        "slot_mapping": slots.to(torch.int64),
        "pool_cache": torch.zeros((tokens // POOL + 1, HEAD), dtype=torch.bfloat16),
        "query_rows": (torch.randn((tokens * INDEX_HEADS, HEAD), generator=gen)
                       ).to(torch.bfloat16),
    }


def ape_tensor():
    torch = _torch()
    return torch.randn((POOL, HEAD), generator=torch.Generator().manual_seed(5)) * 0.1


# ------------------------------------------------------------------------------------
# the two trees
# ------------------------------------------------------------------------------------


class Tree:
    """One tree's prefill entry points."""

    def __init__(self, name, chunked, kpool):
        self.name = name
        self.chunked = chunked
        self.kpool = kpool

    def kda(self, o):
        intra = self.chunked.kda_intra_chunk(o["q"], o["k"], o["v"], o["beta"], o["gk"])
        inter = self.chunked.kda_inter_chunk(intra.kg, intra.w, intra.u, o["gk"], o["q"],
                                             intra.aqk, state=o["state"])
        return (*intra, *inter)

    def intra(self, o):
        return tuple(self.chunked.kda_intra_chunk(o["q"], o["k"], o["v"], o["beta"],
                                                  o["gk"]))

    def inter(self, o):
        return tuple(self.chunked.kda_inter_chunk(o["kg"], o["w"], o["u"], o["gk"], o["q"],
                                                  o["aqk"], state=o["state"]))

    def pool(self, o, ape):
        return (self.kpool.dsa_kpool_hadamard(o["slot_k"], o["slot_score"], ape),)

    def pool_leg(self, o, ape):
        """``pool_window`` and the pool write of ``Glm5NextDSAIndexer.forward``'s prefill leg."""
        torch = _torch()
        key, gate, slot_mapping = o["key"], o["gate"], o["slot_mapping"]
        tokens = int(key.shape[0])
        pos = torch.arange(tokens, device=key.device)
        offsets = torch.arange(POOL, device=key.device)
        window = (pos - (POOL - 1)).clamp_min(0)[:, None] + offsets[None, :]
        write_mask = (slot_mapping >= 0) & (pos >= POOL - 1)
        pooled = self.kpool.dsa_kpool_hadamard(key[window], gate[window], ape)
        cache = o["pool_cache"]
        trash = int(cache.shape[0]) - 1
        destination = torch.where(write_mask, slot_mapping,
                                  cache.new_full((), trash, dtype=torch.int64))
        return (cache.index_copy(0, destination, pooled.to(cache.dtype)),)

    def qrot(self, o):
        return (self.kpool.dsa_hadamard128(o["query_rows"]),)


def live_tree() -> Tree:
    """This tree's entry points: on the lease's LNC2 runtime ``kda_intra_chunk`` and
    ``kda_inter_chunk`` launch ``chunked_lnc2``'s two-core kernels."""
    from vllm_neuron.functional.dsa import kpool_hadamard
    from vllm_neuron.functional.kda import chunked_recurrence
    return Tree("after", chunked_recurrence, kpool_hadamard)


def base_tree(base) -> Tree:
    return Tree("before", base.chunked_recurrence, base.kpool_hadamard)


# ------------------------------------------------------------------------------------
# graphs
# ------------------------------------------------------------------------------------


def to_device(ops: dict) -> dict:
    return {k: v.to(DEVICE, copy=True) for k, v in ops.items()}


def _layer_graph(method, keys, n_layers, extra=()):
    def fn(*flat):
        res = []
        for i in range(n_layers):
            ops = dict(zip(keys, flat[i * len(keys):(i + 1) * len(keys)]))
            res.extend(method(ops, *extra))
        return tails(*res)
    return fn


def _flat(layers, keys):
    dev = [to_device({k: ops[k] for k in keys}) for ops in layers]
    return tuple(d[k] for d in dev for k in keys)


def _stage_inter(layers, base):
    """The inter-chunk operands, computed once by the CPU oracle (timing does not depend
    on them, agreement is checked on the whole ``kda`` chain)."""
    staged = []
    for ops in layers:
        intra = base.chunked_recurrence.kda_intra_chunk_torch_oracle(
            ops["q"], ops["k"], ops["v"], ops["beta"], ops["gk"])
        staged.append({**ops, "kg": intra.kg.contiguous(), "w": intra.w.contiguous(),
                       "u": intra.u.contiguous(), "aqk": intra.aqk.contiguous()})
    return staged


def _compare(a, b) -> dict:
    torch = _torch()
    rows = []
    for x, y in zip(a, b):
        x, y = x.to("cpu"), y.to("cpu")
        same = bool(torch.equal(x, y))
        diff = float((x.float() - y.float()).abs().max()) if x.numel() else 0.0
        scale = float(y.float().abs().max()) if y.numel() else 0.0
        rows.append({"shape": list(x.shape), "bitwise": same, "max_abs_diff": diff,
                     "ref_abs_max": scale})
    return {"bitwise": all(r["bitwise"] for r in rows),
            "max_abs_diff": max(r["max_abs_diff"] for r in rows), "outputs": rows}


def agreement(trees, kda0, kp0, ape_dev) -> dict:
    torch = _torch()
    out = {}
    for label, method, ops, extra in (("kda", "kda", kda0, ()),
                                      ("kpool", "pool", kp0, (ape_dev,)),
                                      ("pool_leg", "pool_leg", kp0, (ape_dev,))):
        got = {}
        for name, tree in trees.items():
            torch._dynamo.reset()
            keys = sorted(ops)
            fn = compiled(lambda *flat, _t=tree, _m=method, _k=keys: tuple(
                getattr(_t, _m)(dict(zip(_k, flat)), *extra)))
            got[name] = [t.to("cpu") for t in fn(*[ops[k] for k in keys])]
        out[label] = _compare(got["after"], got["before"])
        if label == "pool_leg":
            # The cache's last row is the trash row every masked position writes to: duplicate
            # index_copy destinations, which the device serves in no fixed order.
            out[label]["excluding_trash_row"] = _compare(
                [t[:-1] for t in got["after"]], [t[:-1] for t in got["before"]])
    return out


def run_case(tokens: int, base, args) -> dict:
    torch = _torch()
    torch._dynamo.reset()
    trees = {"before": base_tree(base), "after": live_tree()}
    kda_layers = [kda_operands(tokens, 1000 + 7 * i + tokens) for i in range(args.layers)]
    kp_layers = [kpool_operands(tokens, 2000 + 7 * i + tokens) for i in range(args.layers)]
    ape_dev = ape_tensor().to(DEVICE)

    agree = agreement(trees, to_device(kda_layers[0]), to_device(kp_layers[0]), ape_dev)
    print(json.dumps({"T": tokens, "agreement": {
        k: {"bitwise": v["bitwise"], "max_abs_diff": v["max_abs_diff"],
            **({"excluding_trash_row_bitwise": v["excluding_trash_row"]["bitwise"]}
               if "excluding_trash_row" in v else {})}
        for k, v in agree.items()}}), flush=True)

    kda_keys = ["q", "k", "v", "beta", "gk", "state"]
    inter_keys = ["kg", "w", "u", "gk", "q", "aqk", "state"]
    staged = _stage_inter(kda_layers, base)
    graphs = {"empty": (lambda *flat: tails(*flat), _flat(kda_layers, ["q"]), 1)}
    for name, tree in trees.items():
        graphs[f"kda_{name}"] = (_layer_graph(tree.kda, kda_keys, args.layers),
                                 _flat(kda_layers, kda_keys), args.layers)
        graphs[f"intra_{name}"] = (_layer_graph(tree.intra, kda_keys[:5], args.layers),
                                   _flat(kda_layers, kda_keys[:5]), args.layers)
        graphs[f"inter_{name}"] = (_layer_graph(tree.inter, inter_keys, args.layers),
                                   _flat(staged, inter_keys), args.layers)
        graphs[f"kpool_{name}"] = (
            _layer_graph(tree.pool, ["slot_k", "slot_score"], args.layers, (ape_dev,)),
            _flat(kp_layers, ["slot_k", "slot_score"]), args.layers)
        graphs[f"pool_leg_{name}"] = (
            _layer_graph(tree.pool_leg, ["key", "gate", "slot_mapping", "pool_cache"],
                         args.layers, (ape_dev,)),
            _flat(kp_layers, ["key", "gate", "slot_mapping", "pool_cache"]), args.layers)
    graphs["qrot_before"] = (_layer_graph(trees["before"].qrot, ["query_rows"], args.layers),
                             _flat(kp_layers, ["query_rows"]), args.layers)

    keep = set(args.graphs) if args.graphs else None
    timing = {}
    for gname, (fn, inputs, per) in graphs.items():
        if keep is not None and gname not in keep and gname != "empty":
            continue
        torch._dynamo.reset()
        cfn = compiled(fn)
        started = time.perf_counter()
        cfn(*inputs).to("cpu")
        first = time.perf_counter() - started
        timing[gname] = measure(cfn, inputs, per, args.warmup, args.iterations)
        timing[gname]["first_call_s"] = first
        print(json.dumps({"T": tokens, "graph": gname,
                          "median_us": round(timing[gname]["median_us"], 1)}), flush=True)
        if args.profile_dir is not None and (gname.startswith("kda_")
                                             or gname.startswith("kpool_")):
            _profile(cfn, inputs, args, f"t{tokens}_{gname}")

    row = {"tokens": tokens, "chunk": CHUNK, "n_chunks": tokens // CHUNK,
           "n_pools": tokens, "pool_size": POOL, "head_dim": HEAD, "layers": args.layers,
           "agreement": agree, "timing": timing, "speedup": {}}
    for stage in ("kda", "intra", "inter", "kpool", "pool_leg"):
        b, a = timing.get(f"{stage}_before"), timing.get(f"{stage}_after")
        if b and a:
            row["speedup"][stage] = {"before_median_us": b["median_us"],
                                     "after_median_us": a["median_us"],
                                     "before_p90_us": b["p90_us"], "after_p90_us": a["p90_us"],
                                     "median_speedup": b["median_us"] / a["median_us"],
                                     "after_over_before": a["median_us"] / b["median_us"]}
    return row


def _profile(cfn, inputs, args, tag):
    torch = _torch()
    import libtorch_neuronx_lite.envs as libtorch_envs
    where = args.profile_dir / tag
    where.mkdir(parents=True, exist_ok=True)
    runtime = torch.classes.neuron.Runtime()
    runtime.start_profiling(str(where), ["device_profile", "system_profile"], None, None,
                            libtorch_envs.get_neuron_compile_cache_dir())
    try:
        for _ in range(args.profile_iterations):
            cfn(*inputs).to("cpu")
    finally:
        runtime.stop_profiling()


def _after_kda_two_core() -> bool:
    from vllm_neuron.functional.kda import chunked_lnc2
    return chunked_lnc2.chunked_lnc2_enabled()


def cmd_run(args) -> None:
    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1" or os.environ.get("NKI_SIMULATOR"):
        raise ValueError("Hardware benchmark cannot run in CPU mode or the NKI simulator")
    if not os.environ.get("NEURON_RT_VISIBLE_CORES"):
        raise ValueError("Run under the device lease, which pins NEURON_RT_VISIBLE_CORES")
    import vllm_neuron  # noqa: F401 -- registers the Neuron compilation backend
    if not vllm_neuron.__file__.startswith(str(ROOT) + "/"):
        raise ValueError(f"imported vllm_neuron from {vllm_neuron.__file__}, not {ROOT}")
    base = load_baseline(args.baseline_module.resolve())
    output = args.output.resolve()
    if args.profile_dir is not None:
        args.profile_dir = args.profile_dir.resolve()
    scratch = Path(os.environ.get("NEURON_LIBTORCH_CACHE_ROOT") or tempfile.gettempdir())
    scratch = scratch / "benchmark_cwd"
    scratch.mkdir(parents=True, exist_ok=True)
    os.chdir(scratch)
    report = {}
    if args.merge and output.exists():
        report = json.loads(output.read_text())
    report.setdefault("cases", [])
    report.update({
        "environment": {key: os.environ.get(key) for key in (
            "NEURON_RT_VISIBLE_CORES", "NEURON_LOGICAL_NC_CONFIG", "NEURON_CC_FLAGS",
            "NEURON_PLATFORM_TARGET_OVERRIDE", "NEURON_LIBTORCH_CACHE_ROOT",
            "VLLM_NEURON_KDA_CHUNKED_LNC2")},
        "after_kda_two_core_kernels": _after_kda_two_core(),
        "tree": str(ROOT),
        "baseline": {"commit": base.commit, "directory": base.directory},
        "args": {"layers": args.layers, "warmup": args.warmup,
                 "iterations": args.iterations, "tokens": args.tokens,
                 "profile_dir": None if args.profile_dir is None else str(args.profile_dir),
                 "profile_iterations": args.profile_iterations},
        "unit": "graph wall time / layers, microseconds; 'empty' is per call",
        "synchronization": "graph output (last 8 elements of each output) copied to CPU "
                           "on every timed call",
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    for tokens in args.tokens:
        row = run_case(int(tokens), base, args)
        report["cases"] = [r for r in report["cases"] if r["tokens"] != row["tokens"]] + [row]
        report["cases"].sort(key=lambda r: r["tokens"])
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"T": tokens, "speedup": row["speedup"]}), flush=True)


def cmd_cores(args) -> None:
    from test.hardware.benchmark_dsa_indexer import cmd_cores as indexer_cores
    indexer_cores(args)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tokens", nargs="+", type=int, default=list(DEFAULT_TOKENS))
    r.add_argument("--layers", type=int, default=8)
    r.add_argument("--warmup", type=int, default=5)
    r.add_argument("--iterations", type=int, default=50)
    r.add_argument("--graphs", nargs="*", default=None,
                   help="time only these graphs (plus 'empty'); default all")
    r.add_argument("--baseline-module", type=Path,
                   default=ROOT / "test/hardware/baselines/prefill_cores_342e93e")
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--merge", action="store_true")
    r.add_argument("--profile-dir", type=Path, default=None)
    r.add_argument("--profile-iterations", type=int, default=1)
    c = sub.add_parser("cores")
    c.add_argument("--data", required=True)
    c.add_argument("--name", required=True)
    c.add_argument("--functions", action="store_true")
    args = ap.parse_args()
    (cmd_run if args.cmd == "run" else cmd_cores)(args)


if __name__ == "__main__":
    main()
