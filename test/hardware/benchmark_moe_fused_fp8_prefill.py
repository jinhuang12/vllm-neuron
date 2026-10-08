# SPDX-License-Identifier: Apache-2.0
"""Prefill microbenchmark of the fused FP8 routed experts against a module snapshot.

Run under ``devlease.py slice moe-t`` (one trn2 chip, LNC2). This script does not
select cores. ``neuron:0`` is one logical core (two physical cores, the kernel's
two programs).

What is timed. ``fused_fp8_experts`` (the packed branch of
``Glm5NextRoutedExperts.block_quant_expert_mm``) for one rank's expert shard at
the served prefill buckets: ``--tokens 1024`` gives ``row_ids [49, 256]`` (the
1k-chunk line) and ``--tokens 2048`` gives ``[81, 256]`` (the 2048-token chunk of
the 64k line). "before" is the snapshot package's wrapper and kernel
(``--baseline-module``; its ``moe_fused_fp8.py`` is byte-identical to f3a833f's,
see ``kernel_sources`` in the output); "after" is this tree's.

Inputs. Each token picks ``TOP_K`` distinct experts of ``EXPERTS`` uniformly at
random (seeded) with normalized weights times the routed scaling factor; this
rank keeps its ``LOCAL_EXPERTS`` columns. The model's own block mapping
(``_build_blockwise_mapping_torch``) and padding order then build ``row_ids``,
``expert_ids`` and the affinities exactly as ``block_quant_expert_mm`` does. The
realised routed pairs (``row_ids >= 0``) are reported per seed beside the
expectation ``T * TOP_K * LOCAL_EXPERTS / EXPERTS``. ``--skewed-seeds`` add
seeds whose routing sends most tokens to a few local experts, which exercises
the dense and paired-block paths. Expert weights and scales are synthetic,
shape- and dtype-exact (fp8 values in the finite Trn2 range, non-power-of-two
scales).

Method. Each variant is compiled twice into one graph: 1 call and ``L`` calls
(``--layers``, distinct weight copies per call). Call ``l`` reads call
``l-1``'s output through its SwiGLU bounds operand,
``bounds + 2^-8 * y_{l-1}[:128, :3]`` (a 128x3 elementwise op), so the calls
cannot overlap; the graphs return that 128x3 slice, not the full emission. Per
call = ``(t_L - t_1) / (L - 1)`` per paired sample, with device time taken from
the runtime system trace (LNC2 physical-core intervals merged). Repetitions
(``--reps``) are interleaved before/after; each repetition times all four graphs
with interleaved calls. The noise floor is the spread of the per-repetition
medians of one variant. Numerics: the 1-call graphs' full emissions are saved
per seed and compared bitwise (``torch.equal``).

The compile cache is off unless ``--use-compile-cache``: the graph cache key
ignores kernel bodies, so an edited kernel could otherwise be timed from a
stale NEFF.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import statistics
import sys
import time

# The acceptance command runs the script without PYTHONPATH; Python would put
# test/hardware first and the venv would resolve vllm_neuron to another tree.
REPO = Path(__file__).resolve().parents[2]
if sys.path[:1] != [str(REPO)]:
    sys.path.insert(0, str(REPO))

import torch  # noqa: E402

import vllm_neuron  # noqa: E402,F401 -- registers the Neuron compilation backend
import libtorch_neuronx_lite.envs as libtorch_envs  # noqa: E402

from vllm_neuron.functional.moe.blockwise_fp8_retile import BLOCK_QUANT_SIZE  # noqa: E402
from vllm_neuron.functional.moe.fused_fp8_pack import PackedExperts, pack_experts  # noqa: E402
from vllm_neuron.functional.moe.moe_blockwise import (  # noqa: E402
    _build_blockwise_mapping_torch,
)
from vllm_neuron.functional.moe.moe_blockwise_fp8 import _swiglu_bound_operand  # noqa: E402

# GLM-5.3-Flash at EP=16: 288 routed experts, top-8, 18 local experts with the
# full 512-wide intermediate, hidden 4096, routed scaling 2.5, SwiGLU limit 10.
HIDDEN, EXPERTS, TOP_K, LOCAL_EXPERTS, LOCAL_INTERMEDIATE = 4096, 288, 8, 18, 512
SCALING, SWIGLU_LIMIT, RANK = 2.5, 10.0, 5
DEVICE = "neuron:0"
#: Served blocks per bucket (op inventory shapes ``s32[49, 256]``, ``s32[81, 256]``).
SERVED_BLOCKS = {1024: 49, 2048: 81}
#: Weight of the previous call's output in the next call's bounds.
CHAIN = 2.0 ** -8

#: The served model's neuronx-cc arguments, as in ``benchmark_moe_decode.py``.
MODEL_COMPILER_ARGS = [
    "--auto-cast=none",
    "-O1",
    "--internal-hlo2tensorizer-options=--modular-flow-mac-threshold=10 "
    "--experimental-unsafe-fp8e4m3fn-as-fp8e4m3",
    "--internal-backend-options=--enable-verifier=false --enable-nested-dynamic-loop",
]


def load_baseline(path: Path):
    """Import the snapshot directory as package ``moe_5938748``; return its wrapper."""
    name = "moe_5938748"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, path / "__init__.py", submodule_search_locations=[str(path)])
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load baseline package {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return importlib.import_module(f"{name}.fused_fp8")


def md5(path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


# ---- Inputs ----------------------------------------------------------------- #


def expert_bank(seed: int) -> PackedExperts:
    gen = torch.Generator().manual_seed(seed)
    nh, ni = HIDDEN // 128, LOCAL_INTERMEDIATE // 128

    def fp8(*shape):
        return (torch.randn(*shape, generator=gen) * 48).clamp(-240, 240).to(torch.float8_e4m3fn)

    gate_up = fp8(LOCAL_EXPERTS, HIDDEN, 2 * LOCAL_INTERMEDIATE)
    down = fp8(LOCAL_EXPERTS, LOCAL_INTERMEDIATE, HIDDEN)
    gu_scales = (0.5 + torch.rand(LOCAL_EXPERTS, nh, 2, ni, generator=gen)) * (3.0 / (48 * 64))
    d_scales = (0.5 + torch.rand(LOCAL_EXPERTS, ni, nh, generator=gen)) * (
        1.0 / (48 * LOCAL_INTERMEDIATE ** 0.5))
    return pack_experts(gate_up, down, gu_scales, d_scales)


def local_affinities(tokens: int, seed: int, skewed: bool) -> torch.Tensor:
    """This rank's ``[T, LOCAL_EXPERTS]`` slice of a top-k router output."""
    gen = torch.Generator().manual_seed(seed)
    logits = torch.rand(tokens, EXPERTS, generator=gen)
    if skewed:
        # Three in four tokens pick the same three local experts.
        boost = torch.zeros(EXPERTS)
        boost[[RANK * LOCAL_EXPERTS + e for e in (1, 2, 9)]] = 2.0
        hot = (torch.rand(tokens, generator=gen) < 0.75).to(torch.float32)
        logits += hot.unsqueeze(1) * boost
    picks = logits.topk(TOP_K, dim=1).indices
    weight = torch.rand(tokens, TOP_K, generator=gen) + 0.1
    routed = torch.zeros(tokens, EXPERTS)
    routed.scatter_(1, picks, weight / weight.sum(1, keepdim=True) * SCALING)
    return routed[:, RANK * LOCAL_EXPERTS:(RANK + 1) * LOCAL_EXPERTS].contiguous()


def routing(tokens: int, seed: int, skewed: bool):
    """``row_ids``, ``expert_ids`` and padded affinities as the model builds them."""
    affinities = local_affinities(tokens, seed, skewed)
    mask = (affinities != 0).to(torch.float32)
    positions, block_to_expert, blocks = _build_blockwise_mapping_torch(
        expert_mask=mask, num_local_experts=LOCAL_EXPERTS, num_experts_per_token=TOP_K,
        block_size=BLOCK_QUANT_SIZE, total_tokens=tokens, tp_degree=1, moe_group=None)
    rows = min(tokens, BLOCK_QUANT_SIZE)
    row_ids = positions.to(torch.int32).reshape(-1, BLOCK_QUANT_SIZE)[:, :rows].contiguous()
    expert_ids = block_to_expert.to(torch.int32).reshape(-1, 1).contiguous()
    padded = torch.cat([affinities, torch.zeros(1, LOCAL_EXPERTS)]).contiguous()
    return row_ids, expert_ids, padded, blocks


def hidden_states(tokens: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    scale = torch.ones(HIDDEN)
    scale[torch.randperm(HIDDEN, generator=gen)[:24]] = 20.0
    x = (torch.randn(tokens + 1, HIDDEN, generator=gen) * scale).to(torch.bfloat16)
    x[-1].zero_()
    return x


# ---- Graphs ----------------------------------------------------------------- #


def compiler_args():
    return os.environ.get("NEURON_CC_FLAGS") or MODEL_COMPILER_ARGS


def compile_fn(fn):
    return torch.compile(fn, backend="neuron_libtorch", fullgraph=True, dynamic=False,
                         options={"compiler_args": compiler_args()})


def chain_graph(experts_fn, calls: int):
    def chained(x, row_ids, expert_ids, affinity, bounds, *bank):
        current = bounds
        for call in range(calls):
            packed = PackedExperts(bank[2 * call], bank[2 * call + 1])
            y = experts_fn(x, packed, row_ids, expert_ids, affinity, current)
            current = bounds + CHAIN * y[:128, :3]
        return y[:128, :3]

    return compile_fn(chained)


def emission_graph(experts_fn):
    def emit(x, row_ids, expert_ids, affinity, bounds, weights, scales):
        return experts_fn(x, PackedExperts(weights, scales), row_ids, expert_ids,
                          affinity, bounds)

    return compile_fn(emit)


# ---- Timing ----------------------------------------------------------------- #


def device_intervals(events_json: str) -> list[float]:
    """Per-execution device time in us, LNC2 physical-core intervals merged."""
    starts, intervals = {}, {}
    for event in json.loads(events_json)["events"]:
        if event["event_type"] != "nc_exec_running":
            continue
        key = (event["nc_idx"], event["tracking_id"])
        if event["phase"] == "start":
            starts[key] = event
        elif event["phase"] == "stop":
            start = starts.pop(key)
            intervals.setdefault(start["data"]["exec_id"], []).append(
                (start["data"]["nc_timestamp_ns"], event["data"]["nc_timestamp_ns"]))
    return [(max(e for _, e in intervals[k]) - min(b for b, _ in intervals[k])) / 1000.0
            for k in sorted(intervals)]


def time_graphs(graphs: dict, inputs: dict, warmup: int, iterations: int) -> dict:
    """Interleave every graph's timed calls; device samples in us per graph."""
    from nrtpy._nrtpy import SystemTraceSession

    names = list(graphs)
    for _ in range(warmup):
        for name in names:
            graphs[name](*inputs[name]).to("cpu")
    order = []
    with SystemTraceSession() as trace:
        for iteration in range(iterations):
            shift = iteration % len(names)
            for name in names[shift:] + names[:shift]:
                graphs[name](*inputs[name]).to("cpu")
                order.append(name)
        events_json = trace.fetch_events_json()
    device_all = device_intervals(events_json)
    if len(device_all) != len(order):
        raise AssertionError(f"system trace has {len(device_all)} executions for {len(order)} calls")
    device = {name: [] for name in names}
    for name, value in zip(order, device_all):
        device[name].append(value)
    return device


def stats_us(samples):
    ordered = sorted(samples)
    return {"median_us": statistics.median(ordered),
            "p10_us": ordered[int(0.1 * (len(ordered) - 1))],
            "p90_us": ordered[int(0.9 * (len(ordered) - 1))],
            "min_us": ordered[0], "max_us": ordered[-1], "samples": len(ordered)}


def capture_profile(graphs, inputs, names, directory: Path, repeats: int) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    runtime = torch.classes.neuron.Runtime()
    runtime.start_profiling(str(directory), ["device_profile", "system_profile"], None, None,
                            libtorch_envs.get_neuron_compile_cache_dir())
    try:
        for _ in range(repeats):
            for name in names:
                graphs[name](*inputs[name]).to("cpu")
    finally:
        runtime.stop_profiling()
    return str(directory)


# ---- Case ------------------------------------------------------------------- #


def bucket_case(tokens: int, entries: dict, args) -> dict:
    torch._dynamo.reset()
    layers = args.layers
    seeds = [("uniform", s) for s in args.seeds] + [("skewed", s) for s in args.skewed_seeds]
    bounds = _swiglu_bound_operand(SWIGLU_LIMIT, SWIGLU_LIMIT, torch.device("cpu")).to(DEVICE)
    banks = [expert_bank(23 + layer) for layer in range(layers)]
    bank_dev = [t.to(DEVICE) for b in banks for t in (b.weights, b.scales)]
    expected_pairs = tokens * TOP_K * LOCAL_EXPERTS / EXPERTS
    case = {"tokens": tokens, "layers": layers, "expected_pairs": expected_pairs,
            "seeds": [], "reps": []}
    emit = {variant: emission_graph(fn) for variant, fn in entries.items()}
    graphs = {f"{v}_k{n}": chain_graph(fn, n) for v, fn in entries.items() for n in (1, layers)}
    timing_inputs = None
    for kind, seed in seeds:
        row_ids, expert_ids, affinity, blocks = routing(tokens, seed, kind == "skewed")
        if blocks != SERVED_BLOCKS.get(tokens, blocks):
            raise AssertionError(f"T={tokens}: mapping built {blocks} blocks, served "
                                 f"bucket has {SERVED_BLOCKS[tokens]}")
        x = hidden_states(tokens, 101 + seed)
        dev = [t.to(DEVICE) for t in (x, row_ids, expert_ids, affinity)]
        outputs = {v: g(*dev, bounds, *bank_dev[:2]).to("cpu") for v, g in emit.items()}
        record = {"routing": kind, "seed": seed, "row_ids_shape": list(row_ids.shape),
                  "realised_pairs": int((row_ids >= 0).sum()),
                  "blocks_with_rows": int((row_ids >= 0).any(1).sum()),
                  "max_rows_per_block": int((row_ids >= 0).sum(1).max())}
        for variant, out in outputs.items():
            if not torch.isfinite(out).all():
                raise AssertionError(f"T={tokens} seed {seed}: {variant} has nonfinite values")
        if "before" in outputs and "after" in outputs:
            before, after = outputs["before"], outputs["after"]
            record["bit_equal"] = bool(torch.equal(before.view(torch.int32), after.view(torch.int32)))
            record["max_abs_after_minus_before"] = float((after - before).abs().max())
            record["max_abs_before"] = float(before.abs().max())
        if args.save_dir is not None:
            for variant, out in outputs.items():
                path = args.save_dir / f"emission_T{tokens}_{kind}{seed}_{variant}.pt"
                torch.save(out, path)
                record[f"{variant}_pt"] = str(path)
        case["seeds"].append(record)
        print(json.dumps({"tokens": tokens, **record}), flush=True)
        if timing_inputs is None and kind == "uniform":
            timing_inputs = (dev, record["realised_pairs"])
    if timing_inputs is None:
        return case
    dev, pairs = timing_inputs
    case["timed_seed_realised_pairs"] = pairs
    inputs = {name: (*dev, bounds, *bank_dev[:2 * (1 if name.endswith("_k1") else layers)])
              for name in graphs}
    per_rep = {variant: [] for variant in entries}
    for rep in range(args.reps):
        # Alternate which variant leads, so drift does not favour one side.
        order = list(entries) if rep % 2 == 0 else list(entries)[::-1]
        names = [f"{v}_k{n}" for v in order for n in (1, layers)]
        samples = time_graphs({n: graphs[n] for n in names}, inputs, args.warmup, args.iterations)
        rep_record = {"rep": rep, "order": order}
        for variant in order:
            slopes = [(long - one) / (layers - 1) for long, one in
                      zip(samples[f"{variant}_k{layers}"], samples[f"{variant}_k1"])]
            rep_record[variant] = {"per_call_device": stats_us(slopes),
                                   "graph_k1_device": stats_us(samples[f"{variant}_k1"])}
            per_rep[variant].append(rep_record[variant]["per_call_device"]["median_us"])
        case["reps"].append(rep_record)
        print(json.dumps({"tokens": tokens, "rep": rep,
                          **{v: rep_record[v]["per_call_device"]["median_us"] for v in order}}),
              flush=True)
    summary = {}
    for variant, medians in per_rep.items():
        middle = statistics.median(medians)
        summary[variant] = {"per_rep_median_us": medians, "median_us": middle,
                            "min_us": min(medians), "max_us": max(medians),
                            "noise_floor_rel": (max(medians) - min(medians)) / middle}
    if "before" in summary and "after" in summary:
        summary["after_over_before"] = summary["after"]["median_us"] / summary["before"]["median_us"]
    case["summary"] = summary
    if args.profile_dir is not None:
        case["profile_dir"] = capture_profile(
            graphs, inputs, [f"{v}_k1" for v in entries], args.profile_dir / f"T{tokens}",
            args.profile_iterations)
    return case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline-module", type=Path,
                        default=REPO / "test/hardware/baselines/moe_5938748")
    parser.add_argument("--tokens", type=int, nargs="+", default=[1024, 2048])
    parser.add_argument("--variants", nargs="+", default=["before", "after"],
                        choices=["before", "after"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--skewed-seeds", type=int, nargs="*", default=[4])
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--save-dir", type=Path,
                        help="save each seed's full 1-call emissions as .pt files here")
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--profile-iterations", type=int, default=2)
    parser.add_argument("--use-compile-cache", action="store_true",
                        help="reuse cached NEFFs (off by default; see the module docstring)")
    parser.add_argument("--time-limit", type=int, default=3600,
                        help="seconds before the run aborts")
    args = parser.parse_args()
    signal.alarm(args.time_limit)
    if not args.use_compile_cache:
        os.environ["NEURON_LIBTORCH_DISABLE_COMPILE_CACHE"] = "1"
    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1":
        raise ValueError("Hardware benchmark cannot run in VLLM_NEURON_CPU_MODE=1")
    if not os.environ.get("NEURON_RT_VISIBLE_CORES"):
        raise ValueError("Run under devlease.py, which pins NEURON_RT_VISIBLE_CORES")
    if args.layers < 2 or args.reps < 1 or args.iterations < 1:
        raise ValueError("Use --layers >= 2, --reps >= 1 and --iterations >= 1")
    if not Path(vllm_neuron.__file__).resolve().is_relative_to(REPO):
        raise RuntimeError(f"imported {vllm_neuron.__file__}, not this tree ({REPO})")
    from vllm_neuron.functional.moe import fused_fp8 as current

    baseline = load_baseline(args.baseline_module.resolve())
    entries = {"before": baseline.fused_fp8_experts, "after": current.fused_fp8_experts}
    entries = {v: entries[v] for v in args.variants}
    for path in ("out", "save_dir", "profile_dir"):
        if getattr(args, path) is not None:
            setattr(args, path, getattr(args, path).resolve())
    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)
    # The compiler drivers write per-kernel work directories into the cwd.
    workdir = Path(os.environ.get("NEURON_LIBTORCH_CACHE_ROOT") or "/tmp") / "benchmark_workdir"
    workdir.mkdir(parents=True, exist_ok=True)
    os.chdir(workdir)
    kernel_dir = Path(current.__file__).parent
    report = {
        "environment": {k: os.environ.get(k) for k in (
            "NEURON_RT_VISIBLE_CORES", "NEURON_LOGICAL_NC_CONFIG", "NEURON_CC_FLAGS",
            "NEURON_PLATFORM_TARGET_OVERRIDE", "NEURON_LIBTORCH_CACHE_ROOT",
            "NEURON_LIBTORCH_DISABLE_COMPILE_CACHE")},
        "compiler_args": compiler_args(),
        "kernel_sources": {
            "before": {"file": str(args.baseline_module / "moe_fused_fp8.py"),
                       "md5": md5(args.baseline_module / "moe_fused_fp8.py")},
            "after": {"file": str(kernel_dir / "moe_fused_fp8.py"),
                      "md5": md5(kernel_dir / "moe_fused_fp8.py")},
        },
        "method": __doc__.split("Method.")[1].split("The compile cache")[0].strip(),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "cases": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for tokens in args.tokens:
        report["cases"].append(bucket_case(tokens, entries, args))
        report["elapsed_s"] = time.time() - started
        args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
