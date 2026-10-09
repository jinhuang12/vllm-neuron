# SPDX-License-Identifier: Apache-2.0
"""mHC Sinkhorn on Neuron: the batched kernel against a baseline snapshot of it.

Set the Neuron cores before launching (``devlease.py slice`` does, with
``NEURON_LOGICAL_NC_CONFIG=2``). This script does not select cores. References are
computed on the CPU; compilation and warmup are excluded from the timings.

At each token count ``T`` (``--tokens``; default the served decode batches 1-64 and
the served prefill chunks 128-2048) the call is
``sinkhorn_normalise_blocks(f32[T, S, S]) -> f32[T, S, S]`` with ``S = MHC_STREAMS``
and :data:`~vllm_neuron.functional.mhc.sinkhorn.SINKHORN_ITERS` iterations, the mHC
layer's own call. These variants are timed, interleaved in one process:

* ``before``: ``sinkhorn_normalise_blocks`` of the snapshot in ``--baseline-module``.
* ``after``: this tree's ``sinkhorn_normalise_blocks``.
* ``after_aa``: the same call compiled into separate graphs, an A/A pair whose
  difference from ``after`` is the noise floor.
* ``after_iters1``: this tree's kernel at one iteration, so
  ``(after - after_iters1) / (SINKHORN_ITERS - 1)`` is the cost of one iteration.
* ``floor_copy``: the blocks loaded into SBUF and stored back with no compute, in
  the tokens-on-partitions layout, so the DMA-only floor of one launch.
* ``floor_launch``: one block in and out, the fixed cost of an NKI kernel inside a
  graph.
* every entry of ``PROBES`` in ``--probe-module``, a design study's candidates:
  ``PROBES = {name: callable(blocks) -> blocks}``, or ``(callable, normalising)``
  for a probe whose output is not the normalisation (its numerics are skipped).

Per-call device time is the slope ``(T_L - T_1) / (L - 1)`` between an
``L``-call graph and a one-call graph (``--chain``), paired per iteration, from the
runtime system trace (LNC2 physical-core intervals merged). A graph execution
carries a fixed offset of about 10 us that some executions skip, so one pair can
be off by that much; over ``L - 1 = 31`` calls it is about 0.3 us. Each call consumes the
previous call's output, which serializes the calls as the layer's data flow does
(one site's Sinkhorn feeds its combine, whose streams feed the next site's
projection); a doubly stochastic block is a valid positive affinity, so the chain
stays on the kernel's admitted inputs. Each graph runs ``--iterations`` timed
executions, rotated across graphs.

Numerics run first, on the one-call graphs at ``--seeds``: every variant that
computes the normalisation against ``before`` bit for bit (max abs difference and
the count of differing elements) and against a float64 CPU oracle, and each such
graph is run ``--emissions`` times on the same input to show its output does not
change from run to run. The tensors are saved under ``--pt-dir``.

The compile cache is off unless ``--use-compile-cache``: the graph cache key
ignores kernel bodies, so an edited kernel could otherwise be timed from a stale
NEFF. The baseline snapshot is imported under a module name that carries its
directory name, so it never shares a cache key with this tree's kernel.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import signal
import statistics
import sys
import time
from pathlib import Path

#: This file's repository. Run as a script, Python puts ``test/hardware`` on
#: ``sys.path`` and the venv resolves ``vllm_neuron`` to another checkout; the
#: "after" side must be this tree.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

import vllm_neuron  # noqa: E402,F401 -- registers the Neuron compilation backend
import nki  # noqa: E402
import nki.isa as nisa  # noqa: E402
import nki.language as nl  # noqa: E402
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki  # noqa: E402

from vllm_neuron.functional.mhc import sinkhorn  # noqa: E402

DEVICE = "neuron:0"
STREAMS = sinkhorn.MHC_STREAMS
ITERS = sinkhorn.SINKHORN_ITERS
PARTITIONS = sinkhorn.PARTITION_MAX
#: The mHC layer's epsilon after the comb softmax (``text_config.hc_eps``).
HC_EPS = 1e-6
#: Served decode batches (``num_seqs_buckets`` of the bs=64 line) and prefill chunks.
SERVED_TOKENS = (1, 2, 4, 8, 16, 32, 64, 128, 512, 1024, 2048)


def load_baseline(directory: Path):
    """Import ``sinkhorn.py`` from a snapshot directory, under a name of its own."""
    tag = re.sub(r"\W", "_", directory.resolve().name)
    name = f"_sinkhorn_baseline_{tag}"
    spec = importlib.util.spec_from_file_location(name, directory / "sinkhorn.py")
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load baseline snapshot {directory / 'sinkhorn.py'}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_probes(path: Path | None) -> dict:
    """``PROBES`` of a design-study module, or nothing."""
    if path is None:
        return {}
    spec = importlib.util.spec_from_file_location(f"_sinkhorn_probes_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load probe module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return dict(module.PROBES)


def _tokens_on_partitions(tokens: int) -> tuple[int, int, int]:
    """``(per_part, full_parts, tail)`` of the tokens-on-partitions layout."""
    per_part = (tokens + PARTITIONS - 1) // PARTITIONS
    full_parts = tokens // per_part
    return per_part, full_parts, tokens - full_parts * per_part


@nki.jit
def copy_floor_kernel(blocks):
    """The tokens-on-partitions load and store, no compute: ``out = blocks``."""
    tokens, rows, cols = blocks.shape
    per_part, full_parts, tail = _tokens_on_partitions(tokens)
    run = per_part * rows * cols
    out = nl.ndarray((tokens, rows, cols), dtype=nl.float32, buffer=nl.shared_hbm)
    flat_in = blocks.reshape((tokens * rows * cols,))
    flat_out = out.reshape((tokens * rows * cols,))
    parts = full_parts
    if tail > 0:
        parts = full_parts + 1
    work = nl.ndarray((parts, run), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=work[0:full_parts, 0:run],
                  src=flat_in.ap(pattern=[[run, full_parts], [1, run]], offset=0))
    if tail > 0:
        nisa.dma_copy(dst=work[full_parts:full_parts + 1, 0:tail * rows * cols],
                      src=flat_in.ap(pattern=[[run, 1], [1, tail * rows * cols]],
                                     offset=full_parts * run))
    nisa.dma_copy(dst=flat_out.ap(pattern=[[run, full_parts], [1, run]], offset=0),
                  src=work[0:full_parts, 0:run])
    if tail > 0:
        nisa.dma_copy(dst=flat_out.ap(pattern=[[run, 1], [1, tail * rows * cols]],
                                      offset=full_parts * run),
                      src=work[full_parts:full_parts + 1, 0:tail * rows * cols])
    return out


@nki.jit
def launch_floor_kernel(blocks):
    """An NKI kernel's fixed cost: one block in and out."""
    tokens, rows, cols = blocks.shape
    out = nl.ndarray((tokens, rows, cols), dtype=nl.float32, buffer=nl.shared_hbm)
    one = nl.ndarray((1, rows * cols), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=one, src=blocks.reshape((tokens, rows * cols))[0:1, 0:rows * cols])
    nisa.dma_copy(dst=out.reshape((tokens, rows * cols))[0:1, 0:rows * cols], src=one)
    return out


def variants(baseline) -> dict:
    """Every timed variant: ``name -> (call, computes the normalisation)``."""

    def before(blocks):
        return baseline.sinkhorn_normalise_blocks(blocks, iters=ITERS)

    def after(blocks):
        return sinkhorn.sinkhorn_normalise_blocks(blocks, iters=ITERS)

    def after_aa(blocks):
        return sinkhorn.sinkhorn_normalise_blocks(blocks, iters=ITERS)

    def after_iters1(blocks):
        return sinkhorn.sinkhorn_normalise_blocks(blocks, iters=1)

    def floor_copy(blocks):
        return wrap_nki(copy_floor_kernel)(blocks)

    def floor_launch(blocks):
        return wrap_nki(launch_floor_kernel)(blocks)

    return {"before": (before, True), "after": (after, True), "after_aa": (after_aa, True),
            "after_iters1": (after_iters1, False), "floor_copy": (floor_copy, False),
            "floor_launch": (floor_launch, False)}


def compile_fn(fn):
    return torch.compile(fn, backend="neuron_libtorch", fullgraph=True, dynamic=False)


def chain_graph(call, links: int):
    def chain(blocks):
        for _ in range(links):
            blocks = call(blocks)
        return blocks
    return compile_fn(chain)


def stats_us(samples: list[float]) -> dict:
    ordered = sorted(samples)

    def quantile(q):
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

    return {"n": len(ordered), "median_us": statistics.median(ordered), "min_us": ordered[0],
            "p10_us": quantile(0.1), "p90_us": quantile(0.9), "max_us": ordered[-1]}


def device_intervals(events_json: str) -> list[float]:
    """Per-execution device time in us, in execution order, LNC2 core intervals merged."""
    starts = {}
    intervals: dict[int, list[tuple[int, int]]] = {}
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
    return [(max(end for _, end in intervals[e]) - min(b for b, _ in intervals[e])) / 1000.0
            for e in sorted(intervals)]


def time_graphs(graphs: dict, inputs: dict, warmup: int, iterations: int) -> dict:
    """Every graph's device samples in us, the calls rotated across graphs per iteration."""
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


def site_blocks(tokens: int, seed: int) -> torch.Tensor:
    """The layer's own input distribution: ``softmax(logits) + hc_eps`` per block."""
    generator = torch.Generator().manual_seed(seed)
    logits = 2.0 * torch.randn((tokens, STREAMS, STREAMS), generator=generator)
    return torch.softmax(logits, dim=-1) + HC_EPS


def oracle(blocks: torch.Tensor) -> torch.Tensor:
    """Per-block Sinkhorn in float64: the kernel's row-then-column schedule."""
    work = blocks.to(torch.float64) + sinkhorn.SINKHORN_DENOM_EPS
    for _ in range(ITERS):
        work = work / work.sum(dim=-1, keepdim=True)
        work = work / work.sum(dim=-2, keepdim=True)
    return work


def numerics(tokens: int, seeds, graphs: dict, emissions: int, pt_dir: Path) -> list[dict]:
    """Each normalising variant against ``before`` bit for bit, and against the oracle.

    Every graph runs ``emissions`` times on one device copy of the input; the first
    output is the one compared, and the rest must equal it bit for bit.
    """
    rows = []
    for seed in seeds:
        blocks = site_blocks(tokens, seed)
        on_device = blocks.to(DEVICE)
        runs = {name: torch.stack([graph(on_device).to("cpu") for _ in range(emissions)])
                for name, graph in graphs.items()}
        outputs = {name: stacked[0] for name, stacked in runs.items()}
        reference = oracle(blocks)
        path = pt_dir / f"numerics_t{tokens}_seed{seed}.pt"
        torch.save({"seed": seed, "blocks": blocks, **outputs,
                    "emissions": runs}, path)
        row = {"seed": seed, "pt": str(path)}
        for name, out in outputs.items():
            differ = out != outputs["before"]
            row[name] = {
                "bit_equal_to_before": not bool(differ.any()),
                "elements_differing_from_before": int(differ.sum()),
                "max_abs_vs_before": float((out - outputs["before"]).abs().max()),
                "max_abs_vs_oracle": float((out.double() - reference).abs().max()),
                "emissions": emissions,
                "emissions_bit_identical": bool((runs[name] == out).all()),
            }
        rows.append(row)
        print(json.dumps({"tokens": tokens, **row}), flush=True)
    return rows


def token_case(tokens: int, calls: dict, args) -> dict:
    links = args.chain
    blocks = site_blocks(tokens, 1000 + tokens).to(DEVICE)
    graphs, inputs = {}, {}
    for name, (call, _) in calls.items():
        graphs[f"{name}|1"] = chain_graph(call, 1)
        graphs[f"{name}|chain"] = chain_graph(call, links)
        inputs[f"{name}|1"] = inputs[f"{name}|chain"] = (blocks,)
    for key, graph in graphs.items():
        started = time.time()
        graph(*inputs[key]).to("cpu")
        print(f"compiled {key} T={tokens} in {time.time() - started:.1f}s", flush=True)
    normalising = {name: graphs[f"{name}|1"] for name, (_, norm) in calls.items() if norm}
    rows = numerics(tokens, args.seeds, normalising, args.emissions, args.pt_dir)
    device = time_graphs(graphs, inputs, args.warmup, args.iterations)
    results = {}
    slopes = {}
    for name in calls:
        one, many = device[f"{name}|1"], device[f"{name}|chain"]
        slopes[name] = [(m - o) / (links - 1) for m, o in zip(many, one)]
        results[name] = {
            "one_call_graph": stats_us(one),
            "per_call_chain": stats_us(slopes[name]),
            "samples": {"one": one, "chain": many},
        }
    aa = [abs(a - b) for a, b in zip(slopes["after"], slopes["after_aa"])]
    per_iteration = [(a - b) / (ITERS - 1) for a, b in zip(slopes["after"], slopes["after_iters1"])]
    return {
        "tokens": tokens, "blocks": [tokens, STREAMS, STREAMS], "dtype": "float32",
        "iters": ITERS, "links": links, "variants": results,
        "noise_floor_chain_aa_abs_diff": stats_us(aa),
        "per_iteration_after": stats_us(per_iteration),
        "numerics": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-module", type=Path, required=True,
                        help="directory holding the baseline snapshot's sinkhorn.py")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pt-dir", type=Path, help="numerics tensors (default: --out's directory)")
    parser.add_argument("--probe-module", type=Path,
                        help="a design study's module defining PROBES = {name: callable}")
    parser.add_argument("--tokens", type=int, nargs="+", default=list(SERVED_TOKENS))
    parser.add_argument("--chain", type=int, default=32)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--emissions", type=int, default=8,
                        help="runs of each one-call graph per seed, compared bit for bit")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
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
        raise ValueError("Pin NEURON_RT_VISIBLE_CORES before running the benchmark")
    if os.environ.get("NEURON_LOGICAL_NC_CONFIG") != "2":
        raise ValueError("The served configuration is NEURON_LOGICAL_NC_CONFIG=2")
    if args.chain < 2 or args.iterations < 5 or args.emissions < 2 or not args.seeds:
        raise ValueError("Use --chain >= 2, --iterations >= 5, --emissions >= 2 and a seed")
    if any(t < 1 for t in args.tokens):
        raise ValueError("Token counts must be positive")
    if not Path(sinkhorn.__file__).resolve().is_relative_to(REPO_ROOT):
        raise RuntimeError(f"{sinkhorn.__name__} imported from {sinkhorn.__file__}")
    baseline = load_baseline(args.baseline_module)
    calls = variants(baseline)
    for name, probe in load_probes(args.probe_module).items():
        calls[name] = probe if isinstance(probe, tuple) else (probe, True)
    args.out = args.out.resolve()
    args.pt_dir = (args.pt_dir or args.out.parent).resolve()
    args.pt_dir.mkdir(parents=True, exist_ok=True)
    torch._dynamo.config.cache_size_limit = 256
    # The NKI and neuronx-cc drivers write artifacts into the working directory.
    scratch = Path(os.environ.get("NEURON_LIBTORCH_CACHE_ROOT") or "/tmp") / "bench-cwd"
    scratch.mkdir(parents=True, exist_ok=True)
    os.chdir(scratch)
    report = {
        "environment": {key: os.environ.get(key) for key in (
            "NEURON_RT_VISIBLE_CORES", "NEURON_LOGICAL_NC_CONFIG",
            "NEURON_PLATFORM_TARGET_OVERRIDE", "NEURON_LIBTORCH_CACHE_ROOT",
            "NEURON_LIBTORCH_DISABLE_COMPILE_CACHE")},
        "device": "neuron:0 = one logical core (LNC2: 2 physical cores) of the slice",
        "after_module": sinkhorn.__file__,
        "baseline_module": baseline.__file__,
        "kernel_identity": list(sinkhorn.blocks_kernel_identity()),
        "probes": sorted(set(calls) - set(variants(baseline))),
        "method": ("per-call = (L-call chain graph - 1-call graph) / (L - 1), paired per "
                   "iteration; device time from the runtime system trace (LNC2 physical-core "
                   "intervals merged); each call consumes the previous output"),
        "iterations": args.iterations, "warmup": args.warmup, "cases": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for tokens in args.tokens:
        torch._dynamo.reset()
        case = token_case(tokens, calls, args)
        report["cases"].append(case)
        args.out.write_text(json.dumps(report, indent=1) + "\n")
        print(json.dumps({
            "tokens": tokens,
            **{name: round(case["variants"][name]["per_call_chain"]["median_us"], 2)
               for name in calls},
            "noise_floor_us": round(case["noise_floor_chain_aa_abs_diff"]["median_us"], 2),
        }), flush=True)


if __name__ == "__main__":
    main()
