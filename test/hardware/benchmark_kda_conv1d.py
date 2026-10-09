# SPDX-License-Identifier: Apache-2.0
"""KDA prefill depthwise conv1d on Neuron: nkilib's kernel against this tree's.

Set the Neuron cores before launching (``devlease.py slice`` does, with
``NEURON_LOGICAL_NC_CONFIG=2``). This script does not select cores. References are
computed on the CPU; compilation and warmup are excluded from the timings.

At each served prefill shape, ``[1, C, 1, Q + S - 1] -> [1, C, 1, Q]`` in float32
with ``C = 384`` and ``S = 4`` (``--columns``, default 1024 and 2048), five variants
are timed, interleaved in one process:

* ``before``: ``nkilib``'s ``depthwise_conv1d_implicit_gemm`` launched as the
  replaced wrap launched it at this shape (LNC2 grid, ``feature_group_count = C``).
* ``after``: this tree's public ``depthwise_conv1d``.
* ``after_aa``: the same call compiled into separate graphs, an A/A pair whose
  difference from ``after`` is the noise floor.
* ``floor_copy``: the kernel's own loads and stores with no compute, so the
  DMA-only floor of its tiling.
* ``floor_launch``: one row in and out per program, the fixed cost of an NKI
  kernel inside a graph.

Per-call device time is the slope ``(T_L - T_1) / (L - 1)`` between an
``L``-call graph and a one-call graph (``--links``), paired per iteration, from the
runtime system trace (LNC2 physical-core intervals merged). Two ``L``-call graphs
per variant: a dependency chain, each call consuming the previous output, which
serializes the calls as the model's data flow does; and ``L`` independent calls,
which the device may overlap. A chain link shortens the sequence by ``S - 1``, so
links ``2 .. L`` run ``Q - 3, Q - 6, ...`` columns, ``1 - 3L/(2Q)`` of the served
work on average; the report states the bias. Each graph runs ``--iterations``
timed executions, rotated across graphs.

Numerics run first, on the one-call graphs at ``--seeds``: device ``after`` and
``before`` against the CPU torch reference, against the per-element float32
accumulation bound the unit tests use, and against each other. The tensors are
saved under ``--pt-dir``.

The compile cache is off unless ``--use-compile-cache``: the graph cache key
ignores kernel bodies, so an edited kernel could otherwise be timed from a stale
NEFF.
"""

from __future__ import annotations

import argparse
import json
import os
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
from nkilib.experimental.conv.depthwise_conv1d import (  # noqa: E402
    depthwise_conv1d_implicit_gemm,
)

from vllm_neuron.functional.kda import depthwise_conv1d as conv  # noqa: E402
from vllm_neuron.functional.kda import depthwise_conv1d_kernel as kernel  # noqa: E402

DEVICE = "neuron:0"
#: The served KDA conv: q, k and v channels of one rank stacked, four taps.
CHANNELS = 384
TAPS = 4
PARTITIONS = kernel.PARTITION_MAX


def _div_ceil(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


@nki.jit
def copy_floor_kernel(img, filt, col_cap):
    """The kernel's loads and stores at its own tiling, no compute: ``out = img[..., :Q]``.

    ``col_cap`` is the kernel's column tile cap at this tap count and unit stride.
    """
    n_batch, channels, _, width = img.shape
    taps = filt.shape[3]
    columns = width - taps + 1
    out = nl.ndarray((n_batch, channels, 1, columns), dtype=img.dtype, buffer=nl.shared_hbm)
    img_rows = img.reshape((n_batch * channels, width))
    out_rows = out.reshape((n_batch * channels, columns))
    share = _div_ceil(columns, nl.num_programs(axes=0))
    first = nl.program_id(axis=0) * share
    owned = min(share, columns - first)
    n_col = _div_ceil(owned, col_cap)
    span = _div_ceil(owned, n_col)
    for col in range(n_col):
        c0 = first + col * span
        cn = min(span, first + owned - c0)
        for tile in nl.affine_range(n_batch * channels // PARTITIONS):
            row0 = tile * PARTITIONS
            x = nl.ndarray((PARTITIONS, cn + taps - 1), dtype=img.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=x, src=img_rows[row0:row0 + PARTITIONS, c0:c0 + cn + taps - 1],
                          dge_mode=nisa.dge_mode.none)
            nisa.dma_copy(dst=out_rows[row0:row0 + PARTITIONS, c0:c0 + cn],
                          src=x[0:PARTITIONS, 0:cn], dge_mode=nisa.dge_mode.none)
    return out


@nki.jit
def launch_floor_kernel(img, filt):
    """An NKI kernel's fixed cost: one row of the program's columns in and out."""
    n_batch, channels, _, width = img.shape
    columns = width - filt.shape[3] + 1
    out = nl.ndarray((n_batch, channels, 1, columns), dtype=img.dtype, buffer=nl.shared_hbm)
    share = _div_ceil(columns, nl.num_programs(axes=0))
    first = nl.program_id(axis=0) * share
    owned = min(share, columns - first)
    x = nl.ndarray((1, owned), dtype=img.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=x, src=img.reshape((n_batch * channels, width))[0:1, first:first + owned])
    nisa.dma_copy(dst=out.reshape((n_batch * channels, columns))[0:1, first:first + owned], src=x)
    return out


def before_call(img, filt):
    """The replaced wrap's launch at the served shape."""
    return wrap_nki(depthwise_conv1d_implicit_gemm)[conv.LNC_SHARDS](
        img_ref=img, filter_ref=filt, padding=conv.NO_PADDING, stride=conv.UNIT_STRIDE,
        rhs_dilation=conv.UNIT_DILATION, lhs_dilation=conv.UNIT_DILATION,
        feature_group_count=int(img.shape[1]), batch_group_count=1)


def after_call(img, filt):
    return conv.depthwise_conv1d(img, filt)


def after_aa_call(img, filt):
    """``after_call`` again, a distinct function so its graphs compile on their own."""
    return conv.depthwise_conv1d(img, filt)


def copy_floor_call(img, filt):
    col_cap = kernel.column_tile_cap(int(filt.shape[3]), 1)
    return wrap_nki(copy_floor_kernel)[conv.LNC_SHARDS](img, filt, col_cap)


def launch_floor_call(img, filt):
    return wrap_nki(launch_floor_kernel)[conv.LNC_SHARDS](img, filt)


VARIANTS = {"before": before_call, "after": after_call, "after_aa": after_aa_call,
            "floor_copy": copy_floor_call, "floor_launch": launch_floor_call}


def compile_fn(fn):
    return torch.compile(fn, backend="neuron_libtorch", fullgraph=True, dynamic=False)


def chain_graph(call, links: int):
    def chain(filt, img):
        x = img
        for _ in range(links):
            x = call(x, filt)
        return x
    return compile_fn(chain)


def independent_graph(call, links: int):
    def independent(filt, *imgs):
        outs = []
        for img in imgs:
            outs.append(call(img, filt))
        return tuple(outs)
    return compile_fn(independent)


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


def _sync(result):
    (result[0] if isinstance(result, tuple) else result).to("cpu")


def time_graphs(graphs: dict, inputs: dict, warmup: int, iterations: int) -> dict:
    """Every graph's device samples in us, the calls rotated across graphs per iteration."""
    from nrtpy._nrtpy import SystemTraceSession

    names = list(graphs)
    for _ in range(warmup):
        for name in names:
            _sync(graphs[name](*inputs[name]))
    order = []
    with SystemTraceSession() as trace:
        for iteration in range(iterations):
            shift = iteration % len(names)
            for name in names[shift:] + names[:shift]:
                _sync(graphs[name](*inputs[name]))
                order.append(name)
        events_json = trace.fetch_events_json()
    device_all = device_intervals(events_json)
    if len(device_all) != len(order):
        raise AssertionError(f"system trace has {len(device_all)} executions for {len(order)} calls")
    device = {name: [] for name in names}
    for name, value in zip(order, device_all):
        device[name].append(value)
    return device


def direct_sum(img: torch.Tensor, filt: torch.Tensor):
    """``(exact, magnitude)`` in float64 from the definition, unpadded and unit-stride."""
    x = img.double()[:, :, 0, :]
    w = filt.double()[:, 0, 0, :]
    columns = x.shape[-1] - w.shape[-1] + 1
    exact = torch.zeros(x.shape[0], x.shape[1], columns, dtype=torch.float64)
    magnitude = torch.zeros_like(exact)
    for s in range(w.shape[-1]):
        term = w[None, :, s:s + 1] * x[:, :, s:s + columns]
        exact += term
        magnitude += term.abs()
    return exact[:, :, None, :], magnitude[:, :, None, :]


def compare(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    residual = (actual.double() - expected.double()).abs()
    scale = expected.double().abs().max().item()
    return {"max_abs": residual.max().item(), "max_rel": residual.max().item() / scale,
            "max_abs_reference": scale}


def numerics(columns: int, seeds, graphs: dict, pt_dir: Path) -> list[dict]:
    """Device ``after`` and ``before`` at each seed against the CPU reference and the bound."""
    rows = []
    for seed in seeds:
        generator = torch.Generator().manual_seed(seed)
        img = torch.randn(1, CHANNELS, 1, columns + TAPS - 1, generator=generator)
        filt = torch.randn(CHANNELS, 1, 1, TAPS, generator=generator)
        dfilt, dimg = filt.to(DEVICE), img.to(DEVICE)
        after = graphs["after"](dfilt, dimg).to("cpu")
        before = graphs["before"](dfilt, dimg).to("cpu")
        reference = conv.depthwise_conv1d_torch_reference(img, filt)
        exact, magnitude = direct_sum(img, filt)
        bound = TAPS * torch.finfo(torch.float32).eps * magnitude
        path = pt_dir / f"numerics_q{columns}_seed{seed}.pt"
        torch.save({"seed": seed, "img": img, "filt": filt, "after": after, "before": before,
                    "reference": reference}, path)
        rows.append({
            "seed": seed, "pt": str(path),
            "after_vs_reference": compare(after, reference),
            "before_vs_reference": compare(before, reference),
            "after_vs_before": compare(after, before),
            "after_worst_bound_ratio": ((after.double() - exact).abs() / bound).max().item(),
            "before_worst_bound_ratio": ((before.double() - exact).abs() / bound).max().item(),
            "reference_worst_bound_ratio": ((reference.double() - exact).abs() / bound).max().item(),
        })
        print(json.dumps({"columns": columns, **rows[-1]}), flush=True)
    return rows


def shape_case(columns: int, args) -> dict:
    links = args.links
    generator = torch.Generator().manual_seed(columns)
    width = columns + TAPS - 1
    filt = torch.randn(CHANNELS, 1, 1, TAPS, generator=generator).to(DEVICE)
    imgs = [torch.randn(1, CHANNELS, 1, width, generator=generator).to(DEVICE)
            for _ in range(links)]
    graphs, inputs = {}, {}
    for name, call in VARIANTS.items():
        graphs[f"{name}|1"] = chain_graph(call, 1)
        inputs[f"{name}|1"] = (filt, imgs[0])
        graphs[f"{name}|chain"] = chain_graph(call, links)
        inputs[f"{name}|chain"] = (filt, imgs[0])
        graphs[f"{name}|independent"] = independent_graph(call, links)
        inputs[f"{name}|independent"] = (filt, *imgs)
    for key, graph in graphs.items():
        started = time.time()
        _sync(graph(*inputs[key]))
        print(f"compiled {key} q={columns} in {time.time() - started:.1f}s", flush=True)
    one_call = {name: graphs[f"{name}|1"] for name in ("after", "before")}
    rows = numerics(columns, args.seeds, one_call, args.pt_dir)
    device = time_graphs(graphs, inputs, args.warmup, args.iterations)
    variants = {}
    for name in VARIANTS:
        one = device[f"{name}|1"]
        slopes = {}
        for mode in ("chain", "independent"):
            many = device[f"{name}|{mode}"]
            slopes[mode] = [(m - o) / (links - 1) for m, o in zip(many, one)]
        variants[name] = {
            "one_call_graph": stats_us(one),
            "per_call_chain": stats_us(slopes["chain"]),
            "per_call_independent": stats_us(slopes["independent"]),
            "samples": {"one": one, "chain": device[f"{name}|chain"],
                        "independent": device[f"{name}|independent"]},
        }
    aa = [abs(a - b) for a, b in zip(
        [(m - o) / (links - 1) for m, o in zip(device["after|chain"], device["after|1"])],
        [(m - o) / (links - 1) for m, o in zip(device["after_aa|chain"], device["after_aa|1"])])]
    return {
        "columns": columns, "img": [1, CHANNELS, 1, width], "filt": [CHANNELS, 1, 1, TAPS],
        "dtype": "float32", "links": links,
        "chain_work_fraction": 1 - (TAPS - 1) * links / (2 * columns),
        "variants": variants,
        "noise_floor_chain_aa_abs_diff": stats_us(aa),
        "numerics": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pt-dir", type=Path, help="numerics tensors (default: --out's directory)")
    parser.add_argument("--columns", type=int, nargs="+", default=[1024, 2048])
    parser.add_argument("--links", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--use-compile-cache", action="store_true",
                        help="reuse cached NEFFs (off by default; see the module docstring)")
    parser.add_argument("--time-limit", type=int, default=900,
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
    if args.links < 2 or args.iterations < 5 or len(args.seeds) < 1:
        raise ValueError("Use --links >= 2, --iterations >= 5 and at least one seed")
    if CHANNELS % kernel.PARTITION_MAX:
        raise ValueError("copy_floor_kernel tiles whole channel tiles only")
    if not Path(conv.__file__).resolve().is_relative_to(REPO_ROOT):
        raise RuntimeError(f"{conv.__name__} imported from {conv.__file__}")
    args.out = args.out.resolve()
    args.pt_dir = (args.pt_dir or args.out.parent).resolve()
    args.pt_dir.mkdir(parents=True, exist_ok=True)
    torch._dynamo.config.cache_size_limit = 64
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
        "after_module": conv.__file__,
        "kernel_identity": list(conv.kernel_identity()),
        "before_kernel": [depthwise_conv1d_implicit_gemm.func.__module__,
                          depthwise_conv1d_implicit_gemm.func.__qualname__],
        "method": ("per-call = (L-call graph - 1-call graph) / (L - 1), paired per iteration; "
                   "device time from the runtime system trace (LNC2 physical-core intervals "
                   "merged); chain = each call consumes the previous output; independent = "
                   "L calls on distinct inputs"),
        "iterations": args.iterations, "warmup": args.warmup, "cases": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for columns in args.columns:
        case = shape_case(columns, args)
        report["cases"].append(case)
        args.out.write_text(json.dumps(report, indent=1) + "\n")
        print(json.dumps({
            "columns": columns,
            **{name: round(case["variants"][name]["per_call_chain"]["median_us"], 2)
               for name in VARIANTS},
            "noise_floor_us": round(case["noise_floor_chain_aa_abs_diff"]["median_us"], 2),
        }), flush=True)


if __name__ == "__main__":
    main()
