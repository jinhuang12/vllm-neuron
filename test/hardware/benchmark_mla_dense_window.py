# SPDX-License-Identifier: Apache-2.0
"""Time the dense-window MLA prefill kernel on Neuron, next to its entitlement.

Run it ONLY through the device lease. The lease pins the cores (``NEURON_RT_VISIBLE_CORES``)
and sets LNC2; this script refuses to run without the pinned cores and does not select
cores itself::

    export NEURON_LIBTORCH_CACHE_ROOT=$(mktemp -d)   # a new, empty compile cache
    python3 /home/ubuntu/glm53f-wt/devlease.py slice spare1 -- timeout 570 \\
        python test/hardware/benchmark_mla_dense_window.py --output dense_window.json

The script sets ``NEURON_PLATFORM_TARGET_OVERRIDE=trn2`` itself when it is unset.

Geometry: the per-rank TP=64 shape of GLM-5.3-Flash prefill on LNC2. One head, latent 512,
page 128, bf16 query and cache, a 16-page (2048-row) window, softmax scale 1/16. A case is
``tokens,active,ctx``: ``tokens`` operator rows of which the first ``active`` are queries at
positions ``ctx - active .. ctx - 1``; the window rows from ``ctx - active`` on are the
chunk's own (``written``). The default cases are the served full 1024-row chunk, a padded
1024-row chunk with 1000 queries (the last chunk of a prompt), and a 2048-row chunk.

Graphs, each compiled on its own; a sample is one dispatch with the output copied to CPU:

* ``dense``: ``mla_dense_window_attention`` on every row (a full chunk).
* ``sparse``: ``mla_sparse_attention`` on causal-prefix index rows, the call the dense path
  replaces (full chunks only; the indexer that makes the rows is not timed).
* ``helper``: ``attend_dense_window`` with ``active < tokens``, the padded chunk as the
  model calls it (the kernel writes the zero rows).
* ``dense_active``: the seam on the ``active`` rows only (inputs sliced on the host).
* ``floor``: a one-element add, the fixed cost of one execution.

The device time of a sample is the runtime system trace's ``nc_exec_running`` interval, the
two physical cores merged. Each case carries its ENTITLEMENT at the rates of
``glm53f-wt3/reports/entitlement.json`` (meta.hardware, per logical core): the op roofline
(causal FLOPs ``4 * latent * sum(seq_lens)`` at peak, or the bytes of the query, the
attended rows and the fp32 output at HBM bandwidth, whichever is longer) and the kernel's
own Tensor-engine instruction time (PE columns per 128-row tile, the tiles dealt over the
two programs, plus the window transposes).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

#: The worktree root, ahead of any installed copy (the lease sets no PYTHONPATH).
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

HEADS, LATENT, PAGE = 1, 512, 128
SOFTMAX_SCALE = 1.0 / 16
WINDOW_PAGES = 16
BANK_ROWS = 65536
#: The DSA selector's dials at GLM-5.3-Flash: index_topk 2048 tokens in pools of 4.
INDEX_TOPK, INDEX_KPOOL = 2048, 4
#: (tokens, active, ctx) per default case.
CASES = ((1024, 1024, 1024), (1024, 1000, 1000), (2048, 2048, 2048))
#: Every device output against the float64 oracle, as the CPU tests bound it.
ORACLE_ROW_REL = 1e-4
#: entitlement.json meta.hardware, per logical core (two physical cores at LNC2).
TE_PEAK_FLOPS = 79e12
HBM_BYTES_PER_S = 716e9
#: One PE column: a 128 x 128 x 2 FLOP step at the rate of one physical core.
PE_COLUMN_NS = 128 * 128 * 2 / (TE_PEAK_FLOPS / 2) * 1e9
#: The pinned-core marker the device lease sets.
LEASE_MARKER = "NEURON_RT_VISIBLE_CORES"


def make_case(tokens: int, active: int, ctx: int, seed: int):
    """``(q, bank, table, written, offset, seq_lens, window)``: the operands and their window.

    A padded chunk's rows from ``active`` on repeat the last real row's cache entry (the
    model's clamp) and carry ``seq_lens = ctx``; the kernel does not read them.
    """
    import torch

    gen = torch.Generator().manual_seed(seed)
    start = ctx - active
    q = (torch.randn(tokens, HEADS, LATENT, generator=gen) * 0.5).to(torch.bfloat16)
    bank = torch.randn(BANK_ROWS, LATENT, generator=gen).to(torch.bfloat16)
    table = torch.randperm(BANK_ROWS // PAGE, generator=gen)[:WINDOW_PAGES]
    table = table.reshape(WINDOW_PAGES, 1).to(torch.int32)
    written = torch.randn(tokens, LATENT, generator=gen).to(torch.bfloat16)
    written[active:] = written[active - 1]
    offset = torch.tensor([[start]], dtype=torch.int32)
    seq_lens = torch.arange(tokens, dtype=torch.int32) + start + 1
    seq_lens[active:] = ctx
    window = bank.reshape(-1, PAGE, LATENT)[table[:, 0].long()]
    window = window.reshape(WINDOW_PAGES * PAGE, LATENT).clone()
    window[start:start + tokens] = written
    return q, bank, table, written, offset, seq_lens, window


def prefix_indices(seq_lens, width: int):
    """``[tokens, width]`` int32 index rows naming each query's whole causal prefix, -1 after."""
    import torch

    columns = torch.arange(width, dtype=torch.int32).reshape(1, width)
    return torch.where(columns < seq_lens.reshape(-1, 1), columns, -1).to(torch.int32)


def oracle(q, window, seq_lens, rows: int):
    """Float64 causal attention of the first ``rows`` queries over ``window``."""
    import torch

    qd, cd = q[:rows].double(), window.double()
    out = torch.zeros(q.shape, dtype=torch.float64)
    for s in range(rows):
        keys = cd[:int(seq_lens[s])]
        weights = torch.softmax(torch.einsum("hl,kl->hk", qd[s], keys) * SOFTMAX_SCALE, dim=-1)
        out[s] = weights @ keys
    return out


def row_rel(got, want, rows: int) -> float:
    """Largest per-row ``||got - want|| / ||want||`` over the first ``rows`` query rows."""
    got = got[:rows].double().reshape(rows, -1)
    want = want[:rows].double().reshape(rows, -1)
    return float(((got - want).norm(dim=1) / want.norm(dim=1).clamp_min(1e-30)).max())


def entitlement(tokens: int, active: int, ctx: int, row_tile: int, latent_tile: int,
                key_chunk: int, programs: int) -> dict:
    """The case's op roofline and the kernel's Tensor-engine instruction time, in us."""
    window = WINDOW_PAGES * PAGE
    start = ctx - active
    pairs = sum(range(start + 1, ctx + 1))
    flops = 4 * LATENT * pairs
    nbytes = active * LATENT * 2 + ctx * LATENT * 2 + tokens * LATENT * 4
    # PE columns of one 128-row tile at this window. An fp32 operand streams at a quarter
    # of the bf16 rate, so an fp32 transpose of n columns costs 4n: the query transpose,
    # MM1 (one pass per latent tile over the window), the p transpose, and MM2's hi and
    # lo passes (one per key chunk over the latent).
    tile_columns = {
        "q_transpose_fp32": 4 * LATENT,
        "mm1": (LATENT // latent_tile) * window,
        "p_transpose_fp32": 4 * window,
        "mm2_hi_lo": 2 * (window // key_chunk) * LATENT,
    }
    per_tile = sum(tile_columns.values())
    window_columns = (window // key_chunk) * (LATENT // latent_tile) * row_tile
    jobs = -(-active // row_tile)
    per_program = -(-jobs // programs)
    return {
        "op_roofline_us": max(flops / TE_PEAK_FLOPS, nbytes / HBM_BYTES_PER_S) * 1e6,
        "op_flops": flops,
        "op_bytes": nbytes,
        "instruction_us": (per_program * per_tile + window_columns) * PE_COLUMN_NS / 1e3,
        "tile_pe_columns": tile_columns,
        "window_transpose_pe_columns": window_columns,
        "row_tiles": jobs,
        "row_tiles_per_program": per_program,
        "programs": programs,
    }


def device_intervals(events_json: str) -> list[float]:
    """Device us of each execution, in execution order, the physical-core intervals merged."""
    starts, intervals = {}, {}
    for event in json.loads(events_json)["events"]:
        if event["event_type"] != "nc_exec_running":
            continue
        key = (event["nc_idx"], event["tracking_id"])
        if event["phase"] == "start":
            starts[key] = event
        elif event["phase"] == "stop":
            begin = starts.pop(key)
            intervals.setdefault(begin["data"]["exec_id"], []).append(
                (begin["data"]["nc_timestamp_ns"], event["data"]["nc_timestamp_ns"]))
    return [(max(end for _, end in spans) - min(begin for begin, _ in spans)) / 1e3
            for _, spans in sorted(intervals.items())]


def summary(samples: list[float]) -> dict:
    """Median, p10, p90, min and max of ``samples``, in us."""
    ordered = sorted(samples)
    return {"n": len(ordered), "median_us": statistics.median(ordered),
            "p10_us": ordered[int(0.1 * len(ordered))],
            "p90_us": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
            "min_us": ordered[0], "max_us": ordered[-1]}


def time_graphs(graphs: dict, inputs: dict, warmup: int, iterations: int) -> dict:
    """Device and host time of every graph, the call order rotated each iteration."""
    from nrtpy._nrtpy import SystemTraceSession

    names = list(graphs)
    for _ in range(warmup):
        for name in names:
            graphs[name](*inputs[name]).to("cpu")
    order, host = [], {name: [] for name in names}
    with SystemTraceSession() as trace:
        for iteration in range(iterations):
            first = iteration % len(names)
            for name in names[first:] + names[:first]:
                started = time.perf_counter_ns()
                graphs[name](*inputs[name]).to("cpu")
                host[name].append((time.perf_counter_ns() - started) / 1e3)
                order.append(name)
        events = trace.fetch_events_json()
    device_all = device_intervals(events)
    if len(device_all) != len(order):
        raise AssertionError(f"the trace has {len(device_all)} executions for {len(order)} calls")
    device = {name: [] for name in names}
    for name, value in zip(order, device_all):
        device[name].append(value)
    return {name: {"device": summary(device[name]), "host": summary(host[name])}
            for name in names}


def run_case(tokens: int, active: int, ctx: int, args, width: int) -> dict:
    """Compile, check and time one case's graphs."""
    import torch

    from vllm_neuron.functional.attention import mla_dense_window as dense_window
    from vllm_neuron.functional.attention.mla_sparse import mla_sparse_attention
    from vllm_neuron.model.glm5_next.dsa_dense_window import attend_dense_window

    def compiled(fn):
        return torch.compile(fn, backend="neuron_libtorch", fullgraph=True, dynamic=False)

    def dense(q, bank, seq_lens, table, written, offset):
        return dense_window.mla_dense_window_attention(
            q, bank, seq_lens, SOFTMAX_SCALE, block_table_row=table, written=written,
            write_offset=offset, page_size=PAGE)

    def sparse(q, bank, indices, table, written, offset):
        return mla_sparse_attention(q, bank, indices, SOFTMAX_SCALE, block_table_row=table,
                                    written=written, write_offset=offset, page_size=PAGE)

    def helper(q, bank, seq_lens, table, written, offset):
        return attend_dense_window(q, bank, seq_lens, SOFTMAX_SCALE, block_table_row=table,
                                   written=written, write_offset=offset, page_size=PAGE,
                                   active_rows=active)

    def floor(x):
        return x + 1

    q, bank, table, written, offset, seq_lens, window = make_case(
        tokens, active, ctx, seed=tokens + 7 * active)

    def on_device(*tensors):
        return tuple(t.to("neuron:0") for t in tensors)

    graphs, inputs = {}, {}
    if active == tokens:
        graphs["dense"] = compiled(dense)
        inputs["dense"] = on_device(q, bank, seq_lens, table, written, offset)
        if args.sparse:
            graphs["sparse"] = compiled(sparse)
            inputs["sparse"] = on_device(q, bank, prefix_indices(seq_lens, width), table,
                                         written, offset)
    else:
        graphs["helper"] = compiled(helper)
        inputs["helper"] = on_device(q, bank, seq_lens, table, written, offset)
        graphs["dense_active"] = compiled(dense)
        inputs["dense_active"] = on_device(q[:active].clone(), bank, seq_lens[:active].clone(),
                                           table, written, offset)
    graphs["floor"] = compiled(floor)
    inputs["floor"] = on_device(torch.zeros(1, dtype=torch.int32))

    dense_window.reset_mla_dense_window_dispatch_counters()
    want = oracle(q, window, seq_lens, active)
    outputs, first_call_s, accuracy = {}, {}, {}
    for name, graph in graphs.items():
        started = time.perf_counter()
        outputs[name] = graph(*inputs[name]).to("cpu")
        first_call_s[name] = time.perf_counter() - started
        if name != "floor":
            accuracy[name] = {"row_rel_vs_float64": row_rel(outputs[name], want, active)}
    if "helper" in outputs:
        accuracy["helper"]["padding_rows_zero"] = bool((outputs["helper"][active:] == 0).all())
        accuracy["helper"]["prefix_max_abs_vs_dense_active"] = float(
            (outputs["helper"][:active] - outputs["dense_active"]).abs().max())
    if "sparse" in outputs:
        accuracy["dense"]["max_abs_vs_sparse"] = float(
            (outputs["dense"] - outputs["sparse"]).abs().max())
        accuracy["dense"]["row_rel_vs_sparse"] = row_rel(outputs["dense"], outputs["sparse"],
                                                         active)
    for name, one in accuracy.items():
        if one["row_rel_vs_float64"] > ORACLE_ROW_REL or not one.get("padding_rows_zero", True):
            raise AssertionError(f"t{tokens} a{active} ctx{ctx}: {name} {one}")

    rows = tokens * HEADS
    programs = 2 if rows > dense_window.ROW_TILE else 1
    budget = entitlement(tokens, active, ctx, dense_window.ROW_TILE, dense_window.LATENT_TILE,
                         dense_window.KEY_CHUNK, programs)
    repeats = [time_graphs(graphs, inputs, args.warmup, args.iterations)
               for _ in range(args.repeats)]
    ratios = {}
    for name in graphs:
        if name in ("dense", "helper", "dense_active"):
            median = statistics.median(r[name]["device"]["median_us"] for r in repeats)
            ratios[name] = {"median_us": median,
                            "over_instruction": median / budget["instruction_us"],
                            "over_op_roofline": median / budget["op_roofline_us"]}
    return {"case": f"t{tokens}_a{active}_ctx{ctx}", "tokens": tokens, "active_rows": active,
            "ctx": ctx, "entitlement": budget, "ratios": ratios, "first_call_s": first_call_s,
            "accuracy": accuracy, "repeats": repeats,
            "dense_route_counts": dense_window.mla_dense_window_route_counts()}


def parse_case(text: str) -> tuple[int, int, int]:
    """``tokens,active,ctx``: ``1 <= active <= tokens``, ``active <= ctx``, and the chunk's
    ``tokens`` rows, written from window row ``ctx - active`` on, inside the window."""
    try:
        tokens, active, ctx = (int(v) for v in text.split(","))
    except ValueError as err:
        raise argparse.ArgumentTypeError(f"expected tokens,active,ctx; got {text!r}") from err
    window = WINDOW_PAGES * PAGE
    if not (1 <= active <= tokens and active <= ctx and ctx - active + tokens <= window):
        raise argparse.ArgumentTypeError(
            f"need 1 <= active <= tokens, active <= ctx and ctx - active + tokens <= {window}; "
            f"got {text!r}")
    return tokens, active, ctx


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="JSON report path")
    parser.add_argument("--case", type=parse_case, action="append",
                        help="tokens,active,ctx (repeatable; default: the three served cases)")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20,
                        help="timed calls per graph in one repeat")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--no-sparse", dest="sparse", action="store_false",
                        help="do not compile and time the sparse call on full chunks")
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1 or args.repeats < 1:
        parser.error("use warmup >= 0, iterations >= 1 and repeats >= 1")

    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1" or os.environ.get("NKI_SIMULATOR"):
        raise SystemExit("hardware benchmark: unset VLLM_NEURON_CPU_MODE and NKI_SIMULATOR")
    if not os.environ.get(LEASE_MARKER):
        raise SystemExit(f"run under devlease.py slice <name>, which sets {LEASE_MARKER}")
    if os.environ.get("NEURON_LOGICAL_NC_CONFIG") != "2":
        raise SystemExit("the served geometry is LNC2: NEURON_LOGICAL_NC_CONFIG must be 2")
    cache_root = os.environ.get("NEURON_LIBTORCH_CACHE_ROOT")
    if not cache_root:
        raise SystemExit("set NEURON_LIBTORCH_CACHE_ROOT to a new, empty directory")
    os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

    import vllm_neuron  # noqa: F401 -- registers the Neuron compilation backend
    from vllm_neuron.functional.attention import mla_dense_window as dense_window
    from vllm_neuron.functional.dsa.decode_bypass import bypass_max_context

    if not vllm_neuron.__file__.startswith(str(ROOT) + "/"):
        raise SystemExit(f"imported vllm_neuron from {vllm_neuron.__file__}, not {ROOT}")
    bound = bypass_max_context(INDEX_TOPK, INDEX_KPOOL)
    width = -(-bound // dense_window.ROW_TILE) * dense_window.ROW_TILE
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # The compiler writes its logs to the working directory: keep them with the cache.
    scratch = Path(cache_root) / "benchmark_cwd"
    scratch.mkdir(parents=True, exist_ok=True)
    os.chdir(scratch)

    sources = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in (
        "vllm_neuron/functional/attention/mla_dense_window.py",
        "vllm_neuron/functional/attention/mla_sparse.py",
        "vllm_neuron/model/glm5_next/dsa_dense_window.py")}
    report = {
        "environment": {key: os.environ.get(key) for key in (
            LEASE_MARKER, "NEURON_LOGICAL_NC_CONFIG", "NEURON_PLATFORM_TARGET_OVERRIDE",
            "NEURON_LIBTORCH_CACHE_ROOT", "NEURON_CC_FLAGS")},
        "tree": str(ROOT),
        "source_sha256": sources,
        "dense_source_digest": dense_window.SOURCE_DIGEST,
        "shape": {"heads": HEADS, "latent": LATENT, "page": PAGE,
                  "softmax_scale": SOFTMAX_SCALE, "bank_rows": BANK_ROWS,
                  "window_rows": WINDOW_PAGES * PAGE, "sparse_index_width": width,
                  "dtype": "bfloat16"},
        "rates": {"te_peak_flops": TE_PEAK_FLOPS, "hbm_bytes_per_s": HBM_BYTES_PER_S,
                  "pe_column_ns": PE_COLUMN_NS,
                  "source": "glm53f-wt3/reports/entitlement.json meta.hardware"},
        "timing": {"warmup": args.warmup, "iterations": args.iterations,
                   "repeats": args.repeats,
                   "device": "system trace nc_exec_running, physical cores merged",
                   "synchronization": "graph output copied to CPU on every call"},
        "cases": [],
    }
    for tokens, active, ctx in args.case or CASES:
        row = run_case(tokens, active, ctx, args, width)
        report["cases"].append(row)
        output.write_text(json.dumps(report, indent=1) + "\n")
        print(json.dumps({"case": row["case"],
                          "device_median_us": {
                              name: [round(r[name]["device"]["median_us"], 1)
                                     for r in row["repeats"]]
                              for name in row["repeats"][0]},
                          "ratios": row["ratios"], "accuracy": row["accuracy"]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
