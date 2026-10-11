# SPDX-License-Identifier: Apache-2.0
"""Cost of the sparse MLA kernel's gather at one row per offset and at one 4-row pool per offset.

Run it ONLY through the device lease, which pins the cores and sets LNC2; the script refuses
to run without them and selects no cores itself::

    export NEURON_LIBTORCH_CACHE_ROOT=<an empty directory>
    python3 /home/ubuntu/glm53f-wt/devlease.py slice <slice> -- python3 \\
        test/hardware/benchmark_gather_granularity.py --out gather.json

The low-precision sparse MLA body gathers each query's selected rows with one indirect
gather-transpose (``nisa.dma_transpose`` on the software DGE) that lands them with the
latent on partitions: :data:`KEYS` offsets, one per selected row, each moving one
:data:`LATENT`-element bf16 row (``mla_sparse.py``, ``_attention_body_row_tiled_lowp``).
The rows come from ``dsa_index_expand``, which writes column ``4 g + o`` of an index row as
``4 * pool_id[g] + o`` (``index_expand.py``, the history loop and its torch reference): the
four rows of a selected pool are consecutive window rows, and a pool never straddles a
page. So the same gather can name whole pools, one offset per :data:`POOL` rows, with the
window viewed as ``[rows / POOL, POOL * LATENT]`` and each gathered element as
``[1, POOL * LATENT / 128, 128]``. (The view makes the offset unit a pool under both readings
of an indirect offset: the access pattern's outer stride and the tensor's row stride are then
the same; the NKI simulator applies the latter.) Two pools are adjacent only by chance (the
selector returns them in score order), so no larger block is contiguous.

The kernel here does that gather alone, for every query of a call split over the two
programs of a logical core, in two forms:

* ``row``: :data:`KEYS` single-row offsets, the served form, sentinel columns filled by
  the served rule (the index in the same partition of the first chunk, else row 0);
* ``pool``: ``KEYS / POOL`` pool offsets: the selected pools, the open pool the row's tail
  lies in, and sentinel pools filled by the same rule.

Each form runs at :data:`SMALL_QUERIES` and :data:`SERVED_QUERIES` queries, on two index
patterns of the 8k line (chunk 1, mostly sentinels; chunk 8, every pool selected), with a
one-element add as the fixed cost of one execution. A sample is one execution with its
output copied to CPU; its device time is the runtime system trace's ``nc_exec_running``
interval, the two physical cores merged. The order of every graph rotates each call.

The per-gather cycle of one physical core is ``(t(SERVED) - t(SMALL)) / ((SERVED - SMALL)
/ PROGRAMS)``, and the per-offset cycle that over the offsets of one gather. Every
execution stores each query's first :data:`STORED_COLUMNS` columns of its gathered tile's
first latent tile, checked bit for bit against the host's gather of the same rows. The
store keeps every gather live: the compiler removes a gather whose tile nothing reads. ``--probe`` runs and checks every graph once and times nothing, so a
lease script can read the device's fault log before the timed run.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

#: The worktree root, ahead of any installed copy (the lease sets no PYTHONPATH).
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True

import nki
import nki.isa as nisa
import nki.language as nl
from benchmark_sparse_mla_prefill import KPOOL as POOL
from benchmark_sparse_mla_prefill import (
    LATENT,
    LEASE_MARKER,
    SELECT_K,
    device_intervals,
)
from benchmark_sparse_mla_prefill import WIDTH as KEYS

#: SBUF partitions; one gather offset tile column holds this many offsets.
PARTITIONS = 128
#: The gather-transpose moves 128 elements of a row onto the partitions per latent tile.
LATENT_TILE = 128
#: Queries per unrolled block of the device loop, and gathered-tile buffers in the ring:
#: the served body's query block and its QUERY_BUFFERS default.
QUERY_BLOCK, BUFFERS = 16, 2
#: Programs of one logical core at LNC2: the served launch grid.
PROGRAMS = 2
#: Window rows: the 8k line's 64-page window (``mla_sparse_attention`` gathers from it).
WINDOW_ROWS = 8192
#: Columns of each query's stored check tile: its first offsets, 128 latent elements each.
STORED_COLUMNS = 8
#: The 8k line's chunk length, and the two chunks whose index patterns are measured.
CHUNK, PATTERN_CHUNKS = 1024, (1, 8)
SMALL_QUERIES, SERVED_QUERIES = 64, 1024


def _aligned(n: int, quantum: int = 8) -> int:
    """``n`` rounded up to whole 32-byte lines of 4-byte elements."""
    return -(-n // quantum) * quantum


@nki.jit
def gather_kernel(window_hbm, offsets_hbm):
    """Gather-transpose one window block per offset for every query.

    Args:
        window_hbm: ``[blocks, rows_per_offset * LATENT]`` bf16, the staged window viewed
            with ``rows_per_offset`` consecutive rows (1 or POOL) per block.
        offsets_hbm: ``[queries, 128, n_cols]`` int32, each query's block indices
            chunk-major (element ``(p, c)`` is block ``c * 128 + p``), every one in bounds.

    Returns:
        ``[queries, 128, STORED_COLUMNS]`` bf16: for every query, columns
        ``0 .. STORED_COLUMNS - 1`` of its gathered tile's first latent tile.
    """
    queries, _, n_cols = offsets_hbm.shape
    block_elements = window_hbm.shape[1]
    n_blocks = KEYS * LATENT // block_elements
    tiles = block_elements // LATENT_TILE
    out = nl.ndarray((queries, PARTITIONS, STORED_COLUMNS),
                     dtype=window_hbm.dtype, buffer=nl.shared_hbm)
    raw, offsets, gathered = [], [], []
    for _ in range(BUFFERS):
        raw.append(nl.ndarray((PARTITIONS, _aligned(n_cols)), dtype=nl.int32, buffer=nl.sbuf))
        offsets.append(nl.ndarray((PARTITIONS, _aligned(n_cols)), dtype=nl.uint32,
                                  buffer=nl.sbuf))
        gathered.append(nl.ndarray((LATENT_TILE, 1, tiles, n_blocks), dtype=window_hbm.dtype,
                                   buffer=nl.sbuf))
    n_prgs = nl.num_programs(axes=0)
    prg = nl.program_id(0)
    for qb in nl.affine_range(prg, queries // QUERY_BLOCK, n_prgs):
        for qi in range(QUERY_BLOCK):
            b = qi % BUFFERS
            q = qb * QUERY_BLOCK + qi
            nisa.dma_copy(dst=raw[b][:, 0:n_cols],
                          src=offsets_hbm.ap(pattern=[[n_cols, PARTITIONS], [1, n_cols]],
                                             offset=q * PARTITIONS * n_cols),
                          dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
            nisa.tensor_copy(dst=offsets[b][:, 0:n_cols], src=raw[b][:, 0:n_cols],
                             engine=nisa.engine.vector)
            nisa.dma_transpose(
                dst=gathered[b],
                src=window_hbm.ap(pattern=[[block_elements, n_blocks], [0, 1],
                                           [LATENT_TILE, tiles], [1, LATENT_TILE]],
                                  vector_offset=offsets[b][:, 0:n_cols], indirect_dim=0),
                dge_mode=nisa.dge_mode.swdge)
            nisa.dma_copy(dst=out.ap(pattern=[[STORED_COLUMNS, PARTITIONS], [1, STORED_COLUMNS]],
                                     offset=q * PARTITIONS * STORED_COLUMNS),
                          src=gathered[b][:, 0, 0, 0:STORED_COLUMNS],
                          dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)
    return out


def index_rows(chunk: int, queries: int, gen):
    """Each query's selection at 8k-line chunk ``chunk``: ``(pools, tails)``.

    Query ``t`` sees ``start + t + 1`` rows; it selects up to SELECT_K of its complete
    pools in a random (score) order, and its tail is the rows of its open pool it sees.
    """
    import torch

    start = (chunk - 1) * CHUNK
    pools, tails = [], []
    for t in range(queries):
        seen = start + t % CHUNK + 1
        pools.append(torch.randperm(seen // POOL, generator=gen)[:SELECT_K])
        tails.append(torch.arange(seen - seen % POOL, seen))
    return pools, tails


def filled(blocks):
    """The served sentinel rule on a ``[n]`` block row (-1 = sentinel): a sentinel at
    ``c * 128 + p`` takes block ``p``'s index, or 0 where that is a sentinel too."""
    import torch

    n = int(blocks.numel())
    fill = blocks[:PARTITIONS].clamp_min(0)
    rule = fill.repeat(-(-n // PARTITIONS))[:n]
    return torch.where(blocks >= 0, blocks, rule)


def offsets_of(pools, tails, rows_per_offset: int):
    """``[queries, 128, n_cols]`` int32 chunk-major offsets of one form, and the flat rows.

    ``row``: the expanded index row (pool rows in selection order, the tail, -1 to KEYS).
    ``pool``: the selected pools, the open pool when the row has a tail, -1 to KEYS / POOL.
    """
    import torch

    n_blocks = KEYS // rows_per_offset
    n_cols = -(-n_blocks // PARTITIONS)
    rows = torch.full((len(pools), n_blocks), -1, dtype=torch.int64)
    for t, (chosen, tail) in enumerate(zip(pools, tails)):
        if rows_per_offset == 1:
            expanded = (chosen.reshape(-1, 1) * POOL + torch.arange(POOL)).reshape(-1)
            rows[t, :expanded.numel()] = expanded
            rows[t, SELECT_K * POOL:SELECT_K * POOL + tail.numel()] = tail
        else:
            rows[t, :chosen.numel()] = chosen
            if tail.numel():
                rows[t, SELECT_K] = tail[0] // POOL
    rows = torch.stack([filled(r) for r in rows])
    padded = torch.zeros((len(pools), n_cols * PARTITIONS), dtype=torch.int64)
    padded[:, :n_blocks] = rows
    chunk_major = padded.reshape(len(pools), n_cols, PARTITIONS).transpose(1, 2)
    return chunk_major.to(torch.int32).contiguous(), rows


def expected(window, rows, rows_per_offset: int):
    """Every query's stored check tile, gathered on the host."""
    import torch

    first = rows[:, :STORED_COLUMNS]
    # Latent tile 0 of block b: row ``rows_per_offset * b``'s first 128 elements (the
    # pool form's first latent tile is the pool's first row).
    picked = window[first * rows_per_offset, :LATENT_TILE]
    return picked.transpose(1, 2).contiguous().to(torch.bfloat16)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="JSON report path")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=4, help="calls per graph per repeat")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--probe", action="store_true",
                        help="run and check every graph once, then stop")
    args = parser.parse_args()
    if min(args.reps, args.iterations) < 1 or args.warmup < 0:
        parser.error("use reps and iterations >= 1 and warmup >= 0")
    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1" or os.environ.get("NKI_SIMULATOR"):
        raise SystemExit("hardware benchmark: unset VLLM_NEURON_CPU_MODE and NKI_SIMULATOR")
    if not os.environ.get(LEASE_MARKER):
        raise SystemExit(f"run under devlease.py slice <name>, which sets {LEASE_MARKER}")
    if os.environ.get("NEURON_LOGICAL_NC_CONFIG") != "2":
        raise SystemExit("the served geometry is LNC2: NEURON_LOGICAL_NC_CONFIG must be 2")
    cache_root = os.environ.get("NEURON_LIBTORCH_CACHE_ROOT")
    if not cache_root:
        raise SystemExit("set NEURON_LIBTORCH_CACHE_ROOT to a compile cache directory")
    os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")

    import torch
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
    from nrtpy._nrtpy import SystemTraceSession

    import vllm_neuron  # registers the Neuron compilation backend

    if not vllm_neuron.__file__.startswith(str(ROOT) + "/"):
        raise SystemExit(f"imported vllm_neuron from {vllm_neuron.__file__}, not {ROOT}")
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(cache_root) / "benchmark_cwd"
    scratch.mkdir(parents=True, exist_ok=True)
    os.chdir(scratch)

    forms = {"row": 1, "pool": POOL}
    # Four graphs share one Python function (two window views at two query counts): never
    # let a recompile past dynamo's limit fall back to eager.
    torch._dynamo.config.fail_on_recompile_limit_hit = True

    def graph_of():
        def call(window, offsets):
            return wrap_nki(gather_kernel)[PROGRAMS](window, offsets)

        return torch.compile(call, backend="neuron_libtorch", fullgraph=True, dynamic=False)

    gen = torch.Generator().manual_seed(2176)
    window = torch.randn(WINDOW_ROWS, LATENT, generator=gen).to(torch.bfloat16)
    window_dev = window.to("neuron:0")
    views = {form: window_dev.reshape(WINDOW_ROWS // rows, rows * LATENT)
             for form, rows in forms.items()}
    graphs, inputs, checks = {}, {}, {}
    for chunk in PATTERN_CHUNKS:
        pools, tails = index_rows(chunk, SERVED_QUERIES, gen)
        for form, rows_per_offset in forms.items():
            offsets, rows = offsets_of(pools, tails, rows_per_offset)
            for queries in (SMALL_QUERIES, SERVED_QUERIES):
                name = f"{form}_c{chunk}_q{queries}"
                graphs[name] = graph_of()
                inputs[name] = (views[form], offsets[:queries].contiguous().to("neuron:0"))
                checks[name] = expected(window, rows[:queries], rows_per_offset)
    graphs["floor"] = torch.compile(lambda x: x + 1, backend="neuron_libtorch",
                                    fullgraph=True, dynamic=False)
    inputs["floor"] = (torch.zeros(1, dtype=torch.int32).to("neuron:0"),)

    failures, identical = [], {name: [0, 0] for name in checks}
    for name, graph in graphs.items():
        got = graph(*inputs[name]).to("cpu")
        if name in checks and not torch.equal(got, checks[name]):
            failures.append(f"{name}: the gathered rows differ from the host's")
    if args.probe:
        out.write_text(json.dumps({"probe": list(graphs), "failures": failures}, indent=1) + "\n")
        for failure in failures:
            print(f"FAIL {failure}", flush=True)
        return 1 if failures else 0
    for _ in range(args.warmup):
        for name, graph in graphs.items():
            graph(*inputs[name]).to("cpu")
    names = list(graphs)
    medians = {name: [] for name in names}
    for _ in range(args.reps):
        order, per = [], {name: [] for name in names}
        with SystemTraceSession() as trace:
            for it in range(args.iterations):
                first = it % len(names)
                for name in names[first:] + names[:first]:
                    got = graphs[name](*inputs[name]).to("cpu")
                    order.append(name)
                    if name in checks:
                        identical[name][0] += 1
                        identical[name][1] += int(torch.equal(got, checks[name]))
            events = trace.fetch_events_json()
        device = device_intervals(events)
        if len(device) != len(order):
            raise AssertionError(f"the trace has {len(device)} executions for {len(order)} calls")
        for name, value in zip(order, device):
            per[name].append(value)
        for name in names:
            medians[name].append(statistics.median(per[name]))
    for name, (emitted, same) in identical.items():
        if same != emitted:
            failures.append(f"{name}: {emitted - same} of {emitted} emissions differ")

    timing = {}
    for name, values in medians.items():
        median = statistics.median(values)
        timing[name] = {"rep_medians_us": values, "median_us": median,
                        "rep_spread": (max(values) - min(values)) / median}
    cycles = {}
    per_core = (SERVED_QUERIES - SMALL_QUERIES) / PROGRAMS
    for chunk in PATTERN_CHUNKS:
        for form, rows_per_offset in forms.items():
            small = timing[f"{form}_c{chunk}_q{SMALL_QUERIES}"]
            served = timing[f"{form}_c{chunk}_q{SERVED_QUERIES}"]
            gather_us = (served["median_us"] - small["median_us"]) / per_core
            cycles[f"{form}_c{chunk}"] = {
                "offsets_per_gather": KEYS // rows_per_offset,
                "rows_per_offset": rows_per_offset,
                "bytes_per_gather": KEYS * LATENT * 2,
                "per_call_us_served": served["median_us"],
                "per_gather_cycle_us": gather_us,
                "per_offset_cycle_ns": gather_us * 1e3 / (KEYS // rows_per_offset),
                "noise_floor": max(small["rep_spread"], served["rep_spread"])}
    for chunk in PATTERN_CHUNKS:
        cycles[f"pool_over_row_c{chunk}"] = (cycles[f"pool_c{chunk}"]["per_gather_cycle_us"]
                                             / cycles[f"row_c{chunk}"]["per_gather_cycle_us"])
    report = {
        "environment": {key: os.environ.get(key) for key in (
            LEASE_MARKER, "NEURON_LOGICAL_NC_CONFIG", "NEURON_PLATFORM_TARGET_OVERRIDE",
            "NEURON_LIBTORCH_CACHE_ROOT", "NEURON_CC_FLAGS")},
        "tree": str(ROOT),
        "geometry": {"keys": KEYS, "latent": LATENT, "pool": POOL, "window_rows": WINDOW_ROWS,
                     "programs": PROGRAMS, "query_block": QUERY_BLOCK, "buffers": BUFFERS,
                     "small_queries": SMALL_QUERIES, "served_queries": SERVED_QUERIES,
                     "pattern_chunks": list(PATTERN_CHUNKS)},
        "timing_method": {"reps": args.reps, "iterations": args.iterations,
                          "warmup": args.warmup, "order": "rotated every call",
                          "device": "system trace nc_exec_running, physical cores merged",
                          "noise_floor": "largest (max - min) / median of a graph's rep medians"},
        "timing": timing, "cycles": cycles, "identical": identical, "failures": failures,
    }
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(cycles, indent=1), flush=True)
    for failure in failures:
        print(f"FAIL {failure}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
