# SPDX-License-Identifier: Apache-2.0
"""Device A/B of the DSA prefill selection chain: one rank's sharded rows against all rows.

Run it through the device lease, which pins the cores and the LNC; this script selects no
cores::

    python3 devlease.py slice dsa -- python test/vllm_neuron/functional/dsa/indexer_shard_device.py \\
        --cands 16384 --label dsa-r1 --output /tmp/indexer_shard_device/c16384_r1.json

For one candidate width ``C`` it compiles and times three graphs on one logical core:

* ``replicated`` -- the as-built chain on all ``T`` rows of a chunk:
  ``Glm5NextDSAIndexer.score_pools`` then ``select_bounded_pools`` (score GEMM, causal
  bound, top-k, sentinel, order) -> ``[T, k]`` int32 pool ids.
* ``sharded`` -- one rank's share at degree ``d``: ``indexer_shard.select_local_rows`` on
  the same ``T``-row operands (row index, three ``index_select``, the same two stages on
  ``R = ceil(T / d)`` rows) and the float32 cast the all-gather sends. The rank is a device
  operand, as in the model. The all-gather is not timed: one chip has no TP group.
* ``launch`` -- a graph that only scales a one-element input: the launch and the
  synchronising copy every timed call pays, subtracted from the two graphs above.

Each graph returns its ids and a float32 sum of them; a timed call copies only the sum to
the CPU, which waits for the whole graph. The chunk sits at the end of a
``index_kpool * C``-token context, so every row sees all ``C`` pools. After timing, the
sharded ids must equal the replicated ids on the rank's rows as per-row sets, and every
stage must have taken the NKI route. One JSON object is written per run.

``--skip-bound`` times the chain without the causal bound: ``_select_bounded`` (the tail
``select_bounded_pools`` shares with ``forward_requests``) on the raw scores. At
``C = 65536`` the bound's whole-width SBUF row does not compile on either path, so that
width is measured this way. The first call there compiles for about 7 minutes; the lease
allows 10, so run one width per lease.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

#: The worktree root, ahead of any installed copy: the lease command sets no PYTHONPATH.
#: Bytecode is not written, so a run leaves no ``__pycache__`` in the tree.
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import torch  # noqa: E402

import vllm_neuron  # noqa: E402,F401 -- registers the Neuron compilation backend
from vllm_neuron.functional.dsa.indexer_shard import row_shard, select_local_rows  # noqa: E402
from test.vllm_neuron.functional.dsa import indexer_shard_case as case  # noqa: E402

DEVICE = "neuron:0"
#: The prefill chunk every calibrated number is per (``rules.json`` reference point p1).
CHUNK = 1024


def compiled(fn):
    return torch.compile(fn, backend="neuron_libtorch", fullgraph=True, dynamic=False,
                         options={"compiler_args": os.environ.get("NEURON_CC_FLAGS", "")})


def operands(tokens: int, cands: int, seed: int):
    """``case_operands`` with the chunk moved to the end of an ``index_kpool * cands``
    context: ``(query [T, H, D] bf16, keys [C, D] bf16, weights [T, H] fp32, seq_lens [T]
    int32)``."""
    query, keys, weights, _ = case.case_operands(tokens, cands, seed)
    end = cands * case.make_indexer(cands).index_kpool
    seq_lens = torch.arange(end - tokens + 1, end + 1, dtype=torch.int32)
    return query, keys, weights, seq_lens


def measure(fn, inputs, warmup: int, iterations: int) -> dict:
    """Wall time of one call that waits for the graph, in microseconds."""
    for _ in range(warmup):
        fn(*inputs)[1].to("cpu")
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        fn(*inputs)[1].to("cpu")
        samples.append((time.perf_counter_ns() - started) / 1000.0)
    samples.sort()
    return {"iterations": iterations, "median_us": statistics.median(samples),
            "p10_us": samples[int(0.1 * (len(samples) - 1))],
            "p90_us": samples[int(0.9 * (len(samples) - 1))],
            "min_us": samples[0], "max_us": samples[-1]}


def timed(name, graph, inputs, args, out) -> torch.Tensor:
    """Compile ``graph``, time it into ``out[name]``, and return its ids on the CPU."""
    torch._dynamo.reset()
    case.reset_stage_counters()
    fn = compiled(graph)
    started = time.perf_counter()
    ids = fn(*inputs)[0].to("cpu")
    first = time.perf_counter() - started
    out[name] = {"first_call_s": first, "routes": case.stage_counters(),
                 **measure(fn, inputs, args.warmup, args.iterations)}
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cands", type=int, required=True)
    parser.add_argument("--degree", type=int, default=64)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--skip-bound", action="store_true",
                        help="time the chain without the causal bound (see the docstring)")
    parser.add_argument("--label", required=True, help="lease name and repeat, for the record")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("NKI_SIMULATOR") not in (None, "", "0") or os.environ.get(
            "VLLM_NEURON_CPU_MODE") == "1":
        raise ValueError("a device run needs NKI_SIMULATOR and VLLM_NEURON_CPU_MODE off")
    if not os.environ.get("NEURON_RT_VISIBLE_CORES"):
        raise ValueError("run under the device lease, which pins NEURON_RT_VISIBLE_CORES")
    if not vllm_neuron.__file__.startswith(str(ROOT) + "/"):
        raise ValueError(f"imported vllm_neuron from {vllm_neuron.__file__}, not {ROOT}")
    if not 0 <= args.rank < args.degree:
        raise ValueError(f"rank {args.rank} is not a rank of degree {args.degree}")

    indexer = case.make_indexer(args.cands)
    shard = row_shard(CHUNK, args.degree)
    query, keys, weights, seq_lens = (t.to(DEVICE) for t in operands(CHUNK, args.cands,
                                                                       args.seed))
    rank = torch.full((1,), args.rank, dtype=torch.int32, device=DEVICE)
    probe = torch.ones(1, dtype=torch.float32, device=DEVICE)

    def select(rows_query, rows_weights, rows_seq_lens):
        scores = indexer.score_pools(rows_query, keys, rows_weights)
        if args.skip_bound:
            return indexer._select_bounded(scores)
        return indexer.select_bounded_pools(scores, rows_seq_lens)

    def replicated(q, w, s):
        ids = select(q, w, s)
        return ids, ids.to(torch.float32).sum()

    def sharded(q, w, s, r):
        ids = select_local_rows(select, q, w, s, shard, r).to(torch.float32)
        return ids, ids.sum()

    def launch(x):
        y = x * 2.0
        return y, y.sum()

    record = {"label": args.label, "cands": args.cands, "tokens": CHUNK,
              "degree": args.degree, "rows": shard.rows, "rank": args.rank,
              "causal_bound": not args.skip_bound,
              "select_k": int(indexer.select_k()), "heads": int(indexer.index_n_heads),
              "head_dim": int(indexer.index_head_dim),
              "visible_cores": os.environ["NEURON_RT_VISIBLE_CORES"],
              "lnc": os.environ.get("NEURON_LOGICAL_NC_CONFIG"),
              "neuron_cc_flags": os.environ.get("NEURON_CC_FLAGS", ""), "graphs": {}}
    graphs = record["graphs"]
    timed("launch", launch, (probe,), args, graphs)
    whole = timed("replicated", replicated, (query, weights, seq_lens), args, graphs)
    local = timed("sharded", sharded, (query, weights, seq_lens, rank), args, graphs)

    rows = range(args.rank * shard.rows, min((args.rank + 1) * shard.rows, CHUNK))
    mine = local.to(torch.int64)
    same = sum(int(set(mine[i].tolist()) == set(whole[row].to(torch.int64).tolist()))
               for i, row in enumerate(rows))
    launch_us = graphs["launch"]["median_us"]
    record["check"] = {"rows": len(rows), "same_set_rows": same,
                       "routes_off_nki": {g: {k: v for k, v in body["routes"].items()
                                              if v[1] != 0}
                                          for g, body in graphs.items()}}
    record["net_ms"] = {g: (graphs[g]["median_us"] - launch_us) / 1000.0
                        for g in ("replicated", "sharded")}
    record["net_ms"]["saved"] = record["net_ms"]["replicated"] - record["net_ms"]["sharded"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"label": args.label, "cands": args.cands, "net_ms": record["net_ms"],
                      "check": record["check"],
                      "first_call_s": {g: round(b["first_call_s"], 1)
                                       for g, b in graphs.items()}}), flush=True)
    if same != len(rows):
        raise SystemExit(f"sharded ids differ from replicated on {len(rows) - same} rows")
    off_nki = {g: v for g, v in record["check"]["routes_off_nki"].items() if v}
    if off_nki:
        raise SystemExit(f"a stage left the NKI route: {off_nki}")


if __name__ == "__main__":
    main()
