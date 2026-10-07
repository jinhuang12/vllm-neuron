# SPDX-License-Identifier: Apache-2.0
"""Cost model for the query-sharded DSA prefill selection (DERIVED; no device number yet).

What it prices, per 1024-row prefill chunk on TP = 64 (every chunk runs 1024 operator rows,
``prefill_calibrated.md`` section 7), for the 11 DSA layers together:

* **as built** -- every rank runs the score GEMM, the causal bound, the top-k and the
  sentinel ordering on all ``T`` rows over ``C = max_model_len // 4`` candidates. The
  values are worker-3's calibrated buckets (``/home/ubuntu/glm53f-wt3/calib/prefill``,
  read only): MEASURED at ``C`` 1024 and 2048, a stated linear extrapolation above.
* **query-sharded at degree d** -- each rank runs the same chain on ``R = ceil(T / d)``
  rows, then one all-gather of the ``[R, select_k]`` ids per layer.
* **two candidate-sharded fallbacks** -- each rank scores ``C / d`` candidates, takes a
  local top-k and the pieces are merged, either replicated after an all-gather (A) or
  row-sharded after an all-to-all (B).

How each sharded kernel scales (every rule is read off the kernel's loop structure):

* score GEMM and causal bound walk query rows in 128-row tiles and each tile costs the
  same whatever its height (a tile's matmuls stream the same 512 moving columns, and the
  vector ops run on the partitions in parallel). Per-rank time is the fit's per-call
  intercept plus the slope part scaled by ``ceil(R / 128) / ceil(T / 128)``. So 16 rows cost
  what 128 rows cost; that is the "d = 8 full tiles against d = 64 small tiles" question.
* sentinel ordering walks the same 128-row tiles; no intercept is known, so it scales by
  the tile ratio alone.
* top-k: the vendored kernel's own cycle model (``_estimate_rotational_cost``, what its
  factory minimises) for the config the seam builds, fitted to the two measured points as
  ``ms = alpha + beta * Mcycles`` per call. ``beta`` comes out at 0.728 ms/Mcycle, within 2%
  of one cycle at 1.4 GHz, so ``alpha`` is a per-call fixed part. The sharded value is the
  calibrated as-built value times ``(alpha + beta M(R)) / (alpha + beta M(T))``.
* the all-gather: half the calibrated all-reduce (``trn2_constants_glm_cal.json``: 1.5 us per
  hop, log2 hops, 70.77 GB/s bus), ``log2(d) * 1.5 us + (d - 1) / d * bytes / 70.77 GB/s``.
* glue: the row index, three ``index_select`` and two casts per layer. No device number
  exists; the central value is ASSUMED (``GLUE_US_PER_LAYER``) and the band spans 5-100 us.

``python -m test.vllm_neuron.functional.dsa.indexer_shard_cost --write PATH`` writes the
report JSON; ``test_indexer_shard_cost.py`` pins the arithmetic.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys

CALIB_DIR = "/home/ubuntu/glm53f-wt3/calib/prefill"
PLANNER_DIR = "/home/ubuntu/glm53f-wt3/planner"
CAL_CONSTANTS = os.path.join(PLANNER_DIR, "configs", "trn2_constants_glm_cal.json")

TP = 64
CHUNK = 1024
SELECT_K = 512
INDEX_KPOOL = 4
PARTITION = 128
#: Score and causal-bound buckets carry a per-call intercept in worker-3's fit.
FIT_BUCKETS = ("score_gemm", "causal_bound")
#: The sentinel ordering's measured per-chunk ms (11 layers, T = 1024), from the round-2
#: breakdown's master table (``/home/ubuntu/glm53f-wt2/reports/prefill_breakdown.json``):
#: p1 at C = 1024; p2 and p3 at C = 2048.
SENTINEL_KEY = "dsa/sentinel_order.py"
#: ASSUMED glue per DSA layer: 7 small ops (row index: arange, scale, add, clamp; three
#: index_select; two casts) at about 3 us each. The device test (D3) must replace it.
GLUE_US_PER_LAYER = 20.0
GLUE_BAND_US = (5.0, 100.0)
#: Contexts the headline prices (max_model_len) and the TTFT prompts.
CONTEXTS = {"8k": 8192, "64k": 65536, "256k": 262144}
TTFT_PROMPTS = {"1k": 1024, "8k": 8192, "64k": 65536, "256k": 262144}
#: The bs = 1 line's own max_model_len, used when the prompt is shorter.
MIN_CONTEXT = 4096
DEGREES = (8, 64)


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def calibration_available() -> bool:
    return os.path.exists(os.path.join(CALIB_DIR, "prefill_model.py")) and os.path.exists(
        CAL_CONSTANTS)


def load_model():
    """worker-3's calibrated prefill model, imported read-only from its own directory."""
    return _load("_w3_prefill_model", os.path.join(CALIB_DIR, "prefill_model.py")).Model()


def collective_constants() -> dict:
    cal = json.load(open(CAL_CONSTANTS))
    assert cal["collective_hops_model"] == "log2", cal["collective_hops_model"]
    return {"per_hop_s": float(cal["collective_per_hop_latency_s"]),
            "bus_bytes_per_s": float(cal["collective_bw_bytes_per_s"]),
            "source": CAL_CONSTANTS}


def allgather_ms(group: int, total_bytes: float, cc: dict | None = None) -> float:
    """One all-gather whose OUTPUT is ``total_bytes``: the all-gather half of an all-reduce."""
    if group <= 1:
        return 0.0
    cc = cc or collective_constants()
    return 1e3 * (math.log2(group) * cc["per_hop_s"]
                  + (group - 1) / group * total_bytes / cc["bus_bytes_per_s"])


def all_to_all_ms(group: int, bytes_per_rank: float, cc: dict | None = None) -> float:
    """One all-to-all, each rank sending ``bytes_per_rank`` in all: ``group - 1`` pairwise
    hops at the calibrated hop latency, the volume at the calibrated bus bandwidth. Not
    calibrated for an all-to-all (no measured one exists)."""
    if group <= 1:
        return 0.0
    cc = cc or collective_constants()
    return 1e3 * ((group - 1) * cc["per_hop_s"]
                  + (group - 1) / group * bytes_per_rank / cc["bus_bytes_per_s"])


def topk_mcycles(rows: int, width: int, k: int = SELECT_K) -> float | None:
    """The vendored top-k's cycle estimate (per program) for the config the seam builds at
    ``[rows, width]``; ``None`` when the factory refuses the geometry."""
    import nki.language as nl

    from vllm_neuron.functional.dsa.topk_select import _nki_config
    from vllm_neuron.functional.vendored_kernels.rotational_topk.rotational_topk_utils import (
        HW_PARAMS,
        _estimate_rotational_cost,
    )

    try:
        cfg = _nki_config(int(rows), int(width), int(k), nl.float32)
    except AssertionError:
        return None
    per_tile = _estimate_rotational_cost(int(k), int(width), cfg.n_stages) + HW_PARAMS.tile_overhead
    return cfg.n_bxs_tiles * per_tile / 1e6


def row_tiles(rows: int) -> int:
    return -(-int(rows) // PARTITION)


class IndexerCost:
    """The per-chunk prices; every value is ms per chunk for all DSA layers unless named."""

    def __init__(self, model=None, cc: dict | None = None, glue_us: float = GLUE_US_PER_LAYER):
        self.m = model if model is not None else load_model()
        self.cc = cc or collective_constants()
        self.layers = int(self.m.r["architecture"]["dsa_layers"])
        self.glue_us = float(glue_us)
        # top-k: ms per call per layer = alpha + beta * Mcycles, on the two measured points.
        y1 = self.m.idx_value("rotational_topk", 1024) / self.layers
        y2 = self.m.idx_value("rotational_topk", 2048) / self.layers
        m1, m2 = topk_mcycles(CHUNK, 1024), topk_mcycles(CHUNK, 2048)
        self.beta = (y2 - y1) / (m2 - m1)
        self.alpha = y1 - self.beta * m1
        mt = {p: self.m.pts[p]["master_table_ms"][SENTINEL_KEY] for p in ("p1", "p2", "p3")}
        self.sentinel_ms = {1024: mt["p1"], 2048: 0.5 * (mt["p2"] + mt["p3"])}

    # ------------------------------------------------------------------ as built
    def _scaled(self, raw: float) -> float:
        """The calibrated model's token-proportional share at T = 1024 (fixed part carved)."""
        return raw * self.m.tok_scale

    def sentinel_raw(self, cands: int) -> float:
        return self.sentinel_ms[1024] if cands <= 1024 else self.sentinel_ms[2048]

    def as_built(self, cands: int) -> dict:
        out = {bid: self._scaled(self.m.idx_value(bid, cands))
               for bid in ("score_gemm", "rotational_topk", "causal_bound")}
        out["sentinel_order"] = self._scaled(self.sentinel_raw(cands))
        return out

    def intercept(self, bid: str) -> float:
        return self._scaled(self.m.idx_fit[bid][0])

    def topk_ms_layer(self, rows: int, width: int, k: int = SELECT_K) -> float | None:
        cycles = topk_mcycles(rows, width, k)
        return None if cycles is None else self.alpha + self.beta * cycles

    def _topk_scaled(self, cands: int, rows: int, width: int, k: int = SELECT_K) -> float | None:
        """The calibrated as-built top-k at ``[CHUNK, cands]`` moved to ``[rows, width]`` by
        the cycle model's ratio."""
        mine, ref = self.topk_ms_layer(rows, width, k), self.topk_ms_layer(CHUNK, cands)
        if mine is None or ref is None:
            return None
        return self.as_built(cands)["rotational_topk"] * mine / ref

    def glue_ms(self) -> float:
        return self.layers * self.glue_us / 1e3

    # ------------------------------------------------------------------ query-sharded
    def query_sharded(self, cands: int, degree: int, tokens: int = CHUNK) -> dict:
        base = self.as_built(cands)
        rows = -(-tokens // degree)
        ratio = row_tiles(rows) / row_tiles(tokens)
        out = {}
        for bid in FIT_BUCKETS:
            fixed = self.intercept(bid)
            out[bid] = fixed + (base[bid] - fixed) * ratio
        out["sentinel_order"] = base["sentinel_order"] * ratio
        out["rotational_topk"] = (base["rotational_topk"] if degree == 1
                                  else self._topk_scaled(cands, rows, cands))
        compute_before = sum(base.values())
        compute_after = sum(out.values())
        comm = self.layers * allgather_ms(degree, tokens * SELECT_K * 4, self.cc)
        glue = self.glue_ms() if degree > 1 else 0.0
        return {"design": "query", "degree": degree, "rows_per_rank": rows, "cands": cands,
                "as_built_ms": base, "sharded_ms": out, "compute_before_ms": compute_before,
                "compute_after_ms": compute_after,
                "compute_saved_ms": compute_before - compute_after, "comm_added_ms": comm,
                "glue_added_ms": glue, "net_saved_ms": compute_before - compute_after - comm - glue,
                "gather_bytes_per_layer": tokens * SELECT_K * 4 if degree > 1 else 0}

    # ------------------------------------------------------------------ fallbacks
    def candidate_sharded(self, cands: int, degree: int, merge: str, tokens: int = CHUNK) -> dict:
        """``merge`` is ``"gather"`` (A: all-gather, every rank merges every row) or
        ``"alltoall"`` (B: all-to-all, each rank merges its own rows, then the id gather)."""
        base = self.as_built(cands)
        local = -(-cands // degree)
        kept = min(SELECT_K, local)
        col_ratio = (-(-local // 512)) / (-(-cands // 512))
        out = {}
        for bid in FIT_BUCKETS:
            fixed = self.intercept(bid)
            out[bid] = fixed + (base[bid] - fixed) * col_ratio
        local_topk = 0.0
        if local > SELECT_K:
            local_topk = self._topk_scaled(cands, tokens, local)
        width = degree * kept
        rows = tokens if merge == "gather" else -(-tokens // degree)
        merge_topk = self._topk_scaled(cands, rows, width)
        out["rotational_topk"] = None if (local_topk is None or merge_topk is None) else (
            local_topk + merge_topk)
        out["sentinel_order"] = base["sentinel_order"] * (
            1.0 if merge == "gather" else row_tiles(rows) / row_tiles(tokens))
        pair = 8  # an fp32 value and its id travel together
        if merge == "gather":
            comm = self.layers * allgather_ms(degree, tokens * width * pair, self.cc)
        else:
            comm = self.layers * (all_to_all_ms(degree, tokens * kept * pair, self.cc)
                                  + allgather_ms(degree, tokens * SELECT_K * 4, self.cc))
        compute_before = sum(base.values())
        compute_after = None if out["rotational_topk"] is None else sum(out.values())
        saved = None if compute_after is None else compute_before - compute_after
        return {"design": f"candidates+{merge}", "degree": degree, "cands_per_rank": local,
                "kept_per_rank": kept, "merge_width": width, "merge_rows": rows,
                "as_built_ms": base, "sharded_ms": out, "compute_before_ms": compute_before,
                "compute_after_ms": compute_after, "compute_saved_ms": saved,
                "comm_added_ms": comm, "glue_added_ms": self.glue_ms(),
                "net_saved_ms": None if saved is None else saved - comm - self.glue_ms()}

    # ------------------------------------------------------------------ TTFT
    def ttft(self, prompt: int, column: str, degree: int = TP) -> dict:
        """TTFT on the bs = 1 line at ``max_model_len = prompt``; a prompt that fits the line's
        own 4096 runs on that line unchanged (the p1 anchor: context 4096, segment 1024)."""
        context = None if int(prompt) <= MIN_CONTEXT else int(prompt)
        res = self.m.ttft(TP, int(prompt), CHUNK, "bs1", column, context=context)
        context = int(res["context"])
        cands = context // INDEX_KPOOL
        saved = self.query_sharded(cands, degree)["net_saved_ms"]
        return {"prompt": int(prompt), "context": context, "cands": cands, "column": column,
                "n_chunks": res["n_chunks"], "servable_as_built": res["servable_as_built"],
                "before_ms": res["ttft_ms"], "saved_per_chunk_ms": saved,
                "after_ms": res["ttft_ms"] - res["n_chunks"] * saved}

    # ------------------------------------------------------------------ follow-up: C sized
    def prefix_sized_selection_ms(self, prompt: int, degree: int) -> dict:
        """The selection's device ms over a whole prompt if ``C`` followed the real prefix.

        A follow-up, not built here: chunk ``i`` would score ``C_i`` candidates, the
        smallest power of two that holds the ``(i + 1) * CHUNK // 4`` pools its last row can
        see (at least ``SELECT_K * 2``, so the selecting regime holds), capped at the
        context's ``C``. Compared with every chunk at the context's ``C``.
        """
        context = max(int(prompt), MIN_CONTEXT)
        full = context // INDEX_KPOOL
        chunks = -(-int(prompt) // CHUNK)
        fixed = sized = 0.0
        for i in range(chunks):
            need = max(2 * SELECT_K, -(-((i + 1) * CHUNK) // INDEX_KPOOL))
            cands = min(full, 1 << (need - 1).bit_length())
            fixed += self.query_sharded(full, degree)["compute_after_ms"]
            sized += self.query_sharded(cands, degree)["compute_after_ms"]
        return {"prompt": int(prompt), "degree": degree, "chunks": chunks,
                "fixed_c_ms": fixed, "prefix_sized_ms": sized, "saved_ms": fixed - sized}

    # ------------------------------------------------------------------ memory
    @staticmethod
    def transient_bytes(rows: int, cands: int) -> dict:
        """Per-layer selection buffers one rank holds: the fp32 score and bounded score
        ``[rows, C]``, and the selector's fp32 values and int32 ids ``[rows, select_k]``."""
        return {"scores": rows * cands * 4, "bounded": rows * cands * 4,
                "topk_out": rows * SELECT_K * 8}


def report(cost: IndexerCost | None = None) -> dict:
    cost = cost or IndexerCost()
    out = {
        "what": "DERIVED cost of the query-sharded DSA prefill selection, per 1024-row chunk, "
                "TP=64, all 11 DSA layers (per_layer = /11). Source of every as-built value: "
                f"{CALIB_DIR} (worker-3 calibrated buckets).",
        "dsa_layers": cost.layers,
        "constants": {"alpha_ms_per_call_layer": cost.alpha,
                      "beta_ms_per_mcycle": cost.beta, "tok_scale": cost.m.tok_scale,
                      "glue_us_per_layer_assumed": cost.glue_us,
                      "glue_band_us": list(GLUE_BAND_US), "collective": cost.cc,
                      "sentinel_order_measured_ms": cost.sentinel_ms,
                      "fit_intercepts_ms": {b: cost.intercept(b) for b in FIT_BUCKETS}},
        "contexts": {},
        "ttft": {},
    }
    for label, context in CONTEXTS.items():
        cands = context // INDEX_KPOOL
        entry = {"context": context, "cands": cands}
        for d in DEGREES:
            entry[f"query_d{d}"] = cost.query_sharded(cands, d)
        entry["candidates_gather_d64"] = cost.candidate_sharded(cands, TP, "gather")
        entry["candidates_alltoall_d64"] = cost.candidate_sharded(cands, TP, "alltoall")
        lo, hi = (IndexerCost(cost.m, cost.cc, g).query_sharded(cands, TP)["net_saved_ms"]
                  for g in GLUE_BAND_US)
        entry["query_d64_net_saved_band_ms"] = [hi, lo]
        entry["memory_per_rank_per_layer_bytes"] = {
            "as_built": IndexerCost.transient_bytes(CHUNK, cands),
            "query_d64": IndexerCost.transient_bytes(-(-CHUNK // TP), cands),
            "query_d8": IndexerCost.transient_bytes(-(-CHUNK // 8), cands)}
        out["contexts"][label] = entry
    for label, prompt in TTFT_PROMPTS.items():
        out["ttft"][label] = {col: cost.ttft(prompt, col) for col in ("asbuilt", "withbranches")}
    out["followup_prefix_sized_c"] = {
        label: {f"d{d}": cost.prefix_sized_selection_ms(prompt, d) for d in (1, TP)}
        for label, prompt in TTFT_PROMPTS.items() if prompt >= 65536}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", default=None, help="write the report JSON here")
    args = ap.parse_args(argv)
    data = report()
    text = json.dumps(data, indent=1, sort_keys=True, default=float)
    if args.write:
        with open(args.write, "w") as handle:
            handle.write(text + "\n")
    for label, entry in data["contexts"].items():
        q = entry["query_d64"]
        print(f"{label}: C={entry['cands']} as-built {q['compute_before_ms']:.1f} ms/chunk, "
              f"d64 {q['compute_after_ms']:.1f}, saved {q['compute_saved_ms']:.1f}, comm "
              f"{q['comm_added_ms']:.2f}, glue {q['glue_added_ms']:.2f}, net {q['net_saved_ms']:.1f}"
              f" ({q['net_saved_ms'] / data['dsa_layers']:.2f}/layer); d8 net "
              f"{entry['query_d8']['net_saved_ms']:.1f}; fallback A "
              f"{entry['candidates_gather_d64']['net_saved_ms']}, B "
              f"{entry['candidates_alltoall_d64']['net_saved_ms']}")
    for label, cols in data["ttft"].items():
        print(label, {c: (round(v["before_ms"]), round(v["after_ms"])) for c, v in cols.items()})
    for label, by_d in data["followup_prefix_sized_c"].items():
        print("prefix-sized C", label, {d: round(v["saved_ms"]) for d, v in by_d.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
