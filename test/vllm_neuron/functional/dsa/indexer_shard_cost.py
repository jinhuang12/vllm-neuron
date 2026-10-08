# SPDX-License-Identifier: Apache-2.0
"""Cost model and device record of the query-sharded DSA prefill selection.

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
  hop, log2 hops, 70.77 GB/s bus), ``log2(d) * 1.5 us + (d - 1) / d * bytes / 70.77 GB/s``,
  of the int32 ids.
* glue: the row cut, one NKI launch per layer (``shard_rows.dsa_take_rank_rows``), MEASURED
  on the device: its segment in the profiled ``cut`` graph, the median over the profiled
  runs. The band is the lowest and highest ``sharded - precut`` of every device repeat: the
  cut together with every other difference between the two compiled chains (the compiler
  places the score kernel's DMAs on other engines after the cut, for one).

Three MEASURED sections come from the durable device records (``DEVICE_RECORDS_DIR``):

* ``device_ab``: per candidate width, the medians over the repeats of
  ``indexer_shard_device.py`` (device clock, less the launch graph) against the DERIVED
  per-layer saving above;
* ``perf``: per graph, the device time of each op from the run's ``neuron-explorer``
  profile (``indexer_shard_profile.py``), next to its entitlement: the larger of
  ``FLOPs / te_peak_flops`` and ``HBM bytes / hbm_bw_bytes_per_s``
  (``entitlement.json`` ``meta.hardware``), with the bytes and FLOPs of each op derived
  from the record's shapes (:func:`op_work`);
* ``allgather_two_ranks``: the runtime's own all-gather of int32 against float32 between
  two ranks of one chip (``nccom-test``, output verified), at the sizes up to the one the
  sharded selection gathers.

``python -m test.vllm_neuron.functional.dsa.indexer_shard_cost --write PATH`` writes the
report JSON; ``test_indexer_shard_cost.py`` pins the arithmetic.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import os
import re
import statistics
import sys

#: Host-local inputs of this campaign, read only. This module, its test and the device and
#: profile scripts are report scaffolding, not part of the plugin: the test skips where
#: these paths are absent.
CALIB_DIR = "/home/ubuntu/glm53f-wt3/calib/prefill"
PLANNER_DIR = "/home/ubuntu/glm53f-wt3/planner"
CAL_CONSTANTS = os.path.join(PLANNER_DIR, "configs", "trn2_constants_glm_cal.json")
#: The hardware rates every entitlement uses (``meta.hardware``: te_peak_flops,
#: hbm_bw_bytes_per_s, per logical core).
ENTITLEMENT_JSON = "/home/ubuntu/glm53f-wt3/reports/entitlement.json"
#: The device A/B records (``indexer_shard_device.py``), their logs and per-op profiles
#: (``<name>.ops.json``, ``indexer_shard_profile.py``), kept beside the report.
DEVICE_RECORDS_DIR = "/home/ubuntu/glm53f-wt5/reports/indexer_shard_device"

#: The calibrated operating point (``rules.json`` reference point p1): TP = 64 ranks,
#: 1024-row prefill chunks. Every as-built value below is read at this point.
TP = 64
CHUNK = 1024
#: Bytes of one float32 score or value.
FP32_BYTES = 4


def _model_dials() -> tuple[int, int]:
    """``(index_kpool, select_k)`` from the model's own config."""
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    cfg = Glm5NextTextConfig()
    return int(cfg.index_kpool), int(cfg.index_topk) // int(cfg.index_kpool)


INDEX_KPOOL, SELECT_K = _model_dials()


def _id_bytes() -> int:
    """Bytes of one pool id as the all-gather moves it (``indexer_shard.POOL_ID_DTYPE``)."""
    from vllm_neuron.functional.dsa.indexer_shard import POOL_ID_DTYPE

    return int(POOL_ID_DTYPE.itemsize)


ID_BYTES = _id_bytes()
#: The two candidate widths the round-2 breakdown measured the indexer at: p1
#: (max_model_len 4096) and p2 / p3 (8192). worker-3's fits and the top-k fit use both.
FIT_CANDS = (1024, 2048)
#: Score and causal-bound buckets carry a per-call intercept in worker-3's fit.
FIT_BUCKETS = ("score_gemm", "causal_bound")
#: The sentinel ordering's measured per-chunk ms (11 layers, T = 1024), from the round-2
#: breakdown's master table (``/home/ubuntu/glm53f-wt2/reports/prefill_breakdown.json``):
#: p1 at C = 1024; p2 and p3 at C = 2048.
SENTINEL_KEY = "dsa/sentinel_order.py"
#: The device runs the report summarises, by name: ``(cands, causal bound)``. Each has
#: ``DEVICE_REPEATS`` records ``<name>_r<i>.json``; repeat 1 also has its per-op profile.
#: At C = 65536 only the chain without the causal bound compiles (as built), and
#: ``c65536_<path>_compile.log`` hold each path's compiler error.
DEVICE_RUNS = {"c2048": (2048, True), "c16384": (16384, True), "c2048nb": (2048, False),
               "c16384nb": (16384, False), "c65536nb": (65536, False)}
DEVICE_REPEATS = 3
#: The compiler-error logs of the full chain at C = 65536, by path.
COMPILE_FAILURE_LOGS = {path: f"c65536_{path}_compile.log" for path in ("sharded", "replicated")}
#: The dtypes of the two-rank all-gather check, each in ``allgather_<dtype>.json`` (the
#: command is in ``allgather_command.txt``).
ALLGATHER_DTYPES = ("int32", "fp32")
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
    """Whether this host holds every input of the report: the calibrated model, the
    hardware rates and the device records."""
    return (os.path.exists(os.path.join(CALIB_DIR, "prefill_model.py"))
            and os.path.exists(CAL_CONSTANTS) and os.path.exists(ENTITLEMENT_JSON)
            and os.path.isdir(DEVICE_RECORDS_DIR))


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
    """Query-row tiles the selection kernels walk for ``rows`` rows."""
    from vllm_neuron.functional.dsa.indexer_shard import ROW_TILE

    return -(-int(rows) // ROW_TILE)


def cand_tiles(cands: int) -> int:
    """Candidate tiles of the score GEMM's moving operand for ``cands`` candidates."""
    from vllm_neuron.functional.dsa.score_gemm import CAND_TILE

    return -(-int(cands) // CAND_TILE)


# ---------------------------------------------------------------------------------- device
def device_record(name: str) -> dict:
    """One record of the device A/B: ``DEVICE_RECORDS_DIR/<name>.json``."""
    with open(os.path.join(DEVICE_RECORDS_DIR, f"{name}.json")) as handle:
        return json.load(handle)


def device_repeats(run: str) -> list[dict]:
    """The ``DEVICE_REPEATS`` records of one run of :data:`DEVICE_RUNS`."""
    return [device_record(f"{run}_r{i}") for i in range(1, DEVICE_REPEATS + 1)]


def profiled_ops(run: str) -> dict:
    """The per-op profile of repeat 1 of a run (``indexer_shard_profile.py`` output)."""
    with open(os.path.join(DEVICE_RECORDS_DIR, f"{run}_r1.ops.json")) as handle:
        return json.load(handle)


def measured_glue_us() -> float:
    """The row cut's device time, us per layer: its segment in the profiled ``cut`` graph
    (the median over the executions), the median over the profiled runs."""
    return statistics.median(
        statistics.median(_segment_ops(e)[0]["row cut"]
                          for e in profiled_ops(run)["graphs"]["cut"]["executions"])
        for run in DEVICE_RUNS)


def measured_glue_band_us() -> list[float]:
    """The lowest and the highest ``sharded - precut`` of every device repeat, us."""
    values = [1e3 * r["net_ms"]["cut_in_chain"] for run in DEVICE_RUNS
              for r in device_repeats(run)]
    return [min(values), max(values)]


def hardware_rates() -> dict:
    """The per-logical-core rates every entitlement uses (``entitlement.json``)."""
    with open(ENTITLEMENT_JSON) as handle:
        hardware = json.load(handle)["meta"]["hardware"]
    return {"te_peak_flops": float(hardware["te_peak_flops"]),
            "hbm_bw_bytes_per_s": float(hardware["hbm_bw_bytes_per_s"]),
            "source": ENTITLEMENT_JSON}


def compiler_error(log_name: str) -> dict:
    """The first neuronx-cc error code in a compile log and the line that carries it."""
    with open(os.path.join(DEVICE_RECORDS_DIR, log_name)) as handle:
        for line in handle:
            found = re.search(r"NCC_[A-Z]+[0-9]+", line)
            if found:
                return {"code": found.group(0), "line": line.strip()[:400], "log": log_name}
    raise ValueError(f"{log_name} holds no neuronx-cc error code")


def allgather_check() -> dict:
    """The two-rank all-gather check: p50 us per gathered (output) bytes, by dtype, and
    int32 over float32 per size. The sizes must include the chunk's ``[T, select_k]`` ids."""
    p50 = {}
    for dtype in ALLGATHER_DTYPES:
        with open(os.path.join(DEVICE_RECORDS_DIR, f"allgather_{dtype}.json")) as handle:
            rows = json.load(handle)["results"]
        if any(row["type"] != dtype for row in rows):
            raise ValueError(f"allgather_{dtype}.json holds another dtype")
        p50[dtype] = {str(row["size(B)"]): row["time:p50(us)"] for row in rows}
    if str(CHUNK * SELECT_K * ID_BYTES) not in p50["int32"]:
        raise ValueError("the all-gather check does not reach the chunk's id bytes")
    return {"p50_us": p50,
            "int32_over_fp32": {size: us / p50["fp32"][size]
                                for size, us in p50["int32"].items()},
            "chunk_ids_bytes": CHUNK * SELECT_K * ID_BYTES,
            "check_files": [f"allgather_{dtype}.{ext}" for dtype in ALLGATHER_DTYPES
                      for ext in ("json", "log")]}


class IndexerCost:
    """The per-chunk prices; every value is ms per chunk for all DSA layers unless named."""

    def __init__(self, model=None, cc: dict | None = None, glue_us: float | None = None):
        self.m = model if model is not None else load_model()
        self.cc = cc or collective_constants()
        self.layers = int(self.m.r["architecture"]["dsa_layers"])
        self.glue_us = float(measured_glue_us() if glue_us is None else glue_us)
        # top-k: ms per call per layer = alpha + beta * Mcycles, on the two measured points.
        lo, hi = FIT_CANDS
        y1 = self.m.idx_value("rotational_topk", lo) / self.layers
        y2 = self.m.idx_value("rotational_topk", hi) / self.layers
        m1, m2 = topk_mcycles(CHUNK, lo), topk_mcycles(CHUNK, hi)
        self.beta = (y2 - y1) / (m2 - m1)
        self.alpha = y1 - self.beta * m1
        mt = {p: self.m.pts[p]["master_table_ms"][SENTINEL_KEY] for p in ("p1", "p2", "p3")}
        self.sentinel_ms = {lo: mt["p1"], hi: 0.5 * (mt["p2"] + mt["p3"])}

    # ------------------------------------------------------------------ as built
    def _scaled(self, raw: float) -> float:
        """The calibrated model's token-proportional share at T = 1024 (fixed part carved)."""
        return raw * self.m.tok_scale

    def sentinel_raw(self, cands: int) -> float:
        lo, hi = FIT_CANDS
        return self.sentinel_ms[lo] if cands <= lo else self.sentinel_ms[hi]

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
        comm = self.layers * allgather_ms(degree, tokens * SELECT_K * ID_BYTES, self.cc)
        glue = self.glue_ms() if degree > 1 else 0.0
        return {"design": "query", "degree": degree, "rows_per_rank": rows, "cands": cands,
                "as_built_ms": base, "sharded_ms": out, "compute_before_ms": compute_before,
                "compute_after_ms": compute_after,
                "compute_saved_ms": compute_before - compute_after, "comm_added_ms": comm,
                "glue_added_ms": glue, "net_saved_ms": compute_before - compute_after - comm - glue,
                "gather_bytes_per_layer": tokens * SELECT_K * ID_BYTES if degree > 1 else 0}

    # ------------------------------------------------------------------ fallbacks
    def candidate_sharded(self, cands: int, degree: int, merge: str, tokens: int = CHUNK) -> dict:
        """``merge`` is ``"gather"`` (A: all-gather, every rank merges every row) or
        ``"alltoall"`` (B: all-to-all, each rank merges its own rows, then the id gather)."""
        base = self.as_built(cands)
        local = -(-cands // degree)
        kept = min(SELECT_K, local)
        col_ratio = cand_tiles(local) / cand_tiles(cands)
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
        pair = FP32_BYTES + ID_BYTES  # a value and its id travel together
        if merge == "gather":
            comm = self.layers * allgather_ms(degree, tokens * width * pair, self.cc)
        else:
            comm = self.layers * (all_to_all_ms(degree, tokens * kept * pair, self.cc)
                                  + allgather_ms(degree, tokens * SELECT_K * ID_BYTES, self.cc))
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
        return {"scores": rows * cands * FP32_BYTES, "bounded": rows * cands * FP32_BYTES,
                "topk_out": rows * SELECT_K * (FP32_BYTES + ID_BYTES)}


# ------------------------------------------------------------------------- device sections
def device_ab(cost: IndexerCost) -> dict:
    """Per run of :data:`DEVICE_RUNS`: the medians over its repeats against the DERIVED
    per-layer saving (``compute_saved_ms / layers``, less the causal bound's share for a
    chain without it), and every repeat's checks."""
    out = {}
    for run, (cands, bound) in DEVICE_RUNS.items():
        repeats = device_repeats(run)
        for record in repeats:
            if (record["cands"], record["causal_bound"], record["tokens"],
                    record["degree"]) != (cands, bound, CHUNK, TP):
                raise ValueError(f"{record['label']} is not the run {run} names")
        net = {key: statistics.median(r["net_ms"][key] for r in repeats)
               for key in ("replicated", "sharded", "precut", "cut", "saved", "cut_in_chain")}
        q = cost.query_sharded(cands, TP)
        derived = q["compute_saved_ms"]
        if not bound:
            derived -= q["as_built_ms"]["causal_bound"] - q["sharded_ms"]["causal_bound"]
        derived /= cost.layers
        checks = []
        for record in repeats:
            check = record["check"]
            checks.append(check["same_set_rows"] == check["rows"]
                          and all(check["cut_exact_by_rank"].values())
                          and not any(check["routes_off_nki"].values())
                          and check["cut_routes"]["sharded"] == [1, 0]
                          and check["cut_routes"]["cut"] == [1, 0]
                          and check["sharded_equals_precut"])
        saved = [r["net_ms"]["saved"] for r in repeats]
        out[run] = {
            "cands": cands, "with_causal_bound": bound, "repeats": len(repeats),
            "measured_replicated_ms": net["replicated"], "measured_sharded_ms": net["sharded"],
            "measured_precut_ms": net["precut"], "measured_cut_ms": net["cut"],
            "measured_saved_ms": net["saved"],
            "measured_sharded_less_precut_ms": net["cut_in_chain"],
            "measured_saved_range_ms": [min(saved), max(saved)],
            "wall_saved_ms": statistics.median(r["wall_net_ms"]["saved"] for r in repeats),
            "launch_us": statistics.median(r["graphs"]["launch"]["device"]["median_us"]
                                           for r in repeats),
            "derived_saved_ms": derived, "measured_over_derived": net["saved"] / derived,
            "checks_pass": all(checks),
            "record_files": [f"{run}_r{i}.json" for i in range(1, DEVICE_REPEATS + 1)],
        }
    out["compile_failures"] = {path: compiler_error(log)
                               for path, log in COMPILE_FAILURE_LOGS.items()}
    return out


#: The selection chain's ops as ``indexer_shard_profile.py`` names their segments.
PROFILE_OPS = {"shard_rows.py": "row cut", "score_gemm.py": "score GEMM",
               "causal_bound.py._causal_bound_nki": "causal bound", "rotational_topk": "top-k",
               "causal_bound.py._causal_sentinel_nki": "causal sentinel",
               "sentinel_order.py": "sentinel order", "preamble": "preamble",
               "epilogue": "epilogue"}
#: What each op is. A compiler-op segment is named by the op before it: after the top-k it
#: is the selector's id conversion (``topk_select``: uint32 -> int64, ``_select_bounded``:
#: -> int32, which neuronx-cc emits as one 32-bit copy); after the order it is the copy of
#: the graph's output ids and the A/B checksum's load.
OP_KINDS = {
    "row cut": "NKI kernel, this branch (shard_rows.py)",
    "score GEMM": "NKI kernel, in tree (score_gemm.py)",
    "causal bound": "NKI kernel, in tree (causal_bound.py)",
    "top-k": "NKI kernel, vendored in tree (rotational_topk)",
    "id converts": "compiler op, as built on both paths (topk_select .to(int64), "
                   "_select_bounded .to(int32))",
    "causal sentinel": "NKI kernel, in tree (causal_bound.py)",
    "sentinel order": "NKI kernel, in tree (sentinel_order.py)",
    "output copy and checksum": "graph boundary: the A/B graph returns the ids and sums them",
    "compiler op (other)": "compiler op",
    "preamble": "program start and the kernels' first loads, before the first core barrier",
    "epilogue": "graph boundary: output copies and the A/B checksum",
}
#: Bytes per element of the chain's operands (the model's dtypes, ``case_operands``): query
#: and keys bf16, gate fp32, lengths and ids int32, scores and values fp32.
_BF16, _FP32, _I32 = 2, FP32_BYTES, ID_BYTES


def op_work(op: str, rows: int, cands: int, heads: int, head_dim: int,
            k: int = SELECT_K) -> tuple[int, int] | None:
    """``(FLOPs, HBM bytes)`` one op needs at least, from its shapes; None for an op that
    is not selection work (program start, graph boundary)."""
    if op == "row cut":  # read and write one row of each operand
        return 0, 2 * rows * (heads * head_dim * _BF16 + heads * _FP32 + _I32)
    if op == "score GEMM":  # q.k per head, then the gated sum over heads
        flops = 2 * rows * heads * head_dim * cands
        moved = (rows * heads * head_dim * _BF16 + cands * head_dim * _BF16
                 + rows * heads * _FP32 + rows * cands * _FP32)
        return flops, moved
    if op == "causal bound":
        return 0, 2 * rows * cands * _FP32 + rows * _I32
    if op == "top-k":
        return 0, rows * cands * _FP32 + rows * k * (_FP32 + _I32)
    if op == "id converts":
        return 0, 2 * rows * k * _I32
    if op == "causal sentinel":
        return 0, rows * k * (_FP32 + _I32) + rows * k * _I32
    if op == "sentinel order":
        return 0, 2 * rows * k * _I32
    return None


#: The compiler-op segments, by the op of the segment before them.
_AFTER = {"top-k": "id converts", "sentinel order": "output copy and checksum"}


def _segment_ops(execution: dict) -> tuple[dict[str, float], dict[str, list[str]]]:
    """One profiled execution's segment time by op name (:data:`PROFILE_OPS`), and the
    profile's names of the segments each op took."""
    times: dict[str, float] = {}
    names: dict[str, list[str]] = {}
    previous = None
    for segment in execution["segments"]:
        name = segment["op"]
        if name in PROFILE_OPS:
            op = PROFILE_OPS[name]
        elif name and name.startswith("compiler "):
            op = _AFTER.get(previous, "compiler op (other)")
        else:
            op = "compiler op (other)"
        times[op] = times.get(op, 0.0) + segment["length_us"]
        names.setdefault(op, []).append(name)
        if op not in (*_AFTER.values(), "compiler op (other)"):
            previous = op
    return times, names


def _copy_transfer_us(execution: dict, segment_names: list[str]) -> tuple[float | None, int]:
    """``(us from the first copy's issue to the last copy's last packet, most bursts that fit
    one copy)`` of the compiler copies in ``segment_names`` (``indexer_shard_profile.py``:
    a copy's transfer runs past its instruction); None when a copy has no fitting burst."""
    copies = [c for name in segment_names
              for c in execution["compiler_ops"].get(name, {}).get("copies", [])]
    if not copies or any(c["transfer_us"] is None for c in copies):
        return None, 0
    done = max(c["issued_us"] + c["transfer_us"] for c in copies)
    return done - min(c["issued_us"] for c in copies), max(c["matches"] for c in copies)


def _compiler_inventory(executions: list[dict], split: list[tuple]) -> dict:
    """Every compiler op of a profiled graph (instructions with a BIR name and no NKI
    source): the op whose segment holds it, its opcodes and HBM bytes, its instruction time
    and, for a copy, its time from issue to the last packet (medians over the executions)."""
    out = {}
    for name in sorted({n for e in executions for n in e["compiler_ops"]}):
        found = [(e["compiler_ops"][name], segs) for e, (_, segs) in zip(executions, split)
                 if name in e["compiler_ops"]]
        first, segs = found[0]
        where = first["segment_op"]
        label = PROFILE_OPS.get(where) or next(
            (op for op, names in segs.items() if where in names), where)
        body = {"segment": label, "opcodes": first["opcodes"],
                "instructions": first["instructions"],
                "instruction_us": statistics.median(e["instruction_us"] for e, _ in found),
                "hbm_read_bytes": first["hbm_read_bytes"],
                "hbm_write_bytes": first["hbm_write_bytes"]}
        took = [_copy_transfer_us({"compiler_ops": {name: e}}, [name])[0] for e, _ in found]
        if all(t is not None for t in took):
            body["transfer_us"] = statistics.median(took)
        out[name] = body
    return out


def _traced_nodes(graph_ops: list[dict]) -> list[dict]:
    """A record's traced nodes, one entry per distinct (target, outputs), in trace order."""
    counts: dict[str, dict] = {}
    for node in graph_ops:
        key = json.dumps([node["target"], node["out"]])
        counts.setdefault(key, {"target": node["target"], "out": node["out"], "count": 0})
        counts[key]["count"] += 1
    return list(counts.values())


def perf(ab: dict, cost: IndexerCost) -> dict:
    """Per profiled run and graph: each op's device time against its entitlement; the id
    converts also from issue to their last packet (``transfer_us``), and the top-k against
    the cost model's cycle fit (``model_us``)."""
    rates = hardware_rates()
    out = {"rates": rates}
    for run in DEVICE_RUNS:
        record = device_record(f"{run}_r1")
        profiled = profiled_ops(run)
        entry = {"profile": f"{run}_r1.ops.json"}
        for graph in ("replicated", "sharded", "precut", "cut"):
            executions = profiled["graphs"][graph]["executions"]
            split = [_segment_ops(e) for e in executions]
            per = [times for times, _ in split]
            names = sorted({op for times in per for op in times})
            rows = CHUNK if graph == "replicated" else int(record["rows"])
            ops = {}
            for op in names:
                body = {"kind": OP_KINDS[op],
                        "measured_us": statistics.median(t.get(op, 0.0) for t in per)}
                work = op_work(op, rows, int(record["cands"]), int(record["heads"]),
                               int(record["head_dim"]), int(record["select_k"]))
                if work is not None:
                    flops, moved = work
                    entitled = 1e6 * max(flops / rates["te_peak_flops"],
                                         moved / rates["hbm_bw_bytes_per_s"])
                    body.update({"flops": flops, "bytes": moved, "entitlement_us": entitled,
                                 "over_entitlement": body["measured_us"] / entitled})
                if op == "id converts":
                    took = [_copy_transfer_us(e, segs.get(op, []))
                            for e, (_, segs) in zip(executions, split)]
                    if all(t is not None for t, _ in took):
                        body["transfer_us"] = statistics.median(t for t, _ in took)
                        body["transfer_over_entitlement"] = body["transfer_us"] / entitled
                        body["transfer_matches"] = max(m for _, m in took)
                if op == "top-k":
                    model = cost.topk_ms_layer(rows, int(record["cands"]),
                                               int(record["select_k"]))
                    if model is not None:
                        body["model_us"] = 1e3 * model
                        body["measured_over_model"] = body["measured_us"] / body["model_us"]
                ops[op] = body
            if graph == "sharded":
                # The row cut runs beside the score GEMM's first loads, so it has no segment
                # of its own here (its own time is the cut graph's); what the sharded chain
                # adds over the precut one is sharded - precut on the device clock.
                flops, moved = op_work("row cut", rows, int(record["cands"]),
                                       int(record["heads"]), int(record["head_dim"]))
                entitled = 1e6 * moved / rates["hbm_bw_bytes_per_s"]
                cut = ops.setdefault("row cut", {"kind": OP_KINDS["row cut"], "flops": flops,
                                                 "bytes": moved, "entitlement_us": entitled})
                cut["sharded_less_precut_us"] = 1e3 * ab[run]["measured_sharded_less_precut_ms"]
            entry[graph] = {"rows": rows, "executions": len(executions),
                            "execution_us": statistics.median(e["execution_us"]
                                                              for e in executions),
                            "ops": ops,
                            "compiler_ops": _compiler_inventory(executions, split),
                            "traced_nodes": _traced_nodes(record["graphs"][graph]["graph_ops"])}
        out[run] = entry
    return out


#: The kind and source of every field of the report JSON, by field name. A value's label
#: is that of the deepest field on its path that is listed here (so ``as_built_ms`` labels
#: every bucket under it). INPUT marks an operating point or shape, not a measurement.
_BUCKETS = f"worker-3 calibrated buckets ({CALIB_DIR}), per chunk for the 11 DSA layers"
LABELS = {
    "dsa_layers": f"INPUT: the calibrated model's DSA layer count ({CALIB_DIR})",
    "context": "INPUT: max_model_len priced",
    "cands": "INPUT: candidate pools, context // index_kpool",
    "prompt": "INPUT: prompt tokens",
    "column": "INPUT: worker-3 TTFT column (asbuilt, withbranches)",
    "degree": "INPUT: ranks sharing the selection",
    "design": "INPUT: design name",
    "n_chunks": "DERIVED: ceil(prompt / 1024)",
    "chunks": "DERIVED: ceil(prompt / 1024)",
    "rows_per_rank": "DERIVED: ceil(1024 / degree)",
    "cands_per_rank": "DERIVED: cands / degree",
    "kept_per_rank": "DERIVED: min(select_k, cands_per_rank)",
    "merge_rows": "DERIVED: merge top-k rows: 1024 (gather) or ceil(1024 / degree)",
    "merge_width": "DERIVED: merge top-k width, degree x kept_per_rank",
    "gather_bytes_per_layer": "DERIVED: 1024 x select_k x 4 bytes",
    "as_built_ms": f"DERIVED: {_BUCKETS}; MEASURED at cands 1024 and 2048, linear above",
    "compute_before_ms": "DERIVED: the sum of as_built_ms",
    "sharded_ms": "DERIVED: as_built_ms scaled by the rules of the module docstring",
    "compute_after_ms": "DERIVED: the sum of sharded_ms",
    "compute_saved_ms": "DERIVED: compute_before_ms - compute_after_ms",
    "comm_added_ms": f"DERIVED: allgather_ms / all_to_all_ms on {CAL_CONSTANTS}, 11 layers",
    "glue_added_ms": "DERIVED: glue_us_per_layer x 11 layers",
    "net_saved_ms": "DERIVED: compute_saved_ms - comm_added_ms - glue_added_ms",
    "query_d64_net_saved_band_ms": "DERIVED: net_saved_ms at the two ends of glue_band_us",
    "memory_per_rank_per_layer_bytes": "DERIVED: buffer shapes x 4 bytes",
    "before_ms": f"DERIVED: worker-3 prefill_model.py ({CALIB_DIR}); equal to "
                 "prefill_calibrated.json where that file has the row",
    "saved_per_chunk_ms": "DERIVED: net_saved_ms of query_d64 at the prompt's cands",
    "after_ms": "DERIVED: before_ms - n_chunks x saved_per_chunk_ms",
    "servable_as_built": f"DERIVED: worker-3 prefill_model.py admission ({CALIB_DIR})",
    "fixed_c_ms": "DERIVED: compute_after_ms summed over the chunks at the full cands",
    "prefix_sized_ms": "DERIVED: compute_after_ms summed over the chunks at prefix cands",
    "saved_ms": "DERIVED: fixed_c_ms - prefix_sized_ms",
    "alpha_ms_per_call_layer": "DERIVED: top-k fit to the two MEASURED as-built points",
    "beta_ms_per_mcycle": "DERIVED: top-k fit to the two MEASURED as-built points",
    "tok_scale": f"DERIVED: worker-3 calibration token scale ({CALIB_DIR})",
    "collective": f"DERIVED: calibrated collective constants ({CAL_CONSTANTS})",
    "fit_intercepts_ms": f"DERIVED: per-call intercepts of worker-3's fits ({CALIB_DIR})",
    "glue_us_per_layer": f"MEASURED: the row cut's segment in the profiled cut graph, us, "
                         f"median over the profiled runs ({DEVICE_RECORDS_DIR})",
    "glue_band_us": "MEASURED: the lowest and highest sharded - precut of every device "
                    "repeat (the cut and every other difference of the two compiled chains)",
    "sentinel_order_measured_ms": "MEASURED: round-2 breakdown master table, by cands",
    # device_ab: indexer_shard_device.py records in DEVICE_RECORDS_DIR, one chip (lease dsa)
    "with_causal_bound": "INPUT: whether the run's chain has the causal bound",
    "repeats": "INPUT: device runs (processes) per width",
    "measured_replicated_ms": "MEASURED: replicated net_ms (device clock less the launch "
                              "graph), median over the repeats",
    "measured_sharded_ms": "MEASURED: sharded net_ms, median over the repeats",
    "measured_precut_ms": "MEASURED: precut net_ms, median over the repeats",
    "measured_cut_ms": "MEASURED: the row cut alone, net_ms, median over the repeats",
    "measured_saved_ms": "MEASURED: replicated - sharded per repeat, median",
    "measured_sharded_less_precut_ms": "MEASURED: sharded - precut per repeat, median",
    "measured_saved_range_ms": "MEASURED: the lowest and highest saved of the repeats",
    "wall_saved_ms": "MEASURED: saved on the host clock (wall_net_ms), median",
    "launch_us": "MEASURED: the launch graph's device median, subtracted from every graph",
    "derived_saved_ms": "DERIVED: compute_saved_ms of query_d64 / dsa_layers, less the "
                        "causal_bound bucket's saving for a chain without the bound",
    "measured_over_derived": "DERIVED: measured_saved_ms / derived_saved_ms",
    "checks_pass": "MEASURED: in every repeat the rank's 16 rows select the replicated "
                   "per-row sets, the cut is bit-exact for 3 ranks, every stage took NKI, "
                   "sharded == precut",
    "record_files": f"INPUT: the run's records in {DEVICE_RECORDS_DIR}",
    "compile_failures": "MEASURED: the neuronx-cc error of the full chain at C = 65536, per "
                        "path (--compile-only logs)",
    # perf: indexer_shard_profile.py on repeat 1 of each run
    "profile": f"INPUT: the per-op profile file in {DEVICE_RECORDS_DIR}",
    "rows": "INPUT: query rows the graph selects",
    "executions": "INPUT: profiled executions",
    "execution_us": "MEASURED: device us of one profiled execution, median",
    "kind": "INPUT: what the op is (OP_KINDS)",
    "measured_us": "MEASURED: the op's profile segments, us, median over the executions",
    "sharded_less_precut_us": "MEASURED: measured_sharded_less_precut_ms of the run, us: "
                              "the cut and every other difference of the two compiled chains",
    "flops": "DERIVED: op_work FLOPs from the record's shapes",
    "bytes": "DERIVED: op_work HBM bytes from the record's shapes",
    "entitlement_us": "DERIVED: max(flops / te_peak_flops, bytes / hbm_bw_bytes_per_s)",
    "over_entitlement": "DERIVED: measured_us / entitlement_us",
    "transfer_us": "MEASURED: the id converts' copies from issue to their last DMA packet "
                   "(indexer_shard_profile.py copies), us, median over the executions",
    "transfer_over_entitlement": "DERIVED: transfer_us / entitlement_us",
    "transfer_matches": "MEASURED: most DMA bursts that fit one copy's bytes (1 = unambiguous)",
    "model_us": "DERIVED: the cost model's top-k per call and layer, alpha + beta x "
                "topk_mcycles(rows, cands), us",
    "measured_over_model": "DERIVED: measured_us / model_us",
    "rates": f"INPUT: entitlement.json meta.hardware ({ENTITLEMENT_JSON})",
    "compiler_ops": "MEASURED: every compiler op in the graph's profile (BIR group, no NKI "
                    "source): segment it sits in, opcodes, instruction count, HBM bytes, "
                    "instruction time and copy transfer (us, median over the executions)",
    "traced_nodes": "MEASURED: the traced graph's nodes (record graph_ops) by target and "
                    "outputs, with counts",
    # allgather_two_ranks: nccom-test on lease dsa (DEVICE_RECORDS_DIR/allgather_*)
    "allgather_two_ranks": "MEASURED: nccom-test allg, 2 ranks of one chip, output verified "
                           "(-c random), 50 iterations after 10 warmup (allgather_command.txt)",
    "p50_us": "MEASURED: the p50 time per call, us, by dtype and gathered (output) bytes",
    "int32_over_fp32": "DERIVED: the int32 p50 / the fp32 p50, per gathered bytes",
    "chunk_ids_bytes": "DERIVED: the bytes one layer's all-gather returns, tokens x select_k "
                       "x 4 (int32)",
    "check_files": f"INPUT: the check's outputs in {DEVICE_RECORDS_DIR}",
}


def report(cost: IndexerCost | None = None) -> dict:
    cost = cost or IndexerCost()
    band = measured_glue_band_us()
    ab = device_ab(cost)
    out = {
        "what": "The query-sharded DSA prefill selection: its DERIVED cost per 1024-row "
                "chunk, TP=64, all 11 DSA layers (per_layer = /11; source of every as-built "
                f"value: {CALIB_DIR}, worker-3 calibrated buckets), its MEASURED device A/B "
                f"on one chip (device_ab), its per-op device time against entitlement "
                f"(perf) and a two-rank all-gather check (allgather_two_ranks), all from "
                f"{DEVICE_RECORDS_DIR}. This is the evidence file of "
                "/home/ubuntu/glm53f-wt5/reports/indexer_shard.md, not an integrator row "
                "file: 'labels' gives the kind and source of every field, by field name.",
        "labels": LABELS,
        "dsa_layers": cost.layers,
        "device_ab": ab,
        "perf": perf(ab, cost),
        "allgather_two_ranks": allgather_check(),
        "constants": {"alpha_ms_per_call_layer": cost.alpha,
                      "beta_ms_per_mcycle": cost.beta, "tok_scale": cost.m.tok_scale,
                      "glue_us_per_layer": cost.glue_us,
                      "glue_band_us": band, "collective": cost.cc,
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
                  for g in band)
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
