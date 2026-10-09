# SPDX-License-Identifier: Apache-2.0
"""The DCP candidate merge, ``dsa_dcp_merge_select``, on Neuron: identity, exactness, time.

Set the Neuron cores before launching (``devlease.py slice`` does, with
``NEURON_LOGICAL_NC_CONFIG=2``). This script does not select cores. References are computed on
the CPU; compilation and warmup are excluded from the timings.

**Inputs.** A case builds what the ``dcp_size`` ranks of a KV group would hand the merge. Full
score rows ``[R, N]`` over ``N = CP * --local-pools`` global pools are bounded the way the causal
bound bounds them (``BOUND_FILL`` past a row's complete pools), split by ownership (128-token
block ``b`` on rank ``b % CP``, ``block_size // index_kpool`` pools a block; the map is written
out here, not read from the module), and selected per rank by one of two local rules:

* ``prefill``: the real ``dsa_topk_select`` on the CPU, then the causal sentinel (its torch
  reference): values and ids in the selector's descending order, ``-1`` where a slot reaches
  no complete pool;
* ``decode``: the decode select's rule (``decode_select.py``: values ``> v`` and, of the values
  equal to the ``k``-th ``v``, the lowest local index first; nothing at or below
  ``BOUND_FILL_MARK``), emitted in ascending local pool order with the pads last, as the
  decode select emits them.

The patterns: ``random`` (bfloat16-rounded scores, so values tie), ``tie_two_ranks`` and
``tie_one_rank`` (fp32 scores with a group of equal values planted at the ``k``-th place, its
pools owned by two ranks or by the ``k``-th pool's own rank), ``short`` (every row has fewer
than ``k`` complete pools, so every rank pads), ``empty_some`` (few enough complete pools that
the last rank owns none), ``all_empty`` (none) and ``causal`` (row ``i`` of ``R`` sees
``(i + 1) * N / R`` pools: every kind in one tensor). ``garbage_pads`` is ``random`` with every
pad's value replaced by a large positive number, which the merge must not read.
``one_core_ties`` plants the two-rank tie only in the rows the second core of an LNC2 pair
owns, so at ``R >= 2`` one core's tie branch is taken and the other's is not: the two cores
must still take the branch together (a core alone in it waits on the loop's two-core barrier
for good). Every case records each core's own branch condition and ``cores_disagree``.
``value_pads`` is ``short`` with every pad given a real local id the row does not hold, so
only its ``BOUND_FILL`` value marks it. Every case records its pads per row (id ``-1`` or a
value at or below ``BOUND_FILL_MARK``) and ``pads_emitted``, the output slots that hold an id
where the input held a pad, which must be 0. The two rules give the two input orders: value
descending (``prefill``) and ascending local id (``decode``).

**Identity.** Every rank's call, one graph per rank of the public entry point, is compared with
``dsa_dcp_merge_select_torch_oracle`` bit for bit at every ``(R, CP)``, pattern, rule and seed;
the ``random`` case of the first seed runs ``--emissions`` times per rank, every emission bit
for bit equal. The dispatch counters, which count when a graph is traced, must show each
rank's graph tracing the NKI kernel once and never retracing. Every case's device outputs go
to ``--pt-dir``, with its operands up to ``--save-inputs-rows`` rows.

**Exactness against CP=1.** The ranks' outputs, mapped back to global ids, form the merged
set; CP=1's set is the same rule over the whole row. Where CP=1's ``k``-th value has no tie at
its boundary (or the row has fewer than ``k`` complete pools) the sets must be equal; at a
boundary tie the selected scores must be equal as a multiset. Under the decode rule the sets
must be equal everywhere (each rank's lowest local indices are its lowest global ids). The
check runs on the device outputs and on the oracle's; the tie statistics are recorded.

**Glue.** ``dsa_dcp_select_candidates`` runs on the device with a stand-in group whose
``all_gather`` concatenates every rank's operands, and must equal the kernel's output.

**Time.** Per shape, on inputs without an excess tie (``fast``) and with one in every row
(``ties``, the tie-break branch taken): a one-call graph, an ``--links``-call graph of
independent calls on distinct operand copies, and an A/A copy of the one-call graph whose
difference from the first is the noise floor. Per call = ``(T_L - T_1) / (L - 1)``, paired per
iteration, from the runtime system trace (LNC2 physical-core intervals merged). Floors, timed
the same way: ``floor_copy`` (the merge's own loads at its own tiling, no compute, and every
loaded tile stored back, so twice the operand bytes) and ``floor_launch`` (one row in and out
per program).

Every run compiles from scratch: ``--cache-root`` must name a new or empty directory, which
becomes ``NEURON_LIBTORCH_CACHE_ROOT``, and the compile cache is disabled on top.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import statistics
import sys
import time
from pathlib import Path

#: This file's repository. Run as a script, Python puts ``test/hardware`` on ``sys.path`` and the
#: venv resolves ``vllm_neuron`` to another checkout; the module under test must be this tree's.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

import vllm_neuron  # noqa: E402,F401 -- registers the Neuron compilation backend
import nki  # noqa: E402
import nki.isa as nisa  # noqa: E402
import nki.language as nl  # noqa: E402
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki  # noqa: E402

from vllm_neuron.functional.dsa import causal_bound as cb  # noqa: E402
from vllm_neuron.functional.dsa import dcp_candidate_merge as merge  # noqa: E402
from vllm_neuron.functional.dsa.topk_select import dsa_topk_select  # noqa: E402

DEVICE = "neuron:0"
PATTERNS = ("random", "tie_two_ranks", "tie_one_rank", "short", "empty_some", "all_empty",
            "causal", "garbage_pads", "one_core_ties", "value_pads")
RULES = ("prefill", "decode")
#: Pools of a planted boundary tie.
PLANTED_TIE = 8
#: The value a ``garbage_pads`` pad carries: far above every score.
GARBAGE = 1.0e4


# ---------------------------------------------------------------------------------------------
# Ownership, written out independently of the module
# ---------------------------------------------------------------------------------------------


def owner_columns(n_global: int, cp: int, ppb: int) -> list[torch.Tensor]:
    """Per rank, the global pool ids it holds in local order: block ``b`` of ``ppb`` pools on
    rank ``b % cp`` as its local block ``b // cp``."""
    blocks = torch.arange(n_global) // ppb
    cols = []
    for rank in range(cp):
        cols.append(torch.nonzero(blocks % cp == rank).flatten())
    return cols


# ---------------------------------------------------------------------------------------------
# Score rows and local selections
# ---------------------------------------------------------------------------------------------


def second_program_rows(rows: int) -> list[int]:
    """The rows the second program of an LNC2 launch stores (``merge.program_groups``)."""
    owned = []
    for stores, tiles in merge.program_groups(rows, 2)[1]:
        if stores:
            for t0, n in tiles:
                owned.extend(range(t0, t0 + n))
    return owned


def branch_flags(values: torch.Tensor, ids: torch.Tensor, k: int) -> list[list[bool]]:
    """Per program of an LNC2 launch, per group, whether a row of the group has an excess tie
    (more than ``k`` candidates at or above a ``k``-th value some valid candidate holds): the
    kernel's tie-branch condition on each core before the two cores combine it."""
    cp, rows, _ = values.shape
    vals = values.permute(1, 0, 2).reshape(rows, cp * k)
    valid = ids.permute(1, 0, 2).reshape(rows, cp * k) >= 0
    vals = torch.where(valid, vals, torch.full_like(vals, merge.LOWEST))
    kth = torch.topk(vals, k, dim=1).values[:, -1]
    excess = ((vals >= kth[:, None]).sum(1) > k) & (kth > merge.LOWEST)
    return [[bool(any(excess[t0:t0 + n].any() for t0, n in tiles)) for _, tiles in groups]
            for groups in merge.program_groups(rows, 2)]


def score_rows(pattern: str, rows: int, n_global: int, cp: int, ppb: int, k: int,
               seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(scores [rows, n_global] fp32, complete [rows])``: a pattern's raw scores and each row's
    count of complete pools."""
    g = torch.Generator().manual_seed(seed)
    scores = torch.randn(rows, n_global, generator=g)
    complete = torch.full((rows,), n_global, dtype=torch.int64)
    if pattern in ("random", "garbage_pads", "short", "empty_some", "causal", "value_pads"):
        scores = scores.to(torch.bfloat16).float()
    if pattern in ("short", "value_pads"):
        complete = torch.randint(1, k, (rows,), generator=g)
    elif pattern == "empty_some":
        complete = torch.randint(1, (cp - 1) * ppb + 1, (rows,), generator=g)
    elif pattern == "all_empty":
        complete = torch.zeros(rows, dtype=torch.int64)
    elif pattern == "causal":
        complete = (torch.arange(1, rows + 1) * n_global) // rows
    elif pattern in ("tie_two_ranks", "tie_one_rank", "one_core_ties"):
        owner = (torch.arange(n_global) // ppb) % cp
        order = torch.argsort(scores, dim=1, descending=True)
        tied_rows = range(rows)
        if pattern == "one_core_ties":
            tied_rows = second_program_rows(rows)
        for row in tied_rows:
            kth = int(order[row, k - 1])
            ranks = (int(owner[kth]),)
            if pattern != "tie_one_rank":
                ranks = (int(owner[kth]), (int(owner[kth]) + 1) % cp)
            below = order[row, k:]
            picked = []
            for rank in ranks:
                mine = below[owner[below] == rank]
                picked.append(mine[:PLANTED_TIE // len(ranks)])
            scores[row, torch.cat(picked)] = float(scores[row, kth])
    return scores, complete


def bounded(scores: torch.Tensor, complete: torch.Tensor) -> torch.Tensor:
    """``BOUND_FILL`` at every pool past a row's complete ones, the causal bound's output."""
    cols = torch.arange(scores.shape[1])[None, :]
    return scores.masked_fill(cols >= complete[:, None], cb.BOUND_FILL)


def select_prefill(local: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The prefill rule: ``dsa_topk_select`` (its CPU route) then the causal sentinel's torch
    reference (its gate does not look at the device, so the entry point would launch NKI on
    CPU tensors)."""
    values, indices = dsa_topk_select(local, k)
    ids = cb.dsa_causal_sentinel_torch_oracle(values, indices.to(torch.int32),
                                              int(local.shape[1]))
    return values.contiguous(), ids.contiguous()


def select_decode(local: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The decode rule (lowest local index first among equal values, nothing at or below
    ``BOUND_FILL_MARK``), emitted in ascending pool order with the pads last."""
    width = int(local.shape[1])
    # Stable: equal values keep ascending index order.
    order = torch.sort(local, dim=1, descending=True, stable=True).indices[:, :k]
    values = torch.gather(local, 1, order)
    real = values > cb.BOUND_FILL_MARK
    key = torch.where(real, order, torch.full_like(order, width))
    emit = torch.argsort(key, dim=1, stable=True)
    ids = torch.where(real, order, torch.full_like(order, cb.SENTINEL)).gather(1, emit)
    values = torch.where(real, values, torch.full_like(values, cb.BOUND_FILL)).gather(1, emit)
    return values.contiguous(), ids.to(torch.int32).contiguous()


SELECT = {"prefill": select_prefill, "decode": select_decode}


def relabel_pads(ids: torch.Tensor, width: int) -> torch.Tensor:
    """``ids`` with every ``-1`` replaced by a local id of ``[0, width)`` its row does not hold,
    the lowest free ones in order: pads that only their ``BOUND_FILL`` value marks."""
    out = ids.clone()
    for row in range(int(ids.shape[0])):
        pad = torch.nonzero(ids[row] < 0).flatten()
        if pad.numel() == 0:
            continue
        held = torch.zeros(width, dtype=torch.bool)
        held[ids[row][ids[row] >= 0].to(torch.int64)] = True
        out[row, pad] = torch.nonzero(~held).flatten()[:pad.numel()].to(ids.dtype)
    return out


def pad_mask(values: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """``[CP, R, k]`` bool: the pads, an id of ``-1`` or a value at or below ``BOUND_FILL_MARK``."""
    return (ids < 0) | (values <= cb.BOUND_FILL_MARK)


def build_case(pattern: str, rule: str, rows: int, cp: int, k: int, local_pools: int, ppb: int,
               seed: int) -> dict:
    """Every rank's merge operands for one case, and the CP=1 reference over the whole row."""
    n_global = cp * local_pools
    scores, complete = score_rows(pattern, rows, n_global, cp, ppb, k, seed)
    row_scores = bounded(scores, complete)
    cols = owner_columns(n_global, cp, ppb)
    values, ids = [], []
    for rank in range(cp):
        v, i = SELECT[rule](row_scores[:, cols[rank]].contiguous(), k)
        if pattern == "garbage_pads":
            v = torch.where(i < 0, torch.full_like(v, GARBAGE), v)
        if pattern == "value_pads":
            i = relabel_pads(i, int(cols[rank].numel()))
        values.append(v)
        ids.append(i)
    _, ref_ids = SELECT[rule](row_scores, k)
    return {"values": torch.stack(values).contiguous(), "ids": torch.stack(ids).contiguous(),
            "row_scores": row_scores, "ref_ids": ref_ids, "cols": cols, "n_global": n_global}


# ---------------------------------------------------------------------------------------------
# Exactness
# ---------------------------------------------------------------------------------------------


def global_mask(per_rank: list[torch.Tensor], cols: list[torch.Tensor],
                n_global: int) -> torch.Tensor:
    """``[R, n_global]`` bool: the global pools the ranks' outputs (local ids, ``-1`` elsewhere)
    select."""
    rows = int(per_rank[0].shape[0])
    mask = torch.zeros(rows, n_global, dtype=torch.bool)
    for rank, out in enumerate(per_rank):
        real = out >= 0
        gid = cols[rank][out.clamp(min=0).to(torch.int64)]
        mask[torch.nonzero(real, as_tuple=True)[0], gid[real]] = True
    return mask


def exactness(merged: torch.Tensor, case: dict, rule: str, k: int) -> dict:
    """The criterion on one case, with its tie statistics."""
    scores = case["row_scores"]
    rows, n_global = scores.shape
    ref = torch.zeros(rows, n_global, dtype=torch.bool)
    real = case["ref_ids"] >= 0
    ref[torch.nonzero(real, as_tuple=True)[0], case["ref_ids"][real].to(torch.int64)] = True
    valid = scores > cb.BOUND_FILL_MARK
    n_valid = valid.sum(1)
    kth = torch.where(n_valid >= k,
                      torch.topk(torch.where(valid, scores, torch.full_like(scores, -torch.inf)),
                                 k, dim=1).values[:, k - 1],
                      torch.full((rows,), -torch.inf))
    at_least = (valid & (scores >= kth[:, None])).sum(1)
    tie_rows = (n_valid >= k) & (at_least > k)
    tie_size = (valid & (scores == kth[:, None])).sum(1)
    same_set = (merged == ref).all(1)

    def chosen_scores(mask):
        return torch.topk(torch.where(mask, scores, torch.full_like(scores, -torch.inf)), k,
                          dim=1).values

    same_scores = (chosen_scores(merged) == chosen_scores(ref)).all(1)
    same_multiset = (merged.sum(1) == ref.sum(1)) & same_scores
    if rule == "decode":
        ok = bool(same_set.all())
    else:
        ok = bool((same_set | (tie_rows & same_multiset)).all())
    return {"ok": ok, "rows": rows, "rows_with_boundary_tie": int(tie_rows.sum()),
            "boundary_tie_size_max": int(tie_size[tie_rows].max()) if bool(tie_rows.any()) else 0,
            "boundary_tie_size_mean": (float(tie_size[tie_rows].float().mean())
                                       if bool(tie_rows.any()) else 0.0),
            "rows_set_differs": int((~same_set).sum()),
            "rows_set_differs_without_tie": int((~same_set & ~tie_rows).sum()),
            "rows_multiset_differs": int((~same_multiset).sum()),
            "rows_short": int((n_valid < k).sum()), "rows_empty": int((n_valid == 0).sum())}


# ---------------------------------------------------------------------------------------------
# Floors
# ---------------------------------------------------------------------------------------------


@nki.jit
def copy_floor(values_hbm, ids_hbm, cp_rank):
    """The merge's loads at its own tiling and fold, and no compute; every loaded tile is
    stored back, so no load is left without a reader. It moves twice the operand bytes."""
    cp = values_hbm.shape[0]
    rows = values_hbm.shape[1]
    k = values_hbm.shape[2]
    vals_out = nl.ndarray((cp, rows, k), dtype=nl.float32, buffer=nl.shared_hbm)
    ids_out = nl.ndarray((cp, rows, k), dtype=nl.int32, buffer=nl.shared_hbm)
    for group in merge.program_groups(rows, nl.num_programs(axes=0))[nl.program_id(0)]:
        for tile in group[1]:
            r0 = tile[0]
            n = tile[1]
            f = merge.tile_fold(n, cp, k)
            pieces = merge.fold_groups(f, cp)[2]
            width = (cp * k) // f
            values_runs = values_hbm.reshape((cp, rows * pieces, k // pieces))
            ids_runs = ids_hbm.reshape((cp, rows * pieces, k // pieces))
            vals_back = vals_out.reshape((cp, rows * pieces, k // pieces))
            ids_back = ids_out.reshape((cp, rows * pieces, k // pieces))
            vals = nl.ndarray((n * f, width), dtype=nl.float32, buffer=nl.sbuf)
            lids = nl.ndarray((n * f, width), dtype=nl.int32, buffer=nl.sbuf)
            for rank in range(cp):
                slot = merge.rank_slot(f, cp, k, n, rank)
                nisa.dma_copy(dst=vals[slot[0]:slot[0] + slot[1], slot[2]:slot[2] + slot[3]],
                              src=values_runs[rank, r0 * pieces:(r0 + n) * pieces, 0:slot[3]])
                nisa.dma_copy(dst=lids[slot[0]:slot[0] + slot[1], slot[2]:slot[2] + slot[3]],
                              src=ids_runs[rank, r0 * pieces:(r0 + n) * pieces, 0:slot[3]])
            if group[0]:
                for rank in range(cp):
                    slot = merge.rank_slot(f, cp, k, n, rank)
                    nisa.dma_copy(
                        dst=vals_back[rank, r0 * pieces:(r0 + n) * pieces, 0:slot[3]],
                        src=vals[slot[0]:slot[0] + slot[1], slot[2]:slot[2] + slot[3]])
                    nisa.dma_copy(
                        dst=ids_back[rank, r0 * pieces:(r0 + n) * pieces, 0:slot[3]],
                        src=lids[slot[0]:slot[0] + slot[1], slot[2]:slot[2] + slot[3]])
    return vals_out, ids_out


@nki.jit
def launch_floor(values_hbm, ids_hbm, cp_rank):
    """An NKI kernel's fixed cost at the merge's operands: one row in and out per program."""
    rows = values_hbm.shape[1]
    k = values_hbm.shape[2]
    out_hbm = nl.ndarray((rows, k), dtype=nl.int32, buffer=nl.shared_hbm)
    first = merge.program_groups(rows, nl.num_programs(axes=0))[nl.program_id(0)][0][1][0][0]
    row = nl.ndarray((1, k), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=row, src=ids_hbm[cp_rank, first:first + 1, 0:k])
    nisa.dma_copy(dst=out_hbm[first:first + 1, 0:k], src=row)
    return out_hbm


# ---------------------------------------------------------------------------------------------
# Graphs and timing
# ---------------------------------------------------------------------------------------------


def compile_fn(fn):
    return torch.compile(fn, backend="neuron_libtorch", fullgraph=True, dynamic=False)


def merge_call(cp_rank: int, ppb: int):
    """The public entry point for one rank."""
    def call(values, ids):
        return merge.dsa_dcp_merge_select(values, ids, cp_rank=cp_rank, pools_per_block=ppb)
    return call


def floor_call(kernel, cp_rank: int):
    def call(values, ids):
        launch = wrap_nki(kernel)
        if merge.merge_programs() == 2:
            launch = launch[2]
        return launch(values, ids, cp_rank)
    return call


def many_graph(call, links: int):
    """``links`` independent calls on distinct operand copies, every output returned."""
    def many(*operands):
        return tuple(call(operands[2 * i], operands[2 * i + 1]) for i in range(links))
    return compile_fn(many)


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


def _first(result):
    while isinstance(result, tuple):
        result = result[0]
    return result


def time_graphs(graphs: dict, inputs: dict, warmup: int, iterations: int) -> dict:
    """Every graph's device samples in us, the calls rotated across graphs per iteration."""
    from nrtpy._nrtpy import SystemTraceSession

    names = list(graphs)
    for _ in range(warmup):
        for name in names:
            _first(graphs[name](*inputs[name])).to("cpu")
    order = []
    with SystemTraceSession() as trace:
        for iteration in range(iterations):
            shift = iteration % len(names)
            for name in names[shift:] + names[:shift]:
                _first(graphs[name](*inputs[name])).to("cpu")
                order.append(name)
        events_json = trace.fetch_events_json()
    device_all = device_intervals(events_json)
    if len(device_all) != len(order):
        raise AssertionError(
            f"system trace has {len(device_all)} executions for {len(order)} calls")
    device = {name: [] for name in names}
    for name, value in zip(order, device_all):
        device[name].append(value)
    return device


def rep_stats(reps: list[list[float]]) -> dict:
    medians = [statistics.median(samples) for samples in reps]
    return {**stats_us([v for samples in reps for v in samples]), "rep_medians_us": medians,
            "rep_range_us": [min(medians), max(medians)]}


# ---------------------------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------------------------


def digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


def identity_section(args, rows: int, cp: int) -> dict:
    """Device against oracle for every pattern, rule and seed at one shape; exactness against
    CP=1; the emissions; the tensors of the first case to ``--pt-dir``."""
    k, ppb = args.k, args.pools_per_block
    graphs = [compile_fn(merge_call(rank, ppb)) for rank in range(cp)]
    out = {"rows": rows, "cp": cp, "cases": [], "emissions": {}, "routes": []}
    saved = {"device": {}}
    for pattern in PATTERNS:
        for rule in RULES:
            for seed in args.seeds:
                case = build_case(pattern, rule, rows, cp, k, args.local_pools, ppb, seed)
                v_dev = case["values"].to(DEVICE)
                i_dev = case["ids"].to(DEVICE)
                device, oracle = [], []
                for rank in range(cp):
                    # The counters count at trace time: a graph's first call traces it, and a
                    # later call that retraced would count again.
                    merge.reset_dcp_merge_select_dispatch_counters()
                    device.append(graphs[rank](v_dev, i_dev).to("cpu"))
                    counters = merge.dcp_merge_select_dispatch_counters()
                    if len(out["routes"]) <= rank:
                        kernel = merge.dcp_merge_select_kernel_identity() or ()
                        out["routes"].append({"cp_rank": rank,
                                              "counters_at_trace": list(counters),
                                              "kernel": list(kernel)})
                    elif counters != (0, 0):
                        out["routes"][rank]["retraced"] = True
                    oracle.append(merge.dsa_dcp_merge_select_torch_oracle(
                        case["values"], case["ids"], cp_rank=rank, pools_per_block=ppb))
                mismatches = [int((d != o).sum()) for d, o in zip(device, oracle)]
                flags = branch_flags(case["values"], case["ids"], k)
                pads = pad_mask(case["values"], case["ids"])
                pads_per_row = pads.sum(dim=(0, 2))
                record = {
                    "pattern": pattern, "rule": rule, "seed": seed,
                    "branch_flags_per_program": flags, "cores_disagree": flags[0] != flags[1],
                    "pads_per_row": {"min": int(pads_per_row.min()),
                                     "max": int(pads_per_row.max()),
                                     "mean": float(pads_per_row.float().mean())},
                    "rows_fewer_than_k_real": int((cp * k - pads_per_row < k).sum()),
                    "pads_emitted": sum(int(((d >= 0) & pads[r]).sum())
                                        for r, d in enumerate(device)),
                    "device_equals_oracle": all(m == 0 for m in mismatches),
                    "mismatches_per_rank": mismatches,
                    "exact_device": exactness(global_mask(device, case["cols"], case["n_global"]),
                                              case, rule, k),
                    "exact_oracle": exactness(global_mask(oracle, case["cols"], case["n_global"]),
                                              case, rule, k),
                    "output_digests": [digest(d) for d in device],
                }
                out["cases"].append(record)
                saved["device"][f"{pattern}/{rule}/{seed}"] = torch.stack(device)
                if rows <= args.save_inputs_rows:
                    saved[f"{pattern}/{rule}/{seed}/values"] = case["values"]
                    saved[f"{pattern}/{rule}/{seed}/ids"] = case["ids"]
                if pattern == "random" and rule == "prefill" and seed == args.seeds[0]:
                    for rank in range(cp):
                        first = device[rank]
                        same = [bool(torch.equal(graphs[rank](v_dev, i_dev).to("cpu"), first))
                                for _ in range(args.emissions - 1)]
                        out["emissions"][str(rank)] = {
                            "emissions": args.emissions, "all_identical": all(same),
                            "equal_oracle": bool(torch.equal(first, oracle[rank]))}
                print(json.dumps({"rows": rows, "cp": cp, "pattern": pattern, "rule": rule,
                                  "seed": seed, "equal": record["device_equals_oracle"],
                                  "exact": record["exact_device"]["ok"],
                                  "tie_rows": record["exact_device"]["rows_with_boundary_tie"],
                                  "cores_disagree": record["cores_disagree"]}),
                      flush=True)
    path = args.pt_dir / f"identity_r{rows}_cp{cp}.pt"
    torch.save(saved, path)
    out["pt"] = str(path)
    return out


class StandInGroup:
    """A KV group of ``len(operands)`` ranks for one process: ``all_gather`` concatenates every
    rank's operand of the same kind on dim 0, this rank's being the one it is handed."""

    def __init__(self, rank: int, values: list, ids: list):
        self.rank_in_group = rank
        self.world_size = len(values)
        self._by_dtype = {torch.float32: values, torch.int32: ids}

    def all_gather(self, x, dim=0):
        parts = list(self._by_dtype[x.dtype])
        parts[self.rank_in_group] = x
        return torch.cat(parts, dim=dim)


def glue_section(args, rows: int, cp: int) -> dict:
    """``dsa_dcp_select_candidates`` against the kernel, one rank's graph per rank."""
    k, ppb = args.k, args.pools_per_block
    case = build_case("random", "prefill", rows, cp, k, args.local_pools, ppb, args.seeds[0])
    values = [case["values"][r].to(DEVICE) for r in range(cp)]
    ids = [case["ids"][r].to(DEVICE) for r in range(cp)]
    equal = []
    for rank in range(cp):
        def glue(*operands, rank=rank):
            group = StandInGroup(rank, list(operands[:cp]), list(operands[cp:]))
            return merge.dsa_dcp_select_candidates(
                operands[rank], operands[cp + rank], dcp_group=group, cp_rank=rank,
                dcp_size=cp, block_size=args.block_size, index_kpool=args.index_kpool)
        got = compile_fn(glue)(*values, *ids).to("cpu")
        want = merge.dsa_dcp_merge_select_torch_oracle(case["values"], case["ids"], cp_rank=rank,
                                                      pools_per_block=ppb)
        equal.append(bool(torch.equal(got, want)))
    return {"rows": rows, "cp": cp, "block_size": args.block_size,
            "index_kpool": args.index_kpool, "equal_oracle_per_rank": equal}


def perf_operands(path: str, rows: int, cp: int, args) -> tuple[torch.Tensor, torch.Tensor]:
    """``fast``: fp32 scores, no excess tie; ``ties``: an excess tie in every row."""
    pattern = "random" if path == "fast" else "tie_two_ranks"
    case = build_case(pattern, "prefill", rows, cp, args.k, args.local_pools,
                      args.pools_per_block, args.seeds[0])
    if path == "fast":
        g = torch.Generator().manual_seed(args.seeds[0])
        values = torch.randn(case["values"].shape, generator=g)
        values = torch.where(case["ids"] < 0, case["values"], values)
        return values.contiguous(), case["ids"]
    return case["values"], case["ids"]


def perf_section(args, rows: int, cp: int) -> dict:
    """Per-call time of rank ``cp - 1``'s merge and of the floors at one shape."""
    rank = cp - 1
    calls = {"merge": merge_call(rank, args.pools_per_block),
             "floor_copy": floor_call(copy_floor, rank),
             "floor_launch": floor_call(launch_floor, rank)}
    graphs, inputs = {}, {}
    for path in ("fast", "ties"):
        values, ids = perf_operands(path, rows, cp, args)
        copies = []
        for _ in range(args.links):
            copies += [values.to(DEVICE), ids.to(DEVICE)]
        for name, call in calls.items():
            if name != "merge" and path == "ties":
                continue
            label = f"{name}_{path}" if name == "merge" else name
            graphs[f"{label}|1"] = compile_fn(call)
            inputs[f"{label}|1"] = tuple(copies[:2])
            graphs[f"{label}|L"] = many_graph(call, args.links)
            inputs[f"{label}|L"] = tuple(copies)
            if name == "merge":
                graphs[f"{label}_aa|1"] = compile_fn(call)
                inputs[f"{label}_aa|1"] = tuple(copies[:2])
        got = graphs[f"merge_{path}|1"](*inputs[f"merge_{path}|1"]).to("cpu")
        want = merge.dsa_dcp_merge_select_torch_oracle(values, ids, cp_rank=rank,
                                                      pools_per_block=args.pools_per_block)
        if not torch.equal(got, want):
            raise AssertionError(f"perf operands {path} at R={rows} CP={cp}: kernel != oracle")
    for name in list(graphs):
        _first(graphs[name](*inputs[name])).to("cpu")
    reps = [time_graphs(graphs, inputs, args.warmup, args.iterations) for _ in range(args.reps)]
    out = {"rows": rows, "cp": cp, "cp_rank": rank, "links": args.links, "variants": {}}
    labels = sorted({name.split("|")[0] for name in graphs if not name.endswith("_aa|1")})
    for label in labels:
        one = [d[f"{label}|1"] for d in reps]
        many = [d[f"{label}|L"] for d in reps]
        per_call = [[(m - o) / (args.links - 1) for m, o in zip(m_rep, o_rep)]
                    for m_rep, o_rep in zip(many, one)]
        entry = {"one_call_graph": rep_stats(one), "per_call": rep_stats(per_call)}
        if label.startswith("merge_"):
            aa = [d[f"{label}_aa|1"] for d in reps]
            entry["one_call_graph_aa"] = rep_stats(aa)
            entry["noise_floor_one_call_abs_diff"] = stats_us(
                [abs(a - b) for a_rep, b_rep in zip(one, aa) for a, b in zip(a_rep, b_rep)])
        out["variants"][label] = entry
    print(json.dumps({"rows": rows, "cp": cp, **{label: round(v["per_call"]["median_us"], 1)
                                                for label, v in out["variants"].items()}}),
          flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pt-dir", type=Path, help="tensors (default: --out's directory)")
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8, 64, 512, 1024, 2048])
    parser.add_argument("--cp", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--perf-rows", type=int, nargs="*", default=[1, 8, 64, 1024, 2048])
    parser.add_argument("--k", type=int, default=512, help="index_topk // index_kpool")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--index-kpool", type=int, default=4)
    parser.add_argument("--local-pools", type=int, default=2048,
                        help="pools each rank holds per row (the scores' width per rank)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--emissions", type=int, default=8)
    parser.add_argument("--save-inputs-rows", type=int, default=64,
                        help="save every case's operands to --pt-dir at row counts up to this")
    parser.add_argument("--links", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--skip-identity", action="store_true")
    parser.add_argument("--cache-root", type=Path, required=True,
                        help="a new or empty directory: NEURON_LIBTORCH_CACHE_ROOT for this run")
    parser.add_argument("--time-limit", type=int, default=7200,
                        help="seconds before the run aborts")
    args = parser.parse_args()
    signal.alarm(args.time_limit)
    args.cache_root = args.cache_root.resolve()
    if args.cache_root.exists() and any(args.cache_root.iterdir()):
        raise ValueError(f"--cache-root {args.cache_root} is not empty; name a fresh directory")
    args.cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["NEURON_LIBTORCH_CACHE_ROOT"] = str(args.cache_root)
    os.environ["NEURON_LIBTORCH_DISABLE_COMPILE_CACHE"] = "1"
    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1":
        raise ValueError("Hardware benchmark cannot run in VLLM_NEURON_CPU_MODE=1")
    if not os.environ.get("NEURON_RT_VISIBLE_CORES"):
        raise ValueError("Pin NEURON_RT_VISIBLE_CORES before running the benchmark")
    if os.environ.get("NEURON_LOGICAL_NC_CONFIG") != "2":
        raise ValueError("The served configuration is NEURON_LOGICAL_NC_CONFIG=2")
    if args.links < 2 or args.iterations < 5 or args.reps < 1 or args.emissions < 2:
        raise ValueError("Use --links >= 2, --iterations >= 5, --reps >= 1, --emissions >= 2")
    if args.block_size % args.index_kpool != 0:
        raise ValueError("--block-size must be a whole number of --index-kpool pools")
    if not Path(merge.__file__).resolve().is_relative_to(REPO_ROOT):
        raise RuntimeError(f"{merge.__name__} imported from {merge.__file__}")
    args.pools_per_block = args.block_size // args.index_kpool
    args.out = args.out.resolve()
    args.pt_dir = (args.pt_dir or args.out.parent).resolve()
    args.pt_dir.mkdir(parents=True, exist_ok=True)
    torch._dynamo.config.cache_size_limit = 1024
    # The NKI and neuronx-cc drivers write artifacts into the working directory.
    scratch = args.cache_root / "bench-cwd"
    scratch.mkdir(parents=True, exist_ok=True)
    os.chdir(scratch)
    report = {
        "environment": {key: os.environ.get(key) for key in (
            "NEURON_RT_VISIBLE_CORES", "NEURON_LOGICAL_NC_CONFIG",
            "NEURON_PLATFORM_TARGET_OVERRIDE", "NEURON_LIBTORCH_CACHE_ROOT",
            "NEURON_LIBTORCH_DISABLE_COMPILE_CACHE")},
        "device": "neuron:0 = one logical core (LNC2: 2 physical cores) of the slice",
        "module": merge.__file__, "source_digest": merge.SOURCE_DIGEST,
        "k": args.k, "block_size": args.block_size, "index_kpool": args.index_kpool,
        "local_pools": args.local_pools, "seeds": args.seeds,
        "method": ("per-call = (L-call graph - 1-call graph) / (L - 1), L independent calls on "
                   "distinct operand copies, paired per iteration; device time from the runtime "
                   "system trace (LNC2 physical-core intervals merged)"),
        "iterations": args.iterations, "warmup": args.warmup, "reps": args.reps,
        "identity": [], "glue": [], "perf": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def write():
        args.out.write_text(json.dumps(report, indent=1) + "\n")

    t0 = time.time()
    if not args.skip_identity:
        for cp in args.cp:
            report["glue"].append(glue_section(args, 64, cp))
            write()
            for rows in args.rows:
                report["identity"].append(identity_section(args, rows, cp))
                write()
    for cp in args.cp:
        for rows in args.perf_rows:
            report["perf"].append(perf_section(args, rows, cp))
            write()
    report["seconds"] = round(time.time() - t0, 1)
    write()
    bad = [(c["rows"], c["cp"], case["pattern"], case["rule"], case["seed"])
           for c in report["identity"] for case in c["cases"]
           if not (case["device_equals_oracle"] and case["exact_device"]["ok"]
                   and case["exact_oracle"]["ok"] and case["pads_emitted"] == 0)]
    routes_ok = all(r["counters_at_trace"] == [1, 0] and not r.get("retraced")
                    for c in report["identity"] for r in c["routes"])
    print(json.dumps({"identity_cases": sum(len(c["cases"]) for c in report["identity"]),
                      "cores_disagree_cases": sum(case["cores_disagree"]
                                                  for c in report["identity"]
                                                  for case in c["cases"]),
                      "failed": bad, "routes_nki_once_per_graph": routes_ok,
                      "emissions_identical": all(e["all_identical"] and e["equal_oracle"]
                                                 for c in report["identity"]
                                                 for e in c["emissions"].values()),
                      "glue_equal": all(all(g["equal_oracle_per_rank"]) for g in report["glue"])}),
          flush=True)


if __name__ == "__main__":
    main()
