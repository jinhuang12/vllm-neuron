# SPDX-License-Identifier: Apache-2.0
"""Qualification of the DCP candidate merge at the decode-context-parallel widths.

``dsa_dcp_merge_select`` (``vllm_neuron/functional/dsa/dcp_candidate_merge.py``) merges the
``dcp_size`` ranks' local top-``k`` pool selections into the selection one rank makes over
the whole pool axis. This file checks the kernel, rank by rank and bit for bit, against a
CPU reference written here from the rule alone (:func:`reference_merge`), at the shapes
decode context parallelism of 2, 4 and 8 serves:

* ``k = index_topk // index_kpool`` (:data:`SELECT_K`, 512) candidates a rank, so ``CP * k``
  is 1024, 2048 and 4096 a row; the widest ``CP * k`` under ``MAX_CANDIDATES`` at CP 8 and
  at CP 2 (:data:`EDGE_WIDTHS`, 4248 each); and, at each size of :data:`PROBE_SIZES`, the
  first ``k`` the width rule refuses and the ``k`` below it (:func:`merge_refuses`), which the
  host must refuse and serve alike, the refused one before it builds a kernel;
* rows: one (a decode step at batch one), 1024 (a 1024-token prefill chunk), and the row
  counts that select each tile layout the kernel has (:data:`ROW_COUNTS`);
* local pool ids up to ``MAX_DECODE_INDEX_CANDIDATES - 1`` (16383, a rank holding 65536
  tokens), so global pool ids reach 131071 at CP 8.

**Inputs** (:func:`build_case`). The patterns of :data:`PATTERNS`: bfloat16-rounded scores,
which tie; the two local rules over whole score rows (the decode select's ascending ids
with pads last, and value-descending order with ``-1`` at incomplete pools); the decode
rule's lists with their slots permuted; a tie planted at each row's ``k``-th value across
two ranks; every candidate equal, so the ids alone decide; the same at the top of the local
id range; ids that straddle bits 14, 15 and 16 of the global id; ``+0.0`` beside ``-0.0`` at
the boundary; pads by id, by value, at ``BOUND_FILL_MARK`` itself and next to it, and pads
whose value is above every score; rows with fewer than ``k`` real candidates and rows with
none; values from the finite extremes of fp32; ties in only the rows the second core of an
LNC2 pair owns, so the two cores' tie-branch conditions differ; and ties in only one of the two
tiles a core selects together, so the tie step runs over a tile with no excess tie beside one
with them.

**Reference.** A numpy ``lexsort`` over each row's ``CP * k`` candidates: real before pad,
value descending (``-0.0 == +0.0``), global pool id ascending; the first ``k`` real ones are
the selection. Global ids come from a table built by walking the global pool axis block by
block (:func:`global_table`), not from the module's formula. It is checked against a
per-row Python sort and against the module's torch oracle. Where the local lists come from
whole score rows under the decode rule, the merged set must also equal the one-rank
selection over the whole row (:func:`cp1_selection`).

Two modes:

* **CPU** (pytest; every test name contains ``cpu``). The kernel runs on the NKI simulator
  under the LNC2 grid at the shapes above; the reference, the patterns and the host
  refusals are checked; and neuronx-cc, in child processes that open no device, builds the
  kernel at the served and edge widths, every matmul a single fp32 instruction outside the
  double-row mode (none where a program runs more than one group of unfolded tiles), with no
  second SBUF allocation pass
  (:data:`CONTROL_BUILD` is the control that takes one).
* **Device** (``python test/hardware/test_dcp_candidate_merge_qualification.py device
  --out OUT.json --cache-root EMPTY_DIR``, with ``NEURON_RT_VISIBLE_CORES`` naming the cores
  and ``NEURON_LOGICAL_NC_CONFIG=2``). One compiled graph per shape and rank, built with the
  model runner's compiler arguments (:data:`RUNNER_COMPILER_ARGS`), every case compared bit
  for bit, the first case re-run to check that every emission is identical; a graph of
  :data:`MULTI_CALLS` independent merges, each output checked; and a graph that runs every
  rank's ``dsa_decode_select_pools`` at 16384 local pools and then the merge, as the decode
  step composes them, each kernel's output checked. The dispatch counters must show the NKI
  kernel traced once per graph and no torch fallback. Writes one JSON record; exits 1 on any
  mismatch. With ``--sandbox`` the same legs and checks run eagerly on the CPU through the NKI
  simulator at :data:`SANDBOX_ROWS` (a CPU test runs it), so the entry is exercised before any
  device time.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import glob
import hashlib
import json
import os
import pathlib
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time

#: This file's repository. Run as a script, Python puts ``test/hardware`` on ``sys.path`` and
#: the interpreter may resolve ``vllm_neuron`` to another checkout; the module under test must
#: be this tree's.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from vllm_neuron.functional.dsa import dcp_candidate_merge as merge  # noqa: E402
from vllm_neuron.functional.dsa import decode_select  # noqa: E402
from vllm_neuron.functional.dsa.causal_bound import BOUND_FILL, BOUND_FILL_MARK  # noqa: E402
from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig  # noqa: E402

# ---------------------------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------------------------

_CONFIG = Glm5NextTextConfig()
#: Tokens per indexer pool.
INDEX_KPOOL = int(_CONFIG.index_kpool)
#: Pools each rank selects and hands the merge: ``index_topk // index_kpool``.
SELECT_K = int(_CONFIG.index_topk) // INDEX_KPOOL
#: Tokens per KV block, the DCP interleave: the hybrid KV cache's page for this model.
BLOCK_SIZE = 128
#: Pools per ownership block, the kernel's ``pools_per_block``.
PPB = BLOCK_SIZE // INDEX_KPOOL
#: Pools a rank holds at most: the decode indexer's widest candidate axis, 65536 tokens.
LOCAL_POOLS = int(decode_select.MAX_DECODE_INDEX_CANDIDATES)
#: Decode context parallel sizes qualified here.
DCP_SIZES = (2, 4, 8)
#: Row counts and the tile layout each selects under the LNC2 grid (two programs, rows split
#: evenly): 1 folds the row over 128 partitions (the second program repeats the first's group
#: without storing); 8 folds 4-row tiles over 32; 128 folds 64-row tiles over 2, at 512 or more
#: candidates a partition; 129 gives the two programs 65 rows unfolded and 64 rows folded;
#: 1024 runs unfolded 128-row tiles two at a time, the prefill chunk.
ROW_COUNTS = (1, 8, 128, 129, 1024)
#: ``(CP, k)`` pairs at the widest ``CP * k`` under ``MAX_CANDIDATES`` (4248 each).
EDGE_WIDTHS = ((8, 531), (2, 2124))

# The merge's width rule. The next three names are the only places this file states the
# rule; a change to the merge's width plan changes the three together.


def merge_refuses(cp: int, k: int) -> bool:
    """True when the host refuses ``cp`` ranks of ``k`` candidates a row: no tile serves
    ``CP * k`` (``merge.tile_rows`` is ``None``), because no fold of a row over the partitions
    leaves each partition few enough columns for two tiles in flight to fit its SBUF."""
    return merge.tile_rows(cp, k) is None


#: Text every refusal of a width carries.
REFUSAL_TEXT = "fit no tile"
#: ``(CP, rows, k)``: a build past the SBUF budget, so the allocator must take a second pass and
#: spill; it shows that the build test can tell a fitting build from a spilling one. The merge
#: serves this width in tiles of fewer rows, so the control is built in 128-row tiles
#: (:func:`unfolded_tile_rows`). At CP 16 that build takes the second pass but logs no spill
#: traffic, so the control is CP 32.
CONTROL_BUILD = (32, 1024, SELECT_K)

#: Sizes at which the refusal test probes the width rule's edge (:func:`first_refused_k`).
PROBE_SIZES = (2, 4, 8, 16)


def first_refused_k(cp: int) -> int:
    """The smallest ``k`` :func:`merge_refuses` refuses at ``cp``, scanned up from 1."""
    return next(k for k in range(1, 1 << 16) if merge_refuses(cp, k))


#: Independent merges in the device run's multi-call graph.
MULTI_CALLS = 4

#: The model runner's neuronx-cc arguments for this checkpoint (``neuron_model_runner.py``,
#: ``compile_options["compiler_args"]``): fp8 weights on trn2, the platform's default
#: optimization level 1. Not included: the optional all-reduce fusion flag the runner appends
#: from ``collective_policy.fuse_compiler_args()``, empty unless
#: ``VLLM_NEURON_TP_ALLREDUCE_FUSE=1``.
RUNNER_COMPILER_ARGS = (
    "--auto-cast=none",
    "--verbose=35",
    "-O1",
    "--internal-hlo2tensorizer-options=--modular-flow-mac-threshold=10 "
    "--experimental-unsafe-fp8e4m3fn-as-fp8e4m3",
    "--internal-backend-options=--enable-verifier=false --enable-nested-dynamic-loop",
)

#: The pad mark the kernel compares in fp32.
MARK32 = np.float32(BOUND_FILL_MARK)
#: The value of a pad that must never be read: above every score.
GARBAGE = 1.0e4

PATTERNS = ("random", "decode_rule", "prefill_rule", "permuted", "boundary_tie", "all_equal",
            "max_ids", "id_bit_edges", "signed_zero", "pads", "short", "empty", "extremes",
            "one_core_ties", "first_tile_ties", "second_tile_ties")
#: Patterns built from whole score rows, whose merged set must equal one rank's selection.
WHOLE_ROW_PATTERNS = ("decode_rule", "prefill_rule", "permuted")


# ---------------------------------------------------------------------------------------------
# Ownership, written out from the block rule
# ---------------------------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def global_table(cp: int, local_pools: int = LOCAL_POOLS, ppb: int = PPB) -> np.ndarray:
    """``[cp, local_pools]`` int64: the global pool id of each rank's local pool.

    Walks the global axis a block of ``ppb`` pools at a time: block ``b`` goes to rank
    ``b % cp`` as that rank's next local block. A rank holds whole blocks, so
    ``local_pools`` is a multiple of ``ppb``.
    """
    if local_pools % ppb:
        raise ValueError(f"local_pools={local_pools} is not a whole number of {ppb}-pool blocks")
    table = np.full((cp, local_pools), -1, dtype=np.int64)
    filled = [0] * cp
    for block in range(cp * local_pools // ppb):
        rank = block % cp
        table[rank, filled[rank]:filled[rank] + ppb] = np.arange(block * ppb, (block + 1) * ppb)
        filled[rank] += ppb
    return table


# ---------------------------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------------------------


def _real(values: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """A candidate is real when its id is not ``-1`` and its value is above the fp32 mark."""
    return (ids >= 0) & (values > MARK32)


def _merge_by_rule(values: torch.Tensor, ids: torch.Tensor, *, tie_to_lower_id: bool,
                   value_pads: bool) -> torch.Tensor:
    cp, rows, k = (int(d) for d in values.shape)
    v = values.numpy().astype(np.float32).transpose(1, 0, 2).reshape(rows, cp * k)
    lid = ids.numpy().astype(np.int64).transpose(1, 0, 2).reshape(rows, cp * k)
    if bool((lid >= LOCAL_POOLS).any()):
        raise ValueError(f"a local id is at or past {LOCAL_POOLS}")
    real = _real(v, lid) if value_pads else lid >= 0
    rank = np.broadcast_to(np.repeat(np.arange(cp), k), (rows, cp * k))
    gid = global_table(cp)[rank, np.clip(lid, 0, None)]
    id_key = gid if tie_to_lower_id else -gid
    order = np.lexsort((np.where(real, id_key, np.iinfo(np.int64).max),
                        np.where(real, -v.astype(np.float64), 0.0),
                        np.where(real, 0, 1)), axis=1)
    taken = np.zeros((rows, cp * k), dtype=bool)
    np.put_along_axis(taken, order[:, :k], True, axis=1)
    out = np.where(taken & real, lid, -1).astype(np.int32)
    return torch.from_numpy(np.ascontiguousarray(out.reshape(rows, cp, k).transpose(1, 0, 2)))


def reference_merge(values: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """``[CP, R, k]`` int32: every rank's merge output.

    Rank ``r``'s row ``i`` holds ``ids[r, i, j]`` where that candidate is among the row's
    ``k`` selected and ``-1`` in every other slot; the selection is the first ``k`` real
    candidates of the row's ``CP * k`` in the order real first, value descending, global id
    ascending.
    """
    return _merge_by_rule(values, ids, tie_to_lower_id=True, value_pads=True)


def case_reference(case: dict) -> torch.Tensor:
    """:func:`reference_merge` of a case, computed once."""
    if "reference" not in case:
        case["reference"] = reference_merge(case["values"], case["ids"])
    return case["reference"]


def reference_merge_python(values: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """:func:`reference_merge` as one Python sort per row, for small shapes."""
    cp, rows, k = (int(d) for d in values.shape)
    table = global_table(cp)
    out = torch.full((cp, rows, k), -1, dtype=torch.int32)
    for i in range(rows):
        cands = []
        for r in range(cp):
            for j in range(k):
                lid, val = int(ids[r, i, j]), float(values[r, i, j])
                if lid >= 0 and np.float32(val) > MARK32:
                    cands.append((-val, int(table[r, lid]), r, j))
        for _, _, r, j in sorted(cands)[:k]:
            out[r, i, j] = ids[r, i, j]
    return out


def merged_global_mask(per_rank: torch.Tensor, width: int) -> np.ndarray:
    """``[R, width]`` bool: the global pools the ranks' outputs select."""
    cp, rows, _ = (int(d) for d in per_rank.shape)
    table = global_table(cp)
    mask = np.zeros((rows, width), dtype=bool)
    for r in range(cp):
        out = per_rank[r].numpy().astype(np.int64)
        row, slot = np.nonzero(out >= 0)
        mask[row, table[r, out[row, slot]]] = True
    return mask


def cp1_selection(global_rows: torch.Tensor, k: int) -> np.ndarray:
    """``[R, N]`` bool: one rank's selection over whole bounded rows: the ``k`` largest
    values, ties to the lowest pool id, nothing at or below the mark."""
    rows, n = (int(d) for d in global_rows.shape)
    v = global_rows.numpy().astype(np.float32)
    order = np.lexsort((np.broadcast_to(np.arange(n), (rows, n)), -v.astype(np.float64)), axis=1)
    mask = np.zeros((rows, n), dtype=bool)
    np.put_along_axis(mask, order[:, :k], True, axis=1)
    return mask & (v > MARK32)


# ---------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------


def _generator(*key) -> torch.Generator:
    digest = hashlib.sha256(repr(key).encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(digest[:8], "little"))


def _distinct_ids(g: torch.Generator, cp: int, rows: int, k: int, lo: int = 0,
                  hi: int = LOCAL_POOLS) -> torch.Tensor:
    """``[cp, rows, k]`` int32: ``k`` distinct local ids of ``[lo, hi)`` per rank and row, in
    random slot order."""
    picks = []
    for start in range(0, cp * rows, 256):
        keys = torch.rand(min(256, cp * rows - start), hi - lo, generator=g)
        picks.append(torch.topk(keys, k, dim=1).indices + lo)
    return torch.cat(picks).view(cp, rows, k).to(torch.int32).contiguous()


def local_rows(global_rows: torch.Tensor, cp: int, local_pools: int) -> torch.Tensor:
    """``[cp, R, local_pools]``: each rank's local pool axis of ``[R, cp * local_pools]`` rows."""
    table = torch.from_numpy(global_table(cp, local_pools))
    return torch.stack([global_rows[:, table[r]] for r in range(cp)]).contiguous()


def decode_rule_lists(local: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The decode select's local top-``k`` of bounded rows ``[..., C]``: the ``k`` largest
    values with ties to the lowest index, in ascending id order, ``(BOUND_FILL, -1)`` pads last
    where fewer than ``k`` are above the mark."""
    width = int(local.shape[-1])
    order = torch.sort(local, dim=-1, descending=True, stable=True).indices[..., :k]
    vals = torch.gather(local, -1, order)
    real = vals > float(MARK32)
    key = torch.where(real, order, torch.full_like(order, width))
    emit = torch.argsort(key, dim=-1, stable=True)
    ids = torch.where(real, order, torch.full_like(order, -1)).gather(-1, emit)
    vals = torch.where(real, vals, torch.full_like(vals, BOUND_FILL)).gather(-1, emit)
    return vals.contiguous(), ids.to(torch.int32).contiguous()


def prefill_rule_lists(local: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A value-descending local top-``k``, ``-1`` where the value is at or below the mark and
    the value kept, as the prefill selection and its causal sentinel hand them over."""
    order = torch.sort(local, dim=-1, descending=True, stable=True).indices[..., :k]
    vals = torch.gather(local, -1, order)
    ids = torch.where(vals > float(MARK32), order, torch.full_like(order, -1))
    return vals.contiguous(), ids.to(torch.int32).contiguous()


def _whole_rows(g: torch.Generator, rows: int, n: int, k_floor: int) -> torch.Tensor:
    """``[rows, n]`` bfloat16-rounded scores, ``BOUND_FILL`` past each row's complete pools
    (between ``k_floor // 2`` and ``n``)."""
    scores = torch.randn(rows, n, generator=g).to(torch.bfloat16).float()
    complete = torch.randint(max(1, k_floor // 2), n + 1, (rows,), generator=g)
    cols = torch.arange(n)[None, :]
    return scores.masked_fill(cols >= complete[:, None], BOUND_FILL)


def _plant_boundary_tie(values: torch.Tensor, ids: torch.Tensor, rows, g) -> None:
    """In each row of ``rows``, give eight candidates below the ``k``-th value, four on each of
    two ranks, the ``k``-th value itself: an excess tie at the boundary across two ranks."""
    cp, _, k = (int(d) for d in values.shape)
    for i in rows:
        flat = values[:, i, :].reshape(-1)
        order = torch.argsort(flat, descending=True)
        kth = float(flat[order[k - 1]])
        owner = int(order[k - 1]) // k
        below = order[k:]
        for rank in (owner, (owner + 1) % cp):
            mine = below[below // k == rank][:4]
            flat[mine] = kth
        values[:, i, :] = flat.view(cp, k)


def build_case(pattern: str, cp: int, rows: int, k: int, seed: int) -> dict:
    """Every rank's merge operands for one pattern: ``values [cp, rows, k]`` fp32 and
    ``ids [cp, rows, k]`` int32, rank-major as the all-gather hands them over; for the
    whole-row patterns also the bounded rows ``global_rows [rows, cp * local_pools]``."""
    g = _generator(pattern, cp, rows, k, seed)
    case: dict = {"pattern": pattern, "cp": cp, "rows": rows, "k": k, "seed": seed}
    if pattern in WHOLE_ROW_PATTERNS:
        # Narrower axes past 8 rows keep the case small; a rank holds whole blocks.
        local_pools = LOCAL_POOLS if rows <= 8 else -(-max(2 * k, 1024) // PPB) * PPB
        global_rows = _whole_rows(g, rows, cp * local_pools, k)
        local = local_rows(global_rows, cp, local_pools)
        rule = prefill_rule_lists if pattern == "prefill_rule" else decode_rule_lists
        values, ids = rule(local, k)
        if pattern == "permuted":
            perm = torch.argsort(torch.rand(cp, rows, k, generator=g), dim=-1)
            values, ids = values.gather(-1, perm), ids.gather(-1, perm)
        case["global_rows"] = global_rows
        case["values"], case["ids"] = values.contiguous(), ids.contiguous()
        return case

    ids = _distinct_ids(g, cp, rows, k)
    values = torch.randn(cp, rows, k, generator=g)
    if pattern == "random":
        values = values.to(torch.bfloat16).float()
    elif pattern == "boundary_tie":
        _plant_boundary_tie(values, ids, range(rows), g)
    elif pattern == "all_equal":
        values = torch.full((cp, rows, k), 0.25)
    elif pattern == "max_ids":
        perm = torch.argsort(torch.rand(cp, rows, k, generator=g), dim=-1)
        ids = (perm + (LOCAL_POOLS - k)).to(torch.int32).contiguous()
        values = torch.full((cp, rows, k), -0.75)
    elif pattern == "id_bit_edges":
        table = global_table(cp)
        picks = []
        for r in range(cp):
            near = []
            for edge in (1 << 14, 1 << 15, 1 << 16, cp * LOCAL_POOLS - 1):
                at = int(np.searchsorted(table[r], min(edge, int(table[r, -1]))))
                lo = max(0, min(LOCAL_POOLS - k // 4, at - k // 8))
                near.extend(range(lo, lo + k // 4))
            near = sorted(set(near))[:k]
            held = set(near)
            rest = [x for x in torch.randperm(LOCAL_POOLS, generator=g).tolist()
                    if x not in held][:k - len(near)]
            picks.append(near + rest)
        base = torch.tensor(picks, dtype=torch.int32)
        perm = torch.argsort(torch.rand(cp, rows, k, generator=g), dim=-1)
        ids = base[:, None, :].expand(cp, rows, k).gather(-1, perm).contiguous()
        values = torch.full((cp, rows, k), 1.5)
    elif pattern == "signed_zero":
        zeros = torch.where(torch.rand(cp, rows, k, generator=g) < 0.5, 0.0, -0.0)
        few = torch.rand(cp, rows, k, generator=g) < (k // 4) / (cp * k)
        values = torch.where(few, torch.rand(cp, rows, k, generator=g) + 0.5, zeros)
        values = torch.where(torch.rand(cp, rows, k, generator=g) < 0.25, -1.0, values)
    elif pattern == "pads":
        values = values.to(torch.bfloat16).float()
        # The first half of the rows (at least one) is mostly pads, so it holds fewer than k
        # real candidates and the one just above the mark is selected; the rest is mostly
        # real. Slots 0 and 1 keep their ids: the value at the mark, and the one above it.
        u = torch.rand(cp, rows, k, generator=g)
        u[:, :, :2] = 1.0
        sparse = (torch.arange(rows) < max(1, rows // 2)).view(1, rows, 1)
        id_pad = torch.where(sparse, u < 0.6, u < 0.25)
        value_pad = ~id_pad & torch.where(sparse, u < 0.9, u < 0.375)
        ids = torch.where(id_pad, torch.full_like(ids, -1), ids)
        values = torch.where(id_pad, torch.full_like(values, GARBAGE), values)
        values = torch.where(value_pad, torch.full_like(values, BOUND_FILL), values)
        values[:, :, 0] = float(MARK32)
        values[:, :, 1] = float(np.nextafter(MARK32, np.float32(0.0)))
    elif pattern == "short":
        keep = torch.rand(cp, rows, k, generator=g) < 0.5 / cp
        ids = torch.where(keep, ids, torch.full_like(ids, -1))
        values = torch.where(keep, values, torch.full_like(values, BOUND_FILL))
    elif pattern == "empty":
        ids = torch.where(torch.rand(cp, rows, k, generator=g) < 0.5, ids,
                          torch.full_like(ids, -1))
        values = torch.where(ids < 0, torch.full_like(values, GARBAGE),
                             torch.full_like(values, BOUND_FILL))
    elif pattern == "extremes":
        table = torch.tensor([3.0e38, -3.0e38, -9.0e28, 1.4e-45, -1.4e-45, 2.0 ** -126,
                              1.0, -1.0, 0.0], dtype=torch.float32)
        values = table[torch.randint(0, len(table), (cp, rows, k), generator=g)]
        values = torch.where(values < float(MARK32), torch.full_like(values, -9.0e28), values)
    elif pattern == "one_core_ties":
        _plant_boundary_tie(values, ids, second_program_rows(rows), g)
    elif pattern == "first_tile_ties":
        _plant_boundary_tie(values, ids, first_program_tiles(rows)[0], g)
    elif pattern == "second_tile_ties":
        _plant_boundary_tie(values, ids, first_program_tiles(rows)[1], g)
    else:
        raise ValueError(f"unknown pattern {pattern!r}")
    case["values"], case["ids"] = values.to(torch.float32).contiguous(), ids.contiguous()
    return case


def second_program_rows(rows: int) -> range:
    """The rows the second program of a two-program launch owns: the upper ``rows // 2``."""
    return range((rows + 1) // 2, rows)


def first_program_tiles(rows: int) -> list[range]:
    """The first program's row tiles: its share, the first ``ceil(rows / 2)`` rows, in tiles of
    up to 128 rows. The kernel selects a program's tiles two at a time."""
    share = (rows + 1) // 2
    return [range(start, min(share, start + 128)) for start in range(0, share, 128)]


def pattern_applies(pattern: str, rows: int) -> bool:
    """Whether ``pattern`` has the rows it needs: a second program's rows, or a second tile."""
    if pattern == "one_core_ties":
        return rows >= 2
    if pattern == "second_tile_ties":
        return len(first_program_tiles(rows)) >= 2
    return True


def excess_tie_rows(values: torch.Tensor, ids: torch.Tensor) -> np.ndarray:
    """``[R]`` bool: rows with more than ``k`` real candidates at or above the ``k``-th real
    value, the kernel's tie-branch condition."""
    cp, rows, k = (int(d) for d in values.shape)
    v = values.numpy().transpose(1, 0, 2).reshape(rows, cp * k).astype(np.float32)
    lid = ids.numpy().transpose(1, 0, 2).reshape(rows, cp * k)
    real = _real(v, lid)
    out = np.zeros(rows, dtype=bool)
    for i in range(rows):
        reals = np.sort(v[i][real[i]])[::-1]
        if len(reals) >= k:
            out[i] = (reals >= reals[k - 1]).sum() > k
    return out


def cases_for(cp: int, rows: int, k: int, seeds) -> list[dict]:
    """Every pattern that applies to the shape, at each seed."""
    out = []
    for pattern in PATTERNS:
        if not pattern_applies(pattern, rows):
            continue
        for seed in seeds:
            out.append(build_case(pattern, cp, rows, k, seed))
    return out


def check_case(case: dict, got: torch.Tensor, cp_rank: int) -> dict:
    """``got`` (one rank's output) against the reference: mismatching slots, pads emitted, and
    the first mismatching row."""
    want = case_reference(case)[cp_rank]
    differ = got != want
    pads = torch.from_numpy(~_real(case["values"][cp_rank].numpy(),
                                   case["ids"][cp_rank].numpy()))
    rows = torch.nonzero(differ.any(dim=1)).flatten()
    return {"equal": bool(torch.equal(got, want)), "mismatched_slots": int(differ.sum()),
            "first_mismatched_row": int(rows[0]) if rows.numel() else None,
            "pads_emitted": int(((got >= 0) & pads).sum()),
            "kept": int((want >= 0).sum())}


# ---------------------------------------------------------------------------------------------
# CPU: the reference and the inputs
# ---------------------------------------------------------------------------------------------


def test_cpu_module_under_test_is_this_tree() -> None:
    assert pathlib.Path(merge.__file__).resolve().is_relative_to(REPO_ROOT), merge.__file__


def test_cpu_global_table_matches_the_block_rule() -> None:
    for cp in DCP_SIZES:
        table = global_table(cp)
        assert sorted(table.reshape(-1).tolist()) == list(range(cp * LOCAL_POOLS))
        gid = table.reshape(-1)
        owner = np.repeat(np.arange(cp), LOCAL_POOLS)
        assert ((gid // PPB) % cp == owner).all()
        assert (np.diff(table, axis=1) > 0).all()
    assert int(global_table(8)[7, LOCAL_POOLS - 1]) == 8 * LOCAL_POOLS - 1 == 131071


def test_cpu_reference_agrees_with_a_per_row_python_sort() -> None:
    for cp in DCP_SIZES:
        for pattern in PATTERNS:
            rows = next(r for r in (1, 2, 257) if pattern_applies(pattern, r))
            k = 24 if pattern not in WHOLE_ROW_PATTERNS else 16
            case = build_case(pattern, cp, rows, k, 0)
            assert torch.equal(reference_merge(case["values"], case["ids"]),
                               reference_merge_python(case["values"], case["ids"])), pattern
    case = build_case("random", 8, 1, SELECT_K, 0)
    assert torch.equal(reference_merge(case["values"], case["ids"]),
                       reference_merge_python(case["values"], case["ids"]))


def test_cpu_reference_agrees_with_the_module_oracle() -> None:
    for cp in DCP_SIZES:
        for rows in (1, 8):
            for case in cases_for(cp, rows, SELECT_K, (0, 1)):
                want = reference_merge(case["values"], case["ids"])
                for rank in range(cp):
                    got = merge.dsa_dcp_merge_select_torch_oracle(
                        case["values"], case["ids"], cp_rank=rank, pools_per_block=PPB)
                    assert torch.equal(got, want[rank]), (case["pattern"], cp, rows, rank)


def test_cpu_whole_row_merges_equal_the_one_rank_selection() -> None:
    for cp in DCP_SIZES:
        for rows in (1, 8, 129):
            for pattern in WHOLE_ROW_PATTERNS:
                case = build_case(pattern, cp, rows, SELECT_K, 0)
                merged = merged_global_mask(reference_merge(case["values"], case["ids"]),
                                            int(case["global_rows"].shape[1]))
                assert (merged == cp1_selection(case["global_rows"], SELECT_K)).all(), (
                    pattern, cp, rows)


def test_cpu_patterns_hold_what_they_claim() -> None:
    """Each pattern carries the hazard it is named for, so a pass on it is not vacuous."""
    for cp in DCP_SIZES:
        cases = {c["pattern"]: c for c in cases_for(cp, 8, SELECT_K, (0,))}
        ref = {name: reference_merge(c["values"], c["ids"]) for name, c in cases.items()}
        for name in ("boundary_tie", "all_equal", "max_ids", "id_bit_edges", "signed_zero"):
            assert excess_tie_rows(cases[name]["values"], cases[name]["ids"]).all(), name
        for name in ("all_equal", "max_ids"):
            assert int((ref[name] >= 0).sum()) == 8 * SELECT_K, name
        top = cases["max_ids"]
        gids = global_table(cp)[np.arange(cp)[:, None, None], top["ids"].numpy()]
        assert int(gids.max()) == cp * LOCAL_POOLS - 1
        assert int(gids.min()) == int(global_table(cp)[0, LOCAL_POOLS - SELECT_K])
        edges = cases["id_bit_edges"]
        gids = global_table(cp)[np.arange(cp)[:, None, None], edges["ids"].numpy()]
        for bit in (14, 15, 16):
            if cp * LOCAL_POOLS > (1 << bit):
                assert ((gids >> bit) & 1).any() and (((gids >> bit) & 1) == 0).any(), bit
        sz = cases["signed_zero"]
        kept_vals = sz["values"][ref["signed_zero"] >= 0]
        kept_zeros = kept_vals[kept_vals == 0]
        assert bool(torch.signbit(kept_zeros).any()) and not bool(torch.signbit(kept_zeros).all())
        pads = cases["pads"]
        assert bool((pads["ids"] < 0).any()) and bool((pads["values"] == BOUND_FILL).any())
        assert bool((pads["values"][:, :, 0] == float(MARK32)).all())
        assert int((ref["pads"][:, :, 0] >= 0).sum()) == 0
        assert bool((ref["pads"][:, :4, 1] >= 0).all())
        short = cases["short"]
        real = _real(short["values"].numpy(), short["ids"].numpy()).sum(axis=(0, 2))
        assert (real < SELECT_K).all() and int((ref["short"] >= 0).sum()) == int(real.sum())
        assert int((ref["empty"] >= 0).sum()) == 0
        ties = excess_tie_rows(cases["one_core_ties"]["values"], cases["one_core_ties"]["ids"])
        assert ties[list(second_program_rows(8))].all() and not ties[:4].any()
        assert "second_tile_ties" not in cases
    for cp in DCP_SIZES:
        tiles = first_program_tiles(1024)
        assert [(t.start, t.stop) for t in tiles[:2]] == [(0, 128), (128, 256)]
        for name, tile in (("first_tile_ties", tiles[0]), ("second_tile_ties", tiles[1])):
            case = build_case(name, cp, 1024, SELECT_K, 0)
            want = np.zeros(1024, dtype=bool)
            want[list(tile)] = True
            assert (excess_tie_rows(case["values"], case["ids"]) == want).all(), (name, cp)
        perm = cases["permuted"]["ids"]
        assert not bool((perm[..., 1:] >= perm[..., :-1]).all())
        dec = cases["decode_rule"]["ids"]
        assert bool(((dec[..., 1:] > dec[..., :-1]) | (dec[..., 1:] < 0)).all())


def test_cpu_the_comparison_rejects_a_wrong_rule() -> None:
    """A merge that breaks ties by the higher global id, or that keeps a value pad, fails
    :func:`check_case`: the comparison discriminates the rule's two hazards."""
    for cp in DCP_SIZES:
        for pattern in ("all_equal", "max_ids", "boundary_tie", "signed_zero"):
            tie = build_case(pattern, cp, 1, SELECT_K, 0)
            wrong = _merge_by_rule(tie["values"], tie["ids"], tie_to_lower_id=False,
                                   value_pads=True)
            assert not all(check_case(tie, wrong[r], r)["equal"] for r in range(cp)), pattern
        pads = build_case("pads", cp, 1, SELECT_K, 0)
        loose = _merge_by_rule(pads["values"], pads["ids"], tie_to_lower_id=True,
                               value_pads=False)
        assert not all(check_case(pads, loose[r], r)["equal"] for r in range(cp))


def test_cpu_host_refuses_exactly_the_widths_the_width_rule_refuses() -> None:
    assert merge.MAX_CANDIDATES == merge.SBUF_BYTES_PER_PARTITION // (
        merge._TILES_IN_FLIGHT * merge._BYTES_PER_CANDIDATE) == 4249
    for cp in DCP_SIZES:
        assert not merge_refuses(cp, SELECT_K), cp
    for cp, k in EDGE_WIDTHS:
        assert not merge_refuses(cp, k), (cp, k)
    probes = [*EDGE_WIDTHS, *((cp, SELECT_K) for cp in PROBE_SIZES)]
    probes += [(cp, first_refused_k(cp) + step) for cp in PROBE_SIZES for step in (-1, 0)]
    for cp, k in probes:
        values, ids = torch.zeros(cp, 1, k), torch.zeros(cp, 1, k, dtype=torch.int32)
        if not merge_refuses(cp, k):
            assert merge._validate(values, ids, 0, PPB) == (cp, 1, k), (cp, k)
            continue
        try:
            merge.dsa_dcp_merge_select(values, ids, cp_rank=0, pools_per_block=PPB)
        except merge.DcpCandidateMergeError as refusal:
            assert REFUSAL_TEXT in str(refusal), refusal
        else:
            raise AssertionError(f"CP * k = {cp} * {k} was not refused")


# ---------------------------------------------------------------------------------------------
# CPU: the kernel on the NKI simulator
# ---------------------------------------------------------------------------------------------


def _simulate(monkeypatch) -> None:
    monkeypatch.setenv("NKI_SIMULATOR", "1")
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    assert merge.can_run_dsa_dcp_merge_select(torch.zeros(1), torch.zeros(1)), (
        "the simulator route is not open: VLLM_NEURON_CPU_MODE=1 is required")
    assert merge.merge_programs() == 2


def _run_kernel(case: dict, rank: int) -> tuple[torch.Tensor, tuple[int, int]]:
    merge.reset_dcp_merge_select_dispatch_counters()
    got = merge.dsa_dcp_merge_select(case["values"], case["ids"], cp_rank=rank,
                                     pools_per_block=PPB)
    return got, merge.dcp_merge_select_dispatch_counters()


def _ranks(cp: int, rows: int) -> range | tuple:
    """Every rank at the small row counts; the first and last at the large ones."""
    return range(cp) if rows <= 129 else (0, cp - 1)


def _patterns(rows: int) -> tuple:
    chosen = PATTERNS if rows <= 129 else ("random", "boundary_tie", "max_ids", "pads",
                                           "one_core_ties", "decode_rule", "first_tile_ties",
                                           "second_tile_ties")
    return tuple(p for p in chosen if pattern_applies(p, rows))


def test_cpu_simulator_kernel_matches_the_reference_at_the_served_widths(monkeypatch) -> None:
    _simulate(monkeypatch)
    failures = []
    for cp in DCP_SIZES:
        for rows in ROW_COUNTS:
            for pattern in _patterns(rows):
                case = build_case(pattern, cp, rows, SELECT_K, 0)
                for rank in _ranks(cp, rows):
                    got, counters = _run_kernel(case, rank)
                    result = check_case(case, got, rank)
                    if counters != (1, 0) or not result["equal"] or result["pads_emitted"]:
                        failures.append((cp, rows, pattern, rank, counters, result))
    assert not failures, failures[:5]


def test_cpu_simulator_kernel_matches_the_reference_at_the_edge_widths(monkeypatch) -> None:
    _simulate(monkeypatch)
    failures = []
    for cp, k in EDGE_WIDTHS:
        for rows in (1, 129):
            for pattern in ("random", "all_equal", "max_ids", "boundary_tie", "pads"):
                case = build_case(pattern, cp, rows, k, 0)
                for rank in range(cp):
                    got, counters = _run_kernel(case, rank)
                    result = check_case(case, got, rank)
                    if counters != (1, 0) or not result["equal"]:
                        failures.append((cp, k, rows, pattern, rank, counters, result))
    assert not failures, failures[:5]


def chain_case(cp: int, context: int, seed: int) -> dict:
    """One decode row at a ``context``-token sequence over ``cp`` ranks: each rank's bounded
    scores on its own pool axis (``LOCAL_POOLS`` wide, ``BOUND_FILL`` past its complete pools)
    and its owned length."""
    g = _generator("chain", cp, context, seed)
    complete = context // INDEX_KPOOL
    scores = torch.randn(1, cp * LOCAL_POOLS, generator=g).to(torch.bfloat16).float()
    scores[:, complete:] = BOUND_FILL
    owned = [0] * cp
    for block in range(-(-context // BLOCK_SIZE)):
        owned[block % cp] += min(BLOCK_SIZE, context - block * BLOCK_SIZE)
    return {"global_rows": scores, "local": local_rows(scores, cp, LOCAL_POOLS),
            "owned": torch.tensor(owned, dtype=torch.int32).view(cp, 1), "cp": cp,
            "context": context}


#: Contexts of the chain: the widest a rank holds (65536 tokens a rank, one pool short of
#: whole, so the open pool is in the last block) and one where most ranks hold fewer than
#: ``SELECT_K`` complete pools.
def chain_contexts(cp: int) -> tuple[int, int]:
    return (cp * LOCAL_POOLS * INDEX_KPOOL - 3, cp * BLOCK_SIZE * 3 + 70)


def test_cpu_simulator_decode_select_then_merge_equals_the_one_rank_selection(
        monkeypatch) -> None:
    """Each rank's ``dsa_decode_select_pools`` at 16384 local pools, then the merge, on the
    simulator: the composed selection is the one-rank selection over the whole row."""
    _simulate(monkeypatch)
    for cp in DCP_SIZES:
        for context in chain_contexts(cp):
            case = chain_case(cp, context, 0)
            lists = [decode_select.dsa_decode_select_pools(
                case["local"][r], case["owned"][r], select_k=SELECT_K, pool_size=INDEX_KPOOL)
                for r in range(cp)]
            values = torch.stack([v for _, v, _ in lists]).contiguous()
            ids = torch.stack([i for _, _, i in lists]).contiguous()
            want_v, want_i = decode_rule_lists(case["local"], SELECT_K)
            assert torch.equal(values, want_v) and torch.equal(ids, want_i), (cp, context)
            outs = []
            for rank in range(cp):
                got, counters = _run_kernel({"values": values, "ids": ids}, rank)
                assert counters == (1, 0)
                outs.append(got)
            merged = merged_global_mask(torch.stack(outs), cp * LOCAL_POOLS)
            assert (merged == cp1_selection(case["global_rows"], SELECT_K)).all(), (cp, context)


# ---------------------------------------------------------------------------------------------
# CPU: neuronx-cc builds of the kernel, in child processes that open no device
# ---------------------------------------------------------------------------------------------

#: ``(CP, rows, k)`` builds: the served widths at a decode row and a prefill chunk, the edge
#: widths, ``k = 532`` at CP 8 (4256 candidates, one ``k`` past the edge, still one SBUF
#: pass), and :data:`CONTROL_BUILD` in 128-row tiles.
BUILDS = ([(cp, rows, SELECT_K) for cp in DCP_SIZES for rows in (1, 1024)]
          + [(cp, 1024, k) for cp, k in EDGE_WIDTHS] + [(8, 1024, 532), CONTROL_BUILD])
_BUILD_ROW = "dcp_merge_build"
_DROP = ("NKI_SIMULATOR", "NKI_PRECISE_FP", "VLLM_NEURON_CPU_MODE", "NEURON_RT_VISIBLE_CORES")
_PIN = {"VLLM_NEURON_CPU_COMPILE": "1", "NEURON_PLATFORM_TARGET_OVERRIDE": "trn2",
        "NEURON_LOGICAL_NC_CONFIG": "2", "PYTHONDONTWRITEBYTECODE": "1"}


def unfolded_tile_rows(cp: int, k: int) -> int:
    """The control's tile height: 128 rows whatever the width. A module-level function, so the
    NKI front end resolves it by name."""
    return merge.PARTITIONS


def _build_one(cp: int, rows: int, k: int, work: str, unfolded: bool) -> None:
    """Trace and build the kernel on the LNC2 grid to ``work/kernel.neff``; print one row.
    ``unfolded`` builds 128-row tiles whatever the width."""
    from dataclasses import replace

    from nki.compiler.driver import _compile_bir_to_neff
    from nki.compiler.frontend import resolve_frontend_cls
    from nki.framework.compiled import CompileKernel, compile_kernel_to_nir

    inputs = {"values_hbm": np.zeros((cp, rows, k), dtype=np.float32),
              "ids_hbm": np.zeros((cp, rows, k), dtype=np.int32), "cp_rank": cp - 1,
              "pools_per_block": PPB, "source_digest": merge.SOURCE_DIGEST}
    print(f"{_BUILD_ROW}|tree|module={merge.__file__}", flush=True)
    if unfolded:
        merge.tile_rows = unfolded_tile_rows
    frontend = resolve_frontend_cls()
    compiling = merge.dsa_dcp_merge_select_kernel[2]._to_subclass(CompileKernel,
                                                                  _frontend_cls=frontend)
    neff = os.path.join(work, "kernel.neff")
    options = replace(compiling._compile_opts(), artifacts_dir=work, output_path=neff)
    nir = compile_kernel_to_nir(compiling, inputs=inputs, compile_opts=options,
                                frontend=frontend(), enable_cache=False)
    _compile_bir_to_neff(nir, options, {n: v for n, v in inputs.items()
                                        if isinstance(v, np.ndarray)})
    print(f"{_BUILD_ROW}|built|lnc={options.lnc}|neff_bytes={os.path.getsize(neff)}", flush=True)


def build_facts(work: str) -> dict:
    """What a build's artifacts show: its matmuls (count, modes, operand dtypes, zero region)
    from ``module.mlir``, and whether the SBUF allocator took a second pass or spilled, from
    ``log-neuron-cc.txt``."""
    mlir = pathlib.Path(work, "module.mlir").read_text()
    matmuls = [line for line in mlir.splitlines() if "nisa.matmul(" in line]
    log = pathlib.Path(work, "log-neuron-cc.txt").read_text()
    spill = [int(n) for n in re.findall(r"\[Spill Optimization\] reduced DMA traffic (\d+)", log)]
    return {
        "matmuls": len(matmuls),
        "perf_modes": sorted(set(re.findall(r"perf_opt=(\w+)", "\n".join(matmuls)))),
        "operand_dtypes": sorted(set(re.findall(r"memref<[0-9x]+x(\w+), #nisa\.mem<",
                                                "\n".join(matmuls)))),
        "zero_regions": sorted(set(re.findall(r"psum_zero_region=(\w+)", "\n".join(matmuls)))),
        "allocator_retry": "FORWARD_ALLOCATE : 0" in log,
        "spill_traffic_reduced": spill,
    }


def run_builds(builds) -> dict:
    """Every build in its own child process, all at once; ``{(cp, rows, k): facts}``."""
    environment = {k: v for k, v in os.environ.items() if k not in _DROP}
    environment.update(_PIN, PYTHONPATH=os.pathsep.join(
        [str(REPO_ROOT), *filter(None, os.environ.get("PYTHONPATH", "").split(os.pathsep))]))
    environment["PATH"] = os.pathsep.join(
        [str(pathlib.Path(sys.executable).parent), environment.get("PATH", "")])
    facts = {}
    with tempfile.TemporaryDirectory(prefix="dcp_merge_build_") as scratch:
        children = []
        for cp, rows, k in builds:
            work = os.path.join(scratch, f"cp{cp}_r{rows}_k{k}")
            os.makedirs(work)
            children.append(((cp, rows, k), work, subprocess.Popen(
                [sys.executable, "-B", str(pathlib.Path(__file__).resolve()), "build-one",
                 str(cp), str(rows), str(k), work,
                 *(["--unfolded"] if (cp, rows, k) == CONTROL_BUILD else [])],
                cwd=work, env=environment,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)))
        for key, work, child in children:
            out, err = child.communicate(timeout=1800)
            assert child.returncode == 0, (key, err[-3000:])
            printed = [line for line in out.splitlines() if line.startswith(_BUILD_ROW + "|")]
            assert any(f"module={REPO_ROOT}/" in line for line in printed), printed
            assert any("|built|" in line for line in printed), printed
            facts[key] = build_facts(work)
    return facts


def test_cpu_neuronx_cc_builds_the_merge_at_the_served_and_edge_widths() -> None:
    facts = run_builds(BUILDS)
    control = facts.pop(CONTROL_BUILD)
    assert control["allocator_retry"] and max(control["spill_traffic_reduced"]) > 0, (
        f"the 128-row-tile build past the cap shows no second allocation pass, so the "
        f"probe cannot tell a fitting build from a spilling one: {control}")
    for key, fact in facts.items():
        assert not fact["allocator_retry"], (key, fact)
        assert fact["spill_traffic_reduced"] and max(fact["spill_traffic_reduced"]) == 0, (
            key, fact)
        # A build sums by matmul a folded tile's row counts and a one-group program's joint
        # tie flag; a program of more groups of unfolded tiles has neither.
        cp, rows, k = key
        groups = merge.program_groups_at(rows, 2, merge.tile_rows(cp, k))[0]
        if len(groups) > 1 and all(merge.tile_fold(n, cp, k) == 1
                                   for _, tiles in groups for _, n in tiles):
            assert fact["matmuls"] == 0, (key, fact)
            continue
        assert fact["matmuls"] > 0, (key, fact)
        assert fact["perf_modes"] == ["none_"], (key, fact)
        assert fact["operand_dtypes"] == ["f32"], (key, fact)
        assert fact["zero_regions"] == ["size2048"], (key, fact)


# ---------------------------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Target:
    """Where the device entry runs: the Neuron device, its graphs compiled with the runner's
    arguments; or the sandbox, the same legs eagerly on the CPU through the NKI simulator."""

    device: str
    compiled: bool

    def compile(self, fn):
        if not self.compiled:
            return fn
        from vllm_neuron.envs import get_compile_backend_name

        return torch.compile(fn, backend=get_compile_backend_name(), fullgraph=True,
                             dynamic=False,
                             options={"alias_meta_to_neuron": True,
                                      "compiler_args": list(RUNNER_COMPILER_ARGS)})


#: The kernel the merge seam must dispatch: ``(module, qualname)``.
MERGE_KERNEL = ("vllm_neuron.functional.dsa.dcp_candidate_merge", "dsa_dcp_merge_select_kernel")

DEVICE = Target("neuron:0", compiled=True)
SANDBOX = Target("cpu", compiled=False)
#: The sandbox's shapes: every leg and check of the device entry at the row counts the
#: simulator runs quickly.
SANDBOX_ROWS = (1, 8)


def _digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()[:16]


def _merge_fn(rank: int):
    def call(values, ids):
        return merge.dsa_dcp_merge_select(values, ids, cp_rank=rank, pools_per_block=PPB)
    return call


def device_identity(target: Target, cp: int, rows: int, k: int, seeds, emissions: int) -> dict:
    """One graph per rank, every case of the shape, each rank's output against the
    reference; the first case run ``emissions`` times.

    The dispatch counters are read after the first call, which traces the graph, and again,
    from zero, after every other call: a compiled graph dispatches nothing from Python once
    traced, so the second reading is ``(0, 0)`` on the device and one NKI dispatch a call in
    the sandbox."""
    cases = cases_for(cp, rows, k, seeds if rows <= 129 else seeds[:1])
    first_operands = (cases[0]["values"].to(target.device), cases[0]["ids"].to(target.device))
    calls_after = (emissions - 1) + (len(cases) - 1)
    expected_after = [0, 0] if target.compiled else [calls_after, 0]
    out = {"cp": cp, "rows": rows, "k": k, "ranks": []}
    for rank in range(cp):
        t0 = time.time()
        graph = target.compile(_merge_fn(rank))
        merge.reset_dcp_merge_select_dispatch_counters()
        first = graph(*first_operands).to("cpu")
        at_trace = list(merge.dcp_merge_select_dispatch_counters())
        kernel = list(merge.dcp_merge_select_kernel_identity() or ())
        compile_s = round(time.time() - t0, 1)
        merge.reset_dcp_merge_select_dispatch_counters()
        again = [bool(torch.equal(graph(*first_operands).to("cpu"), first))
                 for _ in range(emissions - 1)]
        results = []
        for case in cases:
            got = first if case is cases[0] else graph(case["values"].to(target.device),
                                                       case["ids"].to(target.device)).to("cpu")
            results.append({"pattern": case["pattern"], "seed": case["seed"],
                            **check_case(case, got, rank), "digest": _digest(got)})
        out["ranks"].append({
            "cp_rank": rank, "counters_at_trace": at_trace,
            "counters_after": list(merge.dcp_merge_select_dispatch_counters()),
            "counters_expected_after": expected_after,
            "kernel": kernel,
            "compile_and_first_call_s": compile_s, "emissions_identical": all(again),
            "cases": results})
        print(json.dumps({"leg": "identity", "cp": cp, "rows": rows, "k": k, "rank": rank,
                          "equal": all(r["equal"] for r in results),
                          "counters": at_trace, "s": compile_s}), flush=True)
    return out


def device_multi(target: Target, cp: int, rows: int, ranks) -> dict:
    """A graph of :data:`MULTI_CALLS` independent merges on distinct operands, each output
    checked."""
    cases = [build_case(p, cp, rows, SELECT_K, 0)
             for p in ("random", "boundary_tie", "max_ids", "pads")[:MULTI_CALLS]]
    flat = []
    for c in cases:
        flat += [c["values"].to(target.device), c["ids"].to(target.device)]
    out = {"cp": cp, "rows": rows, "ranks": []}
    for rank in ranks:
        def many(*operands, rank=rank):
            return tuple(merge.dsa_dcp_merge_select(operands[2 * i], operands[2 * i + 1],
                                                    cp_rank=rank, pools_per_block=PPB)
                         for i in range(len(operands) // 2))
        outputs = target.compile(many)(*flat)
        results = [check_case(c, g.to("cpu"), rank) for c, g in zip(cases, outputs)]
        out["ranks"].append({"cp_rank": rank, "calls": results})
        print(json.dumps({"leg": "multi", "cp": cp, "rows": rows, "rank": rank,
                          "equal": [r["equal"] for r in results]}), flush=True)
    return out


def device_chain(target: Target, cp: int, seed: int) -> dict:
    """Per rank, one graph of every rank's decode select at 16384 local pools and the merge;
    each select's lists, each merge output, and the composed set against the one-rank
    selection."""
    out = {"cp": cp, "contexts": []}
    for context in chain_contexts(cp):
        case = chain_case(cp, context, seed)
        want_v, want_i = decode_rule_lists(case["local"], SELECT_K)
        local = case["local"].to(target.device)
        owned = case["owned"].to(target.device)
        outs, ranks = [], []
        for rank in range(cp):
            def chain(local, owned, rank=rank):
                lists = [decode_select.dsa_decode_select_pools(
                    local[r], owned[r], select_k=SELECT_K, pool_size=INDEX_KPOOL)
                    for r in range(cp)]
                values = torch.stack([v for _, v, _ in lists])
                ids = torch.stack([i for _, _, i in lists])
                return values, ids, merge.dsa_dcp_merge_select(values, ids, cp_rank=rank,
                                                               pools_per_block=PPB)
            values, ids, merged = (t.to("cpu") for t in target.compile(chain)(local, owned))
            lists_equal = bool(torch.equal(values, want_v) and torch.equal(ids, want_i))
            result = check_case({"values": want_v, "ids": want_i}, merged, rank)
            ranks.append({"cp_rank": rank, "lists_equal": lists_equal, **result})
            outs.append(merged)
        composed = bool((merged_global_mask(torch.stack(outs), cp * LOCAL_POOLS)
                         == cp1_selection(case["global_rows"], SELECT_K)).all())
        out["contexts"].append({"context": context, "owned": case["owned"].flatten().tolist(),
                                "ranks": ranks, "composed_equals_one_rank": composed})
        print(json.dumps({"leg": "chain", "cp": cp, "context": context,
                          "equal": [r["equal"] and r["lists_equal"] for r in ranks],
                          "composed": composed}), flush=True)
    return out


def compile_commands(cache_root: pathlib.Path) -> dict:
    """The neuronx-cc commands the run's graphs were built with: how many, and whether each
    carries every runner argument verbatim."""
    commands = sorted(glob.glob(str(cache_root / "**" / "command.txt"), recursive=True))
    carrying = 0
    for path in commands:
        argv = shlex.split(pathlib.Path(path).read_text())
        carrying += all(arg in argv for arg in RUNNER_COMPILER_ARGS)
    return {"commands": len(commands), "carrying_every_runner_arg": carrying}


def identity_shapes(sandbox: bool) -> list[tuple[int, int, int]]:
    """``(CP, R, k)`` of the identity leg: the served widths at every row count, the edge
    widths at one row and at the prefill chunk; the sandbox's at :data:`SANDBOX_ROWS`."""
    row_counts = SANDBOX_ROWS if sandbox else ROW_COUNTS
    edge_rows = SANDBOX_ROWS[:1] if sandbox else (1, 1024)
    shapes = [(cp, rows, SELECT_K) for cp in DCP_SIZES for rows in row_counts]
    return shapes + [(cp, rows, k) for cp, k in EDGE_WIDTHS for rows in edge_rows]


def test_cpu_every_device_identity_case_builds() -> None:
    """Every case the device identity leg runs builds, with the operand shapes and dtypes the
    merge validates: an input bug fails here, not on the device."""
    for cp, rows, k in identity_shapes(sandbox=False):
        for case in cases_for(cp, rows, k, (0, 1) if rows <= 129 else (0,)):
            assert tuple(case["values"].shape) == tuple(case["ids"].shape) == (cp, rows, k)
            assert case["values"].dtype == torch.float32 and case["ids"].dtype == torch.int32
            assert merge._validate(case["values"], case["ids"], 0, PPB) == (cp, rows, k)


def failures(report: dict, compiled: bool) -> list:
    """Every check of ``report`` that did not hold, one tuple each."""
    failed = []
    for shape in report["identity"]:
        for rank in shape["ranks"]:
            if rank["counters_at_trace"] != [1, 0] \
                    or rank["counters_after"] != rank["counters_expected_after"] \
                    or rank["kernel"] != list(MERGE_KERNEL) or not rank["emissions_identical"]:
                failed.append(("identity-route", shape["cp"], shape["rows"], shape["k"],
                               rank["cp_rank"]))
            for case in rank["cases"]:
                if not case["equal"] or case["pads_emitted"]:
                    failed.append(("identity", shape["cp"], shape["rows"], shape["k"],
                                   rank["cp_rank"], case["pattern"], case["seed"]))
    for shape in report["multi"]:
        for rank in shape["ranks"]:
            for i, call in enumerate(rank["calls"]):
                if not call["equal"] or call["pads_emitted"]:
                    failed.append(("multi", shape["cp"], shape["rows"], rank["cp_rank"], i))
    for shape in report["chain"]:
        for ctx in shape["contexts"]:
            if not ctx["composed_equals_one_rank"]:
                failed.append(("chain-composed", shape["cp"], ctx["context"]))
            for rank in ctx["ranks"]:
                if not (rank["equal"] and rank["lists_equal"]):
                    failed.append(("chain", shape["cp"], ctx["context"], rank["cp_rank"]))
    if compiled:
        cc = report["compile_commands"]
        if cc["commands"] == 0 or cc["carrying_every_runner_arg"] != cc["commands"]:
            failed.append(("compile-commands", cc["commands"], cc["carrying_every_runner_arg"]))
    return failed


def _check_environment(sandbox: bool) -> None:
    """The device run needs the device route; the sandbox needs the simulator route."""
    if sandbox:
        for name in ("VLLM_NEURON_CPU_MODE", "NKI_SIMULATOR"):
            if os.environ.get(name) != "1":
                raise ValueError(f"the sandbox needs {name}=1 (the NKI simulator route)")
    else:
        for name in ("VLLM_NEURON_CPU_MODE", "NKI_SIMULATOR"):
            if os.environ.get(name) not in (None, "", "0"):
                raise ValueError(f"{name}={os.environ[name]!r}: the device run needs it unset")
        if not os.environ.get("NEURON_RT_VISIBLE_CORES"):
            raise ValueError("NEURON_RT_VISIBLE_CORES must name the cores (set by the caller)")
    if os.environ.get("NEURON_LOGICAL_NC_CONFIG") != "2":
        raise ValueError("the run needs NEURON_LOGICAL_NC_CONFIG=2, the LNC2 grid")
    if not pathlib.Path(merge.__file__).resolve().is_relative_to(REPO_ROOT):
        raise RuntimeError(f"{merge.__name__} imported from {merge.__file__}")


def device_main(args) -> int:
    """The device entry, or with ``args.sandbox`` the same legs and checks on the CPU at
    :data:`SANDBOX_ROWS`. Writes ``args.out`` after every shape; returns 1 on any failure."""
    _check_environment(args.sandbox)
    target = SANDBOX if args.sandbox else DEVICE
    cache_root = args.cache_root.resolve()
    if cache_root.exists() and any(cache_root.iterdir()):
        raise ValueError(f"--cache-root {cache_root} is not empty")
    cache_root.mkdir(parents=True, exist_ok=True)
    scratch = cache_root / "cwd"
    scratch.mkdir()
    if target.compiled:
        os.environ["NEURON_LIBTORCH_CACHE_ROOT"] = str(cache_root)
        torch._dynamo.config.cache_size_limit = 4096
    os.chdir(scratch)
    head = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    report = {"target": dataclasses.asdict(target),
              "environment": {n: os.environ.get(n) for n in (
                  "NEURON_RT_VISIBLE_CORES", "NEURON_LOGICAL_NC_CONFIG",
                  "NEURON_PLATFORM_TARGET_OVERRIDE", "NEURON_LIBTORCH_CACHE_ROOT")},
              "module": merge.__file__, "source_digest": merge.SOURCE_DIGEST, "head": head,
              "compiler_args": list(RUNNER_COMPILER_ARGS), "select_k": SELECT_K, "ppb": PPB,
              "local_pools": LOCAL_POOLS, "seeds": args.seeds, "legs": args.legs,
              "identity": [], "multi": [], "chain": []}

    def write():
        args.out.write_text(json.dumps(report, indent=1) + "\n")

    multi_rows = SANDBOX_ROWS[:1] if args.sandbox else (1, 1024)
    t0 = time.time()
    if "identity" in args.legs:
        for cp, rows, k in identity_shapes(args.sandbox):
            report["identity"].append(
                device_identity(target, cp, rows, k, args.seeds, args.emissions))
            write()
    if "multi" in args.legs:
        for cp in DCP_SIZES:
            for rows in multi_rows:
                report["multi"].append(device_multi(target, cp, rows, (0, cp - 1)))
                write()
    if "chain" in args.legs:
        for cp in DCP_SIZES:
            report["chain"].append(device_chain(target, cp, args.seeds[0]))
            write()
    report["seconds"] = round(time.time() - t0, 1)
    report["compile_commands"] = compile_commands(cache_root)
    report["failed"] = failures(report, target.compiled)
    write()
    print(json.dumps({"failed": len(report["failed"]), "first": report["failed"][:5],
                      "seconds": report["seconds"],
                      "compile_commands": report["compile_commands"]}), flush=True)
    return 1 if report["failed"] else 0


def device_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    device = sub.add_parser("device", help="the device run (see the module docstring)")
    device.add_argument("--out", type=pathlib.Path, required=True)
    device.add_argument("--cache-root", type=pathlib.Path, required=True,
                        help="a new or empty directory: NEURON_LIBTORCH_CACHE_ROOT for the run")
    device.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    device.add_argument("--emissions", type=int, default=4)
    device.add_argument("--legs", nargs="+", default=["identity", "multi", "chain"],
                        choices=["identity", "multi", "chain"])
    device.add_argument("--sandbox", action="store_true",
                        help="run the legs eagerly on the CPU through the NKI simulator")
    device.add_argument("--time-limit", type=int, default=18000,
                        help="seconds of wall time before the run ends itself (0: no limit)")
    build = sub.add_parser("build-one", help="one CPU build (the build test's child)")
    build.add_argument("cp", type=int)
    build.add_argument("rows", type=int)
    build.add_argument("k", type=int)
    build.add_argument("work")
    build.add_argument("--unfolded", action="store_true",
                       help="128-row tiles whatever the width (the build test's control)")
    return parser


def test_cpu_sandbox_runs_the_device_entry(tmp_path, monkeypatch) -> None:
    """The device entry's legs and checks, eagerly on the simulator: it exits 0 and its record
    holds every leg."""
    _simulate(monkeypatch)
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "sandbox.json"
    args = device_parser().parse_args(["device", "--sandbox", "--out", str(out),
                                       "--cache-root", str(tmp_path / "cache"),
                                       "--seeds", "0", "--emissions", "2"])
    assert device_main(args) == 0
    report = json.loads(out.read_text())
    assert report["failed"] == [] and report["target"] == {"device": "cpu", "compiled": False}
    assert len(report["identity"]) == len(DCP_SIZES) * len(SANDBOX_ROWS) + len(EDGE_WIDTHS)
    assert len(report["multi"]) == len(DCP_SIZES) and len(report["chain"]) == len(DCP_SIZES)
    for shape in report["identity"]:
        for rank in shape["ranks"]:
            assert rank["counters_after"][0] == len(rank["cases"]) > 0
            assert rank["kernel"] == list(MERGE_KERNEL)


def main() -> None:
    args = device_parser().parse_args()
    if args.mode == "build-one":
        _build_one(args.cp, args.rows, args.k, args.work, args.unfolded)
        return
    if args.time_limit:
        # A hung device call would hold the cores; SIGALRM's default action ends the run.
        signal.alarm(args.time_limit)
    args.out = args.out.resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sys.exit(device_main(args))


if __name__ == "__main__":
    main()
