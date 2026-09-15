# SPDX-License-Identifier: Apache-2.0
"""Acceptance for ``dsa_index_expand``: pool ids expanded to token indices, bit-identically.

WHAT IS BEING ASSERTED, and what each reading is worth. The expansion is INTEGER INDEX ARITHMETIC, so
there is no tolerance to spend: the kernel is either bit-identical to the reference or it is wrong. Every
declared case therefore asserts ``max abs diff == 0`` and, separately, that every produced entry is
either the ``-1`` sentinel or lies in ``[0, seq_len)`` for ITS OWN ROW.

**THE CALLER PRECONDITION, stated here because the in-range assertion only means something under it.**

    every non-negative pool id satisfies  0 <= pool_ids[row, g] < seq_lens[row] // pool_size

This is upstream's contract, not an addition: its kernel gates only on ``pool_ids >= 0``
(``kpool_compress.py:846-848``) and never compares an expanded index against the sequence length, so a
pool id past the row's last pool expands past the end of the sequence and BOTH the kernel and the
reference do exactly that. Every declared fixture below satisfies the precondition, and one item checks
that on the fixtures themselves rather than trusting the comment. Under the precondition the in-range
count is a LIVE GATE on the ``pid * pool_size + o`` arithmetic: an off-by-one in the multiply or the
offset pushes an index outside the row's range and the count moves off zero.

**A ``-1`` IS A VALUE, NOT AN OUT-OF-BOUNDS INDEX.** It means "this column selects no token", and it
arises for two independent reasons -- a ``-1`` pool id, or a tail column past the row's tail count. It is
excluded BY NAME from the in-range population, and the bit-identical assertion is what covers it: a
sentinel that appeared or vanished in the wrong place would move the diff off zero.

THE ZEROS HERE OWN FIRING CONTROLS. The supplementary item at the bottom feeds a pool id PAST its row's
last pool on purpose and reads a NON-ZERO out-of-range count from the same reader the declared cases read
zero from, so "0 out of range" is a measurement rather than a reader that cannot fire. The torch-fallback
zero is controlled the same way, by handing the seam a non-power-of-two ``pool_size``.

ONE ITEM PER COUNTED CONJUNCT and no ``parametrize`` (plan section 6, rules 4b and 6), so a failure names
the conjunct that failed. Counters are reset at the START of each declared case (section 4b), and the
supplementary case runs in its OWN reset window and is excluded from the declared total of 4.

**THE WIDTH THIS FILE ASSERTS MOVED AT ``inc-glm53f-102``, and the reason is one sentence.** The
expansion used to emit the RAW width -- ``n_groups * pool_size + pool_size - 1``, the history columns plus
the forced tail -- and that width is never a whole number of ``KEY_CHUNK`` columns for any ``pool_size``
above one, so the sparse attention kernel that consumes it refused every width this module could produce.
It now emits the raw width rounded UP to ``KEY_CHUNK``, with the padded columns carrying the same ``-1``
sentinel a tail column already carries. The declared cases above therefore state their shape as
``index_expand_width(...)`` rather than as the raw formula, and the ADMISSIBILITY block at the bottom of
this file -- five items, selected with ``-k admissible`` -- is what settles that the new width is served,
that the padding is sentinel on both routes, that the landed declarations moved with the rule, and that
the emitted width is the width upstream allocates for the same geometry by its own expression.
"""

from __future__ import annotations

import ast
import pathlib

import pytest
import torch

from vllm_neuron.functional.attention import mla_sparse as ms
from vllm_neuron.functional.attention.mla_sparse import (
    KEY_CHUNK,
    MlaSparseAttentionError,
)
from vllm_neuron.functional.dsa import index_expand as mod
from vllm_neuron.functional.dsa.index_expand import (
    INDEX_KPOOL,
    IndexExpandError,
    _dsa_index_expand_torch,
    can_run_dsa_index_expand,
    dsa_index_expand,
    index_expand_dispatch_counters,
    index_expand_kernel_identity,
    index_expand_raw_width,
    index_expand_width,
    is_power_of_two,
    reset_index_expand_dispatch_counters,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

POOL_SIZE = 4
"""The pool size every declared case uses -- this checkpoint's compress ratio, and a power of two."""

N_GROUPS_SMALL = 8
"""Selected pools per row in cases 1 to 3. Small on purpose: the history loop is ``pool_size`` long
however wide the selection is, so a larger count adds simulator time without a distinct reading. Case 4
carries the production width instead."""

N_GROUPS_C4 = 512
"""Selected pools per row in case 4 -- the landed ``dsa_topk_select``'s ``select_k``, so case 4 is the
width that actually ships: 2048 history columns, 2051 raw columns and 2176 emitted."""

N_GROUPS_TINY = 2
"""The tiny geometry the admissibility items carry: k = 2 GROUPS at pool 4, raw width 11, emitted 128.

THE UNIT IS NAMED IN THE NAME, and that is not decoration. Read as a group count this is 11 raw columns
(``4 * (2 + 1) - 1``); read as a token budget it would be 5 (``2 + 4 - 1``), which is what a ruling
draft said before the reading corrected it -- and which is not even a geometry either side can build,
since 2 tokens over a pool of 4 is half a group. ``probe-102-tiny-geometry.out`` settles it 17/17."""

SEQ_TINY = 12
"""The tiny geometry's sequence length: three complete pools at pool 4, so selecting 2 is meaningful.
This is ``-051``'s declared minimum and it corroborates the group reading independently."""

SEQS_TINY = [SEQ_TINY, SEQ_TINY + 3]
"""The tiny geometry's two rows, and the SECOND ROW IS THREE TOKENS LONGER ON PURPOSE.

At 12 tokens the tail count is 0, so every column from the history width out to the emitted width is
``-1`` and a tail sentinel is indistinguishable from a padding sentinel -- the tiny geometry would then
carry no reading about WHERE the padding starts. At 15 tokens the tail count is 3, so raw column 10 is
the real token 14 and column 11 is the first padded ``-1``. That pair is what makes an off-by-one in the
padding boundary visible at the cheap geometry as well as at the production one."""


def _emit(tag: str, **values: object) -> None:
    """Print one MACHINE-READABLE reading line, for the driver to re-check independently.

    The pattern is the landed one at ``test_score_gemm.py:102-110``: the item asserts, and then PRINTS
    the value it asserted on, so the driver that owns the transcript can check the same number without
    trusting this file's own verdict.
    """
    body = " ".join(f"{k}={v}" for k, v in values.items())
    print(f"S048|{tag}|{body}", flush=True)


def _tensors(pids: list[list[int]], seqs: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """One case's two inputs at the shapes the seam declares: ``[rows, n_groups]`` and ``[rows]``."""
    return (torch.tensor(pids, dtype=torch.int32), torch.tensor(seqs, dtype=torch.int32))


def _diff(got: torch.Tensor, want: torch.Tensor) -> int:
    """Max abs difference in int64, so nothing wraps. Reported whether or not the item passes."""
    return int((got.to(torch.int64) - want.to(torch.int64)).abs().max().item())


def _range_counts(got: torch.Tensor, seqs: list[int]) -> tuple[int, int, int, int]:
    """``(in_range, out_of_bounds, sentinels, population)`` with the sentinel excluded BY NAME.

    ``in_range`` and ``out_of_bounds`` are counted over NON-SENTINEL entries only, per row against that
    row's own sequence length. ``population`` is every entry, so the two counts are always readable
    against the total they came from.
    """
    in_range = out_of_bounds = sentinels = 0
    for r, limit in enumerate(seqs):
        for value in got[r].tolist():
            value = int(value)
            if value == -1:
                sentinels += 1
            elif 0 <= value < limit:
                in_range += 1
            else:
                out_of_bounds += 1
    return (in_range, out_of_bounds, sentinels, int(got.numel()))


def _precondition_violations(
    pids: list[list[int]], seqs: list[int], pool_size: int
) -> list[tuple[int, int]]:
    """Every ``(row, pool_id)`` that breaks the caller precondition. Empty means the fixture is legal."""
    return [
        (r, int(p))
        for r in range(len(pids))
        for p in pids[r]
        if int(p) != -1 and not (0 <= int(p) < seqs[r] // pool_size)
    ]


def _python_oracle(pids: list[list[int]], seqs: list[int], pool_size: int) -> list[list[int]]:
    """A THIRD spelling of the expansion, in plain python loops with no torch and no vectorisation.

    The module's ``_dsa_index_expand_torch`` is upstream's ``where`` form and the kernel is a closed
    form in max and min; this is neither. It exists so that one item can check the REFERENCE itself
    instead of only checking the kernel against it -- if the reference drifted, every other item here
    would drift with it silently.

    IT NOW BUILDS AT THE EMITTED WIDTH, AND IT ASKS THE MODULE FOR IT RATHER THAN RE-DERIVING IT. That
    looks like it weakens the independence and it does not: this oracle's independence is about the
    VALUES -- a third spelling of the index arithmetic -- and the width has two dedicated items of its
    own. Spelling the ceiling a fourth time here would add a place to disagree without adding a reading.
    The padded columns need no new branch: for any column at or past the raw width the tail offset is at
    least ``pool_size - 1`` and the tail count is at most ``pool_size - 1``, so the existing ``else``
    already appends ``-1``.
    """
    rows = len(pids)
    n_groups = len(pids[0])
    topk = n_groups * pool_size
    rowsout = []
    for r in range(rows):
        seq = seqs[r]
        tail_start = (seq // pool_size) * pool_size
        tail_count = seq - tail_start
        row = []
        for col in range(index_expand_width(n_groups, pool_size)):
            if col < topk:
                pid = pids[r][col // pool_size]
                row.append(pid * pool_size + col % pool_size if pid >= 0 else -1)
            else:
                offset = col - topk
                row.append(tail_start + offset if offset < tail_count else -1)
        rowsout.append(row)
    return rowsout


def _module_source() -> str:
    """The module's own committed bytes. Read from disk, so the reading is about what ships."""
    return pathlib.Path(mod.__file__).read_text(encoding="utf-8")


# THE FOUR DECLARED CASES, exactly as `predictions-048-sizing.txt` declares them.
C1_PIDS = [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]]
C1_SEQS = [32, 40]
"""CASE 1 -- even tail, every pool valid. pool_len 8 and 10, tail_count 0 and 0: every tail column -1."""

C2_PIDS = [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]]
C2_SEQS = [33, 35]
"""CASE 2 -- non-multiple tail. tail_count 1 and 3, so both ends of ``[1, pool_size)`` are covered."""

C3_PIDS = [[0, 1, 2, 3, 4, 5, 6, -1], [-1, 1, 2, -1, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]]
C3_SEQS = [33, 34, 39]
"""CASE 3 -- negative pool ids present on two rows, tail_count 1, 2 and 3. Both sentinel sources are
live in one case: a negative pool id, and a tail column past the tail count."""

C4_PIDS = [list(range(N_GROUPS_C4)), list(range(N_GROUPS_C4))]
C4_SEQS = [2050, 2051]
"""CASE 4 -- production width. topk 2048, raw_cols 2051, out_cols 2176, tail_count 2 and 3."""


# ---------------------------------------------------------------------------------------------
# Declared case 1 of 4 -- even tail
# ---------------------------------------------------------------------------------------------


def test_declared_case_1_bit_identical():
    """DECLARED CASE 1: the expansion is bit-identical to the reference, max abs diff exactly 0."""
    reset_index_expand_dispatch_counters()
    pids, seqs = _tensors(C1_PIDS, C1_SEQS)
    got = dsa_index_expand(pids, seqs, POOL_SIZE)
    want = _dsa_index_expand_torch(pids, seqs, POOL_SIZE)
    worst = _diff(got, want)
    _emit("CASE1_BITIDENTICAL", rows=len(C1_SEQS), n_groups=N_GROUPS_SMALL, pool_size=POOL_SIZE,
          seq_lens=C1_SEQS, max_abs_diff=worst, population=got.numel())
    # RE-DECLARED at `-102` from the raw width to the emitted one, and asked of the module rather than
    # spelled here. The old expectation was `N_GROUPS_SMALL * POOL_SIZE + POOL_SIZE - 1` = 35 columns,
    # which `mla_sparse_attention` refuses; the admissibility items below show that refusal.
    assert got.shape == (len(C1_SEQS), index_expand_width(N_GROUPS_SMALL, POOL_SIZE))
    assert got.dtype is torch.int32
    assert worst == 0


def test_declared_case_1_every_entry_is_sentinel_or_in_range():
    """DECLARED CASE 1's in-range reading: 0 out of bounds, with the sentinel excluded by name."""
    reset_index_expand_dispatch_counters()
    pids, seqs = _tensors(C1_PIDS, C1_SEQS)
    got = dsa_index_expand(pids, seqs, POOL_SIZE)
    in_range, oob, sentinels, population = _range_counts(got, C1_SEQS)
    _emit("CASE1_INRANGE", in_range=in_range, out_of_bounds=oob, sentinels=sentinels,
          population=population, nonsentinel_population=in_range + oob)
    assert oob == 0
    assert in_range + oob + sentinels == population


def test_declared_case_1_tail_columns_are_all_sentinel():
    """DECLARED CASE 1's own distinguishing reading: tail_count is 0 on both rows, so every tail is -1."""
    pids, seqs = _tensors(C1_PIDS, C1_SEQS)
    got = dsa_index_expand(pids, seqs, POOL_SIZE)
    topk = N_GROUPS_SMALL * POOL_SIZE
    tail = got[:, topk:]
    _emit("CASE1_TAIL", tail_columns=tail.tolist(),
          tail_counts=[s % POOL_SIZE for s in C1_SEQS], population=tail.numel())
    assert [s % POOL_SIZE for s in C1_SEQS] == [0, 0]
    assert bool((tail == -1).all().item())


def test_declared_case_1_route_predicate_one_dispatch():
    """DECLARED CASE 1's route predicate, form R-1: gate True, 1 NKI dispatch, 0 torch fallbacks."""
    reset_index_expand_dispatch_counters()
    pids, seqs = _tensors(C1_PIDS, C1_SEQS)
    admitted = can_run_dsa_index_expand(pids, seqs, POOL_SIZE)
    dsa_index_expand(pids, seqs, POOL_SIZE)
    nki_n, fallback_n = index_expand_dispatch_counters()
    _emit("CASE1_ROUTE", can_run=admitted, nki_dispatch=nki_n, torch_fallback=fallback_n)
    assert admitted is True
    assert nki_n == 1
    assert fallback_n == 0


# ---------------------------------------------------------------------------------------------
# Declared case 2 of 4 -- the non-multiple tail
# ---------------------------------------------------------------------------------------------


def test_declared_case_2_bit_identical():
    """DECLARED CASE 2: a non-multiple tail expands bit-identically, max abs diff exactly 0."""
    reset_index_expand_dispatch_counters()
    pids, seqs = _tensors(C2_PIDS, C2_SEQS)
    got = dsa_index_expand(pids, seqs, POOL_SIZE)
    want = _dsa_index_expand_torch(pids, seqs, POOL_SIZE)
    worst = _diff(got, want)
    _emit("CASE2_BITIDENTICAL", seq_lens=C2_SEQS, tail_counts=[s % POOL_SIZE for s in C2_SEQS],
          max_abs_diff=worst, population=got.numel())
    assert worst == 0


def test_declared_case_2_every_entry_is_sentinel_or_in_range():
    """DECLARED CASE 2's in-range reading: 0 out of bounds over the non-sentinel population."""
    reset_index_expand_dispatch_counters()
    pids, seqs = _tensors(C2_PIDS, C2_SEQS)
    got = dsa_index_expand(pids, seqs, POOL_SIZE)
    in_range, oob, sentinels, population = _range_counts(got, C2_SEQS)
    _emit("CASE2_INRANGE", in_range=in_range, out_of_bounds=oob, sentinels=sentinels,
          population=population, nonsentinel_population=in_range + oob)
    assert oob == 0


def test_declared_case_2_tail_counts_are_one_and_three():
    """DECLARED CASE 2's distinguishing reading: the tail carries 1 token on row 0 and 3 on row 1.

    Both ends of the open interval ``[1, pool_size)`` are covered, which is what makes this the
    non-multiple-tail case rather than a second copy of case 1.
    """
    pids, seqs = _tensors(C2_PIDS, C2_SEQS)
    got = dsa_index_expand(pids, seqs, POOL_SIZE)
    topk = N_GROUPS_SMALL * POOL_SIZE
    tail = got[:, topk:]
    carried = [int((row != -1).sum().item()) for row in tail]
    _emit("CASE2_TAIL", tail_columns=tail.tolist(), tokens_carried=carried,
          expected=[s % POOL_SIZE for s in C2_SEQS])
    assert carried == [1, 3]
    assert carried == [s % POOL_SIZE for s in C2_SEQS]


def test_declared_case_2_route_predicate_one_dispatch():
    """DECLARED CASE 2's route predicate, form R-1: gate True, 1 NKI dispatch, 0 torch fallbacks."""
    reset_index_expand_dispatch_counters()
    pids, seqs = _tensors(C2_PIDS, C2_SEQS)
    admitted = can_run_dsa_index_expand(pids, seqs, POOL_SIZE)
    dsa_index_expand(pids, seqs, POOL_SIZE)
    nki_n, fallback_n = index_expand_dispatch_counters()
    _emit("CASE2_ROUTE", can_run=admitted, nki_dispatch=nki_n, torch_fallback=fallback_n)
    assert admitted is True
    assert nki_n == 1
    assert fallback_n == 0


# ---------------------------------------------------------------------------------------------
# Declared case 3 of 4 -- negative pool ids
# ---------------------------------------------------------------------------------------------


def test_declared_case_3_bit_identical():
    """DECLARED CASE 3: negative pool ids expand bit-identically, max abs diff exactly 0."""
    reset_index_expand_dispatch_counters()
    pids, seqs = _tensors(C3_PIDS, C3_SEQS)
    got = dsa_index_expand(pids, seqs, POOL_SIZE)
    want = _dsa_index_expand_torch(pids, seqs, POOL_SIZE)
    worst = _diff(got, want)
    _emit("CASE3_BITIDENTICAL", seq_lens=C3_SEQS, max_abs_diff=worst, population=got.numel(),
          negative_pool_ids=sum(1 for r in C3_PIDS for p in r if p < 0),
          pool_id_population=sum(len(r) for r in C3_PIDS))
    assert worst == 0


def test_declared_case_3_every_entry_is_sentinel_or_in_range():
    """DECLARED CASE 3's in-range reading: 0 out of bounds even with sentinels from both sources."""
    reset_index_expand_dispatch_counters()
    pids, seqs = _tensors(C3_PIDS, C3_SEQS)
    got = dsa_index_expand(pids, seqs, POOL_SIZE)
    in_range, oob, sentinels, population = _range_counts(got, C3_SEQS)
    _emit("CASE3_INRANGE", in_range=in_range, out_of_bounds=oob, sentinels=sentinels,
          population=population, nonsentinel_population=in_range + oob)
    assert oob == 0


def test_declared_case_3_negative_pool_ids_become_exactly_minus_one():
    """DECLARED CASE 3's distinguishing reading: every column of a ``-1`` pool id reads exactly ``-1``.

    This is the closed form's whole claim -- ``max(pid * pool_size + o, -1)`` pins a negative pool id
    to the sentinel without a compare and without a select. A leak would show up here as a value like
    ``-4`` or ``-3`` rather than ``-1``.
    """
    pids, seqs = _tensors(C3_PIDS, C3_SEQS)
    got = dsa_index_expand(pids, seqs, POOL_SIZE)
    leaked: list[int] = []
    expected = 0
    for r, row in enumerate(C3_PIDS):
        for g, pid in enumerate(row):
            if pid >= 0:
                continue
            expected += POOL_SIZE
            block = got[r, g * POOL_SIZE:(g + 1) * POOL_SIZE].tolist()
            leaked.extend(int(v) for v in block if int(v) != -1)
    _emit("CASE3_SENTINEL_BLOCKS", negative_id_columns=expected, leaked_values=leaked,
          leaked_count=len(leaked))
    assert expected == 12
    assert leaked == []


def test_declared_case_3_route_predicate_one_dispatch():
    """DECLARED CASE 3's route predicate, form R-1: gate True, 1 NKI dispatch, 0 torch fallbacks."""
    reset_index_expand_dispatch_counters()
    pids, seqs = _tensors(C3_PIDS, C3_SEQS)
    admitted = can_run_dsa_index_expand(pids, seqs, POOL_SIZE)
    dsa_index_expand(pids, seqs, POOL_SIZE)
    nki_n, fallback_n = index_expand_dispatch_counters()
    _emit("CASE3_ROUTE", can_run=admitted, nki_dispatch=nki_n, torch_fallback=fallback_n)
    assert admitted is True
    assert nki_n == 1
    assert fallback_n == 0


# ---------------------------------------------------------------------------------------------
# Declared case 4 of 4 -- the production width
# ---------------------------------------------------------------------------------------------


def test_declared_case_4_bit_identical_at_production_width():
    """DECLARED CASE 4: 512 selected pools, 2051 raw columns emitted as 2176, max abs diff exactly 0.

    This is the width that ships. The history loop is ``pool_size`` iterations here exactly as it is in
    case 1 -- that independence from the pool count is why this case costs what case 1 costs.

    THE DECLARED WIDTH MOVED AT ``-102``, from 2051 to 2176. The 2051 columns still carry every
    meaningful value; the 125 after them are the ``-1`` padding that makes the width a whole number of
    ``KEY_CHUNK`` columns so ``mla_sparse_attention`` admits it. Upstream allocates the identical 2176
    (``models/glm5next/nvidia/model.py:594-599``), which the parity item below asserts.
    """
    reset_index_expand_dispatch_counters()
    pids, seqs = _tensors(C4_PIDS, C4_SEQS)
    got = dsa_index_expand(pids, seqs, POOL_SIZE)
    want = _dsa_index_expand_torch(pids, seqs, POOL_SIZE)
    worst = _diff(got, want)
    _emit("CASE4_BITIDENTICAL", rows=len(C4_SEQS), n_groups=N_GROUPS_C4, seq_lens=C4_SEQS,
          raw_cols=index_expand_raw_width(N_GROUPS_C4, POOL_SIZE),
          out_cols=int(got.shape[1]), max_abs_diff=worst, population=got.numel())
    # RE-DECLARED at `-102`: the formula spelling asks the module, and the literal moves 2051 -> 2176.
    # Both spellings are kept on purpose -- the derived one catches a change in the rule, the literal
    # catches a change in the rule that happens to leave the derivation self-consistent.
    assert got.shape == (2, index_expand_width(N_GROUPS_C4, POOL_SIZE))
    assert int(got.shape[1]) == 2176
    assert index_expand_raw_width(N_GROUPS_C4, POOL_SIZE) == 2051
    assert worst == 0


def test_declared_case_4_every_entry_is_sentinel_or_in_range():
    """DECLARED CASE 4's in-range reading: 0 out of bounds over two rows of the padded width.

    RE-DECLARED at ``-102`` alongside the three declarations in the test above. This reading counted
    its own entries by writing the total down, and the total is two rows of the emitted width -- so
    it moved when the emitted width moved, and the counted run is what found it. The count now asks
    the module for the width instead of naming a number, which is the same treatment the other
    declarations got and the reason the thirty-five readings that never wrote their widths down
    survived this increment untouched.
    """
    reset_index_expand_dispatch_counters()
    pids, seqs = _tensors(C4_PIDS, C4_SEQS)
    got = dsa_index_expand(pids, seqs, POOL_SIZE)
    in_range, oob, sentinels, population = _range_counts(got, C4_SEQS)
    _emit("CASE4_INRANGE", in_range=in_range, out_of_bounds=oob, sentinels=sentinels,
          population=population, nonsentinel_population=in_range + oob)
    assert oob == 0
    assert population == 2 * index_expand_width(N_GROUPS_C4, POOL_SIZE)


def test_declared_case_4_tail_counts_are_two_and_three():
    """DECLARED CASE 4's distinguishing reading: the tail carries 2 tokens on row 0 and 3 on row 1."""
    pids, seqs = _tensors(C4_PIDS, C4_SEQS)
    got = dsa_index_expand(pids, seqs, POOL_SIZE)
    topk = N_GROUPS_C4 * POOL_SIZE
    tail = got[:, topk:]
    carried = [int((row != -1).sum().item()) for row in tail]
    _emit("CASE4_TAIL", tail_columns=tail.tolist(), tokens_carried=carried,
          expected=[s % POOL_SIZE for s in C4_SEQS])
    assert carried == [2, 3]
    assert carried == [s % POOL_SIZE for s in C4_SEQS]


def test_declared_case_4_route_predicate_one_dispatch():
    """DECLARED CASE 4's route predicate, form R-1: gate True, 1 NKI dispatch, 0 torch fallbacks."""
    reset_index_expand_dispatch_counters()
    pids, seqs = _tensors(C4_PIDS, C4_SEQS)
    admitted = can_run_dsa_index_expand(pids, seqs, POOL_SIZE)
    dsa_index_expand(pids, seqs, POOL_SIZE)
    nki_n, fallback_n = index_expand_dispatch_counters()
    _emit("CASE4_ROUTE", can_run=admitted, nki_dispatch=nki_n, torch_fallback=fallback_n)
    assert admitted is True
    assert nki_n == 1
    assert fallback_n == 0


# ---------------------------------------------------------------------------------------------
# The declared case set as a whole
# ---------------------------------------------------------------------------------------------


def test_declared_case_set_total_dispatches_is_four():
    """The declared total: four cases, one NKI dispatch each, zero torch fallbacks, ONE reset window."""
    reset_index_expand_dispatch_counters()
    for pids_raw, seqs_raw in ((C1_PIDS, C1_SEQS), (C2_PIDS, C2_SEQS),
                               (C3_PIDS, C3_SEQS), (C4_PIDS, C4_SEQS)):
        pids, seqs = _tensors(pids_raw, seqs_raw)
        dsa_index_expand(pids, seqs, POOL_SIZE)
    nki_n, fallback_n = index_expand_dispatch_counters()
    _emit("DECLARED_TOTAL", cases=4, nki_dispatch=nki_n, torch_fallback=fallback_n)
    assert nki_n == 4
    assert fallback_n == 0


def test_every_declared_fixture_satisfies_the_caller_precondition():
    """The precondition is checked on the FIXTURES, not only asserted about the output.

    A case that silently violated it would still pass the bit-identical assertion -- the reference
    reproduces upstream's out-of-range expansion faithfully -- while quietly weakening the in-range
    gate to nothing. So the fixtures are checked directly.
    """
    violations = {
        "C1": _precondition_violations(C1_PIDS, C1_SEQS, POOL_SIZE),
        "C2": _precondition_violations(C2_PIDS, C2_SEQS, POOL_SIZE),
        "C3": _precondition_violations(C3_PIDS, C3_SEQS, POOL_SIZE),
        "C4": _precondition_violations(C4_PIDS, C4_SEQS, POOL_SIZE),
    }
    _emit("PRECONDITION", violations={k: len(v) for k, v in violations.items()},
          pool_id_population=sum(len(r) for c in (C1_PIDS, C2_PIDS, C3_PIDS, C4_PIDS) for r in c))
    assert violations == {"C1": [], "C2": [], "C3": [], "C4": []}


def test_control_the_precondition_reader_fires_on_a_violating_fixture():
    """CONTROL: the precondition reader is not a function that always returns empty."""
    violations = _precondition_violations([[0, 99]], [32], POOL_SIZE)
    _emit("PRECONDITION_CONTROL", violations=violations)
    assert violations == [(0, 99)]


def test_reference_agrees_with_an_independently_spelled_python_oracle():
    """The REFERENCE itself is checked, in a third spelling: plain python loops, no torch.

    The kernel is a closed form in max and min, the module's reference is upstream's ``where`` form,
    and this is neither. If the reference had drifted, every case above would have drifted with it and
    said nothing.
    """
    pids, seqs = _tensors(C3_PIDS, C3_SEQS)
    want = _dsa_index_expand_torch(pids, seqs, POOL_SIZE)
    plain = _python_oracle(C3_PIDS, C3_SEQS, POOL_SIZE)
    _emit("REFERENCE_CROSSCHECK", rows=len(C3_SEQS), agree=want.tolist() == plain,
          population=want.numel())
    assert want.tolist() == plain


def test_control_the_fixture_discriminates_the_sentinel_rule():
    """CONTROL: case 3 can tell the closed form from a plausible wrong one.

    The wrong form drops the ``max(..., -1)`` and lets a negative pool id expand arithmetically. If
    case 3's fixture could not tell the two apart, its agreement would be worth nothing.

    THE EXPECTED COUNT IS 9 AND NOT 12, and the missing 3 are worth naming: for a ``-1`` pool id the
    naive form gives ``-4, -3, -2, -1`` across the four offsets, so the LAST offset coincides with the
    sentinel by arithmetic accident. Three negative ids x three differing offsets is 9. A control whose
    expected count was reasoned from "3 ids x 4 columns" would have been wrong about its own instrument.
    """
    pids, seqs = _tensors(C3_PIDS, C3_SEQS)
    want = _dsa_index_expand_torch(pids, seqs, POOL_SIZE)
    wrong = _python_oracle(C3_PIDS, C3_SEQS, POOL_SIZE)
    for r, row in enumerate(C3_PIDS):
        for g, pid in enumerate(row):
            if pid >= 0:
                continue
            for o in range(POOL_SIZE):
                wrong[r][g * POOL_SIZE + o] = pid * POOL_SIZE + o
    differing = sum(1 for r in range(len(C3_SEQS))
                    for c in range(want.shape[1])
                    if int(want[r, c].item()) != wrong[r][c])
    _emit("DISCRIMINATION_CONTROL", differing=differing, population=want.numel(),
          negative_ids=3, differing_offsets_per_id=POOL_SIZE - 1)
    assert differing == 9


# ---------------------------------------------------------------------------------------------
# The labelled supplementary item -- OUTSIDE the declared total, its own reset window
# ---------------------------------------------------------------------------------------------


def test_supplementary_out_of_range_pool_id_reproduces_upstream_behaviour():
    """SUPPLEMENTARY, outside the declared total of 4: a pool id PAST its row's last pool.

    This measures the EDGE of the in-range gate instead of assuming it. Upstream gates only on
    ``pool_ids >= 0``, so a pool id past the row's last pool expands to token indices past the end of
    the sequence, and the kernel and the reference must agree on doing exactly that. It also serves as
    the FIRING CONTROL for every declared case's ``out_of_bounds == 0``: the same reader returns a
    non-zero count here.
    """
    reset_index_expand_dispatch_counters()
    pids_raw = [[0, 1, 2, 3, 4, 5, 6, 20]]
    seqs_raw = [33]
    violations = _precondition_violations(pids_raw, seqs_raw, POOL_SIZE)
    pids, seqs = _tensors(pids_raw, seqs_raw)
    got = dsa_index_expand(pids, seqs, POOL_SIZE)
    want = _dsa_index_expand_torch(pids, seqs, POOL_SIZE)
    worst = _diff(got, want)
    in_range, oob, sentinels, population = _range_counts(got, seqs_raw)
    nki_n, fallback_n = index_expand_dispatch_counters()
    _emit("SUPPLEMENTARY_OUT_OF_RANGE", violates_precondition=len(violations),
          max_abs_diff=worst, out_of_bounds=oob, in_range=in_range, sentinels=sentinels,
          population=population, nki_dispatch=nki_n, torch_fallback=fallback_n)
    assert violations == [(0, 20)]
    assert worst == 0
    assert oob == 4
    assert nki_n == 1
    assert fallback_n == 0


# ---------------------------------------------------------------------------------------------
# The counters, and the controls that prove each zero can move
# ---------------------------------------------------------------------------------------------


def test_torch_reference_dispatches_zero_nki_kernels():
    """Calling the reference directly moves NEITHER counter, which is what makes it usable as oracle."""
    reset_index_expand_dispatch_counters()
    pids, seqs = _tensors(C2_PIDS, C2_SEQS)
    _dsa_index_expand_torch(pids, seqs, POOL_SIZE)
    nki_n, fallback_n = index_expand_dispatch_counters()
    _emit("REFERENCE_COUNTS", nki_dispatch=nki_n, torch_fallback=fallback_n)
    assert nki_n == 0
    assert fallback_n == 0


def test_control_the_dispatch_counter_can_move():
    """CONTROL: the NKI dispatch counter is not a constant zero."""
    reset_index_expand_dispatch_counters()
    before, _ = index_expand_dispatch_counters()
    pids, seqs = _tensors(C1_PIDS, C1_SEQS)
    dsa_index_expand(pids, seqs, POOL_SIZE)
    after, _ = index_expand_dispatch_counters()
    _emit("DISPATCH_CONTROL", before=before, after=after)
    assert before == 0
    assert after == 1


def test_control_torch_fallback_counter_fires_on_a_non_power_of_two_pool_size():
    """CONTROL: the fallback zero can move -- a ``pool_size`` of 3 is refused and served by torch.

    3 is chosen rather than a bad dtype because it is the refusal this module exists to make: the
    tail derivation is exact only for a power of two, and the gate is what keeps the wrong answer
    unreachable.
    """
    reset_index_expand_dispatch_counters()
    pids, seqs = _tensors([[0, 1, 2]], [9])
    admitted = can_run_dsa_index_expand(pids, seqs, 3)
    got = dsa_index_expand(pids, seqs, 3)
    want = _dsa_index_expand_torch(pids, seqs, 3)
    nki_n, fallback_n = index_expand_dispatch_counters()
    _emit("FALLBACK_CONTROL", pool_size=3, can_run=admitted, nki_dispatch=nki_n,
          torch_fallback=fallback_n, max_abs_diff=_diff(got, want))
    assert admitted is False
    assert nki_n == 0
    assert fallback_n == 1


def test_kernel_identity_is_none_before_any_dispatch():
    """Before any dispatch the identity is ``None`` -- "no kernel ran" is distinguishable."""
    reset_index_expand_dispatch_counters()
    identity = index_expand_kernel_identity()
    _emit("IDENTITY_BEFORE", identity=identity)
    assert identity is None


def test_kernel_identity_after_dispatch_names_the_nki_kernel():
    """After a dispatch the identity names THIS module's kernel, derived through the seam (D13.1)."""
    reset_index_expand_dispatch_counters()
    pids, seqs = _tensors(C1_PIDS, C1_SEQS)
    dsa_index_expand(pids, seqs, POOL_SIZE)
    identity = index_expand_kernel_identity()
    _emit("IDENTITY_AFTER", identity=identity)
    assert identity is not None
    module_name, qualname = identity
    assert module_name.endswith("index_expand")
    assert qualname == "_index_expand_nki"


def test_the_kernel_gate_reads_can_run_kernel():
    """The gate consults the house predicate, so a host that cannot run kernels is served by torch."""
    pids, seqs = _tensors(C1_PIDS, C1_SEQS)
    house = bool(can_run_kernel())
    admitted = can_run_dsa_index_expand(pids, seqs, POOL_SIZE)
    _emit("GATE_HOUSE_PREDICATE", can_run_kernel=house, can_run_dsa_index_expand=admitted)
    assert admitted is house


# ---------------------------------------------------------------------------------------------
# The gate, and the power-of-two rule it enforces
# ---------------------------------------------------------------------------------------------


def test_gate_admits_the_declared_shape():
    """The gate admits every declared case's shape and dtype."""
    verdicts = []
    for pids_raw, seqs_raw in ((C1_PIDS, C1_SEQS), (C2_PIDS, C2_SEQS),
                               (C3_PIDS, C3_SEQS), (C4_PIDS, C4_SEQS)):
        pids, seqs = _tensors(pids_raw, seqs_raw)
        verdicts.append(can_run_dsa_index_expand(pids, seqs, POOL_SIZE))
    _emit("GATE_ADMITS", verdicts=verdicts, population=len(verdicts))
    assert verdicts == [True, True, True, True]


def test_gate_refuses_a_non_power_of_two_pool_size():
    """The gate refuses ``pool_size`` 6, because ``seq & (pool_size - 1)`` is not its remainder."""
    pids, seqs = _tensors([[0, 1]], [12])
    admitted = can_run_dsa_index_expand(pids, seqs, 6)
    _emit("GATE_REFUSES_POOL_SIZE", pool_size=6, can_run=admitted,
          is_power_of_two=is_power_of_two(6))
    assert admitted is False


def test_gate_refuses_an_unadmitted_index_dtype():
    """The gate refuses int64 indices; the reference serves them."""
    pids = torch.tensor(C1_PIDS, dtype=torch.int64)
    seqs = torch.tensor(C1_SEQS, dtype=torch.int32)
    admitted = can_run_dsa_index_expand(pids, seqs, POOL_SIZE)
    _emit("GATE_REFUSES_DTYPE", dtype=str(pids.dtype), can_run=admitted)
    assert admitted is False


def test_power_of_two_rule_and_its_control():
    """``is_power_of_two`` accepts 1, 2, 4, 8, 512 and refuses 0, 3, 6, 12, -4.

    One rule, one spelling, so the gate and the docstring cannot drift apart.
    """
    accepted = [n for n in (1, 2, 4, 8, 512) if is_power_of_two(n)]
    refused = [n for n in (0, 3, 6, 12, -4) if not is_power_of_two(n)]
    _emit("POWER_OF_TWO", accepted=accepted, refused=refused)
    assert accepted == [1, 2, 4, 8, 512]
    assert refused == [0, 3, 6, 12, -4]


def test_index_kpool_records_the_checkpoint_compress_ratio():
    """``INDEX_KPOOL`` is the recorded checkpoint value 4, and it is a power of two."""
    _emit("INDEX_KPOOL", value=INDEX_KPOOL, is_power_of_two=is_power_of_two(INDEX_KPOOL))
    assert INDEX_KPOOL == 4
    assert is_power_of_two(INDEX_KPOOL)


# ---------------------------------------------------------------------------------------------
# Malformed calls
# ---------------------------------------------------------------------------------------------


def test_refuses_a_non_2d_pool_ids():
    """A 1-D ``pool_ids`` is a malformed call, not a shape to guess at."""
    with pytest.raises(IndexExpandError, match="2-D"):
        dsa_index_expand(torch.tensor([0, 1], dtype=torch.int32),
                         torch.tensor([8], dtype=torch.int32), POOL_SIZE)


def test_refuses_a_non_1d_seq_lens():
    """A 2-D ``seq_lens`` is refused: upstream's shape is ``[rows]`` and the seam owns the reshape."""
    with pytest.raises(IndexExpandError, match="1-D"):
        dsa_index_expand(torch.tensor([[0, 1]], dtype=torch.int32),
                         torch.tensor([[8]], dtype=torch.int32), POOL_SIZE)


def test_refuses_a_seq_lens_that_does_not_match_the_row_count():
    """One sequence length per row, or the call is malformed."""
    with pytest.raises(IndexExpandError, match="one length per row"):
        dsa_index_expand(torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
                         torch.tensor([8], dtype=torch.int32), POOL_SIZE)


def test_refuses_a_non_positive_pool_size():
    """A ``pool_size`` of 0 is malformed rather than merely unadmitted."""
    with pytest.raises(IndexExpandError, match="pool_size must be positive"):
        dsa_index_expand(torch.tensor([[0, 1]], dtype=torch.int32),
                         torch.tensor([8], dtype=torch.int32), 0)


# ---------------------------------------------------------------------------------------------
# Source-level prohibitions, each with a firing control
# ---------------------------------------------------------------------------------------------


def test_module_source_calls_no_mx_primitive():
    """Kickoff rule 3: no code path calls ``nc_matmul_mx`` or any MX-quantised nkilib kernel."""
    source = _module_source()
    hits = [name for name in ("nc_matmul_mx", "quantize_mx") if f"{name}(" in source]
    _emit("MX_SCREEN", hits=hits, population=len(source.splitlines()))
    assert hits == []


def test_control_the_mx_reader_fires_on_a_planted_call():
    """CONTROL: the MX reader is not a reader that cannot fire."""
    planted = "x = nisa.nc_matmul_mx(dst=d)\n"
    hits = [name for name in ("nc_matmul_mx", "quantize_mx") if f"{name}(" in planted]
    _emit("MX_SCREEN_CONTROL", hits=hits)
    assert hits == ["nc_matmul_mx"]


# ---------------------------------------------------------------------------------------------
# `inc-glm53f-102` -- EXPANSION WIDTH ADMISSIBILITY. Selected with `-k admissible`, five items.
#
# WHY THESE EXIST. Every width this module could emit before `-102` was refused by the kernel that
# consumes it: the raw width is `pool_size - 1` mod `pool_size`, so it is never a whole number of
# `KEY_CHUNK` columns for any `pool_size >= 2`, and `mla_sparse_attention` refuses a selected-row count
# that is not (`mla_sparse.py:1162-1168`). The module now emits the raw width rounded up, padded with the
# same `-1` sentinel, which is what upstream does for the same reason.
#
# THE STREAM MARKER STAYS `S048`. It marks this FILE for the driver that reads the transcript, not the
# increment that added a line, and re-marking half the file would break the landed driver's reader for
# no reading.
#
# THE GATE IS CALLED, NEVER RE-IMPLEMENTED. `ms._require_admissible` is the private entry that
# `mla_sparse_attention` itself calls; reaching for it from a test is the landed precedent at
# `test_mla_sparse.py:1695` and `:1728`. A re-implementation here would assert that my copy of the rule
# agrees with my copy of the rule.
# ---------------------------------------------------------------------------------------------


def _admit(topk: int) -> None:
    """Ask the sparse kernel's own gate whether it serves a selected-row count of ``topk``.

    Every other axis is held at a value the gate admits -- this checkpoint's latent rank and RoPE width,
    one query, one cache row, one head -- so a raise can only be about ``topk``. That is not an
    assumption: the emitted-width calls below do not raise, which is what proves the rest of the
    geometry is admissible, and the raw-width calls then raise from the same starting point.
    """
    ms._require_admissible(
        seq=1,
        heads=1,
        latent=ms.TARGET_LATENT_RANK,
        rope=ms.TARGET_ROPE_WIDTH,
        topk=topk,
        s_kv=1,
        softmax_scale=1.0,
    )


def _test_source() -> str:
    """This test file's own committed bytes, for the item that reads its own declarations."""
    return pathlib.Path(__file__).read_text(encoding="utf-8")


def test_admissible_emitted_width_is_admitted_by_the_sparse_kernel_gate():
    """ITEM 1: the emitted width is a positive multiple of ``KEY_CHUNK`` and the sparse GATE admits it,
    at the tiny geometry and at the production geometry, with the raw width as a doctored control the
    same gate refuses.

    The control is the point of the item. "The gate admits 2176" alone would be satisfied by a gate that
    admits everything; the pair says the gate is discriminating and that the width this module used to
    emit is on the wrong side of it.
    """
    reset_index_expand_dispatch_counters()
    for label, n_groups, seqs in (
        ("tiny", N_GROUPS_TINY, SEQS_TINY),
        ("production", N_GROUPS_C4, C4_SEQS),
    ):
        pids = [list(range(n_groups)) for _ in seqs]
        got = dsa_index_expand(*_tensors(pids, seqs), POOL_SIZE)
        emitted = int(got.shape[1])
        raw = index_expand_raw_width(n_groups, POOL_SIZE)
        _emit(
            f"ADMISSIBLE_GATE_{label}",
            n_groups=n_groups, pool_size=POOL_SIZE, raw_cols=raw, emitted_cols=emitted,
            key_chunk=KEY_CHUNK, emitted_mod_chunk=emitted % KEY_CHUNK, raw_mod_chunk=raw % KEY_CHUNK,
            chunks=emitted // KEY_CHUNK, population=got.numel(),
        )
        # the emitted width is what the module says it is, and it is a whole number of chunks
        assert emitted == index_expand_width(n_groups, POOL_SIZE)
        assert emitted > 0 and emitted % KEY_CHUNK == 0
        # the gate ADMITS it -- called, not re-implemented
        _admit(emitted)
        # THE DOCTORED CONTROL: the raw width this module used to emit is refused by that same gate
        assert raw % KEY_CHUNK != 0
        with pytest.raises(MlaSparseAttentionError, match="positive multiple"):
            _admit(raw)


def test_admissible_padding_is_sentinel_and_the_two_routes_stay_bit_identical():
    """ITEM 2: the first ``width_raw`` columns are unchanged, every column beyond is exactly ``-1``, on
    BOTH routes, and the two routes are bit-identical to each other.

    "UNCHANGED" IS MEASURED AGAINST THE PYTHON ORACLE, and that substitution is disclosed rather than
    hidden. The parent commit's kernel cannot be called from inside this process, but the oracle is a
    third spelling of the same arithmetic whose value-producing body this increment did not touch -- only
    the column range it loops over moved. So agreement with the oracle over ``[0, width_raw)`` is the
    parent-equivalence claim, taken against code that has no reason to have drifted with the kernel.
    """
    reset_index_expand_dispatch_counters()
    for label, n_groups, seqs in (
        ("tiny", N_GROUPS_TINY, SEQS_TINY),
        ("production", N_GROUPS_C4, C4_SEQS),
    ):
        pids = [list(range(n_groups)) for _ in seqs]
        tensors = _tensors(pids, seqs)
        kernel = dsa_index_expand(*tensors, POOL_SIZE)
        torch_route = _dsa_index_expand_torch(*tensors, POOL_SIZE)
        oracle = torch.tensor(_python_oracle(pids, seqs, POOL_SIZE), dtype=torch.int32)
        raw = index_expand_raw_width(n_groups, POOL_SIZE)

        head_kernel = _diff(kernel[:, :raw], oracle[:, :raw])
        head_torch = _diff(torch_route[:, :raw], oracle[:, :raw])
        pad_kernel = kernel[:, raw:]
        pad_torch = torch_route[:, raw:]
        routes = _diff(kernel, torch_route)
        # THE BOUNDARY, read on the row whose tail count is non-zero: the LAST raw column carries a real
        # token and the FIRST padded column is the sentinel. Row 1 of each geometry is chosen for this
        # because its tail count fills the tail region completely -- 3 of 3 at the tiny geometry, 3 of 3
        # at production -- so raw - 1 is a token index rather than a tail sentinel. An off-by-one in the
        # memset start clobbers that token; an off-by-one the other way leaves a column unwritten.
        last_raw = int(kernel[1, raw - 1].item())
        first_pad = int(kernel[1, raw].item())
        last_raw_torch = int(torch_route[1, raw - 1].item())
        _emit(
            f"ADMISSIBLE_PADDING_{label}",
            n_groups=n_groups, raw_cols=raw, emitted_cols=int(kernel.shape[1]),
            padded_cols=int(pad_kernel.shape[1]),
            head_diff_kernel_vs_oracle=head_kernel, head_diff_torch_vs_oracle=head_torch,
            route_diff=routes,
            pad_non_sentinel_kernel=int((pad_kernel != -1).sum().item()),
            pad_non_sentinel_torch=int((pad_torch != -1).sum().item()),
            last_raw_column=last_raw, first_padded_column=first_pad,
            last_raw_column_torch=last_raw_torch,
            tail_count_row1=seqs[1] - (seqs[1] // POOL_SIZE) * POOL_SIZE,
            population=kernel.numel(),
        )
        # the meaningful columns did not move, on either route
        assert head_kernel == 0
        assert head_torch == 0
        # every padded column is exactly the sentinel, on either route
        assert int(pad_kernel.shape[1]) == int(kernel.shape[1]) - raw
        assert bool((pad_kernel == -1).all().item())
        assert bool((pad_torch == -1).all().item())
        # and the padding starts at exactly the raw width, not one column either side of it
        assert last_raw != -1
        assert last_raw_torch == last_raw
        assert first_pad == -1
        # and the two routes agree everywhere, which is the claim two mechanisms have to earn
        assert routes == 0
        assert kernel.shape == torch_route.shape


_SPARSE_TEST = pathlib.Path(__file__).resolve().parents[1] / "attention" / "test_mla_sparse.py"


def _cited_from_098(*names: str) -> dict:
    """The named definitions from ``-098``'s landed test file, taken from disk and never retyped.

    THE BLOCK REQUIRES ``-098``'S PAIR TO BE CITED AND NOT RESTATED, and this is that citation made
    mechanical. The tolerance pair, the float64 live-columns reference, its attending control, the case
    builder and the geometry all arrive as ``-098``'s own bytes. Retyping any of them would create a
    second spelling that can drift from the first -- and a tolerance retyped is a tolerance that can be
    nudged to reach green, which ``-098`` forbids in its own words in the comment above ``RTOL``.

    READING A SIBLING TEST FILE THROUGH THE PARSER RATHER THAN IMPORTING IT IS DELIBERATE. An import
    would depend on how pytest inserts two sibling directories onto ``sys.path``, and it would drag in
    that file's module-level ``nki`` imports and committed digests for no reading here. Parsing a source
    file to read what it declares is landed precedent in this repo, including in ``-098``'s own file,
    which imports ``ast`` and ``inspect`` to do it.

    The namespace holds ``torch`` and nothing else, so an extracted definition that quietly depended on
    something else in its home module fails loudly here instead of picking up a stand-in.
    """
    src = _SPARSE_TEST.read_text(encoding="utf-8")
    ns: dict = {"torch": torch}
    for node in ast.parse(src).body:
        got = None
        if isinstance(node, ast.FunctionDef):
            got = node.name
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            got = node.targets[0].id
        if got in names:
            exec(ast.get_source_segment(src, node), ns)  # noqa: S102 -- see the docstring
    missing = [n for n in names if n not in ns]
    assert not missing, f"not declared at top level in {_SPARSE_TEST.name}: {missing}"
    return ns


def test_admissible_padded_rows_through_the_sparse_kernel_match_a_float64_reference():
    """ITEM 3: the padded rows this module now emits, fed through ``mla_sparse_attention``, equal a
    float64 reference over their non-sentinel columns at ``-098``'s tolerance pair -- the integration
    this increment exists for.

    WHY THE GEOMETRY NEEDED NO INVENTING, and it is the cleanest evidence that the two increments meet.
    The tiny geometry's EMITTED width is 128, and 128 is exactly the selected-row count ``-098``'s own
    item (1) already runs at (``SENTINEL_UNTILED``). So the padded output of this module is a drop-in for
    the index tensor that item already exercises; the only difference is where the ``-1`` columns come
    from. Here they come from padding a real expansion instead of being sown into a random selection.
    The first assertion below states that equality rather than assuming it.

    WHAT WOULD MAKE THIS ITEM RED, ordered by what it would mean. If the kernel attended a padded column
    the reference dropped, the agreement fails -- and that is ``-098``'s mask failing on rows this module
    produced, which is a finding about the pair and not about either one alone. If the CONTROL stops
    firing, the item is blind and says so instead of passing.

    ONE READING I AM NOT MAKING. Every row here has at least one live column, so the wholly-sentinel row
    never occurs and its exact-zeros rule is not exercised. That rule is ``-098``'s item (2) and it is
    already read there; manufacturing a second reading of it here would add no reading.
    """
    cited = _cited_from_098("RTOL", "ATOL", "SENTINEL_UNTILED", "case_scale", "make_case",
                            "sparse_mla_torch_reference", "sentinel_reference", "attending_reference")
    case = dict(cited["SENTINEL_UNTILED"])
    rtol, atol = cited["RTOL"], cited["ATOL"]

    # THE JOIN, ASSERTED: this module's emitted width IS the selected-row count -098 already runs at.
    emitted = index_expand_width(N_GROUPS_TINY, POOL_SIZE)
    assert emitted == case["topk"]

    # -098's inputs, from -098's builder. Its index tensor is DISCARDED on purpose -- the whole point of
    # this item is that the index rows come from this module's padded expansion instead.
    q_lift, c_kv, _discarded_idx, q_pe, k_pe = cited["make_case"](**case, seed=102)
    assert q_pe is None and k_pe is None, "the sentinel geometry is at rope width 0"

    rows = case["seq"]
    pids = [list(range(N_GROUPS_TINY)) for _ in range(rows)]
    seqs = [SEQS_TINY[r % len(SEQS_TINY)] for r in range(rows)]
    reset_index_expand_dispatch_counters()
    ms.reset_mla_sparse_dispatch_counters()
    ms.reset_mla_sparse_tiled_dispatch_counters()
    idx = dsa_index_expand(*_tensors(pids, seqs), POOL_SIZE)
    assert idx.shape == (rows, emitted)

    live_per_row = [int((idx[r] >= 0).sum().item()) for r in range(rows)]
    scale = cited["case_scale"](case)
    assert ms.can_run_mla_sparse_attention(
        q_lift, case["seq"], case["heads"], case["latent"], case["rope"],
        case["topk"], case["s_kv"], scale,
    ), "the sparse gate refused the padded width, so nothing below would mean anything"

    got = ms.mla_sparse_attention(q_lift, c_kv, idx, scale)
    ref = cited["sentinel_reference"](q_lift, c_kv, idx, scale)
    attending = cited["attending_reference"](q_lift, c_kv, idx, scale)
    err = float((got - ref).abs().max())
    wrong = float((got - attending).abs().max())
    ix_nki, ix_fallback = index_expand_dispatch_counters()
    sp_nki, sp_fallback = ms.mla_sparse_dispatch_counters()
    tiled_nki, tiled_fallback = ms.mla_sparse_tiled_dispatch_counters()
    _emit(
        "ADMISSIBLE_INTEGRATION",
        emitted_cols=emitted, raw_cols=index_expand_raw_width(N_GROUPS_TINY, POOL_SIZE),
        seq_lens=seqs, live_columns_per_row=live_per_row,
        sentinel_columns=int((idx < 0).sum().item()), index_population=idx.numel(),
        rows_wholly_sentinel=int((idx < 0).all(dim=1).sum().item()),
        rtol=rtol, atol=atol, maxabs_vs_live_column_reference=f"{err:.3e}",
        control_sentinel_attended_as_row_0=f"{wrong:.3e}", control_fires=int(wrong > atol),
        index_expand_nki=ix_nki, index_expand_fallback=ix_fallback,
        sparse_seam_nki=sp_nki, sparse_seam_fallback=sp_fallback,
        sparse_tiled_nki=tiled_nki, sparse_tiled_fallback=tiled_fallback,
        out_shape=list(got.shape),
    )
    # the padding is real and is most of the width, which is what makes this a padded-row reading
    assert int((idx < 0).sum().item()) == idx.numel() - sum(live_per_row)
    assert min(live_per_row) > 0, "no row is wholly sentinel here -- that rule is -098's item (2)"
    assert sum(live_per_row) < idx.numel()
    # THE AGREEMENT, at -098's quoted pair. Neither number is authored here.
    torch.testing.assert_close(got, ref, rtol=rtol, atol=atol)
    # THE CONTROL: a reference that ATTENDS the padded columns as cache row 0 must disagree, or this
    # item cannot see whether they were masked at all.
    assert wrong > atol, (
        f"the reference that attends the padded columns agrees with the kernel to {wrong:.3e}, so this "
        f"item is blind. Report it; never widen the tolerance to absorb it"
    )
    # THE ROUTE PREDICATE, both seams. This module dispatched once with no fallback, and the sparse seam
    # took its untiled body, which is what SENTINEL_UNTILED's latent 128 and topk 128 select.
    assert (ix_nki, ix_fallback) == (1, 0)
    assert (sp_nki, sp_fallback) == (1, 0)
    assert (tiled_nki, tiled_fallback) == (0, 0)
    assert got.shape == (case["seq"], case["heads"], case["latent"])


def _assert_statements(src: str, name: str) -> list[str]:
    """The ASSERT statements of one named top-level function, normalised through the parser.

    THREE HAZARDS ARE CLOSED BY GOING THROUGH THE AST rather than counting over file text, and every one
    of them has already bitten this lap. A file-wide count matches the string literals in THIS item's own
    body, so a self-scan cannot tell a declaration it relies on from one it is quoting. A
    whole-function count matches the COMMENT above case 1, which quotes the old expectation verbatim on
    purpose. And a line-number read goes stale the moment anything above it moves -- it already did, in
    the plan text this increment was dispatched from. Asking the parser for the assert statements of a
    named function reads exactly what item (4) is about: what those tests DECLARE, with the prose about
    them out of scope by construction rather than by escaping.

    Statement order is not relied upon -- these are counts over a set.

    ONE FRAGILITY, STATED RATHER THAN HIDDEN: the caller compares fixed strings against ``ast.unparse``
    OUTPUT, so a change in how a Python version formats an assert would move them. That failure mode
    announces itself instead of lying, for two reasons -- the item prints every statement it read, and its
    control runs through this same formatter, so a formatting change fails the control too and names the
    reader rather than the code.
    """
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return [ast.unparse(n) for n in ast.walk(node) if isinstance(n, ast.Assert)]
    raise AssertionError(f"no top-level function named {name!r} in the source read")


def test_admissible_the_landed_width_declarations_moved_with_the_rule():
    """ITEM 4: the two landed width declarations were RE-DECLARED to the emitted width, the literal 2051
    moved to the concept it actually names, and the module spells the ceiling exactly once.

    This is a SOURCE-level item on purpose, in the shape of the landed MX screen above. The behavioural
    items would all still pass if a declaration had been left behind as a stale literal that happened to
    agree with the new width; only reading the declarations catches a number that stopped being derived.

    2051 IS NOT ASSERTED TO BE GONE, because it is not gone and claiming so would be false. It stopped
    being the emitted shape and became the RAW width, which is the honest statement of what changed: the
    2051 meaningful columns are all still there, and 125 sentinel columns follow them.
    """
    tsrc = _test_source()
    msrc = _module_source()
    declared = (_assert_statements(tsrc, "test_declared_case_1_bit_identical")
                + _assert_statements(tsrc, "test_declared_case_4_bit_identical_at_production_width"))
    text = "\n".join(declared)
    reads = {
        "case1_asks_the_module": text.count("index_expand_width(N_GROUPS_SMALL, POOL_SIZE)"),
        "case4_asks_the_module": text.count("index_expand_width(N_GROUPS_C4, POOL_SIZE)"),
        "shape_literal_at_2176": text.count("int(got.shape[1]) == 2176"),
        "shape_literal_left_at_2051": text.count("int(got.shape[1]) == 2051"),
        "2051_now_names_the_raw_width": text.count(
            "index_expand_raw_width(N_GROUPS_C4, POOL_SIZE) == 2051"),
        "raw_formula_still_declared": text.count("POOL_SIZE + POOL_SIZE - 1"),
    }
    ceilings = msrc.count("// KEY_CHUNK) * KEY_CHUNK")
    retyped = msrc.count("KEY_CHUNK = ")

    # THE CONTROL, run through the SAME reader: the parent's own declarations, as they read before this
    # increment. Without it, every zero above could be a reader that cannot reach an assert at all.
    parent = (
        "def test_declared_case_4_bit_identical_at_production_width():\n"
        "    assert got.shape == (2, N_GROUPS_C4 * POOL_SIZE + POOL_SIZE - 1)\n"
        "    assert int(got.shape[1]) == 2051\n"
    )
    ptext = "\n".join(
        _assert_statements(parent, "test_declared_case_4_bit_identical_at_production_width"))
    control = {
        "parent_raw_formula": ptext.count("POOL_SIZE + POOL_SIZE - 1"),
        "parent_shape_literal_2051": ptext.count("int(got.shape[1]) == 2051"),
        "parent_asks_the_module": ptext.count("index_expand_width("),
    }

    _emit(
        "ADMISSIBLE_DECLARATIONS",
        declarations_read=len(declared), ceiling_definitions=ceilings,
        key_chunk_redefinitions=retyped, test_population=len(tsrc.splitlines()),
        module_population=len(msrc.splitlines()), **reads,
    )
    _emit("ADMISSIBLE_DECLARATIONS_CONTROL", **control)
    for statement in declared:
        _emit("ADMISSIBLE_DECLARATION", statement=statement)

    # both landed tests now ask the module for the width instead of spelling the rule again
    assert reads["case1_asks_the_module"] == 1
    assert reads["case4_asks_the_module"] == 1
    # the shape literal moved, and no raw-width formula is declared as a shape any more
    assert reads["shape_literal_at_2176"] == 1
    assert reads["shape_literal_left_at_2051"] == 0
    assert reads["raw_formula_still_declared"] == 0
    # 2051 was not deleted -- it was re-attached to the raw width, where it is correct
    assert reads["2051_now_names_the_raw_width"] == 1
    # ONE definition of the ceiling in the module, and KEY_CHUNK defined nowhere in it -- it is imported
    assert ceilings == 1
    assert retyped == 0
    # and the reader can see all three of the things it just reported absent
    assert control == {"parent_raw_formula": 1, "parent_shape_literal_2051": 1,
                       "parent_asks_the_module": 0}


def test_admissible_emitted_width_equals_upstreams_buffer_width():
    """ITEM 5: the emitted width equals upstream's allocated buffer width, computed by UPSTREAM'S OWN
    expression, at the production geometry and at the tiny one.

    Upstream, at ``878631b6``, ``models/glm5next/nvidia/model.py:594-599``::

        buffer_width = topk_tokens + (kpool - 1 if kpool > 1 else 0)
        buffer_width = ceil(buffer_width / 128) * 128

    written out here in that form rather than by calling this module's function, because the whole
    reading is that two independently-spelled expressions land on the same number. Re-gated on the
    READING (DECISIONS section 70), not on a budget rule: the forced tail pool sits OUTSIDE the
    ``index_topk`` budget on both sides, so no budget rule was adopted and the agreement is by
    construction rather than by convention.

    The raw widths are DERIVED, never typed -- 11 at the tiny geometry and 2051 at production are
    printed as readings, not spelled as constants, which is the guard against the unit confusion that
    made a ruling draft say 5.

    THE ITEM CARRIES TWO READINGS OF UNEQUAL STRENGTH AND SAYS WHICH IS WHICH. The RAW widths agreeing is
    the strong one: it is an exact comparison of two independently-spelled expressions, and an off-by-one
    in either breaks it. The EMITTED widths agreeing is the one the block asks for, and it is weaker,
    because rounding to ``KEY_CHUNK`` absorbs a one-column error. Both are asserted, the weakness is
    proved rather than glossed, and its control is a wrong TILE constant -- the realistic mistake at this
    seam -- instead of a one-column perturbation that cannot fail.
    """

    def upstream_raw_width(topk_tokens: int, kpool: int) -> int:
        """``model.py:594``, before the rounding."""
        return topk_tokens + (kpool - 1 if kpool > 1 else 0)

    def upstream_buffer_width(topk_tokens: int, kpool: int, block: int = 128) -> int:
        """``model.py:594-599``, the allocated width."""
        return ((upstream_raw_width(topk_tokens, kpool) + block - 1) // block) * block

    for label, n_groups in (("tiny", N_GROUPS_TINY), ("production", N_GROUPS_C4)):
        topk_tokens = n_groups * POOL_SIZE
        fork_raw = index_expand_raw_width(n_groups, POOL_SIZE)
        fork = index_expand_width(n_groups, POOL_SIZE)
        up_raw = upstream_raw_width(topk_tokens, POOL_SIZE)
        upstream = upstream_buffer_width(topk_tokens, POOL_SIZE)
        dropped_minus_one = topk_tokens + POOL_SIZE
        wrong_tile = upstream_buffer_width(topk_tokens, POOL_SIZE, block=KEY_CHUNK // 2)
        absorbed = ((up_raw + 1 + 127) // 128) * 128
        _emit(
            f"ADMISSIBLE_UPSTREAM_PARITY_{label}",
            n_groups=n_groups, topk_tokens=topk_tokens, pool_size=POOL_SIZE,
            fork_raw=fork_raw, upstream_raw=up_raw, fork_emitted=fork,
            upstream_buffer_width=upstream, raw_agree=int(fork_raw == up_raw),
            emitted_agree=int(fork == upstream), control_dropped_minus_one=dropped_minus_one,
            control_wrong_tile=wrong_tile, one_column_error_rounds_to=absorbed,
        )
        # THE STRONG READING: the two RAW widths are one expression rearranged --
        # pool * (g + 1) - 1 == g * pool + pool - 1 -- so the fork was already emitting upstream's
        # column count and this increment adds only upstream's rounding step.
        assert fork_raw == up_raw
        assert fork_raw == topk_tokens + POOL_SIZE - 1
        # and an off-by-one there really does disagree, so that equality is a comparison
        assert dropped_minus_one != up_raw
        # THE READING THE BLOCK ASKS FOR: the emitted width is what upstream allocates
        assert fork == upstream
        # with a control that fires -- the wrong tile constant, which is the realistic mistake here
        assert wrong_tile != upstream
        # AND THE WEAKNESS OF THAT SECOND READING, DISCLOSED AND PROVED RATHER THAN ASSERTED AWAY:
        # the rounding absorbs a one-column error, so `fork == upstream` alone would not catch one.
        # That is what the raw-width arms above are for. An earlier draft of this item used a
        # one-column control here and it could not fail; `probe-102-authoring-r1.out` is the round
        # that read FAIL on exactly that, before any host time was spent on it.
        assert absorbed == upstream
    # WHY THE ABSORPTION IS STRUCTURAL AND NOT A COINCIDENCE OF THESE TWO GEOMETRIES: at an even pool
    # size the raw width is always ODD, and every multiple of KEY_CHUNK is even, so the raw width never
    # lands on a chunk boundary and no one-column error can ever cross one.
    parities = {index_expand_raw_width(n, POOL_SIZE) % 2 for n in range(1, N_GROUPS_C4 + 1)}
    on_boundary = [n for n in range(1, N_GROUPS_C4 + 1)
                   if index_expand_raw_width(n, POOL_SIZE) % KEY_CHUNK == 0]
    _emit("ADMISSIBLE_UPSTREAM_PARITY_STRUCTURE", pool_size=POOL_SIZE,
          group_counts_swept=N_GROUPS_C4, raw_width_parities=sorted(parities),
          group_counts_on_a_chunk_boundary=len(on_boundary))
    assert sorted(parities) == [1]
    assert on_boundary == []
