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
"""

from __future__ import annotations

import pathlib

import pytest
import torch

from vllm_neuron.functional.dsa import index_expand as mod
from vllm_neuron.functional.dsa.index_expand import (
    INDEX_KPOOL,
    IndexExpandError,
    _dsa_index_expand_torch,
    can_run_dsa_index_expand,
    dsa_index_expand,
    index_expand_dispatch_counters,
    index_expand_kernel_identity,
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
width that actually ships: 2048 history columns and 2051 output columns."""


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
        for col in range(topk + pool_size - 1):
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
"""CASE 4 -- production width. topk 2048, out_cols 2051, tail_count 2 and 3."""


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
    assert got.shape == (len(C1_SEQS), N_GROUPS_SMALL * POOL_SIZE + POOL_SIZE - 1)
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
    """DECLARED CASE 4: 512 selected pools, 2051 output columns, max abs diff exactly 0.

    This is the width that ships. The history loop is ``pool_size`` iterations here exactly as it is in
    case 1 -- that independence from the pool count is why this case costs what case 1 costs.
    """
    reset_index_expand_dispatch_counters()
    pids, seqs = _tensors(C4_PIDS, C4_SEQS)
    got = dsa_index_expand(pids, seqs, POOL_SIZE)
    want = _dsa_index_expand_torch(pids, seqs, POOL_SIZE)
    worst = _diff(got, want)
    _emit("CASE4_BITIDENTICAL", rows=len(C4_SEQS), n_groups=N_GROUPS_C4, seq_lens=C4_SEQS,
          out_cols=int(got.shape[1]), max_abs_diff=worst, population=got.numel())
    assert got.shape == (2, N_GROUPS_C4 * POOL_SIZE + POOL_SIZE - 1)
    assert int(got.shape[1]) == 2051
    assert worst == 0


def test_declared_case_4_every_entry_is_sentinel_or_in_range():
    """DECLARED CASE 4's in-range reading: 0 out of bounds over 4102 entries."""
    reset_index_expand_dispatch_counters()
    pids, seqs = _tensors(C4_PIDS, C4_SEQS)
    got = dsa_index_expand(pids, seqs, POOL_SIZE)
    in_range, oob, sentinels, population = _range_counts(got, C4_SEQS)
    _emit("CASE4_INRANGE", in_range=in_range, out_of_bounds=oob, sentinels=sentinels,
          population=population, nonsentinel_population=in_range + oob)
    assert oob == 0
    assert population == 4102


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
