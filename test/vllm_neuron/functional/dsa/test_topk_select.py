# SPDX-License-Identifier: Apache-2.0
"""Tests for the DSA top-k select against ``torch.topk``.

Two selection widths are driven over a 4-row, 4096-column score tensor:
``index_topk`` (2048) and ``select_k`` (512). Scores are a scaled permutation,
so every score is distinct and the top-k index set is uniquely defined; index
sets are compared as sets and values are compared numerically.

A further test drives an odd row count, which leaves the kernel's last
program a partial tile -- the shape that once produced NaN out of an
uninitialised partition.
"""

from __future__ import annotations

import torch

from vllm_neuron.functional.dsa.topk_select import (
    can_run_dsa_topk_select,
    dsa_topk_select,
    reset_topk_select_dispatch_counters,
    topk_select_dispatch_counters,
    topk_select_kernel_identity,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

ROWS = 4
WIDTH = 4096
INDEX_TOPK = 2048
SELECT_K = 512

# rtol is 0 because the selected values are gathered rather than computed, so
# the measured difference is exact zero; atol gives float32 rounding room.
VALUE_RTOL = 0.0
VALUE_ATOL = 1e-5


def _scores(k: int) -> torch.Tensor:
    """``[ROWS, WIDTH]`` float32 scores, all distinct, seeded per selection width."""
    gen = torch.Generator().manual_seed(43_000 + k)
    flat = torch.randperm(ROWS * WIDTH, generator=gen)
    return flat.reshape(ROWS, WIDTH).to(torch.float32) / (ROWS * WIDTH)


def _rows_whose_index_sets_agree(got: torch.Tensor, expected: torch.Tensor) -> int:
    return sum(
        1 for r in range(got.shape[0]) if set(got[r].tolist()) == set(expected[r].tolist())
    )


def _assert_index_sets_match(k: int) -> None:
    """Drive one selection width and require the returned index set to match ``torch.topk``."""
    scores = _scores(k)
    _, indices = dsa_topk_select(scores, k)
    _, expected_indices = torch.topk(scores, k, dim=-1)
    agree = _rows_whose_index_sets_agree(indices, expected_indices)

    assert agree == ROWS, (
        f"selected index sets must match torch.topk on every row at k={k}; "
        f"{agree} of {ROWS} rows agree"
    )
    assert indices.shape == (ROWS, k)
    assert indices.dtype == torch.int64


def test_index_sets_match_torch_at_index_topk_2048() -> None:
    """Exact index-set equality at the wider selection width."""
    _assert_index_sets_match(INDEX_TOPK)


def test_index_sets_match_torch_at_select_k_512() -> None:
    """Exact index-set equality at the narrower selection width; not monotonic in k."""
    _assert_index_sets_match(SELECT_K)


def test_selected_values_match_torch_at_atol_1e_5() -> None:
    """Selected values match the torch reference at both selection widths."""
    for k in (INDEX_TOPK, SELECT_K):
        scores = _scores(k)
        values, _ = dsa_topk_select(scores, k)
        expected_values, _ = torch.topk(scores, k, dim=-1)
        torch.testing.assert_close(values, expected_values, rtol=VALUE_RTOL, atol=VALUE_ATOL)


def test_every_case_takes_the_kernel_and_not_the_torch_path() -> None:
    """Each selection width dispatches the kernel exactly once and never enters the fallback."""
    for k in (INDEX_TOPK, SELECT_K):
        scores = _scores(k)
        reset_topk_select_dispatch_counters()
        gate = can_run_dsa_topk_select(scores, k)
        dsa_topk_select(scores, k)
        nki_dispatch, torch_fallback = topk_select_dispatch_counters()
        assert nki_dispatch == 1, f"expected exactly 1 NKI dispatch at k={k}; got {nki_dispatch}"
        assert torch_fallback == 0, (
            f"the torch fallback must not be entered at k={k}; it ran {torch_fallback} time(s)"
        )
        assert can_run_kernel(scores) is True
        assert gate is True


def test_k_equals_width_is_refused_by_the_gate_and_served_by_torch() -> None:
    """k == width is outside the kernel's envelope; the torch path must still select correctly."""
    scores = _scores(SELECT_K)
    violating_k = WIDTH
    reset_topk_select_dispatch_counters()
    gate = can_run_dsa_topk_select(scores, violating_k)
    values, indices = dsa_topk_select(scores, violating_k)
    nki_dispatch, torch_fallback = topk_select_dispatch_counters()

    assert gate is False, "k == width must be refused by the gate"
    assert torch_fallback == 1, f"the torch path must run exactly once; got {torch_fallback}"
    assert nki_dispatch == 0, f"the kernel must not be reached; got {nki_dispatch} dispatch(es)"
    expected_values, _ = torch.topk(scores, violating_k, dim=-1)
    torch.testing.assert_close(values, expected_values, rtol=VALUE_RTOL, atol=VALUE_ATOL)
    assert indices.shape == (ROWS, violating_k)


def test_kernel_identity_reports_the_dispatched_kernel() -> None:
    """The identity is None before any dispatch and names the vendored kernel after one."""
    reset_topk_select_dispatch_counters()
    assert topk_select_kernel_identity() is None

    dsa_topk_select(_scores(SELECT_K), SELECT_K)
    after = topk_select_kernel_identity()
    assert after is not None
    module, qualname = after
    assert module == (
        "vllm_neuron.functional.vendored_kernels.rotational_topk.rotational_topk"
    ), f"the seam must dispatch the vendored rotational kernel; got module {module}"
    assert qualname == "rotational_topk", (
        f"the seam must dispatch rotational_topk; got qualname {qualname}"
    )


# An odd row count: 4 splits evenly across the seam's two programs, so no
# program ever gets a partial tile. 5 does not.
ROWS_ODD = 5


def _scores_at_rows(rows: int, k: int) -> torch.Tensor:
    """``[rows, WIDTH]`` float32 scores, scaled so they stay exactly representable and distinct."""
    gen = torch.Generator().manual_seed(44_000 + rows * 10_000 + k)
    flat = torch.randperm(rows * WIDTH, generator=gen)
    return flat.reshape(rows, WIDTH).to(torch.float32) / float(2**15)


def test_index_sets_match_torch_at_an_odd_row_count() -> None:
    """An odd row count leaves the last program a partial tile; values must stay finite and match torch.

    At rows == 4 both programs get 2 rows and every partition of the kernel's
    staging buffer is written. At rows == 5 the last program gets a partial
    tile, and before a guard the leftover partitions reached the kernel's
    cross-partition matmul uninitialised, turning one non-finite word into NaN
    across every output partition.
    """
    from vllm_neuron.functional.dsa.topk_select import _NUM_PROGRAMS

    rows_per_program = -(-ROWS_ODD // _NUM_PROGRAMS)
    last_program_rows = ROWS_ODD - rows_per_program * (_NUM_PROGRAMS - 1)
    assert last_program_rows < rows_per_program, (
        f"this test needs a partial last tile: {ROWS_ODD} rows over {_NUM_PROGRAMS} "
        f"programs gives {rows_per_program} per program and {last_program_rows} on the last"
    )

    scores = _scores_at_rows(ROWS_ODD, SELECT_K)
    reset_topk_select_dispatch_counters()
    gate = can_run_dsa_topk_select(scores, SELECT_K)
    values, indices = dsa_topk_select(scores, SELECT_K)
    nki_dispatch, torch_fallback = topk_select_dispatch_counters()

    expected_values, expected_indices = torch.topk(scores, SELECT_K, dim=-1)
    agree = _rows_whose_index_sets_agree(indices, expected_indices)

    # The route is asserted first: if the gate refused this shape, the seam
    # would answer from torch.topk, which has no partitions and nothing to catch.
    assert nki_dispatch == 1 and torch_fallback == 0 and gate is True, (
        f"this test must read the kernel, not the torch fallback: gate={gate} "
        f"nki_dispatch={nki_dispatch} torch_fallback={torch_fallback}"
    )
    # Finiteness first: an all-NaN return would otherwise fail the set
    # comparison below with a message about index sets and say nothing about NaN.
    assert torch.isfinite(values).all(), "every selected value must be finite at an odd row count"
    assert agree == ROWS_ODD, (
        f"selected index sets must match torch.topk on every row at rows={ROWS_ODD}; "
        f"{agree} of {ROWS_ODD} rows agree"
    )
    assert torch.equal(values, expected_values), (
        "selected values must be bit-equal to the torch reference at an odd row count"
    )
    assert indices.shape == (ROWS_ODD, SELECT_K)
    assert indices.dtype == torch.int64
