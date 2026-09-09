# SPDX-License-Identifier: Apache-2.0
"""Tier N acceptance for `inc-glm53f-043` -- the DSA indexer top-k WRAP.

Declared command (D1 Tier N, the block's shorthand expanded to the tier's byte form)::

    PYTHONDONTWRITEBYTECODE=1 VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \\
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \\
    python -m pytest test/vllm_neuron/functional/dsa/test_topk_select.py \\
        --timeout 60 -v -s -p no:cacheprovider

ONE ITEM PER COUNTED CONJUNCT AND NO ``parametrize`` (§6 rule 4b / rule 6), so the item
count is derivable before the run. NO FIXTURE SETS ``NEURON_PLATFORM_TARGET_OVERRIDE`` OR
``NKI_SIMULATOR`` (§6 rule 3, D2): both resolve at import time and belong in the process
invocation, so a fixture that set either would have measured the wrong thing.

THE GEOMETRY, and why it is one candidate width and two selection widths. The DSA indexer
scores a context of candidate positions for each query token, keeps ``index_topk`` of them,
and narrows to ``select_k``. So the fixture is ONE candidate axis of 4,096 positions read at
both declared selection widths -- ``k == 2048`` and ``k == 512`` -- over 4 query-token rows.
N in the block's "N/N rows" is therefore 4.

WHY ``float32`` AND WHY THE SCORES ARE A SCALED PERMUTATION. A top-k index SET is well
defined only when no tie straddles the selection boundary. ``torch.randperm`` over
``4 * 4096 == 16384`` values divided by ``16384`` is exactly representable in ``float32``
and yields 16,384 DISTINCT scores, so the correct answer is unique and an exact set
comparison is meaningful. The same scores in ``bfloat16`` collapse to 1,025 distinct values,
which makes the comparison ill-defined rather than merely harder -- measured, and recorded
in the module docstring and in the evidence record. The distinctness is ASSERTED below, in
the two set-equality items, rather than assumed: a fixture that lost its distinctness would
otherwise weaken both items silently.
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

# Value tolerance, ORDER NAMED INLINE (D3): (rtol, atol) == (0.0, 1e-5). rtol is 0 because
# the selected values are GATHERED and not computed, so the measured difference is exact
# zero and a relative term would only loosen a bound that does not need loosening.
VALUE_RTOL = 0.0
VALUE_ATOL = 1e-5


def _scores(k: int) -> torch.Tensor:
    """``[ROWS, WIDTH]`` float32 scores, all distinct, seeded per selection width."""
    gen = torch.Generator().manual_seed(43_000 + k)
    flat = torch.randperm(ROWS * WIDTH, generator=gen)
    return flat.reshape(ROWS, WIDTH).to(torch.float32) / (ROWS * WIDTH)


def _rows_whose_index_sets_agree(got: torch.Tensor, want: torch.Tensor) -> int:
    return sum(1 for r in range(got.shape[0]) if set(got[r].tolist()) == set(want[r].tolist()))


def _index_set_case(label: str, k: int) -> None:
    """Drive one declared case and assert the index sets, printing every reading."""
    scores = _scores(k)
    distinct = int(torch.unique(scores).numel())
    print(f"[fixture] case={label} k={k} rows={ROWS} width={WIDTH} "
          f"distinct_scores={distinct}/{ROWS * WIDTH} dtype={scores.dtype}")
    # The fixture's own precondition: without distinctness the set comparison below is not
    # a well-defined question, so it is asserted rather than trusted.
    assert distinct == ROWS * WIDTH, (
        f"the fixture must hold {ROWS * WIDTH} distinct scores for the top-k index set to "
        f"be unique; got {distinct}"
    )

    reset_topk_select_dispatch_counters()
    values, indices = dsa_topk_select(scores, k)
    nki_dispatch, torch_fallback = topk_select_dispatch_counters()

    want_values, want_indices = torch.topk(scores, k, dim=-1)
    agree = _rows_whose_index_sets_agree(indices, want_indices)
    value_max_abs_diff = (values - want_values).abs().max().item()

    print(f"[acceptance] case={label} k={k} index_set_rows_agree={agree}/{ROWS} "
          f"value_max_abs_diff={value_max_abs_diff:.3e} out_shape={tuple(values.shape)} "
          f"idx_dtype={indices.dtype}")
    print(f"[route-predicate] case={label} k={k} nki_dispatch={nki_dispatch} "
          f"torch_fallback={torch_fallback} can_run_kernel={can_run_kernel(scores)} "
          f"certifies=vllm_neuron.functional.dsa.topk_select.dsa_topk_select")

    assert agree == ROWS, (
        f"selected index sets must match torch.topk on every row at k={k}; "
        f"{agree} of {ROWS} rows agree"
    )
    assert indices.shape == (ROWS, k)
    assert indices.dtype == torch.int64


def test_index_sets_match_torch_at_index_topk_2048() -> None:
    """Conjunct 1 -- exact index-set equality at ``index_topk == 2048``, 4/4 rows.

    Certifies: ``vllm_neuron/functional/dsa/topk_select.py::dsa_topk_select`` -- the wrap
    seam this increment authors, driven at the wider of the two declared selection widths.
    """
    _index_set_case("index_topk", INDEX_TOPK)


def test_index_sets_match_torch_at_select_k_512() -> None:
    """Conjunct 2 -- exact index-set equality at ``select_k == 512``, 4/4 rows.

    Certifies: the same seam at the narrower declared selection width. Both widths are
    driven because the kernel's staging is NOT monotonic in ``k``: passing at one width
    establishes nothing about the other.
    """
    _index_set_case("select_k", SELECT_K)


def test_selected_values_match_torch_at_atol_1e_5() -> None:
    """Conjunct 3 -- selected VALUES match the torch reference at ``(rtol, atol) == (0.0, 1e-5)``.

    Certifies: the same seam's value output at both declared widths. The index comparison
    above says the right candidates were chosen; this says the values handed back are the
    scores those candidates carried, which is a different failure and needs its own item.
    """
    for label, k in (("index_topk", INDEX_TOPK), ("select_k", SELECT_K)):
        scores = _scores(k)
        reset_topk_select_dispatch_counters()
        values, _ = dsa_topk_select(scores, k)
        want_values, _ = torch.topk(scores, k, dim=-1)
        diff = (values - want_values).abs().max().item()
        print(f"[values] case={label} k={k} max_abs_diff={diff:.3e} rtol={VALUE_RTOL} "
              f"atol={VALUE_ATOL} certifies="
              f"vllm_neuron.functional.dsa.topk_select.dsa_topk_select")
        torch.testing.assert_close(values, want_values, rtol=VALUE_RTOL, atol=VALUE_ATOL)


def test_route_predicate_one_dispatch_per_case_and_no_torch_fallback() -> None:
    """Route predicate (D13 form R-1) -- 1 NKI dispatch per declared case, fallback 0.

    Certifies: the ``wrap_nki`` dispatch site inside
    ``vllm_neuron/functional/dsa/topk_select.py::dsa_topk_select``. A pure-torch
    implementation of this module reads ``nki_dispatch == 0`` on every case and therefore
    cannot pass this item, which is the whole point of the form.

    THE COUNTED ZERO NAMES ITS POPULATION: ``calls_made`` is printed beside
    ``torch_fallback``, so the zero is read over a stated number of opportunities to be
    non-zero rather than over an unstated one. Its non-vacuity control is the separate item
    ``test_torch_fallback_counter_is_not_vacuous``, which drives the SAME counter non-zero.
    """
    gate_readings: list[bool] = []
    for label, k in (("index_topk", INDEX_TOPK), ("select_k", SELECT_K)):
        scores = _scores(k)
        # Per-case reset, per §4b's convention: the counter is read at the end of the case
        # it was zeroed at the start of.
        reset_topk_select_dispatch_counters()
        gate = can_run_dsa_topk_select(scores, k)
        gate_readings.append(gate)
        dsa_topk_select(scores, k)
        calls_made = 1
        nki_dispatch, torch_fallback = topk_select_dispatch_counters()
        print(f"[route-predicate] case={label} k={k} calls_made={calls_made} "
              f"nki_dispatch={nki_dispatch} torch_fallback={torch_fallback} "
              f"can_run_kernel={can_run_kernel(scores)} "
              f"can_run_dsa_topk_select={gate} "
              f"certifies=vllm_neuron.functional.dsa.topk_select.dsa_topk_select")
        assert nki_dispatch == calls_made, (
            f"expected exactly {calls_made} NKI dispatch at k={k}; got {nki_dispatch}"
        )
        assert torch_fallback == 0, (
            f"the torch fallback must not be entered at k={k}; it ran {torch_fallback} "
            f"time(s) out of {calls_made} call(s)"
        )
        assert can_run_kernel(scores) is True
        assert gate is True

    print(f"[route-predicate] gate_true_on_all_declared_cases="
          f"{sum(gate_readings)}/{len(gate_readings)}")
    assert all(gate_readings)


def test_torch_fallback_counter_is_not_vacuous() -> None:
    """D1.5 control -- the counted zero above is a zero the instrument CAN move.

    Certifies: the fallback limb of
    ``vllm_neuron/functional/dsa/topk_select.py::dsa_topk_select``, reached through
    ``can_run_dsa_topk_select`` returning False. The violating input is ``k == width``,
    which the kernel refuses for sorted output; no environment variable is touched to
    produce it, so this control cannot itself perturb the tier's pinned invocation (D2).

    A zero that reads zero either way is decoration. This item is what makes the previous
    item's ``torch_fallback == 0`` a measurement.
    """
    scores = _scores(SELECT_K)
    violating_k = WIDTH  # k == width: out of envelope by the kernel's own refusal
    reset_topk_select_dispatch_counters()
    gate = can_run_dsa_topk_select(scores, violating_k)
    values, indices = dsa_topk_select(scores, violating_k)
    nki_dispatch, torch_fallback = topk_select_dispatch_counters()
    print(f"[control] case=k_equals_width k={violating_k} width={WIDTH} calls_made=1 "
          f"nki_dispatch={nki_dispatch} torch_fallback={torch_fallback} "
          f"can_run_dsa_topk_select={gate} "
          f"certifies=vllm_neuron.functional.dsa.topk_select._dsa_topk_select_torch")
    assert gate is False, "k == width must be refused by the gate"
    assert torch_fallback == 1, (
        f"the control must drive the torch fallback exactly once; got {torch_fallback}"
    )
    assert nki_dispatch == 0, (
        f"the control must not reach the NKI seam; got {nki_dispatch} dispatch(es)"
    )
    # The fallback is still the oracle, so it must return the right answer, not merely run.
    want_values, _ = torch.topk(scores, violating_k, dim=-1)
    torch.testing.assert_close(values, want_values, rtol=VALUE_RTOL, atol=VALUE_ATOL)
    assert indices.shape == (ROWS, violating_k)


def test_kernel_identity_is_derived_through_the_seam() -> None:
    """The kernel the seam DISPATCHED is the vendored NKI top-k, read through the seam.

    Certifies: the identity recorded at the dispatch site in
    ``vllm_neuron/functional/dsa/topk_select.py::dsa_topk_select``. D13.1 admits a
    ``kernel_identity`` reading as route evidence only when it is derived through the seam
    the test actually drove; a reading taken from a module-level import certifies what was
    imported and nothing about what ran. So this item reads ``None`` BEFORE any dispatch --
    which is what distinguishes "no kernel ran" from "some kernel ran" -- and the vendored
    kernel after one.

    This item is also the increment's substrate evidence: the WRAP declaration says the
    kernel is the substrate's, and this is the reading that shows it.
    """
    reset_topk_select_dispatch_counters()
    before = topk_select_kernel_identity()
    print(f"[identity] before_any_dispatch={before}")
    assert before is None

    dsa_topk_select(_scores(SELECT_K), SELECT_K)
    after = topk_select_kernel_identity()
    print(f"[identity] after_one_dispatch={after} "
          f"certifies=vllm_neuron.functional.dsa.topk_select.dsa_topk_select")
    assert after is not None
    module, qualname = after
    assert module == (
        "vllm_neuron.functional.vendored_kernels.rotational_topk.rotational_topk"
    ), f"the seam must dispatch the vendored rotational kernel; got module {module}"
    assert qualname == "rotational_topk", (
        f"the seam must dispatch rotational_topk; got qualname {qualname}"
    )


# ---------------------------------------------------------------------------------------------
# `inc-glm53f-103`, rev 268 (`design-20260909-bh`). SECOND WRITER on this file, partitioned by
# REGION: everything above is `-043`'s landed acceptance and is byte-unchanged; this section is
# `-103`'s and adds one item.
# ---------------------------------------------------------------------------------------------

#: An ODD row count, which is the whole point of the item below. `-043`'s acceptance runs
#: `ROWS = 4`, and 4 splits evenly across the seam's two programs, so no program ever receives a
#: partial tile. 5 does not, and a partial tile was the defect.
ROWS_ODD = 5


def _scores_at_rows(rows: int, k: int) -> torch.Tensor:
    """``[rows, WIDTH]`` float32 scores, all distinct -- ``_scores`` with the row count freed.

    Same scaled-permutation construction, with one deliberate difference. The module
    docstring's reason for the divisor is EXACT REPRESENTABILITY in ``float32``, which holds
    for ``_scores`` because ``4 * 4096`` is a power of two. ``5 * 4096`` is not, so this
    divides by the first power of two at or above it (``2 ** 15``) and keeps the property the
    docstring's reasoning actually depends on.
    """
    gen = torch.Generator().manual_seed(44_000 + rows * 10_000 + k)
    flat = torch.randperm(rows * WIDTH, generator=gen)
    return flat.reshape(rows, WIDTH).to(torch.float32) / float(2**15)


def test_index_sets_match_torch_at_an_odd_row_count() -> None:
    """Conjunct (rev 268) -- exact index-set equality at an ODD row count, 5/5 rows.

    Certifies: ``vllm_neuron/functional/vendored_kernels/rotational_topk/rotational_topk.py``
    -- the ragged-tile guard in ``_topk_rotated_core``, driven through the same seam the items
    above drive.

    WHAT THIS READS THAT NOTHING ABOVE READS. The seam shards rows across two programs, so a
    program's tile holds ``ceil(rows / 2)`` rows. At ``rows == 4`` both programs get exactly 2
    and every partition of the kernel's staging buffer is written. At ``rows == 5`` the second
    program gets 2 rows against a 3-row tile, and before the guard the leftover partitions
    reached ``rotate``'s cross-partition ``nisa.nc_matmul`` uninitialised -- which, because that
    matmul sums every partition with a 0/1 weight, turned one non-finite word into NaN across
    every output partition and left every index at its zero default. So this item's three
    readings are the three symptoms: values finite, index sets exact, values bit-equal.

    THE FINITENESS READING IS FIRST ON PURPOSE. It is the one that names the defect, and it is
    asserted before the set comparison because an all-NaN return would otherwise fail the set
    comparison with a message about index sets and say nothing about NaN.

    THE ROUTE IS ASSERTED, or this item could pass vacuously: if the gate refused this geometry
    the seam would quietly answer from ``torch.topk`` -- which has no stages, no partitions and
    no defect to catch -- and every assertion below would hold while reading nothing.
    """
    # The split is READ from the seam's own program count, not typed, so the premise of this
    # item is a value the code supplies rather than a claim in a docstring.
    from vllm_neuron.functional.dsa.topk_select import _NUM_PROGRAMS

    rows_per_program = -(-ROWS_ODD // _NUM_PROGRAMS)
    last_program_rows = ROWS_ODD - rows_per_program * (_NUM_PROGRAMS - 1)

    scores = _scores_at_rows(ROWS_ODD, SELECT_K)
    distinct = int(torch.unique(scores).numel())
    print(f"[fixture] case=odd_rows k={SELECT_K} rows={ROWS_ODD} width={WIDTH} "
          f"distinct_scores={distinct}/{ROWS_ODD * WIDTH} dtype={scores.dtype}")
    print(f"[geometry] programs={_NUM_PROGRAMS} rows_per_program={rows_per_program} "
          f"last_program_rows={last_program_rows} "
          f"tile_is_ragged={last_program_rows < rows_per_program} "
          f"certifies=vllm_neuron.functional.vendored_kernels.rotational_topk."
          f"rotational_topk._topk_rotated_core")
    assert distinct == ROWS_ODD * WIDTH, (
        f"the fixture must hold {ROWS_ODD * WIDTH} distinct scores for the top-k index set "
        f"to be unique; got {distinct}"
    )
    # If this ever reads False the item has stopped exercising the condition it exists for,
    # and it must fail loudly rather than pass on a geometry with nothing ragged about it.
    assert last_program_rows < rows_per_program, (
        f"this item requires a RAGGED tile: {ROWS_ODD} rows over {_NUM_PROGRAMS} programs "
        f"gives {rows_per_program} per program and {last_program_rows} on the last, which is "
        f"not ragged -- the item is no longer reading the guard"
    )

    reset_topk_select_dispatch_counters()
    gate = can_run_dsa_topk_select(scores, SELECT_K)
    values, indices = dsa_topk_select(scores, SELECT_K)
    nki_dispatch, torch_fallback = topk_select_dispatch_counters()

    want_values, want_indices = torch.topk(scores, SELECT_K, dim=-1)
    finite = int(torch.isfinite(values).sum())
    nan_count = int(torch.isnan(values).sum())
    agree = _rows_whose_index_sets_agree(indices, want_indices)
    distinct_indices = [len(set(indices[r].tolist())) for r in range(ROWS_ODD)]

    print(f"[acceptance] case=odd_rows k={SELECT_K} finite={finite}/{values.numel()} "
          f"nan={nan_count} index_set_rows_agree={agree}/{ROWS_ODD} "
          f"values_bit_equal={torch.equal(values, want_values)} "
          f"distinct_indices_per_row={distinct_indices} out_shape={tuple(values.shape)}")
    print(f"[route-predicate] case=odd_rows k={SELECT_K} calls_made=1 "
          f"nki_dispatch={nki_dispatch} torch_fallback={torch_fallback} "
          f"can_run_dsa_topk_select={gate} "
          f"certifies=vllm_neuron.functional.dsa.topk_select.dsa_topk_select")

    assert nki_dispatch == 1 and torch_fallback == 0 and gate is True, (
        f"this item must read the NKI kernel, not the torch oracle: gate={gate} "
        f"nki_dispatch={nki_dispatch} torch_fallback={torch_fallback}"
    )
    assert nan_count == 0 and finite == values.numel(), (
        f"every selected value must be finite at an odd row count; got {nan_count} NaN and "
        f"{values.numel() - finite} non-finite of {values.numel()}"
    )
    assert agree == ROWS_ODD, (
        f"selected index sets must match torch.topk on every row at rows={ROWS_ODD}; "
        f"{agree} of {ROWS_ODD} rows agree"
    )
    assert torch.equal(values, want_values), (
        "selected values must be bit-equal to the torch reference at an odd row count; "
        f"max_abs_diff={(values - want_values).abs().max().item():.3e}"
    )
    assert indices.shape == (ROWS_ODD, SELECT_K)
    assert indices.dtype == torch.int64
