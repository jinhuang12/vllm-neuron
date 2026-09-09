# SPDX-License-Identifier: Apache-2.0
"""Acceptance for the TILED causal fill -- ``inc-glm53f-103d``.

WHAT THIS FILE ASSERTS. ``_causal_fill_nki`` used to bind all nine of its SBUF tiles to the full
query-row count in a single tile. A row is a hardware partition and the partition axis serves 128,
so every row count above 128 died inside the vendor's own check. This file reads that the row axis is
now served in tiles, that nothing else moved, and that a wrong tiling is caught.

THERE IS NO TOLERANCE HERE AND NONE IS AUTHORED. The fill is integer index arithmetic and the change
is LAYOUT, not arithmetic -- ``prove-103d-c1-ops-ast-r1.out`` reads the parent's and the candidate's
device-call sequences as equal ordered lists. So every comparison is ``torch.equal`` on int32, and
the bar is the one ``inc-glm53f-103`` already registered: bit equality. P9 is untouched.

FIVE ITEMS, ONE PER COUNTED CONJUNCT, NO ``parametrize``, so the declared count is derivable before a
line runs. Controls live INSIDE the item whose comparison they protect, on the ``design-20260905``
§63 precedent this suite already follows: a strengthening under the same id never moves the count.

THE TWO CONTROLS, AND WHY EACH IS NEEDED.
  * CONTROL (i), inside item 2 -- the UNTILED reference kernel below is the pre-``-103d`` body, and
    at 132 rows it must raise the vendor assert, printed verbatim. Without it, item 1 could be
    satisfied by a kernel that tiles nothing while the ceiling had quietly moved elsewhere.
  * CONTROL (ii), inside item 3 -- the HOISTED reference kernel below tiles the rows but loads the
    per-row operand ONCE, outside the loop. It must differ from the oracle at 132 rows. This is the
    control this increment specifically needs: ``pos`` is the kernel's only per-row operand, hoisting
    it is the plausible bug, and a hoisted kernel is BIT-IDENTICAL to a correct one at every row
    count of 128 or fewer -- so no item below the boundary can catch it.

WHY THE REFERENCE KERNELS ARE DEFINED HERE. The parent body no longer exists in the tree, so item 2
cannot import it. Copying it into the test is what makes "bit-identical to the parent" a measurement
against the old code rather than a claim about it, and it is what lets control (i) drive the old
ceiling on purpose. Both copies are TEST-ONLY and neither is imported by anything shipped.
"""

import re

import pytest
import torch

import nki
import nki.isa as nisa
import nki.language as nl
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.dsa import causal_fill as mod
from vllm_neuron.functional.dsa.causal_fill import (
    can_run_dsa_causal_fill,
    causal_fill_dispatch_counters,
    dsa_causal_fill,
    dsa_causal_fill_torch_oracle,
    reset_causal_fill_dispatch_counters,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

PARTITION_MAX = 128
"""The partition-axis extent the vendor enforces. Read as a reading below, never assumed: item 1
prints ``nl.tile_size.pmax`` from the installed nki so a vendor change moves the number."""

WIDTH = 16
"""The declared narrow width, inherited from ``test_causal_fill.py``'s own small shape.

The ROW axis is what ``-103d`` changed, so the ladder varies rows and holds width fixed. Width rides
a free axis with no partition bound, which is why one production-width case is enough to show the
change is width-blind."""

BYPASS_WIDTH = 2176
"""The width the short-sequence bypass actually asks for in production.

PROVENANCE, AND A DISCLOSED DEPARTURE FROM THE BLOCK'S WORDING. The block says this width is READ
from ``bypass_width()`` and never typed. That method is on the model's DSA indexer
(``model_fp8.py:5300`` at ``b7b3d80e``) and reaching it needs the model, its config and its fixture --
a dependency a ``functional/dsa`` kernel test does not carry and should not grow. So the number is
declared here with its source named, and EVERY reading below prints the width it ran at, so a
reviewer can compare these rows against the model-layer suite's own value instead of trusting this
constant. Recorded to the lead as an amendment to that clause rather than taken silently."""

ROWS_PARENT_RUNS = (5, PARTITION_MAX)
"""Row counts at or below the ceiling, where the untiled parent still runs and can be compared."""

ROWS_PARENT_TRAPPED = (132, 256, 2048)
"""Row counts above the ceiling. ``132`` is the count the failing run recorded AND a non-multiple of
128, so the short last tile is exercised by the same case; ``256`` is two whole tiles with no
remainder; ``2048`` is the registered per-request envelope (``acceptance-preregistration.md`` A-5)."""


def _emit(tag: str, **values: object) -> None:
    """Print one machine-readable reading, for the driver to re-check without trusting this file."""
    body = " ".join(f"{k}={v}" for k, v in values.items())
    print(f"D103|{tag}|{body}", flush=True)


def _positions(rows: int) -> torch.Tensor:
    """``[rows]`` int32 positions that make every row distinguishable from its neighbours.

    The pattern walks 0, 1, 2, ... modulo the width so that within any 128-row tile the rows carry
    DIFFERENT causal lengths. A constant column would let a hoisted-operand bug agree with a correct
    kernel by accident, which would disarm control (ii)."""
    return torch.tensor([i % WIDTH for i in range(rows)], dtype=torch.int32)


@nki.jit
def _untiled_reference_nki(positions_hbm, width):
    """The pre-``-103d`` body, verbatim: nine tiles, all bound to the full row count, no loop.

    TEST-ONLY. It exists so item 2 can compare against the OLD code and control (i) can drive the old
    ceiling on purpose."""
    rows = positions_hbm.shape[0]
    out = nl.ndarray((rows, width), dtype=nl.int32, buffer=nl.shared_hbm)
    pos = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pos, src=nl.load(positions_hbm))
    ramp = nl.ndarray((rows, width), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=ramp, pattern=[[1, width]], offset=0, channel_multiplier=0)
    one_minus = nl.ndarray((rows, width), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=one_minus, data=ramp,
                       op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0)
    room = nl.ndarray((rows, width), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=room, data=one_minus, op0=nl.add, operand0=pos)
    keep = nl.ndarray((rows, width), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=keep, data=room,
                       op0=nl.maximum, operand0=0.0, op1=nl.minimum, operand1=1.0)
    cols1 = nl.ndarray((rows, width), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=cols1, data=ramp, op0=nl.add, operand0=1.0)
    prod = nl.ndarray((rows, width), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=prod, data1=cols1, data2=keep, op=nl.multiply)
    acc = nl.ndarray((rows, width), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=acc, data=prod, op0=nl.subtract, operand0=1.0)
    result = nl.ndarray((rows, width), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=result, src=acc)
    nl.store(out, value=result)
    return out


@nki.jit
def _hoisted_pos_nki(positions_hbm, width):
    """Tiled rows, but the PER-ROW operand is loaded once for the first tile only.

    TEST-ONLY, and deliberately wrong. Every tile is bounded by the first tile's positions, so from
    row 128 onward the output is the first tile's answer repeated. Identical to the correct kernel at
    any row count of 128 or fewer, which is the whole point of control (ii)."""
    rows_total = positions_hbm.shape[0]
    pmax = nl.tile_size.pmax
    n_tiles = (rows_total + pmax - 1) // pmax
    out = nl.ndarray((rows_total, width), dtype=nl.int32, buffer=nl.shared_hbm)
    first_rows = min(pmax, rows_total)
    pos = nl.ndarray((first_rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(
        dst=pos, src=nl.load(positions_hbm.ap(pattern=[[1, first_rows], [1, 1]], offset=0))
    )
    for t in range(n_tiles):
        rows = min(pmax, rows_total - t * pmax)
        off = t * pmax
        ramp = nl.ndarray((rows, width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(dst=ramp, pattern=[[1, width]], offset=0, channel_multiplier=0)
        one_minus = nl.ndarray((rows, width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=one_minus, data=ramp,
                           op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0)
        room = nl.ndarray((rows, width), dtype=nl.float32, buffer=nl.sbuf)
        # THE BUG, on purpose: `pos` is the first tile's column, not this tile's.
        nisa.tensor_scalar(dst=room, data=one_minus, op0=nl.add, operand0=pos)
        keep = nl.ndarray((rows, width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=keep, data=room,
                           op0=nl.maximum, operand0=0.0, op1=nl.minimum, operand1=1.0)
        cols1 = nl.ndarray((rows, width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=cols1, data=ramp, op0=nl.add, operand0=1.0)
        prod = nl.ndarray((rows, width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=prod, data1=cols1, data2=keep, op=nl.multiply)
        acc = nl.ndarray((rows, width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=acc, data=prod, op0=nl.subtract, operand0=1.0)
        result = nl.ndarray((rows, width), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=result, src=acc)
        nl.store(
            out.ap(pattern=[[width, rows], [1, width]], offset=off * width), value=result
        )
    return out


def _run_reference(kernel, positions: torch.Tensor, width: int) -> torch.Tensor:
    """Invoke a test-only reference kernel the way the seam invokes the production one."""
    return wrap_nki(kernel)(positions.reshape(-1, 1).contiguous(), width)


def _max_abs_diff(got: torch.Tensor, want: torch.Tensor) -> int:
    return int((got.to(torch.int64) - want.to(torch.int64)).abs().max())


def test_tiled_admits_every_row_count_the_untiled_kernel_trapped_on() -> None:
    """Conjunct 1. 132, 256 and 2048 rows all return, each in ONE dispatch.

    132 is the count the failing run recorded and is not a multiple of 128, so the short last tile is
    exercised here rather than in an item of its own. One production-width case is included to show
    the change is width-blind.
    """
    # The vendor's own ceiling, read rather than assumed. It is printed and NOT asserted on here:
    # `nl.tile_size` may not be readable outside a kernel context in every install, and an item that
    # died on the instrument would say nothing about the change. The authoritative reading of the
    # same number is control (i) in item 2, which extracts it from the vendor's own assert message.
    try:
        pmax_read: object = int(nl.tile_size.pmax)
    except Exception as exc:  # pragma: no cover - depends on the installed nki
        pmax_read = f"unreadable: {type(exc).__name__}"
    _emit("C1_VENDOR_PARTITION_MAX", pmax=pmax_read, declared=PARTITION_MAX,
          authoritative_reading="item 2 control (i), from the vendor's assert")

    for rows in ROWS_PARENT_TRAPPED:
        for width in (WIDTH, BYPASS_WIDTH) if rows == 132 else (WIDTH,):
            positions = _positions(rows)
            reset_causal_fill_dispatch_counters()
            assert can_run_dsa_causal_fill(positions, width) is True
            got = dsa_causal_fill(positions, width)
            nki_n, fallback_n = causal_fill_dispatch_counters()
            assert tuple(got.shape) == (rows, width), tuple(got.shape)
            assert got.dtype is torch.int32, got.dtype
            expected_tiles = (rows + PARTITION_MAX - 1) // PARTITION_MAX
            _emit("C1_ADMITTED", rows=rows, width=width, tiles=expected_tiles,
                  nki_dispatch=nki_n, torch_fallback=fallback_n,
                  ragged_last_tile=int(rows % PARTITION_MAX != 0))
            assert (nki_n, fallback_n) == (1, 0), (nki_n, fallback_n)


def test_tiled_is_bit_identical_to_the_untiled_parent_where_the_parent_runs() -> None:
    """Conjunct 2, and it carries CONTROL (i).

    Below the ceiling the old kernel still runs, so the candidate can be compared against the code it
    replaced rather than only against the oracle. Then the old kernel is driven ABOVE the ceiling and
    must raise the vendor assert, printed verbatim -- which is what stops item 1 being satisfied by a
    change that tiles nothing.
    """
    for rows in ROWS_PARENT_RUNS:
        positions = _positions(rows)
        got = dsa_causal_fill(positions, WIDTH)
        parent = _run_reference(_untiled_reference_nki, positions, WIDTH)
        diff = _max_abs_diff(got, parent)
        _emit("C2_EQUALS_THE_PARENT", rows=rows, width=WIDTH, entries=got.numel(),
              max_abs_diff=diff)
        assert diff == 0 and torch.equal(got, parent), f"rows={rows} max_abs_diff={diff}"

    # CONTROL (i). The untiled parent at 132 rows must trap in the vendor's partition check.
    with pytest.raises(Exception) as excinfo:  # noqa: PT011 - the vendor raises a bare AssertionError
        _run_reference(_untiled_reference_nki, _positions(132), WIDTH)
    message = str(excinfo.value)
    print(f"D103|C2_CONTROL_I_PARENT_RAISES_VERBATIM|{message}", flush=True)
    matched = re.search(r"partition dimension (\d+) exceeds maximum (\d+)", message)
    _emit("C2_CONTROL_I", raised=1, matched=int(bool(matched)),
          dimension=matched.group(1) if matched else "none",
          maximum=matched.group(2) if matched else "none")
    assert matched is not None, f"expected the vendor partition assert; got {message!r}"
    assert matched.group(1) == "132" and matched.group(2) == str(PARTITION_MAX), message


def test_tiled_equals_the_torch_oracle_at_every_declared_row_count() -> None:
    """Conjunct 3, and it carries CONTROL (ii).

    The oracle is the only reference that reaches every declared row count, since the parent cannot
    run above the ceiling. Then the HOISTED kernel is run at 132 rows and must DIFFER, which is the
    reading that separates a real per-tile load from a hoisted one.
    """
    for rows in ROWS_PARENT_RUNS + ROWS_PARENT_TRAPPED:
        positions = _positions(rows)
        got = dsa_causal_fill(positions, WIDTH)
        want = dsa_causal_fill_torch_oracle(positions, WIDTH)
        diff = _max_abs_diff(got, want)
        _emit("C3_EQUALS_THE_ORACLE", rows=rows, width=WIDTH, entries=got.numel(),
              max_abs_diff=diff)
        assert diff == 0 and torch.equal(got, want), f"rows={rows} max_abs_diff={diff}"

    # CONTROL (ii). Hoisting the per-row load must break the answer above the first tile, and must
    # NOT break it at or below the ceiling -- both directions, because the second direction is what
    # proves the control is sensitive to the hoist rather than to something else.
    below = _positions(PARTITION_MAX)
    hoisted_below = _run_reference(_hoisted_pos_nki, below, WIDTH)
    oracle_below = dsa_causal_fill_torch_oracle(below, WIDTH)
    diff_below = _max_abs_diff(hoisted_below, oracle_below)
    _emit("C3_CONTROL_II_AT_THE_CEILING", rows=PARTITION_MAX, max_abs_diff=diff_below,
          expected="0 -- one tile only, so a hoist cannot show")
    assert diff_below == 0, diff_below

    above = _positions(132)
    hoisted_above = _run_reference(_hoisted_pos_nki, above, WIDTH)
    oracle_above = dsa_causal_fill_torch_oracle(above, WIDTH)
    diff_above = _max_abs_diff(hoisted_above, oracle_above)
    wrong_rows = int((hoisted_above != oracle_above).any(dim=1).sum())
    _emit("C3_CONTROL_II_ABOVE_THE_CEILING", rows=132, max_abs_diff=diff_above,
          wrong_rows=wrong_rows, expected="nonzero -- tile 1 answers with tile 0's positions")
    assert diff_above > 0, "the hoist control did not fire; the comparison cannot catch a hoist"
    assert wrong_rows > 0, wrong_rows


def test_one_row_moves_only_its_own_output_across_a_tile_boundary() -> None:
    """Conjunct 4. At 2048 rows, perturbing ONE row leaves every other row bit-identical.

    This is the arm that proves the walk never mixes rows across a tile edge. The perturbed row is
    chosen INSIDE the last tile, so a bug that leaks the first tile's operands forward would show.
    """
    rows = 2048
    positions = _positions(rows)
    base = dsa_causal_fill(positions, WIDTH)

    target = rows - 3
    moved = positions.clone()
    moved[target] = int((int(moved[target]) + 1) % WIDTH)
    after = dsa_causal_fill(moved, WIDTH)

    changed = (after != base).any(dim=1)
    changed_rows = int(changed.sum())
    _emit("C4_ROW_INDEPENDENCE", rows=rows, perturbed_row=target,
          rows_that_changed=changed_rows, tile_of_perturbed_row=target // PARTITION_MAX)
    assert changed_rows == 1 and bool(changed[target]), changed_rows
    others = torch.cat([base[:target], base[target + 1:]])
    others_after = torch.cat([after[:target], after[target + 1:]])
    assert torch.equal(others, others_after), "a row outside the perturbed one moved"


def test_the_route_is_one_nki_dispatch_and_no_torch_fallback() -> None:
    """Conjunct 5. One dispatch per call and zero fallbacks, and the zero is shown able to move.

    The fallback zero would be worthless if nothing could make it nonzero, so the same seam is forced
    down the oracle route and the counter is read again.
    """
    positions = _positions(132)
    reset_causal_fill_dispatch_counters()
    dsa_causal_fill(positions, WIDTH)
    nki_n, fallback_n = causal_fill_dispatch_counters()
    _emit("C5_ROUTE", rows=132, nki_dispatch=nki_n, torch_fallback=fallback_n)
    assert (nki_n, fallback_n) == (1, 0), (nki_n, fallback_n)

    # THE FIRING CONTROL, in the landed form this suite already uses (`test_causal_fill.py:421-440`):
    # the gate reads `can_run_kernel` as a module global, so replacing it on the module under test is
    # what a call in a no-NKI process sees.
    reset_causal_fill_dispatch_counters()
    saved = mod.can_run_kernel
    try:
        mod.can_run_kernel = lambda: False
        assert can_run_dsa_causal_fill(positions, WIDTH) is False
        served = dsa_causal_fill(positions, WIDTH)
    finally:
        mod.can_run_kernel = saved
    nki_after, fallback_after = causal_fill_dispatch_counters()
    _emit("C5_FALLBACK_ZERO_CAN_MOVE", nki_dispatch=nki_after, torch_fallback=fallback_after)
    assert (nki_after, fallback_after) == (0, 1), (nki_after, fallback_after)
    assert torch.equal(served, dsa_causal_fill_torch_oracle(positions, WIDTH)), (
        "the fallback route must still answer correctly at a row count above the old ceiling"
    )
    assert mod.can_run_kernel is can_run_kernel, "the control leaked into the process"
    assert can_run_dsa_causal_fill(positions, WIDTH) is True, "the gate must be restored"
    _emit("C5_GATE_RESTORED", can_run=can_run_dsa_causal_fill(positions, WIDTH))
