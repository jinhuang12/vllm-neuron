# SPDX-License-Identifier: Apache-2.0
"""Tier N acceptance for `inc-glm53f-026` -- the dense-half blockwise fp8 GEMM.

Acceptance command (plan block ``#### inc-glm53f-026``, with D1's Tier N env)::

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
    python -m pytest test/vllm_neuron/functional/test_blockwise_fp8_mm.py \
      -q -s --timeout 60 -p no:cacheprovider

The two DECLARED arms, and why there are two
--------------------------------------------
1. the **tolerance arm** -- simulated NKI output against a torch
   block-dequant-then-matmul oracle authored below, ``assert_close(rtol=3e-2,
   atol=1e-5)`` per output tile, all tiles passing, worst tile error reported as
   a number;
2. the **exact arm** -- the same comparison with every block scale ``1.0``,
   which isolates the dequantisation arithmetic from the quantisation error, and
   which the plan requires to pass at **single-op tolerance, 1e-5**.

The two arms run at DIFFERENT gates, and that difference is the point
------------------------------------------------------------------------
Arm 1 gates at the declared pair ``rtol=3e-2, atol=1e-5``. Arm 2 gates at
``1e-5`` on both terms (``SINGLE_OP_TOL`` below), because that is the figure its
own bullet declares.

Until repair batch R4 both arms called the same helper with the same pair, and
the module said so here. Finding ``B19-026`` found that this made arm 2 certify
nothing arm 1 did not already certify: ``assert_close`` allows
``|got - want| <= atol + rtol * |want|``, and on this fixture ``|want|`` reaches
about ``1.45e+02``, so ``rtol=3e-2`` alone allowed about ``4.3`` of absolute
error and the ``atol`` term was swallowed whole. A bf16 accumulator inside the
kernel moves the result by roughly ``4e-3`` relative -- inside ``3e-2``, so both
arms stayed green, and outside ``1e-5``, so the arm the plan created to catch it
would have caught it. ``test_exact_arm_gate_is_armed_at_single_op_tolerance``
now measures both of those readings directly.

No tolerance number is invented, widened or narrowed anywhere in this file:
``RTOL`` and ``ATOL`` below are the plan's, ``SINGLE_OP_TOL`` is bound to
``ATOL`` rather than written again, and widening one to absorb remapping error
is the user's election (§11.B.7 option (b)), never this test's.

Why the route predicate is an acceptance criterion and not a diagnostic (F1)
---------------------------------------------------------------------------
The numeric expectation compares simulated NKI output against a torch oracle. If
the seam silently took its torch path, *both* sides would be torch and the
comparison would pass green while measuring nothing about a kernel -- and the
exact arm is the one a torch fallback satisfies most easily, which is why the
plan requires the count on that arm too. So each declared case reads three route
instruments and reports each as a number:

1. the seam's own module-level dispatch counter (form R-1) -- ``nki_dispatch ==
   1``, ``torch_fallback == 0``;
2. ``can_run_kernel()`` -- ``True``;
3. real ``nki.simulator.simulate_kernel`` invocations on the F1 chain -- ``1``
   per kernel call. Instrument 3 counts the VENDOR entry point, so a bug in
   instrument 1 cannot fake it.

Every zero above is armed. ``test_route_control_fallback_counter_discriminates``
shows instrument 1 reading ``(0, 1)`` and instrument 3 reading ``0`` on the
fallback path, and ``test_route_control_simulator_is_load_bearing`` shows the
chain RAISING rather than quietly computing torch when the simulator is off.

Why the numeric comparison here IS a check on the scale mapping
--------------------------------------------------------------
Unlike `inc-glm53f-025`, whose kernel and vendor oracle read the scale tensor
through the same convention, the oracle below dequantises **first** and never
consults :func:`~vllm_neuron.functional.blockwise_fp8_mm.flat_scale_index`. With
distinct per-block scales a transposed flattening therefore shows up as a numeric
disagreement. ``test_flat_scale_index_maps_block_to_predicted_output_columns``
adds an oracle-free one-hot reading on top, and
``test_numeric_comparison_is_armed`` shows the comparison itself can fail.
"""

from __future__ import annotations

import importlib
import os
import sys

import pytest
import torch

import nki
import nki.simulator

from vllm_neuron.functional.blockwise_fp8_mm import (
    BLOCK_QUANT_SIZE,
    TILE_SIZE,
    BlockwiseFp8MmError,
    blockwise_fp8_mm,
    blockwise_fp8_mm_torch_oracle,
    can_run_blockwise_fp8_mm,
    dispatch_counters,
    flat_scale_index,
    kernel_identity,
    kernel_scale_shape,
    reset_dispatch_counters,
    scale_grid_shape,
    to_kernel_scale_layout,
)
from vllm_neuron.functional.moe.blockwise_fp8_retile import (
    DOWN,
    is_pow2_exact,
    retile_block_scales,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

# --------------------------------------------------------------------------- #
# The tiny config. Every extent is forced by the kernel's own tiling.           #
# --------------------------------------------------------------------------- #
M = 256   # tokens; a whole number of TILE_SIZE PSUM partition tiles
K = 512   # contraction; a whole number of BLOCK_QUANT_SIZE scale blocks
N = 512   # output width; a whole number of BLOCK_QUANT_SIZE scale blocks

M_TILES = M // TILE_SIZE            # 2
K_BLOCKS = K // BLOCK_QUANT_SIZE    # 2
N_BLOCKS = N // BLOCK_QUANT_SIZE    # 2
OUTPUT_TILES = M_TILES * N_BLOCKS   # 4 -- the "per output tile" population

#: The declared tolerance pair, from the plan block. Not moved anywhere below.
RTOL = 3e-2
ATOL = 1e-5

#: The exact arm's own declared gate: the plan's "a 1e-5 exact-scale case ...
#: which must pass at single-op tolerance". It is BOUND to ``ATOL`` rather than
#: written as a second literal, because ``ATOL`` already IS the plan's 1e-5.
#: Nothing new is introduced here and nothing existing moves: this name only
#: routes the plan's own figure to the arm whose bullet declares it.
#: Added by ``inc-glm53f-026``'s repair for finding ``B19-026`` (repair batch R4).
SINGLE_OP_TOL = ATOL

_FP8 = torch.float8_e4m3fn
_MODULE = "vllm_neuron.functional.blockwise_fp8_mm"


class RouteInstrumentError(AssertionError):
    """A route reading that is not what the plan declares."""


class F1PreconditionError(AssertionError):
    """The pow2 losslessness precondition did not hold on this case's scales."""


class ScaleMappingError(AssertionError):
    """The observed block-to-scale mapping is not the one this module declares."""


class VacuousControlError(AssertionError):
    """A control whose input could not have made it fail.

    A zero over vacuous input measures nothing, so the control refuses to report
    a pass it did not earn.
    """


# --------------------------------------------------------------------------- #
# Route instrumentation. Counts the VENDOR entry point, so it is independent    #
# of the seam counter it cross-checks.                                          #
# --------------------------------------------------------------------------- #
class _SimulatorCounter:
    """Counts real ``nki.simulator.simulate_kernel`` calls for the duration."""

    def __init__(self) -> None:
        self.calls = 0
        self._real = None

    def __enter__(self) -> "_SimulatorCounter":
        self._real = nki.simulator.simulate_kernel
        real = self._real

        def counting(*args, **kwargs):
            self.calls += 1
            return real(*args, **kwargs)

        nki.simulator.simulate_kernel = counting
        return self

    def __exit__(self, *exc_info) -> None:
        nki.simulator.simulate_kernel = self._real


def _assert_route(sim: _SimulatorCounter, expected_dispatches: int, label: str) -> str:
    """Read all three route instruments and return the reading for the transcript."""
    nki_dispatch, torch_fallback = dispatch_counters()
    gate = can_run_kernel(torch.zeros(1))
    reading = (
        f"[{label}] nki_dispatch={nki_dispatch} torch_fallback={torch_fallback} "
        f"can_run_kernel={gate} simulate_kernel_calls={sim.calls}"
    )
    print(reading)
    if nki_dispatch != expected_dispatches:
        raise RouteInstrumentError(
            f"{label}: seam dispatch counter read {nki_dispatch}, declared "
            f"{expected_dispatches}. {reading}"
        )
    if torch_fallback != 0:
        raise RouteInstrumentError(
            f"{label}: torch-fallback counter read {torch_fallback}, declared "
            f"exactly 0 -- a fallback pass would compare torch against torch. "
            f"{reading}"
        )
    if gate is not True:
        raise RouteInstrumentError(
            f"{label}: can_run_kernel() read {gate!r}, declared True. {reading}"
        )
    if sim.calls != expected_dispatches:
        raise RouteInstrumentError(
            f"{label}: nki.simulator.simulate_kernel ran {sim.calls} times, "
            f"declared {expected_dispatches}. A numeric pass without a simulator "
            f"call is the F1 false green. {reading}"
        )
    return reading


# --------------------------------------------------------------------------- #
# Case construction.                                                           #
#                                                                              #
# The retile is `inc-glm53f-024`'s LANDED producer, not a re-implementation:    #
# this test consumes its `block_scales` / `retiled_weights` / `records` and     #
# never its MoE `consumer_scales`, so no MoE flat-index semantics enter the     #
# dense path. The one claim that reuse makes -- the axis mapping from the       #
# producer's (E, i_256, h_256) to this module's [k_block, n_block] -- is        #
# SETTLED by a bit-exact dequantisation-invariance reading, not assumed:        #
# `test_retile_reuse_is_dequantisation_invariant`.                             #
# --------------------------------------------------------------------------- #
#: Per-``256``-block exponent of the RETAINED scale, and the per-constituent
#: offsets around it. DETERMINISTIC rather than sampled, because four properties
#: of this fixture are load-bearing and a draw can only satisfy them by luck:
#: (i) every entry an exact power of two, (ii) the four block scales DISTINCT so a
#: transposed flat index is numerically visible, (iii) the block-scale matrix
#: ASYMMETRIC so the transposed-mapping control in
#: :func:`test_retile_reuse_is_dequantisation_invariant` can fail, and (iv) every
#: constituent ratio NON-UNIT for three of four positions so conjunct 1 is
#: exercised rather than satisfied by uniformity.
_BLOCK_EXPONENTS = (-3, 1, 2, -1)
#: Offsets at ``(d_k, d_n)``. ``(0, 0)`` is ``0`` because `inc-glm53f-024` retains
#: the block's ``(0, 0)`` constituent as the block scale. The others are bounded
#: at ONE exponent so that rescaling weights on ``m/8`` (``m in 1..7``) by the
#: ratio stays inside e4m3's NORMAL range -- ``1/16 = 2**-4`` against a minimum
#: normal of ``2**-6`` -- which is what makes ``inexact_rescales == 0`` a
#: property of the construction rather than of the seed.
_RATIO_OFFSETS = ((0, 1), (-1, 1))


def _pow2_checkpoint_scales(uniform_one: bool = False) -> torch.Tensor:
    """``(1, K//128, N//128)`` fp32 checkpoint scales, every entry an exact pow2.

    Built so the F1 COMPLETE condition holds NON-trivially: the retained
    ``256``-block scale is ``2 ** e_block`` with ``e_block`` varying per block,
    and the other three constituents are ``2 ** (e_block + d)`` with
    ``d in {-1, 0, 1}`` -- mutually pow2-related without being uniform.

    ``uniform_one=True`` returns all ``1.0`` -- the exact arm. ``1.0`` is
    ``2 ** 0`` (all-zero significand under the bit-pattern test), so that arm
    already satisfies both conjuncts by construction and needs no repair,
    exactly as the plan block states.
    """
    grid = (1, K // TILE_SIZE, N // TILE_SIZE)
    if uniform_one:
        return torch.ones(grid, dtype=torch.float32)

    exponents = torch.zeros(grid[1:], dtype=torch.int64)
    for k_block in range(K_BLOCKS):
        for n_block in range(N_BLOCKS):
            base = _BLOCK_EXPONENTS[
                (k_block * N_BLOCKS + n_block) % len(_BLOCK_EXPONENTS)
            ]
            for d_k in range(2):
                for d_n in range(2):
                    exponents[k_block * 2 + d_k, n_block * 2 + d_n] = (
                        base + _RATIO_OFFSETS[d_k][d_n]
                    )
    return torch.ldexp(torch.ones(grid, dtype=torch.float32), exponents.unsqueeze(0))


def _fp8_grid(seed: int, *shape: int, signed: bool = False) -> torch.Tensor:
    """Values already on the fp8-e4m3 grid, so every cast in the fixture is exact.

    ``signed=False`` is the default and it is a CONDITIONING choice carried from
    `inc-glm53f-025`, whose attempt 1 read ``max_rel_error=8.32e+01`` against
    ``rtol=3e-2`` from catastrophic cancellation in a SIGNED fixture over a
    512-wide contraction -- not from a kernel defect. With signed values the
    reference lands arbitrarily close to zero while the terms that built it are
    ~1e2, so a pointwise RELATIVE tolerance is dominated by cancellation and no
    correct kernel can satisfy it.

    The hazard is sharper here because the oracle is authored in this same test,
    so the declared arms run on a well-conditioned (positive) fixture -- which is
    what makes ``rtol=3e-2`` a statement about the kernel -- and signed coverage
    is kept as its own arm at the SAME declared tolerance, compared in a
    cancellation-robust norm:
    :func:`test_signed_fixture_agrees_in_norm_under_cancellation`. The tolerance
    is UNCHANGED; only the world it is measured in is narrowed, exactly as the
    plan's F1 clause does for the scales.
    """
    generator = torch.Generator().manual_seed(seed)
    low = -7 if signed else 1
    return torch.randint(low, 8, shape, generator=generator).to(torch.float32) / 8.0


def _build_case(uniform_one: bool = False, signed: bool = False) -> dict:
    """The tiny config, retiled through `inc-glm53f-024`'s landed producer."""
    checkpoint = _pow2_checkpoint_scales(uniform_one=uniform_one)
    weights = _fp8_grid(21, 1, K, N, signed=signed)

    result = retile_block_scales(weights.to(_FP8), checkpoint, projection=DOWN)

    # The producer's block_scales are (E, i_256, h_256) with rows=K and cols=N,
    # so the transpose lands on this module's [k_block, n_block]. The mapping is
    # settled by test_retile_reuse_is_dequantisation_invariant, not asserted here.
    block_scales = result.block_scales[0].t().contiguous()
    weight = result.retiled_weights[0].contiguous()
    x = _fp8_grid(31, M, K, signed=signed).to(torch.bfloat16)

    return {
        "x": x,
        "weight": weight,
        "weight_scale": block_scales,
        "checkpoint": checkpoint,
        "raw_weights": weights,
        "result": result,
    }


def _tile(index: int) -> tuple[slice, slice]:
    """Output tile ``index`` as ``(row slice, column slice)``.

    Tiles are ``(m_tile, n_block)`` pairs: ``TILE_SIZE`` rows by
    ``BLOCK_QUANT_SIZE`` columns, which is exactly the region the kernel
    accumulates and stores in one pass.
    """
    m_tile, n_block = divmod(index, N_BLOCKS)
    return (
        slice(m_tile * TILE_SIZE, (m_tile + 1) * TILE_SIZE),
        slice(n_block * BLOCK_QUANT_SIZE, (n_block + 1) * BLOCK_QUANT_SIZE),
    )


def _max_rel_error(got: torch.Tensor, want: torch.Tensor) -> float:
    """``max |got - want| / (|want| + ATOL)`` -- a number, not a verdict."""
    return float(((got - want).abs() / (want.abs() + ATOL)).max())


def _compare_per_output_tile(
    got: torch.Tensor,
    want: torch.Tensor,
    label: str,
    *,
    rtol: float = RTOL,
    atol: float = ATOL,
) -> float:
    """Assert a declared tolerance per output tile; return the worst error.

    Reports every tile's error as a number and the worst one, which is the
    plan's declared expected result for both arms.

    ``rtol`` and ``atol`` default to the declared pair, which is arm 1's gate.
    The exact arm passes ``SINGLE_OP_TOL`` for both, which is the gate ITS own
    bullet declares. The gate in force is printed on every line below, so a
    reader never has to infer which arm ran at which numbers -- that inference
    is what finding ``B19-026`` had to make by hand.
    """
    nonzero = int((want.abs().sum(-1) > 0).sum())
    if nonzero == 0:
        raise VacuousControlError(
            f"{label}: the oracle produced an all-zero reference, so the "
            f"comparison would pass over empty input; refusing to report a pass"
        )
    print(
        f"[{label}] oracle_nonzero_rows={nonzero} of {M} "
        f"want_absmax={float(want.abs().max()):.6e} "
        f"want_absmin={float(want.abs().min()):.6e} "
        f"got_absmax={float(got.abs().max()):.6e}"
    )

    worst, worst_tile, passed = -1.0, -1, 0
    for index in range(OUTPUT_TILES):
        rows, cols = _tile(index)
        rel = _max_rel_error(got[rows, cols], want[rows, cols])
        print(
            f"[{label}] tile={index} rows={rows.start}:{rows.stop} "
            f"cols={cols.start}:{cols.stop} max_rel_error={rel:.6e}"
        )
        if rel > worst:
            worst, worst_tile = rel, index
        torch.testing.assert_close(
            got[rows, cols], want[rows, cols], rtol=rtol, atol=atol
        )
        passed += 1

    print(
        f"[{label}] tiles_passing={passed}/{OUTPUT_TILES} "
        f"worst_max_rel_error={worst:.6e} worst_tile={worst_tile} "
        f"rtol={rtol} atol={atol} "
        f"worst_max_abs_error={float((got - want).abs().max()):.6e}"
    )
    assert passed == OUTPUT_TILES, f"{passed}/{OUTPUT_TILES} tiles passed"
    return worst


# --------------------------------------------------------------------------- #
# DECLARED ARM 1 -- the tolerance arm.                                         #
# --------------------------------------------------------------------------- #
def test_output_matches_torch_oracle_per_output_tile() -> None:
    """Simulated NKI output vs the torch dequant-then-matmul oracle, per tile.

    The plan's declared Expected: ``assert_close(rtol=3e-2, atol=1e-5)`` per
    output tile, all tiles passing, worst tile error reported as a number.
    """
    case = _build_case()
    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        got = blockwise_fp8_mm(case["x"], case["weight"], case["weight_scale"])
    _assert_route(sim, 1, "tolerance-arm")

    want = blockwise_fp8_mm_torch_oracle(
        case["x"], case["weight"], case["weight_scale"]
    )
    _compare_per_output_tile(got.to(torch.float32), want, "tolerance-arm")


# --------------------------------------------------------------------------- #
# DECLARED ARM 2 -- the exact arm, all block scales 1.0.                       #
# --------------------------------------------------------------------------- #
def test_exact_scale_arm_all_block_scales_one() -> None:
    """Every block scale ``1.0``: isolates dequant arithmetic from quant error.

    THE GATE HERE IS ``1e-5`` ON BOTH TERMS, NOT THE DECLARED PAIR. The plan's
    bullet for this arm says "a 1e-5 exact-scale case where all block scales are
    1.0, which must pass at single-op tolerance". Running it at ``rtol=3e-2``
    would let about ``4.3`` of absolute error through on this fixture, which is
    more than arm 1 already allows, so the arm would isolate nothing. Finding
    ``B19-026`` found exactly that. The measured error on this arm is
    ``0.000000e+00`` on all four tiles, so the tighter gate is passable as it
    stands and no number had to move to reach it.

    The route count matters MOST here. With unit scales the dequantisation is a
    no-op, so this is the arm a silent torch fallback would satisfy most easily
    -- which is why the plan requires ``1`` dispatch on this arm too and not only
    on the tolerance arm.
    """
    case = _build_case(uniform_one=True)
    scale = case["weight_scale"]
    assert torch.equal(scale, torch.ones_like(scale)), (
        "the exact arm's block scales are not all 1.0, so this arm is not the "
        f"arm the plan declares; got unique values {scale.unique().tolist()}"
    )
    # 1.0 is 2 ** 0: F1's second conjunct holds by construction on this arm.
    assert is_pow2_exact(1.0)

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        got = blockwise_fp8_mm(case["x"], case["weight"], scale)
    _assert_route(sim, 1, "exact-arm")

    want = blockwise_fp8_mm_torch_oracle(case["x"], case["weight"], scale)
    _compare_per_output_tile(
        got.to(torch.float32),
        want,
        "exact-arm",
        rtol=SINGLE_OP_TOL,
        atol=SINGLE_OP_TOL,
    )


# --------------------------------------------------------------------------- #
# NON-VACUITY of the EXACT ARM'S gate specifically (finding B19-026).           #
# --------------------------------------------------------------------------- #
#: The precision the kernel module names as load-bearing at
#: ``blockwise_fp8_mm.py:42-44`` -- "the PSUM tile and the SBUF accumulator are
#: fp32, and the kernel returns fp32". This control degrades a REAL kernel result
#: to bf16 and back, which is the observable consequence of that choice being
#: reversed, without editing the kernel.
DEGRADED_ACCUMULATOR_DTYPE = torch.bfloat16


def test_exact_arm_gate_is_armed_at_single_op_tolerance() -> None:
    """A bf16-degraded result must pass the declared pair and FAIL ``1e-5``.

    WHY THIS ARM EXISTS. The exact arm's own error is ``0.000000e+00``, and a
    zero is indistinguishable from an unwired gate. Finding ``B19-026`` found
    that the gate really was unwired: both arms ran at ``rtol=3e-2``, which on
    this fixture allows about ``4.3`` of absolute error, so the exact arm could
    not fail on anything arm 1 would not already have failed on.

    WHAT IT MEASURES, AS TWO COUNTED READINGS. It takes the exact arm's real
    simulated-NKI output and rounds it through bf16 and back. bf16 keeps 8
    significand bits, so this is about ``2e-3`` of relative error -- the same
    order as reversing the kernel's fp32 accumulator, which is the failure the
    finding named. Then:

    1. that degraded result PASSES at ``rtol=RTOL, atol=ATOL``. This is the
       reading that proves the old gate was blind; if it ever fails, the fixture
       has changed and this control's premise is gone.
    2. that same degraded result FAILS at ``rtol=atol=SINGLE_OP_TOL``. This is
       the reading that proves the new gate bites.

    NO NUMBER MOVES HERE. Both gates are the file's existing constants. The
    control changes the OUTPUT it feeds them, never a tolerance.

    A SOURCE MUTATION IS STILL THE SHARPER CONTROL, and the repair's host
    transcript carries one: ``acc`` and ``out`` in the kernel switched to bf16,
    the exact arm red, restored, green. That cannot live in the suite, because it
    edits shipped source. This arm is the part that can.
    """
    case = _build_case(uniform_one=True)
    scale = case["weight_scale"]
    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        got = blockwise_fp8_mm(case["x"], case["weight"], scale)
    _assert_route(sim, 1, "exact-arm-armed-control")

    want = blockwise_fp8_mm_torch_oracle(case["x"], case["weight"], scale)
    exact = got.to(torch.float32)
    degraded = exact.to(DEGRADED_ACCUMULATOR_DTYPE).to(torch.float32)

    changed = int((degraded != exact).sum())
    if changed == 0:
        raise VacuousControlError(
            "rounding the kernel result through "
            f"{DEGRADED_ACCUMULATOR_DTYPE} changed no element, so this control "
            "degrades nothing and its two readings below would be vacuous"
        )

    degraded_rel = _max_rel_error(degraded, want)
    degraded_abs = float((degraded - want).abs().max())
    print(
        f"[exact-arm-armed] elements_changed={changed}/{exact.numel()} "
        f"degraded_max_rel_error={degraded_rel:.6e} "
        f"degraded_max_abs_error={degraded_abs:.6e} "
        f"want_absmax={float(want.abs().max()):.6e}"
    )

    passes_declared_pair = True
    try:
        torch.testing.assert_close(degraded, want, rtol=RTOL, atol=ATOL)
    except AssertionError:
        passes_declared_pair = False

    passes_single_op = True
    try:
        torch.testing.assert_close(
            degraded, want, rtol=SINGLE_OP_TOL, atol=SINGLE_OP_TOL
        )
    except AssertionError:
        passes_single_op = False

    print(
        f"[exact-arm-armed] passes_at_declared_pair={passes_declared_pair} "
        f"(rtol={RTOL} atol={ATOL}) "
        f"passes_at_single_op={passes_single_op} "
        f"(rtol=atol={SINGLE_OP_TOL})"
    )

    assert passes_declared_pair, (
        f"a {DEGRADED_ACCUMULATOR_DTYPE} result was rejected by the declared "
        f"pair rtol={RTOL} atol={ATOL} (max rel error {degraded_rel:.6e}). That "
        f"contradicts finding B19-026's premise, so the reading below no longer "
        f"shows what the repair claims it shows"
    )
    assert not passes_single_op, (
        f"a {DEGRADED_ACCUMULATOR_DTYPE} result PASSED at single-op tolerance "
        f"{SINGLE_OP_TOL} (max rel error {degraded_rel:.6e}, max abs error "
        f"{degraded_abs:.6e}). The exact arm's gate is therefore still not "
        f"armed against the precision loss the plan created it to catch"
    )


# --------------------------------------------------------------------------- #
# NON-VACUITY of the numeric comparison (D1.5): it must be able to FAIL.        #
# --------------------------------------------------------------------------- #
def test_numeric_comparison_is_armed() -> None:
    """Feed the kernel a WRONG block scale; the declared comparison must fail.

    Without this arm a worst-tile error of ``0`` would be indistinguishable from
    an unwired comparison. Here one block scale is doubled on the kernel's side
    only, and the same ``assert_close(rtol=RTOL, atol=ATOL)`` is required to
    RAISE. No tolerance is changed: the control uses the declared pair.
    """
    case = _build_case()
    true_scale = case["weight_scale"]
    # Perturb the block carrying the LARGEST scale. Chosen by reading the
    # fixture rather than hardcoded: doubling a block whose scale is far below
    # its column neighbour's could move the sum by less than RTOL, which would
    # make this control fail for a reason that says nothing about the kernel.
    flat_argmax = int(true_scale.reshape(-1).argmax())
    target_k, target_n = divmod(flat_argmax, N_BLOCKS)
    wrong_scale = true_scale.clone()
    wrong_scale[target_k, target_n] = wrong_scale[target_k, target_n] * 2.0
    if torch.equal(wrong_scale, true_scale):
        raise VacuousControlError("injection changed nothing -- control is vacuous")

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        got = blockwise_fp8_mm(case["x"], case["weight"], wrong_scale).to(
            torch.float32
        )
    _assert_route(sim, 1, "armed-control")

    want = blockwise_fp8_mm_torch_oracle(case["x"], case["weight"], true_scale)
    rel = _max_rel_error(got, want)
    print(
        f"[armed-control] scale[{target_k},{target_n}]="
        f"{float(true_scale[target_k, target_n])} doubled: "
        f"max_rel_error={rel:.6e} vs rtol={RTOL}"
    )
    assert rel > RTOL, (
        f"doubling one block scale moved the comparison by only {rel:.6e}, "
        f"which is inside the declared rtol {RTOL}; the comparison is therefore "
        f"not a discriminator and a green tolerance arm would mean nothing"
    )
    with pytest.raises(AssertionError):
        torch.testing.assert_close(got, want, rtol=RTOL, atol=ATOL)


# --------------------------------------------------------------------------- #
# F1 PRECONDITION -- asserted HERE, not inherited from `inc-glm53f-024`.        #
# --------------------------------------------------------------------------- #
def test_f1_precondition_complete_condition_n_over_n() -> None:
    """Both losslessness conjuncts hold on the tolerance arm's scales, N/N tiles.

    Conjunct 1: the four constituent ``[128,128]`` scales are mutually
    power-of-two-related. Conjunct 2: the retained ``256``-block scale is itself
    a power of two. Both by the **bit-pattern** predicate
    :func:`~vllm_neuron.functional.moe.blockwise_fp8_retile.is_pow2_exact`,
    never ``log2``.

    Recomputed from the producer's emitted records rather than read off its own
    ``lossless`` flag, so this is an independent reading.
    """
    case = _build_case()
    result = case["result"]
    records = result.records
    if not records:
        raise VacuousControlError(
            "the producer emitted 0 block records, so an N/N reading would be "
            "0/0 -- vacuous"
        )

    lossless = 0
    ratio_count = 0
    non_unit_ratios = 0
    for record in records:
        conjunct1 = all(is_pow2_exact(ratio) for ratio in record.ratios)
        conjunct2 = is_pow2_exact(record.block_scale)
        ratio_count += len(record.ratios)
        non_unit_ratios += sum(1 for ratio in record.ratios if ratio != 1.0)
        if conjunct1 and conjunct2:
            lossless += 1
        else:
            raise F1PreconditionError(
                f"block {record.key}: conjunct1(quad mutually pow2)={conjunct1} "
                f"ratios={record.ratios} conjunct2(block scale pow2)="
                f"{conjunct2} block_scale={record.block_scale!r}. The tolerance "
                f"{RTOL}/{ATOL} certifies KERNEL error only when both hold; "
                f"widening it to absorb remapping error is the user's election, "
                f"not this test's."
            )

    print(
        f"[f1] complete_condition_blocks={lossless}/{len(records)} (N/N required) "
        f"ratios_tested={ratio_count} non_unit_ratios={non_unit_ratios} "
        f"inexact_rescales={result.inexact_rescales} "
        f"emitted_unsupplied={result.emitted_unsupplied} "
        f"input_scales_dropped={result.input_scales_dropped}"
    )
    assert lossless == len(records), (
        f"{lossless}/{len(records)} blocks satisfy both conjuncts"
    )
    assert len(records) == K_BLOCKS * N_BLOCKS, (
        f"expected {K_BLOCKS * N_BLOCKS} block records, got {len(records)}"
    )
    # The retile must be bit-exact on the weights it rescaled, or the kernel
    # would be fed a weight the checkpoint does not contain.
    assert result.inexact_rescales == 0, (
        f"inexact_rescales={result.inexact_rescales}, declared 0"
    )
    # Conjunct 1 must be exercised, not satisfied by a uniform quad: if every
    # ratio were 1.0 the mutual-pow2 conjunct would be a tautology here.
    assert non_unit_ratios > 0, (
        "every constituent ratio is 1.0, so conjunct 1 is satisfied by "
        "uniformity and this reading does not exercise it"
    )


def test_f1_detector_catches_a_non_pow2_block_scale() -> None:
    """DETECTOR: the F1 predicate must FAIL a block that violates the condition.

    Without this arm, ``N/N`` above is a tautology over scales this test itself
    chose. The detector feeds the real predicate a real violating scale and
    requires a real negative.
    """
    clean = _build_case()["result"]
    clean_bad = [
        record.key
        for record in clean.records
        if not (
            all(is_pow2_exact(r) for r in record.ratios)
            and is_pow2_exact(record.block_scale)
        )
    ]
    if clean_bad:
        raise VacuousControlError(
            f"the clean arm already violates F1 at {clean_bad[:3]}, so the "
            f"detector cannot attribute a negative to its injection"
        )

    # 3.0 has a non-zero significand field, so it is not a power of two: the
    # retained block scale for block (0, 0) violates conjunct 2 and the three
    # ratios taken against it violate conjunct 1.
    checkpoint = _pow2_checkpoint_scales()
    poisoned = checkpoint.clone()
    poisoned[0, 0, 0] = 3.0
    if torch.equal(poisoned, checkpoint):
        raise VacuousControlError("injection changed nothing -- control is vacuous")

    dirty = retile_block_scales(
        _fp8_grid(21, 1, K, N).to(_FP8), poisoned, projection=DOWN
    )
    violations = [
        record.key
        for record in dirty.records
        if not (
            all(is_pow2_exact(r) for r in record.ratios)
            and is_pow2_exact(record.block_scale)
        )
    ]
    print(
        f"[f1-detector] clean_violations={len(clean_bad)} "
        f"poisoned_violations={len(violations)} first={violations[:1]} "
        f"poisoned_inexact_rescales={dirty.inexact_rescales}"
    )
    assert violations, (
        "the F1 predicate passed a non-pow2 block scale (3.0); it is therefore "
        "not a discriminator and the N/N reading above means nothing"
    )
    assert (0, 0, 0) in violations, f"expected block (0,0,0) flagged, got {violations}"
    assert not is_pow2_exact(3.0)
    assert is_pow2_exact(2.0) and is_pow2_exact(0.25)


# --------------------------------------------------------------------------- #
# The `inc-glm53f-024` reuse, settled rather than assumed.                      #
# --------------------------------------------------------------------------- #
def test_retile_reuse_is_dequantisation_invariant() -> None:
    """The retiled (weight, block scale) pair dequantises to the SAME product.

    This is the one reading the reuse of `inc-glm53f-024`'s MoE-shaped producer
    needs: it settles the axis mapping from the producer's
    ``(E, i_256, h_256)`` block scales onto this module's
    ``[k_block, n_block]`` grid **without** any claim about which axis is which.
    A transposed mapping changes the dequantised product wherever the blocks
    carry distinct scales, and this fixture's blocks do.

    Required BIT-EXACT, not within a tolerance: both sides scale the same fp8
    weight by powers of two, which is exact in fp32 barring overflow.
    """
    case = _build_case()
    retiled = case["weight"].to(torch.float32) * case[
        "weight_scale"
    ].repeat_interleave(BLOCK_QUANT_SIZE, 0).repeat_interleave(BLOCK_QUANT_SIZE, 1)
    checkpoint = case["raw_weights"][0] * case["checkpoint"][0].repeat_interleave(
        TILE_SIZE, 0
    ).repeat_interleave(TILE_SIZE, 1)

    distinct = case["weight_scale"].unique().numel()
    print(
        f"[retile-reuse] distinct_block_scales={distinct} of "
        f"{K_BLOCKS * N_BLOCKS} bit_exact={bool(torch.equal(retiled, checkpoint))} "
        f"max_abs_diff={float((retiled - checkpoint).abs().max()):.6e}"
    )
    if distinct < 2:
        raise VacuousControlError(
            "every block carries the same scale, so a transposed axis mapping "
            "would be invisible to this reading"
        )
    assert torch.equal(retiled, checkpoint), (
        "the retiled weight and block scale do not dequantise to the checkpoint "
        "product bit-exactly, so either the axis mapping onto [k_block, "
        "n_block] is wrong or the rescale was inexact"
    )

    # Control: the reading must move when the mapping is wrong.
    transposed = case["weight_scale"].t().contiguous()
    mistaken = case["weight"].to(torch.float32) * transposed.repeat_interleave(
        BLOCK_QUANT_SIZE, 0
    ).repeat_interleave(BLOCK_QUANT_SIZE, 1)
    print(
        f"[retile-reuse] control transposed_mapping_bit_exact="
        f"{bool(torch.equal(mistaken, checkpoint))}"
    )
    assert not torch.equal(mistaken, checkpoint), (
        "a transposed scale mapping dequantises identically, so this reading "
        "cannot settle the mapping and the control is unarmed"
    )


# --------------------------------------------------------------------------- #
# The flat index, read off the kernel rather than off the arithmetic.           #
# --------------------------------------------------------------------------- #
def test_flat_scale_index_maps_block_to_predicted_output_columns() -> None:
    """A one-hot block scale must move exactly the output columns it owns.

    Oracle-free: it compares two KERNEL runs, so it cannot be satisfied by a
    shared misreading between kernel and oracle. Under the declared index
    ``flat = k_block * n_n_blocks + n_block``, raising the scale of block
    ``(k_block, n_block)`` changes only output columns
    ``[n_block * 256, (n_block + 1) * 256)`` -- the contraction block ``k_block``
    contributes to every column of that range and to no other.
    """
    case = _build_case(uniform_one=True)
    unit = case["weight_scale"]
    probed_scale = unit.clone()
    target_k, target_n = 1, 1
    probed_scale[target_k, target_n] = 2.0
    if torch.equal(probed_scale, unit):
        raise VacuousControlError("injection changed nothing -- control is vacuous")

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        baseline = blockwise_fp8_mm(case["x"], case["weight"], unit).to(torch.float32)
        probed = blockwise_fp8_mm(case["x"], case["weight"], probed_scale).to(
            torch.float32
        )
    # SUPPLEMENTARY case: two kernel runs, so 2 dispatches BY CONSTRUCTION. The
    # plan's "1 per declared case" governs the two declared arms above.
    _assert_route(sim, 2, "flat-index-probe")

    delta = (probed - baseline).abs()
    per_block = [
        float(
            delta[
                :, n * BLOCK_QUANT_SIZE : (n + 1) * BLOCK_QUANT_SIZE
            ].max()
        )
        for n in range(N_BLOCKS)
    ]
    print(
        f"[flat-index] flat_scale_index({target_k},{target_n},{N_BLOCKS})="
        f"{flat_scale_index(target_k, target_n, N_BLOCKS)} "
        f"delta_max_per_n_block={['%.6e' % v for v in per_block]}"
    )
    if max(per_block) == 0.0:
        raise VacuousControlError(
            "raising a block scale changed no output at all, so this probe "
            "identifies nothing; the instrument is unarmed"
        )
    for n in range(N_BLOCKS):
        moved = per_block[n] > 0.0
        if moved != (n == target_n):
            raise ScaleMappingError(
                f"block (k={target_k}, n={target_n}) moved output columns of "
                f"n_block {n} = {moved}, expected {n == target_n}. Observed "
                f"deltas {per_block}. The declared index is k_block-major; this "
                f"reading contradicts it and routes to the lead as a design "
                f"contradiction, never a silent re-index."
            )
    print("[flat-index] SETTLED: k_block-major; the n_block owns its columns")


# --------------------------------------------------------------------------- #
# SUPPLEMENTARY -- signed coverage, kept after the declared arms were           #
# conditioned. Same declared tolerance, cancellation-robust norm.               #
# --------------------------------------------------------------------------- #
def test_signed_fixture_agrees_in_norm_under_cancellation() -> None:
    """Signed weights and activations, compared in per-tile relative L2.

    Kept so sign handling stays covered after the declared arms moved to a
    well-conditioned fixture (the `inc-glm53f-025` carry). Over a 512-wide
    contraction a signed fixture cancels, so this arm applies the SAME declared
    ``RTOL`` to a per-tile relative L2 norm, which cancellation does not distort.
    No new tolerance number is introduced: the bound is ``RTOL``.

    The pointwise numbers are printed alongside so the conditioning effect is
    visible in the transcript rather than described in prose.
    """
    case = _build_case(signed=True)
    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        got = blockwise_fp8_mm(case["x"], case["weight"], case["weight_scale"]).to(
            torch.float32
        )
    _assert_route(sim, 1, "signed-norm")
    want = blockwise_fp8_mm_torch_oracle(
        case["x"], case["weight"], case["weight_scale"]
    )

    within = (got - want).abs() <= (ATOL + RTOL * want.abs())
    denominator = float((want * want).sum())
    best_fit = float((got * want).sum()) / denominator if denominator > 0 else float("nan")
    print(
        f"[signed-norm] pointwise_within_declared_tol="
        f"{int(within.sum())}/{within.numel()} "
        f"pointwise_max_rel_error={_max_rel_error(got, want):.6e} "
        f"best_fit_scale={best_fit:.6f} "
        f"want_absmax={float(want.abs().max()):.6e} "
        f"want_absmin={float(want.abs().min()):.6e}"
    )

    for index in range(OUTPUT_TILES):
        rows, cols = _tile(index)
        residual = float((got[rows, cols] - want[rows, cols]).pow(2).sum().sqrt())
        reference = float(want[rows, cols].pow(2).sum().sqrt())
        if reference == 0.0:
            raise VacuousControlError(
                f"tile {index}: the reference has zero norm, so a relative "
                f"comparison against it is vacuous"
            )
        rel_l2 = residual / reference
        print(f"[signed-norm] tile={index} rel_L2={rel_l2:.6e} bound={RTOL}")
        assert rel_l2 <= RTOL, (
            f"tile {index}: relative L2 {rel_l2:.6e} exceeds the declared rtol "
            f"{RTOL}; that is a structural disagreement, not cancellation"
        )


# --------------------------------------------------------------------------- #
# ROUTE CONTROLS -- so every zero above is a measurement.                       #
# --------------------------------------------------------------------------- #
def test_route_control_fallback_counter_discriminates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the simulator disabled the seam takes the torch path, and it is COUNTED.

    This is the arm that makes ``torch_fallback == 0`` above meaningful: the
    counter is shown reading ``1`` and ``nki_dispatch`` reading ``0``, through
    the real gate rather than a mock. It is also the measured form of the plan's
    claim that a pure-torch implementation yields ``0`` dispatches.
    """
    case = _build_case()
    monkeypatch.setitem(os.environ, "NKI_SIMULATOR", "0")
    assert can_run_kernel(torch.zeros(1)) is False, (
        "the gate did not flip with NKI_SIMULATOR=0, so this control is unarmed"
    )

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        out = blockwise_fp8_mm(case["x"], case["weight"], case["weight_scale"])
    nki_dispatch, torch_fallback = dispatch_counters()
    print(
        f"[route-control] nki_dispatch={nki_dispatch} "
        f"torch_fallback={torch_fallback} simulate_kernel_calls={sim.calls}"
    )
    assert nki_dispatch == 0, f"expected 0 NKI dispatches, got {nki_dispatch}"
    assert torch_fallback == 1, f"expected 1 torch fallback, got {torch_fallback}"
    assert sim.calls == 0, f"the simulator ran {sim.calls} times with it disabled"
    assert tuple(out.shape) == (M, N)


def test_route_control_simulator_is_load_bearing() -> None:
    """The NKI chain RAISES without the simulator rather than computing torch.

    Recorded because it is what forecloses the F1 false green BELOW this
    repository's seam: if the HOP silently degraded to a torch path, a green
    numeric comparison could not be attributed to a kernel at all, and no
    counter of this module's could detect it.
    """
    case = _build_case()
    saved = os.environ.get("NKI_SIMULATOR")
    os.environ["NKI_SIMULATOR"] = "0"
    try:
        from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

        from vllm_neuron.functional.blockwise_fp8_mm import blockwise_fp8_mm_kernel

        scale_t = to_kernel_scale_layout(case["weight_scale"], K, N)
        with pytest.raises(RuntimeError) as excinfo:
            wrap_nki(blockwise_fp8_mm_kernel)(
                x=case["x"], weight=case["weight"], weight_scale_t=scale_t
            )
    finally:
        if saved is None:
            os.environ.pop("NKI_SIMULATOR", None)
        else:
            os.environ["NKI_SIMULATOR"] = saved

    message = str(excinfo.value)
    print(f"[route-control] simulator_off_raise={message[:160]!r}")
    assert "simulator" in message.lower(), message


# --------------------------------------------------------------------------- #
# The counters are MODULE-LEVEL state -- `inc-glm53f-033` depends on it.        #
# --------------------------------------------------------------------------- #
def test_dispatch_counters_are_module_level_state_reachable_from_elsewhere() -> None:
    """Another module can zero and read these counters. `-033` needs exactly this.

    `inc-glm53f-033`'s route predicate is form R-2 over THIS seam: its own test
    module resets and reads the counters this module owns. A test-local counter,
    or one only this file could reset, would pass this increment and break that
    one.

    Measured rather than asserted by inspection: the module is re-acquired
    through ``importlib`` -- the same mechanism another test module's import
    uses -- one reference RESETS, the seam is driven, and the OTHER reference
    READS. Then the reset is shown to be load-bearing, so "the counter reads 1"
    cannot be a counter that never moves.
    """
    foreign = importlib.import_module(_MODULE)
    assert foreign is sys.modules[_MODULE]
    # Identity, not just equality: both references address one counter object.
    assert foreign.dispatch_counters is dispatch_counters
    assert foreign.reset_dispatch_counters is reset_dispatch_counters

    case = _build_case(uniform_one=True)
    foreign.reset_dispatch_counters()
    assert dispatch_counters() == (0, 0), (
        "a reset through the foreign reference did not zero the counters this "
        "test reads, so the state is not shared and -033's R-2 predicate cannot "
        "be taken over this seam"
    )
    with _SimulatorCounter() as sim:
        foreign.blockwise_fp8_mm(case["x"], case["weight"], case["weight_scale"])
    after_one = dispatch_counters()
    with _SimulatorCounter() as sim2:
        foreign.blockwise_fp8_mm(case["x"], case["weight"], case["weight_scale"])
    after_two = foreign.dispatch_counters()
    print(
        f"[cross-module] after_reset=(0, 0) after_one_call={after_one} "
        f"after_two_calls={after_two} sim_calls={sim.calls + sim2.calls}"
    )
    assert after_one == (1, 0), f"expected (1, 0) after one dispatch, got {after_one}"
    assert after_two == (2, 0), (
        f"expected (2, 0) after two dispatches, got {after_two}; the counter "
        f"does not accumulate across calls, so it cannot count a caller's "
        f"dispatches"
    )
    foreign.reset_dispatch_counters()
    assert dispatch_counters() == (0, 0)


# --------------------------------------------------------------------------- #
# Seam identity, geometry refusals and the bridge.                              #
# --------------------------------------------------------------------------- #
def test_seam_dispatches_to_the_kernel_this_increment_authors() -> None:
    """The seam dispatches to THIS module's kernel, read off the object.

    SCRATCH is checkable here: the kernel's module is this repository's, not
    ``nkilib``'s, which is the difference between this increment and
    `inc-glm53f-025`.
    """
    module, qualname = kernel_identity()
    print(f"[identity] kernel={module}.{qualname}")
    assert module == _MODULE, module
    assert qualname == "blockwise_fp8_mm_kernel", qualname
    assert not module.startswith("nkilib"), (
        "the seam dispatches to a vendor kernel, but this increment is SCRATCH: "
        "G1 found no blockwise member in the substrate's QuantizationType"
    )


@pytest.mark.parametrize(
    "tokens,rows,cols,needle",
    [
        (200, 512, 512, "M=200 is not a positive multiple of TILE_SIZE"),
        (256, 384, 512, "K=384 is not a positive multiple of"),
        (256, 512, 300, "N=300 is not a positive multiple of"),
        (0, 512, 512, "M=0 is not a positive multiple of TILE_SIZE"),
    ],
)
def test_refuses_inadmissible_geometry_by_name(
    tokens: int, rows: int, cols: int, needle: str
) -> None:
    """Every refusal is a NAMED error carrying the offending extent."""
    with pytest.raises(BlockwiseFp8MmError) as excinfo:
        can_run_blockwise_fp8_mm(torch.zeros(1), rows, cols, tokens)
    message = str(excinfo.value)
    assert needle in message, f"[M={tokens},K={rows},N={cols}] message was: {message}"


def test_to_kernel_scale_layout_refuses_and_conserves() -> None:
    """The bridge refuses a mis-sized grid and invents no slot.

    A mis-sized grid is the defect that cannot be caught downstream: it can
    flatten onto a different block-to-scale assignment with no error at all.
    """
    good = torch.ones(scale_grid_shape(K, N), dtype=torch.float32)
    bridged = to_kernel_scale_layout(good, K, N)
    print(
        f"[bridge] grid={tuple(good.shape)} operand={tuple(bridged.shape)} "
        f"expected={kernel_scale_shape(K, N)}"
    )
    assert tuple(bridged.shape) == kernel_scale_shape(K, N)
    assert bridged.shape[0] == TILE_SIZE
    assert bridged.shape[1] == K_BLOCKS * N_BLOCKS

    # Each column is one block's scale, replicated down the partition axis.
    distinct = torch.arange(
        K_BLOCKS * N_BLOCKS, dtype=torch.float32
    ).reshape(K_BLOCKS, N_BLOCKS) + 1.0
    operand = to_kernel_scale_layout(distinct, K, N)
    for k_block in range(K_BLOCKS):
        for n_block in range(N_BLOCKS):
            column = operand[:, flat_scale_index(k_block, n_block, N_BLOCKS)]
            assert torch.equal(column, column[0].expand(TILE_SIZE)), (
                "the operand column is not a constant replication, so "
                "tensor_scalar would broadcast a non-uniform scale"
            )
            assert float(column[0]) == float(distinct[k_block, n_block])

    with pytest.raises(BlockwiseFp8MmError) as excinfo:
        to_kernel_scale_layout(torch.ones((K_BLOCKS, N_BLOCKS + 1)), K, N)
    assert "mis-sized" in str(excinfo.value)

    with pytest.raises(BlockwiseFp8MmError):
        to_kernel_scale_layout(good.to(torch.bfloat16), K, N)
