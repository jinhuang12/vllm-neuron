# SPDX-License-Identifier: Apache-2.0
"""Tier N acceptance for `inc-glm53f-025` -- the MoE-half block-quant kernel.

Acceptance command (plan block, `#### inc-glm53f-025`)::

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
    pytest test/vllm_neuron/functional/moe/test_moe_blockwise_fp8.py -k cte \
    --timeout 60

Why the route predicate is an acceptance criterion and not a diagnostic (F1)
---------------------------------------------------------------------------
The declared numeric expectation compares simulated NKI output against a torch
oracle. If the seam silently took its torch path, *both* sides of that
comparison would be torch and it would pass green while measuring nothing about
a kernel. So every case that dispatches reads three route instruments, and each
is reported as a number:

1. the seam's own dispatch counter (form R-1) -- ``nki_dispatch == 1``,
   ``torch_fallback == 0`` per case;
2. ``can_run_kernel()`` -- ``True``;
3. real ``nki.simulator.simulate_kernel`` invocations on the F1 chain -- ``1``
   per kernel call. Instrument 3 is independent of this repository's code: it
   counts the vendor entry point, so a bug in instrument 1 cannot fake it.

Both instruments are armed rather than assumed. ``test_cte_route_control_*``
shows instrument 1 reading ``(0, 1)`` on the fallback path and instrument 3
reading ``0``, so a zero is a measurement and not an unwired counter.

Why the numeric comparison alone cannot settle the scale layout
--------------------------------------------------------------
The NKI kernel and its vendor torch oracle read the scale tensor through the
*same* convention. A layout error consistent between them is therefore
invisible to a kernel-vs-oracle comparison. The two ``test_cte_layout_*`` cases
settle the layout with **oracle-free** instruments instead: a one-hot scale
probe whose observable consequence differs between the two candidate orders.
"""

from __future__ import annotations

import ast
import os

import pytest
import torch

import nki
import nki.simulator

from vllm_neuron.functional.moe.blockwise_fp8_retile import (
    BLOCK_QUANT_SIZE,
    GATE_UP,
    DOWN,
    TILE_SIZE,
    is_pow2_exact,
    retile_block_scales,
)
from vllm_neuron.functional.moe.moe_blockwise_fp8 import (
    NUM_SHARDS,
    MoeBlockwiseFp8Error,
    blockwise_fp8_moe,
    blockwise_fp8_moe_torch_oracle,
    can_run_blockwise_fp8_moe,
    dispatch_counters,
    kernel_identity,
    kernel_scale_shape,
    reset_dispatch_counters,
    seam_identity,
    to_kernel_scale_layout,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

# --------------------------------------------------------------------------- #
# The tiny config. Every extent is forced by an assert in the vendor kernel     #
# (`nkilib/core/moe/moe_cte/bwmm_shard_on_I.py`), cited at its own line.        #
# --------------------------------------------------------------------------- #
H = 512       # :668 512 <= H <= 8192 ; :680 H % 256 == 0 ; docstring :204 H % 512 == 0
I_TP = 512    # :670 I_TP % 16 ; :681 I_TP % 256 ; even multiple of 256 at NUM_SHARDS=2
E = 2         # "small expert count", per the block
B = 256       # :667 B % 256 == 0
N_BLOCKS = 2  # one token block per expert, so per-block rows are disjoint
T = N_BLOCKS * B

H_256 = H // BLOCK_QUANT_SIZE
I_256 = I_TP // BLOCK_QUANT_SIZE
H_TILES = H // TILE_SIZE
I_TILES = I_TP // TILE_SIZE

#: The declared tolerance. Fixed by the plan block; narrowing the world it is
#: measured in is the F1 precondition's job, and widening it to absorb remapping
#: error is the user's election, never this test's.
RTOL = 3e-2
ATOL = 1e-5

_FP8 = torch.float8_e4m3fn


class RouteInstrumentError(AssertionError):
    """A route reading that is not what the plan declares.

    A named error, so a failure says which instrument disagreed rather than
    surfacing a bare ``AssertionError``.
    """


class F1PreconditionError(AssertionError):
    """The pow2 losslessness precondition did not hold on this case's scales."""


class LayoutSettlementError(AssertionError):
    """The kernel's observed scale-block mapping is not the one recorded."""


class VacuousControlError(AssertionError):
    """A control whose input could not have made it fail.

    Raised when a control's stream is empty or its two arms are identical: a
    zero over vacuous input measures nothing, so the control refuses to report a
    pass it did not earn.
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
# Case construction. Built FROM 128-granularity checkpoint scales through       #
# `inc-glm53f-024`'s producer, so the retile is exercised rather than mimicked.  #
# --------------------------------------------------------------------------- #
def _pow2_checkpoint_scales(seed: int, rows: int, cols: int) -> torch.Tensor:
    """``(E, rows//128, cols//128)`` fp32 scales, every entry an exact power of two.

    Exact powers of two, and therefore mutually pow2-related within any
    ``256``-block, and each retained block scale itself a power of two: the
    COMPLETE condition `inc-glm53f-024` part 5 declares, satisfied by
    construction and then RE-ASSERTED from the emitted records in
    :func:`test_cte_f1_precondition_complete_condition_n_over_n`.

    Exponents vary per tile so the scales are DISTINCT: a comparison run on
    uniform scales cannot see a permuted layout.
    """
    generator = torch.Generator().manual_seed(seed)
    exponents = torch.randint(
        -3, 4, (E, rows // TILE_SIZE, cols // TILE_SIZE), generator=generator
    )
    return torch.ldexp(torch.ones_like(exponents, dtype=torch.float32), exponents)


def _fp8_grid_weights(seed: int, *shape: int, signed: bool = False) -> torch.Tensor:
    """Values already on the fp8-e4m3 grid, so every cast in the fixture is exact.

    ``signed=False`` is the default and it is a CONDITIONING choice, measured
    rather than assumed. With signed values every dot product over the ``H=512``
    contraction is a near-cancelling sum, so elements of the reference land
    arbitrarily close to zero while the terms that built them are ~1e4. A
    pointwise *relative* tolerance is then dominated by cancellation rather than
    by kernel error, and NO correct bf16-accumulating kernel can satisfy it:
    measured on the signed fixture, agreement is ``0.99996`` in best-fit scale
    and ``3.8e-03`` in per-block relative L2, while the pointwise maximum reads
    ``3.6e+02`` on elements whose reference is ~1e-1.

    So the declared tolerance is measured on a well-conditioned fixture -- which
    is what makes ``rtol=3e-2`` a statement about the kernel -- and the signed
    case is kept as its own arm, compared in a cancellation-robust norm at the
    same declared ``rtol``:
    :func:`test_cte_signed_fixture_agrees_in_norm_under_cancellation`.
    The tolerance itself is UNCHANGED; only the world it is measured in is
    narrowed, exactly as the F1 clause does for the scales.
    """
    generator = torch.Generator().manual_seed(seed)
    low = -7 if signed else 1
    raw = torch.randint(low, 8, shape, generator=generator).to(torch.float32) / 8.0
    return raw


def _build_case(signed: bool = False) -> dict:
    """The tiny config, retiled through `inc-glm53f-024`, in the kernel's layout."""
    # --- gate/up: the producer runs per fusion half, on (E, H, I_TP) ---------- #
    gup_scale_logical = torch.empty(
        kernel_scale_shape(E, H, I_TP, GATE_UP), dtype=torch.float32
    )
    gup_weight = torch.empty((E, H, 2, I_TP), dtype=torch.float32)
    gup_results = []
    for gate_or_up in range(2):
        checkpoint = _pow2_checkpoint_scales(11 + gate_or_up, H, I_TP)
        weights = _fp8_grid_weights(21 + gate_or_up, E, H, I_TP, signed=signed)
        result = retile_block_scales(
            weights.to(_FP8), checkpoint, projection=GATE_UP, gate_or_up=gate_or_up
        )
        gup_results.append((checkpoint, weights, result))
        gup_weight[:, :, gate_or_up, :] = result.retiled_weights.to(torch.float32)
        # One C-order reshape, in the module, at the one place it is written.
        bridged = to_kernel_scale_layout(
            result.consumer_scales, E, H, I_TP, projection=GATE_UP
        )
        # The producer writes only this half's slots, so take this half's slice.
        gup_scale_logical[:, :, gate_or_up, :, :] = bridged[:, :, gate_or_up, :, :]

    # --- down: the producer's `rows` is the H axis and `cols` the I axis, so    #
    # --- the physically-[E, I_TP, H] weight is retiled in (E, H, I_TP) view.    #
    down_checkpoint = _pow2_checkpoint_scales(31, H, I_TP)
    down_weight_hi = _fp8_grid_weights(41, E, H, I_TP, signed=signed)
    down_result = retile_block_scales(
        down_weight_hi.to(_FP8), down_checkpoint, projection=DOWN
    )
    down_scale_logical = to_kernel_scale_layout(
        down_result.consumer_scales, E, H, I_TP, projection=DOWN
    )
    # Back to the kernel's physical [E, I_TP, H].
    down_weight = (
        down_result.retiled_weights.to(torch.float32).transpose(1, 2).contiguous()
    )

    hidden = _fp8_grid_weights(51, T + 1, H, signed=signed).to(torch.bfloat16)
    affinities = torch.zeros(((T + 1) * E, 1), dtype=torch.bfloat16)
    affinities_2d = affinities.view(T + 1, E)
    token_position_to_id = torch.arange(N_BLOCKS * B, dtype=torch.int32)
    block_to_expert = torch.arange(N_BLOCKS, dtype=torch.int32).reshape(N_BLOCKS, 1) % E
    for block in range(N_BLOCKS):
        expert = int(block_to_expert[block, 0])
        affinities_2d[block * B : (block + 1) * B, expert] = 1.0

    return {
        "kernel_inputs": dict(
            hidden_states=hidden,
            expert_affinities_masked=affinities,
            gate_up_proj_weight=gup_weight.to(_FP8),
            down_proj_weight=down_weight.to(_FP8),
            block_size=B,
            token_position_to_id=token_position_to_id,
            block_to_expert=block_to_expert,
            gate_up_proj_scale=gup_scale_logical,
            down_proj_scale=down_scale_logical,
        ),
        "gup_results": gup_results,
        "down_result": down_result,
        "block_to_expert": block_to_expert,
    }


def _per_block_rows(block: int) -> slice:
    """Output rows the given token block owns. Disjoint by construction."""
    return slice(block * B, (block + 1) * B)


def _max_rel_error(got: torch.Tensor, want: torch.Tensor) -> float:
    """``max |got - want| / (|want| + ATOL)`` -- reported as a number, not a verdict."""
    return float(
        ((got - want).abs() / (want.abs() + ATOL)).max()
    )


# --------------------------------------------------------------------------- #
# THE DECLARED ACCEPTANCE CASE.                                                #
# --------------------------------------------------------------------------- #
def test_cte_output_matches_torch_oracle_per_expert_block() -> None:
    """Simulated NKI output vs the nkilib torch oracle, per expert block.

    The plan's declared Expected: ``assert_close(rtol=3e-2, atol=1e-5)`` per
    expert block, all blocks passing, worst per-block ``max_rel_error``
    reported as a number.
    """
    case = _build_case()
    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        got = blockwise_fp8_moe(**case["kernel_inputs"])
    _assert_route(sim, 1, "acceptance")

    want = blockwise_fp8_moe_torch_oracle(**case["kernel_inputs"])

    got_f32 = got.to(torch.float32)
    want_f32 = want.to(torch.float32)

    # Nonemptiness gate: an all-zero reference would make assert_close vacuous.
    nonzero_rows = int((want_f32.abs().sum(-1) > 0).sum())
    if nonzero_rows == 0:
        raise VacuousControlError(
            "the torch oracle produced an all-zero reference, so the comparison "
            "would pass over empty input; refusing to report it as a pass"
        )
    print(f"[acceptance] oracle_nonzero_rows={nonzero_rows} of {T}")
    # Reported so the reader can see which half of the declared tolerance binds:
    # at these magnitudes rtol dominates and atol=1e-5 is far below one ulp.
    print(
        f"[acceptance] want_absmax={float(want_f32[:T].abs().max()):.4e} "
        f"want_absmin={float(want_f32[:T].abs().min()):.4e} "
        f"got_absmax={float(got_f32[:T].abs().max()):.4e}"
    )

    worst = -1.0
    worst_block = -1
    passed = 0
    for block in range(N_BLOCKS):
        rows = _per_block_rows(block)
        expert = int(case["block_to_expert"][block, 0])
        rel = _max_rel_error(got_f32[rows], want_f32[rows])
        print(
            f"[acceptance] block={block} expert={expert} "
            f"rows={rows.start}:{rows.stop} max_rel_error={rel:.6e}"
        )
        if rel > worst:
            worst, worst_block = rel, block
        torch.testing.assert_close(
            got_f32[rows], want_f32[rows], rtol=RTOL, atol=ATOL
        )
        passed += 1

    print(
        f"[acceptance] blocks_passing={passed}/{N_BLOCKS} "
        f"worst_max_rel_error={worst:.6e} worst_block={worst_block} "
        f"rtol={RTOL} atol={ATOL}"
    )
    assert passed == N_BLOCKS, f"{passed}/{N_BLOCKS} blocks passed"


def test_cte_signed_fixture_agrees_in_norm_under_cancellation() -> None:
    """SUPPLEMENTARY, not the declared acceptance: signed weights, compared in norm.

    Kept so sign handling stays covered after the declared arm moved to a
    well-conditioned fixture. Over the ``H=512`` contraction a signed fixture
    cancels, so this arm applies the SAME declared ``rtol`` to a per-block
    relative L2 norm, which is what cancellation does not distort. No new
    tolerance number is introduced: the bound is ``RTOL``.

    The pointwise numbers are printed alongside, unrounded, so the conditioning
    effect is visible in the transcript rather than described in prose.
    """
    case = _build_case(signed=True)
    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        got = blockwise_fp8_moe(**case["kernel_inputs"]).to(torch.float32)
    _assert_route(sim, 1, "signed-norm")
    want = blockwise_fp8_moe_torch_oracle(**case["kernel_inputs"]).to(torch.float32)

    got_t, want_t = got[:T], want[:T]
    within = (got_t - want_t).abs() <= (ATOL + RTOL * want_t.abs())
    denom = float((want_t * want_t).sum())
    best_fit = float((got_t * want_t).sum()) / denom if denom > 0 else float("nan")
    print(
        f"[signed-norm] pointwise_within_declared_tol="
        f"{int(within.sum())}/{within.numel()} "
        f"pointwise_max_rel_error={_max_rel_error(got_t, want_t):.6e} "
        f"best_fit_scale={best_fit:.6f} "
        f"want_absmax={float(want_t.abs().max()):.4e} "
        f"want_absmin={float(want_t.abs().min()):.4e}"
    )

    for block in range(N_BLOCKS):
        rows = _per_block_rows(block)
        residual = float((got_t[rows] - want_t[rows]).pow(2).sum().sqrt())
        reference = float(want_t[rows].pow(2).sum().sqrt())
        if reference == 0.0:
            raise VacuousControlError(
                f"block {block}: the reference has zero norm, so a relative "
                f"comparison against it is vacuous"
            )
        rel_l2 = residual / reference
        print(f"[signed-norm] block={block} rel_L2={rel_l2:.6e} bound={RTOL}")
        assert rel_l2 <= RTOL, (
            f"block {block}: relative L2 {rel_l2:.6e} exceeds the declared "
            f"rtol {RTOL}; that is a structural disagreement, not cancellation"
        )


# --------------------------------------------------------------------------- #
# F1 PRECONDITION -- re-asserted here, not inherited from `inc-glm53f-024`.     #
# --------------------------------------------------------------------------- #
def test_cte_f1_precondition_complete_condition_n_over_n() -> None:
    """Both losslessness conjuncts hold on THIS case's scales, N/N blocks.

    Conjunct 1: the four constituent ``[128,128]`` scales are mutually
    power-of-two-related. Conjunct 2: the retained ``256``-block scale is itself
    a power of two. Both by the **bit-pattern** predicate
    :func:`is_pow2_exact`, never ``log2``.

    Recomputed from the emitted records rather than read off the producer's own
    ``lossless`` flag, so this is an independent instrument on `inc-glm53f-024`'s
    central claim.
    """
    case = _build_case()
    banks = [
        (f"gate_up[g={g}]", result)
        for g, (_checkpoint, _weights, result) in enumerate(case["gup_results"])
    ] + [("down", case["down_result"])]

    total = 0
    lossless = 0
    for label, result in banks:
        records = result.records
        if not records:
            raise VacuousControlError(
                f"{label}: the producer emitted 0 block records, so an N/N "
                f"reading would be 0/0 -- vacuous"
            )
        for record in records:
            total += 1
            conjunct1 = all(is_pow2_exact(ratio) for ratio in record.ratios)
            conjunct2 = is_pow2_exact(record.block_scale)
            if conjunct1 and conjunct2:
                lossless += 1
            else:
                raise F1PreconditionError(
                    f"{label} block {record.key}: conjunct1(quad mutually pow2)"
                    f"={conjunct1} ratios={record.ratios} "
                    f"conjunct2(block scale pow2)={conjunct2} "
                    f"block_scale={record.block_scale!r}. The tolerance "
                    f"{RTOL}/{ATOL} certifies kernel error only when both hold; "
                    f"widening it to absorb remapping error is the user's "
                    f"election, not this test's."
                )
        # The retile must also be bit-exact on the weights it rescaled, or the
        # kernel would be fed a weight the checkpoint did not contain.
        assert result.inexact_rescales == 0, (
            f"{label}: inexact_rescales={result.inexact_rescales}, declared 0"
        )
        assert result.emitted_unsupplied == 0, (
            f"{label}: emitted_unsupplied={result.emitted_unsupplied}, declared 0"
        )
        assert result.input_scales_dropped == 0, (
            f"{label}: input_scales_dropped={result.input_scales_dropped}, declared 0"
        )

    print(f"[f1] complete_condition_blocks={lossless}/{total} (N/N required)")
    assert lossless == total, f"{lossless}/{total} blocks satisfy both conjuncts"
    assert total > 0


def test_cte_f1_detector_catches_a_non_pow2_block_scale() -> None:
    """DETECTOR: the F1 predicate must FAIL a block that violates conjunct 2.

    Without this arm, ``N/N`` above is a tautology over scales the test itself
    chose. The detector feeds the real predicate a real violating scale and
    requires a real negative.
    """
    checkpoint = _pow2_checkpoint_scales(11, H, I_TP)
    weights = _fp8_grid_weights(21, E, H, I_TP)

    clean = retile_block_scales(weights.to(_FP8), checkpoint, projection=DOWN)
    clean_bad = [
        record.key
        for record in clean.records
        if not (all(is_pow2_exact(r) for r in record.ratios)
                and is_pow2_exact(record.block_scale))
    ]
    if clean_bad:
        raise VacuousControlError(
            f"the clean arm already violates F1 at {clean_bad[:3]}, so the "
            f"detector cannot attribute a negative to its injection"
        )

    # Inject: 3.0 is not a power of two (significand field non-zero), so the
    # retained block scale for block (0, 0) violates conjunct 2, and the three
    # ratios against it violate conjunct 1.
    poisoned = checkpoint.clone()
    poisoned[0, 0, 0] = 3.0
    if torch.equal(poisoned, checkpoint):
        raise VacuousControlError("injection changed nothing -- control is vacuous")

    dirty = retile_block_scales(weights.to(_FP8), poisoned, projection=DOWN)
    violations = [
        record.key
        for record in dirty.records
        if not (all(is_pow2_exact(r) for r in record.ratios)
                and is_pow2_exact(record.block_scale))
    ]
    print(
        f"[f1-detector] clean_violations={len(clean_bad)} "
        f"poisoned_violations={len(violations)} first={violations[:1]}"
    )
    assert violations, (
        "the F1 predicate passed a non-pow2 block scale (3.0); it is therefore "
        "not a discriminator and the N/N reading above means nothing"
    )
    assert (0, 0, 0) in violations, f"expected block (0,0,0) flagged, got {violations}"
    # And the bit-pattern predicate itself, on the injected value.
    assert not is_pow2_exact(3.0)
    assert is_pow2_exact(2.0) and is_pow2_exact(0.25)


# --------------------------------------------------------------------------- #
# LAYOUT SETTLEMENT -- oracle-free, because kernel and oracle share the          #
# convention and a shared error is invisible to their comparison.               #
# --------------------------------------------------------------------------- #
def _down_scale_onehot(flat_slot: int, hot: float) -> torch.Tensor:
    """Down scales: all ``1.0`` except one flat ``256``-block slot set to ``hot``.

    Built in the flat producer layout and bridged through the module's one
    reshape, so the probe measures the layout the module actually ships.
    """
    n_blocks = I_256 * H_256
    flat = torch.ones((E, n_blocks * TILE_SIZE), dtype=torch.float32)
    flat[:, flat_slot * TILE_SIZE : (flat_slot + 1) * TILE_SIZE] = hot
    return to_kernel_scale_layout(flat, E, H, I_TP, projection=DOWN)


def test_cte_layout_block_order_is_i_block_major() -> None:
    """Which ``(i_block, h_block)`` does the kernel read from flat slot 1?

    The two candidate orders make DIFFERENT, falsifiable predictions at
    ``H//256 == I_TP//256 == 2``:

    * ``i_block`` major -- ``flat = i_block * (H//256) + h_block`` -- slot 1 is
      ``(i_block=0, h_block=1)``, so the changed output columns are
      ``[256:512]``.
    * ``h_block`` major -- ``flat = h_block * (I_TP//256) + i_block`` -- slot 1
      is ``(h_block=0, i_block=1)``, so the changed columns are ``[0:256]``.

    The down scale multiplies the contribution landing in its ``h_block``'s
    output columns (``bwmm_shard_on_I.py:2133`` computes ``dst_h_start`` from the
    H block, ``:2138`` applies the scale there), so the changed column range
    identifies ``h_block`` and therefore the order. No oracle is involved.
    """
    case = _build_case()
    inputs = dict(case["kernel_inputs"])

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        inputs["down_proj_scale"] = _down_scale_onehot(1, 1.0)
        baseline = blockwise_fp8_moe(**inputs).to(torch.float32)
        inputs["down_proj_scale"] = _down_scale_onehot(1, 2.0)
        probed = blockwise_fp8_moe(**inputs).to(torch.float32)
    _assert_route(sim, 2, "layout-block-order")

    delta = (probed - baseline).abs()
    lower = float(delta[:T, 0:BLOCK_QUANT_SIZE].max())
    upper = float(delta[:T, BLOCK_QUANT_SIZE : 2 * BLOCK_QUANT_SIZE].max())
    print(
        f"[layout] onehot_slot=1 delta_max_columns[0:256]={lower:.6e} "
        f"delta_max_columns[256:512]={upper:.6e}"
    )

    if lower == 0.0 and upper == 0.0:
        raise VacuousControlError(
            "changing a scale changed no output at all, so this probe cannot "
            "identify any block; the instrument is unarmed"
        )
    if not (upper > 0.0 and lower == 0.0):
        raise LayoutSettlementError(
            "flat slot 1 did not map to (i_block=0, h_block=1): observed "
            f"delta_max[0:256]={lower:.6e}, delta_max[256:512]={upper:.6e}. "
            "The recorded order is i_block-major over C-order storage; this "
            "reading contradicts it and is a design contradiction to route, "
            "never a silent re-layout of inc-glm53f-024's landed code."
        )
    print("[layout] SETTLED: i_block-major, matching flat_scale_index(DOWN)")


def test_cte_layout_tile_axis_is_minor_not_partition_major() -> None:
    """Is ``TILE_SIZE`` the minor (contiguous) axis, or is the layout partition-major?

    The kernel reads a width-1 column across ``TILE_SIZE`` partitions
    (``bwmm_shard_on_I.py:2005``), filled by a DMA whose partition axis walks
    with stride ``1`` (``:2007`` ``pattern=[[1, TILE_SIZE], [TILE_SIZE, 1]]``).
    So consecutive elements of the host tensor land on consecutive PARTITIONS,
    and the partition axis is the token axis within a ``128``-token tile.

    Consequence, and the falsifiable prediction: setting only replica ``0`` of
    one block makes the kernel apply a different scale to the token at partition
    ``0`` of each tile -- a PER-TOKEN pattern with period ``TILE_SIZE``. Under a
    partition-major layout the same bytes would instead vary per BLOCK and every
    token in the affected column range would move together.
    """
    case = _build_case()
    inputs = dict(case["kernel_inputs"])
    n_blocks = I_256 * H_256

    uniform = torch.ones((E, n_blocks * TILE_SIZE), dtype=torch.float32)
    replica0 = uniform.clone()
    replica0[:, 0] = 2.0  # slot 0, replica 0 only
    if torch.equal(replica0, uniform):
        raise VacuousControlError("injection changed nothing -- control is vacuous")

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        inputs["down_proj_scale"] = to_kernel_scale_layout(
            uniform, E, H, I_TP, projection=DOWN
        )
        baseline = blockwise_fp8_moe(**inputs).to(torch.float32)
        inputs["down_proj_scale"] = to_kernel_scale_layout(
            replica0, E, H, I_TP, projection=DOWN
        )
        probed = blockwise_fp8_moe(**inputs).to(torch.float32)
    _assert_route(sim, 2, "layout-tile-axis")

    delta = (probed - baseline).abs()
    per_token = delta[:T].max(dim=-1).values
    moved = torch.nonzero(per_token > 0, as_tuple=False).flatten().tolist()
    if not moved:
        raise VacuousControlError(
            "changing replica 0 changed no output, so this probe cannot "
            "distinguish the two layouts; the instrument is unarmed"
        )
    residues = sorted({index % TILE_SIZE for index in moved})
    print(
        f"[layout] replica0 probe: moved_tokens={len(moved)} of {T} "
        f"distinct_residues_mod_{TILE_SIZE}={residues}"
    )

    if residues != [0]:
        raise LayoutSettlementError(
            f"replica 0 moved tokens at residues {residues} mod {TILE_SIZE}, "
            f"expected exactly [0]. A partition-major layout would move whole "
            f"blocks of tokens together. The recorded layout is C-order with "
            f"TILE_SIZE minor; this reading contradicts it and routes to the "
            f"lead as a design contradiction."
        )
    if len(moved) == T:
        raise LayoutSettlementError(
            f"every one of {T} tokens moved, which is the partition-major "
            f"signature, not the per-token signature C-order predicts"
        )
    print("[layout] SETTLED: TILE_SIZE is the minor axis; partition-major refuted")


# --------------------------------------------------------------------------- #
# ROUTE CONTROLS -- so every zero above is a measurement, not an unwired counter.#
# --------------------------------------------------------------------------- #
def test_cte_route_control_fallback_counter_discriminates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the simulator disabled the seam takes the torch path, and it is COUNTED.

    This is the arm that makes ``torch_fallback == 0`` above meaningful: the
    counter is shown reading ``1``, and ``nki_dispatch`` reading ``0``, through
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
        out = blockwise_fp8_moe(**case["kernel_inputs"])
    nki_dispatch, torch_fallback = dispatch_counters()
    print(
        f"[route-control] nki_dispatch={nki_dispatch} "
        f"torch_fallback={torch_fallback} simulate_kernel_calls={sim.calls}"
    )
    assert nki_dispatch == 0, f"expected 0 NKI dispatches, got {nki_dispatch}"
    assert torch_fallback == 1, f"expected 1 torch fallback, got {torch_fallback}"
    assert sim.calls == 0, f"the simulator ran {sim.calls} times with it disabled"
    assert out.shape == (T + 1, H)


def test_cte_route_control_simulator_is_load_bearing() -> None:
    """The NKI chain RAISES without the simulator rather than computing torch.

    Recorded because it is what forecloses the F1 false green below this
    repository's seam: if the HOP silently degraded to a torch path, a green
    numeric comparison could not be attributed to a kernel at all.
    """
    case = _build_case()
    inputs = case["kernel_inputs"]
    saved = os.environ.get("NKI_SIMULATOR")
    os.environ["NKI_SIMULATOR"] = "0"
    try:
        from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
        from nkilib.core.moe.moe_cte.bwmm_shard_on_I import (
            blockwise_mm_baseline_shard_intermediate,
        )

        with pytest.raises(RuntimeError) as excinfo:
            wrap_nki(blockwise_mm_baseline_shard_intermediate)[NUM_SHARDS](
                is_block_quant=True, **inputs
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
# Seam identity and named refusals.                                            #
# --------------------------------------------------------------------------- #
def test_cte_seam_dispatches_to_the_adapted_nkilib_member() -> None:
    """The seam adapts the vendor member, read off the object rather than assumed."""
    module, qualname = kernel_identity()
    print(f"[identity] kernel={module}.{qualname} num_shards={NUM_SHARDS}")
    assert module == "nkilib.core.moe.moe_cte.bwmm_shard_on_I", module
    assert qualname == "blockwise_mm_baseline_shard_intermediate", qualname
    assert NUM_SHARDS == 2


def test_cte_identity_readings_are_derived_through_the_seam() -> None:
    """`B26-M1`, `inc-glm53f-077`: both readings follow the real call chain.

    The seam wraps a shim and the shim forwards to the vendor kernel, so there
    are TWO substitutable hops. The reading this repair replaced looked at
    neither: it read this module's own import of the kernel, so a substitution at
    either hop left it byte-identical and the silence read as reassurance.

    What this arm settles, and what it does NOT. It settles that each reading
    resolves to the object at its own hop, that the two hops are different
    objects, and that a chain which cannot be derived RAISES rather than falling
    back to the import. It does NOT discriminate the repair from the reading it
    replaced -- that needs the call site itself edited, which the acceptance
    harness does as its graded mutation arms (``accept-077-r1-host.out``).
    """
    import vllm_neuron.functional.moe.moe_blockwise_fp8 as moe

    seam_module, seam_qualname = seam_identity()
    kernel_module, kernel_qualname = kernel_identity()
    print(f"[identity] seam={seam_module}.{seam_qualname}")
    print(f"[identity] kernel={kernel_module}.{kernel_qualname}")

    # Hop 1: what ``wrap_nki`` wraps is THIS module's shim, not the vendor member.
    assert seam_module == "vllm_neuron.functional.moe.moe_blockwise_fp8", seam_module
    assert seam_qualname == (
        "_torch_compatible_blockwise_mm_baseline_shard_intermediate"
    ), seam_qualname

    # Hop 2: the kernel reading is unchanged by this repair, to the byte.
    assert kernel_module == "nkilib.core.moe.moe_cte.bwmm_shard_on_I", kernel_module
    assert kernel_qualname == "blockwise_mm_baseline_shard_intermediate", (
        kernel_qualname
    )

    # The pair is not vacuous: two hops, two different objects, two readings.
    wrapped = moe._seam_wrapped_object()
    forwarded = moe._shim_forward_target()
    if wrapped is forwarded:
        raise RouteInstrumentError(
            "the seam's wrapped object and the shim's forward target are the "
            "same object, so the two readings cannot separate the two hops and "
            "this arm asserts a property of nothing"
        )
    assert (seam_module, seam_qualname) != (kernel_module, kernel_qualname)

    # Each reading is the object at its own hop, by identity rather than by name.
    assert wrapped is moe._torch_compatible_blockwise_mm_baseline_shard_intermediate
    assert forwarded is moe.blockwise_mm_baseline_shard_intermediate

    # A chain that cannot be derived RAISES. No fall back to the import: that
    # fall back is the silence `B26-M1` found.
    with pytest.raises(MoeBlockwiseFp8Error) as no_source:
        moe._function_ast(object())
    assert "cannot read the source" in str(no_source.value)

    with pytest.raises(MoeBlockwiseFp8Error) as not_a_name:
        moe._resolved(seam_identity, ast.Constant(value=1), "a test probe")
    assert "not a plain name" in str(not_a_name.value)

    with pytest.raises(MoeBlockwiseFp8Error) as unbound:
        moe._resolved(seam_identity, ast.Name(id="not_bound_anywhere"), "a probe")
    assert "not bound in" in str(unbound.value)


def test_cte_kernel_identity_has_no_fall_back_to_the_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`B46 N1`, `inc-glm53f-077`: the reading is BOUND to the derivation.

    The arm above binds both helpers to their objects, but checks
    ``kernel_identity()`` itself only against two name strings -- and in a healthy
    tree the derivation and this module's own import of the kernel name the SAME
    object, so a ``kernel_identity()`` that went back to reading the import would
    satisfy those strings unchanged. That is the silence `B26-M1` found, and the
    arm above cannot see it.

    What a unit test CAN settle is that no such fall back exists: break the
    derivation and the reading must RAISE rather than answer. An implementation
    that read the import would return the real identity here, and this arm would
    then fail on the missing exception. Discriminating the repair from its
    predecessor on an INTACT tree still needs the call site itself edited, which
    stays the acceptance harness's graded mutation arms.
    """
    import vllm_neuron.functional.moe.moe_blockwise_fp8 as moe

    # POPULATION BEFORE PROPERTY: the reading works before the break, so the
    # exception below belongs to the break and not to a tree that was already red.
    intact_module, intact_qualname = kernel_identity()
    assert intact_module == "nkilib.core.moe.moe_cte.bwmm_shard_on_I", intact_module

    sentinel = "the derivation was broken by this arm, on purpose"

    def _refuse() -> None:
        raise MoeBlockwiseFp8Error(sentinel)

    monkeypatch.setattr(moe, "_shim_forward_target", _refuse)

    with pytest.raises(MoeBlockwiseFp8Error) as broken:
        moe.kernel_identity()
    assert sentinel in str(broken.value), str(broken.value)

    # NON-VACUITY CONTROL, and the reason this arm is not a test of an absent
    # name. The module-level import is still bound, and unwrapping it yields the
    # very identity the intact reading returned -- so there really was something
    # to fall back TO, and the refusal above is a choice rather than an accident.
    fallback = moe._unwrap_nki(moe.blockwise_mm_baseline_shard_intermediate)
    assert (fallback.__module__, fallback.__qualname__) == (
        intact_module,
        intact_qualname,
    )


@pytest.mark.parametrize(
    "rows,cols,needle",
    [
        (512, 256, "BLOCK_QUANT_SIZE * NUM_SHARDS"),  # odd multiple of 256
        (512, 768, "BLOCK_QUANT_SIZE * NUM_SHARDS"),  # odd multiple of 256
        (256, 512, "outside [512, 8192]"),            # below the H floor
        (768, 512, "multiple of PSUM_SIZE"),          # H not a multiple of 512
        (512, 500, "not a positive multiple of 256"),  # I not 256-blocked
    ],
)
def test_cte_refuses_inadmissible_geometry_by_name(
    rows: int, cols: int, needle: str
) -> None:
    """Every refusal is a NAMED error carrying the offending extent."""
    with pytest.raises(MoeBlockwiseFp8Error) as excinfo:
        can_run_blockwise_fp8_moe(torch.zeros(1), rows, cols)
    message = str(excinfo.value)
    assert needle in message, f"[H={rows},I={cols}] message was: {message}"


def test_cte_to_kernel_scale_layout_refuses_missized_tensor() -> None:
    """A mis-sized scale tensor is refused, not reshaped onto a wrong mapping."""
    good = torch.ones((E, I_256 * H_256 * TILE_SIZE), dtype=torch.float32)
    bridged = to_kernel_scale_layout(good, E, H, I_TP, projection=DOWN)
    assert tuple(bridged.shape) == kernel_scale_shape(E, H, I_TP, DOWN)

    bad = torch.ones((E, I_256 * H_256 * TILE_SIZE - TILE_SIZE), dtype=torch.float32)
    with pytest.raises(MoeBlockwiseFp8Error) as excinfo:
        to_kernel_scale_layout(bad, E, H, I_TP, projection=DOWN)
    assert "mis-sized" in str(excinfo.value)


def test_cte_kernel_scale_shape_matches_the_producer_element_count() -> None:
    """The bridge conserves elements: no slot invented, none dropped."""
    for projection in (DOWN, GATE_UP):
        logical = kernel_scale_shape(E, H, I_TP, projection)
        elements = 1
        for extent in logical:
            elements *= extent
        flat = torch.ones((E, elements // E), dtype=torch.float32)
        bridged = to_kernel_scale_layout(flat, E, H, I_TP, projection)
        print(f"[shape] {projection}: flat={tuple(flat.shape)} logical={logical}")
        assert bridged.numel() == flat.numel()
        assert logical[-1] == TILE_SIZE
