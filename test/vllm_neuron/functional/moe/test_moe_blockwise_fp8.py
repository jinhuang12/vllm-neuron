# SPDX-License-Identifier: Apache-2.0
"""Tier N acceptance for `inc-glm53f-025` -- the MoE-half block-quant kernel.

Acceptance command (plan block, `#### inc-glm53f-025`)::

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
    pytest test/vllm_neuron/functional/moe/test_moe_blockwise_fp8.py -k cte \
    --timeout 60

What this file covers, and what it stopped covering
---------------------------------------------------
Every item here is a ``cte_128`` item: it measures the three limbs the routed
bank actually calls -- ``moe_gate_up_blockwise_fp8``,
``moe_swiglu_transposed`` and ``moe_down_blockwise_fp8`` -- which index the
checkpoint's own ``[128, 128]`` scale grid.

The items that measured the vendor member at ``256`` granularity were retired
once no product path fed it: the routed bank calls the three limbs above and
nothing in this tree calls ``blockwise_fp8_moe``. Removing the vendor entry
point and its flat-scale adapter is recorded as a separate debt, not done here.

Why the route predicate is an acceptance criterion and not a diagnostic (F1)
---------------------------------------------------------------------------
The declared numeric expectation compares simulated NKI output against a torch
oracle. If a limb silently took a torch path, *both* sides of that comparison
would be torch and it would pass green while measuring nothing about a kernel.
So every case that dispatches reads three route instruments, and each is
reported as a number:

1. the limb's own dispatch counter (form R-1) -- ``nki_dispatch == 1``,
   ``torch_fallback == 0`` per case;
2. ``can_run_kernel()`` -- ``True``;
3. real ``nki.simulator.simulate_kernel`` invocations on the F1 chain -- ``1``
   per kernel call. Instrument 3 is independent of this repository's code: it
   counts the vendor entry point, so a bug in instrument 1 cannot fake it.

The zeros are armed rather than assumed:
``test_cte_128_route_control_the_gate_up_limb_has_no_torch_route`` shows what
the counters read when no kernel runs, so a zero is a measurement and not an
unwired counter.

Why the numeric comparison alone cannot settle the scale layout
--------------------------------------------------------------
A kernel and a torch oracle that read the scale tensor through the *same*
convention share any layout error, so the comparison cannot see it.
``test_cte_128_scale_operand_order_is_the_checkpoint_grid_order`` and
``test_cte_128_a_one_hot_scale_moves_only_its_own_output_columns`` settle the
order with **oracle-free** instruments instead: a one-hot scale probe whose
observable consequence differs between the two candidate orders.
"""

from __future__ import annotations

import ast
import math
import os

import pytest
import torch

import nki
import nki.isa as nisa
import nki.language as nl
import nki.simulator

from vllm_neuron.functional.moe.blockwise_fp8_retile import (
    BLOCK_QUANT_SIZE,
    GATE_UP,
    DOWN,
    TILE_SIZE,
)
from vllm_neuron.functional.moe.moe_blockwise_fp8 import (
    GATE_UP_FUSION,
    GATE_UP_SCALE_BLOCK,
    MoeBlockwiseFp8Error,
    can_run_blockwise_fp8_moe,
    can_run_moe_down_blockwise_fp8,
    can_run_moe_gate_up_blockwise_fp8,
    down_dispatch_counters,
    down_kernel_identity,
    gate_up_dispatch_counters,
    gate_up_flat_scale_index,
    gate_up_kernel_identity,
    gate_up_kernel_scale_shape,
    kernel_scale_shape,
    moe_down_blockwise_fp8,
    moe_gate_up_blockwise_fp8,
    moe_swiglu_transposed,
    reset_down_dispatch_counters,
    reset_gate_up_dispatch_counters,
    reset_swiglu_dispatch_counters,
    swiglu_dispatch_counters,
    swiglu_kernel_identity,
    to_down_kernel_scale_operand,
    to_gate_up_kernel_scale_operand,
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


def _max_rel_error(got: torch.Tensor, want: torch.Tensor) -> float:
    """``max |got - want| / (|want| + ATOL)`` -- reported as a number, not a verdict."""
    return float(
        ((got - want).abs() / (want.abs() + ATOL)).max()
    )


# --------------------------------------------------------------------------- #
# Seam identity and named refusals.                                            #
# --------------------------------------------------------------------------- #
def test_cte_128_no_limb_seam_reaches_the_vendor_member() -> None:
    """Every limb this campaign runs dispatches into THIS module, and none into nkilib.

    The positive half -- that each identity names the kernel authored here -- is
    settled by the two items further down this file. This one settles the NEGATIVE
    half, which those cannot: that no limb on the product path resolves into
    ``nkilib`` at all. Before the switch the block seam did, and its own docstring
    said so; an item that only checks the new names would pass just as well on a tree
    where one limb had slipped back.
    """
    readings = {
        "gate_up": gate_up_kernel_identity(),
        "swiglu": swiglu_kernel_identity(),
        "down": down_kernel_identity(),
    }
    for limb, (module, qualname) in readings.items():
        print(f"[identity] {limb}={module}.{qualname}")
    mine = "vllm_neuron.functional.moe.moe_blockwise_fp8"
    foreign = {
        limb: module for limb, (module, _) in readings.items() if module != mine
    }
    assert foreign == {}, (
        f"these limbs resolve outside this module: {foreign}. Every kernel the "
        f"routed MoE path runs is authored here"
    )
    assert len({qualname for _, qualname in readings.values()}) == len(readings), (
        f"two limbs resolved to the same kernel: {readings}. Three distinct limbs "
        f"must be three distinct objects, or one of these readings is not deriving "
        f"anything"
    )


def test_cte_128_limb_identities_are_derived_through_their_own_seams() -> None:
    """`B26-M1`'s property, on the limbs: each reading follows its own call chain.

    A reading that looked at this module's import of a kernel instead of at the seam
    that calls it is byte-identical on a healthy tree and blind to a substitution at
    the seam. So the property is checked the only way a unit test can: break ONE
    limb's derivation and that limb's reading must RAISE, while the other two are
    unmoved. An implementation reading the import would answer for all three.
    """
    import vllm_neuron.functional.moe.moe_blockwise_fp8 as moe

    # POPULATION BEFORE PROPERTY: all three answer before the break, so the
    # exception below belongs to the break and not to an already-red tree.
    intact = {
        "gate_up": gate_up_kernel_identity(),
        "swiglu": swiglu_kernel_identity(),
        "down": down_kernel_identity(),
    }
    sentinel = "the derivation was broken by this arm, on purpose"

    def _refuse(seam, what):
        raise MoeBlockwiseFp8Error(sentinel)

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(moe, "_wrapped_object_of", _refuse)
        for limb, reader in (
            ("gate_up", moe.gate_up_kernel_identity),
            ("swiglu", moe.swiglu_kernel_identity),
            ("down", moe.down_kernel_identity),
        ):
            with pytest.raises(MoeBlockwiseFp8Error) as broken:
                reader()
            assert sentinel in str(broken.value), f"[{limb}] {broken.value}"
    finally:
        monkeypatch.undo()

    # NON-VACUITY: there really was something to fall back TO. The module-level
    # kernel names are still bound and unwrapping them yields the same identities,
    # so the refusals above are a choice rather than an absence.
    for limb, kernel in (
        ("gate_up", moe.moe_gate_up_blockwise_fp8_kernel),
        ("swiglu", moe.moe_swiglu_transposed_kernel),
        ("down", moe.moe_down_blockwise_fp8_kernel),
    ):
        fallback = moe._unwrap_nki(kernel)
        assert (fallback.__module__, fallback.__qualname__) == intact[limb], limb


def test_cte_128_a_broken_limb_derivation_refuses_instead_of_answering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MEASURED: break the gate/up derivation and the reading RAISES, never falls back.

    The seam resolves its kernel through this module's own name at call time, so
    rebinding that name moves the call and the reading together: an arm asking the
    reading to survive a substitution would be asking for something the seam does not
    do. What a unit test can settle is that there is no fall back to the import --
    break the derivation and the reading must refuse -- and the control below shows
    the attribute it could have fallen back to was bound all along, so the refusal is
    a choice rather than an absent name.
    """
    import vllm_neuron.functional.moe.moe_blockwise_fp8 as moe

    intact = gate_up_kernel_identity()
    assert intact[0] == moe.__name__, intact

    sentinel = "the derivation was broken by this arm, on purpose"

    def _refuse(*_args, **_kwargs):
        raise MoeBlockwiseFp8Error(sentinel)

    monkeypatch.setattr(moe, "_wrapped_object_of", _refuse)
    with pytest.raises(MoeBlockwiseFp8Error) as broken:
        moe.gate_up_kernel_identity()
    assert sentinel in str(broken.value), str(broken.value)

    fallback = moe._unwrap_nki(moe.moe_gate_up_blockwise_fp8_kernel)
    reading = (fallback.__module__, fallback.__qualname__)
    print(f"[identity] intact={intact} available_fall_back={reading}")
    assert reading == intact, (
        f"the attribute this reading could have fallen back to names {reading}, not "
        f"{intact}, so the refusal above would be a missing name rather than a "
        f"refused fall back"
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


# ===========================================================================
# `inc-glm53f-113a` -- the campaign's OWN gate/up kernel, at [128, 128].
# ===========================================================================
#
# WHAT IS COMPARED, AND WHY IT IS AN EQUALITY RATHER THAN A TOLERANCE. The
# compared tensor is the fp32 PRE-ACTIVATION gate/up matmul output, per expert,
# against a reference derived from the model's own quantities: the checkpoint's
# fp8 weights dequantised by the checkpoint's own `128 x 128` scales. On the
# fixture below every value is exactly representable at every step -- the weights
# and activations sit on the fp8-e4m3 grid as `k/8`, and each scale is a
# two-mantissa-bit value times a power of two -- so the reference and the kernel
# must agree BIT FOR BIT and the expectation is `torch.equal` with
# `max_abs_diff == 0` printed as a number. That exactness is not assumed: every
# item that reads an equality prints the fp64-versus-fp32 precondition FIRST, and
# also prints that the flat and the block-wise reference forms agree, which is
# what makes the kernel's summation order (per 128-block, scaled, then added)
# irrelevant to the reading rather than merely convenient.
#
# WHY THE FAILING CONTROL COLLAPSES QUADS AND WHY EACH QUAD MUST MIX MANTISSA
# FAMILIES. The defect this increment removes is a lossy retile of the
# checkpoint's own grid onto `256` blocks. The control therefore applies that
# mapping -- each `2 x 2` quad of `128`-blocks takes the quad maximum -- inside
# this test file, never in the producer, and requires that the equality above
# CANNOT then be reached. `2.5 == 2 * 1.25` and `3.5 == 2 * 1.75`, so a quad drawn
# from one mantissa family retiles LOSSLESSLY and a control built from it would
# read `max_abs_diff == 0` while proving nothing. The grid builder mixes both
# families in every quad and the control MEASURES that it did before it reads its
# own result.
#
# EVERY ITEM BELOW CARRIES `cte_128` IN ITS NAME, which is what the plan's
# acceptance command selects (`-k cte_128`), and every reading is emitted on a
# tagged row (`E113|`) so a transcript reader can anchor on the rows this
# increment produced.

#: Extents. `B` tokens per expert block, `H` contraction, `I` per fusion half --
#: the same numbers the `-025` case above uses, so nothing here invents a shape.
G128_TOKENS = B
G128_H = H
G128_I = I_TP
G128_H_BLOCKS = G128_H // GATE_UP_SCALE_BLOCK
G128_I_BLOCKS = G128_I // GATE_UP_SCALE_BLOCK
G128_BLOCKS = G128_H_BLOCKS * GATE_UP_FUSION * G128_I_BLOCKS


def _one_block_routing(tokens: int) -> tuple:
    """``(row_index, expert_index, block)`` for one block of one expert.

    The identity row index and a single zero expert, which is the mapping a
    single-expert comparison implies. ``block`` is the whole token count, so the
    routed kernels see exactly one block and their block arithmetic reduces to the
    single-expert form these items were written against.
    """
    return (
        torch.arange(tokens, dtype=torch.int32).reshape(-1, 1),
        torch.zeros((1, 1), dtype=torch.int32),
        tokens,
    )


def _gate_up_one_block(hidden, fused_weight, scale_operand):
    """The routed gate/up seam at the single-expert, single-block shape."""
    tokens = int(hidden.shape[0])
    padded = torch.cat(
        [hidden, torch.zeros((1, int(hidden.shape[1])), dtype=hidden.dtype)]
    )
    return moe_gate_up_blockwise_fp8(
        padded,
        fused_weight.unsqueeze(0),
        scale_operand.unsqueeze(0),
        *_one_block_routing(tokens),
    )


def _affinity_bank(affinity):
    """One expert's ``[B, 1]`` affinities as the seam's ``[(T + 1) * E, 1]`` bank."""
    return torch.cat([affinity, torch.zeros((1, 1), dtype=affinity.dtype)])


def _down_one_block(intermediate_t, down_weight, scale_operand, affinity):
    """The routed down seam at the single-expert, single-block shape.

    The affinity arrives as ``[B, 1]`` for one expert and the bank the seam takes is
    ``[(T + 1) * E, 1]``; at ``E = 1`` that is this column with the padding token's
    zero appended, which is what the mapping itself would hand over.
    """
    tokens = int(intermediate_t.shape[1])
    bank = torch.cat([affinity, torch.zeros((1, 1), dtype=affinity.dtype)])
    return moe_down_blockwise_fp8(
        intermediate_t,
        down_weight.unsqueeze(0),
        scale_operand.unsqueeze(0),
        bank,
        *_one_block_routing(tokens),
        tokens,
    )

#: The four scale values, and the two mantissa families they fall into. Powers of
#: two multiply them per quad, which keeps every block scale distinct without
#: moving any value between families.
_G128_SCALE_VALUES = (1.25, 1.75, 2.5, 3.5)
_G128_FAMILY = {1.25: "A", 2.5: "A", 1.75: "B", 3.5: "B"}


class GateUpExactnessError(AssertionError):
    """The bit-exactness precondition did not hold on this fixture."""


def _emit_128(item: str, body: str) -> None:
    """One tagged reading per line, so the transcript can be anchored."""
    print(f"E113|{item}|{body}", flush=True)


def _mantissa_family_128(value: float) -> str:
    """``"A"`` for the ``1.25`` family, ``"B"`` for the ``1.75`` family.

    The significand is recovered by exact halving and doubling rather than by
    ``log2``, so the classification is a bit-level statement about the value.
    """
    scaled = float(value)
    while scaled >= 2.0:
        scaled /= 2.0
    while scaled < 1.0:
        scaled *= 2.0
    if scaled not in _G128_FAMILY:
        raise VacuousControlError(
            f"{value!r} has significand {scaled!r}, which is not one of "
            f"{sorted(_G128_FAMILY)}; the family classification is undefined and "
            f"the control cannot state that its quads mix families"
        )
    return _G128_FAMILY[scaled]


def _g128_scale_grid() -> torch.Tensor:
    """``[E, H//128, 2, I//128]`` fp32 -- the checkpoint's own grid, deterministic.

    Two properties are built in and both are measured where they are used: within
    every ``2 x 2`` quad the four scales come from BOTH mantissa families (so the
    lossy-``256`` control cannot pass for the wrong reason), and the power-of-two
    factor varies per quad (so every block scale is distinct and a permuted
    block-to-scale assignment cannot read as exact).
    """
    grid = torch.empty(
        (E, G128_H_BLOCKS, GATE_UP_FUSION, G128_I_BLOCKS), dtype=torch.float32
    )
    for expert in range(E):
        for h_block in range(G128_H_BLOCKS):
            for gate_or_up in range(GATE_UP_FUSION):
                for i_block in range(G128_I_BLOCKS):
                    value = _G128_SCALE_VALUES[
                        (h_block % 2) * 2 + (i_block % 2)
                    ]
                    quad = (h_block // 2) * (G128_I_BLOCKS // 2) + (i_block // 2)
                    exponent = ((quad + gate_or_up + expert) % 4) - 2
                    grid[expert, h_block, gate_or_up, i_block] = value * (
                        2.0**exponent
                    )
    return grid


def _g128_fp8_values(seed: int, *shape: int) -> torch.Tensor:
    """``k/8`` for ``k`` in ``1..7``: on the fp8-e4m3 grid, so every cast is exact.

    Unsigned, for the reason `inc-glm53f-025`'s own fixture records at ``:205``:
    over a 512-long contraction a signed fixture cancels, and this item reads an
    EQUALITY rather than a relative tolerance, so cancellation would make the
    reading fragile for a reason that has nothing to do with the kernel.
    """
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randint(1, 8, shape, generator=generator).to(torch.float32) / 8.0
    return raw


def _build_gate_up_128_case() -> dict:
    """One token block and one fused gate/up weight per expert, plus the operands."""
    grid = _g128_scale_grid()
    hidden = _g128_fp8_values(101, E, G128_TOKENS, G128_H).to(torch.bfloat16)
    # `[E, H, 2, I]` is the checkpoint's own layout; `[E, H, 2*I]` is its plain
    # reshape, which is what the kernel consumes -- so no copy stands between the
    # checkpoint and the operand.
    weight_fused = _g128_fp8_values(
        102, E, G128_H, GATE_UP_FUSION * G128_I
    ).to(_FP8)
    operands = [
        to_gate_up_kernel_scale_operand(grid[expert], G128_H, G128_I)
        for expert in range(E)
    ]
    return {
        "grid": grid,
        "hidden": hidden,
        "weight_fused": weight_fused,
        "operands": operands,
    }


def _g128_expanded_scales(grid: torch.Tensor, expert: int) -> torch.Tensor:
    """``[H, 2*I]`` -- one scale per element, expanded from the block grid."""
    per_block = grid[expert]                                    # [nh, 2, ni]
    expanded = per_block.repeat_interleave(GATE_UP_SCALE_BLOCK, dim=0)
    expanded = expanded.repeat_interleave(GATE_UP_SCALE_BLOCK, dim=2)
    return expanded.reshape(G128_H, GATE_UP_FUSION * G128_I)


def _g128_reference_flat(case: dict, expert: int) -> torch.Tensor:
    """The model-derived reference: dequantise every element, then one matmul."""
    weight = case["weight_fused"][expert].to(torch.float32)
    dequantised = weight * _g128_expanded_scales(case["grid"], expert)
    return case["hidden"][expert].to(torch.float32) @ dequantised


def _g128_reference_blockwise(case: dict, expert: int) -> torch.Tensor:
    """The same reference in the kernel's own order: per block, scale, then add.

    Written out so the summation order is a MEASURED non-issue rather than an
    assumption: the item below prints that this form and the flat form above are
    bit-equal, which is what lets a single equality certify a kernel that
    accumulates per block.
    """
    hidden = case["hidden"][expert].to(torch.float32)
    weight = case["weight_fused"][expert].to(torch.float32)
    accumulator = torch.zeros(
        (G128_TOKENS, GATE_UP_FUSION * G128_I), dtype=torch.float32
    )
    for h_block in range(G128_H_BLOCKS):
        rows = slice(
            h_block * GATE_UP_SCALE_BLOCK, (h_block + 1) * GATE_UP_SCALE_BLOCK
        )
        partial = hidden[:, rows] @ weight[rows, :]
        column_scales = (
            case["grid"][expert, h_block]
            .reshape(-1)
            .repeat_interleave(GATE_UP_SCALE_BLOCK)
        )
        accumulator += partial * column_scales
    return accumulator


def _g128_precondition(case: dict, expert: int) -> tuple[bool, float, bool]:
    """``(fp64_agrees, gap, forms_agree)`` -- printed BEFORE any equality is read."""
    weight64 = case["weight_fused"][expert].to(torch.float64)
    scales64 = _g128_expanded_scales(case["grid"], expert).to(torch.float64)
    reference64 = case["hidden"][expert].to(torch.float64) @ (weight64 * scales64)
    flat32 = _g128_reference_flat(case, expert)
    block32 = _g128_reference_blockwise(case, expert)
    gap = float((reference64 - flat32.to(torch.float64)).abs().max())
    return (
        bool(torch.equal(reference64, flat32.to(torch.float64))),
        gap,
        bool(torch.equal(flat32, block32)),
    )


def _assert_gate_up_route_128(
    sim: _SimulatorCounter, expected_dispatches: int, label: str
) -> str:
    """The three declared route values for THIS limb, each read as a number.

    ``torch_fallback`` can only read ``0`` because the limb has no torch
    projection route at all -- an inadmissible geometry raises (P13) -- so the
    zero is stated here rather than assumed, and the control below shows the gate
    itself flipping so the zero is armed.
    """
    nki_dispatch, torch_fallback = gate_up_dispatch_counters()
    gate = can_run_kernel(torch.zeros(1))
    reading = (
        f"gate_up_nki_dispatch={nki_dispatch} "
        f"gate_up_torch_fallback={torch_fallback} can_run_kernel={gate} "
        f"simulate_kernel_calls={sim.calls}"
    )
    _emit_128(label, reading)
    if nki_dispatch != expected_dispatches:
        raise RouteInstrumentError(
            f"{label}: the gate/up dispatch counter read {nki_dispatch}, declared "
            f"{expected_dispatches}. {reading}"
        )
    if torch_fallback != 0:
        raise RouteInstrumentError(
            f"{label}: the gate/up torch-fallback counter read {torch_fallback}, "
            f"declared exactly 0. {reading}"
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
# THE DECLARED ACCEPTANCE CASE for `inc-glm53f-113a`.                           #
# --------------------------------------------------------------------------- #
def test_cte_128_gate_up_matches_the_model_reference_per_expert_block() -> None:
    """Pre-activation gate/up output equals the model-derived reference, bit for bit.

    The plan's declared Expected for this increment: ``torch.equal`` on the fp32
    pre-activation result, ``N/N`` expert blocks, ``max_abs_diff == 0`` printed as
    a number, and the fp64-versus-fp32 exactness precondition printed first.
    """
    case = _build_gate_up_128_case()
    reset_gate_up_dispatch_counters()

    for expert in range(E):
        exact, gap, forms_agree = _g128_precondition(case, expert)
        _emit_128(
            "precondition",
            f"expert={expert} fp64_vs_fp32_bit_equal={int(exact)} "
            f"max_gap={gap:.6e} flat_form_equals_block_form={int(forms_agree)}",
        )
        if not exact:
            raise GateUpExactnessError(
                f"expert {expert}: the fp64 and fp32 references disagree on this "
                f"fixture (max gap {gap:.6e}), so an equality taken against the "
                f"fp32 reference would be a statement about rounding rather than "
                f"about the kernel"
            )
        if not forms_agree:
            raise GateUpExactnessError(
                f"expert {expert}: the flat and block-wise reference forms differ, "
                f"so the kernel's per-block summation order is not neutral on this "
                f"fixture and the equality below could not attribute a difference"
            )

    with _SimulatorCounter() as sim:
        outputs = [
            _gate_up_one_block(
                case["hidden"][expert],
                case["weight_fused"][expert],
                case["operands"][expert],
            )
            for expert in range(E)
        ]
    _assert_gate_up_route_128(sim, E, "acceptance")

    passed = 0
    worst = -1.0
    for expert in range(E):
        got = outputs[expert].to(torch.float32)
        want = _g128_reference_flat(case, expert)
        assert tuple(got.shape) == (G128_TOKENS, GATE_UP_FUSION * G128_I), (
            f"expert {expert}: kernel returned {tuple(got.shape)}, expected "
            f"{(G128_TOKENS, GATE_UP_FUSION * G128_I)}"
        )
        # A reference of all zeros would make an equality vacuous.
        if float(want.abs().max()) == 0.0:
            raise VacuousControlError(
                f"expert {expert}: the reference is all zero, so an equality "
                f"against it measures nothing"
            )
        max_abs_diff = float((got - want).abs().max())
        worst = max(worst, max_abs_diff)
        _emit_128(
            "equality",
            f"expert={expert} bit_equal={int(bool(torch.equal(got, want)))} "
            f"max_abs_diff={max_abs_diff} "
            f"want_absmax={float(want.abs().max()):.6e} "
            f"got_absmax={float(got.abs().max()):.6e}",
        )
        assert torch.equal(got, want), (
            f"expert {expert}: the kernel and the model-derived reference are not "
            f"bit-equal; max_abs_diff={max_abs_diff}"
        )
        assert max_abs_diff == 0
        passed += 1

    _emit_128(
        "verdict",
        f"expert_blocks_passing={passed}/{E} worst_max_abs_diff={worst}",
    )
    assert passed == E, f"{passed}/{E} expert blocks reached bit equality"


# --------------------------------------------------------------------------- #
# THE MUST-FAIL CONTROL: the lossy `256` retile cannot reach exactness.          #
# --------------------------------------------------------------------------- #
def _block_slice(tile: int) -> slice:
    """The weight rows, or columns, that one ``128`` scale block covers."""
    return slice(tile * GATE_UP_SCALE_BLOCK, (tile + 1) * GATE_UP_SCALE_BLOCK)


def _lossy_256_retile(
    weight_fp32: torch.Tensor, scale2d: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """A REAL ``256`` retile of one weight matrix: its bytes move with its scale.

    BOTH HALVES RUN, and that is the whole point. A retile keeps ONE scale per
    ``2 x 2`` quad of ``128`` blocks and re-expresses the other three tiles' WEIGHT
    BYTES against it, ``w * s_tile / s_retained``, cast back to fp8. Rewriting only
    the scale would corrupt the product by whole factors, and a control built that
    way passes on gross scale damage rather than on the retile's own loss -- so it
    would read a difference for every kernel and could never be unarmed.

    Where the loss lives: within a quad the four scales come from both mantissa
    families, so two of the four ratios are not powers of two, ``w * ratio`` leaves
    the fp8 grid and the cast rounds. That rounding IS the retile's loss and it is
    the only difference between this pair and the checkpoint's own.

    The quad MAXIMUM is retained, so every ratio is ``<= 1`` and no rescaled byte
    can overflow e4m3; an overflow would end the item before a number printed.

    Returns:
        The rescaled fp8 weight, the coarsened grid expressed on the ``128`` grid
        the kernel indexes, and how many tiles the fp8 cast could not express
        exactly.
    """
    retained = scale2d.clone()
    rescaled = weight_fp32.clone()
    inexact_tiles = 0
    n_row_blocks, n_col_blocks = scale2d.shape
    for row_quad in range(n_row_blocks // 2):
        for col_quad in range(n_col_blocks // 2):
            r0, c0 = row_quad * 2, col_quad * 2
            keep = float(scale2d[r0 : r0 + 2, c0 : c0 + 2].max())
            for d_row in range(2):
                for d_col in range(2):
                    r_tile, c_tile = r0 + d_row, c0 + d_col
                    ratio = float(scale2d[r_tile, c_tile]) / keep
                    rows, cols = _block_slice(r_tile), _block_slice(c_tile)
                    exact = weight_fp32[rows, cols] * ratio
                    as_fp8 = exact.to(_FP8)
                    if not torch.equal(as_fp8.to(torch.float32), exact):
                        inexact_tiles += 1
                    rescaled[rows, cols] = as_fp8.to(torch.float32)
                    retained[r_tile, c_tile] = keep
    return rescaled.to(_FP8), retained, inexact_tiles


def _lossy_256_gate_up(
    case: dict, expert: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """The retile applied to one expert's fused gate/up weight, a half at a time.

    The checkpoint grid holds a separate ``2 x 2`` neighbourhood per fused half, so
    the two halves retile independently and the fused weight is reassembled from
    them. Doing it half by half keeps the quads the grid's own rather than inventing
    a neighbourhood that straddles the fusion axis.
    """
    grid = case["grid"]
    weight = case["weight_fused"][expert].to(torch.float32)
    lossy_weight = weight.clone()
    lossy_grid = grid[expert].clone()
    inexact_tiles = 0
    for gate_or_up in range(GATE_UP_FUSION):
        half = slice(gate_or_up * G128_I, (gate_or_up + 1) * G128_I)
        bytes_fp8, retained, inexact = _lossy_256_retile(
            weight[:, half], grid[expert, :, gate_or_up, :]
        )
        lossy_weight[:, half] = bytes_fp8.to(torch.float32)
        lossy_grid[:, gate_or_up, :] = retained
        inexact_tiles += inexact
    return lossy_weight.to(_FP8), lossy_grid, inexact_tiles


def test_cte_128_a_lossy_256_retile_must_not_reach_exactness() -> None:
    """The `256` mapping must NOT reach ``max_abs_diff == 0``, and it must be armed.

    Three readings before the result, because a control that could not have failed
    proves nothing: every quad is shown to MIX both mantissa families (a
    single-family quad retiles losslessly, and then a zero here would be correct
    rather than a false green); the mapping is shown to have moved the grid AND the
    weight bytes, because a scale swap on unchanged bytes moves the output for any
    kernel that reads its scales and can never be unarmed; and at least one tile's
    rescaled bytes are shown to have left the fp8 grid, which is where the loss is.
    """
    case = _build_gate_up_128_case()
    grid = case["grid"]

    quads = 0
    for expert in range(E):
        for gate_or_up in range(GATE_UP_FUSION):
            for quad_h in range(G128_H_BLOCKS // 2):
                for quad_i in range(G128_I_BLOCKS // 2):
                    values = grid[
                        expert,
                        2 * quad_h : 2 * quad_h + 2,
                        gate_or_up,
                        2 * quad_i : 2 * quad_i + 2,
                    ].reshape(-1)
                    families = {_mantissa_family_128(float(v)) for v in values}
                    if families != {"A", "B"}:
                        raise VacuousControlError(
                            f"quad (expert={expert}, g={gate_or_up}, "
                            f"h={quad_h}, i={quad_i}) draws from families "
                            f"{sorted(families)} only, values "
                            f"{[float(v) for v in values]}. A single-family quad "
                            f"retiles losslessly, so this control could read "
                            f"max_abs_diff=0 for the wrong reason."
                        )
                    quads += 1
    _emit_128("control-arming", f"quads_mixing_both_families={quads}/{quads}")
    if quads == 0:
        raise VacuousControlError("no quads were examined, so nothing is armed")

    lossy_weight, lossy, inexact_tiles = _lossy_256_gate_up(case, 0)
    if torch.equal(lossy, grid[0]):
        raise VacuousControlError(
            "the lossy 256 mapping changed no scale, so the control cannot "
            "distinguish the two granularities"
        )
    if torch.equal(
        lossy_weight.to(torch.float32), case["weight_fused"][0].to(torch.float32)
    ):
        raise VacuousControlError(
            "the lossy 256 mapping changed no weight byte, so it is a scale swap "
            "and not a retile, and it would move the output for every kernel"
        )
    if inexact_tiles == 0:
        raise VacuousControlError(
            "every rescaled tile stayed exactly on the fp8 grid, so this mapping is "
            "a LOSSLESS retile and cannot show that the 128 grid buys anything"
        )
    changed = int((lossy != grid[0]).sum())
    _emit_128(
        "control-arming",
        f"scales_changed_by_the_256_mapping={changed} of {grid[0].numel()} "
        f"tiles_the_fp8_cast_rounded={inexact_tiles}",
    )

    reset_gate_up_dispatch_counters()
    with _SimulatorCounter() as sim:
        got = _gate_up_one_block(
            case["hidden"][0],
            lossy_weight,
            to_gate_up_kernel_scale_operand(lossy, G128_H, G128_I),
        ).to(torch.float32)
    _assert_gate_up_route_128(sim, 1, "control-route")

    want = _g128_reference_flat(case, 0)
    max_abs_diff = float((got - want).abs().max())
    _emit_128(
        "control",
        f"lossy_256_max_abs_diff={max_abs_diff} "
        f"reached_zero={int(max_abs_diff == 0)}",
    )
    assert max_abs_diff != 0, (
        "the lossy 256 retile reached bit equality against the 128-granular "
        "reference, so the declared equality does not discriminate the two "
        "granularities and this increment's acceptance would pass on the defect "
        "it exists to remove"
    )


# --------------------------------------------------------------------------- #
# ROUTE CONTROL: the zero above is armed, and there is no torch route to take.   #
# --------------------------------------------------------------------------- #
def test_cte_128_route_control_the_gate_up_limb_has_no_torch_route() -> None:
    """With the simulator off the gate reads False and the call RAISES.

    This is the arm that makes ``gate_up_torch_fallback == 0`` a measurement: the
    limb is shown to have no torch projection path to fall into, which is P13's
    requirement for kernel-class work, and the gate is shown to flip through the
    real environment read rather than through a mock.
    """
    case = _build_gate_up_128_case()
    saved = os.environ.get("NKI_SIMULATOR")
    os.environ["NKI_SIMULATOR"] = "0"
    reset_gate_up_dispatch_counters()
    try:
        gate = can_run_moe_gate_up_blockwise_fp8(
            torch.zeros(1), G128_TOKENS, G128_H, G128_I
        )
        assert gate is False, (
            f"the gate read {gate!r} with NKI_SIMULATOR=0, so this control is "
            f"unarmed"
        )
        # The exception TYPE is the substrate's to choose, so it is printed rather
        # than asserted: what this arm settles is that the call did not return a
        # tensor computed some other way.
        with pytest.raises(Exception) as excinfo:  # noqa: B017 - see above
            _gate_up_one_block(
                case["hidden"][0], case["weight_fused"][0], case["operands"][0]
            )
    finally:
        if saved is None:
            os.environ.pop("NKI_SIMULATOR", None)
        else:
            os.environ["NKI_SIMULATOR"] = saved

    message = str(excinfo.value)
    nki_dispatch, torch_fallback = gate_up_dispatch_counters()
    _emit_128(
        "route-control",
        f"gate_with_simulator_off={gate} "
        f"raised={type(excinfo.value).__name__}:{message[:80]!r} "
        f"names_the_simulator={int('simulator' in message.lower())} "
        f"gate_up_nki_dispatch={nki_dispatch} "
        f"gate_up_torch_fallback={torch_fallback}",
    )
    assert message, "the call raised without a message, so nothing can be read"
    # The counter counts ENTRIES into the NKI route, so the attempted dispatch is
    # counted even though it raised -- and the fallback counter stays 0, which is
    # the reading this arm exists to take: nothing computed torch instead.
    assert nki_dispatch == 1, nki_dispatch
    assert torch_fallback == 0, torch_fallback

    # AND THE SOURCE-LEVEL STATEMENT, which no environment can move: the seam
    # returns the result of exactly one call, and that call is the ``wrap_nki``
    # dispatch. A torch limb would have to be a second return.
    import vllm_neuron.functional.moe.moe_blockwise_fp8 as moe

    _fn, tree = moe._function_ast(moe_gate_up_blockwise_fp8)
    returns = [node for node in ast.walk(tree) if isinstance(node, ast.Return)]
    call_returns = [
        node for node in returns if isinstance(node.value, ast.Call)
    ]
    wrap_returns = [
        node
        for node in call_returns
        if isinstance(node.value.func, ast.Call)
        and isinstance(node.value.func.func, ast.Name)
        and node.value.func.func.id == "wrap_nki"
    ]
    _emit_128(
        "route-control",
        f"seam_returns={len(returns)} call_returns={len(call_returns)} "
        f"wrap_nki_returns={len(wrap_returns)}",
    )
    assert len(returns) == 1, (
        f"the gate/up seam has {len(returns)} return statements; a second one is "
        f"where a torch fallback would live, and this limb is declared to have "
        f"none (P13)"
    )
    assert len(wrap_returns) == 1


# --------------------------------------------------------------------------- #
# IDENTITY: the kernel under test is authored in this campaign.                  #
# --------------------------------------------------------------------------- #
def _seam_without_wrap_nki():
    """A probe seam that wraps nothing, so the derivation's guard can be armed."""
    return 1


def _seam_with_two_wrap_nki_calls():
    """A probe seam that wraps twice, so the ambiguity refusal can be armed."""
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    first = wrap_nki(moe_gate_up_blockwise_fp8_kernel_probe_a)
    second = wrap_nki(moe_gate_up_blockwise_fp8_kernel_probe_b)
    return first, second


def moe_gate_up_blockwise_fp8_kernel_probe_a() -> None:
    """Named target for the two-call probe above. Never traced."""


def moe_gate_up_blockwise_fp8_kernel_probe_b() -> None:
    """Named target for the two-call probe above. Never traced."""


def test_cte_128_kernel_identity_is_authored_in_this_campaign() -> None:
    """``gate_up_kernel_identity()`` names THIS module's kernel, derived through the seam.

    The reading is taken through the seam's own ``wrap_nki`` argument rather than
    off a module-level name, which is `B26-M1`'s finding: a reading taken off an
    import stays byte-identical when the seam is substituted. Three things are
    settled here -- the identity, that it is not the vendor member, and that a
    derivation which cannot be made RAISES instead of answering.
    """
    import vllm_neuron.functional.moe.moe_blockwise_fp8 as moe

    module, qualname = gate_up_kernel_identity()
    _emit_128("identity", f"gate_up_kernel={module}.{qualname}")
    assert module == "vllm_neuron.functional.moe.moe_blockwise_fp8", module
    assert qualname == "moe_gate_up_blockwise_fp8_kernel", qualname

    # By object identity, not by name: the derivation resolves the live binding.
    derived = moe._unwrap_nki(
        moe._wrapped_object_of(moe.moe_gate_up_blockwise_fp8, "the gate/up seam")
    )
    assert derived is moe._unwrap_nki(moe.moe_gate_up_blockwise_fp8_kernel)

    # NON-VACUITY: the vendor member is still bound in this module and its
    # identity DIFFERS, so this reading discriminates the two rather than
    # restating whatever it finds.
    vendor = moe._unwrap_nki(moe.blockwise_mm_baseline_shard_intermediate)
    _emit_128("identity", f"vendor_member={vendor.__module__}.{vendor.__qualname__}")
    assert (module, qualname) != (vendor.__module__, vendor.__qualname__)
    assert "nkilib" not in module

    # The guard is armed at both refusals.
    with pytest.raises(MoeBlockwiseFp8Error) as none_wrapped:
        moe._wrapped_object_of(_seam_without_wrap_nki, "a probe")
    assert "makes 0 `wrap_nki(...)` calls" in str(none_wrapped.value)

    with pytest.raises(MoeBlockwiseFp8Error) as two_wrapped:
        moe._wrapped_object_of(_seam_with_two_wrap_nki_calls, "a probe")
    assert "makes 2 `wrap_nki(...)` calls" in str(two_wrapped.value)

    with pytest.raises(MoeBlockwiseFp8Error) as no_source:
        moe._wrapped_object_of(object(), "a probe")
    assert "cannot read the source" in str(no_source.value)


# --------------------------------------------------------------------------- #
# NAMED REFUSALS -- no geometry is coerced and no fallback is shipped.           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "tokens,rows,cols,needle",
    [
        (200, 512, 512, "B=200 is not a positive multiple"),
        (0, 512, 512, "B=0 is not a positive multiple"),
        (256, 500, 512, "H=500 is not a positive multiple"),
        (256, 512, 500, "I=500 is not a positive multiple"),
    ],
)
def test_cte_128_refuses_inadmissible_geometry_by_name(
    tokens: int, rows: int, cols: int, needle: str
) -> None:
    """Every refusal names the offending extent and the loop that needs it."""
    with pytest.raises(MoeBlockwiseFp8Error) as excinfo:
        can_run_moe_gate_up_blockwise_fp8(torch.zeros(1), tokens, rows, cols)
    message = str(excinfo.value)
    _emit_128("refusal", f"B={tokens} H={rows} I={cols} message={message[:100]!r}")
    assert needle in message, f"[B={tokens},H={rows},I={cols}] message was: {message}"


def test_cte_128_seam_refuses_wrong_operands_by_name() -> None:
    """Rank, orientation, fusion width and operand shape are each refused by name.

    The orientation arm is the one that matters most: a ``[2*I, H]`` weight is the
    likely mistake, and reshaping it silently would compute a different function
    rather than fail.
    """
    case = _build_gate_up_128_case()
    hidden = case["hidden"][0]
    weight = case["weight_fused"][0]
    operand = case["operands"][0]

    with pytest.raises(MoeBlockwiseFp8Error) as rank:
        moe_gate_up_blockwise_fp8(
            hidden.unsqueeze(0),
            weight.unsqueeze(0),
            operand.unsqueeze(0),
            *_one_block_routing(G128_TOKENS),
        )
    assert "must be [T + 1, H]" in str(rank.value)

    # Built at the wrong shape rather than transposed or sliced from the fp8
    # fixture: a transpose-then-contiguous on fp8 would make this arm depend on a
    # cast path it is not about.
    transposed = torch.zeros(
        (GATE_UP_FUSION * G128_I, G128_H), dtype=torch.float32
    ).to(_FP8)
    with pytest.raises(MoeBlockwiseFp8Error) as orientation:
        moe_gate_up_blockwise_fp8(
            hidden,
            transposed.unsqueeze(0),
            operand.unsqueeze(0),
            *_one_block_routing(G128_TOKENS),
        )
    assert "contraction-major" in str(orientation.value)

    odd_width = torch.zeros(
        (G128_H, GATE_UP_FUSION * G128_I - 1), dtype=torch.float32
    ).to(_FP8)
    with pytest.raises(MoeBlockwiseFp8Error) as fusion:
        moe_gate_up_blockwise_fp8(
            hidden,
            odd_width.unsqueeze(0),
            operand.unsqueeze(0),
            *_one_block_routing(G128_TOKENS),
        )
    assert "GATE_UP_FUSION" in str(fusion.value)

    with pytest.raises(MoeBlockwiseFp8Error) as shape:
        moe_gate_up_blockwise_fp8(
            hidden,
            weight.unsqueeze(0),
            operand[:, :-1].contiguous().unsqueeze(0),
            *_one_block_routing(G128_TOKENS),
        )
    assert "to_gate_up_kernel_scale_operand" in str(shape.value)

    with pytest.raises(MoeBlockwiseFp8Error) as grid:
        to_gate_up_kernel_scale_operand(
            case["grid"][0][:, :, :-1].contiguous(), G128_H, G128_I
        )
    assert "mis-sized" in str(grid.value)
    _emit_128("refusal", "seam_refusals=5 all_named")


# --------------------------------------------------------------------------- #
# THE SCALE OPERAND -- host-side order, and then the kernel's own reading of it.  #
# --------------------------------------------------------------------------- #
def test_cte_128_scale_operand_order_is_the_checkpoint_grid_order() -> None:
    """Every block's scale sits in the column the flat index names, replicated.

    Oracle-free and kernel-free: this is a statement about the bridge alone, over
    every block of the declared geometry rather than over a sample.
    """
    grid = _g128_scale_grid()
    operand = to_gate_up_kernel_scale_operand(grid[0], G128_H, G128_I)
    assert tuple(operand.shape) == gate_up_kernel_scale_shape(G128_H, G128_I)
    assert tuple(operand.shape) == (TILE_SIZE, G128_BLOCKS)

    checked = 0
    for h_block in range(G128_H_BLOCKS):
        for gate_or_up in range(GATE_UP_FUSION):
            for i_block in range(G128_I_BLOCKS):
                column = gate_up_flat_scale_index(
                    h_block, gate_or_up, i_block, G128_I_BLOCKS
                )
                want = float(grid[0, h_block, gate_or_up, i_block])
                got = operand[:, column]
                assert float(got.min()) == want and float(got.max()) == want, (
                    f"block (h={h_block}, g={gate_or_up}, i={i_block}) should sit "
                    f"in column {column} as {want}; that column reads "
                    f"[{float(got.min())}, {float(got.max())}]"
                )
                checked += 1
    _emit_128(
        "operand",
        f"blocks_placed_correctly={checked}/{G128_BLOCKS} "
        f"shape={tuple(operand.shape)}",
    )
    assert checked == G128_BLOCKS

    # The index itself refuses a half selector it cannot place.
    with pytest.raises(MoeBlockwiseFp8Error):
        gate_up_flat_scale_index(0, GATE_UP_FUSION, 0, G128_I_BLOCKS)


def test_cte_128_a_one_hot_scale_moves_only_its_own_output_columns() -> None:
    """The kernel reads the column the flat index names, settled without an oracle.

    A single block's scale is doubled and the output delta must be confined to
    that block's ``128`` output columns. A kernel that read the scale operand in a
    different order would move a different column range, and the equality item
    could not say which of the two -- kernel or reference -- had moved.
    """
    case = _build_gate_up_128_case()
    hot_h, hot_g, hot_i = 1, 1, 2
    ones = torch.ones(
        (G128_H_BLOCKS, GATE_UP_FUSION, G128_I_BLOCKS), dtype=torch.float32
    )
    doubled = ones.clone()
    doubled[hot_h, hot_g, hot_i] = 2.0
    if torch.equal(doubled, ones):
        raise VacuousControlError("the injection changed nothing")

    reset_gate_up_dispatch_counters()
    with _SimulatorCounter() as sim:
        baseline = _gate_up_one_block(
            case["hidden"][0],
            case["weight_fused"][0],
            to_gate_up_kernel_scale_operand(ones, G128_H, G128_I),
        ).to(torch.float32)
        probed = _gate_up_one_block(
            case["hidden"][0],
            case["weight_fused"][0],
            to_gate_up_kernel_scale_operand(doubled, G128_H, G128_I),
        ).to(torch.float32)
    _assert_gate_up_route_128(sim, 2, "one-hot-route")

    delta = (probed - baseline).abs()
    hot_start = hot_g * G128_I + hot_i * GATE_UP_SCALE_BLOCK
    hot_stop = hot_start + GATE_UP_SCALE_BLOCK
    inside = float(delta[:, hot_start:hot_stop].max())
    outside = max(
        float(delta[:, :hot_start].max()) if hot_start > 0 else 0.0,
        float(delta[:, hot_stop:].max())
        if hot_stop < delta.shape[1]
        else 0.0,
    )
    _emit_128(
        "one-hot",
        f"hot_block=(h={hot_h},g={hot_g},i={hot_i}) columns={hot_start}:{hot_stop} "
        f"delta_inside={inside:.6e} delta_outside={outside:.6e}",
    )
    if inside == 0.0:
        raise VacuousControlError(
            "doubling a block scale moved nothing, so this probe cannot identify "
            "any column range and the instrument is unarmed"
        )
    assert outside == 0.0, (
        f"the delta escaped the hot block's own columns "
        f"({hot_start}:{hot_stop}): outside={outside:.6e}. The kernel is reading "
        f"the scale operand in a different order than "
        f"gate_up_flat_scale_index declares, which is a design contradiction to "
        f"route rather than a layout to re-guess here."
    )


# --------------------------------------------------------------------------- #
# LOAD-BEARING DEFAULTS -- each kept choice is shown to change the answer.        #
# --------------------------------------------------------------------------- #
def test_cte_128_the_folded_dequantisation_is_load_bearing() -> None:
    """Feed the shipped kernel a scale operand of ones: the answer must move.

    The cheapest possible falsifier for the fold, and it needs no second kernel:
    if the block scales did not reach the arithmetic, an all-ones operand would
    produce the same numbers as the real one and the equality above would be
    reading a kernel that ignores the checkpoint's scales entirely.
    """
    case = _build_gate_up_128_case()
    ones = torch.ones(
        (G128_H_BLOCKS, GATE_UP_FUSION, G128_I_BLOCKS), dtype=torch.float32
    )
    if torch.equal(ones, case["grid"][0]):
        raise VacuousControlError(
            "the fixture's own scale grid is all ones, so replacing it with ones "
            "changes nothing and this control is unarmed"
        )

    reset_gate_up_dispatch_counters()
    with _SimulatorCounter() as sim:
        without_scales = _gate_up_one_block(
            case["hidden"][0],
            case["weight_fused"][0],
            to_gate_up_kernel_scale_operand(ones, G128_H, G128_I),
        ).to(torch.float32)
    _assert_gate_up_route_128(sim, 1, "fold-control-route")

    want = _g128_reference_flat(case, 0)
    max_abs_diff = float((without_scales - want).abs().max())
    finite = math.isfinite(max_abs_diff)
    _emit_128(
        "default",
        f"disabled='the folded block dequantisation' max_abs_diff={max_abs_diff} "
        f"finite={int(finite)} reached_zero={int(max_abs_diff == 0)}",
    )
    if not finite:
        raise VacuousControlError(
            f"the ones-operand run produced {max_abs_diff}, which is not a number, "
            f"so this arm cannot say whether the fold is load-bearing"
        )
    assert max_abs_diff != 0, (
        "the kernel produced the same numbers with an all-ones scale operand, so "
        "the checkpoint's block scales do not reach its arithmetic and the "
        "equality above certifies nothing about the dequantisation"
    )


@nki.jit
def _variant_128_untransposed_kernel(hidden, fused_weight, scale_operand):
    """The shipped body with ONE line changed: the activation is not transposed.

    A copy rather than a switch on the shipped kernel, because the shipped body
    must carry no flag a caller could flip: a control that ran the shipped body
    could not falsify one of its choices. Only the activation load differs -- with
    a plain ``nl.load`` the contraction extent lands on the free axis and the
    engine contracts the token axis instead, which is a different function.
    """
    tokens, h_extent = hidden.shape
    _, fused_cols = fused_weight.shape
    i_extent = fused_cols // GATE_UP_FUSION
    n_h_blocks = h_extent // GATE_UP_SCALE_BLOCK
    n_i_blocks = i_extent // GATE_UP_SCALE_BLOCK
    n_col_blocks = GATE_UP_FUSION * n_i_blocks

    out = nl.ndarray((tokens, fused_cols), dtype=nl.float32, buffer=nl.shared_hbm)
    scale_sb = nl.load(scale_operand)

    for m_tile in range(tokens // TILE_SIZE):
        m0 = m_tile * TILE_SIZE
        for i_block in range(n_i_blocks):
            gate_col = i_block * GATE_UP_SCALE_BLOCK
            up_col = i_extent + i_block * GATE_UP_SCALE_BLOCK
            gate_acc = nl.ndarray(
                (TILE_SIZE, GATE_UP_SCALE_BLOCK), dtype=nl.float32, buffer=nl.sbuf
            )
            up_acc = nl.ndarray(
                (TILE_SIZE, GATE_UP_SCALE_BLOCK), dtype=nl.float32, buffer=nl.sbuf
            )
            for h_block in range(n_h_blocks):
                gate_psum = nl.ndarray(
                    (TILE_SIZE, GATE_UP_SCALE_BLOCK),
                    dtype=nl.float32,
                    buffer=nl.psum,
                )
                up_psum = nl.ndarray(
                    (TILE_SIZE, GATE_UP_SCALE_BLOCK),
                    dtype=nl.float32,
                    buffer=nl.psum,
                )
                h0 = h_block * GATE_UP_SCALE_BLOCK
                # THE ONE CHANGED LINE: no DMA-side transpose.
                hidden_t = nl.load(hidden[m0 : m0 + TILE_SIZE, h0 : h0 + TILE_SIZE])
                gate_w = nl.load(
                    fused_weight[
                        h0 : h0 + TILE_SIZE, gate_col : gate_col + GATE_UP_SCALE_BLOCK
                    ],
                    dtype=nl.bfloat16,
                )
                up_w = nl.load(
                    fused_weight[
                        h0 : h0 + TILE_SIZE, up_col : up_col + GATE_UP_SCALE_BLOCK
                    ],
                    dtype=nl.bfloat16,
                )
                nisa.nc_matmul(
                    dst=gate_psum[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                    stationary=hidden_t,
                    moving=gate_w,
                    accumulate=False,
                )
                nisa.nc_matmul(
                    dst=up_psum[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                    stationary=hidden_t,
                    moving=up_w,
                    accumulate=False,
                )
                gate_flat = h_block * n_col_blocks + i_block
                up_flat = h_block * n_col_blocks + n_i_blocks + i_block
                if h_block == 0:
                    nisa.tensor_scalar(
                        dst=gate_acc[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                        data=gate_psum[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                        op0=nl.multiply,
                        operand0=scale_sb[0:TILE_SIZE, gate_flat : gate_flat + 1],
                    )
                    nisa.tensor_scalar(
                        dst=up_acc[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                        data=up_psum[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                        op0=nl.multiply,
                        operand0=scale_sb[0:TILE_SIZE, up_flat : up_flat + 1],
                    )
                else:
                    nisa.scalar_tensor_tensor(
                        dst=gate_acc[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                        data=gate_psum[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                        op0=nl.multiply,
                        operand0=scale_sb[0:TILE_SIZE, gate_flat : gate_flat + 1],
                        op1=nl.add,
                        operand1=gate_acc[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                    )
                    nisa.scalar_tensor_tensor(
                        dst=up_acc[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                        data=up_psum[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                        op0=nl.multiply,
                        operand0=scale_sb[0:TILE_SIZE, up_flat : up_flat + 1],
                        op1=nl.add,
                        operand1=up_acc[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
                    )
            nl.store(
                out[m0 : m0 + TILE_SIZE, gate_col : gate_col + GATE_UP_SCALE_BLOCK],
                value=gate_acc[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
            )
            nl.store(
                out[m0 : m0 + TILE_SIZE, up_col : up_col + GATE_UP_SCALE_BLOCK],
                value=up_acc[0:TILE_SIZE, 0:GATE_UP_SCALE_BLOCK],
            )
    return out


def test_cte_128_the_activation_transpose_is_load_bearing() -> None:
    """Without the DMA-side transpose the kernel computes a different function.

    THREE OUTCOMES, NOT TWO, which is `inc-glm53f-112`'s form: a finite non-zero
    ``max_abs_diff`` falsifies the choice, a zero means it bought nothing, and a
    ``nan`` means the variant itself is broken -- reported as such rather than
    counted as a falsification.
    """
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    case = _build_gate_up_128_case()
    with _SimulatorCounter() as sim:
        got = wrap_nki(_variant_128_untransposed_kernel)(
            case["hidden"][0],
            case["weight_fused"][0],
            case["operands"][0],
        )
    want = _g128_reference_flat(case, 0)
    max_abs_diff = float((got.to(torch.float32) - want).abs().max())
    finite = math.isfinite(max_abs_diff)
    _emit_128(
        "default",
        f"disabled='the DMA-side activation transpose' "
        f"max_abs_diff={max_abs_diff} finite={int(finite)} "
        f"reached_zero={int(max_abs_diff == 0)} "
        f"simulate_kernel_calls={sim.calls}",
    )
    if not finite:
        raise VacuousControlError(
            f"the untransposed variant produced {max_abs_diff}, which is not a "
            f"number: the variant is broken, so this arm cannot say whether the "
            f"transpose is load-bearing"
        )
    assert max_abs_diff != 0, (
        "the untransposed variant reached bit equality, so the engine is not "
        "contracting the axis this kernel's operand orientation assumes and the "
        "orientation argument in the module comment is wrong"
    )


# ===========================================================================
# `inc-glm53f-113b` -- the activation and the down projection, both in NKI.
# ===========================================================================
#
# WHAT EACH READING SETTLES. The down projection carries the same EXACT equality
# `-113a` does, on the same kind of fixture, and it carries it in two arms: with
# every affinity at `1.0`, where the compared tensor is literally the plan's pinned
# pre-affinity matmul output, and with exactly-representable affinities, where the
# affinity multiply is inside the equality and cannot hide. The activation cannot
# be read by an equality in general -- a device sigmoid is not reproducible by a
# torch reference bit for bit -- so it is read twice instead: at the file's ALREADY
# DECLARED tolerance on the general fixture (no new tolerance pair, P9), and at
# BIT EQUALITY on a zero-gate fixture, where `SiLU(0) * up` is exactly `+0.0` and
# any plumbing error still shows.
#
# WHAT IS NOT HERE. The block seam is not switched, so the four landed items that
# assert the vendor identity stay byte-unchanged; the module comment names the
# three device-side constructs the switch needs and the campaign rule that says
# they are measured before they are used.

#: The affinity values, all exactly representable, so the affinity arm of the
#: equality stays bit-exact rather than becoming a tolerance reading.
_G128_AFFINITIES = (1.0, 0.5, 0.75, 0.25)


def _g128_down_grid() -> torch.Tensor:
    """``[E, I//128, H//128]`` fp32 down scales, mixing both families in every quad."""
    n_i, n_h = G128_I // GATE_UP_SCALE_BLOCK, G128_H // GATE_UP_SCALE_BLOCK
    grid = torch.empty((E, n_i, n_h), dtype=torch.float32)
    for expert in range(E):
        for i_block in range(n_i):
            for h_block in range(n_h):
                value = _G128_SCALE_VALUES[(i_block % 2) * 2 + (h_block % 2)]
                quad = (i_block // 2) * (n_h // 2) + (h_block // 2)
                grid[expert, i_block, h_block] = value * (
                    2.0 ** (((quad + expert) % 4) - 2)
                )
    return grid


def _build_down_128_case() -> dict:
    """One transposed intermediate, one down weight and one affinity column per expert."""
    grid = _g128_down_grid()
    # `[E, I, B]`: the activation seam's own output orientation.
    intermediate_t = _g128_fp8_values(201, E, G128_I, G128_TOKENS)
    down_weight = _g128_fp8_values(202, E, G128_I, G128_H).to(_FP8)
    affinity = torch.empty((E, G128_TOKENS, 1), dtype=torch.float32)
    for expert in range(E):
        for token in range(G128_TOKENS):
            affinity[expert, token, 0] = _G128_AFFINITIES[
                (token + expert) % len(_G128_AFFINITIES)
            ]
    return {
        "grid": grid,
        "intermediate_t": intermediate_t,
        "down_weight": down_weight,
        "affinity": affinity,
        "ones": torch.ones((E, G128_TOKENS, 1), dtype=torch.float32),
        "operands": [
            to_down_kernel_scale_operand(grid[expert], G128_I, G128_H)
            for expert in range(E)
        ],
    }


def _down_expanded_scales(grid: torch.Tensor, expert: int) -> torch.Tensor:
    """``[I, H]`` -- one scale per weight element."""
    per_block = grid[expert]
    expanded = per_block.repeat_interleave(GATE_UP_SCALE_BLOCK, dim=0)
    return expanded.repeat_interleave(GATE_UP_SCALE_BLOCK, dim=1)


def _down_reference(case: dict, expert: int, affinity_key: str) -> torch.Tensor:
    """The model-derived reference: dequantise, one matmul, then the affinity."""
    weight = case["down_weight"][expert].to(torch.float32)
    dequantised = weight * _down_expanded_scales(case["grid"], expert)
    intermediate = case["intermediate_t"][expert].t().to(torch.float32)
    return (intermediate @ dequantised) * case[affinity_key][expert]


def _down_reference_blockwise(case: dict, expert: int) -> torch.Tensor:
    """The same reference in the kernel's order: per contraction block, scale, add."""
    weight = case["down_weight"][expert].to(torch.float32)
    intermediate = case["intermediate_t"][expert].t().to(torch.float32)
    accumulator = torch.zeros((G128_TOKENS, G128_H), dtype=torch.float32)
    for i_block in range(G128_I // GATE_UP_SCALE_BLOCK):
        rows = slice(
            i_block * GATE_UP_SCALE_BLOCK, (i_block + 1) * GATE_UP_SCALE_BLOCK
        )
        partial = intermediate[:, rows] @ weight[rows, :]
        column_scales = case["grid"][expert, i_block].repeat_interleave(
            GATE_UP_SCALE_BLOCK
        )
        accumulator += partial * column_scales
    return accumulator * case["ones"][expert]


def _down_precondition(case: dict, expert: int) -> tuple[bool, float, bool]:
    """``(fp64_agrees, gap, forms_agree)`` for the down projection, printed first."""
    weight64 = case["down_weight"][expert].to(torch.float64)
    scales64 = _down_expanded_scales(case["grid"], expert).to(torch.float64)
    intermediate64 = case["intermediate_t"][expert].t().to(torch.float64)
    reference64 = intermediate64 @ (weight64 * scales64)
    flat32 = _down_reference(case, expert, "ones")
    gap = float((reference64 - flat32.to(torch.float64)).abs().max())
    return (
        bool(torch.equal(reference64, flat32.to(torch.float64))),
        gap,
        bool(torch.equal(flat32, _down_reference_blockwise(case, expert))),
    )


def _assert_limb_route_128(counters, sim, expected_dispatches: int, label: str) -> str:
    """The three declared route values for one `-113b` limb, each read as a number."""
    nki_dispatch, torch_fallback = counters()
    gate = can_run_kernel(torch.zeros(1))
    reading = (
        f"nki_dispatch={nki_dispatch} torch_fallback={torch_fallback} "
        f"can_run_kernel={gate} simulate_kernel_calls={sim.calls}"
    )
    _emit_128(label, reading)
    if nki_dispatch != expected_dispatches:
        raise RouteInstrumentError(
            f"{label}: the limb's dispatch counter read {nki_dispatch}, declared "
            f"{expected_dispatches}. {reading}"
        )
    if torch_fallback != 0:
        raise RouteInstrumentError(
            f"{label}: the torch-fallback counter read {torch_fallback}, declared "
            f"exactly 0. {reading}"
        )
    if gate is not True:
        raise RouteInstrumentError(f"{label}: can_run_kernel() read {gate!r}. {reading}")
    if sim.calls != expected_dispatches:
        raise RouteInstrumentError(
            f"{label}: nki.simulator.simulate_kernel ran {sim.calls} times, declared "
            f"{expected_dispatches}. A numeric pass without a simulator call is the "
            f"F1 false green. {reading}"
        )
    return reading


def test_cte_128_down_matches_the_model_reference_per_expert_block() -> None:
    """The down projection equals the model-derived reference, bit for bit, twice.

    Arm 1 sets every affinity to ``1.0``, so the compared tensor is exactly the
    plan's pinned pre-affinity matmul output. Arm 2 uses exactly-representable
    affinities, so the affinity multiply is inside the equality and a wrong
    placement or a wrong broadcast cannot pass.
    """
    case = _build_down_128_case()
    reset_down_dispatch_counters()

    for expert in range(E):
        exact, gap, forms_agree = _down_precondition(case, expert)
        _emit_128(
            "down-precondition",
            f"expert={expert} fp64_vs_fp32_bit_equal={int(exact)} "
            f"max_gap={gap:.6e} flat_form_equals_block_form={int(forms_agree)}",
        )
        if not exact or not forms_agree:
            raise GateUpExactnessError(
                f"expert {expert}: exactness precondition failed "
                f"(fp64 agrees={exact}, forms agree={forms_agree}, gap={gap:.6e}), "
                f"so an equality here would be a statement about rounding"
            )

    with _SimulatorCounter() as sim:
        outputs = {
            (expert, key): _down_one_block(
                case["intermediate_t"][expert],
                case["down_weight"][expert],
                case["operands"][expert],
                case[key][expert],
            )
            for expert in range(E)
            for key in ("ones", "affinity")
        }
    _assert_limb_route_128(down_dispatch_counters, sim, 2 * E, "down-acceptance")

    passed = 0
    worst = -1.0
    for (expert, key), output in outputs.items():
        got = output.to(torch.float32)
        want = _down_reference(case, expert, key)
        assert tuple(got.shape) == (G128_TOKENS, G128_H), tuple(got.shape)
        if float(want.abs().max()) == 0.0:
            raise VacuousControlError(
                f"expert {expert} [{key}]: the reference is all zero"
            )
        max_abs_diff = float((got - want).abs().max())
        worst = max(worst, max_abs_diff)
        _emit_128(
            "down-equality",
            f"expert={expert} affinity={key} "
            f"bit_equal={int(bool(torch.equal(got, want)))} "
            f"max_abs_diff={max_abs_diff} want_absmax={float(want.abs().max()):.6e}",
        )
        assert torch.equal(got, want), (
            f"expert {expert} [{key}]: the down projection and the model-derived "
            f"reference are not bit-equal; max_abs_diff={max_abs_diff}"
        )
        assert max_abs_diff == 0
        passed += 1

    _emit_128(
        "down-verdict",
        f"expert_block_arms_passing={passed}/{2 * E} worst_max_abs_diff={worst}",
    )
    assert passed == 2 * E


def test_cte_128_down_a_lossy_256_retile_must_not_reach_exactness() -> None:
    """The `256` quad-maximum mapping must not reach exactness on the down grid either.

    Armed the same three ways as the gate/up control, and it runs the SAME retile:
    every quad is shown to mix both mantissa families, the mapping is shown to have
    moved both the grid and the weight bytes, and at least one tile is shown to have
    left the fp8 grid. The down grid has no fusion axis, so the retile applies to it
    whole rather than a half at a time.
    """
    case = _build_down_128_case()
    grid = case["grid"]
    n_i, n_h = grid.shape[1], grid.shape[2]

    quads = 0
    for expert in range(E):
        for quad_i in range(n_i // 2):
            for quad_h in range(n_h // 2):
                values = grid[
                    expert,
                    2 * quad_i : 2 * quad_i + 2,
                    2 * quad_h : 2 * quad_h + 2,
                ].reshape(-1)
                families = {_mantissa_family_128(float(v)) for v in values}
                if families != {"A", "B"}:
                    raise VacuousControlError(
                        f"down quad (expert={expert}, i={quad_i}, h={quad_h}) draws "
                        f"from {sorted(families)} only, values "
                        f"{[float(v) for v in values]}; a single-family quad "
                        f"retiles losslessly and this control would read zero for "
                        f"the wrong reason"
                    )
                quads += 1
    _emit_128("down-control-arming", f"quads_mixing_both_families={quads}/{quads}")

    lossy_weight, lossy, inexact_tiles = _lossy_256_retile(
        case["down_weight"][0].to(torch.float32), grid[0]
    )
    if torch.equal(lossy, grid[0]):
        raise VacuousControlError("the lossy 256 mapping changed no down scale")
    if torch.equal(
        lossy_weight.to(torch.float32), case["down_weight"][0].to(torch.float32)
    ):
        raise VacuousControlError(
            "the lossy 256 mapping changed no down weight byte, so it is a scale "
            "swap and not a retile, and it would move the output for every kernel"
        )
    if inexact_tiles == 0:
        raise VacuousControlError(
            "every rescaled down tile stayed exactly on the fp8 grid, so this "
            "mapping is a LOSSLESS retile and shows nothing about the 128 grid"
        )
    _emit_128(
        "down-control-arming",
        f"scales_changed_by_the_256_mapping={int((lossy != grid[0]).sum())} "
        f"of {grid[0].numel()} tiles_the_fp8_cast_rounded={inexact_tiles}",
    )

    reset_down_dispatch_counters()
    with _SimulatorCounter() as sim:
        got = _down_one_block(
            case["intermediate_t"][0],
            lossy_weight,
            to_down_kernel_scale_operand(lossy, G128_I, G128_H),
            case["ones"][0],
        ).to(torch.float32)
    _assert_limb_route_128(down_dispatch_counters, sim, 1, "down-control-route")

    max_abs_diff = float((got - _down_reference(case, 0, "ones")).abs().max())
    _emit_128(
        "down-control",
        f"lossy_256_max_abs_diff={max_abs_diff} "
        f"reached_zero={int(max_abs_diff == 0)}",
    )
    assert max_abs_diff != 0, (
        "the lossy 256 retile reached bit equality on the down projection, so this "
        "limb's equality does not discriminate the two granularities"
    )


def test_cte_128_the_affinity_scaling_is_load_bearing() -> None:
    """Changing the affinities must change the down result, by exactly their ratio.

    Two readings rather than one: the two arms differ (so the affinity reaches the
    arithmetic at all), and their ratio is exactly the affinity column (so it is
    applied per TOKEN and not per anything else).
    """
    case = _build_down_128_case()
    reset_down_dispatch_counters()
    with _SimulatorCounter() as sim:
        plain = _down_one_block(
            case["intermediate_t"][0],
            case["down_weight"][0],
            case["operands"][0],
            case["ones"][0],
        ).to(torch.float32)
        scaled = _down_one_block(
            case["intermediate_t"][0],
            case["down_weight"][0],
            case["operands"][0],
            case["affinity"][0],
        ).to(torch.float32)
    _assert_limb_route_128(down_dispatch_counters, sim, 2, "affinity-route")

    if torch.equal(case["affinity"][0], case["ones"][0]):
        raise VacuousControlError("the two affinity columns are identical")
    moved = float((scaled - plain).abs().max())
    ratio_exact = bool(torch.equal(scaled, plain * case["affinity"][0]))
    _emit_128(
        "affinity",
        f"max_abs_change={moved} scaled_equals_plain_times_affinity="
        f"{int(ratio_exact)}",
    )
    assert moved != 0, (
        "the affinity column changed nothing, so it does not reach the kernel's "
        "arithmetic and the equality above certifies nothing about it"
    )
    assert ratio_exact, (
        "the two arms differ but not by the affinity column, so the affinity is "
        "applied along the wrong axis"
    )


def test_cte_128_swiglu_matches_torch_and_is_exact_where_it_can_be() -> None:
    """The activation, read twice: at the declared tolerance, and at bit equality.

    A device sigmoid is not reproducible by a torch reference bit for bit, so the
    general arm is bounded by the tolerance this file ALREADY declares (no new
    pair, P9) and reports its own error as a number. The zero-gate arm is exact:
    ``SiLU(0) * up`` is ``+0.0``, so plumbing errors still have nowhere to hide.
    """
    gate_up = torch.empty(
        (G128_TOKENS, GATE_UP_FUSION * G128_I), dtype=torch.float32
    )
    gate_up[:, :G128_I] = _g128_fp8_values(301, G128_TOKENS, G128_I) - 0.5
    gate_up[:, G128_I:] = _g128_fp8_values(302, G128_TOKENS, G128_I)

    reset_swiglu_dispatch_counters()
    with _SimulatorCounter() as sim:
        got = moe_swiglu_transposed(gate_up).to(torch.float32)
    _assert_limb_route_128(swiglu_dispatch_counters, sim, 1, "swiglu-route")

    assert tuple(got.shape) == (G128_I, G128_TOKENS), (
        f"the activation seam returned {tuple(got.shape)}, expected the transposed "
        f"{(G128_I, G128_TOKENS)} its contract declares"
    )
    want = (
        torch.nn.functional.silu(gate_up[:, :G128_I]) * gate_up[:, G128_I:]
    ).t().contiguous()
    max_rel = _max_rel_error(got, want)
    _emit_128(
        "swiglu",
        f"max_rel_error={max_rel:.6e} rtol={RTOL} atol={ATOL} "
        f"want_absmax={float(want.abs().max()):.6e}",
    )
    if float(want.abs().max()) == 0.0:
        raise VacuousControlError("the activation reference is all zero")
    torch.testing.assert_close(got, want, rtol=RTOL, atol=ATOL)

    # The exact arm: a zero gate makes SiLU(gate) * up exactly +0.0.
    zero_gate = gate_up.clone()
    zero_gate[:, :G128_I] = 0.0
    reset_swiglu_dispatch_counters()
    with _SimulatorCounter() as sim:
        zeros = moe_swiglu_transposed(zero_gate).to(torch.float32)
    _assert_limb_route_128(swiglu_dispatch_counters, sim, 1, "swiglu-zero-route")
    max_abs_diff = float(zeros.abs().max())
    _emit_128(
        "swiglu",
        f"zero_gate_max_abs={max_abs_diff} "
        f"bit_equal_to_positive_zero="
        f"{int(bool(torch.equal(zeros, torch.zeros_like(zeros))))} "
        f"signbits_set={int((torch.signbit(zeros)).sum())}",
    )
    assert torch.equal(zeros, torch.zeros_like(zeros)), max_abs_diff
    # +0.0 and not -0.0: the sign bit is a real distinction the campaign has been
    # bitten by before (`functional/dsa/ragged_pack.py` records the measurement).
    assert int(torch.signbit(zeros).sum()) == 0


def test_cte_128_swiglu_and_down_identities_are_authored_in_this_campaign() -> None:
    """Both new seams dispatch to kernels this module authors, derived through the seam."""
    import vllm_neuron.functional.moe.moe_blockwise_fp8 as moe

    for label, reading, expected in (
        ("swiglu", swiglu_kernel_identity(), "moe_swiglu_transposed_kernel"),
        ("down", down_kernel_identity(), "moe_down_blockwise_fp8_kernel"),
    ):
        module, qualname = reading
        _emit_128("identity", f"{label}_kernel={module}.{qualname}")
        assert module == "vllm_neuron.functional.moe.moe_blockwise_fp8", module
        assert qualname == expected, qualname
        assert "nkilib" not in module

    # Non-vacuity: the vendor member is still bound here and reads differently.
    vendor = moe._unwrap_nki(moe.blockwise_mm_baseline_shard_intermediate)
    assert (vendor.__module__, vendor.__qualname__) != (
        "vllm_neuron.functional.moe.moe_blockwise_fp8",
        "moe_down_blockwise_fp8_kernel",
    )
    # And the three campaign limbs are three different kernels, not one read thrice.
    identities = {
        gate_up_kernel_identity(),
        swiglu_kernel_identity(),
        down_kernel_identity(),
    }
    _emit_128("identity", f"distinct_campaign_kernels={len(identities)}/3")
    assert len(identities) == 3


@pytest.mark.parametrize(
    "tokens,rows,cols,needle",
    [
        (200, 512, 512, "B=200 is not a positive multiple"),
        (256, 500, 512, "I=500 is not a positive multiple"),
        (256, 512, 500, "H=500 is not a positive multiple"),
    ],
)
def test_cte_128_down_refuses_inadmissible_geometry_by_name(
    tokens: int, rows: int, cols: int, needle: str
) -> None:
    """Every down refusal names the offending extent."""
    with pytest.raises(MoeBlockwiseFp8Error) as excinfo:
        can_run_moe_down_blockwise_fp8(torch.zeros(1), tokens, rows, cols)
    assert needle in str(excinfo.value), str(excinfo.value)


def test_cte_128_down_and_swiglu_refuse_wrong_operands_by_name() -> None:
    """Rank, orientation, operand shape and affinity shape are each refused by name."""
    case = _build_down_128_case()
    inter, weight = case["intermediate_t"][0], case["down_weight"][0]
    operand, affinity = case["operands"][0], case["ones"][0]

    with pytest.raises(MoeBlockwiseFp8Error) as rank:
        moe_down_blockwise_fp8(
            inter.unsqueeze(0),
            weight.unsqueeze(0),
            operand.unsqueeze(0),
            _affinity_bank(affinity),
            *_one_block_routing(G128_TOKENS),
            G128_TOKENS,
        )
    assert "must have rank 2" in str(rank.value)

    # THE WRONG-WAY WEIGHT IS BUILT AT A CONTRACTION THIS CASE DOES NOT USE, and
    # that is forced rather than chosen. `H` and `I_TP` are both 512 here, so the
    # transpose of this case's own `[I, H]` weight carries the SAME shape as the
    # weight itself, and a seam that checks axis 0 has nothing to see. Doubling
    # the contraction separates the two orientations: `[2*H, I]` is the shape a
    # valid `[I, 2*H]` weight would have if it were handed over transposed.
    wrong_way = torch.zeros((2 * G128_H, G128_I), dtype=torch.float32).to(_FP8)
    with pytest.raises(MoeBlockwiseFp8Error) as orientation:
        moe_down_blockwise_fp8(
            inter,
            wrong_way.unsqueeze(0),
            operand.unsqueeze(0),
            _affinity_bank(affinity),
            *_one_block_routing(G128_TOKENS),
            G128_TOKENS,
        )
    assert "contraction-major" in str(orientation.value)

    with pytest.raises(MoeBlockwiseFp8Error) as shape:
        moe_down_blockwise_fp8(
            inter,
            weight.unsqueeze(0),
            operand[:, :-1].contiguous().unsqueeze(0),
            _affinity_bank(affinity),
            *_one_block_routing(G128_TOKENS),
            G128_TOKENS,
        )
    assert "to_down_kernel_scale_operand" in str(shape.value)

    with pytest.raises(MoeBlockwiseFp8Error) as columns:
        moe_down_blockwise_fp8(
            inter,
            weight.unsqueeze(0),
            operand.unsqueeze(0),
            _affinity_bank(affinity)[:-1],
            *_one_block_routing(G128_TOKENS),
            G128_TOKENS,
        )
    assert "affinity_bank must be [(T + 1) * E_local, 1]" in str(columns.value)
    # AND THE ORIENTATION ARM ABOVE IS NOT HOLLOW, which this line is what says.
    # The call here carries the case's OWN correctly-oriented weight and reached
    # the affinity guard, so it got PAST the orientation guard -- which makes the
    # message that arm reads the orientation guard's own, rather than something
    # every refused call to this seam happens to carry.
    assert "contraction-major" not in str(columns.value)

    with pytest.raises(MoeBlockwiseFp8Error) as grid:
        to_down_kernel_scale_operand(case["grid"][0][:, :-1].contiguous(), G128_I, G128_H)
    assert "mis-sized" in str(grid.value)

    with pytest.raises(MoeBlockwiseFp8Error) as activation:
        moe_swiglu_transposed(torch.zeros((G128_TOKENS, 3), dtype=torch.float32))
    assert "GATE_UP_FUSION" in str(activation.value)
    _emit_128("refusal", "down_and_swiglu_refusals=6 all_named")
