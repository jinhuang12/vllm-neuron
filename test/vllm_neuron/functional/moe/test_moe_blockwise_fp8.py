# SPDX-License-Identifier: Apache-2.0
"""Tests for the routed block-quant MoE limbs at the checkpoint's 128-block scales.

Covers ``moe_gate_up_blockwise_fp8``, ``moe_swiglu_transposed`` and
``moe_down_blockwise_fp8`` under the NKI simulator, plus the vendor seam's named
refusals and its scale-layout bridge.

The matmul fixtures are built so the kernel and the torch reference agree bit for
bit: weights and activations are ``k/8`` (on the fp8-e4m3 grid) and every block
scale is a two-mantissa-bit value times a power of two. Only the activation, whose
device sigmoid is not reproducible exactly, is compared at ``RTOL=3e-2, ATOL=1e-5``.
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

# Extents the vendor kernel (`nkilib/core/moe/moe_cte/bwmm_shard_on_I.py`) asserts:
# 512 <= H <= 8192 with H % 512 == 0; I % 256 == 0 and an even multiple at two
# shards; B % 256 == 0.
H = 512
I_TP = 512
E = 2
B = 256

H_256 = H // BLOCK_QUANT_SIZE
I_256 = I_TP // BLOCK_QUANT_SIZE

# Used for the activation only; the matmul limbs are compared with torch.equal.
RTOL = 3e-2
ATOL = 1e-5

_FP8 = torch.float8_e4m3fn


class DispatchRouteError(AssertionError):
    """A limb did not take the NKI route exactly as many times as expected."""


class VacuousCaseError(AssertionError):
    """A case whose input could not have made it fail: empty, or arms identical."""


class ReferenceExactnessError(AssertionError):
    """The fixture did not make the fp32 reference exact, so torch.equal is unsafe."""


class _SimulatorCounter:
    """Counts ``nki.simulator.simulate_kernel`` calls made while it is active."""

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


# --------------------------------------------------------------------------- #
# Kernel identity.                                                              #
# --------------------------------------------------------------------------- #
def _seam_without_wrap_nki():
    """A probe seam that makes no ``wrap_nki`` call."""
    return 1


def _seam_with_two_wrap_nki_calls():
    """A probe seam that makes two ``wrap_nki`` calls."""
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    first = wrap_nki(moe_gate_up_blockwise_fp8_kernel_probe_a)
    second = wrap_nki(moe_gate_up_blockwise_fp8_kernel_probe_b)
    return first, second


def moe_gate_up_blockwise_fp8_kernel_probe_a() -> None:
    """Target for the two-call probe. Never traced."""


def moe_gate_up_blockwise_fp8_kernel_probe_b() -> None:
    """Target for the two-call probe. Never traced."""


def test_the_limb_identities_name_three_distinct_kernels_of_this_module() -> None:
    """Each limb dispatches to its own kernel in this module, none to the vendor's."""
    import vllm_neuron.functional.moe.moe_blockwise_fp8 as moe

    mine = "vllm_neuron.functional.moe.moe_blockwise_fp8"
    readings = {
        "gate_up": gate_up_kernel_identity(),
        "swiglu": swiglu_kernel_identity(),
        "down": down_kernel_identity(),
    }
    foreign = {
        limb: module for limb, (module, _) in readings.items() if module != mine
    }
    assert foreign == {}, f"these limbs resolve outside this module: {foreign}"
    assert all("nkilib" not in module for module, _ in readings.values())
    assert readings["gate_up"][1] == "moe_gate_up_blockwise_fp8_kernel"
    assert readings["swiglu"][1] == "moe_swiglu_transposed_kernel"
    assert readings["down"][1] == "moe_down_blockwise_fp8_kernel"
    assert len({qualname for _, qualname in readings.values()}) == len(readings), (
        f"two limbs resolved to the same kernel: {readings}"
    )

    # By object identity, not by name: the derivation resolves the live binding.
    derived = moe._unwrap_nki(
        moe._wrapped_object_of(moe.moe_gate_up_blockwise_fp8, "the gate/up seam")
    )
    assert derived is moe._unwrap_nki(moe.moe_gate_up_blockwise_fp8_kernel)

    vendor = moe._unwrap_nki(moe.blockwise_mm_baseline_shard_intermediate)
    assert (vendor.__module__, vendor.__qualname__) != readings["gate_up"]
    assert (vendor.__module__, vendor.__qualname__) not in set(readings.values())


def test_the_seam_reader_refuses_a_seam_it_cannot_read() -> None:
    """``_wrapped_object_of`` raises on zero calls, two calls, or unreadable source."""
    import vllm_neuron.functional.moe.moe_blockwise_fp8 as moe

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
# The vendor seam: named refusals and the scale-layout bridge.                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "rows,cols,needle",
    [
        (512, 256, "BLOCK_QUANT_SIZE * NUM_SHARDS"),  # odd multiple of 256
        (512, 768, "BLOCK_QUANT_SIZE * NUM_SHARDS"),  # odd multiple of 256
        (256, 512, "outside [512, 8192]"),  # below the H floor
        (768, 512, "multiple of PSUM_SIZE"),  # H not a multiple of 512
        (512, 500, "not a positive multiple of 256"),  # I not 256-blocked
    ],
)
def test_the_vendor_seam_refuses_inadmissible_geometry_by_name(
    rows: int, cols: int, needle: str
) -> None:
    """Every refusal is a named error carrying the offending extent."""
    with pytest.raises(MoeBlockwiseFp8Error) as excinfo:
        can_run_blockwise_fp8_moe(torch.zeros(1), rows, cols)
    message = str(excinfo.value)
    assert needle in message, f"[H={rows},I={cols}] message was: {message}"


def test_to_kernel_scale_layout_refuses_a_missized_tensor() -> None:
    """A mis-sized scale tensor is refused, not reshaped onto a wrong mapping."""
    good = torch.ones((E, I_256 * H_256 * TILE_SIZE), dtype=torch.float32)
    bridged = to_kernel_scale_layout(good, E, H, I_TP, projection=DOWN)
    assert tuple(bridged.shape) == kernel_scale_shape(E, H, I_TP, DOWN)

    bad = torch.ones((E, I_256 * H_256 * TILE_SIZE - TILE_SIZE), dtype=torch.float32)
    with pytest.raises(MoeBlockwiseFp8Error) as excinfo:
        to_kernel_scale_layout(bad, E, H, I_TP, projection=DOWN)
    assert "mis-sized" in str(excinfo.value)


def test_kernel_scale_shape_matches_the_producer_element_count() -> None:
    """The bridge conserves elements: no slot invented, none dropped."""
    for projection in (DOWN, GATE_UP):
        logical = kernel_scale_shape(E, H, I_TP, projection)
        elements = 1
        for extent in logical:
            elements *= extent
        flat = torch.ones((E, elements // E), dtype=torch.float32)
        bridged = to_kernel_scale_layout(flat, E, H, I_TP, projection)
        assert bridged.numel() == flat.numel()
        assert logical[-1] == TILE_SIZE


# --------------------------------------------------------------------------- #
# The gate/up kernel at [128, 128] scales.                                       #
#                                                                               #
# The compared tensor is the fp32 pre-activation matmul output. Weights and     #
# activations are k/8 and every scale is a two-mantissa-bit value times a power #
# of two, so every product and partial sum is exact in fp32 and the kernel must #
# match the reference under torch.equal. The lossy-256 case collapses each      #
# 2 x 2 quad of 128-blocks to its maximum; 2.5 == 2 * 1.25 and 3.5 == 2 * 1.75, #
# so a quad drawn from one mantissa family would retile losslessly, and the     #
# grid therefore mixes both families in every quad.                             #
# --------------------------------------------------------------------------- #
G128_TOKENS = B
G128_H = H
G128_I = I_TP
G128_H_BLOCKS = G128_H // GATE_UP_SCALE_BLOCK
G128_I_BLOCKS = G128_I // GATE_UP_SCALE_BLOCK
G128_BLOCKS = G128_H_BLOCKS * GATE_UP_FUSION * G128_I_BLOCKS


def _one_block_routing(tokens: int) -> tuple:
    """``(row_index, expert_index, block)`` for one block of one expert.

    The identity row index and a single zero expert; ``block`` is the whole token
    count, so the routed kernels see exactly one block.
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
    """The routed down seam at the single-expert, single-block shape."""
    tokens = int(intermediate_t.shape[1])
    return moe_down_blockwise_fp8(
        intermediate_t,
        down_weight.unsqueeze(0),
        scale_operand.unsqueeze(0),
        _affinity_bank(affinity),
        *_one_block_routing(tokens),
        tokens,
    )


# The four scale values and their two mantissa families. Powers of two multiply them
# per quad, which keeps every block scale distinct without moving a value between
# families.
_G128_SCALE_VALUES = (1.25, 1.75, 2.5, 3.5)
_G128_FAMILY = {1.25: "A", 2.5: "A", 1.75: "B", 3.5: "B"}


def _mantissa_family_128(value: float) -> str:
    """``"A"`` for the ``1.25`` family, ``"B"`` for the ``1.75`` family."""
    # Recovered by exact halving and doubling rather than log2, so the classification
    # is a bit-level statement about the value.
    scaled = float(value)
    while scaled >= 2.0:
        scaled /= 2.0
    while scaled < 1.0:
        scaled *= 2.0
    if scaled not in _G128_FAMILY:
        raise VacuousCaseError(
            f"{value!r} has significand {scaled!r}, which is not one of "
            f"{sorted(_G128_FAMILY)}; the family classification is undefined and "
            f"the case cannot state that its quads mix families"
        )
    return _G128_FAMILY[scaled]


def _g128_scale_grid() -> torch.Tensor:
    """``[E, H//128, 2, I//128]`` fp32: the checkpoint's own grid, deterministic.

    Within every ``2 x 2`` quad the four scales come from both mantissa families,
    and the power-of-two factor varies per quad so every block scale is distinct.
    """
    grid = torch.empty(
        (E, G128_H_BLOCKS, GATE_UP_FUSION, G128_I_BLOCKS), dtype=torch.float32
    )
    for expert in range(E):
        for h_block in range(G128_H_BLOCKS):
            for gate_or_up in range(GATE_UP_FUSION):
                for i_block in range(G128_I_BLOCKS):
                    value = _G128_SCALE_VALUES[(h_block % 2) * 2 + (i_block % 2)]
                    quad = (h_block // 2) * (G128_I_BLOCKS // 2) + (i_block // 2)
                    exponent = ((quad + gate_or_up + expert) % 4) - 2
                    grid[expert, h_block, gate_or_up, i_block] = value * (
                        2.0**exponent
                    )
    return grid


def _g128_fp8_values(seed: int, *shape: int) -> torch.Tensor:
    """``k/8`` for ``k`` in ``1..7``: on the fp8-e4m3 grid, so every cast is exact.

    Unsigned, because over a 512-long contraction a signed fixture cancels and the
    equality readings would become fragile for a reason unrelated to the kernel.
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(1, 8, shape, generator=generator).to(torch.float32) / 8.0


def _build_gate_up_128_case() -> dict:
    """One token block and one fused gate/up weight per expert, plus the operands."""
    grid = _g128_scale_grid()
    hidden = _g128_fp8_values(101, E, G128_TOKENS, G128_H).to(torch.bfloat16)
    # `[E, H, 2, I]` is the checkpoint's own layout; `[E, H, 2*I]` is its plain
    # reshape, which is what the kernel consumes.
    weight_fused = _g128_fp8_values(102, E, G128_H, GATE_UP_FUSION * G128_I).to(_FP8)
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
    """``[H, 2*I]``: one scale per element, expanded from the block grid."""
    per_block = grid[expert]  # [nh, 2, ni]
    expanded = per_block.repeat_interleave(GATE_UP_SCALE_BLOCK, dim=0)
    expanded = expanded.repeat_interleave(GATE_UP_SCALE_BLOCK, dim=2)
    return expanded.reshape(G128_H, GATE_UP_FUSION * G128_I)


def _g128_reference_flat(case: dict, expert: int) -> torch.Tensor:
    """The reference: dequantise every element, then one matmul."""
    weight = case["weight_fused"][expert].to(torch.float32)
    dequantised = weight * _g128_expanded_scales(case["grid"], expert)
    return case["hidden"][expert].to(torch.float32) @ dequantised


def _g128_reference_blockwise(case: dict, expert: int) -> torch.Tensor:
    """The same reference in the kernel's own order: per block, scale, then add."""
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
    """``(fp64_agrees, gap, forms_agree)``, checked before any equality is read."""
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


def _assert_nki_route(
    counters, sim: _SimulatorCounter, expected_dispatches: int, label: str
) -> None:
    """Every call took the NKI route: the counters, the gate and the simulator agree."""
    nki_dispatch, torch_fallback = counters()
    gate = can_run_kernel(torch.zeros(1))
    reading = (
        f"nki_dispatch={nki_dispatch} torch_fallback={torch_fallback} "
        f"can_run_kernel={gate} simulate_kernel_calls={sim.calls}"
    )
    if nki_dispatch != expected_dispatches:
        raise DispatchRouteError(
            f"{label}: the dispatch counter read {nki_dispatch}, expected "
            f"{expected_dispatches}. {reading}"
        )
    if torch_fallback != 0:
        raise DispatchRouteError(
            f"{label}: the torch-fallback counter read {torch_fallback}, expected 0. "
            f"{reading}"
        )
    if gate is not True:
        raise DispatchRouteError(
            f"{label}: can_run_kernel() read {gate!r}, expected True. {reading}"
        )
    if sim.calls != expected_dispatches:
        raise DispatchRouteError(
            f"{label}: nki.simulator.simulate_kernel ran {sim.calls} times, expected "
            f"{expected_dispatches}; a numeric pass without a simulator call would "
            f"compare torch with torch. {reading}"
        )


def test_gate_up_matches_the_reference_per_expert_block() -> None:
    """The fp32 pre-activation gate/up output equals the reference bit for bit."""
    case = _build_gate_up_128_case()
    reset_gate_up_dispatch_counters()

    for expert in range(E):
        exact, gap, forms_agree = _g128_precondition(case, expert)
        if not exact:
            raise ReferenceExactnessError(
                f"expert {expert}: the fp64 and fp32 references disagree on this "
                f"fixture (max gap {gap:.6e}), so an equality against the fp32 "
                f"reference would measure rounding rather than the kernel"
            )
        if not forms_agree:
            raise ReferenceExactnessError(
                f"expert {expert}: the flat and block-wise reference forms differ, "
                f"so the kernel's per-block summation order is not neutral on this "
                f"fixture"
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
    _assert_nki_route(gate_up_dispatch_counters, sim, E, "gate/up")

    for expert in range(E):
        got = outputs[expert].to(torch.float32)
        expected = _g128_reference_flat(case, expert)
        assert tuple(got.shape) == (G128_TOKENS, GATE_UP_FUSION * G128_I), (
            f"expert {expert}: kernel returned {tuple(got.shape)}, expected "
            f"{(G128_TOKENS, GATE_UP_FUSION * G128_I)}"
        )
        # A reference of all zeros would make an equality vacuous.
        if float(expected.abs().max()) == 0.0:
            raise VacuousCaseError(
                f"expert {expert}: the reference is all zero, so an equality "
                f"against it measures nothing"
            )
        max_abs_diff = float((got - expected).abs().max())
        assert torch.equal(got, expected), (
            f"expert {expert}: the kernel and the reference are not bit-equal; "
            f"max_abs_diff={max_abs_diff}"
        )
        assert max_abs_diff == 0


def _block_slice(tile: int) -> slice:
    """The weight rows, or columns, that one ``128`` scale block covers."""
    return slice(tile * GATE_UP_SCALE_BLOCK, (tile + 1) * GATE_UP_SCALE_BLOCK)


def _lossy_256_retile(
    weight_fp32: torch.Tensor, scale2d: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """A real ``256`` retile of one weight matrix: its bytes move with its scale.

    One scale per ``2 x 2`` quad of ``128`` blocks is kept (the quad maximum, so no
    rescaled byte can overflow e4m3) and the other three tiles' weight bytes are
    re-expressed against it as ``w * s_tile / s_retained``, cast back to fp8. Within
    a quad two of the four ratios are not powers of two, so the cast rounds; that
    rounding is the retile's loss. Rewriting only the scale would instead corrupt the
    product by whole factors and show a difference for any kernel.

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
    the two halves retile independently and the fused weight is reassembled.
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


def test_a_lossy_256_retile_does_not_reach_exactness() -> None:
    """A quad-maximum 256 retile of the same weights cannot reach bit equality."""
    case = _build_gate_up_128_case()
    grid = case["grid"]

    # A single-family quad retiles losslessly, so every quad must mix families.
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
                        raise VacuousCaseError(
                            f"quad (expert={expert}, g={gate_or_up}, "
                            f"h={quad_h}, i={quad_i}) draws from families "
                            f"{sorted(families)} only, values "
                            f"{[float(v) for v in values]}. A single-family quad "
                            f"retiles losslessly, so this case could read "
                            f"max_abs_diff=0 for the wrong reason."
                        )
                    quads += 1
    if quads == 0:
        raise VacuousCaseError("no quads were examined, so nothing is checked")

    lossy_weight, lossy, inexact_tiles = _lossy_256_gate_up(case, 0)
    if torch.equal(lossy, grid[0]):
        raise VacuousCaseError(
            "the lossy 256 mapping changed no scale, so the case cannot "
            "distinguish the two granularities"
        )
    if torch.equal(
        lossy_weight.to(torch.float32), case["weight_fused"][0].to(torch.float32)
    ):
        raise VacuousCaseError(
            "the lossy 256 mapping changed no weight byte, so it is a scale swap "
            "and not a retile, and it would move the output for every kernel"
        )
    if inexact_tiles == 0:
        raise VacuousCaseError(
            "every rescaled tile stayed exactly on the fp8 grid, so this mapping is "
            "a lossless retile and cannot show that the 128 grid buys anything"
        )

    reset_gate_up_dispatch_counters()
    with _SimulatorCounter() as sim:
        got = _gate_up_one_block(
            case["hidden"][0],
            lossy_weight,
            to_gate_up_kernel_scale_operand(lossy, G128_H, G128_I),
        ).to(torch.float32)
    _assert_nki_route(gate_up_dispatch_counters, sim, 1, "lossy-256 gate/up")

    expected = _g128_reference_flat(case, 0)
    max_abs_diff = float((got - expected).abs().max())
    assert max_abs_diff != 0, (
        "the lossy 256 retile reached bit equality against the 128-granular "
        "reference, so the equality does not discriminate the two granularities"
    )


def _is_wrap_nki_dispatch(callee: ast.AST) -> bool:
    """``wrap_nki(kernel)`` or ``wrap_nki(kernel)[shards]``: the seam's one dispatch."""
    if isinstance(callee, ast.Subscript):
        callee = callee.value
    return (
        isinstance(callee, ast.Call)
        and isinstance(callee.func, ast.Name)
        and callee.func.id == "wrap_nki"
    )


def test_the_gate_up_limb_raises_when_the_kernel_cannot_run() -> None:
    """With the simulator off the gate reads False and the call raises."""
    case = _build_gate_up_128_case()
    saved = os.environ.get("NKI_SIMULATOR")
    os.environ["NKI_SIMULATOR"] = "0"
    reset_gate_up_dispatch_counters()
    try:
        gate = can_run_moe_gate_up_blockwise_fp8(
            torch.zeros(1), G128_TOKENS, G128_H, G128_I
        )
        assert gate is False, (
            f"the gate read {gate!r} with NKI_SIMULATOR=0, so this case is "
            f"not exercising the refusal"
        )
        # The exception type is the substrate's to choose; what matters is that no
        # tensor computed some other way came back.
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
    assert message, "the call raised without a message, so nothing can be read"
    # The counter counts entries into the NKI route, so the attempted dispatch is
    # counted even though it raised, and the fallback counter stays 0.
    assert nki_dispatch == 1, nki_dispatch
    assert torch_fallback == 0, torch_fallback

    # At the source level: the seam returns the result of exactly one call, and that
    # call is the wrap_nki dispatch. A torch limb would have to be a second return.
    import vllm_neuron.functional.moe.moe_blockwise_fp8 as moe

    _fn, tree = moe._function_ast(moe_gate_up_blockwise_fp8)
    returns = [node for node in ast.walk(tree) if isinstance(node, ast.Return)]
    call_returns = [node for node in returns if isinstance(node.value, ast.Call)]
    wrap_returns = [
        node for node in call_returns if _is_wrap_nki_dispatch(node.value.func)
    ]
    assert len(returns) == 1, (
        f"the gate/up seam has {len(returns)} return statements; a second one is "
        f"where a torch fallback would live, and this limb has none"
    )
    assert len(wrap_returns) == 1


@pytest.mark.parametrize(
    "tokens,rows,cols,needle",
    [
        (200, 512, 512, "B=200 is not a positive multiple"),
        (0, 512, 512, "B=0 is not a positive multiple"),
        (256, 500, 512, "H=500 is not a positive multiple"),
        (256, 512, 500, "I=500 is not a positive multiple"),
    ],
)
def test_gate_up_refuses_inadmissible_geometry_by_name(
    tokens: int, rows: int, cols: int, needle: str
) -> None:
    """Every refusal names the offending extent and the loop that needs it."""
    with pytest.raises(MoeBlockwiseFp8Error) as excinfo:
        can_run_moe_gate_up_blockwise_fp8(torch.zeros(1), tokens, rows, cols)
    message = str(excinfo.value)
    assert needle in message, f"[B={tokens},H={rows},I={cols}] message was: {message}"


def test_gate_up_refuses_wrong_operands_by_name() -> None:
    """Rank, orientation, fusion width and operand shape are each refused by name."""
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

    # A `[2*I, H]` weight is the likely mistake, and reshaping it silently would
    # compute a different function. Built at the wrong shape rather than transposed
    # from the fp8 fixture, so this arm does not depend on an fp8 cast path.
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


# --------------------------------------------------------------------------- #
# The scale operand: host-side order, then the kernel's own reading of it.       #
# --------------------------------------------------------------------------- #
def test_the_scale_operand_order_is_the_checkpoint_grid_order() -> None:
    """Every block's scale sits in the column the flat index names, replicated."""
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
                expected = float(grid[0, h_block, gate_or_up, i_block])
                got = operand[:, column]
                assert float(got.min()) == expected and float(got.max()) == expected, (
                    f"block (h={h_block}, g={gate_or_up}, i={i_block}) should sit "
                    f"in column {column} as {expected}; that column reads "
                    f"[{float(got.min())}, {float(got.max())}]"
                )
                checked += 1
    assert checked == G128_BLOCKS

    # The index itself refuses a half selector it cannot place.
    with pytest.raises(MoeBlockwiseFp8Error):
        gate_up_flat_scale_index(0, GATE_UP_FUSION, 0, G128_I_BLOCKS)


def test_a_one_hot_scale_moves_only_its_own_output_columns() -> None:
    """Doubling one block's scale changes only that block's 128 output columns."""
    # Settled without a reference: a kernel and a reference that read the scale
    # tensor through the same convention would share any layout error.
    case = _build_gate_up_128_case()
    hot_h, hot_g, hot_i = 1, 1, 2
    ones = torch.ones(
        (G128_H_BLOCKS, GATE_UP_FUSION, G128_I_BLOCKS), dtype=torch.float32
    )
    doubled = ones.clone()
    doubled[hot_h, hot_g, hot_i] = 2.0
    if torch.equal(doubled, ones):
        raise VacuousCaseError("the injection changed nothing")

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
    _assert_nki_route(gate_up_dispatch_counters, sim, 2, "one-hot gate/up")

    delta = (probed - baseline).abs()
    hot_start = hot_g * G128_I + hot_i * GATE_UP_SCALE_BLOCK
    hot_stop = hot_start + GATE_UP_SCALE_BLOCK
    inside = float(delta[:, hot_start:hot_stop].max())
    outside = max(
        float(delta[:, :hot_start].max()) if hot_start > 0 else 0.0,
        float(delta[:, hot_stop:].max()) if hot_stop < delta.shape[1] else 0.0,
    )
    if inside == 0.0:
        raise VacuousCaseError(
            "doubling a block scale moved nothing, so this probe cannot identify "
            "any column range"
        )
    assert outside == 0.0, (
        f"the delta escaped the hot block's own columns "
        f"({hot_start}:{hot_stop}): outside={outside:.6e}. The kernel is reading "
        f"the scale operand in a different order than "
        f"gate_up_flat_scale_index declares."
    )


# --------------------------------------------------------------------------- #
# Each kept kernel choice is shown to change the answer.                         #
# --------------------------------------------------------------------------- #
def test_the_folded_dequantisation_changes_the_result() -> None:
    """An all-ones scale operand must move the output, or the scales never fold in."""
    case = _build_gate_up_128_case()
    ones = torch.ones(
        (G128_H_BLOCKS, GATE_UP_FUSION, G128_I_BLOCKS), dtype=torch.float32
    )
    if torch.equal(ones, case["grid"][0]):
        raise VacuousCaseError(
            "the fixture's own scale grid is all ones, so replacing it with ones "
            "changes nothing"
        )

    reset_gate_up_dispatch_counters()
    with _SimulatorCounter() as sim:
        without_scales = _gate_up_one_block(
            case["hidden"][0],
            case["weight_fused"][0],
            to_gate_up_kernel_scale_operand(ones, G128_H, G128_I),
        ).to(torch.float32)
    _assert_nki_route(gate_up_dispatch_counters, sim, 1, "all-ones gate/up")

    expected = _g128_reference_flat(case, 0)
    max_abs_diff = float((without_scales - expected).abs().max())
    if not math.isfinite(max_abs_diff):
        raise VacuousCaseError(
            f"the ones-operand run produced {max_abs_diff}, which is not a number, "
            f"so this arm cannot say whether the fold changes the result"
        )
    assert max_abs_diff != 0, (
        "the kernel produced the same numbers with an all-ones scale operand, so "
        "the checkpoint's block scales do not reach its arithmetic"
    )


@nki.jit
def _variant_128_untransposed_kernel(hidden, fused_weight, scale_operand):
    """The shipped gate/up body with one line changed: the activation is not transposed.

    A copy rather than a switch on the shipped kernel, so the shipped body carries
    no flag a caller could flip. With a plain ``nl.load`` the contraction extent
    lands on the free axis and the engine contracts the token axis instead.
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
                # The one changed line: no DMA-side transpose.
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


def test_the_activation_transpose_changes_the_result() -> None:
    """Without the DMA-side transpose the kernel computes a different function."""
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    case = _build_gate_up_128_case()
    got = wrap_nki(_variant_128_untransposed_kernel)(
        case["hidden"][0],
        case["weight_fused"][0],
        case["operands"][0],
    )
    expected = _g128_reference_flat(case, 0)
    max_abs_diff = float((got.to(torch.float32) - expected).abs().max())
    # Three outcomes: a finite non-zero difference is the expected result, a zero
    # means the transpose bought nothing, and a nan means the variant itself is
    # broken and says nothing about the transpose.
    if not math.isfinite(max_abs_diff):
        raise VacuousCaseError(
            f"the untransposed variant produced {max_abs_diff}, which is not a "
            f"number: the variant is broken, so this arm cannot say whether the "
            f"transpose changes the result"
        )
    assert max_abs_diff != 0, (
        "the untransposed variant reached bit equality, so the engine is not "
        "contracting the axis this kernel's operand orientation assumes"
    )


# --------------------------------------------------------------------------- #
# The activation and the down projection.                                       #
#                                                                               #
# The down projection carries the same exact equality as gate/up, in two arms:  #
# with every affinity at 1.0, and with exactly representable affinities so the  #
# affinity multiply sits inside the equality. The activation is compared at     #
# RTOL/ATOL on the general fixture (a device sigmoid is not reproducible bit    #
# for bit) and at bit equality on a zero-gate fixture, where SiLU(0) * up is    #
# exactly +0.0.                                                                 #
# --------------------------------------------------------------------------- #
# All exactly representable, so the affinity arm of the equality stays bit-exact.
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
    """A transposed intermediate, a down weight and an affinity column per expert."""
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
    """``[I, H]``: one scale per weight element."""
    per_block = grid[expert]
    expanded = per_block.repeat_interleave(GATE_UP_SCALE_BLOCK, dim=0)
    return expanded.repeat_interleave(GATE_UP_SCALE_BLOCK, dim=1)


def _down_reference(case: dict, expert: int, affinity_key: str) -> torch.Tensor:
    """The reference: dequantise, one matmul, then the affinity."""
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
    """``(fp64_agrees, gap, forms_agree)`` for the down projection."""
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


def test_down_matches_the_reference_per_expert_block() -> None:
    """The down projection equals the reference bit for bit, with and without gating."""
    case = _build_down_128_case()
    reset_down_dispatch_counters()

    for expert in range(E):
        exact, gap, forms_agree = _down_precondition(case, expert)
        if not exact or not forms_agree:
            raise ReferenceExactnessError(
                f"expert {expert}: exactness precondition failed "
                f"(fp64 agrees={exact}, forms agree={forms_agree}, gap={gap:.6e}), "
                f"so an equality here would measure rounding"
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
    _assert_nki_route(down_dispatch_counters, sim, 2 * E, "down")

    for (expert, key), output in outputs.items():
        got = output.to(torch.float32)
        expected = _down_reference(case, expert, key)
        assert tuple(got.shape) == (G128_TOKENS, G128_H), tuple(got.shape)
        if float(expected.abs().max()) == 0.0:
            raise VacuousCaseError(
                f"expert {expert} [{key}]: the reference is all zero"
            )
        max_abs_diff = float((got - expected).abs().max())
        assert torch.equal(got, expected), (
            f"expert {expert} [{key}]: the down projection and the reference are "
            f"not bit-equal; max_abs_diff={max_abs_diff}"
        )
        assert max_abs_diff == 0


def test_down_a_lossy_256_retile_does_not_reach_exactness() -> None:
    """The quad-maximum 256 retile cannot reach bit equality on the down grid either."""
    case = _build_down_128_case()
    grid = case["grid"]
    n_i, n_h = grid.shape[1], grid.shape[2]

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
                    raise VacuousCaseError(
                        f"down quad (expert={expert}, i={quad_i}, h={quad_h}) draws "
                        f"from {sorted(families)} only, values "
                        f"{[float(v) for v in values]}; a single-family quad "
                        f"retiles losslessly and this case would read zero for "
                        f"the wrong reason"
                    )

    # The down grid has no fusion axis, so the retile applies to it whole.
    lossy_weight, lossy, inexact_tiles = _lossy_256_retile(
        case["down_weight"][0].to(torch.float32), grid[0]
    )
    if torch.equal(lossy, grid[0]):
        raise VacuousCaseError("the lossy 256 mapping changed no down scale")
    if torch.equal(
        lossy_weight.to(torch.float32), case["down_weight"][0].to(torch.float32)
    ):
        raise VacuousCaseError(
            "the lossy 256 mapping changed no down weight byte, so it is a scale "
            "swap and not a retile, and it would move the output for every kernel"
        )
    if inexact_tiles == 0:
        raise VacuousCaseError(
            "every rescaled down tile stayed exactly on the fp8 grid, so this "
            "mapping is a lossless retile and shows nothing about the 128 grid"
        )

    reset_down_dispatch_counters()
    with _SimulatorCounter() as sim:
        got = _down_one_block(
            case["intermediate_t"][0],
            lossy_weight,
            to_down_kernel_scale_operand(lossy, G128_I, G128_H),
            case["ones"][0],
        ).to(torch.float32)
    _assert_nki_route(down_dispatch_counters, sim, 1, "lossy-256 down")

    max_abs_diff = float((got - _down_reference(case, 0, "ones")).abs().max())
    assert max_abs_diff != 0, (
        "the lossy 256 retile reached bit equality on the down projection, so this "
        "limb's equality does not discriminate the two granularities"
    )


def test_the_affinity_scaling_scales_the_down_result() -> None:
    """Changing the affinities scales the down result by exactly the per-token ratio."""
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
    _assert_nki_route(down_dispatch_counters, sim, 2, "affinity down")

    if torch.equal(case["affinity"][0], case["ones"][0]):
        raise VacuousCaseError("the two affinity columns are identical")
    moved = float((scaled - plain).abs().max())
    ratio_exact = bool(torch.equal(scaled, plain * case["affinity"][0]))
    assert moved != 0, (
        "the affinity column changed nothing, so it does not reach the kernel's "
        "arithmetic"
    )
    assert ratio_exact, (
        "the two arms differ but not by the affinity column, so the affinity is "
        "applied along the wrong axis"
    )


def test_the_activation_matches_torch_and_is_exact_where_it_can_be() -> None:
    """The activation matches torch at RTOL/ATOL and is exactly +0.0 for a zero gate."""
    gate_up = torch.empty(
        (G128_TOKENS, GATE_UP_FUSION * G128_I), dtype=torch.float32
    )
    gate_up[:, :G128_I] = _g128_fp8_values(301, G128_TOKENS, G128_I) - 0.5
    gate_up[:, G128_I:] = _g128_fp8_values(302, G128_TOKENS, G128_I)

    reset_swiglu_dispatch_counters()
    with _SimulatorCounter() as sim:
        got = moe_swiglu_transposed(gate_up).to(torch.float32)
    _assert_nki_route(swiglu_dispatch_counters, sim, 1, "activation")

    assert tuple(got.shape) == (G128_I, G128_TOKENS), (
        f"the activation seam returned {tuple(got.shape)}, expected the transposed "
        f"{(G128_I, G128_TOKENS)} its contract declares"
    )
    expected = (
        torch.nn.functional.silu(gate_up[:, :G128_I]) * gate_up[:, G128_I:]
    ).t().contiguous()
    if float(expected.abs().max()) == 0.0:
        raise VacuousCaseError("the activation reference is all zero")
    torch.testing.assert_close(got, expected, rtol=RTOL, atol=ATOL)

    # The exact arm: a zero gate makes SiLU(gate) * up exactly +0.0.
    zero_gate = gate_up.clone()
    zero_gate[:, :G128_I] = 0.0
    reset_swiglu_dispatch_counters()
    with _SimulatorCounter() as sim:
        zeros = moe_swiglu_transposed(zero_gate).to(torch.float32)
    _assert_nki_route(swiglu_dispatch_counters, sim, 1, "zero-gate activation")
    max_abs_diff = float(zeros.abs().max())
    assert torch.equal(zeros, torch.zeros_like(zeros)), max_abs_diff
    # +0.0 and not -0.0: torch.equal treats the two as equal, so the sign bit is
    # checked on its own.
    assert int(torch.signbit(zeros).sum()) == 0


@pytest.mark.parametrize(
    "tokens,rows,cols,needle",
    [
        (200, 512, 512, "B=200 is not a positive multiple"),
        (256, 500, 512, "I=500 is not a positive multiple"),
        (256, 512, 500, "H=500 is not a positive multiple"),
    ],
)
def test_down_refuses_inadmissible_geometry_by_name(
    tokens: int, rows: int, cols: int, needle: str
) -> None:
    """Every down refusal names the offending extent."""
    with pytest.raises(MoeBlockwiseFp8Error) as excinfo:
        can_run_moe_down_blockwise_fp8(torch.zeros(1), tokens, rows, cols)
    assert needle in str(excinfo.value), str(excinfo.value)


def test_down_and_activation_refuse_wrong_operands_by_name() -> None:
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

    # H and I are both 512 here, so the transpose of this case's own [I, H] weight
    # has the same shape as the weight itself. Doubling the contraction separates
    # the two orientations: [2*H, I] is the shape a valid [I, 2*H] weight would have
    # if it were handed over transposed.
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
    # This call carried the correctly oriented weight and reached the affinity
    # guard, so the orientation message above is that guard's own and not one every
    # refused call carries.
    assert "contraction-major" not in str(columns.value)

    with pytest.raises(MoeBlockwiseFp8Error) as grid:
        to_down_kernel_scale_operand(
            case["grid"][0][:, :-1].contiguous(), G128_I, G128_H
        )
    assert "mis-sized" in str(grid.value)

    with pytest.raises(MoeBlockwiseFp8Error) as activation:
        moe_swiglu_transposed(torch.zeros((G128_TOKENS, 3), dtype=torch.float32))
    assert "GATE_UP_FUSION" in str(activation.value)
