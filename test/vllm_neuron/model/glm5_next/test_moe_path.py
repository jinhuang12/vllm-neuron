# SPDX-License-Identifier: Apache-2.0
"""The blockwise-FP8 MoE call site: routing, expert bank, shared expert, reduction.

Every arm compares against a torch reference built in the same file and counts the
kernel dispatches the path is supposed to make.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import torch

import nki
import nki.simulator

from vllm_neuron.functional.moe.blockwise_fp8_retile import (
    BLOCK_QUANT_SIZE,
    DOWN,
    GATE_UP,
    I_TILES_PER_BLOCK,
    TILE_SIZE,
    consumer_scale_shape,
    flat_scale_index,
    is_pow2_exact,
    retile_block_scales,
)
from vllm_neuron.functional.moe.moe_blockwise_fp8 import (
    blockwise_fp8_moe,
    blockwise_fp8_moe_torch_oracle,
    dispatch_counters,
    down_dispatch_counters,
    down_kernel_identity,
    gate_up_dispatch_counters,
    gate_up_kernel_identity,
    kernel_scale_shape,
    reset_down_dispatch_counters,
    reset_dispatch_counters,
    reset_gate_up_dispatch_counters,
    reset_swiglu_dispatch_counters,
    swiglu_dispatch_counters,
    swiglu_kernel_identity,
    to_down_kernel_scale_operand,
    to_gate_up_kernel_scale_operand,
)
from vllm_neuron.functional.moe.moe_blockwise_fp8 import (
    to_kernel_scale_layout as moe_to_kernel_scale_layout,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

#: The block-quant seam module. The attribution target of instrument 4, resolved as a
#: file path off the imported module rather than spelled as a string, so a moved
#: module fails the import instead of silently making the attribution unmatchable.
import vllm_neuron.functional.moe.moe_blockwise_fp8 as _seam_module

_SEAM_FILE = _seam_module.__file__

# ---------------------------------------------------------------------------
# The tiny config. Every extent is forced, and by what is cited next to it.
# ---------------------------------------------------------------------------

#: ``H`` and ``I_TP`` are the smallest pair the seam's five admission gates
#: accept (``moe_blockwise_fp8.py:_require_blocked``): ``H % 256 == 0``,
#: ``512 <= H <= 8192``, ``H % PSUM_SIZE(512) == 0``, ``I_TP % 256 == 0`` and
#: ``I_TP % (256 * NUM_SHARDS) == 0``. Chosen to admit the kernel, which is
#: the kernel's carry: a geometry it refuses raises rather than falling
#: back, so an inadmissible tiny config would measure the refusal path.
H = 512
I_TP = 512

#: This rank's local expert count. ``4`` rather than ``2`` so that
#: ``block_to_expert`` is not the identity and a permuted block-to-expert
#: mapping is observable, and so ``num_blocks`` exceeds the number of occupied
#: blocks -- the empty-block case the vendor kernel handles by indexing the
#: padding slot. Both facts are measured in
#: :func:`test_moe_path_mapping_shape_is_the_one_the_seam_consumes`.
E = 4

#: Experts per token. ``2`` keeps every expert's occupancy at ``T * K / E = 128``
#: tokens, comfortably inside one ``block_size`` so the mapping needs exactly one
#: block per expert.
K = 2

#: Real tokens the call site is handed. ``256`` is the extent
#: ``build_blockwise_mapping`` sees, because the call site builds the mapping over
#: the real token count and appends the kernel's padding slot afterwards. Both of
#: the mapping's kernel gates turn on that extent being even, so the order is
#: load-bearing; see
#: :func:`test_moe_path_call_site_maps_before_padding_and_dispatches_nki`, which
#: measures it, and
#: :func:`test_moe_path_mapping_order_moves_no_numbers`, which measures that the
#: two orders produce the same tensors.
T = 256

#: Tokens per block. ``256`` is the vendor kernel's ``B % 256 == 0`` assert
#: (``bwmm_shard_on_I.py``) at its minimum.
B = 256

H_256 = H // BLOCK_QUANT_SIZE
I_256 = I_TP // BLOCK_QUANT_SIZE

#: The declared tolerance, order named inline. Never widened here:
#: widening a declared value is the user's election through the lead, not this
#: file's.
RTOL = 3e-2
ATOL = 1e-5


#: The declared dispatch count for the route arm: ``1/1`` calls.
#:
#: what this file declares and what it does not. This file declares the seam reading
#: only: one dispatch, no torch fallback, and the attributed share equal to the
#: dispatch count, in ``1/1`` calls. No total simulator-entry count and no mapping
#: dispatch count anywhere, so the total clause inside :func:`_assert_route` is this
#: file's instrument against a vacuous pass: at least one entry, then every entry
#: attributed to a measured source.
#:
#: the reading is now per limb, one ``(nki_dispatch, torch_fallback)`` pair for the
#: gate/up, activation and down limbs in launch order: the routed composition runs
#: three kernels for a whole MoE layer. A triple rather than a sum, deliberately --
#: a summed ``3`` is also what one limb dispatching three times reads, which is the
#: false green this file's counter clause exists to exclude.
DECLARED_LIMB_DISPATCHES = ((1, 0), (1, 0), (1, 0))

#: The same reading when the route was refused before any limb was entered.
NO_LIMB_DISPATCHES = ((0, 0), (0, 0), (0, 0))

_FP8 = torch.float8_e4m3fn

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "config.json"

#: Fixture-only constants. This file declares the tolerances, the case count and
#: the tiny-config shape family; it declares nothing about how the fixture is
#: built, so these are labelled as this file's own choices.
WEIGHT_SEED_GATE = 271
WEIGHT_SEED_UP = 272
WEIGHT_SEED_DOWN = 273
HIDDEN_SEED = 274
SCALE_SEED_GATE = 275
SCALE_SEED_UP = 276
SCALE_SEED_DOWN = 277

#: Dyadic affinity values -- multiples of ``1/8``, hence exact in bf16, which is
#: the dtype the call site casts the masked affinities to before the kernel
#: multiplies them into the hidden states. Five values, not four: with a table
#: length equal to ``E`` the value index would collapse onto the expert index
#: and every one of an expert's 128 tokens would carry the same affinity, so a
#: token-permuted affinity assignment would be invisible. Five is coprime to
#: ``E = 4``, which decouples the two.
AFFINITY_VALUES = (0.5, 0.25, 0.375, 0.75, 0.625)

#: Tokens each expert must receive, by construction. Asserted, not assumed.
TOKENS_PER_EXPERT = T * K // E


# ---------------------------------------------------------------------------
# Named errors. A failure must say which instrument disagreed; a bare
# ``AssertionError`` from three different causes is one message for three bugs.
# ---------------------------------------------------------------------------
class RouteInstrumentError(AssertionError):
    """A route reading that is not what this file declares."""


class VacuousControlError(AssertionError):
    """A control whose input could not have made it fail. """


class ReferenceShapeError(AssertionError):
    """The comparator was handed an operand of a shape it cannot mean."""


class ExportSurfaceError(AssertionError):
    """The export hub does not resolve, or resolves a colliding name."""


class FixtureConditioningError(AssertionError):
    """The fixture is not conditioned the way the tolerance is measured against."""


# ---------------------------------------------------------------------------
# Instrument 4, with attribution.
# ---------------------------------------------------------------------------
class _AttributedSimulatorCounter:
    """Counts ``nki.simulator.simulate_kernel`` entries and attributes each one. """

    def __init__(self, seam_file: str = _SEAM_FILE) -> None:
        self.total = 0
        self.through_seam = 0
        self.elsewhere = 0
        self._seam_file = os.path.realpath(seam_file)
        self._real = None

    def __enter__(self) -> "_AttributedSimulatorCounter":
        self._real = nki.simulator.simulate_kernel
        real = self._real
        seam_file = self._seam_file

        def counting(*args, **kwargs):
            self.total += 1
            frame = sys._getframe(1)
            attributed = False
            while frame is not None:
                if os.path.realpath(frame.f_code.co_filename) == seam_file:
                    attributed = True
                    break
                frame = frame.f_back
            if attributed:
                self.through_seam += 1
            else:
                self.elsewhere += 1
            return real(*args, **kwargs)

        nki.simulator.simulate_kernel = counting
        return self

    def __exit__(self, *exc_info) -> None:
        nki.simulator.simulate_kernel = self._real


def _reset_limb_counters() -> None:
    """Zero the three limb counters and the vendor seam's, in the same window."""
    reset_gate_up_dispatch_counters()
    reset_swiglu_dispatch_counters()
    reset_down_dispatch_counters()
    reset_dispatch_counters()


def _limb_counters() -> tuple[tuple[int, int], ...]:
    """The three limbs' ``(nki_dispatch, torch_fallback)`` pairs, in launch order."""
    return (
        gate_up_dispatch_counters(),
        swiglu_dispatch_counters(),
        down_dispatch_counters(),
    )


def _assert_route(
    sim: _AttributedSimulatorCounter,
    expected: tuple[tuple[int, int], ...],
    label: str,
    *,
    mapping_count: int,
) -> str:
    """Read all four route instruments and return the reading."""
    limbs = _limb_counters()
    vendor = dispatch_counters()
    expected_total = sum(dispatches for dispatches, _fallback in expected)
    gate = can_run_kernel(torch.zeros(1))
    reading = (
        f"[{label}] limb_counters={limbs} vendor_seam_counters={vendor} "
        f"declared_limb_counters={expected} can_run_kernel={gate} "
        f"simulate_kernel_total={sim.total} "
        f"simulate_kernel_through_025_seam={sim.through_seam} "
        f"simulate_kernel_elsewhere={sim.elsewhere} "
        f"measured_mapping_count={mapping_count} "
        f"attribution_sum={sim.through_seam + mapping_count}"
    )
    if limbs != expected:
        raise RouteInstrumentError(
            f"{label}: the limb counters read {limbs}, declared {expected}. A limb "
            f"short means the composition reached it some other way; a nonzero "
            f"fallback means a limb grew a torch path. {reading}"
        )
    if vendor != (0, 0):
        raise RouteInstrumentError(
            f"{label}: the vendor block seam's counters read {vendor}, declared "
            f"(0, 0). The routed composition does not enter that seam, so any "
            f"reading here means the call site went back to it. {reading}"
        )
    if gate is not True:
        raise RouteInstrumentError(
            f"{label}: can_run_kernel() read {gate!r}, declared True. {reading}"
        )
    if sim.total < 1:
        raise RouteInstrumentError(
            f"{label}: nki.simulator.simulate_kernel ran {sim.total} times, so "
            f"no kernel was simulated at all. A numeric pass without a simulator "
            f"entry is the vacuous pass this arm screens for. {reading}"
        )
    if sim.total != sim.through_seam + mapping_count:
        raise RouteInstrumentError(
            f"{label}: {sim.total} simulator entries do not add up. "
            f"{sim.through_seam} attributed to the block-quant seam plus "
            f"{mapping_count} measured for the token-block mapping alone is "
            f"{sim.through_seam + mapping_count}. Either some component nobody "
            f"measured dispatched, or the call site handed the mapping an extent "
            f"whose NKI gates refuse -- which is what padding before mapping "
            f"does. {reading}"
        )
    if sim.through_seam != expected_total:
        raise RouteInstrumentError(
            f"{label}: {sim.through_seam} of {sim.total} simulator entries were "
            f"attributed to the limb module ({_SEAM_FILE}), declared "
            f"{expected_total}. "
            f"An unattributed dispatch means some OTHER component produced it, "
            f"which is not what R-2 counts. {reading}"
        )
    return reading


def _run_mapping(expert_affinities: torch.Tensor, label: str) -> dict:
    """Run the token-block mapping alone under the file's own counter. """
    from vllm_neuron.functional import build_blockwise_mapping

    with _AttributedSimulatorCounter() as mapping_sim:
        masked, token_position_to_id, block_to_expert, conditions = (
            build_blockwise_mapping(
                expert_affinities=expert_affinities,
                num_local_experts=E,
                num_experts_per_token=K,
                block_size=B,
                moe_group=None,
                tp_degree=1,
            )
        )
    if mapping_sim.through_seam != 0:
        raise RouteInstrumentError(
            f"mapping-{label}: {mapping_sim.through_seam} of "
            f"{mapping_sim.total} entries produced by the token-block mapping "
            f"were attributed to the block-quant seam ({_SEAM_FILE}). The mapping is a "
            f"different file, so this reading would make the attribution "
            f"identity double-count."
        )
    return {
        "masked": masked,
        "token_position_to_id": token_position_to_id,
        "block_to_expert": block_to_expert,
        "conditions": conditions,
        "total": mapping_sim.total,
    }


def _measure_mapping_count(expert_affinities: torch.Tensor, label: str) -> int:
    """The mapping's measured simulator entry count, for the attribution identity."""
    return _run_mapping(expert_affinities, label)["total"]


def _envelope_affinities(tokens: int) -> torch.Tensor:
    """This file's fixture scatter pattern, at an arbitrary token count. """
    out = torch.zeros(tokens, E, dtype=torch.float32)
    for token in range(tokens):
        for slot in range(K):
            out[token, (token + slot) % E] = AFFINITY_VALUES[
                (token + slot) % len(AFFINITY_VALUES)
            ]
    return out


# ---------------------------------------------------------------------------
# The quantisation policy, read through the real recognition path.
# ---------------------------------------------------------------------------
def _pinned_raw_config() -> dict:
    """The checkpoint config the recognition path is driven with."""
    return json.loads(FIXTURE_PATH.read_text())


def _block_quant_config():
    """``Glm5NextQuantConfig`` for the pinned checkpoint -- nothing hand-fed. """
    from vllm_neuron.model.glm5_next.config import Glm5NextConfig
    from vllm_neuron.model.glm5_next.model_fp8 import Glm5NextQuantConfig

    return Glm5NextQuantConfig.from_model_config(
        Glm5NextConfig.from_configs(_pinned_raw_config())
    )


def _build_bank():
    """The routed-expert bank at this rank's tiny partition, plus its config."""
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig
    from vllm_neuron.model.glm5_next.model_fp8 import Glm5NextRoutedExperts

    text_config = Glm5NextTextConfig(
        hidden_size=H,
        moe_intermediate_size=I_TP,
        n_routed_experts=E,
        num_experts_per_tok=K,
    )
    bank = Glm5NextRoutedExperts(text_config, world_size=1)
    if int(bank.num_local_experts) != E:
        raise VacuousControlError(
            f"the bank reports num_local_experts={bank.num_local_experts}, but "
            f"the fixture is built for {E}; the call site's own extent check "
            f"would fire before any numerics ran"
        )
    return bank, text_config


# ---------------------------------------------------------------------------
# The fixture. Built through the retile producer, so the
# producer is exercised rather than mimicked, and the scales the comparator
# reads are the ones the kernel is handed.
# ---------------------------------------------------------------------------
def _pow2_checkpoint_scales(seed: int, rows: int, cols: int) -> torch.Tensor:
    """``(E, rows//128, cols//128)`` fp32 scales, every entry an exact power of 2. """
    generator = torch.Generator().manual_seed(seed)
    exponents = torch.randint(
        -3, 4, (E, rows // TILE_SIZE, cols // TILE_SIZE), generator=generator
    )
    return torch.ldexp(torch.ones_like(exponents, dtype=torch.float32), exponents)


def _fp8_grid_values(seed: int, *shape: int) -> torch.Tensor:
    """Values already on the fp8-e4m3 grid: multiples of ``1/8`` in ``[1/8, 7/8]``. """
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(1, 8, shape, generator=generator).to(torch.float32) / 8.0


def _vendor_256_pair(
    weights: torch.Tensor,
    scales: torch.Tensor,
    projection: str,
    gate_or_up: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The vendor seam's ``256``-granular pair, coarsened here and by no module. """
    experts, rows, cols = weights.shape
    h_256, i_256 = rows // BLOCK_QUANT_SIZE, cols // BLOCK_QUANT_SIZE
    flat = torch.full(
        consumer_scale_shape(experts, rows, cols, projection),
        float("nan"),
        dtype=torch.float32,
    )
    rescaled = weights.to(torch.float32).clone()
    for expert in range(experts):
        for h_tile in range(rows // TILE_SIZE):
            for i_tile in range(cols // TILE_SIZE):
                kept = float(
                    scales[
                        expert,
                        h_tile - h_tile % I_TILES_PER_BLOCK,
                        i_tile - i_tile % I_TILES_PER_BLOCK,
                    ]
                )
                ratio = float(scales[expert, h_tile, i_tile]) / kept
                window = (
                    expert,
                    slice(h_tile * TILE_SIZE, (h_tile + 1) * TILE_SIZE),
                    slice(i_tile * TILE_SIZE, (i_tile + 1) * TILE_SIZE),
                )
                rescaled[window] = (rescaled[window] * ratio).to(_FP8).to(torch.float32)
                block = flat_scale_index(
                    h_tile, i_tile, h_256, i_256, projection, gate_or_up
                )
                flat[expert, block * TILE_SIZE : (block + 1) * TILE_SIZE] = kept
    assert bool(torch.isfinite(rescaled).all()), (
        "a rescaled byte left the fp8 range, so this fixture measures saturation "
        "rather than the seam"
    )
    return rescaled.to(_FP8), flat


def _build_case() -> dict:
    """The tiny config in the form the call site takes -- not the seam's form. """
    # --- gate/up: the mapping runs per fusion half, on (E, H, I_TP). -------- #
    # the two halves are merged in the flat domain, which is legitimate because
    # ``moe_to_kernel_scale_layout`` is a documented C-order reshape
    # (``moe_to_kernel_scale_layout``'s own docstring): a ``view`` onto the shape
    # addresses exactly the elements that reshape would.
    gup_flat = torch.full(
        consumer_scale_shape(E, H, I_TP, GATE_UP), float("nan"), dtype=torch.float32
    )
    gup_logical_view = gup_flat.view(*kernel_scale_shape(E, H, I_TP, GATE_UP))
    gup_weight = torch.empty((E, H, 2, I_TP), dtype=torch.float32)
    gup_original = torch.empty((E, H, 2, I_TP), dtype=torch.float32)
    gup_grid = torch.empty(
        (E, H // TILE_SIZE, 2, I_TP // TILE_SIZE), dtype=torch.float32
    )
    for gate_or_up, (weight_seed, scale_seed) in enumerate(
        ((WEIGHT_SEED_GATE, SCALE_SEED_GATE), (WEIGHT_SEED_UP, SCALE_SEED_UP))
    ):
        checkpoint = _pow2_checkpoint_scales(scale_seed, H, I_TP)
        weights = _fp8_grid_values(weight_seed, E, H, I_TP)
        retiled_weights, consumer_scales = _vendor_256_pair(
            weights.to(_FP8), checkpoint, GATE_UP, gate_or_up
        )
        gup_weight[:, :, gate_or_up, :] = retiled_weights.to(torch.float32)
        gup_original[:, :, gate_or_up, :] = weights
        gup_grid[:, :, gate_or_up, :] = checkpoint
        bridged = moe_to_kernel_scale_layout(
            consumer_scales, E, H, I_TP, projection=GATE_UP
        )
        # The mapping writes only this half's slots and leaves the other half
        # NaN on purpose, so take this half's slice. Merging the two is the
        # weight loader's step in production; here it is the fixture's.
        gup_logical_view[:, :, gate_or_up, :, :] = bridged[:, :, gate_or_up, :, :]

    # NaN survives arithmetic, so an unwritten slot would poison the output
    # rather than pass quietly -- but a poisoned output fails the numeric arm
    # for a fixture reason. Refuse here, where the cause is legible.
    if not bool(torch.isfinite(gup_flat).all()):
        raise VacuousControlError(
            f"{int((~torch.isfinite(gup_flat)).sum())} gate/up scale slots are "
            f"still NaN after both fusion halves were written; the fixture, not "
            f"the call site, is wrong"
        )

    # --- down: the mapping's ``rows`` is the H axis and ``cols`` the I axis, ---
    # --- so the physically-[E, I_TP, H] weight is coarsened in (E, H, I_TP).
    down_checkpoint = _pow2_checkpoint_scales(SCALE_SEED_DOWN, H, I_TP)
    down_weight_hi = _fp8_grid_values(WEIGHT_SEED_DOWN, E, H, I_TP)
    down_retiled, down_flat = _vendor_256_pair(
        down_weight_hi.to(_FP8), down_checkpoint, DOWN
    )
    if not bool(torch.isfinite(down_flat).all()):
        raise VacuousControlError(
            "down-projection scale slots contain NaN; one mapping pass covers "
            "every DOWN slot, so a NaN here is a fixture defect"
        )
    # Back to the kernel's physical [E, I_TP, H].
    down_weight = down_retiled.to(torch.float32).transpose(1, 2).contiguous()

    # --- activations and router scores, in the call site's own form. -------- #
    hidden = _fp8_grid_values(HIDDEN_SEED, T, H).to(torch.bfloat16)
    affinities = torch.zeros(T, E, dtype=torch.float32)
    for token in range(T):
        for slot in range(K):
            expert = (token + slot) % E
            affinities[token, expert] = AFFINITY_VALUES[(token + slot) % len(AFFINITY_VALUES)]

    down_original = down_weight_hi.transpose(1, 2).contiguous()
    down_grid = down_checkpoint.transpose(1, 2).contiguous()
    gup_operands = torch.stack(
        [to_gate_up_kernel_scale_operand(gup_grid[expert], H, I_TP) for expert in range(E)]
    )
    down_operands = torch.stack(
        [to_down_kernel_scale_operand(down_grid[expert], I_TP, H) for expert in range(E)]
    )

    return {
        "call_site_inputs": dict(
            hidden_states=hidden,
            expert_affinities=affinities,
            gate_up_proj_weight=gup_original.to(_FP8),
            down_proj_weight=down_original.to(_FP8),
            gate_up_scale_operands=gup_operands,
            down_scale_operands=down_operands,
        ),
        "gup_grid": gup_grid,
        "down_grid": down_grid,
        "gup_retiled": gup_weight.to(_FP8),
        "down_retiled": down_weight.to(_FP8),
        "gup_logical": gup_logical_view.clone(),
        "down_logical": moe_to_kernel_scale_layout(
            down_flat, E, H, I_TP, projection=DOWN
        ),
    }


# ---------------------------------------------------------------------------
# The comparator: a pure-torch reference MoE over the same weights and scores.
# ---------------------------------------------------------------------------
def _dequantise(weight_fp8: torch.Tensor, block_scale: torch.Tensor) -> torch.Tensor:
    """``weight[k, n] * scale[k // g, n // g]``, expanded, in fp32. """
    if weight_fp8.dim() != 2 or block_scale.dim() != 2:
        raise ReferenceShapeError(
            f"expected 2-D weight and 2-D block scale, got "
            f"{tuple(weight_fp8.shape)} and {tuple(block_scale.shape)}"
        )
    rows, cols = weight_fp8.shape
    # The granularity is read off the scale, not named here. This reference now serves
    # the checkpoint's own [128, 128] grid, and naming one number would make the
    # function silently wrong for the other rather than refuse.
    block_rows = rows // int(block_scale.shape[0]) if int(block_scale.shape[0]) else 0
    block_cols = cols // int(block_scale.shape[1]) if int(block_scale.shape[1]) else 0
    if (block_rows, block_cols) == (0, 0) or tuple(block_scale.shape) != (
        rows // block_rows,
        cols // block_cols,
    ):
        raise ReferenceShapeError(
            f"block scale {tuple(block_scale.shape)} does not tile a "
            f"{rows}x{cols} weight at any whole granularity"
        )
    if block_rows != block_cols:
        raise ReferenceShapeError(
            f"block scale {tuple(block_scale.shape)} tiles {rows}x{cols} only with "
            f"unequal blocks {block_rows}x{block_cols}; both this checkpoint's grid "
            f"and the consumer's are square"
        )
    expanded = block_scale.repeat_interleave(block_rows, dim=0).repeat_interleave(
        block_cols, dim=1
    )
    return weight_fp8.to(torch.float32) * expanded


def torch_reference_moe(
    hidden_states: torch.Tensor,
    expert_affinities: torch.Tensor,
    gate_up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    gate_up_block_scale: torch.Tensor,
    down_block_scale: torch.Tensor,
    *,
    swiglu_limit: float,
    post_scale: bool = True,
    clamp: bool = True,
) -> torch.Tensor:
    """A pure-torch block-quant MoE. """
    if hidden_states.dim() != 2 or expert_affinities.dim() != 2:
        raise ReferenceShapeError(
            f"expected [T, H] hidden and [T, E] affinities, got "
            f"{tuple(hidden_states.shape)} and {tuple(expert_affinities.shape)}"
        )
    tokens, hidden = hidden_states.shape
    num_experts = expert_affinities.shape[1]
    if expert_affinities.shape[0] != tokens:
        raise ReferenceShapeError(
            f"affinities cover {expert_affinities.shape[0]} tokens, hidden "
            f"states cover {tokens}"
        )
    output = torch.zeros(tokens, hidden, dtype=torch.float32)
    for expert in range(num_experts):
        gate_weight = _dequantise(
            gate_up_proj_weight[expert, :, 0, :],
            gate_up_block_scale[expert, :, 0, :],
        )
        up_weight = _dequantise(
            gate_up_proj_weight[expert, :, 1, :],
            gate_up_block_scale[expert, :, 1, :],
        )
        down_weight = _dequantise(
            down_proj_weight[expert], down_block_scale[expert]
        )
        rows = torch.nonzero(expert_affinities[:, expert], as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        scale = expert_affinities[rows, expert].to(hidden_states.dtype).unsqueeze(1)
        if post_scale:
            local = hidden_states[rows].to(torch.float32)
        else:
            local = (scale * hidden_states[rows]).to(torch.float32)
        gate_act = local @ gate_weight
        up_act = local @ up_weight
        if clamp:
            # The model's own asymmetry, not a tidier symmetric bound
            # (``modeling_glm5_next.py``).
            gate_act = gate_act.clamp(min=None, max=swiglu_limit)
            up_act = up_act.clamp(min=-swiglu_limit, max=swiglu_limit)
        intermediate = torch.nn.functional.silu(gate_act) * up_act
        contribution = intermediate @ down_weight
        if post_scale:
            contribution = contribution * scale.to(torch.float32)
        output[rows] += contribution.to(torch.bfloat16).to(torch.float32)
    return output


def _configured_reference(case: dict, bank) -> torch.Tensor:
    """The comparator on this case's operands, at the call site's two overrides."""
    return torch_reference_moe(
        hidden_states=case["call_site_inputs"]["hidden_states"],
        expert_affinities=case["call_site_inputs"]["expert_affinities"],
        gate_up_proj_weight=case["call_site_inputs"]["gate_up_proj_weight"],
        down_proj_weight=case["call_site_inputs"]["down_proj_weight"],
        gate_up_block_scale=case["gup_grid"],
        down_block_scale=case["down_grid"],
        swiglu_limit=bank.swiglu_limit,
        post_scale=True,
        clamp=True,
    )


def _nonempty_or_raise(reference: torch.Tensor, label: str) -> int:
    """Refuse a comparison whose reference is all zeros (the control)."""
    nonzero_rows = int((reference.abs().sum(-1) > 0).sum())
    if nonzero_rows == 0:
        raise VacuousControlError(
            f"{label}: every reference row is zero, so assert_close would pass "
            f"on a function that returns zeros"
        )
    return nonzero_rows


class _TorchLimbs:
    """The three limbs' contracts computed in torch. """

    def __init__(self, gate_up_grid: torch.Tensor, down_grid: torch.Tensor) -> None:
        self.gate_up_grid = gate_up_grid
        self.down_grid = down_grid

    @staticmethod
    def _rows(row_index: torch.Tensor, pad_row: int) -> torch.Tensor:
        """Block-order row addresses with the mapping's ``-1`` resolved to the pad row."""
        rows = row_index.reshape(-1).long()
        return torch.where(rows < 0, torch.full_like(rows, pad_row), rows)

    def gate_up(
        self, hidden_states: torch.Tensor, weight_bank: torch.Tensor,
        scale_bank: torch.Tensor, row_index: torch.Tensor,
        expert_index: torch.Tensor, block: int,
    ) -> torch.Tensor:
        """``[T + 1, H]`` in, ``[P, 2*I]`` fp32 out, block order."""
        experts, contraction = int(weight_bank.shape[0]), int(weight_bank.shape[1])
        fused = weight_bank.reshape(experts, contraction, -1).shape[2]
        half = fused // 2
        positions = int(row_index.shape[0])
        rows = self._rows(row_index, int(hidden_states.shape[0]) - 1)
        out = torch.zeros(positions, fused, dtype=torch.float32)
        for position in range(positions // block):
            expert = int(expert_index.reshape(-1)[position])
            span = slice(position * block, (position + 1) * block)
            local = hidden_states[rows[span]].to(torch.float32)
            slab = weight_bank[expert].reshape(contraction, 2, half)
            for half_index in range(2):
                grid = self.gate_up_grid[expert, :, half_index, :]
                weight = _dequantise(slab[:, half_index, :], grid)
                columns = slice(half_index * half, (half_index + 1) * half)
                out[span, columns] = local @ weight
        return out

    @staticmethod
    def swiglu(
        gate_up: torch.Tensor, gate_upper: float | None = None,
        up_upper: float | None = None,
    ) -> torch.Tensor:
        """``[B, 2*I]`` in, ``[I, B]`` fp32 out. The transpose is the limb's contract."""
        half = int(gate_up.shape[1]) // 2
        gate_act = gate_up[:, :half].to(torch.float32)
        up_act = gate_up[:, half:].to(torch.float32)
        if gate_upper is not None:
            gate_act = gate_act.clamp(min=None, max=float(gate_upper))
        if up_upper is not None:
            up_act = up_act.clamp(min=-float(up_upper), max=float(up_upper))
        intermediate = torch.nn.functional.silu(gate_act) * up_act
        return intermediate.transpose(0, 1).contiguous()

    def down(
        self, intermediate_t: torch.Tensor, weight_bank: torch.Tensor,
        scale_bank: torch.Tensor, affinity_bank: torch.Tensor,
        row_index: torch.Tensor, expert_index: torch.Tensor, block: int,
        tokens: int,
    ) -> torch.Tensor:
        """``[I, P]`` in, ``[P, H]`` fp32 out, block order, affinity applied here."""
        experts, cols = int(weight_bank.shape[0]), int(weight_bank.shape[2])
        positions = int(intermediate_t.shape[1])
        rows = self._rows(row_index, tokens)
        affinity = affinity_bank.reshape(-1, experts)
        out = torch.zeros(positions, cols, dtype=torch.float32)
        for position in range(positions // block):
            expert = int(expert_index.reshape(-1)[position])
            span = slice(position * block, (position + 1) * block)
            weight = _dequantise(weight_bank[expert], self.down_grid[expert])
            local = intermediate_t[:, span].transpose(0, 1).to(torch.float32)
            scale = affinity[rows[span], expert].to(torch.float32).unsqueeze(1)
            out[span] = (local @ weight) * scale
        return out


# ===========================================================================
# the declared acceptance case. Both check, one call, 1/1.
# ===========================================================================
def test_moe_path_output_matches_pure_torch_reference() -> None:
    """The MoE call site's output vs a pure-torch reference, and the route. """
    bank, _text_config = _build_bank()
    quant_config = _block_quant_config()
    case = _build_case()

    _reset_limb_counters()
    with _AttributedSimulatorCounter() as sim:
        got = bank.block_quant_expert_mm(
            quant_config=quant_config, block_size=B, **case["call_site_inputs"]
        )
    # The mapping's own share, measured on the same inputs the call site hands it
    # (the real token count -- the padding slot is appended after the mapping),
    # under the same instrument, in this same run. A reading, never a literal.
    mapping_count = _measure_mapping_count(
        case["call_site_inputs"]["expert_affinities"], "acceptance"
    )
    _assert_route(
        sim, DECLARED_LIMB_DISPATCHES, "acceptance", mapping_count=mapping_count
    )

    if tuple(got.shape) != (T, H):
        raise ReferenceShapeError(
            f"the call site returned {tuple(got.shape)}, declared ({T}, {H}) -- "
            f"the padding-token row must be sliced off"
        )

    # The reference is configured from the bank the call site used, so the two
    # sides cannot disagree about the bound by construction.
    want = _configured_reference(case, bank)
    got_f32 = got.to(torch.float32)

    torch.testing.assert_close(got_f32, want, rtol=RTOL, atol=ATOL)


# ===========================================================================
# the comparator's own provenance, at the same declared tolerances.
# ===========================================================================
def test_moe_path_reference_agrees_with_vendor_torch_oracle() -> None:
    """This file's reference vs ``nkilib``'s own, on the same case. """
    from vllm_neuron.functional import build_blockwise_mapping
    from vllm_neuron.functional.moe.moe_blockwise_fp8 import ExpertAffinityScaleMode

    bank, _text_config = _build_bank()
    limit = float(bank.swiglu_limit)
    case = _build_case()
    inputs = case["call_site_inputs"]
    hidden = inputs["hidden_states"]
    affinities = inputs["expert_affinities"]

    padded_hidden = torch.cat(
        [hidden, torch.zeros(1, H, dtype=hidden.dtype)], dim=0
    )
    padded_affinities = torch.cat(
        [affinities, torch.zeros(1, E, dtype=affinities.dtype)], dim=0
    )
    masked, token_position_to_id, block_to_expert, _conditions = (
        build_blockwise_mapping(
            expert_affinities=padded_affinities,
            num_local_experts=E,
            num_experts_per_token=K,
            block_size=B,
            moe_group=None,
            tp_degree=1,
        )
    )
    def _oracle(**configuration) -> torch.Tensor:
        return blockwise_fp8_moe_torch_oracle(
            hidden_states=padded_hidden,
            expert_affinities_masked=masked.to(hidden.dtype),
            gate_up_proj_weight=case["gup_retiled"],
            down_proj_weight=case["down_retiled"],
            block_size=B,
            token_position_to_id=token_position_to_id,
            block_to_expert=block_to_expert.reshape(-1, 1),
            gate_up_proj_scale=case["gup_logical"],
            down_proj_scale=case["down_logical"],
            **configuration,
        ).to(torch.float32)[:T]

    def _mine(**configuration) -> torch.Tensor:
        return torch_reference_moe(
            hidden_states=hidden,
            expert_affinities=affinities,
            gate_up_proj_weight=case["gup_retiled"],
            down_proj_weight=case["down_retiled"],
            gate_up_block_scale=case["gup_logical"][..., 0],
            down_block_scale=case["down_logical"][..., 0],
            swiglu_limit=limit,
            **configuration,
        )

    vendor_default = _oracle()
    vendor_configured = _oracle(
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
        gate_clamp_upper_limit=limit,
        gate_clamp_lower_limit=None,
        up_clamp_upper_limit=limit,
        up_clamp_lower_limit=-limit,
    )
    separation = float((vendor_configured - vendor_default).abs().max())
    if not separation > 0.0:
        raise VacuousControlError(
            f"the vendor oracle returned the same numbers with and without the "
            f"call site's scaling mode and four clamp limits (max abs difference "
            f"{separation!r}), so those keywords were accepted and ignored and "
            f"the configured comparison below would pass on arithmetic that is "
            f"not the configured arithmetic"
        )
    _nonempty_or_raise(vendor_default, "reference-provenance")
    _nonempty_or_raise(vendor_configured, "reference-provenance-configured")

    mine_default = _mine(post_scale=False, clamp=False)
    mine_configured = _mine(post_scale=True, clamp=True)
    torch.testing.assert_close(mine_default, vendor_default, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(
        mine_configured, vendor_configured, rtol=RTOL, atol=ATOL
    )


# ===========================================================================
# The override instrument: each of the call site's two overrides, reverted
# one at a time, must turn the acceptance comparison red.
# ===========================================================================
def test_moe_path_f1_pre_scale_unclamped_reference_must_fail() -> None:
    """measured: revert either override in the reference and the numbers diverge. """
    bank, _text_config = _build_bank()
    quant_config = _block_quant_config()
    case = _build_case()
    limit = float(bank.swiglu_limit)

    _reset_limb_counters()
    got = bank.block_quant_expert_mm(
        quant_config=quant_config, block_size=B, **case["call_site_inputs"]
    ).to(torch.float32)
    limbs = _limb_counters()

    def _reference(post_scale: bool, clamp: bool) -> torch.Tensor:
        return torch_reference_moe(
            hidden_states=case["call_site_inputs"]["hidden_states"],
            expert_affinities=case["call_site_inputs"]["expert_affinities"],
            gate_up_proj_weight=case["call_site_inputs"]["gate_up_proj_weight"],
            down_proj_weight=case["call_site_inputs"]["down_proj_weight"],
            gate_up_block_scale=case["gup_grid"],
            down_block_scale=case["down_grid"],
            swiglu_limit=limit,
            post_scale=post_scale,
            clamp=clamp,
        )

    configured = _reference(post_scale=True, clamp=True)
    _nonempty_or_raise(configured, "override-pair")
    reverted = {
        "scaling_point_reverted": _reference(post_scale=False, clamp=True),
        "clamp_reverted": _reference(post_scale=True, clamp=False),
        "both_reverted_the_pre_r6_reference": _reference(
            post_scale=False, clamp=False
        ),
    }

    assert limbs == DECLARED_LIMB_DISPATCHES, (
        f"expected the limb reading {DECLARED_LIMB_DISPATCHES}, got "
        f"{limbs}; this pair is about the route the "
        f"acceptance measures, so it must run on that route"
    )
    for label, other in reverted.items():
        separation = float((other - configured).abs().max())
        if not separation > 0.0:
            raise VacuousControlError(
                f"{label}: reverting the override left the reference numerically "
                f"identical on this fixture (max abs difference {separation!r}), "
                f"so this arm cannot fail for the reason it claims"
            )
        with pytest.raises(AssertionError):
            torch.testing.assert_close(got, other, rtol=RTOL, atol=ATOL)

    # The configured reference is asserted last, so a failure here reads as the
    # call site disagreeing with the model rather than as an unarmed control.
    torch.testing.assert_close(got, configured, rtol=RTOL, atol=ATOL)


# ===========================================================================
# the routing is an operand, so a wrong operand must redden the acceptance.
# ===========================================================================
def test_moe_path_planted_routing_operands_must_fail() -> None:
    """measured: two plantings in the mapping's output each turn the numbers red. """
    from vllm_neuron import functional as functional_hub

    bank, _text_config = _build_bank()
    quant_config = _block_quant_config()
    case = _build_case()
    real_mapping = functional_hub.build_blockwise_mapping

    def _shift_real_slots(positions, block_to_expert):
        # Rolling the whole vector is inert on this fixture: its blocks are half
        # padding, so every real row stays inside its own block and only padding
        # crosses. Rolling the real slots alone carries the last real row of each
        # block into the next one, and the number of rows that land under a
        # different expert is returned so the arm can refuse a vacuous planting.
        flat = positions.reshape(-1)
        real = torch.nonzero(flat >= 0, as_tuple=False).reshape(-1)
        shifted = flat.clone()
        shifted[real] = flat[real].roll(1, 0)
        experts = block_to_expert.reshape(-1)[real.div(B, rounding_mode="floor")]
        return shifted.reshape(positions.shape), int((experts.roll(-1) != experts).sum())

    def _plant(shift_rows: bool, next_expert: bool):
        def mapping(**kwargs):
            masked, positions, block_to_expert, conditions = real_mapping(**kwargs)
            if shift_rows:
                positions, mapping.crossed = _shift_real_slots(positions, block_to_expert)
            if next_expert:
                block_to_expert = block_to_expert.clone()
                flat = block_to_expert.reshape(-1)
                flat[0] = (int(flat[0]) + 1) % E
            return masked, positions, block_to_expert, conditions

        mapping.crossed = 0
        return mapping

    want = _configured_reference(case, bank)
    _nonempty_or_raise(want, "planted-routing")

    for label, planting in (
        ("real_rows_shifted_one_real_slot", dict(shift_rows=True, next_expert=False)),
        ("block_0_expert_incremented", dict(shift_rows=False, next_expert=True)),
    ):
        planted = _plant(**planting)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(functional_hub, "build_blockwise_mapping", planted)
            _reset_limb_counters()
            got = bank.block_quant_expert_mm(
                quant_config=quant_config, block_size=B, **case["call_site_inputs"]
            ).to(torch.float32)
        counters = _limb_counters()
        separation = float((got - want).abs().max())
        assert counters == DECLARED_LIMB_DISPATCHES, (
            f"{label}: the limb counters read {counters}, declared "
            f"{DECLARED_LIMB_DISPATCHES}; this arm must fail for the planted "
            f"operand, not because the route stopped running"
        )
        if planting["shift_rows"] and not planted.crossed > 0:
            raise VacuousControlError(
                f"{label}: the shift left every real row under the expert it "
                f"already had (crossings {planted.crossed!r}), measured on the real "
                f"mapping's own operands, so it cannot change the output at all"
            )
        if not separation > 0.0:
            raise VacuousControlError(
                f"{label}: the planting left the output numerically identical "
                f"(max abs difference {separation!r}), so this arm cannot fail "
                f"for the reason it claims"
            )
        with pytest.raises(AssertionError):
            torch.testing.assert_close(got, want, rtol=RTOL, atol=ATOL)


# ===========================================================================
# the three limbs trace whole, over routing operands built before the trace.
# ===========================================================================
_DYNAMO_REFUSAL_NAMES = ("Unsupported", "GraphBreakError", "FullGraphError")


def _dynamo_refusal_classes() -> tuple[type, ...]:
    """Resolve, by name, the classes dynamo raises when a traced region breaks."""
    from torch._dynamo import exc as dynamo_exc

    found = tuple(
        getattr(dynamo_exc, name)
        for name in _DYNAMO_REFUSAL_NAMES
        if isinstance(getattr(dynamo_exc, name, None), type)
    )
    if not found:
        raise VacuousControlError(
            f"none of {_DYNAMO_REFUSAL_NAMES} names a class in torch._dynamo.exc "
            f"on this torch, so requiring the refusal by name would accept anything"
        )
    return found


#: Where a data-dependent guard refusal is named: module path, then class name. A
#: branch on a value read to the host raises one of these, wrapped or bare, and a
#: name this torch does not carry is skipped rather than assumed.
_DATA_DEPENDENT_REFUSALS = (
    ("torch._dynamo.exc", "UserError"),
    ("torch.fx.experimental.symbolic_shapes", "GuardOnDataDependentSymNode"),
)


def _steering_refusal_classes() -> tuple[type, ...]:
    """The classes a Python branch on a host-read value raises, dynamo's own included."""
    import importlib

    found = list(_dynamo_refusal_classes())
    for module_path, name in _DATA_DEPENDENT_REFUSALS:
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            continue
        candidate = getattr(module, name, None)
        if isinstance(candidate, type):
            found.append(candidate)
    return tuple(found)


def _refusal_in_chain(raised, classes: tuple[type, ...]):
    """The first exception in ``raised``'s chain that is one of ``classes``, or ``None``.
    """
    seen: set[int] = set()
    current = raised
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, classes):
            return current
        current = current.__cause__ or current.__context__
    return None


#: The three limbs in launch order, under the names the call site resolves off
#: this module at call time.
_LIMB_LAUNCH_ORDER = (
    "moe_gate_up_blockwise_fp8",
    "moe_swiglu_transposed",
    "moe_down_blockwise_fp8",
)

#: Where the mapping's row index sits in two limbs' argument lists. The capture
#: below asserts the two positions hold the same object before anything reads it.
_ROW_INDEX_IN_GATE_UP = 3
_ROW_INDEX_IN_DOWN = 4


#: The spelling torch itself prints for a captured kernel-wrapper call target.
_KERNEL_CALL_NAME = "nki_kernel_wrapper"


def _kernel_calls_in(graph) -> list[str]:
    """The kernel call targets one captured graph holds, in graph order."""
    return [
        str(node.target)
        for node in graph.graph.nodes
        if node.op == "call_function" and _KERNEL_CALL_NAME in str(node.target)
    ]


def _fullgraph(traced, graphs: list):
    """``traced`` traced whole, every captured graph kept in ``graphs``. """
    from torch import _dynamo
    from vllm_neuron import envs as neuron_envs

    if neuron_envs.VLLM_NEURON_DEBUG_MODE:
        raise VacuousControlError(
            "VLLM_NEURON_DEBUG_MODE is set, so the runner would compile with "
            "fullgraph=False and this reading would measure nothing"
        )

    def keep_graph(graph, _example_inputs):
        graphs.append(graph)
        return graph.forward

    # Two tests compile the same closure body; a cache hit would answer one of them
    # with the other one's trace.
    _dynamo.reset()
    return torch.compile(traced, fullgraph=True, backend=keep_graph)


def _capture_limb_calls(bank, quant_config, case):
    """Run the call site once and record the limb operands and the last answer. """
    real = {name: getattr(_seam_module, name) for name in _LIMB_LAUNCH_ORDER}
    recorded: dict[str, tuple] = {}
    answers: dict[str, torch.Tensor] = {}

    def recorder(name):
        def record(*args):
            answers[name] = real[name](*args)
            recorded[name] = args
            return answers[name]

        return record

    with pytest.MonkeyPatch.context() as patch:
        for name in _LIMB_LAUNCH_ORDER:
            patch.setattr(_seam_module, name, recorder(name))
        bank.block_quant_expert_mm(
            quant_config=quant_config, block_size=B, **case["call_site_inputs"]
        )
    missing = [name for name in _LIMB_LAUNCH_ORDER if name not in recorded]
    if missing:
        raise RouteInstrumentError(
            f"the call site never reached {missing}, so there are no recorded "
            f"operands for the limbs to run over"
        )
    gate_up, swiglu, down = (recorded[name] for name in _LIMB_LAUNCH_ORDER)
    if gate_up[_ROW_INDEX_IN_GATE_UP] is not down[_ROW_INDEX_IN_DOWN]:
        raise RouteInstrumentError(
            f"argument {_ROW_INDEX_IN_GATE_UP} of the gate/up limb is not the row "
            f"index object the down limb was handed, so a plant reading it would "
            f"be reading something else"
        )
    return (gate_up, swiglu[1:], down[1:]), answers[_LIMB_LAUNCH_ORDER[2]]


def _limbs_over(gate_up):
    """The three limbs composed over recorded operands, gate/up handed in."""
    swiglu = getattr(_seam_module, _LIMB_LAUNCH_ORDER[1])
    down = getattr(_seam_module, _LIMB_LAUNCH_ORDER[2])

    def limbs(gate_up_args, swiglu_tail, down_tail):
        return down(swiglu(gate_up(*gate_up_args), *swiglu_tail), *down_tail)

    return limbs


def test_moe_path_routed_limbs_trace_under_fullgraph() -> None:
    """measured: the three limbs trace whole under ``fullgraph=True`` and return. """
    bank, _text_config = _build_bank()
    quant_config = _block_quant_config()
    case = _build_case()
    operands, answer = _capture_limb_calls(bank, quant_config, case)
    want = answer.to(torch.float32)
    _nonempty_or_raise(want, "fullgraph")

    _reset_limb_counters()
    intact = getattr(_seam_module, _LIMB_LAUNCH_ORDER[0])
    graphs: list = []
    got = _fullgraph(_limbs_over(intact), graphs)(*operands).to(torch.float32)
    # The counters are folded off the traced graph: each adds one per trace, at trace time, and
    # nothing on a cache hit. One compiled call is one trace, so the read equals the declared
    # triple here; a second call of the same shapes would leave it unchanged.
    counters = _limb_counters()
    kernel_calls = [call for graph in graphs for call in _kernel_calls_in(graph)]
    declared_calls = sum(nki for nki, _fallback in DECLARED_LIMB_DISPATCHES)
    assert len(graphs) == 1, (
        f"{len(graphs)} graphs reached the backend, want the one whole trace; "
        f"under fullgraph a second graph cannot happen and none means no trace"
    )
    assert len(kernel_calls) == declared_calls, (
        f"the captured graph holds {kernel_calls}, want {declared_calls} kernel "
        f"calls; every call target it holds is "
        f"{[str(node.target) for node in graphs[0].graph.nodes]}"
    )
    assert counters == DECLARED_LIMB_DISPATCHES, (
        f"the compiled limbs read {counters}, declared {DECLARED_LIMB_DISPATCHES}; "
        f"a trace that returns without running the limbs proves nothing (one count per trace, "
        f"not per call)"
    )
    torch.testing.assert_close(got, want, rtol=RTOL, atol=ATOL)


def test_moe_path_planted_host_read_breaks_the_limb_trace() -> None:
    """measured: a host read that steers a Python branch turns the test above red. """
    bank, _text_config = _build_bank()
    quant_config = _block_quant_config()
    case = _build_case()
    operands, answer = _capture_limb_calls(bank, quant_config, case)
    want = answer.to(torch.float32)
    _nonempty_or_raise(want, "planted-host-read")
    intact = getattr(_seam_module, _LIMB_LAUNCH_ORDER[0])
    read: list[tuple] = []

    def planted(*args):
        steered = args[_ROW_INDEX_IN_GATE_UP].reshape(-1)[0].item()
        # The plant: a Python branch on the value just read to the host. Both arms
        # hand the same operands to the same limb, so nothing numeric moves and the
        # steering alone is what a whole-graph trace has to answer for.
        if steered >= 0:
            arm = "nonnegative"
        else:
            arm = "negative"
        read.append((steered, arm))
        return intact(*args)

    refusals = _steering_refusal_classes()
    eager = _limbs_over(planted)(*operands).to(torch.float32)
    torch.testing.assert_close(eager, want, rtol=RTOL, atol=ATOL)

    graphs: list = []
    declared_calls = sum(nki for nki, _fallback in DECLARED_LIMB_DISPATCHES)
    raised = None
    try:
        _fullgraph(_limbs_over(planted), graphs)(*operands)
    except Exception as refused:  # Noqa: BLE001 -- the chain is the reading
        raised = refused
    whole = [
        graph for graph in graphs if len(_kernel_calls_in(graph)) == declared_calls
    ]
    refusal = _refusal_in_chain(raised, refusals)
    from torch._dynamo import config as dynamo_config

    assert not whole, (
        f"the steered arm still left {len(whole)} whole graph(s) holding "
        f"{declared_calls} kernel calls, which is what the test above passes on"
    )
    assert refusal is not None, (
        f"the steered arm raised {type(raised).__name__} with no class among "
        f"{[cls.__name__ for cls in refusals]} anywhere in its chain, so nothing "
        f"says the trace was refused for the steering"
    )


# ===========================================================================
# diagnostic, never gating: what the mapping alone does under a fullgraph trace.
# ===========================================================================
#: the two serving shapes read beside the tests' own: the pinned prefill bucket,
#: and the one-token decode step at ``max-num-seqs 1``.
_DIAG_PREFILL_TOKENS = 2048
_DIAG_DECODE_TOKENS = 1


def _mapping_affinities(tokens: int, experts: int, top_k: int) -> torch.Tensor:
    """``[tokens, experts]`` router scores: ``top_k`` slots per token, walked."""
    affinities = torch.zeros(tokens, experts, dtype=torch.float32)
    for token in range(tokens):
        for slot in range(top_k):
            affinities[token, (token + slot) % experts] = 1.0 / top_k
    return affinities


def _routing_by_expert(mapping: tuple, experts: int, block: int) -> dict:
    """Which token ids each expert is handed, read off one mapping's own outputs."""
    _, token_position_to_id, block_to_expert, _ = mapping
    blocks = token_position_to_id.reshape(-1, block)
    return {
        expert: sorted(
            int(token)
            for token in blocks[block_to_expert == expert].reshape(-1).tolist()
            if int(token) != -1
        )
        for expert in range(experts)
    }


def _diag_mapping_row(tokens: int, experts: int, top_k: int) -> str:
    """Trace the mapping alone and report the attempt as one row, never raising."""
    from torch import _dynamo
    from vllm_neuron import functional as functional_hub

    affinities = _mapping_affinities(tokens, experts, top_k)
    graphs: list = []

    def keep_graph(graph, _example_inputs):
        """Take the traced graph and hand back the callable dynamo will run."""
        graphs.append(graph)
        return graph.forward

    def mapping(scores):
        return functional_hub.build_blockwise_mapping(
            expert_affinities=scores,
            num_local_experts=experts,
            num_experts_per_token=top_k,
            block_size=B,
            moe_group=None,
            tp_degree=1,
        )

    _dynamo.reset()
    error = ""
    try:
        torch.compile(mapping, fullgraph=True, backend=keep_graph)(affinities)
    except Exception as refused:  # Noqa: BLE001 -- the row is the reading
        head = str(refused).splitlines() or [type(refused).__name__]
        error = head[0]
    # A graph in hand is what says the trace completed: the backend is reached only
    # after one, so reading the exception would call a failed run a failed trace.
    if not graphs:
        return f"traced=no|error={error or 'no graph reached the backend'}"
    return f"traced=yes|error={error}"


@pytest.mark.parametrize(
    "tokens,from_the_pinned_config",
    [
        pytest.param(T, False, id="tests_own_mask"),
        pytest.param(_DIAG_PREFILL_TOKENS, True, id="prefill_bucket"),
        pytest.param(_DIAG_DECODE_TOKENS, True, id="decode_step"),
    ],
)
def test_moe_path_mapping_traces_at_every_declared_shape(
    tokens: int, from_the_pinned_config: bool
) -> None:
    """measured: the mapping traces whole at this file's shape and both serving shapes.
    """
    experts, top_k = E, K
    if from_the_pinned_config:
        text_config = _pinned_raw_config()["text_config"]
        experts = int(text_config["n_routed_experts"])
        top_k = int(text_config["num_experts_per_tok"])
    row = _diag_mapping_row(tokens=tokens, experts=experts, top_k=top_k)
    assert row.startswith("traced=yes"), row


# ===========================================================================
# the construction a captured graph gets must route what the vendor's routes.
# ===========================================================================
@pytest.mark.parametrize(
    "tokens,from_the_pinned_config",
    [
        pytest.param(T, False, id="tests_own_mask"),
        pytest.param(_DIAG_PREFILL_TOKENS, True, id="prefill_bucket"),
    ],
)
def test_moe_path_capture_safe_mapping_routes_what_the_vendor_routes(
    monkeypatch: pytest.MonkeyPatch, tokens: int, from_the_pinned_config: bool
) -> None:
    """measured: the construction capture gets hands every expert the vendor's tokens.
    """
    from vllm_neuron import functional as functional_hub
    from vllm_neuron.functional.moe import moe_blockwise

    experts, top_k = E, K
    if from_the_pinned_config:
        text_config = _pinned_raw_config()["text_config"]
        experts = int(text_config["n_routed_experts"])
        top_k = int(text_config["num_experts_per_tok"])
    affinities = _mapping_affinities(tokens, experts, top_k)

    def build() -> tuple:
        return functional_hub.build_blockwise_mapping(
            expert_affinities=affinities,
            num_local_experts=experts,
            num_experts_per_token=top_k,
            block_size=B,
            moe_group=None,
            tp_degree=1,
        )

    # The vendor subkernels run here, under the simulator and at the serving shape.
    # These rows are the only thing that distinguishes a slow reference from a hang
    # while the step is still running; they carry their own tag, so the rows the host
    # counts as this test's verdict stay two.
    reference = build()
    monkeypatch.setattr(moe_blockwise, "can_run_kernel", lambda *_a, **_k: False)
    candidate = build()
    reference_routing = _routing_by_expert(reference, experts, B)
    candidate_routing = _routing_by_expert(candidate, experts, B)
    routed = sum(len(rows) for rows in reference_routing.values())
    if routed != tokens * top_k:
        raise VacuousControlError(
            f"the reference routed {routed} positions at {tokens} tokens and top-k "
            f"{top_k}, so the comparison below would pass on a mapping that routes "
            f"almost none of them"
        )
    assert torch.equal(candidate[0], reference[0])
    assert candidate_routing == reference_routing


# ===========================================================================
# The arm that says why the counter clause is load-bearing.
# ===========================================================================
def test_moe_path_f1_numeric_arm_alone_cannot_discriminate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """measured: a torch composition passes the numbers while no kernel runs. """
    bank, _text_config = _build_bank()
    quant_config = _block_quant_config()
    case = _build_case()
    limbs = _TorchLimbs(case["gup_grid"], case["down_grid"])

    # The call site imports the three names inside the method, so it resolves them
    # off this module at call time and patching the module is what reaches it.
    for name, replacement in (
        ("moe_gate_up_blockwise_fp8", limbs.gate_up),
        ("moe_swiglu_transposed", limbs.swiglu),
        ("moe_down_blockwise_fp8", limbs.down),
    ):
        monkeypatch.setattr(_seam_module, name, replacement)

    _reset_limb_counters()
    with _AttributedSimulatorCounter() as sim:
        got = bank.block_quant_expert_mm(
            quant_config=quant_config, block_size=B, **case["call_site_inputs"]
        )
    counters = _limb_counters()
    want = _configured_reference(case, bank)
    got_f32 = got.to(torch.float32)
    _nonempty_or_raise(want, "f1-hazard")
    assert counters == NO_LIMB_DISPATCHES, (
        f"the limb counters read {counters}, expected {NO_LIMB_DISPATCHES}; the "
        f"substitution did not take effect, so this arm would be measuring the "
        f"shipped route and not the hazard"
    )
    # Not ``sim.total == 0``: the token-block mapping dispatches its own NKI
    # subkernels inside this call, which is the reading :func:`_run_mapping`
    # requires. Zero is the share attributed to the limb module.
    assert sim.through_seam == 0, (
        f"{sim.through_seam} of {sim.total} simulator entries were attributed to "
        f"{_SEAM_FILE} while all three limbs were substituted, so the "
        f"attribution is matching frames it should not"
    )
    torch.testing.assert_close(got_f32, want, rtol=RTOL, atol=ATOL)


def test_moe_path_call_site_maps_before_padding_and_dispatches_nki() -> None:
    """measured through the call site: the mapping runs its NKI flow, not torch. """
    bank, _text_config = _build_bank()
    quant_config = _block_quant_config()
    case = _build_case()

    seen: list[torch.Tensor] = []
    nested: list[_AttributedSimulatorCounter] = []

    import vllm_neuron.functional as NF

    real_mapping = NF.build_blockwise_mapping

    def spy(expert_affinities, *args, **kwargs):
        seen.append(expert_affinities.detach().clone())
        # A nested counter, so the mapping's own share is read inside the real
        # call-site call rather than reconstructed from a separate run. Nesting
        # composes: this counter's "real" is the outer counter's wrapper, so the
        # outer total still sees every entry.
        with _AttributedSimulatorCounter() as inner:
            result = real_mapping(expert_affinities, *args, **kwargs)
        nested.append(inner)
        return result

    _reset_limb_counters()
    NF.build_blockwise_mapping = spy
    try:
        with _AttributedSimulatorCounter() as outer:
            out = bank.block_quant_expert_mm(
                quant_config=quant_config, block_size=B, **case["call_site_inputs"]
            )
    finally:
        NF.build_blockwise_mapping = real_mapping

    if len(seen) != 1 or len(nested) != 1:
        raise VacuousControlError(
            f"the call site entered the mapping {len(seen)} times, not once; "
            f"every reading here assumes exactly one entry"
        )
    handed, inner = seen[0], nested[0]
    counters = _limb_counters()

    assert tuple(handed.shape) == (T, E), (
        f"the call site handed the mapping {tuple(handed.shape)}; it must hand "
        f"the REAL token count ({T}, {E}) and append the kernel's padding slot "
        f"afterwards. ({T + 1}, {E}) is the pad-first order defect "
        f"reports, and it turns both of the mapping's NKI gates off."
    )
    assert inner.total > 0, (
        f"the mapping dispatched {inner.total} kernels inside a real call-site "
        f"call, so it took the torch fallback for per-token device work the "
        f"fork ships NKI subkernels for"
    )
    assert inner.through_seam == 0, (
        f"{inner.through_seam} of the mapping's {inner.total} entries were "
        f"attributed to the block-quant seam, so the attribution sum double-counts"
    )
    assert outer.through_seam == sum(
        dispatches for dispatches, _fallback in DECLARED_LIMB_DISPATCHES
    ), (
        f"the limb module was entered {outer.through_seam} times, declared "
        f"{DECLARED_LIMB_DISPATCHES}"
    )
    assert outer.total == outer.through_seam + inner.total, (
        f"{outer.total} entries in the whole call do not add up: "
        f"{outer.through_seam} through the seam plus {inner.total} in the "
        f"mapping is {outer.through_seam + inner.total}. Some component nobody "
        f"measured dispatched."
    )
    assert counters == DECLARED_LIMB_DISPATCHES, (
        f"seam counters {counters}: the mapping's route must not change how many "
        f"times the block-quant kernel is entered, nor take a fallback"
    )
    assert tuple(out.shape) == (T, H)
    assert bool(torch.isfinite(out).all())
    assert float(out.abs().max()) > 0.0


def test_moe_path_mapping_order_moves_no_numbers() -> None:
    """The two pad orders give the same tensors, so no declared value moves. """
    case = _build_case()
    affinities = case["call_site_inputs"]["expert_affinities"]
    padded = torch.cat(
        [affinities, torch.zeros(1, E, dtype=affinities.dtype)], dim=0
    )

    pad_first = _run_mapping(padded, "pad-first-257")
    pad_after = _run_mapping(affinities, "pad-after-256")

    assert pad_first["total"] == 0, (
        f"the pad-first order dispatched {pad_first['total']} kernels; finding "
        f"its mechanism is that both NKI gates refuse at T + 1 = {T + 1}"
    )
    assert pad_after["total"] > 0, (
        f"the pad-after order dispatched {pad_after['total']} kernels, so the "
        f"repair did not turn the mapping's NKI flow on"
    )

    pad_flat = torch.zeros(E, 1, dtype=pad_after["masked"].dtype)
    repaired_masked = torch.cat([pad_after["masked"], pad_flat], dim=0)
    assert tuple(repaired_masked.shape) == tuple(pad_first["masked"].shape)
    assert torch.equal(repaired_masked, pad_first["masked"]), (
        "appending E zero entries to the flat masked tensor is not the same "
        "tensor as appending a zero row before the view, so the pad-order "
        "change would move what the seam consumes"
    )

    for key in ("token_position_to_id", "block_to_expert", "conditions"):
        left, right = pad_first[key], pad_after[key]
        equal = tuple(left.shape) == tuple(right.shape) and bool(
            torch.equal(left, right)
        )
        assert equal, (
            f"{key} differs between the two pad orders, so turning the NKI flow "
            f"on changes what the seam consumes and the declared tolerance is "
            f"no longer the same measurement"
        )


@pytest.mark.parametrize("tokens", [T, 512, 1024, 2048])
def test_moe_path_mapping_dispatches_nki_across_the_token_envelope(
    tokens: int,
) -> None:
    """The mapping's NKI flow is on at every token count in the review's bar. """
    real = _envelope_affinities(tokens)
    padded = torch.cat([real, torch.zeros(1, E, dtype=real.dtype)], dim=0)

    after = _run_mapping(real, f"envelope-{tokens}-real")
    first = _run_mapping(padded, f"envelope-{tokens}-padded")

    assert after["total"] > 0, (
        f"at {tokens} real tokens the mapping dispatched {after['total']} "
        f"kernels, so its NKI flow is off at this extent"
    )
    assert first["total"] == 0, (
        f"at {tokens} + 1 tokens the mapping dispatched {first['total']} "
        f"kernels; the contrast that makes the reading above meaningful is that "
        f"the pad-first extent refuses both gates"
    )


# ===========================================================================
# the route selector: no enum member, a named refusal instead.
# ===========================================================================
def test_moe_path_unquantised_config_raises_by_name() -> None:
    """An unquantised config must raise, not reach the substrate's none default. """
    from vllm_neuron.model.glm5_next.model_fp8 import (
        Glm5NextBlockQuantRouteError,
        Glm5NextQuantConfig,
    )

    bank, _text_config = _build_bank()
    case = _build_case()
    unquantised = Glm5NextQuantConfig(None)
    if unquantised.is_block_quantized:
        raise VacuousControlError(
            "Glm5NextQuantConfig(None) reports is_block_quantized=True, so this "
            "control's input is not the unquantised world it claims to be"
        )
    _reset_limb_counters()
    with pytest.raises(Glm5NextBlockQuantRouteError, match="block-quant route"):
        bank.block_quant_expert_mm(
            quant_config=unquantised, block_size=B, **case["call_site_inputs"]
        )
    assert _limb_counters() == NO_LIMB_DISPATCHES, (
        f"the refusal must fire BEFORE any dispatch, read "
        f"{_limb_counters()}"
    )


def test_moe_path_foreign_checkpoint_block_shape_raises_by_name() -> None:
    """A checkpoint block shape the retile does not bridge must raise. """
    from vllm_neuron.model.glm5_next.model_fp8 import Glm5NextBlockQuantRouteError

    bank, _text_config = _build_bank()
    case = _build_case()
    real = _block_quant_config()
    if tuple(real.block_shape) != (TILE_SIZE, TILE_SIZE):
        raise VacuousControlError(
            f"the pinned checkpoint declares block_shape={real.block_shape}, "
            f"not ({TILE_SIZE}, {TILE_SIZE}); this control's premise is stale"
        )

    class _ForeignShapeMethod:
        block_shape = (64, 64)

    class _ForeignShapeConfig:
        is_block_quantized = True
        method = _ForeignShapeMethod()
        block_shape = (64, 64)

    _reset_limb_counters()
    with pytest.raises(Glm5NextBlockQuantRouteError, match="weight_block_size"):
        bank.block_quant_expert_mm(
            quant_config=_ForeignShapeConfig(),
            block_size=B,
            **case["call_site_inputs"],
        )
    assert _limb_counters() == NO_LIMB_DISPATCHES


def _imported_names(source: str, filename: str) -> set[str]:
    """Every name an ``import`` statement binds in ``source``, by AST. """
    import ast

    bound: set[str] = set()
    tree = ast.parse(source, filename=filename)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add(alias.asname or alias.name.split(".")[0])
    return bound


def test_moe_path_package_binds_no_quantisation_enum(tmp_path: Path) -> None:
    """Mechanically: no module of this arch package imports ``QuantizationType``.
    """
    from vllm_neuron.model.glm5_next import model_fp8

    package_dir = Path(model_fp8.__file__).resolve().parent
    modules = sorted(package_dir.glob("*.py"))
    if not modules:
        raise VacuousControlError(
            f"no modules found under {package_dir}, so this scan read nothing"
        )
    offenders = {}
    for module_path in modules:
        bound = _imported_names(
            module_path.read_text(), filename=str(module_path)
        )
        if "QuantizationType" in bound:
            offenders[module_path.name] = sorted(bound & {"QuantizationType"})
    assert offenders == {}, (
        f"a module of this arch package binds the substrate's QuantizationType, "
        f"which is the precondition for adding a member to it: {offenders}"
    )

    control = tmp_path / "control_binds_the_enum.py"
    control.write_text(
        "from vllm_neuron.functional.quantization import QuantizationType\n"
    )
    control_bound = _imported_names(control.read_text(), filename=str(control))
    if "QuantizationType" not in control_bound:
        raise VacuousControlError(
            f"the collector did not find QuantizationType in source that "
            f"imports it (found {sorted(control_bound)}), so the empty offender "
            f"set above is not a measurement"
        )


def test_moe_path_exports_resolve_to_their_own_modules() -> None:
    """Both MoE kernels resolve through the hub, and are the module's own objects."""
    import vllm_neuron.functional as functional
    import vllm_neuron.functional.moe as functional_moe
    from vllm_neuron.functional.blockwise_fp8_mm import (
        blockwise_fp8_mm as dense_seam,
    )
    from vllm_neuron.functional.moe.blockwise_fp8_retile import (
        retile_block_scales as retile_producer,
    )

    if functional.blockwise_fp8_moe is not blockwise_fp8_moe:
        raise ExportSurfaceError(
            "vllm_neuron.functional.blockwise_fp8_moe is not the block-quant seam object"
        )
    if functional.blockwise_fp8_mm is not dense_seam:
        raise ExportSurfaceError(
            "vllm_neuron.functional.blockwise_fp8_mm is not the dense seam object"
        )
    if functional_moe.blockwise_fp8_moe is not blockwise_fp8_moe:
        raise ExportSurfaceError(
            "vllm_neuron.functional.moe.blockwise_fp8_moe is not the block-quant seam"
        )
    if functional_moe.retile_block_scales is not retile_producer:
        raise ExportSurfaceError(
            "vllm_neuron.functional.moe.retile_block_scales is not the retile producer"
        )
    for name in ("blockwise_fp8_mm", "blockwise_fp8_moe"):
        if name not in functional.__all__:
            raise ExportSurfaceError(f"{name!r} missing from functional.__all__")
    assert functional.__all__ == sorted(functional.__all__), (
        "functional.__all__ is no longer alphabetical, which is the convention "
        "the file itself declares"
    )
    assert functional_moe.__all__ == sorted(functional_moe.__all__)


def test_moe_path_colliding_helper_names_are_not_flat_exported() -> None:
    """The name collisions stay qualified, and the collision is real. """
    import vllm_neuron.functional as functional
    import vllm_neuron.functional.moe as functional_moe
    from vllm_neuron.functional.blockwise_fp8_mm import (
        to_kernel_scale_layout as dense_bridge,
    )

    # The collision is real: same name, different arity, both live.
    dense_arity = dense_bridge.__code__.co_argcount
    moe_arity = moe_to_kernel_scale_layout.__code__.co_argcount
    if dense_bridge is moe_to_kernel_scale_layout or dense_arity == moe_arity:
        raise VacuousControlError(
            f"the two to_kernel_scale_layout helpers are the same object or "
            f"share an arity ({dense_arity} vs {moe_arity}), so there is no "
            f"collision for the hub to guard against and this arm asserts a "
            f"property of nothing"
        )
    for name in (
        "to_kernel_scale_layout",
        "flat_scale_index",
        "kernel_scale_shape",
        "dispatch_counters",
        "reset_dispatch_counters",
        "kernel_identity",
        "BLOCK_QUANT_SIZE",
        "TILE_SIZE",
    ):
        for hub, label in (
            (functional, "vllm_neuron.functional"),
            (functional_moe, "vllm_neuron.functional.moe"),
        ):
            if name in getattr(hub, "__all__", ()):
                raise ExportSurfaceError(
                    f"{label}.__all__ flat-exports the colliding name {name!r}"
                )
            if hasattr(hub, name):
                raise ExportSurfaceError(
                    f"{label} has attribute {name!r}, which resolves one of two "
                    f"different-arity definitions by import order"
                )


def test_moe_path_fixture_conditioning_is_measured_not_assumed() -> None:
    """The conditioning the declared tolerance is measured against, asserted. """
    case = _build_case()
    inputs = case["call_site_inputs"]
    hidden = inputs["hidden_states"]
    affinities = inputs["expert_affinities"]
    gup = inputs["gate_up_proj_weight"]
    down = inputs["down_proj_weight"]

    hidden_min = float(hidden.to(torch.float32).min())
    gup_min = float(gup.to(torch.float32).min())
    down_min = float(down.to(torch.float32).min())
    per_expert = (affinities != 0).sum(dim=0).tolist()
    distinct_gup_scales = int(torch.unique(case["gup_logical"]).numel())
    distinct_down_scales = int(torch.unique(case["down_logical"]).numel())
    affinity_roundtrip = float(
        (
            affinities - affinities.to(torch.bfloat16).to(torch.float32)
        ).abs().max()
    )
    hidden_roundtrip = float(
        (
            hidden.to(torch.float32)
            - hidden.to(torch.float32).to(_FP8).to(torch.float32)
        ).abs().max()
    )
    if hidden_min <= 0.0 or gup_min <= 0.0 or down_min <= 0.0:
        raise FixtureConditioningError(
            f"the fixture contains non-positive values (hidden_min={hidden_min}, "
            f"gate_up_min={gup_min}, down_min={down_min}); the declared "
            f"pointwise rtol={RTOL} is then dominated by catastrophic "
            f"cancellation over the H={H} contraction rather than by kernel "
            f"error, which is how an early block-quant kernel failed"
        )
    if affinity_roundtrip != 0.0 or hidden_roundtrip != 0.0:
        raise FixtureConditioningError(
            f"a fixture value does not round-trip exactly "
            f"(affinity={affinity_roundtrip}, hidden={hidden_roundtrip}); the "
            f"tolerance would be absorbing fixture cast error"
        )
    if per_expert != [TOKENS_PER_EXPERT] * E:
        raise FixtureConditioningError(
            f"expert occupancy is {per_expert}, declared "
            f"{[TOKENS_PER_EXPERT] * E}; an empty expert would make its share "
            f"of the comparison vacuous"
        )
    if distinct_gup_scales < 2 or distinct_down_scales < 2:
        raise FixtureConditioningError(
            f"the scales are effectively uniform (gate_up distinct="
            f"{distinct_gup_scales}, down distinct={distinct_down_scales}); a "
            f"comparison on uniform scales cannot observe a permuted layout"
        )


def test_moe_path_mapping_shape_is_the_one_the_seam_consumes() -> None:
    """The mapping the call site builds has the extents the seam declares. """
    from vllm_neuron.functional import build_blockwise_mapping

    case = _build_case()
    padded_affinities = torch.cat(
        [
            case["call_site_inputs"]["expert_affinities"],
            torch.zeros(1, E, dtype=torch.float32),
        ],
        dim=0,
    )
    masked, token_position_to_id, block_to_expert, conditions = (
        build_blockwise_mapping(
            expert_affinities=padded_affinities,
            num_local_experts=E,
            num_experts_per_token=K,
            block_size=B,
            moe_group=None,
            tp_degree=1,
        )
    )
    num_blocks = int(block_to_expert.numel())
    empty_blocks = int((conditions == 0).sum())
    covered = sorted({int(value) for value in block_to_expert.tolist()})
    assert tuple(masked.shape) == ((T + 1) * E, 1)
    assert tuple(token_position_to_id.shape) == (num_blocks * B,)
    assert token_position_to_id.dtype is torch.int32
    assert block_to_expert.dtype is torch.int32
    assert covered == list(range(E)), (
        f"block_to_expert covers experts {covered}, not every local expert; "
        f"an uncovered expert's weights would never be read"
    )
    assert empty_blocks >= 1, (
        "no empty block in this configuration, so the padding-slot path the "
        "call site's appended row exists for is not exercised"
    )


def test_moe_path_kernel_identity_is_the_three_limb_kernels() -> None:
    """Read off the objects: this call site's three kernels are authored here. """
    readings = {"gate_up": gate_up_kernel_identity(),
                "swiglu": swiglu_kernel_identity(),
                "down": down_kernel_identity()}
    for limb, (module, qualname) in readings.items():
        assert module == _seam_module.__name__, (
            f"the {limb} limb's kernel lives in {module!r}, not the limb module "
            f"{_seam_module.__name__!r}"
        )
        assert "bwmm_shard_on_I" not in module and "moe_cte" not in qualname, (
            f"the {limb} limb reaches {module!r}.{qualname!r}, a vendor member "
            f"rather than a kernel authored here"
        )
    assert len(set(readings.values())) == 3, (
        f"two limbs report the same kernel: {readings}; the composition launches "
        f"three distinct ones"
    )


def test_moe_path_block_size_must_be_block_quant_granular() -> None:
    """A ``block_size`` the vendor kernel's ``B % 256 == 0`` assert refuses."""
    from vllm_neuron.model.glm5_next.model_fp8 import Glm5NextBlockQuantRouteError

    bank, _text_config = _build_bank()
    quant_config = _block_quant_config()
    case = _build_case()
    _reset_limb_counters()
    with pytest.raises(Glm5NextBlockQuantRouteError, match="BLOCK_QUANT_SIZE"):
        bank.block_quant_expert_mm(
            quant_config=quant_config,
            block_size=BLOCK_QUANT_SIZE + 1,
            **case["call_site_inputs"],
        )
    assert _limb_counters() == NO_LIMB_DISPATCHES


def test_moe_path_expert_count_disagreement_raises_by_name() -> None:
    """A weight bank that disagrees with the partition must raise. """
    from vllm_neuron.model.glm5_next.model_fp8 import Glm5NextBlockQuantRouteError

    bank, _text_config = _build_bank()
    quant_config = _block_quant_config()
    case = _build_case()
    inputs = dict(case["call_site_inputs"])
    inputs["gate_up_proj_weight"] = inputs["gate_up_proj_weight"][: E - 1]
    _reset_limb_counters()
    with pytest.raises(Glm5NextBlockQuantRouteError, match="experts"):
        bank.block_quant_expert_mm(
            quant_config=quant_config, block_size=B, **inputs
        )
    assert _limb_counters() == NO_LIMB_DISPATCHES


# ===========================================================================
# the dispatch step: global router columns -> this rank's slice.
# Router affinities are global, not local.
# ===========================================================================
#
# Why this section exists. Every arm above builds the bank through
# :func:`_build_bank`, which passes ``world_size=1``. At degree 1 the global and
# the local expert counts coincide at ``E``, so that is the one configuration in
# which the call site's affinity contract cannot be wrong -- and no arm above can
# see whether the site consumes the router's global ``[T, E]`` or this rank's
# local ``[T, E_local]``. These two arms run at a degree above 1, where the two
# shapes differ and the question has an answer.
#
# Nothing here is hand-chosen. The global expert width and the router's ``top_k``
# are the pinned checkpoint's own values, read off the same digest-verified
# fixture the rest of this file routes on, and the degree is derived from them.


#: This rank's local expert extent, at the degree below. ``E`` rather than a new
#: number, because ``E`` is the extent :func:`_build_case` already builds weights
#: and scales for, and the dispatch is a shape question that must not be answered
#: by rebuilding the fixture around it.
EP_LOCAL_EXPERTS = E

#: Fixture-only seed for this section's router inputs. This file's own choice.
EP_ROUTER_SEED = 9027


def _ep_partition() -> tuple[int, int, int]:
    """``(global experts, expert-parallel degree, router top_k)``, all measured. """
    text = _pinned_raw_config()["text_config"]
    global_experts = int(text["n_routed_experts"])
    top_k = int(text["num_experts_per_tok"])
    if global_experts % EP_LOCAL_EXPERTS:
        raise VacuousControlError(
            f"the pinned config's n_routed_experts={global_experts} is not "
            f"divisible by the fixture's local extent {EP_LOCAL_EXPERTS}, so no "
            f"uniform degree gives this rank the weights _build_case builds"
        )
    degree = global_experts // EP_LOCAL_EXPERTS
    if degree <= 1:
        raise VacuousControlError(
            f"the derived expert-parallel degree is {degree}; this section's "
            f"whole subject is a degree ABOVE 1 and cannot be tested at 1"
        )
    return global_experts, degree, top_k


def _build_ep_bank():
    """A bank at a degree above 1 whose local extent is the fixture's ``E``. """
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig
    from vllm_neuron.model.glm5_next.model_fp8 import Glm5NextRoutedExperts

    global_experts, degree, top_k = _ep_partition()
    text_config = Glm5NextTextConfig(
        hidden_size=H,
        moe_intermediate_size=I_TP,
        n_routed_experts=global_experts,
        num_experts_per_tok=top_k,
    )
    bank = Glm5NextRoutedExperts(text_config, world_size=degree, ep_degree=degree)
    if int(bank.num_local_experts) != EP_LOCAL_EXPERTS:
        raise VacuousControlError(
            f"the bank reports num_local_experts={bank.num_local_experts} at "
            f"degree {degree}, but this section's fixture is built for "
            f"{EP_LOCAL_EXPERTS}; the weight extent check would fire before any "
            f"affinity was looked at"
        )
    if int(bank.num_routed_experts) == int(bank.num_local_experts):
        raise VacuousControlError(
            "the global and local expert counts coincide, so this bank is the "
            "degree-1 case again and the dispatch step would be a no-op"
        )
    # The partition, checked as a partition before this arm indexes with
    # it. If it did not cover every global column exactly once, the per-rank
    # column comparisons below would be against the wrong reference.
    seen: list[int] = []
    for rank in range(degree):
        seen.extend(bank.local_expert_indices(rank))
    if sorted(seen) != list(range(global_experts)):
        raise VacuousControlError(
            f"local_expert_indices over {degree} ranks does not cover "
            f"0..{global_experts - 1} exactly once, so the columns this arm "
            f"compares against are not a partition"
        )
    return bank, text_config, global_experts, degree


def _ep_route(bank, global_experts: int, text_config):
    """Run ``route_tokens`` for real and return its own output tensors. """
    gen = torch.Generator().manual_seed(EP_ROUTER_SEED)
    hidden_bsh = (torch.randn(1, T, H, generator=gen) * 0.5).to(torch.bfloat16)
    gamma = torch.ones(1, H, dtype=torch.bfloat16)
    bank.router_weight = torch.nn.Parameter(
        (torch.randn(H, global_experts, generator=gen) * 0.1).to(torch.bfloat16),
        requires_grad=False,
    )
    bank.router_bias = torch.nn.Parameter(
        (torch.randn(global_experts, generator=gen) * 0.05).to(torch.bfloat16),
        requires_grad=False,
    )
    _logits, expert_index, affinities = bank.route_tokens(
        hidden_bsh, gamma, text_config
    )
    if tuple(affinities.shape) != (T, global_experts):
        raise VacuousControlError(
            f"route_tokens returned {tuple(affinities.shape)}; this section's "
            f"whole subject is its GLOBAL [T={T}, E={global_experts}] form"
        )
    if int((affinities != 0).sum()) == 0:
        raise VacuousControlError(
            "route_tokens returned an all-zero affinity tensor, so every "
            "reading below would hold for the wrong reason"
        )
    return affinities, expert_index


class _MappingAffinitySpy:
    """Captures the affinity tensor ``build_blockwise_mapping`` is handed. """

    def __init__(self) -> None:
        self.seen: list[torch.Tensor] = []
        self._module = None
        self._real = None

    def __enter__(self) -> "_MappingAffinitySpy":
        import vllm_neuron.functional as NF

        self._module = NF
        self._real = NF.build_blockwise_mapping
        real = self._real

        def spy(expert_affinities, *args, **kwargs):
            self.seen.append(expert_affinities.detach().clone())
            return real(expert_affinities, *args, **kwargs)

        NF.build_blockwise_mapping = spy
        return self

    def __exit__(self, *_exc) -> bool:
        self._module.build_blockwise_mapping = self._real
        return False

    @property
    def only(self) -> torch.Tensor:
        if len(self.seen) != 1:
            raise VacuousControlError(
                f"the mapping was entered {len(self.seen)} times, not once; "
                f"every per-rank reading here assumes exactly one entry"
            )
        return self.seen[0]


def test_moe_path_dispatch_maps_global_router_output_at_degree_above_one() -> None:
    """``route_tokens``' global output reaches the kernel as ``[T, E_local]``. """
    bank, text_config, global_experts, degree = _build_ep_bank()
    quant_config = _block_quant_config()
    case = _build_case()
    affinities, _expert_index = _ep_route(bank, global_experts, text_config)

    # Rank choice is measured: the three ranks carrying the most routing mass,
    # so no probed slice is all zero and no two are trivially equal.
    occupancy = {}
    for rank in range(degree):
        cols = torch.tensor(bank.local_expert_indices(rank), dtype=torch.int64)
        occupancy[rank] = int((affinities[:, cols] != 0).sum())
    probed = sorted(occupancy, key=lambda r: (-occupancy[r], r))[:3]
    assert len(probed) == 3, (
        f"only {len(probed)} ranks exist at degree {degree}; the rank-liveness "
        f"reading below needs three distinct column sets"
    )
    assert min(occupancy[r] for r in probed) > 0, (
        "a probed rank owns no routed tokens at all, so its slice is all zero "
        "and the gather reading below would hold for the wrong reason"
    )

    gathered: dict[int, torch.Tensor] = {}
    for rank in probed:
        inputs = dict(case["call_site_inputs"])
        inputs["expert_affinities"] = affinities
        # The no-slicing claim, stated as a reading rather than an assumption:
        # What goes in is the global width, which is not this rank's extent.
        assert tuple(inputs["expert_affinities"].shape) == (T, global_experts)
        assert global_experts != EP_LOCAL_EXPERTS
        _reset_limb_counters()
        with _MappingAffinitySpy() as spy:
            out = bank.block_quant_expert_mm(
                quant_config=quant_config,
                block_size=B,
                expert_parallel_rank=rank,
                **inputs,
            )
        seam_affinities = spy.only
        counters = _limb_counters()
        columns = torch.tensor(
            bank.local_expert_indices(rank), dtype=torch.int64
        )
        # The expectation is built by plain advanced indexing, which is a
        # different mechanism from the ``torch.gather`` the call site uses, so
        # this is a comparison and not a restatement.
        want = affinities[:, columns]
        got = seam_affinities
        assert tuple(seam_affinities.shape) == (T, EP_LOCAL_EXPERTS), (
            f"the mapping was handed {tuple(seam_affinities.shape)}; the whole "
            f"point of the dispatch step is that it sees "
            f"[T={T}, E_local={EP_LOCAL_EXPERTS}], and the padding slot is "
            f"appended after the mapping, not before it"
        )
        assert torch.equal(got, want), (
            f"rank {rank}'s gathered slice is not its own global columns "
            f"{tuple(columns.tolist())}"
        )
        assert tuple(out.shape) == (T, H)
        assert bool(torch.isfinite(out).all())
        assert float(out.abs().max()) > 0.0, (
            "the kernel returned an all-zero output, so nothing downstream of "
            "the dispatch actually computed"
        )
        assert counters == DECLARED_LIMB_DISPATCHES, (
            f"seam counters {counters}: the dispatch must not change how many "
            f"times the block-quant kernel is entered, nor take a fallback"
        )
        gathered[rank] = got

    # The rank argument is live: disjoint column sets give different slices.
    for index, left in enumerate(probed):
        for right in probed[index + 1 :]:
            assert not torch.equal(gathered[left], gathered[right]), (
                f"ranks {left} and {right} received the same slice over "
                f"disjoint columns, so the gather is not reading its rank"
            )

    # The default rank is documented as 0, and it is the same call as an
    # explicit 0. Read rather than asserted from the signature.
    inputs = dict(case["call_site_inputs"])
    inputs["expert_affinities"] = affinities
    with _MappingAffinitySpy() as spy_default:
        bank.block_quant_expert_mm(
            quant_config=quant_config, block_size=B, **inputs
        )
    with _MappingAffinitySpy() as spy_zero:
        bank.block_quant_expert_mm(
            quant_config=quant_config,
            block_size=B,
            expert_parallel_rank=0,
            **inputs,
        )
    same = bool(torch.equal(spy_default.only, spy_zero.only))
    assert same, "the default expert_parallel_rank is documented as rank 0"


def test_moe_path_dispatch_refuses_a_caller_sliced_local_form_above_degree_one() -> None:
    """Above degree 1 the call site refuses a caller-sliced ``[T, E_local]``. """
    from vllm_neuron.model.glm5_next.model_fp8 import Glm5NextBlockQuantRouteError

    bank, text_config, global_experts, _degree = _build_ep_bank()
    quant_config = _block_quant_config()
    case = _build_case()
    affinities, _expert_index = _ep_route(bank, global_experts, text_config)
    columns = torch.tensor(bank.local_expert_indices(1), dtype=torch.int64)
    pre_sliced = affinities[:, columns]
    assert tuple(pre_sliced.shape) == (T, EP_LOCAL_EXPERTS), (
        "the control must hand in exactly the local form a slicing caller "
        "would produce"
    )
    inputs = dict(case["call_site_inputs"])
    inputs["expert_affinities"] = pre_sliced
    _reset_limb_counters()
    with pytest.raises(Glm5NextBlockQuantRouteError, match="GLOBAL router width"):
        bank.block_quant_expert_mm(
            quant_config=quant_config,
            block_size=B,
            expert_parallel_rank=1,
            **inputs,
        )
    counters = _limb_counters()
    assert counters == NO_LIMB_DISPATCHES, (
        "the refusal must happen before the block-quant kernel is entered"
    )
