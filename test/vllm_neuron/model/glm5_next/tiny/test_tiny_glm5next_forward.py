"""Forward-pass tests for the tiny GLM-5.3-Flash configuration: the dense MLP.

This module also holds what the other ``tiny/`` test modules import: the fp8
block-scaled operand builders, the kernel dispatch counters and route check,
the torch references, and the tiny decoder stack and root fixtures.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
import torch

from test.vllm_neuron.model.glm5_next.test_mla_decode import paged_operands
from vllm_neuron.functional.blockwise_fp8_mm import SCALE_BLOCK_SIZE
from vllm_neuron.functional.moe.blockwise_fp8_retile import BLOCK_QUANT_SIZE, TILE_SIZE
from vllm_neuron.model.glm5_next.weight_loaders_fp8 import (
    compensate_block_scales,
    downscale_fp8_weight_bytes,
)

DENSE_BLOCK = SCALE_BLOCK_SIZE
_DENSE_BLOCKS_PER_REGIME = BLOCK_QUANT_SIZE // DENSE_BLOCK


def _per_dense_block(regimes: tuple[int, ...]) -> tuple[int, ...]:
    """Each scale exponent repeated over the dense blocks it covers."""
    return tuple(e for e in regimes for _ in range(_DENSE_BLOCKS_PER_REGIME))


pytestmark = [pytest.mark.fast]

_FP8 = torch.float8_e4m3fn

# The quantisation policy under test is read from this checkpoint config.
FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "config.json"

HIDDEN_SIZE = 256
INTERMEDIATE_SIZE = 1024
NUM_KEY_VALUE_HEADS = 2
MAX_HEAD_DIM = 128
MAX_PARAMETERS = 32_000_000

TOKENS = 128

RTOL = 1e-2
ATOL = 1e-5

MOE_RTOL = 3e-2
MOE_ATOL = 1e-5

SEED_HIDDEN = 5401
SEED_GATE = 5402
SEED_UP = 5403
SEED_DOWN = 5404

# Hidden states are fp8-grid values times 2**-3, so they are exact in bf16.
HIDDEN_SCALE_EXPONENT = -3


class VacuousControlError(AssertionError):
    """A precondition the test fixture itself must satisfy did not hold."""


class ReferenceShapeError(AssertionError):
    """A reference was handed operands it cannot combine."""


def _impl():
    """Import the modeling module lazily, inside test bodies."""
    from vllm_neuron.model.glm5_next import model_fp8

    return model_fp8


_SEAM_REGISTRY = {
    "blockwise_fp8_mm": (
        "vllm_neuron.functional.blockwise_fp8_mm",
        "dispatch_counters", "reset_dispatch_counters"),
    "blockwise_fp8_moe": (
        "vllm_neuron.functional.moe.moe_blockwise_fp8",
        "dispatch_counters", "reset_dispatch_counters"),
    "moe_fused": (
        "vllm_neuron.functional.moe.fused_fp8",
        "fused_dispatch_counters", "reset_fused_dispatch_counters"),
    "moe_gate_up": (
        "vllm_neuron.functional.moe.moe_blockwise_fp8",
        "gate_up_dispatch_counters", "reset_gate_up_dispatch_counters"),
    "moe_swiglu": (
        "vllm_neuron.functional.moe.moe_blockwise_fp8",
        "swiglu_dispatch_counters", "reset_swiglu_dispatch_counters"),
    "moe_down": (
        "vllm_neuron.functional.moe.moe_blockwise_fp8",
        "down_dispatch_counters", "reset_down_dispatch_counters"),
    "noaux_tc_router": (
        "vllm_neuron.functional.moe.router",
        "noaux_tc_dispatch_counters", "reset_noaux_tc_counters"),
    "mla_projection": (
        "vllm_neuron.functional.attention.mla_projections",
        "mla_projection_dispatch_counters",
        "reset_mla_projection_dispatch_counters"),
    "mla_absorb": (
        "vllm_neuron.functional.attention.mla_absorb",
        "mla_absorb_dispatch_counters", "reset_mla_absorb_dispatch_counters"),
    "mla_sparse": (
        "vllm_neuron.functional.attention.mla_sparse",
        "mla_sparse_dispatch_counters", "reset_mla_sparse_dispatch_counters"),
    "mla_sparse_tiled": (
        "vllm_neuron.functional.attention.mla_sparse",
        "mla_sparse_tiled_dispatch_counters",
        "reset_mla_sparse_tiled_dispatch_counters"),
    "mla_sparse_row_tiled": (
        "vllm_neuron.functional.attention.mla_sparse",
        "mla_sparse_row_tiled_dispatch_counters",
        "reset_mla_sparse_row_tiled_dispatch_counters"),
    "dsa_kpool_hadamard": (
        "vllm_neuron.functional.dsa.kpool_hadamard",
        "kpool_hadamard_dispatch_counters",
        "reset_kpool_hadamard_dispatch_counters"),
    "dsa_paged_gather": (
        "vllm_neuron.functional.dsa.paged_gather",
        "paged_gather_dispatch_counters", "reset_paged_gather_dispatch_counters"),
    "dsa_score_gemm": (
        "vllm_neuron.functional.dsa.score_gemm",
        "score_gemm_dispatch_counters", "reset_score_gemm_dispatch_counters"),
    "dsa_topk_select": (
        "vllm_neuron.functional.dsa.topk_select",
        "topk_select_dispatch_counters", "reset_topk_select_dispatch_counters"),
    "dsa_index_expand": (
        "vllm_neuron.functional.dsa.index_expand",
        "index_expand_dispatch_counters", "reset_index_expand_dispatch_counters"),
    "dsa_decode_tail_update": (
        "vllm_neuron.functional.dsa.decode_tail_update",
        "decode_tail_dispatch_counters", "reset_decode_tail_dispatch_counters"),
    "dsa_ragged_pack": (
        "vllm_neuron.functional.dsa.ragged_pack",
        "ragged_pack_dispatch_counters", "reset_ragged_pack_dispatch_counters"),
    "dsa_causal_fill": (
        "vllm_neuron.functional.dsa.causal_fill",
        "causal_fill_dispatch_counters", "reset_causal_fill_dispatch_counters"),
    "dsa_causal_bound": (
        "vllm_neuron.functional.dsa.causal_bound",
        "causal_bound_dispatch_counters", "reset_causal_bound_dispatch_counters"),
    "dsa_causal_sentinel": (
        "vllm_neuron.functional.dsa.causal_bound",
        "causal_sentinel_dispatch_counters",
        "reset_causal_sentinel_dispatch_counters"),
    "dsa_sentinel_order": (
        "vllm_neuron.functional.dsa.sentinel_order",
        "sentinel_order_dispatch_counters",
        "reset_sentinel_order_dispatch_counters"),
    "kda_chunked_recurrence": (
        "vllm_neuron.functional.kda.chunked_recurrence",
        "dispatch_counters", "reset_dispatch_counters"),
    "kda_chunked_recurrence_inter": (
        "vllm_neuron.functional.kda.chunked_recurrence",
        "inter_dispatch_counters", "reset_inter_dispatch_counters"),
    "kda_decode_state": (
        "vllm_neuron.functional.kda.decode_state",
        "decode_dispatch_counters", "reset_decode_dispatch_counters"),
    "kda_depthwise_conv1d": (
        "vllm_neuron.functional.kda.depthwise_conv1d",
        "dispatch_counters", "reset_dispatch_counters"),
    "kda_gate_clamp": (
        "vllm_neuron.functional.kda.gate_clamp",
        "gate_clamp_dispatch_counters", "reset_gate_clamp_dispatch_counters"),
    "mhc_hyper_connection": (
        "vllm_neuron.functional.mhc.hyper_connection",
        "dispatch_counters", "reset_dispatch_counters"),
    "mhc_sinkhorn": (
        "vllm_neuron.functional.mhc.sinkhorn",
        "dispatch_counters", "reset_dispatch_counters"),
    "vision_patch_embed": (
        "vllm_neuron.functional.vision.patch_embed",
        "dispatch_counters", "reset_dispatch_counters"),
}
_SEAMS = tuple(_SEAM_REGISTRY)

_COUNTER_SUFFIX = "dispatch_counters"


def _seam_counter_api(name: str) -> tuple:
    """``(read, reset)`` counter functions for one registered family."""
    path, reader, reset = _SEAM_REGISTRY[name]
    module = importlib.import_module(path)
    return getattr(module, reader), getattr(module, reset)


def _read_seam_counters() -> dict:
    """``{seam: (nki_dispatch, torch_fallback)}`` as each family reports itself."""
    return {name: tuple(int(v) for v in _seam_counter_api(name)[0]())
            for name in _SEAMS}


def _reset_seam_counters() -> None:
    """Zero every registered counter family."""
    for name in _SEAMS:
        _seam_counter_api(name)[1]()


def _assert_every_counter_family_is_registered() -> None:
    """Every counter family in every registered module is claimed by a registry row."""
    claimed: dict[str, set] = {}
    for name in _SEAMS:
        path, reader, _reset = _SEAM_REGISTRY[name]
        claimed.setdefault(path, set()).add(reader)
    for path, readers in claimed.items():
        module = importlib.import_module(path)
        found = {
            attribute for attribute in dir(module)
            if attribute.endswith(_COUNTER_SUFFIX)
            and not attribute.startswith("reset_")
        }
        if found != readers:
            raise VacuousControlError(
                f"{path} reports the counter families {sorted(found)} and this "
                f"file's registry claims {sorted(readers)}. An unclaimed family "
                f"is a kernel whose dispatches and torch fallbacks no route check "
                f"reads"
            )


def _assert_no_unregistered_counter_family() -> None:
    """No counter family anywhere in ``vllm_neuron.functional`` is unregistered."""
    import re

    package = importlib.import_module("vllm_neuron.functional")
    root = Path(package.__file__).resolve().parent
    pattern = re.compile(r"^def (\w*%s)\(" % _COUNTER_SUFFIX, re.M)
    found: set[tuple[str, str]] = set()
    for file in sorted(root.rglob("*.py")):
        dotted = "vllm_neuron.functional." + ".".join(
            file.relative_to(root).with_suffix("").parts
        )
        for reader in pattern.findall(file.read_text(errors="replace")):
            if reader.startswith("reset_"):
                continue
            found.add((dotted, reader))
    claimed = {(path, reader) for path, reader, _reset in _SEAM_REGISTRY.values()}
    if found != claimed:
        unclaimed = sorted(found - claimed)
        phantom = sorted(claimed - found)
        raise VacuousControlError(
            f"the package defines {len(found)} counter families and this file's "
            f"registry claims {len(claimed)}. Families no row claims: {unclaimed}. "
            f"Rows the package does not define: {phantom}. An unclaimed family is "
            f"a kernel whose torch fallbacks no route check totals"
        )


def _declare_bound_and_sentinel(expected: dict) -> None:
    """The three causal families dispatch once per top-k selection."""
    selector = expected.get("dsa_topk_select")
    if selector is None:
        return
    expected["dsa_causal_bound"] = selector
    expected["dsa_causal_sentinel"] = selector
    expected["dsa_sentinel_order"] = selector


def _assert_route_predicate(label: str, expected: dict, before: dict, after: dict) -> None:
    """The kernel route was live, no family fell back to torch, and exactly ``expected`` fired."""
    from vllm_neuron.utils.neuron_utils import can_run_kernel

    _assert_every_counter_family_is_registered()
    _assert_no_unregistered_counter_family()
    gate = bool(can_run_kernel(torch.zeros(1)))
    fired = {}
    fallbacks = 0
    for name in _SEAMS:
        dispatched = after[name][0] - before[name][0]
        fell_back = after[name][1] - before[name][1]
        fallbacks += fell_back
        if dispatched:
            fired[name] = dispatched
    if not gate:
        raise VacuousControlError(
            f"{label}: can_run_kernel() is False, so this forward ran on torch "
            f"rather than the NKI route and the comparison would pass either way. "
            f"Run with VLLM_NEURON_CPU_MODE=1 and NKI_SIMULATOR=1"
        )
    if fallbacks != 0:
        raise VacuousControlError(
            f"{label}: the torch-fallback counters total {fallbacks} across "
            f"{list(_SEAMS)}; expected exactly 0"
        )
    if not fired:
        raise VacuousControlError(
            f"{label}: no registered kernel dispatched at all, so this forward "
            f"exercised torch end to end and not the kernels"
        )
    if fired != expected:
        raise VacuousControlError(
            f"{label}: the kernels that fired were {sorted(fired.items())}; "
            f"expected {sorted(expected.items())}"
        )


def _pinned_raw_config() -> dict:
    """The checkpoint config these tests read their quantisation policy from."""
    return json.loads(FIXTURE_PATH.read_text())


def _quant_config():
    """``Glm5NextQuantConfig`` resolved from the pinned checkpoint config."""
    from vllm_neuron.model.glm5_next.config import Glm5NextConfig

    return _impl().Glm5NextQuantConfig.from_model_config(
        Glm5NextConfig.from_configs(_pinned_raw_config())
    )


def _tiny_text_config():
    """The tiny text config."""
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    return Glm5NextTextConfig(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_key_value_heads=NUM_KEY_VALUE_HEADS,
    )


def _fp8_grid_values(seed: int, *shape: int) -> torch.Tensor:
    """Values already on the fp8-e4m3 grid: multiples of ``1/8`` in ``[1/8, 7/8]``."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(1, 8, shape, generator=generator).to(torch.float32) / 8.0


def _pow2_scales(exponents: tuple[int, ...], rows: int) -> torch.Tensor:
    """A ``[rows, len(exponents)]`` fp32 grid of exact powers of two."""
    row = torch.tensor([float(2.0**e) for e in exponents], dtype=torch.float32)
    return row.repeat(rows, 1)


def _dequantise(weight_fp8: torch.Tensor, block_scale: torch.Tensor) -> torch.Tensor:
    """``weight[k, n] * scale[k // block_rows, n // block_cols]``, expanded, in fp32."""
    if weight_fp8.dim() != 2 or block_scale.dim() != 2:
        raise ReferenceShapeError(
            f"expected 2-D weight and 2-D block scale, got "
            f"{tuple(weight_fp8.shape)} and {tuple(block_scale.shape)}"
        )
    rows, cols = weight_fp8.shape
    grid_rows, grid_cols = block_scale.shape
    if grid_rows <= 0 or grid_cols <= 0 or rows % grid_rows or cols % grid_cols:
        raise ReferenceShapeError(
            f"block scale {tuple(block_scale.shape)} does not tile a "
            f"{rows}x{cols} weight at any whole granularity"
        )
    block_rows, block_cols = rows // grid_rows, cols // grid_cols
    if block_rows not in (DENSE_BLOCK, BLOCK_QUANT_SIZE) or block_cols not in (
        DENSE_BLOCK,
        BLOCK_QUANT_SIZE,
    ):
        raise ReferenceShapeError(
            f"block scale {tuple(block_scale.shape)} tiles a {rows}x{cols} weight at "
            f"{(block_rows, block_cols)}, which is neither the dense consumer's "
            f"{DENSE_BLOCK} nor the routed bank's {BLOCK_QUANT_SIZE}"
        )
    expanded = compensate_block_scales(block_scale).scale_inv.repeat_interleave(
        block_rows, dim=0
    ).repeat_interleave(block_cols, dim=1)
    return downscale_fp8_weight_bytes(weight_fp8).to(torch.float32) * expanded


def _scale_grid_attribute(leaf: str) -> str:
    """The attribute name the model stores a weight leaf's scale grid under."""
    return _impl().Glm5NextForConditionalGeneration._sibling_scale_grid_name(leaf)


def _attach(
    module,
    leaf: str,
    weight: torch.Tensor,
    grid: torch.Tensor,
    *,
    prep_will_compensate: bool = False,
) -> None:
    """Bind one fp8 weight and its block-scale grid to ``module``."""
    setattr(
        module,
        leaf,
        torch.nn.Parameter(
            downscale_fp8_weight_bytes(weight).clone(), requires_grad=False
        ),
    )
    delivered = grid if prep_will_compensate else compensate_block_scales(grid).scale_inv
    setattr(module, _scale_grid_attribute(leaf), delivered.clone())


def _prep_operands_from_the_module(module, leaves, fixture: dict) -> tuple:
    """The six arguments ``prepare_scale_operands`` takes, read off the module."""
    weights = tuple(getattr(module, leaf) for leaf in leaves)
    grids = tuple(getattr(module, _scale_grid_attribute(leaf)) for leaf in leaves)
    for leaf, weight, grid in zip(leaves, weights, grids):
        if weight is fixture[leaf][0] or grid is fixture[leaf][1]:
            raise VacuousControlError(
                f"{leaf} reached the prep as the fixture's own tensor, so the load's "
                f"squeeze and compensation were skipped and the test would measure "
                f"the checkpoint tensors rather than what a load stores"
            )
    return weights + grids


def _dense_operands() -> dict:
    """The dense MLP's three weights and three public block-scale grids."""
    blocks = INTERMEDIATE_SIZE // BLOCK_QUANT_SIZE
    if blocks != 4:
        raise VacuousControlError(
            f"this fixture's exponent choices are written for 4 column blocks; "
            f"INTERMEDIATE_SIZE={INTERMEDIATE_SIZE} gives {blocks}"
        )
    if TOKENS % TILE_SIZE:
        raise VacuousControlError(
            f"TOKENS={TOKENS} is not a whole number of TILE_SIZE={TILE_SIZE} rows; "
            f"the dense kernel would refuse this geometry before any numerics ran"
        )

    hidden = _fp8_grid_values(SEED_HIDDEN, TOKENS, HIDDEN_SIZE) * float(
        2.0**HIDDEN_SCALE_EXPONENT
    )

    gate_w = _fp8_grid_values(SEED_GATE, HIDDEN_SIZE, INTERMEDIATE_SIZE)
    up_w = _fp8_grid_values(SEED_UP, HIDDEN_SIZE, INTERMEDIATE_SIZE)
    # One negated block keeps the lower clamp on ``up`` live.
    up_w[:, -BLOCK_QUANT_SIZE:] = -up_w[:, -BLOCK_QUANT_SIZE:]
    down_w = _fp8_grid_values(SEED_DOWN, INTERMEDIATE_SIZE, HIDDEN_SIZE)

    k_blocks_parallel = HIDDEN_SIZE // DENSE_BLOCK
    k_blocks_down = INTERMEDIATE_SIZE // DENSE_BLOCK
    gate_s = _pow2_scales(_per_dense_block((-3, -1, 1, 3)), k_blocks_parallel)
    up_s = _pow2_scales(_per_dense_block((1, 3, -3, 1)), k_blocks_parallel)
    down_s = _pow2_scales((-3,) * (HIDDEN_SIZE // DENSE_BLOCK), k_blocks_down)
    down_s[-_DENSE_BLOCKS_PER_REGIME:, :] = float(2.0**3)

    return {
        "hidden": hidden.to(torch.bfloat16),
        "gate_proj_weight": (gate_w.to(_FP8), gate_s),
        "up_proj_weight": (up_w.to(_FP8), up_s),
        "down_proj_weight": (down_w.to(_FP8), down_s),
    }


def _clamped(
    tensor: torch.Tensor, low: float | None, high: float | None
) -> torch.Tensor:
    """``tensor`` clamped to the bounds given; no clamp at all when neither is."""
    if low is None and high is None:
        return tensor
    return tensor.clamp(min=low, max=high)


def _dense_output(
    operands: dict,
    gate_max: float | None,
    up_min: float | None,
    up_max: float | None,
) -> dict:
    """The dense MLP in torch, in fp32, with the clamp branches switchable."""
    from torch.nn.functional import silu

    x = operands["hidden"].to(torch.float32)
    gate = x @ _dequantise(*operands["gate_proj_weight"])
    up = x @ _dequantise(*operands["up_proj_weight"])

    gate_c = _clamped(gate, None, gate_max)
    up_c = _clamped(up, up_min, up_max)
    activated = silu(gate_c) * up_c

    down = activated.to(torch.bfloat16).to(torch.float32) @ _dequantise(
        *operands["down_proj_weight"]
    )
    return {"out": down, "gate": gate, "up": up}


def test_tiny_dense_mlp_forward_matches_the_reference() -> None:
    """The dense MLP on one layer, against the reference's clamped SwiGLU."""
    model_fp8 = _impl()
    text_config = _tiny_text_config()
    module = model_fp8.Glm5NextDenseMLP(text_config)

    limit = float(module.swiglu_limit)
    if limit != float(text_config.swiglu_limit):
        raise VacuousControlError(
            f"the module resolved swiglu_limit={limit} but the config declares "
            f"{text_config.swiglu_limit}; the bound under test would not be the "
            f"checkpoint's"
        )

    operands = _dense_operands()
    for leaf in ("gate_proj_weight", "up_proj_weight", "down_proj_weight"):
        _attach(module, leaf, *operands[leaf])

    parameters = sum(int(p.numel()) for p in module.parameters() if p is not None)
    if parameters >= MAX_PARAMETERS:
        raise VacuousControlError(
            f"the tiny dense MLP holds {parameters} parameters, at or above the "
            f"tiny-config bound of {MAX_PARAMETERS}"
        )

    reference = _dense_output(operands, limit, -limit, limit)

    above_gate = int((reference["gate"] > limit).sum())
    below_gate = int((reference["gate"] <= limit).sum())
    above_up = int((reference["up"] > limit).sum())
    within_up = int(((reference["up"] >= -limit) & (reference["up"] <= limit)).sum())
    below_up = int((reference["up"] < -limit).sum())
    for name, count in (
        ("gate above the bound", above_gate),
        ("gate at or below the bound", below_gate),
        ("up above the bound", above_up),
        ("up inside the bound", within_up),
        ("up below the negated bound", below_up),
    ):
        if count == 0:
            raise VacuousControlError(
                f"no element has {name}, so the test cannot tell the reference's "
                f"clamp from its absence"
            )

    for name, variant in (
        ("gate upper clamp", _dense_output(operands, None, -limit, limit)),
        ("up upper clamp", _dense_output(operands, limit, -limit, None)),
        ("up lower clamp", _dense_output(operands, limit, None, limit)),
    ):
        moved = not torch.allclose(
            variant["out"], reference["out"], rtol=RTOL, atol=ATOL
        )
        if not moved:
            raise VacuousControlError(
                f"removing the {name} leaves the result inside rtol={RTOL}, "
                f"atol={ATOL}; the test would pass with that branch deleted from "
                f"the module"
            )

    _reset_seam_counters()
    before = _read_seam_counters()
    got = module.forward(operands["hidden"], quant_config=_quant_config())
    after = _read_seam_counters()
    _assert_route_predicate("dense MLP", {"blockwise_fp8_mm": 3}, before, after)

    if tuple(got.shape) != (TOKENS, HIDDEN_SIZE):
        raise ReferenceShapeError(
            f"the forward returned {tuple(got.shape)}, expected "
            f"{(TOKENS, HIDDEN_SIZE)}"
        )
    torch.testing.assert_close(
        got.float(), reference["out"].float(), rtol=RTOL, atol=ATOL
    )

    # The same weights arranged the way the checkpoint loader leaves them:
    # transposed, with a scale grid at the checkpoint's tile granularity.
    tiles_per_block = DENSE_BLOCK // TILE_SIZE
    loaded = model_fp8.Glm5NextDenseMLP(text_config)
    for leaf in ("gate_proj_weight", "up_proj_weight", "down_proj_weight"):
        compute_weight, public_grid = operands[leaf]
        checkpoint_grid = public_grid.repeat_interleave(
            tiles_per_block, dim=0
        ).repeat_interleave(tiles_per_block, dim=1)
        _attach(
            loaded,
            leaf,
            compute_weight.t().contiguous(),
            checkpoint_grid.t().contiguous(),
            prep_will_compensate=True,
        )

    with pytest.raises(
        model_fp8.Glm5NextDenseMLPRouteError, match=r"gate_proj_weight must be \[H="
    ):
        loaded.forward(operands["hidden"], quant_config=_quant_config())

    retiled = loaded.retile_checkpoint_scale_grids()
    if retiled != 3:
        raise VacuousControlError(
            f"the load-path prep published {retiled} projections, not 3; at "
            f"[{HIDDEN_SIZE},{INTERMEDIATE_SIZE}] every extent is a whole "
            f"{DENSE_BLOCK} block, so a skip means it could not read them"
        )
    health = getattr(loaded, loaded.DENSE_RETILE_HEALTH_ATTR)
    for leaf in ("gate_proj_weight", "up_proj_weight", "down_proj_weight"):
        compute_weight, public_grid = operands[leaf]
        assert leaf in health, f"the retile health record has no entry for {leaf}"
        published = getattr(loaded, leaf)
        published_grid = getattr(loaded, _scale_grid_attribute(leaf))
        if tuple(published.shape) != tuple(compute_weight.shape):
            raise ReferenceShapeError(
                f"after the prep {leaf} is {tuple(published.shape)}; the frame the "
                f"kernel multiplies in is {tuple(compute_weight.shape)}"
            )
        if tuple(published_grid.shape) != tuple(public_grid.shape):
            raise ReferenceShapeError(
                f"after the prep {leaf}'s grid is {tuple(published_grid.shape)}; "
                f"the public grid for this weight is {tuple(public_grid.shape)}"
            )

    _reset_seam_counters()
    before_loaded = _read_seam_counters()
    got_loaded = loaded.forward(operands["hidden"], quant_config=_quant_config())
    after_loaded = _read_seam_counters()
    _assert_route_predicate(
        "dense MLP from the loader frame",
        {"blockwise_fp8_mm": 3},
        before_loaded,
        after_loaded,
    )
    torch.testing.assert_close(
        got_loaded.float(), reference["out"].float(), rtol=RTOL, atol=ATOL
    )

    # Compensating the grid once too few or once too many must change the result.
    for label in ("one_compensation_too_few", "one_compensation_too_many"):
        wrong = model_fp8.Glm5NextDenseMLP(text_config)
        for leaf in ("gate_proj_weight", "up_proj_weight", "down_proj_weight"):
            compute_weight, public_grid = operands[leaf]
            checkpoint_grid = public_grid.repeat_interleave(
                tiles_per_block, dim=0
            ).repeat_interleave(tiles_per_block, dim=1)
            loader_weight = compute_weight.t().contiguous()
            loader_grid = checkpoint_grid.t().contiguous()
            if label == "one_compensation_too_few":
                setattr(
                    wrong,
                    leaf,
                    torch.nn.Parameter(loader_weight.clone(), requires_grad=False),
                )
                setattr(wrong, _scale_grid_attribute(leaf), loader_grid.clone())
            else:
                _attach(wrong, leaf, loader_weight, loader_grid)
        if wrong.retile_checkpoint_scale_grids() != 3:
            raise VacuousControlError(
                f"the {label} control retiled fewer than 3 projections, so it is "
                f"not the arrangement this control means to refuse"
            )
        got_wrong = wrong.forward(operands["hidden"], quant_config=_quant_config())
        with pytest.raises(AssertionError):
            torch.testing.assert_close(
                got_wrong.float(), reference["out"].float(), rtol=RTOL, atol=ATOL
            )


ROUTED_HIDDEN_SIZE = 512
ROUTED_INTERMEDIATE_SIZE = 1024
ROUTED_EXPERTS = 16
ROUTED_EXPERTS_PER_TOKEN = 8

ROUTED_AFFINITIES = (0.75, 0.5, 0.375, 0.3125, 0.25, 0.1875, 0.09375, 0.03125)

ROUTED_GATE_EXPONENTS = (0, -2, -3, 0)
ROUTED_GATE_NEGATED_BLOCK = 3
ROUTED_UP_EXPONENTS = (-2, 0, 1, -1)
ROUTED_UP_NEGATED_BLOCK = 2

# One down block carries a very different scale, so a wrong H/I retile is visible.
ROUTED_DOWN_EXPONENT_BLOCK3 = 10
ROUTED_DOWN_EXPONENT_OTHER = -3

ROUTED_MAX_CONDITION = 4.0
ROUTED_MAX_BLOCK_SHARE = 1.5

SEED_ROUTED_HIDDEN = 5411
SEED_ROUTED_GATE = 5412
SEED_ROUTED_UP = 5413
SEED_ROUTED_DOWN = 5414

_POST_SCALE = "POST_SCALE"
_PRE_SCALE = "PRE_SCALE"


def _coarsen_to_256(grid: torch.Tensor) -> torch.Tensor:
    """A ``TILE_SIZE`` scale grid read back at ``BLOCK_QUANT_SIZE``."""
    step = BLOCK_QUANT_SIZE // TILE_SIZE
    if grid.dim() != 2 or grid.shape[0] % step or grid.shape[1] % step:
        raise ReferenceShapeError(
            f"a {TILE_SIZE} grid of shape {tuple(grid.shape)} does not tile at "
            f"{BLOCK_QUANT_SIZE}"
        )
    coarse = grid[::step, ::step].clone()
    rebuilt = coarse.repeat_interleave(step, dim=0).repeat_interleave(step, dim=1)
    if not torch.equal(rebuilt, grid):
        raise VacuousControlError(
            f"the {TILE_SIZE} scale grid is not uniform inside its "
            f"{BLOCK_QUANT_SIZE} blocks, so the load-time retile would have to "
            f"rescale inexactly and this fixture's exact-power-of-two premise is "
            f"gone"
        )
    return coarse


def _routed_tile_grid(
    exponents: tuple[int, ...],
    *,
    i_first: bool,
    intermediate: int = ROUTED_INTERMEDIATE_SIZE,
    hidden: int = ROUTED_HIDDEN_SIZE,
    experts: int = ROUTED_EXPERTS,
) -> torch.Tensor:
    """``[E, I/128, H/128]`` powers of two, one exponent per 256-block of I."""
    step = BLOCK_QUANT_SIZE // TILE_SIZE
    i_tiles = intermediate // TILE_SIZE
    h_tiles = hidden // TILE_SIZE
    per_tile = [float(2.0 ** exponents[tile // step]) for tile in range(i_tiles)]
    grid = torch.tensor(per_tile, dtype=torch.float32).reshape(i_tiles, 1)
    grid = grid.repeat(1, h_tiles)
    if not i_first:
        grid = grid.t().contiguous()
    return grid.unsqueeze(0).repeat(experts, 1, 1).contiguous()


def _routed_affinities() -> torch.Tensor:
    """``[T, E]`` scattered top-8 router scores, as ``route_tokens`` returns them."""
    affinities = torch.zeros(TOKENS, ROUTED_EXPERTS, dtype=torch.float32)
    for token in range(TOKENS):
        for role, weight in enumerate(ROUTED_AFFINITIES):
            affinities[token, (token + role) % ROUTED_EXPERTS] = weight
    selected = int((affinities != 0).sum(dim=1).min())
    if selected != ROUTED_EXPERTS_PER_TOKEN:
        raise VacuousControlError(
            f"a token carries {selected} nonzero router columns and this fixture "
            f"declares top-{ROUTED_EXPERTS_PER_TOKEN}; the block mapping is built "
            f"from that count and would disagree with the mask"
        )
    return affinities


def _routed_operands() -> dict:
    """The bank's three weights, three ``TILE_SIZE`` grids, hidden states, affinities."""
    blocks = ROUTED_INTERMEDIATE_SIZE // BLOCK_QUANT_SIZE
    if blocks != len(ROUTED_GATE_EXPONENTS) or blocks != len(ROUTED_UP_EXPONENTS):
        raise VacuousControlError(
            f"this fixture declares {len(ROUTED_GATE_EXPONENTS)} gate and "
            f"{len(ROUTED_UP_EXPONENTS)} up regimes for {blocks} blocks of I"
        )
    if ROUTED_HIDDEN_SIZE % BLOCK_QUANT_SIZE or ROUTED_HIDDEN_SIZE // BLOCK_QUANT_SIZE < 2:
        raise VacuousControlError(
            f"ROUTED_HIDDEN_SIZE={ROUTED_HIDDEN_SIZE} must be a multiple of "
            f"{BLOCK_QUANT_SIZE} and give at least two blocks along H, or the down "
            f"retile's H and I roles are indistinguishable and the test would pass "
            f"with that mapping wrong"
        )

    hidden = _fp8_grid_values(
        SEED_ROUTED_HIDDEN, TOKENS, ROUTED_HIDDEN_SIZE
    ) * float(2.0**HIDDEN_SCALE_EXPONENT)

    gate_w = _fp8_grid_values(
        SEED_ROUTED_GATE, ROUTED_EXPERTS, ROUTED_INTERMEDIATE_SIZE, ROUTED_HIDDEN_SIZE
    )
    up_w = _fp8_grid_values(
        SEED_ROUTED_UP, ROUTED_EXPERTS, ROUTED_INTERMEDIATE_SIZE, ROUTED_HIDDEN_SIZE
    )
    down_w = _fp8_grid_values(
        SEED_ROUTED_DOWN, ROUTED_EXPERTS, ROUTED_HIDDEN_SIZE, ROUTED_INTERMEDIATE_SIZE
    )
    for negated, weight in ((ROUTED_GATE_NEGATED_BLOCK, gate_w),
                            (ROUTED_UP_NEGATED_BLOCK, up_w)):
        rows = slice(negated * BLOCK_QUANT_SIZE, (negated + 1) * BLOCK_QUANT_SIZE)
        weight[:, rows, :] = -weight[:, rows, :]

    down_exponents = tuple(
        ROUTED_DOWN_EXPONENT_BLOCK3 if block == ROUTED_GATE_NEGATED_BLOCK
        else ROUTED_DOWN_EXPONENT_OTHER
        for block in range(blocks)
    )
    return {
        "hidden": hidden.to(torch.bfloat16),
        "expert_affinities": _routed_affinities(),
        "gate_proj_weight": (gate_w.to(_FP8),
                             _routed_tile_grid(ROUTED_GATE_EXPONENTS, i_first=True)),
        "up_proj_weight": (up_w.to(_FP8),
                           _routed_tile_grid(ROUTED_UP_EXPONENTS, i_first=True)),
        "down_proj_weight": (down_w.to(_FP8),
                             _routed_tile_grid(down_exponents, i_first=False)),
    }


def _routed_output(
    operands: dict,
    *,
    mode: str,
    gate_max: float | None,
    gate_min: float | None,
    up_max: float | None,
    up_min: float | None,
) -> dict:
    """The routed expert bank in torch, in fp32, with the scaling point and the four clamp bounds switchable."""
    from torch.nn.functional import silu

    if mode not in (_POST_SCALE, _PRE_SCALE):
        raise ReferenceShapeError(f"unknown scaling point {mode!r}")
    hidden = operands["hidden"].to(torch.float32)
    affinities = operands["expert_affinities"]
    gate_w, gate_grid = operands["gate_proj_weight"]
    up_w, up_grid = operands["up_proj_weight"]
    down_w, down_grid = operands["down_proj_weight"]
    tokens, hidden_size = hidden.shape[0], hidden.shape[1]
    experts = int(affinities.shape[1])
    blocks = down_w.shape[2] // BLOCK_QUANT_SIZE

    terms = [torch.zeros(tokens, hidden_size, dtype=torch.float32)
             for _ in range(blocks)]
    gates, ups = [], []
    for expert in range(experts):
        weight = affinities[:, expert:expert + 1]
        activations = hidden * weight if mode == _PRE_SCALE else hidden
        gate = activations @ _dequantise(
            gate_w[expert], _coarsen_to_256(gate_grid[expert])
        ).t()
        up = activations @ _dequantise(
            up_w[expert], _coarsen_to_256(up_grid[expert])
        ).t()
        gates.append(gate)
        ups.append(up)

        clamped = silu(_clamped(gate, gate_min, gate_max)) * _clamped(
            up, up_min, up_max
        )
        clamped = clamped.to(torch.bfloat16).to(torch.float32)
        down = _dequantise(down_w[expert], _coarsen_to_256(down_grid[expert]))
        for block in range(blocks):
            columns = slice(block * BLOCK_QUANT_SIZE, (block + 1) * BLOCK_QUANT_SIZE)
            term = clamped[:, columns] @ down[:, columns].t()
            terms[block] += term * weight if mode == _POST_SCALE else term

    stack = torch.stack(terms)
    out = stack.sum(dim=0)
    scale = float(out.abs().max())
    return {
        "out": out,
        "blocks": stack,
        "gate": torch.stack(gates),
        "up": torch.stack(ups),
        "condition": float(stack.abs().sum(dim=0).max()) / scale if scale else float("inf"),
        "share": float(stack.abs().max()) / scale if scale else float("inf"),
    }


def _routed_text_config():
    """The text config at the routed bank's widths."""
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    return Glm5NextTextConfig(
        hidden_size=ROUTED_HIDDEN_SIZE,
        intermediate_size=ROUTED_INTERMEDIATE_SIZE,
        num_key_value_heads=NUM_KEY_VALUE_HEADS,
        n_routed_experts=ROUTED_EXPERTS,
        num_experts_per_tok=ROUTED_EXPERTS_PER_TOKEN,
    )


SHARED_AT_ROUTED_GATE_EXPONENTS = (0, -3, 0, -3)
SHARED_AT_ROUTED_UP_EXPONENTS = (-3, 0, -3, 0)
SHARED_AT_ROUTED_UP_NEGATED_BLOCK = 3
SHARED_AT_ROUTED_DOWN_EXPONENT = -3

SEED_SHARED_GATE = 5421
SEED_SHARED_UP = 5422
SEED_SHARED_DOWN = 5423
SEED_MOE_HIDDEN = 5424
SEED_MOE_ROUTER = 5425

MOE_GAMMA_VALUES = (1.0, 1.25, 1.5, 1.75)

MOE_ROUTER_WEIGHT_SCALE = 0.1
MOE_ROUTER_BIAS_SCALE = 0.05


def _shared_at_routed_operands(
    *,
    seed_offset: int = 0,
    gate_exponents: tuple = SHARED_AT_ROUTED_GATE_EXPONENTS,
    up_exponents: tuple = SHARED_AT_ROUTED_UP_EXPONENTS,
    down_exponent: int = SHARED_AT_ROUTED_DOWN_EXPONENT,
) -> dict:
    """A shared expert's three weights and public grids at the bank's hidden size."""
    h = ROUTED_HIDDEN_SIZE
    i = ROUTED_INTERMEDIATE_SIZE
    blocks = i // BLOCK_QUANT_SIZE
    if blocks != len(gate_exponents) or blocks != len(up_exponents):
        raise VacuousControlError(
            f"this call declares {len(gate_exponents)} gate and "
            f"{len(up_exponents)} up regimes for {blocks} blocks of I"
        )

    gate_w = _fp8_grid_values(SEED_SHARED_GATE + seed_offset, h, i)
    up_w = _fp8_grid_values(SEED_SHARED_UP + seed_offset, h, i)
    columns = slice(
        SHARED_AT_ROUTED_UP_NEGATED_BLOCK * BLOCK_QUANT_SIZE,
        (SHARED_AT_ROUTED_UP_NEGATED_BLOCK + 1) * BLOCK_QUANT_SIZE,
    )
    up_w[:, columns] = -up_w[:, columns]
    down_w = _fp8_grid_values(SEED_SHARED_DOWN + seed_offset, i, h)

    h_blocks = h // DENSE_BLOCK
    i_blocks = i // DENSE_BLOCK
    return {
        "gate_proj_weight": (
            gate_w.to(_FP8),
            _pow2_scales(_per_dense_block(gate_exponents), h_blocks),
        ),
        "up_proj_weight": (
            up_w.to(_FP8),
            _pow2_scales(_per_dense_block(up_exponents), h_blocks),
        ),
        "down_proj_weight": (
            down_w.to(_FP8),
            _pow2_scales((down_exponent,) * h_blocks, i_blocks),
        ),
    }


def _ffn_gamma() -> torch.Tensor:
    """``[H]`` the FFN norm's gain, repeating :data:`MOE_GAMMA_VALUES` along H."""
    row = torch.tensor(MOE_GAMMA_VALUES, dtype=torch.float32)
    return row.repeat(ROUTED_HIDDEN_SIZE // len(MOE_GAMMA_VALUES))


def _ffn_norm(hidden: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm in fp32 with a gain, cast back to the input dtype."""
    x = hidden.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    normed = x * torch.rsqrt(variance + eps)
    normed = normed * gamma.to(torch.float32)
    return normed.to(hidden.dtype)


MLA_HIDDEN_SIZE = 256
MLA_HEADS = 4
MLA_Q_LORA_RANK = 128
MLA_KV_LORA_RANK = 128
MLA_QK_NOPE_HEAD_DIM = 64
MLA_QK_ROPE_HEAD_DIM = 0
MLA_V_HEAD_DIM = 64
MLA_INDEX_N_HEADS = 4
MLA_INDEX_HEAD_DIM = 128
MLA_INDEX_KPOOL = 4
MLA_TOPK_POOLS = 2
MLA_PAGE_SIZE = 4
MLA_PAGES = 8

# Not a multiple of the pool or the page, so partial pools and pages are exercised.
MLA_TOKENS = 35

SEED_MLA = 5431

MLA_SOFTMAX_SCALE = float(
    (MLA_QK_NOPE_HEAD_DIM + MLA_QK_ROPE_HEAD_DIM) ** -0.5
)


def _mla_text_config(**overrides):
    """The checkpoint's text config at the tiny MLA geometry."""
    from dataclasses import replace

    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    dials = {
        "hidden_size": MLA_HIDDEN_SIZE,
        "num_key_value_heads": NUM_KEY_VALUE_HEADS,
        "num_attention_heads": MLA_HEADS,
        "q_lora_rank": MLA_Q_LORA_RANK,
        "kv_lora_rank": MLA_KV_LORA_RANK,
        "qk_nope_head_dim": MLA_QK_NOPE_HEAD_DIM,
        "qk_rope_head_dim": MLA_QK_ROPE_HEAD_DIM,
        "v_head_dim": MLA_V_HEAD_DIM,
        "index_n_heads": MLA_INDEX_N_HEADS,
        "index_head_dim": MLA_INDEX_HEAD_DIM,
        "index_kpool": MLA_INDEX_KPOOL,
        "index_topk": MLA_TOPK_POOLS * MLA_INDEX_KPOOL,
    }
    dials.update(overrides)
    return replace(Glm5NextTextConfig(), **dials)


def _mla_contract(x: torch.Tensor, weight_out_in: torch.Tensor) -> torch.Tensor:
    """One projection, from the checkpoint-shaped ``[out, in]`` leaf, in fp32."""
    return x.to(torch.float32) @ weight_out_in.to(torch.float32).t()


def _mla_latent_norm(
    x: torch.Tensor, gain: torch.Tensor, eps: float
) -> torch.Tensor:
    """RMSNorm on a latent: ``x / sqrt(mean(x**2) + eps) * gain``, fp32, no cast."""
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + float(eps)) * gain.to(torch.float32)


def _materialise_mla_attention(
    attention, cfg, *, seed: int, output_damping: float = 1.0
) -> tuple[dict, dict]:
    """Fill every leaf of a built attention module, run both preps, and fill its indexer."""
    gen = torch.Generator().manual_seed(int(seed))

    raw: dict = {}
    for name, in_features, out_features in attention.projection_widths():
        weight = torch.randn(
            out_features, in_features, generator=gen, dtype=torch.float32
        ) * (in_features ** -0.5)
        if name == "o_proj":
            weight = weight * float(output_damping)
        raw[name] = weight
        setattr(attention, f"{name}_weight", torch.nn.Parameter(weight))

    gains: dict = {}
    for gain_name, width in (
        ("q_a_layernorm_weight", int(cfg.q_lora_rank)),
        ("kv_a_layernorm_weight", int(cfg.kv_lora_rank)),
    ):
        gain = 1.0 + torch.randn(width, generator=gen, dtype=torch.float32) * 0.05
        gains[gain_name] = gain
        setattr(attention, gain_name, torch.nn.Parameter(gain))

    prepared = attention.prepare_projection_weights()
    absorbed = attention.prepare_absorb_weights()
    if (prepared, absorbed) != (5, 2):
        raise VacuousControlError(
            f"the load-time preps built {prepared} projection(s) and {absorbed} "
            f"absorb operand(s); this geometry declares five and two"
        )
    _materialise_mla_indexer(attention.indexer, gen)
    return raw, gains


def _mla_attention_fixture(*, seed: int = SEED_MLA):
    """One MLA attention module, every leaf materialised and both preps run."""
    model_fp8 = _impl()
    cfg = _mla_text_config()
    attention = model_fp8.Glm5NextMLAAttention(cfg)
    raw, gains = _materialise_mla_attention(attention, cfg, seed=seed)

    parameters = sum(int(p.numel()) for p in attention.parameters())
    if parameters >= MAX_PARAMETERS:
        raise VacuousControlError(
            f"the attention module carries {parameters} parameters, at or past "
            f"the tiny-config bound of {MAX_PARAMETERS}"
        )
    widths = {
        "qk_nope_head_dim": int(cfg.qk_nope_head_dim),
        "qk_rope_head_dim": int(cfg.qk_rope_head_dim),
        "v_head_dim": int(cfg.v_head_dim),
        "index_head_dim": int(cfg.index_head_dim),
    }
    over = {name: width for name, width in widths.items() if width > MAX_HEAD_DIM}
    if over:
        raise VacuousControlError(
            f"{sorted(over.items())} exceeds the tiny-config bound "
            f"head_dim <= {MAX_HEAD_DIM}"
        )
    return attention, raw, gains, cfg


def _materialise_mla_indexer(indexer, gen: torch.Generator) -> None:
    """The indexer's seven leaves, then its own load-time prep."""
    for name, in_features, out_features in indexer.projection_widths():
        weight = torch.randn(
            out_features, in_features, generator=gen, dtype=torch.float32
        ) * (in_features ** -0.5)
        setattr(
            indexer, indexer.PROJECTION_PARAMETERS[name], torch.nn.Parameter(weight)
        )
    head_dim = int(indexer.index_head_dim)
    pool = int(indexer.index_kpool)
    indexer.k_norm_weight = torch.nn.Parameter(
        1.0 + torch.randn(head_dim, generator=gen, dtype=torch.float32) * 0.05
    )
    indexer.k_norm_bias = torch.nn.Parameter(
        torch.randn(head_dim, generator=gen, dtype=torch.float32) * 0.02
    )
    indexer.index_kpool_compress_ape = torch.nn.Parameter(
        (torch.randn(pool, head_dim, generator=gen, dtype=torch.float32) * 0.1).to(
            torch.bfloat16
        ),
        requires_grad=False,
    )
    prepared = indexer.prepare_projection_weights()
    if prepared != 4:
        raise VacuousControlError(
            f"the indexer prep built {prepared} projections, not four"
        )


def _mla_selection_operands(
    *, tokens: int = MLA_TOKENS, pages: int = MLA_PAGES
) -> dict:
    """The operands the selection stage needs, each derived from its own rule."""
    slots = torch.full((tokens,), -1, dtype=torch.int32)
    for position in range(tokens):
        if (position + 1) % MLA_INDEX_KPOOL == 0:
            slots[position] = position // MLA_INDEX_KPOOL
    candidates = tokens // MLA_INDEX_KPOOL
    rows = pages * MLA_PAGE_SIZE
    if candidates <= MLA_TOPK_POOLS:
        raise VacuousControlError(
            f"{candidates} candidate pool(s) is not more than the "
            f"{MLA_TOPK_POOLS} selected, and the indexer serves the bypass "
            f"regime there instead of selecting"
        )
    if candidates > rows - 1:
        raise VacuousControlError(
            f"{candidates} candidates leaves no trash row above them in {rows} "
            f"pooled-key rows"
        )
    return {
        "slot_mapping": slots,
        "seq_lens": torch.arange(1, tokens + 1, dtype=torch.int32),
        "candidates": candidates,
        "pool_rows": rows,
    }


def _mla_pool_cache(*, pages: int = MLA_PAGES) -> torch.Tensor:
    """The pooled-key store, ``[rows, index_head_dim]`` bf16, written in place."""
    return torch.zeros(
        pages * MLA_PAGE_SIZE, MLA_INDEX_HEAD_DIM, dtype=torch.bfloat16
    )


def _mla_latent_cache(attention, *, tokens: int = MLA_TOKENS) -> torch.Tensor:
    """The latent cache at the layer's own declared spec, in whole blocks."""
    return torch.zeros(
        -(-int(tokens) // MLA_PAGE_SIZE) * MLA_PAGE_SIZE,
        attention.NUM_LATENT_KV_HEADS, int(attention.head_size),
        dtype=torch.float32,
    )


def _mla_dense_reference(
    attention,
    raw: dict,
    gains: dict,
    normed: torch.Tensor,
    latent_cache: torch.Tensor,
    topk_indices: torch.Tensor,
    *,
    softmax_scale: float,
    honour_sentinels: bool = True,
) -> torch.Tensor:
    """Dense MLA attention in torch, from the same weights. ``[tokens, hidden]``."""
    tokens = int(normed.shape[0])
    heads = int(attention.num_attention_heads)
    nope = int(attention.qk_nope_head_dim)
    vdim = int(attention.v_head_dim)
    eps = float(attention.rms_norm_eps)
    x = normed.to(torch.float32)

    q_latent = _mla_latent_norm(
        _mla_contract(x, raw["q_a_proj"]), gains["q_a_layernorm_weight"], eps
    )
    query = _mla_contract(q_latent, raw["q_b_proj"]).reshape(tokens, heads, nope)
    kv_latent = _mla_latent_norm(
        _mla_contract(x, raw["kv_a_proj_with_mqa"]),
        gains["kv_a_layernorm_weight"],
        eps,
    )

    latent_cache[0:tokens, 0, :] = kv_latent.to(latent_cache.dtype)
    c_kv = latent_cache[:tokens, 0, :].to(torch.float32)

    key_value = _mla_contract(c_kv, raw["kv_b_proj"]).reshape(
        tokens, heads, nope + vdim
    )
    key_nope = key_value[..., :nope]
    value = key_value[..., nope:]

    index = topk_indices.to(torch.int64)
    keep = index >= 0
    rows = index.clamp(min=0)
    out = torch.empty(tokens, heads, vdim, dtype=torch.float32)
    for token in range(tokens):
        gathered_key = key_nope[rows[token]]
        gathered_value = value[rows[token]]
        scores = torch.einsum("hd,khd->hk", query[token], gathered_key)
        if honour_sentinels:
            scores = scores.masked_fill(~keep[token], float("-inf"))
        weights = torch.nan_to_num(
            torch.softmax(scores * float(softmax_scale), dim=-1)
        )
        out[token] = torch.einsum("hk,khv->hv", weights, gathered_value)

    flat = out.reshape(tokens, heads * vdim)
    return _mla_contract(flat, raw["o_proj"]).to(normed.dtype)


STACK_HIDDEN_SIZE = ROUTED_HIDDEN_SIZE

STACK_LAYERS = 3

STACK_FIRST_K_DENSE = 2
STACK_DENSE_LAYERS = STACK_FIRST_K_DENSE
STACK_MOE_LAYERS = STACK_LAYERS - STACK_FIRST_K_DENSE

STACK_TOKENS = TOKENS

STACK_VOCAB_SIZE = 256

STACK_PAGES = 16

STACK_DENSE_INTERMEDIATE_SIZE = ROUTED_INTERMEDIATE_SIZE

STACK_MOE_INTERMEDIATE_SIZE = 512

STACK_EXPERTS = ROUTED_EXPERTS
STACK_EXPERTS_PER_TOKEN = ROUTED_EXPERTS_PER_TOKEN

# Keeps the attention half small next to the residual streams.
STACK_ATTENTION_DAMPING = 0.125

STACK_BANK_GATE_EXPONENTS = (-3, -3)
STACK_BANK_UP_EXPONENTS = (-3, -3)
STACK_BANK_DOWN_EXPONENTS = (-3, -3)

STACK_MAX_CONDITION = 32.0

SEED_STACK_EMBED = 5441
SEED_STACK_IDS = 5442
SEED_STACK_ATTENTION = 5443
SEED_STACK_BANK_GATE = 5461
SEED_STACK_BANK_UP = 5462
SEED_STACK_BANK_DOWN = 5463
SEED_STACK_ROUTER = 5464
SEED_STACK_MHC = 5471

MHC_SITES_PER_LAYER = 2

STACK_DENSE_SEED_OFFSETS = (100, 200)

# The dense layers' scales are shifted down so the SwiGLU clamp stays live
# without saturating every row into one constant.
STACK_DENSE_GATE_UP_SHIFT = -3
STACK_DENSE_DOWN_SHIFT = -6
STACK_DENSE_GATE_EXPONENTS = tuple(
    e + STACK_DENSE_GATE_UP_SHIFT for e in SHARED_AT_ROUTED_GATE_EXPONENTS
)
STACK_DENSE_UP_EXPONENTS = tuple(
    e + STACK_DENSE_GATE_UP_SHIFT for e in SHARED_AT_ROUTED_UP_EXPONENTS
)
STACK_DENSE_DOWN_EXPONENT = SHARED_AT_ROUTED_DOWN_EXPONENT + STACK_DENSE_DOWN_SHIFT

_STACK_GAIN_VALUES = (1.0, 1.25, 1.5, 1.75, 0.75, 1.125, 0.875, 1.375)


def _stack_gain(site: int) -> torch.Tensor:
    """``[H]`` the norm gain for one site, its own rotation of the eight values."""
    row = torch.tensor(_STACK_GAIN_VALUES, dtype=torch.float32).roll(int(site))
    return row.repeat(STACK_HIDDEN_SIZE // len(_STACK_GAIN_VALUES))


def _stack_text_config(**overrides):
    """The tiny stack's text config: the MLA geometry plus the stack's own fields."""
    from vllm_neuron.model.glm5_next.config import DSA_LAYER_TYPE

    layer_types = overrides.pop("layer_types", [DSA_LAYER_TYPE] * STACK_LAYERS)
    return _mla_text_config(
        hidden_size=STACK_HIDDEN_SIZE,
        intermediate_size=STACK_DENSE_INTERMEDIATE_SIZE,
        moe_intermediate_size=STACK_MOE_INTERMEDIATE_SIZE,
        num_hidden_layers=STACK_LAYERS,
        layer_types=layer_types,
        first_k_dense_replace=STACK_FIRST_K_DENSE,
        n_routed_experts=STACK_EXPERTS,
        num_experts_per_tok=STACK_EXPERTS_PER_TOKEN,
        n_shared_experts=0,
        vocab_size=STACK_VOCAB_SIZE,
        **overrides,
    )


def _stack_geometry_preconditions() -> None:
    """Every extent this stack chose, checked against the kernel that constrains it."""
    from vllm_neuron.functional.moe.moe_blockwise_fp8 import (
        MAX_HIDDEN,
        MIN_HIDDEN,
        NUM_SHARDS,
        PSUM_SIZE,
    )

    problems = []
    if not MIN_HIDDEN <= STACK_HIDDEN_SIZE <= MAX_HIDDEN:
        problems.append(
            f"hidden {STACK_HIDDEN_SIZE} outside the MoE kernel's "
            f"[{MIN_HIDDEN}, {MAX_HIDDEN}]"
        )
    if STACK_HIDDEN_SIZE % PSUM_SIZE or STACK_HIDDEN_SIZE % BLOCK_QUANT_SIZE:
        problems.append(
            f"hidden {STACK_HIDDEN_SIZE} is not a multiple of PSUM_SIZE={PSUM_SIZE} "
            f"and of BLOCK_QUANT_SIZE={BLOCK_QUANT_SIZE}"
        )
    if STACK_MOE_INTERMEDIATE_SIZE % (BLOCK_QUANT_SIZE * NUM_SHARDS):
        problems.append(
            f"bank I_TP {STACK_MOE_INTERMEDIATE_SIZE} is not a multiple of "
            f"BLOCK_QUANT_SIZE * NUM_SHARDS = {BLOCK_QUANT_SIZE * NUM_SHARDS}"
        )
    if STACK_TOKENS % TILE_SIZE:
        problems.append(
            f"tokens {STACK_TOKENS} is not a whole number of TILE_SIZE={TILE_SIZE} "
            f"rows, which the dense kernel refuses and does not pad"
        )
    if STACK_HIDDEN_SIZE % len(_STACK_GAIN_VALUES):
        problems.append(
            f"hidden {STACK_HIDDEN_SIZE} is not a multiple of "
            f"{len(_STACK_GAIN_VALUES)}, so the gain pattern does not tile it"
        )
    if problems:
        raise VacuousControlError(
            "this stack's geometry is inadmissible: " + "; ".join(problems)
        )


def _stack_bank_operands() -> dict:
    """The bank's three weights and three ``TILE_SIZE`` grids at the stack's widths."""
    h = STACK_HIDDEN_SIZE
    i = STACK_MOE_INTERMEDIATE_SIZE
    e = STACK_EXPERTS
    blocks = i // BLOCK_QUANT_SIZE
    for label, exponents in (
        ("gate", STACK_BANK_GATE_EXPONENTS),
        ("up", STACK_BANK_UP_EXPONENTS),
        ("down", STACK_BANK_DOWN_EXPONENTS),
    ):
        if len(exponents) != blocks:
            raise VacuousControlError(
                f"the bank's {label} projection declares {len(exponents)} scale "
                f"regimes for {blocks} blocks of I={i}"
            )
    grid_extents = dict(intermediate=i, hidden=h, experts=e)
    return {
        "gate_proj_weight": (
            _fp8_grid_values(SEED_STACK_BANK_GATE, e, i, h).to(_FP8),
            _routed_tile_grid(
                STACK_BANK_GATE_EXPONENTS, i_first=True, **grid_extents),
        ),
        "up_proj_weight": (
            _fp8_grid_values(SEED_STACK_BANK_UP, e, i, h).to(_FP8),
            _routed_tile_grid(
                STACK_BANK_UP_EXPONENTS, i_first=True, **grid_extents),
        ),
        "down_proj_weight": (
            _fp8_grid_values(SEED_STACK_BANK_DOWN, e, h, i).to(_FP8),
            _routed_tile_grid(
                STACK_BANK_DOWN_EXPONENTS, i_first=False, **grid_extents),
        ),
    }


def _stack_leaf_shapes(cfg) -> dict:
    """The checkpoint shape of each of the six mHC leaves, derived from the config."""
    from vllm_neuron.model.glm5_next.weight_loaders_fp8 import MHC_LEAVES

    hc_mult = int(cfg.hc_mult)
    hidden = int(cfg.hidden_size)
    mix = (2 + hc_mult) * hc_mult
    by_role = {"fn": (mix, hc_mult * hidden), "base": (mix,), "scale": (3,)}
    return {leaf: by_role[leaf.split("_")[2]] for leaf in MHC_LEAVES}


def _stack_load_the_six(layer, cfg, *, seed: int) -> dict:
    """Place a random tensor on each of one layer's six mHC leaves."""
    generator = torch.Generator().manual_seed(seed)
    placed = {}
    for leaf, shape in sorted(_stack_leaf_shapes(cfg).items()):
        tensor = torch.randn(shape, generator=generator, dtype=torch.float32) * 0.1
        setattr(layer, leaf, torch.nn.Parameter(tensor, requires_grad=False))
        placed[leaf] = tensor
    return placed


def _stack_fixture(model=None) -> dict:
    """The whole tiny stack, every mapped tensor bound and every load-time prep run."""
    model_fp8 = _impl()
    from vllm_neuron.model.glm5_next.config import Glm5NextConfig

    _stack_geometry_preconditions()
    cfg = _stack_text_config()
    if model is None:
        model = model_fp8.Glm5NextModel(Glm5NextConfig(text_config=cfg), 1)
    layers = list(model.layers)
    if len(layers) != STACK_LAYERS:
        raise VacuousControlError(
            f"the config declares {STACK_LAYERS} layers and the tree built "
            f"{len(layers)}"
        )

    dense_at = [
        index for index, layer in enumerate(layers)
        if isinstance(layer.mlp, model_fp8.Glm5NextDenseMLP)
    ]
    moe_at = [
        index for index, layer in enumerate(layers)
        if isinstance(layer.mlp, model_fp8.Glm5NextMoEBlock)
    ]
    if dense_at != list(range(STACK_DENSE_LAYERS)) or moe_at != list(
        range(STACK_DENSE_LAYERS, STACK_LAYERS)
    ):
        raise VacuousControlError(
            f"_build_mlp put dense MLPs at {dense_at} and expert blocks at "
            f"{moe_at}; the references are written for the first "
            f"{STACK_DENSE_LAYERS} dense and the rest sparse"
        )
    if len(dense_at) != len(STACK_DENSE_SEED_OFFSETS):
        raise VacuousControlError(
            f"{len(dense_at)} dense layers and {len(STACK_DENSE_SEED_OFFSETS)} "
            f"seed offsets; two layers sharing a draw cannot show that each read "
            f"its own weights"
        )

    table = (
        _fp8_grid_values(SEED_STACK_EMBED, STACK_VOCAB_SIZE, STACK_HIDDEN_SIZE)
        * float(2.0**HIDDEN_SCALE_EXPONENT)
    ).to(torch.bfloat16)
    model.embed_tokens_weight = torch.nn.Parameter(table, requires_grad=False)
    final_gain = _stack_gain(2 * STACK_LAYERS)
    model.norm_weight = torch.nn.Parameter(final_gain, requires_grad=False)

    hc_mult = int(cfg.hc_mult)
    if hc_mult < 2:
        raise VacuousControlError(
            f"the config declares hc_mult={hc_mult}; one stream would make every "
            f"comparison below a plain residual add, which is the network the "
            f"hyper-connection route exists to replace"
        )

    attention_operands = []
    mlp_operands: dict = {}
    mhc_operands: dict = {}
    for index, layer in enumerate(layers):
        layer.input_layernorm_weight = torch.nn.Parameter(
            _stack_gain(2 * index), requires_grad=False
        )
        layer.post_attention_layernorm_weight = torch.nn.Parameter(
            _stack_gain(2 * index + 1), requires_grad=False
        )
        mhc_operands[index] = _stack_load_the_six(
            layer, cfg, seed=SEED_STACK_MHC + index
        )
        bound = layer.bind_hyper_connection_sites(cfg, torch.device("cpu"))
        if bound != MHC_SITES_PER_LAYER:
            raise VacuousControlError(
                f"layer {index} bound {bound} mHC sites and this stack needs "
                f"{MHC_SITES_PER_LAYER}: one for the attention half and one for the "
                f"feed-forward half; a layer short of a site refuses its own forward"
            )
        attention_operands.append(
            _materialise_mla_attention(
                layer.self_attn,
                cfg,
                seed=SEED_STACK_ATTENTION + index,
                output_damping=STACK_ATTENTION_DAMPING,
            )
        )
        if index in dense_at:
            operands = _shared_at_routed_operands(
                seed_offset=STACK_DENSE_SEED_OFFSETS[dense_at.index(index)],
                gate_exponents=STACK_DENSE_GATE_EXPONENTS,
                up_exponents=STACK_DENSE_UP_EXPONENTS,
                down_exponent=STACK_DENSE_DOWN_EXPONENT,
            )
            for leaf in ("gate_proj_weight", "up_proj_weight", "down_proj_weight"):
                _attach(layer.mlp, leaf, *operands[leaf])
            mlp_operands[index] = operands
            continue

        operands = _stack_bank_operands()
        for leaf in ("gate_proj_weight", "up_proj_weight", "down_proj_weight"):
            _attach(layer.mlp.experts, leaf, *operands[leaf])
        built = layer.mlp.experts.prepare_scale_operands(
            *_prep_operands_from_the_module(
                layer.mlp.experts, ("gate_proj_weight", "up_proj_weight", "down_proj_weight"), operands
            )
        )
        if built != 2:
            raise VacuousControlError(
                f"the bank's load-time prep built {built} operands; its forward "
                f"looks up 2"
            )
        if getattr(layer.mlp, "shared_experts", None) is not None:
            raise VacuousControlError(
                "the block built a shared expert at n_shared_experts=0, so the "
                "sparse layer's reference would be missing a term"
            )
        generator = torch.Generator().manual_seed(SEED_STACK_ROUTER)
        layer.mlp.experts.router_weight = torch.nn.Parameter(
            (
                torch.randn(
                    STACK_HIDDEN_SIZE, STACK_EXPERTS, generator=generator
                )
                * MOE_ROUTER_WEIGHT_SCALE
            ).to(torch.bfloat16),
            requires_grad=False,
        )
        layer.mlp.experts.router_bias = torch.nn.Parameter(
            (
                torch.randn(STACK_EXPERTS, generator=generator)
                * MOE_ROUTER_BIAS_SCALE
            ).to(torch.bfloat16),
            requires_grad=False,
        )
        mlp_operands[index] = operands

    for index, layer in enumerate(layers):
        health = getattr(layer, _impl().MHC_BIND_HEALTH_ATTR, None)
        if health is None:
            raise VacuousControlError(
                f"layer {index} carries no mHC bind record, so the bind above did not "
                f"run on this layer and its two sites are not this fixture's"
            )

    parameters = sum(int(p.numel()) for p in model.parameters() if p is not None)
    widths = {
        "qk_nope_head_dim": int(cfg.qk_nope_head_dim),
        "qk_rope_head_dim": int(cfg.qk_rope_head_dim),
        "v_head_dim": int(cfg.v_head_dim),
        "index_head_dim": int(cfg.index_head_dim),
    }
    if parameters >= MAX_PARAMETERS:
        raise VacuousControlError(
            f"the tiny stack holds {parameters} parameters, at or past the "
            f"tiny-config bound of {MAX_PARAMETERS}"
        )
    over = {name: width for name, width in widths.items() if width > MAX_HEAD_DIM}
    if over:
        raise VacuousControlError(
            f"{sorted(over.items())} exceeds the tiny-config bound "
            f"head_dim <= {MAX_HEAD_DIM}"
        )
    return {
        "model": model,
        "cfg": cfg,
        "layers": layers,
        "table": table,
        "final_gain": final_gain,
        "attention_operands": attention_operands,
        "mlp_operands": mlp_operands,
        "mhc_operands": mhc_operands,
        "hc_mult": hc_mult,
        "dense_at": dense_at,
        "moe_at": moe_at,
    }


def _stack_carriers(layers, selection: dict) -> list:
    """One carrier mapping per layer, in stack order, each with its own caches."""
    banks = [
        _mla_latent_cache(layer.self_attn, tokens=STACK_TOKENS) for layer in layers
    ]
    return [
        {
            "latent_cache": bank,
            "pool_cache": _mla_pool_cache(pages=STACK_PAGES),
            "seq_lens": selection["seq_lens"],
            "start_position": 0,
            "softmax_scale": MLA_SOFTMAX_SCALE,
            "max_seq_len": STACK_TOKENS,
            "page_size": MLA_PAGE_SIZE,
            "slot_mapping": selection["slot_mapping"],
            **paged_operands(bank, 0, STACK_TOKENS, page=MLA_PAGE_SIZE),
        }
        for bank in banks
    ]


def _stack_mhc_site(layer, which: str, label: str):
    """The bound mHC site for one half of a layer."""
    impl = _impl()
    sites = getattr(layer, impl.MHC_SITES_ATTR, {})
    if not sites or which not in sites:
        raise VacuousControlError(
            f"{label}: this layer has {sorted(sites)} bound and this comparison needs "
            f"the {which!r} site. The streams route runs one site per half, so an "
            f"unbound site means the fixture's load-time bind did not reach this layer"
        )
    return sites[which]


def _stack_mhc_pre(site, streams, label: str):
    """``(post_mix, comb_mix, layer_input)`` for one half, from the site's ``mhc_pre``."""
    post_mix, comb_mix, layer_input = site.mhc_pre(streams)
    tokens, hc_mult, hidden = (int(v) for v in streams.shape)
    expected = ((tokens, hc_mult, 1), (tokens, hc_mult, hc_mult), (tokens, hidden))
    got = (tuple(post_mix.shape), tuple(comb_mix.shape), tuple(layer_input.shape))
    if got != expected:
        raise ReferenceShapeError(
            f"{label}: the site's pre returned {got} for {tuple(streams.shape)} "
            f"streams and this comparison is written for {expected}; the post gate is one "
            f"weight per stream, the combine is stream by stream and the collapsed "
            f"input is the single stream the sublayer takes"
        )
    return post_mix, comb_mix, layer_input


def _stack_mhc_post(site, half, streams, post_mix, comb_mix):
    """The mixed streams in float32, from the site's own combine."""
    return site.mhc_post(half.float(), streams.float(), post_mix, comb_mix)


def _stack_attention_half(layer, raw, gains, hidden, cfg, selection):
    """One layer's attention half in torch, from the tensor that layer received."""
    normed = _ffn_norm(
        hidden, layer.input_layernorm_weight, float(cfg.rms_norm_eps)
    )
    attention = layer.self_attn
    q_latent = attention.project_query_latent(normed)
    topk_indices = attention.indexer(
        normed,
        q_latent,
        _mla_pool_cache(pages=STACK_PAGES),
        selection["seq_lens"],
        max_seq_len=STACK_TOKENS,
        page_size=MLA_PAGE_SIZE,
        slot_mapping=selection["slot_mapping"],
    )
    attended = _mla_dense_reference(
        attention,
        raw,
        gains,
        normed.float(),
        _mla_latent_cache(attention, tokens=STACK_TOKENS),
        topk_indices,
        softmax_scale=MLA_SOFTMAX_SCALE,
    )
    return normed, topk_indices, attended


def _stack_ffn_half(layer, hidden, cfg, operands, *, routed: bool, gain=None,
                    router_input=None, expert_input=None) -> dict:
    """One layer's FFN half in torch, from the tensor that layer's MLP received."""
    limit = float(cfg.swiglu_limit)
    if gain is None:
        gain = layer.post_attention_layernorm_weight
    normed = _ffn_norm(hidden, gain, float(cfg.rms_norm_eps))
    if expert_input is not None:
        normed = expert_input
    if not routed:
        return _dense_output({**operands, "hidden": normed}, limit, -limit, limit)
    if router_input is None:
        router_input = hidden
    _logits, _index, affinities = layer.mlp.experts.route_tokens(
        router_input.unsqueeze(0), gain, cfg
    )
    return _routed_output(
        {**operands, "hidden": normed, "expert_affinities": affinities},
        mode=_POST_SCALE,
        gate_max=limit,
        gate_min=None,
        up_max=limit,
        up_min=-limit,
    )


# Small head scale keeps the logits well conditioned.
ROOT_HEAD_SCALE_EXPONENT = -8

SEED_ROOT_HEAD = 5481

# The last row, the first row, and one position asked for twice.
ROOT_SAMPLING_POSITIONS = (STACK_TOKENS - 1, 0, 7, 7)

ROOT_MAX_CONDITION = STACK_MAX_CONDITION


def _root_config(**overrides):
    """The root's ``Glm5NextConfig``: the tiny stack plus the pinned checkpoint's quantisation fields."""
    from vllm_neuron.model.glm5_next.config import Glm5NextConfig

    raw = _pinned_raw_config()["quantization_config"]
    text = _stack_text_config(**overrides)
    exempt = raw.get("modules_to_not_convert")
    return Glm5NextConfig(
        text_config=text,
        tie_word_embeddings=bool(text.tie_word_embeddings),
        quant_method=raw["quant_method"],
        activation_scheme=raw["activation_scheme"],
        weight_block_size=list(raw["weight_block_size"]),
        modules_to_not_convert=None if exempt is None else list(exempt),
        fmt=raw["fmt"],
    )


def _root_head_weight() -> torch.Tensor:
    """``[vocab, hidden]`` head weight, on the unsigned fp8 grid and exact in bf16."""
    return (
        _fp8_grid_values(SEED_ROOT_HEAD, STACK_VOCAB_SIZE, STACK_HIDDEN_SIZE)
        * float(2.0**ROOT_HEAD_SCALE_EXPONENT)
    ).to(torch.bfloat16)


def _root_fixture(**overrides) -> dict:
    """The root module with the tiny stack bound inside it, plus its head tensor."""
    model_fp8 = _impl()
    config = _root_config(**overrides)
    root = model_fp8.Glm5NextForConditionalGeneration(config)
    if int(root.world_size) != 1:
        raise VacuousControlError(
            f"the root resolved world_size={root.world_size}; every width in this "
            f"fixture is written for a single rank"
        )
    fixture = _stack_fixture(model=root.model)
    cfg = fixture["cfg"]
    mismatched = {
        name: (getattr(root.text_config, name), getattr(cfg, name))
        for name in ("hidden_size", "num_hidden_layers", "vocab_size",
                     "first_k_dense_replace")
        if getattr(root.text_config, name) != getattr(cfg, name)
    }
    if mismatched:
        raise VacuousControlError(
            f"the root's text config and this fixture's disagree on "
            f"{sorted(mismatched.items())}, so the references are written for a "
            f"different tree than the one that ran"
        )

    tied = bool(root.text_config.tie_word_embeddings)
    declared = root.declared_parameter_names()
    if tied:
        head = fixture["table"]
    else:
        if "lm_head_weight" not in declared:
            raise VacuousControlError(
                "the untied root declares no lm_head_weight, so there is no head "
                "tensor to bind"
            )
        head = _root_head_weight()
        root.lm_head_weight = torch.nn.Parameter(head, requires_grad=False)

    parameters = sum(int(p.numel()) for p in root.parameters() if p is not None)
    if parameters >= MAX_PARAMETERS:
        raise VacuousControlError(
            f"the root holds {parameters} parameters with its head bound, at or "
            f"past the tiny-config bound of {MAX_PARAMETERS}"
        )
    fixture.update(root=root, config=config, head=head, tied=tied)
    return fixture


def _root_reference(hidden: torch.Tensor, head: torch.Tensor,
                    positions) -> torch.Tensor:
    """``[rows, vocab]`` logits in fp32: gather the named rows, then project."""
    rows = torch.stack([hidden[int(index)].float() for index in positions])
    return rows @ head.float().t()
