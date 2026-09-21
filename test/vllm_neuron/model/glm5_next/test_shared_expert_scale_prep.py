# SPDX-License-Identifier: Apache-2.0
"""The shared-expert kernel scale operand is built once, at load time.

``to_kernel_scale_layout`` allocates and then scatters one element at a time, and the
shared-expert path enters that seam three times per call, so at production geometry the
operand was rebuilt by hundreds of one-element device writes on every forward step. A
block scale never changes after a checkpoint load, so the operand is built once at load
time and handed to the seam by keyword.

The build counter below counts builds the seam performs; ``prepare_scale_operands``
calls the bridge directly, so its own one-time build is not counted. Withholding the
operand at the seam, with the prep still running, is what moves that count to 3.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from vllm_neuron.functional.blockwise_fp8_mm import (
    SCALE_BLOCK_SIZE,
    TILE_SIZE,
    BlockwiseFp8MmError,
    blockwise_fp8_mm,
    dispatch_counters,
    kernel_scale_shape,
    reset_dispatch_counters,
    reset_scale_layout_builds,
    scale_grid_shape,
    scale_layout_builds,
    to_kernel_scale_layout,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel


def _impl():
    """Import the modeling module inside a test body, never at import time.

    This directory's convention, so a test that looks the architecture class up lazily is
    not defeated by a module-level import here. The functional imports at the top of this
    file are a different module and stay there.
    """
    from vllm_neuron.model.glm5_next import model_fp8

    return model_fp8


# --------------------------------------------------------------------------- #
# The geometry, and why it is not the production one.
# --------------------------------------------------------------------------- #
# Every reading here is a count, an exact equality or a named refusal, and none varies
# with extent, so the extents only have to be admissible. A small admissible geometry
# keeps the kernel simulator inside a 60-second timeout. Both extents are derived from
# the kernel's own block size rather than typed, so they follow it if it changes.
HIDDEN = 4 * SCALE_BLOCK_SIZE  # 512: four whole scale blocks
INTERMEDIATE = 4 * SCALE_BLOCK_SIZE  # 512: four whole scale blocks
TOKENS = TILE_SIZE  # one whole tile of rows
FP8 = torch.float8_e4m3fn

PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def _pinned_fixture() -> dict:
    """The checkpoint config the quantisation policy under test is read from."""
    fixture = Path(__file__).resolve().parent / "fixtures" / "config.json"
    return json.loads(fixture.read_text())


def _fixture(seed: int = 0):
    """One shared-expert module and one set of operands, seeded.

    The weights are fp8 bytes and the scales are the public grids ``scale_grid_shape``
    declares. The values are seeded, so every equality below is reproducible.

    The SwiGLU bound is left at the config's default: no reading here measures it, and
    both arms of every comparison share whatever it is.
    """
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    torch.manual_seed(seed)
    text_config = Glm5NextTextConfig(
        hidden_size=HIDDEN,
        moe_intermediate_size=INTERMEDIATE,
        n_shared_experts=1,
    )
    module = _impl().Glm5NextSharedExperts(text_config)
    gu_grid = scale_grid_shape(HIDDEN, INTERMEDIATE)
    dn_grid = scale_grid_shape(INTERMEDIATE, HIDDEN)
    operands = {
        "hidden_states": torch.randn(TOKENS, HIDDEN, dtype=torch.bfloat16),
        "gate_proj_weight": torch.randn(HIDDEN, INTERMEDIATE).to(FP8),
        "up_proj_weight": torch.randn(HIDDEN, INTERMEDIATE).to(FP8),
        "down_proj_weight": torch.randn(INTERMEDIATE, HIDDEN).to(FP8),
        "gate_proj_scale": torch.rand(*gu_grid, dtype=torch.float32) + 0.5,
        "up_proj_scale": torch.rand(*gu_grid, dtype=torch.float32) + 0.5,
        "down_proj_scale": torch.rand(*dn_grid, dtype=torch.float32) + 0.5,
    }
    return module, operands


def _quant_config() -> object:
    """``Glm5NextQuantConfig`` for the pinned checkpoint, with nothing hand-fed.

    The policy comes from the digest-checked fixture through
    ``Glm5NextConfig.from_configs``, so this file cannot route on a block shape the
    checkpoint does not carry.
    """
    from vllm_neuron.model.glm5_next.config import Glm5NextConfig

    return _impl().Glm5NextQuantConfig.from_model_config(
        Glm5NextConfig.from_configs(_pinned_fixture())
    )


def _run_prepared(module, operands, quant_config) -> torch.Tensor:
    """One shared-expert call with the operands built once, beforehand."""
    module.prepare_scale_operands(
        operands["gate_proj_weight"],
        operands["up_proj_weight"],
        operands["down_proj_weight"],
        operands["gate_proj_scale"],
        operands["up_proj_scale"],
        operands["down_proj_scale"],
    )
    return module.shared_expert_mm(
        hidden_states=operands["hidden_states"],
        gate_proj_weight=operands["gate_proj_weight"],
        up_proj_weight=operands["up_proj_weight"],
        down_proj_weight=operands["down_proj_weight"],
        gate_proj_scale=operands["gate_proj_scale"],
        up_proj_scale=operands["up_proj_scale"],
        down_proj_scale=operands["down_proj_scale"],
        quant_config=quant_config,
    )


def _withhold_at_the_seam(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the seam build the operand itself, without skipping the prep.

    ``shared_expert_mm`` imports the seam inside its own body, so replacing the module
    attribute replaces what the three call sites reach. The replacement forwards every
    argument except ``prebuilt_scale_t``, so the same three dispatches run over the same
    operands and only the builder of the scale operand differs.

    The module is fetched with ``import_module`` rather than ``import ... as``, because
    the package re-exports the seam function under the submodule's own name, so the
    plain import would bind the function instead of the module the call sites read.
    """
    import importlib

    seam = importlib.import_module("vllm_neuron.functional.blockwise_fp8_mm")
    real = seam.blockwise_fp8_mm

    def without_prebuilt(x, weight, weight_scale, *, prebuilt_scale_t=None):
        return real(x, weight, weight_scale)

    monkeypatch.setattr(seam, "blockwise_fp8_mm", without_prebuilt)


# --------------------------------------------------------------------------- #
# The per-forward build count.
# --------------------------------------------------------------------------- #
def test_the_seam_builds_no_scale_operand_per_forward_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The seam builds no scale layout per forward step, and three when it must."""
    quant_config = _quant_config()

    module, operands = _fixture()
    reset_scale_layout_builds()
    reset_dispatch_counters()
    _run_prepared(module, operands, quant_config)
    prepared_builds = scale_layout_builds()
    prepared_dispatch = dispatch_counters()

    control_module, control_operands = _fixture()
    with monkeypatch.context() as patched:
        _withhold_at_the_seam(patched)
        reset_scale_layout_builds()
        reset_dispatch_counters()
        _run_prepared(control_module, control_operands, quant_config)
        control_builds = scale_layout_builds()
        control_dispatch = dispatch_counters()

    assert prepared_builds == 0, (
        "the prebuilt operand was supplied at all three sites, so the seam must "
        f"not have built one; it built {prepared_builds}"
    )
    # The count has to move: a counter that never advances would also report zero.
    assert control_builds == 3, (
        "withholding the operand at the seam must make the seam build one per "
        f"projection, so 3; it built {control_builds}"
    )
    # Both arms took the kernel route the same number of times, so the zero is
    # not a zero reached by skipping the seam.
    assert prepared_dispatch == control_dispatch == (3, 0)


# --------------------------------------------------------------------------- #
# The operand and the output are the same bytes either way.
# --------------------------------------------------------------------------- #
def test_the_prebuilt_operand_and_the_output_are_the_same_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Moving the build to load time moves no bit of the arithmetic."""
    quant_config = _quant_config()

    module, operands = _fixture()
    prepared_output = _run_prepared(module, operands, quant_config)

    # Every operand, against the bridge run directly on the same public grid.
    equal_rows = 0
    max_operand_diff = 0.0
    for name, (rows, cols) in (
        ("gate_proj", (HIDDEN, INTERMEDIATE)),
        ("up_proj", (HIDDEN, INTERMEDIATE)),
        ("down_proj", (INTERMEDIATE, HIDDEN)),
    ):
        expected = to_kernel_scale_layout(operands[f"{name}_scale"], rows, cols)
        got = module._prepared_scale_operand(name)
        max_operand_diff = max(
            max_operand_diff, float((got - expected).abs().max())
        )
        if torch.equal(got, expected):
            equal_rows += 1

    control_module, control_operands = _fixture()
    with monkeypatch.context() as patched:
        _withhold_at_the_seam(patched)
        control_output = _run_prepared(
            control_module, control_operands, quant_config
        )

    output_max_abs_diff = float((prepared_output - control_output).abs().max())

    # The prepared operand carries the shape the kernel helper declares, and the output
    # is finite; both were only ever reported before.
    assert (
        tuple(module._prepared_scale_operand("gate_proj").shape)
        == kernel_scale_shape(HIDDEN, INTERMEDIATE)
    )
    assert bool(torch.isfinite(prepared_output).all())

    # Non-vacuity: an all-zero output would satisfy a difference of zero.
    assert float(prepared_output.abs().max()) > 0.0
    assert equal_rows == 3, f"only {equal_rows} of 3 operands matched the bridge"
    assert max_operand_diff == 0.0
    # EXACT, and no tolerance is introduced: both paths compute one function
    # from one set of bytes, so anything but zero is a real defect.
    assert output_max_abs_diff == 0.0


# --------------------------------------------------------------------------- #
# The two refusals.
# --------------------------------------------------------------------------- #
def test_both_named_refusals_fire_by_name() -> None:
    """A caller that skips the prep, and one that prepares the wrong shape, both raise."""
    quant_config = _quant_config()
    module, operands = _fixture()
    route_error = _impl().Glm5NextSharedExpertRouteError

    # The prep did not run.
    with pytest.raises(route_error) as skipped:
        module.shared_expert_mm(
            hidden_states=operands["hidden_states"],
            gate_proj_weight=operands["gate_proj_weight"],
            up_proj_weight=operands["up_proj_weight"],
            down_proj_weight=operands["down_proj_weight"],
            gate_proj_scale=operands["gate_proj_scale"],
            up_proj_scale=operands["up_proj_scale"],
            down_proj_scale=operands["down_proj_scale"],
            quant_config=quant_config,
        )
    names_the_method = "prepare_scale_operands()" in str(skipped.value)

    # The operand was prepared for a different weight.
    wrong = torch.zeros(TILE_SIZE, 99, dtype=torch.float32)
    with pytest.raises(BlockwiseFp8MmError) as mis_shaped:
        blockwise_fp8_mm(
            operands["hidden_states"],
            operands["gate_proj_weight"],
            operands["gate_proj_scale"],
            prebuilt_scale_t=wrong,
        )
    names_the_shape_helper = "kernel_scale_shape" in str(mis_shaped.value)

    assert names_the_method, str(skipped.value)
    assert names_the_shape_helper, str(mis_shaped.value)


# --------------------------------------------------------------------------- #
# The operand still reaches the kernel.
# --------------------------------------------------------------------------- #
def test_the_route_predicate_counts_one_dispatch_and_no_fallback_per_call() -> None:
    """One kernel dispatch per call and no fallback, three per shared-expert call.

    An operand moved to load time that stopped reaching the kernel would satisfy every
    exact reading above; this is the reading that would break.
    """
    module, operands = _fixture()
    module.prepare_scale_operands(
        operands["gate_proj_weight"],
        operands["up_proj_weight"],
        operands["down_proj_weight"],
        operands["gate_proj_scale"],
        operands["up_proj_scale"],
        operands["down_proj_scale"],
    )

    # Per call.
    reset_dispatch_counters()
    blockwise_fp8_mm(
        operands["hidden_states"],
        operands["gate_proj_weight"],
        operands["gate_proj_scale"],
        prebuilt_scale_t=module._prepared_scale_operand("gate_proj"),
    )
    per_call = dispatch_counters()

    # And over one whole shared-expert call.
    reset_dispatch_counters()
    _run_prepared(module, operands, _quant_config())
    per_shared_expert_call = dispatch_counters()

    route_available = can_run_kernel(operands["hidden_states"])

    assert route_available is True
    assert per_call == (1, 0), f"per call the seam read {per_call}, wanted (1, 0)"
    assert per_shared_expert_call == (3, 0)
