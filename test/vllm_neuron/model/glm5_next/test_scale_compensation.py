"""The dense and shared scale grids are compensated exactly once."""

from __future__ import annotations

import pytest
import torch

from vllm_neuron.functional.blockwise_fp8_mm import SCALE_BLOCK_SIZE
from vllm_neuron.model.glm5_next.quantization import DEFAULT_WEIGHT_BLOCK_SIZE
from vllm_neuron.model.glm5_next.weight_loaders_fp8 import (
    MAPPED_KEY_QUANTISED_WEIGHT,
    BlockScaleCompensation,
    classify_mapped_keys,
    compensate_block_scales,
    dequantise_blockwise,
    loader_for_mapped_keys,
    needs_240_downscale,
)

#: The two magnitudes the pair is built from -- the legacy and ocp e4m3 maxima.
FP8_E4M3_MAX = 240.0
FP8_E4M3FN_MAX = 448.0

#: What a missing compensation costs, per projection and through one MLP.
#:
#: A later round made the squeeze the largest power of two that fits 448 inside 240
#: rather than the ratio of the two, so this is 0.5 and not 240/448. It is written as the
#: literal it is: the ratio would be the wrong relation, and the load path's own constant is
#: pinned against the same literal in ``test_weight_loaders.py``.
#:
#: which comparisons become exact. ``x 1/2`` is a pure exponent shift, so every fp8 value at
#: or above ``2**-5`` survives the round trip bit-exactly; the eight that do not are the odd
#: multiples of ``2**-9``. :func:`_ratio` is a ratio of sums weighted by magnitude, so those
#: eight contribute essentially nothing and every reading below becomes exact up to fp32
#: summation noise, where at 240/448 the same statistic sat about 1.4% high. The bands are
#: not Re-tuned for that -- a band is not this file's to move, and gaining margin is not a
#: reason to touch one. The exact-element count is asserted instead, so the gain is
#: visible as a number rather than folded into a tolerance.
SQUEEZE = 0.5                                     # was FP8_E4M3_MAX / FP8_E4M3FN_MAX
MLP_SQUEEZE = SQUEEZE**3                          # 0.125, was 0.1537445335276968

BLOCK = 256
#: the checkpoint's TILE, and the grid the module carries.
#: imported from the dense kernel rather than typed: the publish now hands the
#: checkpoint's own grid to ``blockwise_fp8_mm``, which indexes at
#: ``SCALE_BLOCK_SIZE``, so a literal here would be a second place for that number to
#: live and would go stale the day the kernel's granularity moves again.
TILE = SCALE_BLOCK_SIZE
HIDDEN = 2 * BLOCK          # 512, a whole number of 256-blocks so the retile runs
INTERMEDIATE = 2 * BLOCK    # 512
FP8 = torch.float8_e4m3fn
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def _impl():
    """Import the modeling module inside a test body -- this package's convention."""
    from vllm_neuron.model.glm5_next import model_fp8

    return model_fp8


@pytest.fixture(autouse=True)
def _the_gate_is_engaged() -> None:
    """Fail rather than report on a platform where the squeeze is a no-op."""
    if not needs_240_downscale():
        pytest.fail(
            "needs_240_downscale() is False, so the 240 squeeze and the compensation are "
            "both no-ops and every value in this file would pass without measuring "
            "anything. The clamp maximum is resolved at import, so a fixture cannot "
            "change it: run with NEURON_PLATFORM_TARGET_OVERRIDE=trn2 in the invocation."
        )


def _checkpoint_tensors(rows: int, cols: int, seed: int):
    """One projection in checkpoint format: fp8-e4m3fn bytes and its own 128-tile grid.
    """
    torch.manual_seed(seed)
    raw_bytes = (torch.randn(rows, cols) * 40.0).to(FP8)
    grid = torch.rand(rows // TILE, cols // TILE, dtype=torch.float32) * 0.02 + 0.01
    return raw_bytes, grid


def _reference_matrix(raw_bytes: torch.Tensor, raw_grid: torch.Tensor) -> torch.Tensor:
    """The checkpoint's own numbers: raw bytes against the raw grid, un-squeezed. """
    return dequantise_blockwise(raw_bytes, raw_grid, DEFAULT_WEIGHT_BLOCK_SIZE).to(
        torch.float32
    )


def _stored_bytes_through_the_real_loader(raw_bytes: torch.Tensor) -> torch.Tensor:
    """The bytes the actual loader stores, taken from the actual dispatch. """
    keys = [
        "model.language_model.layers.0.mlp.gate_proj.weight",
        "model.language_model.layers.0.mlp.gate_proj.weight_scale_inv",
    ]
    assert classify_mapped_keys(keys) == MAPPED_KEY_QUANTISED_WEIGHT, (
        "this reading depends on the quantised-weight branch of loader_for_mapped_keys; "
        "the mapping above no longer classifies as one"
    )
    loader = loader_for_mapped_keys(keys, param_name="gate_proj_weight")
    stored = loader.load([raw_bytes], 0)
    assert stored.dtype is FP8, f"the loader stored {stored.dtype}, not fp8 bytes"
    return stored


def _loaded_module(kind: str):
    """A shared expert or a dense MLP, built from the package's own config type."""
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    # Both intermediate widths are set. The shared expert reads
    # ``moe_intermediate_size`` and the dense MLP reads ``intermediate_size``, and while
    # neither decides anything here -- the prep takes its extents off the attached weight
    # -- leaving one at its 12288 default would put a module's own declared width at odds
    # with the tensors this test hands it, which reads like an oversight.
    text_config = Glm5NextTextConfig(
        hidden_size=HIDDEN,
        moe_intermediate_size=INTERMEDIATE,
        intermediate_size=INTERMEDIATE,
        n_shared_experts=1,
    )
    if kind == "shared":
        return _impl().Glm5NextSharedExperts(text_config)
    return _impl().Glm5NextDenseMLP(text_config)


def _attach_checkpoint_format(module, seed: int) -> dict[str, tuple]:
    """Put checkpoint-format weights and raw grids on the module, as the load path does.
    """
    truth = {}
    for index, leaf in enumerate(PROJECTIONS):
        rows, cols = (
            (INTERMEDIATE, HIDDEN) if leaf == "down_proj" else (HIDDEN, INTERMEDIATE)
        )
        raw_bytes, raw_grid = _checkpoint_tensors(rows, cols, seed + index)
        stored = _stored_bytes_through_the_real_loader(raw_bytes)
        weight_name = f"{leaf}_weight"
        grid_name = _impl().Glm5NextForConditionalGeneration._sibling_scale_grid_name(
            weight_name
        )
        setattr(module, weight_name, torch.nn.Parameter(stored, requires_grad=False))
        setattr(module, grid_name, raw_grid.clone())
        truth[leaf] = (raw_bytes, raw_grid, weight_name, grid_name)
    return truth


def _effective_matrix(module, weight_name: str, grid_name: str) -> torch.Tensor:
    """Rebuild what the kernel will multiply, undoing only the compute-frame transpose.
    """
    weight = getattr(module, weight_name)
    grid = getattr(module, grid_name)
    return dequantise_blockwise(
        weight.data.t().contiguous(), grid.t().contiguous(), (TILE, TILE)
    ).to(torch.float32)


def _ratio(got: torch.Tensor, want: torch.Tensor) -> float:
    """One number for how far the load path is from the checkpoint, by total magnitude.
    """
    return float(got.abs().sum() / want.abs().sum())


def _prepped(kind: str, seed: int = 11):
    """Drive the real seam: attach checkpoint format, then call the module's own prep."""
    module = _loaded_module(kind)
    truth = _attach_checkpoint_format(module, seed)
    retiled = module.retile_checkpoint_scale_grids()
    return module, truth, retiled


# --------------------------------------------------------------------------- #
# 1. The effective matrix matches the checkpoint, through the real seam.       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["shared", "dense"])
def test_the_effective_matrix_matches_the_checkpoint_not_the_squeezed_copy(kind) -> None:
    module, truth, retiled = _prepped(kind)
    assert retiled == len(PROJECTIONS), (
        f"the prep retiled {retiled} projections, not {len(PROJECTIONS)}; a skipped "
        f"retile would make the comparison below read a granularity nobody serves"
    )
    for leaf, (raw_bytes, raw_grid, weight_name, grid_name) in truth.items():
        want = _reference_matrix(raw_bytes, raw_grid)
        got = _effective_matrix(module, weight_name, grid_name)
        ratio = _ratio(got, want)
        assert ratio == pytest.approx(1.0, rel=0.02), (
            f"{kind} {leaf}: the effective matrix is {ratio:.7f} of the checkpoint's. "
            f"A missing compensation reads {SQUEEZE:.7f}"
        )


# --------------------------------------------------------------------------- #
# 2. The compensation is applied exactly once per grid -- neither 0 nor 2.     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["shared", "dense"])
def test_the_compensation_is_counted_once_per_grid(kind) -> None:
    module, truth, _ = _prepped(kind)
    # The health attribute is read off the module's own class constant, never guessed by
    # name shape: each class declares where its record lands and a sniffing test would
    # start passing on the wrong dictionary the moment either constant changed.
    health_attr = (
        module.SHARED_RETILE_HEALTH_ATTR
        if kind == "shared"
        else module.DENSE_RETILE_HEALTH_ATTR
    )
    health = getattr(module, health_attr)
    for leaf in truth:
        record = health[f"{leaf}_weight"]
        applied = record["scale_compensations_applied"]
        assert applied == 1, f"{kind} {leaf} compensated {applied} times, not once"
        assert record["scale_compensated"] is True, (
            f"{kind} {leaf}: the platform gate reports the multiply did not happen, so "
            f"the count of 1 is a call that did nothing"
        )


# --------------------------------------------------------------------------- #
# 4. One MLP output, so the reading is not only about a stored tensor.         #
# --------------------------------------------------------------------------- #
def test_one_mlp_output_carries_no_leftover_squeeze() -> None:
    """y = down @ ((gate @ x) * (up @ x)), the load path against the checkpoint. """
    module, truth, _ = _prepped("shared")
    torch.manual_seed(7)
    x = torch.randn(HIDDEN, 8, dtype=torch.float32)

    def mlp(matrices: dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = (matrices["gate_proj"] @ x) * (matrices["up_proj"] @ x)
        return matrices["down_proj"] @ hidden

    want = mlp(
        {leaf: _reference_matrix(t[0], t[1]).t() for leaf, t in truth.items()}
    )
    got = mlp(
        {
            leaf: _effective_matrix(module, t[2], t[3]).t()
            for leaf, t in truth.items()
        }
    )
    ratio = _ratio(got, want)
    assert ratio == pytest.approx(1.0, rel=0.05), (
        f"the MLP output is {ratio:.7f} of the checkpoint's; an uncompensated load "
        f"reads {MLP_SQUEEZE:.7f}"
    )
