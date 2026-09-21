# SPDX-License-Identifier: Apache-2.0
"""Tests for the GLM-5.3-Flash vision patch-merge adapter.

The adapter is compared with transformers' own chain -- ``post_layernorm``, the strided
``downsample``, then the ``merger`` -- run on the same input with the reference's weights
copied into the fork module. A second arm packs the merge block in the wrong index order and
must disagree, and a third shows that the regroup plus one linear is the strided convolution
itself: kernel equal to stride with no padding visits each input element once, so the
convolution is a single matrix multiply.

The comparisons run in float32. The tower serves bf16, which carries about three decimal
digits, so at ``atol=1e-5`` a bf16 comparison would measure rounding rather than the
composition under test -- and the composition is dtype-independent.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.testing import assert_close

# transformers is a hard floor for this fork: requirements/core.txt declares
# transformers>=5.16.1,<6.0.0 because the glm5_next classes arrived in 5.16.1. The skip is
# taken here rather than in a fixture, so an older transformers skips this file by name
# instead of failing every other file's collection with an import error. Skipping is the
# only alternative to importing: nothing below means anything without the reference, so
# this file must never report a pass when it could not measure against it.
pytest.importorskip(
    "transformers.models.glm5_next",
    reason="transformers>=5.16.1 carries the glm5_next reference this file measures against",
)

from transformers.models.glm5_next import (  # noqa: E402
    configuration_glm5_next,
    modeling_glm5_next,
)

from vllm_neuron.model.glm5_next.config import Glm5NextVisionConfig
from vllm_neuron.model.glm5_next.vision_adapter import Glm5NextVisionAdapter

RTOL = 1e-2
ATOL = 1e-5

# A tiny config. hidden_size and spatial_merge_size are both greater than 1 on purpose: at 1
# the correct and the permuted regroup coincide and the permuted arm could not disagree.
TINY = dict(
    hidden_size=8,
    out_hidden_size=16,
    spatial_merge_size=2,
    projection_intermediate_size=24,
    hidden_act="silu",
    swiglu_limit=10.0,
    rms_norm_eps=1e-05,
)

# (name, patch_rows, patch_cols) -- both multiples of spatial_merge_size, so the merge never
# crosses a token-group bound. Token counts 4, 8 and 24 give 1, 2 and 6 merged tokens.
MERGE_GRIDS = [
    ("g1_single_block_2x2", 2, 2),
    ("g2_two_blocks_4x2", 4, 2),
    ("g3_six_blocks_4x6", 4, 6),
]

DTYPE = torch.float32


def _tiny_config() -> Glm5NextVisionConfig:
    return Glm5NextVisionConfig(**TINY)


def _reference_modules(cfg: Glm5NextVisionConfig, generator: torch.Generator):
    """Build the reference chain as ``Glm5NextVisionModel.__init__`` builds it.

    Every learned tensor is then filled with random values: ``Glm5NextRMSNorm`` starts its
    weight at ones and ``LayerNorm`` at weight one, bias zero, and at those values a mis-wired
    weight multiply is invisible.
    """
    modeling = modeling_glm5_next
    ref_norm = modeling.Glm5NextRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(DTYPE)
    ref_conv = nn.Conv2d(
        in_channels=cfg.hidden_size,
        out_channels=cfg.out_hidden_size,
        kernel_size=cfg.spatial_merge_size,
        stride=cfg.spatial_merge_size,
    ).to(DTYPE)
    ref_merger = modeling.Glm5NextVisionPatchMerger(
        dim=cfg.out_hidden_size,
        context_dim=cfg.projection_intermediate_size,
        hidden_act=cfg.hidden_act,
        swiglu_limit=cfg.swiglu_limit,
    ).to(DTYPE)

    for module in (ref_norm, ref_conv, ref_merger):
        for param in module.parameters():
            param.data.copy_(
                torch.randn(param.shape, generator=generator, dtype=DTYPE) * 0.25
            )
    return ref_norm, ref_conv, ref_merger


def _reference_chain(cfg, ref_norm, ref_conv, ref_merger, hidden_states):
    """Run the reference's own five steps."""
    h = ref_norm(hidden_states)
    h = h.view(-1, cfg.spatial_merge_size, cfg.spatial_merge_size, h.shape[-1])
    h = h.permute(0, 3, 1, 2)
    h = ref_conv(h).view(-1, cfg.out_hidden_size)
    return ref_merger(h)


def _fork_adapter_with_copied_weights(cfg, ref_norm, ref_conv, ref_merger, *, permuted_regroup=False):
    """Build the fork adapter and copy the reference's weights into it.

    Args:
        permuted_regroup: when True the conv kernel is flattened without moving the channel
            axis last, i.e. in ``(c, i, j)`` order instead of ``(i, j, c)``. Everything else
            is copied correctly, so the only difference from the passing configuration is the
            index pairing.
    """
    adapter = Glm5NextVisionAdapter(cfg, dtype=DTYPE)
    with torch.no_grad():
        adapter.post_layernorm.weight.copy_(ref_norm.weight)
        if permuted_regroup:
            wrong = ref_conv.weight.reshape(ref_conv.weight.shape[0], -1).t().contiguous()
            adapter.downsample_weight.copy_(wrong)
        else:
            adapter.downsample_weight.copy_(
                adapter.downsample_weight_from_conv(ref_conv.weight)
            )
        adapter.downsample_bias.copy_(ref_conv.bias)
        adapter.merger.proj.weight.copy_(ref_merger.proj.weight)
        adapter.merger.post_projection_norm.weight.copy_(ref_merger.post_projection_norm.weight)
        adapter.merger.post_projection_norm.bias.copy_(ref_merger.post_projection_norm.bias)
        adapter.merger.gate_proj.weight.copy_(ref_merger.gate_proj.weight)
        adapter.merger.up_proj.weight.copy_(ref_merger.up_proj.weight)
        adapter.merger.down_proj.weight.copy_(ref_merger.down_proj.weight)
    return adapter


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    # Detached before the scalar read: some of these tensors are module parameters, and torch
    # warns when a tensor that still carries requires_grad is converted to a python float.
    return float((a.detach() - b.detach()).abs().max())


def _case(name, rows, cols, seed):
    cfg = _tiny_config()
    generator = torch.Generator().manual_seed(seed)
    ref_norm, ref_conv, ref_merger = _reference_modules(cfg, generator)
    hidden_states = torch.randn(rows * cols, cfg.hidden_size, generator=generator, dtype=DTYPE)
    return cfg, ref_norm, ref_conv, ref_merger, hidden_states


@pytest.mark.parametrize("name,rows,cols", MERGE_GRIDS, ids=[g[0] for g in MERGE_GRIDS])
def test_adapter_matches_reference_chain(name, rows, cols):
    """Adapter output equals ``post_layernorm -> downsample -> merger`` with the same weights."""
    cfg, ref_norm, ref_conv, ref_merger, hidden_states = _case(name, rows, cols, seed=1104)
    adapter = _fork_adapter_with_copied_weights(cfg, ref_norm, ref_conv, ref_merger)

    with torch.no_grad():
        expected = _reference_chain(cfg, ref_norm, ref_conv, ref_merger, hidden_states)
        actual = adapter(hidden_states)

    merged_tokens = (rows * cols) // cfg.spatial_merge_size**2
    assert expected.shape == (merged_tokens, cfg.out_hidden_size), (
        f"the reference itself produced {tuple(expected.shape)} for grid {rows}x{cols}, not the "
        f"{(merged_tokens, cfg.out_hidden_size)} this case is built on"
    )
    assert actual.shape == expected.shape
    assert expected.numel() > 0
    assert_close(actual, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("name,rows,cols", MERGE_GRIDS, ids=[g[0] for g in MERGE_GRIDS])
def test_permuted_regroup_fails_the_same_comparison(name, rows, cols):
    """Flattening the block in ``(c, i, j)`` order instead of ``(i, j, c)`` must be caught.

    Only the index pairing differs from the passing configuration: the same kernel numbers,
    the same bias, the same merger weights, the same input.
    """
    cfg, ref_norm, ref_conv, ref_merger, hidden_states = _case(name, rows, cols, seed=1104)
    wrong = _fork_adapter_with_copied_weights(
        cfg, ref_norm, ref_conv, ref_merger, permuted_regroup=True
    )

    with torch.no_grad():
        expected = _reference_chain(cfg, ref_norm, ref_conv, ref_merger, hidden_states)
        actual = wrong(hidden_states)

    assert actual.shape == expected.shape, (
        "the permuted regroup changed the output shape, so this case is caught by a shape "
        "check and does not exercise the index pairing"
    )
    gap = _max_abs(actual, expected)
    raised = False
    try:
        assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    except AssertionError:
        raised = True
    assert raised, (
        f"the permuted regroup was accepted at rtol={RTOL} atol={ATOL} with a max abs error "
        f"of {gap:.6e}, so the comparison does not measure the index pairing"
    )


@pytest.mark.parametrize("name,rows,cols", MERGE_GRIDS, ids=[g[0] for g in MERGE_GRIDS])
def test_regroup_and_one_linear_equals_conv2d(name, rows, cols):
    """The composed form equals ``F.conv2d`` on the copied kernel, at the same tolerance."""
    cfg, ref_norm, ref_conv, ref_merger, hidden_states = _case(name, rows, cols, seed=1104)
    adapter = _fork_adapter_with_copied_weights(cfg, ref_norm, ref_conv, ref_merger)

    with torch.no_grad():
        actual = adapter.regroup_and_downsample(hidden_states)
        nchw = hidden_states.view(
            -1, cfg.spatial_merge_size, cfg.spatial_merge_size, cfg.hidden_size
        ).permute(0, 3, 1, 2)
        expected = F.conv2d(
            nchw, ref_conv.weight, ref_conv.bias, stride=cfg.spatial_merge_size
        ).view(-1, cfg.out_hidden_size)

    assert expected.numel() > 0
    assert actual.shape == expected.shape
    assert_close(actual, expected, rtol=RTOL, atol=ATOL)


def test_the_two_kernel_packings_differ_for_this_config():
    """``downsample_weight_from_conv`` is not a plain transpose of the conv kernel."""
    cfg = _tiny_config()
    assert cfg.hidden_size > 1
    assert cfg.spatial_merge_size > 1
    generator = torch.Generator().manual_seed(1104)
    _, ref_conv, _ = _reference_modules(cfg, generator)
    adapter = Glm5NextVisionAdapter(cfg, dtype=DTYPE)
    right = adapter.downsample_weight_from_conv(ref_conv.weight)
    wrong = ref_conv.weight.reshape(ref_conv.weight.shape[0], -1).t().contiguous()
    assert right.shape == wrong.shape
    assert not torch.equal(right, wrong), (
        "the two packings are identical for this config, so a permuted regroup could not be "
        "told apart from the correct one"
    )


def test_the_reference_declares_the_fields_the_adapter_reads():
    """The adapter reads these off the reference's vision config, so all seven must exist.

    ``rms_norm_eps`` is declared twice in the reference's config file, once for the vision
    tower and once for the text decoder; the tower's is the vision one. If a later transformers
    release moves or renames any of these, this says so instead of the adapter silently reading
    a default.
    """
    configuration = configuration_glm5_next
    vision_cls = configuration.Glm5NextVisionConfig
    declared = set(getattr(vision_cls, "__annotations__", {}))
    for field in (
        "hidden_size",
        "out_hidden_size",
        "spatial_merge_size",
        "projection_intermediate_size",
        "hidden_act",
        "swiglu_limit",
        "rms_norm_eps",
    ):
        assert field in declared, (
            f"the reference's vision config no longer declares {field!r}; the adapter reads it "
            f"from Glm5NextVisionConfig and would fall back to a default"
        )
