# SPDX-License-Identifier: Apache-2.0
"""Tests for the GLM-5.3-Flash vision patch-merge adapter (``inc-glm53f-104``).

WHAT IS MEASURED. Three things, and the third is the reason the first two can be trusted.

  C01  The adapter's output against the transformers 5.16.1 reference chain -- ``post_layernorm`` then the
       strided ``downsample`` then the ``merger`` -- run on the same input with the reference's own weights
       COPIED into the fork module, over 3/3 merge grids. transformers is the executable specification for
       this checkpoint, so the reference is EXECUTED here, never restated as a formula.
  C02  The same comparison, with the block regroup deliberately packed in the wrong index order. It must
       FAIL. A wrong regroup produces a tensor of the right shape, right dtype and plausible magnitude, so
       without this control C01 would also pass for an adapter that pairs every weight with the wrong
       feature.
  C03  The fork's regroup-plus-one-linear against ``torch.nn.functional.conv2d`` on the same kernel. This is
       the equivalence the whole file rests on: the reference projects with
       ``nn.Conv2d(in_channels=hidden_size, out_channels=out_hidden_size, kernel_size=spatial_merge_size,
       stride=spatial_merge_size)`` (``modeling_glm5_next.py:1757-1762``) and declares NO padding, NO
       dilation and NO grouping. Kernel equal to stride with no padding means each input element is visited
       exactly once, so the convolution is a single matrix multiply and the composed form is not an
       approximation of it -- it is the same arithmetic.

ONE TOLERANCE PAIR. Every comparison above uses ``rtol=1e-2, atol=1e-5``, the pair registered for this
block. No comparison in this file uses any other pair.

WHY THE COMPARISON RUNS IN FLOAT32. The tower runs bf16 in service, but bf16 carries about three decimal
digits, so at ``atol=1e-5`` a bf16 comparison would measure rounding noise instead of the thing under test.
What is under test here is the COMPOSITION -- the index order of the regroup, the repack of the conv kernel,
the stage order of the merger and its asymmetric clamp -- and that is dtype-independent. The adapter's own
default dtype is unchanged; these tests instantiate it at float32 deliberately. Dtype fidelity of the
assembled tower belongs to the block that assembles it.

WHY THE RANDOM INIT IS RANDOMISED FURTHER. ``Glm5NextRMSNorm`` starts its weight at ones and ``LayerNorm``
starts at weight one, bias zero. Left at those values a mis-wired weight multiply is invisible, so every
learned tensor on the reference side is filled with random values before it is copied across.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.testing import assert_close

# The oracle is imported plainly and on purpose. transformers is a hard floor for this fork --
# ``requirements/core.txt:18`` declares ``transformers>=5.16.1,<6.0.0`` precisely because the glm5_next
# classes arrived in 5.16.1 -- so an absent reference is a broken environment, not a reason to pass. A
# conditional import would let this whole file go green without ever executing the specification it claims
# to be measured against, which is the failure mode worth being loud about.
from transformers.models.glm5_next import configuration_glm5_next, modeling_glm5_next

from vllm_neuron.model.glm5_next.config import Glm5NextVisionConfig
from vllm_neuron.model.glm5_next.vision_adapter import Glm5NextVisionAdapter

# The pair registered for this block. Used by C01, C02 and C03 alike.
RTOL = 1e-2
ATOL = 1e-5

# A tiny config. hidden_size and spatial_merge_size are both > 1 on purpose: if either were 1 the correct
# and the permuted regroup would coincide and C02 could not fail. C04 asserts that precondition rather than
# leaving it to a reader.
TINY = dict(
    hidden_size=8,
    out_hidden_size=16,
    spatial_merge_size=2,
    projection_intermediate_size=24,
    hidden_act="silu",
    swiglu_limit=10.0,
    rms_norm_eps=1e-05,
)

# (name, patch_rows, patch_cols) -- both multiples of spatial_merge_size, so the merge never crosses a
# token-group bound. Token counts 4, 8 and 24 give 1, 2 and 6 merged tokens.
MERGE_GRIDS = [
    ("g1_single_block_2x2", 2, 2),
    ("g2_two_blocks_4x2", 4, 2),
    ("g3_six_blocks_4x6", 4, 6),
]

DTYPE = torch.float32


def _tiny_config() -> Glm5NextVisionConfig:
    return Glm5NextVisionConfig(**TINY)


def _reference_modules(cfg: Glm5NextVisionConfig, generator: torch.Generator):
    """Build the reference chain exactly as ``Glm5NextVisionModel.__init__`` builds it.

    The three constructions are the reference's own, argument for argument:
    ``Glm5NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)`` (``modeling_glm5_next.py:1763``),
    ``nn.Conv2d(...)`` (``:1757-1762``) and ``Glm5NextVisionPatchMerger(dim=config.out_hidden_size,
    context_dim=config.projection_intermediate_size, hidden_act=config.hidden_act,
    swiglu_limit=config.swiglu_limit)`` (``:1751-1756``).
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

    # Randomise every learned tensor, including the ones whose init would hide a mis-wiring.
    for module in (ref_norm, ref_conv, ref_merger):
        for param in module.parameters():
            param.data.copy_(
                torch.randn(param.shape, generator=generator, dtype=DTYPE) * 0.25
            )
    return ref_norm, ref_conv, ref_merger


def _reference_chain(cfg, ref_norm, ref_conv, ref_merger, hidden_states):
    """Run the reference's own five steps (``modeling_glm5_next.py:1812-1820``)."""
    h = ref_norm(hidden_states)
    h = h.view(-1, cfg.spatial_merge_size, cfg.spatial_merge_size, h.shape[-1])
    h = h.permute(0, 3, 1, 2)
    h = ref_conv(h).view(-1, cfg.out_hidden_size)
    return ref_merger(h)


def _fork_adapter_with_copied_weights(cfg, ref_norm, ref_conv, ref_merger, *, permuted_regroup=False):
    """Build the fork adapter and copy the reference's weights into it.

    Args:
        permuted_regroup: when True the conv kernel is flattened WITHOUT moving the channel axis last,
            i.e. in ``(c, i, j)`` order instead of ``(i, j, c)``. Everything else is copied correctly, so
            the only difference from the passing configuration is the index pairing. This is C02's control.
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


def _emit(tag: str, **values) -> None:
    """Print one reading as a bracketed line so the acceptance transcript carries the number itself.

    A test that only says "passed" leaves the reader to trust it. These lines put the measured error, the
    tolerance pair it was judged against and the control's gap into the transcript, where the acceptance
    driver reads them back.
    """
    body = " ".join(f"{k}={v}" for k, v in values.items())
    print(f"[{tag}] {body}", flush=True)


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    # Detached before the scalar read: some of these tensors are module parameters, and torch warns when a
    # tensor that still carries requires_grad is converted to a python float.
    return float((a.detach() - b.detach()).abs().max())


def _case(name, rows, cols, seed):
    cfg = _tiny_config()
    generator = torch.Generator().manual_seed(seed)
    ref_norm, ref_conv, ref_merger = _reference_modules(cfg, generator)
    hidden_states = torch.randn(rows * cols, cfg.hidden_size, generator=generator, dtype=DTYPE)
    return cfg, ref_norm, ref_conv, ref_merger, hidden_states


# ---------------------------------------------------------------------------
# C00 -- the module under test is the tree this run means to measure.
# ---------------------------------------------------------------------------
def _repo_root_from_this_file() -> Path:
    """Name the checkout this test file itself ships in, without asking the environment.

    The test file and the module it tests ship in the same checkout, so the file's own path already
    names the tree. The walk stops at the first parent holding both halves of that checkout.

    The plan declares this derivation for the sibling block as the fixed index
    ``Path(__file__).resolve().parents[4]`` (increment-plan.md L1123). A walk and that index name the
    same directory at this file's depth, and the assertion below says so as a reading rather than
    leaving me to claim it. If the file is ever moved to another depth the assertion fires, which is
    the right outcome: it means the plan's declared form no longer describes this file.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "vllm_neuron" / "__init__.py").is_file() and (parent / "test").is_dir():
            assert parent == here.parents[4], (
                f"the derived root {parent} is not the plan's declared parents[4] {here.parents[4]}; "
                "this file has moved depth and the declared derivation needs re-reading"
            )
            return parent
    raise AssertionError(f"no checkout root above {here} holds both vllm_neuron/ and test/")


def test_c00_the_module_under_test_is_the_scratch_candidate():
    """The import must resolve under the tree this run is meant to measure.

    The venv holds an editable install pointing at a different checkout, and importing vllm already imports
    this plugin through its platform entry point, so PYTHONPATH has to be set before the first import rather
    than checked afterwards. This asserts the outcome of that.

    THERE ARE TWO HONEST WAYS TO NAME THE TREE AND THIS TEST TAKES BOTH. When the runner declares
    ``GLM53F_CANDIDATE_ROOT``, the declaration is what gets enforced, so a run that says which tree it
    measured is held to its own word. When nothing is declared, the tree is derived from this file's own
    path instead. The comparison-suite run unsets the variable on purpose, so that a test coupled to it
    reads red rather than being hidden by the runner's environment; deriving the root keeps the assertion
    real in that run instead of turning it into a skip, and a skip in added lines is itself a finding.
    """
    import vllm_neuron

    from vllm_neuron.model.glm5_next import vision_adapter as module_under_test

    declared_root = os.environ.get("GLM53F_CANDIDATE_ROOT")
    root = Path(declared_root).resolve() if declared_root else _repo_root_from_this_file()
    _emit("c00 import-origin", root=root, root_source="declared" if declared_root else "derived")
    for name, module in (("vllm_neuron", vllm_neuron), ("vision_adapter", module_under_test)):
        resolved = Path(module.__file__).resolve()
        _emit(
            "c00 import-origin",
            module=name,
            IS_UNDER_THE_SCRATCH_CANDIDATE=int(root in resolved.parents),
            resolved=resolved,
        )
        assert root in resolved.parents, f"{name} resolved to {resolved}, which is not under {root}"


# ---------------------------------------------------------------------------
# C01 -- 3/3 merge grids: the adapter equals the reference chain.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,rows,cols", MERGE_GRIDS, ids=[g[0] for g in MERGE_GRIDS])
def test_c01_adapter_matches_reference_chain(name, rows, cols):
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
    _emit(
        f"c01 {name}",
        merged_tokens=merged_tokens,
        elements=expected.numel(),
        rtol=RTOL,
        atol=ATOL,
        max_abs_error=f"{_max_abs(actual, expected):.6e}",
        reference_max_abs=f"{float(expected.abs().max()):.6e}",
    )
    assert_close(actual, expected, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# C02 -- the control: a permuted regroup must FAIL the same comparison.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,rows,cols", MERGE_GRIDS, ids=[g[0] for g in MERGE_GRIDS])
def test_c02_permuted_regroup_fails_the_same_comparison(name, rows, cols):
    """Flattening the block in ``(c, i, j)`` order instead of ``(i, j, c)`` must be caught.

    Only the index pairing differs from the configuration C01 passes on: the same kernel numbers, the same
    bias, the same merger weights, the same input. If this comparison were to pass, C01 would not be
    measuring the pairing at all.
    """
    cfg, ref_norm, ref_conv, ref_merger, hidden_states = _case(name, rows, cols, seed=1104)
    wrong = _fork_adapter_with_copied_weights(
        cfg, ref_norm, ref_conv, ref_merger, permuted_regroup=True
    )

    with torch.no_grad():
        expected = _reference_chain(cfg, ref_norm, ref_conv, ref_merger, hidden_states)
        actual = wrong(hidden_states)

    assert actual.shape == expected.shape, (
        "the permuted regroup changed the OUTPUT SHAPE, so this control would be caught by a shape check "
        "and does not exercise the index pairing"
    )
    gap = _max_abs(actual, expected)
    raised = False
    try:
        assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    except AssertionError:
        raised = True
    _emit(
        f"c02 {name}",
        rtol=RTOL,
        atol=ATOL,
        control_max_abs_error=f"{gap:.6e}",
        assert_close_raised=raised,
    )
    assert raised, (
        f"the permuted regroup was accepted at rtol={RTOL} atol={ATOL} with a max abs error of {gap:.6e}; "
        f"C01 therefore does not measure the index pairing"
    )


# ---------------------------------------------------------------------------
# C03 -- regroup plus one linear is the strided convolution.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,rows,cols", MERGE_GRIDS, ids=[g[0] for g in MERGE_GRIDS])
def test_c03_regroup_and_one_linear_equals_conv2d(name, rows, cols):
    """The composed form equals ``F.conv2d`` on the copied kernel, at the same pair."""
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
    _emit(
        f"c03 {name}",
        elements=expected.numel(),
        rtol=RTOL,
        atol=ATOL,
        conv_vs_composed_max_abs=f"{_max_abs(actual, expected):.6e}",
        conv_output_max_abs=f"{float(expected.abs().max()):.6e}",
    )
    assert_close(actual, expected, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# C04 -- the control in C02 is not vacuous, and the reference fields are the ones read.
# ---------------------------------------------------------------------------
def test_c04_the_permuted_control_can_actually_differ():
    """``hidden_size`` and ``spatial_merge_size`` must both exceed 1, or C02 is empty.

    With either equal to 1 the ``(i, j, c)`` and ``(c, i, j)`` flattenings are the same permutation and C02
    would pass for the wrong reason.
    """
    cfg = _tiny_config()
    assert cfg.hidden_size > 1
    assert cfg.spatial_merge_size > 1
    generator = torch.Generator().manual_seed(1104)
    _, ref_conv, _ = _reference_modules(cfg, generator)
    adapter = Glm5NextVisionAdapter(cfg, dtype=DTYPE)
    right = adapter.downsample_weight_from_conv(ref_conv.weight)
    wrong = ref_conv.weight.reshape(ref_conv.weight.shape[0], -1).t().contiguous()
    assert right.shape == wrong.shape
    _emit(
        "c04 packing-precondition",
        hidden_size=cfg.hidden_size,
        spatial_merge_size=cfg.spatial_merge_size,
        packings_differ=not torch.equal(right, wrong),
        packing_max_abs_difference=f"{_max_abs(right, wrong):.6e}",
    )
    assert not torch.equal(right, wrong), (
        "the two packings are identical for this config, so C02 cannot fail for the reason it claims"
    )


def test_c05_the_reference_declares_the_fields_the_adapter_reads():
    """Guards the field names, including the one this increment added to the fork config.

    ``rms_norm_eps`` is declared TWICE in the reference's config file -- once for the vision tower and once
    for the text decoder -- and the tower's is the vision one. If a later transformers release moves or
    renames any of these, this test says so instead of the adapter silently reading a default.
    """
    configuration = configuration_glm5_next
    vision_cls = configuration.Glm5NextVisionConfig
    declared = set(getattr(vision_cls, "__annotations__", {}))
    _emit(
        "c05 reference-fields",
        declared_count=len(declared),
        has_rms_norm_eps=int("rms_norm_eps" in declared),
        has_projection_intermediate_size=int("projection_intermediate_size" in declared),
    )
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
            f"the reference's vision config no longer declares {field!r}; the adapter reads it from "
            f"Glm5NextVisionConfig and would fall back to a default"
        )
    assert len(declared) > 1
