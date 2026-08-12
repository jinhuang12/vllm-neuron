# SPDX-License-Identifier: Apache-2.0
"""Stage 4 dense MLP and MoE component gates for GLM-5.2."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig

from vllm_neuron.model.glm_moe_dsa.config import GlmMoeDsaConfig
from vllm_neuron.model.glm_moe_dsa.mlp import GlmMoeDsaSwiGLUMLP
from vllm_neuron.model.glm_moe_dsa.moe import (
    GlmMoeDsaMoE,
    GlmMoeDsaNoAuxRouter,
    GlmMoeDsaRoutedExperts,
)

MODEL_PATH_VALUE = os.environ.get("GLM52_MODEL_PATH")
MODEL_PATH = Path(MODEL_PATH_VALUE or ".")


@pytest.fixture(scope="module")
def config() -> GlmMoeDsaConfig:
    if MODEL_PATH_VALUE is None:
        pytest.skip("GLM52_MODEL_PATH is required for pinned-model tests")
    hf_config = AutoConfig.from_pretrained(
        MODEL_PATH, local_files_only=True, trust_remote_code=False
    )
    return GlmMoeDsaConfig.from_configs(hf_config)


def test_config_freezes_glm52_moe_contract(config: GlmMoeDsaConfig) -> None:
    assert config.hidden_act == "silu"
    assert config.n_routed_experts == 256
    assert config.n_shared_experts == 1
    assert config.num_experts_per_tok == 8
    assert config.norm_topk_prob is True
    assert config.routed_scaling_factor == 2.5
    assert config.n_group == 1
    assert config.topk_group == 1
    assert config.topk_method == "noaux_tc"
    assert config.scoring_func == "sigmoid"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_shared_experts", 2),
        ("norm_topk_prob", False),
        ("routed_scaling_factor", 1.0),
        ("topk_method", "greedy"),
        ("scoring_func", "softmax"),
        ("hidden_act", "gelu"),
    ],
)
def test_config_rejects_wrong_moe_contract(
    config: GlmMoeDsaConfig, field: str, value: object
) -> None:
    raw = AutoConfig.from_pretrained(
        MODEL_PATH, local_files_only=True, trust_remote_code=False
    ).to_dict()
    raw[field] = value
    with pytest.raises(ValueError, match=field):
        GlmMoeDsaConfig.from_configs(raw)


def test_dense_swiglu_matches_reference() -> None:
    torch.manual_seed(11)
    module = GlmMoeDsaSwiGLUMLP(
        hidden_size=16,
        intermediate_size=24,
        tensor_parallel_size=2,
        dtype=torch.float32,
    )
    hidden = torch.randn(7, 16)
    expected = torch.nn.functional.linear(
        torch.nn.functional.silu(
            torch.nn.functional.linear(hidden, module.gate_proj.weight)
        )
        * torch.nn.functional.linear(hidden, module.up_proj.weight),
        module.down_proj.weight,
    )
    torch.testing.assert_close(module(hidden), expected, rtol=0, atol=0)


def test_production_tp64_and_ep64_parameter_shapes(
    config: GlmMoeDsaConfig,
) -> None:
    dense = GlmMoeDsaSwiGLUMLP.dense_from_config(
        config, tensor_parallel_size=64, device="meta"
    )
    assert dense.gate_proj.weight.shape == (192, 6144)
    assert dense.up_proj.weight.shape == (192, 6144)
    assert dense.down_proj.weight.shape == (6144, 192)

    moe = GlmMoeDsaMoE(
        config,
        tensor_parallel_size=64,
        expert_parallel_size=64,
        expert_parallel_rank=63,
        device="meta",
    )
    assert moe.router.gate.weight.shape == (256, 6144)
    assert moe.router.correction_bias.shape == (256,)
    assert moe.router.selection_identity.dtype is torch.float32
    assert moe.experts.num_local_experts == 4
    assert moe.experts.global_expert_ids == (252, 253, 254, 255)
    assert moe.experts.experts[0].gate_proj.weight.shape == (2048, 6144)
    assert moe.experts.experts[0].down_proj.weight.shape == (6144, 2048)
    assert moe.shared_experts.gate_proj.weight.shape == (32, 6144)
    assert moe.shared_experts.down_proj.weight.shape == (6144, 32)


def test_noaux_router_bias_changes_selection_not_weight() -> None:
    router = GlmMoeDsaNoAuxRouter(
        hidden_size=4,
        num_experts=4,
        top_k=2,
        routed_scaling_factor=2.5,
    )
    with torch.no_grad():
        router.gate.weight.zero_()
        router.correction_bias.copy_(torch.tensor([-0.2, 0.1, 0.4, 0.8]))
    affinities, selected = router(torch.ones(3, 4))

    assert {2, 3} == set(selected[0].tolist())
    torch.testing.assert_close(
        affinities[:, 2:], torch.full((3, 2), 1.25), rtol=0, atol=0
    )
    torch.testing.assert_close(affinities[:, :2], torch.zeros(3, 2), rtol=0, atol=0)
    torch.testing.assert_close(
        affinities.sum(dim=-1), torch.full((3,), 2.5), rtol=0, atol=0
    )


def test_local_routed_experts_match_explicit_reference() -> None:
    torch.manual_seed(23)
    experts = GlmMoeDsaRoutedExperts(
        hidden_size=8,
        intermediate_size=12,
        num_experts=8,
        top_k=3,
        expert_parallel_size=2,
        expert_parallel_rank=1,
        dtype=torch.float32,
    )
    hidden = torch.randn(5, 8)
    affinities = torch.zeros(5, 8)
    affinities[:, 4] = 0.3
    affinities[:, 6] = 0.7
    selected = torch.tensor([[4, 6, 0]] * 5, dtype=torch.int32)

    expected = (
        experts.experts[0](hidden) * affinities[:, 4:5]
        + experts.experts[2](hidden) * affinities[:, 6:7]
    )
    actual = experts(hidden, affinities, selected, is_decode=True)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_shared_expert_is_separate_tp_contribution(config: GlmMoeDsaConfig) -> None:
    shared = GlmMoeDsaSwiGLUMLP.shared_from_config(
        config, tensor_parallel_size=64, device="meta"
    )
    assert shared.intermediate_size == 2048
    assert shared.local_intermediate_size == 32


class _HardwareMoE(nn.Module):
    def __init__(self, config: GlmMoeDsaConfig, *, is_decode: bool) -> None:
        super().__init__()
        self.is_decode = is_decode
        self.moe = GlmMoeDsaMoE(
            config,
            tensor_parallel_size=64,
            expert_parallel_size=64,
            expert_parallel_rank=0,
        )
        with torch.no_grad():
            self.moe.router.gate.weight.zero_()
            correction = torch.linspace(
                1.0, 0.0, config.n_routed_experts, dtype=torch.float32
            )
            self.moe.router.correction_bias.copy_(correction)
            for parameter in self.moe.experts.parameters():
                parameter.normal_(mean=0.0, std=0.002)
            for parameter in self.moe.shared_experts.parameters():
                parameter.normal_(mean=0.0, std=0.002)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.moe(hidden_states, is_decode=self.is_decode)


class _HardwareMoEComponents(nn.Module):
    """Expose the production NKI component boundaries for numeric comparison."""

    def __init__(self, module: _HardwareMoE) -> None:
        super().__init__()
        self.module = module

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        moe = self.module.moe
        affinities, selected_experts = moe.router(hidden_states)
        routed = moe.experts(
            hidden_states,
            affinities,
            selected_experts,
            is_decode=self.module.is_decode,
        )
        shared = moe.shared_experts(hidden_states)
        return selected_experts, routed, shared, routed + shared


@torch.no_grad()
def _production_shape_reference(
    module: _HardwareMoE, hidden_states: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Independent torch reference using the exact initialized module tensors."""
    moe = module.moe
    scores = torch.sigmoid(
        F.linear(hidden_states.to(torch.float32), moe.router.gate.weight)
    )
    selection_scores = scores + moe.router.correction_bias
    selected_experts = torch.topk(
        selection_scores,
        moe.router.top_k,
        dim=-1,
        sorted=False,
    ).indices.to(torch.int32)
    selected_weights = torch.gather(scores, -1, selected_experts.to(torch.long))
    selected_weights = selected_weights / selected_weights.sum(dim=-1, keepdim=True)
    selected_weights = selected_weights * moe.router.routed_scaling_factor
    affinities = torch.zeros_like(scores).scatter(
        -1, selected_experts, selected_weights
    )

    routed = torch.zeros_like(hidden_states)
    for local_id, global_id in enumerate(moe.experts.global_expert_ids):
        expert = moe.experts.experts[local_id]
        expert_output = F.linear(
            F.silu(F.linear(hidden_states, expert.gate_proj.weight))
            * F.linear(hidden_states, expert.up_proj.weight),
            expert.down_proj.weight,
        )
        routed = routed + expert_output * affinities[:, global_id : global_id + 1].to(
            hidden_states.dtype
        )

    shared_mlp = moe.shared_experts
    shared = F.linear(
        F.silu(F.linear(hidden_states, shared_mlp.gate_proj.weight))
        * F.linear(hidden_states, shared_mlp.up_proj.weight),
        shared_mlp.down_proj.weight,
    )
    return selected_experts, routed, shared, routed + shared


def _assert_bf16_component_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Check BF16 kernel output against the independent reference."""
    actual_fp32 = actual.to(torch.float32)
    expected_fp32 = expected.to(torch.float32)
    # Eight percent allows BF16 rounding and different accelerator accumulation
    # order. The 3e-5 floor covers values near zero while remaining over 10x
    # smaller than the smallest component maximum in this deterministic gate.
    torch.testing.assert_close(
        actual_fp32,
        expected_fp32,
        rtol=0.08,
        atol=3e-5,
    )


@pytest.mark.skipif(
    os.getenv("GLM_STAGE4_HARDWARE") != "1",
    reason="explicit scoped Neuron Stage 4 compile smoke",
)
@pytest.mark.parametrize(
    ("token_count", "is_decode"),
    [(16, False), (2048, False), (1, True), (32, True)],
)
def test_neuron_compile_and_activate_stage4_moe(
    config: GlmMoeDsaConfig, token_count: int, is_decode: bool
) -> None:
    torch.manual_seed(29 + token_count)
    hidden = torch.randn(token_count, config.hidden_size, dtype=torch.bfloat16)
    device = torch.device("neuron:0")
    module = _HardwareMoE(config, is_decode=is_decode).eval().to(device)
    compile_root = Path(os.environ["GLM_STAGE4_COMPILE_DIR"])
    variant = f"{'decode' if is_decode else 'prefill'}-{token_count}"
    compiled = torch.compile(
        module,
        backend="vllm_neuron",
        fullgraph=True,
        dynamic=False,
        options={"compiler_workdir": str(compile_root / variant)},
    )
    actual = compiled(hidden.to(device)).cpu()
    assert actual.shape == hidden.shape
    assert torch.isfinite(actual).all()
    assert actual.abs().max().item() > 0


@pytest.mark.skipif(
    os.getenv("GLM_STAGE4_HARDWARE") != "1",
    reason="explicit scoped Neuron Stage 4 numeric comparison",
)
@pytest.mark.parametrize(
    ("token_count", "is_decode"),
    [(1, True), (16, False)],
    ids=["decode1", "prefill16"],
)
def test_neuron_stage4_moe_matches_production_shape_reference(
    config: GlmMoeDsaConfig, token_count: int, is_decode: bool
) -> None:
    torch.manual_seed(29 + token_count)
    hidden = torch.randn(token_count, config.hidden_size, dtype=torch.bfloat16)
    module = _HardwareMoE(config, is_decode=is_decode).eval()
    with torch.no_grad():
        # Exercise the production router matmul and sigmoid. The correction
        # remains large enough to include all four experts local to EP rank zero
        # while the nonzero logits vary the remaining selected global experts.
        torch.manual_seed(1031 + token_count)
        module.moe.router.gate.weight.normal_(mean=0.0, std=0.0005)

    expected = _production_shape_reference(module, hidden)
    expected_selected = torch.sort(expected[0].to(torch.long), dim=-1).values
    for local_expert in module.moe.experts.global_expert_ids:
        assert torch.any(expected_selected == local_expert)
    assert torch.any(expected_selected >= module.moe.experts.num_local_experts)

    device = torch.device("neuron:0")
    compiled_module = _HardwareMoEComponents(module).eval().to(device)
    compile_root = Path(os.environ["GLM_STAGE4_COMPILE_DIR"])
    variant = f"numeric-{'decode' if is_decode else 'prefill'}-{token_count}"
    compiled = torch.compile(
        compiled_module,
        backend="vllm_neuron",
        fullgraph=True,
        dynamic=False,
        options={"compiler_workdir": str(compile_root / variant)},
    )
    actual = tuple(tensor.cpu() for tensor in compiled(hidden.to(device)))

    actual_selected = torch.sort(actual[0].to(torch.long), dim=-1).values
    assert torch.equal(actual_selected, expected_selected)
    for actual_component, expected_component in zip(
        actual[1:], expected[1:], strict=True
    ):
        assert actual_component.shape == expected_component.shape
        assert torch.isfinite(actual_component).all()
        _assert_bf16_component_close(actual_component, expected_component)
