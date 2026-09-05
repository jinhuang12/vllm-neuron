# SPDX-License-Identifier: Apache-2.0
"""Stage 6 FP8 storage, scale binding, execution, and HBM gates."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from torch.fx.experimental.proxy_tensor import make_fx

from vllm_neuron.model.glm_moe_dsa.block_fp8 import (
    FP8_INVERSE_SCALE_ADJUSTMENT,
    FP8_STORAGE_SCALE,
    BlockFP8Linear,
    block_fp8_linear,
    dequantize_block_fp8,
)
from vllm_neuron.model.glm_moe_dsa.config import GlmMoeDsaConfig
from vllm_neuron.model.glm_moe_dsa.model import GlmMoeDsaForCausalLM
from vllm_neuron.model.glm_moe_dsa.moe import GlmMoeDsaNoAuxRouter
from vllm_neuron.model.glm_moe_dsa.weight_loaders import (
    Disposition,
    load_checkpoint_manifest,
)

CHECKPOINT_DIR_VALUE = os.environ.get("GLM52_MODEL_PATH")
CHECKPOINT_DIR = Path(CHECKPOINT_DIR_VALUE or ".")


def _production_config() -> GlmMoeDsaConfig:
    if CHECKPOINT_DIR_VALUE is None:
        pytest.skip("GLM52_MODEL_PATH is required for pinned-checkpoint tests")
    with (CHECKPOINT_DIR / "config.json").open() as config_file:
        return GlmMoeDsaConfig.from_configs(json.load(config_file))


def _rank17_meta_model() -> GlmMoeDsaForCausalLM:
    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_world_size", return_value=64),
        patch("torch.distributed.get_rank", return_value=17),
        torch.device("meta"),
    ):
        tp_group = SimpleNamespace(device_group=object())
        return GlmMoeDsaForCausalLM(
            _production_config(),
            64,
            expert_parallel_rank=17,
            tp_group=tp_group,
        )


def test_router_routed_scale_preserves_eager_math_and_fp32_dtype() -> None:
    torch.manual_seed(67)
    router = GlmMoeDsaNoAuxRouter(16, 8, 3, routed_scaling_factor=2.5)
    hidden = torch.randn(5, 16)

    scores = torch.sigmoid(F.linear(hidden, router.gate.weight).float())
    selection_scores = scores + router.correction_bias
    _, expected_experts = torch.topk(
        selection_scores, router.top_k, dim=-1, sorted=False
    )
    expected_experts = expected_experts.to(torch.int32)
    expected_weights = torch.gather(scores, -1, expected_experts.to(torch.long))
    expected_weights = expected_weights / expected_weights.sum(dim=-1, keepdim=True)
    expected_weights = expected_weights * torch.full(
        (), router.routed_scaling_factor, dtype=scores.dtype
    )
    expected_affinities = torch.zeros_like(scores).scatter(
        -1, expected_experts, expected_weights
    )

    with patch(
        "vllm_neuron.model.glm_moe_dsa.moe.can_run_kernel",
        return_value=False,
    ):
        actual_affinities, actual_experts = router(hidden)

    assert actual_affinities.dtype is torch.float32
    assert torch.equal(actual_experts, expected_experts)
    torch.testing.assert_close(actual_affinities, expected_affinities)


def test_router_routed_scale_captures_with_meta_faketensors() -> None:
    router = GlmMoeDsaNoAuxRouter(16, 8, 3, device="meta")
    parameters = dict(router.named_parameters())
    buffers = dict(router.named_buffers())
    hidden = torch.randn(2, 16, device="meta")

    def functional_router(
        params: dict[str, torch.Tensor],
        module_buffers: dict[str, torch.Tensor],
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return cast(
            tuple[torch.Tensor, torch.Tensor],
            torch.func.functional_call(router, (params, module_buffers), (inputs,)),
        )

    with patch(
        "vllm_neuron.model.glm_moe_dsa.moe.can_run_kernel",
        return_value=False,
    ):
        graph_module = make_fx(functional_router, tracing_mode="fake")(
            parameters, buffers, hidden
        )

    new_full_nodes = [
        node
        for node in graph_module.graph.nodes
        if node.target is torch.ops.aten.new_full.default
    ]
    assert len(new_full_nodes) == 1
    assert new_full_nodes[0].meta["val"].dtype is torch.float32
    tensor_dtypes = [
        node.meta["val"].dtype
        for node in graph_module.graph.nodes
        if isinstance(node.meta.get("val"), torch.Tensor)
    ]
    assert torch.float64 not in tensor_dtypes


def test_block_dequant_reference_covers_both_tp_boundary_offsets() -> None:
    values = torch.arange(192 * 192, dtype=torch.float32).reshape(192, 192)
    weight = ((values % 31) - 15).to(torch.float8_e4m3fn)
    scales = torch.tensor([[0.5, 1.0], [2.0, 4.0]])
    actual = dequantize_block_fp8(weight, scales, row_offset=64, col_offset=64)
    rows = (torch.arange(192) + 64) // 128
    cols = (torch.arange(192) + 64) // 128
    expected = weight.float() * scales[rows[:, None], cols[None, :]]
    torch.testing.assert_close(actual, expected)


def test_cpu_block_fp8_linear_matches_explicit_reference() -> None:
    torch.manual_seed(61)
    hidden = torch.randn(7, 192, dtype=torch.bfloat16)
    weight = (torch.randn(192, 192) * 0.5).to(torch.float8_e4m3fn)
    scales = torch.rand(2, 2) + 0.25
    expected = F.linear(
        hidden,
        dequantize_block_fp8(weight, scales, row_offset=64, col_offset=64).to(
            torch.bfloat16
        ),
    )
    actual = block_fp8_linear(
        hidden,
        weight,
        scales,
        row_offset=64,
        col_offset=64,
    )
    torch.testing.assert_close(actual, expected)


def test_block_fp8_module_retains_fp8_weight_and_grid() -> None:
    module = BlockFP8Linear(192, 192, row_offset=64, col_offset=64, device="meta")
    assert module.weight.dtype is torch.float8_e4m3fn
    assert module.weight_scale_inv.dtype is torch.float32
    assert module.weight.shape == (192, 192)
    assert module.weight_scale_inv.shape == (2, 2)


def test_meta_router_identities_materialize_before_device_transfer() -> None:
    model = _rank17_meta_model()
    before = {
        name: buffer
        for name, buffer in model.named_buffers()
        if name.endswith("router.selection_identity")
    }
    assert len(before) == 75
    assert all(buffer.is_meta for buffer in before.values())
    before_rank_inputs = {
        name: buffer
        for name, buffer in model.named_buffers()
        if name.endswith("experts.expert_parallel_rank_tensor")
    }
    assert len(before_rank_inputs) == 75
    assert all(buffer.is_meta for buffer in before_rank_inputs.values())

    model._materialize_router_selection_identities(torch.device("cpu"))

    after = {
        name: buffer
        for name, buffer in model.named_buffers()
        if name.endswith("router.selection_identity")
    }
    assert after.keys() == before.keys()
    assert all(not buffer.is_meta for buffer in after.values())
    assert all(buffer.device.type == "cpu" for buffer in after.values())
    assert all(buffer.dtype is torch.float32 for buffer in after.values())
    assert all(buffer.shape == (256, 256) for buffer in after.values())
    expected = torch.eye(256, dtype=torch.float32)
    assert all(torch.equal(buffer, expected) for buffer in after.values())
    assert all(name not in model.state_dict() for name in after)

    rank_inputs = {
        name: buffer
        for name, buffer in model.named_buffers()
        if name.endswith("experts.expert_parallel_rank_tensor")
    }
    assert len(rank_inputs) == 75
    assert all(not buffer.is_meta for buffer in rank_inputs.values())
    assert all(buffer.device.type == "cpu" for buffer in rank_inputs.values())
    assert all(buffer.dtype is torch.int32 for buffer in rank_inputs.values())
    assert all(buffer.tolist() == [[17]] for buffer in rank_inputs.values())
    assert all(name not in model.state_dict() for name in rank_inputs)


def test_load_lifecycle_materializes_identity_and_preserves_cpu_router_selection() -> (
    None
):
    model = _rank17_meta_model()
    manifest = object()
    report = object()
    with (
        patch(
            "vllm_neuron.model.glm_moe_dsa.model.load_checkpoint_manifest",
            return_value=manifest,
        ),
        patch.object(model, "checkpoint_mappings", return_value={}),
        patch.object(model, "_account_manifest", return_value=report),
        patch.object(model, "_install_weight_loaders"),
        patch.object(model, "load_state_dict") as load_state_dict,
        patch(
            "vllm_neuron.model.glm_moe_dsa.model.dist.is_initialized",
            return_value=False,
        ),
        patch(
            "vllm_neuron.model.glm_moe_dsa.model.SafetensorsCheckpoint"
        ) as checkpoint_class,
    ):
        checkpoint_class.return_value.load_sharded.return_value = SimpleNamespace(
            state_dict={}
        )
        model.load_weights("/unused/checkpoint", torch.device("cpu"))

    load_state_dict.assert_called_once_with({}, strict=True, assign=True)
    assert model.last_load_report is report
    identities = {
        name: buffer
        for name, buffer in model.named_buffers()
        if name.endswith("router.selection_identity")
    }
    assert len(identities) == 75
    assert all(not buffer.is_meta for buffer in identities.values())
    assert all(buffer.device.type == "cpu" for buffer in identities.values())

    router = model.model.layers[3].mlp.router
    router.gate.weight = torch.nn.Parameter(torch.zeros(256, 6144))
    router.gate.e_score_correction_bias = torch.nn.Parameter(
        torch.linspace(-1.0, 1.0, 256, dtype=torch.float32)
    )

    affinities, selected = router(torch.zeros(2, 6144, dtype=torch.float32))

    expected = torch.arange(248, 256, dtype=torch.int64)
    assert torch.equal(torch.sort(selected[0].to(torch.int64)).values, expected)
    assert torch.equal(torch.sort(selected[1].to(torch.int64)).values, expected)
    torch.testing.assert_close(
        affinities.sum(dim=-1), torch.full((2,), 2.5), rtol=0, atol=0
    )


def test_rank17_manifest_binds_every_fp8_grid_and_has_hbm_headroom() -> None:
    model = _rank17_meta_model()
    manifest = load_checkpoint_manifest(CHECKPOINT_DIR / "model.safetensors.index.json")
    mappings = model.checkpoint_mappings(manifest)
    report = model._account_manifest(manifest, mappings)

    parameters = dict(model.named_parameters())
    dtype_counts = Counter(parameter.dtype for parameter in parameters.values())
    assert len(parameters) == report.local_keys == report.mapped_sources == 3_660
    assert report.load_targets == 2_094
    assert report.fp8_scales == 1_566
    assert dtype_counts[torch.float8_e4m3fn] == 1_566
    assert dtype_counts[torch.bfloat16] == 453
    assert dtype_counts[torch.float32] == 1_641
    assert sum(name.endswith(".weight_scale_inv") for name in parameters) == 1_566

    for parameter_name, checkpoint_key in mappings.items():
        parameter = parameters[parameter_name]
        if isinstance(checkpoint_key, list):
            assert len(checkpoint_key) == 2
            weight_key, scale_key = checkpoint_key
            assert parameter_name.startswith("model.layers.")
            assert ".mlp.experts.experts." in parameter_name
            assert scale_key == weight_key + "_scale_inv"
            if parameter_name.endswith(".weight_scale_inv"):
                assert parameter.dtype is torch.float32
                assert parameter.ndim == 1
            else:
                assert parameter.dtype is torch.float8_e4m3fn
                assert parameter.ndim == 2
            continue
        entry = manifest.by_key[checkpoint_key]
        if entry.info.disposition is Disposition.FP8_SCALE:
            assert parameter.dtype is torch.float32
            assert parameter_name.endswith(".weight_scale_inv")
        elif entry.header.dtype == "F8_E4M3":
            assert parameter.dtype is torch.float8_e4m3fn
        elif entry.header.dtype == "BF16":
            assert parameter.dtype is torch.bfloat16
        elif entry.header.dtype == "F32":
            assert parameter.dtype is torch.float32
        else:
            raise AssertionError(f"unexpected checkpoint dtype {entry.header.dtype}")

    rank_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in parameters.values()
    )
    assert rank_bytes / 2**30 < 12.5
    assert 24.0 - rank_bytes / 2**30 > 11.5


def test_weight_and_scale_loaders_preserve_storage_and_value_contract() -> None:
    model = _rank17_meta_model()
    weight = torch.arange(384 * 256, dtype=torch.float32).reshape(384, 256)
    weight = ((weight % 29) - 14).to(torch.float8_e4m3fn)
    scale = torch.arange(6, dtype=torch.float32).reshape(3, 2) + 1.0

    weight_loader = model._fp8_weight_loader(
        "model.layers.0.mlp.gate_proj.weight", (384, 256), 0
    )
    scale_loader = model._fp8_scale_loader(
        "model.layers.0.mlp.gate_proj.weight_scale_inv", (384, 256), 0
    )
    local_weight = weight_loader.load([weight], rank=17)
    local_scale = scale_loader.load([scale], rank=17)

    assert local_weight.dtype is torch.float8_e4m3fn
    torch.testing.assert_close(
        local_weight.float(),
        (weight[102:108].float() * FP8_STORAGE_SCALE).to(torch.float8_e4m3fn).float(),
    )
    torch.testing.assert_close(
        local_scale,
        scale[0:1] * FP8_INVERSE_SCALE_ADJUSTMENT,
    )


def test_routed_expert_pair_loaders_emit_row_fp8_without_bf16_state() -> None:
    model = _rank17_meta_model()
    values = torch.arange(256 * 384, dtype=torch.float32).reshape(256, 384)
    weight = ((values % 29) - 14).to(torch.float8_e4m3fn)
    scale = torch.tensor([[0.25, 0.5, 1.0], [0.75, 1.25, 1.5]], dtype=torch.float32)
    weight_key = "model.layers.3.mlp.experts.68.gate_proj.weight"
    scale_key = weight_key + "_scale_inv"

    row_weight = model._row_fp8_pair_loader(
        weight_key, scale_key, (256, 384), return_scale=False
    ).load([weight, scale], rank=17)
    row_scale = model._row_fp8_pair_loader(
        weight_key, scale_key, (256, 384), return_scale=True
    ).load([weight, scale], rank=17)

    assert row_weight.dtype is torch.float8_e4m3fn
    assert row_weight.shape == weight.shape
    assert row_scale.dtype is torch.float32
    assert row_scale.shape == (256,)
    reconstructed = row_weight.float() * row_scale[:, None]
    stored = (weight.float() * FP8_STORAGE_SCALE).to(torch.float8_e4m3fn)
    adjusted_scale = scale * FP8_INVERSE_SCALE_ADJUSTMENT
    expected = (
        stored.float()
        * adjusted_scale[
            (torch.arange(256) // 128)[:, None],
            (torch.arange(384) // 128)[None, :],
        ]
    )
    torch.testing.assert_close(reconstructed, expected, rtol=0.13, atol=0.1)
