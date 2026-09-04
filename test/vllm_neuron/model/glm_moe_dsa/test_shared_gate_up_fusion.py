# SPDX-License-Identifier: Apache-2.0
"""Focused gates for the GLM-5.2 shared gate/up fusion candidate."""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from safetensors import safe_open

from vllm_neuron.model.glm_moe_dsa import block_fp8
from vllm_neuron.model.glm_moe_dsa.block_fp8 import (
    block_fp8_linear,
    shared_gate_up_block_fp8_linear,
)
from vllm_neuron.model.glm_moe_dsa.config import GlmMoeDsaConfig
from vllm_neuron.model.glm_moe_dsa.mlp import GlmMoeDsaSwiGLUMLP
from vllm_neuron.model.glm_moe_dsa.model import GlmMoeDsaForCausalLM
from vllm_neuron.model.glm_moe_dsa.weight_loaders import (
    load_checkpoint_manifest,
    local_load_plan,
)
from vllm_neuron.utils.weight_loader import get_weight_loader

FLAG = "GLM_ENABLE_SHARED_GATE_UP_FUSION"
MODEL_PATH_VALUE = os.environ.get("GLM52_MODEL_PATH")


def _fp8_fixture(
    rows: int, columns: int, *, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(rows, columns, generator=generator) * 0.2
    scales = (
        torch.rand(
            (rows + 127) // 128,
            (columns + 127) // 128,
            generator=generator,
        )
        * 0.03
        + 0.005
    )
    row_blocks = torch.arange(rows) // 128
    column_blocks = torch.arange(columns) // 128
    expanded = scales[row_blocks[:, None], column_blocks[None, :]]
    quantized = (weight / expanded).clamp(-240.0, 240.0)
    return quantized.to(torch.float8_e4m3fn), scales.to(torch.float32)


def _production_gate_up_fixture(
    token_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden = torch.randn(
        token_count,
        6144,
        dtype=torch.bfloat16,
        generator=torch.Generator().manual_seed(9000 + token_count),
    )
    gate_weight, gate_scale = _fp8_fixture(32, 6144, seed=91)
    up_weight, up_scale = _fp8_fixture(32, 6144, seed=92)
    return (
        hidden,
        torch.stack((gate_weight, up_weight)),
        torch.stack((gate_scale, up_scale)),
    )


def _two_call_reference(
    hidden: torch.Tensor,
    packed_weight: torch.Tensor,
    packed_scale: torch.Tensor,
) -> torch.Tensor:
    return torch.stack(
        tuple(
            block_fp8_linear(hidden, packed_weight[index], packed_scale[index])
            for index in range(2)
        )
    )


def test_default_path_retains_two_independent_projections(monkeypatch) -> None:
    monkeypatch.delenv(FLAG, raising=False)
    module = GlmMoeDsaSwiGLUMLP(
        6144,
        2048,
        tensor_parallel_size=64,
        fp8_weights=True,
        fuse_shared_gate_up=os.getenv(FLAG) == "1",
        device="meta",
    )

    assert hasattr(module, "gate_proj")
    assert hasattr(module, "up_proj")
    assert not hasattr(module, "gate_up_weights")
    assert module.down_proj.weight.shape == (6144, 32)


def test_flag_packs_gate_up_without_changing_down_projection(monkeypatch) -> None:
    monkeypatch.setenv(FLAG, "1")
    config = GlmMoeDsaConfig(quantization_config={"weight_block_size": [128, 128]})
    module = GlmMoeDsaSwiGLUMLP.shared_from_config(
        config,
        tensor_parallel_size=64,
        tensor_parallel_rank=17,
        device="meta",
    )
    dense = GlmMoeDsaSwiGLUMLP.dense_from_config(
        config,
        tensor_parallel_size=64,
        tensor_parallel_rank=17,
        device="meta",
    )

    assert not hasattr(module, "gate_proj")
    assert not hasattr(module, "up_proj")
    assert module.gate_up_weights.shape == (2, 32, 6144)
    assert module.gate_up_scales.shape == (2, 1, 48)
    assert module.down_proj.weight.shape == (6144, 32)
    assert module.down_proj.weight_scale_inv.shape == (48, 1)
    assert hasattr(dense, "gate_proj")
    assert hasattr(dense, "up_proj")
    assert not hasattr(dense, "gate_up_weights")


def test_cpu_fused_gate_up_matches_existing_two_call_path() -> None:
    hidden, packed_weight, packed_scale = _production_gate_up_fixture(32)
    expected = _two_call_reference(hidden, packed_weight, packed_scale)

    actual = shared_gate_up_block_fp8_linear(hidden, packed_weight, packed_scale)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_production_shared_mlp_preserves_silu_product_and_down_projection() -> None:
    hidden, packed_weight, packed_scale = _production_gate_up_fixture(32)
    down_weight, down_scale = _fp8_fixture(6144, 32, seed=93)
    legacy = GlmMoeDsaSwiGLUMLP(
        6144,
        2048,
        tensor_parallel_size=64,
        fp8_weights=True,
    )
    fused = GlmMoeDsaSwiGLUMLP(
        6144,
        2048,
        tensor_parallel_size=64,
        fp8_weights=True,
        fuse_shared_gate_up=True,
    )
    with torch.no_grad():
        legacy.gate_proj.weight.copy_(packed_weight[0])
        legacy.gate_proj.weight_scale_inv.copy_(packed_scale[0])
        legacy.up_proj.weight.copy_(packed_weight[1])
        legacy.up_proj.weight_scale_inv.copy_(packed_scale[1])
        legacy.down_proj.weight.copy_(down_weight)
        legacy.down_proj.weight_scale_inv.copy_(down_scale)
        fused.gate_up_weights.copy_(packed_weight)
        fused.gate_up_scales.copy_(packed_scale)
        fused.down_proj.weight.copy_(down_weight)
        fused.down_proj.weight_scale_inv.copy_(down_scale)

    with patch.object(block_fp8, "can_run_kernel", return_value=False):
        torch.testing.assert_close(fused(hidden), legacy(hidden), rtol=0, atol=0)
    assert fused.gate_up_weights.numel() == (
        legacy.gate_proj.weight.numel() + legacy.up_proj.weight.numel()
    )
    assert fused.gate_up_scales.numel() == (
        legacy.gate_proj.weight_scale_inv.numel()
        + legacy.up_proj.weight_scale_inv.numel()
    )


def test_gate_and_up_keep_independent_block_scale_grids() -> None:
    hidden, packed_weight, packed_scale = _production_gate_up_fixture(1)
    packed_scale[0].fill_(0.007)
    packed_scale[1].fill_(0.019)

    actual = shared_gate_up_block_fp8_linear(hidden, packed_weight, packed_scale)
    expected = _two_call_reference(hidden, packed_weight, packed_scale)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    gate_changed_scale = packed_scale.clone()
    gate_changed_scale[0] *= 2
    gate_changed = shared_gate_up_block_fp8_linear(
        hidden, packed_weight, gate_changed_scale
    )
    torch.testing.assert_close(gate_changed[1], actual[1], rtol=0, atol=0)
    assert not torch.equal(gate_changed[0], actual[0])


def test_kernel_contract_assigns_one_projection_to_each_lnc2_program() -> None:
    source = inspect.getsource(block_fp8._shared_gate_up_block_fp8_linear_nki)
    assert "nl.num_programs(axes=0) == 2" in source
    assert "projection_id = nl.program_id(axis=0)" in source
    assert source.count("projection_id") >= 4
    assert "src=weight[" in source
    assert "weight_scale_inv[" in source


@pytest.mark.skipif(
    os.environ.get("NKI_SIMULATOR") != "1",
    reason="requires the SDK 2.32 NKI CPU simulator",
)
@pytest.mark.parametrize("token_count", [1, 32, 127, 128, 129])
def test_nki_simulator_matches_existing_two_call_path(token_count: int) -> None:
    hidden, packed_weight, packed_scale = _production_gate_up_fixture(token_count)
    expected = _two_call_reference(hidden, packed_weight, packed_scale)

    with patch.object(block_fp8, "can_run_kernel", return_value=True):
        actual = shared_gate_up_block_fp8_linear(hidden, packed_weight, packed_scale)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _real_checkpoint_sources(
    model_path: Path, keys: list[str]
) -> tuple[list[torch.Tensor], dict[str, object]]:
    with (model_path / "model.safetensors.index.json").open() as index_file:
        weight_map = json.load(index_file)["weight_map"]
    sources = []
    by_key: dict[str, object] = {}
    for key in keys:
        filename = weight_map[key]
        with safe_open(model_path / filename, framework="pt", device="cpu") as shard:
            tensor = shard.get_tensor(key)
        sources.append(tensor)
        by_key[key] = SimpleNamespace(
            header=SimpleNamespace(shape=tuple(tensor.shape), dtype="F8_E4M3")
        )
    return sources, by_key


@pytest.mark.skipif(
    MODEL_PATH_VALUE is None,
    reason="GLM52_MODEL_PATH is required for the real-checkpoint packing gate",
)
@pytest.mark.parametrize("rank", [0, 17, 63])
def test_real_checkpoint_packing_is_byte_exact(rank: int) -> None:
    model_path = Path(MODEL_PATH_VALUE or ".")
    keys = GlmMoeDsaForCausalLM._shared_gate_up_checkpoint_keys(3)
    sources, by_key = _real_checkpoint_sources(model_path, keys)
    owner = object.__new__(GlmMoeDsaForCausalLM)
    owner.tensor_parallel_size = 64
    owner.rank = rank

    weight_loader = owner._shared_gate_up_block_fp8_loader(
        keys, by_key, return_scale=False
    )
    scale_loader = owner._shared_gate_up_block_fp8_loader(
        keys, by_key, return_scale=True
    )
    actual_weights = weight_loader.load(sources, rank)
    actual_scales = scale_loader.load(sources, rank)

    expected_weights = []
    expected_scales = []
    for offset in (0, 2):
        weight_key = keys[offset]
        scale_key = keys[offset + 1]
        shape = by_key[weight_key].header.shape
        expected_weights.append(
            owner._fp8_weight_loader(weight_key, shape, 0).load([sources[offset]], rank)
        )
        expected_scales.append(
            owner._fp8_scale_loader(scale_key, shape, 0).load(
                [sources[offset + 1]], rank
            )
        )

    assert torch.equal(
        actual_weights.view(torch.uint8),
        torch.stack(expected_weights).view(torch.uint8),
    )
    assert torch.equal(
        actual_scales.view(torch.uint8), torch.stack(expected_scales).view(torch.uint8)
    )
    assert actual_weights.shape == (2, 32, 6144)
    assert actual_scales.shape == (2, 1, 48)


@pytest.mark.skipif(
    MODEL_PATH_VALUE is None,
    reason="GLM52_MODEL_PATH is required for exact manifest coverage",
)
def test_fused_mapping_exactly_covers_rank_local_manifest(monkeypatch) -> None:
    model_path = Path(MODEL_PATH_VALUE or ".")
    with (model_path / "config.json").open() as config_file:
        config = GlmMoeDsaConfig.from_configs(json.load(config_file))
    monkeypatch.setenv(FLAG, "1")
    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_world_size", return_value=64),
        patch("torch.distributed.get_rank", return_value=17),
        torch.device("meta"),
    ):
        model = GlmMoeDsaForCausalLM(
            config,
            64,
            expert_parallel_rank=17,
            tp_group=SimpleNamespace(device_group=object()),
        )

    manifest = load_checkpoint_manifest(model_path / "model.safetensors.index.json")
    mappings = model.checkpoint_mappings(manifest)
    model._install_weight_loaders(manifest, mappings)
    report = model._account_manifest(manifest, mappings)
    expected_sources = {entry.key for entry in local_load_plan(manifest, ep_rank=17)}
    actual_sources = {
        key
        for mapped in mappings.values()
        for key in (mapped if isinstance(mapped, list) else [mapped])
    }

    assert actual_sources == expected_sources
    assert report.mapped_sources == report.local_keys == len(expected_sources)
    prefix = "model.layers.3.mlp.shared_experts."
    expected = GlmMoeDsaForCausalLM._shared_gate_up_checkpoint_keys(3)
    assert mappings[prefix + "gate_up_weights"] == expected
    assert mappings[prefix + "gate_up_scales"] == expected
    assert prefix + "gate_proj.weight" not in mappings
    assert prefix + "up_proj.weight" not in mappings
    parameters = dict(model.named_parameters())
    assert get_weight_loader(parameters[prefix + "gate_up_weights"]).transform
    assert get_weight_loader(parameters[prefix + "gate_up_scales"]).transform
