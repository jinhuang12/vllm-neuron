# SPDX-License-Identifier: Apache-2.0
"""Focused tests for prepacked GLM-5.2 row-FP8 routed experts."""

from __future__ import annotations

import ast
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import vllm_neuron.model.glm_moe_dsa.moe as moe_module
from vllm_neuron.model.glm_moe_dsa.block_fp8 import quantize_block_fp8_to_row
from vllm_neuron.model.glm_moe_dsa.moe import GlmMoeDsaRoutedExperts
from vllm_neuron.model.glm_moe_dsa.config import GlmMoeDsaConfig
from vllm_neuron.model.glm_moe_dsa.model import GlmMoeDsaForCausalLM
from vllm_neuron.model.glm_moe_dsa.packed_row_fp8 import (
    pack_down_row_fp8_bank,
    pack_gate_up_row_fp8_bank,
)
from vllm_neuron.model.glm_moe_dsa.weight_loaders import (
    load_checkpoint_manifest,
    local_load_plan,
)

CHECKPOINT_DIR_VALUE = os.environ.get("GLM52_MODEL_PATH")


def _block_pair(rows: int, columns: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    weight = (torch.randn(rows, columns, generator=generator) * 3).to(
        torch.float8_e4m3fn
    )
    scale = torch.rand(
        rows // 128,
        columns // 128,
        generator=generator,
        dtype=torch.float32,
    )
    return weight, scale


def test_load_time_packers_preserve_exact_row_conversion_and_layout() -> None:
    expert_pairs = [
        (_block_pair(128, 256, 1), _block_pair(128, 256, 2)),
        (_block_pair(128, 256, 3), _block_pair(128, 256, 4)),
    ]
    down_pairs = [_block_pair(256, 128, 5), _block_pair(256, 128, 6)]

    gate_up, gate_up_scale = pack_gate_up_row_fp8_bank(expert_pairs)
    down, down_scale = pack_down_row_fp8_bank(down_pairs)

    assert gate_up.shape == (2, 256, 2, 128)
    assert gate_up_scale.shape == (2, 2, 128)
    assert down.shape == (2, 128, 256)
    assert down_scale.shape == (2, 256)
    for expert_id, (gate_pair, up_pair) in enumerate(expert_pairs):
        expected_gate = quantize_block_fp8_to_row(*gate_pair)
        expected_up = quantize_block_fp8_to_row(*up_pair)
        assert torch.equal(gate_up[expert_id, :, 0], expected_gate[0].T)
        assert torch.equal(gate_up[expert_id, :, 1], expected_up[0].T)
        assert torch.equal(gate_up_scale[expert_id, 0], expected_gate[1])
        assert torch.equal(gate_up_scale[expert_id, 1], expected_up[1])
    for expert_id, pair in enumerate(down_pairs):
        expected_down = quantize_block_fp8_to_row(*pair)
        assert torch.equal(down[expert_id], expected_down[0].T)
        assert torch.equal(down_scale[expert_id], expected_down[1])


def test_packed_bank_loader_matches_old_pair_loader_byte_for_byte() -> None:
    model = object.__new__(GlmMoeDsaForCausalLM)
    torch.nn.Module.__init__(model)
    model.tensor_parallel_size = 1
    model.rank = 0
    checkpoint_keys = []
    sources = []
    by_key = {}
    pairs = []
    for expert_id in range(2):
        expert_pairs = []
        for projection_id, projection in enumerate(("gate_proj", "up_proj")):
            weight_key = f"experts.{expert_id}.{projection}.weight"
            scale_key = weight_key + "_scale_inv"
            pair = _block_pair(128, 256, 10 + expert_id * 2 + projection_id)
            checkpoint_keys.extend((weight_key, scale_key))
            sources.extend(pair)
            by_key[weight_key] = SimpleNamespace(
                header=SimpleNamespace(shape=(128, 256))
            )
            expert_pairs.append(pair)
        pairs.append(expert_pairs)

    actual_weights = model._packed_row_fp8_bank_loader(
        checkpoint_keys, by_key, "gate_up_weights"
    ).load(sources, rank=0)
    actual_scales = model._packed_row_fp8_bank_loader(
        checkpoint_keys, by_key, "gate_up_scales"
    ).load(sources, rank=0)

    expected_weights = []
    expected_scales = []
    for expert_id, expert_pairs in enumerate(pairs):
        row_pairs = []
        for projection_id, pair in enumerate(expert_pairs):
            weight_key = checkpoint_keys[expert_id * 4 + projection_id * 2]
            scale_key = weight_key + "_scale_inv"
            row_weight = model._row_fp8_pair_loader(
                weight_key,
                scale_key,
                (128, 256),
                return_scale=False,
            ).load(list(pair), rank=0)
            row_scale = model._row_fp8_pair_loader(
                weight_key,
                scale_key,
                (128, 256),
                return_scale=True,
            ).load(list(pair), rank=0)
            row_pairs.append((row_weight, row_scale))
        expected_weights.append(
            torch.stack((row_pairs[0][0].T, row_pairs[1][0].T), dim=1)
        )
        expected_scales.append(torch.stack((row_pairs[0][1], row_pairs[1][1]), dim=0))

    assert torch.equal(actual_weights, torch.stack(expected_weights, dim=0))
    assert torch.equal(actual_scales, torch.stack(expected_scales, dim=0))


@pytest.mark.skipif(
    CHECKPOINT_DIR_VALUE is None,
    reason="GLM52_MODEL_PATH is required for exact manifest coverage",
)
def test_packed_mapping_exactly_covers_rank_local_manifest(monkeypatch) -> None:
    checkpoint_dir = Path(CHECKPOINT_DIR_VALUE or ".")
    with (checkpoint_dir / "config.json").open() as config_file:
        config = GlmMoeDsaConfig.from_configs(json.load(config_file))
    monkeypatch.setenv("GLM_ENABLE_PACKED_ROW_FP8_MOE", "1")
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

    manifest = load_checkpoint_manifest(checkpoint_dir / "model.safetensors.index.json")
    mappings = model.checkpoint_mappings(manifest)
    report = model._account_manifest(manifest, mappings)
    expected_sources = {entry.key for entry in local_load_plan(manifest, ep_rank=17)}
    actual_sources = {
        key
        for mapped in mappings.values()
        for key in (mapped if isinstance(mapped, list) else [mapped])
    }

    assert actual_sources == expected_sources
    assert report.mapped_sources == report.local_keys == len(expected_sources)
    prefix = "model.layers.3.mlp.experts.packed_row_fp8."
    expected_gate_up = [
        key
        for expert_id in range(68, 72)
        for projection in ("gate_proj", "up_proj")
        for key in (
            f"model.layers.3.mlp.experts.{expert_id}.{projection}.weight",
            f"model.layers.3.mlp.experts.{expert_id}.{projection}.weight_scale_inv",
        )
    ]
    expected_down = [
        key
        for expert_id in range(68, 72)
        for key in (
            f"model.layers.3.mlp.experts.{expert_id}.down_proj.weight",
            f"model.layers.3.mlp.experts.{expert_id}.down_proj.weight_scale_inv",
        )
    ]
    assert mappings[prefix + "gate_up_weights"] == expected_gate_up
    assert mappings[prefix + "gate_up_scales"] == expected_gate_up
    assert mappings[prefix + "down_weights"] == expected_down
    assert mappings[prefix + "down_scales"] == expected_down


def test_packed_row_flag_allocates_only_final_kernel_banks(monkeypatch) -> None:
    monkeypatch.setenv("GLM_ENABLE_PACKED_ROW_FP8_MOE", "1")
    experts = GlmMoeDsaRoutedExperts(
        hidden_size=256,
        intermediate_size=128,
        num_experts=4,
        top_k=2,
        expert_parallel_size=1,
        fp8_weights=True,
        device="meta",
    )

    assert len(experts.experts) == 0
    packed = experts.packed_row_fp8
    assert packed is not None
    assert packed.gate_up_weights.shape == (4, 256, 2, 128)
    assert packed.down_weights.shape == (4, 128, 256)
    assert packed.gate_up_scales.shape == (4, 2, 128)
    assert packed.down_scales.shape == (4, 256)
    assert packed.gate_up_weights.dtype is torch.float8_e4m3fn
    assert packed.down_weights.dtype is torch.float8_e4m3fn
    assert packed.gate_up_scales.dtype is torch.float32
    assert packed.down_scales.dtype is torch.float32
    assert set(dict(experts.named_parameters())) == {
        "packed_row_fp8.gate_up_weights",
        "packed_row_fp8.down_weights",
        "packed_row_fp8.gate_up_scales",
        "packed_row_fp8.down_scales",
    }


def test_packed_row_dispatch_passes_banks_directly_once(monkeypatch) -> None:
    monkeypatch.setenv("GLM_ENABLE_PACKED_ROW_FP8_MOE", "1")
    experts = GlmMoeDsaRoutedExperts(
        hidden_size=256,
        intermediate_size=128,
        num_experts=4,
        top_k=2,
        expert_parallel_size=1,
        fp8_weights=True,
        device="meta",
    )
    hidden = torch.empty((32, 256), dtype=torch.bfloat16, device="meta")
    affinities = torch.empty((32, 4), dtype=torch.float32, device="meta")
    selected = torch.empty((32, 2), dtype=torch.int32, device="meta")
    expected = torch.empty_like(hidden)
    calls = []

    monkeypatch.setattr(moe_module, "can_run_kernel", lambda _: True)
    monkeypatch.setattr(
        experts,
        "_kernel_weights",
        lambda: pytest.fail("packed dispatch rebuilt expert weights"),
    )

    def fake_moe_tkg(**kwargs):
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(moe_module.NF, "moe_tkg", fake_moe_tkg)
    actual = experts(hidden, affinities, selected, is_decode=True)

    assert actual.shape == expected.shape
    assert len(calls) == 1
    packed = experts.packed_row_fp8
    assert packed is not None
    assert calls[0]["expert_gate_up_weights"] is packed.gate_up_weights
    assert calls[0]["expert_down_weights"] is packed.down_weights
    assert calls[0]["expert_gate_up_weights_scale"] is packed.gate_up_scales
    assert calls[0]["expert_down_weights_scale"] is packed.down_scales
    assert calls[0]["expert_affinities"].shape == affinities.shape
    assert calls[0]["expert_index"].shape == selected.shape
    assert calls[0]["is_all_expert"] is True
    assert calls[0]["mask_unselected_experts"] is True


def test_packed_row_prefill_keeps_moe_tkg_128_token_limit(monkeypatch) -> None:
    monkeypatch.setenv("GLM_ENABLE_PACKED_ROW_FP8_MOE", "1")
    experts = GlmMoeDsaRoutedExperts(
        hidden_size=256,
        intermediate_size=128,
        num_experts=4,
        top_k=2,
        expert_parallel_size=1,
        fp8_weights=True,
        device="meta",
    )
    hidden = torch.empty((257, 256), dtype=torch.bfloat16, device="meta")
    affinities = torch.empty((257, 4), dtype=torch.float32, device="meta")
    selected = torch.empty((257, 2), dtype=torch.int32, device="meta")
    calls = []

    monkeypatch.setattr(moe_module, "can_run_kernel", lambda _: True)

    def fake_moe_tkg(**kwargs):
        calls.append(kwargs)
        return torch.empty_like(kwargs["hidden_input"])

    monkeypatch.setattr(moe_module.NF, "moe_tkg", fake_moe_tkg)
    output = experts(hidden, affinities, selected, is_decode=False)

    assert output.shape == hidden.shape
    assert [call["hidden_input"].shape[0] for call in calls] == [128, 128, 1]
    packed = experts.packed_row_fp8
    assert packed is not None
    assert all(
        call["expert_gate_up_weights"] is packed.gate_up_weights for call in calls
    )
    assert all(call["expert_down_weights"] is packed.down_weights for call in calls)


def test_packed_row_runtime_has_no_layout_or_requantization_ops() -> None:
    forbidden_calls = {"stack", "transpose"}
    source = inspect.getsource(GlmMoeDsaRoutedExperts._packed_row_fp8_nki)
    tree = ast.parse(inspect.cleandoc(source))
    call_names = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert call_names.isdisjoint(forbidden_calls)
    assert not any(
        isinstance(node, ast.Attribute) and node.attr == "T" for node in ast.walk(tree)
    )
    assert "dequant" not in source
    assert "quantize" not in source


def test_packed_row_cpu_execution_fails_closed(monkeypatch) -> None:
    monkeypatch.setenv("GLM_ENABLE_PACKED_ROW_FP8_MOE", "1")
    experts = GlmMoeDsaRoutedExperts(
        hidden_size=256,
        intermediate_size=128,
        num_experts=4,
        top_k=2,
        expert_parallel_size=1,
        fp8_weights=True,
    )
    hidden = torch.empty((1, 256), dtype=torch.bfloat16)
    affinities = torch.empty((1, 4), dtype=torch.float32)
    selected = torch.empty((1, 2), dtype=torch.int32)

    monkeypatch.setattr(moe_module, "can_run_kernel", lambda _: False)
    with pytest.raises(RuntimeError, match="requires Neuron NKI execution"):
        experts(hidden, affinities, selected, is_decode=True)
