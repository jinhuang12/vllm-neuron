# SPDX-License-Identifier: Apache-2.0
"""Focused Stage 2 tests for GLM-5.2 checkpoint metadata and sharding."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import pytest
import torch

from vllm_neuron.model.glm_moe_dsa.quantization import Fp8BlockQuantization
from vllm_neuron.model.glm_moe_dsa.weight_loaders import (
    PINNED_EP,
    PINNED_INDEX_KEY_COUNT,
    PINNED_SHARD_COUNT,
    PINNED_TOTAL_SIZE,
    Disposition,
    TensorCategory,
    TPShardSpec,
    UnexpectedCheckpointKey,
    classify_checkpoint_key,
    fp8_scale_coverage_for_key,
    load_checkpoint_index,
    load_checkpoint_manifest,
    local_load_plan,
    tp_shard_spec_for_key,
)

CHECKPOINT_DIR_VALUE = os.environ.get("GLM52_MODEL_PATH")
CHECKPOINT_DIR = Path(CHECKPOINT_DIR_VALUE or ".")


def _require_checkpoint_dir() -> Path:
    if CHECKPOINT_DIR_VALUE is None:
        pytest.skip("GLM52_MODEL_PATH is required for pinned-checkpoint tests")
    return CHECKPOINT_DIR


@pytest.fixture(scope="module")
def manifest():
    index_path = _require_checkpoint_dir() / "model.safetensors.index.json"
    assert index_path.is_file()
    return load_checkpoint_manifest(index_path)


def test_pinned_fp8_config_is_dynamic_e4m3_block_128() -> None:
    with (_require_checkpoint_dir() / "config.json").open() as config_file:
        raw = json.load(config_file)["quantization_config"]
    quant = Fp8BlockQuantization.from_hf_config(raw)
    assert quant.format == "e4m3"
    assert quant.activation_scheme == "dynamic"
    assert quant.weight_block_size == (128, 128)
    assert quant.scale_suffix == "weight_scale_inv"
    assert quant.torch_weight_dtype is torch.float8_e4m3fn


def test_lightweight_index_accounts_for_every_key_without_shard_headers() -> None:
    index = load_checkpoint_index(
        _require_checkpoint_dir() / "model.safetensors.index.json"
    )
    assert len(index.key_to_shard) == PINNED_INDEX_KEY_COUNT
    assert len(index.shard_names) == PINNED_SHARD_COUNT
    assert index.total_size == PINNED_TOTAL_SIZE


def test_full_manifest_accounts_for_all_keys_and_only_mtp_is_skipped(manifest) -> None:
    assert len(manifest.entries) == PINNED_INDEX_KEY_COUNT
    assert len({entry.key for entry in manifest.entries}) == PINNED_INDEX_KEY_COUNT
    assert sum(entry.info.is_scale for entry in manifest.entries) == 59_044
    assert (
        sum(
            entry.info.is_scale
            and entry.info.disposition is Disposition.INTENTIONAL_SKIP
            for entry in manifest.entries
        )
        == 778
    )
    assert all(
        entry.key.endswith(".weight_scale_inv")
        for entry in manifest.entries
        if entry.info.is_scale
    )

    assert manifest.disposition_counts == Counter(
        {
            Disposition.LOAD_TARGET: 58_794,
            Disposition.FP8_SCALE: 58_266,
            Disposition.INTENTIONAL_SKIP: 1_569,
        }
    )
    assert manifest.category_counts == Counter(
        {
            TensorCategory.ROUTED_EXPERT: 115_200,
            TensorCategory.ATTENTION: 1_083,
            TensorCategory.MTP: 1_569,
            TensorCategory.SHARED_EXPERT: 450,
            TensorCategory.LAYER_NORM: 156,
            TensorCategory.ROUTER: 150,
            TensorCategory.DENSE_MLP: 18,
            TensorCategory.OUTER: 3,
        }
    )

    skipped = [
        entry
        for entry in manifest.entries
        if entry.info.disposition is Disposition.INTENTIONAL_SKIP
    ]
    assert len(skipped) == 1_569
    assert all(entry.key.startswith("model.layers.78.") for entry in skipped)
    assert {entry.key for entry in skipped} == {
        entry.key
        for entry in manifest.entries
        if entry.key.startswith("model.layers.78.")
    }


@pytest.mark.parametrize(
    ("key", "dtype", "shape"),
    [
        ("model.layers.0.self_attn.q_a_proj.weight", "F8_E4M3", (2048, 6144)),
        ("model.layers.0.self_attn.q_a_proj.weight_scale_inv", "F32", (16, 48)),
        ("model.layers.0.self_attn.q_b_proj.weight", "F8_E4M3", (16384, 2048)),
        ("model.layers.0.self_attn.q_b_proj.weight_scale_inv", "F32", (128, 16)),
        (
            "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
            "F8_E4M3",
            (576, 6144),
        ),
        (
            "model.layers.0.self_attn.kv_a_proj_with_mqa.weight_scale_inv",
            "F32",
            (5, 48),
        ),
        ("model.layers.0.self_attn.kv_b_proj.weight", "F8_E4M3", (28672, 512)),
        ("model.layers.0.self_attn.kv_b_proj.weight_scale_inv", "F32", (224, 4)),
        ("model.layers.0.self_attn.o_proj.weight", "F8_E4M3", (6144, 16384)),
        ("model.layers.0.self_attn.o_proj.weight_scale_inv", "F32", (48, 128)),
        ("model.layers.0.mlp.gate_proj.weight", "F8_E4M3", (12288, 6144)),
        ("model.layers.0.mlp.gate_proj.weight_scale_inv", "F32", (96, 48)),
        ("model.layers.3.mlp.gate.weight", "BF16", (256, 6144)),
        ("model.layers.3.mlp.gate.e_score_correction_bias", "F32", (256,)),
        (
            "model.layers.3.mlp.experts.0.gate_proj.weight",
            "F8_E4M3",
            (2048, 6144),
        ),
        (
            "model.layers.3.mlp.experts.0.gate_proj.weight_scale_inv",
            "F32",
            (16, 48),
        ),
        (
            "model.layers.3.mlp.shared_experts.down_proj.weight",
            "F8_E4M3",
            (6144, 2048),
        ),
        (
            "model.layers.3.mlp.shared_experts.down_proj.weight_scale_inv",
            "F32",
            (48, 16),
        ),
    ],
)
def test_representative_real_headers(manifest, key, dtype, shape) -> None:
    entry = manifest.by_key[key]
    assert entry.header.dtype == dtype
    assert entry.header.shape == shape


@pytest.mark.parametrize(
    ("key", "shape", "local_shape", "shard_dim"),
    [
        ("model.layers.0.self_attn.q_a_proj.weight", (2048, 6144), (2048, 6144), None),
        ("model.layers.0.self_attn.q_b_proj.weight", (16384, 2048), (256, 2048), 0),
        (
            "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
            (576, 6144),
            (576, 6144),
            None,
        ),
        ("model.layers.0.self_attn.kv_b_proj.weight", (28672, 512), (448, 512), 0),
        ("model.layers.0.self_attn.o_proj.weight", (6144, 16384), (6144, 256), 1),
        ("model.layers.0.mlp.gate_proj.weight", (12288, 6144), (192, 6144), 0),
        ("model.layers.0.mlp.up_proj.weight", (12288, 6144), (192, 6144), 0),
        ("model.layers.0.mlp.down_proj.weight", (6144, 12288), (6144, 192), 1),
        (
            "model.layers.3.mlp.experts.7.gate_proj.weight",
            (2048, 6144),
            (2048, 6144),
            None,
        ),
        (
            "model.layers.3.mlp.shared_experts.gate_proj.weight",
            (2048, 6144),
            (32, 6144),
            0,
        ),
        (
            "model.layers.3.mlp.shared_experts.down_proj.weight",
            (6144, 2048),
            (6144, 32),
            1,
        ),
    ],
)
def test_representative_tp64_shapes(key, shape, local_shape, shard_dim) -> None:
    spec = tp_shard_spec_for_key(key, shape)
    assert spec.shard_dim == shard_dim
    assert spec.local_shape == local_shape


def test_tp_slices_have_exact_synthetic_values_and_shapes() -> None:
    column = torch.arange(32).reshape(8, 4)
    column_spec = tp_shard_spec_for_key(
        "model.layers.0.self_attn.q_b_proj.weight", column.shape, world_size=4
    )
    assert torch.equal(column_spec.load_slice(column, 2), column[4:6, :])

    row = torch.arange(32).reshape(4, 8)
    row_spec = tp_shard_spec_for_key(
        "model.layers.0.self_attn.o_proj.weight", row.shape, world_size=4
    )
    assert torch.equal(row_spec.load_slice(row, 1), row[:, 2:4])

    dense = torch.arange(32).reshape(8, 4)
    dense_spec = tp_shard_spec_for_key(
        "model.layers.0.mlp.gate_proj.weight", dense.shape, world_size=4
    )
    assert torch.equal(dense_spec.load_slice(dense, 3), dense[6:8, :])

    with pytest.raises(ValueError, match="not divisible"):
        TPShardSpec((7, 4), 0, 4)
    with pytest.raises(ValueError, match="Local shard shape"):
        dense_spec.load_slice(dense, 0, expected_local_shape=(3, 4))


def test_tp64_inverse_scale_coverage_handles_block_boundaries() -> None:
    q_b = fp8_scale_coverage_for_key(
        "model.layers.0.self_attn.q_b_proj.weight", (16384, 2048), rank=17
    )
    assert q_b.local_scale_shape == (2, 16)
    assert q_b.weight_offset_in_first_block == 0

    kv_b = fp8_scale_coverage_for_key(
        "model.layers.0.self_attn.kv_b_proj.weight", (28672, 512), rank=1
    )
    assert kv_b.local_scale_shape == (4, 4)
    assert kv_b.weight_offset_in_first_block == 64

    dense = fp8_scale_coverage_for_key(
        "model.layers.0.mlp.gate_proj.weight", (12288, 6144), rank=1
    )
    assert dense.local_scale_shape == (2, 48)
    assert dense.weight_offset_in_first_block == 64

    o_proj = fp8_scale_coverage_for_key(
        "model.layers.0.self_attn.o_proj.weight", (6144, 16384), rank=9
    )
    assert o_proj.local_scale_shape == (48, 2)
    assert o_proj.weight_offset_in_first_block == 0


def test_ep64_owns_exactly_four_routed_experts_and_keeps_shared_separate(
    manifest,
) -> None:
    assert PINNED_EP.experts_per_rank == 4
    assert PINNED_EP.routed_expert_tp_size == 1
    assert PINNED_EP.routed_experts_for_rank(0) == (0, 1, 2, 3)
    assert PINNED_EP.routed_experts_for_rank(1) == (4, 5, 6, 7)
    assert PINNED_EP.routed_experts_for_rank(63) == (252, 253, 254, 255)
    assert {
        expert
        for rank in range(64)
        for expert in PINNED_EP.routed_experts_for_rank(rank)
    } == set(range(256))

    routed = classify_checkpoint_key("model.layers.3.mlp.experts.7.gate_proj.weight")
    shared = classify_checkpoint_key(
        "model.layers.3.mlp.shared_experts.gate_proj.weight"
    )
    assert routed.category is TensorCategory.ROUTED_EXPERT
    assert routed.expert_index == 7
    assert PINNED_EP.rank_loads(routed, 1)
    assert not PINNED_EP.rank_loads(routed, 0)
    assert shared.category is TensorCategory.SHARED_EXPERT
    assert shared.expert_index is None
    assert all(PINNED_EP.rank_loads(shared, rank) for rank in range(64))
    with pytest.raises(ValueError, match="outside EP=64"):
        local_load_plan(manifest, ep_rank=-1)
    with pytest.raises(ValueError, match="outside EP=64"):
        local_load_plan(manifest, ep_rank=64)

    routed_spec = tp_shard_spec_for_key(
        routed.key, manifest.by_key[routed.key].header.shape
    )
    assert routed_spec.shard_dim is None
    assert routed_spec.local_shape == (2048, 6144)
    routed_scales = fp8_scale_coverage_for_key(
        routed.key, routed_spec.global_shape, rank=1
    )
    assert routed_scales.local_scale_shape == (16, 48)
    assert routed_scales.weight_offset_in_first_block == 0

    shared_spec = tp_shard_spec_for_key(
        shared.key, manifest.by_key[shared.key].header.shape
    )
    assert shared_spec.shard_dim == 0
    assert shared_spec.local_shape == (32, 6144)
    shared_scales = fp8_scale_coverage_for_key(
        shared.key, shared_spec.global_shape, rank=1
    )
    assert shared_scales.local_scale_shape == (1, 48)
    assert shared_scales.weight_offset_in_first_block == 32

    plan = local_load_plan(manifest, ep_rank=1)
    assert len(plan) == 3_660
    assert (
        sum(entry.info.category is TensorCategory.ROUTED_EXPERT for entry in plan)
        == 1_800
    )
    assert (
        sum(entry.info.category is TensorCategory.SHARED_EXPERT for entry in plan)
        == 450
    )
    assert not any(
        entry.info.disposition is Disposition.INTENTIONAL_SKIP for entry in plan
    )


@pytest.mark.parametrize(
    "key",
    [
        "unexpected.weight",
        "model.layers.79.input_layernorm.weight",
        "model.layers.4.self_attn.indexer.wk.weight",
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.3.mlp.experts.256.gate_proj.weight",
        "model.layers.3.mlp.fake_projection.weight",
        "model.layers.77.eh_proj.weight",
        "model.layers.78.not_an_mtp_field.weight",
    ],
)
def test_invalid_or_unexpected_keys_fail_closed(key) -> None:
    with pytest.raises(UnexpectedCheckpointKey):
        classify_checkpoint_key(key)
