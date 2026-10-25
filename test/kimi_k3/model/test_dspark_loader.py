# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 DSpark drafter: checkpoint mapping, TP 32 shards, the shared head and the cache spec.

Reads the draft checkpoint's header and the slices each check needs, never the whole
file. Set ``KIMI_K3_DSPARK_CHECKPOINT`` to the draft directory (default
``~/models/Inferact/Kimi-K3-DSpark-Block5``) and ``KIMI_K3_CHECKPOINT`` to the target
directory (default ``~/models/moonshotai/Kimi-K3``). Run with ``VLLM_NEURON_CPU_MODE=1``.
"""

import contextlib
import dataclasses
import os
import types
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform
from vllm.transformers_utils.config import get_config
from vllm.v1.core.kv_cache_utils import get_kv_cache_groups

from vllm_neuron.model.kimi_k3 import dspark
from vllm_neuron.model.kimi_k3 import model as kimi_model
from vllm_neuron.model.kv_cache import KeyOnlyAttentionSpec

DRAFT = Path(
    os.environ.get(
        "KIMI_K3_DSPARK_CHECKPOINT",
        Path.home() / "models" / "Inferact" / "Kimi-K3-DSpark-Block5",
    )
)
TARGET = Path(
    os.environ.get("KIMI_K3_CHECKPOINT", Path.home() / "models" / "moonshotai" / "Kimi-K3")
)
WEIGHTS = DRAFT / "model.safetensors"

pytestmark = pytest.mark.skipif(
    not (WEIGHTS.is_file() and (TARGET / "config.json").is_file()),
    reason=f"DSpark checkpoint at {DRAFT} or Kimi K3 config at {TARGET} not found",
)

TP = 32
TARGET_LAYERS = 93
LATENT = 576
# Per-rank slice of each sharded tensor kind at TP 32: (dim, size per rank). The
# plan's rules: q_b, kv_b and o_proj by head (3 heads of 192, 256 and 128 values),
# the MLP by its width, q_a, kv_a and the context projection by their input width,
# the second Markov weight by vocabulary. Every other tensor loads whole.
SHARDS = {
    "self_attn.q_a_proj.weight": (1, 224),
    "self_attn.kv_a_proj_with_mqa.weight": (1, 224),
    "self_attn.q_b_proj.weight": (0, 3 * 192),
    "self_attn.kv_b_proj.weight": (0, 3 * 256),
    "self_attn.o_proj.weight": (1, 3 * 128),
    "mlp.gate_proj.weight": (0, 448),
    "mlp.up_proj.weight": (0, 448),
    "mlp.down_proj.weight": (1, 448),
    "context_proj.weight": (1, 672),
    "markov_head.markov_w2.weight": (0, 5120),
}


@contextlib.contextmanager
def _tp_rank(rank: int, size: int):
    """Make the drafter and the target see a tensor-parallel group of ``size`` with this ``rank``.

    At size 32 the world is the served form, TP 32 by DP 2, and the wide expert group
    spans its 64 ranks. At size 1 the world is that one rank, so the experts stay on the
    tensor-parallel group.
    """
    group = types.SimpleNamespace(world_size=size, rank_in_group=rank, device_group=None)
    world_size = 2 * size if size > 1 else 1
    world = types.SimpleNamespace(world_size=world_size, rank_in_group=rank, device_group=None)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(dspark, "_tp_group", lambda: group)
        patch.setattr(kimi_model, "_tp_group", lambda: group)
        patch.setattr(kimi_model, "get_world_group", lambda: world)
        patch.setattr(kimi_model, "get_wide_ep_group", lambda: world)
        yield


@pytest.fixture(scope="module")
def draft_config():
    """vLLM's HF config for the draft checkpoint, as the proposer receives it."""
    return get_config(str(DRAFT), trust_remote_code=False)


def _draft(draft_config, size: int = TP) -> dspark.K3DSparkForCausalLM:
    """The drafter on the meta device, built on rank 0 of a ``size``-way group."""
    with _tp_rank(0, size), torch.device("meta"):
        return dspark.K3DSparkForCausalLM.from_configs(
            config=draft_config, start_layer_idx=TARGET_LAYERS
        )


def test_checkpoint_tensors_map_to_one_parameter_or_a_declared_skip(draft_config):
    with safe_open(WEIGHTS, framework="pt") as f:
        stored = set(f.keys())
    params = [name for name, _ in _draft(draft_config).named_parameters()]
    skipped = {k for k in stored if k.startswith(dspark.SKIPPED_PREFIXES)}
    assert (len(stored), len(params), len(set(params))) == (44, 41, 41)
    assert sorted(skipped) == [
        "confidence_head.proj.bias", "confidence_head.proj.weight", "embed_tokens.weight"
    ]
    unmapped = stored - skipped - set(params)
    missing = set(params) - stored
    assert (len(unmapped), len(missing)) == (0, 0)


def _expected_slice(stored, name: str, rank: int) -> torch.Tensor:
    """The part of a stored tensor that ``rank`` holds at TP 32, by the rules above."""
    rule = next((r for kind, r in SHARDS.items() if name.endswith(kind)), None)
    if rule is None:
        return stored[:]
    dim, size = rule
    index = [slice(None)] * len(stored.get_shape())
    index[dim] = slice(rank * size, (rank + 1) * size)
    return stored[tuple(index)]


def _differing_bytes(a: torch.Tensor, b: torch.Tensor) -> int:
    """Bytes that differ between two tensors of one shape and dtype."""
    assert (a.shape, a.dtype) == (b.shape, b.dtype)
    return int((a.contiguous().view(torch.uint8) != b.contiguous().view(torch.uint8)).sum())


def test_tp32_shards_of_ranks_0_and_31_equal_their_checkpoint_slices(draft_config):
    checked, differing = 0, 0
    with safe_open(WEIGHTS, framework="pt") as f:
        for rank in (0, TP - 1):
            with _tp_rank(rank, TP):
                with torch.device("meta"):
                    draft = dspark.K3DSparkForCausalLM.from_configs(
                        config=draft_config, start_layer_idx=TARGET_LAYERS
                    )
                draft.load_weights(str(DRAFT), torch.device("cpu"))
            expected = {}
            for name, param in draft.named_parameters():
                expected[name] = _expected_slice(f.get_slice(name), name, rank)
                differing += _differing_bytes(param, expected[name])
                checked += 1
            # The context KV weight: the three layers' kv_a slices stacked, in fp32.
            kv_a = [expected[f"layers.{i}.self_attn.kv_a_proj_with_mqa.weight"] for i in range(3)]
            differing += _differing_bytes(draft.context_kv_weight, torch.cat(kv_a).float())
    assert (checked, differing) == (2 * 41, 0)


@pytest.fixture(scope="module")
def target_on_meta():
    """The 93-layer target on the meta device, on one rank of a 32-way TP group."""
    with _tp_rank(0, TP), torch.device("meta"):
        return kimi_model.KimiK3ForConditionalGeneration.from_configs(
            get_config(str(TARGET), trust_remote_code=False)
        )


def test_alias_makes_the_target_embedding_and_lm_head_the_drafters(draft_config):
    with _tp_rank(0, 1), torch.device("meta"):
        target = kimi_model.KimiK3ForConditionalGeneration.from_configs(
            get_config(str(TARGET), trust_remote_code=False)
        )
    # Real storage for the embedding and the LM head only; the load step stacks the head.
    embed, head = target.model.embed_tokens, target.lm_head
    for module in (embed, head):
        shape = module.weight.shape
        module.weight = torch.nn.Parameter(torch.empty(shape, dtype=torch.bfloat16), False)
    draft = _draft(draft_config, size=1)
    with pytest.raises(ValueError, match="prepare_after_load"):
        draft.alias_target_head(target)
    assert draft.embed_tokens is None and draft.head_weights is None
    with _tp_rank(0, 1):
        target.prepare_after_load()
    halves = list(target.head_weights)
    draft.alias_target_head(target)
    assert draft.embed_tokens.weight is embed.weight
    assert [id(w) for w in draft.head_weights] == [id(w) for w in halves]
    allocated = [t for t in draft.state_dict(keep_vars=True).values() if not t.is_meta]
    ptrs = sorted(t.untyped_storage().data_ptr() for t in allocated)
    assert ptrs == sorted(t.untyped_storage().data_ptr() for t in (embed.weight, *halves))


@pytest.fixture(scope="module")
def served_config():
    """The target's served engine config after the platform's block-size step."""
    config = EngineArgs(
        model=str(TARGET),
        tensor_parallel_size=TP,
        data_parallel_size=2,
        enable_expert_parallel=True,
        max_model_len=131072,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.90,
    ).create_engine_config()
    current_platform.update_block_size_for_backend(config)
    return config


def test_kv_spec_is_one_key_only_group_at_the_target_block_size(
    draft_config, target_on_meta, served_config
):
    layers = _draft(draft_config).get_kv_spec().layers
    names = [f"layers.{TARGET_LAYERS + i}.self_attn" for i in range(3)]
    assert [layer.name for layer in layers] == names
    target_latent = next(s for s in target_on_meta.get_kv_spec().layers if s.key_only)
    for layer in layers:
        assert (layer.key_only, layer.num_kv_heads, layer.head_size) == (True, 1, LATENT)
        assert dataclasses.replace(layer, name=target_latent.name) == target_latent
    block = served_config.cache_config.block_size
    specs = {
        layer.name: KeyOnlyAttentionSpec(
            block_size=block, num_kv_heads=layer.num_kv_heads,
            head_size=layer.head_size, dtype=layer.dtype,
        )
        for layer in layers
    }
    groups = get_kv_cache_groups(served_config, specs)
    assert [sorted(g.layer_names) for g in groups] == [names]
    assert (groups[0].kv_cache_spec.block_size, block) == (256, 256)
