# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 hybrid KV cache: layer specs, page alignment, state banks and the memory budget.

The config is the served one: TP 32 by DP 2, 128K context, prefix caching off. Set
``KIMI_K3_CHECKPOINT`` to the checkpoint directory; the default is
``~/models/moonshotai/Kimi-K3``. Run with ``VLLM_NEURON_CPU_MODE=1``.
"""

import os
import types
from collections import Counter
from math import prod
from pathlib import Path

import pytest
import torch
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.platforms import current_platform
from vllm.transformers_utils.config import get_config
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import MultipleOf
from vllm.v1.attention.backends.utils import resolve_kv_cache_layout
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_config_from_groups,
    get_kv_cache_configs,
    get_kv_cache_groups,
)
from vllm.v1.kv_cache_interface import MambaSpec, MLAAttentionSpec

from vllm_neuron.model.kimi_k3 import model as kimi_model
from vllm_neuron.model.kv_cache import KeyOnlyAttentionSpec
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.vllm.attention.attn import NeuronAttentionBackend
from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner
from vllm_neuron.vllm.worker.neuron_worker import NeuronWorker

CHECKPOINT = Path(
    os.environ.get(
        "KIMI_K3_CHECKPOINT", Path.home() / "models" / "moonshotai" / "Kimi-K3"
    )
)

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "config.json").is_file(),
    reason=f"Kimi K3 checkpoint not found at {CHECKPOINT}",
)

TP, DP = 32, 2
LATENT = 576  # kv_lora_rank 512 + rope 64, one latent head
KDA_HEADS, KDA_HEAD_DIM, CONV_TAPS = 96, 128, 4
NEURON_DEFAULT_BLOCK = 32
# One LNC 2 trn3 core's HBM region, as the runtime reports it.
HBM_PER_RANK = 36 * 1024**3
# Weight bytes per rank of the served TP 32 by EP 64 layout (parallelism planner).
WEIGHT_BYTES_PER_RANK = 27_367_507_968
USERS_PER_REPLICA, CONTEXT = 32 // DP, 8192


@pytest.fixture(scope="module")
def vllm_config():
    """The served engine config after the platform's block-size step."""
    config = EngineArgs(
        model=str(CHECKPOINT),
        tensor_parallel_size=TP,
        data_parallel_size=DP,
        enable_expert_parallel=True,
        max_model_len=131072,
        enable_prefix_caching=False,
        gpu_memory_utilization=0.90,
    ).create_engine_config()
    current_platform.update_block_size_for_backend(config)
    return config


@pytest.fixture(scope="module")
def model():
    """The 93-layer model on the meta device, built on one rank of a 32-way TP group."""
    group = types.SimpleNamespace(world_size=TP, rank_in_group=0, device_group=None)
    with pytest.MonkeyPatch.context() as patch, torch.device("meta"):
        patch.setattr(kimi_model, "_tp_group", lambda: group)
        return kimi_model.KimiK3ForConditionalGeneration.from_configs(
            get_config(str(CHECKPOINT), trust_remote_code=False)
        )


def _runner(vllm_config, model) -> NeuronModelRunner:
    """A model runner holding only what KV spec and cache setup read."""
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.vllm_config = vllm_config
    runner.model = model
    runner.speculative_config = None
    runner.neuron_config = NeuronConfig()
    runner.max_num_reqs = USERS_PER_REPLICA
    runner.max_model_len = vllm_config.model_config.max_model_len
    runner.max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
    runner.device = torch.device("cpu")
    runner.vocab_size = model.text_config.vocab_size
    runner.is_pooling_model = False
    runner._kv_cache_full_tensors = {}
    runner._block_table_groups = {}
    runner._swa_groups = {}
    return runner


@pytest.fixture(scope="module")
def specs(vllm_config, model):
    """The runner's layer specs, with the KV layout resolved as the engine core does."""
    specs = _runner(vllm_config, model).get_kv_cache_spec()
    layouts = NeuronWorker.__new__(NeuronWorker).get_supported_kv_cache_layouts()
    resolve_kv_cache_layout(vllm_config, [layouts], list(specs.values()))
    return specs


@pytest.fixture(scope="module")
def served_kv_config(vllm_config, specs):
    """vLLM's KV cache config for the worker's budget on one rank."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
        patch.delenv("VLLM_NEURON_RESERVED_HEADROOM_GIB", raising=False)
        worker = NeuronWorker.__new__(NeuronWorker)
        worker.cache_config = vllm_config.cache_config
        budget = worker._compute_kv_budget(HBM_PER_RANK, WEIGHT_BYTES_PER_RANK)
    return get_kv_cache_configs(vllm_config, [specs], [budget])[0]


def _upstream_kda_state(vllm_config):
    """vLLM's KDA state shapes and dtypes for one rank."""
    shapes = MambaStateShapeCalculator.kda_state_shape(
        TP, KDA_HEADS, KDA_HEAD_DIM, conv_kernel_size=CONV_TAPS
    )
    dtypes = MambaStateDtypeCalculator.kda_state_dtype(
        torch.bfloat16, vllm_config.cache_config.mamba_cache_dtype
    )
    return tuple(shapes), tuple(dtypes)


def test_mla_page_is_block_times_576_bf16(vllm_config, model, specs):
    block = vllm_config.cache_config.block_size
    mla = {n: s for n, s in specs.items() if isinstance(s, KeyOnlyAttentionSpec)}
    assert sorted(mla) == sorted(
        f"layers.{i}.self_attn" for i in model.text_config.mla_layer_indices
    )
    assert len(mla) == 24
    for spec in mla.values():
        assert (spec.block_size, spec.num_kv_heads, spec.head_size) == (block, 1, LATENT)
        assert spec.dtype == torch.bfloat16
        assert spec.page_size_bytes == block * LATENT * 2


def test_kda_state_slot_is_the_conv_and_recurrent_state(vllm_config, model, specs):
    cache = vllm_config.cache_config
    shapes, dtypes = _upstream_kda_state(vllm_config)
    kda = {n: s for n, s in specs.items() if isinstance(s, MambaSpec)}
    assert sorted(kda) == sorted(
        f"layers.{i}.self_attn" for i in model.text_config.kda_layer_indices
    )
    assert len(kda) == 69
    for spec in kda.values():
        assert (spec.shapes, spec.dtypes) == (shapes, dtypes)
        slot = sum(prod(s) * d.itemsize for s, d in zip(spec.shapes, spec.dtypes))
        assert slot == 3 * 128 * 128 * 4 + 3 * 1152 * 2
        assert (spec.block_size, spec.mamba_cache_mode) == (cache.mamba_block_size, "none")
        assert spec.page_size_bytes == cache.mamba_page_size_padded


def test_two_specs_in_four_cache_groups(specs, served_kv_config):
    groups = served_kv_config.kv_cache_groups
    assert len(groups) == 4
    assert len({g.kv_cache_spec for g in groups}) == 2
    assert sorted(len(g.layer_names) for g in groups) == [23, 23, 23, 24]
    assert sorted(n for g in groups for n in g.layer_names) == sorted(specs)


def test_unified_page_equals_upstreams_alignment_rule(vllm_config, served_kv_config):
    # Upstream raises the block size until one attention page holds a state, in
    # steps of the kernel alignment, then pads the state page to the attention page.
    token_page = MLAAttentionSpec(
        block_size=1, num_kv_heads=1, head_size=LATENT, dtype=torch.bfloat16
    ).page_size_bytes
    shapes, dtypes = _upstream_kda_state(vllm_config)
    state_page = MambaSpec(shapes=shapes, dtypes=dtypes, block_size=-1).page_size_bytes
    kernel = min(
        s.base if isinstance(s, MultipleOf) else s
        for s in NeuronAttentionBackend.get_supported_kernel_block_sizes()
    )
    align = max(kernel, NEURON_DEFAULT_BLOCK, 128)
    block = align * cdiv(state_page, align * token_page)
    page = block * token_page
    assert (block, page) == (256, 294_912)
    cache = vllm_config.cache_config
    assert (cache.block_size, cache.mamba_page_size_padded) == (block, page)
    assert {g.kv_cache_spec.page_size_bytes for g in served_kv_config.kv_cache_groups} == {page}
    assert {t.block_stride for t in served_kv_config.kv_cache_tensors} == {page}


def test_cache_setup_binds_69_state_banks_and_24_latent_caches(vllm_config, model, specs):
    blocks, page = 4, vllm_config.cache_config.mamba_page_size_padded
    groups = get_kv_cache_groups(vllm_config, specs)
    slots = max(len(g.layer_names) for g in groups)
    config = get_kv_cache_config_from_groups(vllm_config, groups, blocks * slots * page)
    assert config.num_blocks == blocks
    caches = _runner(vllm_config, model).initialize_kv_cache(config)
    assert Counter(len(v) for v in caches.values()) == {2: 69, 1: 24}

    storages = set()
    for i, layer in enumerate(model.model.layers):
        attn, bound = layer.self_attn, caches[f"layers.{i}.self_attn"]
        if layer.is_kda:
            conv, state = attn.conv_state, attn.recurrent_state
            assert conv is bound[0] and state is bound[1]
            assert (conv.shape, conv.dtype) == ((blocks, 3, 1152), torch.bfloat16)
            assert (state.shape, state.dtype) == ((blocks, 3, 128, 128), torch.float32)
            assert (conv.stride(0) * 2, state.stride(0) * 4) == (page, page)
            storages.update(t.untyped_storage().data_ptr() for t in bound)
        else:
            assert attn.kv_cache.shape == (blocks, 256, LATENT)
            assert attn.kv_cache.dtype == torch.bfloat16
            storages.add(attn.kv_cache.untyped_storage().data_ptr())
    assert len(storages) == slots == 24

    # One sequence's state fills the first bytes of its block's page: conv rows, then the state.
    attn = model.model.layers[0].self_attn
    attn.conv_state[2].view(torch.uint8).fill_(1)
    attn.recurrent_state[2].view(torch.uint8).fill_(1)
    storage = attn.conv_state.untyped_storage()
    raw = torch.empty(0, dtype=torch.uint8).set_(storage, 0, (storage.nbytes(),), (1,))
    written = raw.nonzero().flatten()
    slot = 3 * 1152 * 2 + 3 * 128 * 128 * 4
    assert torch.equal(written, torch.arange(2 * page, 2 * page + slot))


def test_kv_budget_at_8k_32_users_fits_a_rank_with_the_weights(vllm_config, served_kv_config):
    config = served_kv_config
    pool = max(t.offset + t.size for t in config.kv_cache_tensors)
    per_block = pool // config.num_blocks
    per_request = sum(
        g.kv_cache_spec.max_num_blocks_per_req(vllm_config, CONTEXT)
        for g in config.kv_cache_groups
    )
    need = USERS_PER_REPLICA * per_request + 1  # block 0 is the null block
    usable = vllm_config.cache_config.gpu_memory_utilization * HBM_PER_RANK
    assert round(usable / 1e9, 1) == 34.8
    assert need <= config.num_blocks
    assert WEIGHT_BYTES_PER_RANK + need * per_block <= usable
    assert WEIGHT_BYTES_PER_RANK + pool <= usable
