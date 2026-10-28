# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 verify step at DCP 4: the warm-up attention metadata has the run-time shapes.

Warm-up compiles each verify graph from synthetic attention metadata, and a step whose tensors
differ in shape misses that graph. A stand-in runner holds the served cache groups: a state group
of 1 + 4 pages per request and a latent group of 256-token blocks, split over 4 DCP ranks. The
cache setup, the input batch and both metadata builders are the runner's own, and no model loads.
Run with ``VLLM_NEURON_CPU_MODE=1``.
"""

import types

import numpy as np
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheTensor, MambaSpec
from vllm.v1.worker.gpu_input_batch import CachedRequestState

from vllm_neuron.model.kv_cache import KeyOnlyAttentionSpec
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

DCP, DRAFTS, MAX_LEN, BLOCK, BLOCKS = 4, 4, 32768, 256, 16
BUCKETS = [2048, 8192, 16384, 28672]
STATE, LATENT = "layers.0.self_attn", "layers.3.self_attn"
PROMPT = 8


def _kv_cache_config() -> KVCacheConfig:
    """The served groups over one allocation: the state page is padded to the latent page."""
    latent = KeyOnlyAttentionSpec(block_size=BLOCK, num_kv_heads=1, head_size=576,
                                  dtype=torch.bfloat16)
    state = MambaSpec(block_size=MAX_LEN, shapes=((3, 16), (2, 8, 8)),
                      dtypes=(torch.bfloat16, torch.float32), mamba_cache_mode="none",
                      num_speculative_blocks=DRAFTS, page_size_padded=latent.page_size_bytes)
    page = latent.page_size_bytes
    tensors = [KVCacheTensor(size=BLOCKS * page, layers=[name], layer_stride=BLOCKS * page,
                             block_stride=page) for name in (STATE, LATENT)]
    return KVCacheConfig(num_blocks=BLOCKS, kv_cache_tensors=tensors,
                         kv_cache_groups=[KVCacheGroupSpec([STATE], state),
                                          KVCacheGroupSpec([LATENT], latent)])


def _runner() -> NeuronModelRunner:
    """A runner at DCP rank 0 of 4 with 4 drafts per step, its KV cache set up."""
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.vllm_config = types.SimpleNamespace(
        parallel_config=types.SimpleNamespace(decode_context_parallel_size=DCP),
        cache_config=types.SimpleNamespace(mamba_cache_mode="none", block_size=BLOCK))
    runner.model = types.SimpleNamespace(bind_kv_cache=lambda caches: None)
    runner.speculative_config = types.SimpleNamespace(num_speculative_tokens=DRAFTS,
                                                      use_eagle=lambda: False)
    runner.drafter = types.SimpleNamespace(num_speculative_tokens=DRAFTS)
    runner.neuron_config = NeuronConfig(decode_context_length_buckets=list(BUCKETS))
    runner.is_dflash_spec, runner.is_dspark_spec = False, True
    runner.max_num_reqs, runner.max_model_len, runner.max_num_batched_tokens = 2, MAX_LEN, 8192
    runner.device, runner.vocab_size, runner.is_pooling_model = torch.device("cpu"), 1024, False
    runner._kv_cache_full_tensors, runner._block_table_groups, runner._swa_groups = {}, {}, {}
    runner._constant_tensors, runner._kv_snapshot_enabled = {}, False
    runner._is_synthetic_model = False
    runner._dcp_size = runner.cp_world_size = DCP
    runner.cp_kv_cache_interleave_size, runner._cp_rank = 1, 0
    runner.initialize_kv_cache(_kv_cache_config())
    return runner


def _verify_step_metadata(runner: NeuronModelRunner) -> dict:
    """The run-time attention metadata of one request's verify step after an 8-token prompt."""
    runner.input_batch.add_request(CachedRequestState(
        req_id="a", prompt_token_ids=list(range(1, PROMPT + 1)), mm_features=[],
        sampling_params=SamplingParams(), generator=None, num_computed_tokens=PROMPT,
        output_token_ids=[7], block_ids=([1, 2, 3, 4, 5], [6])))
    positions = np.arange(PROMPT, PROMPT + 1 + DRAFTS)
    runner._compute_slot_mappings(positions, np.zeros(len(positions), dtype=np.int64))
    runner._save_block_table_to_device()
    runner._save_slot_mapping_to_device(len(positions))
    return runner._build_attention_metadata(
        padded_num_reqs=1, total_num_scheduled_tokens=len(positions), max_query_len=1 + DRAFTS,
        max_num_draft_tokens=DRAFTS, cached_seq_len=PROMPT, max_decode_ctx_len=PROMPT)


def _shapes(metadata: dict) -> dict:
    """Each tensor's shape, keyed by layer and field."""
    return {(layer, key): tuple(value.shape) for layer, fields in metadata.items()
            for key, value in fields.items() if isinstance(value, torch.Tensor)}


def test_verify_warmup_metadata_has_the_run_time_shapes():
    """A verify step at the 2,048-token bucket: every tensor has its warm-up shape."""
    runner = _runner()
    run = _verify_step_metadata(runner)
    assert run[LATENT]["full_block_table_tensor"].shape[1] == MAX_LEN // (BLOCK * DCP)
    warm = runner._build_warmup_attention_metadata(num_tokens=1 + DRAFTS, num_reqs=1,
                                                   ctx_bucket=BUCKETS[0])
    assert _shapes(warm) == _shapes(run)
