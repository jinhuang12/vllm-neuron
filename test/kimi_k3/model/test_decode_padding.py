# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 KDA decode on the KV cache's page buffer, at a step padded past its requests.

The runner allocates and binds the cache, and its block table comes from vLLM's input
batch after a request leaves. The decode step runs the landed kernels in the simulator on
that table, padded from 2 requests to 4 rows, and again with the padding rows on free blocks
of their own. Set ``KIMI_K3_CHECKPOINT`` to the checkpoint directory
(default ``~/models/moonshotai/Kimi-K3``); run with ``VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1``.
"""

import copy
import os
import types
from pathlib import Path

import pytest
import torch
from torch import nn

from vllm_neuron.model.kimi_k3 import model as m
from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

CHECKPOINT = Path(
    os.environ.get("KIMI_K3_CHECKPOINT", Path.home() / "models" / "moonshotai" / "Kimi-K3")
)

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "config.json").is_file(),
    reason=f"Kimi K3 checkpoint not found at {CHECKPOINT}",
)

LAYER, BLOCKS, PAGE, MAX_LEN = "layers.0.self_attn", 9, 294_912, 4096
REQUESTS, PADDED = {"a": 5, "b": 3, "c": 7}, 4


def _kda_layer_model(gen: torch.Generator) -> m.KimiK3ForConditionalGeneration:
    """A one-layer KDA model at one rank's widths, with random weights, laid out as loaded."""
    from vllm.transformers_utils.config import get_config

    config = copy.deepcopy(get_config(str(CHECKPOINT), trust_remote_code=False).to_dict())
    text = config["text_config"]
    text.update(num_hidden_layers=1, num_attention_heads=3, intermediate_size=1056,
                vocab_size=5120, first_k_dense_replace=1)
    text["linear_attn_config"].update(num_heads=3, kda_layers=[1], full_attn_layers=[])
    with torch.device("meta"):
        model = m.KimiK3ForConditionalGeneration.from_configs(config)
    for name, p in list(model.named_parameters()):
        path, _, leaf = name.rpartition(".")
        if "norm" in name:
            value = 1 + 0.1 * torch.randn(p.shape, generator=gen)
        else:
            scale = 0.5 if leaf in ("A_log", "dt_bias") or "conv" in name else 0.02
            value = scale * torch.randn(p.shape, generator=gen)
        param = nn.Parameter(value.to(p.dtype), requires_grad=False)
        setattr(model.get_submodule(path), leaf, param)
    model.prepare_after_load()
    return model


def _runner(model: nn.Module) -> NeuronModelRunner:
    """A model runner holding only what cache setup and the block-table upload read."""
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    cache = types.SimpleNamespace(block_size=MAX_LEN, mamba_cache_mode="none")
    runner.vllm_config = types.SimpleNamespace(cache_config=cache)
    runner.model, runner.speculative_config = model, None
    runner.max_num_reqs, runner.max_model_len, runner.max_num_batched_tokens = 8, MAX_LEN, MAX_LEN
    runner.device, runner.vocab_size, runner.is_pooling_model = torch.device("cpu"), 5120, False
    runner._kv_cache_full_tensors, runner._block_table_groups, runner._swa_groups = {}, {}, {}
    runner._kv_snapshot_enabled = False
    return runner


def _state_bytes(kda: m.KimiK3KDA, slot: int) -> torch.Tensor:
    """The slot's conv rows and recurrent state as bytes, read through the bound state views."""
    conv, state = kda.conv_state[slot].contiguous(), kda.recurrent_state[slot].contiguous()
    return torch.cat([conv.view(torch.uint8).flatten(), state.view(torch.uint8).flatten()])


@pytest.fixture(scope="module")
def steps() -> dict:
    """The block table, the cache before and after each step, and each step's output."""
    from vllm.sampling_params import SamplingParams
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
        MambaSpec,
    )
    from vllm.v1.worker.gpu_input_batch import CachedRequestState

    gen = torch.Generator().manual_seed(0)
    model = _kda_layer_model(gen)
    kda = model.model.layers[0].self_attn
    spec = MambaSpec(shapes=((3, 1152), (3, 128, 128)), dtypes=(torch.bfloat16, torch.float32),
                     block_size=MAX_LEN, page_size_padded=PAGE)
    tensor = KVCacheTensor(size=BLOCKS * PAGE, layers=[LAYER], layer_stride=BLOCKS * PAGE,
                           block_stride=PAGE)
    runner = _runner(model)
    runner.initialize_kv_cache(KVCacheConfig(num_blocks=BLOCKS, kv_cache_tensors=[tensor],
                                             kv_cache_groups=[KVCacheGroupSpec([LAYER], spec)]))
    cache = runner._state_pages[LAYER]
    cache.copy_(torch.randint(-128, 128, cache.shape, dtype=torch.int8, generator=gen))
    kda.conv_state.copy_(0.5 * torch.randn(kda.conv_state.shape, generator=gen))
    kda.recurrent_state.copy_(0.1 * torch.randn(kda.recurrent_state.shape, generator=gen))

    # Three requests join; the second leaves, and the batch closes the gap.
    batch = runner.input_batch
    for req_id, block in REQUESTS.items():
        batch.add_request(CachedRequestState(
            req_id=req_id, prompt_token_ids=[1, 2, 3], mm_features=[],
            sampling_params=SamplingParams(), generator=None, block_ids=([block],),
            num_computed_tokens=3, output_token_ids=[4]))
    batch.remove_request("b")
    batch.condense()
    runner._save_block_table_to_device()
    table = batch.block_table[0].get_device_tensor(PADDED)

    x = (0.5 * torch.randn(PADDED, 7168, generator=gen)).bfloat16()
    gain = (1 + 0.1 * torch.randn(7168, generator=gen)).bfloat16()
    n, before = batch.num_reqs, cache.clone()
    out_served = kda.forward_decode(x, None, gain, {"block_table_tensor": table})
    after = cache.clone()
    served = {b: _state_bytes(kda, b) for b in range(BLOCKS)}
    cache.copy_(before)
    held = table[:n, 0].tolist()
    own = table.clone()
    own[n:, 0] = torch.tensor([b for b in range(1, BLOCKS) if b not in held][: PADDED - n])
    out_own = kda.forward_decode(x, None, gain, {"block_table_tensor": own})
    return {"slots": m._state_slots({"block_table_tensor": table}).tolist(), "num_reqs": n,
            "before": before, "after": after, "served": served, "out_served": out_served,
            "own": {b: _state_bytes(kda, b) for b in range(BLOCKS)}, "out_own": out_own}


def test_padding_rows_write_no_request_slot(steps):
    """Padding rows take a slot no request holds and leave the requests' steps unchanged."""
    n, slots = steps["num_reqs"], steps["slots"]
    held, padding = slots[:n], slots[n:]
    assert not set(padding) & set(held), f"padding rows on request slots: {slots}"
    for s in held:
        start = steps["before"][s, : steps["served"][s].numel()].view(torch.uint8)
        assert not torch.equal(steps["served"][s], start), f"slot {s} not written in the cache"
        assert torch.equal(steps["served"][s], steps["own"][s]), f"slot {s} differs"
    served, own = steps["out_served"][:n], steps["out_own"][:n]
    assert torch.equal(served.view(torch.int16), own.view(torch.int16)), "request outputs differ"
    others = [b for b in range(BLOCKS) if b not in held + padding]
    assert torch.equal(steps["after"][others], steps["before"][others])
