# SPDX-License-Identifier: Apache-2.0
"""Increment 3 (revision 1): multi-layer eagle3 drafter backbone CPU tests.

Generalizes the plugin's hard-wired 1-layer fusion backbone to upstream's
N-layer architecture (upstream reference
``vllm/model_executor/models/llama_eagle3.py`` v0.21.0):
  - logical layer 0 = eagle3 fusion layer (2x-width qkv, hidden_norm, embeds
    concat — ``midlayer.*`` in the checkpoint);
  - logical layers 1..N-1 = plain Llama decoder layers (1x-width qkv, no
    hidden_norm).

All CPU-only (VLLM_NEURON_CPU_MODE) on a single-rank gloo group, following the
inc-3 ``test_eagle3_parallel_forward.py`` bootstrap. The NKI attention kernel
falls back to torch on CPU, so the real backbone forward runs end-to-end.

Coverage:
  (a) construction: 1-layer and 4-layer fixtures — fusion vs plain layer
      count, hidden_norm presence, and fusion-vs-plain qkv input widths.
  (b) strict weight load: synthetic safetensors for BOTH shapes — 1-layer
      (``midlayer.*``) and 4-layer (``midlayer.*`` + ``layers.{1,2,3}.*``);
      ``load_state_dict(strict=True)`` must succeed.
  (c) sequential forward shape: 1-layer and 4-layer — both traverse ALL N
      layers per step (multi-layer sequential is an upstream-supported config).
  (d) parallel forward shape: 4-layer single-pass P-EAGLE forward.
"""
import os

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "12496")

import pytest
import torch
from safetensors.torch import save_file

# s_prior (= max_blocks_per_seq * block_size) must be divisible by the NKI mask
# kernel's P_MAX (128); the torch fallback shares the same mask builder.
BLOCK_SIZE = 128
PTD = 201020
START_LAYER = 1  # drafter layer_idx offset (past the target's layers)


@pytest.fixture(scope="module", autouse=True)
def _neuron_cpu_parallel_state():
    """Single-rank distributed + Neuron parallel state for the whole module."""
    import torch.distributed as dist
    from vllm.config import VllmConfig, set_current_vllm_config

    from vllm_neuron.parallel.neuron_parallel_state import (
        initialize_neuron_parallel_state,
        is_initialized,
    )

    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", world_size=1, rank=0)
    cfg = VllmConfig()
    ctx = set_current_vllm_config(cfg)
    ctx.__enter__()
    if not is_initialized():
        initialize_neuron_parallel_state(tp_global_ranks=[0], local_rank=0)
    yield
    ctx.__exit__(None, None, None)


def _make_drafter(
    n_layers, hidden=16, n_heads=4, n_kv=2, head_dim=4, vocab=32, K=2
):
    from vllm_neuron.model.llama3.config import LlamaConfig
    from vllm_neuron.model.llama3.eagle3_model import Eagle3LlamaForCausalLM

    lc = LlamaConfig(
        vocab_size=vocab,
        draft_vocab_size=None,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv,
        head_dim=head_dim,
        torch_dtype=torch.float32,
    )
    model = Eagle3LlamaForCausalLM(lc, start_layer_idx=START_LAYER)
    model.num_speculative_tokens = K
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            if not p.is_meta:
                p.copy_(torch.randn_like(p) * 0.05)
    return model, lc


def _write_checkpoint(tmp_path, n_layers, hidden=16, n_heads=4, n_kv=2, head_dim=4,
                      vocab=32, with_mask_hidden=False):
    """Write a synthetic drafter checkpoint: midlayer.* + layers.{1..N-1}.*."""
    inter = hidden * 2
    qout = n_heads * head_dim
    kvout = n_kv * head_dim
    t = {
        "embed_tokens.weight": torch.randn(vocab, hidden),
        "norm.weight": torch.randn(hidden),
        "fc.weight": torch.randn(hidden, 3 * hidden),
        "lm_head.weight": torch.randn(vocab, hidden),
    }
    if with_mask_hidden:
        t["mask_hidden"] = torch.randn(1, 3 * hidden)

    def _layer(prefix, fusion):
        inw = 2 * hidden if fusion else hidden
        t[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(qout, inw)
        t[f"{prefix}.self_attn.k_proj.weight"] = torch.randn(kvout, inw)
        t[f"{prefix}.self_attn.v_proj.weight"] = torch.randn(kvout, inw)
        t[f"{prefix}.self_attn.o_proj.weight"] = torch.randn(hidden, qout)
        t[f"{prefix}.mlp.gate_proj.weight"] = torch.randn(inter, hidden)
        t[f"{prefix}.mlp.up_proj.weight"] = torch.randn(inter, hidden)
        t[f"{prefix}.mlp.down_proj.weight"] = torch.randn(hidden, inter)
        t[f"{prefix}.input_layernorm.weight"] = torch.randn(hidden)
        t[f"{prefix}.post_attention_layernorm.weight"] = torch.randn(hidden)
        if fusion:
            t[f"{prefix}.hidden_norm.weight"] = torch.randn(hidden)

    _layer("midlayer", True)
    for i in range(1, n_layers):
        _layer(f"layers.{i}", False)

    d = tmp_path / f"ckpt_{n_layers}L{'_mh' if with_mask_hidden else ''}"
    d.mkdir()
    save_file(t, str(d / "model.safetensors"))
    return str(d)


def _layer_names(n_layers):
    return [f"layers.{START_LAYER + i}.self_attn" for i in range(n_layers)]


def _bind_cache(model, n_layers, num_blocks=8):
    # bind_kv_cache loops ALL model layers and requires every layer name in a
    # single dict, so build caches for all N layers at once.
    caches = {}
    for i in range(n_layers):
        attn = model.model.layers[i].self_attn
        k = torch.zeros(
            num_blocks, attn.num_key_value_heads_per_rank, BLOCK_SIZE, attn.head_dim
        )
        v = torch.zeros_like(k)
        caches[f"layers.{START_LAYER + i}.self_attn"] = [k, v]
    model.bind_kv_cache(caches)


def _decode_meta(n_layers, bs, max_query_len):
    """Decode attn_metadata dict keyed for all N drafter layers."""
    block_table = torch.arange(bs, dtype=torch.int32).view(bs, 1)
    slot = (block_table.view(-1) * BLOCK_SIZE).to(torch.int64)
    meta = {}
    for ln in _layer_names(n_layers):
        meta[ln] = {
            "block_table_tensor": block_table,
            "slot_mapping": slot,
            "max_query_len": max_query_len,
            "block_size": BLOCK_SIZE,
            "max_blocks_per_seq": 1,
            "decode_token_threshold": max(max_query_len, 1),
        }
    return meta


# --------------------------------------------------------------------------
# (a) construction
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_layers", [1, 4])
def test_construction_layer_modes(n_layers):
    model, lc = _make_drafter(n_layers)
    layers = model.model.layers
    assert len(layers) == n_layers
    # logical layer 0 = fusion; the rest plain.
    assert layers[0].is_fusion_layer is True
    assert hasattr(layers[0], "hidden_norm")
    for i in range(1, n_layers):
        assert layers[i].is_fusion_layer is False
        assert not hasattr(layers[i], "hidden_norm")
    # absolute layer_idx offset past the target's layers.
    for i, layer in enumerate(layers):
        assert layer.layer_idx == START_LAYER + i


@pytest.mark.parametrize("n_layers", [1, 4])
def test_fusion_vs_plain_qkv_widths(n_layers):
    model, lc = _make_drafter(n_layers)
    H = lc.hidden_size
    # fusion qkv accepts 2*hidden; plain accepts 1*hidden. qkv_proj_weight is
    # [in, qkv_size] (transposed storage).
    assert model.model.layers[0].self_attn.qkv_proj_weight.shape[0] == 2 * H
    for i in range(1, n_layers):
        assert model.model.layers[i].self_attn.qkv_proj_weight.shape[0] == H


# --------------------------------------------------------------------------
# (b) strict weight load — BOTH shapes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_layers", [1, 4])
def test_strict_weight_load(tmp_path, n_layers):
    model, lc = _make_drafter(n_layers)
    ckpt = _write_checkpoint(tmp_path, n_layers, hidden=lc.hidden_size)
    # strict=True inside load_weights must succeed for both shapes.
    model.load_weights(ckpt, torch.device("cpu"))


# --------------------------------------------------------------------------
# (c) sequential forward — traverses ALL N layers per step
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_layers", [1, 4])
@pytest.mark.parametrize("K", [2, 3])
def test_sequential_forward_shape(n_layers, K):
    model, lc = _make_drafter(n_layers, K=K)
    _bind_cache(model, n_layers)
    bs = 2
    H = lc.hidden_size
    input_ids = torch.arange(1, bs + 1, dtype=torch.int32)
    positions = torch.zeros(bs, dtype=torch.long)
    tgt_hidden = torch.randn(bs, H * 3)
    sampling_positions = torch.arange(bs, dtype=torch.long)
    meta = _decode_meta(n_layers, bs, max_query_len=1)
    rank = torch.tensor(0, dtype=torch.int32)
    stacked, drafts_only, logits = model(
        input_ids=input_ids,
        positions=positions,
        initial_target_hidden_states=tgt_hidden,
        attn_metadata=meta,
        sampling_positions=sampling_positions,
        rank=rank,
    )
    # No bonus token -> stacked == drafts_only == [bs, K].
    assert drafts_only.shape == (bs, K)
    assert stacked.shape == (bs, K)
    assert drafts_only.dtype == torch.int32
    assert not torch.isnan(drafts_only.float()).any()


# --------------------------------------------------------------------------
# (d) parallel (P-EAGLE) forward over the 4-layer backbone
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_layers", [1, 4])
@pytest.mark.parametrize("K", [2, 4])
def test_parallel_forward_shape_multilayer(n_layers, K):
    model, lc = _make_drafter(n_layers, K=K)
    model.parallel_drafting = True
    model.ptd_token_id = PTD
    model.register_buffer(
        "mask_hidden", torch.randn(1, lc.hidden_size * 3), persistent=False
    )
    _bind_cache(model, n_layers)
    bs = 2
    H = lc.hidden_size
    input_ids = torch.arange(1, bs + 1, dtype=torch.int32)
    positions = torch.zeros(bs, dtype=torch.long)
    tgt_hidden = torch.randn(bs, H * 3)
    sampling_positions = torch.arange(bs, dtype=torch.long)
    # _forward_parallel reads base_meta from the fusion-layer key, then builds
    # its own per-layer metadata for all N layers.
    meta = _decode_meta(n_layers, bs, max_query_len=1)
    rank = torch.tensor(0, dtype=torch.int32)
    stacked, drafts_only, logits = model(
        input_ids=input_ids,
        positions=positions,
        initial_target_hidden_states=tgt_hidden,
        attn_metadata=meta,
        sampling_positions=sampling_positions,
        rank=rank,
    )
    assert drafts_only.shape == (bs, K)
    assert stacked.shape == (bs, K)
    assert logits is None
    assert drafts_only.dtype == torch.int32
    assert not torch.isnan(drafts_only.float()).any()
