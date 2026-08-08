# SPDX-License-Identifier: Apache-2.0
"""Increment 4 (P-EAGLE dispatch) CPU-only unit tests — group B: routing.

Proves the DESIGN-1 dispatch actually works end-to-end on the synthetic-input
path that warmup/graph_extract use:

  (b) When ``parallel_drafting`` is threaded onto the drafter, driving the
      EagleProposer's synthetic-input path (``_build_synthetic_inputs`` ->
      ``propose`` -> ``Eagle3LlamaForCausalLM.forward``) with the DESIGN-1
      input window ``bs * (1 + K)`` ROUTES INTO ``_forward_parallel`` (spy
      confirms the call). With the flag off it does NOT — the sequential
      recurrent path runs. This is the trace-level proof the lead asked for
      (not an assumption that threading works).

  (a) The routed parallel propose returns the sequential output contract:
      ``drafts_only`` is ``[bs, K]`` (K draft ids per request), matching what
      the sequential path returns, so ``_propose_draft_token_ids`` consumers
      are unchanged.

Runs on CPU via the same single-rank gloo + Neuron-parallel-state bootstrap the
increment-3 forward test uses (the drafter attention layer needs Neuron parallel
groups to instantiate; the NKI kernel falls back to torch under
VLLM_NEURON_CPU_MODE).
"""
import os

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "12496")

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm_neuron.vllm.spec_decode.eagle import EagleProposer

BLOCK_SIZE = 128
PTD = 201020


@pytest.fixture(scope="module", autouse=True)
def _neuron_cpu_parallel_state():
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


def _make_drafter_model(
    hidden=16, n_heads=4, n_kv=2, head_dim=4, vocab=32, K=2, n_layers=1
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
        torch_dtype=torch.bfloat16,
    )
    model = Eagle3LlamaForCausalLM(lc, start_layer_idx=1)
    model.num_speculative_tokens = K
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            if not p.is_meta:
                p.copy_(torch.randn_like(p) * 0.05)
    # KV cache bind for ALL N drafter layers (single block per request, pos 0).
    caches = {}
    for layer in model.model.layers:
        attn = layer.self_attn
        k = torch.zeros(
            8, attn.num_key_value_heads_per_rank, BLOCK_SIZE, attn.head_dim,
            dtype=torch.bfloat16,
        )
        v = torch.zeros_like(k)
        caches[f"layers.{layer.layer_idx}.self_attn"] = [k, v]
    model.bind_kv_cache(caches)
    return model, lc


def _make_bare_proposer(model, lc, K, parallel):
    """A bare EagleProposer (bypassing __init__/gloo-heavy construction) with
    exactly the attributes ``_build_synthetic_inputs`` + ``propose`` read."""
    p = EagleProposer.__new__(EagleProposer)
    p.model = model
    p.device = torch.device("cpu")
    p.num_speculative_tokens = K
    p.on_device_sampling = False
    # One attn layer name per drafter layer (mirrors EagleProposer.load_model,
    # eagle.py:191-196, which derives these from draft num_hidden_layers).
    p.attn_layer_names = [
        f"layers.{layer.layer_idx}.self_attn" for layer in model.model.layers
    ]
    p.rank_tensor = torch.tensor(0, dtype=torch.int32)
    p.parallel_drafting = parallel
    p.ptd_token_id = PTD if parallel else None
    p.speculative_config = SimpleNamespace(model="test-drafter")
    p.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=False),
        model_config=SimpleNamespace(model="test-target"),
    )
    # Parallel forward needs the mask_hidden buffer + the flag on the model.
    model.parallel_drafting = parallel
    model.ptd_token_id = PTD if parallel else None
    if parallel and "mask_hidden" not in dict(model.named_buffers()):
        model.register_buffer(
            "mask_hidden", torch.randn(1, lc.hidden_size * 3, dtype=torch.bfloat16), persistent=False
        )
    return p


def _decode_attn_metadata(bs, model=None):
    block_table = torch.arange(bs, dtype=torch.int32).view(bs, 1)
    slot = (block_table.view(-1) * BLOCK_SIZE).to(torch.int64)
    entry = {
        "block_table_tensor": block_table,
        "slot_mapping": slot,
        "max_query_len": 1,
        "block_size": BLOCK_SIZE,
        "max_blocks_per_seq": 1,
        "decode_token_threshold": 1,
    }
    # Default 1-layer key preserves the original single-layer callers; when a
    # model is supplied, key an entry per drafter layer (N-layer backbone).
    if model is None:
        return {"layers.1.self_attn": dict(entry)}
    return {
        f"layers.{layer.layer_idx}.self_attn": dict(entry)
        for layer in model.model.layers
    }


@pytest.mark.parametrize("bs", [1, 2])
@pytest.mark.parametrize("K", [2, 4])
def test_synthetic_path_routes_to_parallel_forward(bs, K):
    """(b)+(a) parallel on: synthetic input-window path hits _forward_parallel
    and returns [bs, K] drafts."""
    model, lc = _make_drafter_model(K=K)
    p = _make_bare_proposer(model, lc, K, parallel=True)

    num_tokens = p.draft_graph_input_tokens(bs)  # bs*(1+K), DESIGN 1
    assert num_tokens == bs * (1 + K)
    kwargs = p._build_synthetic_inputs(num_tokens, bs)
    # Synthetic builder keyed to the DECODE branch (K+1 cols) at this window.
    assert kwargs["raw_sampled_token_ids"].shape == (bs, K + 1)
    meta = _decode_attn_metadata(bs)

    real_parallel = model._forward_parallel
    with mock.patch.object(
        model, "_forward_parallel", side_effect=real_parallel, autospec=False
    ) as spy:
        draft_token_ids, drafts_only = p.propose(
            attn_metadata=meta, is_warmup=True, **kwargs
        )
    assert spy.call_count == 1, "parallel flag must route to _forward_parallel"
    # (a) K draft ids per request, sequential output contract.
    assert drafts_only.shape == (bs, K)
    assert drafts_only.dtype == torch.int32


@pytest.mark.parametrize("bs", [1, 2])
@pytest.mark.parametrize("K", [2, 4])
def test_synthetic_path_sequential_does_not_route_parallel(bs, K):
    """Flag off: the same synthetic path must NOT call _forward_parallel."""
    model, lc = _make_drafter_model(K=K)
    p = _make_bare_proposer(model, lc, K, parallel=False)

    num_tokens = p.draft_graph_input_tokens(bs)
    kwargs = p._build_synthetic_inputs(num_tokens, bs)
    meta = _decode_attn_metadata(bs)

    real_parallel = model._forward_parallel
    with mock.patch.object(
        model, "_forward_parallel", side_effect=real_parallel, autospec=False
    ) as spy:
        draft_token_ids, drafts_only = p.propose(
            attn_metadata=meta, is_warmup=True, **kwargs
        )
    assert spy.call_count == 0, "sequential path must not touch _forward_parallel"
    # Sequential also yields [bs, K] drafts (contract parity).
    assert drafts_only.shape == (bs, K)


@pytest.mark.parametrize("bs,K", [(1, 2), (2, 4)])
def test_warmup_entrypoint_traces_parallel(bs, K):
    """The public warmup() entrypoint (used by neuron_worker/runner) drives the
    parallel path end-to-end without error when the flag is on."""
    model, lc = _make_drafter_model(K=K)
    p = _make_bare_proposer(model, lc, K, parallel=True)
    num_tokens = p.draft_graph_input_tokens(bs)
    meta = _decode_attn_metadata(bs)

    real_parallel = model._forward_parallel
    with mock.patch.object(
        model, "_forward_parallel", side_effect=real_parallel, autospec=False
    ) as spy:
        # warmup builds synthetic inputs internally and calls propose().
        p.warmup(num_tokens=num_tokens, num_reqs=bs, attn_metadata=meta)
    assert spy.call_count == 1


# ---------------------------------------------------------------------------
# Increment 5 (rev 1): warmup trace-spy on the 4-layer (P-EAGLE) backbone
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bs,K", [(1, 2), (2, 4)])
def test_warmup_traces_parallel_on_4layer_backbone(bs, K):
    """Warmup routes to _forward_parallel exactly once on the 4-layer backbone.

    Same trace-spy proof as the 1-layer case, but the drafter is the P-EAGLE
    shape (1 fusion + 3 plain layers). Confirms the DESIGN-1 dispatch is
    unchanged by backbone depth: the input window is still bs*(1+K), warmup
    still hits _forward_parallel once, and the single masked pass now traverses
    all 4 layers internally (verified end-to-end since the spy uses the real
    _forward_parallel via side_effect)."""
    model, lc = _make_drafter_model(K=K, n_layers=4)
    assert len(model.model.layers) == 4
    assert model.model.layers[0].is_fusion_layer is True
    p = _make_bare_proposer(model, lc, K, parallel=True)
    num_tokens = p.draft_graph_input_tokens(bs)
    assert num_tokens == bs * (1 + K)  # depth-independent window
    meta = _decode_attn_metadata(bs, model=model)  # keys all 4 layers

    real_parallel = model._forward_parallel
    with mock.patch.object(
        model, "_forward_parallel", side_effect=real_parallel, autospec=False
    ) as spy:
        p.warmup(num_tokens=num_tokens, num_reqs=bs, attn_metadata=meta)
    assert spy.call_count == 1


@pytest.mark.parametrize("bs,K", [(1, 2), (2, 4)])
def test_synthetic_path_routes_parallel_on_4layer(bs, K):
    """Synthetic-input propose path routes to _forward_parallel and returns the
    [bs, K] contract on the 4-layer backbone."""
    model, lc = _make_drafter_model(K=K, n_layers=4)
    p = _make_bare_proposer(model, lc, K, parallel=True)
    num_tokens = p.draft_graph_input_tokens(bs)
    kwargs = p._build_synthetic_inputs(num_tokens, bs)
    assert kwargs["raw_sampled_token_ids"].shape == (bs, K + 1)
    meta = _decode_attn_metadata(bs, model=model)

    real_parallel = model._forward_parallel
    with mock.patch.object(
        model, "_forward_parallel", side_effect=real_parallel, autospec=False
    ) as spy:
        draft_token_ids, drafts_only = p.propose(
            attn_metadata=meta, is_warmup=True, **kwargs
        )
    assert spy.call_count == 1
    assert drafts_only.shape == (bs, K)
    assert drafts_only.dtype == torch.int32
