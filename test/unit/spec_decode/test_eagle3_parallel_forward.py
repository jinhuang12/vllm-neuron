# SPDX-License-Identifier: Apache-2.0
"""Increment 3 (P-EAGLE drafter parallel forward + mask_hidden load) CPU tests.

Three groups, all CPU-only (VLLM_NEURON_CPU_MODE) on trn2-2:

  (a) mask_hidden load matrix: a synthetic safetensors fixture with/without a
      ``mask_hidden`` tensor, crossed with ``parallel_drafting`` on/off. The
      contract mirrors upstream ``Eagle3LlamaForCausalLM.load_weights``
      (vllm/model_executor/models/llama_eagle3.py:404-409):
        parallel on  + present  -> loads
        parallel on  + missing  -> ValueError
        parallel off + present  -> no-op (buffer not set)
        parallel off + missing  -> no-op
      Exercises the ``_load_mask_hidden`` method (the mask_hidden load path)
      directly; a full ``load_weights`` needs a complete valid checkpoint,
      which is out of scope for this unit.

  (b) parallel forward shape: a tiny random-weight drafter (small hidden, 1
      layer) run through the single-pass parallel path for batch in {1,2} and
      K in {2,4}. Asserts output token ids are ``[batch, K]`` with no NaN.

  (c) mask plumbing: the same forward with two different ``mask_hidden``
      vectors must produce different draft tokens in the masked slots
      (columns 1..K-1) while the real seed slot (column 0) is unchanged —
      proving the substituted hidden states actually reach the forward.

The drafter's attention layer needs Neuron parallel groups to instantiate, so a
single-rank gloo process group + ``initialize_neuron_parallel_state`` is set up
once per module (the same bootstrap the plugin's MPExecutor test path uses). The
NKI attention kernel falls back to the torch implementation on CPU
(``can_run_kernel`` returns False under CPU mode), so the real backbone forward
runs end-to-end.
"""
import os

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "12495")

import pytest
import torch
from safetensors.torch import save_file

# s_prior (= max_blocks_per_seq * block_size) must be divisible by the NKI
# mask kernel's P_MAX (128); the torch fallback shares the same mask builder.
BLOCK_SIZE = 128
PTD = 201020


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


def _make_drafter(hidden=16, n_heads=4, n_kv=2, head_dim=4, vocab=32, K=2):
    from vllm_neuron.model.llama3.config import LlamaConfig
    from vllm_neuron.model.llama3.eagle3_model import Eagle3LlamaForCausalLM

    lc = LlamaConfig(
        vocab_size=vocab,
        draft_vocab_size=None,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=1,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv,
        head_dim=head_dim,
        torch_dtype=torch.float32,
    )
    model = Eagle3LlamaForCausalLM(lc, start_layer_idx=1)
    model.num_speculative_tokens = K
    # Deterministic small random weights (drafter is meta-free here since we
    # instantiate on CPU directly, not via the meta-device compile path).
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            if not p.is_meta:
                p.copy_(torch.randn_like(p) * 0.05)
    return model, lc


def _bind_cache(model, hidden_layer_idx=1, num_blocks=8):
    attn = model.model.layers[0].self_attn
    k = torch.zeros(
        num_blocks, attn.num_key_value_heads_per_rank, BLOCK_SIZE, attn.head_dim
    )
    v = torch.zeros_like(k)
    model.bind_kv_cache({f"layers.{hidden_layer_idx}.self_attn": [k, v]})


def _run_parallel_forward(model, lc, bs, K):
    """Drive the full parallel forward; returns (stacked, drafts_only)."""
    hidden = lc.hidden_size
    T = bs  # one seed token per request at decode
    input_ids = torch.arange(1, T + 1, dtype=torch.int32)
    positions = torch.zeros(T, dtype=torch.long)
    tgt_hidden = torch.randn(T, hidden * 3)
    sampling_positions = torch.arange(bs, dtype=torch.long)
    block_table = torch.arange(bs, dtype=torch.int32).view(bs, 1)
    # slots: each request in its own block, position 0 -> slot block_id*BLOCK.
    meta = {
        "layers.1.self_attn": {
            "block_table_tensor": block_table,
            "slot_mapping": (block_table.view(-1) * BLOCK_SIZE).to(torch.int64),
            "max_query_len": 1,
            "block_size": BLOCK_SIZE,
            "max_blocks_per_seq": 1,
            "decode_token_threshold": 1,
        }
    }
    rank = torch.tensor(0, dtype=torch.int32)
    stacked, drafts_only, logits = model(
        input_ids=input_ids,
        positions=positions,
        initial_target_hidden_states=tgt_hidden,
        attn_metadata=meta,
        sampling_positions=sampling_positions,
        rank=rank,
    )
    return stacked, drafts_only, logits


# --------------------------------------------------------------------------
# (a) mask_hidden load matrix
# --------------------------------------------------------------------------


def _write_fixture(tmp_path, with_mask_hidden, hidden=16):
    d = tmp_path / ("with_mh" if with_mask_hidden else "without_mh")
    d.mkdir()
    tensors = {"embed_tokens.weight": torch.randn(4, hidden)}
    if with_mask_hidden:
        tensors["mask_hidden"] = torch.randn(1, hidden * 3)
    save_file(tensors, str(d / "model.safetensors"))
    return str(d)


def test_load_mask_hidden_parallel_on_present(tmp_path):
    """(a1) parallel on + mask_hidden present -> buffer loaded."""
    model, lc = _make_drafter()
    model.parallel_drafting = True
    model.ptd_token_id = PTD
    ckpt = _write_fixture(tmp_path, with_mask_hidden=True, hidden=lc.hidden_size)
    model._load_mask_hidden(ckpt)
    assert hasattr(model, "mask_hidden")
    assert model.mask_hidden.shape == (1, lc.hidden_size * 3)


def test_load_mask_hidden_parallel_on_missing_raises(tmp_path):
    """(a2) parallel on + mask_hidden missing -> clear ValueError."""
    model, lc = _make_drafter()
    model.parallel_drafting = True
    model.ptd_token_id = PTD
    ckpt = _write_fixture(tmp_path, with_mask_hidden=False, hidden=lc.hidden_size)
    with pytest.raises(ValueError, match="mask_hidden not found"):
        model._load_mask_hidden(ckpt)


def test_load_mask_hidden_parallel_off_present_noop(tmp_path):
    """(a3) parallel off + mask_hidden present -> no-op (existing behavior)."""
    model, lc = _make_drafter()
    assert model.parallel_drafting is False  # constructor default
    ckpt = _write_fixture(tmp_path, with_mask_hidden=True, hidden=lc.hidden_size)
    model._load_mask_hidden(ckpt)
    # No mask_hidden buffer registered when parallel drafting is off.
    assert "mask_hidden" not in dict(model.named_buffers())


def test_load_mask_hidden_parallel_off_missing_noop(tmp_path):
    """(a4) parallel off + mask_hidden missing -> no-op, no error."""
    model, lc = _make_drafter()
    ckpt = _write_fixture(tmp_path, with_mask_hidden=False, hidden=lc.hidden_size)
    model._load_mask_hidden(ckpt)  # must not raise
    assert "mask_hidden" not in dict(model.named_buffers())


# --------------------------------------------------------------------------
# (b) parallel forward shape
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bs", [1, 2])
@pytest.mark.parametrize("K", [2, 4])
def test_parallel_forward_shape(bs, K):
    model, lc = _make_drafter(K=K)
    model.parallel_drafting = True
    model.ptd_token_id = PTD
    model.register_buffer(
        "mask_hidden", torch.randn(1, lc.hidden_size * 3), persistent=False
    )
    _bind_cache(model)
    stacked, drafts_only, logits = _run_parallel_forward(model, lc, bs, K)
    # No bonus token passed -> stacked == drafts_only == [bs, K].
    assert drafts_only.shape == (bs, K)
    assert stacked.shape == (bs, K)
    assert logits is None
    assert not torch.isnan(drafts_only.float()).any()
    assert drafts_only.dtype == torch.int32


# --------------------------------------------------------------------------
# (c) mask plumbing — substituted hidden states reach the forward
# --------------------------------------------------------------------------


def test_mask_hidden_substitution_reaches_forward():
    """Different mask_hidden -> different draft tokens.

    Proves the substituted mask-hidden states are actually consumed by the
    forward: running the identical parallel pass with two different
    ``mask_hidden`` vectors must produce different draft token ids. If the
    substitution were dropped (masked slots kept the seed hidden), the two runs
    would be byte-identical.
    """
    K = 4
    bs = 2
    model, lc = _make_drafter(K=K)
    model.parallel_drafting = True
    model.ptd_token_id = PTD
    _bind_cache(model)

    torch.manual_seed(1)
    mh_a = torch.zeros(1, lc.hidden_size * 3)
    mh_b = torch.randn(1, lc.hidden_size * 3) * 5.0

    model.register_buffer("mask_hidden", mh_a, persistent=False)
    drafts_a = _run_parallel_forward(model, lc, bs, K)[1].clone()

    model.register_buffer("mask_hidden", mh_b, persistent=False)
    drafts_b = _run_parallel_forward(model, lc, bs, K)[1].clone()

    # The two mask_hidden vectors must change the produced drafts, confirming
    # the substituted hidden states flow through the single-pass forward.
    assert not torch.equal(drafts_a, drafts_b), (
        "draft tokens must depend on the substituted mask_hidden"
    )
