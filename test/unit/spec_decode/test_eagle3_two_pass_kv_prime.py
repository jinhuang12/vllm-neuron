# SPDX-License-Identifier: Apache-2.0
"""Increment 8 (P-EAGLE two-pass parallel drafting KV-prime) CPU tests.

Background (benefit.md 7.7 defect): the single-pass parallel branch ran the
drafter over ONLY ``bs*K`` slots seeded from the bonus token, so the full
prompt (prefill) and the interior accepted decode tokens NEVER got drafter KV
written from real per-token hidden states. The approved fix (design.md rev 2)
adds a KV-prime pass BEFORE ``_forward_parallel``: it runs the full incoming
window through ``self.model(...)`` exactly like the sequential path, discards
the hidden-state outputs, and keeps only the in-kernel KV writes from real
hidden states.

These CPU-only tests assert the FIXED behaviour (the NKI attention kernel falls
back to torch under VLLM_NEURON_CPU_MODE, and the fallback writes the paged KV
caches in place via ``index_put_``, so the bound caches are directly
inspectable):

  (i)   Two-step decode KV lifecycle (AL>1): interior accepted positions hold
        mask-derived KV after step t, then step t+1's KV-prime pass OVERWRITES
        them with real-hidden-derived KV. Proves interior accepted tokens no
        longer keep mask_hidden-derived KV forever.
  (ii)  Prefill KV: a multi-token prompt window writes drafter KV for ALL
        prompt positions from real hiddens (not just the K seed slots).
  (iii) Rejected-position pollution: the KV-prime pass writes KV at positions
        >= the next seed p'; assert (a) pass 2's decode attention mask does NOT
        attend those positions (iota < min_pos == p'), and (b) a subsequent
        step's KV-prime pass overwrites them.

All follow the inc-3 ``test_eagle3_parallel_forward.py`` CPU bootstrap.
"""

import os

os.environ.setdefault("VLLM_NEURON_CPU_MODE", "1")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "12498")

import pytest
import torch

BLOCK_SIZE = 128
PTD = 201020
START_LAYER = 1


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


def _make_drafter(hidden=16, n_heads=4, n_kv=2, head_dim=4, vocab=32, K=3, n_layers=1):
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
    model.parallel_drafting = True
    model.ptd_token_id = PTD
    torch.manual_seed(0)
    with torch.no_grad():
        for p in model.parameters():
            if not p.is_meta:
                p.copy_(torch.randn_like(p) * 0.05)
    torch.manual_seed(7)
    model.register_buffer(
        "mask_hidden", torch.randn(1, lc.hidden_size * 3) * 3.0, persistent=False
    )
    return model, lc


def _make_caches(model, num_blocks=8):
    caches = {}
    for layer in model.model.layers:
        attn = layer.self_attn
        k = torch.zeros(
            num_blocks, attn.num_key_value_heads_per_rank, BLOCK_SIZE, attn.head_dim
        )
        v = torch.zeros_like(k)
        caches[f"layers.{layer.layer_idx}.self_attn"] = [k, v]
    return caches


def _fusion_layer_name(model):
    return f"layers.{model.model.layers[0].layer_idx}.self_attn"


def _decode_meta(model, positions, block_id=0):
    """Decode-mode incoming metadata over ``len(positions)`` tokens, bs=1.

    max_query_len == decode_token_threshold == T so ``is_prefill`` is False;
    the window routes through forward_decode (update_cache=True path).
    """
    T = positions.shape[0]
    block_table = torch.tensor([[block_id]], dtype=torch.int32)
    slot = (block_table.view(-1)[0] * BLOCK_SIZE + positions % BLOCK_SIZE).to(
        torch.int64
    )
    meta = {}
    for layer in model.model.layers:
        meta[f"layers.{layer.layer_idx}.self_attn"] = {
            "block_table_tensor": block_table,
            "slot_mapping": slot,
            "max_query_len": T,
            "block_size": BLOCK_SIZE,
            "max_blocks_per_seq": 1,
            "decode_token_threshold": T,
        }
    return meta


def _prefill_meta(model, positions, block_id=0):
    """Prefill-mode incoming metadata: max_query_len > decode_token_threshold."""
    T = positions.shape[0]
    block_table = torch.tensor([[block_id]], dtype=torch.int32)
    slot = (block_table.view(-1)[0] * BLOCK_SIZE + positions % BLOCK_SIZE).to(
        torch.int64
    )
    meta = {}
    for layer in model.model.layers:
        meta[f"layers.{layer.layer_idx}.self_attn"] = {
            "block_table_tensor": block_table,
            "slot_mapping": slot,
            "max_query_len": T,
            "block_size": BLOCK_SIZE,
            "max_blocks_per_seq": 1,
            "decode_token_threshold": 1,
        }
    return meta


def _kv_at(caches, layer_name, position, block_id=0):
    """Return (k, v) cache contents at a sequential position for a block, as
    flattened [heads*head_dim] tensors."""
    k, v = caches[layer_name]
    slot = position % BLOCK_SIZE
    k_out = k[block_id, :, slot, :].reshape(-1).clone()
    v_out = v[block_id, :, slot, :].reshape(-1).clone()
    return k_out, v_out


def _forward(model, caches, input_ids, positions, sampling_positions, meta):
    model.bind_kv_cache(caches)
    rank = torch.tensor(0, dtype=torch.int32)
    hidden = model.config.hidden_size
    tgt_hidden = torch.randn(positions.shape[0], hidden * 3)
    out = model(
        input_ids=input_ids.to(torch.int32),
        positions=positions,
        initial_target_hidden_states=tgt_hidden,
        attn_metadata=meta,
        sampling_positions=sampling_positions,
        rank=rank,
    )
    return out, tgt_hidden


# --------------------------------------------------------------------------
# (i) Two-step decode KV lifecycle, AL > 1
# --------------------------------------------------------------------------


def test_two_step_interior_accepted_kv_is_real_after_fix():
    """Interior accepted decode tokens get real-hidden KV, not stuck on mask.

    Step t: seed at position p=10, K=3. _forward_parallel writes K slots at
    positions 10,11,12; slot 0 (pos 10) from the real seed hidden, slots 1..2
    (pos 11,12) from mask_hidden. So after step t the drafter KV at the
    INTERIOR positions 11,12 is MASK-derived.

    Step t+1: the target accepted tokens 11,12 (AL>1). The incoming decode
    window covers positions [11,12,13] (the accepted run + new seed). The
    KV-prime pass (pass 1) runs this full window through the backbone and
    OVERWRITES positions 11,12 with real-hidden-derived KV.

    Assert: KV at interior position 11 CHANGES between step t (mask-derived)
    and step t+1 (real-derived) -- proving the fix rewrites interior accepted
    KV from real hidden states. Under the defect (pass 2 only) step t+1 would
    touch only positions 13,14,15, leaving 11,12 frozen at their step-t
    mask-derived values.
    """
    model, lc = _make_drafter(K=3)
    ln = _fusion_layer_name(model)
    caches = _make_caches(model)

    torch.manual_seed(100)
    # Step t: seed at pos 10 (single decode token window).
    p = 10
    pos_t = torch.tensor([p], dtype=torch.long)
    ids_t = torch.tensor([5], dtype=torch.int32)
    sp_t = torch.tensor([0], dtype=torch.long)
    _forward(model, caches, ids_t, pos_t, sp_t, _decode_meta(model, pos_t))

    # After step t: pass 2 wrote K=3 slots at positions 10,11,12.
    k_interior_step_t, v_interior_step_t = _kv_at(caches, ln, 11)
    assert k_interior_step_t.abs().sum() > 0, (
        "step t should have written K-slot KV at interior position 11"
    )

    # Step t+1: accepted run 11,12 + new seed 13 -> decode window [11,12,13].
    torch.manual_seed(200)
    pos_t1 = torch.tensor([11, 12, 13], dtype=torch.long)
    ids_t1 = torch.tensor([6, 7, 8], dtype=torch.int32)
    sp_t1 = torch.tensor([2], dtype=torch.long)  # seed = last accepted (pos 13)
    _forward(model, caches, ids_t1, pos_t1, sp_t1, _decode_meta(model, pos_t1))

    k_interior_step_t1, v_interior_step_t1 = _kv_at(caches, ln, 11)

    # The KV-prime pass overwrote interior position 11 from real hiddens; it
    # must differ from the step-t mask-derived KV.
    assert not torch.allclose(k_interior_step_t, k_interior_step_t1), (
        "interior accepted position 11 K-cache must be rewritten from real "
        "hidden states on step t+1 (was frozen at mask-derived under the defect)"
    )


def test_two_step_interior_kv_matches_independent_real_projection():
    """The rewritten interior KV equals an independent real-window backbone run.

    Cross-check for (i): run the SAME step-t+1 window through the backbone
    alone (the KV-prime pass in isolation) into a fresh cache; the two-pass
    forward's interior KV must match it bit-for-bit, confirming the interior
    KV derives from the real per-token hidden states written by pass 1 (pass 2
    never touches positions < seed)."""
    model, lc = _make_drafter(K=3)
    ln = _fusion_layer_name(model)

    torch.manual_seed(321)
    pos = torch.tensor([11, 12, 13], dtype=torch.long)
    ids = torch.tensor([6, 7, 8], dtype=torch.int32)
    sp = torch.tensor([2], dtype=torch.long)
    tgt_hidden = torch.randn(pos.shape[0], lc.hidden_size * 3)
    rank = torch.tensor(0, dtype=torch.int32)
    meta = _decode_meta(model, pos)

    # Full two-pass forward.
    caches_full = _make_caches(model)
    model.bind_kv_cache(caches_full)
    model(
        input_ids=ids,
        positions=pos,
        initial_target_hidden_states=tgt_hidden,
        attn_metadata=meta,
        sampling_positions=sp,
        rank=rank,
    )
    k_full, v_full = _kv_at(caches_full, ln, 11)

    # KV-prime pass in isolation: the backbone over the same window with the
    # already fc-combined hiddens (exactly what forward() feeds pass 1).
    caches_prime = _make_caches(model)
    model.bind_kv_cache(caches_prime)
    combined = model.model.combine_hidden_states(tgt_hidden)
    model.model(
        input_ids=ids,
        positions=pos,
        target_hidden_states=combined,
        attn_metadata=meta,
        rank=rank,
    )
    k_prime, v_prime = _kv_at(caches_prime, ln, 11)

    assert torch.allclose(k_full, k_prime, atol=1e-5), (
        "two-pass interior K-cache must equal the isolated real-window "
        "KV-prime write"
    )
    assert torch.allclose(v_full, v_prime, atol=1e-5)
    assert k_prime.abs().sum() > 0


# --------------------------------------------------------------------------
# (ii) Prefill: drafter KV written for ALL prompt positions
# --------------------------------------------------------------------------


def test_prefill_writes_kv_for_all_prompt_positions():
    """A multi-token prompt window writes drafter KV at every prompt position.

    Under the defect the parallel branch early-returned into pass 2, which only
    ever writes the K seed slots (positions p..p+K-1). With the KV-prime pass a
    T-token prompt writes real-hidden KV at all T positions 0..T-1."""
    model, lc = _make_drafter(K=3)
    ln = _fusion_layer_name(model)
    caches = _make_caches(model)

    T = 6
    torch.manual_seed(11)
    pos = torch.arange(T, dtype=torch.long)
    ids = torch.arange(1, T + 1, dtype=torch.int32)
    sp = torch.tensor([T - 1], dtype=torch.long)  # seed at last prompt token
    _forward(model, caches, ids, pos, sp, _prefill_meta(model, pos))

    # Every prompt position 0..T-1 must have non-zero drafter KV.
    for position in range(T):
        k, v = _kv_at(caches, ln, position)
        assert k.abs().sum() > 0, (
            f"prompt position {position} K-cache must be written from real "
            f"hidden states (only K seed slots were written under the defect)"
        )
        assert v.abs().sum() > 0, f"prompt position {position} V-cache unwritten"

    # A position beyond BOTH the prompt window AND pass-2's K seed slots
    # (seed at T-1, slots T-1..T-1+K-1) is untouched.
    k_beyond, _ = _kv_at(caches, ln, T - 1 + model.num_speculative_tokens + 10)
    assert k_beyond.abs().sum() == 0


# --------------------------------------------------------------------------
# (iii) Rejected-position pollution: masked out + overwritten next step
# --------------------------------------------------------------------------


def test_pass2_mask_excludes_kv_prime_written_positions():
    """Pass 2 decode attention does NOT attend positions the KV-prime pass
    wrote at >= the next seed p'.

    The KV-prime pass writes KV at all incoming-window positions, including
    positions >= p' (rejected / seed-region tokens). Pass 2 builds a decode
    mask with min_pos == p' (its query positions are p'..p'+K-1), so the prior
    region admits only positions < p' (iota < min_pos). Assert the shuffled
    mask is 0 at the linear indices for sequential positions p' and p'+1, and
    1 at p'-1 -- i.e. pass-1's writes at >= p' are excluded from pass-2 attn."""
    from vllm_neuron.functional.attention.attention_decode_mask import (
        _build_seq_to_linear_map,
        _resize_block_len,
        gen_attention_decode_mask,
    )

    K = 3
    p_prime = 20
    s_prior = BLOCK_SIZE  # one block, divisible by P_MAX=128
    block_len = BLOCK_SIZE
    # The torch impl resizes block_len internally; use the SAME resized value
    # (and the same seq->linear map it uses) to locate sequential positions in
    # the shuffled [s_prior] output layout.
    resized_block_len = _resize_block_len(block_len, 1, 1, K, s_prior)
    seq_to_linear = _build_seq_to_linear_map(s_prior, resized_block_len)

    # Pass-2 query positions: p'..p'+K-1 (mirrors _forward_parallel).
    q_positions = torch.arange(p_prime, p_prime + K, dtype=torch.float32).view(1, K)
    mask = gen_attention_decode_mask(
        pos_ids=q_positions,
        bs=1,
        q_head=1,
        s_active=K,
        s_prior=s_prior,
        start_pos=None,
        block_len=block_len,
    )  # [s_prior, bs, q_head, s_active]

    # Prior region (non-active slots): attended iff seq_pos < min_pos == p'.
    # Positions p' and p'+1 (written by KV-prime) must be masked OUT.
    for seq_pos, expect in [(p_prime - 1, 1.0), (p_prime, 0.0), (p_prime + 1, 0.0)]:
        lin = int(seq_to_linear[seq_pos])
        val = mask[lin, 0, 0, :]  # across all K active queries
        assert torch.all(val == expect), (
            f"prior seq_pos {seq_pos}: expected mask {expect}, got {val.tolist()} "
            f"(pass-2 must not attend KV-prime writes at >= p'={p_prime})"
        )


def test_next_step_kv_prime_overwrites_rejected_positions():
    """A subsequent step's KV-prime pass overwrites the >= p' positions.

    Step t writes KV at positions 20,21,22 (K=3 slots at seed p'=20). Some of
    these correspond to rejected drafts. Step t+1 re-verifies with a window
    covering positions 21,22,23; its KV-prime pass overwrites 21,22 with fresh
    real-hidden KV. Assert the cache at position 21 changes between steps."""
    model, lc = _make_drafter(K=3)
    ln = _fusion_layer_name(model)
    caches = _make_caches(model)

    torch.manual_seed(400)
    pos_t = torch.tensor([20], dtype=torch.long)
    ids_t = torch.tensor([9], dtype=torch.int32)
    sp_t = torch.tensor([0], dtype=torch.long)
    _forward(model, caches, ids_t, pos_t, sp_t, _decode_meta(model, pos_t))
    k_21_t, _ = _kv_at(caches, ln, 21)
    assert k_21_t.abs().sum() > 0  # written by pass 2 (mask-derived)

    torch.manual_seed(500)
    pos_t1 = torch.tensor([21, 22, 23], dtype=torch.long)
    ids_t1 = torch.tensor([3, 4, 5], dtype=torch.int32)
    sp_t1 = torch.tensor([2], dtype=torch.long)
    _forward(model, caches, ids_t1, pos_t1, sp_t1, _decode_meta(model, pos_t1))
    k_21_t1, _ = _kv_at(caches, ln, 21)

    assert not torch.allclose(k_21_t, k_21_t1), (
        "position 21 KV must be overwritten by step t+1 KV-prime pass"
    )
