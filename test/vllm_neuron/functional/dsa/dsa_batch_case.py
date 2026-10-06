# SPDX-License-Identifier: Apache-2.0
"""One DSA/MLA layer's decode step for ``B`` requests, batched and one request at a time.

Shared by the batched-indexer tests and the hardware benchmark, so both measure and check
the same composition.

* :func:`batch_operands` builds one decode step of ``B`` requests with unequal lengths on
  the runner's side-cache layout: one latent bank of pages (each request holds its own
  shuffled pages), and two DSA banks addressed by a state slot per request -- the pooled
  keys ``[slots, rows, 128]`` and the rings ``[slots, 2, 4, 128]``.
* :func:`batched_layer` is the layer at ``B``: the projections once on ``B`` rows,
  ``Glm5NextDSAIndexer.forward_requests`` for every request's selection, and
  ``mla_decode_attention`` for every request's attention. It is wave 2's decode
  composition (``Glm5NextMLAAttention._attend_requests``) with the per-request indexer
  loop replaced by the batched one.
* :func:`per_request_layer` is the wave-1 layer, ``Glm5NextMLAAttention.forward``, once
  per request on that request's own bank views, as the runner serves ``B = 1``.
"""

from __future__ import annotations

import torch

from test.vllm_neuron.functional.dsa.dsa_decode_case import PAGE

#: Position of ``attended`` (the attention output before absorb-out) in the collector a
#: one-request ``forward`` fills: the selected indices, the latent and its rows, the
#: bank, then ``attended``.
COLLECTOR_INDICES = 0
COLLECTOR_ATTENDED = 4


def pool_rows(max_seq_len: int, pool: int) -> int:
    """Rows per slot of the pooled-key bank: every candidate plus the trash row, in pages.

    Whole pages because the one-request path reads its store through the paged gather.
    """
    return -(-(int(max_seq_len) // int(pool) + 1) // PAGE) * PAGE


def batch_operands(cfg, lengths, *, max_seq_len: int, seed: int = 7, spare_slots: int = 3,
                   spare_pages: int = 3, device=None) -> dict:
    """One decode step of ``len(lengths)`` requests; ``lengths[b]`` counts this step's token.

    Every request holds ``max_seq_len // PAGE`` table entries (``-1`` past its own pages)
    over pages no other request holds, and a state slot no other request holds; the
    spare slots and pages belong to nobody.
    """
    gen = torch.Generator().manual_seed(int(seed))
    batch = len(lengths)
    pool = int(cfg.index_kpool)
    head_dim = int(cfg.index_head_dim)
    latent = int(cfg.kv_lora_rank)
    pages = int(max_seq_len) // PAGE
    used = [-(-int(n) // PAGE) for n in lengths]
    bank_pages = sum(used) + int(spare_pages)
    order = torch.randperm(bank_pages, generator=gen).to(torch.int32)
    table = torch.full((batch, pages), -1, dtype=torch.int32)
    start = 0
    for b, n in enumerate(used):
        table[b, :n] = order[start:start + n]
        start += n
    position = torch.tensor([int(n) - 1 for n in lengths], dtype=torch.int64)
    latent_slots = torch.stack([
        table[b, int(position[b]) // PAGE].to(torch.int64) * PAGE + int(position[b]) % PAGE
        for b in range(batch)])
    slots_total = batch + int(spare_slots)
    state_slots = torch.randperm(slots_total, generator=gen)[:batch].to(torch.int32)
    rows = pool_rows(max_seq_len, pool)
    ops = {
        "hidden": (torch.randn(batch, int(cfg.hidden_size), generator=gen) * 0.5
                   ).to(torch.bfloat16),
        "latent_cache": (torch.randn(bank_pages * PAGE, 1, latent, generator=gen) * 0.5
                         ).to(torch.bfloat16),
        "pool_bank": (torch.randn(slots_total, rows, head_dim, generator=gen) * 0.5
                      ).to(torch.bfloat16),
        "tail_bank": (torch.randn(slots_total, 2, pool, head_dim, generator=gen) * 0.5
                      ).to(torch.bfloat16),
        "state_slots": state_slots,
        "seq_lens": torch.tensor([int(n) for n in lengths], dtype=torch.int32),
        "position": position,
        "block_table": table,
        "latent_slots": latent_slots,
        "softmax_scale": float(int(cfg.qk_nope_head_dim) + int(cfg.qk_rope_head_dim)) ** -0.5,
        "max_seq_len": int(max_seq_len),
    }
    if device is not None:
        ops = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in ops.items()}
    return ops


def cloned(ops: dict) -> dict:
    """The same operands with every tensor copied, so a run's in-place writes stay local."""
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in ops.items()}


def batched_layer(module, ops: dict, *, taps: dict | None = None) -> torch.Tensor:
    """The layer for every request at once. ``[B, hidden_size]``; banks written in place.

    ``taps``, when given, receives ``projected`` (the indexer's four projections),
    ``indices`` (``[B, width]``, None when nothing selects) and ``attended``
    (``[B, 1, latent]`` fp32, before absorb-out).
    """
    from vllm_neuron.functional.attention.mla_absorb import mla_absorb
    from vllm_neuron.functional.attention.mla_decode import mla_decode_attention
    from vllm_neuron.functional.dsa.decode_bypass import (
        selection_bound,
        selection_is_a_no_op,
    )

    hidden = ops["hidden"]
    indexer = module.indexer
    window = int(ops["block_table"].shape[1]) * PAGE
    bound = selection_bound(int(ops["max_seq_len"]), window)
    dense = selection_is_a_no_op(bound, indexer.index_topk, indexer.index_kpool)
    q_latent = module.project_query_latent(hidden)
    projected = indexer.project_stage(hidden, q_latent, query_side=not dense)
    indices = indexer.forward_requests(
        hidden, q_latent, ops["pool_bank"], ops["tail_bank"], ops["state_slots"],
        ops["seq_lens"], ops["position"],
        max_seq_len=bound if dense else int(ops["max_seq_len"]),
        indices_wanted=not dense, projected=projected)
    query, kv_latent = module.project_query_and_latent(hidden)
    cache = ops["latent_cache"]
    written = kv_latent.to(cache.dtype)
    cache[:, 0, :].index_copy_(0, ops["latent_slots"], written)
    q_lift = mla_absorb(query, module._absorb_weight("W_UK"))
    attended = mla_decode_attention(
        q_lift, cache[:, 0, :], ops["block_table"], ops["position"], written,
        float(ops["softmax_scale"]), PAGE, topk_indices=None if dense else indices)
    if taps is not None:
        taps["projected"] = projected
        taps["indices"] = indices
        taps["attended"] = attended
    reduced = mla_absorb(attended.to(hidden.dtype), module._absorb_weight("W_UV"))
    return module.project_output(reduced, None)


def per_request_layer(module, ops: dict, *, taps: dict | None = None,
                      projected: tuple | None = None) -> torch.Tensor:
    """The wave-1 layer once per request, on its own bank views. ``[B, hidden_size]``.

    ``taps``, when given, receives ``indices`` (one ``[1, width]`` row per request) and
    ``attended`` (one ``[1, 1, latent]`` row per request), read from each forward's own
    collector.

    ``projected`` (the batched run's ``taps["projected"]``) serves row ``b`` of the
    indexer's projections to request ``b``'s indexer instead of projecting one row: the
    projection kernels round one row and ``B`` rows differently in their last fp32 bit,
    and this compares the indexer and the attention on the same operands.
    """
    outs, indices, attended = [], [], []
    indexer = module.indexer
    for b in range(int(ops["hidden"].shape[0])):
        slot = int(ops["state_slots"][b])
        collector: list = []
        if projected is not None:
            rows = tuple(None if part is None else part[b:b + 1] for part in projected)
            indexer.project_stage = lambda *_a, _rows=rows, **_k: _rows
        try:
            out = module(
                ops["hidden"][b:b + 1],
                latent_cache=ops["latent_cache"],
                pool_cache=ops["pool_bank"][slot],
                seq_lens=ops["seq_lens"][b:b + 1],
                start_position=ops["position"][b],
                softmax_scale=float(ops["softmax_scale"]),
                max_seq_len=int(ops["max_seq_len"]),
                page_size=PAGE,
                block_table_row=ops["block_table"][b].reshape(-1, 1),
                latent_slots=ops["latent_slots"][b:b + 1],
                tail=ops["tail_bank"][slot],
                position=ops["position"][b],
                collector=collector,
            )
        finally:
            if projected is not None:
                del indexer.project_stage
        outs.append(out)
        indices.append(collector[COLLECTOR_INDICES])
        attended.append(collector[COLLECTOR_ATTENDED])
    if taps is not None:
        taps["indices"] = indices
        taps["attended"] = attended
    return torch.cat(outs)
