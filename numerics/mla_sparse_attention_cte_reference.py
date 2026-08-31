# SPDX-License-Identifier: Apache-2.0
"""Plain-torch reference for the ``mla_sparse_attention_cte`` numerics declaration.

Triad LD-76 (port-plan §19.2 Amendment 11; port-assessment §17.4). Lives in the
top-level ``numerics/`` namespace package, OUTSIDE ``vllm_neuron``. Prereg:
TRIADS-Z0-PREREGISTRATION.md §D6 (sealed 2026-08-31, sha 5f5006b9…),
tolerances fixed there BEFORE any measurement.

Scope: EXACTLY the declared harness case family ``c1-core`` — 3-D contiguous
caches ``[1, slots, 512]`` in a plain (non-fp8, unscaled) storage dtype,
``positions=None`` (pos = arange(T) causal prefill), ``seq_ids=None`` (flat
single-sequence addressing), ``compress_ratio=0`` (topk indices are raw row
ids, no causal cap on the compressed leg), no current-row operands, no
chunking. Anything else raises loudly.

Math (independent spelling of dsv4_ref/kernel.py:328-350 over the reference's
single concatenated index list, dsv4_ref/model.py:520): per query token,
gather the sliding-window band ``clamp(pos-window+1, 0) + arange(window)``
(entries > pos absent) from the SWA cache and the top-k rows (−1 = absent)
from the compressed cache; ONE fp32 softmax over [window ++ topk] with the
per-head sink logit in the DENOMINATOR only; gathered rows are key AND value;
single cast to ``out_dtype`` at the end. Fully-masked queries return 0.
"""

import torch

MASK_NEG = -1.0e30
LATENT_DIM = 512


def mla_sparse_attention_cte_reference(
    q: torch.Tensor,
    compressed_k_cache: torch.Tensor,
    swa_k_cache: torch.Tensor,
    topk_indices: torch.Tensor,
    attn_sink,
    scale: float = None,
    window: int = None,
    *,
    compress_ratio: int = 0,
    **unused_kwargs,
) -> torch.Tensor:
    """Reference output for the declared CTE cases.

    Args mirror the op's call form for the declared cases: five positional
    tensors, then scalar kwargs.
    """
    assert q.dim() == 3 and q.shape[-1] == LATENT_DIM, (
        f"reference domain: q [T, H, 512]; got {tuple(q.shape)}"
    )
    assert compressed_k_cache.dim() == 3 and compressed_k_cache.shape[0] == 1, (
        "reference domain: 3-D contiguous compressed cache [1, slots, 512]"
    )
    assert swa_k_cache.dim() == 3 and swa_k_cache.shape[0] == 1, (
        "reference domain: 3-D contiguous swa cache [1, slots, 512]"
    )
    assert compress_ratio == 0, "reference domain: compress_ratio == 0"
    assert scale is not None and window is not None, "scale/window required"
    assert not unused_kwargs, f"out-of-domain kwargs: {sorted(unused_kwargs)}"

    num_tokens, num_heads, head_dim = q.shape
    swa_flat = swa_k_cache[0].to(torch.float32)  # [slots, 512]
    comp_flat = compressed_k_cache[0].to(torch.float32)  # [slots, 512]
    pos = torch.arange(num_tokens, dtype=torch.int64)

    # Sliding-window band (dsv4_ref/model.py:268-270).
    offsets = torch.arange(window, dtype=torch.int64).unsqueeze(0)
    win_idx = (pos.reshape(-1, 1) - window + 1).clamp_min(0) + offsets
    win_valid = win_idx <= pos.reshape(-1, 1)

    # Top-k rows: -1 marks absent; compress_ratio == 0 → raw row ids, no
    # causal cap (the indexer owns causality at ratio 0).
    comp_idx = topk_indices.reshape(num_tokens, -1).to(torch.int64)
    comp_valid = comp_idx >= 0

    idx = torch.cat((win_idx, comp_idx), dim=1)  # [T, W+K]
    valid = torch.cat((win_valid, comp_valid), dim=1)
    safe = torch.where(idx < 0, torch.zeros_like(idx), idx)

    win_rows = swa_flat[safe[:, :window].reshape(-1)].view(
        num_tokens, window, LATENT_DIM
    )
    comp_rows = comp_flat[
        safe[:, window:].clamp(0, comp_flat.shape[0] - 1).reshape(-1)
    ].view(num_tokens, -1, LATENT_DIM)
    latent = torch.cat((win_rows, comp_rows), dim=1)  # [T, W+K, 512]

    scores = torch.bmm(q.to(torch.float32), latent.transpose(1, 2)) * scale
    scores = scores + torch.where(
        valid,
        torch.zeros((), dtype=torch.float32),
        torch.full((), MASK_NEG, dtype=torch.float32),
    ).unsqueeze(1)

    row_max = scores.max(dim=-1, keepdim=True).values
    probs = torch.exp(scores - row_max)
    denom = probs.sum(dim=-1, keepdim=True)
    if attn_sink is not None:
        sink_f = attn_sink.reshape(-1)[:num_heads].reshape(1, num_heads, 1)
        denom = denom + torch.exp(sink_f.to(torch.float32) - row_max)
    keep = valid.any(dim=-1, keepdim=True).to(torch.float32).unsqueeze(-1)
    out = (torch.bmm(probs, latent) / denom) * keep
    return out.to(torch.bfloat16)
