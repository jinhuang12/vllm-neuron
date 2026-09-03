# SPDX-License-Identifier: Apache-2.0
"""Plain-torch reference for the ``mla_decode_tkg`` numerics declaration.

Triad LD-75 (port-plan §19.2 Amendment 11; port-assessment §17.4). Lives in the
top-level ``numerics/`` namespace package, OUTSIDE ``vllm_neuron`` — the
harness (scripts/triad_numerics.py) refuses a reference resolving inside the
port package. Prereg: TRIADS-Z0-PREREGISTRATION.md §D6 (sealed 2026-08-31,
sha 5f5006b9…), tolerances fixed there BEFORE any measurement.

Scope: EXACTLY the declared harness case family ``d1-swa-warm`` —
3-D contiguous caches ``[B, slots, 512]`` in a plain (non-fp8, unscaled)
storage dtype, ``positions=None``/``context_lens=None`` (warm-cache window:
the whole window is populated, plain arange addressing), ``topk_indices=None``
(dense compressed pool scan), ``compress_ratio=0`` (no causal cap), no
current-row operands, no cache write. Anything else raises loudly.

Math (independent spelling of dsv4_ref/kernel.py:328-350): one fp32 softmax
over the concatenated [window ++ compressed-pool] latents, the gathered latent
rows serving as key AND value (absorbed MLA), optional per-head sink logit in
the DENOMINATOR only, single cast to ``out_dtype`` at the end.
"""

import torch

# Pinned as literals per the numerics convention (R-24-adjacent: the reference
# must not import the port package for its constants).
MASK_NEG = -1.0e30
LATENT_DIM = 512


def mla_decode_tkg_reference(
    q: torch.Tensor,
    latent_cache: torch.Tensor,
    swa_cache: torch.Tensor,
    scale: float = None,
    sink=None,
    *,
    window: int = 128,
    compress_ratio: int = 0,
    **unused_kwargs,
) -> torch.Tensor:
    """Reference output for the declared decode cases.

    Args mirror the op's call form for the declared cases: three positional
    tensors (q, latent_cache, swa_cache), then scalar kwargs.
    """
    assert q.dim() == 3 and q.shape[-1] == LATENT_DIM, (
        f"reference domain: q [B, H, 512]; got {tuple(q.shape)}"
    )
    assert latent_cache.dim() == 3 and swa_cache.dim() == 3, (
        "reference domain: 3-D contiguous caches [B, slots, 512]"
    )
    assert compress_ratio == 0, "reference domain: compress_ratio == 0"
    assert scale is not None, "scale is required"
    assert not unused_kwargs, f"out-of-domain kwargs: {sorted(unused_kwargs)}"

    batch, num_heads, head_dim = q.shape
    slots_w = swa_cache.shape[1]
    slots_c = latent_cache.shape[1]
    assert slots_w >= window, (
        "warm-cache window addressing needs slots >= window "
        f"(slots {slots_w}, window {window})"
    )

    # Warm-cache window: plain arange over the first `window` slots per
    # sequence (mla_decode positions=None branch). Compressed pool: the whole
    # per-sequence pool, all valid (topk_indices=None, compress_ratio=0).
    win = swa_cache[:, :window, :].to(torch.float32)  # [B, W, 512]
    comp = latent_cache[:, :slots_c, :].to(torch.float32)  # [B, S, 512]
    latent = torch.cat((win, comp), dim=1)  # [B, W+S, 512]

    # fp32 scores, explicit max/exp/sum/divide spelling (kernel.py:332-348).
    scores = torch.bmm(q.to(torch.float32), latent.transpose(1, 2)) * scale
    row_max = scores.max(dim=-1, keepdim=True).values
    probs = torch.exp(scores - row_max)
    denom = probs.sum(dim=-1, keepdim=True)
    if sink is not None:
        sink_f = sink.reshape(1, num_heads, 1).to(torch.float32)
        denom = denom + torch.exp(sink_f - row_max)
    out = torch.bmm(probs, latent) / denom  # every slot valid: keep == 1
    return out.to(torch.bfloat16)
