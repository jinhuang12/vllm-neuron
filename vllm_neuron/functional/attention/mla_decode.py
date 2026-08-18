# SPDX-License-Identifier: Apache-2.0
"""DSv4-Flash MLA attention for the single-token generation (decode) path.

WHY torch and not a kernel: the plan's decode A/B against
``core.attention.attention_tkg`` is excluded by that kernel's hard
``kernel_assert(d_head <= 128)`` (the MLA latent is 512 wide), and no
DSv4-specific nkilib kernel exists in the installed wheel (interface contract
§0). This ladder row therefore takes its recorded same-row torch-composition
rung. Traceable static-shape torch only: python loops over statically known
counts, ``torch.gather``/``index_select`` gathers, no ``.item()``, no
``.tolist()``, no ``nonzero()``, no boolean-mask indexing, no data-dependent
shapes, no python ``if`` on tensor values.

The decode path is the SAME math as the prefill path — DeepSeek's reference
calls one ``sparse_attn`` for both and only changes how the index list is built
(``dsv4_ref/model.py:523-538``). So this module reuses the gather + sink-softmax
core from ``mla_sparse_attention``; keeping one core is deliberate, because
prefill/decode numeric drift is the classic way a sink-augmented softmax goes
wrong.

Reference evidence for the decode specifics:
  * ``dsv4_ref/model.py:535`` — the sliding window is a RING buffer of length
    ``window`` written at ``start_pos % window``; ``:262-264`` reads it back as
    a rotated arange, i.e. the same ``window`` slots as the prefill band.
  * ``dsv4_ref/model.py:538`` — decode attends over
    ``self.kv_cache[:bsz]``, the single tensor holding the window region
    ``[:window]`` and the compressed region ``[window:]``
    (``:479-480``, ``:497``).
  * ``dsv4_ref/model.py:277`` — with no indexer (C128 layers) the compressed
    index list is the plain causal arange ``arange((start_pos + 1) // ratio)``;
    with the indexer (C4) it is the top-k of ``:433-438``.
  * ``dsv4_ref/kernel.py:308-350`` — fp32 softmax statistics, per-head sink
    added to the DENOMINATOR only after the key loop, bf16 store.
"""

from typing import List, Optional, Sequence

import torch
from torch import Tensor

from vllm_neuron.functional.attention.mla_sparse_attention import (
    LATENT_NOPE_DIM,
    LATENT_ROPE_DIM,
    _default_widths,
    _gather_latent,
    _mla_gathered_attention,
    _window_local_indices,
)


def _compressed_pool_span(
    cache: Tensor,
    block_table: Optional[Tensor],
    max_compressed_slots: Optional[int],
) -> int:
    """Static per-sequence span of the compressed pool to scan.

    Used only when the caller has no ``topk_indices`` (the C128-dense layers,
    ``dsv4_ref/model.py:519``): the index list is then the whole causal
    compressed pool. The span must be a python int so the traced shape is
    static — it comes from the block table (``max_blocks_per_seq * block_size``)
    or from the contiguous cache's slot count.
    """
    if max_compressed_slots is not None:
        return int(max_compressed_slots)
    if cache.dim() == 4:
        assert block_table is not None, (
            "a paged compressed cache needs either max_compressed_slots or a "
            "block table to bound the pool span"
        )
        return int(block_table.shape[1]) * int(cache.shape[2])
    return int(cache.shape[1])


def mla_decode_attention(
    q: Tensor,
    latent_cache: Tensor,
    swa_cache: Tensor,
    scale: float,
    sink: Optional[Tensor],
    *,
    positions: Optional[Tensor] = None,
    context_lens: Optional[Tensor] = None,
    window: int = 128,
    compress_ratio: int = 0,
    topk_indices: Optional[Tensor] = None,
    topk_index_offset: int = 0,
    max_compressed_slots: Optional[int] = None,
    latent_v_cache: Optional[Tensor] = None,
    latent_rope_cache: Optional[Tensor] = None,
    latent_scale_cache: Optional[Tensor] = None,
    latent_widths: Optional[Sequence[int]] = None,
    latent_block_table: Optional[Tensor] = None,
    swa_v_cache: Optional[Tensor] = None,
    swa_scale_cache: Optional[Tensor] = None,
    swa_widths: Optional[Sequence[int]] = None,
    swa_block_table: Optional[Tensor] = None,
    swa_pos_offset: Optional[Tensor] = None,
    swa_ring: bool = False,
    nope_dim: int = LATENT_NOPE_DIM,
    rope_dim: int = LATENT_ROPE_DIM,
    quant_group_size: int = 64,
    softmax_dtype: torch.dtype = torch.float32,
    out_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Single-token-generation attention for the DSv4-Flash MLA layers.

    Does a paged read of the compressed-latent cache and of the sliding-window
    cache through their block tables, then ONE sink-augmented softmax over the
    union of the two index ranges, with all softmax statistics accumulated in
    fp32 and a single cast to bf16 at the end.

    Absorbed-MLA form: the 512-wide latent (448 NoPE + 64 RoPE) is both key and
    value — there is no separate V projection, so the output is 512 wide and its
    RoPE dims are de-rotated afterwards by the O-projection
    (``dsv4_ref/model.py:480``, ``:538``, ``:539``).

    Args (plan-fixed positional order, contract §5):
        q: ``[B, H, 512]`` (or ``[B, 1, H, 512]``) queries for the one token
            each sequence is generating.
        latent_cache: compressed-latent cache — 4-D paged
            ``[num_blocks, 1, block_size, width]`` (contract §4; pass the other
            physical pieces via ``latent_v_cache``/``latent_rope_cache``) or 3-D
            ``[B, num_slots, 512]`` (reference layout,
            ``dsv4_ref/model.py:497``).
        swa_cache: sliding-window latent cache, same two layouts. Pass
            ``swa_ring=True`` for the reference's ``pos % window`` ring buffer
            (``dsv4_ref/model.py:535``).
        scale: softmax scale, ``head_dim ** -0.5`` (``dsv4_ref/model.py:470``).
        sink: ``[H]`` fp32 per-query-head sink logit
            (``dsv4_ref/model.py:462``), or None. It is an extra logit that
            enters the softmax DENOMINATOR only — no value vector
            (``dsv4_ref/kernel.py:345-348``).

    Keyword-only extensions (all defaulted, so the recorded positional call form
    ``mla_decode_attention(q, latent_cache, swa_cache, scale, sink)`` stays
    valid):
        positions: ``[B]`` absolute position of the token being generated. With
            it, the sliding-window band and the compressed causal cap are exact
            for cold sequences. Without it (and without ``context_lens``) the
            code assumes a warm cache: every window slot and every ``-1``-free
            compressed index counts.
        context_lens: ``[B]`` number of cached tokens; used as
            ``positions = context_lens - 1`` when ``positions`` is absent.
        window: sliding-window width, 128 (``dsv4_ref/model.py:458``).
        compress_ratio: layer compression ratio (4 or 128). Used for the causal
            cap ``compressed_slot < (pos + 1) // compress_ratio``
            (``dsv4_ref/model.py:277``, ``:435``). 0 disables the cap.
        topk_indices: ``[B, topk]`` int DSA top-k compressed-slot indices,
            ``-1`` = absent (``dsv4_ref/model.py:433-438``). When None, the
            whole causal compressed pool is scanned, which is what the C128
            layers do (``dsv4_ref/model.py:519``).
        topk_index_offset: subtracted from ``topk_indices`` first; the reference
            offsets them by ``window`` so they address its single fused buffer
            (``dsv4_ref/model.py:515``).
        max_compressed_slots: static pool span for the no-``topk_indices`` case.
        latent_*/swa_*: the other physical cache pieces, their column widths,
            their fp32 group-64 dequant scales, and their block tables.
        swa_pos_offset: ``[B]`` int per-sequence window-start offset for a
            SWA block table the runner TRIMMED to the window-relevant blocks.
            Subtracted from ``positions`` for the sliding-window leg ONLY, so
            that leg addresses its trimmed table in the trimmed frame while
            the compressed leg keeps the absolute positions its causal cap
            needs. This is NOT optional bookkeeping on a real serve: at decode
            the runner replaces a ``SlidingWindowSpec`` group's
            ``block_table_tensor`` with ``_compute_swa_decode_tensors``'
            window-trimmed gather and publishes the matching
            ``swa_kv_pos_offset`` (``neuron_model_runner.py:3966-3985``,
            ``:3786-3815``), so a caller that keeps feeding absolute positions
            reads the wrong blocks (or out of range) for every sequence whose
            context has grown past the trimmed span. ``gpt_oss`` applies the
            same shift (``model_bf16.py:1533-1552``). Negative results are
            clamped to 0 — a padded/freed row carries position 0 against a
            positive offset, and the clamp collapses its window instead of
            marking a stale row valid, which is exactly the guard
            ``model_bf16.py:1539-1552`` records.
        nope_dim / rope_dim: latent split, 448 + 64.
        quant_group_size: fp8 group size of the scale caches, 64.
        softmax_dtype: fp32 accumulation dtype for the softmax statistics.
        out_dtype: bf16 output dtype.

    Returns:
        ``[B, H, 512]`` (or ``[B, 1, H, 512]`` if ``q`` was 4-D).
    """
    q_was_4d = q.dim() == 4
    if q_was_4d:
        batch, q_len, num_heads, head_dim = q.shape
        assert q_len == 1, (
            "mla_decode_attention generates one token per sequence; got "
            f"q_len={q_len} (use mla_sparse_attention for multi-token queries)"
        )
        q_flat = q.reshape(batch, num_heads, head_dim)
    else:
        assert q.dim() == 3, f"q must be [B, H, D] or [B, 1, H, D], got {tuple(q.shape)}"
        q_flat = q
        batch, num_heads, head_dim = q_flat.shape
    assert head_dim == nope_dim + rope_dim, (
        f"q head_dim {head_dim} != nope_dim + rope_dim ({nope_dim} + {rope_dim}); "
        "the absorbed-MLA latent is 512 = 448 NoPE + 64 RoPE"
    )
    device = q_flat.device

    # One token per sequence, so token index == sequence index.
    seq_ids = torch.arange(batch, device=device, dtype=torch.int64)

    pos: Optional[Tensor] = None
    if positions is not None:
        pos = positions.reshape(-1).to(torch.int64)
    elif context_lens is not None:
        pos = (context_lens.reshape(-1).to(torch.int64) - 1).clamp_min(0)

    # ── Sliding-window leg ───────────────────────────────────────────────
    if pos is None:
        # Warm-cache assumption: the whole window is populated. Ring order is
        # irrelevant to the softmax, so a plain arange addresses all `window`
        # slots (reference reads the same set, rotated: model.py:262-264).
        win_idx = (
            torch.arange(window, device=device, dtype=torch.int64)
            .unsqueeze(0)
            .expand(batch, window)
        )
        win_valid = torch.ones_like(win_idx, dtype=torch.bool)
    else:
        # Shift into the trimmed block table's frame when the runner trimmed
        # it; see ``swa_pos_offset`` in the docstring. The band's geometry is
        # translation-invariant, so this is exact rather than approximate.
        swa_pos = pos
        if swa_pos_offset is not None:
            swa_pos = (pos - swa_pos_offset.reshape(-1).to(torch.int64)).clamp_min(0)
        win_idx, win_valid = _window_local_indices(swa_pos, window, ring=swa_ring)

    swa_caches: List[Tensor] = [swa_cache]
    if swa_v_cache is not None:
        swa_caches.append(swa_v_cache)
    win_latent = _gather_latent(
        swa_caches,
        _default_widths(swa_widths, len(swa_caches), nope_dim, rope_dim),
        win_idx,
        seq_ids=seq_ids,
        block_table=swa_block_table,
        scale_cache=swa_scale_cache,
        dtype=softmax_dtype,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        quant_group_size=quant_group_size,
    )

    # ── Compressed-latent leg ────────────────────────────────────────────
    if topk_indices is not None:
        comp_idx = topk_indices.reshape(batch, -1).to(torch.int64) - topk_index_offset
        comp_valid = comp_idx >= 0
    else:
        # C128-dense layers: scan the whole causal compressed pool
        # (dsv4_ref/model.py:277, :519).
        span = _compressed_pool_span(latent_cache, latent_block_table, max_compressed_slots)
        comp_idx = (
            torch.arange(span, device=device, dtype=torch.int64)
            .unsqueeze(0)
            .expand(batch, span)
        )
        comp_valid = torch.ones_like(comp_idx, dtype=torch.bool)

    if pos is not None and compress_ratio > 0:
        # Causal cap: a query at `pos` may only see compressed slots that
        # already exist (dsv4_ref/model.py:277, :435).
        comp_valid = comp_valid & (comp_idx < ((pos.reshape(-1, 1) + 1) // compress_ratio))
    elif context_lens is not None and compress_ratio > 0:
        comp_valid = comp_valid & (
            comp_idx < (context_lens.reshape(-1, 1).to(torch.int64) // compress_ratio)
        )

    latent_caches: List[Tensor] = [latent_cache]
    if latent_v_cache is not None:
        latent_caches.append(latent_v_cache)
    if latent_rope_cache is not None:
        latent_caches.append(latent_rope_cache)
    comp_latent = _gather_latent(
        latent_caches,
        _default_widths(latent_widths, len(latent_caches), nope_dim, rope_dim),
        comp_idx,
        seq_ids=seq_ids,
        block_table=latent_block_table,
        scale_cache=latent_scale_cache,
        dtype=softmax_dtype,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        quant_group_size=quant_group_size,
    )

    # ── ONE softmax over [window ++ compressed], sink in the denominator ──
    latent = torch.cat([win_latent, comp_latent], dim=1)
    valid = torch.cat([win_valid, comp_valid], dim=1)
    sink_f = None if sink is None else sink.reshape(-1)[:num_heads]

    out = _mla_gathered_attention(
        q_flat,
        latent,
        valid,
        sink_f,
        scale,
        softmax_dtype=softmax_dtype,
        out_dtype=out_dtype,
    )
    if q_was_4d:
        return out.view(batch, 1, num_heads, head_dim)
    return out
