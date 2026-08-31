# SPDX-License-Identifier: Apache-2.0
"""DSv4-Flash sparse MLA attention for the context-encoding (prefill) path.

WHY torch and not a kernel: the plan's ``mla_sparse_attention_cte`` nkilib
kernel is absent from the installed neuron wheel (interface contract §0), so
this ladder row takes its recorded same-row torch-composition rung. The code
stays inside the traceable static-shape subset: python loops over statically
known counts, ``torch.gather``/``index_select`` instead of boolean masking, no
``.item()``, no ``.tolist()``, no ``nonzero()``, no data-dependent shapes, no
python ``if`` on tensor values. All masks are additive float masks.

ABSORBED-MLA FORM (important, and the reason there is no ``v`` argument):
DeepSeek's own reference implementation keeps ONE latent tensor per position and
uses it as both key and value — ``dsv4_ref/model.py:480`` allocates a single
``kv_cache[max_batch, window + max_seq_len // compress_ratio, head_dim]`` and
``dsv4_ref/model.py:533``/``:538`` pass exactly that one tensor into
``sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale)``. The latent is 512
wide = 448 NoPE + 64 RoPE (``dsv4_ref/model.py:453-455``,
``args.head_dim=512``/``args.rope_head_dim=64``). There is NO separate V
projection at this stage: the attention output is the 512-wide latent mixture,
and the RoPE dims are de-rotated afterwards by the O-projection
(``dsv4_ref/model.py:539`` ``apply_rotary_emb(o[..., -rd:], freqs_cis, True)``).
The ``wo_a``/``wo_b`` pair is what finally maps it to hidden size.

Reference math this module reproduces (PRIMARY evidence, pinned checkpoint
revision):
  * ``dsv4_ref/kernel.py:294-352`` — ``sparse_attn_kernel``: gather the KV rows
    named by ``topk_idxs``, treat index ``-1`` as an absent slot with an
    ``-inf`` logit (``:323-327``), score with ``q @ kv^T * scale`` (``:328-330``),
    accumulate the softmax statistics in fp32 (``:308-314``), add the per-head
    sink to the DENOMINATOR ONLY after the key loop (``:345-346``), divide and
    store bf16 (``:347-350``).
  * ``dsv4_ref/model.py:513-538`` — the index list is
    ``cat([window_idxs, compressed_topk_idxs], dim=-1)``: ONE softmax over the
    sliding window and the gathered compressed slots together, not two
    attentions merged afterwards. Same structure as upstream vLLM's single
    fused call with ``extra_k_cache``/``extra_indices_in_kvcache``
    (``deepseek_v4_attention.py:897-913``).
  * ``dsv4_ref/model.py:261-271`` — the window index list
    (``clamp(pos - window + 1, 0) + arange(window)``, ``-1`` past the query).
  * ``dsv4_ref/model.py:275-282`` / ``:430-438`` — compressed indices are
    causally capped and marked ``-1`` where the compressed slot does not exist
    yet for that query.
  * ``dsv4_ref/model.py:517`` / ``:519`` — ``topk_idxs`` come from the DSA
    indexer (C4 layers) or from a plain causal arange (C128 layers); either way
    they are a STATIC-SHAPE int tensor, which is why this op gathers with
    ``index_select``/``torch.gather`` and never with a boolean mask.
"""

from typing import List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from vllm_neuron.functional.attention.swa_attention import (
    _mask_from_allowed,
    _sink_softmax_attend,
    dequant_group_scales,
)

# Latent geometry (DSv4-Flash). Kept as module constants so the docstrings and
# the default kwargs cannot drift apart. dsv4_ref/model.py:453-455.
LATENT_NOPE_DIM: int = 448
LATENT_ROPE_DIM: int = 64
LATENT_DIM: int = LATENT_NOPE_DIM + LATENT_ROPE_DIM  # 512


def _flat_rows(cache: Tensor) -> Tuple[Tensor, int]:
    """Flatten a latent cache to ``[rows, width]`` plus its rows-per-sequence.

    Two layouts are accepted, both static-shape:
      * 4-D paged ``[num_blocks, num_kv_heads, block_size, width]`` (the port's
        KV-cache convention, contract §4; ``num_kv_heads`` is 1 for every DSv4
        cache). ``rows_per_seq`` is 0 — row ids are physical slot ids and the
        caller must translate through a block table.
      * 3-D contiguous ``[batch, num_slots, width]`` — the reference layout
        (``dsv4_ref/model.py:480``). ``rows_per_seq = num_slots``, so row id
        ``= seq * num_slots + local_index``.

    NO dtype conversion happens here — the cache is flattened in its STORAGE
    dtype and the caller converts only the rows it actually gathered
    (LD-78 / plan §19.2 Rule 2, F-241). An earlier revision converted the WHOLE
    cache first (``cache.to(dtype)``), justified by the claim that "fp8 tensors
    cannot be fancy-indexed (``attention_decode.py:610-620``)". **That claim was
    false** (F-5, probe ep9-P3: ``torch.index_select`` on ``float8_e4m3fn``
    lowers cleanly through ``convert_fx_to_hlo``), and the whole-cache convert
    was itself a compile-breaking defect: it materializes every 88 MB fp8 cache
    as a 352 MB fp32 mid-graph value — ep19 ITER-20 P76b counted 1,991 such
    converts as the effective readers keeping NCC_EOOM002 alive (peak 28.23 GB
    vs the 24.00 GB Trn2 limit). Elementwise conversion commutes with row
    gather EXACTLY, so gather-first is bitwise-identical for every caller.
    """
    if cache.dim() == 4:
        num_blocks, num_kv_heads, block_size, width = cache.shape
        assert num_kv_heads == 1, (
            "DSv4 latent caches are declared with num_kv_heads=1 (contract §4); "
            f"got {num_kv_heads}"
        )
        return cache[:, 0].reshape(num_blocks * block_size, width), 0
    assert cache.dim() == 3, (
        "latent cache must be 4-D paged [num_blocks, num_kv_heads, block_size, "
        f"width] or 3-D contiguous [batch, num_slots, width]; got {tuple(cache.shape)}"
    )
    batch, num_slots, width = cache.shape
    return cache.reshape(batch * num_slots, width), num_slots


def _slot_ids(
    local_idx: Tensor,
    *,
    seq_ids: Optional[Tensor],
    block_table: Optional[Tensor],
    block_size: int,
    rows_per_seq: int,
) -> Tensor:
    """Translate per-token *sequence-local* slot indices into flat row ids.

    ``local_idx``: ``[T, S]`` int64. Returns ``[T, S]`` int64 row ids into the
    table produced by ``_flat_rows``.

    Paged form (``block_table`` given, ``[B, max_blocks_per_seq]``):
    ``row = block_table[seq, local // block_size] * block_size + local % block_size``
    — the port-wide paged addressing of ``dataflow-shapes.md`` §D, read the same
    way ``attention_decode.py:603-635`` reads it. The block lookup uses
    ``torch.gather`` (never boolean masking) and the block index is clamped into
    range; out-of-range entries are masked out by the caller's validity map.

    Contiguous form (``rows_per_seq > 0``): ``row = seq * rows_per_seq + local``.
    """
    num_tokens = local_idx.shape[0]
    if block_table is not None:
        max_blocks = block_table.shape[1]
        table = block_table.to(torch.int64)
        if seq_ids is not None:
            table = torch.index_select(table, 0, seq_ids.reshape(-1).to(torch.int64))
        else:
            table = table[:1].expand(num_tokens, max_blocks)
        block_of = (local_idx // block_size).clamp(0, max_blocks - 1)
        blocks = torch.gather(table, 1, block_of)
        blocks = torch.where(blocks < 0, torch.zeros_like(blocks), blocks)
        return blocks * block_size + (local_idx % block_size)
    if rows_per_seq > 0 and seq_ids is not None:
        return seq_ids.reshape(-1, 1).to(torch.int64) * rows_per_seq + local_idx
    return local_idx


def _gather_latent(
    caches: Sequence[Tensor],
    widths: Sequence[int],
    local_idx: Tensor,
    *,
    seq_ids: Optional[Tensor],
    block_table: Optional[Tensor],
    scale_cache: Optional[Tensor],
    dtype: torch.dtype,
    nope_dim: int,
    rope_dim: int,
    quant_group_size: int,
) -> Tensor:
    """Gather ``[T, S, nope_dim + rope_dim]`` latents named by ``local_idx``.

    ``caches`` are the physical pieces of one logical latent (contract §4 splits
    it as NoPE ``[0:224]`` + NoPE ``[224:448]`` + 64 RoPE dims across three
    cache tensors; a single pre-assembled 512-wide cache is also accepted) and
    ``widths`` says how many columns each piece contributes. The python loop
    runs over a statically known piece count.

    Negative indices (the reference's ``-1`` absent-slot marker,
    ``dsv4_ref/kernel.py:323-325``) are clamped to row 0 here and zeroed/masked
    by the caller, mirroring the kernel's ``if_then_else(idx != -1, kv, 0)``.
    """
    num_tokens, span = local_idx.shape
    safe_local = torch.where(local_idx < 0, torch.zeros_like(local_idx), local_idx)

    parts: List[Tensor] = []
    for cache, width in zip(caches, widths):
        flat, rows_per_seq = _flat_rows(cache)
        block_size = cache.shape[2] if cache.dim() == 4 else 0
        rows = _slot_ids(
            safe_local,
            seq_ids=seq_ids,
            block_table=block_table,
            block_size=block_size,
            rows_per_seq=rows_per_seq,
        )
        rows = rows.clamp(0, flat.shape[0] - 1).reshape(-1)
        # LD-78 / plan §19.2 Rule 2 (F-241): gather the STORAGE-dtype rows
        # FIRST, convert only the gathered rows. Bitwise-identical to the old
        # convert-then-gather (elementwise convert commutes with row gather);
        # the whole-cache convert was the dominant NCC_EOOM002 reader class
        # (ep19 ITER-20 P76b: 1,991 whole-cache converts).
        part = (
            torch.index_select(flat, 0, rows).to(dtype).view(num_tokens, span, -1)
        )
        parts.append(part[..., :width])

    latent = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
    assert latent.shape[-1] == nope_dim + rope_dim, (
        f"assembled latent width {latent.shape[-1]} != nope_dim + rope_dim "
        f"({nope_dim} + {rope_dim}); check the cache widths"
    )

    if scale_cache is not None:
        # The NoPE half is stored fp8 with group-64 UE8M0 scales; the RoPE half
        # is stored bf16 unscaled (contract §4; write side
        # fused_compress_quant_cache.py:156-214, reference dsv4_ref/model.py:378
        # act_quant(kv[..., :-rd], 64, ...)).
        flat_s, rows_per_seq_s = _flat_rows(scale_cache)
        block_size_s = scale_cache.shape[2] if scale_cache.dim() == 4 else 0
        rows_s = _slot_ids(
            safe_local,
            seq_ids=seq_ids,
            block_table=block_table,
            block_size=block_size_s,
            rows_per_seq=rows_per_seq_s,
        )
        rows_s = rows_s.clamp(0, flat_s.shape[0] - 1).reshape(-1)
        # LD-78 / Rule 2 again: gather storage-dtype scale rows, then convert
        # only the gathered rows to fp32 (bitwise-commuting with the old
        # whole-cache ``.to(float32)``).
        scales = (
            torch.index_select(flat_s, 0, rows_s)
            .to(torch.float32)
            .view(num_tokens, span, -1)
        )
        nope = dequant_group_scales(
            latent[..., :nope_dim],
            scales,
            group_size=quant_group_size,
            num_groups=nope_dim // quant_group_size,
        )
        latent = torch.cat([nope, latent[..., nope_dim:]], dim=-1)

    return latent


def _window_local_indices(
    q_pos: Tensor,
    window: int,
    *,
    ring: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Per-query sliding-window index list and its validity map.

    Reproduces ``dsv4_ref/model.py:268-270`` (fresh-sequence branch of
    ``get_window_topk_idxs``):
    ``idx = clamp(pos - window + 1, min=0) + arange(window)`` with entries
    ``> pos`` marked absent. Absolute positions are returned; when ``ring`` is
    set they are folded modulo ``window``, which is how the reference addresses
    its decode-time window buffer (write at ``start_pos % window``,
    ``dsv4_ref/model.py:535``; read via the rotated arange of ``:262-264``).

    Returns ``(local_idx [T, window] int64, valid [T, window] bool)``.
    """
    device = q_pos.device
    offsets = torch.arange(window, device=device, dtype=torch.int64).unsqueeze(0)
    base = q_pos.reshape(-1, 1).to(torch.int64)
    idx = (base - window + 1).clamp_min(0) + offsets
    valid = idx <= base
    if ring:
        idx = idx % window
    return idx, valid


def _mla_gathered_attention(
    q: Tensor,
    latent: Tensor,
    valid: Tensor,
    sink: Optional[Tensor],
    scale: float,
    *,
    softmax_dtype: torch.dtype,
    out_dtype: torch.dtype,
) -> Tensor:
    """One softmax over the per-token gathered latents (the reference kernel).

    Args:
        q: ``[T, H, D]`` queries.
        latent: ``[T, S, D]`` gathered latents — key AND value (absorbed MLA).
        valid: ``[T, S]`` bool map of slots that really exist.
        sink: ``[H]`` fp32 per-query-head sink, or None.
        scale: ``head_dim ** -0.5`` (``dsv4_ref/model.py:470``), passed in.

    Mirrors ``dsv4_ref/kernel.py:328-350``: fp32 scores, ``-inf`` (here: a large
    finite negative) on absent slots, fp32 row max / sum, the sink added to the
    denominator only, one cast to bf16 at the end.
    """
    num_tokens, num_heads, head_dim = q.shape
    latent_f = latent.to(softmax_dtype)

    # scores: [T, H, S] — batched over tokens, since the gathered KV differs per
    # token. num_kv_heads is 1 for MLA, so there is no GQA expansion here.
    scores = torch.bmm(q.to(softmax_dtype), latent_f.transpose(1, 2)) * scale
    scores = scores + _mask_from_allowed(valid, dtype=softmax_dtype).unsqueeze(1)

    sink_g = None
    if sink is not None:
        sink_g = sink.reshape(1, num_heads, 1).to(softmax_dtype)

    keep = valid.any(dim=-1, keepdim=True).to(softmax_dtype).unsqueeze(-1)  # [T,1,1]
    out = _sink_softmax_attend(scores, latent_f, sink_g, keep)  # [T, H, D] fp32
    return out.to(out_dtype)


def mla_sparse_attention(
    q: Tensor,
    compressed_k_cache: Tensor,
    swa_k_cache: Tensor,
    topk_indices: Tensor,
    attn_sink: Optional[Tensor],
    scale: float,
    window: int,
    *,
    positions: Optional[Tensor] = None,
    seq_ids: Optional[Tensor] = None,
    compressed_v_cache: Optional[Tensor] = None,
    compressed_rope_cache: Optional[Tensor] = None,
    compressed_scale_cache: Optional[Tensor] = None,
    compressed_widths: Optional[Sequence[int]] = None,
    compressed_block_table: Optional[Tensor] = None,
    topk_index_offset: int = 0,
    compress_ratio: int = 0,
    swa_v_cache: Optional[Tensor] = None,
    swa_scale_cache: Optional[Tensor] = None,
    swa_widths: Optional[Sequence[int]] = None,
    swa_block_table: Optional[Tensor] = None,
    swa_ring: bool = False,
    nope_dim: int = LATENT_NOPE_DIM,
    rope_dim: int = LATENT_ROPE_DIM,
    quant_group_size: int = 64,
    chunk_size: Optional[int] = None,
    softmax_dtype: torch.dtype = torch.float32,
    out_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Prefill / context-encoding attention for the compressed DSv4 layers.

    Attends, in ONE softmax, over

      1. the compressed-latent slots named by ``topk_indices`` (the DSA
         indexer's top-512 for C4 layers, the whole causal compressed pool for
         C128 layers), and
      2. the causal sliding window of width ``window`` (128),

    with the per-query-head ``attn_sink`` participating in the softmax
    DENOMINATOR only. This is the reference's
    ``sparse_attn(q, kv, attn_sink, cat([window_idxs, topk_idxs], -1), scale)``
    (``dsv4_ref/model.py:513-533``, ``dsv4_ref/kernel.py:294-352``) written as a
    torch composition, with the two index ranges kept in their own physical
    caches instead of one fused buffer.

    Absorbed-MLA form: the latent is 512 wide (448 NoPE + 64 RoPE) and serves as
    BOTH key and value; there is no separate V projection at this stage, so the
    output is 512 wide and its RoPE dims are de-rotated later by the
    O-projection (``dsv4_ref/model.py:539``).

    Args (plan-fixed positional order, contract §5):
        q: ``[T, H, 512]`` (or ``[B, S, H, 512]``) queries, already per-head
            RMSNorm'd and RoPE'd by ``mla_qkv``.
        compressed_k_cache: compressed-latent cache. Either 4-D paged
            ``[num_blocks, 1, block_size, width]`` (contract §4 — pass the
            companion pieces via ``compressed_v_cache`` /
            ``compressed_rope_cache``) or 3-D ``[B, num_slots, 512]``
            (reference layout, ``dsv4_ref/model.py:497``).
        swa_k_cache: sliding-window latent cache, same two layouts. For a fresh
            prefill chunk this is simply the current chunk's per-token latent
            ``[B, S, 512]`` (``dsv4_ref/model.py:531`` concatenates exactly
            that).
        topk_indices: ``[T, topk]`` (or ``[B, S, topk]``) int tensor of
            *sequence-local compressed-slot* indices, ``-1`` = absent. Static
            shape; gathered with ``index_select``/``torch.gather``, never with a
            boolean mask.
        attn_sink: ``[H]`` fp32 per-query-head sink logit (reference
            ``dsv4_ref/model.py:462``), or None.
        scale: softmax scale, ``head_dim ** -0.5`` (``dsv4_ref/model.py:470``).
        window: sliding-window width, 128 (``dsv4_ref/model.py:458``).

    Keyword-only extensions (every one defaulted, so the recorded positional
    call form stays valid):
        positions: ``[T]`` absolute positions. Default ``arange(T)`` per
            sequence-flat token order.
        seq_ids: ``[T]`` sequence id per token, for packed multi-sequence
            prefill (selects the block-table row / contiguous-cache batch row).
        compressed_v_cache / compressed_rope_cache: the other physical pieces of
            the compressed latent (contract §4: NoPE ``[224:448]`` and the 64
            RoPE dims).
        compressed_scale_cache: fp32 group-64 dequant scales for the NoPE half.
        compressed_widths: columns taken from each compressed piece. Default
            ``(512,)`` for a single cache, ``(224, 224, 64)`` for the contract's
            three-way split.
        compressed_block_table: ``[B, max_blocks_per_seq]`` for the compressed
            pool when it is paged.
        topk_index_offset: subtracted from ``topk_indices`` before use. The
            reference adds an offset so the indices address one fused buffer
            (``dsv4_ref/model.py:515``, ``:436``); pass it here to undo that.
        compress_ratio: when > 0 and ``positions`` is given, additionally cap
            the compressed indices causally at ``(pos + 1) // compress_ratio``
            (``dsv4_ref/model.py:280-281``, ``:435``). Default 0 = trust the
            indexer's ``-1`` marking, exactly as the reference kernel does.
        swa_v_cache / swa_scale_cache / swa_widths / swa_block_table: same idea
            for the sliding-window latent. ``swa_block_table`` must address
            ABSOLUTE positions (``row = table[seq, pos // block_size] *
            block_size + pos % block_size``); if the runner hands out a rotating
            window table instead, pass ``swa_ring=True`` with a 3-D
            ``[B, window, 512]`` buffer.
        swa_ring: the window cache is a ring of length ``window`` addressed at
            ``pos % window`` (the reference's decode-time layout,
            ``dsv4_ref/model.py:535``).
        nope_dim / rope_dim: latent split, 448 + 64.
        quant_group_size: fp8 group size for the scale cache, 64.
        chunk_size: process the query tokens in chunks of this many, to bound
            the peak size of the gathered latent workspace (upstream chunks the
            same gather, ``deepseek_v4_attention.py:966-973``). The chunk count
            is a python constant, so tracing stays static.
        softmax_dtype: fp32 accumulation dtype for the softmax statistics.
        out_dtype: bf16 output dtype.

    Returns:
        ``[T, H, 512]`` (or ``[B, S, H, 512]`` if ``q`` was 4-D) attention out.
    """
    # ── Normalise q / topk_indices to the flat [T, ...] token layout ──────
    q_was_4d = q.dim() == 4
    if q_was_4d:
        batch, seq_len, num_heads, head_dim = q.shape
        q_flat = q.reshape(batch * seq_len, num_heads, head_dim)
    else:
        assert q.dim() == 3, f"q must be [T, H, D] or [B, S, H, D], got {tuple(q.shape)}"
        batch, seq_len = 0, 0
        q_flat = q
    num_tokens, num_heads, head_dim = q_flat.shape
    assert head_dim == nope_dim + rope_dim, (
        f"q head_dim {head_dim} != nope_dim + rope_dim ({nope_dim} + {rope_dim}); "
        "the absorbed-MLA latent is 512 = 448 NoPE + 64 RoPE"
    )
    device = q_flat.device

    topk = topk_indices.reshape(num_tokens, -1).to(torch.int64) - topk_index_offset

    if positions is None:
        if q_was_4d:
            pos = (
                torch.arange(seq_len, device=device, dtype=torch.int64)
                .unsqueeze(0)
                .expand(batch, seq_len)
                .reshape(-1)
            )
        else:
            pos = torch.arange(num_tokens, device=device, dtype=torch.int64)
    else:
        pos = positions.reshape(-1).to(torch.int64)

    if seq_ids is None and q_was_4d:
        seq_ids = (
            torch.arange(batch, device=device, dtype=torch.int64)
            .unsqueeze(1)
            .expand(batch, seq_len)
            .reshape(-1)
        )

    # ── Cache piece lists and their widths ───────────────────────────────
    comp_caches: List[Tensor] = [compressed_k_cache]
    if compressed_v_cache is not None:
        comp_caches.append(compressed_v_cache)
    if compressed_rope_cache is not None:
        comp_caches.append(compressed_rope_cache)
    comp_widths = _default_widths(compressed_widths, len(comp_caches), nope_dim, rope_dim)

    swa_caches: List[Tensor] = [swa_k_cache]
    if swa_v_cache is not None:
        swa_caches.append(swa_v_cache)
    swa_w = _default_widths(swa_widths, len(swa_caches), nope_dim, rope_dim)

    sink_f = None if attn_sink is None else attn_sink.reshape(-1)[:num_heads]

    # ── Chunked over query tokens (statically known chunk count) ─────────
    step = num_tokens if chunk_size is None else min(chunk_size, num_tokens)
    step = max(step, 1)
    outputs: List[Tensor] = []
    for start in range(0, num_tokens, step):
        end = min(start + step, num_tokens)
        chunk_seq = None if seq_ids is None else seq_ids.reshape(-1)[start:end]
        chunk_pos = pos[start:end]

        # Compressed leg: the indexer already marks absent slots with -1
        # (dsv4_ref/model.py:436); optionally re-apply the causal cap.
        comp_idx = topk[start:end]
        comp_valid = comp_idx >= 0
        if compress_ratio > 0:
            comp_valid = comp_valid & (
                comp_idx < ((chunk_pos.reshape(-1, 1) + 1) // compress_ratio)
            )
        comp_latent = _gather_latent(
            comp_caches,
            comp_widths,
            comp_idx,
            seq_ids=chunk_seq,
            block_table=compressed_block_table,
            scale_cache=compressed_scale_cache,
            dtype=softmax_dtype,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
            quant_group_size=quant_group_size,
        )

        # Sliding-window leg.
        win_idx, win_valid = _window_local_indices(chunk_pos, window, ring=swa_ring)
        win_latent = _gather_latent(
            swa_caches,
            swa_w,
            win_idx,
            seq_ids=chunk_seq,
            block_table=swa_block_table,
            scale_cache=swa_scale_cache,
            dtype=softmax_dtype,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
            quant_group_size=quant_group_size,
        )

        # ONE softmax over [window ++ compressed], mirroring the reference's
        # single concatenated index list (dsv4_ref/model.py:520).
        latent = torch.cat([win_latent, comp_latent], dim=1)
        valid = torch.cat([win_valid, comp_valid], dim=1)
        outputs.append(
            _mla_gathered_attention(
                q_flat[start:end],
                latent,
                valid,
                sink_f,
                scale,
                softmax_dtype=softmax_dtype,
                out_dtype=out_dtype,
            )
        )

    out = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
    if q_was_4d:
        return out.view(batch, seq_len, num_heads, head_dim)
    return out


def _default_widths(
    widths: Optional[Sequence[int]],
    num_parts: int,
    nope_dim: int,
    rope_dim: int,
) -> Sequence[int]:
    """Pick the per-piece column counts when the caller did not state them.

    One piece → the whole latent (``nope_dim + rope_dim``). Three pieces → the
    contract §4 compressed split ``(224, 224, 64)`` generalised as
    ``(nope/2, nope/2, rope)``. Two pieces → the NoPE half split in two, with
    the RoPE dims riding in the first piece
    (``(nope // 2 + rope, nope // 2)``) — the layout the SWA cache pair takes
    when the runner splits the 512-wide latent across ``k_cache``/``v_cache``.
    """
    if widths is not None:
        assert len(widths) == num_parts, "one width per cache piece"
        assert sum(widths) == nope_dim + rope_dim, (
            f"cache widths {tuple(widths)} must sum to {nope_dim + rope_dim}"
        )
        return widths
    if num_parts == 1:
        return (nope_dim + rope_dim,)
    if num_parts == 2:
        return (nope_dim // 2 + rope_dim, nope_dim // 2)
    if num_parts == 3:
        return (nope_dim // 2, nope_dim // 2, rope_dim)
    raise AssertionError(f"unsupported latent piece count {num_parts}")
