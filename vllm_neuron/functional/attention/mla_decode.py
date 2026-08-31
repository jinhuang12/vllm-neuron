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
from vllm_neuron.functional.attention.swa_attention import dequant_group_scales

#: The runner's padded-slot sentinel (``neuron_model_runner.py:92``,
#: ``PAD_SLOT_ID = -1``) — the same value the model side compares against
#: (``attention._PAD_SLOT_ID``). Redeclared here because ``functional`` must
#: not import from ``model``.
_PAD_SLOT_ID: int = -1

#: Explicit max for int64 index clamps — a min-only s64 clamp lowers with a
#: synthesized ``iinfo(int64).max`` operand, which NCC_ESFH001 rejects (ITER-21).
_INT32_MAX: int = 2**31 - 1


def _masked_write_rows(cache: Tensor, slots: Tensor, rows: Tensor) -> None:
    """Write one whole slot per row into a paged ``cache``, skipping ``-1``.

    The op-owned decode cache write (LD-75 ``update_cache`` torch fallback;
    sibling contract ``attention_decode.py:770-838``). Deliberately the SAME
    arithmetic as the model side's ``_masked_scatter_rows`` — clamp the padded
    destinations to slot 0, ``index_select`` that slot's current content and
    ``torch.where`` it back so the redirect is a no-op write — so a row
    written here is BITWISE what the model-side scatter would have stored.
    Traced AFTER every cache read in this op, the functionalized scatter feeds
    only the aliased root outputs (plan §19.2 Rules 3-4); the LD-75 NKI kernel
    replaces it with an in-kernel write.
    """
    num_blocks, num_kv_heads, block_size, width = cache.shape
    src = rows.to(cache.dtype)
    extra = width - src.shape[-1]
    assert extra >= 0, f"{src.shape[-1]} columns do not fit stored width {width}"
    if extra > 0:
        src = torch.cat(
            (
                src,
                torch.zeros(
                    src.shape[0], extra, dtype=src.dtype, device=src.device
                ),
            ),
            dim=-1,
        )
    flat = cache.view(num_blocks * num_kv_heads * block_size, width)
    valid = (slots > _PAD_SLOT_ID).unsqueeze(-1)
    dest = torch.clamp(slots, min=0).to(torch.long)
    existing = torch.index_select(flat, 0, dest)
    flat.index_put_((dest,), torch.where(valid, src, existing))


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
    current_latent_rows: Optional[Tensor] = None,
    current_scale_rows: Optional[Tensor] = None,
    current_compressed_rows: Optional[Tensor] = None,
    current_slot_ids: Optional[Tensor] = None,
    update_cache: bool = True,
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
        current_latent_rows: ``[B, nope_dim + rope_dim]`` CACHE-FORM
            sliding-window row of the token each sequence is generating — the
            exact tensor the cache write stores (already cast to the cache
            dtype). Plan §19.2 Rules 1/3 (LD-75, F-13): with the KV-dataflow
            restructure the model traces NO cache write before this op, so the
            cache parameters hold only prior rows; the current token's row
            rides here as an explicit operand (the F-5 shape — the window's
            right edge IS this token) and, under ``update_cache``, this op owns
            writing it.
        current_scale_rows: ``[B, num_scale_groups]`` CACHE-FORM group-64
            scale row companion (when ``swa_scale_cache`` is in play).
        current_compressed_rows: ``[B, nope_dim + rope_dim (+ groups)]``
            CACHE-FORM compressed row — NoPE codes ++ RoPE columns ++ scale
            columns — for the compression group this token CLOSES, when it
            closes one (F-240). Ignored (and unwritten) for sequences whose
            ``current_slot_ids[:, 1]`` is ``-1``.
        current_slot_ids: ``[B, 3]`` int PHYSICAL write frame, the writer's
            own (F-13): column 0 the raw sliding-window slot, column 1 the
            compressed-latent coarse slot, column 2 the rope/scale coarse
            slot; ``-1`` where the writer masked (padding, or no group
            closing). Sourcing an in-flight row requires the matching column
            to be a real slot.
        update_cache: this op OWNS the decode cache writes (sibling contract:
            ``gpt_oss`` ``attention_decode.py:770-838``, model_bf16.py:900-931
            ``update_cache=True`` returns output only). The torch fallback
            writes via ``index_put_`` AFTER all reads, so the functionalized
            scatter feeds only the aliased root outputs (Rule 4 topology);
            the LD-75 NKI kernel writes in place and removes the scatter
            entirely (Rule 3). With every ``current_*`` operand ``None``
            nothing is written and the pre-restructure call form is unchanged.
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
        pos = (context_lens.reshape(-1).to(torch.int64) - 1).clamp(0, _INT32_MAX)

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
            swa_pos = (pos - swa_pos_offset.reshape(-1).to(torch.int64)).clamp(
                0, _INT32_MAX
            )
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

    # ── Current-token provenance merge, window leg (Rules 1/3, LD-75) ────
    num_scale_groups = nope_dim // quant_group_size
    slots_b: Optional[Tensor] = None
    if current_slot_ids is not None:
        slots_b = current_slot_ids.reshape(batch, -1).to(torch.long)
    if current_latent_rows is not None:
        assert slots_b is not None, (
            "current_latent_rows needs current_slot_ids (the writer's frame)"
        )
        assert pos is not None, (
            "the current-row provenance split needs positions (or "
            "context_lens) to place the window's right edge"
        )
        assert not swa_ring, (
            "current-row provenance is defined for the paged/absolute window "
            "frame; the ring layout has no writer frame to validate against"
        )
        # Identical dequant path to _gather_latent's, so the in-flight row and
        # its written-then-read twin are BITWISE equal (D1).
        cur_f = current_latent_rows.reshape(batch, -1)[
            ..., : nope_dim + rope_dim
        ].to(softmax_dtype)
        if swa_scale_cache is not None:
            assert current_scale_rows is not None, (
                "a scale-quantized window cache needs current_scale_rows"
            )
            cur_s = current_scale_rows.reshape(batch, -1)[
                ..., :num_scale_groups
            ].to(torch.float32)
            cur_lat = torch.cat(
                (
                    dequant_group_scales(
                        cur_f[..., :nope_dim],
                        cur_s,
                        group_size=quant_group_size,
                        num_groups=num_scale_groups,
                    ),
                    cur_f[..., nope_dim:],
                ),
                dim=-1,
            )
        else:
            cur_lat = cur_f
        # The window's right edge IS the current token (the F-5 shape). Every
        # other band entry is strictly prior and stays cache-sourced; the
        # right edge is sourced from the operand iff the writer's frame says
        # a real slot is being written (F-13; padded sequences carry -1).
        edge = swa_pos.reshape(-1, 1)
        swa_slot_ok = (slots_b[:, 0] > _PAD_SLOT_ID).unsqueeze(1)
        use_cur = win_valid & (win_idx == edge) & swa_slot_ok
        win_latent = torch.where(
            use_cur.unsqueeze(-1), cur_lat.unsqueeze(1), win_latent
        )
        win_valid = win_valid & ((win_idx < edge) | use_cur)

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

    # ── Current-token provenance merge, compressed leg (F-240, LD-75) ────
    if current_compressed_rows is not None:
        assert slots_b is not None, (
            "current_compressed_rows needs current_slot_ids"
        )
        assert pos is not None and compress_ratio > 0, (
            "the compressed provenance split needs positions and "
            "compress_ratio to place the group this token closes"
        )
        lat_w = nope_dim + rope_dim
        cur_c_f = current_compressed_rows.reshape(batch, -1)[..., :lat_w].to(
            softmax_dtype
        )
        if latent_scale_cache is not None:
            cur_cs = current_compressed_rows.reshape(batch, -1)[
                ..., lat_w : lat_w + num_scale_groups
            ].to(torch.float32)
            cur_clat = torch.cat(
                (
                    dequant_group_scales(
                        cur_c_f[..., :nope_dim],
                        cur_cs,
                        group_size=quant_group_size,
                        num_groups=num_scale_groups,
                    ),
                    cur_c_f[..., nope_dim:],
                ),
                dim=-1,
            )
        else:
            cur_clat = cur_c_f
        # A group is PRIOR iff it closed at an earlier step (g < pos // ratio,
        # written by that step's forward). The group closing at THIS token
        # (g == pos // ratio, admissible only when (pos+1) % ratio == 0 — the
        # causal cap already encodes that) is sourced from the operand iff the
        # writer's frame confirms a real coarse slot (F-13; -1 when no group
        # closes or the sequence is padded).
        gid = torch.div(pos, compress_ratio, rounding_mode="floor").reshape(-1, 1)
        lat_slot_ok = (slots_b[:, 1] > _PAD_SLOT_ID).unsqueeze(1)
        use_cur_c = comp_valid & (comp_idx == gid) & lat_slot_ok
        comp_latent = torch.where(
            use_cur_c.unsqueeze(-1), cur_clat.unsqueeze(1), comp_latent
        )
        comp_valid = comp_valid & ((comp_idx < gid) | use_cur_c)

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

    # ── Op-owned cache writes (LD-75 update_cache, Rules 3-4) ────────────
    # Traced strictly AFTER every cache read above, so the functionalized
    # scatters feed only the aliased root outputs — zero effective readers
    # (the S34-cheap topology). The LD-75 NKI kernel moves these writes
    # in-kernel and removes the scatters from the graph entirely.
    if update_cache and slots_b is not None:
        if current_latent_rows is not None:
            _masked_write_rows(
                swa_cache, slots_b[:, 0], current_latent_rows.reshape(batch, -1)
            )
            if swa_scale_cache is not None and current_scale_rows is not None:
                _masked_write_rows(
                    swa_scale_cache,
                    slots_b[:, 0],
                    current_scale_rows.reshape(batch, -1),
                )
        if current_compressed_rows is not None:
            # Column layout mirrors the model's _write_compressed_cache: NoPE
            # code pieces (columns [0, nope_dim)) land in the latent pair at
            # the coarse-latent slot; the RoPE piece and the scale columns
            # land in the rope pair at the coarse-rope slot.
            bundle = current_compressed_rows.reshape(batch, -1)
            widths = _default_widths(
                latent_widths, len(latent_caches), nope_dim, rope_dim
            )
            col = 0
            for cache_piece, piece_w in zip(latent_caches, widths):
                slot_col = 1 if col + piece_w <= nope_dim else 2
                _masked_write_rows(
                    cache_piece,
                    slots_b[:, slot_col],
                    bundle[:, col : col + piece_w],
                )
                col += piece_w
            if latent_scale_cache is not None:
                _masked_write_rows(
                    latent_scale_cache,
                    slots_b[:, 2],
                    bundle[:, col : col + num_scale_groups],
                )

    if q_was_4d:
        return out.view(batch, 1, num_heads, head_dim)
    return out
