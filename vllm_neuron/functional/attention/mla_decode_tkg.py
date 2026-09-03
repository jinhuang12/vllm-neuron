# SPDX-License-Identifier: Apache-2.0
"""LD-75 ``mla_decode_tkg`` — NKI decode kernel for the DSv4-Flash MLA layers.

Triad (port-plan §19.2 Amendment 11, ladder decision LD-75; prereg
TRIADS-Z0-PREREGISTRATION.md §D1/§D3, sealed 2026-08-31 before any edit):

  * public op ``mla_decode_tkg`` — EXACT signature mirror of the recorded
    torch composition ``mla_decode_attention`` (every positional arg and
    keyword-only default unchanged; no v1 extensions);
  * gate ``_can_use_mla_decode_tkg`` — cheap, total, shape-only envelope;
    when it declines the call delegates VERBATIM to ``mla_decode_attention``
    (the family-19 composition, untouched — it stays the recorded fallback);
  * kernel ``_mla_decode_tkg_nki`` — one fused NKI kernel per decode step:
    indirect row gathers from the paged fp8 caches (oob-skip), current-token
    operand overlay (the F-240/F-13 provenance merge), in-kernel group-64
    ue8m0 dequant, fp32 sink softmax (sink in the DENOMINATOR only,
    ``dsv4_ref/kernel.py:345-348``), bf16 out — and the in-kernel must-alias
    cache write of the current rows (§17.4 kernel rung: NO scatter node on
    the decode path of the traced torch graph; the write lives inside the
    kernel and threads to the caller's cache through the HOP's
    ``operand_output_aliases``, ``nki_hop.py:529``).

Wrapper/kernel boundary (prereg D3): the torch wrapper computes ONLY small
index/mask tensors — flat row ids per cache piece (reusing the fallback
modules' own ``_slot_ids``/``_window_local_indices`` so index semantics are
shared), operand-overlay ids, additive fp32 validity masks and per-piece
write ids. Cache pieces are passed to the kernel WHOLE and UNRESHAPED so the
write alias threads to the root parameter; the kernel flattens internally via
zero-copy ``.reshape``. The wrapper never reads, converts, or scatters into a
cache-shaped tensor.

Scatter/gather idiom at the deployed pins (NKI 0.5.0+28631259367): the
per-token block scatter of ``nkilib`` ``_update_block_cache_vectorized``
(render TRIADS-RAW-R2-blockscatter-render.txt) —
``nisa.dma_copy(dst=CACHE.reshape((rows, w)).ap(pattern=[[w, tile],[1, w]],
vector_offset=idx, indirect_dim=0), src=tile, oob_mode=oob_mode.skip)`` with
uint32-max as the skip sentinel — used here for both the gathers (src
indirect) and the write (dst indirect).
"""

from typing import Optional, Sequence

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa import oob_mode
from nkilib.core.utils.kernel_assert import kernel_assert

import torch
from torch import Tensor

from vllm_neuron.functional.attention.mla_decode import (
    _PAD_SLOT_ID,
    _compressed_pool_span,
    mla_decode_attention,
)
from vllm_neuron.functional.attention.mla_sparse_attention import (
    LATENT_NOPE_DIM,
    LATENT_ROPE_DIM,
    _default_widths,
    _slot_ids,
    _window_local_indices,
)
from vllm_neuron.functional.attention.swa_attention import MASK_NEG
from vllm_neuron.nki.nki_hop import can_run_kernel, wrap_nki

#: uint32 skip sentinel — out-of-bounds under ``oob_mode.skip`` on every cache
#: piece the kernel touches (R2 render: ``_k_update_fp8_packed`` casts int32
#: ``-1`` to uint32-max for exactly this purpose).
_SKIP32: int = 0xFFFFFFFF

_PMAX: int = 128
_PSUM_F: int = 512  # gen3 PSUM free-dim limit (prereg D4)

#: Explicit max for int64 index clamps — a min-only s64 clamp lowers with a
#: synthesized ``iinfo(int64).max`` operand, which NCC_ESFH001 rejects (ITER-21).
_INT32_MAX: int = 2**31 - 1


# ---------------------------------------------------------------------------
# Gate (prereg D8: cheap, total, monotonic; shape/dtype/layout only — never
# tensor values, so it is trace-safe under FakeTensor)
# ---------------------------------------------------------------------------


def _can_use_mla_decode_tkg(
    q: Tensor,
    latent_cache: Tensor,
    swa_cache: Tensor,
    scale: float,
    sink: Optional[Tensor],
    **kw,
) -> bool:
    """Envelope check for the LD-75 kernel; False → torch fallback."""
    if not can_run_kernel(q):
        return False
    if kw.get("swa_ring", False):
        return False
    if kw.get("nope_dim", LATENT_NOPE_DIM) != 448:
        return False
    if kw.get("rope_dim", LATENT_ROPE_DIM) != 64:
        return False
    if kw.get("quant_group_size", 64) != 64:
        return False
    if kw.get("softmax_dtype", torch.float32) is not torch.float32:
        return False
    if kw.get("out_dtype", torch.bfloat16) not in (torch.bfloat16, torch.float32):
        return False

    if q.dim() == 4:
        if q.shape[1] != 1:
            return False
        batch, _, num_heads, head_dim = q.shape
    elif q.dim() == 3:
        batch, num_heads, head_dim = q.shape
    else:
        return False
    if head_dim != 512 or not (1 <= num_heads <= _PMAX) or not (1 <= batch <= 1024):
        return False
    if q.dtype not in (torch.bfloat16, torch.float32):
        return False

    window = int(kw.get("window", 128))
    if not (1 <= window <= 1024):
        return False

    topk = kw.get("topk_indices")
    if topk is not None:
        if topk.dim() != 2 or not (1 <= topk.shape[1] <= 1024):
            return False
    else:
        try:
            span = _compressed_pool_span(
                latent_cache,
                kw.get("latent_block_table"),
                kw.get("max_compressed_slots"),
            )
        except AssertionError:
            return False
        if not (1 <= span <= 1024):
            return False

    # SWA leg: single 512-wide piece (the only call-site form; contract §4).
    if kw.get("swa_v_cache") is not None:
        return False
    sw = kw.get("swa_widths")
    if sw is not None and (len(sw) != 1 or int(sw[0]) != 512):
        return False
    if swa_cache.dim() not in (3, 4):
        return False
    if swa_cache.dim() == 4 and (
        swa_cache.shape[1] != 1 or kw.get("swa_block_table") is None
    ):
        return False
    if int(swa_cache.shape[-1]) < 512:
        return False

    # Latent leg: (224,224,64) trio or a single 512-wide piece. The SWA-only
    # decode class passes the swa tensors in the latent slots (attention.py
    # `_decode_attention`); the kernel folds that to ONE operand, so the gate
    # requires the scale slots to share the same identity.
    shared = latent_cache is swa_cache
    lat_v = kw.get("latent_v_cache")
    lat_r = kw.get("latent_rope_cache")
    if shared:
        if lat_v is not None or lat_r is not None:
            return False
        if kw.get("latent_scale_cache") is not kw.get("swa_scale_cache"):
            return False
        # the kernel folds the shared pair to one operand and therefore has
        # no distinct latent pieces to write a compressed bundle into — the
        # fallback would write them, so this form must not take the kernel
        if kw.get("current_compressed_rows") is not None:
            return False
    npieces = 1 + (lat_v is not None) + (lat_r is not None)
    if npieces == 2:
        return False
    lw = kw.get("latent_widths")
    if lw is not None:
        if len(lw) != npieces:
            return False
        if npieces == 3 and tuple(int(x) for x in lw) != (224, 224, 64):
            return False
        if npieces == 1 and int(lw[0]) != 512:
            return False
    if latent_cache.dim() not in (3, 4):
        return False
    if latent_cache.dim() == 4 and (
        latent_cache.shape[1] != 1 or kw.get("latent_block_table") is None
    ):
        return False
    for piece in (lat_v, lat_r):
        if piece is not None and piece.dim() != latent_cache.dim():
            return False

    lat_scale = kw.get("latent_scale_cache")
    swa_scale = kw.get("swa_scale_cache")
    if lat_scale is not None and lat_scale.dim() != latent_cache.dim():
        return False
    if swa_scale is not None and swa_scale.dim() != swa_cache.dim():
        return False

    ok_dtypes = (torch.float8_e4m3fn, torch.bfloat16, torch.float32)
    for piece in (latent_cache, swa_cache, lat_v, lat_r, lat_scale, swa_scale):
        if piece is not None and piece.dtype not in ok_dtypes:
            return False
    return True


# ---------------------------------------------------------------------------
# Torch-fallback leg of the LD-75 triad (c14 leg b). The numerics authority
# is UNCHANGED: this delegates 1:1 to the plan-recorded restructured torch
# composition ``mla_decode_attention`` (port-assessment.md:3625 —
# fallback=restructured-torch-composition, NOT re-authored on this visit).
# The delegator exists so the triad module carries its own module-level
# fallback definition, which is what `scripts/finalize_branch.sh` c14 leg b
# checks (`^def <fallback>(` over the triad module — G5 RAW r1, verdict FAIL
# with the bare import). Behavior identical: one extra python frame.
# ---------------------------------------------------------------------------


def _torch_mla_decode_tkg(
    q: Tensor,
    latent_cache: Tensor,
    swa_cache: Tensor,
    scale: float,
    sink: Optional[Tensor],
    **kw,
) -> Tensor:
    """Fallback leg: the family-19 restructured composition, by delegation."""
    return mla_decode_attention(q, latent_cache, swa_cache, scale, sink, **kw)


# ---------------------------------------------------------------------------
# Public op — exact signature mirror of mla_decode_attention (prereg D1)
# ---------------------------------------------------------------------------


def mla_decode_tkg(
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
    """LD-75 decode attention: NKI kernel when the gate admits, else the
    recorded torch fallback ``mla_decode_attention`` verbatim. Same math, same
    contract — see the fallback's docstring for full argument semantics."""
    kw = dict(
        positions=positions,
        context_lens=context_lens,
        window=window,
        compress_ratio=compress_ratio,
        topk_indices=topk_indices,
        topk_index_offset=topk_index_offset,
        max_compressed_slots=max_compressed_slots,
        latent_v_cache=latent_v_cache,
        latent_rope_cache=latent_rope_cache,
        latent_scale_cache=latent_scale_cache,
        latent_widths=latent_widths,
        latent_block_table=latent_block_table,
        swa_v_cache=swa_v_cache,
        swa_scale_cache=swa_scale_cache,
        swa_widths=swa_widths,
        swa_block_table=swa_block_table,
        swa_pos_offset=swa_pos_offset,
        swa_ring=swa_ring,
        current_latent_rows=current_latent_rows,
        current_scale_rows=current_scale_rows,
        current_compressed_rows=current_compressed_rows,
        current_slot_ids=current_slot_ids,
        update_cache=update_cache,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        quant_group_size=quant_group_size,
        softmax_dtype=softmax_dtype,
        out_dtype=out_dtype,
    )
    if not _can_use_mla_decode_tkg(q, latent_cache, swa_cache, scale, sink, **kw):
        return _torch_mla_decode_tkg(q, latent_cache, swa_cache, scale, sink, **kw)
    return _mla_decode_tkg_dispatch(q, latent_cache, swa_cache, scale, sink, **kw)


# ---------------------------------------------------------------------------
# Kernel-path wrapper: small index/mask tensors ONLY (prereg D3). Every line
# mirrors the fallback's index arithmetic 1:1 (mla_decode.py:280-465) so the
# two paths share one index semantics; the heavy math moves into the kernel.
# ---------------------------------------------------------------------------


def _piece_geometry(piece: Tensor) -> tuple[int, int]:
    """(block_size, rows_per_seq) for ``_slot_ids`` — from shapes only, never
    touching cache data (D3: the wrapper does no cache read, not even a view)."""
    if piece.dim() == 4:
        return int(piece.shape[2]), 0
    return int(piece.shape[1]), int(piece.shape[1])


def _mla_decode_tkg_dispatch(
    q: Tensor,
    latent_cache: Tensor,
    swa_cache: Tensor,
    scale: float,
    sink: Optional[Tensor],
    **kw,
) -> Tensor:
    q_was_4d = q.dim() == 4
    q_flat = q.reshape(q.shape[0], q.shape[-2], q.shape[-1]) if q_was_4d else q
    batch, num_heads, head_dim = q_flat.shape
    device = q_flat.device
    nope_dim = kw["nope_dim"]
    rope_dim = kw["rope_dim"]
    quant_group_size = kw["quant_group_size"]
    num_scale_groups = nope_dim // quant_group_size
    window = int(kw["window"])
    compress_ratio = int(kw["compress_ratio"])
    positions = kw["positions"]
    context_lens = kw["context_lens"]
    lat_dim = nope_dim + rope_dim

    seq_ids = torch.arange(batch, device=device, dtype=torch.int64)
    # s64 −1 → uint32-max 0xFFFFFFFF at every consumer's ``.to(torch.uint32)``;
    # NKI oob_mode.skip sentinel unchanged; ESFH001 fix, ITER-21.
    skip = torch.full((1,), -1, device=device, dtype=torch.int64)

    pos: Optional[Tensor] = None
    if positions is not None:
        pos = positions.reshape(-1).to(torch.int64)
    elif context_lens is not None:
        pos = (context_lens.reshape(-1).to(torch.int64) - 1).clamp(0, _INT32_MAX)

    # ── window leg indices (mirrors mla_decode.py:289-306) ──────────────
    if pos is None:
        win_idx = (
            torch.arange(window, device=device, dtype=torch.int64)
            .unsqueeze(0)
            .expand(batch, window)
        )
        win_valid = torch.ones_like(win_idx, dtype=torch.bool)
        swa_pos = None
    else:
        swa_pos = pos
        if kw["swa_pos_offset"] is not None:
            swa_pos = (
                pos - kw["swa_pos_offset"].reshape(-1).to(torch.int64)
            ).clamp(0, _INT32_MAX)
        win_idx, win_valid = _window_local_indices(swa_pos, window, ring=False)

    swa_bs, swa_rps = _piece_geometry(swa_cache)
    win_rows64 = _slot_ids(
        win_idx,
        seq_ids=seq_ids,
        block_table=kw["swa_block_table"],
        block_size=swa_bs,
        rows_per_seq=swa_rps,
    )

    # ── window current-token merge (mirrors mla_decode.py:327-380) ──────
    slots_b: Optional[Tensor] = None
    if kw["current_slot_ids"] is not None:
        slots_b = kw["current_slot_ids"].reshape(batch, -1).to(torch.long)
    use_cur = torch.zeros_like(win_valid)
    cur_win = kw["current_latent_rows"]
    if cur_win is not None:
        assert slots_b is not None, (
            "current_latent_rows needs current_slot_ids (the writer's frame)"
        )
        assert pos is not None, (
            "the current-row provenance split needs positions (or "
            "context_lens) to place the window's right edge"
        )
        edge = swa_pos.reshape(-1, 1)
        swa_slot_ok = (slots_b[:, 0] > _PAD_SLOT_ID).unsqueeze(1)
        use_cur = win_valid & (win_idx == edge) & swa_slot_ok
        win_valid = win_valid & ((win_idx < edge) | use_cur)

    win_gather = torch.where(win_valid & ~use_cur, win_rows64, skip).to(torch.uint32)
    win_ovl = torch.where(
        use_cur, seq_ids.reshape(-1, 1).expand(batch, window), skip
    ).to(torch.uint32)
    win_mask = torch.where(
        win_valid,
        torch.zeros((), device=device, dtype=torch.float32),
        torch.full((), MASK_NEG, device=device, dtype=torch.float32),
    )

    # ── compressed leg indices (mirrors mla_decode.py:382-410) ──────────
    topk_indices = kw["topk_indices"]
    if topk_indices is not None:
        comp_idx = (
            topk_indices.reshape(batch, -1).to(torch.int64)
            - kw["topk_index_offset"]
        )
        comp_valid = comp_idx >= 0
    else:
        span = _compressed_pool_span(
            latent_cache, kw["latent_block_table"], kw["max_compressed_slots"]
        )
        comp_idx = (
            torch.arange(span, device=device, dtype=torch.int64)
            .unsqueeze(0)
            .expand(batch, span)
        )
        comp_valid = torch.ones_like(comp_idx, dtype=torch.bool)

    if pos is not None and compress_ratio > 0:
        comp_valid = comp_valid & (
            comp_idx < ((pos.reshape(-1, 1) + 1) // compress_ratio)
        )
    elif context_lens is not None and compress_ratio > 0:
        comp_valid = comp_valid & (
            comp_idx
            < (context_lens.reshape(-1, 1).to(torch.int64) // compress_ratio)
        )

    # ── compressed current-token merge (mirrors mla_decode.py:412-465) ──
    use_cur_c = torch.zeros_like(comp_valid)
    cur_bundle = kw["current_compressed_rows"]
    if cur_bundle is not None:
        assert slots_b is not None, "current_compressed_rows needs current_slot_ids"
        assert pos is not None and compress_ratio > 0, (
            "the compressed provenance split needs positions and "
            "compress_ratio to place the group this token closes"
        )
        gid = torch.div(pos, compress_ratio, rounding_mode="floor").reshape(-1, 1)
        lat_slot_ok = (slots_b[:, 1] > _PAD_SLOT_ID).unsqueeze(1)
        use_cur_c = comp_valid & (comp_idx == gid) & lat_slot_ok
        comp_valid = comp_valid & ((comp_idx < gid) | use_cur_c)

    lat_bs, lat_rps = _piece_geometry(latent_cache)
    comp_rows64 = _slot_ids(
        comp_idx,
        seq_ids=seq_ids,
        block_table=kw["latent_block_table"],
        block_size=lat_bs,
        rows_per_seq=lat_rps,
    )
    comp_span = comp_idx.shape[1]
    comp_gather = torch.where(comp_valid & ~use_cur_c, comp_rows64, skip).to(
        torch.uint32
    )
    comp_ovl = torch.where(
        use_cur_c, seq_ids.reshape(-1, 1).expand(batch, comp_span), skip
    ).to(torch.uint32)
    comp_mask = torch.where(
        comp_valid,
        torch.zeros((), device=device, dtype=torch.float32),
        torch.full((), MASK_NEG, device=device, dtype=torch.float32),
    )

    # keep factor: 1.0 iff any real key survives (fallback
    # _sink_softmax_attend's ``keep = valid.any``) — rides as the last aux col.
    keep = (
        torch.cat([win_valid, comp_valid], dim=1).any(dim=1).to(torch.float32)
    )
    aux = torch.cat([win_mask, comp_mask, keep.unsqueeze(1)], dim=1).contiguous()

    # ── operands: current rows in cache form; write ids (prereg D3) ─────
    shared_lat = latent_cache is swa_cache
    lat_v = kw["latent_v_cache"]
    lat_r = kw["latent_rope_cache"]
    lat_pieces = [latent_cache] + [c for c in (lat_v, lat_r) if c is not None]
    lat_widths = tuple(
        int(x)
        for x in _default_widths(
            kw["latent_widths"], len(lat_pieces), nope_dim, rope_dim
        )
    )

    cur_win_t = None
    cur_scale_t = None
    cur_bundle_t = None
    if cur_win is not None:
        cur_win_t = (
            cur_win.reshape(batch, -1)[:, :lat_dim].to(swa_cache.dtype).contiguous()
        )
        if kw["swa_scale_cache"] is not None:
            assert kw["current_scale_rows"] is not None, (
                "a scale-quantized window cache needs current_scale_rows"
            )
            cur_scale_t = (
                kw["current_scale_rows"]
                .reshape(batch, -1)[:, :num_scale_groups]
                .to(kw["swa_scale_cache"].dtype)
                .contiguous()
            )
    if cur_bundle is not None:
        cur_bundle_t = (
            cur_bundle.reshape(batch, -1).to(latent_cache.dtype).contiguous()
        )

    wr_ids = None
    if kw["update_cache"] and slots_b is not None and (
        cur_win_t is not None or cur_bundle_t is not None
    ):
        wr_ids = torch.where(
            slots_b > _PAD_SLOT_ID, slots_b, skip.reshape(1, 1)
        ).to(torch.uint32).contiguous()

    sink_t = None
    if sink is not None:
        sink_t = sink.reshape(-1)[:num_heads].to(torch.float32).contiguous()

    wrapped = wrap_nki(_mla_decode_tkg_nki)
    res = wrapped[2](
        q=q_flat.contiguous(),
        aux=aux,
        win_rows=win_gather.contiguous(),
        win_ovl=win_ovl.contiguous(),
        comp_rows=comp_gather.contiguous(),
        comp_ovl=comp_ovl.contiguous(),
        swa_k=swa_cache,
        swa_scale=kw["swa_scale_cache"],
        lat0=None if shared_lat else lat_pieces[0],
        lat1=None if shared_lat or len(lat_pieces) < 2 else lat_pieces[1],
        lat2=None if shared_lat or len(lat_pieces) < 3 else lat_pieces[2],
        lat_scale=None if shared_lat else kw["latent_scale_cache"],
        cur_win=cur_win_t,
        cur_scale=cur_scale_t,
        cur_bundle=cur_bundle_t,
        sink=sink_t,
        wr_ids=wr_ids,
        scale=float(scale),
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        quant_group_size=quant_group_size,
        lat_widths=lat_widths,
        shared_lat=shared_lat,
        out_fp32=(kw["out_dtype"] is torch.float32),
    )
    out = res[0] if isinstance(res, (tuple, list)) else res
    if q_was_4d:
        return out.view(batch, 1, num_heads, head_dim)
    return out


# ---------------------------------------------------------------------------
# NKI kernel (prereg D4 API surface only; deployed pins R1/R2)
# ---------------------------------------------------------------------------


def _flat2(piece):
    """Zero-copy [rows, width] view of a cache piece inside the kernel
    (kernel-side ``.reshape`` on HBM inputs is legal at the deployed pins —
    R1/R2 renders; the WRAPPER never reshapes a cache, D3)."""
    if len(piece.shape) == 4:
        nb, kvh, bs, w = piece.shape
        return piece.reshape((nb * kvh * bs, w))
    b0, s0, w = piece.shape
    return piece.reshape((b0 * s0, w))


def _load_ids(rows_hbm, base, tw):
    """[tw, 1] uint32 id column from a flat offset into a 2-D id tensor."""
    ids = nl.ndarray((tw, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=ids, src=rows_hbm.ap(pattern=[[1, tw], [1, 1]], offset=base)
    )
    return ids


def _gather_leg_chunk(
    ids,
    ovl,
    tw,
    lat_dim,
    pieces,          # [(flat_cache, dst_col, width)] — prior rows
    scale_flat,      # flat scale cache or None
    ovl_srcs,        # [(rows_2d, src_col, dst_col, width)] — current rows
    ovl_scale,       # (rows_2d, src_col) or None
    nsg,
    qgs,
    stage_dtype,
    scale_dtype,
):
    """memset-0 staging → indirect cache gather (oob skip) → operand overlay
    (second indirect DMA — this IS the provenance merge, F-240/F-13) →
    fp8→fp32 convert → group-64 dequant. Returns the [tw, lat_dim] fp32 tile.

    Skipped rows (uint32-max ids) stay 0 and are killed later by the additive
    mask — same masked result as the fallback's clamped-gather-then-mask."""
    stage = nl.ndarray((tw, lat_dim), dtype=stage_dtype, buffer=nl.sbuf)
    nisa.memset(stage, 0.0)
    # P4-r3 fix: the deployed NKI 0.5.0 parser rejects tuple-unpacking
    # for-loop targets ("expecting simple variable",
    # TRIADS-RAW-G3-p4compile-r3.txt) — index loops throughout.
    for _pi in range(len(pieces)):
        flat = pieces[_pi][0]
        dst_col = pieces[_pi][1]
        w = pieces[_pi][2]
        nisa.dma_copy(
            dst=stage[0:tw, dst_col : dst_col + w],
            src=flat.ap(
                pattern=[[w, tw], [1, w]],
                offset=0,
                vector_offset=ids,
                indirect_dim=0,
            ),
            oob_mode=oob_mode.skip,
        )
    for _oi in range(len(ovl_srcs)):
        rows_2d = ovl_srcs[_oi][0]
        src_col = ovl_srcs[_oi][1]
        dst_col = ovl_srcs[_oi][2]
        w = ovl_srcs[_oi][3]
        row_w = rows_2d.shape[1]
        nisa.dma_copy(
            dst=stage[0:tw, dst_col : dst_col + w],
            src=rows_2d.ap(
                pattern=[[row_w, tw], [1, w]],
                offset=src_col,
                vector_offset=ovl,
                indirect_dim=0,
            ),
            oob_mode=oob_mode.skip,
        )

    f32 = nl.ndarray((tw, lat_dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=f32, src=stage)

    if scale_flat is not None:
        sst = nl.ndarray((tw, nsg), dtype=scale_dtype, buffer=nl.sbuf)
        nisa.memset(sst, 0.0)
        sw = scale_flat.shape[1]
        nisa.dma_copy(
            dst=sst,
            src=scale_flat.ap(
                pattern=[[sw, tw], [1, nsg]],
                offset=0,
                vector_offset=ids,
                indirect_dim=0,
            ),
            oob_mode=oob_mode.skip,
        )
        if ovl_scale is not None:
            rows_2d, src_col = ovl_scale
            row_w = rows_2d.shape[1]
            nisa.dma_copy(
                dst=sst,
                src=rows_2d.ap(
                    pattern=[[row_w, tw], [1, nsg]],
                    offset=src_col,
                    vector_offset=ovl,
                    indirect_dim=0,
                ),
                oob_mode=oob_mode.skip,
            )
        s32 = nl.ndarray((tw, nsg), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=s32, src=sst)
        # group-64 ue8m0 dequant over the NoPE columns; RoPE columns ride
        # unscaled (identical to swa_attention.dequant_group_scales).
        for gi in range(nsg):
            nisa.tensor_scalar(
                dst=f32[0:tw, gi * qgs : (gi + 1) * qgs],
                data=f32[0:tw, gi * qgs : (gi + 1) * qgs],
                op0=nl.multiply,
                operand0=s32[0:tw, gi : gi + 1],
            )
    return f32


def _leg_scores(
    scores_sb,
    leg_off,
    count,
    b,
    rows_hbm,
    ovl_hbm,
    qT,
    lat_dim,
    pieces,
    scale_flat,
    ovl_srcs,
    ovl_scale,
    nsg,
    qgs,
    stage_dtype,
    scale_dtype,
):
    """One attention leg for token ``b``: gather+dequant all row chunks, build
    latT tiles, matmul scores into ``scores_sb[:, leg_off:leg_off+count]``.
    Returns the list of (f32_rows, chunk_off, tw) for the PV pass."""
    n_dc = lat_dim // _PMAX
    H = scores_sb.shape[0]
    # P4-r2 fix (TRIADS-RAW-G3-p4compile-r2.txt line 674 "unsupported
    # expression"): the deployed NKI 0.5.0 parser rejects list comprehensions
    # in kernel code (no kernel-side comprehension exists anywhere in the
    # installed nkilib either — torch-side files only). Explicit append loop;
    # list literals + .append have kernel-side precedent (nkilib ssd_block).
    latT = []
    for _dc in range(n_dc):
        latT.append(
            nl.ndarray((_PMAX, count), dtype=nl.float32, buffer=nl.sbuf)
        )
    chunks = []
    for c0 in range(0, count, _PMAX):
        tw = min(_PMAX, count - c0)
        ids = _load_ids(rows_hbm, b * count + c0, tw)
        ovl = _load_ids(ovl_hbm, b * count + c0, tw)
        f32 = _gather_leg_chunk(
            ids, ovl, tw, lat_dim, pieces, scale_flat, ovl_srcs, ovl_scale,
            nsg, qgs, stage_dtype, scale_dtype,
        )
        for dc in range(n_dc):
            pt = nl.ndarray((_PMAX, tw), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(
                dst=pt[0:_PMAX, 0:tw],
                data=f32[0:tw, dc * _PMAX : (dc + 1) * _PMAX],
            )
            nisa.tensor_copy(dst=latT[dc][0:_PMAX, c0 : c0 + tw], src=pt)
        chunks.append((f32, c0, tw))

    for p0 in range(0, count, _PSUM_F):
        pn = min(_PSUM_F, count - p0)
        ps = nl.ndarray((H, pn), dtype=nl.float32, buffer=nl.psum)
        for dc in range(n_dc):
            nisa.nc_matmul(
                dst=ps,
                stationary=qT[dc],
                moving=latT[dc][0:_PMAX, p0 : p0 + pn],
                accumulate=(dc > 0),
            )
        nisa.tensor_copy(
            dst=scores_sb[0:H, leg_off + p0 : leg_off + p0 + pn], src=ps
        )
    return chunks


@nki.jit
def _mla_decode_tkg_nki(
    q,                # [B, H, 512] bf16/fp32
    aux,              # [B, W + S + 1] fp32: win mask ++ comp mask ++ keep
    win_rows,         # [B, W] uint32 flat swa rows (uint32-max = skip)
    win_ovl,          # [B, W] uint32 rows into cur_win (uint32-max = skip)
    comp_rows,        # [B, S] uint32 flat latent rows
    comp_ovl,         # [B, S] uint32 rows into cur_bundle
    swa_k=None,       # swa cache piece, whole/unreshaped (write-aliased)
    swa_scale=None,
    lat0=None,        # latent pieces (None when shared_lat)
    lat1=None,
    lat2=None,
    lat_scale=None,
    cur_win=None,     # [B, 512] cache-dtype current window rows
    cur_scale=None,   # [B, nsg] current scale rows
    cur_bundle=None,  # [B, 512 + nsg] current compressed bundle
    sink=None,        # [H] fp32
    wr_ids=None,      # [B, 3] uint32 write slots (swa | coarse-lat | coarse-rope)
    scale=1.0,
    nope_dim=448,
    rope_dim=64,
    quant_group_size=64,
    lat_widths=(224, 224, 64),
    shared_lat=False,
    out_fp32=False,
):
    """Fused DSv4-Flash decode attention with the in-kernel cache write.

    Order of effects: ALL cache gathers happen before ANY cache write (the
    current token's own contribution comes from the operand overlay, never
    from re-reading the cache after the write — no in-kernel RAW on cache;
    prereg D3). The NKI dependency tracker orders the write DMAs after the
    gather DMAs through the shared root buffers.
    """
    B, H, D = q.shape
    W = win_rows.shape[1]
    S = comp_rows.shape[1]
    kernel_assert(D == nope_dim + rope_dim, "latent width mismatch")
    kernel_assert(D % _PMAX == 0, "latent width must be a multiple of 128")
    kernel_assert(H <= _PMAX, "query heads exceed one partition tile")
    nsg = nope_dim // quant_group_size
    n_dc = D // _PMAX

    swa_flat = _flat2(swa_k)
    swa_pieces = [(swa_flat, 0, D)]
    swa_scale_flat = _flat2(swa_scale) if swa_scale is not None else None
    if shared_lat:
        lat_pieces = swa_pieces
        lat_scale_flat = swa_scale_flat
    else:
        lat_pieces = []
        _lat_srcs = [lat0, lat1, lat2]
        col = 0
        # P4-r3 fix: no zip / no tuple-unpack targets (parser).
        for _li in range(len(lat_widths)):
            piece = _lat_srcs[_li]
            w = lat_widths[_li]
            if piece is not None:
                lat_pieces.append((_flat2(piece), col, w))
                col += w
        lat_scale_flat = _flat2(lat_scale) if lat_scale is not None else None

    win_ovl_srcs = [(cur_win, 0, 0, D)] if cur_win is not None else []
    win_ovl_scale = (
        (cur_scale, 0)
        if (cur_scale is not None and swa_scale_flat is not None)
        else None
    )
    comp_ovl_srcs = [(cur_bundle, 0, 0, D)] if cur_bundle is not None else []
    comp_ovl_scale = (
        (cur_bundle, D)
        if (cur_bundle is not None and lat_scale_flat is not None)
        else None
    )

    out = nl.ndarray(
        (B, H, D),
        dtype=nl.float32 if out_fp32 else nl.bfloat16,
        buffer=nl.shared_hbm,
        name="mla_decode_tkg_out",
    )
    out_flat = out.reshape((B * H, D))
    q_flat = q.reshape((B * H, D))

    sink_sb = None
    if sink is not None:
        sink_sb = nl.ndarray((H, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=sink_sb, src=sink.ap(pattern=[[1, H], [1, 1]], offset=0))

    for b in range(B):
        # per-token aux row broadcast across the H partitions (mask varies
        # along the free dim, so tensor ops cannot broadcast it; H is small
        # at every venue — 64/world_size in production)
        aux_sb = nl.ndarray((H, W + S + 1), dtype=nl.float32, buffer=nl.sbuf)
        for h in range(H):
            nisa.dma_copy(
                dst=aux_sb[h : h + 1, 0 : W + S + 1],
                src=aux[b : b + 1, 0 : W + S + 1],
            )

        # q_b: load, fp32, pre-scale (fallback scales scores; same product
        # within fp32 rounding, covered by the D5 tolerances), transpose
        q_sb = nl.ndarray((H, D), dtype=q.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=q_sb, src=q_flat[b * H : (b + 1) * H, 0:D])
        q32 = nl.ndarray((H, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=q32, src=q_sb)
        nisa.tensor_scalar(dst=q32, data=q32, op0=nl.multiply, operand0=scale)
        qT = []
        for dc in range(n_dc):
            pt = nl.ndarray((_PMAX, H), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(
                dst=pt[0:_PMAX, 0:H], data=q32[0:H, dc * _PMAX : (dc + 1) * _PMAX]
            )
            qs = nl.ndarray((_PMAX, H), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=qs, src=pt)
            qT.append(qs)

        scores_sb = nl.ndarray((H, W + S), dtype=nl.float32, buffer=nl.sbuf)
        win_chunks = _leg_scores(
            scores_sb, 0, W, b, win_rows, win_ovl, qT, D,
            swa_pieces, swa_scale_flat, win_ovl_srcs, win_ovl_scale,
            nsg, quant_group_size, swa_k.dtype,
            swa_scale.dtype if swa_scale is not None else nl.float32,
        )
        comp_chunks = _leg_scores(
            scores_sb, W, S, b, comp_rows, comp_ovl, qT, D,
            lat_pieces, lat_scale_flat, comp_ovl_srcs, comp_ovl_scale,
            nsg, quant_group_size,
            swa_k.dtype if shared_lat else lat0.dtype,
            (lat_scale.dtype if lat_scale is not None else nl.float32)
            if not shared_lat
            else (swa_scale.dtype if swa_scale is not None else nl.float32),
        )

        # ── ONE fp32 sink softmax over [window ++ compressed] ────────────
        nisa.tensor_tensor(
            dst=scores_sb, data1=scores_sb, data2=aux_sb[0:H, 0 : W + S], op=nl.add
        )
        negmax = nl.ndarray((H, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(
            dst=negmax, op=nl.maximum, data=scores_sb, axis=1, negate=True
        )
        probs = nl.ndarray((H, W + S), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=probs, op=nl.exp, data=scores_sb, bias=negmax)
        denom = nl.ndarray((H, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=denom, op=nl.add, data=probs, axis=1)
        if sink_sb is not None:
            # sink enters the DENOMINATOR only (dsv4_ref/kernel.py:345-348)
            sink_exp = nl.ndarray((H, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=sink_exp, op=nl.exp, data=sink_sb, bias=negmax)
            nisa.tensor_tensor(
                dst=denom, data1=denom, data2=sink_exp, op=nl.add
            )
        recip = nl.ndarray((H, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=recip, data=denom)
        # fold the keep factor (all-masked query → 0 output) into 1/denom
        nisa.tensor_tensor(
            dst=recip,
            data1=recip,
            data2=aux_sb[0:H, W + S : W + S + 1],
            op=nl.multiply,
        )

        # ── PV: out[b] = (probs @ latent) / denom ────────────────────────
        outp = nl.ndarray((H, D), dtype=nl.float32, buffer=nl.psum)
        first = True
        # P4-r3 fix: index loops, no tuple-unpack targets (parser).
        _leg_offs = [0, W]
        _leg_chunks = [win_chunks, comp_chunks]
        for _leg in range(2):
            leg_off = _leg_offs[_leg]
            chunks = _leg_chunks[_leg]
            for _ci in range(len(chunks)):
                f32c = chunks[_ci][0]
                c0 = chunks[_ci][1]
                tw = chunks[_ci][2]
                pT = nl.ndarray((tw, H), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_transpose(
                    dst=pT[0:tw, 0:H],
                    data=probs[0:H, leg_off + c0 : leg_off + c0 + tw],
                )
                pTs = nl.ndarray((tw, H), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=pTs, src=pT)
                nisa.nc_matmul(
                    dst=outp,
                    stationary=pTs[0:tw, 0:H],
                    moving=f32c[0:tw, 0:D],
                    accumulate=(not first),
                )
                first = False

        outs = nl.ndarray((H, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=outs, src=outp)
        nisa.tensor_scalar(
            dst=outs, data=outs, op0=nl.multiply, operand0=recip
        )
        outb = nl.ndarray(
            (H, D), dtype=nl.float32 if out_fp32 else nl.bfloat16, buffer=nl.sbuf
        )
        nisa.tensor_copy(dst=outb, src=outs)
        nisa.dma_copy(dst=out_flat[b * H : (b + 1) * H, 0:D], src=outb)

    # ── in-kernel must-alias cache write (R2 idiom), traced after all
    # gathers; mirrors _masked_write_rows piece/slot/source mapping and its
    # zero-padding to the stored width, so written bytes are BITWISE equal ──
    written = []
    if wr_ids is not None:
        plan = []
        if cur_win is not None:
            plan.append((swa_flat, 0, cur_win, 0, cur_win.shape[1]))
            written.append(swa_k)
            if swa_scale_flat is not None and cur_scale is not None:
                plan.append((swa_scale_flat, 0, cur_scale, 0, nsg))
                written.append(swa_scale)
        if cur_bundle is not None and not shared_lat:
            col = 0
            # P4-r3 fix: no enumerate / no tuple-unpack targets (parser).
            for _wp in range(len(lat_pieces)):
                flat = lat_pieces[_wp][0]
                w = lat_pieces[_wp][2]
                slot_col = 1 if col + w <= nope_dim else 2
                plan.append((flat, slot_col, cur_bundle, col, w))
                written.append(_lat_srcs[_wp])
                col += w
            if lat_scale_flat is not None:
                plan.append((lat_scale_flat, 2, cur_bundle, col, nsg))
                written.append(lat_scale)
        for tb0 in range(0, B, _PMAX):
            tb = min(_PMAX, B - tb0)
            for _pl in range(len(plan)):
                flat = plan[_pl][0]
                slot_col = plan[_pl][1]
                src2d = plan[_pl][2]
                src_col = plan[_pl][3]
                w_src = plan[_pl][4]
                wid = nl.ndarray((tb, 1), dtype=nl.uint32, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=wid,
                    src=wr_ids.ap(
                        pattern=[[wr_ids.shape[1], tb], [1, 1]],
                        offset=tb0 * wr_ids.shape[1] + slot_col,
                    ),
                )
                piece_w = flat.shape[1]
                kernel_assert(w_src <= piece_w, "write row wider than piece")
                wst = nl.ndarray((tb, piece_w), dtype=flat.dtype, buffer=nl.sbuf)
                nisa.memset(wst, 0.0)
                nisa.dma_copy(
                    dst=wst[0:tb, 0:w_src],
                    src=src2d[tb0 : tb0 + tb, src_col : src_col + w_src],
                )
                nisa.dma_copy(
                    dst=flat.ap(
                        pattern=[[piece_w, tb], [1, piece_w]],
                        offset=0,
                        vector_offset=wid,
                        indirect_dim=0,
                    ),
                    src=wst[0:tb, 0:piece_w],
                    oob_mode=oob_mode.skip,
                )

    # returning the mutated inputs keeps the output arity identical on
    # the simulator and HOP paths (sibling precedent:
    # attention_decode.py:1240-1246 "always returns (output, K, V)").
    # P4-r3 fix: starred expansion is parser-illegal ("tuple expansion is
    # not supported") — explicit arity chain over the static write count
    # (at most 6 written pieces: swa, swa_scale, lat0/1/2, lat_scale).
    n_w = len(written)
    if n_w == 0:
        return out
    if n_w == 1:
        return out, written[0]
    if n_w == 2:
        return out, written[0], written[1]
    if n_w == 3:
        return out, written[0], written[1], written[2]
    if n_w == 4:
        return out, written[0], written[1], written[2], written[3]
    if n_w == 5:
        return (out, written[0], written[1], written[2], written[3],
                written[4])
    return (out, written[0], written[1], written[2], written[3], written[4],
            written[5])
