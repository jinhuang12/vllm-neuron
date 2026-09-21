# SPDX-License-Identifier: Apache-2.0
"""Sparse MLA latent attention over topk-selected cache rows, as NKI kernels.

Per query, over the selected cache rows::

    c_g     = c_kv[topk_indices[q]]                  # [K, L], gathered
    scores  = q_lift[q] @ c_g.T   (+ RoPE limb)      # [H, K]
    weights = softmax(scores * softmax_scale)        # [H, K]
    out[q]  = weights @ c_g                          # [H, L]

This is the absorbed-latent path: attention runs in the latent rank L and the V-up
projection belongs to the caller. The output is natural row-major ``[S, H, L]``
float32, and arithmetic is fp32 throughout.

The latent rides the partition axis because ``nc_matmul`` contracts that axis and
``nc_n_gather`` gathers within a partition, so one gather per latent tile selects K
columns for all 128 latent components at once; MM2 contracts K, so both operands are
transposed on chip between the two matmuls. A RoPE width of zero is a first-class
path: the limb is elided at trace time rather than sized to zero. The gather uses
``nc_n_gather`` rather than tensor indirection, which needs NeuronCore-v4.

Three kernel bodies serve three geometries -- untiled, latent-tiled and
selected-row-tiled -- and the seam picks one from the shapes alone. The row-tiled
entries take compile-time ``BLOCK_N`` (a multiple of 128 up to 512) and ``STREAM_KV``
options. There is no torch attention fallback; an inadmissible geometry raises.
:func:`mla_sparse_attention_torch_oracle` is a CPU reference for tests only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.utils.neuron_utils import can_run_kernel, values_are_readable

logger = logging.getLogger(__name__)

#: Partition-axis extent, ``nl.tile_size.pmax``. The latent axis is tiled to this in
#: MM1 because ``nc_matmul`` contracts the partition axis, and the gather then reads
#: 128 latent components per instruction.
LATENT_TILE = 128

#: Partition-axis extent again, under the name of the axis that occupies it in MM2.
#: MM2 contracts K, so the selected keys move onto partitions and K is chunked here.
#: Equal to :data:`LATENT_TILE` but a different axis, so kept as its own name.
KEY_CHUNK = 128

#: Stationary free-axis extent, ``nl.tile_size.gemm_stationary_fmax``. The head axis
#: rides the stationary operand in both matmuls, so this bounds H.
HEAD_MAX = 128

#: Free-axis alignment, in float32 elements, of the per-block Q tiles the matmuls
#: read one query at a time: their width is kept at a whole 32 bytes. Eight float32
#: elements are 32 bytes.
DMA_TRANSPOSE_ALIGN = 8
#: Elements per staged row block: 16 rows of a 2-byte or a 4-byte dtype are whole 32-byte lines.
STAGE_ALIGN = 16

#: Source rows per DMA transpose. The hardware descriptor generator takes a transpose
#: whose source is 16 rows of a 2-byte dtype, at most 128 wide, so every transpose
#: here lands 16 source rows and its destination slice starts every 16 columns --
#: 32 bytes apart for a 2-byte source, 64 for float32. The runtime refuses a
#: host-expanded transpose descriptor whose destination offset is not 32-byte
#: aligned ("transpose dest offset <n> must be 32B aligned").
DGE_TRANSPOSE_ROWS = 16

#: Moving free-axis extent, ``nl.tile_size.gemm_moving_fmax``. K rides the moving
#: operand in MM1 and the latent rides it in MM2; both axes are tiled to this width.
MOVING_MAX = 512

#: Where the tile extents above come from; quoted in refusal messages.
_TILE_PROVENANCE = "nl.tile_size.{pmax, gemm_stationary_fmax, gemm_moving_fmax}"

#: The target checkpoint's latent rank, ``kv_lora_rank``. Informational only: the
#: kernel accepts any admissible L and does not compare against this.
TARGET_LATENT_RANK = 512

#: The target checkpoint's MLA RoPE width, ``qk_rope_head_dim``. Zero is a real value
#: here, not a placeholder.
TARGET_ROPE_WIDTH = 0

#: The index the selector writes for a selected-row column that carries no token, and
#: the only negative index this seam admits. ``functional/dsa/index_expand.py`` emits
#: it and every consumer must mask on it.
SENTINEL_INDEX = -1

#: How far below its row maximum a masked score is pushed, in exponent units. The bias
#: the kernels add is this divided by ``softmax_scale``, because ``nisa.activation``
#: computes ``exp(scale * data + bias)``; a bias fixed in score units would shrink with
#: the scale and a small scale would give a masked column real weight. 200 is well past
#: fp32's exp underflow point -- ``exp(-104)`` is already 0.0 there.
_SENTINEL_EXP_FLOOR = 200.0


class MlaSparseAttentionError(ValueError):
    """Raised for a geometry this kernel does not serve; there is no torch fallback."""


@dataclass
class _MlaSparseDispatchCounters:
    """Per-process record of how the seam below was reached."""

    nki_dispatch: int = 0
    torch_fallback: int = 0


_MLA_SPARSE_COUNTERS = _MlaSparseDispatchCounters()


def reset_mla_sparse_dispatch_counters() -> None:
    """Zero this seam's counters."""
    _MLA_SPARSE_COUNTERS.nki_dispatch = 0
    _MLA_SPARSE_COUNTERS.torch_fallback = 0


def mla_sparse_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last reset.

    ``torch_fallback`` is always ``0``: this module has no torch attention route, and
    an inadmissible geometry raises instead.
    """
    return (
        _MLA_SPARSE_COUNTERS.nki_dispatch,
        _MLA_SPARSE_COUNTERS.torch_fallback,
    )


@torch._dynamo.assume_constant_result
def _count_nki_dispatch() -> None:
    """Count one kernel dispatch, off the traced graph: a store Dynamo traces becomes a guard."""
    _MLA_SPARSE_COUNTERS.nki_dispatch += 1


# The tile helpers take positional shapes only. The NKI compiler drops a keyword-only
# parameter instead of applying its default, so `def _sbuf(*shape, dtype=nl.float32)`
# leaves `dtype` unbound ("unbound variable 'dtype'"), and a `**kwargs` form is
# refused as well. Each dtype therefore gets its own helper name.
def _sbuf(*shape: int):
    return nl.ndarray(tuple(shape), dtype=nl.float32, buffer=nl.sbuf)


def _aligned(width: int, block: int = DMA_TRANSPOSE_ALIGN) -> int:
    """``width`` rounded up to a whole ``block``: :data:`DMA_TRANSPOSE_ALIGN` elements by default."""
    blocks = (width + block - 1) // block
    return blocks * block


def _stage(hbm, parts: int, width: int):
    """A ``[parts, width]`` view of an SBUF tile in ``hbm``'s own dtype, its row padded to whole 32-byte lines."""
    return nl.ndarray((parts, _aligned(width, STAGE_ALIGN)), dtype=hbm.dtype, buffer=nl.sbuf)[:, 0:width]


def _scalar(parts: int):
    """A ``[parts, 1]`` float32 view of a tile whose row is one whole 32-byte line."""
    return _sbuf(parts, DMA_TRANSPOSE_ALIGN)[:, 0:1]


def _transpose_rows(dst, src_hbm, row_stride, rows, width, offset):
    """Transpose ``rows`` source rows of ``width`` elements onto partitions, 16 rows per DMA."""
    for r0 in range(0, rows, DGE_TRANSPOSE_ROWS):
        n = min(DGE_TRANSPOSE_ROWS, rows - r0)
        nisa.dma_transpose(
            dst=dst[:, r0:r0 + n],
            src=src_hbm.ap(pattern=[[row_stride, n], [1, width]],
                           offset=offset + r0 * row_stride),
        )


#: Window rows one staging DMA moves: the SBUF partition bound. A page wider than this
#: is staged in that many pieces, and the bank is addressed in whole pieces, so every
#: runtime offset is a piece index and no access pattern carries a trace-time offset
#: and a tensor-borne one at once.
STAGE_ROWS = LATENT_TILE


def _clamped_page(table_hbm, entry: int):
    """A ``[1, 1]`` int32 tile holding ``block_table[entry]``, the -1 pad clamped onto page 0.

    Each tile is declared a whole 32-byte line wide and one column of it is used: the
    allocator packs tiles back to back per partition, so a four-byte row would move
    every tile placed after it off the line.
    """
    raw = _sbuf_i32(1, _aligned(1))[:, 0:1]
    nisa.dma_copy(dst=raw, src=table_hbm.ap(pattern=[[1, 1], [1, 1]], offset=entry))
    floor = _sbuf_i32(1, _aligned(1))[:, 0:1]
    nisa.memset(dst=floor, value=SENTINEL_INDEX)
    live = _sbuf_i32(1, _aligned(1))[:, 0:1]
    nisa.tensor_tensor(dst=live, data1=floor, data2=raw, op=nl.less)
    held = _sbuf_i32(1, _aligned(1))[:, 0:1]
    nisa.tensor_tensor(dst=held, data1=raw, data2=live, op=nl.multiply)
    return held


def _piece_of_page(page, pieces: int, piece: int):
    """A ``[1, 1]`` int32 tile holding ``page * pieces + piece``: which whole piece of the bank to read."""
    held = _sbuf_i32(1, _aligned(1))[:, 0:1]
    nisa.tensor_scalar(dst=held, data=page, op0=nl.multiply, operand0=pieces)
    nisa.tensor_scalar(dst=held, data=held, op0=nl.add, operand0=piece)
    return held


def _write_row(offset_hbm, ahead: int):
    """A ``[1, 1]`` int32 tile holding this step's own write row, plus ``ahead`` rows."""
    held = _sbuf_i32(1, _aligned(1))[:, 0:1]
    nisa.dma_copy(dst=held, src=offset_hbm.ap(pattern=[[1, 1], [1, 1]], offset=0))
    if ahead > 0:
        nisa.tensor_scalar(dst=held, data=held, op0=nl.add, operand0=ahead)
    return held


#: Rows the staged window carries beyond the window itself; nothing selects, reduces or
#: contracts them. The overlay's destination row is a runtime value, and the tracer
#: reads such a pattern's end as an index into the staged tile, so a step whose rows
#: fill the window would reach one element past the tile and be refused. These rows
#: keep that end inside the tile. The pad is a whole :data:`STAGE_ALIGN` block because
#: the bodies size their own SBUF tiles from this tile's row count, and 16 rows of a
#: 2- or 4-byte dtype are whole 32-byte lines.
STAGE_PAD = STAGE_ALIGN


def _staged_window(bank_hbm, table_hbm, written_hbm, offset_hbm, page_size: int):
    """Assemble the window a block table names in HBM, with this step's own rows overlaid.

    The bodies read the returned tile exactly as an unpaged cache. It carries
    :data:`STAGE_PAD` rows beyond the window, which no selected row reaches: the seam
    bounds a selected row by the window, ``pages * page_size``, never by this tile's
    row count.

    The window is staged in HBM rather than straight into SBUF because the overlay's
    destination row is a runtime value; one HBM tile makes one runtime offset legal for
    the page reads and the overlay alike. The pages are read before the overlay: this
    step's rows sit at positions the pages also cover, so a page copy issued afterwards
    would put stale cache back over them.
    """
    latent = bank_hbm.shape[1]
    pages = table_hbm.shape[0]
    tokens = 0 if written_hbm is None else written_hbm.shape[0]
    span = page_size if page_size < STAGE_ROWS else STAGE_ROWS
    pieces = page_size // span
    banked = bank_hbm.reshape((bank_hbm.shape[0] // span, span, latent))
    staged = nl.ndarray((pages * page_size + STAGE_PAD, latent), dtype=bank_hbm.dtype,
                        buffer=nl.private_hbm)
    for entry in range(pages):
        page = _clamped_page(table_hbm, entry)
        for piece in range(pieces):
            hold = nl.ndarray((span, latent), dtype=bank_hbm.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=hold,
                          src=banked.ap(pattern=[[latent, span], [1, latent]], offset=0,
                                        scalar_offset=_piece_of_page(page, pieces, piece),
                                        indirect_dim=0))
            nisa.dma_copy(dst=staged.ap(pattern=[[latent, span], [1, latent]],
                                        offset=(entry * page_size + piece * span) * latent),
                          src=hold)
    for start in range(0, tokens, STAGE_ROWS):
        rows = tokens - start if tokens - start < STAGE_ROWS else STAGE_ROWS
        fresh = nl.ndarray((rows, latent), dtype=bank_hbm.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=fresh, src=written_hbm.ap(pattern=[[latent, rows], [1, latent]],
                                                    offset=start * latent))
        nisa.dma_copy(dst=staged.ap(pattern=[[latent, rows], [1, latent]], offset=0,
                                    scalar_offset=_write_row(offset_hbm, start), indirect_dim=0),
                      src=fresh)
    return staged


def _window_of(c_kv_hbm, table_hbm, written_hbm, offset_hbm, page_size: int):
    """The cache the body reads: the staged window when a block table is given, else the cache itself."""
    if table_hbm is None:
        return c_kv_hbm
    return _staged_window(c_kv_hbm, table_hbm, written_hbm, offset_hbm, page_size)


def _queries_per_block(seq: int, heads: int) -> int:
    """Queries whose Q rows fill one 16-row transpose, when the head count and sequence allow."""
    if DGE_TRANSPOSE_ROWS % heads == 0 and seq % (DGE_TRANSPOSE_ROWS // heads) == 0:
        return DGE_TRANSPOSE_ROWS // heads
    return 1


def _sbuf_u32(*shape: int):
    return nl.ndarray(tuple(shape), dtype=nl.uint32, buffer=nl.sbuf)


# The sentinel mask needs the signed tile: -1 read as uint32 is 4,294,967,295, which
# no comparison against -1 sees.
def _sbuf_i32(*shape: int):
    return nl.ndarray(tuple(shape), dtype=nl.int32, buffer=nl.sbuf)


def _sentinel_scratch(parts, width, heads):
    """The sentinel working set, shared by all three bodies.

    Returns, in order: the -1 comparand, the signed index tile, the 0/1 valid tile,
    the clamped index tile, the valid mask in float, the additive score bias, the
    masked scores, the masked probabilities.

    The mask is built at ``parts`` and sliced down to ``heads``: every partition of
    the index tile holds the same K offsets, so partitions ``0:heads`` are exactly the
    per-column mask a score tile needs, and ``heads <= HEAD_MAX == LATENT_TILE`` keeps
    that slice available. Every parameter is positional (see :func:`_sbuf`).
    """
    comparand = _sbuf_i32(parts, width)
    nisa.memset(dst=comparand, value=SENTINEL_INDEX)
    return (comparand, _sbuf_i32(parts, width), _sbuf_i32(parts, width),
            _sbuf_i32(parts, width), _sbuf(heads, width), _sbuf(heads, width),
            _sbuf(heads, width), _sbuf(heads, width))


def _mask_sentinel(topk_hbm, offset, width, heads, sentinel_bias, sen, idx_sb):
    """Read one query's selected rows, clamp the sentinel columns, build their mask.

    ``-1 < index`` is 1 for a real cache row and 0 for the sentinel, and the clamp is
    that mask times the index, so a sentinel column's offset becomes 0 and the gather
    stays inside the cache. What it gathers is irrelevant, because its probability is
    forced to zero; all that matters is that the load is legal.

    The bias is ``(valid - 1) * sentinel_bias``: exactly 0.0 where a token lives and
    ``-sentinel_bias`` where none does, by two scalar ops against Python constants --
    ``tensor_scalar``'s ``operand0`` must be a Python scalar and never a tile.
    """
    nisa.tensor_copy(
        dst=sen[1][:, 0:width],
        src=nl.load(
            topk_hbm.ap(pattern=[[0, LATENT_TILE], [1, width]], offset=offset),
            dtype=nl.int32,
        ),
    )
    nisa.tensor_tensor(dst=sen[2][:, 0:width], data1=sen[0][:, 0:width],
                       data2=sen[1][:, 0:width], op=nl.less)
    nisa.tensor_tensor(dst=sen[3][:, 0:width], data1=sen[1][:, 0:width],
                       data2=sen[2][:, 0:width], op=nl.multiply)
    nisa.tensor_copy(dst=idx_sb[:, 0:width], src=sen[3][:, 0:width])
    nisa.tensor_copy(dst=sen[4][:, 0:width], src=sen[2][0:heads, 0:width])
    nisa.tensor_scalar(dst=sen[5][:, 0:width], data=sen[4][:, 0:width],
                       op0=nl.add, operand0=-1.0)
    nisa.tensor_scalar(dst=sen[5][:, 0:width], data=sen[5][:, 0:width],
                       op0=nl.multiply, operand0=sentinel_bias)


def _psum(*shape: int):
    return nl.ndarray(tuple(shape), dtype=nl.float32, buffer=nl.psum)


def _attention_body(q_lift_hbm, c_kv_hbm, topk_hbm, softmax_scale, out_hbm,
                    q_pe_hbm=None, k_pe_hbm=None):
    """Trace the sparse latent attention. Shared by both jit entry points.

    The RoPE limb is elided at trace time, not masked at run time: with no RoPE half
    (``q_pe_hbm`` and ``k_pe_hbm`` are ``None``) no RoPE buffer is allocated, no RoPE
    gather is issued and MM1 emits one fewer matmul. A zero-extent SBUF tile cannot be
    allocated, so a kernel that kept the limb and sized it 0 would not trace.

    A column of ``topk_hbm`` holding :data:`SENTINEL_INDEX` carries no token: it is
    gathered from cache row 0 so the load is legal, then given exactly zero
    probability. A query whose every column is the sentinel produces exact zeros.

    Shapes:
        q_lift_hbm  [S, H, L]      the absorbed Q latent, per head
        c_kv_hbm    [S_kv, L]      the latent KV cache
        topk_hbm    [S, K] int32   the selected cache rows, per query; -1 means none
        q_pe_hbm    [S, H, R]      present only when R > 0
        k_pe_hbm    [S_kv, R]      present only when R > 0
        out_hbm     [S, H, L]      written once per query
    """
    seq, heads, latent = q_lift_hbm.shape
    s_kv = c_kv_hbm.shape[0]
    topk = topk_hbm.shape[1]
    n_latent = latent // LATENT_TILE
    n_chunks = topk // KEY_CHUNK
    rope = 0 if q_pe_hbm is None else q_pe_hbm.shape[2]

    # ---- the cache, transposed onto partitions once for the whole call ----------
    # HBM holds it [S_kv, L]; MM1 and the gather both need [L_partition, S_kv]. The
    # transposes land in the source dtype, 16 rows per DMA, and one copy per latent
    # tile widens to float32; the cache is read-only for the rest of the call, so
    # this cost is per call and not per query.
    #
    # Built by a loop, not a comprehension: a comprehension inside a traced kernel is
    # refused at specialization with "unsupported expression".
    c_sb = []
    for _ in range(n_latent):
        c_sb.append(_sbuf(LATENT_TILE, s_kv))
    c_stage = _stage(c_kv_hbm, LATENT_TILE, s_kv)
    for li in range(n_latent):
        _transpose_rows(c_stage, c_kv_hbm, latent, s_kv, LATENT_TILE, li * LATENT_TILE)
        nisa.tensor_copy(dst=c_sb[li], src=c_stage)
    k_pe_sb = None
    if rope > 0:
        k_pe_sb = _sbuf(rope, s_kv)
        k_pe_stage = _stage(k_pe_hbm, rope, s_kv)
        _transpose_rows(k_pe_stage, k_pe_hbm, rope, s_kv, rope, 0)
        nisa.tensor_copy(dst=k_pe_sb, src=k_pe_stage)

    # ---- per-query working set, allocated once and reused across the loop -------
    qpb = _queries_per_block(seq, heads)
    block = qpb * heads
    idx_sb = _sbuf_u32(LATENT_TILE, topk)
    q_stage = []
    for _ in range(n_latent):
        q_stage.append(_stage(q_lift_hbm, LATENT_TILE, block))
    q_lift_t = _sbuf(LATENT_TILE, n_latent, _aligned(block))
    c_g = _sbuf(LATENT_TILE, n_latent, topk)
    c_g_t = _sbuf(KEY_CHUNK, n_chunks, latent)
    p_t = _sbuf(KEY_CHUNK, n_chunks, _aligned(heads))
    p = _sbuf(heads, topk)
    neg_row_max = _scalar(heads)
    exp_bias = _scalar(heads)
    row_sum = _scalar(heads)
    recip = _scalar(heads)
    out_sb = _sbuf(heads, latent)
    q_pe_t = _sbuf(rope, _aligned(block)) if rope > 0 else None
    q_pe_stage = _stage(q_pe_hbm, rope, block) if rope > 0 else None
    k_pe_g = _sbuf(rope, topk) if rope > 0 else None

    # The sentinel bias is in exponent units divided by the scale, because `activation`
    # below computes ``exp(scale * data + bias)``. `softmax_scale` is a Python float
    # inside the traced body, so this division happens at trace time and emits nothing.
    sen = _sentinel_scratch(LATENT_TILE, topk, heads)
    valid_f = sen[4]
    mask_bias = sen[5]
    scores_m = sen[6]
    p_m = sen[7]
    sentinel_bias = _SENTINEL_EXP_FLOOR / softmax_scale

    # ---- the queries, in blocks whose Q rows fill one 16-row transpose ----------
    # One transpose per latent tile lands the block's Q rows in the source dtype and
    # one copy widens them; each query then reads its own columns of the block tile.
    for qb in nl.affine_range(seq // qpb):
        q0 = qb * qpb
        for li in range(n_latent):
            _transpose_rows(q_stage[li], q_lift_hbm, latent, block, LATENT_TILE,
                            q0 * heads * latent + li * LATENT_TILE)
            nisa.tensor_copy(dst=q_lift_t[:, li, 0:block], src=q_stage[li])
        if rope > 0:
            _transpose_rows(q_pe_stage, q_pe_hbm, rope, block, rope, q0 * heads * rope)
            nisa.tensor_copy(dst=q_pe_t[:, 0:block], src=q_pe_stage)
        for qi in range(qpb):
            q_idx = q0 + qi
            h0 = qi * heads
            # ---- this query's selected rows, replicated to every partition ----------
            # `nc_n_gather` gathers within a partition and reads its offsets from the
            # same partition, so all 128 latent partitions need the same K offsets. A
            # zero-stride partition read is one DMA and replicates them for free; the
            # alternative is a shuffle plus a fan-out copy.
            _mask_sentinel(topk_hbm, q_idx * topk, topk, heads, sentinel_bias, sen, idx_sb)

            # ---- gather the latent cache rows: one instruction per latent tile ------
            for li in range(n_latent):
                nisa.nc_n_gather(dst=c_g[:, li, :], data=c_sb[li], indices=idx_sb)

            # ---- MM1: scores[H, K] = sum over latent tiles of q_lift_t.T @ c_g ------
            scores_ps = _psum(heads, topk)
            for li in range(n_latent):
                nisa.nc_matmul(
                    dst=scores_ps,
                    stationary=q_lift_t[:, li, h0:h0 + heads],
                    moving=c_g[:, li, :],
                    accumulate=(li > 0),
                )
            if rope > 0:
                nisa.nc_n_gather(dst=k_pe_g, data=k_pe_sb, indices=idx_sb[0:rope, :])
                nisa.nc_matmul(dst=scores_ps, stationary=q_pe_t[:, h0:h0 + heads],
                               moving=k_pe_g,
                               accumulate=True)

            # ---- the gathered cache, transposed for MM2 -----------------------------
            # Hoisted above the softmax on purpose: it depends only on the gather, so it
            # is work the engines can overlap with the softmax chain rather than wait on.
            for ck in range(n_chunks):
                ks = ck * KEY_CHUNK
                for li in range(n_latent):
                    c_g_t_ps = _psum(KEY_CHUNK, LATENT_TILE)
                    nisa.nc_transpose(dst=c_g_t_ps, data=c_g[:, li, ks:ks + KEY_CHUNK])
                    nisa.tensor_copy(
                        dst=c_g_t[:, ck, li * LATENT_TILE:(li + 1) * LATENT_TILE],
                        src=c_g_t_ps,
                    )

            # ---- the sentinel columns leave the softmax, before the max -------------
            # Applied to the raw scores because `activation` scales its data, which caps
            # a masked column's exponent at -`_SENTINEL_EXP_FLOOR` whatever scale the
            # caller passed; and before the max, because a max taken over unmasked
            # scores could be a sentinel column's and would drag every real column's
            # exponent down with it. With no sentinel both masking steps are exact
            # identities in fp32: `mask_bias` is 0.0 and `valid_f` is 1.0.
            nisa.tensor_tensor(dst=scores_m, data1=scores_ps, data2=mask_bias, op=nl.add)

            # ---- softmax over K, per head row --------------------------------------
            # The max is taken negated and then scaled, so `activation` can fold the
            # subtraction into its bias and produce the row sum in the same pass. The
            # scale multiplies the raw scores, so the bias must carry the same factor.
            nisa.tensor_reduce(dst=neg_row_max, op=nl.maximum, data=scores_m, axis=1,
                               negate=True)
            nisa.tensor_scalar(dst=exp_bias, data=neg_row_max, op0=nl.multiply,
                               operand0=softmax_scale, engine=nisa.engine.vector)
            nisa.activation(dst=p, op=nl.exp, data=scores_m, bias=exp_bias,
                            scale=softmax_scale, reduce_op=nl.add, reduce_res=row_sum,
                            reduce_cmd=nisa.reduce_cmd.reset_reduce)
            nisa.reciprocal(dst=recip, data=row_sum)

            # Masked probabilities are exactly zero, not merely small. The bias alone
            # underflows the exp for a row with at least one real column, but a wholly
            # sentinel row's own maximum rebases its exponent to 0 and the exp returns 1,
            # so the zero is multiplied in. That row's output is then exact zeros with no
            # divide-by-zero: its numerator is 0 while `row_sum` stays at least 1. `p` is
            # left alone and MM2 reads `p_m`, avoiding a write back onto an operand.
            nisa.tensor_tensor(dst=p_m, data1=p, data2=valid_f, op=nl.multiply)

            # ---- MM2: out[H, L] = p[H, K] @ c_g[K, L], K contracted on partitions ---
            for ck in range(n_chunks):
                ks = ck * KEY_CHUNK
                p_t_ps = _psum(KEY_CHUNK, heads)
                nisa.nc_transpose(dst=p_t_ps, data=p_m[:, ks:ks + KEY_CHUNK])
                nisa.tensor_copy(dst=p_t[:, ck, 0:heads], src=p_t_ps)

            pv_ps = _psum(heads, latent)
            for ck in range(n_chunks):
                nisa.nc_matmul(dst=pv_ps, stationary=p_t[:, ck, 0:heads],
                               moving=c_g_t[:, ck, :], accumulate=(ck > 0))

            # The softmax denominator is applied here rather than to `p`, so it costs one
            # pass over [H, L] instead of one over [H, K], and no division happens before
            # the accumulation.
            nisa.tensor_scalar(dst=out_sb, data=pv_ps, op0=nl.multiply, operand0=recip,
                               engine=nisa.engine.vector)
            nl.store(
                out_hbm.ap(pattern=[[latent, heads], [1, latent]],
                           offset=q_idx * heads * latent),
                value=out_sb,
            )


@nki.jit
def mla_sparse_attention_nope_kernel(q_lift_hbm, c_kv_hbm, topk_hbm, softmax_scale,
                                     block_table_hbm=None, written_hbm=None,
                                     write_offset_hbm=None, page_size=0):
    """The R == 0 entry point: sparse latent attention with no RoPE limb.

    A separate entry point rather than a flag because the limb is elided at trace
    time; see :func:`_attention_body`.
    """
    seq, heads, latent = q_lift_hbm.shape
    out_hbm = nl.ndarray((seq, heads, latent), dtype=nl.float32, buffer=nl.shared_hbm)
    window_hbm = _window_of(c_kv_hbm, block_table_hbm, written_hbm, write_offset_hbm,
                            page_size)
    _attention_body(q_lift_hbm, window_hbm, topk_hbm, softmax_scale, out_hbm)
    return out_hbm


@nki.jit
def mla_sparse_attention_rope_kernel(q_lift_hbm, q_pe_hbm, c_kv_hbm, k_pe_hbm,
                                     topk_hbm, softmax_scale):
    """The R > 0 entry point, for a checkpoint that carries an MLA RoPE half.

    Shares every line of arithmetic with the NoPE entry point above.
    """
    seq, heads, latent = q_lift_hbm.shape
    out_hbm = nl.ndarray((seq, heads, latent), dtype=nl.float32, buffer=nl.shared_hbm)
    _attention_body(q_lift_hbm, c_kv_hbm, topk_hbm, softmax_scale, out_hbm,
                    q_pe_hbm=q_pe_hbm, k_pe_hbm=k_pe_hbm)
    return out_hbm


# --------------------------------------------------------------------------- #
# The latent-tiled path, for a latent rank that does not fit one tile.
#
# The body above requires the latent rank to be a multiple of 128 and to fit one
# MM2 moving tile of 512. This body tiles both axes instead, each with a ragged
# last tile: the partition axis in tiles of 128 (MM1 contracts the latent, so 2,051
# needs 17 tiles, the last 3 deep) and the MM2 moving free axis in tiles of 512
# (the latent is MM2's output free axis, so 2,051 needs 5 tiles, the last 3 wide).
# Nothing is padded: every primitive used here accepts a ragged extent, and adding
# exact zeros to a short contraction would be bit-identical anyway.
# --------------------------------------------------------------------------- #


@dataclass
class _MlaSparseTiledDispatchCounters:
    """Per-process record of how the latent-tiled path was reached.

    Additive to the seam counter, not exclusive: a tiled call also counts as a seam
    dispatch, so the two compose rather than partition the traffic.
    """

    nki_dispatch: int = 0
    torch_fallback: int = 0


_MLA_SPARSE_TILED_COUNTERS = _MlaSparseTiledDispatchCounters()


def reset_mla_sparse_tiled_dispatch_counters() -> None:
    """Zero the latent-tiled counters; the seam reset does not touch these."""
    _MLA_SPARSE_TILED_COUNTERS.nki_dispatch = 0
    _MLA_SPARSE_TILED_COUNTERS.torch_fallback = 0


def mla_sparse_tiled_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` for the latent-tiled path since its last reset.

    ``torch_fallback`` is always ``0``: there is no torch attention route here.
    """
    return (
        _MLA_SPARSE_TILED_COUNTERS.nki_dispatch,
        _MLA_SPARSE_TILED_COUNTERS.torch_fallback,
    )


@torch._dynamo.assume_constant_result
def _count_tiled_nki_dispatch() -> None:
    """Count one kernel dispatch, off the traced graph: a store Dynamo traces becomes a guard."""
    _MLA_SPARSE_TILED_COUNTERS.nki_dispatch += 1


def _latent_tiles(latent: int) -> tuple[tuple[int, int], ...]:
    """``(offset, extent)`` per partition-axis latent tile; the last may be ragged.

    At 2,051 this returns 17 tiles, the last of extent 3.
    """
    tiles = []
    offset = 0
    while offset < latent:
        tiles.append((offset, min(LATENT_TILE, latent - offset)))
        offset += LATENT_TILE
    return tuple(tiles)


def _output_tiles(latent: int) -> tuple[tuple[int, int], ...]:
    """``(offset, extent)`` per MM2 moving-axis output tile; the last may be ragged.

    A different tiling of the same axis: the latent rides partitions in MM1 and the
    moving free axis in MM2, and those two axes have different extents. At 2,051 this
    returns 5 tiles, the last of extent 3.
    """
    tiles = []
    offset = 0
    while offset < latent:
        tiles.append((offset, min(MOVING_MAX, latent - offset)))
        offset += MOVING_MAX
    return tuple(tiles)


def _attention_body_tiled(q_lift_hbm, c_kv_hbm, topk_hbm, softmax_scale, out_hbm,
                          q_pe_hbm=None, k_pe_hbm=None):
    """Trace sparse latent attention with the latent rank tiled on both its axes.

    Same arithmetic as the untiled body and the same RoPE elision at trace time. Every
    latent-indexed partition buffer is a list of tiles whose last member may be ragged,
    instead of one uniform 3-D buffer, because a ragged tile cannot live in a uniform
    buffer.

    Shapes are the untiled body's, with the latent rank unconstrained above 1:
        q_lift_hbm  [S, H, L]      the absorbed Q latent, per head
        c_kv_hbm    [S_kv, L]      the latent KV cache
        topk_hbm    [S, K] int32   the selected cache rows, per query
        q_pe_hbm    [S, H, R]      present only when R > 0
        k_pe_hbm    [S_kv, R]      present only when R > 0
        out_hbm     [S, H, L]      written once per query
    """
    seq, heads, latent = q_lift_hbm.shape
    s_kv = c_kv_hbm.shape[0]
    topk = topk_hbm.shape[1]
    n_chunks = topk // KEY_CHUNK
    rope = 0 if q_pe_hbm is None else q_pe_hbm.shape[2]
    lat_tiles = _latent_tiles(latent)
    out_tiles = _output_tiles(latent)

    # ---- the cache, transposed onto partitions once for the whole call ----------
    # One staged transpose per latent tile, with the last tile's extent the remainder.
    #
    # Loops rather than comprehensions, and an index rather than a tuple target, here
    # and in every tile loop below: the NKI compiler refuses a comprehension at
    # specialization ("unsupported expression"), a for-target that is not a plain name
    # at compilation ("expecting simple variable"), and `enumerate` outright ("failed
    # to resolve name 'builtins.enumerate'"). Each list keeps its own loop so the
    # allocation order is preserved.
    c_sb = []
    for li in range(len(lat_tiles)):
        c_sb.append(_sbuf(lat_tiles[li][1], s_kv))
    for li in range(len(lat_tiles)):
        offset = lat_tiles[li][0]
        extent = lat_tiles[li][1]
        c_stage = _stage(c_kv_hbm, extent, s_kv)
        _transpose_rows(c_stage, c_kv_hbm, latent, s_kv, extent, offset)
        nisa.tensor_copy(dst=c_sb[li], src=c_stage)
    k_pe_sb = None
    if rope > 0:
        k_pe_sb = _sbuf(rope, s_kv)
        k_pe_stage = _stage(k_pe_hbm, rope, s_kv)
        _transpose_rows(k_pe_stage, k_pe_hbm, rope, s_kv, rope, 0)
        nisa.tensor_copy(dst=k_pe_sb, src=k_pe_stage)

    # ---- per-query working set, allocated once and reused across the loop -------
    # `c_g_t` and `out_sb` stay single buffers with the latent on their free axis,
    # where a ragged extent needs no special case; only the partition-axis buffers
    # become lists.
    qpb = _queries_per_block(seq, heads)
    block = qpb * heads
    idx_sb = _sbuf_u32(LATENT_TILE, topk)
    q_stage = []
    for li in range(len(lat_tiles)):
        q_stage.append(_stage(q_lift_hbm, lat_tiles[li][1], block))
    q_lift_t = []
    for li in range(len(lat_tiles)):
        q_lift_t.append(_sbuf(lat_tiles[li][1], _aligned(block)))
    c_g = []
    for li in range(len(lat_tiles)):
        c_g.append(_sbuf(lat_tiles[li][1], topk))
    c_g_t = _sbuf(KEY_CHUNK, n_chunks, latent)
    p_t = _sbuf(KEY_CHUNK, n_chunks, _aligned(heads))
    p = _sbuf(heads, topk)
    neg_row_max = _scalar(heads)
    exp_bias = _scalar(heads)
    row_sum = _scalar(heads)
    recip = _scalar(heads)
    out_sb = _sbuf(heads, latent)
    q_pe_t = _sbuf(rope, _aligned(block)) if rope > 0 else None
    q_pe_stage = _stage(q_pe_hbm, rope, block) if rope > 0 else None
    k_pe_g = _sbuf(rope, topk) if rope > 0 else None

    # The sentinel working set is the untiled body's exactly: the mask is a fact about
    # the selected-row axis, which is not the axis this body tiles. The index tile
    # stays the full 128 partitions and each gather slices it, so one clamp serves
    # every latent tile including the ragged one.
    sen = _sentinel_scratch(LATENT_TILE, topk, heads)
    valid_f = sen[4]
    mask_bias = sen[5]
    scores_m = sen[6]
    p_m = sen[7]
    sentinel_bias = _SENTINEL_EXP_FLOOR / softmax_scale

    # ---- the queries, in blocks whose Q rows fill one 16-row transpose ----------
    # One transpose per latent tile lands the block's Q rows in the source dtype and
    # one copy widens them; each query then reads its own columns of the block tile.
    for qb in nl.affine_range(seq // qpb):
        q0 = qb * qpb
        for li in range(len(lat_tiles)):
            offset = lat_tiles[li][0]
            extent = lat_tiles[li][1]
            _transpose_rows(q_stage[li], q_lift_hbm, latent, block, extent,
                            q0 * heads * latent + offset)
            nisa.tensor_copy(dst=q_lift_t[li][:, 0:block], src=q_stage[li])
        if rope > 0:
            _transpose_rows(q_pe_stage, q_pe_hbm, rope, block, rope, q0 * heads * rope)
            nisa.tensor_copy(dst=q_pe_t[:, 0:block], src=q_pe_stage)
        for qi in range(qpb):
            q_idx = q0 + qi
            h0 = qi * heads
            # ---- this query's selected rows, replicated to every partition ----------
            _mask_sentinel(topk_hbm, q_idx * topk, topk, heads, sentinel_bias, sen, idx_sb)

            # ---- gather the latent cache rows: one instruction per latent tile ------
            # The index tile is sliced to the data tile's extent. `nc_n_gather` reads its
            # offsets from the same partition it writes, so a 3-deep data tile needs a
            # 3-deep index tile and not the full 128.
            for li in range(len(lat_tiles)):
                extent = lat_tiles[li][1]
                nisa.nc_n_gather(dst=c_g[li], data=c_sb[li], indices=idx_sb[0:extent, :])

            # ---- MM1: scores[H, K] = sum over latent tiles of q_lift_t.T @ c_g ------
            # The ragged tail tile contributes a shorter contraction to the same
            # accumulation as the full ones; nothing is masked or padded.
            scores_ps = _psum(heads, topk)
            for li in range(len(lat_tiles)):
                nisa.nc_matmul(
                    dst=scores_ps,
                    stationary=q_lift_t[li][:, h0:h0 + heads],
                    moving=c_g[li],
                    accumulate=(li > 0),
                )
            if rope > 0:
                nisa.nc_n_gather(dst=k_pe_g, data=k_pe_sb, indices=idx_sb[0:rope, :])
                nisa.nc_matmul(dst=scores_ps, stationary=q_pe_t[:, h0:h0 + heads],
                               moving=k_pe_g,
                               accumulate=True)

            # ---- the gathered cache, transposed for MM2 -----------------------------
            # Each latent tile transposes into its own slice of the latent free axis, so
            # the ragged tail lands as a 3-wide slice rather than a padded 128-wide one.
            for ck in range(n_chunks):
                ks = ck * KEY_CHUNK
                for li in range(len(lat_tiles)):
                    offset = lat_tiles[li][0]
                    extent = lat_tiles[li][1]
                    c_g_t_ps = _psum(KEY_CHUNK, extent)
                    nisa.nc_transpose(dst=c_g_t_ps, data=c_g[li][:, ks:ks + KEY_CHUNK])
                    nisa.tensor_copy(
                        dst=c_g_t[:, ck, offset:offset + extent],
                        src=c_g_t_ps,
                    )

            # ---- softmax over K, per head row --------------------------------------
            # Untouched by the tiling: the scores tile is [H, K] whatever the latent rank,
            # so this is the untiled chain, including the two masking steps.
            nisa.tensor_tensor(dst=scores_m, data1=scores_ps, data2=mask_bias, op=nl.add)
            nisa.tensor_reduce(dst=neg_row_max, op=nl.maximum, data=scores_m, axis=1,
                               negate=True)
            nisa.tensor_scalar(dst=exp_bias, data=neg_row_max, op0=nl.multiply,
                               operand0=softmax_scale, engine=nisa.engine.vector)
            nisa.activation(dst=p, op=nl.exp, data=scores_m, bias=exp_bias,
                            scale=softmax_scale, reduce_op=nl.add, reduce_res=row_sum,
                            reduce_cmd=nisa.reduce_cmd.reset_reduce)
            nisa.reciprocal(dst=recip, data=row_sum)
            nisa.tensor_tensor(dst=p_m, data1=p, data2=valid_f, op=nl.multiply)

            # ---- MM2: out[H, L] = p[H, K] @ c_g[K, L], K contracted on partitions ---
            for ck in range(n_chunks):
                ks = ck * KEY_CHUNK
                p_t_ps = _psum(KEY_CHUNK, heads)
                nisa.nc_transpose(dst=p_t_ps, data=p_m[:, ks:ks + KEY_CHUNK])
                nisa.tensor_copy(dst=p_t[:, ck, 0:heads], src=p_t_ps)

            # The second tiling: the latent is MM2's moving free axis, so the output is
            # produced 512 columns at a time and the last tile may be narrow. The
            # denominator is applied per output tile, one pass over [H, L] rather than
            # one over [H, K]. A plain loop target read by subscript, because the NKI
            # compiler refuses a tuple target.
            for otile in out_tiles:
                offset = otile[0]
                extent = otile[1]
                pv_ps = _psum(heads, extent)
                for ck in range(n_chunks):
                    nisa.nc_matmul(dst=pv_ps, stationary=p_t[:, ck, 0:heads],
                                   moving=c_g_t[:, ck, offset:offset + extent],
                                   accumulate=(ck > 0))
                nisa.tensor_scalar(dst=out_sb[:, offset:offset + extent], data=pv_ps,
                                   op0=nl.multiply, operand0=recip,
                                   engine=nisa.engine.vector)
            nl.store(
                out_hbm.ap(pattern=[[latent, heads], [1, latent]],
                           offset=q_idx * heads * latent),
                value=out_sb,
            )


@nki.jit
def mla_sparse_attention_nope_tiled_kernel(q_lift_hbm, c_kv_hbm, topk_hbm,
                                           softmax_scale, block_table_hbm=None,
                                           written_hbm=None, write_offset_hbm=None,
                                           page_size=0):
    """The latent-tiled R == 0 entry point."""
    seq, heads, latent = q_lift_hbm.shape
    out_hbm = nl.ndarray((seq, heads, latent), dtype=nl.float32, buffer=nl.shared_hbm)
    window_hbm = _window_of(c_kv_hbm, block_table_hbm, written_hbm, write_offset_hbm,
                            page_size)
    _attention_body_tiled(q_lift_hbm, window_hbm, topk_hbm, softmax_scale, out_hbm)
    return out_hbm


@nki.jit
def mla_sparse_attention_rope_tiled_kernel(q_lift_hbm, q_pe_hbm, c_kv_hbm, k_pe_hbm,
                                           topk_hbm, softmax_scale):
    """The latent-tiled R > 0 entry point.

    The RoPE limb contracts its own axis into the same score tile, so it is
    latent-independent and shares every line with the entry point above.
    """
    seq, heads, latent = q_lift_hbm.shape
    out_hbm = nl.ndarray((seq, heads, latent), dtype=nl.float32, buffer=nl.shared_hbm)
    _attention_body_tiled(q_lift_hbm, c_kv_hbm, topk_hbm, softmax_scale, out_hbm,
                          q_pe_hbm=q_pe_hbm, k_pe_hbm=k_pe_hbm)
    return out_hbm


# --------------------------------------------------------------------------- #
# The selected-row-tiled path, for a topk past one MM1 moving tile.
#
# The bodies above bound K at one MM1 moving tile of 512. This body tiles K on
# MM1's moving free axis in tiles of BLOCK_N (four score tiles at 2,048) and on
# MM2's partition axis in chunks of 128 within each tile. The softmax is the
# content: a tile cannot be exponentiated against a max nobody has seen yet, so
# each tile is exponentiated against its own row max, and the running denominator
# and output accumulator are rescaled when a later tile raises that max. Both
# rescale factors are exp of a non-positive number by construction, so neither can
# overflow; that is why the max is carried instead of every tile using the first
# tile's bias.
#
# Combining the two tilings in one call is refused at the gate, so this body may
# assume the latent is an exact multiple of the partition tile and fits one MM2
# moving tile.
# --------------------------------------------------------------------------- #


@dataclass
class _MlaSparseRowTiledDispatchCounters:
    """Per-process record of how the row-tiled path was reached.

    Additive to the seam counter, not exclusive: a row-tiled call also counts as a
    seam dispatch, so the counters compose rather than partition the traffic.
    """

    nki_dispatch: int = 0
    torch_fallback: int = 0


_MLA_SPARSE_ROW_TILED_COUNTERS = _MlaSparseRowTiledDispatchCounters()


def reset_mla_sparse_row_tiled_dispatch_counters() -> None:
    """Zero the row-tiled counters; the other two resets do not touch these."""
    _MLA_SPARSE_ROW_TILED_COUNTERS.nki_dispatch = 0
    _MLA_SPARSE_ROW_TILED_COUNTERS.torch_fallback = 0


def mla_sparse_row_tiled_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` for the row-tiled path since its last reset.

    ``torch_fallback`` is always ``0``: there is no torch attention route here.
    """
    return (
        _MLA_SPARSE_ROW_TILED_COUNTERS.nki_dispatch,
        _MLA_SPARSE_ROW_TILED_COUNTERS.torch_fallback,
    )


@torch._dynamo.assume_constant_result
def _count_row_tiled_nki_dispatch() -> None:
    """Count one kernel dispatch, off the traced graph: a store Dynamo traces becomes a guard."""
    _MLA_SPARSE_ROW_TILED_COUNTERS.nki_dispatch += 1


def _score_tiles(topk: int, block_n: int = MOVING_MAX) -> tuple[tuple[int, int], ...]:
    """``(offset, extent)`` per MM1 moving-axis score tile; at 2,048 this is 4 x 512.

    The last tile may be narrower but is never a partial key chunk: the gate admits
    only a multiple of :data:`KEY_CHUNK` and ``block_n`` must be one too, so every
    extent is a whole number of MM2 key chunks (``topk=640`` gives tiles of 512 and
    128). That is what lets the body chunk each tile without a partial-chunk case.
    """
    tiles = []
    offset = 0
    while offset < topk:
        tiles.append((offset, min(block_n, topk - offset)))
        offset += block_n
    return tuple(tiles)


def _load_selected_rows(dst, cache_hbm, indices_hbm, offset, width):
    """Gather one 128-row tile from HBM, widening to FP32 on the DMA.

    The signed clamp makes every DMA address valid before the attention mask
    removes sentinel columns. The gather preserves index order and duplicates.
    """
    rows = dst.shape[0]
    raw = _sbuf_i32(rows, DMA_TRANSPOSE_ALIGN)[:, 0:1]
    safe = _sbuf_i32(rows, DMA_TRANSPOSE_ALIGN)[:, 0:1]
    nisa.dma_copy(dst=raw, src=indices_hbm.ap(
        pattern=[[1, rows], [1, 1]], offset=offset))
    nisa.tensor_scalar(dst=safe, data=raw, op0=nl.maximum, operand0=0)
    nisa.dma_copy(dst=dst, src=cache_hbm.ap(
        pattern=[[width, rows], [1, width]], vector_offset=safe, indirect_dim=0))


def _attention_body_row_tiled(q_lift_hbm, c_kv_hbm, topk_hbm, softmax_scale, out_hbm,
                              q_pe_hbm=None, k_pe_hbm=None,
                              BLOCK_N=MOVING_MAX, STREAM_KV=True):
    """Online sparse attention over score tiles, with input-derived dimensions.

    ``BLOCK_N`` is the selected-key tile width. It must be a multiple of
    ``KEY_CHUNK`` and fit the matmul moving axis. Changing it changes the softmax
    merge grouping; the default keeps the untiled body's arithmetic order.

    Streaming gathers ``[KEY_CHUNK, latent]`` FP32 rows straight from HBM. One
    transpose supplies MM1, and MM2 reuses the gathered rows. Staging instead loads
    the full cache once and gathers from SBUF. Both preserve duplicate indices and
    mask -1.

    The geometry gate restricts this body to exact latent tiles that fit one MM2
    moving tile; other latent shapes take the other bodies.
    """
    seq, heads, latent = q_lift_hbm.shape
    s_kv = c_kv_hbm.shape[0]
    topk = topk_hbm.shape[1]
    n_latent = latent // LATENT_TILE
    rope = 0 if q_pe_hbm is None else q_pe_hbm.shape[2]
    assert KEY_CHUNK <= BLOCK_N <= MOVING_MAX and BLOCK_N % KEY_CHUNK == 0, (
        "BLOCK_N must be a multiple of KEY_CHUNK in [KEY_CHUNK, MOVING_MAX]")
    tiles = _score_tiles(topk, BLOCK_N)
    single = len(tiles) == 1
    # Not ``max(extent for _, extent in tiles)``: the NKI compiler refuses a generator
    # expression at specialization ("unsupported expression") and a tuple for-target at
    # compilation ("expecting simple variable"), so the target is a plain name and the
    # pair is read by subscript.
    tile_max = 0
    for tile in tiles:
        if tile[1] > tile_max:
            tile_max = tile[1]
    chunk_max = tile_max // KEY_CHUNK

    # The optional staged path reuses the full cache across queries. Streaming
    # omits these allocations, so its SBUF use does not grow with cache length.
    c_sb = []
    if not STREAM_KV:
        for _ in range(n_latent):
            c_sb.append(_sbuf(LATENT_TILE, s_kv))
        c_stage = _stage(c_kv_hbm, LATENT_TILE, s_kv)
        for li in range(n_latent):
            _transpose_rows(c_stage, c_kv_hbm, latent, s_kv, LATENT_TILE, li * LATENT_TILE)
            nisa.tensor_copy(dst=c_sb[li], src=c_stage)
    k_pe_sb = None
    if rope > 0 and not STREAM_KV:
        k_pe_sb = _sbuf(rope, s_kv)
        k_pe_stage = _stage(k_pe_hbm, rope, s_kv)
        _transpose_rows(k_pe_stage, k_pe_hbm, rope, s_kv, rope, 0)
        nisa.tensor_copy(dst=k_pe_sb, src=k_pe_stage)

    # ---- the working set, sized to one score tile and reused by every tile -------
    # Every buffer below is the size the untiled body uses at `topk == MOVING_MAX`,
    # whatever K the caller passes.
    qpb = _queries_per_block(seq, heads)
    block = qpb * heads
    idx_sb = _sbuf_u32(LATENT_TILE, tile_max)
    q_stage = []
    for _ in range(n_latent):
        q_stage.append(_stage(q_lift_hbm, LATENT_TILE, block))
    q_lift_t = _sbuf(LATENT_TILE, n_latent, _aligned(block))
    c_g = _sbuf(LATENT_TILE, n_latent, tile_max)
    c_g_t = _sbuf(KEY_CHUNK, chunk_max, latent)
    p_t = _sbuf(KEY_CHUNK, chunk_max, _aligned(heads))
    p = _sbuf(heads, tile_max)
    neg_row_max = _scalar(heads)
    exp_bias = _scalar(heads)
    tile_sum = _scalar(heads)
    recip = _scalar(heads)
    out_sb = _sbuf(heads, latent)
    q_pe_t = _sbuf(rope, _aligned(block)) if rope > 0 else None
    q_pe_stage = _stage(q_pe_hbm, rope, block) if rope > 0 else None
    k_pe_g = _sbuf(rope, tile_max) if rope > 0 else None
    k_pe_rows = _sbuf(KEY_CHUNK, _aligned(rope))[:, 0:rope] if rope > 0 and STREAM_KV else None

    # ---- the running state carried across score tiles ----------------------------
    # `run_pos` holds `softmax_scale * (running row max)` in positive form, because
    # merging two maxima uses `tensor_tensor(op=nl.maximum)`. The softmax chain
    # produces the negated, scaled max, so it is negated once per tile.
    run_pos = _scalar(heads)
    run_sum = _scalar(heads)
    acc = _sbuf(heads, latent)
    # Merge scratch: fresh destinations, never a write back onto an operand.
    tile_pos = _scalar(heads)
    new_pos = _scalar(heads)
    d_acc = _scalar(heads)
    d_tile = _scalar(heads)
    c_acc = _scalar(heads)
    c_tile = _scalar(heads)
    sum_kept = _scalar(heads)
    sum_added = _scalar(heads)
    sum_new = _scalar(heads)
    acc_kept = _sbuf(heads, latent)
    pv_added = _sbuf(heads, latent)
    acc_new = _sbuf(heads, latent)

    # The sentinel working set, sized to one score tile like every other buffer here
    # and sliced per tile the same way, because the mask is a fact about the axis this
    # body tiles.
    #
    # The merge needs no special case for sentinels because the bias is large in
    # exponent units. A wholly-sentinel tile's `tile_pos` lands `_SENTINEL_EXP_FLOOR`
    # below any real tile's, so the merge's rescale factor for it is
    # ``exp(very negative) == 0``, which annihilates both its summed denominator and
    # its already-zero numerator whichever order the tiles arrive in. A wholly-sentinel
    # first tile initialises the running state, and the next real tile's rescale wipes
    # what it left.
    sen = _sentinel_scratch(LATENT_TILE, tile_max, heads)
    valid_f = sen[4]
    mask_bias = sen[5]
    scores_m = sen[6]
    p_m = sen[7]
    sentinel_bias = _SENTINEL_EXP_FLOOR / softmax_scale

    # ---- the queries, in blocks whose Q rows fill one 16-row transpose ----------
    # One transpose per latent tile lands the block's Q rows in the source dtype and
    # one copy widens them; each query then reads its own columns of the block tile.
    for qb in nl.affine_range(seq // qpb):
        q0 = qb * qpb
        for li in range(n_latent):
            _transpose_rows(q_stage[li], q_lift_hbm, latent, block, LATENT_TILE,
                            q0 * heads * latent + li * LATENT_TILE)
            nisa.tensor_copy(dst=q_lift_t[:, li, 0:block], src=q_stage[li])
        if rope > 0:
            _transpose_rows(q_pe_stage, q_pe_hbm, rope, block, rope, q0 * heads * rope)
            nisa.tensor_copy(dst=q_pe_t[:, 0:block], src=q_pe_stage)
        for qi in range(qpb):
            q_idx = q0 + qi
            h0 = qi * heads
            # Not ``for ti, (ks, extent) in enumerate(tiles)``: the NKI compiler refuses
            # a tuple target ("expecting simple variable") and `enumerate` ("failed to
            # resolve name 'builtins.enumerate'"), so the tiles are indexed by range.
            for ti in range(len(tiles)):
                ks = tiles[ti][0]
                extent = tiles[ti][1]
                n_chunks = extent // KEY_CHUNK

                # ---- this tile's selected rows, replicated to every partition --------
                # The untiled body loads all K rows of the query; this loads the tile's
                # slice of them, signed and clamped before the gather reads it.
                _mask_sentinel(topk_hbm, q_idx * topk + ks, extent, heads, sentinel_bias,
                               sen, idx_sb)

                # ---- gather this tile's cache rows: one instruction per latent tile ---
                if STREAM_KV:
                    for ck in range(n_chunks):
                        cs = ck * KEY_CHUNK
                        _load_selected_rows(c_g_t[:, ck, :], c_kv_hbm, topk_hbm,
                                            q_idx * topk + ks + cs, latent)
                        for li in range(n_latent):
                            gathered_ps = _psum(LATENT_TILE, KEY_CHUNK)
                            nisa.nc_transpose(dst=gathered_ps, data=c_g_t[
                                :, ck, li * LATENT_TILE:(li + 1) * LATENT_TILE])
                            nisa.tensor_copy(dst=c_g[:, li, cs:cs + KEY_CHUNK],
                                             src=gathered_ps)
                        if rope > 0:
                            _load_selected_rows(k_pe_rows, k_pe_hbm, topk_hbm,
                                                q_idx * topk + ks + cs, rope)
                            rope_ps = _psum(rope, KEY_CHUNK)
                            nisa.nc_transpose(dst=rope_ps, data=k_pe_rows)
                            nisa.tensor_copy(dst=k_pe_g[:, cs:cs + KEY_CHUNK], src=rope_ps)
                else:
                    for li in range(n_latent):
                        nisa.nc_n_gather(dst=c_g[:, li, 0:extent], data=c_sb[li],
                                         indices=idx_sb[:, 0:extent])

                # ---- MM1 over this tile: scores[H, extent] ---------------------------
                scores_ps = _psum(heads, extent)
                for li in range(n_latent):
                    nisa.nc_matmul(
                        dst=scores_ps,
                        stationary=q_lift_t[:, li, h0:h0 + heads],
                        moving=c_g[:, li, 0:extent],
                        accumulate=(li > 0),
                    )
                if rope > 0:
                    if not STREAM_KV:
                        nisa.nc_n_gather(dst=k_pe_g[:, 0:extent], data=k_pe_sb,
                                         indices=idx_sb[0:rope, 0:extent])
                    nisa.nc_matmul(dst=scores_ps, stationary=q_pe_t[:, h0:h0 + heads],
                                   moving=k_pe_g[:, 0:extent], accumulate=True)

                # ---- the gathered cache, transposed for MM2 --------------------------
                if not STREAM_KV:
                    for ck in range(n_chunks):
                        cs = ck * KEY_CHUNK
                        for li in range(n_latent):
                            c_g_t_ps = _psum(KEY_CHUNK, LATENT_TILE)
                            nisa.nc_transpose(dst=c_g_t_ps,
                                              data=c_g[:, li, cs:cs + KEY_CHUNK])
                            nisa.tensor_copy(
                                dst=c_g_t[:, ck, li * LATENT_TILE:(li + 1) * LATENT_TILE],
                                src=c_g_t_ps,
                            )

                # ---- softmax over this tile's keys: the untiled chain ----------------
                # Against the tile's own max, the only max available yet; the merge below
                # is what makes that legitimate. The two masking steps sit where they sit
                # above: the bias before the tile's max, the zeroing before the tile's MM2.
                nisa.tensor_tensor(dst=scores_m[:, 0:extent], data1=scores_ps,
                                   data2=mask_bias[:, 0:extent], op=nl.add)
                nisa.tensor_reduce(dst=neg_row_max, op=nl.maximum,
                                   data=scores_m[:, 0:extent], axis=1, negate=True)
                nisa.tensor_scalar(dst=exp_bias, data=neg_row_max, op0=nl.multiply,
                                   operand0=softmax_scale, engine=nisa.engine.vector)
                nisa.activation(dst=p[:, 0:extent], op=nl.exp,
                                data=scores_m[:, 0:extent],
                                bias=exp_bias, scale=softmax_scale, reduce_op=nl.add,
                                reduce_res=tile_sum,
                                reduce_cmd=nisa.reduce_cmd.reset_reduce)
                nisa.tensor_tensor(dst=p_m[:, 0:extent], data1=p[:, 0:extent],
                                   data2=valid_f[:, 0:extent], op=nl.multiply)

                # ---- MM2 over this tile: pv[H, L] = p[H, extent] @ c_g[extent, L] ----
                for ck in range(n_chunks):
                    cs = ck * KEY_CHUNK
                    p_t_ps = _psum(KEY_CHUNK, heads)
                    nisa.nc_transpose(dst=p_t_ps, data=p_m[:, cs:cs + KEY_CHUNK])
                    nisa.tensor_copy(dst=p_t[:, ck, 0:heads], src=p_t_ps)
                pv_ps = _psum(heads, latent)
                for ck in range(n_chunks):
                    nisa.nc_matmul(dst=pv_ps, stationary=p_t[:, ck, 0:heads],
                                   moving=c_g_t[:, ck, :], accumulate=(ck > 0))

                # ---- the merge ------------------------------------------------------
                if single:
                    # One tile: no merge is emitted, so the arithmetic is the untiled
                    # body's exactly.
                    nisa.tensor_copy(dst=acc, src=pv_ps)
                    nisa.tensor_copy(dst=run_sum, src=tile_sum)
                    continue

                nisa.tensor_scalar(dst=tile_pos, data=exp_bias, op0=nl.multiply,
                                   operand0=-1.0, engine=nisa.engine.vector)
                if ti == 0:
                    # The first tile initialises the running state; there is nothing to
                    # rescale against yet.
                    nisa.tensor_copy(dst=run_pos, src=tile_pos)
                    nisa.tensor_copy(dst=acc, src=pv_ps)
                    nisa.tensor_copy(dst=run_sum, src=tile_sum)
                    continue

                # The running max, and the two rescale factors it implies. Both are `exp` of
                # a non-positive number: `run_pos - new_pos` is <= 0 because `new_pos` is
                # the maximum of the two, and so is `tile_pos - new_pos`. So neither can
                # overflow, whatever the score spread.
                nisa.tensor_tensor(dst=new_pos, data1=run_pos, data2=tile_pos,
                                   op=nl.maximum)
                nisa.tensor_tensor(dst=d_acc, data1=run_pos, data2=new_pos, op=nl.subtract)
                nisa.activation(dst=c_acc, op=nl.exp, data=d_acc)
                nisa.tensor_tensor(dst=d_tile, data1=tile_pos, data2=new_pos,
                                   op=nl.subtract)
                nisa.activation(dst=c_tile, op=nl.exp, data=d_tile)

                # The denominator: what was already summed, rebased, plus this tile's sum,
                # rebased. Two multiplies and an add, all on [H, 1].
                nisa.tensor_tensor(dst=sum_kept, data1=run_sum, data2=c_acc,
                                   op=nl.multiply)
                nisa.tensor_tensor(dst=sum_added, data1=tile_sum, data2=c_tile,
                                   op=nl.multiply)
                nisa.tensor_tensor(dst=sum_new, data1=sum_kept, data2=sum_added,
                                   op=nl.add)

                # The output accumulator, rebased the same way. The denominator is applied
                # once after the last tile, not per tile: one pass over [H, L] instead of
                # one over [H, K], and it keeps the accumulator in the numerator's scale so
                # a rescale is a single multiply.
                nisa.tensor_scalar(dst=acc_kept, data=acc, op0=nl.multiply, operand0=c_acc,
                                   engine=nisa.engine.vector)
                nisa.tensor_scalar(dst=pv_added, data=pv_ps, op0=nl.multiply,
                                   operand0=c_tile, engine=nisa.engine.vector)
                nisa.tensor_tensor(dst=acc_new, data1=acc_kept, data2=pv_added, op=nl.add)

                nisa.tensor_copy(dst=run_pos, src=new_pos)
                nisa.tensor_copy(dst=run_sum, src=sum_new)
                nisa.tensor_copy(dst=acc, src=acc_new)

            # ---- one normalisation per query, after the last tile --------------------
            nisa.reciprocal(dst=recip, data=run_sum)
            nisa.tensor_scalar(dst=out_sb, data=acc, op0=nl.multiply, operand0=recip,
                               engine=nisa.engine.vector)
            nl.store(
                out_hbm.ap(pattern=[[latent, heads], [1, latent]],
                           offset=q_idx * heads * latent),
                value=out_sb,
            )


@nki.jit
def mla_sparse_attention_nope_row_tiled_kernel(q_lift_hbm, c_kv_hbm, topk_hbm,
                                               softmax_scale, block_table_hbm=None,
                                               written_hbm=None, write_offset_hbm=None,
                                               page_size=0, BLOCK_N=MOVING_MAX,
                                               STREAM_KV=True):
    """Row-tiled NoPE sparse attention; the paged operands and tile parameters are optional.

    The paged operands are declared before the tile parameters so a paged call reaches
    ``page_size`` positionally without passing a tile parameter.
    """
    seq, heads, latent = q_lift_hbm.shape
    out_hbm = nl.ndarray((seq, heads, latent), dtype=nl.float32, buffer=nl.shared_hbm)
    window_hbm = _window_of(c_kv_hbm, block_table_hbm, written_hbm, write_offset_hbm,
                            page_size)
    _attention_body_row_tiled(q_lift_hbm, window_hbm, topk_hbm, softmax_scale, out_hbm,
                              BLOCK_N=BLOCK_N, STREAM_KV=STREAM_KV)
    return out_hbm


@nki.jit
def mla_sparse_attention_rope_row_tiled_kernel(q_lift_hbm, q_pe_hbm, c_kv_hbm, k_pe_hbm,
                                               topk_hbm, softmax_scale,
                                               BLOCK_N=MOVING_MAX, STREAM_KV=True):
    """Row-tiled sparse attention with paired RoPE operands and compile-time tile options."""
    seq, heads, latent = q_lift_hbm.shape
    out_hbm = nl.ndarray((seq, heads, latent), dtype=nl.float32, buffer=nl.shared_hbm)
    _attention_body_row_tiled(q_lift_hbm, c_kv_hbm, topk_hbm, softmax_scale, out_hbm,
                              q_pe_hbm=q_pe_hbm, k_pe_hbm=k_pe_hbm,
                              BLOCK_N=BLOCK_N, STREAM_KV=STREAM_KV)
    return out_hbm


def _require_admissible(seq: int, heads: int, latent: int, rope: int, topk: int,
                        s_kv: int, softmax_scale: float) -> None:
    """Raise unless a kernel body serves this geometry; there is no fallback.

    Every bound that names a hardware axis says which one; a tiled axis keeps only
    positivity. There is no bound on the RoPE width being positive, and none on S or
    S_kv: the query loop and the cache free axis walk whatever they are given.
    """
    if seq < 1 or s_kv < 1:
        raise MlaSparseAttentionError(
            f"mla_sparse_attention needs at least one query and one cache row; got "
            f"seq={seq}, s_kv={s_kv}"
        )
    if heads < 1 or heads > HEAD_MAX:
        raise MlaSparseAttentionError(
            f"heads rides the matmul stationary free axis, bounded at {HEAD_MAX} "
            f"({_TILE_PROVENANCE}); got heads={heads}"
        )
    if latent < 1:
        raise MlaSparseAttentionError(
            f"the latent rank must be positive; got latent={latent}. The kernel "
            f"TILES both axes this used to be bounded on -- the partition axis in "
            f"tiles of {LATENT_TILE} with a ragged tail, and MM2's moving free axis in "
            f"tiles of {MOVING_MAX} ({_TILE_PROVENANCE}) -- so neither a multiple-of "
            f"nor an upper bound is asserted here any more. This checkpoint's "
            f"kv_lora_rank is {TARGET_LATENT_RANK} and takes the untiled body"
        )
    if rope < 0 or rope > HEAD_MAX:
        raise MlaSparseAttentionError(
            f"the RoPE width rides the partition axis in the RoPE matmul, bounded at "
            f"{HEAD_MAX}; got rope={rope}. Zero IS admissible and is this "
            f"checkpoint's value"
        )
    if topk < 1 or topk % KEY_CHUNK != 0:
        raise MlaSparseAttentionError(
            f"the selected-row count rides the partition axis in MM2 in chunks of "
            f"{KEY_CHUNK}, so it must be a positive multiple of it; got topk={topk}. "
            f"A width that is not a multiple is refused rather than padded"
        )
    if topk > MOVING_MAX and (latent % LATENT_TILE != 0 or latent > MOVING_MAX):
        raise MlaSparseAttentionError(
            f"tiling the selected-row axis and the latent axis in the SAME call is not "
            f"served: got topk={topk}, which is past the moving free-axis tile width "
            f"{MOVING_MAX} ({_TILE_PROVENANCE}), together with latent={latent}, which "
            f"is not an exact fit for the {LATENT_TILE} partition tile. Each axis is "
            f"tiled on its own -- the rows at an exact-fit latent, the latent at topk "
            f"no wider than one tile -- and the combination is refused, not promised"
        )
    if not softmax_scale > 0:
        raise MlaSparseAttentionError(
            f"softmax_scale must be positive; got {softmax_scale}"
        )


def can_run_mla_sparse_attention(reference: Tensor, seq: int, heads: int, latent: int,
                                 rope: int, topk: int, s_kv: int,
                                 softmax_scale: float) -> bool:
    """True when the NKI route is available and serves this geometry."""
    if not can_run_kernel(reference):
        return False
    try:
        _require_admissible(seq, heads, latent, rope, topk, s_kv, softmax_scale)
    except MlaSparseAttentionError:
        return False
    return True


def _kernel_operand(t: Tensor) -> Tensor:
    """The dtype the kernel reads: a 2-byte float as stored, FP8 widened to bfloat16 (exact), else float32."""
    t = t.contiguous()
    if t.dtype in (torch.bfloat16, torch.float16):
        return t
    if t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return t.to(torch.bfloat16)
    return t.to(torch.float32)


def _require_paged(c_kv: Tensor, block_table_row: Tensor, written: Tensor | None,
                   write_offset: Tensor | None, page_size: int,
                   k_pe: Tensor | None) -> int:
    """Raise unless these paged operands name a window this kernel can stage; return its length."""
    if k_pe is not None:
        raise MlaSparseAttentionError(
            "a paged window and a RoPE half are refused together: k_pe is a SECOND window "
            "beside the latent one, one row per cache row, and nothing here pages it. This "
            "checkpoint's RoPE width is 0 and the limb is elided at trace time, so pass "
            "block_table_row with no q_pe/k_pe, or the gathered window with them"
        )
    page_size = int(page_size)
    if page_size < 1:
        raise MlaSparseAttentionError(
            f"page_size is the served block size and travels as a trace-time int; got "
            f"{page_size}. It is specialised into the graph, so a block-size change is a "
            f"recompile rather than a runtime operand"
        )
    if page_size % STAGE_ROWS != 0 and STAGE_ROWS % page_size != 0:
        raise MlaSparseAttentionError(
            f"a page is staged in whole pieces of {STAGE_ROWS} rows, the SBUF partition "
            f"bound, so the block size must be a multiple of it or divide it; got "
            f"page_size={page_size}"
        )
    if block_table_row.ndim != 2 or int(block_table_row.shape[1]) != 1:
        raise MlaSparseAttentionError(
            f"block_table_row must be [pages, 1], the column the page number is read out "
            f"of; got shape {tuple(block_table_row.shape)}"
        )
    pages = int(block_table_row.shape[0])
    if pages < 1:
        raise MlaSparseAttentionError(
            "block_table_row must name at least one page; got none"
        )
    bank_rows, bank_latent = (int(d) for d in c_kv.shape)
    if bank_rows % page_size != 0:
        raise MlaSparseAttentionError(
            f"c_kv is the whole latent BANK when a block table is present, so its rows must "
            f"be whole pages; got {bank_rows} rows against page_size={page_size}"
        )
    window = pages * page_size
    if values_are_readable(block_table_row):
        lo, hi = int(block_table_row.min()), int(block_table_row.max())
        if lo < SENTINEL_INDEX or hi >= bank_rows // page_size:
            raise MlaSparseAttentionError(
                f"every page number must index the bank or be the {SENTINEL_INDEX} pad; got "
                f"the range [{lo}, {hi}] against {bank_rows // page_size} pages in the bank"
            )
    tokens = 0 if written is None else int(written.shape[0])
    if written is not None:
        if written.ndim != 2 or int(written.shape[1]) != bank_latent:
            raise MlaSparseAttentionError(
                f"written must be [tokens, latent] with c_kv's latent rank {bank_latent}; "
                f"got shape {tuple(written.shape)}"
            )
        if tokens > window:
            raise MlaSparseAttentionError(
                f"written carries more rows than the window holds; got {tokens} rows "
                f"against a window of {window} = {pages} pages of {page_size}"
            )
        if _kernel_operand(written).dtype != _kernel_operand(c_kv).dtype:
            raise MlaSparseAttentionError(
                f"written is overlaid onto the staged window and is copied without a cast, "
                f"so it must read as c_kv's dtype; got {written.dtype} against {c_kv.dtype}"
            )
    if tokens > 0 and write_offset is None:
        raise MlaSparseAttentionError(
            f"written carries {tokens} rows and no write_offset says where they sit in the "
            f"window; pass a [1, 1] int32 row offset"
        )
    if write_offset is not None:
        if write_offset.ndim != 2 or tuple(int(d) for d in write_offset.shape) != (1, 1):
            raise MlaSparseAttentionError(
                f"write_offset must be a [1, 1] int32 tensor, read on device; got shape "
                f"{tuple(write_offset.shape)}"
            )
        # The position is checked eagerly and nowhere else: a tracer makes the value
        # unreadable, so an extracted graph carries no refusal for it. In a graph the
        # runner's sizing holds instead -- the window spans this step's context plus its
        # query rows -- and a configuration that clips that sizing to the table width
        # would overlay past the staged window with nothing to refuse it.
        if values_are_readable(write_offset):
            at = int(write_offset[0, 0])
            if at < 0 or at + tokens > window:
                raise MlaSparseAttentionError(
                    f"this step's {tokens} written rows must land inside the window; got "
                    f"write_offset={at} against a window of {window}"
                )
    return window


def mla_sparse_attention(q_lift: Tensor, c_kv: Tensor, topk_indices: Tensor,
                         softmax_scale: float, q_pe: Tensor | None = None,
                         k_pe: Tensor | None = None,
                         block_table_row: Tensor | None = None,
                         written: Tensor | None = None,
                         write_offset: Tensor | None = None,
                         page_size: int = 0) -> Tensor:
    """Sparse MLA attention over each query's selected cache rows.

    Args:
        q_lift: ``[S, H, L]`` -- the absorbed Q latent, per head.
        c_kv: ``[S_kv, L]`` -- the latent KV cache, or the whole latent bank when
            ``block_table_row`` is given.
        topk_indices: ``[S, K]`` integer -- selected cache rows per query. ``-1``
            marks a column that carries no token; it is masked, not read.
        softmax_scale: positive; multiplies the raw scores.
        q_pe: ``[S, H, R]`` -- the RoPE half of Q. Present together with ``k_pe`` or
            not at all.
        k_pe: ``[S_kv, R]`` -- the RoPE half of the cache.
        block_table_row: ``[pages, 1]`` int32 -- makes the call paged. The window is
            the named ``page_size``-row blocks of the bank, in order, and a selected
            row indexes that window (``pages * page_size`` rows) exactly as it would
            index ``c_kv`` in an unpaged call.
        written: ``[tokens, L]`` -- this step's own rows, overlaid onto the window at
            row ``write_offset``.
        write_offset: ``[1, 1]`` int32 -- read on device.
        page_size: rows per page; a trace-time constant.

    Returns:
        ``[S, H, L]`` float32.

    Raises:
        MlaSparseAttentionError: for a malformed call or a geometry no kernel body
            serves. The selected-row range is checked only when the values are
            readable, so a traced call relies on the producer's contract instead.
    """
    if q_lift.ndim != 3:
        raise MlaSparseAttentionError(
            f"q_lift must be [seq, heads, latent]; got shape {tuple(q_lift.shape)}"
        )
    if c_kv.ndim != 2:
        raise MlaSparseAttentionError(
            f"c_kv must be [s_kv, latent]; got shape {tuple(c_kv.shape)}"
        )
    if topk_indices.ndim != 2:
        raise MlaSparseAttentionError(
            f"topk_indices must be [seq, topk]; got shape {tuple(topk_indices.shape)}"
        )
    if (q_pe is None) != (k_pe is None):
        raise MlaSparseAttentionError(
            "the RoPE half is both tensors or neither; got q_pe="
            f"{None if q_pe is None else tuple(q_pe.shape)} and k_pe="
            f"{None if k_pe is None else tuple(k_pe.shape)}"
        )

    seq, heads, latent = (int(d) for d in q_lift.shape)
    s_kv, cache_latent = (int(d) for d in c_kv.shape)
    if cache_latent != latent:
        raise MlaSparseAttentionError(
            f"q_lift and c_kv must share the latent rank; got {latent} against "
            f"{cache_latent}"
        )
    if int(topk_indices.shape[0]) != seq:
        raise MlaSparseAttentionError(
            f"topk_indices must carry one row per query; got "
            f"{int(topk_indices.shape[0])} rows against seq={seq}"
        )
    topk = int(topk_indices.shape[1])
    if block_table_row is not None:
        s_kv = _require_paged(c_kv, block_table_row, written, write_offset, page_size, k_pe)

    rope = 0
    if q_pe is not None:
        if q_pe.ndim != 3 or k_pe.ndim != 2:
            raise MlaSparseAttentionError(
                f"q_pe must be [seq, heads, rope] and k_pe [s_kv, rope]; got "
                f"{tuple(q_pe.shape)} and {tuple(k_pe.shape)}"
            )
        rope = int(q_pe.shape[2])
        if (int(q_pe.shape[0]), int(q_pe.shape[1])) != (seq, heads):
            raise MlaSparseAttentionError(
                f"q_pe's leading dims must match q_lift's [seq, heads] = "
                f"{(seq, heads)}; got {tuple(q_pe.shape)[:2]}"
            )
        if (int(k_pe.shape[0]), int(k_pe.shape[1])) != (s_kv, rope):
            raise MlaSparseAttentionError(
                f"k_pe must be [s_kv, rope] = {(s_kv, rope)}; got "
                f"{tuple(k_pe.shape)}"
            )

    _require_admissible(seq, heads, latent, rope, topk, s_kv, float(softmax_scale))

    # An out-of-range selected row reads memory the cache never held, and the gather's
    # out-of-bound behaviour is undefined, so it is refused here where the message can
    # name the value. -1 is admitted because it is the selector's sentinel for an empty
    # column (:data:`SENTINEL_INDEX`) and the kernels mask it; -2 and below are refused.
    # The check reads the values, so it runs in eager calls only: a traced or meta-built
    # call has none to read and relies on the producer's contract instead.
    if values_are_readable(topk_indices):
        lo, hi = int(topk_indices.min()), int(topk_indices.max())
        if lo < SENTINEL_INDEX or hi >= s_kv:
            raise MlaSparseAttentionError(
                f"every selected row must index the cache or be the {SENTINEL_INDEX} "
                f"sentinel; got the range [{lo}, {hi}] against s_kv={s_kv}"
            )

    _count_nki_dispatch()

    # The body is chosen from the shapes alone; no caller passes a flag. An exact-fit
    # latent keeps the untiled body; a ragged or wider-than-one-tile latent takes the
    # latent-tiled one; a selected-row count wider than one MM1 moving tile takes the
    # row-tiled one. The gate above refuses the last two together, so the three bodies
    # are exclusive. The seam counter above counts every dispatch whichever body runs,
    # and the per-body counters below are additive to it.
    tiled = latent % LATENT_TILE != 0 or latent > MOVING_MAX
    if tiled:
        _count_tiled_nki_dispatch()

    rows_tiled = topk > MOVING_MAX
    if rows_tiled:
        _count_row_tiled_nki_dispatch()

    if rows_tiled:
        nope_entry = mla_sparse_attention_nope_row_tiled_kernel
        rope_entry = mla_sparse_attention_rope_row_tiled_kernel
    elif tiled:
        nope_entry = mla_sparse_attention_nope_tiled_kernel
        rope_entry = mla_sparse_attention_rope_tiled_kernel
    else:
        nope_entry = mla_sparse_attention_nope_kernel
        rope_entry = mla_sparse_attention_rope_kernel

    q_lift_in = _kernel_operand(q_lift)
    c_kv_in = _kernel_operand(c_kv)
    topk_i32 = topk_indices.contiguous().to(torch.int32)
    if block_table_row is not None:
        # A step that writes nothing hands no overlay operands rather than empty ones: the
        # kernel front end builds one tile per tensor operand and refuses a zero-extent shape.
        overlaid = written is not None and int(written.shape[0]) > 0
        rows = _kernel_operand(written) if overlaid else None
        at = write_offset.contiguous().to(torch.int32) if overlaid else None
        return wrap_nki(nope_entry)(
            q_lift_in, c_kv_in, topk_i32, float(softmax_scale),
            block_table_row.contiguous().to(torch.int32), rows, at, int(page_size)
        )
    if rope == 0:
        return wrap_nki(nope_entry)(
            q_lift_in, c_kv_in, topk_i32, float(softmax_scale)
        )
    return wrap_nki(rope_entry)(
        q_lift_in,
        _kernel_operand(q_pe),
        c_kv_in,
        _kernel_operand(k_pe),
        topk_i32,
        float(softmax_scale),
    )


def mla_sparse_attention_torch_oracle(q_lift: Tensor, c_kv: Tensor,
                                      topk_indices: Tensor, softmax_scale: float,
                                      q_pe: Tensor | None = None,
                                      k_pe: Tensor | None = None) -> Tensor:
    """CPU reference for tests. Not a fallback: nothing in this module dispatches to it.

    It carries the sentinel semantics so it agrees with the kernels on the selector's
    ordinary output. The clamp is load-bearing: torch reads a negative index as a
    wrap-around, so an unclamped -1 would silently attend the last cache row.
    """
    q = q_lift.to(torch.float32)
    cache = c_kv.to(torch.float32)
    idx = topk_indices.to(torch.int64)
    keep = idx >= 0                                       # [S, K] -- False at -1
    rows = idx.clamp(min=0)                               # -1 -> 0, in bounds
    seq, heads, latent = q.shape
    out = torch.empty(seq, heads, latent, dtype=torch.float32)
    for s in range(seq):
        gathered = cache[rows[s]]                         # [K, L]
        scores = q[s] @ gathered.t()                      # [H, K]
        if q_pe is not None:
            rope_gathered = k_pe.to(torch.float32)[rows[s]]  # [K, R]
            scores = scores + q_pe.to(torch.float32)[s] @ rope_gathered.t()
        scaled = scores.masked_fill(~keep[s], float("-inf"))
        # A wholly-sentinel row is all -inf, and softmax of that is NaN rather than
        # zero -- so the zeros the kernels produce for it are written here too.
        weights = torch.nan_to_num(torch.softmax(scaled * softmax_scale, dim=-1))
        out[s] = weights @ gathered                       # [H, L]
    return out


def mla_sparse_kernel_identity() -> tuple[tuple[str, str], ...]:
    """``(module, qualname)`` of each kernel entry point this module defines.

    ``nki.jit`` returns a wrapper whose own ``__module__`` is the decorator's, so the
    wrapped function at ``.func`` is read instead.
    """
    identities = []
    for entry in (mla_sparse_attention_nope_kernel,
                  mla_sparse_attention_rope_kernel,
                  mla_sparse_attention_nope_tiled_kernel,
                  mla_sparse_attention_rope_tiled_kernel,
                  mla_sparse_attention_nope_row_tiled_kernel,
                  mla_sparse_attention_rope_row_tiled_kernel):
        func = getattr(entry, "func", None)
        target = func if func is not None else entry
        identities.append((target.__module__, target.__qualname__))
    return tuple(identities)
