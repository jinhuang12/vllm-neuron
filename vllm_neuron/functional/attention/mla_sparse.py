# SPDX-License-Identifier: Apache-2.0
"""Sparse MLA latent attention, authored in NKI for this checkpoint's geometry.

`inc-glm53f-040`. Per query, over the topk-selected cache rows::

    c_g     = c_kv[topk_indices[q]]                  # [K, L], gathered
    scores  = q_lift[q] @ c_g.T   (+ RoPE limb)      # [H, K]
    weights = softmax(scores * softmax_scale)        # [H, K]
    out[q]  = weights @ c_g                          # [H, L]

This is the absorbed-latent value path: the attention runs in the latent rank L and
the V-up projection is the CALLER's, not this kernel's.

WHY THIS IS ADAPTED AND NOT CALLED. The substrate ships a sparse-MLA context-encoding
member for the same computation, and it refuses this checkpoint on TWO independent
counts, both measured rather than argued. THE MEMBER IS NOT NAMED HERE: its symbol
and its vendor module path are held in the acceptance file, which counts their
occurrences in this source and requires zero, so writing one here would be a defect
rather than a courtesy to the reader.

  1. ITS RoPE WIDTH IS MANDATORY. Its validator asserts `0 < R <= 128`. This
     checkpoint is NoPE on the MLA half -- `qk_rope_head_dim` is `0`, and the config
     states that 0 is a VALUE here and not a placeholder -- so R == 0 is refused
     outright. Removing that constraint is this increment's declared content, which
     is why the acceptance runs AT R == 0: a run at R > 0 would prove nothing about
     this target.
  2. ITS GATHER IS A GEN4 INSTRUCTION. It gathers the selected cache rows through
     the Tensor Indirection view `NkiTensor.indirect()`, and on this campaign's trn2
     target that path refuses with `tensor_copy tensor indirection requires
     NeuronCore-v4 or later, got NeuronCore-v3`. The reading is in
     `probe-040-indirect-r3.out` in this campaign's increments directory, taken
     twice: once with the wrong index dtype, which produced a different and
     misleading refusal, and then with the substrate's own uint16 index tile, which
     produced the version bound above. So the gather is re-derived too, on
     `nisa.nc_n_gather` -- the GpSimd per-partition gather, which serves gen3.
     `probe-040-primitives-r2.out` records it gathering the right columns on two
     different partitions, not merely running.

NEITHER REFUSAL IS RESTATED FROM MEMORY: the acceptance re-takes both live, against
the installed substrate, every time it runs.

WHY THIS IS A KERNEL AND NOT TORCH GLUE (P13). Sparse attention is the per-token
device attention path on a model whose context ambition is 1M tokens. The plan block
classifies this increment KERNEL-CLASS, and a torch attention fallback here would be
the defect P13 names rather than a safety net. So this module carries NO torch
attention route: an inadmissible geometry RAISES. Section 4 of the plan permits a
`functional/` module a torch path that is either (a) the CPU oracle for the test or
(b) the constraint-violation fallback the pin's own dispatchers carry. Only (a) is
present, by name: :func:`mla_sparse_attention_torch_oracle`. Region (b) is
DELIBERATELY ABSENT, as in the landed `functional/kda/gate_clamp.py` and
`functional/attention/mla_projections.py`.

THE MX 4-PACK OUTPUT PERMUTE IS DELIBERATELY NOT CARRIED OVER. The substrate writes
each head's latent columns pre-permuted into a 4-pack order, and it does so to honour
a cross-kernel layout contract with an o_proj kernel whose projections are
unconditionally MX. This campaign does not use that consumer and is not MX-gated, so
reproducing the permute here would add an untested coupling to a kernel nothing
calls, and would put MX vocabulary on this increment's added lines where the
changeset scan requires none. The output is therefore NATURAL row-major `[S, H, L]`.
The consumer that reads it is the fork's own decode path at `inc-glm53f-042`.

WHY THE LATENT RIDES THE PARTITION AXIS. `nc_matmul` contracts the PARTITION axis, so
MM1 (which contracts the latent) needs both `q_lift` and the gathered cache latent
partition-major. That is also the layout the free-axis gather wants: `nc_n_gather`
gathers WITHIN a partition over the free dimensions, so with the latent on partitions
and the cache sequence on the free axis, one gather instruction per latent tile
selects K columns for all 128 latent components at once. MM2 contracts K instead, so
the two operands are transposed on chip between the two matmuls.

ARITHMETIC IS fp32 HERE, and the substrate's is bf16. This increment is the geometry
gate, not a precision or throughput increment: no tolerance, threshold or registered
value is this module's to move, and the landed `mla_projections.py` kernel this
campaign already carries is fp32 on the same target. A bf16 pass is a later question
with its own reading; taking it here would mix two changes under one acceptance.

Tier N harness -- the NKI simulator on a host CPU, no device and no lease::

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
    python -m pytest test/vllm_neuron/functional/attention/test_mla_sparse.py \
        -k geometry -v -s -p no:cacheprovider
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

from vllm_neuron.utils.neuron_utils import can_run_kernel

logger = logging.getLogger(__name__)

#: Partition-axis extent, ``nl.tile_size.pmax``. The latent axis is tiled to this in
#: MM1 because ``nc_matmul`` contracts the partition axis, and the gather then reads
#: 128 latent components per instruction.
LATENT_TILE = 128

#: Partition-axis extent again, under the name of the axis that occupies it in MM2.
#: MM2 contracts K, so the selected keys move onto partitions and K is chunked here.
#: Equal to :data:`LATENT_TILE` on this image, and NOT the same quantity: they bound
#: two different axes and a future image could separate them.
KEY_CHUNK = 128

#: Stationary free-axis extent, ``nl.tile_size.gemm_stationary_fmax``. The head axis
#: rides the stationary operand in both matmuls, so this bounds H.
HEAD_MAX = 128

#: Moving free-axis extent, ``nl.tile_size.gemm_moving_fmax``. K rides the moving
#: operand in MM1 and the latent rides it in MM2, so this bounds both -- and BOTH are
#: now TILED to it rather than bounded by it: the latent by `inc-glm53f-041` and the
#: selected-row count K by `inc-glm53f-093`. So this is the TILE WIDTH of each of those
#: two paths, not a ceiling on either axis. It remains a ceiling on nothing at all,
#: which is why no refusal quotes it as one any more.
MOVING_MAX = 512

#: The provenance of the four numbers above, recorded rather than the numbers being
#: presented as arbitrary. They are read from this image's own ``nl.tile_size`` and
#: the acceptance asserts each against it, so an image change reddens the test
#: instead of silently loosening the kernel.
_TILE_PROVENANCE = "nl.tile_size.{pmax, gemm_stationary_fmax, gemm_moving_fmax}"

#: This checkpoint's latent rank, `kv_lora_rank`. Held as a named constant ONLY so
#: the acceptance can say which value it exercised; the kernel itself accepts any
#: admissible L and does not compare against this.
TARGET_LATENT_RANK = 512

#: This checkpoint's MLA RoPE width, `qk_rope_head_dim`. Zero is a value here, not a
#: placeholder -- see the module docstring. Held for the same reason as above.
TARGET_ROPE_WIDTH = 0


class MlaSparseAttentionError(ValueError):
    """Raised for a geometry this kernel does not serve.

    Inadmissibility RAISES rather than falling back, because a torch attention
    fallback for kernel-class work is the P13 defect and not the remedy.
    """


@dataclass
class _MlaSparseDispatchCounters:
    """How the seam below was reached, per process.

    Private to this module, so a test can attribute a dispatch to THIS seam and to
    no other -- the discipline the landed KDA modules and `mla_projections.py` each
    use with their own counter objects.
    """

    nki_dispatch: int = 0
    torch_fallback: int = 0


_MLA_SPARSE_COUNTERS = _MlaSparseDispatchCounters()


def reset_mla_sparse_dispatch_counters() -> None:
    """Zero this seam's counters. Call immediately before a case's first call."""
    _MLA_SPARSE_COUNTERS.nki_dispatch = 0
    _MLA_SPARSE_COUNTERS.torch_fallback = 0


def mla_sparse_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last reset.

    ``torch_fallback`` can only ever read ``0``, because this module has no torch
    attention route to increment it: an inadmissible geometry raises instead (P13).
    The counter is kept so a test can STATE that reading rather than assume it, which
    is what makes the zero a measurement instead of an argument.
    """
    return (
        _MLA_SPARSE_COUNTERS.nki_dispatch,
        _MLA_SPARSE_COUNTERS.torch_fallback,
    )


def _sbuf(*shape: int, dtype=nl.float32):
    return nl.ndarray(tuple(shape), dtype=dtype, buffer=nl.sbuf)


def _psum(*shape: int):
    return nl.ndarray(tuple(shape), dtype=nl.float32, buffer=nl.psum)


def _attention_body(q_lift_hbm, c_kv_hbm, topk_hbm, softmax_scale, out_hbm,
                    q_pe_hbm=None, k_pe_hbm=None):
    """Trace the sparse latent attention. Shared by both jit entry points.

    THE RoPE LIMB IS ELIDED AT TRACE TIME, not masked at run time. When there is no
    RoPE half, ``q_pe_hbm`` and ``k_pe_hbm`` are ``None``, no RoPE buffer is
    allocated, no RoPE gather is issued and MM1 emits one fewer matmul. That is what
    makes R == 0 a first-class path rather than a zero-width special case: a
    zero-extent SBUF tile is not a thing this image allocates, so a kernel that tried
    to keep the limb and size it 0 would not trace at all.

    Shapes:
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
    n_latent = latent // LATENT_TILE
    n_chunks = topk // KEY_CHUNK
    rope = 0 if q_pe_hbm is None else q_pe_hbm.shape[2]

    # ---- the cache, transposed onto partitions ONCE for the whole call ----------
    # HBM holds it [S_kv, L]; MM1 and the gather both want [L_partition, S_kv]. One
    # `dma_transpose` per latent tile does it, and the cache is read-only for the
    # rest of the call, so this cost is per call and not per query.
    c_sb = [_sbuf(LATENT_TILE, s_kv) for _ in range(n_latent)]
    for li in range(n_latent):
        nisa.dma_transpose(
            dst=c_sb[li],
            src=c_kv_hbm.ap(pattern=[[latent, s_kv], [1, LATENT_TILE]],
                            offset=li * LATENT_TILE),
        )
    k_pe_sb = None
    if rope > 0:
        k_pe_sb = _sbuf(rope, s_kv)
        nisa.dma_transpose(
            dst=k_pe_sb,
            src=k_pe_hbm.ap(pattern=[[rope, s_kv], [1, rope]], offset=0),
        )

    # ---- per-query working set, allocated once and reused across the loop -------
    idx_sb = _sbuf(LATENT_TILE, topk, dtype=nl.uint32)
    q_lift_t = _sbuf(LATENT_TILE, n_latent, heads)
    c_g = _sbuf(LATENT_TILE, n_latent, topk)
    c_g_t = _sbuf(KEY_CHUNK, n_chunks, latent)
    p_t = _sbuf(KEY_CHUNK, n_chunks, heads)
    p = _sbuf(heads, topk)
    neg_row_max = _sbuf(heads, 1)
    exp_bias = _sbuf(heads, 1)
    row_sum = _sbuf(heads, 1)
    recip = _sbuf(heads, 1)
    out_sb = _sbuf(heads, latent)
    q_pe_t = _sbuf(rope, heads) if rope > 0 else None
    k_pe_g = _sbuf(rope, topk) if rope > 0 else None

    for q_idx in nl.affine_range(seq):
        # ---- this query's selected rows, replicated to every partition ----------
        # `nc_n_gather` gathers within a partition and reads its offsets from the
        # SAME partition, so all 128 latent partitions need the same K offsets. A
        # zero-stride partition read is one DMA and replicates them for free; the
        # alternative is a shuffle plus a fan-out copy.
        nisa.tensor_copy(
            dst=idx_sb,
            src=nl.load(
                topk_hbm.ap(pattern=[[0, LATENT_TILE], [1, topk]],
                            offset=q_idx * topk),
                dtype=nl.uint32,
            ),
        )

        # ---- gather the latent cache rows: one instruction per latent tile ------
        for li in range(n_latent):
            nisa.nc_n_gather(dst=c_g[:, li, :], data=c_sb[li], indices=idx_sb)

        # ---- this query's Q latent, transposed onto partitions ------------------
        for li in range(n_latent):
            nisa.dma_transpose(
                dst=q_lift_t[:, li, :],
                src=q_lift_hbm.ap(pattern=[[latent, heads], [1, LATENT_TILE]],
                                  offset=q_idx * heads * latent + li * LATENT_TILE),
            )

        # ---- MM1: scores[H, K] = sum over latent tiles of q_lift_t.T @ c_g ------
        scores_ps = _psum(heads, topk)
        for li in range(n_latent):
            nisa.nc_matmul(
                dst=scores_ps,
                stationary=q_lift_t[:, li, :],
                moving=c_g[:, li, :],
                accumulate=(li > 0),
            )
        if rope > 0:
            nisa.nc_n_gather(dst=k_pe_g, data=k_pe_sb, indices=idx_sb[0:rope, :])
            nisa.dma_transpose(
                dst=q_pe_t,
                src=q_pe_hbm.ap(pattern=[[rope, heads], [1, rope]],
                                offset=q_idx * heads * rope),
            )
            nisa.nc_matmul(dst=scores_ps, stationary=q_pe_t, moving=k_pe_g,
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

        # ---- softmax over K, per head row --------------------------------------
        # The max is taken NEGATED and then scaled, so `activation` can fold the
        # subtraction into its bias and produce the row sum in the same pass. The
        # scale multiplies the raw scores, so the bias must carry the same factor --
        # that is why the negated max is scaled here and not left raw.
        nisa.tensor_reduce(dst=neg_row_max, op=nl.maximum, data=scores_ps, axis=1,
                           negate=True)
        nisa.tensor_scalar(dst=exp_bias, data=neg_row_max, op0=nl.multiply,
                           operand0=softmax_scale, engine=nisa.engine.vector)
        nisa.activation(dst=p, op=nl.exp, data=scores_ps, bias=exp_bias,
                        scale=softmax_scale, reduce_op=nl.add, reduce_res=row_sum,
                        reduce_cmd=nisa.reduce_cmd.reset_reduce)
        nisa.reciprocal(dst=recip, data=row_sum)

        # ---- MM2: out[H, L] = p[H, K] @ c_g[K, L], K contracted on partitions ---
        for ck in range(n_chunks):
            ks = ck * KEY_CHUNK
            p_t_ps = _psum(KEY_CHUNK, heads)
            nisa.nc_transpose(dst=p_t_ps, data=p[:, ks:ks + KEY_CHUNK])
            nisa.tensor_copy(dst=p_t[:, ck, :], src=p_t_ps)

        pv_ps = _psum(heads, latent)
        for ck in range(n_chunks):
            nisa.nc_matmul(dst=pv_ps, stationary=p_t[:, ck, :],
                           moving=c_g_t[:, ck, :], accumulate=(ck > 0))

        # The softmax denominator is applied HERE rather than to `p`, so it costs one
        # pass over [H, L] instead of one over [H, K] plus the precision of dividing
        # before the accumulation.
        nisa.tensor_scalar(dst=out_sb, data=pv_ps, op0=nl.multiply, operand0=recip,
                           engine=nisa.engine.vector)
        nl.store(
            out_hbm.ap(pattern=[[latent, heads], [1, latent]],
                       offset=q_idx * heads * latent),
            value=out_sb,
        )


@nki.jit
def mla_sparse_attention_nope_kernel(q_lift_hbm, c_kv_hbm, topk_hbm, softmax_scale):
    """The R == 0 entry point: sparse latent attention with NO RoPE limb.

    This is THIS CHECKPOINT'S path and the one the acceptance exercises. It is a
    separate entry point rather than a flag because the limb is elided at trace time
    -- see :func:`_attention_body`.
    """
    seq, heads, latent = q_lift_hbm.shape
    out_hbm = nl.ndarray((seq, heads, latent), dtype=nl.float32, buffer=nl.shared_hbm)
    _attention_body(q_lift_hbm, c_kv_hbm, topk_hbm, softmax_scale, out_hbm)
    return out_hbm


@nki.jit
def mla_sparse_attention_rope_kernel(q_lift_hbm, q_pe_hbm, c_kv_hbm, k_pe_hbm,
                                     topk_hbm, softmax_scale):
    """The R > 0 entry point, for a checkpoint that does carry an MLA RoPE half.

    KEPT ON PURPOSE, and the reason is that this increment REMOVES the substrate's
    `0 < R <= 128` constraint rather than inverting it. A kernel that served only
    R == 0 would have swapped one refused geometry for another, and the next
    checkpoint would pay for it. It shares every line of arithmetic with the NoPE
    entry point above.
    """
    seq, heads, latent = q_lift_hbm.shape
    out_hbm = nl.ndarray((seq, heads, latent), dtype=nl.float32, buffer=nl.shared_hbm)
    _attention_body(q_lift_hbm, c_kv_hbm, topk_hbm, softmax_scale, out_hbm,
                    q_pe_hbm=q_pe_hbm, k_pe_hbm=k_pe_hbm)
    return out_hbm


# --------------------------------------------------------------------------- #
# `inc-glm53f-041` -- the tiling path for a latent rank that does not fit one tile.
#
# EVERYTHING THIS INCREMENT ADDS IS IN THIS ONE CONTIGUOUS BLOCK, on purpose. Two
# increments write this file and the plan partitions them by concern (§11.A):
# `-040` owns the kernel above, the seam and the seam's own counter; `-041` owns the
# tiling path behind that seam and its own counter. Keeping the block contiguous
# means `-040`'s body is provably untouched -- the acceptance digests
# `_attention_body`'s own source and requires the value the landed commit had -- and
# a reviewer can read this increment as one unit instead of hunting for it.
#
# WHAT WAS BOUNDED AND IS NOT ANY MORE. `-040` required the latent rank to be a
# positive multiple of 128 AND to fit one MM2 moving tile of 512, because it tiled
# neither axis. Both bounds are now tiled instead of asserted:
#
#   * the PARTITION axis, in tiles of 128 with a RAGGED LAST TILE. MM1 contracts the
#     latent, so the latent rides partitions there and 2,051 needs 17 tiles whose
#     last one is 3 deep.
#   * the MM2 MOVING free axis, in tiles of 512, again with a ragged last tile: the
#     latent is the OUTPUT free axis in MM2, so 2,051 needs 5 tiles whose last one is
#     3 wide.
#
# NOTHING IS PADDED, AND THAT IS A MEASUREMENT RATHER THAN A PREFERENCE. Before this
# was designed, every primitive the tiled path uses was run on a 3-extent tile on the
# leased host at the landed commit: a 3-partition SBUF tile, a 3-deep `nc_matmul`
# contraction, a 3-wide `nc_matmul` moving axis, `nc_transpose` in both orientations,
# `dma_transpose` into a 3-partition tail tile, and `nc_n_gather` on a 3-partition
# data tile with its index tile sliced to match. All ten readings agreed with torch to
# 0.000e+00 (`probe-041-ragged-tiles.out`, `RPROBE_OK=10`). So this image accepts
# ragged extents and a pad would be dead weight. The same probe also measured the one
# fact the acceptance's exactness claim rests on: adding 125 exact zeros to a 3-deep
# contraction is BIT-IDENTICAL, not merely close.
# --------------------------------------------------------------------------- #


@dataclass
class _MlaSparseTiledDispatchCounters:
    """How the TILED path was reached, per process.

    A separate object from `-040`'s, because the plan gives each increment its own
    counted value and has neither read the other's. This one is ADDITIVE and not
    exclusive: a tiled call increments `-040`'s seam counter as well, since that
    counter counts every dispatch through the seam whatever body runs. `-042`'s decode
    predicate reads both, so the two must compose rather than partition the traffic.
    """

    nki_dispatch: int = 0
    torch_fallback: int = 0


_MLA_SPARSE_TILED_COUNTERS = _MlaSparseTiledDispatchCounters()


def reset_mla_sparse_tiled_dispatch_counters() -> None:
    """Zero the TILED counters. `-040`'s reset does not touch these and vice versa."""
    _MLA_SPARSE_TILED_COUNTERS.nki_dispatch = 0
    _MLA_SPARSE_TILED_COUNTERS.torch_fallback = 0


def mla_sparse_tiled_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` for the TILED path since its last reset.

    ``torch_fallback`` can only read ``0`` for the same reason `-040`'s can: there is
    no torch attention route in this module to increment it, an inadmissible geometry
    raises (P13), and the counter exists so a test can STATE the zero rather than
    assume it.
    """
    return (
        _MLA_SPARSE_TILED_COUNTERS.nki_dispatch,
        _MLA_SPARSE_TILED_COUNTERS.torch_fallback,
    )


def _latent_tiles(latent: int) -> tuple[tuple[int, int], ...]:
    """``(offset, extent)`` per PARTITION-axis latent tile. The last one may be ragged.

    Derived from the latent rank rather than assumed uniform, which is the whole
    difference from `-040`: at 2,051 this returns 17 tiles, the last of extent 3.
    """
    tiles = []
    offset = 0
    while offset < latent:
        tiles.append((offset, min(LATENT_TILE, latent - offset)))
        offset += LATENT_TILE
    return tuple(tiles)


def _output_tiles(latent: int) -> tuple[tuple[int, int], ...]:
    """``(offset, extent)`` per MM2 MOVING-axis output tile. The last may be ragged.

    A different tiling of the same axis, because the latent rides partitions in MM1
    and the moving free axis in MM2, and those two axes have different extents on this
    image. At 2,051 this returns 5 tiles, the last of extent 3.
    """
    tiles = []
    offset = 0
    while offset < latent:
        tiles.append((offset, min(MOVING_MAX, latent - offset)))
        offset += MOVING_MAX
    return tuple(tiles)


def _attention_body_tiled(q_lift_hbm, c_kv_hbm, topk_hbm, softmax_scale, out_hbm,
                          q_pe_hbm=None, k_pe_hbm=None):
    """Trace sparse latent attention with the latent axis TILED on both its axes.

    Same arithmetic as `-040`'s body and the same RoPE elision at trace time; the
    difference is that every latent-indexed buffer is a LIST of tiles whose last
    member may be ragged, instead of one uniform 3-D buffer. A ragged tile cannot live
    in a uniform buffer, which is why the shape changes and not just the loop bound.

    Shapes are `-040`'s, with the latent rank now unconstrained above 1:
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

    # ---- the cache, transposed onto partitions ONCE for the whole call ----------
    # One `dma_transpose` per latent tile, exactly as `-040`, except the last tile's
    # extent is the remainder. The tail transpose was measured on its own before this
    # was written, because it was the reading least likely to be legal.
    c_sb = [_sbuf(extent, s_kv) for _, extent in lat_tiles]
    for li, (offset, extent) in enumerate(lat_tiles):
        nisa.dma_transpose(
            dst=c_sb[li],
            src=c_kv_hbm.ap(pattern=[[latent, s_kv], [1, extent]], offset=offset),
        )
    k_pe_sb = None
    if rope > 0:
        k_pe_sb = _sbuf(rope, s_kv)
        nisa.dma_transpose(
            dst=k_pe_sb,
            src=k_pe_hbm.ap(pattern=[[rope, s_kv], [1, rope]], offset=0),
        )

    # ---- per-query working set, allocated once and reused across the loop -------
    # `c_g_t` and `out_sb` stay single buffers with the latent on their FREE axis,
    # where a ragged extent needs no special case; only the partition-axis buffers
    # become lists.
    idx_sb = _sbuf(LATENT_TILE, topk, dtype=nl.uint32)
    q_lift_t = [_sbuf(extent, heads) for _, extent in lat_tiles]
    c_g = [_sbuf(extent, topk) for _, extent in lat_tiles]
    c_g_t = _sbuf(KEY_CHUNK, n_chunks, latent)
    p_t = _sbuf(KEY_CHUNK, n_chunks, heads)
    p = _sbuf(heads, topk)
    neg_row_max = _sbuf(heads, 1)
    exp_bias = _sbuf(heads, 1)
    row_sum = _sbuf(heads, 1)
    recip = _sbuf(heads, 1)
    out_sb = _sbuf(heads, latent)
    q_pe_t = _sbuf(rope, heads) if rope > 0 else None
    k_pe_g = _sbuf(rope, topk) if rope > 0 else None

    for q_idx in nl.affine_range(seq):
        # ---- this query's selected rows, replicated to every partition ----------
        nisa.tensor_copy(
            dst=idx_sb,
            src=nl.load(
                topk_hbm.ap(pattern=[[0, LATENT_TILE], [1, topk]],
                            offset=q_idx * topk),
                dtype=nl.uint32,
            ),
        )

        # ---- gather the latent cache rows: one instruction per latent tile ------
        # The index tile is SLICED to the data tile's extent. `nc_n_gather` reads its
        # offsets from the same partition it writes, so a 3-deep data tile needs a
        # 3-deep index tile and not the full 128.
        for li, (_, extent) in enumerate(lat_tiles):
            nisa.nc_n_gather(dst=c_g[li], data=c_sb[li], indices=idx_sb[0:extent, :])

        # ---- this query's Q latent, transposed onto partitions ------------------
        for li, (offset, extent) in enumerate(lat_tiles):
            nisa.dma_transpose(
                dst=q_lift_t[li],
                src=q_lift_hbm.ap(pattern=[[latent, heads], [1, extent]],
                                  offset=q_idx * heads * latent + offset),
            )

        # ---- MM1: scores[H, K] = sum over latent tiles of q_lift_t.T @ c_g ------
        # The ragged tail tile contributes a 3-deep contraction to the same
        # accumulation as the 16 full ones. Nothing is masked and nothing is padded:
        # a shorter contraction is a shorter contraction.
        scores_ps = _psum(heads, topk)
        for li in range(len(lat_tiles)):
            nisa.nc_matmul(
                dst=scores_ps,
                stationary=q_lift_t[li],
                moving=c_g[li],
                accumulate=(li > 0),
            )
        if rope > 0:
            nisa.nc_n_gather(dst=k_pe_g, data=k_pe_sb, indices=idx_sb[0:rope, :])
            nisa.dma_transpose(
                dst=q_pe_t,
                src=q_pe_hbm.ap(pattern=[[rope, heads], [1, rope]],
                                offset=q_idx * heads * rope),
            )
            nisa.nc_matmul(dst=scores_ps, stationary=q_pe_t, moving=k_pe_g,
                           accumulate=True)

        # ---- the gathered cache, transposed for MM2 -----------------------------
        # Each latent tile transposes into its own slice of the latent free axis, so
        # the ragged tail lands as a 3-wide slice rather than a padded 128-wide one.
        for ck in range(n_chunks):
            ks = ck * KEY_CHUNK
            for li, (offset, extent) in enumerate(lat_tiles):
                c_g_t_ps = _psum(KEY_CHUNK, extent)
                nisa.nc_transpose(dst=c_g_t_ps, data=c_g[li][:, ks:ks + KEY_CHUNK])
                nisa.tensor_copy(
                    dst=c_g_t[:, ck, offset:offset + extent],
                    src=c_g_t_ps,
                )

        # ---- softmax over K, per head row --------------------------------------
        # Untouched by the tiling: the scores tile is [H, K] whatever the latent rank
        # was, so this is `-040`'s chain verbatim.
        nisa.tensor_reduce(dst=neg_row_max, op=nl.maximum, data=scores_ps, axis=1,
                           negate=True)
        nisa.tensor_scalar(dst=exp_bias, data=neg_row_max, op0=nl.multiply,
                           operand0=softmax_scale, engine=nisa.engine.vector)
        nisa.activation(dst=p, op=nl.exp, data=scores_ps, bias=exp_bias,
                        scale=softmax_scale, reduce_op=nl.add, reduce_res=row_sum,
                        reduce_cmd=nisa.reduce_cmd.reset_reduce)
        nisa.reciprocal(dst=recip, data=row_sum)

        # ---- MM2: out[H, L] = p[H, K] @ c_g[K, L], K contracted on partitions ---
        for ck in range(n_chunks):
            ks = ck * KEY_CHUNK
            p_t_ps = _psum(KEY_CHUNK, heads)
            nisa.nc_transpose(dst=p_t_ps, data=p[:, ks:ks + KEY_CHUNK])
            nisa.tensor_copy(dst=p_t[:, ck, :], src=p_t_ps)

        # THE SECOND TILING, and the one `-040`'s bound was really about: the latent
        # is MM2's moving free axis, so the output is produced 512 columns at a time
        # and the last tile is 3 wide. The denominator is applied per output tile, for
        # `-040`'s reason -- one pass over [H, L] rather than one over [H, K].
        for offset, extent in out_tiles:
            pv_ps = _psum(heads, extent)
            for ck in range(n_chunks):
                nisa.nc_matmul(dst=pv_ps, stationary=p_t[:, ck, :],
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
                                           softmax_scale):
    """The tiled R == 0 entry point, and the one this increment's acceptance runs."""
    seq, heads, latent = q_lift_hbm.shape
    out_hbm = nl.ndarray((seq, heads, latent), dtype=nl.float32, buffer=nl.shared_hbm)
    _attention_body_tiled(q_lift_hbm, c_kv_hbm, topk_hbm, softmax_scale, out_hbm)
    return out_hbm


@nki.jit
def mla_sparse_attention_rope_tiled_kernel(q_lift_hbm, q_pe_hbm, c_kv_hbm, k_pe_hbm,
                                           topk_hbm, softmax_scale):
    """The tiled R > 0 entry point.

    KEPT FOR `-040`'S REASON, which applies with more force here: this increment
    removes a latent-width bound, and serving the wide latent only when there is no
    RoPE half would have swapped one refused geometry for another. The alternative was
    to refuse the tiled RoPE geometry, which would have minted a refusal message
    promising a future increment nobody owns -- the defect this file already carries
    twice on the topk axis and which is under design review. The RoPE limb contracts
    its own axis into the same score tile, so it is latent-independent and shares
    every line with the entry point above.
    """
    seq, heads, latent = q_lift_hbm.shape
    out_hbm = nl.ndarray((seq, heads, latent), dtype=nl.float32, buffer=nl.shared_hbm)
    _attention_body_tiled(q_lift_hbm, c_kv_hbm, topk_hbm, softmax_scale, out_hbm,
                          q_pe_hbm=q_pe_hbm, k_pe_hbm=k_pe_hbm)
    return out_hbm


# --------------------------------------------------------------------------- #
# `inc-glm53f-093` -- the SELECTED-ROW tiling: `topk` past one MM1 moving tile.
#
# EVERYTHING THIS INCREMENT ADDS IS IN THIS ONE CONTIGUOUS BLOCK, for `-041`'s reason.
# Three increments write this file and the plan partitions them by concern (§11.A.1):
# `-040` owns the kernel and the seam, `-041` owns the LATENT tiling, and this block
# owns the SELECTED-ROW tiling. `-040`'s `_attention_body` and `-041`'s
# `_attention_body_tiled` are untouched, and the acceptance digests both bodies over
# spans found by symbol and requires the values they had at this block's base commit.
#
# WHAT WAS BOUNDED AND IS NOT ANY MORE. This checkpoint's selector picks 2,048 cache
# rows per query (`index_topk`). `-040` bounded K at one MM1 moving tile of 512 because
# it tiled the axis not at all, so the production row count was refused by the gate and
# no block read the kernel at the width the decode path passes. K is now TILED:
#
#   * MM1's MOVING free axis, in tiles of 512. At 2,048 that is four score tiles.
#   * MM2's PARTITION axis, in chunks of 128, accumulating -- which `-040` already did
#     within one tile and this block does within each tile, four chunks per tile.
#
# THE SOFTMAX IS THE WHOLE CONTENT, and it is why this is kernel-class work rather than
# a loop bound. A tile cannot be exponentiated against a max nobody has seen yet, so
# each tile is exponentiated against ITS OWN row max and the running denominator and the
# running output accumulator are RESCALED when a later tile raises that max. Both
# rescale factors are ``exp`` of a non-positive number by construction, so neither can
# overflow -- which is the reason the max is carried at all instead of every tile being
# exponentiated against the first tile's bias.
#
# THE MERGE ARITHMETIC WAS MEASURED BEFORE IT WAS WRITTEN, in
# `probe-093-merge-algebra.out` (`PROBE_093_MERGE=CLEAN`, 8/8): one tile is BIT-IDENTICAL
# to the one-shot softmax, four tiles agree to fp64 rounding, and each rescale's sign is
# shown to matter by flipping it. Revision 1 of that probe failed its own control,
# because on data whose scores rise with the row index the tile rescale is exactly 1.0
# and a sign flip on it is invisible; the acceptance's inputs are ORDERED for that
# reason, and the reading is kept beside the probe.
#
# THE COMBINATION OF THE TWO TILINGS IS REFUSED, BY NAME, at the gate: a call that is
# both row-tiled and latent-tiled raises. No increment owns it and none is promised. So
# this body may assume what the gate guarantees -- the latent is an exact multiple of
# the partition tile and fits one MM2 moving tile -- and it does.
# --------------------------------------------------------------------------- #


@dataclass
class _MlaSparseRowTiledDispatchCounters:
    """How the ROW-TILED path was reached, per process.

    A third counter object, on `-041`'s form and for its reason: the plan gives each
    increment its own counted value and has none read another's. ADDITIVE, not
    exclusive: a row-tiled call increments `-040`'s seam counter as well, because that
    counter counts every dispatch through the seam whatever body runs. `-042`'s decode
    predicate reads all three, so they must compose rather than partition the traffic.
    """

    nki_dispatch: int = 0
    torch_fallback: int = 0


_MLA_SPARSE_ROW_TILED_COUNTERS = _MlaSparseRowTiledDispatchCounters()


def reset_mla_sparse_row_tiled_dispatch_counters() -> None:
    """Zero the ROW-TILED counters. The other two resets do not touch these."""
    _MLA_SPARSE_ROW_TILED_COUNTERS.nki_dispatch = 0
    _MLA_SPARSE_ROW_TILED_COUNTERS.torch_fallback = 0


def mla_sparse_row_tiled_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` for the ROW-TILED path since its last reset.

    ``torch_fallback`` can only read ``0`` for the reason the other two can: this module
    has no torch attention route to increment it, an inadmissible geometry raises (P13),
    and the counter exists so a test can STATE the zero rather than assume it.
    """
    return (
        _MLA_SPARSE_ROW_TILED_COUNTERS.nki_dispatch,
        _MLA_SPARSE_ROW_TILED_COUNTERS.torch_fallback,
    )


def _score_tiles(topk: int) -> tuple[tuple[int, int], ...]:
    """``(offset, extent)`` per MM1 MOVING-axis score tile. At 2,048 this is 4 x 512.

    A LAST TILE MAY BE NARROWER, and it is never ragged in the sense `-041`'s tail is:
    the gate admits only a multiple of :data:`KEY_CHUNK`, and :data:`MOVING_MAX` is
    itself a multiple of it, so every extent this returns is a whole number of MM2 key
    chunks. `topk=640` returns tiles of 512 and 128. That invariant is what lets the
    body chunk each tile without a partial-chunk case.
    """
    tiles = []
    offset = 0
    while offset < topk:
        tiles.append((offset, min(MOVING_MAX, topk - offset)))
        offset += MOVING_MAX
    return tuple(tiles)


def _attention_body_row_tiled(q_lift_hbm, c_kv_hbm, topk_hbm, softmax_scale, out_hbm,
                              q_pe_hbm=None, k_pe_hbm=None):
    """Trace sparse latent attention with the SELECTED-ROW axis tiled, softmax carried.

    Same arithmetic as `-040`'s body per tile, and the same RoPE elision at trace time.
    The difference is that the per-query work is done once per SCORE TILE and the
    softmax is merged across the tiles, so the denominator and the output accumulator
    are running values instead of one-shot ones.

    AT ONE SCORE TILE THIS IS `-040`'S BODY. The merge block is emitted only when there
    is more than one tile -- a trace-time branch, not a run-time mask -- so a call at
    ``topk <= MOVING_MAX`` traces `-040`'s instruction sequence with the Q transpose
    hoisted above the (single-iteration) tile loop and one exact fp32 copy of the
    accumulator added. Neither changes a value, which is why the acceptance can claim
    BIT-IDENTITY there rather than a tolerance.

    THE GATE'S GUARANTEE IS USED. The seam routes here only when the latent is an exact
    multiple of :data:`LATENT_TILE` and fits one MM2 moving tile, because the
    combination of both tilings is refused. So ``n_latent`` divides exactly and MM2's
    moving extent is the whole latent. A DIRECT call on this entry point with a ragged
    latent is outside what it serves; the seam is the supported caller.

    Shapes are `-040`'s, with the selected-row count now unbounded above 1:
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
    n_latent = latent // LATENT_TILE
    rope = 0 if q_pe_hbm is None else q_pe_hbm.shape[2]
    tiles = _score_tiles(topk)
    single = len(tiles) == 1
    tile_max = max(extent for _, extent in tiles)
    chunk_max = tile_max // KEY_CHUNK

    # ---- the cache, transposed onto partitions ONCE for the whole call ----------
    # `-040`'s exactly. The cache does not depend on which score tile is being served,
    # so this stays outside both loops and the widened K costs nothing here.
    c_sb = [_sbuf(LATENT_TILE, s_kv) for _ in range(n_latent)]
    for li in range(n_latent):
        nisa.dma_transpose(
            dst=c_sb[li],
            src=c_kv_hbm.ap(pattern=[[latent, s_kv], [1, LATENT_TILE]],
                            offset=li * LATENT_TILE),
        )
    k_pe_sb = None
    if rope > 0:
        k_pe_sb = _sbuf(rope, s_kv)
        nisa.dma_transpose(
            dst=k_pe_sb,
            src=k_pe_hbm.ap(pattern=[[rope, s_kv], [1, rope]], offset=0),
        )

    # ---- the working set, sized to ONE SCORE TILE and reused by every tile -------
    # This is the point of tiling rather than widening: every buffer below is the size
    # `-040`'s was at `topk == MOVING_MAX`, whatever K the caller passes.
    idx_sb = _sbuf(LATENT_TILE, tile_max, dtype=nl.uint32)
    q_lift_t = _sbuf(LATENT_TILE, n_latent, heads)
    c_g = _sbuf(LATENT_TILE, n_latent, tile_max)
    c_g_t = _sbuf(KEY_CHUNK, chunk_max, latent)
    p_t = _sbuf(KEY_CHUNK, chunk_max, heads)
    p = _sbuf(heads, tile_max)
    neg_row_max = _sbuf(heads, 1)
    exp_bias = _sbuf(heads, 1)
    tile_sum = _sbuf(heads, 1)
    recip = _sbuf(heads, 1)
    out_sb = _sbuf(heads, latent)
    q_pe_t = _sbuf(rope, heads) if rope > 0 else None
    k_pe_g = _sbuf(rope, tile_max) if rope > 0 else None

    # ---- the running state carried ACROSS score tiles ----------------------------
    # `run_pos` holds `softmax_scale * (running row max)` -- the POSITIVE form, because
    # merging two maxima needs `maximum` and this image's landed kernels use
    # `tensor_tensor(op=nl.maximum)`. `-040`'s chain produces the NEGATED max, so this
    # block negates it once per tile with a scalar multiply.
    run_pos = _sbuf(heads, 1)
    run_sum = _sbuf(heads, 1)
    acc = _sbuf(heads, latent)
    # Merge scratch. FRESH DESTINATIONS, never a write back onto an operand -- the idiom
    # the landed `kda/decode_state.py` records as the reviewed one for this file family.
    tile_pos = _sbuf(heads, 1)
    new_pos = _sbuf(heads, 1)
    d_acc = _sbuf(heads, 1)
    d_tile = _sbuf(heads, 1)
    c_acc = _sbuf(heads, 1)
    c_tile = _sbuf(heads, 1)
    sum_kept = _sbuf(heads, 1)
    sum_added = _sbuf(heads, 1)
    sum_new = _sbuf(heads, 1)
    acc_kept = _sbuf(heads, latent)
    pv_added = _sbuf(heads, latent)
    acc_new = _sbuf(heads, latent)

    for q_idx in nl.affine_range(seq):
        # ---- this query's Q latent, transposed onto partitions ONCE --------------
        # HOISTED ABOVE THE TILE LOOP on purpose: Q does not depend on which selected
        # rows are being scored, so transposing it per tile would repeat the work once
        # per tile for no reading.
        for li in range(n_latent):
            nisa.dma_transpose(
                dst=q_lift_t[:, li, :],
                src=q_lift_hbm.ap(pattern=[[latent, heads], [1, LATENT_TILE]],
                                  offset=q_idx * heads * latent + li * LATENT_TILE),
            )
        if rope > 0:
            nisa.dma_transpose(
                dst=q_pe_t,
                src=q_pe_hbm.ap(pattern=[[rope, heads], [1, rope]],
                                offset=q_idx * heads * rope),
            )

        for ti, (ks, extent) in enumerate(tiles):
            n_chunks = extent // KEY_CHUNK

            # ---- THIS TILE's selected rows, replicated to every partition --------
            # The offset is where the tiling shows: `-040` loaded all K rows of the
            # query, this loads the tile's slice of them.
            nisa.tensor_copy(
                dst=idx_sb[:, 0:extent],
                src=nl.load(
                    topk_hbm.ap(pattern=[[0, LATENT_TILE], [1, extent]],
                                offset=q_idx * topk + ks),
                    dtype=nl.uint32,
                ),
            )

            # ---- gather this tile's cache rows: one instruction per latent tile ---
            for li in range(n_latent):
                nisa.nc_n_gather(dst=c_g[:, li, 0:extent], data=c_sb[li],
                                 indices=idx_sb[:, 0:extent])

            # ---- MM1 over this tile: scores[H, extent] ---------------------------
            scores_ps = _psum(heads, extent)
            for li in range(n_latent):
                nisa.nc_matmul(
                    dst=scores_ps,
                    stationary=q_lift_t[:, li, :],
                    moving=c_g[:, li, 0:extent],
                    accumulate=(li > 0),
                )
            if rope > 0:
                nisa.nc_n_gather(dst=k_pe_g[:, 0:extent], data=k_pe_sb,
                                 indices=idx_sb[0:rope, 0:extent])
                nisa.nc_matmul(dst=scores_ps, stationary=q_pe_t,
                               moving=k_pe_g[:, 0:extent], accumulate=True)

            # ---- the gathered cache, transposed for MM2 --------------------------
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

            # ---- softmax over THIS TILE's keys -- `-040`'s chain, verbatim -------
            # Against the TILE's own max, which is the only max available yet. The
            # merge below is what makes that legitimate.
            nisa.tensor_reduce(dst=neg_row_max, op=nl.maximum, data=scores_ps, axis=1,
                               negate=True)
            nisa.tensor_scalar(dst=exp_bias, data=neg_row_max, op0=nl.multiply,
                               operand0=softmax_scale, engine=nisa.engine.vector)
            nisa.activation(dst=p[:, 0:extent], op=nl.exp, data=scores_ps,
                            bias=exp_bias, scale=softmax_scale, reduce_op=nl.add,
                            reduce_res=tile_sum,
                            reduce_cmd=nisa.reduce_cmd.reset_reduce)

            # ---- MM2 over this tile: pv[H, L] = p[H, extent] @ c_g[extent, L] ----
            for ck in range(n_chunks):
                cs = ck * KEY_CHUNK
                p_t_ps = _psum(KEY_CHUNK, heads)
                nisa.nc_transpose(dst=p_t_ps, data=p[:, cs:cs + KEY_CHUNK])
                nisa.tensor_copy(dst=p_t[:, ck, :], src=p_t_ps)
            pv_ps = _psum(heads, latent)
            for ck in range(n_chunks):
                nisa.nc_matmul(dst=pv_ps, stationary=p_t[:, ck, :],
                               moving=c_g_t[:, ck, :], accumulate=(ck > 0))

            # ---- THE MERGE ------------------------------------------------------
            if single:
                # ONE TILE: no merge is emitted at all. This arm is what makes the
                # acceptance's bit-identity claim against `-040` a claim about the same
                # arithmetic rather than about two implementations that agree closely.
                nisa.tensor_copy(dst=acc, src=pv_ps)
                nisa.tensor_copy(dst=run_sum, src=tile_sum)
                continue

            nisa.tensor_scalar(dst=tile_pos, data=exp_bias, op0=nl.multiply,
                               operand0=-1.0, engine=nisa.engine.vector)
            if ti == 0:
                # The first tile INITIALISES the running state; there is nothing to
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
            # ONCE after the last tile, not per tile -- `-040`'s reason, one pass over
            # [H, L] instead of one over [H, K], and here it also keeps the accumulator
            # in the numerator's scale so a rescale is a single multiply.
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
                                               softmax_scale):
    """The row-tiled R == 0 entry point, and the one this increment's acceptance runs."""
    seq, heads, latent = q_lift_hbm.shape
    out_hbm = nl.ndarray((seq, heads, latent), dtype=nl.float32, buffer=nl.shared_hbm)
    _attention_body_row_tiled(q_lift_hbm, c_kv_hbm, topk_hbm, softmax_scale, out_hbm)
    return out_hbm


@nki.jit
def mla_sparse_attention_rope_row_tiled_kernel(q_lift_hbm, q_pe_hbm, c_kv_hbm, k_pe_hbm,
                                               topk_hbm, softmax_scale):
    """The row-tiled R > 0 entry point.

    KEPT FOR `-040`'S AND `-041`'S REASON: this increment removes a bound on the
    selected-row count, and serving the production row count only when there is no RoPE
    half would swap one refused geometry for another and mint a refusal promising an
    increment nobody owns. The RoPE limb contracts its own axis into the same score
    tile, so it is row-tiled exactly like the latent limb and shares every line above.
    IT IS AUTHORED AND NOT EXERCISED: this checkpoint is NoPE on the MLA half, so the
    acceptance runs at R == 0 and this entry point traces only when a caller passes a
    RoPE half. That is the same disclosure `-040` and `-041` each make.
    """
    seq, heads, latent = q_lift_hbm.shape
    out_hbm = nl.ndarray((seq, heads, latent), dtype=nl.float32, buffer=nl.shared_hbm)
    _attention_body_row_tiled(q_lift_hbm, c_kv_hbm, topk_hbm, softmax_scale, out_hbm,
                              q_pe_hbm=q_pe_hbm, k_pe_hbm=k_pe_hbm)
    return out_hbm


def _require_admissible(seq: int, heads: int, latent: int, rope: int, topk: int,
                        s_kv: int, softmax_scale: float) -> None:
    """Raise unless this kernel serves the geometry, rather than fall back (P13).

    EVERY BOUND HERE THAT NAMES A HARDWARE AXIS SAYS WHICH ONE; the remaining bounds
    are plain positivity, which is all that is left of an axis once it is TILED rather
    than bounded -- the latent rank is that case, under `inc-glm53f-041`. There is
    deliberately NO bound on the RoPE width being positive
    -- that absence is the increment. There is deliberately no bound on S or S_kv
    either: the query loop and the cache free axis both walk whatever they are given.
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
            f"the latent rank must be positive; got latent={latent}. inc-glm53f-041 "
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
            f"A width that is not a multiple is refused rather than padded, and no "
            f"increment is promised for it"
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
    """True when the NKI route is available AND serves this geometry."""
    if not can_run_kernel(reference):
        return False
    try:
        _require_admissible(seq, heads, latent, rope, topk, s_kv, softmax_scale)
    except MlaSparseAttentionError:
        return False
    return True


def mla_sparse_attention(q_lift: Tensor, c_kv: Tensor, topk_indices: Tensor,
                         softmax_scale: float, q_pe: Tensor | None = None,
                         k_pe: Tensor | None = None) -> Tensor:
    """The counted seam. Returns ``[S, H, L]`` float32.

    ``q_lift`` is ``[S, H, L]``, ``c_kv`` is ``[S_kv, L]``, ``topk_indices`` is
    ``[S, K]`` integer. ``q_pe`` ``[S, H, R]`` and ``k_pe`` ``[S_kv, R]`` are the
    RoPE half and are BOTH present or BOTH absent; absent is this checkpoint.
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

    # An out-of-range selected row reads memory the cache never held, and the gather
    # instruction's own out-of-bound behaviour is documented as undefined -- so it is
    # refused here, where the message can name the offending value.
    lo, hi = int(topk_indices.min()), int(topk_indices.max())
    if lo < 0 or hi >= s_kv:
        raise MlaSparseAttentionError(
            f"every selected row must index the cache; got the range [{lo}, {hi}] "
            f"against s_kv={s_kv}"
        )

    _MLA_SPARSE_COUNTERS.nki_dispatch += 1

    # `inc-glm53f-041`'s branch. The seam decides from the WIDTH ALONE: an exact-fit
    # latent keeps `-040`'s body and anything ragged or wider than one MM2 moving tile
    # takes the tiled one. No caller passes a flag, so no caller can pick the wrong
    # path, and `can_run_mla_sparse_attention` needs no second question.
    #
    # THE COUNTER ABOVE IS DELIBERATELY NOT MADE CONDITIONAL. It counts every dispatch
    # through this seam whichever body runs, which is what `inc-glm53f-042`'s decode
    # predicate reads. The tiled counter here is ADDITIVE rather than exclusive, so a
    # tiled call is counted twice over -- once as a seam dispatch and once as a tiled
    # dispatch -- and an exact-fit call increments only the seam counter. That
    # difference is what a test can measure, and it is measured.
    tiled = latent % LATENT_TILE != 0 or latent > MOVING_MAX
    if tiled:
        _MLA_SPARSE_TILED_COUNTERS.nki_dispatch += 1

    # `inc-glm53f-093`'s branch, added beside `-041`'s and on the same terms. The seam
    # decides from the SELECTED-ROW COUNT alone: a count wider than one MM1 moving tile
    # takes the row-tiled body. The two conditions cannot both hold, because the gate
    # above refuses the combination -- so these are three exclusive bodies and not a
    # matrix of four.
    #
    # THIS COUNTER IS ADDITIVE TOO, for the reason `-041`'s is: `-040`'s seam counter
    # above already counted this dispatch, and `-042`'s decode predicate reads all three
    # counters per step and needs them to compose. A row-tiled call therefore reads 1 on
    # the seam counter and 1 here; an exact-fit, narrow call reads 1 and 0.
    rows_tiled = topk > MOVING_MAX
    if rows_tiled:
        _MLA_SPARSE_ROW_TILED_COUNTERS.nki_dispatch += 1

    if rows_tiled:
        nope_entry = mla_sparse_attention_nope_row_tiled_kernel
        rope_entry = mla_sparse_attention_rope_row_tiled_kernel
    elif tiled:
        nope_entry = mla_sparse_attention_nope_tiled_kernel
        rope_entry = mla_sparse_attention_rope_tiled_kernel
    else:
        nope_entry = mla_sparse_attention_nope_kernel
        rope_entry = mla_sparse_attention_rope_kernel

    q_lift_f32 = q_lift.contiguous().to(torch.float32)
    c_kv_f32 = c_kv.contiguous().to(torch.float32)
    topk_i32 = topk_indices.contiguous().to(torch.int32)
    if rope == 0:
        return wrap_nki(nope_entry)(
            q_lift_f32, c_kv_f32, topk_i32, float(softmax_scale)
        )
    return wrap_nki(rope_entry)(
        q_lift_f32,
        q_pe.contiguous().to(torch.float32),
        c_kv_f32,
        k_pe.contiguous().to(torch.float32),
        topk_i32,
        float(softmax_scale),
    )


def mla_sparse_attention_torch_oracle(q_lift: Tensor, c_kv: Tensor,
                                      topk_indices: Tensor, softmax_scale: float,
                                      q_pe: Tensor | None = None,
                                      k_pe: Tensor | None = None) -> Tensor:
    """CPU oracle -- section 4 clause (a), and the ONLY torch arithmetic here.

    This is the region the acceptance excludes BY NAME when it screens this module
    for a torch attention path. It is not a fallback and nothing dispatches to it:
    the seam above never calls it, so no input can reach it except a test's.
    """
    q = q_lift.to(torch.float32)
    cache = c_kv.to(torch.float32)
    idx = topk_indices.to(torch.int64)
    seq, heads, latent = q.shape
    out = torch.empty(seq, heads, latent, dtype=torch.float32)
    for s in range(seq):
        gathered = cache[idx[s]]                          # [K, L]
        scores = q[s] @ gathered.t()                      # [H, K]
        if q_pe is not None:
            rope_gathered = k_pe.to(torch.float32)[idx[s]]  # [K, R]
            scores = scores + q_pe.to(torch.float32)[s] @ rope_gathered.t()
        weights = torch.softmax(scores * softmax_scale, dim=-1)
        out[s] = weights @ gathered                       # [H, L]
    return out


def mla_sparse_kernel_identity() -> tuple[tuple[str, str], ...]:
    """``(module, qualname)`` of each kernel entry point this module authors.

    Read by the acceptance to prove the kernels under test are authored HERE rather
    than imported from the substrate. THE UNWRAP IS THE WHOLE READING: ``nki.jit``
    returns a wrapper whose own ``__module__`` is the substrate's, so reading the
    attribute off the decorated object reports the same answer for an authored kernel
    and an imported one alike. Unwrapping ``.func`` first is what makes the two cases
    differ, and it is the form the landed `mla_projections.py` uses.
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
