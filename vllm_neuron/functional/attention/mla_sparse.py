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
#: operand in MM1 and the latent rides it in MM2, so this bounds both -- which is
#: why L must fit one moving tile here and why widening it is `inc-glm53f-041`'s
#: tiling increment rather than a constant edit.
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


def _require_admissible(seq: int, heads: int, latent: int, rope: int, topk: int,
                        s_kv: int, softmax_scale: float) -> None:
    """Raise unless this kernel serves the geometry, rather than fall back (P13).

    EVERY BOUND HERE IS AN AXIS THE KERNEL PUTS ON A BOUNDED HARDWARE AXIS, and each
    names which one. There is deliberately NO bound on the RoPE width being positive
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
    if latent < 1 or latent % LATENT_TILE != 0:
        raise MlaSparseAttentionError(
            f"the latent rank rides the partition axis in tiles of {LATENT_TILE}, so "
            f"it must be a positive multiple of it; got latent={latent}"
        )
    if latent > MOVING_MAX:
        raise MlaSparseAttentionError(
            f"the latent rank rides the matmul moving free axis in MM2, bounded at "
            f"{MOVING_MAX} ({_TILE_PROVENANCE}); got latent={latent}. This "
            f"checkpoint's kv_lora_rank is {TARGET_LATENT_RANK}, which fits; "
            f"widening past it is inc-glm53f-041's tiling increment"
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
            f"An arbitrary width is inc-glm53f-041's padding increment"
        )
    if topk > MOVING_MAX:
        raise MlaSparseAttentionError(
            f"the selected-row count rides the matmul moving free axis in MM1, "
            f"bounded at {MOVING_MAX} ({_TILE_PROVENANCE}); got topk={topk}. "
            f"Tiling past it is inc-glm53f-041's increment"
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
    q_lift_f32 = q_lift.contiguous().to(torch.float32)
    c_kv_f32 = c_kv.contiguous().to(torch.float32)
    topk_i32 = topk_indices.contiguous().to(torch.int32)
    if rope == 0:
        return wrap_nki(mla_sparse_attention_nope_kernel)(
            q_lift_f32, c_kv_f32, topk_i32, float(softmax_scale)
        )
    return wrap_nki(mla_sparse_attention_rope_kernel)(
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
    for entry in (mla_sparse_attention_nope_kernel, mla_sparse_attention_rope_kernel):
        func = getattr(entry, "func", None)
        target = func if func is not None else entry
        identities.append((target.__module__, target.__qualname__))
    return tuple(identities)
