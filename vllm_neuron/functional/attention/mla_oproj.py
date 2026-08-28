# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 grouped MLA output projection (``wo_a`` bmm + ``wo_b`` row-parallel).

WHY this is a torch composition and not an NKI wrapper: the nkilib kernel this
row cites by name (``mla_vup_oproj_cte``) is ABSENT from the installed neuron
wheel (family interface contract §0), so there is nothing to gate on and no
``_can_use_*`` dispatch pair — unlike ``o_proj.py``, which does have a kernel to
fall back from. The body is real math and stays inside the traceable
static-shape torch subset.

PRIMARY source of truth for the MATH is DeepSeek's own reference implementation
shipped in the pinned checkpoint repo. It outranks the derived spec reports:

  * ``dsv4_ref/model.py:539`` — ``apply_rotary_emb(o[..., -rd:], freqs_cis,
    True)``: the INVERSE RoPE on the attention output happens BEFORE this op.
  * ``dsv4_ref/model.py:542-546`` — ``o = o.view(bsz, seqlen, n_local_groups,
    -1)``; ``wo_a = wo_a.weight.view(n_local_groups, o_lora_rank, -1)``;
    ``o = einsum("bsgd,grd->bsgr", o, wo_a)``. The contracted axis ``d`` is the
    group's whole ``n_heads * head_dim / o_groups`` (= 8 heads x 512 = 4096)
    latent width.
  * ``dsv4_ref/model.py:547`` — ``x = wo_b(o.flatten(2))``.
  * ``dsv4_ref/model.py:172-186`` — ``RowParallelLinear``: shards the INPUT dim,
    then ``y = y.float(); dist.all_reduce(y)`` and only then ``y.type_as(x)``.
    The TP reduction is therefore an FP32 reduction; see ``out_dtype`` below.
  * ``dsv4_ref/model.py:468`` — ``wo_a`` is bf16 in the reference (``:544-545``
    notes the checkpoint tensor is fp8 and an fp8 einsum "could" be used); the
    port takes the fp8 path whenever a scale is supplied.
  * ``dsv4_ref/convert.py:57-59`` + ``:113-116`` — sharding: ``wo_a`` splits
    dim 0 (out), ``wo_b`` splits dim 1 (in), both via
    ``param.narrow(dim, i * shard, shard)``, i.e. rank ``i`` owns a CONTIGUOUS
    slice. That is what fixes which K columns this core's ``wo_b`` consumes.
  * ``dsv4_ref/model.py:114-126, 145, 150`` — weights are ``[out, in]``;
    unquantized is ``F.linear``, fp8 is a 128x128-block-scaled GEMM.

Corroborating upstream-vLLM refs (tag ``v0.21.0``), kept because they pin the
port's INTERFACE rather than the math:
  * ``vllm/model_executor/models/deepseek_v4.py:992-1009`` — ``wo_a`` is a
    ``ColumnParallelLinear`` with ``is_bmm = True`` and
    ``bmm_batch_size = n_local_groups``, mapping
    ``n_heads * head_dim / o_groups -> o_groups * o_lora_rank``; ``wo_b`` is a
    ``RowParallelLinear`` mapping ``o_groups * o_lora_rank -> hidden_size``.
  * ``vllm/model_executor/layers/deepseek_v4_attention.py:336-354`` — the
    grouped ``einsum("bhr,hdr->bhd", o_fp8, wo_a_fp8)`` producing
    ``z: [T, n_local_groups, o_lora_rank]``, flattened into ``wo_b``.
  * ``vllm/v1/attention/ops/rocm_aiter_mla_sparse.py:875-905`` — the bf16
    reference of the same grouped bmm, ``einsum("tgd,grd->tgr", ...)``.
  * ``dataflow-shapes.md`` §A rows ``wo_a``/``wo_b`` and §B steps 13-14.

NOT SETTLED BY THE REFERENCE: both the reference and upstream vLLM assume
``tp_size <= o_groups`` (``model.py:457``: ``n_local_groups = n_groups //
world_size`` must be >= 1), so a whole group's 8 heads always live on one core
and the sum over them is a LOCAL einsum contraction. This port runs
TP=64 > o_groups=8 with one head per core, which splits that contraction across
8 cores and makes it a cross-core reduction. The 8-core intra-group all-reduce
below is therefore the port's own scheme (``dataflow-shapes.md`` GAPS-5), only
algebraically justified by the reference, not exhibited by it.
"""

from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor


def _linear(
    x: Tensor,
    weight: Tensor,
    weight_scale: Optional[Tensor],
    out_dtype: torch.dtype,
) -> Tensor:
    """One ``[out, in]``-oriented linear leg, block-FP8 or plain.

    Deliberately duplicated from ``mla_qkv.py`` rather than imported: keeping
    this module free of sibling imports means it can be loaded (and unit-tested)
    without pulling in the rest of the functional package's import graph.

    Weights in this checkpoint are stored ``[out, in]`` and the reference's
    unquantized path is ``F.linear(x, weight)`` = ``x @ weight.t()``
    (``dsv4_ref/model.py:114-126, 145, 150``) — the same orientation
    ``NF.block_fp8_linear`` wants, so no ``is_storage_transposed`` fiddling
    (family interface contract §1). The reference's quantized path is a
    128x128-block-scaled fp8 GEMM (``model.py:122-124``,
    ``convert.py:126-127``), which is the envelope requested below.

    When ``weight_scale is None`` the weight is taken as unquantized and a plain
    ``matmul`` against ``w.t()`` is used, which is what keeps this module
    unit-testable on ordinary bf16/fp32 tensors with no fp8 support present.

    ``NF.block_fp8_linear`` is resolved LAZILY as an attribute of the functional
    package because it is authored by a different node: importing this module
    must never depend on that op existing yet.
    """
    if weight_scale is None:
        # Cast rather than assert on dtype so unquantized unit tests can mix a
        # bf16 activation with an fp32 reference weight.
        return torch.matmul(x, weight.t().to(x.dtype)).to(out_dtype)

    import vllm_neuron.functional as NF

    return NF.block_fp8_linear(
        x,
        weight,
        weight_scale,
        block_size=(128, 128),
        act_group_size=128,
        accum_dtype=torch.float32,
        out_dtype=out_dtype,
        bias=None,
    )


def _group_topology(
    group_pg, group_rank: "int | Tensor | None"
) -> "tuple[int, int | Tensor]":
    """Resolve ``(group_size, group_rank)`` for the intra-o_group process group.

    Returns ``(1, 0)`` when there is no process group (single-core CPU unit
    tests, or a hypothetical ``tp_size <= o_groups`` deployment where the whole
    group is local and upstream's reduction is already a local sum).
    ``group_size`` is always a Python int resolved at trace time, so nothing
    here becomes a data-dependent shape. ``group_rank`` may pass through as a
    value-free int32 buffer TENSOR (LD-74): the rank then binds at RUNTIME and
    never renders as a graph literal.
    """
    if group_pg is None or not dist.is_available() or not dist.is_initialized():
        return 1, 0
    size = dist.get_world_size(group=group_pg)
    rank = dist.get_rank(group=group_pg) if group_rank is None else group_rank
    return size, rank


def mla_grouped_oproj(
    attn_out: Tensor,
    wo_a_w: Tensor,
    wo_b_w: Tensor,
    o_groups: int,
    group_pg,
    *,
    wo_a_scale: Optional[Tensor] = None,
    wo_b_scale: Optional[Tensor] = None,
    group_rank: "int | Tensor | None" = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Grouped MLA o-projection: local ``wo_a`` partial, group reduce, ``wo_b``.

    WHICH REDUCTION THIS FUNCTION PERFORMS, AND WHICH ONE THE CALLER OWNS
    ---------------------------------------------------------------------
    * THIS FUNCTION performs the **intra-o_group fp32 SUM all-reduce over
      ``group_pg``** (the 8-core subgroup, one head per core at TP=64). That
      reduction is what reconstructs the reference's per-group sum over the
      group's 8 heads inside ``einsum("bsgd,grd->bsgr", o, wo_a)``
      (``dsv4_ref/model.py:543-546``; upstream vLLM's equivalent
      ``einsum("bhr,hdr->bhd", ...)`` at ``deepseek_v4_attention.py:344-352``).
      It is accumulated in fp32 and cast back only after the reduction
      completes, matching how the reference's ``RowParallelLinear`` upcasts
      before its own all-reduce (``model.py:180-186``); summing 8 bf16 partials
      of a 1024-wide activation in bf16 loses precision the fp8 einsum keeps.
    * THE CALLER owns the **TP-wide (64-way) SUM all-reduce** that completes
      ``wo_b``'s row-parallel K reduction. This function returns only this
      core's ``wo_b`` partial, exactly as ``RowParallelLinear`` splits "matmul
      here, reduce at the call site" (``dsv4_ref/model.py:179-186``). Forgetting
      that all-reduce yields a silently wrong (1/64 of the terms) hidden delta.
      The reference reduces that partial in FP32 and casts back afterwards
      (``model.py:182-186``), so to reproduce its arithmetic the caller should
      pass ``out_dtype=torch.float32`` here, all-reduce, then cast to bf16.

    ``attn_out`` must ALREADY be inverse-RoPE'd. The reference applies
    ``apply_rotary_emb(o[..., -rd:], freqs_cis, inverse=True)`` immediately
    before the ``wo_a`` einsum (``dsv4_ref/model.py:539``; upstream fuses it with
    a block-fp8 quant into ``fused_inv_rope_fp8_quant``,
    ``deepseek_v4_attention.py:324-334``). That step needs ``positions`` and the
    cos/sin tables, which the plan-fixed signature here does not carry, so it
    stays with the attention module. The reference applies NO ``act_quant``
    round trip to ``o`` (contrast the KV latent at ``model.py:512``), so none is
    done here either.

    Args:
        attn_out: this core's local head slice of the attention output, either
            ``[T, heads_local, head_dim]`` (``[T, 1, 512]`` at TP=64) or the
            pre-flattened ``[T, heads_local * head_dim]``. Inverse-RoPE'd.
        wo_a_w: ``[o_lora_rank, heads_local * head_dim]`` (``[1024, 512]`` at
            TP=64) — this core's head-slice of its group's ``wo_a`` K axis.
            Stored ``[out, in]`` like the rest of the checkpoint.
        wo_b_w: this core's row-parallel slice of ``wo_b``,
            ``[hidden_size, (o_groups * o_lora_rank) // tp_size]``
            (``[4096, 128]`` at TP=64 — see ``config.block_fp8_linear_plan``).
            The alternative ``[hidden_size, o_lora_rank]`` group-slice layout
            (redundant across the group's cores) is also accepted and gets the
            required ``1 / group_size`` prescale; see below.
        o_groups: number of o-projection groups (8). Used to validate that the
            global ``wo_b`` K axis really is ``o_groups * o_lora_rank``.
        group_pg: the intra-o_group ``torch.distributed`` process group built
            once in the backbone ``__init__`` via ``dist.new_group(ranks)``
            (family interface contract §2). ``None`` disables the intra-group
            reduction, which is correct only when the whole group is local.
        wo_a_scale: block-FP8 weight scale for ``wo_a_w``, fp32
            ``[ceil(N/128), ceil(K/128)]``; ``None`` selects plain matmul.
            KEYWORD-ONLY EXTENSION over the plan-fixed signature.
        wo_b_scale: same, for ``wo_b_w``. KEYWORD-ONLY EXTENSION.
        group_rank: this core's rank WITHIN ``group_pg``, i.e. which
            ``o_lora_rank / group_size`` column block of its group's activation
            this core's ``wo_b`` slice consumes. Defaults to
            ``dist.get_rank(group=group_pg)``. The checkpoint convention is a
            CONTIGUOUS K slice per rank (``dsv4_ref/convert.py:59, 113-116``:
            ``wo_b`` narrowed on dim 1 at ``i * shard``), so at TP=64 rank ``r``
            owns global columns ``[r*128, (r+1)*128)`` = group ``r // 8``'s block
            ``r % 8`` — which is why ``group_rank`` must be ``tp_rank % 8`` and
            the group must be 8 consecutive ranks. That is NOT true under the
            TRN2 8x8 mesh (``functional/process_groups.py:TRN2_8x8_MESH``), so
            the loader's shard order and this argument have to agree.
            Under LD-74 the traced path passes a VALUE-FREE int32 buffer
            ``[[tp_rank % 8]]`` here instead of the Python int: the lane
            extraction below then becomes a runtime-indexed, clamped
            ``index_select`` and the render carries no per-rank slice-start
            literal, so the 8 ranks of one oproj tile share one compile key.
            An int keeps the trace-time slice path (CPU unit tests).
            KEYWORD-ONLY EXTENSION.
        out_dtype: store dtype of the returned partial. Pass
            ``torch.float32`` to let the caller's TP all-reduce accumulate in
            fp32 as ``dsv4_ref/model.py:182-186`` does. KEYWORD-ONLY EXTENSION.

    Returns:
        ``[T, hidden_size]`` — this core's ``wo_b`` partial of the hidden delta,
        pending the caller's TP all-reduce.
    """
    o_lora_rank = wo_a_w.shape[0]
    k_local_a = wo_a_w.shape[1]

    x = attn_out.reshape(attn_out.shape[0], -1)
    assert x.shape[-1] == k_local_a, (
        f"attn_out flattens to width {x.shape[-1]} but wo_a_w expects "
        f"{k_local_a} (heads_local * head_dim)"
    )
    assert wo_a_w.dim() == 2 and wo_b_w.dim() == 2, (
        "wo_a_w and wo_b_w must be 2D [out, in] tensors"
    )

    group_size, resolved_group_rank = _group_topology(group_pg, group_rank)

    # --- Stage A: per-core wo_a partial on the local head slice -------------
    # dsv4_ref/model.py:546's einsum("bsgd,grd->bsgr") contracts d = the group's
    # whole 8-head-wide latent (4096); this core owns 1/group_size of d, so its
    # matmul is a PARTIAL of that group's o_lora activation.
    z_partial = _linear(x, wo_a_w, wo_a_scale, out_dtype)

    # --- Stage A reduction (OWNED HERE): fp32 SUM over the 8-core o_group ---
    z_group = z_partial.to(torch.float32)
    if group_size > 1:
        # .contiguous() because all_reduce is in-place on the buffer it is
        # handed; a non-contiguous view would reduce the wrong strides.
        z_group = z_group.contiguous()
        dist.all_reduce(z_group, op=dist.ReduceOp.SUM, group=group_pg)

    # --- Stage B: row-parallel wo_b on this core's K slice ------------------
    k_local_b = wo_b_w.shape[1]
    # ``o_groups`` pins the GLOBAL wo_b K axis at o_groups * o_lora_rank
    # (deepseek_v4.py:1002-1008). Without tp_size in this signature, the
    # checkable invariant is that the local slice evenly divides that axis.
    assert o_groups >= 1, f"o_groups must be positive, got {o_groups}"
    assert (o_groups * o_lora_rank) % k_local_b == 0, (
        f"wo_b local K width {k_local_b} does not divide the global wo_b K axis "
        f"o_groups * o_lora_rank = {o_groups * o_lora_rank}"
    )

    if k_local_b * group_size == o_lora_rank:
        # Primary path (config.block_fp8_linear_plan's "row" entry, and the
        # reference's own RowParallelLinear + contiguous-narrow shard convention,
        # dsv4_ref/model.py:174-177 + convert.py:113-116): wo_b's K axis
        # (o_groups * o_lora_rank) is split 64 ways, and core `j` of group `g`
        # owns global columns [g*o_lora_rank + j*k_local_b, ...+k_local_b), i.e.
        # exactly the j-th block of ITS OWN group's activation. So no redundancy
        # and no prescale: the caller's 64-way all-reduce sums each distinct K
        # block exactly once.
        if isinstance(resolved_group_rank, Tensor):
            # LADDER-DECISION LD-74 (E3, assessment §16.3; plan §18.2 item 3):
            # the per-rank 128-wide lane extraction on the ``z_group`` stream,
            # runtime-bound. ``resolved_group_rank`` is the value-free int32
            # ``[[tp_rank % 8]]`` buffer, so the index vector is built from a
            # ``get_attr`` plus STATIC arange arguments — no per-rank
            # slice-start literal (``r*128`` ∈ {0,…,896}) survives in the
            # render, and the 8 ranks of an oproj tile trace byte-identical
            # graphs. The index feeds a NEW indirect consumer (a vector-DGE
            # gather, the aws-neuron-sdk#1335 class), so it lands behind the
            # LD-73 sanitize idiom FROM BIRTH (R-A): clamped into
            # ``[0, z_group.size(1)-1]``. The collapse-to-zero half of the
            # idiom is N/A here, WITH REASON: the index set is TOTAL by
            # construction (``lane * k_local_b + [0, k_local_b)`` with
            # ``lane ∈ [0, group_size)`` — no sentinel/padding class exists on
            # this path), so the clamp is defense-in-depth against #1335, not
            # a semantic mask. Numerics: ``index_select``-vs-slice bitwise
            # equality is proven exhaustively over all 8 lanes (plan §18.3
            # item 3(ii)); clamp idempotence on every in-range index is item
            # 3(iii).
            lane_index = resolved_group_rank.reshape(()) * k_local_b + torch.arange(
                0, k_local_b, dtype=torch.int32, device=z_group.device
            )
            lane_index = torch.clamp(lane_index, 0, z_group.size(1) - 1)
            z_in = torch.index_select(z_group, 1, lane_index)
        else:
            start = resolved_group_rank * k_local_b
            z_in = z_group[:, start : start + k_local_b]
        scale = 1.0
    elif k_local_b == o_lora_rank:
        # Fallback layout (dataflow-shapes.md §B step 14 / GAPS-5): every core
        # of a group holds the group's FULL o_lora_rank-wide wo_b slice, so the
        # group's contribution would be counted group_size times by the caller's
        # 64-way all-reduce. Pre-divide to compensate.
        z_in = z_group
        scale = 1.0 / float(group_size)
    else:
        raise ValueError(
            "wo_b_w K width must be either o_lora_rank // group_size "
            f"({o_lora_rank // group_size if group_size else o_lora_rank}) for "
            f"the row-parallel layout or o_lora_rank ({o_lora_rank}) for the "
            f"redundant group-slice layout, got {k_local_b}"
        )

    if scale != 1.0:
        z_in = z_in * scale
    z_in = z_in.to(out_dtype)

    return _linear(z_in, wo_b_w, wo_b_scale, out_dtype)
