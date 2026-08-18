# SPDX-License-Identifier: Apache-2.0
"""Sliding-window attention with a per-head attention sink (torch composition).

WHY this is real torch math and not a kernel wrapper: every nkilib kernel the
DSv4-Flash port plan named for this row (``swa_fused_cte`` in particular) is
absent from the installed neuron wheel, so this ladder row takes its recorded
same-row torch-composition rung. The code therefore has to stay inside the
traceable static-shape subset the Neuron tracer accepts: python loops over
statically known counts only, no ``.item()``, no ``.tolist()``, no
``nonzero()``, no boolean-mask *indexing*, no data-dependent shapes and no
python ``if`` on tensor values. Masks are additive float masks built from
``arange`` comparisons and ``torch.where``.

PRIMARY reference: DeepSeek's own implementation shipped with the pinned
checkpoint revision (``dsv4_ref/model.py``, ``dsv4_ref/kernel.py``). It outranks
the derived spec reports wherever they disagree.
  * sink position in the softmax — ``dsv4_ref/kernel.py:345-348``:
    ``sum_exp[i] += exp(attn_sink[i] - scores_max[i])`` then
    ``acc_o[i, j] /= sum_exp[i]``. The sink is one extra logit per query head
    that enters the DENOMINATOR ONLY, added after the key loop, using the
    running row max taken over the REAL keys only (``kernel.py:332``), and it
    contributes nothing to the numerator (it has no value vector).
  * fp32 softmax statistics, bf16 store — ``dsv4_ref/kernel.py:308-314``
    (``acc_s``/``acc_o``/``scores_max``/``sum_exp`` all FP32) with the output
    tensor bf16 (``kernel.py:297``, ``sparse_attn`` allocating
    ``torch.empty_like(q)``).
  * invalid slots are index ``-1`` and are masked with ``-inf`` instead of
    being dropped — ``dsv4_ref/kernel.py:323-327``.
  * causal band of width ``window`` — ``dsv4_ref/model.py:261-271``
    (``get_window_topk_idxs``): for a fresh sequence the admitted key indices
    are ``clamp(pos - window + 1, min=0) + arange(window)`` with entries
    ``> pos`` replaced by ``-1``; in decode the same window is a RING buffer of
    length ``window`` written at ``start_pos % window``
    (``dsv4_ref/model.py:535``).
  * ``window = args.window_size`` (128 for DSv4-Flash) —
    ``dsv4_ref/model.py:458``.

Corroborating upstream-vLLM references (same math, see
``spec-upstream-dsv4-attention.md``): ``cache_utils.py:535-536``
(``swa_len = min(pos + 1, WINDOW_SIZE)``), ``rocm_aiter_mla_sparse.py:935-950``
and ``:1070-1082`` (the sink folded into the softmax denominator),
``rocm_aiter_mla_sparse.py:929-933``/``:1062-1071`` (fp32 QK matmul, bf16 at
the end), and ``models/deepseek_v4.py:965-971`` (padded heads carry a ``-inf``
sink, i.e. no sink effect).

``_sink_softmax_attend``, ``_mask_from_allowed`` and ``dequant_group_scales``
below are shared with ``mla_sparse_attention.py`` and ``mla_decode.py`` on
purpose: the sink numerics must have exactly ONE definition across the three
DSv4 attention paths, otherwise prefill and decode drift apart.
"""

from typing import Optional

import torch
from torch import Tensor

# Additive mask value. A *finite* large negative number rather than -inf: the
# row-max subtraction below computes ``scores - max`` and ``sink - max``, and a
# true -inf row max would produce NaN (-inf - -inf). With a finite floor every
# intermediate stays a number, and fully masked rows are zeroed explicitly by
# the ``keep`` factor instead (upstream's ``lonely_q_mask``,
# rocm_aiter_mla_sparse.py:949-950).
MASK_NEG: float = -1.0e30


def _mask_from_allowed(allowed: Tensor, *, dtype: torch.dtype = torch.float32) -> Tensor:
    """Turn a boolean *allowed* map into an additive float mask.

    ``torch.where`` on two same-shape constant tensors keeps this traceable and
    avoids boolean-mask indexing (which the Neuron tracer rejects).
    """
    zero = torch.zeros((), dtype=dtype, device=allowed.device)
    neg = torch.full((), MASK_NEG, dtype=dtype, device=allowed.device)
    return torch.where(allowed, zero, neg)


def _sliding_window_mask(
    q_pos: Tensor,
    k_pos: Tensor,
    window: int,
    *,
    causal: bool = True,
) -> Tensor:
    """Additive ``[Tq, S]`` mask for a causal band of width ``window``.

    ``q_pos``: ``[Tq, 1]`` absolute query positions, ``k_pos``: ``[1, S]``
    absolute key positions (both int64 tensors). A key is admitted when
    ``0 <= q_pos - k_pos < window`` — the same band the reference builds as an
    index list in ``dsv4_ref/model.py:268-270``
    (``clamp(pos - window + 1, 0) + arange(window)``, entries ``> pos`` set to
    ``-1``), and upstream's ``swa_len = min(pos + 1, window)``
    (``cache_utils.py:535-536``), expressed as a per-position comparison so the
    shape stays static.
    """
    delta = q_pos - k_pos
    if causal:
        allowed = (delta >= 0) & (delta < window)
    else:
        allowed = (delta > -window) & (delta < window)
    return _mask_from_allowed(allowed)


def _sink_softmax_attend(
    scores: Tensor,
    values: Tensor,
    sink: Optional[Tensor],
    keep: Tensor,
) -> Tensor:
    """Sink-augmented softmax + value matmul, fp32 throughout.

    Args:
        scores: ``[G, Tq, S]`` fp32 logits, already scaled and masked.
        values: ``[G, S, Dv]`` fp32 values.
        sink: fp32 per-query-head sink logit, any shape broadcastable to
            ``[G, Tq, 1]`` (``[G, 1, 1]`` when the head axis is ``G``,
            ``[1, Tq, 1]`` when the head axis is ``Tq``), or None.
        keep: ``[G or 1, Tq, 1]`` fp32 0/1 factor zeroing queries that have no
            valid key at all (upstream ``lonely_q_mask``).

    The sink is an EXTRA LOGIT WITH NO VALUE: it is folded into ``denom`` and
    never into the ``probs @ values`` numerator. This reproduces the reference
    arithmetic of ``dsv4_ref/kernel.py:332-348`` exactly, including the detail
    that the row max is taken over the REAL keys only and is NOT folded with
    the sink logit (``kernel.py:332`` computes ``scores_max`` inside the key
    loop; ``kernel.py:346`` then adds ``exp(attn_sink - scores_max)``).
    Equivalent upstream-vLLM forms:
    ``lse_for_o = logsumexp([orig_lse, attn_sink[head]])``
    (``rocm_aiter_mla_sparse.py:935-947``) and ``out *= sigmoid(lse - sink)``
    (``:1076-1079``).
    """
    # Row max over the REAL keys only (reference kernel.py:332).
    row_max = scores.max(dim=-1, keepdim=True).values  # [G, Tq, 1]

    probs = torch.exp(scores - row_max)  # [G, Tq, S] fp32
    denom = probs.sum(dim=-1, keepdim=True)  # [G, Tq, 1] fp32
    if sink is not None:
        # <-- MODEL-SPECIFIC: DSv4 attention sink (reference kernel.py:345-346).
        # One extra logit per query head, DENOMINATOR ONLY, no value vector.
        denom = denom + torch.exp(sink - row_max)

    out = torch.bmm(probs, values) / denom  # [G, Tq, Dv] fp32
    return out * keep


def dequant_group_scales(
    latent: Tensor,
    scales: Tensor,
    *,
    group_size: int = 64,
    num_groups: int = 7,
) -> Tensor:
    """Apply the group-64 UE8M0 dequant scales to the NoPE half of a latent.

    ``latent``: ``[..., nope_dim]`` fp32, ``scales``: ``[..., >=num_groups]``
    fp32. Mirrors the write-side quantizer's grouping
    (``fused_compress_quant_cache.py:156-186``: ``QUANT_BLOCK=64``,
    ``N_NOPE_BLOCKS = 448 // 64 = 7``) and the port's scale-cache row
    (contract §4: 7 of 32 fp32 slots used).
    """
    lead = latent.shape[:-1]
    grouped = latent.reshape(*lead, num_groups, group_size)
    s = scales[..., :num_groups].reshape(*lead, num_groups, 1).to(grouped.dtype)
    return (grouped * s).reshape(*lead, num_groups * group_size)


def swa_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    window: int,
    sink: Optional[Tensor],
    scale: float,
    *,
    positions: Optional[Tensor] = None,
    kv_positions: Optional[Tensor] = None,
    q_seq_ids: Optional[Tensor] = None,
    kv_seq_ids: Optional[Tensor] = None,
    kv_valid: Optional[Tensor] = None,
    causal: bool = True,
    softmax_dtype: torch.dtype = torch.float32,
    out_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Plain sliding-window attention with a per-head attention sink.

    Used by the DSv4-Flash SWA-only layers (``compress_ratio <= 1``: layers 0,
    1 and the MTP draft layer — ``dataflow-shapes.md`` §C) and, through
    ``mla_sparse_attention``/``mla_decode_attention``, as the SWA leg of the
    compressed layers. Neither the reference implementation nor upstream vLLM
    has a separate module for it: the reference feeds the window as the first
    slice of one concatenated index list into ``sparse_attn``
    (``dsv4_ref/model.py:513-538``) and upstream feeds it as one of the two KV
    sources of the single fused FlashMLA call
    (``deepseek_v4_attention.py:897-913``). This function is the isolated torch
    statement of that leg, for the layers that have ONLY that leg.

    Args:
        q: ``[T, H, D]`` queries (bf16 or fp32). ``T`` is the flat token count;
            a packed multi-sequence batch is supported by passing ``positions``
            and ``q_seq_ids``/``kv_seq_ids``.
        k: ``[S, Hk, D]`` keys, already gathered into a contiguous window.
        v: ``[S, Hk, Dv]`` values. For the DSv4 MLA layers ``v`` IS ``k`` — the
            single 512-wide latent is both key and value (absorbed form, no V
            projection: ``dsv4_ref/model.py:480`` allocates ONE
            ``kv_cache[..., head_dim]`` and ``:533``/``:538`` pass it as the
            only KV tensor). For a non-MLA sliding-window layer they may be
            distinct tensors.
        window: causal band width (128 for DSv4-Flash,
            ``args.window_size`` — ``dsv4_ref/model.py:458``).
        sink: ``[H]`` or ``[H, 1]`` fp32 per-query-head sink logit, or None.
            Reference shape is ``[n_local_heads]`` fp32
            (``dsv4_ref/model.py:462``). Padded heads carry ``-inf`` upstream
            (``models/deepseek_v4.py:965-971``), which this code handles:
            ``exp(-inf - row_max) == 0``.
        scale: softmax scale applied to the QK product.

    Keyword-only extensions (all defaulted, so the plan-fixed positional call
    form ``swa_attention(q, k, v, window, sink, scale)`` stays valid):
        positions: ``[T]`` absolute query positions. Default ``arange(T)``.
        kv_positions: ``[S]`` absolute key positions. Default ``arange(S)``.
        q_seq_ids / kv_seq_ids: ``[T]`` / ``[S]`` sequence ids; when both are
            given, cross-sequence attention is masked out (needed for packed
            prefill, where positions alone do not separate sequences).
        kv_valid: ``[S]`` or ``[B, S]`` 0/1 (or bool) map of populated key
            slots; padded slots are masked out.
        causal: keep the band one-sided (default, and what DSv4 needs).
        softmax_dtype: accumulation dtype for the softmax statistics (fp32).
        out_dtype: dtype of the returned tensor (bf16).

    Returns:
        ``[T, H, Dv]`` attention output in ``out_dtype``.
    """
    assert q.dim() == 3, f"q must be [T, H, D], got {tuple(q.shape)}"
    assert k.dim() == 3, f"k must be [S, Hk, D], got {tuple(k.shape)}"
    assert v.dim() == 3, f"v must be [S, Hk, Dv], got {tuple(v.shape)}"

    num_tokens, num_heads, head_dim = q.shape
    num_kv, num_kv_heads, value_dim = v.shape
    assert k.shape[0] == num_kv, "k and v must cover the same window"
    assert k.shape[2] == head_dim, "q and k must share the score dim"
    assert num_heads % num_kv_heads == 0, "H must be a multiple of Hk (GQA)"
    device = q.device

    # ── Positions and the additive band mask ─────────────────────────────
    if positions is None:
        q_pos = torch.arange(num_tokens, device=device, dtype=torch.int64)
    else:
        q_pos = positions.reshape(-1).to(torch.int64)
    if kv_positions is None:
        k_pos = torch.arange(num_kv, device=device, dtype=torch.int64)
    else:
        k_pos = kv_positions.reshape(-1).to(torch.int64)

    mask = _sliding_window_mask(
        q_pos.unsqueeze(1), k_pos.unsqueeze(0), window, causal=causal
    )  # [T, S] fp32

    if q_seq_ids is not None and kv_seq_ids is not None:
        same_seq = q_seq_ids.reshape(-1, 1).to(torch.int64) == kv_seq_ids.reshape(
            1, -1
        ).to(torch.int64)
        mask = mask + _mask_from_allowed(same_seq)

    if kv_valid is not None:
        valid = kv_valid.reshape(1, -1) != 0
        mask = mask + _mask_from_allowed(valid)

    # ── Scores in fp32; GQA expansion of K/V ─────────────────────────────
    heads_per_kv = num_heads // num_kv_heads
    q_g = q.permute(1, 0, 2).to(softmax_dtype)  # [H, T, D]
    k_g = k.permute(1, 2, 0).to(softmax_dtype)  # [Hk, D, S]
    v_g = v.permute(1, 0, 2).to(softmax_dtype)  # [Hk, S, Dv]
    if heads_per_kv > 1:
        k_g = k_g.repeat_interleave(heads_per_kv, dim=0)
        v_g = v_g.repeat_interleave(heads_per_kv, dim=0)

    scores = torch.bmm(q_g, k_g) * scale  # [H, T, S] fp32
    scores = scores + mask.unsqueeze(0).to(softmax_dtype)

    # ── Sink-augmented softmax ───────────────────────────────────────────
    sink_g = None
    if sink is not None:
        sink_g = sink.reshape(num_heads, 1, 1).to(softmax_dtype)

    # Queries with no admitted key at all are forced to 0, matching upstream's
    # lonely_q_mask (rocm_aiter_mla_sparse.py:949-950) — the sink alone must not
    # drive an output. The reference kernel has no such guard (its every query
    # sees at least its own window slot); this guard only ever fires on padded
    # query rows, which the reference never produces.
    keep = (mask > MASK_NEG / 2).any(dim=-1, keepdim=True)  # [T, 1] bool
    keep_f = keep.to(softmax_dtype).unsqueeze(0)  # [1, T, 1]

    out = _sink_softmax_attend(scores, v_g, sink_g, keep_f)  # [H, T, Dv] fp32
    return out.permute(1, 0, 2).to(out_dtype)  # [T, H, Dv]
