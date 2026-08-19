# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 MLA input projection chain (down-proj + q/kv RMSNorm + RoPE).

WHY this is a torch composition and not an NKI wrapper: the nkilib kernel this
row cites by name (``mla_qkv_cte``) is ABSENT from the installed neuron wheel
(family interface contract §0), so there is no kernel to gate on and therefore
no ``_can_use_*`` dispatch pair here — unlike ``qkv.py`` / ``o_proj.py``, which
do have a kernel to fall back from. The whole body is real math and must stay
inside the traceable static-shape torch subset (no ``.item()``, no
``.tolist()``, no ``nonzero()``, no boolean-mask indexing, no data-dependent
shapes); Python loops over statically known counts would be legal but are not
needed.

PRIMARY source of truth for the MATH is DeepSeek's own reference implementation
shipped in the pinned checkpoint repo (``dsv4_ref/model.py``,
``dsv4_ref/kernel.py``, ``dsv4_ref/convert.py``). It outranks the derived spec
reports wherever they disagree. Every arithmetic step below is cited to it:

  * ``dsv4_ref/model.py:502`` — ``qr = q = q_norm(wq_a(x))``: q down-projection
    then the q-latent RMSNorm.
  * ``dsv4_ref/model.py:503`` — ``q = wq_b(q).unflatten(-1, (n_local_heads,
    head_dim))``.
  * ``dsv4_ref/model.py:504`` — ``q *= rsqrt(q.square().mean(-1, keepdim=True)
    + eps)``: the WEIGHTLESS per-head Q RMSNorm over the full ``head_dim``,
    before RoPE.
  * ``dsv4_ref/model.py:505`` / ``:510`` — ``apply_rotary_emb(q[..., -rd:])``
    and ``apply_rotary_emb(kv[..., -rd:])``: one identical rotation on the LAST
    ``rope_head_dim`` dims of q and of the KV latent.
  * ``dsv4_ref/model.py:508-509`` — ``kv = kv_norm(wkv(x))``.
  * ``dsv4_ref/model.py:512`` — ``act_quant(kv[..., :-rd], 64, "ue8m0",
    float8_e8m0fnu, inplace=True)``: the group-64 fp8 quant/dequant round trip
    on the KV latent's NoPE dims (QAT simulation; the RoPE dims stay bf16).
  * ``dsv4_ref/model.py:197-202`` — the RMSNorm formula.
  * ``dsv4_ref/model.py:238-250`` — ``apply_rotary_emb``: complex multiply over
    ``unflatten(-1, (-1, 2))``, i.e. ADJACENT-pair (GPT-J) rotation.
  * ``dsv4_ref/model.py:114-126, 145, 150`` — weights are ``[out, in]`` and the
    unquantized path is ``F.linear(x, weight)``.
  * ``dsv4_ref/kernel.py:36-37, 41-101, 105-125`` — the block-fp8 act-quant
    scale semantics.

Corroborating upstream-vLLM refs (tag ``v0.21.0``), kept because they pin the
port's INTERFACE (sharding, fusion, cache layout) rather than the math:
  * ``vllm/model_executor/models/deepseek_v4.py:973-991`` — module shapes:
    ``fused_wqa_wkv: hidden_size -> [q_lora_rank, head_dim]`` (built with
    ``disable_tp=True``, i.e. replicated), ``q_norm = RMSNorm(q_lora_rank)``,
    ``wq_b: q_lora_rank -> n_heads * head_dim``,
    ``kv_norm = RMSNorm(head_dim)``.
  * ``vllm/model_executor/layers/deepseek_v4_attention.py:429-436`` — the
    ``split([q_lora_rank, head_dim])`` then the fused q/kv RMSNorm.
  * ``vllm/v1/attention/ops/deepseek_v4_ops/fused_qk_rmsnorm.py:44-54`` — the
    RMSNorm formula, fp32 throughout with a single cast at store.
  * ``deepseek_v4_attention.py:451-455`` and ``:544-557`` — ``wq_b`` then the
    fused weightless per-head Q RMSNorm over the full ``head_dim`` followed by
    GPT-J RoPE on the last ``rope_head_dim`` dims, with the KV latent taking
    the *identical* RoPE.
  * ``dataflow-shapes.md`` §A row "wq_b + q-head-norm + RoPE + kv-insert" and
    §B steps 6, 7 and 20.

The paged KV-cache insert that upstream fuses into the same CUDA kernel is
deliberately NOT part of this op: it needs ``slot_mapping``/``block_size`` and
is owned by the attention module, which writes the returned ``latent_kv``.
"""

from typing import Optional

import torch
from torch import Tensor

from vllm_neuron.utils.dtype_utils import FP8_CLAMP_MAX


def _rms_norm(
    x: Tensor,
    weight: Optional[Tensor],
    eps: float,
    out_dtype: torch.dtype,
) -> Tensor:
    """``y = x * rsqrt(mean(x^2, -1) + eps) * w``, fp32 with one cast at store.

    Reproduces ``dsv4_ref/model.py:197-202`` term for term::

        x = x.float(); var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + eps); return (weight * x).to(dtype)

    i.e. ``x``, the reciprocal RMS and ``w`` are all in fp32 with a single cast
    on store (``fused_qk_rmsnorm.py:44-54`` agrees). The fp32 accumulation is
    load-bearing — doing the reduction in bf16 loses enough precision on a
    1024-wide latent to move logits.

    ``weight=None`` is the *weightless* variant used for the per-head Q norm,
    which the reference writes inline as
    ``q *= rsqrt(q.square().mean(-1, keepdim=True) + eps)``
    (``dsv4_ref/model.py:504``; upstream vLLM builds it as
    ``RMSNorm(head_dim, eps, has_weight=False)``,
    ``deepseek_v4_attention.py:219``).
    """
    x_f32 = x.to(torch.float32)
    variance = x_f32.pow(2).mean(-1, keepdim=True)
    y = x_f32 * torch.rsqrt(variance + eps)
    if weight is not None:
        y = y * weight.to(torch.float32)
    return y.to(out_dtype)


def _apply_gptj_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """GPT-J (interleaved) RoPE on the whole trailing dim of ``x``.

    This is ``dsv4_ref/model.py:238-250`` written without complex tensors::

        x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
        x = torch.view_as_real(x * freqs_cis).flatten(-2)

    ``unflatten(-1, (-1, 2))`` means the rotated pairs are ADJACENT dims
    ``(0,1), (2,3), ...`` — NOT the half-split ``(i, i + d/2)`` pairing that
    ``qkv.py``'s torch reference uses. Getting this wrong is silent: both forms
    produce plausible-looking numbers. Expanding
    ``(x_even + i*x_odd) * (cos + i*sin)`` gives the two lanes below, i.e.
    ``(cos, -sin)`` on the even lane and ``(cos, +sin)`` on the odd lane.

    Complex tensors are avoided deliberately: ``view_as_complex`` /
    ``view_as_real`` are outside the traceable static-shape subset this port
    must stay in, and the reference's in-place ``y.copy_(x)`` is replaced by a
    functional return for the same reason.

    The same pairing shows up in the upstream inverse-RoPE kernel's read pattern
    ``x_partner = load(base + (offsets ^ 1))`` with
    ``cs_idx = rope_local >> 1`` (``fused_inv_rope_fp8_quant.py:84-94``).

    Args:
        x: ``[..., rope_dim]`` with ``rope_dim`` even; the token axis is dim 0.
        cos: ``[T, rope_dim // 2]`` fp32, already gathered at the positions.
        sin: ``[T, rope_dim // 2]`` fp32, already gathered at the positions.

    Returns:
        Tensor shaped like ``x``, dtype ``x.dtype``.
    """
    rope_dim = x.shape[-1]
    half = rope_dim // 2

    # Broadcast the per-token cos/sin over any head axes between the token axis
    # and the rotated-pair axis: [T, half] -> [T, 1, ..., 1, half].
    extra_dims = x.dim() - 2
    cos_b = cos.reshape(cos.shape[0], *([1] * extra_dims), half)
    sin_b = sin.reshape(sin.shape[0], *([1] * extra_dims), half)

    pairs = x.to(torch.float32).reshape(*x.shape[:-1], half, 2)
    x_even = pairs[..., 0]
    x_odd = pairs[..., 1]

    out_even = x_even * cos_b - x_odd * sin_b
    out_odd = x_odd * cos_b + x_even * sin_b

    rotated = torch.stack((out_even, out_odd), dim=-1)
    return rotated.reshape(x.shape).to(x.dtype)


def _gather_rope_tables(
    rope_cos: Tensor,
    rope_sin: Tensor,
    positions: Tensor,
    rope_head_dim: int,
    pregathered: bool = False,
) -> tuple[Tensor, Tensor]:
    """Gather the per-token cos/sin rows at ``positions``, in fp32.

    ``index_select`` with an int64 index tensor is used (never advanced/boolean
    indexing) so the graph keeps a static output shape ``[T, rope_head_dim/2]``.

    Both the half-width table ``[max_pos, rope_head_dim // 2]`` (one entry per
    rotated pair, the natural layout for GPT-J RoPE) and the full-width table
    ``[max_pos, rope_head_dim]`` are accepted; for the latter the leading half
    is taken, matching upstream's single ``cos_sin_cache`` whose cos block is
    ``[0:HALF_ROPE]`` (``fused_inv_rope_fp8_quant.py:88-89``).

    ``pregathered=True`` says the caller has ALREADY produced one row per
    TOKEN, so the position lookup must be skipped. The two conventions are
    indistinguishable from the tensors alone -- both are 2-D and
    ``[T, half]`` is a legal table shape -- so the caller must say which it
    holds, and the default stays the table convention this helper was written
    for. See ``rope_tables_pregathered`` in :func:`mla_qkv`.
    """
    half = rope_head_dim // 2

    if pregathered:
        cos = rope_cos.to(torch.float32)
        sin = rope_sin.to(torch.float32)
        assert cos.shape[0] == positions.reshape(-1).shape[0], (
            "rope_tables_pregathered=True means one cos/sin row per token, so "
            f"rope_cos.shape[0]={cos.shape[0]} must equal the token count "
            f"{positions.reshape(-1).shape[0]}"
        )
    else:
        index = positions.reshape(-1).to(torch.long)
        cos = rope_cos.index_select(0, index).to(torch.float32)
        sin = rope_sin.index_select(0, index).to(torch.float32)

    assert cos.shape[-1] in (half, rope_head_dim), (
        f"rope_cos last dim must be {half} (per-pair) or {rope_head_dim} "
        f"(full width), got {cos.shape[-1]}"
    )
    assert sin.shape[-1] == cos.shape[-1], (
        "rope_cos and rope_sin must have the same trailing width, got "
        f"{cos.shape[-1]} and {sin.shape[-1]}"
    )
    if cos.shape[-1] == rope_head_dim:
        cos = cos[:, :half]
        sin = sin[:, :half]
    return cos, sin


def _round_scale_pow2(amax: Tensor, fp8_max: float = FP8_CLAMP_MAX) -> Tensor:
    """``s = 2 ** ceil(log2(amax / fp8_max))``, in a form that LOWERS to HLO.

    ``fp8_max`` DEFAULTS TO THE PLATFORM-RESOLVED CEILING (240.0 on trn2, 448.0
    on trn3), not the reference's literal 448 -- finding F-7, lead-granted route
    (i); see :func:`_fp8_group_quant_dequant` for the whole argument. The divisor
    here and the clamp in the caller MUST be the same ceiling: a scale computed
    against 448 with a clamp at 240 would saturate roughly half of every group's
    top values, which is a real accuracy loss rather than a re-encoding. The
    restatement below leaves both of those invariants exactly as they were: same
    signature, same default, same divisor, same coupling to the caller's clamp.

    ``dsv4_ref/kernel.py:36-37`` computes the ue8m0 (power-of-two) act-quant
    scale as ``fast_pow2(fast_log2_ceil(amax * (1/fp8_max)))``, and
    ``kernel.py:22-33`` implements those two with bit arithmetic::

        exp = (bits >> 23) & 0xFF; man = bits & ((1 << 23) - 1)
        ceil_log2 = exp - 127 + (1 if man != 0 else 0)
        pow2      = reinterpret_f32((ceil_log2 + 127) << 23)

    THAT BIT FORM REPLACED, AND THE MEASUREMENT IS WHY. It does not lower through
    the production ``vllm_neuron/compile/hlo.py:convert_fx_to_hlo``::

        rshift = torch.ops.aten.__rshift__.Scalar(detach, 23)
        RuntimeError: Expected XLA tensor. Got: XLAIntType

    ``convert_fx_to_hlo`` builds XLA placeholders and RE-EXECUTES the FX graph on
    them, so an op with no torch_xla lowering falls back to CPU and the next op
    rejects the result. Eager execution never lowers, which is why eager
    validation passed this for two iterations.

    THIS COPY IS THE HARDER OF THE TWO THIS BUG HAS, AND NOTHING GATES IT. This
    module has no ``_can_use_*`` dispatch pair at all (module docstring), so there
    is no gate to turn off and no torch/NKI fork to take. The only guard on the
    call path is ``kv_nope_fp8_qat``, which defaults ``True`` in :func:`mla_qkv`
    and is passed explicitly ``True`` at both production call sites
    (``model/deepseek_v4/attention.py`` and ``model/deepseek_v4/dspark_model.py``),
    so this helper is traced UNCONDITIONALLY on the production path. Repairing
    only ``functional/block_fp8_linear.py``'s copy would have left the compile
    blocked here.

    Measured on this venue, against a 2-input matmul control and a 1-input fp32
    elementwise control that both reach a NEFF, every crash retried once and
    deterministic:

        COMPILE : mul add sub div floor ceil log2 minimum maximum pow exp clamp
        SEGFAULT: the fp32->int32 bitcast, and ``torch.exp2``
        ABORT   : ``torch.frexp``, ``torch.ldexp``
        FAIL    : ``>>``, ``bitwise_right_shift``, int32 ``multiply``

    So every bit-trick form, every ``frexp`` form and every ``exp2`` form is
    unavailable here, and the restatement has to live inside the first line.

    WHY THIS IS STILL BIT-EXACT AT EXACT POWERS OF TWO, which is the whole point
    of the bit form and what this docstring used to warn about. ``exp2(ceil(log2
    v))`` is the obvious replacement and it is WRONG: measured 73 of 81 mismatches
    at nextafter-above-a-power-of-two, each off by exactly a factor of 0.5 -- one
    whole exponent of quantization error, the warning confirmed by measurement
    rather than dropped. This form instead uses ``log2`` only for a SEED and then
    removes its error entirely:

      * ``pow(2.0, floor(log2(v)))`` is an EXACT power of two, and ``floor(log2)``
        is within 1 of the true ``floor(log2 v)``.
      * three fixups then run, each EXACT: halve once if ``s > v``; double once if
        ``2s <= v``; double once unless ``s == v``. One step each is provably
        enough because the seed is within one binade -- writing ``v`` in
        ``[2**t, 2**(t+1))`` and the seed as ``2**k`` with ``k`` in
        ``{t-1, t, t+1}``, each of the three cases reaches ``s <= v < 2s`` after
        at most one correction.
      * halving and doubling a power of two is exact in fp32 inside the normal
        range, and the comparisons are exact, so ``log2``'s error cannot reach
        the result.

    Measured BIT-EXACT against the bit trick over the declared domain -- 122
    exact-power-of-two positions included, three probes in every admitted binade
    from ``2**-30`` to ``2**30``, both nextafter neighbours of every power of two,
    4096 random values, ``_AMAX_FLOOR`` and ``bfloat16_max``. That an INEXACT seed
    is NOT rescued by the fixups was measured too: ``exp(k*ln2)`` seeds fail the
    same clause at 225 and 524 positions, which is what makes the exactness of
    the ``pow`` seed load-bearing rather than incidental.

    DECLARED DEVIATION, outside the declared domain. For SUBNORMAL ``v`` the bit
    trick floors at ``2**-126`` (its exponent field is 0, so ``ceil_log2``
    saturates) while this form returns the true ceiling; 3 of 4 subnormal probes
    differ. ``amax`` here is an absolute maximum clamped at ``amax_floor`` by
    :func:`_fp8_group_quant_dequant`, so ``v >= 1e-4/fp8_max = 4.166667e-07`` and
    subnormal ``v`` cannot occur on any path this op has. Recorded rather than
    absorbed; no threshold moved.

    THE REINTERPRETATION IDIOM IS UNAFFECTED WHERE IT IS A LOADER CONCERN
    (family interface contract §1, discrepancy 1: decoding the checkpoint's
    ``float8_e8m0fnu`` scales). That runs at weight load, off the traced graph,
    where the bitcast is legal. Only this on-graph use had to move.

    Every selector is ARITHMETIC on integer-valued or ratio quantities -- no
    boolean tensor, no index, no bitcast, no int32 -- so the graph stays
    static-shape and inside the lowerable subset. 35 fx nodes, reaching a NEFF.

    KEPT IDENTICAL to ``functional/block_fp8_linear.py``'s ``_pow2_ceil_scale``,
    which is a local copy of this helper; the LD-11 numerics declaration grades
    that equality as a clause.
    """
    v = (amax.to(torch.float32) * (1.0 / fp8_max)).contiguous()
    zero = torch.zeros_like(v)
    one = torch.ones_like(v)

    # Exact power-of-two seed, within one binade of the answer.
    s = torch.pow(2.0, torch.floor(torch.log2(v)))

    # Fixup 1 -- force s <= v. sel == 1 iff v/s < 1.
    r = v / s
    sel = torch.minimum(torch.maximum(torch.ceil(one - r), zero), one)
    s = s * (0.5 * sel + (one - sel))

    # Fixup 2 -- force 2s > v, so that s <= v < 2s. sel == 1 iff v/s >= 2.
    r = v / s
    sel = torch.minimum(torch.maximum(torch.floor(r * 0.5), zero), one)
    s = s * (2.0 * sel + (one - sel))

    # Fixup 3 -- the smallest power of two >= v: double unless s == v exactly.
    r = v / s
    sel = torch.minimum(torch.maximum(torch.ceil(r - one), zero), one)
    return s * (2.0 * sel + (one - sel))


def _fp8_group_quant_dequant(
    x: Tensor,
    group_size: int,
    fp8_dtype: torch.dtype,
    fp8_max: float = FP8_CLAMP_MAX,
    amax_floor: float = 1e-4,
) -> Tensor:
    """Group-wise fp8 quant/dequant round trip along the last dim.

    This is ``act_quant(..., inplace=True)``: the reference's QAT simulation,
    which quantizes to fp8 and immediately dequantizes back to bf16 so that
    inference sees the values the model was trained to see
    (``dsv4_ref/kernel.py:84-91`` for the inplace branch,
    ``kernel.py:105-125`` for the wrapper). Per group::

        amax = max(|x|.amax(-1), 1e-4)
        s    = 2 ** ceil(log2(amax / fp8_max))            # scale_fmt="ue8m0"
        y    = f32(fp8(clamp(x / s, -fp8_max, fp8_max))) * s

    The ``1e-4`` amax floor is the reference's literal
    (``kernel.py:47-49, 79``). Skipping this step does not make the result
    "cleaner" — it makes it differ from the trained numerics.

    ``fp8_max`` IS **NOT** THE REFERENCE'S 448 ON THIS VENUE (finding F-7,
    port-assessment.md §2.8; lead-granted route (i)). It defaults to the fork's
    platform-resolved ``FP8_CLAMP_MAX`` — 240.0 on trn2, 448.0 on trn3.

    Why the reference's number is wrong here. The reference runs on OCP
    ``float8_e4m3fn``, whose exponent field 15 is FINITE (256..448). trn2 admits
    only LEGACY ``nl.float8_e4m3``, whose field 15 is reserved for inf/NaN and
    whose amax is 240 (``nki/nki_dtype.py:50-52``). With a ue8m0 (power-of-two)
    scale against 448 the group's largest code lands in ``(224, 448]``, so a
    field-15 byte appears as soon as that maximum reaches 256 — the COMMON case,
    not a tail case. Against 240 the largest code lands in ``[120, 240]`` and no
    byte can carry field 15.

    This is a real hazard even though the round trip is transient and stays in
    torch: ``compile/backend.py:690-714`` injects
    ``--experimental-unsafe-fp8e4m3fn-as-fp8e4m3`` unconditionally on trn2, so
    the fp8 convert inside the COMPILED graph is legacy-reinterpreted too, and
    the non-finite code is multiplied straight back into the DEQUANTIZED bf16
    activation below — the poison escapes the round trip.

    Accuracy cost is nil to first order: ``448/240 = 1.867 < 2``, so the
    power-of-two scale either stays put or exactly doubles, and when it doubles
    every grid value halves into a binade whose absolute step also halves, so
    ``grid_step * s`` is unchanged.
    """
    orig_dtype = x.dtype
    shape = x.shape
    assert shape[-1] % group_size == 0, (
        f"last dim {shape[-1]} must be a multiple of the quant group size "
        f"{group_size}"
    )
    grouped = x.to(torch.float32).reshape(
        *shape[:-1], shape[-1] // group_size, group_size
    )
    amax = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=amax_floor)
    scale = _round_scale_pow2(amax, fp8_max)
    quantized = (grouped / scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    dequantized = quantized.to(torch.float32) * scale
    return dequantized.reshape(shape).to(orig_dtype)


def _linear(
    x: Tensor,
    weight: Tensor,
    weight_scale: Optional[Tensor],
    out_dtype: torch.dtype,
) -> Tensor:
    """One ``[out, in]``-oriented linear leg, block-FP8 or plain.

    Weights in this checkpoint are stored ``[out, in]`` and the reference's
    unquantized path is ``F.linear(x, weight)`` = ``x @ weight.t()``
    (``dsv4_ref/model.py:114-126, 145, 150``) — the same orientation
    ``NF.block_fp8_linear`` wants, so no ``is_storage_transposed`` fiddling
    (family interface contract §1). The reference's quantized path is
    ``fp8_gemm(act_quant(x, 128, "ue8m0"), w, w.scale)`` with a
    ``[ceil(out/128), ceil(in/128)]`` block scale (``model.py:122-124``,
    ``convert.py:126-127``), which is exactly the
    ``block_size=(128, 128), act_group_size=128`` envelope requested below.

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


def mla_qkv(
    hidden: Tensor,
    wqa_wkv_w: Tensor,
    wqb_w: Tensor,
    q_norm_w: Tensor,
    kv_norm_w: Tensor,
    rope_cos: Tensor,
    rope_sin: Tensor,
    positions: Tensor,
    *,
    wqa_wkv_scale: Optional[Tensor] = None,
    wqb_scale: Optional[Tensor] = None,
    eps: float = 1e-6,
    qk_rope_head_dim: int = 64,
    apply_q_head_norm: bool = True,
    kv_nope_fp8_qat: bool = True,
    kv_qat_group_size: int = 64,
    kv_qat_fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    rope_tables_pregathered: bool = False,
    out_dtype: torch.dtype = torch.bfloat16,
) -> tuple[Tensor, Tensor, Tensor]:
    """DeepSeek-V4 MLA input projections: fused down-proj, q/kv norm, RoPE.

    The whole body is ``dsv4_ref/model.py:502-512`` re-expressed functionally:

    1. Fused down-projection ``hidden [T, hidden_size] -> [T, q_lora_rank +
       head_dim]``, split into the q latent ``[T, q_lora_rank]`` and the KV
       latent ``[T, head_dim]``. The reference keeps these as two separate
       ``Linear``s, ``wq_a`` (``model.py:463``, used at ``:502``) and ``wkv``
       (``model.py:466``, used at ``:508``); the port fuses them into ONE
       replicated GEMM (family interface contract §1, discrepancy 2, and
       upstream vLLM's ``MergedColumnParallelLinear(disable_tp=True)``,
       ``deepseek_v4.py:973-980``). Fusing a shared input across two weight
       matrices is exactly associative, so the math is unchanged.
    2. RMSNorm the q latent with ``q_norm_w`` (``model.py:502``) and the KV
       latent with ``kv_norm_w`` (``model.py:509``). The KV norm runs over the
       WHOLE ``head_dim``-wide raw latent, not per compressed head.
    3. Q up-projection through ``wqb_w`` into per-head latents ``[T,
       heads_local, head_dim]`` (``model.py:503``), then the WEIGHTLESS per-head
       RMSNorm over the full ``head_dim`` (``model.py:504``).
    4. GPT-J RoPE on the last ``qk_rope_head_dim`` dims of q (``model.py:505``)
       AND on the last ``qk_rope_head_dim`` dims of the KV latent
       (``model.py:510``) — the same rotation, same ``freqs_cis``, for both.
       Then the group-64 fp8 quant/dequant round trip on the KV latent's NoPE
       dims only (``model.py:512``). Q is finally split into its NoPE and RoPE
       halves for the caller.

    There is no ``wkv_b`` up-projection: the normed ``head_dim``-wide latent is
    itself both the SWA cache content and the compressor's input
    (``model.py:508-531``; upstream spec §2, closing paragraph).

    Args:
        hidden: ``[T, hidden_size]`` post-``attn_norm`` activations. The hidden
            stream is replicated across TP ranks, never sharded.
        wqa_wkv_w: ``[q_lora_rank + head_dim, hidden_size]`` (``[1536, 4096]``
            at production config) — the port's fusion of the checkpoint's
            separate ``attn.wq_a`` and ``attn.wkv`` tensors. Replicated.
        wqb_w: ``[heads_local * head_dim, q_lora_rank]`` (``[512, 1024]`` at
            TP=64, one head per core) — column-parallel over heads.
        q_norm_w: ``[q_lora_rank]`` RMSNorm gain for the q latent.
        kv_norm_w: ``[head_dim]`` RMSNorm gain for the KV latent. Its length is
            what defines ``head_dim`` here.
        rope_cos: ``[max_position, qk_rope_head_dim // 2]`` (or full width) cos
            table for this layer's RoPE theta — i.e. ``freqs_cis.real``.
            DeepSeek-V4 is dual-theta AND dual-YaRN: ``model.py:481-487`` uses
            ``(original_seq_len=65536, compress_rope_theta=160000)`` on
            compressed layers and ``(original_seq_len=0, rope_theta=10000)`` on
            SWA-only layers, where ``original_seq_len=0`` DISABLES YaRN
            interpolation entirely (``model.py:227-230``). Building that table is
            the caller's job; this op only gathers and applies it.
        rope_sin: matching sin table.
        positions: ``[T]`` integer absolute positions used to gather cos/sin.
        wqa_wkv_scale: block-FP8 weight scale for ``wqa_wkv_w``, fp32
            ``[ceil(N/128), ceil(K/128)]``. ``None`` selects the plain-matmul
            path. KEYWORD-ONLY EXTENSION over the plan-fixed signature.
        wqb_scale: same, for ``wqb_w``. KEYWORD-ONLY EXTENSION.
        eps: RMSNorm epsilon (``rms_norm_eps``, 1e-6). KEYWORD-ONLY EXTENSION.
        qk_rope_head_dim: rotated width inside ``head_dim`` (64). The NoPE width
            is ``head_dim - qk_rope_head_dim`` (448). KEYWORD-ONLY EXTENSION.
        apply_q_head_norm: run the weightless per-head Q RMSNorm of step 3.
            Defaults True because the reference applies it unconditionally
            (``model.py:504``); exposed only so a caller that has already normed
            q can skip it. KEYWORD-ONLY EXTENSION.
        kv_nope_fp8_qat: run the group-64 fp8 quant/dequant round trip on the KV
            latent's NoPE dims. Defaults True because the reference applies it
            unconditionally (``model.py:512``) — it is a QAT simulation, so
            skipping it makes inference numerics differ from the trained
            numerics. Set False if the target backend cannot cast to
            ``kv_qat_fp8_dtype``. KEYWORD-ONLY EXTENSION.
        kv_qat_group_size: quant group width for that round trip, 64 per
            ``model.py:512``. 448 NoPE dims / 64 = the 7 group scales the KV
            cache contract (§4) reserves. KEYWORD-ONLY EXTENSION.
        kv_qat_fp8_dtype: storage dtype of the round trip. ``float8_e4m3fn``
            (OCP) per ``kernel.py:47-48, 116``; TRN2's non-OCP fp8 is a
            different type, hence the knob. KEYWORD-ONLY EXTENSION.
        rope_tables_pregathered: set True when ``rope_cos``/``rope_sin`` already
            carry ONE ROW PER TOKEN (``[T, qk_rope_head_dim // 2]``) rather than
            one row per POSITION (``[max_position, ...]``), so the position
            lookup is skipped. KEYWORD-ONLY EXTENSION.

            This is not a convenience: the two conventions are
            indistinguishable from the tensors alone, and feeding a per-token
            table with ``pregathered=False`` is silently correct for exactly
            one input shape -- a single sequence prefilled from position 0,
            where ``positions == arange(T)`` -- and wrong everywhere else. At
            decode, ``positions`` holds context lengths in the thousands while
            the table has ``batch`` rows, so ``index_select`` goes out of
            range. Both in-tree DeepSeek-V4 callers build their tables with
            ``model/deepseek_v4/attention.py``'s ``_cos_sin`` (or the
            equivalent ``DeepseekV4RotaryEmbedding.forward``), both of which
            are per-TOKEN, so both pass True.

            ``sparse_indexer_topk`` takes the identically named flag for the
            same reason; the two ops share the convention so a caller cannot
            get one right and the other wrong.
        out_dtype: store dtype for every returned tensor. KEYWORD-ONLY
            EXTENSION.

    Returns:
        ``(q_nope, q_rope, latent_kv)`` where

        * ``q_nope`` is ``[T, heads_local, head_dim - qk_rope_head_dim]``
          (``[T, 1, 448]``) — the un-rotated content dims of q,
        * ``q_rope`` is ``[T, heads_local, qk_rope_head_dim]``
          (``[T, 1, 64]``) — the rotated dims of q,
        * ``latent_kv`` is the FULL ``[T, head_dim]`` (``[T, 512]``) normed KV
          latent with RoPE already applied to its trailing
          ``qk_rope_head_dim`` dims and the fp8 QAT round trip applied to its
          leading NoPE dims, i.e. dims ``[0:448]`` are NoPE and ``[448:512]``
          are RoPE. The caller splits/quantizes/pages it per the KV cache
          contract (§4: NoPE ``[0:224]`` -> ``k_cache``, NoPE ``[224:448]`` ->
          ``v_cache``, RoPE 64 -> the ``.rope`` pair). Re-quantizing the NoPE
          dims at the cache write with the same group-64 ue8m0 scheme is
          idempotent, so the round trip here does not double-quantize.
    """
    q_lora_rank = q_norm_w.shape[0]
    head_dim = kv_norm_w.shape[0]
    nope_head_dim = head_dim - qk_rope_head_dim

    assert wqa_wkv_w.shape[0] == q_lora_rank + head_dim, (
        "fused wqa/wkv output width must be q_lora_rank + head_dim = "
        f"{q_lora_rank + head_dim}, got {wqa_wkv_w.shape[0]}"
    )
    assert wqa_wkv_w.shape[1] == hidden.shape[-1], (
        f"wqa_wkv_w input width {wqa_wkv_w.shape[1]} != hidden width "
        f"{hidden.shape[-1]}"
    )
    assert wqb_w.shape[1] == q_lora_rank, (
        f"wqb_w input width {wqb_w.shape[1]} != q_lora_rank {q_lora_rank}"
    )
    assert wqb_w.shape[0] % head_dim == 0, (
        f"wqb_w output width {wqb_w.shape[0]} must be a multiple of head_dim "
        f"{head_dim} (it is heads_local * head_dim)"
    )
    assert qk_rope_head_dim % 2 == 0 and 0 < qk_rope_head_dim <= head_dim, (
        f"qk_rope_head_dim must be even and within head_dim, got "
        f"{qk_rope_head_dim} vs {head_dim}"
    )
    heads_local = wqb_w.shape[0] // head_dim

    # --- Step 1: fused q/kv down-projection (one GEMM, replicated) -----------
    # dsv4_ref/model.py:502 (`wq_a(x)`) and :508 (`wkv(x)`) fused into one GEMM,
    # per upstream's MergedColumnParallelLinear (deepseek_v4_attention.py:399-401
    # then :429).
    qr_kv = _linear(hidden, wqa_wkv_w, wqa_wkv_scale, out_dtype)
    q_latent, kv_latent = qr_kv.split([q_lora_rank, head_dim], dim=-1)

    # --- Step 2: q_norm on the q latent, kv_norm on the raw KV latent -------
    # dsv4_ref/model.py:502 and :509. Both legs are the same fp32 RMSNorm
    # (model.py:197-202), which is why they are two calls to one helper.
    q_latent = _rms_norm(q_latent, q_norm_w, eps, out_dtype)
    kv_latent = _rms_norm(kv_latent, kv_norm_w, eps, out_dtype)

    # --- Step 3: q up-projection to per-head latents, then per-head Q norm --
    # dsv4_ref/model.py:503: q = wq_b(qr).unflatten(-1, (n_local_heads, head_dim)).
    q = _linear(q_latent, wqb_w, wqb_scale, out_dtype)
    q = q.reshape(q.shape[0], heads_local, head_dim)
    if apply_q_head_norm:
        # dsv4_ref/model.py:504 — weightless RMSNorm over the FULL head_dim,
        # BEFORE RoPE.
        q = _rms_norm(q, None, eps, out_dtype)

    # --- Step 4: GPT-J RoPE on the trailing rope dims of q and of the KV latent
    # dsv4_ref/model.py:505 and :510 apply ONE rotation (same freqs_cis) to both.
    cos, sin = _gather_rope_tables(
        rope_cos, rope_sin, positions, qk_rope_head_dim, rope_tables_pregathered
    )

    q_nope = q[..., :nope_head_dim]
    q_rope = _apply_gptj_rope(q[..., nope_head_dim:], cos, sin)

    kv_nope = kv_latent[..., :nope_head_dim]
    kv_rope = _apply_gptj_rope(kv_latent[..., nope_head_dim:], cos, sin)
    if kv_nope_fp8_qat:
        # dsv4_ref/model.py:512 — act_quant(kv[..., :-rd], 64, "ue8m0",
        # float8_e8m0fnu, inplace=True): fp8-simulate the NoPE dims to match QAT;
        # the RoPE dims deliberately stay bf16 for positional precision.
        kv_nope = _fp8_group_quant_dequant(
            kv_nope, kv_qat_group_size, kv_qat_fp8_dtype
        )
    latent_kv = torch.cat((kv_nope, kv_rope), dim=-1)

    return q_nope.to(out_dtype), q_rope.to(out_dtype), latent_kv.to(out_dtype)
