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
) -> tuple[Tensor, Tensor]:
    """Gather the per-token cos/sin rows at ``positions``, in fp32.

    ``index_select`` with an int64 index tensor is used (never advanced/boolean
    indexing) so the graph keeps a static output shape ``[T, rope_head_dim/2]``.

    Both the half-width table ``[max_pos, rope_head_dim // 2]`` (one entry per
    rotated pair, the natural layout for GPT-J RoPE) and the full-width table
    ``[max_pos, rope_head_dim]`` are accepted; for the latter the leading half
    is taken, matching upstream's single ``cos_sin_cache`` whose cos block is
    ``[0:HALF_ROPE]`` (``fused_inv_rope_fp8_quant.py:88-89``).
    """
    half = rope_head_dim // 2
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


def _round_scale_pow2(amax: Tensor) -> Tensor:
    """``s = 2 ** ceil(log2(amax / 448))`` via the reference's IEEE-754 bit trick.

    ``dsv4_ref/kernel.py:36-37`` computes the ue8m0 (power-of-two) act-quant
    scale as ``fast_pow2(fast_log2_ceil(amax * (1/448)))``, and
    ``kernel.py:22-33`` implements those two with bit arithmetic::

        exp = (bits >> 23) & 0xFF; man = bits & ((1 << 23) - 1)
        ceil_log2 = exp - 127 + (1 if man != 0 else 0)
        pow2      = reinterpret_f32((ceil_log2 + 127) << 23)

    Reproducing the bit form rather than ``exp2(ceil(log2(v)))`` matters at exact
    powers of two, where fp32 ``log2`` can land a hair above or below the integer
    and flip the ceiling — one whole exponent of quantization error. It is also
    the same reinterpretation idiom the port must use for the checkpoint's
    ``float8_e8m0fnu`` scales (family interface contract §1, discrepancy 1).

    ``(man != 0)`` is used as ARITHMETIC (``.to(int32)``), never as an index, so
    the graph stays static-shape.
    """
    v = (amax.to(torch.float32) * (1.0 / 448.0)).contiguous()
    bits = v.view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    ceil_log2 = exponent - 127 + (mantissa != 0).to(torch.int32)
    return ((ceil_log2 + 127) << 23).view(torch.float32)


def _fp8_group_quant_dequant(
    x: Tensor,
    group_size: int,
    fp8_dtype: torch.dtype,
    fp8_max: float = 448.0,
    amax_floor: float = 1e-4,
) -> Tensor:
    """Group-wise fp8 quant/dequant round trip along the last dim.

    This is ``act_quant(..., inplace=True)``: the reference's QAT simulation,
    which quantizes to fp8 and immediately dequantizes back to bf16 so that
    inference sees the values the model was trained to see
    (``dsv4_ref/kernel.py:84-91`` for the inplace branch,
    ``kernel.py:105-125`` for the wrapper). Per group::

        amax = max(|x|.amax(-1), 1e-4)
        s    = 2 ** ceil(log2(amax / 448))      # scale_fmt="ue8m0"
        y    = f32(fp8(clamp(x / s, -448, 448))) * s

    ``fp8_max = 448`` and the ``1e-4`` amax floor are the reference's literals
    (``kernel.py:47-49, 79``). Skipping this step does not make the result
    "cleaner" — it makes it differ from the trained numerics.
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
    scale = _round_scale_pow2(amax)
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
    cos, sin = _gather_rope_tables(rope_cos, rope_sin, positions, qk_rope_head_dim)

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
