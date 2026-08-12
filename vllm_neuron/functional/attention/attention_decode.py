# SPDX-License-Identifier: Apache-2.0
from nkilib.experimental.transformer.attention_block_tkg import (
    attention_block_tkg as _nki_attention_block_tkg,
)

from typing import Optional, Tuple

import torch
from torch.distributed import ProcessGroup

import nki
from torch import Tensor

from nkilib.core.utils.allocator import SbufManager
from nkilib.core.utils.common_types import QuantizationType

from vllm_neuron.nki.nki_hop import can_run_kernel, wrap_nki

from vllm_neuron.functional.attention.attention_decode_mask import (
    _resize_block_len as _mask_resize_block_len,
    P_MAX as _MASK_P_MAX,
)

_PMAX = 128

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_default_active_mask(
    B: int,
    S_tkg: int,
    q_heads: int,
    device: torch.device,
) -> Tensor:
    """Default active-only causal mask for the fused mask-gen path.

    Shape: [S_tkg, B, q_heads, S_tkg], values in {0, 1}.
    For S_tkg=1 (regular decode) this is a [1, B, q_heads, 1] tensor of ones.
    """
    triu = torch.triu(torch.ones(S_tkg, S_tkg, dtype=torch.float32, device=device))
    return triu[:, None, None, :].expand(S_tkg, B, q_heads, S_tkg).contiguous()


def _maybe_build_sbm(
    sbm_lower_bound: Optional[int],
    sbm_upper_bound: Optional[int],
    sbm_use_auto_alloc: bool,
    sbm_default_stack_alloc: bool,
) -> Optional[SbufManager]:
    if sbm_lower_bound is None and sbm_upper_bound is None:
        return None

    return SbufManager(
        sb_lower_bound=sbm_lower_bound,
        sb_upper_bound=sbm_upper_bound,
        use_auto_alloc=sbm_use_auto_alloc,
        default_stack_alloc=sbm_default_stack_alloc,
    )


# ---------------------------------------------------------------------------
# NKI entry-point (runs on NeuronCore)
# ---------------------------------------------------------------------------


@nki.jit
def _torch_compatible_attention_block_tkg_kernel(
    # -- input
    X: Tensor,
    X_hidden_dim_actual: int = None,
    # -- rmsnorm X
    rmsnorm_X_enabled: bool = False,
    rmsnorm_X_eps: Optional[float] = None,
    rmsnorm_X_gamma: Tensor = None,
    # -- qkv projections
    W_qkv: Tensor = None,
    bias_qkv: Tensor = None,
    quantization_type_qkv: QuantizationType = QuantizationType.NONE,
    weight_dequant_scale_qkv: torch.Tensor = None,
    input_dequant_scale_qkv: torch.Tensor = None,
    # -- Q/K processing: pre-RoPE RMSNorm
    rmsnorm_QK_pre_rope_enabled: bool = False,
    rmsnorm_QK_pre_rope_eps: float = 1e-6,
    rmsnorm_QK_pre_rope_W_Q: Tensor = None,
    rmsnorm_QK_pre_rope_W_K: Tensor = None,
    # -- Q/K processing: RoPE
    cos: Optional[Tensor] = None,
    sin: Optional[Tensor] = None,
    rope_contiguous_layout: bool = True,
    # -- Q/K processing: post-RoPE RMSNorm
    rmsnorm_QK_post_rope_enabled: bool = False,
    rmsnorm_QK_post_rope_eps: float = 1e-6,
    rmsnorm_QK_post_rope_W_Q: Tensor = None,
    rmsnorm_QK_post_rope_W_K: Tensor = None,
    # -- attention
    K_cache_transposed: bool = False,
    active_blocks_table: Tensor = None,
    K_cache: Tensor = None,
    V_cache: Tensor = None,
    attention_mask: Tensor = None,
    sink: Tensor = None,
    softmax_scale: float = None,
    # -- in-kernel mask generation (when pos_ids is provided, the kernel
    #    generates the prior causal/SWA mask on-chip from pos_ids and
    #    attention_mask carries only the active-only portion)
    pos_ids: Tensor = None,
    swa_start_pos_ids: Tensor = None,
    max_context_len: Tensor = None,
    # -- KV cache update
    update_cache: bool = False,
    kv_cache_update_idx: Tensor = None,
    k_scale: Tensor = None,
    v_scale: Tensor = None,
    # -- packed FP8 K cache layout
    fp8_packed: bool = False,
    # -- output projection
    W_out: Tensor = None,
    bias_out: Tensor = None,
    quantization_type_out: QuantizationType = QuantizationType.NONE,
    weight_dequant_scale_out: Tensor = None,
    input_dequant_scale_out: Tensor = None,
    transposed_out: bool = False,
    # -- output control
    out_in_sb: bool = False,
    skip_attention: bool = False,
    # -- STATIC_MX layout flag
    is_h_transposed_by_4: bool = False,
    # -- sbm
    sbm_lower_bound: int = None,
    sbm_upper_bound: int = None,
    sbm_use_auto_alloc: bool = True,
    sbm_default_stack_alloc: bool = True,
):
    """
    Torch-friendly @nki.jit wrapper that reconstructs NKI-specific objects
    (SbufManager, QuantizationType) from plain scalar / tensor args and
    delegates to the library attention_block_tkg kernel.
    """

    sbm = _maybe_build_sbm(
        sbm_lower_bound, sbm_upper_bound, sbm_use_auto_alloc, sbm_default_stack_alloc
    )

    return _nki_attention_block_tkg(
        X=X,
        X_hidden_dim_actual=X_hidden_dim_actual,
        rmsnorm_X_enabled=rmsnorm_X_enabled,
        rmsnorm_X_eps=rmsnorm_X_eps,
        rmsnorm_X_gamma=rmsnorm_X_gamma,
        W_qkv=W_qkv,
        bias_qkv=bias_qkv,
        quantization_type_qkv=quantization_type_qkv,
        weight_dequant_scale_qkv=weight_dequant_scale_qkv,
        input_dequant_scale_qkv=input_dequant_scale_qkv,
        rmsnorm_QK_pre_rope_enabled=rmsnorm_QK_pre_rope_enabled,
        rmsnorm_QK_pre_rope_eps=rmsnorm_QK_pre_rope_eps,
        rmsnorm_QK_pre_rope_W_Q=rmsnorm_QK_pre_rope_W_Q,
        rmsnorm_QK_pre_rope_W_K=rmsnorm_QK_pre_rope_W_K,
        cos=cos,
        sin=sin,
        rope_contiguous_layout=rope_contiguous_layout,
        rmsnorm_QK_post_rope_enabled=rmsnorm_QK_post_rope_enabled,
        rmsnorm_QK_post_rope_eps=rmsnorm_QK_post_rope_eps,
        rmsnorm_QK_post_rope_W_Q=rmsnorm_QK_post_rope_W_Q,
        rmsnorm_QK_post_rope_W_K=rmsnorm_QK_post_rope_W_K,
        K_cache_transposed=K_cache_transposed,
        active_blocks_table=active_blocks_table,
        K_cache=K_cache,
        V_cache=V_cache,
        attention_mask=attention_mask,
        sink=sink,
        softmax_scale=softmax_scale,
        update_cache=update_cache,
        kv_cache_update_idx=kv_cache_update_idx,
        k_scale=k_scale,
        v_scale=v_scale,
        fp8_packed=fp8_packed,
        W_out=W_out,
        bias_out=bias_out,
        quantization_type_out=quantization_type_out,
        weight_dequant_scale_out=weight_dequant_scale_out,
        input_dequant_scale_out=input_dequant_scale_out,
        transposed_out=transposed_out,
        out_in_sb=out_in_sb,
        sbm=sbm,
        skip_attention=skip_attention,
        is_h_transposed_by_4=is_h_transposed_by_4,
        KVDP=1,
        KVDP_replica_group=None,
        pos_ids=pos_ids,
        swa_start_pos_ids=swa_start_pos_ids,
        S_ctx=None,
        max_context_len=max_context_len,
    )


# ---------------------------------------------------------------------------
# Torch helpers for the fallback implementation
# ---------------------------------------------------------------------------


def _torch_rms_norm(
    x: Tensor, eps: float, weight: Optional[Tensor], dim_actual: Optional[int] = None
) -> Tensor:
    """
    RMS normalization: x / sqrt(mean(x^2) + eps), optionally scaled by weight.

    Args:
        x: Input tensor, normalised along the last dimension.
        eps: Epsilon for numerical stability.
        weight: Optional per-element scale (broadcastable to x).
        dim_actual: If the last dimension is zero-padded, supply the real
                    (unpadded) size so the mean is computed correctly.

    Returns:
        Normalised tensor in the same dtype as *x*.
    """
    input_dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    d = dim_actual if dim_actual is not None else x.shape[-1]
    variance = x_fp32.pow(2).sum(-1, keepdim=True) / d
    x_normed = x_fp32 * torch.rsqrt(variance + eps)
    if weight is not None:
        x_normed = x_normed * weight.to(torch.float32)
    return x_normed.to(input_dtype)


def _torch_rms_norm_heads(x: Tensor, eps: float, weight: Optional[Tensor]) -> Tensor:
    """
    Per-head RMS normalization over the last (d_head) dimension.

    Args:
        x: [..., d_head]
        eps: Epsilon.
        weight: [1, d_head] or [d_head] scale. Broadcast across batch/head dims.

    Returns:
        Normalised tensor, same shape and dtype as *x*.
    """
    input_dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    variance = x_fp32.pow(2).mean(-1, keepdim=True)
    x_normed = x_fp32 * torch.rsqrt(variance + eps)
    if weight is not None:
        w = weight.to(torch.float32).view(-1)  # flatten to [d_head]
        x_normed = x_normed * w
    return x_normed.to(input_dtype)


def _torch_rotate_half(x: Tensor) -> Tensor:
    """Rotates half the hidden dims of the input (contiguous layout)."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _torch_rotate_half_interleaved(x: Tensor) -> Tensor:
    """Rotates hidden dims using interleaved layout (pairs of adjacent elements)."""
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def _torch_apply_rope(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    rope_contiguous_layout: bool,
) -> Tuple[Tensor, Tensor]:
    """
    Apply Rotary Position Embedding to Q and K.

    Args:
        q: [B, q_heads, S_tkg, d_head]
        k: [B, kv_heads, S_tkg, d_head]
        cos: [d_head//2, B, S_tkg]  RoPE cosine values.
        sin: [d_head//2, B, S_tkg]  RoPE sine values.
        rope_contiguous_layout: True = first-half/second-half split,
                                False = interleaved pairs.

    Returns:
        (q_rotated, k_rotated) with the same shapes and dtype as inputs.
    """
    # cos, sin: [d_head//2, B, S_tkg] -> [B, 1, S_tkg, d_head//2]
    cos_r = cos.permute(1, 2, 0).unsqueeze(1)  # [B, 1, S_tkg, d_head//2]
    sin_r = sin.permute(1, 2, 0).unsqueeze(1)  # [B, 1, S_tkg, d_head//2]

    if rope_contiguous_layout:
        cos_full = torch.cat([cos_r, cos_r], dim=-1)  # [B, 1, S_tkg, d_head]
        sin_full = torch.cat([sin_r, sin_r], dim=-1)
        rotate_fn = _torch_rotate_half
    else:
        cos_full = torch.stack([cos_r, cos_r], dim=-1).flatten(-2)
        sin_full = torch.stack([sin_r, sin_r], dim=-1).flatten(-2)
        rotate_fn = _torch_rotate_half_interleaved

    cos_full = cos_full.to(q.dtype)
    sin_full = sin_full.to(q.dtype)

    q_rotated = (q * cos_full) + (rotate_fn(q) * sin_full)
    k_rotated = (k * cos_full) + (rotate_fn(k) * sin_full)

    return q_rotated, k_rotated


# ---------------------------------------------------------------------------
# Packed FP8 K-cache swizzle helpers (torch fallback)
# ---------------------------------------------------------------------------
#
# The packed layout interleaves two consecutive sequence positions into the
# trailing size-2 dim so the kernel can bf16-reinterpret + DMA-transpose:
#
#     unpacked: [..., block_len, d_head]
#     packed:   [..., block_len // 2, d_head, 2]   (dim -1: 0=even pos, 1=odd pos)
#
# These mirror the kernel's integration-test swizzle
#   K.reshape(nb, bl // 2, 2, d).transpose(0, 1, 3, 2)
# but operate on whatever leading dims the cache carries (3D/4D/5D).


def _unswizzle_packed_k(k_packed: Tensor) -> Tensor:
    """[..., block_len // 2, d_head, 2] fp8 → [..., block_len, d_head].

    Inverse of :func:`_swizzle_packed_k`. Leaves dtype unchanged.
    """
    *lead, half, d_head, two = k_packed.shape
    assert two == 2, f"packed K last dim must be 2, got {two}"
    # [..., half, d_head, 2] → [..., half, 2, d_head] → [..., half*2, d_head]
    return k_packed.transpose(-1, -2).reshape(*lead, half * 2, d_head).contiguous()


def _swizzle_packed_k(k_unpacked: Tensor) -> Tensor:
    """[..., block_len, d_head] → [..., block_len // 2, d_head, 2].

    Inverse of :func:`_unswizzle_packed_k`. Leaves dtype unchanged.
    """
    *lead, block_len, d_head = k_unpacked.shape
    assert block_len % 2 == 0, f"block_len must be even to pack, got {block_len}"
    # [..., bl, d_head] → [..., bl//2, 2, d_head] → [..., bl//2, d_head, 2]
    return (
        k_unpacked.reshape(*lead, block_len // 2, 2, d_head)
        .transpose(-1, -2)
        .contiguous()
    )


# ---------------------------------------------------------------------------
# PyTorch fallback implementation
# ---------------------------------------------------------------------------


def _torch_attention_decode_impl(
    # -- input
    X: Tensor,
    X_hidden_dim_actual: Optional[int] = None,
    # -- rmsnorm X
    rmsnorm_X_enabled: bool = False,
    rmsnorm_X_eps: Optional[float] = None,
    rmsnorm_X_gamma: Optional[Tensor] = None,
    # -- qkv projections
    W_qkv: Tensor = None,
    bias_qkv: Optional[Tensor] = None,
    # -- Q/K processing: pre-RoPE RMSNorm
    rmsnorm_QK_pre_rope_enabled: bool = False,
    rmsnorm_QK_pre_rope_eps: float = 1e-6,
    rmsnorm_QK_pre_rope_W_Q: Optional[Tensor] = None,
    rmsnorm_QK_pre_rope_W_K: Optional[Tensor] = None,
    # -- Q/K processing: RoPE
    cos: Optional[Tensor] = None,
    sin: Optional[Tensor] = None,
    rope_contiguous_layout: bool = True,
    # -- Q/K processing: post-RoPE RMSNorm
    rmsnorm_QK_post_rope_enabled: bool = False,
    rmsnorm_QK_post_rope_eps: float = 1e-6,
    rmsnorm_QK_post_rope_W_Q: Optional[Tensor] = None,
    rmsnorm_QK_post_rope_W_K: Optional[Tensor] = None,
    # -- attention
    active_blocks_table: Optional[Tensor] = None,
    K_cache: Tensor = None,
    V_cache: Tensor = None,
    attention_mask: Tensor = None,
    sink: Tensor = None,
    softmax_scale: Optional[float] = None,
    # -- KV cache update
    update_cache: bool = False,
    kv_cache_update_idx: Optional[Tensor] = None,
    # -- FP8 KV cache quantization
    k_scale: Optional[Tensor] = None,
    v_scale: Optional[Tensor] = None,
    # -- packed FP8 K cache layout
    fp8_packed: bool = False,
    # -- output projection
    W_out: Optional[Tensor] = None,
    bias_out: Optional[Tensor] = None,
    # -- Attention Dependent DP
    attention_dp: int = 1,
    attention_dp_group: Optional[ProcessGroup] = None,
    attention_dp_rank: int = 0,
    kv_needs_a2a: bool = False,
    # -- DCP Decode (Gather Q + LSE correction)
    dcp_size: int = 1,
    dcp_group=None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    PyTorch fallback implementation of the fused attention block for TKG.

    Supports **block KV cache** layout only, with single or multiple KV heads.

    Cache layout convention (matching the model code):
        - 4D: [num_blocks, kv_heads, block_len, d_head]
        - 3D: [num_blocks, block_len, d_head]  (kv_heads=1, pre-squeezed by caller)

    The implementation follows the same algorithmic stages as the NKI kernel:

        RMSNorm(X) → QKV projection → split Q/K/V → optional pre-RoPE
        RMSNorm → RoPE → optional post-RoPE RMSNorm → GQA attention with
        block KV cache → optional KV cache update → optional output projection.

    The attention mask is expected to come from ``gen_mask`` and therefore
    already contains the active-token causal pattern in its last ``S_tkg``
    rows.

    The KV cache update uses ``index_put_`` with (block, head, position)
    indices, matching the pattern used in the model's ``forward_decode``
    and ``forward_prefill`` methods.

    Returns:
        - update_cache=True: ``output`` only (the caches are written in place,
          so the new K/V tokens are redundant).
        - update_cache=False: ``(output, K_out, V_out)`` where K_out/V_out are
          the new K/V tokens for the caller to write into its cache.

        output:
            - With W_out: [B*S_tkg, H]
            - Without W_out: [B, q_heads, d_head, S_tkg]
        K_out: [d_head, B, S_tkg] new K tokens (update_cache=False only).
        V_out: [B, 1, S_tkg, d_head] new V tokens (update_cache=False only).
    """
    assert active_blocks_table is not None, (
        "PyTorch attention_block fallback only supports block KV cache "
        "(active_blocks_table must be provided)"
    )

    B, S_tkg, H = X.shape

    # ── Determine kv_heads and cache geometry ─────────────────────────────
    # Model convention:
    #   4D cache: [num_blocks, kv_heads_cache, block_len, d_head]
    #   3D cache: [num_blocks, block_len, d_head]  (kv_heads=1, squeezed)
    # Geometry is derived from V_cache because it is never packed; the packed
    # K_cache (fp8_packed=True) stores block_len//2 in a swizzled layout and so
    # cannot be read for block_len/d_head directly.
    if V_cache.dim() == 4:
        num_blocks_total, kv_heads_cache, block_len, d_head = V_cache.shape
    else:
        assert V_cache.dim() == 3, f"V_cache must be 3D or 4D, got {V_cache.dim()}D"
        num_blocks_total, block_len, d_head = V_cache.shape
        kv_heads_cache = 1

    if fp8_packed:
        # Packed K: [num_blocks, (kv_heads,) block_len // 2, d_head, 2] fp8.
        expected = (
            (num_blocks_total, kv_heads_cache, block_len // 2, d_head, 2)
            if V_cache.dim() == 4
            else (num_blocks_total, block_len // 2, d_head, 2)
        )
        assert tuple(K_cache.shape) == expected, (
            f"fp8_packed K_cache shape mismatch: expected {expected}, "
            f"got {tuple(K_cache.shape)}"
        )

    # kv_heads for QKV split: derived from weight. With kv_needs_a2a the weight
    # has fewer KV heads than the cache (weight is dependent-DP-sharded, cache stores
    # the gathered result). Derive from cache + attention_dp to get the projected count.
    kv_heads = kv_heads_cache // attention_dp if kv_needs_a2a else kv_heads_cache

    I = W_qkv.shape[1]
    q_heads = I // d_head - 2 * kv_heads
    num_kv_groups = q_heads // kv_heads  # GQA group size

    num_blocks_per_seq = active_blocks_table.shape[1]
    S_ctx = num_blocks_per_seq * block_len

    # ================================================================
    # Stage 1: Optional RMSNorm on input
    # ================================================================
    hidden = X
    if rmsnorm_X_enabled:
        eps = rmsnorm_X_eps if rmsnorm_X_eps is not None else 1e-3
        hidden = _torch_rms_norm(hidden, eps, rmsnorm_X_gamma, X_hidden_dim_actual)

    # ================================================================
    # Stage 2: QKV projection
    # ================================================================
    qkv = hidden @ W_qkv  # [B, S_tkg, I]
    if bias_qkv is not None:
        qkv = qkv + bias_qkv

    # ================================================================
    # Stage 3: Split Q, K, V and reshape to head layout
    # ================================================================
    q_end = q_heads * d_head
    k_end = q_end + kv_heads * d_head

    q = qkv[..., :q_end]  # [B, S_tkg, q_heads * d_head]
    k = qkv[..., q_end:k_end]  # [B, S_tkg, kv_heads * d_head]
    v = qkv[..., k_end:]  # [B, S_tkg, kv_heads * d_head]

    q = q.view(B, S_tkg, q_heads, d_head).transpose(1, 2)  # [B, q_heads, S_tkg, d_head]
    k = k.view(B, S_tkg, kv_heads, d_head).transpose(
        1, 2
    )  # [B, kv_heads, S_tkg, d_head]
    v = v.view(B, S_tkg, kv_heads, d_head).transpose(
        1, 2
    )  # [B, kv_heads, S_tkg, d_head]

    # ================================================================
    # Stage 4: Optional pre-RoPE RMSNorm on Q/K
    # ================================================================
    if rmsnorm_QK_pre_rope_enabled:
        q = _torch_rms_norm_heads(q, rmsnorm_QK_pre_rope_eps, rmsnorm_QK_pre_rope_W_Q)
        k = _torch_rms_norm_heads(k, rmsnorm_QK_pre_rope_eps, rmsnorm_QK_pre_rope_W_K)

    # ================================================================
    # Stage 5: Dependent DP all-to-all Q + select local K/V
    # Done before RoPE so RoPE only needs local batch's cos/sin.
    # ================================================================
    if attention_dp > 1:
        from vllm_neuron.functional import all_to_all

        # q: [DDP*B_local, q_heads_small, S_tkg, d_head]
        # Contiguous needed: transpose from QKV split leaves q non-contiguous
        # all-to-all: swap batch↔heads
        # → [B_local, DDP*q_heads_small, S_tkg, d_head]
        q = all_to_all(
            q.contiguous(),
            split_dim=0,
            concat_dim=1,
            group=attention_dp_group,
        )

        B_local = q.shape[0]

        if kv_needs_a2a:
            # KV also sharded across attention DP — a2a to gather all KV heads
            k = all_to_all(
                k.contiguous(),
                split_dim=0,
                concat_dim=1,
                group=attention_dp_group,
            )
            v = all_to_all(
                v.contiguous(),
                split_dim=0,
                concat_dim=1,
                group=attention_dp_group,
            )
        else:
            # KV fits in TP — just select local batch
            k = k[attention_dp_rank * B_local : (attention_dp_rank + 1) * B_local]
            v = v[attention_dp_rank * B_local : (attention_dp_rank + 1) * B_local]

        # Update head counts for post-all-to-all state
        q_heads = q.shape[1]
        kv_heads = k.shape[1]
        num_kv_groups = q_heads // kv_heads
        B = B_local

    # ================================================================
    # Stage 6: RoPE (on local batch only when attention DP enabled)
    # ================================================================
    if cos is not None and sin is not None:
        q, k = _torch_apply_rope(q, k, cos, sin, rope_contiguous_layout)

    # ================================================================
    # Stage 6.5: Optional post-RoPE RMSNorm on Q/K
    # ================================================================
    if rmsnorm_QK_post_rope_enabled:
        q = _torch_rms_norm_heads(q, rmsnorm_QK_post_rope_eps, rmsnorm_QK_post_rope_W_Q)
        k = _torch_rms_norm_heads(k, rmsnorm_QK_post_rope_eps, rmsnorm_QK_post_rope_W_K)

    # ================================================================
    # Stage 6.8: DCP AllGather Q across DCP group
    # Each rank has Q for its local heads. After gather, every rank has
    # Q for all heads in the DCP group (the KV replica set).
    # ================================================================
    if dcp_size > 1:
        q = dcp_group.all_gather(q.contiguous(), dim=1)
        q_heads = q.shape[1]
        num_kv_groups = q_heads // kv_heads

    # ================================================================
    # Stage 7: Attention with block KV cache
    # ================================================================

    # 7a. Gather K/V from block cache → [B, kv_heads_cache, S_ctx, d_head]
    # Use kv_heads_cache (not kv_heads) since the cache stores the full
    # per-TP heads, which may differ from the projected count with kv_needs_a2a.
    #
    # torch fallback only: clamp -1 sentinels back to 0 for KV block indexing
    safe_idx = torch.where(
        active_blocks_table < 0,
        torch.zeros_like(active_blocks_table),
        active_blocks_table,
    )
    flat_idx = safe_idx.long().reshape(-1)  # [B * num_blocks_per_seq]

    # When the cache is FP8, cast to compute dtype before indexing since
    # PyTorch CPU does not support fancy indexing on float8 dtypes.
    # Dequantization scales are already fused into softmax_scale (for K)
    # and W_out (for V) by the caller.
    #
    # For the packed FP8 K cache, un-swizzle to the standard
    # [num_blocks, (kv_heads,) block_len, d_head] layout first so the gather
    # logic below is identical to the unpacked path.
    K_cache_read = _unswizzle_packed_k(K_cache) if fp8_packed else K_cache
    K_blocks = K_cache_read.to(X.dtype)[flat_idx]
    V_blocks = V_cache.to(X.dtype)[flat_idx]

    if V_cache.dim() == 4:
        # 4D: [num_blocks, kv_heads_cache, block_len, d_head]
        K_blocks = K_blocks.view(
            B, num_blocks_per_seq, kv_heads_cache, block_len, d_head
        )
        V_blocks = V_blocks.view(
            B, num_blocks_per_seq, kv_heads_cache, block_len, d_head
        )
        K_gathered = K_blocks.permute(0, 2, 1, 3, 4).reshape(
            B, kv_heads_cache, S_ctx, d_head
        )
        V_gathered = V_blocks.permute(0, 2, 1, 3, 4).reshape(
            B, kv_heads_cache, S_ctx, d_head
        )
    else:
        # 3D: [num_blocks, block_len, d_head] (kv_heads=1)
        K_gathered = K_blocks.reshape(B, S_ctx, d_head).unsqueeze(
            1
        )  # [B, 1, S_ctx, d_head]
        V_gathered = V_blocks.reshape(B, S_ctx, d_head).unsqueeze(
            1
        )  # [B, 1, S_ctx, d_head]

    # 7b. Place active K/V into the last S_tkg positions.
    #     The mask's last S_tkg rows (from gen_mask's active_mask) encode
    #     the causal pattern among active tokens; stale cache data in
    #     those positions is masked out.
    #
    #     When FP8 KV cache is enabled, round-trip active k/v through FP8
    #     quantization so the quantization noise matches the kernel path.
    #     k_scale/v_scale are reciprocal scales: quantize via (tensor * scale).

    # k, v: [B, kv_heads, S_tkg, d_head]
    if k_scale is not None:
        from vllm_neuron.utils.dtype_utils import FP8_CLAMP_MAX

        k_fp8 = (
            (k * k_scale).clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX).to(torch.float8_e4m3fn)
        )
        v_fp8 = (
            (v * v_scale).clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX).to(torch.float8_e4m3fn)
        )
        K_gathered[:, :, -S_tkg:, :] = k_fp8.to(X.dtype)
        V_gathered[:, :, -S_tkg:, :] = v_fp8.to(X.dtype)
        k = k_fp8
        v = v_fp8
    else:
        K_gathered[:, :, -S_tkg:, :] = k
        V_gathered[:, :, -S_tkg:, :] = v

    # 7c. GQA expansion: [B, kv_heads, S_ctx, d_head] → [B, q_heads, S_ctx, d_head]
    K_full = K_gathered.repeat_interleave(
        num_kv_groups, dim=1
    )  # [B, q_heads, S_ctx, d_head]
    V_full = V_gathered.repeat_interleave(
        num_kv_groups, dim=1
    )  # [B, q_heads, S_ctx, d_head]

    # 7d. Compute attention scores
    scale = softmax_scale if softmax_scale is not None else d_head**-0.5
    # q: [B, q_heads, S_tkg, d_head] × K^T: [B, q_heads, d_head, S_ctx]
    scores = (
        torch.matmul(q, K_full.transpose(-2, -1)) * scale
    )  # [B, q_heads, S_tkg, S_ctx]

    # 7e. Apply attention mask
    # attention_mask: [S_ctx, B, q_heads, S_tkg] → [B, q_heads, S_tkg, S_ctx]
    if dcp_size > 1:
        # DCP flat mask: already in sequential layout, just permute.
        mask = attention_mask.permute(1, 2, 3, 0)
    else:
        resized_bl = _mask_resize_block_len(block_len, B, q_heads, S_tkg, S_ctx)
        if resized_bl > 0:
            mask = (
                attention_mask.permute(1, 2, 3, 0)
                .reshape(-1, resized_bl, _MASK_P_MAX)
                .swapaxes(-1, -2)
                .reshape(B, q_heads, S_tkg, S_ctx)
            )
        else:
            mask = attention_mask.permute(1, 2, 3, 0)

    scores = scores.to(torch.float32)
    scores = scores.masked_fill(mask == 0, float("-inf"))

    # 7f. Attention sink
    if sink is not None:
        # sink shape: [q_heads, 1] (pre-gathered across attention DP at load time).
        sink_score = (
            sink.to(torch.float32).view(1, q_heads, 1, 1).expand(B, -1, S_tkg, -1)
        )
        scores = torch.cat([scores, sink_score], dim=-1)  # [B, q_heads, S_tkg, S_ctx+1]

    # 7g. Softmax → weighted sum
    # With DCP, extract partial LSE before softmax for cross-rank correction.
    if dcp_size > 1:
        partial_lse = torch.logsumexp(scores, dim=-1)  # [B, q_heads, S_tkg]

    attn_weights = torch.softmax(scores, dim=-1).to(q.dtype)

    # Strip the sink column so the matmul only touches real V positions
    if sink is not None:
        attn_weights = attn_weights[..., :-1]  # [B, q_heads, S_tkg, S_ctx]

    attn_out = torch.matmul(attn_weights, V_full)  # [B, q_heads, S_tkg, d_head]

    # ================================================================
    # Stage 7.5a: DCP LSE correction + ReduceScatter
    # Each rank computed attention over its local KV shard. Combine partials
    # across DCP ranks using LSE-weighted correction, then reduce-scatter
    # to distribute heads back to their owning ranks.
    # ================================================================
    if dcp_size > 1:
        all_lse = dcp_group.all_gather(
            partial_lse.contiguous(), dim=0
        )  # [dcp_size * B, q_heads, S_tkg]
        all_lse = all_lse.view(
            dcp_size, B, q_heads, S_tkg
        )  # [dcp_size, B, q_heads, S_tkg]
        global_lse = torch.logsumexp(all_lse, dim=0)  # [B, q_heads, S_tkg]

        correction = torch.exp(
            partial_lse - global_lse
        )  # [B, q_heads, S_tkg] (float32)
        attn_out = (attn_out.float() * correction.unsqueeze(-1)).to(attn_out.dtype)

        attn_out = dcp_group.reduce_scatter(
            attn_out.contiguous(), dim=1
        )  # [B, q_heads_local, S_tkg, d_head]
        q_heads = attn_out.shape[1]

    # ================================================================
    # Stage 7.5b: Dependent DP reverse all-to-all
    # ================================================================
    if attention_dp > 1:
        # attn_out: [B_local, DDP*q_heads_small, S_tkg, d_head]
        # → [DDP*B_local, q_heads_small, S_tkg, d_head]
        attn_out = all_to_all(
            attn_out, split_dim=1, concat_dim=0, group=attention_dp_group
        )

        # Restore B to DDP*B_local for O projection and return
        B = attn_out.shape[0]
        q_heads = attn_out.shape[1]

    # ================================================================
    # Stage 8: KV cache update
    # ================================================================
    if update_cache:
        # In-place KV cache update via index_put_.
        # kv_cache_update_idx: [B_attn, S_tkg] (caller-sliced when attention_dp > 1
        # — the NKI kernel expects the same B_attn = B/KVDP layout).
        # slot = block_idx * block_len + position_in_block.
        assert kv_cache_update_idx is not None, (
            "kv_cache_update_idx must be provided when update_cache=True"
        )
        # DCP non-owning ranks receive -1 sentinels in slot_mapping. The NKI
        # kernel handles this via oob_mode.skip in scatter DMA. The caller
        # casts to uint32 for the kernel (so -1 becomes 0xFFFFFFFF). Convert
        # back to signed and detect sentinels, then redirect them to the LAST
        # cache slot (block num_blocks_total-1, position block_len-1) rather
        # than slot 0. The cache is over-allocated, so the last slot is never a
        # real token's write target — writing there is a harmless no-op. This
        # matches the pre-#2306 model-level behavior, where a raw -1 index
        # negative-indexed to the last block. (Redirecting to slot 0 instead
        # corrupts the first token's KV.)
        max_slot = num_blocks_total * block_len
        slot_mapping = kv_cache_update_idx.to(torch.int64).reshape(-1)  # [B*S_tkg]
        is_sentinel = (slot_mapping < 0) | (slot_mapping >= max_slot)
        slot_mapping = torch.where(
            is_sentinel,
            torch.full_like(slot_mapping, max_slot - 1),
            slot_mapping,
        )

        block_indices = (slot_mapping // block_len).repeat(kv_heads)
        position_indices = (slot_mapping % block_len).repeat(kv_heads)
        head_indices = torch.arange(
            kv_heads, dtype=torch.long, device=K_cache.device
        ).repeat_interleave(slot_mapping.shape[0])

        # k, v: [B, kv_heads, S_tkg, d_head] → [kv_heads, B*S_tkg, d_head]
        k_flat = k.transpose(0, 1).reshape(-1, d_head).to(K_cache.dtype)
        v_flat = v.transpose(0, 1).reshape(-1, d_head).to(V_cache.dtype)

        # Sentinel slots were redirected to the harmless last cache slot above
        # (not slot 0), so their real K/V values write to a never-read target —
        # no need to zero them out (zeroing slot 0 corrupted the first token's
        # KV).
        if fp8_packed:
            # K_cache is the swizzled packed FP8 layout. Un-swizzle into a
            # standard [num_blocks, (kv_heads,) block_len, d_head] buffer,
            # scatter the new tokens with the same (block, head, position)
            # indices as the unpacked path, then re-swizzle and write back in
            # place. The scatter runs in the compute dtype to avoid fp8 fancy
            # indexing (unsupported on CPU); fp8→compute→fp8 is lossless since
            # k is already fp8-quantized.
            K_unpacked = _unswizzle_packed_k(K_cache).to(X.dtype)
            k_flat_c = k_flat.to(X.dtype)
            if V_cache.dim() == 4:
                K_unpacked.index_put_(
                    (block_indices, head_indices, position_indices), k_flat_c
                )
                V_cache.index_put_(
                    (block_indices, head_indices, position_indices), v_flat
                )
            else:
                K_unpacked.index_put_((block_indices, position_indices), k_flat_c)
                V_cache.index_put_((block_indices, position_indices), v_flat)
            K_cache.copy_(_swizzle_packed_k(K_unpacked.to(K_cache.dtype)))
        elif V_cache.dim() == 4:
            K_cache.index_put_((block_indices, head_indices, position_indices), k_flat)
            V_cache.index_put_((block_indices, head_indices, position_indices), v_flat)
        else:
            # 3D: [num_blocks, block_len, d_head] (kv_heads=1, pre-squeezed)
            K_cache.index_put_((block_indices, position_indices), k_flat)
            V_cache.index_put_((block_indices, position_indices), v_flat)
        K_out = V_out = None
    else:
        # No cache update: return new K/V tokens.
        # Use k.shape[0] (the local batch), NOT B: with attention DP, Stage 5
        # slices k/v to B_local, while Stage 7.5b restores B to DDP*B_local for
        # the output projection. K_new: [d_head, B_local*kv_heads, S_tkg]
        # (matches NKI kernel output).
        B_kv = k.shape[0]
        k_for_return = k.reshape(B_kv * kv_heads, S_tkg, d_head)
        K_out = k_for_return.permute(2, 0, 1).contiguous()
        # V_out: [B_kv, kv_heads, S_tkg, d_head]
        V_out = v

    # ================================================================
    # Stage 9: Output projection
    # ================================================================
    if W_out is not None:
        # attn_out: [B, q_heads, S_tkg, d_head] → [B*S_tkg, q_heads*d_head]
        attn_flat = attn_out.transpose(1, 2).reshape(B * S_tkg, q_heads * d_head)
        output = attn_flat @ W_out  # [B*S_tkg, H]
        if bias_out is not None:
            output = output + bias_out
    else:
        # Without projection: return [B, q_heads, d_head, S_tkg] (matches NKI kernel)
        output = attn_out.permute(0, 1, 3, 2).contiguous()

    # When update_cache=True the caches are written in place, so the new K/V
    # tokens are redundant — return only the attention output. The
    # update_cache=False path still returns the new tokens for the caller to
    # write into its cache.
    if update_cache:
        return output
    return output, K_out, V_out


# ---------------------------------------------------------------------------
# Kernel eligibility check
# ---------------------------------------------------------------------------


def _can_use_attention_block_kernel(
    X: Tensor,
    V_cache: Optional[Tensor],
) -> bool:
    """
    Check whether the NKI attention_block_tkg kernel can be used.

    Returns ``True`` when every kernel constraint is satisfied and the tensors
    reside on a NeuronCore device or CPU with the NKI simulator; ``False`` otherwise (→ PyTorch fallback).
    """
    if not can_run_kernel(X):
        return False

    B, S_tkg, H = X.shape

    # H must be a multiple of 128
    if H % _PMAX != 0:
        return False

    if V_cache.dim() == 3:
        kv_heads = 1
    else:
        _, kv_heads, _, _ = V_cache.shape

    d_head = V_cache.shape[-1]

    if d_head % 2 != 0:
        return False

    if kv_heads > 1:
        return False

    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def attention_decode(
    # -- input
    X: Tensor,
    X_hidden_dim_actual: Optional[int] = None,
    # -- rmsnorm X
    rmsnorm_X_enabled: bool = False,
    rmsnorm_X_eps: Optional[float] = None,
    rmsnorm_X_gamma: Optional[Tensor] = None,
    # -- qkv projections
    W_qkv: Tensor = None,
    bias_qkv: Optional[Tensor] = None,
    quantization_type_qkv: QuantizationType = QuantizationType.NONE,
    weight_dequant_scale_qkv: Optional[Tensor] = None,
    input_dequant_scale_qkv: Optional[Tensor] = None,
    # -- Q/K processing: pre-RoPE RMSNorm
    rmsnorm_QK_pre_rope_enabled: bool = False,
    rmsnorm_QK_pre_rope_eps: float = 1e-6,
    rmsnorm_QK_pre_rope_W_Q: Optional[Tensor] = None,
    rmsnorm_QK_pre_rope_W_K: Optional[Tensor] = None,
    # -- Q/K processing: RoPE
    cos: Optional[Tensor] = None,
    sin: Optional[Tensor] = None,
    rope_contiguous_layout: bool = True,
    # -- Q/K processing: post-RoPE RMSNorm
    rmsnorm_QK_post_rope_enabled: bool = False,
    rmsnorm_QK_post_rope_eps: float = 1e-6,
    rmsnorm_QK_post_rope_W_Q: Optional[Tensor] = None,
    rmsnorm_QK_post_rope_W_K: Optional[Tensor] = None,
    # -- attention
    K_cache_transposed: bool = False,
    active_blocks_table: Optional[Tensor] = None,
    K_cache: Tensor = None,
    V_cache: Tensor = None,
    attention_mask: Optional[Tensor] = None,
    sink: Optional[Tensor] = None,
    softmax_scale: Optional[float] = None,
    # -- in-kernel mask generation (fused path)
    pos_ids: Optional[Tensor] = None,
    swa_start_pos_ids: Optional[Tensor] = None,
    max_context_len: Optional[Tensor] = None,
    # -- KV cache update
    update_cache: bool = False,
    kv_cache_update_idx: Optional[Tensor] = None,
    k_scale: Optional[Tensor] = None,
    v_scale: Optional[Tensor] = None,
    # -- packed FP8 K cache layout
    fp8_packed: bool = False,
    # -- output projection
    W_out: Optional[Tensor] = None,
    bias_out: Optional[Tensor] = None,
    quantization_type_out: QuantizationType = QuantizationType.NONE,
    weight_dequant_scale_out: Optional[Tensor] = None,
    input_dequant_scale_out: Optional[Tensor] = None,
    transposed_out: bool = False,
    # -- output control
    out_in_sb: bool = False,
    skip_attention: bool = False,
    # -- STATIC_MX layout flag
    is_h_transposed_by_4: bool = False,
    # -- sbm control (optional)
    sbm_lower_bound: Optional[int] = None,
    sbm_upper_bound: Optional[int] = None,
    sbm_use_auto_alloc: bool = True,
    sbm_default_stack_alloc: bool = True,
    # -- Attention Dependent DP (decode-only Q/O sharding across DP)
    attention_dp: int = 1,
    attention_dp_group: Optional[ProcessGroup] = None,
    attention_dp_rank: int = 0,
    kv_needs_a2a: bool = False,
    dcp_size: int = 1,
    dcp_group=None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Fused Attention Block for Token Generation (TKG).

    Automatically selects between NKI kernel and PyTorch fallback.

    Dispatches to:
    - NKI kernel: When all constraints are satisfied and running on Neuron
    - PyTorch implementation: When constraints are violated or not on Neuron
      (block KV cache only; raises for unsupported features)

    Performs end-to-end attention: optional RMSNorm → QKV projection →
    optional pre-RoPE RMSNorm → optional RoPE → optional post-RoPE RMSNorm →
    attention → KV-cache update → optional output projection.

    Dimensions:
        B:       Batch size (≤ 16 recommended)
        S_tkg:   Number of new tokens (≤ 8)
        S_ctx:   Current KV-cache sequence length
        S_max:   Maximum KV-cache capacity
        H:       Hidden dimension (multiple of 128)
        d_head:  Head dimension (must be even, ≤ 128)
        q_heads: Number of query heads
        kv_heads: Number of key/value heads (1 for NKI kernel, ≥1 for torch fallback)

    Args:
        X:                  [B, S_tkg, H]  Input hidden states.
        X_hidden_dim_actual: Actual H if X is zero-padded (None = H).
        rmsnorm_X_enabled:  Apply RMSNorm to X before QKV.
        rmsnorm_X_eps:      RMSNorm epsilon (default 1e-3).
        rmsnorm_X_gamma:    [1, H]  RMSNorm gamma weights.
        W_qkv:              [H, d_head*(q_heads+2*kv_heads)]  QKV projection weights.
        bias_qkv:           [1, d_head*(q_heads+2*kv_heads)]  Optional QKV bias.
        quantization_type_qkv: NONE or STATIC.
        weight_dequant_scale_qkv: [PMAX, 1] weight scale (STATIC only).
        input_dequant_scale_qkv:  [PMAX, 1] input scale (STATIC only).
        rmsnorm_QK_pre_rope_enabled:  Pre-RoPE RMSNorm on Q/K.
        rmsnorm_QK_pre_rope_eps:      Epsilon.
        rmsnorm_QK_pre_rope_W_Q:      [1, d_head]  Q gamma.
        rmsnorm_QK_pre_rope_W_K:      [1, d_head]  K gamma.
        cos:                [d_head//2, B, S_tkg]  RoPE cos (None = skip).
        sin:                [d_head//2, B, S_tkg]  RoPE sin (None = skip).
        rope_contiguous_layout: True=contiguous halves, False=interleaved.
        rmsnorm_QK_post_rope_enabled: Post-RoPE RMSNorm on Q/K.
        rmsnorm_QK_post_rope_eps:     Epsilon.
        rmsnorm_QK_post_rope_W_Q:     [1, d_head]  Q gamma.
        rmsnorm_QK_post_rope_W_K:     [1, d_head]  K gamma.
        K_cache_transposed: K-cache layout flag (NKI kernel only).
        active_blocks_table: [B, num_blocks] int32 block indices (block KV).
        K_cache:            Key cache in HBM.
            - 3D: [num_blocks, block_len, d_head]
            - 4D: [num_blocks, kv_heads, block_len, d_head]
            - packed (fp8_packed=True): the swizzled FP8 layout
              [num_blocks, block_len//2, d_head, 2] (4D) or
              [num_blocks, kv_heads, block_len//2, d_head, 2] (5D). Two
              consecutive sequence positions are packed into the trailing
              size-2 dim so the kernel can bf16-reinterpret + DMA-transpose
              instead of using the slower PE-transpose FP8 path.
        V_cache:            Value cache in HBM. Same layout as the *unpacked*
            K_cache (3D/4D [num_blocks, (kv_heads,) block_len, d_head]) even
            when fp8_packed=True — only K is swizzled, V is never packed.
        attention_mask:     Attention mask. Shape depends on whether ``pos_ids``
                            is provided:
                              - ``pos_ids is None``: [S_ctx, B, q_heads, S_tkg]
                                full pre-generated mask (legacy path).
                              - ``pos_ids is not None``: [S_tkg, B, q_heads, S_tkg]
                                active-only causal mask. If ``None``, the
                                wrapper builds a default triu causal mask.
        sink:               [q_heads, 1]  Attention-sink scores (NKI kernel only).
        softmax_scale:      Custom scale (None → 1/√d_head).
        pos_ids:            [B, S_tkg] @ HBM, float32 absolute sequence position
                            of each active token. When provided, the kernel
                            generates the prior causal mask on-chip via
                            ``iota < pos_id`` and ``attention_mask`` carries only
                            the active-only portion. Block-KV only.
        swa_start_pos_ids:  [B, S_tkg] @ HBM, float32 per-query SWA window start
                            (inclusive). Requires ``pos_ids``. When provided,
                            the kernel generates a banded mask
                            ``start_pos <= kv_pos < pos_id``.
        update_cache:       Write new K/V tokens into the cache.
        kv_cache_update_idx: [B, 1] uint32  Cache write positions.
        k_scale:            [PMAX, 1] or [1, 1]  FP8 K quant scale (NKI kernel only).
        v_scale:            [PMAX, 1] or [1, 1]  FP8 V quant scale (NKI kernel only).
        fp8_packed:         When True, K_cache uses the swizzled packed FP8
                            layout (see K_cache above). Requires a float8_e4m3
                            K_cache and an even block_len. V_cache stays
                            unpacked. Typically paired with k_scale/v_scale.
        W_out:              [q_heads*d_head, H]  Output projection weights.
        bias_out:           [1, H]  Output projection bias.
        quantization_type_out: NONE or STATIC.
        weight_dequant_scale_out: [PMAX, 1]  (STATIC only).
        input_dequant_scale_out:  [PMAX, 1]  (STATIC only).
        transposed_out:     Transpose output layout (requires W_out).
        out_in_sb:          Return output in SBUF (NKI kernel only).
        skip_attention:     Skip attention (NKI kernel only).
        sbm_lower_bound:    Optional SBUF lower bound.
        sbm_upper_bound:    Optional SBUF upper bound.
        sbm_use_auto_alloc: Use auto-allocation for SBUF.
        sbm_default_stack_alloc: Default stack allocation.

    Returns:
        - update_cache=True: ``output`` only — the K/V caches are written in
          place (in-kernel scatter DMA, or index_put_ on the torch fallback),
          so no cache tensors are returned.
        - update_cache=False: ``(output, K_out, V_out)`` where K_out/V_out are
          the new K/V tokens for the caller to write into its cache.

        output: Attention output (shape depends on W_out / transposed_out).

    Torch Fallback Constraints (raises NotImplementedError if violated):
        - Requires block KV cache (active_blocks_table must be provided)
        - Quantization not supported
        - FP8 KV cache quantization not supported
        - out_in_sb not supported
        - skip_attention not supported
        - sink not supported
    """

    # The NKI kernel relies on int32 with -1 sentinels for inactive blocks
    if active_blocks_table is not None:
        assert active_blocks_table.dtype == torch.int32, (
            f"active_blocks_table must be int32 with -1 padding, got "
            f"{active_blocks_table.dtype}"
        )

    # Fused mask-gen path: when pos_ids is provided, the kernel generates the
    # prior causal/SWA mask on-chip and consumes only an active-only mask
    # of shape [S_tkg, B, q_heads, S_tkg]. If the caller doesn't pass
    # attention_mask, build the default triu causal active mask here so model
    # code stays free of mask construction.
    if swa_start_pos_ids is not None and pos_ids is None:
        raise ValueError("swa_start_pos_ids requires pos_ids to also be provided")
    if pos_ids is not None:
        if active_blocks_table is None:
            raise ValueError(
                "pos_ids (fused mask-gen) requires block KV cache "
                "(active_blocks_table must be provided)"
            )
        B_x, S_tkg_x, _ = X.shape
        kv_heads = 1 if V_cache.dim() == 3 else V_cache.shape[1]
        d_head_local = V_cache.shape[-1]
        q_heads_local = W_qkv.shape[1] // d_head_local - 2 * kv_heads
        # Derive block_len from V_cache: it is never packed, so its block_len
        # dim is the true logical block length (the packed K_cache stores
        # block_len//2 in its swizzled layout, so K_cache is unreliable here).
        block_len_local = V_cache.shape[1] if V_cache.dim() == 3 else V_cache.shape[2]
        S_ctx_local = active_blocks_table.shape[1] * block_len_local
        # FP8 KV cache: the in-kernel mask gen path produces a small but
        # consistent accuracy regression (BC ~0.988 vs baseline ~0.999) that
        # compounds across decode steps. Until the kernel is fixed, fall back
        # to the external mask path for FP8 KV. BF16/FP16 KV is unaffected
        # and keeps the perf win.
        if k_scale is not None:
            from vllm_neuron.functional.attention.attention_decode_mask import (
                gen_attention_decode_mask,
            )

            attention_mask = gen_attention_decode_mask(
                pos_ids=pos_ids.reshape(1, B_x * S_tkg_x).to(torch.float32),
                bs=B_x,
                q_head=q_heads_local,
                s_active=S_tkg_x,
                s_prior=S_ctx_local,
                start_pos=swa_start_pos_ids.reshape(1, B_x * S_tkg_x).to(torch.float32)
                if swa_start_pos_ids is not None
                else None,
                block_len=block_len_local,
            )
            pos_ids = None
            swa_start_pos_ids = None
        elif attention_mask is None:
            attention_mask = _build_default_active_mask(
                B=B_x,
                S_tkg=S_tkg_x,
                q_heads=q_heads_local,
                device=X.device,
            )
        else:
            assert attention_mask.shape[0] == S_tkg_x, (
                f"With pos_ids, attention_mask dim 0 must equal S_tkg "
                f"({S_tkg_x}); got {tuple(attention_mask.shape)}"
            )
            assert attention_mask.shape[3] == S_tkg_x, (
                f"With pos_ids, attention_mask dim 3 must equal S_tkg "
                f"({S_tkg_x}); got {tuple(attention_mask.shape)}"
            )

    can_use_kernel = _can_use_attention_block_kernel(
        X=X,
        V_cache=V_cache,
    )

    # Dependent DP / DCP: use torch fallback for now (NKI kernel integration is future work)
    if attention_dp > 1 or dcp_size > 1:
        can_use_kernel = False

    if can_use_kernel:
        # attention_block_tkg uses LNC-2 sharding → grid size 2
        wrapped = wrap_nki(_torch_compatible_attention_block_tkg_kernel)
        kernel_out = wrapped[2](
            X,
            X_hidden_dim_actual=X_hidden_dim_actual,
            rmsnorm_X_enabled=rmsnorm_X_enabled,
            rmsnorm_X_eps=rmsnorm_X_eps,
            rmsnorm_X_gamma=rmsnorm_X_gamma,
            W_qkv=W_qkv,
            bias_qkv=bias_qkv,
            quantization_type_qkv=quantization_type_qkv,
            weight_dequant_scale_qkv=weight_dequant_scale_qkv,
            input_dequant_scale_qkv=input_dequant_scale_qkv,
            rmsnorm_QK_pre_rope_enabled=rmsnorm_QK_pre_rope_enabled,
            rmsnorm_QK_pre_rope_eps=rmsnorm_QK_pre_rope_eps,
            rmsnorm_QK_pre_rope_W_Q=rmsnorm_QK_pre_rope_W_Q,
            rmsnorm_QK_pre_rope_W_K=rmsnorm_QK_pre_rope_W_K,
            cos=cos,
            sin=sin,
            rope_contiguous_layout=rope_contiguous_layout,
            rmsnorm_QK_post_rope_enabled=rmsnorm_QK_post_rope_enabled,
            rmsnorm_QK_post_rope_eps=rmsnorm_QK_post_rope_eps,
            rmsnorm_QK_post_rope_W_Q=rmsnorm_QK_post_rope_W_Q,
            rmsnorm_QK_post_rope_W_K=rmsnorm_QK_post_rope_W_K,
            K_cache_transposed=K_cache_transposed,
            active_blocks_table=active_blocks_table,
            K_cache=K_cache,
            V_cache=V_cache,
            attention_mask=attention_mask,
            sink=sink,
            softmax_scale=softmax_scale,
            update_cache=update_cache,
            kv_cache_update_idx=kv_cache_update_idx,
            k_scale=k_scale,
            v_scale=v_scale,
            fp8_packed=fp8_packed,
            W_out=W_out,
            bias_out=bias_out,
            quantization_type_out=quantization_type_out,
            weight_dequant_scale_out=weight_dequant_scale_out,
            input_dequant_scale_out=input_dequant_scale_out,
            transposed_out=transposed_out,
            out_in_sb=out_in_sb,
            skip_attention=skip_attention,
            is_h_transposed_by_4=is_h_transposed_by_4,
            sbm_lower_bound=sbm_lower_bound,
            sbm_upper_bound=sbm_upper_bound,
            sbm_use_auto_alloc=sbm_use_auto_alloc,
            sbm_default_stack_alloc=sbm_default_stack_alloc,
            KVDP=attention_dp,
            pos_ids=pos_ids,
            swa_start_pos_ids=swa_start_pos_ids,
            max_context_len=max_context_len,
        )
        # The NKI kernel always returns (output, K, V). When update_cache=True
        # the K/V outputs are the in-place-written caches (kept live so the FX
        # aliasing pass threads the write back to the caller's cache); they're
        # redundant to return, so expose only the attention output to match the
        # torch fallback's update_cache=True contract.
        if update_cache:
            return kernel_out[0]
        return kernel_out
    else:
        # Validate torch-unsupported features
        if quantization_type_qkv != QuantizationType.NONE:
            raise NotImplementedError(
                "Attention block torch fallback does not support QKV quantization"
            )
        if quantization_type_out != QuantizationType.NONE:
            raise NotImplementedError(
                "Attention block torch fallback does not support output quantization"
            )

        if out_in_sb:
            raise NotImplementedError(
                "Attention block torch fallback does not support out_in_sb=True"
            )
        if skip_attention:
            raise NotImplementedError(
                "Attention block torch fallback does not support skip_attention=True"
            )

        # Torch fallback consumes a full [S_ctx, B, q_heads, S_tkg] mask. When
        # pos_ids is provided (fused mask-gen path), build the equivalent full
        # mask via gen_attention_decode_mask so kernel and torch paths produce
        # identical numerics.
        if pos_ids is not None:
            from vllm_neuron.functional.attention.attention_decode_mask import (
                gen_attention_decode_mask,
            )

            attention_mask = gen_attention_decode_mask(
                pos_ids=pos_ids.reshape(1, B_x * S_tkg_x).to(torch.float32),
                bs=B_x,
                q_head=q_heads_local,
                s_active=S_tkg_x,
                s_prior=S_ctx_local,
                start_pos=swa_start_pos_ids.reshape(1, B_x * S_tkg_x).to(torch.float32)
                if swa_start_pos_ids is not None
                else None,
                block_len=block_len_local,
            )

        return _torch_attention_decode_impl(
            X=X,
            X_hidden_dim_actual=X_hidden_dim_actual,
            rmsnorm_X_enabled=rmsnorm_X_enabled,
            rmsnorm_X_eps=rmsnorm_X_eps,
            rmsnorm_X_gamma=rmsnorm_X_gamma,
            W_qkv=W_qkv,
            bias_qkv=bias_qkv,
            rmsnorm_QK_pre_rope_enabled=rmsnorm_QK_pre_rope_enabled,
            rmsnorm_QK_pre_rope_eps=rmsnorm_QK_pre_rope_eps,
            rmsnorm_QK_pre_rope_W_Q=rmsnorm_QK_pre_rope_W_Q,
            rmsnorm_QK_pre_rope_W_K=rmsnorm_QK_pre_rope_W_K,
            cos=cos,
            sin=sin,
            rope_contiguous_layout=rope_contiguous_layout,
            rmsnorm_QK_post_rope_enabled=rmsnorm_QK_post_rope_enabled,
            rmsnorm_QK_post_rope_eps=rmsnorm_QK_post_rope_eps,
            rmsnorm_QK_post_rope_W_Q=rmsnorm_QK_post_rope_W_Q,
            rmsnorm_QK_post_rope_W_K=rmsnorm_QK_post_rope_W_K,
            active_blocks_table=active_blocks_table,
            K_cache=K_cache,
            V_cache=V_cache,
            attention_mask=attention_mask,
            sink=sink,
            softmax_scale=softmax_scale,
            update_cache=update_cache,
            kv_cache_update_idx=kv_cache_update_idx,
            k_scale=k_scale,
            v_scale=v_scale,
            fp8_packed=fp8_packed,
            W_out=W_out,
            bias_out=bias_out,
            attention_dp=attention_dp,
            attention_dp_group=attention_dp_group,
            attention_dp_rank=attention_dp_rank,
            kv_needs_a2a=kv_needs_a2a,
            dcp_size=dcp_size,
            dcp_group=dcp_group,
        )
