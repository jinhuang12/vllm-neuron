# SPDX-License-Identifier: Apache-2.0
"""Fused RMSNorm and projection with fp32 sums: ``rmsnorm(hidden) * gamma @ W``.

One NKI launch per call, on the decode or the prefill kernel. The split is the
library's: at most 96 tokens per sequence and 128 rows in all take the decode kernel.
Both kernels sum in fp32 and round the output to bf16 once; the prefill kernel also
splits the normed activations into two bf16 passes. There is no torch path: an input
outside the kernels' limits raises.
"""

import nki
import torch
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.attention.qkv_fp32acc_cte import qkv_fp32acc_cte
from vllm_neuron.functional.attention.qkv_fp32acc_tkg import qkv_fp32acc_tkg
from vllm_neuron.utils.hardware_config import get_logical_nc_config

_P = 128
_DECODE_MAX_SEQ = 96
_MAX_COLUMNS = 3072


@nki.jit
def qkv_fp32acc_kernel(hidden, qkv_weights, norm_weights, eps: float):
    """``[B, S, I]`` bf16 on the decode kernel for short inputs, else on the prefill kernel."""
    batch, seq, _ = hidden.shape
    if seq <= _DECODE_MAX_SEQ and batch * seq <= _P:
        return qkv_fp32acc_tkg(hidden, qkv_weights, norm_weights, eps)
    return qkv_fp32acc_cte(hidden, qkv_weights, norm_weights, eps)


_KERNEL = wrap_nki(qkv_fp32acc_kernel)


def qkv_proj_fp32acc(hidden: torch.Tensor, qkv_weights: torch.Tensor,
                     norm_weights: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm of ``hidden`` with gain ``norm_weights``, projected by ``qkv_weights``.

    Args:
        hidden: ``[B, S, H]`` bf16.
        qkv_weights: ``[H, I]`` bf16, ``I <= 3072``.
        norm_weights: ``[H]`` or ``[1, H]`` bf16 norm gain.
        eps: the norm epsilon.

    Returns:
        ``[B, S, I]`` bf16.

    Raises:
        ValueError: an input outside the kernels' contract.
    """
    lnc = get_logical_nc_config()
    if hidden.dim() != 3 or qkv_weights.dim() != 2:
        raise ValueError(f"hidden must be [B, S, H] and weights [H, I], got {tuple(hidden.shape)} "
                         f"and {tuple(qkv_weights.shape)}")
    size = hidden.shape[-1]
    if (hidden.dtype, qkv_weights.dtype, norm_weights.dtype) != (torch.bfloat16,) * 3:
        raise ValueError(f"hidden, weights and norm gain must be bf16, got {hidden.dtype}, "
                         f"{qkv_weights.dtype} and {norm_weights.dtype}")
    if qkv_weights.shape[0] != size or norm_weights.numel() != size:
        raise ValueError(f"H {size} must match weights {tuple(qkv_weights.shape)} and norm gain "
                         f"{tuple(norm_weights.shape)}")
    if size % (_P * lnc) or qkv_weights.shape[1] > _MAX_COLUMNS:
        raise ValueError(f"H {size} must split into 128-row tiles over {lnc} programs, and I "
                         f"{qkv_weights.shape[1]} must be at most {_MAX_COLUMNS}")
    return _KERNEL[lnc](hidden=hidden.contiguous(), qkv_weights=qkv_weights.contiguous(),
                        norm_weights=norm_weights.reshape(1, size).contiguous(), eps=float(eps))
