# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 DSA indexer using Neuron-native PyTorch operations."""

from __future__ import annotations

from dataclasses import dataclass

import nki
import nki.isa as nisa
import nki.language as nl
import torch
import torch.nn.functional as F
from torch import nn

from vllm_neuron.functional.topk import topk as neuron_topk
from vllm_neuron.nki.nki_hop import can_run_kernel, wrap_nki

from .block_fp8 import BlockFP8Linear
from .cache import (
    INDEXER_CACHE_BYTES,
    INDEXER_KEY_DIM,
    MAIN_INDEXER_LAYER_INDICES,
    MAIN_LAYER_COUNT,
)


_DSA_SCORE_TILE_SIZE = 256


def _kernel_assert(condition: bool, error_text: str) -> None:
    assert condition, (
        "[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: " + error_text
    )


@nki.jit
def _pack_ue8m0_nki(keys, eps=1.0e-4):
    """Gen3 NKI pack with arithmetic correction for E4M3FN above 256."""

    _kernel_assert(keys.shape[-1] == 128, "key width must be 128")
    row_count = 1
    for dimension in keys.shape[:-1]:
        row_count *= dimension
    keys_2d = keys.reshape((row_count, 128))
    packed = nl.ndarray(keys.shape[:-1] + (132,), dtype=nl.uint8, buffer=nl.shared_hbm)
    packed_2d = packed.reshape((row_count, 132))
    for tile_index in nl.affine_range((row_count + 127) // 128):
        row_start = tile_index * 128
        row_size = min(128, row_count - row_start)
        values = nl.ndarray((row_size, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=values, src=keys_2d[row_start : row_start + row_size, 0:128])
        absolute = nl.ndarray((row_size, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=absolute, data=values, op=nl.abs)
        row_max = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=row_max, data=absolute, op=nl.maximum, axis=1)
        nisa.tensor_scalar(dst=row_max, data=row_max, op0=nl.maximum, operand0=eps)

        row_max_bytes = row_max.view(nl.uint8)
        byte0 = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
        byte1 = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
        byte2 = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
        byte3 = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=byte0, src=row_max_bytes[0:row_size, 0:1])
        nisa.tensor_copy(dst=byte1, src=row_max_bytes[0:row_size, 1:2])
        nisa.tensor_copy(dst=byte2, src=row_max_bytes[0:row_size, 2:3])
        nisa.tensor_copy(dst=byte3, src=row_max_bytes[0:row_size, 3:4])
        exponent_high_bit = nl.floor(
            nl.multiply(byte2, 1.0 / 128.0, dtype=nl.float32),
            dtype=nl.float32,
        )
        exponent_high_value = nl.ndarray(
            (row_size, 1), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_scalar(
            dst=exponent_high_value,
            data=exponent_high_bit,
            op0=nl.multiply,
            operand0=128.0,
        )
        mantissa_byte2 = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=mantissa_byte2,
            data1=byte2,
            data2=exponent_high_value,
            op=nl.subtract,
        )
        mantissa_high = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
        mantissa_equal = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=mantissa_high,
            data=mantissa_byte2,
            op0=nl.greater,
            operand0=96.0,
        )
        nisa.tensor_scalar(
            dst=mantissa_equal,
            data=mantissa_byte2,
            op0=nl.equal,
            operand0=96.0,
        )
        lower_mantissa = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=lower_mantissa,
            data1=byte0,
            data2=byte1,
            op=nl.add,
        )
        lower_nonzero = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=lower_nonzero,
            data=lower_mantissa,
            op0=nl.greater,
            operand0=0.0,
        )
        equal_and_lower = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=equal_and_lower,
            data1=mantissa_equal,
            data2=lower_nonzero,
            op=nl.multiply,
        )
        mantissa_increment = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=mantissa_increment,
            data1=mantissa_high,
            data2=equal_and_lower,
            op=nl.add,
        )
        biased_exponent_base = nl.ndarray(
            (row_size, 1), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_scalar(
            dst=biased_exponent_base,
            data=byte3,
            op0=nl.multiply,
            operand0=2.0,
        )
        biased_exponent = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=biased_exponent,
            data1=biased_exponent_base,
            data2=exponent_high_bit,
            op=nl.add,
        )
        biased_base = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=biased_base,
            data=biased_exponent,
            op0=nl.subtract,
            operand0=8.0,
        )
        biased = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=biased,
            data1=biased_base,
            data2=mantissa_increment,
            op=nl.add,
        )
        biased_half_floor = nl.floor(
            nl.multiply(biased, 0.5, dtype=nl.float32), dtype=nl.float32
        )
        biased_even = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=biased_even,
            data=biased_half_floor,
            op0=nl.multiply,
            operand0=2.0,
        )
        biased_parity = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=biased_parity,
            data1=biased,
            data2=biased_even,
            op=nl.subtract,
        )
        scale_bytes = nl.ndarray((row_size, 4), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.memset(dst=scale_bytes, value=0)
        nisa.tensor_scalar(
            dst=scale_bytes[0:row_size, 2:3],
            data=biased_parity,
            op0=nl.multiply,
            operand0=128.0,
        )
        nisa.activation(
            dst=scale_bytes[0:row_size, 3:4],
            data=biased_half_floor,
            op=nl.copy,
        )
        scale = scale_bytes.view(nl.float32)
        inverse_scale = nl.ndarray((row_size, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=inverse_scale, data=scale)
        normalized = nl.ndarray((row_size, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=normalized,
            data=values,
            op0=nl.multiply,
            operand0=inverse_scale,
            op1=nl.minimum,
            operand1=448.0,
        )
        nisa.tensor_scalar(
            dst=normalized,
            data=normalized,
            op0=nl.maximum,
            operand0=-448.0,
        )

        hardware_fp8 = nl.ndarray(
            (row_size, 128), dtype=nl.float8_e4m3fn, buffer=nl.sbuf
        )
        nisa.activation(dst=hardware_fp8, data=normalized, op=nl.copy)
        hardware_bytes = hardware_fp8.view(nl.uint8)
        magnitude = nl.ndarray((row_size, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=magnitude, data=normalized, op=nl.abs)
        high_units = nl.ndarray((row_size, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=high_units,
            data=magnitude,
            op0=nl.multiply,
            operand0=1.0 / 32.0,
        )
        high_floor = nl.floor(high_units, dtype=nl.float32)
        high_fraction = nl.ndarray((row_size, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=high_fraction,
            data1=high_units,
            data2=high_floor,
            op=nl.subtract,
        )
        half_floor = nl.floor(
            nl.multiply(high_floor, 0.5, dtype=nl.float32), dtype=nl.float32
        )
        twice_half_floor = nl.ndarray((row_size, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=twice_half_floor,
            data=half_floor,
            op0=nl.multiply,
            operand0=2.0,
        )
        high_odd = nl.ndarray((row_size, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=high_odd,
            data1=high_floor,
            data2=twice_half_floor,
            op=nl.subtract,
        )
        fraction_gt_half = nl.ndarray((row_size, 128), dtype=nl.float32, buffer=nl.sbuf)
        fraction_eq_half = nl.ndarray((row_size, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=fraction_gt_half,
            data=high_fraction,
            op0=nl.greater,
            operand0=0.5,
        )
        nisa.tensor_scalar(
            dst=fraction_eq_half,
            data=high_fraction,
            op0=nl.equal,
            operand0=0.5,
        )
        tie_and_odd = nl.ndarray((row_size, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=tie_and_odd,
            data1=fraction_eq_half,
            data2=high_odd,
            op=nl.multiply,
        )
        round_up = nl.ndarray((row_size, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=round_up,
            data1=fraction_gt_half,
            data2=tie_and_odd,
            op=nl.add,
        )
        high_index = nl.ndarray((row_size, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=high_index,
            data1=high_floor,
            data2=round_up,
            op=nl.add,
        )
        nisa.tensor_scalar(
            dst=high_index,
            data=high_index,
            op0=nl.minimum,
            operand0=14.0,
        )
        sign = nl.ndarray((row_size, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=sign,
            data=normalized,
            op0=nl.less,
            operand0=0.0,
            op1=nl.multiply,
            operand1=128.0,
        )
        high_code = nl.ndarray((row_size, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=high_code, data1=high_index, data2=sign, op=nl.add)
        nisa.tensor_scalar(dst=high_code, data=high_code, op0=nl.add, operand0=112.0)
        high_mask = nl.ndarray((row_size, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=high_mask, data=magnitude, op0=nl.greater, operand0=272.0
        )
        base = nl.ndarray((row_size, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=base, src=hardware_bytes)
        delta = nl.ndarray((row_size, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=delta, data1=high_code, data2=base, op=nl.subtract)
        nisa.tensor_tensor(dst=delta, data1=delta, data2=high_mask, op=nl.multiply)
        encoded = nl.ndarray((row_size, 128), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=encoded, data1=base, data2=delta, op=nl.add)
        nisa.dma_copy(
            dst=packed_2d[row_start : row_start + row_size, 0:128], src=encoded
        )
        nisa.dma_copy(
            dst=packed_2d[row_start : row_start + row_size, 128:132],
            src=scale_bytes,
        )
    return packed


@nki.jit
def _unpack_ue8m0_nki(packed):
    """Gen3 NKI decode for the exact 132-byte cache row."""

    _kernel_assert(packed.shape[-1] == 132, "packed width must be 132")
    row_count = 1
    for dimension in packed.shape[:-1]:
        row_count *= dimension
    packed_2d = packed.reshape((row_count, 132))
    output = nl.ndarray(
        packed.shape[:-1] + (128,), dtype=nl.float32, buffer=nl.shared_hbm
    )
    output_2d = output.reshape((row_count, 128))
    for tile_index in nl.affine_range((row_count + 127) // 128):
        row_start = tile_index * 128
        row_size = min(128, row_count - row_start)
        packed_tile = nl.ndarray((row_size, 132), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=packed_tile,
            src=packed_2d[row_start : row_start + row_size, 0:132],
        )
        encoded = nl.ndarray((row_size, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=encoded, src=packed_tile[0:row_size, 0:128])
        sign_mask = nl.ndarray((row_size, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=sign_mask,
            data=encoded,
            op0=nl.greater,
            operand0=127.0,
        )
        sign_byte = nl.ndarray((row_size, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=sign_byte,
            data=sign_mask,
            op0=nl.multiply,
            operand0=128.0,
        )
        payload = nl.ndarray((row_size, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=payload,
            data1=encoded,
            data2=sign_byte,
            op=nl.subtract,
        )
        safe_payload = nl.ndarray((row_size, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=safe_payload,
            data=payload,
            op0=nl.minimum,
            operand0=119.0,
        )
        safe_encoded = nl.ndarray((row_size, 128), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=safe_encoded,
            data1=safe_payload,
            data2=sign_byte,
            op=nl.add,
        )
        safe_values = safe_encoded.view(nl.float8_e4m3fn)
        base = nl.ndarray((row_size, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=base, data=safe_values, op=nl.copy)
        high_mask = nl.ndarray((row_size, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=high_mask,
            data=payload,
            op0=nl.greater,
            operand0=119.0,
        )
        high_index = nl.ndarray((row_size, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=high_index,
            data=payload,
            op0=nl.subtract,
            operand0=112.0,
        )
        high_magnitude = nl.ndarray((row_size, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=high_magnitude,
            data=high_index,
            op0=nl.multiply,
            operand0=32.0,
        )
        sign_value = nl.ndarray((row_size, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=sign_value,
            data=sign_mask,
            op0=nl.multiply,
            operand0=-2.0,
            op1=nl.add,
            operand1=1.0,
        )
        high_value = nl.ndarray((row_size, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=high_value,
            data1=high_magnitude,
            data2=sign_value,
            op=nl.multiply,
        )
        high_delta = nl.ndarray((row_size, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=high_delta,
            data1=high_value,
            data2=base,
            op=nl.subtract,
        )
        nisa.tensor_tensor(
            dst=high_delta,
            data1=high_delta,
            data2=high_mask,
            op=nl.multiply,
        )
        values = nl.ndarray((row_size, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=values,
            data1=base,
            data2=high_delta,
            op=nl.add,
        )
        scale = packed_tile[0:row_size, 128:132].view(nl.float32)
        restored = nl.ndarray((row_size, 128), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=restored,
            data=values,
            op0=nl.multiply,
            operand0=scale,
            engine=nisa.vector_engine,
        )
        nisa.dma_copy(
            dst=output_2d[row_start : row_start + row_size, 0:128], src=restored
        )
    return output


def rotary_cos_sin(
    positions: torch.Tensor,
    dim: int,
    *,
    theta: float = 8_000_000.0,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build interleaved rotary tables for the pinned default RoPE."""

    if dim <= 0 or dim % 2:
        raise ValueError("rotary dimension must be a positive even number")
    inv_freq = 1.0 / (
        theta
        ** (torch.arange(0, dim, 2, dtype=torch.float32, device=positions.device) / dim)
    )
    angle = positions.to(torch.float32).unsqueeze(-1) * inv_freq
    return angle.cos().to(dtype), angle.sin().to(dtype)


def apply_interleaved_rope(
    values: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply the non-NeoX, pair-interleaved rotary convention used upstream."""

    if values.shape[-1] % 2:
        raise ValueError("rotary input width must be even")
    expected = values.shape[:-1] + (values.shape[-1] // 2,)
    while cos.ndim < len(expected):
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
    if cos.shape != expected and not all(
        left == right or left == 1
        for left, right in zip(cos.shape, expected, strict=True)
    ):
        raise ValueError(
            f"rotary table shape {cos.shape} cannot broadcast to {expected}"
        )
    even = values[..., 0::2]
    odd = values[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2)


def quantize_ue8m0_fp8(
    values: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pinned per-row e4m3 quantization with a UE8M0-compatible scale.

    The upstream indexer uses a power-of-two FP32 scale.  It computes the raw
    absmax/448 scale and rounds upward with ``exp2(ceil(log2(scale)))``.
    """

    if values.shape[-1] <= 0:
        raise ValueError("quantized values must have a positive final dimension")
    quantized, scale = _quantize_ue8m0_values(values, eps=eps)
    return quantized.to(torch.float8_e4m3fn), scale


def _quantize_ue8m0_values(
    values: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact E4M3FN values in FP32 plus power-of-two row scales."""

    fp8_max = 448.0
    absmax = values.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
    raw_scale = absmax / fp8_max
    scale = torch.exp2(torch.ceil(torch.log2(raw_scale)))
    normalized = (values / scale).clamp(-fp8_max, fp8_max)
    magnitude = normalized.abs()
    normal = magnitude >= 2.0**-6
    exponent = torch.floor(torch.log2(magnitude.clamp_min(2.0**-9))).clamp(-6.0, 8.0)
    quantum = torch.where(
        normal,
        torch.exp2(exponent - 3.0),
        torch.full_like(magnitude, 2.0**-9),
    )
    quantized = torch.sign(normalized) * torch.round(magnitude / quantum) * quantum
    return quantized, scale.to(torch.float32)


def _encode_e4m3fn(values: torch.Tensor) -> torch.Tensor:
    """Encode finite E4M3FN values as uint8 without a dtype view."""

    magnitude = values.abs()
    normal = magnitude >= 2.0**-6
    exponent = torch.floor(torch.log2(magnitude.clamp_min(2.0**-9))).clamp(-6.0, 8.0)
    exponent_field = torch.where(normal, exponent + 7.0, torch.zeros_like(exponent))
    normal_mantissa = torch.round(magnitude / torch.exp2(exponent - 3.0)) - 8.0
    subnormal_mantissa = torch.round(magnitude / (2.0**-9))
    mantissa = torch.where(normal, normal_mantissa, subnormal_mantissa)
    sign = (values < 0).to(torch.float32) * 128.0
    return (sign + exponent_field * 8.0 + mantissa).to(torch.uint8)


def _decode_e4m3fn(value_bytes: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    """Decode finite E4M3FN bytes without a dtype view."""

    encoded = value_bytes.to(torch.float32)
    sign = torch.where(encoded >= 128.0, -1.0, 1.0)
    payload = torch.remainder(encoded, 128.0)
    exponent_field = torch.floor(payload / 8.0)
    mantissa = torch.remainder(payload, 8.0)
    normal = exponent_field > 0.0
    magnitude = torch.where(
        normal,
        (8.0 + mantissa) * torch.exp2(exponent_field - 10.0),
        mantissa * (2.0**-9),
    )
    return (sign * magnitude).to(dtype)


def pack_indexer_keys(keys: torch.Tensor) -> torch.Tensor:
    """Pack 128 FP8 key bytes followed by one FP32 inverse-scale value."""

    if keys.shape[-1] != INDEXER_KEY_DIM:
        raise ValueError(f"indexer keys must have width {INDEXER_KEY_DIM}")
    if can_run_kernel(keys):
        return wrap_nki(_pack_ue8m0_nki)[1](keys, 1.0e-4)
    quantized, scale = _quantize_ue8m0_values(keys, eps=1.0e-4)
    value_bytes = _encode_e4m3fn(quantized)
    scale_exponent = torch.round(torch.log2(scale)) + 127.0
    zero_byte = torch.zeros_like(scale_exponent, dtype=torch.uint8)
    scale_byte_2 = (torch.remainder(scale_exponent, 2.0) * 128.0).to(torch.uint8)
    scale_byte_3 = torch.floor(scale_exponent / 2.0).to(torch.uint8)
    scale_bytes = torch.cat((zero_byte, zero_byte, scale_byte_2, scale_byte_3), dim=-1)
    packed = torch.cat((value_bytes, scale_bytes), dim=-1)
    if packed.shape[-1] != INDEXER_CACHE_BYTES:
        raise AssertionError("packed indexer cache has the wrong width")
    return packed


def unpack_indexer_keys(packed: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize the packed 132-byte indexer cache representation."""

    if packed.dtype is not torch.uint8 or packed.shape[-1] != INDEXER_CACHE_BYTES:
        raise ValueError(
            f"packed indexer keys must be uint8 with width {INDEXER_CACHE_BYTES}"
        )
    if can_run_kernel(packed):
        return wrap_nki(_unpack_ue8m0_nki)[1](packed).to(dtype)
    values = _decode_e4m3fn(packed[..., :INDEXER_KEY_DIM], dtype=dtype)
    scale_bytes = packed[..., INDEXER_KEY_DIM:].to(torch.float32)
    scale_exponent = scale_bytes[..., 3:4] * 2.0 + torch.floor(
        scale_bytes[..., 2:3] / 128.0
    )
    scale = torch.exp2(scale_exponent - 127.0).to(dtype)
    return values * scale


def causal_topk_indices(
    queries: torch.Tensor,
    keys: torch.Tensor,
    head_weights: torch.Tensor,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    *,
    topk: int = 2048,
) -> torch.Tensor:
    """Return causal DSA positions, padded with -1 when the prefix is short.

    Scores match the pinned upstream path: Q and K are independently quantized
    to e4m3 using UE8M0 power-of-two scales, each dequantized per-head dot
    product is scaled and passed through ReLU, and learned head weights reduce
    the per-head scores.
    """

    if queries.ndim != 4:
        raise ValueError("queries must have shape [batch, queries, heads, dim]")
    if keys.ndim != 3:
        raise ValueError("keys must have shape [batch, keys, dim]")
    if queries.shape[0] != keys.shape[0] or queries.shape[-1] != keys.shape[-1]:
        raise ValueError("query/key batch and indexer dimensions must match")
    if head_weights.shape != queries.shape[:-1]:
        raise ValueError("head_weights must have shape [batch, queries, heads]")
    if topk <= 0:
        raise ValueError("topk must be positive")

    if query_positions.ndim == 1:
        query_positions = query_positions.unsqueeze(0).expand(queries.shape[0], -1)
    if key_positions.ndim == 1:
        key_positions = key_positions.unsqueeze(0).expand(keys.shape[0], -1)

    # GLM-5.2 selects up to 2048 keys. When the static key bucket is no wider
    # than topk, every causal key is selected, so score ranking cannot change
    # the attention result. Build the selection from absolute positions. This
    # also handles segmented prefill, whose query positions do not restart at
    # zero for each segment, without materializing the DSA score graph.
    if keys.shape[1] <= topk:
        key_indices = torch.arange(
            keys.shape[1], dtype=torch.int64, device=keys.device
        ).view(1, 1, -1)
        causal = key_positions.unsqueeze(1) <= query_positions.unsqueeze(-1)
        selected = torch.where(causal, key_indices, -torch.ones_like(key_indices))
        if keys.shape[1] < topk:
            selected = F.pad(selected, (0, topk - keys.shape[1]), value=-1)
        return selected

    if can_run_kernel(queries):
        q_quant = wrap_nki(_unpack_ue8m0_nki)[1](
            wrap_nki(_pack_ue8m0_nki)[1](queries, 1.0e-10)
        )
        k_quant = wrap_nki(_unpack_ue8m0_nki)[1](
            wrap_nki(_pack_ue8m0_nki)[1](keys, 1.0e-4)
        )
        q_scale = torch.ones_like(queries[..., :1], dtype=torch.float32)
        k_scale = torch.ones_like(keys[..., :1], dtype=torch.float32)
    else:
        q_quant, q_scale = _quantize_ue8m0_values(queries, eps=1.0e-10)
        k_quant, k_scale = _quantize_ue8m0_values(keys, eps=1.0e-4)
    batch, query_count, head_count, head_dim = q_quant.shape
    q_for_matmul = q_quant.float().reshape(batch, query_count * head_count, head_dim)
    scaled_head_weights = head_weights.float() * queries.shape[-2] ** -0.5
    score_floor = torch.full(
        (),
        torch.finfo(torch.float32).min,
        dtype=torch.float32,
        device=queries.device,
    )

    # Keep only the current global top-k while visiting fixed-width key tiles.
    # This bounds the largest QK intermediate to [B, Q, H, 256] instead of
    # materializing [B, Q, H, T].  The Python loop is over static shapes, so
    # torch.compile unrolls a fixed graph for each frozen context bucket.
    candidate_values = None
    candidate_indices = None
    candidate_validity = None
    for key_start in range(0, keys.shape[1], _DSA_SCORE_TILE_SIZE):
        key_stop = min(key_start + _DSA_SCORE_TILE_SIZE, keys.shape[1])
        key_tile = k_quant[:, key_start:key_stop]
        tile_scores_per_head = torch.matmul(
            q_for_matmul,
            key_tile.float().transpose(1, 2),
        ).reshape(batch, query_count, head_count, key_stop - key_start)
        tile_scores_per_head = tile_scores_per_head * q_scale.squeeze(-1).unsqueeze(-1)
        tile_scores_per_head = (
            tile_scores_per_head
            * k_scale[:, key_start:key_stop].squeeze(-1)[:, None, None, :]
        )
        tile_scores_per_head = F.relu(tile_scores_per_head * queries.shape[-1] ** -0.5)
        tile_scores = (tile_scores_per_head * scaled_head_weights.unsqueeze(-1)).sum(
            dim=2
        )
        tile_key_positions = key_positions[:, None, key_start:key_stop]
        tile_validity = tile_key_positions <= query_positions.unsqueeze(-1)
        tile_scores = torch.where(tile_validity, tile_scores, score_floor)
        tile_indices = torch.arange(
            key_start,
            key_stop,
            dtype=torch.int64,
            device=keys.device,
        ).view(1, 1, -1)
        tile_indices = tile_indices.expand(batch, query_count, -1)

        if candidate_values is None:
            candidate_values = tile_scores
            candidate_indices = tile_indices
            candidate_validity = tile_validity
        else:
            candidate_values = torch.cat((candidate_values, tile_scores), dim=-1)
            candidate_indices = torch.cat((candidate_indices, tile_indices), dim=-1)
            candidate_validity = torch.cat((candidate_validity, tile_validity), dim=-1)

        if candidate_values.shape[-1] > topk:
            candidate_values, retained = neuron_topk(
                candidate_values,
                k=topk,
                dim=-1,
                gather_dim=-1,
            )
            retained = retained.to(torch.int64)
            candidate_indices = torch.gather(candidate_indices, -1, retained)
            candidate_validity = torch.gather(candidate_validity, -1, retained)

    assert candidate_indices is not None
    assert candidate_validity is not None
    selected = torch.where(
        candidate_validity,
        candidate_indices,
        -torch.ones_like(candidate_indices),
    )
    available = min(topk, keys.shape[1])
    if available < topk:
        selected = F.pad(selected, (0, topk - available), value=-1)
    return selected


def latest_indexer_layer(layer_index: int) -> int:
    """Return the indexer result reused by one main decoder layer."""

    if not 0 <= layer_index < MAIN_LAYER_COUNT:
        raise ValueError(f"layer {layer_index} is outside main execution")
    return max(layer for layer in MAIN_INDEXER_LAYER_INDICES if layer <= layer_index)


@dataclass(frozen=True)
class IndexerProjection:
    queries: torch.Tensor
    keys: torch.Tensor
    head_weights: torch.Tensor


class GlmMoeDsaIndexer(nn.Module):
    """Projection and sparse-selection module for one scheduled DSA layer."""

    def __init__(
        self,
        *,
        hidden_size: int,
        q_lora_rank: int,
        num_heads: int = 32,
        head_dim: int = 128,
        rope_dim: int = 64,
        topk: int = 2048,
        rope_theta: float = 8_000_000.0,
        fp8_weights: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if head_dim != INDEXER_KEY_DIM:
            raise ValueError(f"GLM-5.2 indexer head_dim must be {INDEXER_KEY_DIM}")
        if not 0 < rope_dim <= head_dim or rope_dim % 2:
            raise ValueError("rope_dim must be positive, even, and at most head_dim")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        self.topk = topk
        self.rope_theta = rope_theta
        if fp8_weights:
            self.wq_b = BlockFP8Linear(q_lora_rank, num_heads * head_dim, device=device)
            self.wk = BlockFP8Linear(hidden_size, head_dim, device=device)
        else:
            self.wq_b = nn.Linear(
                q_lora_rank,
                num_heads * head_dim,
                bias=False,
                dtype=dtype,
                device=device,
            )
            self.wk = nn.Linear(
                hidden_size,
                head_dim,
                bias=False,
                dtype=dtype,
                device=device,
            )
        self.weights_proj = nn.Linear(
            hidden_size,
            num_heads,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.k_norm = nn.LayerNorm(head_dim, eps=1.0e-6, dtype=dtype, device=device)

    def project(
        self,
        hidden_states: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
    ) -> IndexerProjection:
        q = self.wq_b(q_lora).view(*q_lora.shape[:-1], self.num_heads, self.head_dim)
        k = self.k_norm(self.wk(hidden_states))
        q_pe = q[..., : self.rope_dim].contiguous()
        q_nope = q[..., self.rope_dim :].contiguous()
        k_pe = k[..., : self.rope_dim].contiguous()
        k_nope = k[..., self.rope_dim :].contiguous()
        cos, sin = rotary_cos_sin(
            positions, self.rope_dim, theta=self.rope_theta, dtype=q.dtype
        )
        q_pe = apply_interleaved_rope(q_pe, cos, sin)
        k_pe = apply_interleaved_rope(k_pe, cos, sin)
        return IndexerProjection(
            queries=torch.cat((q_pe, q_nope), dim=-1),
            keys=torch.cat((k_pe, k_nope), dim=-1),
            head_weights=self.weights_proj(hidden_states),
        )

    def select(
        self,
        projection: IndexerProjection,
        cached_keys: torch.Tensor,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
    ) -> torch.Tensor:
        return causal_topk_indices(
            projection.queries,
            cached_keys,
            projection.head_weights,
            query_positions,
            key_positions,
            topk=self.topk,
        )
