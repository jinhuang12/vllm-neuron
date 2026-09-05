# SPDX-License-Identifier: Apache-2.0
"""Weight packing helpers for the Qwen3-VL STATIC_MX / native-MX path.

The kernel consumes ``nl.float8_e4m3fn_x4``, which is 4 ``float8_e4m3fn``
bytes packed little-endian into one 32-bit element. ``torch`` lacks an
``x4`` dtype, so we store packed weights as ``torch.uint32``; the in-kernel
wrapper reinterprets the dtype back to ``nl.float8_e4m3fn_x4`` (mirroring
``vllm_neuron/functional/moe/moe_block_tkg_wrapper.py:67-70``).
"""

import torch


def x4_pack_fp8(w: torch.Tensor, *, contraction_axis: int) -> torch.Tensor:
    """Pack 4 consecutive fp8 bytes along ``contraction_axis`` into one uint32.

    Output shape has ``shape[contraction_axis] // 4``; output dtype is
    ``torch.uint32``. The pack is a bit-level reinterpret (no float math), so
    inverse round-trip with ``unpack_float8_e4m3fn_x4`` is bit-exact.

    Args:
        w: ``torch.float8_e4m3fn`` tensor with ``shape[contraction_axis] % 4 == 0``.
        contraction_axis: axis along which 4 consecutive bytes form one x4.

    Returns:
        ``torch.uint32`` tensor of the same rank, with ``shape[contraction_axis]
        // 4``.

    Example:
        >>> import torch
        >>> w = torch.zeros(4, dtype=torch.float8_e4m3fn)
        >>> x4_pack_fp8(w, contraction_axis=0).shape
        torch.Size([1])
    """
    assert w.dtype == torch.float8_e4m3fn, (
        f"x4_pack_fp8 expects torch.float8_e4m3fn, got {w.dtype}"
    )
    contraction_axis = contraction_axis % w.ndim
    n = w.shape[contraction_axis]
    assert n % 4 == 0, (
        f"x4_pack_fp8 requires shape[{contraction_axis}]={n} divisible by 4"
    )

    # Move contraction axis to the last position, make contiguous, then view as
    # uint8 -> uint32. The little-endian view is what nl.float8_e4m3fn_x4
    # expects (byte 0 -> low 8 bits, byte 3 -> high 8 bits); see
    # nkilib/core/utils/mx_torch_common.py::unpack_float8_e4m3fn_x4 lines
    # 114-123 for the canonical reverse transform.
    moved = w.movedim(contraction_axis, -1).contiguous()
    bytes_view = moved.view(torch.uint8)
    new_last = n // 4
    packed = bytes_view.reshape(*moved.shape[:-1], new_last, 4)
    # Bit-pack 4 bytes (little-endian) into one uint32. Done in int64 because
    # older torch builds (e.g. 2.5.1) lack a CPU implementation of lshift on
    # uint32; the result fits in 32 bits, so the final cast is lossless.
    packed_i64 = (
        packed[..., 0].to(torch.int64)
        | (packed[..., 1].to(torch.int64) << 8)
        | (packed[..., 2].to(torch.int64) << 16)
        | (packed[..., 3].to(torch.int64) << 24)
    )
    return packed_i64.to(torch.uint32).movedim(-1, contraction_axis).contiguous()
