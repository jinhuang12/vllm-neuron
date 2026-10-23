# SPDX-License-Identifier: Apache-2.0
"""KDA output at prefill: gated RMSNorm and the ``o_proj`` partial in one NKI launch.

Per head of ``kdim`` columns, in fp32, as the checkpoint's ``FusedRMSNormGated`` with a
sigmoid gate, then ``o_proj``::

    o = core * rsqrt(mean(core^2) + eps) * o_norm * sigmoid(gate)
    out = o @ o_proj.T            # [T, H] fp32, this rank's partial before the TP sum

The elementwise steps run with tokens on the partitions, in that order. The projection
runs on the tensor engine as two bf16 passes accumulated in fp32 PSUM: ``hi = bf16(o)``
and ``lo = bf16(o - hi)``. Every product is exact in fp32, and ``o - hi - lo`` is at most
2^-17 of ``|o|``: ``o - hi`` is below 2^-8 of the leading power of two of ``o``, and its
bf16 rounding leaves at most 2^-9 of that.

Each program takes a contiguous half of the rows (LNC 2) and all ``H`` output columns.
``o_proj^T`` stays resident in SBUF, loaded once by DMA transposes; the rows go through
in groups of ``TILE_GROUP`` 128-row tiles, and each ``[128, 1024]`` output block is
stored as soon as it leaves PSUM.
"""

import nki
import nki.isa as nisa
import nki.language as nl
from nkilib.core.utils.kernel_assert import kernel_assert

import torch
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron.utils.hardware_config import get_logical_nc_config

_P = 128
_BANK = 512  # fp32 columns per PSUM bank
_STORE_BANKS = 2  # PSUM banks per output store
_DGE_ROWS = 16  # o_proj rows per DMA-transpose block


def _gated_rows(core, gate, gain, eps):
    """``core * rsqrt(mean(core^2) + eps) * gain * sigmoid(gate)`` per ``kdim`` group, fp32."""
    rows, groups, kdim = core.shape
    sq = nl.ndarray((rows, groups, kdim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sq, data1=core, data2=core, op=nl.multiply)
    rstd = nl.ndarray((rows, max(groups, 8)), dtype=nl.float32, buffer=nl.sbuf)[:, :groups]
    nisa.tensor_reduce(dst=rstd, op=nl.add, data=sq, axis=(2,))
    # sum * (1 / kdim) is the exact mean for a power-of-two kdim.
    nisa.tensor_scalar(dst=rstd, data=rstd, op0=nl.multiply, operand0=1.0 / float(kdim),
                       op1=nl.add, operand1=float(eps))
    nisa.tensor_scalar(dst=rstd, data=rstd, op0=nl.rsqrt, operand0=0.0,
                       engine=nisa.engine.gpsimd)
    shaped = nl.ndarray((rows, groups, kdim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=shaped, data1=core, data2=rstd.expand_dim(2).broadcast(2, kdim),
                       op=nl.multiply)
    nisa.tensor_tensor(dst=shaped, data1=shaped,
                       data2=gain.expand_dim(1).broadcast(1, groups), op=nl.multiply)
    sig = nl.ndarray((rows, groups, kdim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sig.reshape((rows, groups * kdim)), op=nl.sigmoid,
                    data=gate.reshape((rows, groups * kdim)))
    nisa.tensor_tensor(dst=shaped, data1=shaped, data2=sig, op=nl.multiply)
    return shaped


def _split_bf16(flat):
    """``(hi, lo)`` bf16 tiles with ``hi = bf16(flat)`` and ``lo = bf16(flat - hi)``."""
    rows, cols = flat.shape
    hi = nl.ndarray((rows, cols), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=hi, src=flat)
    rem = nl.ndarray((rows, cols), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=rem, data1=flat, data2=hi, op=nl.subtract)
    lo = nl.ndarray((rows, cols), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=lo, src=rem)
    return hi, lo


@nki.jit
def kda_output_prefill_kernel(core, gate, o_norm, o_proj, HEADS: int = 3,
                              EPS: float = 1e-5, TILE_GROUP: int = 8):
    """``out [T, H]`` fp32: the gated RMSNorm of ``core`` projected by ``o_proj``.

    Args:
        core: ``[T, W]`` fp32 recurrence output, ``W = HEADS * kdim`` in 128-column
            tiles, ``T`` a multiple of 128 per program.
        gate: ``[T, W]`` fp32 output gate, before its sigmoid.
        o_norm: ``[kdim]`` fp32 or bf16 norm gain, ``kdim`` a power of two.
        o_proj: ``[H, W]`` bf16, this rank's row-parallel slice, ``H`` in 1024-column
            output blocks.
        HEADS: heads per rank.
        EPS: the norm epsilon.
        TILE_GROUP: 128-row tiles normed together.

    Returns:
        ``[T, H]`` fp32 partial sum, before the TP reduction.
    """
    tokens, width = core.shape
    hidden, w2 = o_proj.shape
    kdim = width // HEADS
    programs = nl.num_programs(0)
    program = nl.program_id(0)
    kernel_assert(programs in (1, 2), "one or two programs")
    kernel_assert(tokens % (_P * programs) == 0, "T splits into 128-row tiles per program")
    kernel_assert(tuple(gate.shape) == (tokens, width), "gate matches core")
    kernel_assert(w2 == width and width % _P == 0 and kdim * HEADS == width,
                  "W = HEADS * kdim in 128-column tiles")
    kernel_assert(kdim & (kdim - 1) == 0 and tuple(o_norm.shape) == (kdim,),
                  "o_norm is [kdim], kdim a power of two")
    kernel_assert(hidden % (_STORE_BANKS * _BANK) == 0, "H in 1024-column blocks")
    kernel_assert(core.dtype == nl.float32 and gate.dtype == nl.float32, "fp32 core and gate")
    kernel_assert(o_proj.dtype == nl.bfloat16, "bf16 o_proj")
    rows = tokens // programs
    r0 = program * rows
    tiles = rows // _P
    col_tiles = width // _P
    block = _STORE_BANKS * _BANK
    blocks = hidden // block
    out = nl.ndarray((tokens, hidden), dtype=nl.float32, buffer=nl.shared_hbm)

    # o_proj^T, resident: ot[kt][b][j, h] = o_proj[b * block + h, kt * 128 + j].
    ot = [[nl.ndarray((_P, block), dtype=o_proj.dtype, buffer=nl.sbuf) for _ in range(blocks)]
          for _ in range(col_tiles)]
    for kt in range(col_tiles):
        for b in range(blocks):
            nisa.dma_transpose(
                dst=ot[kt][b].reshape((_P, block // _DGE_ROWS, _DGE_ROWS)),
                src=o_proj.ap(pattern=[[width, _DGE_ROWS], [_DGE_ROWS * width, block // _DGE_ROWS],
                                       [1, _P]],
                              offset=b * block * width + kt * _P))
    gain = nl.ndarray((_P, kdim), dtype=o_norm.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=gain, src=o_norm.ap(pattern=[[0, _P], [1, kdim]], offset=0))

    for g0 in range(0, tiles, TILE_GROUP):
        group = min(TILE_GROUP, tiles - g0)
        base = r0 + g0 * _P
        # c[p, i] = core[base + 128 i + p]: the group's tiles side by side on the free axis.
        c = nl.ndarray((_P, group * HEADS, kdim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=c.reshape((_P, group, width)),
                      src=core.ap(pattern=[[width, _P], [_P * width, group], [1, width]],
                                  offset=base * width))
        g = nl.ndarray((_P, group, width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=g, src=gate.ap(pattern=[[width, _P], [_P * width, group], [1, width]],
                                         offset=base * width))
        shaped = _gated_rows(c, g.reshape((_P, group * HEADS, kdim)), gain, EPS)
        hi, lo = _split_bf16(shaped.reshape((_P, group * width)))

        # gt[j, i, kt, part, p]: the stationaries, part 0 = hi, part 1 = lo.
        gt = nl.ndarray((_P, group, col_tiles, 2, _P), dtype=nl.bfloat16, buffer=nl.sbuf)
        for i in range(group):
            for kt in range(col_tiles):
                col = i * width + kt * _P
                gps = nl.ndarray((_P, 2, _P), dtype=nl.bfloat16, buffer=nl.psum)
                nisa.nc_transpose(dst=gps[:, 0, :], data=hi[:, col:col + _P])
                nisa.nc_transpose(dst=gps[:, 1, :], data=lo[:, col:col + _P])
                nisa.tensor_copy(dst=gt[:, i, kt, :, :], src=gps)

        for b in range(blocks):
            for i in range(group):
                rows_out = nl.ndarray((_P, block), dtype=nl.float32, buffer=nl.sbuf)
                for j in range(_STORE_BANKS):
                    acc = nl.ndarray((_P, _BANK), dtype=nl.float32, buffer=nl.psum)
                    for kt in range(col_tiles):
                        for part in range(2):
                            nisa.nc_matmul(dst=acc, stationary=gt[:, i, kt, part, :],
                                           moving=ot[kt][b][:, j * _BANK:(j + 1) * _BANK],
                                           accumulate=(kt > 0 or part > 0))
                    # Vector and Scalar both read PSUM: alternate them.
                    nisa.tensor_copy(dst=rows_out[:, j * _BANK:(j + 1) * _BANK], src=acc,
                                     engine=nisa.engine.vector
                                     if (b * group + i + j) % 2 == 0 else nisa.engine.scalar)
                nisa.dma_copy(dst=out.ap(pattern=[[hidden, _P], [1, block]],
                                         offset=(base + i * _P) * hidden + b * block),
                              src=rows_out)
    return out


_KERNEL = wrap_nki(kda_output_prefill_kernel)


def kda_output_prefill(core: torch.Tensor, gate: torch.Tensor, o_norm_weight: torch.Tensor,
                       o_proj_weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Gated RMSNorm of ``core`` and its ``o_proj`` partial, ``[T, H]`` fp32, on the NKI kernel.

    Args:
        core: ``[T, W]`` fp32 recurrence output of this rank's heads.
        gate: ``[T, W]`` fp32 output-gate projection, before its sigmoid.
        o_norm_weight: ``[kdim]`` fp32 or bf16 norm gain shared by the heads.
        o_proj_weight: ``[H, W]`` bf16 row-parallel slice of ``o_proj``.
        eps: the norm epsilon.

    Returns:
        ``[T, H]`` fp32, this rank's partial before the TP reduction.

    Raises:
        ValueError: an input outside the kernel's contract.
    """
    lnc = get_logical_nc_config()
    tokens, width = core.shape
    hidden = o_proj_weight.shape[0]
    kdim = o_norm_weight.shape[0]
    if core.dtype != torch.float32 or gate.dtype != torch.float32:
        raise ValueError(f"core and gate must be fp32, got {core.dtype} and {gate.dtype}")
    if o_proj_weight.dtype != torch.bfloat16:
        raise ValueError(f"o_proj must be bf16, got {o_proj_weight.dtype}")
    if o_norm_weight.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"o_norm must be fp32 or bf16, got {o_norm_weight.dtype}")
    if tuple(gate.shape) != (tokens, width) or tuple(o_proj_weight.shape) != (hidden, width):
        raise ValueError(f"shapes core {tuple(core.shape)}, gate {tuple(gate.shape)}, "
                         f"o_proj {tuple(o_proj_weight.shape)} disagree")
    if o_norm_weight.dim() != 1 or kdim & (kdim - 1) or width % kdim or width % _P:
        raise ValueError(f"W {width} must be whole heads of a power-of-two kdim {kdim}, "
                         f"in 128-column tiles")
    if tokens % (_P * lnc) or hidden % (_STORE_BANKS * _BANK):
        raise ValueError(f"T {tokens} must split into 128-row tiles over {lnc} programs and "
                         f"H {hidden} into 1024-column blocks")
    return _KERNEL[lnc](core=core.contiguous(), gate=gate.contiguous(),
                        o_norm=o_norm_weight.contiguous(), o_proj=o_proj_weight.contiguous(),
                        HEADS=width // kdim, EPS=float(eps))
