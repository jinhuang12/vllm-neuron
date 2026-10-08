# SPDX-License-Identifier: Apache-2.0
"""Pieces shared by the MTP tail kernels (``tail_in``, ``tail_out``).

Both kernels are a row-wise RMSNorm of a few decode rows followed by a GEMV of
those rows against a row-major bf16 weight whose rows are the OUTPUT features
(``torch.nn.functional.linear`` layout, ``[out, in]``). The layout idioms live here:

* ``rms_rows``: ``[rows, H]`` rows on the partitions, the whole row in each
  partition's free dimension (``rows <= 128``), fp32 interior, rounding on the last
  write -- the arithmetic of ``Glm5NextMultiTokenPredictor._rms_norm``;
* ``transpose_rows``: the normed rows moved onto the contraction axis,
  ``xt[c, kb, r] = x[r, kb * 128 + c]`` (tensor engine transposes, one PSUM bank at
  a time);
* ``load_transposed``: a weight slab ``[cols, K]`` brought in as
  ``wt[c, kb, col] = w[col, kb * 128 + c]`` by the DMA engines' transpose, so the
  GEMV is ``nc_matmul(stationary=xt[:, kb, :], moving=wt[:, kb, :])`` accumulated
  over ``kb`` -- one pass over the weight, which is the bytes the kernel's time is;
* the launch-side helpers: the program count under LNC2, the dispatch counters the
  tests read, and :class:`MtpTailError`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

import nki.isa as nisa
import nki.language as nl

PARTITIONS = nl.tile_size.pmax
#: One PSUM bank: 2 KiB, the widest fp32 ``nc_matmul`` result (512 columns).
BANK_BYTES = 2048
MATMUL_COLS = BANK_BYTES // 4
#: Rows per DMA transpose (the engine transposes 16-row groups of a 2-byte dtype) and
#: the alignment of the transpose's output row stride (32 bytes = 16 bf16).
TRANSPOSE_ROWS = 16
TRANSPOSE_STRIDE = 16


class MtpTailError(ValueError):
    """An operand geometry or dtype the MTP tail kernels do not take."""


def padded(count: int, step: int) -> int:
    """``count`` rounded up to a multiple of ``step``."""
    return -(-count // step) * step


def even(count: int) -> int:
    """``count`` rounded up to even: a bf16 PSUM slot starts 4-byte aligned."""
    return count + (count % 2)


def per_bank(dtype, cols: int) -> int:
    """How many ``cols``-wide transposes of ``dtype`` fill one PSUM bank (>= 1)."""
    size = 2 if dtype == nl.bfloat16 else 4
    return max(1, BANK_BYTES // (size * max(cols, 1)))


def tile(rows: int, cols: int, dtype=nl.float32):
    """An SBUF tile ``[rows, cols]`` backed by at least 8 columns."""
    return nl.ndarray((rows, max(cols, 8)), dtype=dtype, buffer=nl.sbuf)[:, :cols]


def rms_rows(x, rows: int, hidden: int, gain, eps: float, out_dtype):
    """``[rows, hidden]`` -> ``(x * rsqrt(mean(x**2) + eps)) * gain`` in ``out_dtype``.

    ``x`` is an fp32 SBUF tile, ``gain`` an SBUF tile ``[rows, hidden]`` (the gain
    row broadcast over the partitions by its load). ``* (1 / hidden)`` stands for
    the mean's divide: exact for a power-of-two ``hidden``, else off by one fp32 ulp,
    inside the norm's derived error term either way.
    """
    squares = tile(rows, hidden)
    nisa.tensor_tensor(dst=squares, data1=x, data2=x, op=nl.multiply)
    sqsum = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=sqsum, op=nl.add, data=squares, axis=(1,))
    rstd = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rstd, data=sqsum, op0=nl.multiply, operand0=1.0 / hidden,
                       op1=nl.add, operand1=float(eps))
    # rsqrt on GpSimd, the higher-precision engine for it (see functional/norm.py).
    nisa.tensor_scalar(dst=rstd, data=rstd, op0=nl.rsqrt, operand0=0.0,
                       engine=nisa.engine.gpsimd)
    scaled = tile(rows, hidden)
    nisa.tensor_scalar(dst=scaled, data=x, op0=nl.multiply, operand0=rstd)
    normed = tile(rows, hidden, out_dtype)
    nisa.tensor_tensor(dst=normed, data1=scaled, data2=gain, op=nl.multiply)
    return normed


def transpose_rows(dst, x, rows: int, blocks: int, block0: int = 0) -> None:
    """``dst[c, block0 + kb, r] = x[r, kb * 128 + c]`` for ``kb < blocks``.

    ``x`` is ``[rows, blocks * 128]`` with ``rows <= 128``; the transposes go
    through PSUM one bank at a time and are copied out in ``dst``'s dtype.
    """
    slot = even(rows)
    per = per_bank(x.dtype, slot)
    for g0 in range(0, blocks, per):
        gn = min(per, blocks - g0)
        flipped = nl.ndarray((PARTITIONS, gn, slot), dtype=x.dtype, buffer=nl.psum)
        for j in range(gn):
            nisa.nc_transpose(dst=flipped[:, j, 0:rows],
                              data=x[:, (g0 + j) * PARTITIONS:(g0 + j + 1) * PARTITIONS])
        nisa.tensor_copy(dst=dst[:, block0 + g0:block0 + g0 + gn, 0:rows],
                         src=flipped[:, :, 0:rows])


def load_transposed(dst, weight, row0: int, rows: int, blocks: int) -> None:
    """``dst[c, kb, i] = weight[row0 + i, kb * 128 + c]`` for ``i < rows``, ``kb < blocks``.

    ``weight`` is a row-major 2-byte ``[*, blocks * 128]`` HBM tensor; ``dst`` an SBUF
    tile ``[128, blocks, >= rows]`` whose last stride is a multiple of
    :data:`TRANSPOSE_STRIDE` elements (the caller pads it). The DMA engines
    transpose 16 rows at a time.
    """
    width = weight.shape[1]
    for r0 in range(0, rows, TRANSPOSE_ROWS):
        n = min(TRANSPOSE_ROWS, rows - r0)
        nisa.dma_transpose(
            dst=dst[:, :, r0:r0 + n],
            src=weight.ap(pattern=[[width, n], [PARTITIONS, blocks], [1, PARTITIONS]],
                          offset=(row0 + r0) * width))


def launch_programs() -> int:
    """2 programs (both LNC2 cores) under ``NEURON_LOGICAL_NC_CONFIG=2``, else 1."""
    return 2 if os.environ.get("NEURON_LOGICAL_NC_CONFIG") == "2" else 1


@dataclass
class DispatchCounters:
    """``kernel``: launches that returned; ``torch_route``: calls the torch formula served."""

    kernel: int = 0
    torch_route: int = 0

    def reset(self) -> None:
        self.kernel = 0
        self.torch_route = 0

    def pair(self) -> tuple[int, int]:
        return self.kernel, self.torch_route


def count_kernel(counters: DispatchCounters):
    @torch._dynamo.assume_constant_result
    def bump() -> None:
        counters.kernel += 1
    return bump


def count_torch_route(counters: DispatchCounters):
    @torch._dynamo.assume_constant_result
    def bump() -> None:
        counters.torch_route += 1
    return bump
