# SPDX-License-Identifier: Apache-2.0
"""Depthwise conv1d over the sequence axis, as one NKI kernel.

``out[n, c, q] = sum_s filt[c, s] * x_pad[n, c, q * stride_w + s]``, where ``x_pad``
is the input with ``pad_left`` and ``pad_right`` zero columns. Each channel owns its
taps, so there is no contraction across partitions and nothing for the PE array to
do: a channel tile sits on the partitions and the sequence on the free axis, and
every tap is one elementwise pass over a shifted window of the same SBUF tile.

Engine plan, per tile of ``PARTITION_MAX`` channels by up to ``COL_TILE`` columns:

1. One static DMA loads the input window, ``(cn - 1) * stride_w + S`` columns.
   Columns that fall in the padding are memset to zero instead.
2. The taps split into two accumulation chains. ScalarE heads each chain with
   ``activation(copy, scale=filt[c, s])``, a product; VectorE extends it with
   ``scalar_tensor_tensor``, ``x_s * filt[c, s] + partial``. At ``S = 4`` that is two
   passes on each engine, so neither waits on the other's backlog.
3. The DMA engines add the two chain results while storing the tile
   (``dma_compute``), so no engine spends a pass on the merge.

Every pass writes an accumulator slot that none of its operands occupies: each
chain alternates between two slots of one tile, and the tile is allocated whole,
so a destination never aliases a source by construction.

Under LNC2 the wrap launches two programs. Each takes a contiguous share of the
output columns, which keeps all of a tile's partitions busy at any channel count
and leaves every output element computed by the same instructions whichever
program owns it.

The tap weights of every channel tile load once, as float32, into one SBUF tile of
``TAP_SLOTS_MAX`` slots per partition; ScalarE takes its ``scale`` operand in
float32. That load runs on the software DGE queue, apart from the input loads:
on their queue the scheduler may issue it after the first input window, and the
first pass would then wait on the weights alone and rely on the queue completing
in order for its input. Accumulation is float32 throughout, and the store rounds
once to the input dtype.
"""

from __future__ import annotations

import nki
import nki.isa as nisa
import nki.language as nl

#: Partitions per channel tile. A literal rather than ``nl.tile_size.pmax`` read at
#: import time, because this module must import on a host with no NKI device.
PARTITION_MAX = 128

#: Output columns per tile, at most. Measured on trn2 at the served ``[1, 384, 1,
#: 2051]`` call: one 1024-column tile per program costs 21.0 us a call, 512-column
#: tiles 18.3 us, because the first tile's passes start while later tiles load.
#: Narrower tiles add per-instruction overhead (342 columns: 19.4 us).
COL_TILE = 512

#: Input columns one tile's window may span per partition, ``(cn - 1) * stride_w +
#: S``: 16 KiB of float32. At a large stride this, not ``COL_TILE``, caps the tile.
WINDOW_MAX = 4096

#: Float32 tap slots per partition in the weight tile, ``ceil(C / PARTITION_MAX) *
#: S``: 8 KiB. The wrap refuses a call that needs more.
TAP_SLOTS_MAX = 2048

#: Accumulation chains per output: one per engine that can head a chain, so ScalarE
#: and VectorE share the products.
CHAINS = 2

#: Accumulator slots per chain. A chain step reads one slot and writes the other.
SLOTS_PER_CHAIN = 2


def _div_ceil(numerator: int, denominator: int) -> int:
    """``ceil(numerator / denominator)`` in integers, for ``numerator >= 0 < denominator``."""
    return (numerator + denominator - 1) // denominator


def tap_slots(channels: int, taps: int) -> int:
    """Float32 slots per partition the weight tile needs for ``channels`` x ``taps``."""
    return _div_ceil(channels, PARTITION_MAX) * taps


def column_tile_cap(taps: int, stride_w: int) -> int:
    """Widest column tile whose input window fits :data:`WINDOW_MAX` and :data:`COL_TILE`."""
    return min(COL_TILE, (WINDOW_MAX - taps) // stride_w + 1)


def _load_taps(filt, channels, taps):
    """``[PARTITION_MAX, tiles, S]`` float32: channel ``t * PARTITION_MAX + p`` on row ``p``."""
    tiles = _div_ceil(channels, PARTITION_MAX)
    full = channels // PARTITION_MAX
    tail = channels - full * PARTITION_MAX
    rows = filt.reshape((channels, taps))
    weights = nl.ndarray((PARTITION_MAX, tiles, taps), dtype=nl.float32, buffer=nl.sbuf)
    if full > 0:
        nisa.dma_copy(
            dst=weights[0:PARTITION_MAX, 0:full, 0:taps],
            src=rows.ap(pattern=[[taps, PARTITION_MAX], [PARTITION_MAX * taps, full],
                                 [1, taps]], offset=0),
            dge_mode=nisa.dge_mode.swdge)
    if tail > 0:
        nisa.dma_copy(dst=weights[0:tail, full, 0:taps],
                      src=rows[full * PARTITION_MAX:channels, 0:taps],
                      dge_mode=nisa.dge_mode.swdge)
    return weights


def _window(x, rows, tap, cn, stride_w):
    """The ``cn`` window columns tap ``tap`` multiplies."""
    if stride_w == 1:
        return x[0:rows, tap:tap + cn]
    return x[0:rows, tap:tap + (cn - 1) * stride_w + 1:stride_w]


def _conv_tile(img_rows, out_rows, weights, row0, rows, tile, c0, cn, taps, stride_w,
               pad_left):
    """Output columns ``[c0, c0 + cn)`` of image rows ``[row0, row0 + rows)``.

    ``tile`` indexes the channel tile's taps in ``weights``; ``rows`` is
    ``PARTITION_MAX`` except on a channel tail.
    """
    width = img_rows.shape[1]
    window = (cn - 1) * stride_w + taps
    # Input column of the window's first slot; negative inside the left padding.
    start = c0 * stride_w - pad_left
    lead = min(window, max(0, -start))
    loaded = max(0, min(width, start + window) - max(0, start))
    x = nl.ndarray((PARTITION_MAX, window), dtype=img_rows.dtype, buffer=nl.sbuf)
    if lead > 0:
        nisa.memset(dst=x[0:rows, 0:lead], value=0.0)
    if loaded > 0:
        nisa.dma_copy(dst=x[0:rows, lead:lead + loaded],
                      src=img_rows[row0:row0 + rows, start + lead:start + lead + loaded],
                      dge_mode=nisa.dge_mode.none)
    if lead + loaded < window:
        nisa.memset(dst=x[0:rows, lead + loaded:window], value=0.0)

    acc = nl.ndarray((PARTITION_MAX, CHAINS * SLOTS_PER_CHAIN, cn), dtype=nl.float32,
                     buffer=nl.sbuf)
    chain_taps = _div_ceil(taps, CHAINS)
    results = []
    for chain in range(CHAINS):
        first = chain * chain_taps
        stop = min(taps, first + chain_taps)
        if first < stop:
            base = chain * SLOTS_PER_CHAIN
            nisa.activation(dst=acc[0:rows, base, 0:cn], op=nl.copy,
                            data=_window(x, rows, first, cn, stride_w),
                            scale=weights[0:rows, tile, first:first + 1])
            for tap in range(first + 1, stop):
                step = tap - first
                nisa.scalar_tensor_tensor(
                    dst=acc[0:rows, base + step % SLOTS_PER_CHAIN, 0:cn],
                    data=_window(x, rows, tap, cn, stride_w),
                    op0=nl.multiply, operand0=weights[0:rows, tile, tap:tap + 1],
                    op1=nl.add,
                    operand1=acc[0:rows, base + (step - 1) % SLOTS_PER_CHAIN, 0:cn])
            results.append(acc[0:rows, base + (stop - 1 - first) % SLOTS_PER_CHAIN, 0:cn])
    dst = out_rows[row0:row0 + rows, c0:c0 + cn]
    if len(results) == 1:
        nisa.dma_copy(dst=dst, src=results[0], dge_mode=nisa.dge_mode.none)
    else:
        nisa.dma_compute(dst=dst, srcs=results, reduce_op=nl.add)


@nki.jit
def depthwise_conv1d_kernel(img, filt, pad_left, pad_right, stride_w):
    """Depthwise conv1d of ``img`` by ``filt`` along the last axis.

    Args:
        img: ``[N, C, 1, W]`` in HBM, float32, bfloat16 or float16.
        filt: ``[C, 1, 1, S]`` in HBM at the dtype of ``img``, one filter per channel.
        pad_left: zero columns before the input, ``>= 0``.
        pad_right: zero columns after the input, ``>= 0``.
        stride_w: output column step along the input, ``>= 1``.

    Returns:
        ``[N, C, 1, Q]`` in HBM at the dtype of ``img``, with
        ``Q = (W + pad_left + pad_right - S) // stride_w + 1``.

    Notes:
        The caller checks the geometry: ``Q >= 1`` and
        :func:`tap_slots` ``<= TAP_SLOTS_MAX``. Under a grid of ``P`` programs
        each takes ``ceil(Q / P)`` contiguous columns; a program past the last
        column computes nothing.
    """
    n_batch, channels, _, width = img.shape
    taps = filt.shape[3]
    columns = (width + pad_left + pad_right - taps) // stride_w + 1
    out = nl.ndarray((n_batch, channels, 1, columns), dtype=img.dtype, buffer=nl.shared_hbm)
    img_rows = img.reshape((n_batch * channels, width))
    out_rows = out.reshape((n_batch * channels, columns))
    weights = _load_taps(filt, channels, taps)

    full_tiles = channels // PARTITION_MAX
    tail_rows = channels - full_tiles * PARTITION_MAX
    share = _div_ceil(columns, nl.num_programs(axes=0))
    first_column = nl.program_id(axis=0) * share
    owned = min(share, columns - first_column)
    if owned > 0:
        n_col = _div_ceil(owned, column_tile_cap(taps, stride_w))
        span = _div_ceil(owned, n_col)
        for batch in range(n_batch):
            for col in range(n_col):
                c0 = first_column + col * span
                cn = min(span, first_column + owned - c0)
                if full_tiles > 0:
                    for tile in nl.affine_range(full_tiles):
                        _conv_tile(img_rows, out_rows, weights,
                                   batch * channels + tile * PARTITION_MAX, PARTITION_MAX,
                                   tile, c0, cn, taps, stride_w, pad_left)
                if tail_rows > 0:
                    _conv_tile(img_rows, out_rows, weights,
                               batch * channels + full_tiles * PARTITION_MAX, tail_rows,
                               full_tiles, c0, cn, taps, stride_w, pad_left)
    return out
