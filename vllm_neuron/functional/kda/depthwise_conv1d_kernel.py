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

:func:`depthwise_conv1d_rows_kernel` takes the same convolution's input token-major
instead: the left-context rows and the channel blocks as separate ``[T, C_i]``
tensors, which a producer that writes rows emits without a concatenation or a
transpose in HBM. Its step 1 loads 128-row blocks and turns each onto the channel
partitions with one PE transpose, which moves the values unchanged, so the window
holds the same values as the image form's and steps 2 and 3 are shared.

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
#: 2051]`` call: 512-column tiles are faster than one 1024-column tile per program,
#: because the first tile's passes start while later tiles load. Narrower tiles add
#: per-instruction overhead.
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


def _image_window(img_rows, row0, rows, c0, cn, taps, stride_w, pad_left):
    """The input window of output columns ``[c0, c0 + cn)``, image rows ``[row0, row0 + rows)``.

    One static DMA loads the columns inside the image; columns in the padding are zero.
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
    return x


def _rows_window(history, block, b_ch0, h_ch0, c0, cn, taps):
    """The input window of output columns ``[c0, c0 + cn)`` of ``PARTITION_MAX`` channels.

    The convolution's input row ``r`` is ``history[r]`` for ``r < taps - 1``, else
    ``block[r - taps + 1]``; the channels are columns ``b_ch0 ..`` of ``block`` and
    ``h_ch0 ..`` of ``history``. Each run of up to ``PARTITION_MAX`` window rows loads
    token-major and one PE transpose puts it on the channel partitions; the Vector and
    Scalar engines take turns copying it out of PSUM.
    """
    window = cn + taps - 1
    lead = taps - 1
    x = nl.ndarray((PARTITION_MAX, window), dtype=block.dtype, buffer=nl.sbuf)
    for k in range(_div_ceil(window, PARTITION_MAX)):
        b0 = k * PARTITION_MAX
        nb = min(PARTITION_MAX, window - b0)
        r0 = c0 + b0
        from_history = max(0, min(lead - r0, nb))
        rows = nl.ndarray((PARTITION_MAX, PARTITION_MAX), dtype=block.dtype, buffer=nl.sbuf)
        if from_history > 0:
            nisa.dma_copy(dst=rows[0:from_history, 0:PARTITION_MAX],
                          src=history[r0:r0 + from_history, h_ch0:h_ch0 + PARTITION_MAX],
                          dge_mode=nisa.dge_mode.none)
        if nb > from_history:
            s0 = r0 + from_history - lead
            nisa.dma_copy(dst=rows[from_history:nb, 0:PARTITION_MAX],
                          src=block[s0:s0 + nb - from_history, b_ch0:b_ch0 + PARTITION_MAX],
                          dge_mode=nisa.dge_mode.none)
        flipped = nl.ndarray((PARTITION_MAX, PARTITION_MAX), dtype=block.dtype,
                             buffer=nl.psum)
        nisa.nc_transpose(dst=flipped[0:PARTITION_MAX, 0:nb],
                          data=rows[0:nb, 0:PARTITION_MAX], engine=nisa.engine.tensor)
        nisa.tensor_copy(dst=x[0:PARTITION_MAX, b0:b0 + nb],
                         src=flipped[0:PARTITION_MAX, 0:nb],
                         engine=nisa.engine.vector if k % 2 == 0 else nisa.engine.scalar)
    return x


def _conv_tile(x, out_rows, weights, row0, rows, tile, c0, cn, taps, stride_w):
    """Output columns ``[c0, c0 + cn)``, output rows ``[row0, row0 + rows)``, from window ``x``.

    ``tile`` indexes the channel tile's taps in ``weights``; ``rows`` is
    ``PARTITION_MAX`` except on a channel tail.
    """
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
                        row0 = batch * channels + tile * PARTITION_MAX
                        x = _image_window(img_rows, row0, PARTITION_MAX, c0, cn, taps,
                                          stride_w, pad_left)
                        _conv_tile(x, out_rows, weights, row0, PARTITION_MAX, tile, c0, cn,
                                   taps, stride_w)
                if tail_rows > 0:
                    row0 = batch * channels + full_tiles * PARTITION_MAX
                    x = _image_window(img_rows, row0, tail_rows, c0, cn, taps, stride_w,
                                      pad_left)
                    _conv_tile(x, out_rows, weights, row0, tail_rows, full_tiles, c0, cn,
                               taps, stride_w)
    return out


@nki.jit
def depthwise_conv1d_rows_kernel(history, block_a, block_b, block_c, filt):
    """Stride-1 depthwise conv1d of a token-major input given as left context and blocks.

    The convolution is :func:`depthwise_conv1d_kernel`'s with no padding and
    ``stride_w = 1`` on the image ``cat(history, cat(blocks, -1)).t()``, which this
    kernel never builds.

    Args:
        history: ``[S - 1, C]`` in HBM, the input rows before the first output's.
        block_a, block_b, block_c: ``[T, C_i]`` in HBM, the input's channel blocks in
            channel order, ``C_a + C_b + C_c = C``, each a multiple of ``PARTITION_MAX``.
        filt: ``[C, 1, 1, S]`` in HBM at the dtype of the blocks, one filter per channel.

    Returns:
        ``[1, C, 1, T]`` in HBM at the dtype of the blocks, as
        :func:`depthwise_conv1d_kernel` returns for that image.

    Notes:
        The caller checks the geometry and dtypes. Programs split the output columns
        as :func:`depthwise_conv1d_kernel`'s do.
    """
    blocks = (block_a, block_b, block_c)
    tokens = block_a.shape[0]
    channels = history.shape[1]
    taps = filt.shape[3]
    out = nl.ndarray((1, channels, 1, tokens), dtype=block_a.dtype, buffer=nl.shared_hbm)
    out_rows = out.reshape((channels, tokens))
    weights = _load_taps(filt, channels, taps)

    share = _div_ceil(tokens, nl.num_programs(axes=0))
    first_column = nl.program_id(axis=0) * share
    owned = min(share, tokens - first_column)
    if owned > 0:
        n_col = _div_ceil(owned, column_tile_cap(taps, 1))
        span = _div_ceil(owned, n_col)
        for col in range(n_col):
            c0 = first_column + col * span
            cn = min(span, first_column + owned - c0)
            tile = 0
            for index in range(len(blocks)):
                block = blocks[index]
                for block_tile in range(block.shape[1] // PARTITION_MAX):
                    x = _rows_window(history, block, block_tile * PARTITION_MAX,
                                     tile * PARTITION_MAX, c0, cn, taps)
                    _conv_tile(x, out_rows, weights, tile * PARTITION_MAX, PARTITION_MAX,
                               tile, c0, cn, taps, 1)
                    tile += 1
    return out
