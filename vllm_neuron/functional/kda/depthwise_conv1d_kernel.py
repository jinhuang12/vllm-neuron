# SPDX-License-Identifier: Apache-2.0
"""KDA prefill short convolution: a causal depthwise conv1d and SiLU in one NKI kernel.

``out[c, t] = silu(sum_s weight[c, s] * x[t + s, c])``, where ``x`` is the carried
history rows followed by the q, k and v blocks side by side. The input arrives
token-major. Each run of up to 128 window rows loads as one DMA and turns onto the
channel partitions with one PE transpose, which moves the values unchanged. The taps
run as two float32 accumulation chains, each headed on Scalar and extended on Vector;
Vector adds the two chains, Scalar applies SiLU, and one DMA stores the tile
channel-major at the input dtype. Under LNC 2 each program takes a contiguous share
of the tokens.
"""

import nki
import nki.isa as nisa
import nki.language as nl

# A literal, so this module imports on a host with no Neuron device.
PARTITION_MAX = 128

# Output tokens per tile, at most: the first tile's passes start while later
# tiles load.
COL_TILE = 512

# Accumulation chains per output, and accumulator slots per chain: a chain step
# reads one slot and writes the other, so a destination never aliases its source.
CHAINS = 2
SLOTS_PER_CHAIN = 2


def _div_ceil(numerator, denominator):
    """Integer ceiling of ``numerator / denominator``."""
    return (numerator + denominator - 1) // denominator


def _load_taps(weight, channels, taps):
    """``[128, tiles, taps]`` float32: channel ``t * 128 + p`` on partition ``p``."""
    pmax = PARTITION_MAX
    tiles = channels // pmax
    taps_sb = nl.ndarray((pmax, tiles, taps), dtype=nl.float32, buffer=nl.sbuf)
    # The software DGE queue keeps this load off the window loads' queue.
    nisa.dma_copy(
        dst=taps_sb[0:pmax, 0:tiles, 0:taps],
        src=weight.ap(
            pattern=[[taps, pmax], [pmax * taps, tiles], [1, taps]], offset=0
        ),
        dge_mode=nisa.dge_mode.swdge,
    )
    return taps_sb


def _rows_window(history, block, block_ch0, channel0, c0, cn, taps):
    """The ``[128, cn + taps - 1]`` input window of output tokens ``[c0, c0 + cn)``.

    Window row ``r`` is ``history[c0 + r]`` while ``c0 + r < taps - 1``, else token
    ``c0 + r - taps + 1`` of ``block``; the channels are ``block[:, block_ch0:]``
    and ``history[:, channel0:]``.
    """
    pmax = PARTITION_MAX
    window = cn + taps - 1
    lead = taps - 1
    x = nl.ndarray((pmax, window), dtype=block.dtype, buffer=nl.sbuf)
    for run in range(_div_ceil(window, pmax)):
        b0 = run * pmax
        nb = min(pmax, window - b0)
        r0 = c0 + b0
        from_history = max(0, min(lead - r0, nb))
        rows = nl.ndarray((pmax, pmax), dtype=block.dtype, buffer=nl.sbuf)
        if from_history > 0:
            nisa.dma_copy(
                dst=rows[0:from_history, 0:pmax],
                src=history[r0:r0 + from_history, channel0:channel0 + pmax],
                dge_mode=nisa.dge_mode.none,
            )
        if nb > from_history:
            s0 = r0 + from_history - lead
            nisa.dma_copy(
                dst=rows[from_history:nb, 0:pmax],
                src=block[s0:s0 + nb - from_history, block_ch0:block_ch0 + pmax],
                dge_mode=nisa.dge_mode.none,
            )
        flipped = nl.ndarray((pmax, pmax), dtype=block.dtype, buffer=nl.psum)
        nisa.nc_transpose(
            dst=flipped[0:pmax, 0:nb],
            data=rows[0:nb, 0:pmax],
            engine=nisa.engine.tensor,
        )
        nisa.tensor_copy(
            dst=x[0:pmax, b0:b0 + nb],
            src=flipped[0:pmax, 0:nb],
            engine=nisa.engine.vector if run % 2 == 0 else nisa.engine.scalar,
        )
    return x


def _conv_silu_tile(x, out, taps_sb, tile, c0, cn, taps):
    """Output channels of tile ``tile``, tokens ``[c0, c0 + cn)``, from window ``x``."""
    pmax = PARTITION_MAX
    acc = nl.ndarray(
        (pmax, CHAINS * SLOTS_PER_CHAIN, cn), dtype=nl.float32, buffer=nl.sbuf
    )
    chain_taps = _div_ceil(taps, CHAINS)
    results = []
    for chain in range(CHAINS):
        first = chain * chain_taps
        stop = min(taps, first + chain_taps)
        base = chain * SLOTS_PER_CHAIN
        nisa.activation(
            dst=acc[0:pmax, base, 0:cn],
            op=nl.copy,
            data=x[0:pmax, first:first + cn],
            scale=taps_sb[0:pmax, tile, first:first + 1],
        )
        for tap in range(first + 1, stop):
            step = tap - first
            nisa.scalar_tensor_tensor(
                dst=acc[0:pmax, base + step % SLOTS_PER_CHAIN, 0:cn],
                data=x[0:pmax, tap:tap + cn],
                op0=nl.multiply,
                operand0=taps_sb[0:pmax, tile, tap:tap + 1],
                op1=nl.add,
                operand1=acc[0:pmax, base + (step - 1) % SLOTS_PER_CHAIN, 0:cn],
            )
        last = base + (stop - 1 - first) % SLOTS_PER_CHAIN
        results.append(acc[0:pmax, last, 0:cn])
    total = nl.ndarray((pmax, cn), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=total[0:pmax, 0:cn], data1=results[0], data2=results[1], op=nl.add
    )
    y = nl.ndarray((pmax, cn), dtype=out.dtype, buffer=nl.sbuf)
    nisa.activation(dst=y[0:pmax, 0:cn], op=nl.silu, data=total[0:pmax, 0:cn])
    row0 = tile * pmax
    nisa.dma_copy(
        dst=out[row0:row0 + pmax, c0:c0 + cn],
        src=y[0:pmax, 0:cn],
        dge_mode=nisa.dge_mode.none,
    )


@nki.jit
def depthwise_conv1d_silu_kernel(history, q, k, v, weight):
    """SiLU of the causal depthwise conv1d of ``[history; q | k | v]``, channel-major.

    Args:
        history: ``[S - 1, C]`` in HBM, the input rows before the first token.
        q, k, v: ``[T, C_i]`` in HBM at the history's dtype, ``C_q + C_k + C_v = C``,
            each a multiple of ``PARTITION_MAX``.
        weight: ``[C, S]`` float32 in HBM, the taps of each channel in q, k, v order.

    Returns:
        ``[C, T]`` in HBM at the input dtype.
    """
    blocks = (q, k, v)
    tokens = q.shape[0]
    channels = history.shape[1]
    taps = weight.shape[1]
    out = nl.ndarray((channels, tokens), dtype=q.dtype, buffer=nl.shared_hbm)
    taps_sb = _load_taps(weight, channels, taps)

    share = _div_ceil(tokens, nl.num_programs(axes=0))
    first_token = nl.program_id(axis=0) * share
    owned = min(share, tokens - first_token)
    if owned > 0:
        n_col = _div_ceil(owned, COL_TILE)
        span = _div_ceil(owned, n_col)
        for col in range(n_col):
            c0 = first_token + col * span
            cn = min(span, first_token + owned - c0)
            tile = 0
            for block in blocks:
                for block_tile in range(block.shape[1] // PARTITION_MAX):
                    x = _rows_window(
                        history,
                        block,
                        block_tile * PARTITION_MAX,
                        tile * PARTITION_MAX,
                        c0,
                        cn,
                        taps,
                    )
                    _conv_silu_tile(x, out, taps_sb, tile, c0, cn, taps)
                    tile += 1
    return out
