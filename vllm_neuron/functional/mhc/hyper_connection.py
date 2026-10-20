# SPDX-License-Identifier: Apache-2.0
"""mHC combine, the hyper-connection "post" block, as an NKI kernel.

:mod:`vllm_neuron.functional.mhc.sinkhorn` normalises the mixing scores; this
module spends them. Given the ``hc_mult`` residual streams and the sub-block's
single-stream output, it mixes the streams back into ``hc_mult`` streams in one
dispatch per layer call. The torch code here is a CPU oracle, never the shipped
path.

The operation is upstream's ``mhc_post``::

    out_j = post_layer_mix_j * x + sum_i comb_res_mix_ij * residual_i

======================  =====================  ==============================
name                    shape                  what it is
======================  =====================  ==============================
``x``                   ``[T, H]``             the sub-block's single-stream
                                               output
``residual``            ``[T, S, H]``          the ``S = hc_mult`` residual
                                               streams
``post_layer_mix``      ``[T, S, 1]``          per token, per output stream
``comb_res_mix``        ``[T, S, S]``          per token, ``[i, j]`` = input
                                               stream ``i`` -> output ``j``
``out``                 ``[T, S, H]``          the re-mixed streams
======================  =====================  ==============================

``i`` is the summed input stream and ``j`` is the output stream. Upstream states
this twice in spellings that agree bit-for-bit,
``torch.einsum("...ij,...ih->...jh", comb, residual)`` and
``torch.bmm(comb.mT, residual)``; the ``.mT`` in the second is the whole content
of the convention, and a kernel that read ``comb[j, i]`` would be transposed.

There is no matmul in this kernel, because the mixing matrix is per token: the
tensor engine would need a block-diagonal ``[T*S, T*S]`` stationary built from
``T`` different ``S x S`` blocks. The operation is ``S * (S + 1)`` scalar
multiply-adds per hidden element, and the layout decides how many lanes do them.

Layout, picked from the token count. Up to
:data:`HIDDEN_ON_PARTITIONS_MAX_TOKENS` tokens (decode) the hidden axis lies
across the 128 partitions: hidden index ``h`` is on partition ``h // HF`` at free
offset ``h % HF`` (``HF = H / 128 = 32`` at the target's ``H = 4096``, 16 per
program under two programs), so each partition reads one contiguous run of every
row. Tokens and streams ride the free axes. Each token's ``S + S*S`` mixing
scalars are the same for every hidden index, so one DMA broadcasts them to all
partitions and stride-0 views broadcast them along ``HF``. A token chunk is then
``1 + 2*S`` vector instructions (9 at ``S = 4``), each over
``[128, tokens, S, HF]``. Baseline ``mhc_5938748`` put the token on the partition axis,
which at decode (``T = 1``) ran 72 instructions of 2048 elements on one lane of 128.

Above that token count the short ``HF`` runs make the DMAs the bottleneck, so
the tokens go back on the partitions, where every DMA row is a long contiguous
hidden run, and each multiply-add is one instruction: ``S * (1 + S)`` per tile
where baseline ``mhc_5938748`` issued ``S * (1 + 2*S)``.

With bf16 ``x`` and streams (the prefill form) those ``S * (1 + S)`` multiply-adds
per output element are the whole cost on the vector engine, so the combine moves
to the tensor engine. A group of ``G = 128 // (S + 1)`` tokens is one block-diagonal
matmul: the contraction rows are the group's ``G * S`` stream rows and then its
``G`` rows of ``x``, exactly as they lie in HBM, and the stationary operand holds
token ``t``'s ``comb[t]`` and ``post[t]`` on its diagonal block, so the result rows
are the output rows ``(t, j)`` in HBM order. The fp32 mixing weights are split into
:data:`TENSOR_ENGINE_PIECES` bf16 pieces, one matmul pass each, accumulated in fp32
PSUM: bf16 x bf16 products are exact, so only the order of the fp32 additions
differs from the vector form (see :func:`_combine_on_tensor_engine`).

Under an LNC2 launch (``NEURON_LOGICAL_NC_CONFIG=2``, the serving setting) the
kernel runs as two SPMD programs, each on one contiguous half of the hidden axis,
so both physical cores of the logical core work, in either layout.

The kernel computes in fp32 and returns ``residual``'s dtype. ``x`` and the
streams may be fp32 or bf16. In the two vector forms bf16 tiles are upcast on chip
(exact), so a bf16 call does the fp32 products and adds of an fp32 call on the same
values and rounds only the last add to bf16 -- bit for bit the fp32 call followed by
``.to(bf16)``, without the fp32 copies of the operands and the result in HBM. The
tensor-engine form is not bit for bit that call; it keeps the bound stated in
:func:`_combine_on_tensor_engine`. Upstream's ``mhc_post`` also takes bf16 and
returns ``residual.dtype``.

``T`` is unbounded: the hidden layout walks it in chunks sized to a fixed SBUF
budget (:data:`SBUF_CHUNK_ELEMS`), the token layout in tiles of 128 tokens.
``H`` need not divide by 128: in the hidden layout the last partition then holds
a shorter run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.dsa.launch_grid import lnc_pair
from vllm_neuron.functional.mhc.sinkhorn import MHC_STREAMS, PARTITION_MAX
from vllm_neuron.utils.neuron_utils import can_run_kernel

logger = logging.getLogger(__name__)

#: SBUF budget, in fp32 elements per partition, for one token chunk's tiles
#: (x, the streams, the accumulator, the product and the mixing scalars): 64 KiB
#: of the 192 KiB partition. At the target's H = 4096 one program holds 37 tokens
#: per chunk (HF = 32), two programs 71 (HF = 16), so decode is one chunk.
SBUF_CHUNK_ELEMS = 16384

#: Retired: baseline ``mhc_5938748`` tiled the hidden axis 2048 wide on the free axis.
#: The kernel no longer has that tile; the name stays because
#: ``test/hardware/benchmark_sparse_mla_mhc.py`` reports it.
HIDDEN_TILE = None

#: The largest token count served with hidden on the partitions. Above it the
#: kernel puts tokens on the partitions, where every DMA row is a long contiguous
#: run; below it, few tokens on the partitions would leave most lanes idle.
#: Measured on trn2 at H = 4096, two programs: the hidden layout is faster up to 32
#: tokens and the tokens layout from 64.
HIDDEN_ON_PARTITIONS_MAX_TOKENS = 32

#: Hidden slice width, in fp32 values, of the tokens-on-partitions layout: each
#: DMA row is one 2 KiB contiguous run.
TOKEN_LAYOUT_HIDDEN_SLICE = 512

#: bf16 pieces per fp32 mixing weight in the tensor-engine form. A bf16 piece carries
#: 8 significand bits, and each piece takes the rounding error of the ones before it,
#: so three pieces sum to the fp32 weight exactly (24 bits) unless the last piece
#: underflows; fewer would round the weights themselves.
TENSOR_ENGINE_PIECES = 3

#: Token groups whose stationary operands the tensor-engine form builds at once. A
#: group's bf16 pieces take ``2 * TENSOR_ENGINE_PIECES * G * S`` bytes of each
#: partition and two spans are live at once (the next span's pieces are built while
#: the tensor engine runs the current one), so a bounded span keeps the kernel's SBUF
#: use independent of ``T``, and the first span's build is all that runs before the
#: first matmul.
TENSOR_ENGINE_GROUP_SPAN = 16

# `MHC_STREAMS` (the target's `hc_mult`) and `PARTITION_MAX` are imported from
# `sinkhorn.py` rather than restated: both name the same quantity for both halves
# of mHC, so a second copy would be a second thing that can drift.

__all__ = [
    "MHC_STREAMS",
    "PARTITION_MAX",
    "HIDDEN_ON_PARTITIONS_MAX_TOKENS",
    "HyperConnectionError",
    "TENSOR_ENGINE_GROUP_SPAN",
    "TENSOR_ENGINE_PIECES",
    "TOKEN_LAYOUT_HIDDEN_SLICE",
    "can_run_hyper_connection",
    "dispatch_counters",
    "hyper_connection_combine",
    "hyper_connection_kernel",
    "hyper_connection_torch_oracle",
    "kernel_identity",
    "launch_programs",
    "reset_dispatch_counters",
]


class HyperConnectionError(ValueError):
    """A geometry or rank this module refuses, named rather than coerced.

    Raised in preference to letting NKI or numpy trap, because those traps do not
    name the offending argument. Without this check a ``[T, 3, 3]`` mix against 4
    streams gives ``Out-of-bound access for tensor `unnamed` on dimension 1``, a
    mismatched token count gives ``operands could not be broadcast together with
    shapes (7,32) (8,1)``, and a mismatched hidden extent gives a remapped-shape
    ``ValueError``.
    """


@nki.jit
def hyper_connection_kernel(x, residual, post_layer_mix, comb_res_mix):
    """The mHC combine, in NKI. One dispatch per call.

    Args:
        x: ``[T, H]`` fp32 in HBM -- the sub-block's single-stream output.
        residual: ``[T, S, H]`` fp32 in HBM -- the ``S`` residual streams.
        post_layer_mix: ``[T, S, 1]`` fp32 -- per token, per OUTPUT stream.
        comb_res_mix: ``[T, S, S]`` fp32 -- per token, ``[i, j]`` weights input
            stream ``i`` into output stream ``j``.

    Returns:
        ``[T, S, H]`` in ``residual``'s dtype,
        ``out_j = post_layer_mix_j * x + sum_i comb_ij * res_i``, computed in fp32
        and rounded once. ``x`` and ``residual`` may be fp32 or bf16.

    On the vector engine the arithmetic is the baseline kernel's, in its
    order and rounding: ``acc = post_j * x`` and then ``acc = acc + comb_ij * res_i``
    for ``i = 0..S-1``, each product rounded to fp32 before its add. The form is
    picked at trace time from ``T`` and the dtypes:

    * ``T <= HIDDEN_ON_PARTITIONS_MAX_TOKENS`` (decode): hidden on the
      partitions, see :func:`_combine_hidden_on_partitions`. All 128 lanes work
      however few tokens there are, and a token chunk is ``1 + 2*S``
      instructions, where baseline ``mhc_5938748`` issued ``S * (1 + 2*S)`` per
      2048-wide hidden tile on ``T`` lanes (72 at decode, on 1 lane).
    * Larger ``T`` with bf16 ``x`` and streams (prefill): the tensor engine, see
      :func:`_combine_on_tensor_engine`. Its products are exact and its sums are
      fp32, but added in the engine's order, so an output can differ from the
      vector form: by one step of the output dtype where the fp32 value lies within
      that order's rounding of a rounding boundary, and by more steps of its own
      size where the terms cancel. Both forms keep the bound stated there.
    * Larger ``T`` otherwise (fp32 operands: decode and verify batches): tokens on
      the partitions, see :func:`_combine_tokens_on_partitions`. Every DMA moves
      long contiguous hidden runs, which the hidden-on-partitions layout cannot
      (its run is ``H / 128`` values, 64 bytes per program at ``H = 4096``), and a
      tile is ``S * (1 + S)`` instructions. The switch point is measured; see
      :data:`HIDDEN_ON_PARTITIONS_MAX_TOKENS`.

    SPMD: under a 2-program launch each program takes one contiguous half of
    the hidden axis, so both physical cores of an LNC2 logical core work. One
    program covers the whole axis.
    """
    t_extent, s_extent, h_extent = residual.shape
    n_prog = nl.num_programs(axes=0)
    prog = nl.program_id(0)

    out = nl.ndarray(
        (t_extent, s_extent, h_extent), dtype=residual.dtype, buffer=nl.shared_hbm
    )

    # This program's contiguous hidden range [h_lo, h_lo + width).
    share = (h_extent + n_prog - 1) // n_prog
    h_lo = prog * share
    width = h_extent - h_lo
    if width > share:
        width = share

    if t_extent <= HIDDEN_ON_PARTITIONS_MAX_TOKENS:
        _combine_hidden_on_partitions(
            x, residual, post_layer_mix, comb_res_mix, out, h_lo, width
        )
    elif x.dtype == nl.bfloat16 and residual.dtype == nl.bfloat16:
        _combine_on_tensor_engine(
            x, residual, post_layer_mix, comb_res_mix, out, h_lo, width
        )
    else:
        _combine_tokens_on_partitions(
            x, residual, post_layer_mix, comb_res_mix, out, h_lo, width
        )
    return out


def _combine_hidden_on_partitions(
    x, residual, post_layer_mix, comb_res_mix, out, h_lo, width
) -> None:
    """The combine over hidden ``[h_lo, h_lo + width)``, hidden on the partitions.

    Traced inside :func:`hyper_connection_kernel`; writes ``out``. Hidden index
    ``h`` lives on partition ``(h - h_lo) // HF`` at free offset
    ``(h - h_lo) % HF``, ``HF = ceil(width / 128)``, so every partition holds one
    contiguous run of each row and all 128 partitions work. Tokens and streams
    ride the free axes as ``[tokens, S, HF]``. The ``S + S*S`` mixing scalars of
    each token are the same for every hidden index, so they are DMA-broadcast to
    every partition once (a stride-0 partition read) and broadcast along ``HF``
    by stride-0 views. One instruction covers every output stream ``j`` and
    every token of the chunk (``[128, tokens, S, HF]``), so a chunk is
    ``1 + 2*S`` instructions, 9 at ``S = 4``.

    ``T`` is walked in chunks sized by :data:`SBUF_CHUNK_ELEMS`, reusing one
    set of tiles. A ``width`` that 128 does not divide leaves a short last
    partition; its unused tail is zeroed and never stored.
    """
    t_extent, s_extent, h_extent = residual.shape
    hf = (width + PARTITION_MAX - 1) // PARTITION_MAX
    full_parts = width // hf
    tail = width - full_parts * hf
    parts = full_parts
    if tail > 0:
        parts = full_parts + 1

    # Tokens per chunk: as many as SBUF_CHUNK_ELEMS fp32 per partition holds
    # across the six tiles below (x, the streams, acc, term and the scalars).
    chunk = SBUF_CHUNK_ELEMS // (hf * (1 + 3 * s_extent) + s_extent + s_extent * s_extent)
    if chunk < 1:
        chunk = 1
    if chunk > t_extent:
        chunk = t_extent
    n_chunks = (t_extent + chunk - 1) // chunk
    mix = s_extent * s_extent

    x_t = nl.ndarray((parts, chunk, hf), dtype=x.dtype, buffer=nl.sbuf)
    res_t = nl.ndarray((parts, chunk, s_extent, hf), dtype=residual.dtype, buffer=nl.sbuf)
    acc = nl.ndarray((parts, chunk, s_extent, hf), dtype=nl.float32, buffer=nl.sbuf)
    term = nl.ndarray((parts, chunk, s_extent, hf), dtype=nl.float32, buffer=nl.sbuf)
    # The last add rounds into the output dtype; an fp32 output is the accumulator.
    fin = acc
    if out.dtype != nl.float32:
        fin = nl.ndarray((parts, chunk, s_extent, hf), dtype=out.dtype, buffer=nl.sbuf)
    post_b = nl.ndarray((parts, chunk, s_extent), dtype=nl.float32, buffer=nl.sbuf)
    comb_b = nl.ndarray(
        (parts, chunk, s_extent, s_extent), dtype=nl.float32, buffer=nl.sbuf
    )
    if tail > 0:
        # The short last partition's unused columns are computed but never
        # stored; zero them so they hold defined values.
        nisa.memset(dst=x_t, value=0.0)
        nisa.memset(dst=res_t, value=0.0)

    for c in range(n_chunks):
        t0 = c * chunk
        tc = t_extent - t0
        if tc > chunk:
            tc = chunk

        # The chunk's mixing scalars, the same on every partition.
        nisa.dma_copy(
            dst=post_b[0:parts, 0:tc, 0:s_extent],
            src=post_layer_mix.ap(
                pattern=[[0, parts], [s_extent, tc], [1, s_extent]],
                offset=t0 * s_extent,
            ),
        )
        nisa.dma_copy(
            dst=comb_b[0:parts, 0:tc, 0:s_extent, 0:s_extent],
            src=comb_res_mix.ap(
                pattern=[[0, parts], [mix, tc], [s_extent, s_extent], [1, s_extent]],
                offset=t0 * mix,
            ),
        )
        # x and the streams, hidden on the partitions.
        nisa.dma_copy(
            dst=x_t[0:full_parts, 0:tc, 0:hf],
            src=x.ap(
                pattern=[[hf, full_parts], [h_extent, tc], [1, hf]],
                offset=t0 * h_extent + h_lo,
            ),
        )
        nisa.dma_copy(
            dst=res_t[0:full_parts, 0:tc, 0:s_extent, 0:hf],
            src=residual.ap(
                pattern=[
                    [hf, full_parts],
                    [s_extent * h_extent, tc],
                    [h_extent, s_extent],
                    [1, hf],
                ],
                offset=t0 * s_extent * h_extent + h_lo,
            ),
        )
        if tail > 0:
            nisa.dma_copy(
                dst=x_t[full_parts:parts, 0:tc, 0:tail],
                src=x.ap(
                    pattern=[[hf, 1], [h_extent, tc], [1, tail]],
                    offset=t0 * h_extent + h_lo + full_parts * hf,
                ),
            )
            nisa.dma_copy(
                dst=res_t[full_parts:parts, 0:tc, 0:s_extent, 0:tail],
                src=residual.ap(
                    pattern=[
                        [hf, 1],
                        [s_extent * h_extent, tc],
                        [h_extent, s_extent],
                        [1, tail],
                    ],
                    offset=t0 * s_extent * h_extent + h_lo + full_parts * hf,
                ),
            )

        acc_c = acc[0:parts, 0:tc, 0:s_extent, 0:hf]
        term_c = term[0:parts, 0:tc, 0:s_extent, 0:hf]
        fin_c = fin[0:parts, 0:tc, 0:s_extent, 0:hf]
        # acc[.., j, :] = post_j * x: x broadcast over j, post_j over hidden.
        nisa.tensor_tensor(
            dst=acc_c,
            data1=x_t[0:parts, 0:tc, 0:hf].expand_dim(2).broadcast(2, s_extent),
            data2=post_b[0:parts, 0:tc, 0:s_extent].expand_dim(3).broadcast(3, hf),
            op=nl.multiply,
        )
        for i in range(s_extent):
            # term[.., j, :] = comb_ij * res_i, for every j at once.
            nisa.tensor_tensor(
                dst=term_c,
                data1=res_t[0:parts, 0:tc, i, 0:hf].expand_dim(2).broadcast(2, s_extent),
                data2=comb_b[0:parts, 0:tc, i, 0:s_extent].expand_dim(3).broadcast(3, hf),
                op=nl.multiply,
            )
            nisa.tensor_tensor(dst=fin_c if i == s_extent - 1 else acc_c, data1=acc_c,
                               data2=term_c, op=nl.add)

        nisa.dma_copy(
            dst=out.ap(
                pattern=[
                    [hf, full_parts],
                    [s_extent * h_extent, tc],
                    [h_extent, s_extent],
                    [1, hf],
                ],
                offset=t0 * s_extent * h_extent + h_lo,
            ),
            src=fin[0:full_parts, 0:tc, 0:s_extent, 0:hf],
        )
        if tail > 0:
            nisa.dma_copy(
                dst=out.ap(
                    pattern=[
                        [hf, 1],
                        [s_extent * h_extent, tc],
                        [h_extent, s_extent],
                        [1, tail],
                    ],
                    offset=t0 * s_extent * h_extent + h_lo + full_parts * hf,
                ),
                src=fin[full_parts:parts, 0:tc, 0:s_extent, 0:tail],
            )


def _combine_tokens_on_partitions(
    x, residual, post_layer_mix, comb_res_mix, out, h_lo, width
) -> None:
    """The combine over hidden ``[h_lo, h_lo + width)``, tokens on the partitions.

    Traced inside :func:`hyper_connection_kernel`; writes ``out``. Up to 128
    tokens per tile on the partitions, hidden on the free axis in slices of
    :data:`TOKEN_LAYOUT_HIDDEN_SLICE`, so every DMA row is one contiguous run of
    that many values. The mixing scalars are per-partition operands. Per output
    stream ``j``: ``acc_j = post_j * x`` (``tensor_scalar``), then
    ``acc_j = comb_ij * res_i + acc_j`` for each ``i`` in one
    ``scalar_tensor_tensor`` (multiply, then add), so a slice is ``S * (1 + S)``
    instructions where baseline ``mhc_5938748`` issued ``S * (1 + 2*S)``. The tiles are
    allocated per slice so the compiler can overlap one slice's DMA with another's
    compute.
    """
    t_extent, s_extent, h_extent = residual.shape
    n_tiles = (t_extent + PARTITION_MAX - 1) // PARTITION_MAX
    n_slices = (width + TOKEN_LAYOUT_HIDDEN_SLICE - 1) // TOKEN_LAYOUT_HIDDEN_SLICE

    for t in range(n_tiles):
        off = t * PARTITION_MAX
        rows = t_extent - off
        if rows > PARTITION_MAX:
            rows = PARTITION_MAX

        post_t = nl.ndarray((rows, s_extent), dtype=nl.float32, buffer=nl.sbuf)
        comb_t = nl.ndarray((rows, s_extent, s_extent), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=post_t,
            src=post_layer_mix.ap(
                pattern=[[s_extent, rows], [1, s_extent]], offset=off * s_extent
            ),
        )
        nisa.dma_copy(
            dst=comb_t,
            src=comb_res_mix.ap(
                pattern=[
                    [s_extent * s_extent, rows],
                    [s_extent, s_extent],
                    [1, s_extent],
                ],
                offset=off * s_extent * s_extent,
            ),
        )

        for sl in range(n_slices):
            h0 = sl * TOKEN_LAYOUT_HIDDEN_SLICE
            w = width - h0
            if w > TOKEN_LAYOUT_HIDDEN_SLICE:
                w = TOKEN_LAYOUT_HIDDEN_SLICE
            x_t = nl.ndarray((rows, w), dtype=x.dtype, buffer=nl.sbuf)
            res_t = nl.ndarray((rows, s_extent, w), dtype=residual.dtype, buffer=nl.sbuf)
            acc = nl.ndarray((rows, s_extent, w), dtype=nl.float32, buffer=nl.sbuf)
            fin = acc
            if out.dtype != nl.float32:
                fin = nl.ndarray((rows, s_extent, w), dtype=out.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=x_t,
                src=x.ap(
                    pattern=[[h_extent, rows], [1, w]],
                    offset=off * h_extent + h_lo + h0,
                ),
            )
            nisa.dma_copy(
                dst=res_t,
                src=residual.ap(
                    pattern=[[s_extent * h_extent, rows], [h_extent, s_extent], [1, w]],
                    offset=off * s_extent * h_extent + h_lo + h0,
                ),
            )
            for j in range(s_extent):
                nisa.tensor_scalar(
                    dst=acc[0:rows, j, 0:w],
                    data=x_t,
                    op0=nl.multiply,
                    operand0=post_t[0:rows, j:j + 1],
                )
                for i in range(s_extent):
                    nisa.scalar_tensor_tensor(
                        dst=(fin if i == s_extent - 1 else acc)[0:rows, j, 0:w],
                        data=res_t[0:rows, i, 0:w],
                        op0=nl.multiply,
                        operand0=comb_t[0:rows, i, j:j + 1],
                        op1=nl.add,
                        operand1=acc[0:rows, j, 0:w],
                    )
            nisa.dma_copy(
                dst=out.ap(
                    pattern=[[s_extent * h_extent, rows], [h_extent, s_extent], [1, w]],
                    offset=off * s_extent * h_extent + h_lo + h0,
                ),
                src=fin,
            )


def _combine_on_tensor_engine(
    x, residual, post_layer_mix, comb_res_mix, out, h_lo, width
) -> None:
    """The combine over hidden ``[h_lo, h_lo + width)`` as block-diagonal matmuls.

    Traced inside :func:`hyper_connection_kernel` for bf16 ``x`` and streams; writes
    ``out``. Tokens are taken in groups of ``G = 128 // (S + 1)``. For a group the
    moving operand is ``G * (S + 1)`` contraction rows: the group's ``G * S``
    stream rows (``residual`` viewed as ``[T * S, H]``, row ``t * S + i``) and then
    its ``G`` rows of ``x``, each one contiguous hidden run in HBM. The stationary
    operand is ``[G * (S + 1), G * S]`` with ``comb[t, i, j]`` at row ``t * S + i``,
    column ``t * S + j`` and ``post[t, j]`` at row ``G * S + t``, column
    ``t * S + j``, zero elsewhere, so result row ``t * S + j`` is
    ``post_j * x_t + sum_i comb_ij * res_ti``: output row ``(t, j)`` of ``out``
    viewed as ``[T * S, H]``, stored as one contiguous run per row.

    The stationaries are built :data:`TENSOR_ENGINE_GROUP_SPAN` groups at a time (see
    :func:`_stationary_pieces`), each split into :data:`TENSOR_ENGINE_PIECES` bf16
    pieces; the next span's pieces are issued before the current span's groups, so
    they are built while the tensor engine runs the current span. Each piece is one
    matmul pass into the group's fp32 PSUM banks (one per ``gemm_moving_fmax`` hidden
    columns), largest piece first. A bf16 product is exact in fp32, and every result
    element sums ``S + 1`` nonzero products per piece, so the result is the exact
    ``post_j * x + sum_i comb_ij * res_i`` up to fp32 additions in the engine's
    order, rounded once into the output dtype when the scalar engine copies the
    banks out (the vector engine, which builds the pieces, is left out of the
    per-group path).

    A short last group zeroes its moving tile first, so rows past ``T`` contribute
    exact zeros, and its rows past ``T`` are never stored.

    The stream loads and the output stores use static DMA descriptors
    (``dge_mode=nisa.dge_mode.none``): their descriptors are built before the launch,
    not during the call by the software descriptor generator on the GpSimd engine,
    which would otherwise build them for most of the call's bytes and pace the call.
    This is correct for every shape the kernel takes, because each copy's offset and
    access pattern are trace-time constants (``g``, ``tokens``, ``h_lo``, ``width`` and
    the shapes are Python integers): a static descriptor moves the same bytes as a
    generated one, and a copy does not change a value. A program issues one load and
    one store per group, ``ceil(T / G)`` of each.

    The bound. Let ``y`` be the exact ``post_j * x + sum_i comb_ij * res_i``,
    ``m = |post_j * x| + sum_i |comb_ij * res_i|``, ``u = 2**-24`` and
    ``g = (P * (S + 1) - 1) * u`` with ``P =`` :data:`TENSOR_ENGINE_PIECES`. This
    form's fp32 sum and the vector forms' each lie within ``g * m`` of ``y``, so each
    output lies within ``step(|y| + g * m) / 2 + g * m`` of ``y``, where ``step(v)``
    is the output dtype's step at ``v``, and the two forms' outputs within twice
    that of each other. Where the terms cancel (``|y|`` far below ``m``) the
    ``g * m`` term spans many steps of ``|y|``.
    """
    t_extent, s_extent, h_extent = residual.shape
    group = PARTITION_MAX // (s_extent + 1)
    rows_out = group * s_extent
    rows_in = rows_out + group
    n_groups = (t_extent + group - 1) // group
    bank = nl.tile_size.gemm_moving_fmax
    n_banks = (width + bank - 1) // bank

    # mask[k, u] = 1 where contraction row k is token u's: a stream row with
    # 0 <= k - S*u <= S - 1, or the x row k = G*S + u.
    ramp = nl.ndarray((rows_in, group), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=ramp, pattern=[[-s_extent, group]], offset=0, channel_multiplier=1)
    above = nl.ndarray((rows_in, group), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=above, data=ramp, op0=nl.greater_equal, operand0=0.0)
    below = nl.ndarray((rows_in, group), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=below, data=ramp, op0=nl.less_equal,
                       operand0=float(s_extent - 1))
    mask = nl.ndarray((rows_in, group), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=mask, data1=above, data2=below, op=nl.multiply)
    x_ramp = nl.ndarray((rows_in, group), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=x_ramp, pattern=[[-1, group]], offset=-rows_out, channel_multiplier=1)
    x_rows = nl.ndarray((rows_in, group), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=x_rows, data=x_ramp, op0=nl.equal, operand0=0.0)
    nisa.tensor_tensor(dst=mask, data1=mask, data2=x_rows, op=nl.add)

    first = n_groups
    if first > TENSOR_ENGINE_GROUP_SPAN:
        first = TENSOR_ENGINE_GROUP_SPAN
    upcoming = _stationary_pieces(post_layer_mix, comb_res_mix, mask, 0, first)
    for g0 in range(0, n_groups, TENSOR_ENGINE_GROUP_SPAN):
        span = n_groups - g0
        if span > TENSOR_ENGINE_GROUP_SPAN:
            span = TENSOR_ENGINE_GROUP_SPAN
        pieces = upcoming
        # The next span's pieces, issued ahead of this span's groups.
        g1 = g0 + TENSOR_ENGINE_GROUP_SPAN
        if g1 < n_groups:
            later = n_groups - g1
            if later > TENSOR_ENGINE_GROUP_SPAN:
                later = TENSOR_ENGINE_GROUP_SPAN
            upcoming = _stationary_pieces(post_layer_mix, comb_res_mix, mask, g1, later)
        for g in range(g0, g0 + span):
            tokens = t_extent - g * group
            if tokens > group:
                tokens = group
            moving = nl.ndarray((rows_in, width), dtype=residual.dtype, buffer=nl.sbuf)
            if tokens < group:
                nisa.memset(dst=moving, value=0.0)
            nisa.dma_copy(
                dge_mode=nisa.dge_mode.none,
                dst=moving[0:tokens * s_extent, 0:width],
                src=residual.ap(pattern=[[h_extent, tokens * s_extent], [1, width]],
                                offset=g * rows_out * h_extent + h_lo),
            )
            nisa.dma_copy(
                dst=moving[rows_out:rows_out + tokens, 0:width],
                src=x.ap(pattern=[[h_extent, tokens], [1, width]],
                         offset=g * group * h_extent + h_lo),
            )
            # One PSUM tile per bank, so each bank is its own accumulation group.
            accs = []
            for _ in range(n_banks):
                accs.append(nl.ndarray((rows_out, bank), dtype=nl.float32,
                                       buffer=nl.psum))
            for n in range(TENSOR_ENGINE_PIECES):
                for b in range(n_banks):
                    w = width - b * bank
                    if w > bank:
                        w = bank
                    nisa.nc_matmul(
                        dst=accs[b][0:rows_out, 0:w],
                        stationary=pieces[n][0:rows_in, g - g0, 0:rows_out],
                        moving=moving[0:rows_in, b * bank:b * bank + w],
                        accumulate=(n > 0),
                    )
            mixed = nl.ndarray((rows_out, width), dtype=out.dtype, buffer=nl.sbuf)
            for b in range(n_banks):
                w = width - b * bank
                if w > bank:
                    w = bank
                nisa.activation(dst=mixed[0:rows_out, b * bank:b * bank + w],
                                op=nl.copy, data=accs[b][0:rows_out, 0:w])
            nisa.dma_copy(
                dge_mode=nisa.dge_mode.none,
                dst=out.ap(pattern=[[h_extent, tokens * s_extent], [1, width]],
                           offset=g * rows_out * h_extent + h_lo),
                src=mixed[0:tokens * s_extent, 0:width],
            )


def _stationary_pieces(post_layer_mix, comb_res_mix, mask, g0, span):
    """The bf16 pieces of the stationary operands of token groups ``[g0, g0 + span)``.

    Traced inside :func:`_combine_on_tensor_engine`, whose layout this follows;
    ``mask`` is its ``[G * (S + 1), G]`` block mask. Returns
    :data:`TENSOR_ENGINE_PIECES` tiles ``[G * (S + 1), span, G * S]``, largest first,
    ``piece_n[k, g, t * S + j] = wpiece_n[k, g, j] * mask[k, t]``. ``weights[k, g, j]``
    is ``comb[(g0 + g) * G + k // S, k % S, j]`` for ``k < G * S``, then
    ``post[(g0 + g) * G + k - G * S, j]``, loaded with the contraction row on the
    partition (a short last group's missing rows stay zero), and ``wpiece_n =
    bf16(weights - sum of the earlier pieces)``, each difference exact in fp32. The
    split runs on the ``S`` weights of a row, ``1 / G`` of the elements of the
    ``G * S`` columns the mask then spreads them over; a product with a 0/1 mask is
    exact, so the pieces are those of the masked weights.

    The weight loads move one ``S``-value element per contraction row and group, and
    their descriptors are built before the launch (``dge_mode.none``), off the queues
    that carry the streams.
    """
    t_extent, s_extent, _ = comb_res_mix.shape
    rows_in, group = mask.shape
    rows_out = group * s_extent
    first = g0 * group
    full = (t_extent - first) // group
    if full > span:
        full = span
    tail = t_extent - first - full * group
    if full == span:
        tail = 0

    weights = nl.ndarray((rows_in, span, s_extent), dtype=nl.float32, buffer=nl.sbuf)
    if tail > 0:
        nisa.memset(dst=weights, value=0.0)
    if full > 0:
        nisa.dma_copy(
            dge_mode=nisa.dge_mode.none,
            dst=weights[0:rows_out, 0:full, 0:s_extent],
            src=comb_res_mix.ap(
                pattern=[[s_extent, rows_out], [rows_out * s_extent, full],
                         [1, s_extent]],
                offset=first * s_extent * s_extent,
            ),
        )
        nisa.dma_copy(
            dge_mode=nisa.dge_mode.none,
            dst=weights[rows_out:rows_in, 0:full, 0:s_extent],
            src=post_layer_mix.ap(
                pattern=[[s_extent, group], [group * s_extent, full], [1, s_extent]],
                offset=first * s_extent,
            ),
        )
    if tail > 0:
        last = first + full * group
        nisa.dma_copy(
            dge_mode=nisa.dge_mode.none,
            dst=weights[0:tail * s_extent, full, 0:s_extent],
            src=comb_res_mix.ap(pattern=[[s_extent, tail * s_extent], [1, s_extent]],
                                offset=last * s_extent * s_extent),
        )
        nisa.dma_copy(
            dge_mode=nisa.dge_mode.none,
            dst=weights[rows_out:rows_out + tail, full, 0:s_extent],
            src=post_layer_mix.ap(pattern=[[s_extent, tail], [1, s_extent]],
                                  offset=last * s_extent),
        )
    spread = mask.expand_dim(1).expand_dim(3).broadcast(1, span).broadcast(3, s_extent)
    pieces = []
    rest = weights
    for n in range(TENSOR_ENGINE_PIECES):
        wpiece = nl.ndarray((rows_in, span, s_extent), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=wpiece, src=rest)
        piece = nl.ndarray((rows_in, span, rows_out), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=piece.reshape((rows_in, span, group, s_extent)),
            data1=wpiece.expand_dim(2).broadcast(2, group),
            data2=spread,
            op=nl.multiply,
        )
        pieces.append(piece)
        if n < TENSOR_ENGINE_PIECES - 1:
            left = nl.ndarray((rows_in, span, s_extent), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=left, data1=rest, data2=wpiece, op=nl.subtract)
            rest = left
    return pieces


def _require_admissible(
    x: Tensor, residual: Tensor, post_layer_mix: Tensor, comb_res_mix: Tensor
) -> tuple[int, int, int]:
    """Every rank and extent condition the kernel imposes, checked in one place.

    Returns:
        ``(T, S, H)`` once every condition holds.
    """
    problems: list[str] = []

    if residual.dim() != 3:
        raise HyperConnectionError(
            f"residual must be 3-D [T, S, H], got shape {tuple(residual.shape)}; "
            f"T maps onto the partition axis, S is the stream axis and H the free "
            f"axis"
        )
    rows, streams, hidden = (int(v) for v in residual.shape)

    # There is no upper bound on T: the kernel walks the token axis in
    # `PARTITION_MAX` tiles, so that constant is the tile height and not a ceiling.
    if rows <= 0:
        problems.append(f"T={rows} must be positive")
    if streams <= 0:
        problems.append(f"S={streams} must be positive")
    if hidden <= 0:
        problems.append(f"H={hidden} must be positive")

    if x.dim() != 2:
        problems.append(
            f"x must be 2-D [T, H], got shape {tuple(x.shape)}"
        )
    elif tuple(int(v) for v in x.shape) != (rows, hidden):
        problems.append(
            f"x has shape {tuple(x.shape)}, expected [T, H] = [{rows}, {hidden}] "
            f"to match residual"
        )

    if post_layer_mix.dim() != 3 or tuple(
        int(v) for v in post_layer_mix.shape
    ) != (rows, streams, 1):
        problems.append(
            f"post_layer_mix has shape {tuple(post_layer_mix.shape)}, expected "
            f"[T, S, 1] = [{rows}, {streams}, 1] -- one scalar per token per "
            f"OUTPUT stream"
        )

    if comb_res_mix.dim() != 3 or tuple(
        int(v) for v in comb_res_mix.shape
    ) != (rows, streams, streams):
        problems.append(
            f"comb_res_mix has shape {tuple(comb_res_mix.shape)}, expected "
            f"[T, S, S] = [{rows}, {streams}, {streams}] -- per token, [i, j] "
            f"weights input stream i into output stream j"
        )

    if problems:
        raise HyperConnectionError(
            "mHC combine refuses this geometry: " + "; ".join(problems)
        )
    return rows, streams, hidden


@dataclass
class _DispatchCounters:
    """Which path actually ran, counted rather than inferred.

    ``nki_dispatch`` counts entries into the ``wrap_nki`` call, ``torch_fallback``
    entries into the torch path. Two counters rather than one flag, so "the kernel
    ran" and "the fallback did not" are independent readings.
    """

    nki_dispatch: int = 0
    torch_fallback: int = 0


#: Module level so a caller outside this module can reset and read it. A distinct
#: object from Sinkhorn's: a caller that wires both halves of mHC reads two
#: numbers, so the two must not share one counter.
_COUNTERS = _DispatchCounters()


def reset_dispatch_counters() -> None:
    """Zero both counters."""
    _COUNTERS.nki_dispatch = 0
    _COUNTERS.torch_fallback = 0


def dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last reset."""
    return _COUNTERS.nki_dispatch, _COUNTERS.torch_fallback


@torch._dynamo.assume_constant_result
def _count_torch_fallback() -> None:
    """Count one torch-path entry, off the traced graph."""
    _COUNTERS.torch_fallback += 1


@torch._dynamo.assume_constant_result
def _count_nki_dispatch() -> None:
    """Count one kernel dispatch, off the traced graph: a store Dynamo reads becomes a guard."""
    _COUNTERS.nki_dispatch += 1


def can_run_hyper_connection(
    x: Tensor, residual: Tensor, post_layer_mix: Tensor, comb_res_mix: Tensor
) -> bool:
    """Is the NKI path available *and* admissible for these shapes?

    Two independent conditions: ``can_run_kernel`` answers whether a device or
    simulator exists, :func:`_require_admissible` whether this kernel accepts these
    extents. A geometry the kernel cannot serve raises rather than falling back.

    Raises:
        HyperConnectionError: if any rank or extent is inadmissible.
    """
    _require_admissible(x, residual, post_layer_mix, comb_res_mix)
    return can_run_kernel(residual)


def hyper_connection_combine(
    x: Tensor,
    residual: Tensor,
    post_layer_mix: Tensor,
    comb_res_mix: Tensor,
) -> Tensor:
    """The mHC combine: one kernel dispatch per call.

    Argument names and order match upstream's ``mhc_post``, so a layer wires a
    call rather than a translation.

    Args:
        x: ``[T, H]`` -- the sub-block's single-stream output.
        residual: ``[T, S, H]`` -- the ``S = hc_mult`` residual streams.
        post_layer_mix: ``[T, S, 1]`` -- per token, per output stream.
        comb_res_mix: ``[T, S, S]`` -- ``[i, j]``: input stream ``i`` into output
            stream ``j``.

    Returns:
        ``[T, S, H]`` in ``residual``'s dtype, computed in fp32 (fp32 or bf16
        ``x`` and ``residual``; see :func:`hyper_connection_kernel`).

    Raises:
        HyperConnectionError: on an inadmissible rank or extent.
    """
    if not can_run_hyper_connection(x, residual, post_layer_mix, comb_res_mix):
        _count_torch_fallback()
        logger.debug(
            "hyper_connection_combine: NKI route unavailable, using the torch "
            "path (oracle only, not the shipped path)"
        )
        return hyper_connection_torch_oracle(
            x, residual, post_layer_mix, comb_res_mix
        ).to(residual.dtype)

    _count_nki_dispatch()
    call = wrap_nki(hyper_connection_kernel)
    programs = launch_programs(int(residual.shape[2]))
    if programs > 1:
        call = call[programs]
    return call(
        x=x,
        residual=residual,
        post_layer_mix=post_layer_mix,
        comb_res_mix=comb_res_mix,
    )


def launch_programs(hidden: int) -> int:
    """SPMD programs for one combine launch: 2 under LNC2, else 1.

    Under an LNC2 pair (``NEURON_LOGICAL_NC_CONFIG=2``, the serving setting, read
    through :func:`~vllm_neuron.functional.dsa.launch_grid.lnc_pair`) the two
    programs split the hidden axis, one contiguous half each, so both physical
    cores work. A hidden extent narrower than two partitions' worth stays on one
    program.

    Raises:
        LaunchGridError: ``NEURON_LOGICAL_NC_CONFIG`` is set and is not 1 or 2.
    """
    if lnc_pair() and hidden >= 2 * PARTITION_MAX:
        return 2
    return 1


def hyper_connection_torch_oracle(
    x: Tensor,
    residual: Tensor,
    post_layer_mix: Tensor,
    comb_res_mix: Tensor,
) -> Tensor:
    """The same operation in torch, in fp32. The CPU oracle, never shipped.

    Written in upstream's ``einsum`` spelling rather than its
    ``bmm(comb.mT, residual)`` one, so that the two independent statements of the
    ``i``/``j`` convention can be cross-checked instead of one file agreeing with
    itself.

    Independent of the kernel where it matters: ``einsum`` contracts the stream
    axis in one call, where the kernel accumulates ``S`` per-token scalar
    broadcasts in a fixed order, so the two round differently.

    Kept in fp32 rather than cast to ``residual.dtype`` as upstream does, since
    returning bf16 is why upstream's own test compares at ``atol=5e-2``.

    Returns:
        ``[T, S, H]`` fp32.
    """
    mixed_residual = torch.einsum(
        "...ij,...ih->...jh",
        comb_res_mix.to(torch.float32),
        residual.to(torch.float32),
    )
    post_term = post_layer_mix.to(torch.float32) * x.unsqueeze(-2).to(torch.float32)
    return mixed_residual + post_term


def kernel_identity() -> tuple[str, str]:
    """``(module, qualname)`` of the NKI kernel, read off the object.

    Lets a caller check that this module dispatches to the kernel it authors, so a
    substitution shows up as a changed reading.
    """
    func = getattr(hyper_connection_kernel, "func", None)
    target = func if func is not None else hyper_connection_kernel
    return target.__module__, target.__qualname__
