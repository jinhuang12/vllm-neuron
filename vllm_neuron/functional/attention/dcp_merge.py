# SPDX-License-Identifier: Apache-2.0
"""The DCP log-sum-exp merge of per-rank MLA partials, as one NKI kernel.

Under decode context parallelism (DCP) the latent KV of a sequence is interleaved over the
CP ranks of a CP group, and each rank attends with every head of the group (the gathered
query, ``Hq = CP * H``) over the columns it owns. Per (head, row) a rank returns the
**partial contract** (``mla_sparse.mla_sparse_attention_partial``):

* ``partial``: float32, the attention output normalised over this rank's own valid
  columns, head-major ``[Hq, R, L]`` (L = the latent rank, before the W_UV absorb-out);
* ``lse``: float32 ``[Hq, R]``, ``softmax_scale * m + ln(l)`` over those columns, ``m`` the
  row maximum of the raw scores and ``l`` the sum of ``exp(softmax_scale * (s - m))``;
* an **empty** partial (no column of the row is this rank's): ``partial`` exactly 0 and
  ``lse`` = :data:`EMPTY_LSE`.

The collective hands each rank the CP partials of its own H heads, ``[CP, H, R, L]`` with
the source rank leading (an all-to-all over dim 0 of the head-major partials: no transposing
copy), and this kernel merges them::

    m        = max_c lse[c, h, r]
    w[c]     = exp(lse[c, h, r] - m)
    out[r, h] = bf16( sum_c (w[c] / sum_c' w[c']) * partial[c, h, r] )

That is llama3's merge (``attention_decode.py``: ``global_lse = logsumexp_c lse``,
``out = sum_c partial_c * exp(lse_c - global_lse)``), with the log-sum-exp's own max shift
kept apart: ``exp(lse_c - global_lse) = w[c] / sum w``. It is exact in the latent rank
because the absorb-out after it is linear per head. The result is rounded to bf16 once,
where the CP=1 path casts its float32 attention output before the absorb-out.

Empty partials need no special case. A rank with ``lse = EMPTY_LSE`` beside a live rank
has weight ``exp(-1e30 - m) = 0`` exactly. A row empty on every rank (a padded row) has
``m = EMPTY_LSE``, every weight ``exp(0) = 1`` and output ``0 / CP = 0``. The constant is
finite on purpose: with ``-inf`` that row would compute ``-inf - (-inf) = NaN``.

Layout: rows ride the partitions in tiles of :data:`ROW_TILE`; the latent rides the free
axis in tiles of :data:`LATENT_CHUNK` (one 2 KiB float32 run per partition per DMA). The CP
lses of a row tile are loaded rank-major (one contiguous run per rank) and put one row per
partition by a PE transpose, which is exact on NeuronCore-v3 and later. Both cores of an
LNC2 pair take contiguous halves of the row tiles (:func:`merge_programs`).

:func:`dcp_lse_merge` is the seam. There is no torch route; a malformed call raises. Its
signature, the partial contract above and :data:`EMPTY_LSE` are a stable interface: the DCP
collective path calls the seam after its all-to-all, and every partial producer (here
``mla_sparse_attention_partial`` and ``mla_dense_window_attention_partial``) emits the
contract. Change them only together with those callers.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.dsa.launch_grid import lnc_pair

#: The lse of an empty partial: no column of the row is this rank's. The magnitude of the
#: masked score elsewhere in this tree (``mla_decode.MASKED_SCORE``), and finite so that a
#: row empty on every rank merges to 0 instead of NaN.
EMPTY_LSE = -1.0e30

#: Rows per tile: the partition axis, ``nl.tile_size.pmax``. The CP ranks of a row tile also
#: ride the partitions once, in the lse transpose, so CP is bounded by it as well.
ROW_TILE = 128

#: Latent columns per tile: 512 float32 elements are one 2 KiB contiguous run per partition,
#: the DMA size that streams at full bandwidth.
LATENT_CHUNK = 512

#: Free-axis alignment of the small tiles, in float32 elements: one whole 32-byte line.
_LINE = 8

#: The partial dtypes the kernel reads: the contract's float32, and bfloat16 for a caller
#: that narrows the wire (each partial is then rounded once before the merge).
PARTIAL_DTYPES = (torch.float32, torch.bfloat16)

#: This file's content digest, handed to the entry point as a trace-time int: the compiled
#: kernel cache keys on the entry's own source, and the arithmetic lives in a helper.
SOURCE_DIGEST = int(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:7], 16)


class DcpMergeError(ValueError):
    """Raised for operands the merge does not serve; there is no torch fallback."""


def _aligned(width: int) -> int:
    """``width`` rounded up to a whole :data:`_LINE`."""
    return (width + _LINE - 1) // _LINE * _LINE


def _sbuf(parts: int, width: int):
    return nl.ndarray((parts, width), dtype=nl.float32, buffer=nl.sbuf)


def _column(parts: int):
    """A ``[parts, 1]`` float32 view of a tile whose row is one whole 32-byte line."""
    return _sbuf(parts, _LINE)[:, 0:1]


def _merge_rows(partials_hbm, lse_hbm, out_hbm, head, r0, rows_here):
    """Merge rows ``r0 .. r0 + rows_here - 1`` of one head into ``out_hbm``.

    Shapes: ``partials_hbm [CP, H, R, L]``, ``lse_hbm [CP, H, R]`` float32, ``out_hbm
    [R, H, L]`` bfloat16; ``rows_here <= ROW_TILE``.
    """
    cp, heads, rows, latent = partials_hbm.shape

    # ---- the rows' lses: rank-major as stored, then one row per partition --------------
    lse_ranks = _sbuf(cp, ROW_TILE)
    nisa.dma_copy(dst=lse_ranks[:, 0:rows_here],
                  src=lse_hbm.ap(pattern=[[heads * rows, cp], [1, rows_here]],
                                 offset=head * rows + r0))
    lse_ps = nl.ndarray((ROW_TILE, cp), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=lse_ps[0:rows_here, 0:cp], data=lse_ranks[:, 0:rows_here])
    lse_rows = _sbuf(ROW_TILE, _aligned(cp))
    nisa.tensor_copy(dst=lse_rows[0:rows_here, 0:cp], src=lse_ps[0:rows_here, 0:cp])

    # ---- the correction per rank: w_c / sum w, w_c = exp(lse_c - max lse) --------------
    # The max is taken negated so `activation` folds the shift into its bias and sums the
    # weights in the same pass.
    neg_max = _column(ROW_TILE)
    nisa.tensor_reduce(dst=neg_max[0:rows_here, :], op=nl.maximum,
                       data=lse_rows[0:rows_here, 0:cp], axis=1, negate=True)
    weight = _sbuf(ROW_TILE, _aligned(cp))
    total = _column(ROW_TILE)
    nisa.activation(dst=weight[0:rows_here, 0:cp], op=nl.exp, data=lse_rows[0:rows_here, 0:cp],
                    bias=neg_max[0:rows_here, :], reduce_op=nl.add,
                    reduce_res=total[0:rows_here, :], reduce_cmd=nisa.reduce_cmd.reset_reduce)
    recip = _column(ROW_TILE)
    nisa.reciprocal(dst=recip[0:rows_here, :], data=total[0:rows_here, :])
    share = _sbuf(ROW_TILE, _aligned(cp))
    nisa.tensor_scalar(dst=share[0:rows_here, 0:cp], data=weight[0:rows_here, 0:cp],
                       op0=nl.multiply, operand0=recip[0:rows_here, :],
                       engine=nisa.engine.vector)

    # ---- the weighted sum over the ranks, one latent tile at a time --------------------
    # Two float32 accumulators alternate so no instruction writes back onto its own
    # operand; the last rank's multiply-add writes the bf16 tile, the one rounding.
    for c0 in range(0, latent, LATENT_CHUNK):
        width = min(LATENT_CHUNK, latent - c0)
        accs = []
        for _ in range(2):
            accs.append(_sbuf(ROW_TILE, LATENT_CHUNK))
        merged = nl.ndarray((ROW_TILE, LATENT_CHUNK), dtype=nl.bfloat16, buffer=nl.sbuf)
        for c in range(cp):
            part = nl.ndarray((ROW_TILE, LATENT_CHUNK), dtype=partials_hbm.dtype,
                              buffer=nl.sbuf)
            nisa.dma_copy(dst=part[0:rows_here, 0:width],
                          src=partials_hbm.ap(pattern=[[latent, rows_here], [1, width]],
                                              offset=((c * heads + head) * rows + r0) * latent
                                              + c0))
            dst = merged if c == cp - 1 else accs[c % 2]
            if c == 0:
                nisa.tensor_scalar(dst=dst[0:rows_here, 0:width], data=part[0:rows_here, 0:width],
                                   op0=nl.multiply, operand0=share[0:rows_here, c:c + 1],
                                   engine=nisa.engine.vector)
            else:
                nisa.scalar_tensor_tensor(dst=dst[0:rows_here, 0:width],
                                          data=part[0:rows_here, 0:width],
                                          op0=nl.multiply, operand0=share[0:rows_here, c:c + 1],
                                          op1=nl.add,
                                          operand1=accs[(c - 1) % 2][0:rows_here, 0:width])
        nisa.dma_copy(dst=out_hbm.ap(pattern=[[heads * latent, rows_here], [1, width]],
                                     offset=(r0 * heads + head) * latent + c0),
                      src=merged[0:rows_here, 0:width])


@nki.jit
def dcp_lse_merge_kernel(partials_hbm, lse_hbm, source_digest=0):
    """Merge ``[CP, H, R, L]`` partials and ``[CP, H, R]`` lses into ``[R, H, L]`` bfloat16.

    The row tiles split into contiguous shares over the programs of the launch grid; the
    last tile may be ragged. ``source_digest`` is :data:`SOURCE_DIGEST` and only keys the
    kernel cache.
    """
    cp, heads, rows, latent = partials_hbm.shape
    out_hbm = nl.ndarray((rows, heads, latent), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    n_prgs = nl.num_programs(axes=0)
    prg = nl.program_id(0)
    n_tiles = (rows + ROW_TILE - 1) // ROW_TILE
    share = (n_tiles + n_prgs - 1) // n_prgs
    first = prg * share
    last = min(n_tiles, first + share)
    for head in range(heads):
        for t in range(first, last):
            r0 = t * ROW_TILE
            _merge_rows(partials_hbm, lse_hbm, out_hbm, head, r0, min(ROW_TILE, rows - r0))
    return out_hbm


def merge_programs(rows: int) -> int:
    """Programs the merge launches: both cores of an LNC2 pair from two row tiles on."""
    if lnc_pair() and (int(rows) + ROW_TILE - 1) // ROW_TILE >= 2:
        return 2
    return 1


def _validate(partials: Tensor, lse: Tensor) -> tuple[int, int, int, int]:
    if partials.ndim != 4:
        raise DcpMergeError(
            f"partials must be [CP, H, R, L], the CP ranks' head-major partials of this "
            f"rank's H heads with the source rank leading; got {tuple(partials.shape)}")
    cp, heads, rows, latent = (int(d) for d in partials.shape)
    if lse.ndim != 3 or tuple(int(d) for d in lse.shape) != (cp, heads, rows):
        raise DcpMergeError(
            f"lse must be [CP, H, R] = {(cp, heads, rows)}, one per partial row; got "
            f"{tuple(lse.shape)}")
    if lse.dtype != torch.float32:
        raise DcpMergeError(
            f"lse must be float32: the merge's exponents are differences of lses of a few "
            f"tens, which a 2-byte float would round by up to 2**-8 of their size; got "
            f"{lse.dtype}")
    if partials.dtype not in PARTIAL_DTYPES:
        raise DcpMergeError(
            f"partials must be float32 (the partial contract) or bfloat16; got "
            f"{partials.dtype}")
    if min(cp, heads, rows, latent) < 1:
        raise DcpMergeError(
            f"partials must carry at least one rank, head, row and latent column; got "
            f"CP={cp}, H={heads}, R={rows}, L={latent}")
    if cp > ROW_TILE:
        raise DcpMergeError(
            f"CP={cp}: the lse transpose puts the CP ranks of a row tile on partitions, at "
            f"most {ROW_TILE} (nl.tile_size.pmax)")
    return cp, heads, rows, latent


def dcp_lse_merge(partials: Tensor, lse: Tensor) -> Tensor:
    """Merge the CP partials of this rank's heads into their attention output.

    A stable interface (module docstring): the DCP collective path calls it as is.

    Args:
        partials: ``[CP, H, R, L]`` float32 (or bfloat16), rank c's partial of head h for
            row r: attention normalised over rank c's valid columns; exactly 0 when empty.
        lse: ``[CP, H, R]`` float32, ``softmax_scale * m + ln(l)`` over the same columns;
            :data:`EMPTY_LSE` when empty.

    Returns:
        ``[R, H, L]`` bfloat16, ``sum_c exp(lse_c - logsumexp_c lse) * partial_c`` rounded
        once. A row empty on every rank is exactly 0.

    Raises:
        DcpMergeError: for operands of the wrong rank, shape or dtype, or CP past
            :data:`ROW_TILE`.
    """
    _cp, _heads, rows, _latent = _validate(partials, lse)
    call = wrap_nki(dcp_lse_merge_kernel)
    if merge_programs(rows) == 2:
        call = call[2]
    return call(partials.contiguous(), lse.contiguous(), SOURCE_DIGEST)
