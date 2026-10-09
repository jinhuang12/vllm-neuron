# SPDX-License-Identifier: Apache-2.0
"""Low-rank MLA projection ``y[S, O] = x[S, I] @ w[I, O]`` as a tiled NKI matmul.

``nc_matmul`` contracts the partition axis, so both operands must present the
contraction extent there. The seam therefore hands the kernel ``x`` already
transposed to ``[I, S]`` and expects the weight contraction-major as ``[I, O]``,
not in torch's ``nn.Linear`` ``[O, I]`` orientation. Accepting ``[O, I]`` would
cost either a per-tile on-device transpose, which caps the output tile at 128
instead of 512 and so quadruples the matmul count, or a host-side copy of up to
64 MB per call. A projection weight is constant, so the caller transposes it once
at load time instead.

There is no bias parameter: this model's config does not carry ``attention_bias``,
so a bias limb would be unreachable. There is no torch projection path either --
an inadmissible geometry raises. ``mla_projection_torch_oracle`` is the CPU
reference for tests and nothing dispatches to it.
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
from nkilib.core.utils.kernel_assert import kernel_assert

from vllm_neuron.functional.dsa.launch_grid import lnc_pair
from vllm_neuron.utils.neuron_utils import can_run_kernel

logger = logging.getLogger(__name__)

#: Partition-axis extent, ``nl.tile_size.pmax``. The contraction axis is tiled to
#: this because ``nc_matmul`` contracts the partition axis.
CONTRACTION_TILE = 128

#: Stationary free-axis extent, ``nl.tile_size.gemm_stationary_fmax``. The sequence
#: axis rides the stationary operand, so this bounds the sequence tile, not the
#: sequence itself, which the loop below walks.
SEQUENCE_TILE = 128

#: Moving free-axis extent, ``nl.tile_size.gemm_moving_fmax``. The output axis rides
#: the moving operand, so output tiles are four times wider than sequence tiles.
#: That asymmetry is the hardware's, not a choice made here.
OUTPUT_TILE = 512

#: Where the three extents above come from. They are declared here rather than
#: imported from ``vllm_neuron/functional/kda/``: two of them equal a constant that
#: package already holds, but an equal number is not the same quantity.
_TILE_PROVENANCE = "nl.tile_size.{pmax, gemm_stationary_fmax, gemm_moving_fmax}"


class MlaProjectionError(ValueError):
    """Raised for a geometry this kernel does not serve."""


@dataclass
class _MlaProjectionDispatchCounters:
    """Per-process record of how the seam below was reached."""

    nki_dispatch: int = 0
    torch_fallback: int = 0


_MLA_PROJECTION_COUNTERS = _MlaProjectionDispatchCounters()


def reset_mla_projection_dispatch_counters() -> None:
    """Zero this seam's counters."""
    _MLA_PROJECTION_COUNTERS.nki_dispatch = 0
    _MLA_PROJECTION_COUNTERS.torch_fallback = 0


def mla_projection_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last reset.

    ``torch_fallback`` always reads ``0``: this module has no torch projection
    route, and an inadmissible geometry raises instead of falling back.
    """
    return (
        _MLA_PROJECTION_COUNTERS.nki_dispatch,
        _MLA_PROJECTION_COUNTERS.torch_fallback,
    )


@torch._dynamo.assume_constant_result
def _count_nki_dispatch() -> None:
    """Count one kernel dispatch, off the traced graph: a store Dynamo traces becomes a guard."""
    _MLA_PROJECTION_COUNTERS.nki_dispatch += 1


def _sbuf(rows: int, cols: int):
    return nl.ndarray((rows, cols), dtype=nl.float32, buffer=nl.sbuf)


def _psum(rows: int, cols: int):
    return nl.ndarray((rows, cols), dtype=nl.float32, buffer=nl.psum)


@nki.jit
def mla_projection_kernel(xt_hbm, w_hbm):
    """``y[S, O] = x[S, I] @ w[I, O]``, tiled on all three axes.

    ``xt_hbm`` is ``x`` already transposed to ``[I, S]`` and ``w_hbm`` is
    ``[I, O]``, so both operands present the contraction extent on the partition
    axis that ``nc_matmul`` contracts.

    Accumulation is in PSUM across the contraction loop -- ``accumulate`` is False
    on the first contraction tile and True on every later one -- so one output tile
    is written to HBM exactly once, fully summed.

    Loop bounds are ``range`` rather than ``nl.affine_range`` so every extent stays
    a trace-time int and the ragged final tile on each axis can be an ordinary
    ``min``; an affine range would make it a trace value and silently require the
    extent to divide exactly.
    """
    idim, seq = xt_hbm.shape
    _, odim = w_hbm.shape
    out = nl.ndarray((seq, odim), dtype=nl.float32, buffer=nl.shared_hbm)

    for s0 in range(0, seq, SEQUENCE_TILE):
        sw = min(SEQUENCE_TILE, seq - s0)
        for o0 in range(0, odim, OUTPUT_TILE):
            ow = min(OUTPUT_TILE, odim - o0)
            acc_ps = _psum(sw, ow)
            for k0 in range(0, idim, CONTRACTION_TILE):
                kw = min(CONTRACTION_TILE, idim - k0)
                x_tile = _sbuf(kw, sw)
                nisa.tensor_copy(
                    dst=x_tile,
                    src=nl.load(xt_hbm[k0:k0 + kw, s0:s0 + sw], dtype=nl.float32),
                )
                w_tile = _sbuf(kw, ow)
                nisa.tensor_copy(
                    dst=w_tile,
                    src=nl.load(w_hbm[k0:k0 + kw, o0:o0 + ow], dtype=nl.float32),
                )
                nisa.nc_matmul(
                    dst=acc_ps,
                    stationary=x_tile,
                    moving=w_tile,
                    accumulate=(k0 > 0),
                )
            out_sb = _sbuf(sw, ow)
            nisa.tensor_copy(dst=out_sb, src=acc_ps)
            nl.store(out[s0:s0 + sw, o0:o0 + ow], value=out_sb)
    return out


def _require_mla_projection_admissible(seq: int, idim: int, odim: int) -> None:
    """Raise unless this kernel serves the geometry.

    Only positivity is checked: the kernel walks all three axes in tiles, so no
    extent has a magnitude limit.
    """
    if seq < 1 or idim < 1 or odim < 1:
        raise MlaProjectionError(
            f"mla_projection needs a positive extent on every axis; got "
            f"seq={seq}, in_features={idim}, out_features={odim}"
        )


def can_run_mla_projection(reference: Tensor, seq: int, idim: int, odim: int) -> bool:
    """True when the NKI route is available and serves this geometry."""
    if not can_run_kernel(reference):
        return False
    try:
        _require_mla_projection_admissible(seq, idim, odim)
    except MlaProjectionError:
        return False
    return True


def mla_projection(x: Tensor, weight: Tensor) -> Tensor:
    """Project ``x [S, I]`` by ``weight [I, O]``, returning ``[S, O]`` in float32.

    ``weight`` is contraction-major, ``[in_features, out_features]``, and not
    torch's ``nn.Linear`` ``[out_features, in_features]``; the module docstring
    gives the reason.

    Raises:
        MlaProjectionError: on a non-2D operand, a weight whose contraction extent
            does not match ``x``, or a non-positive extent.
    """
    if x.ndim != 2:
        raise MlaProjectionError(
            f"x must be [seq, in_features]; got shape {tuple(x.shape)}"
        )
    if weight.ndim != 2:
        raise MlaProjectionError(
            f"weight must be [in_features, out_features]; got shape "
            f"{tuple(weight.shape)}"
        )
    seq, idim = int(x.shape[0]), int(x.shape[1])
    if int(weight.shape[0]) != idim:
        raise MlaProjectionError(
            f"weight is contraction-major, so weight.shape[0] must equal "
            f"x.shape[1]; got weight {tuple(weight.shape)} against x "
            f"{tuple(x.shape)}. A [out_features, in_features] weight is the "
            f"likely cause -- this seam does not accept that orientation"
        )
    odim = int(weight.shape[1])
    _require_mla_projection_admissible(seq, idim, odim)

    _count_nki_dispatch()
    return wrap_nki(mla_projection_kernel)(
        x.t().contiguous().to(torch.float32),
        weight.to(torch.float32),
    )


def mla_projection_torch_oracle(x: Tensor, weight: Tensor) -> Tensor:
    """CPU reference for tests. Nothing dispatches here; the seam above raises instead."""
    return x.to(torch.float32) @ weight.to(torch.float32)


def mla_projection_kernel_identity() -> tuple[str, str]:
    """``(module, qualname)`` of the projection kernel this module defines.

    ``nki.jit`` returns a wrapper whose own ``__module__`` is the decorator's, so
    the identity has to be read off the wrapped function at ``.func``.
    """
    func = getattr(mla_projection_kernel, "func", None)
    target = func if func is not None else mla_projection_kernel
    return target.__module__, target.__qualname__


# ---------------------------------------------------------------------------
# Low-precision projection: fp8 block-quant or bf16 weights, read as stored.
#
# The fp32 kernel above reads a weight four times the size of the checkpoint's own
# fp8 bytes and runs every matmul as a four-pass fp32 product. The decode step
# (one row per request) is bound by those bytes and passes, not by FLOPs. The
# kernel below reads the weight as stored -- fp8-e4m3 with one fp32 dequant
# multiplier per 128 x 128 block, or bf16 -- and runs single-pass matmuls.
#
# Layout. ``x`` rides the stationary operand (``[128, M]``, M <= 128) and the
# weight rides the moving operand (``[128, 128]`` per block, or up to 512 wide
# when unscaled). So the Tensor Engine streams one weight column per output
# column and the row count M is free: B = 1 and B = 64 cost the same PE time.
#
# Scales. A block's dequant multiplier is folded into the stationary operand,
# ``(x_k * s[k, n]) @ w[k, n]`` = ``x_k @ (w[k, n] * s[k, n])``, so PSUM
# accumulates across every contraction block and no per-block vector op touches
# the output. The fold rounds ``x_k * s[k, n]`` to bf16, the stationary dtype.
#
# Both physical cores of an LNC2 core split the work (``[2]`` grid). A decode batch
# (one sequence tile) splits the output columns: each program loads only its half of
# the weight. A prefill chunk splits its rows instead (:func:`lowp_splits_rows`):
# each program transposes only its own rows of ``x`` once, holds every weight column,
# and picks per weight chunk which operand is stationary
# (:func:`lowp_weight_stationary`). Both splits compute every output element with the
# same products in the same order, so they return the same bits.
# ---------------------------------------------------------------------------

#: Weight rows per scale block, the checkpoint's ``weight_block_size``.
LOWP_SCALE_BLOCK = 128

#: Contraction blocks moved per weight DMA. Four blocks of a 768-column fp8 half
#: are 384 KB, large enough to run at bandwidth, small enough to spread over queues.
LOWP_KB_PER_DMA = 4


@dataclass
class _MlaProjectionLowpDispatchCounters:
    nki_dispatch: int = 0
    scaled_dispatch: int = 0
    two_program_dispatch: int = 0


_MLA_PROJECTION_LOWP_COUNTERS = _MlaProjectionLowpDispatchCounters()


def reset_mla_projection_lowp_counts() -> None:
    """Zero the low-precision seam's counters."""
    _MLA_PROJECTION_LOWP_COUNTERS.nki_dispatch = 0
    _MLA_PROJECTION_LOWP_COUNTERS.scaled_dispatch = 0
    _MLA_PROJECTION_LOWP_COUNTERS.two_program_dispatch = 0


def mla_projection_lowp_counts() -> tuple[int, int, int]:
    """``(nki_dispatch, scaled_dispatch, two_program_dispatch)`` since the last reset."""
    return (
        _MLA_PROJECTION_LOWP_COUNTERS.nki_dispatch,
        _MLA_PROJECTION_LOWP_COUNTERS.scaled_dispatch,
        _MLA_PROJECTION_LOWP_COUNTERS.two_program_dispatch,
    )


@torch._dynamo.assume_constant_result
def _count_lowp_dispatch(scaled: bool, programs: int) -> None:
    _MLA_PROJECTION_LOWP_COUNTERS.nki_dispatch += 1
    if scaled:
        _MLA_PROJECTION_LOWP_COUNTERS.scaled_dispatch += 1
    if programs == 2:
        _MLA_PROJECTION_LOWP_COUNTERS.two_program_dispatch += 1


@nki.jit
def mla_projection_lowp_kernel(x_hbm, w_hbm, scale_hbm=None, col_chunk=0):
    """``y[M, N] = x[M, K] @ dequant(w[K, N])`` with the weight read as stored.

    Args:
        x_hbm: ``[M, K]`` bf16 or fp32, ``K`` a multiple of 128 and at most
            ``128 * 128``.
        w_hbm: ``[K, N]`` fp8-e4m3 (with ``scale_hbm``) or bf16 (without).
        scale_hbm: ``[programs, K // 128, N // (128 * programs)]`` fp32 dequant
            multipliers, program-major (see :func:`lowp_scale_layout`), or None.
        col_chunk: weight columns held in SBUF at once, a trace-time int; 0 holds
            the program's whole share (every column when the rows split). The seam
            sizes it to the SBUF budget.

    Returns:
        ``[M, N]`` fp32.

    Rows split over the programs when :func:`lowp_splits_rows` says so
    (:func:`_lowp_project_rows`); else the output columns do, as below.
    """
    m_ext, k_ext = x_hbm.shape
    n_ext = w_hbm.shape[1]
    n_kb = k_ext // CONTRACTION_TILE
    programs = nl.num_programs(axes=0)
    program = nl.program_id(0)
    width = n_ext // programs
    n_lo = program * width
    scaled = scale_hbm is not None
    chunk = width if col_chunk == 0 or col_chunk > width else col_chunk
    kernel_assert(k_ext % CONTRACTION_TILE == 0 and n_kb <= CONTRACTION_TILE,
                  "in_features must be whole 128-row blocks, at most 128 of them")
    kernel_assert(width * programs == n_ext, "out_features must split evenly")
    if scaled:
        kernel_assert(width % LOWP_SCALE_BLOCK == 0 and chunk % LOWP_SCALE_BLOCK == 0,
                      "each program's columns must be whole scale blocks")
        kernel_assert(scale_hbm.shape[0] == programs and scale_hbm.shape[1] == n_kb
                      and scale_hbm.shape[2] * LOWP_SCALE_BLOCK == width,
                      "the scale grid must be program-major for this launch grid")
    out = nl.ndarray((m_ext, n_ext), dtype=nl.float32, buffer=nl.shared_hbm)
    if lowp_splits_rows(m_ext, programs, _lowp_x_row_bytes(x_hbm)):
        _lowp_project_rows(x_hbm, w_hbm, scale_hbm, col_chunk, out)
        return out
    rows_per_transpose = CONTRACTION_TILE // n_kb
    if rows_per_transpose < 1:
        rows_per_transpose = 1

    for q0 in range(0, width, chunk):
        cw = min(chunk, width - q0)
        col0 = n_lo + q0
        # ---- this chunk's weight columns, resident for every row tile ------------
        w_sb = nl.ndarray((CONTRACTION_TILE, n_kb, cw), dtype=w_hbm.dtype,
                          buffer=nl.sbuf)
        for kb0 in range(0, n_kb, LOWP_KB_PER_DMA):
            group = min(LOWP_KB_PER_DMA, n_kb - kb0)
            nisa.dma_copy(
                dst=w_sb[:, kb0:kb0 + group, :],
                src=w_hbm.ap(
                    pattern=[[n_ext, CONTRACTION_TILE],
                             [CONTRACTION_TILE * n_ext, group],
                             [1, cw]],
                    offset=kb0 * CONTRACTION_TILE * n_ext + col0,
                ),
            )
        n_nb = cw // LOWP_SCALE_BLOCK
        if scaled:
            # One partition-broadcast load: every partition holds this chunk's grid.
            # The grid is program-major, so with one chunk each partition reads one
            # contiguous run rather than one short run per contraction block.
            prog_nb = width // LOWP_SCALE_BLOCK
            s_sb = nl.ndarray((CONTRACTION_TILE, n_kb, n_nb), dtype=nl.float32,
                              buffer=nl.sbuf)
            nisa.dma_copy(
                dst=s_sb,
                src=scale_hbm.ap(
                    pattern=[[0, CONTRACTION_TILE], [prog_nb, n_kb], [1, n_nb]],
                    offset=program * n_kb * prog_nb + q0 // LOWP_SCALE_BLOCK,
                ),
            )

        # ---- row tiles: x^T onto the contraction axis, then the matmuls -----------
        for m0 in range(0, m_ext, SEQUENCE_TILE):
            mw = min(SEQUENCE_TILE, m_ext - m0)
            # xt[p, m, kb] = x[m0 + m, kb * 128 + p]: a [rows, 128] view of x, rows
            # ordered (m, kb), transposed on the PE in whole query rows.
            xt = nl.ndarray((CONTRACTION_TILE, mw, n_kb), dtype=nl.bfloat16,
                            buffer=nl.sbuf)
            for r0 in range(0, mw, rows_per_transpose):
                rw = min(rows_per_transpose, mw - r0)
                x_rows = nl.ndarray((rw * n_kb, CONTRACTION_TILE), dtype=x_hbm.dtype,
                                    buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=x_rows,
                    src=x_hbm.ap(pattern=[[CONTRACTION_TILE, rw * n_kb],
                                          [1, CONTRACTION_TILE]],
                                 offset=(m0 + r0) * k_ext),
                )
                # A PE transpose writes PSUM in its input's dtype, in whole 4-byte
                # words: an odd column count of a 2-byte dtype is transposed in fp32
                # (exact) instead.
                t_dtype = x_hbm.dtype
                if (rw * n_kb) % 2 == 1 and x_hbm.dtype != nl.float32:
                    t_dtype = nl.float32
                    x_wide = nl.ndarray((rw * n_kb, CONTRACTION_TILE), dtype=t_dtype,
                                        buffer=nl.sbuf)
                    nisa.tensor_copy(dst=x_wide, src=x_rows)
                    x_rows = x_wide
                t_ps = nl.ndarray((CONTRACTION_TILE, rw * n_kb), dtype=t_dtype,
                                  buffer=nl.psum)
                nisa.nc_transpose(dst=t_ps, data=x_rows)
                nisa.tensor_copy(
                    dst=xt.ap(pattern=[[mw * n_kb, CONTRACTION_TILE], [1, rw * n_kb]],
                              offset=r0 * n_kb),
                    src=t_ps,
                )
            res = nl.ndarray((mw, cw), dtype=nl.float32, buffer=nl.sbuf)
            if scaled:
                # xs[p, kb, nb, m] = xt[p, m, kb] * s[kb, nb]: the block multiplier
                # folded into the stationary operand, one op for every block.
                xs = nl.ndarray((CONTRACTION_TILE, n_kb, n_nb, mw), dtype=nl.bfloat16,
                                buffer=nl.sbuf)
                nisa.tensor_tensor(
                    dst=xs,
                    data1=xt.ap(pattern=[[mw * n_kb, CONTRACTION_TILE], [1, n_kb],
                                         [0, n_nb], [n_kb, mw]]),
                    data2=s_sb.ap(pattern=[[n_kb * n_nb, CONTRACTION_TILE], [n_nb, n_kb],
                                           [1, n_nb], [0, mw]]),
                    op=nl.multiply,
                )
                # One accumulation group per output block: its own PSUM tile, every
                # contraction block summed into it, then one copy out.
                for nb in range(n_nb):
                    c0 = nb * LOWP_SCALE_BLOCK
                    acc_ps = _psum(mw, LOWP_SCALE_BLOCK)
                    for kb in range(n_kb):
                        nisa.nc_matmul(
                            dst=acc_ps,
                            stationary=xs[:, kb, nb, :],
                            moving=w_sb[:, kb, c0:c0 + LOWP_SCALE_BLOCK],
                            accumulate=(kb > 0),
                        )
                    nisa.tensor_copy(dst=res[:, c0:c0 + LOWP_SCALE_BLOCK], src=acc_ps)
            else:
                # xk[p, kb, m] = xt[p, m, kb], so each block's stationary is contiguous.
                xk = nl.ndarray((CONTRACTION_TILE, n_kb, mw), dtype=nl.bfloat16,
                                buffer=nl.sbuf)
                nisa.tensor_copy(
                    dst=xk,
                    src=xt.ap(pattern=[[mw * n_kb, CONTRACTION_TILE], [1, n_kb],
                                       [n_kb, mw]]),
                )
                for c0 in range(0, cw, OUTPUT_TILE):
                    tw = min(OUTPUT_TILE, cw - c0)
                    acc_ps = _psum(mw, tw)
                    for kb in range(n_kb):
                        nisa.nc_matmul(
                            dst=acc_ps,
                            stationary=xk[:, kb, :],
                            moving=w_sb[:, kb, c0:c0 + tw],
                            accumulate=(kb > 0),
                        )
                    nisa.tensor_copy(dst=res[:, c0:c0 + tw], src=acc_ps)
            nisa.dma_copy(
                dst=out.ap(pattern=[[n_ext, mw], [1, cw]], offset=m0 * n_ext + col0),
                src=res,
            )
    return out


#: Rows of x^T one matmul streams in the row-split path when the weight block is the
#: stationary operand and x^T the moving one: the moving free extent,
#: ``nl.tile_size.gemm_moving_fmax``. Also the rows one output DMA carries.
LOWP_ROW_BLOCK = OUTPUT_TILE

#: Bytes per partition of one PSUM bank (``nl.tile_size.psum_bank_fmax`` fp32 words).
#: A transposed 128-wide block is 128 elements per partition, so a bank holds eight
#: bf16 blocks or four fp32 ones.
_PSUM_BANK_BYTES = 2048

#: Every this-many-th scale fold runs on the Scalar engine, the rest on Vector. A
#: ``[128, 512]`` bf16 fold MEASURED 347 ns on Vector and 711 ns on Scalar in the
#: kernel, so Vector takes two of three. GpSimd is not used: its fold MEASURED 4.1 us,
#: and a Vector fold running beside it 2.0-4.1 us (``reports/mla_projection.md``,
#: engine costs).
LOWP_SCALAR_FOLD_PERIOD = 3

#: Contraction blocks from which the row-split path may make the weight block the
#: stationary operand (see :func:`lowp_weight_stationary`). Swapping adds one fp32 PE
#: transpose per output block and saves PE time on every contraction block. MEASURED
#: per call: x^T stationary is faster at 2 blocks (o_proj) and ties at 12 (q_b), so the
#: bound lies in between (``reports/mla_projection.md``, design measurements).
LOWP_WEIGHT_STATIONARY_MIN_BLOCKS = 8

#: SBUF bytes per partition of fp32 results one output DMA carries in the row-split
#: path: 2 MiB a DMA. MEASURED per call against one DMA per PSUM bank (256 KiB to
#: 1 MiB): wq_b 106 -> 92 us; against one DMA per row block (8 MiB for o_proj),
#: o_proj 61 -> 38 us, its stores overlapping the matmuls (``reports/mla_projection.md``,
#: design measurements).
LOWP_RESULT_SBUF_BYTES = 16 * 1024

#: Every this-many-th PSUM evacuation runs on the Scalar engine. In the kernel a
#: ``[128, 512]`` fp32 PSUM read MEASURED 692 ns on Vector and 587 ns on Scalar, and a
#: bf16 bank of x^T blocks 692 ns and 1012 ns: an even split (``reports/mla_projection.md``,
#: engine costs).
LOWP_SCALAR_EVAC_PERIOD = 2

#: Largest row of ``x``, in bytes, the row-split path reads whole: one row on each
#: partition, so this is the SBUF its row buffer takes. A wider ``x`` keeps the column
#: split (see :func:`lowp_splits_rows`).
LOWP_X_ROW_BYTES = 16 * 1024

#: SBUF bytes per partition of x^T the row-split path holds at once; a program's rows
#: past it are transposed and projected a batch at a time. A third of trn2's 192 KiB
#: partition, beside the weight chunk (:data:`LOWP_WEIGHT_SBUF_BYTES`) and the row
#: buffers: a 2048-row chunk of a 4096-wide ``x`` on an LNC2 pair is one batch.
LOWP_XT_SBUF_BYTES = 64 * 1024


def lowp_splits_rows(rows: int, programs: int, row_bytes: int) -> bool:
    """Whether the low-precision kernel splits ``rows`` rather than the output columns.

    A prefill chunk -- more than one sequence tile, in whole tiles per program --
    splits its rows: each core transposes only its own rows of ``x``, and x^T, the
    larger operand there, is built once rather than once per program. A decode batch
    (one sequence tile) splits the output columns, so each core reads half the
    weight, the larger operand there. ``row_bytes`` is one row of ``x`` in bytes; a
    row wider than :data:`LOWP_X_ROW_BYTES` keeps the column split. Shape arithmetic
    only, so the seam and the kernel decide alike.
    """
    return (rows > SEQUENCE_TILE and rows % (SEQUENCE_TILE * programs) == 0
            and row_bytes <= LOWP_X_ROW_BYTES)


def _lowp_x_row_bytes(x_hbm) -> int:
    """Bytes of one row of ``x_hbm``, which is bf16 or fp32 (the seam admits no other)."""
    return x_hbm.shape[1] * (2 if x_hbm.dtype == nl.bfloat16 else 4)


def _lowp_rows_held(n_kb: int) -> int:
    """Rows of x^T, whole sequence tiles, that :data:`LOWP_XT_SBUF_BYTES` holds."""
    held = LOWP_XT_SBUF_BYTES // (n_kb * 2) // SEQUENCE_TILE * SEQUENCE_TILE
    return max(held, SEQUENCE_TILE)


def _alternate(index: int, period: int):
    """The Scalar engine for every ``period``-th instruction, Vector for the rest."""
    return nisa.scalar_engine if index % period == period - 1 else nisa.vector_engine


def _row_dma(dst, src) -> None:
    """A row-split-path DMA, its descriptors from the Sync engine's hardware DGE ring.

    The default, GpSimd's software DGE, MEASURED 0.65-1.1 us of GpSimd per DMA, and
    GpSimd slows Vector beside it. Per call against it the Sync ring MEASURED 2-12 %
    faster on q_a, kv_a and q_b and 2 % slower on o_proj, and static descriptors 1-18 %
    slower (``reports/mla_projection.md``, design measurements).
    """
    nisa.dma_copy(dst=dst, src=src, dge_mode=nisa.dge_mode.hwdge, engine=nisa.engine.sync)


def _lowp_transpose_rows(x_hbm, row0, rows: int):
    """``xt[p, kb, m] = bf16(x[row0 + m, kb * 128 + p])`` for ``rows`` rows of ``x``.

    Whole rows are read (one contiguous run per partition), rounded to bf16 when
    ``x`` is fp32 -- the transpose is bit-accurate (MEASURED on trn2), so rounding
    first gives the same bf16 as rounding after -- and transposed on the PE, a bank
    of blocks at a time.
    """
    k_ext = x_hbm.shape[1]
    n_kb = k_ext // CONTRACTION_TILE
    per_bank = _PSUM_BANK_BYTES // (SEQUENCE_TILE * 2)
    xt = nl.ndarray((CONTRACTION_TILE, n_kb, rows), dtype=nl.bfloat16, buffer=nl.sbuf)
    evac = 0
    for m0 in range(0, rows, SEQUENCE_TILE):
        x_rows = nl.ndarray((SEQUENCE_TILE, k_ext), dtype=x_hbm.dtype, buffer=nl.sbuf)
        _row_dma(
            dst=x_rows,
            src=x_hbm.ap(pattern=[[k_ext, SEQUENCE_TILE], [1, k_ext]],
                         offset=(row0 + m0) * k_ext),
        )
        if x_hbm.dtype != nl.bfloat16:
            x_bf = nl.ndarray((SEQUENCE_TILE, k_ext), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=x_bf, src=x_rows, engine=_alternate(m0 // SEQUENCE_TILE, 2))
            x_rows = x_bf
        for g0 in range(0, n_kb, per_bank):
            group = min(per_bank, n_kb - g0)
            t_ps = nl.ndarray((CONTRACTION_TILE, group, SEQUENCE_TILE), dtype=nl.bfloat16,
                              buffer=nl.psum)
            for j in range(group):
                k0 = (g0 + j) * CONTRACTION_TILE
                nisa.nc_transpose(dst=t_ps[:, j, :], data=x_rows[:, k0:k0 + CONTRACTION_TILE])
            nisa.tensor_copy(dst=xt[:, g0:g0 + group, m0:m0 + SEQUENCE_TILE], src=t_ps,
                             engine=_alternate(evac, LOWP_SCALAR_EVAC_PERIOD))
            evac += 1
    return xt


def _lowp_grid_index(n_kb: int, prog_nb: int, kb: int, nb: int) -> int:
    """Offset of block ``(kb, nb)`` in the flat program-major grid (:func:`lowp_scale_layout`)."""
    return (nb // prog_nb) * n_kb * prog_nb + kb * prog_nb + nb % prog_nb


def lowp_weight_stationary(n_kb: int, cols: int, scaled: bool) -> bool:
    """Whether the row-split path makes the weight block the stationary operand.

    x^T stationary (the column-split path's orientation) moves at most a PSUM bank of
    output columns per matmul -- one 128-wide scale block when a block's multiplier is
    folded into x, else ``min(cols, 512)``. The weight stationary moves up to
    :data:`LOWP_ROW_BLOCK` rows per matmul instead, at the price of one PE transpose of
    each ``[128, 128]`` output block, which :data:`LOWP_WEIGHT_STATIONARY_MIN_BLOCKS`
    contraction blocks amortise. So the weight is stationary when x^T-stationary
    matmuls would be narrow -- scaled, or fewer than 512 columns -- and the
    contraction is deep enough.
    """
    narrow = scaled or cols < OUTPUT_TILE
    return narrow and n_kb >= LOWP_WEIGHT_STATIONARY_MIN_BLOCKS


def _lowp_rows_matmuls(xt, w_sb, s_sb, prog_nb: int, out, row0, col0: int):
    """``out[row0 + m, col0 + c] = x @ dequant(w)`` for one resident weight chunk.

    ``s_sb`` is the broadcast scale grid of an fp8 weight, or None for bf16. A scaled
    block's multiplier is folded into x^T as in the column-split path,
    ``bf16(bf16(x) * s[kb, nb])``, once per block pair over up to
    :data:`LOWP_ROW_BLOCK` rows. The orientation is :func:`lowp_weight_stationary`'s;
    with the weight stationary the ``[n, m]`` result is transposed back on the PE,
    which is bit-accurate. The products and their order are the same in either
    orientation, so the sums are the same bits (MEASURED bit-identical on trn2).

    Every accumulation group has a PSUM bank of its own: two groups in one bank, even
    one after the other, MEASURED wrong sums on trn2; single-matmul writes such as the
    transposes may share one (``reports/mla_projection.md``, engine probes).
    """
    n_kb, rows = xt.shape[1], xt.shape[2]
    cw = w_sb.shape[2]
    scaled = s_sb is not None
    weight_stationary = lowp_weight_stationary(n_kb, cw, scaled)
    # Output columns per accumulation group: the stationary width when the weight is
    # stationary, one scale block when it is folded into x, else a PSUM bank. Rows per
    # pass: what one moving operand or one fold covers; a bf16 weight's x^T-stationary
    # matmuls need neither, so they take a row tile at a time, one bank per group.
    block = OUTPUT_TILE
    row_block = SEQUENCE_TILE
    if weight_stationary or scaled:
        block = LOWP_SCALE_BLOCK
        row_block = LOWP_ROW_BLOCK
    count = [0, 0]  # folds, PSUM evacuations: each alternates its engines
    for r0 in range(0, rows, row_block):
        rw = min(row_block, rows - r0)
        n_mt = rw // SEQUENCE_TILE
        # One output DMA per span of whole banks, LOWP_RESULT_SBUF_BYTES a partition.
        span = max(OUTPUT_TILE, LOWP_RESULT_SBUF_BYTES // (n_mt * 4) // OUTPUT_TILE
                   * OUTPUT_TILE)
        for s0 in range(0, cw, span):
            sw = min(span, cw - s0)
            # res[m, mt, c]: row tile mt of the row block, the span's columns.
            res = nl.ndarray((SEQUENCE_TILE, n_mt, sw), dtype=nl.float32, buffer=nl.sbuf)
            for g0 in range(0, sw, OUTPUT_TILE):
                _lowp_rows_bank(xt, w_sb, s_sb, prog_nb, res, r0, rw, s0, g0, col0, block,
                                weight_stationary, count)
            _row_dma(
                dst=out.ap(pattern=[[out.shape[1], SEQUENCE_TILE],
                                    [SEQUENCE_TILE * out.shape[1], n_mt], [1, sw]],
                           offset=(row0 + r0) * out.shape[1] + col0 + s0),
                src=res,
            )


def _lowp_rows_bank(xt, w_sb, s_sb, prog_nb: int, res, r0: int, rw: int, s0: int, g0: int,
                    col0: int, block: int, weight_stationary: bool, count) -> None:
    """One PSUM bank's worth of output columns, ``res[:, :, g0 : g0 + 512]``, of the
    span at chunk column ``s0`` for the ``rw`` rows at ``r0`` (see
    :func:`_lowp_rows_matmuls`).
    """
    n_kb = xt.shape[1]
    n_mt = rw // SEQUENCE_TILE
    gw = min(OUTPUT_TILE, res.shape[2] - g0)
    # With the weight stationary the blocks' transposes fill one tile; else one x^T-
    # stationary block covering the bank leaves it whole in its accumulation tile.
    whole = weight_stationary or gw <= block
    held = None
    if whole:
        held = nl.ndarray((SEQUENCE_TILE, n_mt, OUTPUT_TILE), dtype=nl.float32,
                          buffer=nl.psum)
    for b0 in range(0, gw, block):
        bw = min(block, gw - b0)
        c0 = s0 + g0 + b0
        if weight_stationary:
            acc_ps = _psum(bw, rw)
        elif whole:
            acc_ps = held
        else:
            # One bank per row tile, each group in its first ``bw`` words.
            acc_ps = nl.ndarray((SEQUENCE_TILE, n_mt, OUTPUT_TILE), dtype=nl.float32,
                                buffer=nl.psum)
        for kb in range(n_kb):
            x_op = xt[:, kb, r0:r0 + rw]
            if s_sb is not None:
                at = _lowp_grid_index(n_kb, prog_nb, kb, (col0 + c0) // LOWP_SCALE_BLOCK)
                x_op = nl.ndarray((CONTRACTION_TILE, rw), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=x_op, data=xt[:, kb, r0:r0 + rw], op0=nl.multiply,
                                   operand0=s_sb[:, at:at + 1],
                                   engine=_alternate(count[0], LOWP_SCALAR_FOLD_PERIOD))
                count[0] += 1
            w_block = w_sb[:, kb, c0:c0 + bw]
            if weight_stationary:
                nisa.nc_matmul(dst=acc_ps, stationary=w_block, moving=x_op,
                               accumulate=(kb > 0))
            else:
                for mt in range(n_mt):
                    m0 = mt * SEQUENCE_TILE
                    nisa.nc_matmul(dst=acc_ps[:, mt, 0:bw],
                                   stationary=x_op[:, m0:m0 + SEQUENCE_TILE],
                                   moving=w_block, accumulate=(kb > 0))
        if weight_stationary:
            y_t = _sbuf(bw, rw)
            nisa.tensor_copy(dst=y_t, src=acc_ps,
                             engine=_alternate(count[1], LOWP_SCALAR_EVAC_PERIOD))
            count[1] += 1
            for mt in range(n_mt):
                m0 = mt * SEQUENCE_TILE
                nisa.nc_transpose(dst=held[:, mt, b0:b0 + bw],
                                  data=y_t[:, m0:m0 + SEQUENCE_TILE])
        elif not whole:
            nisa.tensor_copy(dst=res[:, :, g0 + b0:g0 + b0 + bw], src=acc_ps[:, :, 0:bw],
                             engine=_alternate(count[1], LOWP_SCALAR_EVAC_PERIOD))
            count[1] += 1
    if whole:
        nisa.tensor_copy(dst=res[:, :, g0:g0 + gw], src=held[:, :, 0:gw],
                         engine=_alternate(count[1], LOWP_SCALAR_EVAC_PERIOD))
        count[1] += 1


def _lowp_project_rows(x_hbm, w_hbm, scale_hbm, col_chunk: int, out) -> None:
    """The row-split body (see :func:`lowp_splits_rows`): this program's rows against
    every output column, :func:`_lowp_rows_held` rows of x^T at a time.

    ``col_chunk`` counts columns of the whole weight here, not of a program's share.
    """
    m_ext, k_ext = x_hbm.shape
    n_ext = w_hbm.shape[1]
    n_kb = k_ext // CONTRACTION_TILE
    programs = nl.num_programs(axes=0)
    rows = m_ext // programs
    row0 = nl.program_id(0) * rows
    chunk = n_ext if col_chunk == 0 or col_chunk > n_ext else col_chunk
    s_sb = None
    if scale_hbm is not None:
        # The whole program-major grid on every partition: one contiguous run each.
        grid = scale_hbm.shape[0] * scale_hbm.shape[1] * scale_hbm.shape[2]
        s_sb = nl.ndarray((CONTRACTION_TILE, grid), dtype=nl.float32, buffer=nl.sbuf)
        _row_dma(dst=s_sb, src=scale_hbm.ap(pattern=[[0, CONTRACTION_TILE], [1, grid]],
                                                 offset=0))
    prog_nb = 0 if scale_hbm is None else scale_hbm.shape[2]
    # One resident chunk is loaded once; several are reloaded per batch of x^T.
    resident = chunk == n_ext
    w_sb = None
    if resident:
        w_sb = _lowp_load_weight(w_hbm, 0, n_ext)
    held = _lowp_rows_held(n_kb)
    for t0 in range(0, rows, held):
        xt = _lowp_transpose_rows(x_hbm, row0 + t0, min(held, rows - t0))
        for col0 in range(0, n_ext, chunk):
            if not resident:
                w_sb = _lowp_load_weight(w_hbm, col0, min(chunk, n_ext - col0))
            _lowp_rows_matmuls(xt, w_sb, s_sb, prog_nb, out, row0 + t0, col0)


def _lowp_load_weight(w_hbm, col0: int, cols: int):
    """``w_sb[p, kb, c] = w[kb * 128 + p, col0 + c]``, :data:`LOWP_KB_PER_DMA` blocks a DMA."""
    n_kb = w_hbm.shape[0] // CONTRACTION_TILE
    n_ext = w_hbm.shape[1]
    w_sb = nl.ndarray((CONTRACTION_TILE, n_kb, cols), dtype=w_hbm.dtype, buffer=nl.sbuf)
    for kb0 in range(0, n_kb, LOWP_KB_PER_DMA):
        group = min(LOWP_KB_PER_DMA, n_kb - kb0)
        _row_dma(
            dst=w_sb[:, kb0:kb0 + group, :],
            src=w_hbm.ap(pattern=[[n_ext, CONTRACTION_TILE],
                                  [CONTRACTION_TILE * n_ext, group], [1, cols]],
                         offset=kb0 * CONTRACTION_TILE * n_ext + col0),
        )
    return w_sb


#: SBUF bytes per partition one weight chunk may hold. A quarter of trn2's 192 KiB
#: partition leaves room for the folded stationary tile and the row buffers; every
#: projection of this checkpoint at TP=64 fits whole.
LOWP_WEIGHT_SBUF_BYTES = 64 * 1024


def _lowp_col_chunk(idim: int, width: int, itemsize: int, scaled: bool) -> int:
    """Columns per resident weight chunk: 0 (whole share) when the share fits."""
    n_kb = idim // CONTRACTION_TILE
    if n_kb * width * itemsize <= LOWP_WEIGHT_SBUF_BYTES:
        return 0
    unit = LOWP_SCALE_BLOCK if scaled else OUTPUT_TILE
    cols = LOWP_WEIGHT_SBUF_BYTES // (n_kb * itemsize)
    cols = (cols // unit) * unit
    return cols if cols >= unit else unit


_LOWP_WEIGHT_DTYPES = (torch.float8_e4m3fn, torch.bfloat16)


def _lowp_programs(odim: int, scaled: bool) -> int:
    """2 on an LNC2 core when the output columns split into whole scale blocks."""
    if not lnc_pair():
        return 1
    unit = 2 * (LOWP_SCALE_BLOCK if scaled else 1)
    return 2 if odim % unit == 0 else 1


def lowp_programs(odim: int, scaled: bool) -> int:
    """The launch grid the low-precision seam picks for ``odim`` output columns."""
    return _lowp_programs(int(odim), bool(scaled))


def lowp_scale_layout(scale: Tensor, programs: int) -> Tensor:
    """``[K/128, N/128]`` grid -> program-major ``[programs, K/128, N/(128*programs)]``.

    Each program reads its own columns' multipliers as one contiguous run per
    partition. A load-time re-layout: callers that prepare weights once call this
    once; the seam calls it per call only when handed the plain grid.
    """
    if scale.ndim != 2 or int(scale.shape[1]) % int(programs):
        raise MlaProjectionError(
            f"the scale grid must be [K/128, N/128] with N/128 divisible by the "
            f"{programs} program(s); got {tuple(scale.shape)}"
        )
    n_kb, n_nb = (int(d) for d in scale.shape)
    return (
        scale.to(torch.float32)
        .reshape(n_kb, int(programs), n_nb // int(programs))
        .permute(1, 0, 2)
        .contiguous()
    )


def lowp_scale_grid(scale: Tensor) -> Tensor:
    """The plain ``[K/128, N/128]`` grid of a plain or program-major scale."""
    if scale.ndim == 2:
        return scale.to(torch.float32)
    programs, n_kb, n_nb = (int(d) for d in scale.shape)
    return scale.to(torch.float32).permute(1, 0, 2).reshape(n_kb, programs * n_nb)


def _require_lowp_admissible(x: Tensor, weight: Tensor, scale: Tensor | None,
                             programs: int) -> None:
    if x.ndim != 2 or weight.ndim != 2:
        raise MlaProjectionError(
            f"x must be [rows, in_features] and weight [in_features, out_features]; "
            f"got {tuple(x.shape)} and {tuple(weight.shape)}"
        )
    rows, idim = (int(d) for d in x.shape)
    if int(weight.shape[0]) != idim:
        raise MlaProjectionError(
            f"weight is contraction-major, so weight.shape[0] must equal x.shape[1]; "
            f"got weight {tuple(weight.shape)} against x {tuple(x.shape)}"
        )
    odim = int(weight.shape[1])
    if rows < 1 or odim < 1:
        raise MlaProjectionError(f"need positive extents; got x {tuple(x.shape)}")
    if idim % CONTRACTION_TILE or idim > CONTRACTION_TILE * CONTRACTION_TILE:
        raise MlaProjectionError(
            f"in_features must be a multiple of {CONTRACTION_TILE} and at most "
            f"{CONTRACTION_TILE * CONTRACTION_TILE}; got {idim}"
        )
    if weight.dtype not in _LOWP_WEIGHT_DTYPES:
        raise MlaProjectionError(
            f"the low-precision route reads fp8-e4m3 or bf16 weights as stored; got "
            f"{weight.dtype}"
        )
    if x.dtype not in (torch.bfloat16, torch.float32):
        raise MlaProjectionError(f"x must be bf16 or fp32; got {x.dtype}")
    fp8 = weight.dtype == torch.float8_e4m3fn
    if fp8 != (scale is not None):
        raise MlaProjectionError(
            "an fp8 weight needs its block dequant multipliers and a bf16 weight takes "
            f"none; got weight {weight.dtype} with scale "
            f"{None if scale is None else tuple(scale.shape)}"
        )
    if scale is not None:
        n_kb, n_nb = idim // LOWP_SCALE_BLOCK, odim // LOWP_SCALE_BLOCK
        plain = (n_kb, n_nb)
        laid = (programs, n_kb, n_nb // programs)
        got = tuple(int(d) for d in scale.shape)
        if odim % LOWP_SCALE_BLOCK or got not in (plain, laid):
            raise MlaProjectionError(
                f"the scale grid must be [in // {LOWP_SCALE_BLOCK}, out // "
                f"{LOWP_SCALE_BLOCK}] = {plain} or its program-major layout {laid} "
                f"for this {programs}-program launch; got {got} for weight "
                f"{tuple(weight.shape)}"
            )


def mla_projection_lowp(x: Tensor, weight: Tensor, scale: Tensor | None = None) -> Tensor:
    """Project ``x [M, K]`` by a stored low-precision ``weight [K, N]``; ``[M, N]`` fp32.

    ``weight`` is contraction-major. An fp8-e4m3 weight comes with ``scale``, its
    ``[K // 128, N // 128]`` fp32 dequant multipliers in the same orientation (or
    that grid already in :func:`lowp_scale_layout`); a bf16 weight comes with none.
    ``x`` is rounded to bf16 for the stationary operand, and an fp8 block's
    multiplier is folded into that bf16 operand: those two roundings are the
    numeric difference from the fp32 kernel.
    """
    scaled = scale is not None
    programs = _lowp_programs(int(weight.shape[1]) if weight.ndim == 2 else 0, scaled)
    _require_lowp_admissible(x, weight, scale, programs)
    if scaled and scale.ndim == 2:
        scale = lowp_scale_layout(scale, programs)
    # A program holds every weight column when the kernel splits rows, its share else.
    held = int(weight.shape[1])
    if not lowp_splits_rows(int(x.shape[0]), programs, int(x.shape[1]) * x.element_size()):
        held //= programs
    chunk = _lowp_col_chunk(int(weight.shape[0]), held, weight.element_size(), scaled)
    # Both counters move: this is a dispatch of the projection family as much as the
    # fp32 kernel's is, so the family reading stays one count per projection site.
    _count_nki_dispatch()
    _count_lowp_dispatch(scaled, programs)
    call = wrap_nki(mla_projection_lowp_kernel)
    if programs == 2:
        call = call[2]
    if scaled:
        return call(x.contiguous(), weight.contiguous(),
                    scale.contiguous().to(torch.float32), chunk)
    return call(x.contiguous(), weight.contiguous(), None, chunk)


def mla_projection_prepared(
    x: Tensor, weight_fp32: Tensor, lowp: tuple[Tensor, Tensor | None] | None
) -> Tensor:
    """The route a load-time prep chose for one site. ``[M, N]`` float32 either way.

    ``lowp`` is ``(weight, scale)`` from :func:`prepare_lowp_projection` when the
    checkpoint stored the weight as fp8-e4m3 or bf16, and None otherwise; then the
    fp32 kernel runs on ``weight_fp32`` exactly as before.
    """
    if lowp is None:
        return mla_projection(x.to(torch.float32), weight_fp32)
    return mla_projection_lowp(x, lowp[0], lowp[1])


def lowp_projection_admits(weight_out_in: Tensor, scale_out_in: Tensor | None) -> bool:
    """Whether the low-precision kernel serves this ``[out, in]`` checkpoint weight.

    Shape and dtype only, so a load-time prep can ask before it builds anything. A
    weight it does not serve -- a narrow test model's, or an fp8 width that is not whole
    scale blocks -- stays on the fp32 route.
    """
    if weight_out_in.ndim != 2:
        return False
    odim, idim = (int(d) for d in weight_out_in.shape)
    if odim < 1 or idim % CONTRACTION_TILE or idim > CONTRACTION_TILE * CONTRACTION_TILE:
        return False
    if weight_out_in.dtype == torch.float8_e4m3fn:
        return (scale_out_in is not None and odim % LOWP_SCALE_BLOCK == 0
                and tuple(int(d) for d in scale_out_in.shape)
                == (odim // LOWP_SCALE_BLOCK, idim // LOWP_SCALE_BLOCK))
    return weight_out_in.dtype == torch.bfloat16 and scale_out_in is None


def prepare_lowp_projection(weight_out_in: Tensor, scale_out_in: Tensor | None = None
                            ) -> tuple[Tensor, Tensor | None]:
    """A checkpoint-oriented weight as the low-precision route's operands, once.

    ``weight_out_in`` is ``[out, in]``: fp8-e4m3 bytes with ``scale_out_in`` their
    ``[out/128, in/128]`` multipliers, or a real dtype with no scale, which is
    stored bf16. Returns ``(weight [in, out], scale)`` with the scale program-major
    for the launch grid the seam will pick. Host tensors in, host tensors out.
    """
    if weight_out_in.dtype == torch.float8_e4m3fn:
        if scale_out_in is None:
            raise MlaProjectionError("an fp8 weight needs its scale grid to be prepared")
        weight = weight_out_in.t().contiguous()
        grid = scale_out_in.to(torch.float32).t().contiguous()
        programs = _lowp_programs(int(weight.shape[1]), True)
        return weight, lowp_scale_layout(grid, programs)
    if scale_out_in is not None:
        raise MlaProjectionError(
            f"a {weight_out_in.dtype} weight carries no block scale; got one"
        )
    return weight_out_in.to(torch.bfloat16).t().contiguous(), None


def mla_projection_lowp_torch_oracle(
    x: Tensor, weight: Tensor, scale: Tensor | None = None
) -> Tensor:
    """fp32 reference: ``x @ (weight * expand(scale))``, the fp32 kernel's arithmetic."""
    dense = weight.to(torch.float32)
    if scale is not None:
        dense = dense * lowp_scale_grid(scale).repeat_interleave(
            LOWP_SCALE_BLOCK, 0
        ).repeat_interleave(LOWP_SCALE_BLOCK, 1)
    return x.to(torch.float32) @ dense
