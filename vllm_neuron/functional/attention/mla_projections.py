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
# Both physical cores of an LNC2 core split the output columns (``[2]`` grid);
# each program loads only its half of the weight.
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
            the program's whole share. The seam sizes it to the SBUF budget.

    Returns:
        ``[M, N]`` fp32.
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
    import os

    if os.environ.get("NEURON_LOGICAL_NC_CONFIG") != "2":
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
    chunk = _lowp_col_chunk(int(weight.shape[0]), int(weight.shape[1]) // programs,
                            weight.element_size(), scaled)
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
