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
