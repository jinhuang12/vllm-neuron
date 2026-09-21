# SPDX-License-Identifier: Apache-2.0
"""Per-head batched matmul ``out[s, h, :] = x[s, h, :] @ w[h]``, written in NKI.

Two steps of the MLA attention chain need it: absorb-in lifts a ``[S, H, 256]``
query to the ``[S, H, 512]`` latent rank the sparse attention seam works in, and
absorb-out brings that seam's ``[S, H, 512]`` result back to the 256 head width,
both at ``H = 64``. This module only multiplies what it is handed -- splitting
``kv_b_proj`` into ``W_UK`` and ``W_UV`` and viewing each per head belongs to the
caller, which prepares both weights once off the per-forward path -- so the gate
below checks geometry and never provenance.

``nc_matmul`` contracts the partition axis, so both operands must present the
contraction extent there. That is why the weight arrives contraction-major per head
as ``[H, K, N]`` and not as ``[H, N, K]``: it already presents the extent, while
each ``x`` slab has to be turned onto it on the PE, which carries the accuracy
caveat ``_transpose_tile`` records. An inadmissible geometry raises rather than
falling back: there is no torch absorb path, and ``mla_absorb_torch_oracle`` is
only the CPU reference for tests.
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

from vllm_neuron.functional.attention.mla_projections import (
    CONTRACTION_TILE,
    OUTPUT_TILE,
    SEQUENCE_TILE,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

logger = logging.getLogger(__name__)

#: The three tile extents are imported from the sibling module rather than retyped
#: here: they are the same quantity, the platform's
#: ``nl.tile_size.{pmax, gemm_stationary_fmax, gemm_moving_fmax}``, and this kernel
#: tiles exactly as that one does. Retyping them would let the two drift apart.
_TILE_SOURCE = "vllm_neuron.functional.attention.mla_projections"


class MlaAbsorbError(ValueError):
    """Raised for a geometry this kernel does not serve."""


@dataclass
class _MlaAbsorbDispatchCounters:
    """Per-process record of how the seam below was reached."""

    nki_dispatch: int = 0
    torch_fallback: int = 0


_MLA_ABSORB_COUNTERS = _MlaAbsorbDispatchCounters()


def reset_mla_absorb_dispatch_counters() -> None:
    """Zero this seam's counters."""
    _MLA_ABSORB_COUNTERS.nki_dispatch = 0
    _MLA_ABSORB_COUNTERS.torch_fallback = 0


def mla_absorb_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last reset.

    ``torch_fallback`` always reads ``0``: this module has no torch absorb route,
    and an inadmissible geometry raises instead of falling back.
    """
    return (
        _MLA_ABSORB_COUNTERS.nki_dispatch,
        _MLA_ABSORB_COUNTERS.torch_fallback,
    )


@torch._dynamo.assume_constant_result
def _count_nki_dispatch() -> None:
    """Count one kernel dispatch, off the traced graph: a store Dynamo traces becomes a guard."""
    _MLA_ABSORB_COUNTERS.nki_dispatch += 1


def _sbuf(*shape: int):
    return nl.ndarray(tuple(shape), dtype=nl.float32, buffer=nl.sbuf)


def _psum(rows: int, cols: int):
    return nl.ndarray((rows, cols), dtype=nl.float32, buffer=nl.psum)


def _transpose_tile(src_sb, rows: int, cols: int):
    """``[rows, cols]`` SBUF -> ``[cols, rows]`` SBUF, turned on the PE.

    The destination is PSUM rather than SBUF because a PSUM destination routes the
    turn to the tensor engine and serves all 128 partitions, while an SBUF one runs
    on Vector and refuses a tile above ``[32, 32]``.

    The turn is a matmul against an identity matrix, which NeuronCore-v2 documents
    as not bit-accurate when the tile holds NaN or Inf: one non-finite element can
    reach every output of its partition column, so it spreads across the whole
    ``[SEQUENCE_TILE, CONTRACTION_TILE]`` slab instead of staying in its own row.
    For finite operands the matmul sees exactly the values the caller holds. Nothing
    scans for finiteness on this path -- the operands are a projected query and the
    sparse attention output, and a per-call scan would cost more than the transport
    it guards.
    """
    ps = _psum(cols, rows)
    nisa.nc_transpose(dst=ps, data=src_sb)
    tile = _sbuf(cols, rows)
    nisa.tensor_copy(dst=tile, src=ps)
    return tile


@nki.jit
def mla_absorb_kernel(x_hbm, w_hbm):
    """``out[s, h, :] = x[s, h, :] @ w[h]``, tiled on all three inner axes.

    ``x_hbm`` is ``x`` as the caller stores it, ``[S, H, K]``, and ``w_hbm`` is
    ``[H, K, N]``. The weight already presents the contraction extent on the
    partition axis; each ``x`` slab is loaded as ``[sw, kw]`` and turned onto it by
    :func:`_transpose_tile`. Every ``nl.load`` names ``dtype=nl.float32``, so a bf16
    ``x`` is widened on device as it lands in SBUF rather than crossing HBM at twice
    its width.

    The turn is hoisted above the output loop: one ``[CONTRACTION_TILE, ktiles, sw]``
    SBUF tile holds every turned slab for this (head, sequence tile), so a slab is
    turned once instead of once per output tile and no transpose PSUM tile is live
    while the accumulator is. A whole contraction column will not fit one SBUF tile
    -- the partition axis stops at 128 -- so that tile carries the contraction-tile
    index on its free axis and the matmul reads back the leading ``kw`` partitions
    of one index.

    Each head is an independent matmul, so the head axis is the outer loop and
    carries no arithmetic of its own. Accumulation is in PSUM across the contraction
    loop, so one output tile is written to HBM exactly once, fully summed. The store
    indexes the sequence axis as a slice and the head axis as a scalar, which is what
    lets the kernel return ``[S, H, N]`` and spares the caller a host-side permute.

    Loop bounds are ``range`` rather than ``nl.affine_range`` so every extent stays a
    trace-time int and the ragged final tile on each axis can be an ordinary ``min``.
    """
    seq, heads, kdim = x_hbm.shape
    odim = w_hbm.shape[2]
    ktiles = (kdim + CONTRACTION_TILE - 1) // CONTRACTION_TILE
    out = nl.ndarray((seq, heads, odim), dtype=nl.float32, buffer=nl.shared_hbm)

    for h in range(heads):
        for s0 in range(0, seq, SEQUENCE_TILE):
            sw = min(SEQUENCE_TILE, seq - s0)
            xt = _sbuf(CONTRACTION_TILE, ktiles, sw)
            for ki in range(ktiles):
                k0 = ki * CONTRACTION_TILE
                kw = min(CONTRACTION_TILE, kdim - k0)
                x_sb = _sbuf(sw, kw)
                nisa.tensor_copy(
                    dst=x_sb,
                    src=nl.load(
                        x_hbm[s0:s0 + sw, h, k0:k0 + kw], dtype=nl.float32
                    ),
                )
                nisa.tensor_copy(
                    dst=xt[0:kw, ki, :], src=_transpose_tile(x_sb, sw, kw)
                )
            for o0 in range(0, odim, OUTPUT_TILE):
                ow = min(OUTPUT_TILE, odim - o0)
                acc_ps = _psum(sw, ow)
                for ki in range(ktiles):
                    k0 = ki * CONTRACTION_TILE
                    kw = min(CONTRACTION_TILE, kdim - k0)
                    w_tile = _sbuf(kw, ow)
                    nisa.tensor_copy(
                        dst=w_tile,
                        src=nl.load(
                            w_hbm[h, k0:k0 + kw, o0:o0 + ow], dtype=nl.float32
                        ),
                    )
                    nisa.nc_matmul(
                        dst=acc_ps,
                        stationary=xt[0:kw, ki, :],
                        moving=w_tile,
                        accumulate=(ki > 0),
                    )
                out_sb = _sbuf(sw, ow)
                nisa.tensor_copy(dst=out_sb, src=acc_ps)
                nl.store(out[s0:s0 + sw, h, o0:o0 + ow], value=out_sb)
    return out


def _require_mla_absorb_admissible(seq: int, heads: int, kdim: int, ndim: int) -> None:
    """Raise unless this kernel serves the geometry.

    Only positivity is checked: the kernel walks the head axis and all three inner
    axes in loops, so no extent has a magnitude limit.
    """
    if seq < 1 or heads < 1 or kdim < 1 or ndim < 1:
        raise MlaAbsorbError(
            f"mla_absorb needs a positive extent on every axis; got seq={seq}, "
            f"heads={heads}, contraction={kdim}, out_features={ndim}"
        )


def can_run_mla_absorb(
    reference: Tensor, seq: int, heads: int, kdim: int, ndim: int
) -> bool:
    """True when the NKI route is available and serves this geometry."""
    if not can_run_kernel(reference):
        return False
    try:
        _require_mla_absorb_admissible(seq, heads, kdim, ndim)
    except MlaAbsorbError:
        return False
    return True


def mla_absorb(x: Tensor, w: Tensor) -> Tensor:
    """Multiply ``x [S, H, K]`` by per-head ``w [H, K, N]``, returning ``[S, H, N]``.

    Accumulation is float32 inside the kernel and the result is cast back to
    ``x.dtype``, so this does not widen the dtype of the chain it sits in. ``w`` is
    contraction-major per head, ``[heads, contraction, out_features]``, and not
    ``[heads, out_features, contraction]``; the module docstring gives the reason.

    Raises:
        MlaAbsorbError: on a non-3D operand, a weight whose head count or
            contraction extent does not match ``x``, or a non-positive extent.
    """
    if x.ndim != 3:
        raise MlaAbsorbError(
            f"x must be [seq, heads, contraction]; got shape {tuple(x.shape)}"
        )
    if w.ndim != 3:
        raise MlaAbsorbError(
            f"w must be [heads, contraction, out_features]; got shape "
            f"{tuple(w.shape)}"
        )
    seq, heads, kdim = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    if int(w.shape[0]) != heads:
        raise MlaAbsorbError(
            f"w must carry one matrix per head, so w.shape[0] must equal "
            f"x.shape[1]; got w {tuple(w.shape)} against x {tuple(x.shape)}"
        )
    if int(w.shape[1]) != kdim:
        raise MlaAbsorbError(
            f"w is contraction-major per head, so w.shape[1] must equal "
            f"x.shape[2]; got w {tuple(w.shape)} against x {tuple(x.shape)}. A "
            f"[heads, out_features, contraction] weight is the likely cause -- "
            f"this seam does not accept that orientation"
        )
    ndim = int(w.shape[2])
    _require_mla_absorb_admissible(seq, heads, kdim, ndim)

    _count_nki_dispatch()
    out = wrap_nki(mla_absorb_kernel)(
        x.contiguous(),
        w.contiguous().to(torch.float32),
    )
    return out.to(x.dtype)


def mla_absorb_torch_oracle(x: Tensor, w: Tensor) -> Tensor:
    """CPU reference for tests. Nothing dispatches here; the seam above raises instead.

    It returns float32 rather than ``x.dtype`` on purpose: a reference that rounded
    to the dtype under test could not detect the kernel rounding badly.
    """
    return torch.einsum("shk,hkn->shn", x.to(torch.float32), w.to(torch.float32))


def mla_absorb_kernel_identity() -> tuple[str, str]:
    """``(module, qualname)`` of the absorb kernel this module defines.

    ``nki.jit`` returns a wrapper whose own ``__module__`` is the decorator's, so
    the identity has to be read off the wrapped function at ``.func``.
    """
    func = getattr(mla_absorb_kernel, "func", None)
    target = func if func is not None else mla_absorb_kernel
    return target.__module__, target.__qualname__
