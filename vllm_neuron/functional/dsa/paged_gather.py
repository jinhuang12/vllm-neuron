# SPDX-License-Identifier: Apache-2.0
"""Gather one row per token out of paged key/value storage.

Paged attention keeps a sequence's context in fixed-size pages that are not
contiguous, so reading the rows for a batch of tokens means following a page
table. This module is that read: given the flattened paged storage and, per
output token, the physical page and the slot inside it, it returns one gathered
row per token. Resolving a block table into (page, slot) stays with the caller,
so only one gather and no torch index arithmetic lands on the traced branch.

The NKI kernel does the ``page * page_size + slot`` multiply-add on device in
int32, which is why the seam takes two index tensors rather than one flat index.
Only bfloat16 reaches the kernel; every other dtype is served by the torch path,
which is also the CPU reference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa import oob_mode

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.utils.neuron_utils import can_run_kernel

logger = logging.getLogger(__name__)

# bfloat16 is the only dtype validated on this platform; anything else is served
# correctly by the torch path rather than refused.
_SUPPORTED_DTYPES = (torch.bfloat16,)

# Index tensors are cast to int32 once at the seam, because the kernel's index
# tiles are int32.
_INDEX_DTYPES = (torch.int32, torch.int64)


class DsaPagedGatherError(ValueError):
    """Raised for an input this module will not hand to the kernel or the torch path."""


@dataclass
class _PagedGatherDispatchCounters:
    """Per-process record of how the seam below was reached."""

    nki_dispatch: int = 0
    torch_fallback: int = 0
    last_kernel: tuple[str, str] | None = None


_COUNTERS = _PagedGatherDispatchCounters()


def reset_paged_gather_dispatch_counters() -> None:
    """Zero this seam's counters."""
    _COUNTERS.nki_dispatch = 0
    _COUNTERS.torch_fallback = 0
    _COUNTERS.last_kernel = None


def paged_gather_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last reset."""
    return (_COUNTERS.nki_dispatch, _COUNTERS.torch_fallback)


@torch._dynamo.assume_constant_result
def _count_nki_dispatch() -> None:
    """Count one kernel dispatch, off the traced graph: a store Dynamo traces becomes a guard."""
    _COUNTERS.nki_dispatch += 1


@torch._dynamo.assume_constant_result
def _count_torch_fallback() -> None:
    """Count one torch-path entry, off the traced graph."""
    _COUNTERS.torch_fallback += 1


def paged_gather_kernel_identity() -> tuple[str, str] | None:
    """``(module, qualname)`` of the kernel the seam last dispatched, or ``None``."""
    return _COUNTERS.last_kernel


def _kernel_identity_of(kernel) -> tuple[str, str]:
    """``(module, qualname)`` of the function a ``@nki.jit`` object wraps.

    Reading those attributes off the decorated object would report the decorator
    instead: ``@nki.jit`` returns an ``nki.framework.kernel.Kernel`` whose
    ``__module__`` is ``"nki.framework.kernel"`` and whose ``__qualname__`` is
    ``None``. The wrapped function is reachable at ``__wrapped__`` (the
    ``functools.wraps`` convention) and at ``.func`` (this decorator's own name).
    """
    inner = getattr(kernel, "__wrapped__", None) or getattr(kernel, "func", None) or kernel
    return (inner.__module__, inner.__qualname__)


@nki.jit
def _paged_gather_nki(pages_hbm, page_idx_hbm, slot_idx_hbm, page_size):
    """Gather one row per token out of paged HBM storage, by indirect DMA.

    Args:
        pages_hbm: ``[num_pages * page_size, width]`` -- paged storage, flattened so a
            physical page and a slot address one row.
        page_idx_hbm: ``[tokens, 1]`` int32 -- the physical page for each output token.
        slot_idx_hbm: ``[tokens, 1]`` int32 -- the slot inside that page.
        page_size: rows per page. A compile-time constant.

    Returns:
        ``[tokens, width]`` in ``pages_hbm``'s dtype.

    The index tensors are column-shaped because the flat row index is computed on
    the partition axis: one token per partition, one int32 per partition, which is
    the layout ``vector_select`` reads its offsets from.
    """
    n_slots, width = pages_hbm.shape
    tokens = page_idx_hbm.shape[0]
    pmax = nl.tile_size.pmax
    out_hbm = nl.ndarray((tokens, width), dtype=pages_hbm.dtype, buffer=nl.shared_hbm)
    n_tiles = (tokens + pmax - 1) // pmax
    for t in range(n_tiles):
        # A short final tile is narrowed rather than padded, so no padded row can
        # reach the output.
        rows = min(pmax, tokens - t * pmax)
        off = t * pmax
        pg = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
        sl = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
        # The index tiles must arrive from HBM through an access pattern: this NKI
        # image has no nl.arange/nl.mgrid/nl.iota to synthesise them on device, and
        # building them by transpose fails because nc_transpose routes through
        # nc_matmul, which refuses int32 stationary operands.
        nisa.tensor_copy(
            dst=pg, src=nl.load(page_idx_hbm.ap(pattern=[[1, rows], [1, 1]], offset=off))
        )
        nisa.tensor_copy(
            dst=sl, src=nl.load(slot_idx_hbm.ap(pattern=[[1, rows], [1, 1]], offset=off))
        )
        # flat = page * page_size + slot, on device and in int32.
        flat = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=flat, data=pg, op0=nl.multiply, operand0=page_size)
        nisa.tensor_tensor(dst=flat, data1=flat, data2=sl, op=nl.add)
        dst = nl.ndarray((rows, width), dtype=pages_hbm.dtype, buffer=nl.sbuf)
        # Zero first, so a row the DMA does not write reads as zero instead of as
        # stale tile data that could happen to match the expected value.
        nisa.memset(dst, 0)
        nisa.dma_copy(
            src=pages_hbm.vector_select(0, flat), dst=dst, oob_mode=oob_mode.error
        )
        nl.store(
            out_hbm.ap(pattern=[[width, rows], [1, width]], offset=off * width), value=dst
        )
    return out_hbm


@torch._dynamo.assume_constant_result
def _record_nki_dispatch(tokens: int, width: int, page_size: int) -> None:
    """Record which kernel the seam dispatched, and log it, off the compiled graph.

    The dispatch branch is traced under ``fullgraph=True``, so a host call Dynamo
    refuses would break it. A folded helper may take ints, strings and dtypes only:
    Dynamo converts every non-tensor argument into a Python constant at trace time,
    and an ``@nki.jit`` kernel is a frozen dataclass it cannot reconstruct. The
    kernel is therefore read as a module global instead of passed in.
    """
    _COUNTERS.last_kernel = _kernel_identity_of(_paged_gather_nki)
    logger.info(
        "[dsa-paged-gather] kernel=nki tokens=%d width=%d page_size=%d",
        tokens,
        width,
        page_size,
    )


def can_run_dsa_paged_gather(
    pages: Tensor, page_indices: Tensor, slot_indices: Tensor, page_size: int
) -> bool:
    """True when the NKI route is available and this module admits this geometry."""
    if not can_run_kernel(pages):
        return False
    if pages.ndim != 2:
        return False
    if pages.dtype not in _SUPPORTED_DTYPES:
        return False
    if page_size <= 0:
        return False
    if page_indices.numel() != slot_indices.numel():
        return False
    if page_indices.numel() == 0:
        return False
    return True


def dsa_paged_gather(
    pages: Tensor, page_indices: Tensor, slot_indices: Tensor, page_size: int
) -> Tensor:
    """Gather one row per token out of paged storage.

    Args:
        pages: ``[num_pages * page_size, width]`` -- paged storage, flattened so that a
            physical page and a slot together address one row. ``bfloat16`` takes the
            NKI route; any other dtype is served by the torch path.
        page_indices: ``[tokens]`` or ``[tokens, 1]`` integer -- the physical page for
            each output token, that is, the page table already applied.
        slot_indices: ``[tokens]`` or ``[tokens, 1]`` integer -- the slot inside that page.
        page_size: rows per page.

    Returns:
        ``[tokens, width]`` in ``pages``' dtype: row ``i`` is
        ``pages[page_indices[i] * page_size + slot_indices[i]]``.

    Raises:
        DsaPagedGatherError: for a malformed call -- a non-2D ``pages``, mismatched or
            empty index tensors, a non-integer index dtype, or a non-positive
            ``page_size``. An out-of-range index is not checked here; it surfaces as the
            DMA's own out-of-bound assertion.
    """
    if pages.ndim != 2:
        raise DsaPagedGatherError(
            f"pages must be 2-D [num_pages * page_size, width]; "
            f"got shape {tuple(pages.shape)}"
        )
    if page_size <= 0:
        raise DsaPagedGatherError(f"page_size must be positive; got {page_size}")
    for name, idx in (("page_indices", page_indices), ("slot_indices", slot_indices)):
        if idx.dtype not in _INDEX_DTYPES:
            raise DsaPagedGatherError(
                f"{name} must be one of {[str(d) for d in _INDEX_DTYPES]}; got {idx.dtype}"
            )
        if idx.ndim not in (1, 2) or (idx.ndim == 2 and int(idx.shape[1]) != 1):
            raise DsaPagedGatherError(
                f"{name} must be [tokens] or [tokens, 1]; got shape {tuple(idx.shape)}"
            )
    if page_indices.numel() != slot_indices.numel():
        raise DsaPagedGatherError(
            f"page_indices and slot_indices must describe the same tokens; got "
            f"{page_indices.numel()} and {slot_indices.numel()}"
        )
    if page_indices.numel() == 0:
        raise DsaPagedGatherError("page_indices is empty; there is nothing to gather")

    if not can_run_dsa_paged_gather(pages, page_indices, slot_indices, page_size):
        return _dsa_paged_gather_torch(pages, page_indices, slot_indices, page_size)

    tokens = int(page_indices.numel())
    width = int(pages.shape[1])
    # Column-shaped int32 is the layout the kernel's index tiles read. Reshaping and
    # casting here keeps torch orchestration out of the kernel.
    pg = page_indices.reshape(-1, 1).to(torch.int32).contiguous()
    sl = slot_indices.reshape(-1, 1).to(torch.int32).contiguous()

    _count_nki_dispatch()
    _record_nki_dispatch(tokens, width, page_size)
    return wrap_nki(_paged_gather_nki)(pages, pg, sl, page_size)


def _dsa_paged_gather_torch(
    pages: Tensor, page_indices: Tensor, slot_indices: Tensor, page_size: int
) -> Tensor:
    """CPU reference and fallback: reached without the simulator, with kernels off, or on an unadmitted dtype."""
    _count_torch_fallback()
    logger.info(
        "[dsa-paged-gather] kernel=torch tokens=%d width=%d page_size=%d "
        "reason=nki-route-unavailable",
        int(page_indices.numel()),
        int(pages.shape[1]),
        page_size,
    )
    flat = (
        page_indices.reshape(-1).to(torch.int64) * page_size
        + slot_indices.reshape(-1).to(torch.int64)
    )
    return torch.index_select(pages, 0, flat)
