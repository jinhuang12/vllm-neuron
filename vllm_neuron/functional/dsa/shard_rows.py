# SPDX-License-Identifier: Apache-2.0
"""One rank's rows of a prefill chunk, gathered on device: the row cut of the sharded selection.

``dsa_take_rank_rows(sources, rank, rows)`` returns ``rows`` rows of each ``T``-row source:
row ``i`` is row ``min(rank * rows + i, T - 1)``, so the ``d`` ranks of a chunk cover rows
``0 .. T - 1`` in order and the trailing pad rows of the last ranks repeat row ``T - 1``
(``indexer_shard.row_shard`` makes the plan). The rank is a device operand, because one
prefill graph serves every rank, so the start row is known only on device and a slice
cannot express the cut.

One kernel launch cuts up to :data:`SOURCES_PER_LAUNCH` sources (the selection's query,
gate weights and lengths take one launch). Per 128-row tile it builds the row index once on
device (an ``iota`` over the partitions, plus ``rank * rows``, clamped to ``T - 1``), then
moves each source's rows with one indirect DMA per free-dim tile into SBUF and one plain DMA
out. The SBUF stage is required: an indirect DMA needs its other side in SBUF (neuronx-cc
refuses an HBM-to-HBM one, ``NCC_IBIR231``). No row is pointed out of bounds: under LNC2
an out-of-bounds dynamic-offset access drops the whole tile (``ragged_pack.py``), and the
clamp keeps every index a real row.

The NKI route takes 2-byte and 4-byte dtypes (the selection's operands are bfloat16, float32
and int32); a call with any other dtype, and every call where NKI cannot run, takes the
torch path (``index_select`` on :func:`rank_row_index`), which is also the CPU reference.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.utils.neuron_utils import can_run_kernel

logger = logging.getLogger(__name__)

#: Dtypes the NKI route moves: a DMA copies bytes, so every 2- and 4-byte element type of
#: the selection's operands is served the same way.
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32, torch.int32)

#: Sources one launch cuts: the kernel's tensor parameters (``first_hbm`` .. ``third_hbm``).
SOURCES_PER_LAUNCH = 3

#: Bytes per partition one DMA tile carries. A contiguous run of at least 2 KiB saturates a
#: DMA queue; 32 KiB keeps the staging tile a small share of an SBUF partition at any width.
FREE_TILE_BYTES = 32 * 1024


class ShardRowsError(ValueError):
    """A row cut this module refuses: no source, a source with no rows, sources of
    different row counts, a non-positive row count, or a rank operand that is not one
    integer element."""


@dataclass
class _ShardRowsDispatchCounters:
    """Per-process record of how :func:`dsa_take_rank_rows` was reached."""

    nki_dispatch: int = 0
    torch_fallback: int = 0


_COUNTERS = _ShardRowsDispatchCounters()


def reset_shard_rows_dispatch_counters() -> None:
    """Zero the seam's counters."""
    _COUNTERS.nki_dispatch = 0
    _COUNTERS.torch_fallback = 0


def shard_rows_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last reset."""
    return (_COUNTERS.nki_dispatch, _COUNTERS.torch_fallback)


@torch._dynamo.assume_constant_result
def _count_nki_dispatch() -> None:
    """Count one kernel dispatch, off the traced graph: a store Dynamo reads becomes a guard."""
    _COUNTERS.nki_dispatch += 1


@torch._dynamo.assume_constant_result
def _count_torch_fallback() -> None:
    """Count one torch-path entry, off the traced graph."""
    _COUNTERS.torch_fallback += 1


@torch._dynamo.assume_constant_result
def _record_nki_dispatch(tokens: int, sources: int, rows: int, dtypes: str) -> None:
    """Log the dispatch off the compiled graph (ints and strings only, see ``score_gemm``)."""
    logger.info("[dsa-shard-rows] kernel=nki tokens=%d sources=%d rows=%d dtypes=%s",
                tokens, sources, rows, dtypes)


@nki.jit
def _take_rank_rows_nki(rank_hbm, rows, free_tile, first_hbm, second_hbm=None, third_hbm=None):
    """Rows ``min(rank * rows + i, T - 1)``, ``i < rows``, of each source.

    Args:
        rank_hbm: ``[1, 1]`` int32, this rank's index.
        rows: rows per rank ``R`` (trace-time int).
        free_tile: elements of a row per DMA tile (trace-time int, ``>= 1``).
        first_hbm, second_hbm, third_hbm: ``[T, F_j]`` sources of any 2- or 4-byte dtype,
            contiguous, one ``T``; the last two may be None.

    Returns:
        A tuple with one ``[R, F_j]`` array per source given, each of its source's dtype.
    """
    # The kernel frontend takes plain ``for`` loops over simple names (no comprehension,
    # no tuple target), so the sources and their outputs are listed by index.
    sources = []
    for candidate in (first_hbm, second_hbm, third_hbm):
        if candidate is not None:
            sources.append(candidate)
    tokens = first_hbm.shape[0]
    outs = []
    for j in range(len(sources)):
        outs.append(nl.ndarray((rows, sources[j].shape[1]), dtype=sources[j].dtype,
                               buffer=nl.shared_hbm))
    pmax = nl.tile_size.pmax
    for t in range((rows + pmax - 1) // pmax):
        held = min(pmax, rows - t * pmax)
        # The rank in every partition: a zero partition stride reads one address ``held`` times.
        rank_bc = nl.ndarray((held, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=rank_bc, src=rank_hbm.ap(pattern=[[0, held], [1, 1]], offset=0))
        start = nl.ndarray((held, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=start, data=rank_bc, op0=nl.multiply, operand0=rows)
        position = nl.ndarray((held, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.iota(dst=position, pattern=[[0, 1]], offset=t * pmax, channel_multiplier=1)
        # ``tensor_tensor`` for the tile sum: a ``(held, 1)`` tile as ``tensor_scalar``'s
        # operand is a per-partition scalar, which the verifier takes, but the sum of two
        # tiles is the plain form (``ragged_pack._add_tile``).
        unclamped = nl.ndarray((held, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=unclamped, data1=position, data2=start, op=nl.add)
        index = nl.ndarray((held, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=index, data=unclamped, op0=nl.minimum, operand0=tokens - 1)
        for j in range(len(sources)):
            src = sources[j]
            out = outs[j]
            width = src.shape[1]
            for f in range((width + free_tile - 1) // free_tile):
                f0 = f * free_tile
                span = min(free_tile, width - f0)
                got = nl.ndarray((held, span), dtype=src.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=got,
                    src=src.ap(pattern=[[width, held], [1, span]], offset=f0,
                               vector_offset=index, indirect_dim=0),
                )
                nisa.dma_copy(
                    dst=out.ap(pattern=[[width, held], [1, span]],
                               offset=t * pmax * width + f0),
                    src=got,
                )
    return tuple(outs)


def rank_row_index(tokens: int, rows: int, rank: Tensor | int, device) -> Tensor:
    """``[rows]`` int64: the source rows rank ``rank`` takes, ``min(rank * rows + i, T - 1)``.

    ``rank`` is a one-element integer tensor (the traced path) or a python int.
    """
    offsets = torch.arange(rows, device=device, dtype=torch.int64)
    if torch.is_tensor(rank):
        _check_rank(rank)
        start = rank.to(device=device, dtype=torch.int64).reshape(()) * rows
    else:
        start = int(rank) * rows
    return (offsets + start).clamp_max(tokens - 1)


def _check_rank(rank: Tensor) -> None:
    if rank.numel() != 1 or rank.is_floating_point() or rank.is_complex():
        raise ShardRowsError(
            f"the rank operand must be one integer element; got shape {tuple(rank.shape)} "
            f"dtype {rank.dtype}"
        )


def can_run_dsa_take_rank_rows(sources: Sequence[Tensor], rank: Tensor | int) -> bool:
    """Whether the NKI kernel serves this call. ``False`` sends it to the torch path."""
    if not all(can_run_kernel(src) for src in sources):
        return False
    device = sources[0].device
    return (all(src.dtype in _SUPPORTED_DTYPES and src.device == device for src in sources)
            and torch.is_tensor(rank) and rank.dtype == torch.int32 and rank.device == device)


def dsa_take_rank_rows(sources: Sequence[Tensor], rank: Tensor | int,
                       rows: int) -> tuple[Tensor, ...]:
    """One rank's ``rows`` rows of each ``T``-row chunk operand in ``sources``.

    Args:
        sources: ``[T, ...]`` tensors of one ``T``, any dtypes; the trailing dims travel
            with each row.
        rank: this rank's index, a one-element int32 device tensor (the traced path) or a
            python int (an eager caller; served by the torch path).
        rows: ``R``, rows per rank (``row_shard(T, d).rows``).

    Returns:
        One ``[R, ...]`` tensor per source, of its dtype: row ``i`` is
        ``source[min(rank * R + i, T - 1)]``.

    Raises:
        ShardRowsError: for no source, a source without a leading row axis, sources of
            different ``T``, ``rows < 1`` or a rank operand that is not one integer element.
    """
    sources = tuple(sources)
    if not sources:
        raise ShardRowsError("the cut needs at least one source")
    if any(src.dim() < 1 or int(src.shape[0]) < 1 for src in sources):
        raise ShardRowsError(
            f"every source needs a leading row axis; got shapes "
            f"{[tuple(src.shape) for src in sources]}"
        )
    tokens = int(sources[0].shape[0])
    if any(int(src.shape[0]) != tokens for src in sources):
        raise ShardRowsError(
            f"the sources must share one row count; got {[int(s.shape[0]) for s in sources]}"
        )
    if int(rows) < 1:
        raise ShardRowsError(f"a rank takes at least one row; got rows={rows}")
    if torch.is_tensor(rank):
        _check_rank(rank)
    rows = int(rows)
    if not can_run_dsa_take_rank_rows(sources, rank):
        _count_torch_fallback()
        index = rank_row_index(tokens, rows, rank, sources[0].device)
        return tuple(src.index_select(0, index) for src in sources)
    # One free tile for every source: sized by the widest element, so no tile passes
    # ``FREE_TILE_BYTES`` per partition.
    free_tile = max(1, FREE_TILE_BYTES // max(src.element_size() for src in sources))
    taken = []
    for first in range(0, len(sources), SOURCES_PER_LAUNCH):
        batch = sources[first:first + SOURCES_PER_LAUNCH]
        _count_nki_dispatch()
        _record_nki_dispatch(tokens, len(batch), rows,
                             ",".join(str(src.dtype) for src in batch))
        # ``reshape`` of a contiguous tensor is a view; the kernel sees ``[T, F]`` rows.
        flat = [src.contiguous().reshape(tokens, -1) for src in batch]
        out = wrap_nki(_take_rank_rows_nki)(rank.reshape(1, 1), rows, free_tile, *flat)
        taken.extend(out if isinstance(out, tuple) else (out,))
    return tuple(cut.reshape(rows, *src.shape[1:]) for cut, src in zip(taken, sources))
