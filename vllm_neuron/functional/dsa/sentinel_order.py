# SPDX-License-Identifier: Apache-2.0
"""The DSA indexer's sentinel ordering: real pool ids first in their own order, then the sentinels.

One NKI kernel performs the whole stable partition on chip -- a mask, two prefix scans, one search
and one gather per 128-row tile -- so the int32 ``[rows, select_k]`` tensor reaches its consumer as
stored. The torch spelling this replaces (two cumsums, a ``where`` and a ``scatter``) left the
compiler to choose the scatter source's layout, and it chose a transposed one: its
InsertOffloadedTransposes pass reports ``load non_local int32 (2, 128, 8, 512) ... # dl =
tensor_op_name: _scatter`` with the 128-wide axis moved last, one DMA transpose per DSA layer on
the prefill path. The kernel emits no scatter, so there is nothing for that pass to lay out.

The partition is position-stable on both sides: real ids keep their order and so do the sentinels.
That is what lets the kernel pad a width that is not a multiple of eight (the search's granularity)
with trailing sentinel columns -- a pad lands after every real column and after every original
sentinel, so the first ``k`` ordered columns are the unpadded answer and only those are stored.
The order is exact for integer ids: the counts and the search key are fp32, and
``can_run_dsa_sentinel_order`` admits at most ``SEARCH_MAX_FREE`` columns -- the widest row whose
tiles fit one SBUF partition, derived from ``sentinel_order_sbuf_bytes`` and
``SBUF_BYTES_PER_PARTITION`` rather than typed -- so neither the padded width nor any value the
kernel computes exceeds a few thousand, far below the 2**24 an fp32 holds exactly; a wider row
takes the torch oracle.
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

PARTITION_MAX = 128
SEARCH_WIDTH = 8
# The SBUF bytes one partition offers a kernel on trn2, as the vendor kernel library states them
# (its ``MAX_AVAILABLE_SBUF_SIZE``): 224 KiB less the reserved head and tail.
SBUF_BYTES_PER_PARTITION = 224 * 1024 - 16384 - 8 - 520
_SUPPORTED_DTYPES = (torch.int32,)


def sentinel_order_sbuf_bytes(k: int) -> int:
    """SBUF bytes one partition needs for a ``k``-wide row: 12 padded tiles, 3 int32, 2 scalars."""
    k_pad = -(-k // SEARCH_WIDTH) * SEARCH_WIDTH
    return 48 * k_pad + 12 * k + 8


def _widest_row_that_fits() -> int:
    """The largest ``k`` whose tiles fit one partition; the footprint only grows with ``k``."""
    k = SBUF_BYTES_PER_PARTITION // 60  # every row needs at least 60 bytes per column
    while sentinel_order_sbuf_bytes(k) > SBUF_BYTES_PER_PARTITION:
        k -= 1
    return k


SEARCH_MAX_FREE = _widest_row_that_fits()


class SentinelOrderError(ValueError):
    """A malformed call: ``pool_ids`` is not ``[rows, select_k]``."""


@dataclass
class _SentinelOrderDispatchCounters:
    """Route-predicate counters for this module's seam."""

    nki_dispatch: int = 0
    torch_fallback: int = 0
    last_kernel: tuple[str, str] | None = None


_COUNTERS = _SentinelOrderDispatchCounters()


def reset_sentinel_order_dispatch_counters() -> None:
    """Zero the counters."""
    _COUNTERS.nki_dispatch = 0
    _COUNTERS.torch_fallback = 0
    _COUNTERS.last_kernel = None


def sentinel_order_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last reset."""
    return (_COUNTERS.nki_dispatch, _COUNTERS.torch_fallback)


@torch._dynamo.assume_constant_result
def _count_torch_fallback() -> None:
    """Count one torch-path entry, off the traced graph."""
    _COUNTERS.torch_fallback += 1


@torch._dynamo.assume_constant_result
def _count_nki_dispatch() -> None:
    """Count one kernel dispatch, off the traced graph: a store Dynamo reads becomes a guard."""
    _COUNTERS.nki_dispatch += 1


def sentinel_order_kernel_identity() -> tuple[str, str] | None:
    """``(module, qualname)`` of the kernel the seam last dispatched, or ``None``."""
    return _COUNTERS.last_kernel


def _kernel_identity_of(kernel) -> tuple[str, str]:
    """``(module, qualname)`` of the function a ``@nki.jit`` object wraps."""
    inner = getattr(kernel, "__wrapped__", None) or getattr(kernel, "func", None) or kernel
    return (inner.__module__, inner.__qualname__)


@nki.jit
def _sentinel_order_nki(pool_ids_hbm):
    """``[rows, k]`` int32 as stored -> the same ids, the negatives moved to the trailing columns.

    Any width up to ``SEARCH_MAX_FREE``: the counted width is padded to a multiple of eight with
    trailing sentinel columns, which the stable partition places after every real and every
    original sentinel.
    """
    rows = pool_ids_hbm.shape[0]
    k = pool_ids_hbm.shape[1]
    # The search reads eight values at a time, so the counted width is padded to a multiple of
    # eight with trailing sentinel columns: they sort after every real column, so the first k
    # ordered columns are unchanged and the pad is never stored.
    k_pad = -(-k // SEARCH_WIDTH) * SEARCH_WIDTH
    out = nl.ndarray((rows, k), dtype=nl.int32, buffer=nl.shared_hbm)
    for r0 in range(0, rows, PARTITION_MAX):
        h = min(PARTITION_MAX, rows - r0)
        ids = nl.ndarray((h, k), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ids, src=nl.load(pool_ids_hbm[r0:r0 + h, 0:k]))
        ones = nl.ndarray((h, k_pad), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=ones, value=1.0)
        zero = nl.ndarray((h, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=zero, value=0.0)
        # real = min(max(id + 1, 0), 1): one where the id is a pool, zero where it is a sentinel;
        # the pad columns read zero, which is a sentinel.
        shifted = nl.ndarray((h, k_pad), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=shifted, value=0.0)
        nisa.tensor_scalar(dst=shifted[:, 0:k], data=ids, op0=nl.add, operand0=1.0,
                           op1=nl.maximum, operand1=0.0)
        real = nl.ndarray((h, k_pad), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=real, data=shifted, op0=nl.minimum, operand0=1.0)
        sentinel = nl.ndarray((h, k_pad), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=sentinel, data=real, op0=nl.multiply, operand0=-1.0,
                           op1=nl.add, operand1=1.0)
        reals = nl.ndarray((h, k_pad), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor_scan(dst=reals, data0=ones, data1=real, initial=zero,
                                op0=nl.multiply, op1=nl.add)
        sentinels = nl.ndarray((h, k_pad), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor_scan(dst=sentinels, data0=ones, data1=sentinel, initial=zero,
                                op0=nl.multiply, op1=nl.add)
        n_real = nl.ndarray((h, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=n_real, src=reals[:, k_pad - 1:k_pad])
        # key = real * reals + sentinel * (sentinels + n_real): each id's destination plus one, so a
        # permutation of 1..k_pad that the search below inverts; the pads take k+1..k_pad.
        real_part = nl.ndarray((h, k_pad), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=real_part, data1=real, data2=reals, op=nl.multiply)
        after_reals = nl.ndarray((h, k_pad), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=after_reals, data=sentinels, op0=nl.add, operand0=n_real)
        sentinel_part = nl.ndarray((h, k_pad), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sentinel_part, data1=sentinel, data2=after_reals, op=nl.multiply)
        key = nl.ndarray((h, k_pad), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=key, data1=real_part, data2=sentinel_part, op=nl.add)
        vals = nl.ndarray((h, k_pad), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(dst=vals, pattern=[[1, k_pad]], offset=1, channel_multiplier=0)
        source = nl.ndarray((h, k_pad), dtype=nl.uint32, buffer=nl.sbuf)
        for c0 in range(0, k_pad, SEARCH_WIDTH):
            nisa.nc_find_index8(dst=source[:, c0:c0 + SEARCH_WIDTH], data=key,
                                vals=vals[:, c0:c0 + SEARCH_WIDTH])
        ordered = nl.ndarray((h, k), dtype=nl.int32, buffer=nl.sbuf)
        nisa.nc_n_gather(dst=ordered, data=ids, indices=source[:, 0:k])
        nl.store(out[r0:r0 + h, 0:k], value=ordered)
    return out


@torch._dynamo.assume_constant_result
def _record_nki_dispatch(rows: int, k: int) -> None:
    """Record which kernel the seam dispatched, and log it, OFF the compiled graph.

    The template is landed and measured: ``score_gemm.py:348-374`` by way of
    ``kpool_hadamard.py:428-457``. Dynamo runs a folded call once at trace time, so the log record
    is written on the host and the ``logging.Logger`` call never enters the graph the runner
    compiles with ``fullgraph=True``. A folded helper takes ints only, so the kernel is read as a
    module global rather than passed in.
    """
    _COUNTERS.last_kernel = _kernel_identity_of(_sentinel_order_nki)
    logger.info("[dsa-sentinel-order] kernel=nki rows=%d select_k=%d", rows, k)


def _validate(pool_ids: Tensor) -> tuple[int, int]:
    """Host-side shape validation from ``.shape`` alone. Returns ``(rows, k)``."""
    if pool_ids.ndim != 2:
        raise SentinelOrderError(
            f"pool_ids must be [rows, select_k] from the sentinel; got {tuple(pool_ids.shape)}"
        )
    return int(pool_ids.shape[0]), int(pool_ids.shape[1])


def can_run_dsa_sentinel_order(pool_ids: Tensor) -> bool:
    """Whether the NKI kernel serves this call. ``False`` sends it to the torch oracle."""
    if not can_run_kernel():
        return False
    if pool_ids.dtype not in _SUPPORTED_DTYPES or pool_ids.ndim != 2:
        return False
    rows, k = int(pool_ids.shape[0]), int(pool_ids.shape[1])
    return rows > 0 and 1 <= k <= SEARCH_MAX_FREE


def dsa_sentinel_order(pool_ids: Tensor) -> Tensor:
    """THE COUNTED SEAM. ``[rows, select_k]`` pool ids, the negatives moved to the trailing columns.

    Real ids keep their relative order and so do the sentinels; the multiset of every row is
    unchanged. int32 with 1 to ``SEARCH_MAX_FREE`` columns takes the NKI route, a width that is
    not a multiple of 8 padded on chip with trailing sentinel columns; any other call is served by
    the torch oracle. Raises ``SentinelOrderError`` for a tensor that is not 2-D.
    """
    rows, k = _validate(pool_ids)
    if not can_run_dsa_sentinel_order(pool_ids):
        _count_torch_fallback()
        return _dsa_sentinel_order_torch(pool_ids)
    _count_nki_dispatch()
    _record_nki_dispatch(rows, k)
    return wrap_nki(_sentinel_order_nki)(pool_ids.contiguous())


def _dsa_sentinel_order_torch(pool_ids: Tensor) -> Tensor:
    """The oracle: the counted torch spelling the kernel replaces, kept as the model wrote it."""
    from vllm_neuron.functional.cumsum import cumsum

    real = (pool_ids >= 0).to(torch.int32)
    sentinel = 1 - real
    reals = cumsum(real, dim=-1)
    sentinels = cumsum(sentinel, dim=-1)
    destination = torch.where(
        real.bool(), reals - real, reals[:, -1:] + sentinels - sentinel
    ).to(torch.int64)
    return torch.zeros_like(pool_ids).scatter(1, destination, pool_ids)
