# SPDX-License-Identifier: Apache-2.0
"""Exact causal index rows for the DSA indexer's short-sequence regime.

Below the selection bound every candidate pool would be selected anyway, so there is nothing
to select and the right answer is each query row's own causal prefix, written directly::

    indices[i, c] = c   where c <= positions[i]
    indices[i, c] = -1  otherwise

The ``-1`` is a sentinel value meaning "this column selects no token", not an out-of-bounds
index; masking it belongs to the consumer, ``mla_sparse_attention``. ``width`` is the caller's
expanded width -- a positive multiple of ``KEY_CHUNK``, from ``index_expand.index_expand_width``
-- and is not recomputed here, so the bypass path and the selecting path hand the consumer the
same shape and the same sentinel convention.

The column ramp comes from ``nisa.iota`` with ``channel_multiplier=0``: this NKI image has no
``nl.arange``, ``nl.mgrid``, ``nl.iota`` or ``nl.affine_select`` to build one with. The rule is
then a closed form over ``maximum`` and ``minimum`` rather than a compare-and-select, and runs
in float32 end to end because that is the dtype ``tensor_scalar`` requires of a tile operand.
"""

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

SENTINEL = -1
"""What a column that selects no token holds. The producer's value, written here, masked there."""

_SUPPORTED_DTYPES = (torch.int32,)
"""Position dtypes that take the NKI route. int32 is what the indexer carries and what the consumer
reads; a wider index type would double SBUF traffic for a range no sequence length reaches."""


class DsaCausalFillError(ValueError):
    """A malformed call: a non-positive ``width``, or ``positions`` of the wrong rank or dtype."""


@dataclass
class _CausalFillDispatchCounters:
    """Per-process record of how the seam below was reached."""

    nki_dispatch: int = 0
    torch_fallback: int = 0
    last_kernel: tuple[str, str] | None = None


_COUNTERS = _CausalFillDispatchCounters()


def reset_causal_fill_dispatch_counters() -> None:
    """Zero the counters."""
    _COUNTERS.nki_dispatch = 0
    _COUNTERS.torch_fallback = 0
    _COUNTERS.last_kernel = None


def causal_fill_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last reset."""
    return (_COUNTERS.nki_dispatch, _COUNTERS.torch_fallback)


@torch._dynamo.assume_constant_result
def _count_torch_fallback() -> None:
    """Count one torch-path entry, off the traced graph."""
    _COUNTERS.torch_fallback += 1


@torch._dynamo.assume_constant_result
def _count_nki_dispatch() -> None:
    """Count one kernel dispatch, off the traced graph: a store Dynamo traces becomes a guard."""
    _COUNTERS.nki_dispatch += 1


def causal_fill_kernel_identity() -> tuple[str, str] | None:
    """``(module, qualname)`` of the kernel the seam last dispatched, or ``None``."""
    return _COUNTERS.last_kernel


def _kernel_identity_of(kernel) -> tuple[str, str]:
    """``(module, qualname)`` of the function a ``@nki.jit`` object wraps.

    Reading those attributes off the decorated object would report the decorator instead:
    ``@nki.jit`` returns a kernel object whose ``__module__`` is ``"nki.framework.kernel"`` and
    whose ``__qualname__`` is ``None``. The wrapped function is reachable at ``__wrapped__`` (the
    ``functools.wraps`` convention) and at ``.func`` (this decorator's own name).
    """
    inner = getattr(kernel, "__wrapped__", None) or getattr(kernel, "func", None) or kernel
    return (inner.__module__, inner.__qualname__)


# ---------------------------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------------------------


@nki.jit
def _causal_fill_nki(positions_hbm, width):
    """Each row's causal prefix, ``-1`` past its own position.

    Args:
        positions_hbm: ``[rows, 1]`` int32 -- each query row's own absolute position, already a
            column, because it reaches ``tensor_scalar`` as a per-row scalar operand.
        width: python int, the expanded width. A trace-time constant, which it must be because it
            sizes the output tile; ``wrap_nki`` passes an int through unchanged.

    Returns:
        ``[rows, width]`` int32, holding ``c`` in column ``c`` of every row whose position is at
        least ``c`` and ``-1`` in every other column.

    The row axis is tiled at ``nl.tile_size.pmax`` because a row is a partition and the partition
    axis serves 128: an untiled fill dies above 128 rows inside the vendor's own check
    (``AssertionError: dma_copy dst partition dimension 132 exceeds maximum 128``). The tiling is
    layout and not arithmetic -- row ``i``'s output depends on ``positions[i]`` and the column index
    and on nothing else, and there is no reduction across rows anywhere -- so a tile boundary may
    fall at any row without changing a single output value.
    """
    rows_total = positions_hbm.shape[0]
    pmax = nl.tile_size.pmax
    n_tiles = (rows_total + pmax - 1) // pmax

    out = nl.ndarray((rows_total, width), dtype=nl.int32, buffer=nl.shared_hbm)

    for t in range(n_tiles):
        # A short final tile is narrowed rather than padded, so no partition without a row can
        # contribute one.
        rows = min(pmax, rows_total - t * pmax)
        off = t * pmax

        # `pos` is the only tile this kernel passes as a `tensor_scalar` operand, and the ISA
        # requires such an operand to be float32; an int32 tile here is refused by the MLIR verifier
        # with `'nisa.tensor_scalar_arith' op 'operand0' must be float32, got 'i32'`. The values are
        # unchanged rather than merely close: the engine casts `data` to float32 and does the
        # arithmetic there regardless, and every quantity involved is a whole number well below
        # 2**24 -- a position is at most `select_k * pool + pool - 2` and a column at most
        # `width - 1` -- so float32 holds each one exactly.
        #
        # Loaded per tile because it is this kernel's only per-row operand: a load hoisted out of
        # the loop would bound every tile by the first tile's row count.
        pos = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=pos, src=nl.load(positions_hbm.ap(pattern=[[1, rows], [1, 1]], offset=off))
        )

        # `channel_multiplier=0` gives every row the same 0..width-1 ramp, so it carries no row
        # state and a tile boundary cannot change it. It stays float32 and is used directly, which
        # is what keeps the closed form below in a single dtype.
        ramp = nl.ndarray((rows, width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(dst=ramp, pattern=[[1, width]], offset=0, channel_multiplier=0)

        # `1 - c`, as one two-scalar chain. Spelling it this way rather than as `-(c - 1)` keeps the
        # tile operand and the scalar operands in separate `tensor_scalar` calls; a chain that mixes
        # the two is not a form this image validates.
        one_minus = nl.ndarray((rows, width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=one_minus, data=ramp,
                           op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0)

        # `room = positions[i] - c + 1`, at least 1 exactly while `c <= positions[i]` and at most 0
        # after it. One tile operand, broadcast along the free axis: a `(1, N)` row operand is
        # refused by the MLIR verifier in this position, which is why `positions` arrives shaped as a
        # column.
        room = nl.ndarray((rows, width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=room, data=one_minus, op0=nl.add, operand0=pos)

        # `keep = clamp(room, 0, 1)`.
        keep = nl.ndarray((rows, width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=keep, data=room,
                           op0=nl.maximum, operand0=0.0, op1=nl.minimum, operand1=1.0)

        # `out = (c + 1) * keep - 1`: `c` where the column is causal, `-1` where it is not. The
        # `+1`/`-1` pair is what lets column 0 survive -- a bare `c * keep` would write 0 for a
        # masked column 0 and 0 is a real token index.
        cols1 = nl.ndarray((rows, width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=cols1, data=ramp, op0=nl.add, operand0=1.0)
        prod = nl.ndarray((rows, width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=prod, data1=cols1, data2=keep, op=nl.multiply)
        acc = nl.ndarray((rows, width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=acc, data=prod, op0=nl.subtract, operand0=1.0)

        # The closed form runs in float32 from the ramp to here, so the result is cast to int32
        # exactly once. Casting with `nisa.tensor_copy` rather than inside `nl.store` keeps the
        # conversion a named instruction instead of an implicit property of the store.
        result = nl.ndarray((rows, width), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=result, src=acc)

        # The strided store places this tile's rows at their own row offset in the output.
        nl.store(
            out.ap(pattern=[[width, rows], [1, width]], offset=off * width), value=result
        )

    return out


# ---------------------------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------------------------


@torch._dynamo.assume_constant_result
def _record_nki_dispatch(rows: int, width: int) -> None:
    """Record which kernel the seam dispatched, and log it, off the compiled graph.

    The dispatch branch is traced under ``fullgraph=True``, so a host call Dynamo refuses would
    break it. A folded helper may take ints, strings and dtypes only: Dynamo turns every non-tensor
    argument into a Python constant at trace time, and an ``@nki.jit`` kernel is a frozen dataclass
    it cannot reconstruct. The kernel is therefore read as a module global instead of passed in.
    """
    _COUNTERS.last_kernel = _kernel_identity_of(_causal_fill_nki)
    logger.info("[dsa-causal-fill] kernel=nki rows=%d width=%d", rows, width)


def _validate(positions: Tensor, width: int) -> int:
    """Host-side validation. Returns ``rows``.

    Reads only ``.shape`` and ``.dtype``, never a tensor value, so nothing here forces a
    device-to-host synchronisation or a data-dependent trace. A bad width or position dtype raises
    instead of declining to the torch path: those are malformed calls, and serving one would hide a
    caller bug behind a correct-looking answer.
    """
    if not isinstance(width, int) or isinstance(width, bool):
        raise DsaCausalFillError(
            f"width must be a python int, because it sizes the output tile at trace time; got "
            f"{type(width).__name__} {width!r}"
        )
    if width < 1:
        raise DsaCausalFillError(
            f"width must be at least 1 column; got width={width}. The admissible expanded width "
            f"comes from index_expand.index_expand_width and is a positive multiple of KEY_CHUNK, "
            f"so a width below 1 is a caller error rather than an empty batch"
        )
    if positions.dtype not in _SUPPORTED_DTYPES:
        raise DsaCausalFillError(
            f"positions must be int32, the dtype the indexer carries and the consumer reads; got "
            f"{positions.dtype}"
        )
    if positions.ndim != 1:
        raise DsaCausalFillError(
            f"positions must be 1-D [rows], one absolute position per query row; got shape "
            f"{tuple(positions.shape)}"
        )
    rows = int(positions.shape[0])
    if rows <= 0:
        raise DsaCausalFillError(f"positions must carry at least one row; got {rows}")
    return rows


def can_run_dsa_causal_fill(positions: Tensor, width: int) -> bool:
    """Whether the NKI kernel serves this call. ``False`` sends it to the torch path.

    Narrow on purpose: every malformed call is already refused by :func:`_validate` before this is
    reached, so all that is left to decide is whether NKI is available at all.
    """
    if not can_run_kernel():
        return False
    return positions.dtype in _SUPPORTED_DTYPES and positions.ndim == 1 and width >= 1


def dsa_causal_fill(positions: Tensor, width: int) -> Tensor:
    """Each query row's causal prefix, ``-1`` past its own position.

    Args:
        positions: ``[rows]`` int32 -- each query row's own absolute position. For a decode step
            that is the new token's position; for a prefill row it is the row's index in its
            sequence.
        width: the expanded width, a positive multiple of ``KEY_CHUNK``, obtained from
            ``index_expand.index_expand_width``. Never the raw expansion width, and never recomputed
            here.

    Returns:
        ``[rows, width]`` int32. Column ``c`` of row ``i`` holds ``c`` when ``c <= positions[i]``
        and :data:`SENTINEL` otherwise. The sentinel is a value that the sparse attention kernel
        masks; this function never masks it.

    Raises:
        DsaCausalFillError: for a malformed call -- a non-int ``width``, a ``width`` below 1, a
            ``positions`` that is not int32 or not 1-D, or an empty ``positions``.
    """
    rows = _validate(positions, width)

    if not can_run_dsa_causal_fill(positions, width):
        _count_torch_fallback()
        return dsa_causal_fill_torch_oracle(positions, width)

    # The per-row position reaches `tensor_scalar` as a column operand, so the reshape happens once
    # here rather than per tile on the device.
    pos_col = positions.reshape(rows, 1).contiguous()

    _count_nki_dispatch()
    _record_nki_dispatch(rows, width)
    return wrap_nki(_causal_fill_nki)(pos_col, width)


# ---------------------------------------------------------------------------------------------
# Torch reference
# ---------------------------------------------------------------------------------------------


def dsa_causal_fill_torch_oracle(positions: Tensor, width: int) -> Tensor:
    """CPU reference for the kernel above; also the path taken when NKI is unavailable.

    Kept in upstream's write-then-mask form -- assign the ramp, then overwrite the columns past each
    row's position -- rather than rewritten into the kernel's closed form, so that agreement between
    the two is evidence rather than one spelling agreeing with itself.
    """
    rows = int(positions.shape[0])
    causal_range = torch.arange(width, device=positions.device, dtype=torch.int32)
    out = causal_range[None, :].expand(rows, width).clone()
    out[causal_range[None, :] > positions.to(torch.int32)[:, None]] = SENTINEL
    return out.to(torch.int32)
