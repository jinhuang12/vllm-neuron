# SPDX-License-Identifier: Apache-2.0
"""Pack a padded ragged batch into a dense buffer, and unpack it back.

A ragged batch arrives as ``[batch, max_len, width]``, where sequence ``b`` occupies only
its first ``lengths[b]`` rows and the rest is padding. ``dsa_ragged_pack`` moves the valid
rows down into a dense ``[sum(lengths), width]`` buffer in sequence order;
``dsa_ragged_unpack`` is its exact inverse and fills every padding position with positive
zero, so a pack followed by an unpack reproduces the input bit for bit.

No row is masked by pointing it out of bounds, because an out-of-bounds dynamic-offset store
under LNC2 drops the whole tile and not just the out-of-bounds rows. The pack sends each
padding row to a real in-range trash row, unique to its ``(sequence, position)`` pair, in a
staging buffer over-allocated for that purpose and never read back; the unpack points each
padding row at a real in-range sentinel row that was memset to zero. The unpack gathers that
row rather than multiplying the payload by a 0/1 mask because ``-3.0 * 0.0`` is ``-0.0``,
whose bit pattern differs from ``+0.0`` while comparing equal numerically.

``lengths`` is a sequence of python ints rather than a tensor: the packed length is derived
from it, and deriving it from tensor data would be a host read inside a region the runner
compiles with ``fullgraph=True``. The per-row bounds test still happens on device, which
keeps the kernels' traces independent of the length pattern. The position iota is handed in
as a tensor because this NKI image has no ``nl.arange``, ``nl.mgrid`` or ``nl.iota``. Only
bfloat16 reaches the kernels; every other dtype is served by the torch path, which is also
the CPU reference.
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

# bfloat16 is the only dtype validated on this platform; anything else is served correctly by
# the torch path rather than refused.
_SUPPORTED_DTYPES = (torch.bfloat16,)


class DsaRaggedPackError(ValueError):
    """Raised for an input this module will not hand to the kernel or the torch path."""


@dataclass
class _RaggedPackDispatchCounters:
    """Per-process record of how the two seams below were reached.

    One ``nki_dispatch`` counter serves both directions, so a pack and its inverse read as two
    dispatches rather than one apiece.
    """

    nki_dispatch: int = 0
    torch_fallback: int = 0
    last_kernel: tuple[str, str] | None = None


_COUNTERS = _RaggedPackDispatchCounters()


def reset_ragged_pack_dispatch_counters() -> None:
    """Zero both seams' counters."""
    _COUNTERS.nki_dispatch = 0
    _COUNTERS.torch_fallback = 0
    _COUNTERS.last_kernel = None


def ragged_pack_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last reset, summed over both directions."""
    return (_COUNTERS.nki_dispatch, _COUNTERS.torch_fallback)


@torch._dynamo.assume_constant_result
def _count_nki_dispatch() -> None:
    """Count one kernel dispatch, off the traced graph: a store Dynamo reads becomes a guard."""
    _COUNTERS.nki_dispatch += 1


@torch._dynamo.assume_constant_result
def _count_torch_fallback() -> None:
    """Count one torch-path entry, off the traced graph."""
    _COUNTERS.torch_fallback += 1


def ragged_pack_kernel_identity() -> tuple[str, str] | None:
    """``(module, qualname)`` of the kernel a seam last dispatched, or ``None``."""
    return _COUNTERS.last_kernel


def _kernel_identity_of(kernel) -> tuple[str, str]:
    """``(module, qualname)`` of the function a ``@nki.jit`` object wraps.

    Reading those attributes off the decorated object would report the decorator instead:
    ``@nki.jit`` returns an ``nki.framework.kernel.Kernel`` whose ``__module__`` is
    ``"nki.framework.kernel"`` and whose ``__qualname__`` is ``None``.
    """
    inner = getattr(kernel, "__wrapped__", None) or getattr(kernel, "func", None) or kernel
    return (inner.__module__, inner.__qualname__)


def _add_tile(rows: int, pos, operand):
    """``pos + operand`` as a fresh ``(rows, 1)`` int32 tile, for a ``(rows, 1)`` device operand.

    ``nisa.tensor_tensor`` and deliberately not ``nisa.tensor_scalar``: when
    ``tensor_scalar``'s ``operand0`` is a tile it is a per-partition scalar and must carry one
    entry per partition of ``dst``, so a ``(1, 1)`` tile is refused by the MLIR verifier
    (``'operand0' partition total elements 1 != 'dst' partition total elements 128``).
    ``NKI_SIMULATOR=1`` does not run that verifier, so the rejected form computes correct
    values under simulation and fails only at compile time.
    """
    out = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=out, data1=pos, data2=operand, op=nl.add)
    return out


def _add_const(rows: int, pos, value: int):
    """``pos + value`` as a fresh ``(rows, 1)`` int32 tile, for a trace-time integer.

    ``tensor_scalar`` is right here: a python int is a true scalar and implies no partition
    count at all. Kept separate from ``_add_tile`` so which rule applies is visible at the call
    site instead of being decided inside by the argument's type.
    """
    out = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=out, data=pos, op0=nl.add, operand0=value)
    return out


def _fill(rows: int, value: int):
    """A fresh ``(rows, 1)`` int32 tile holding one trace-time constant in every row."""
    out = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=out, value=value)
    return out


def _row_index(rows: int, pos, len_bc, inside, outside):
    """The per-row indirect index, computed entirely on device. Shared by both kernels.

    ``index[i] = inside[i]`` while ``pos[i] < len_bc``, and ``index[i] = outside[i]`` otherwise
    -- so a position inside its sequence addresses a real row of the dense buffer, and a
    padding position addresses an escape row.

    The caller builds both candidate index tiles, because the two directions need escape rows
    of a different shape. The pack must escape to a row unique per ``(sequence, position)``, so
    its ``outside`` varies with the row (``pos + packed_len + b*max_len``); a single trace-time
    constant would send every padding row in a tile to one address, colliding, which breaks the
    vendor's requirement that scatter indices within a 128-row tile be distinct. The unpack
    escapes to one shared zeroed sentinel, where a constant is exactly right because a
    duplicate gather source is only a read.

    ``len_bc`` is a ``(rows, 1)`` int32 SBUF tile carrying the sequence length in every
    partition, as ``_broadcast_scalar`` builds it; a ``(1, 1)`` tile passed as
    ``tensor_scalar``'s ``operand0`` does not compile, for the reason ``_add_tile`` records.

    ``1 - valid`` is built as ``valid * -1 + 1`` rather than with a reverse-subtract op, because
    ``nl.multiply`` and ``nl.add`` are both proven on this image and a reverse-subtract is not.
    """
    valid = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=valid, data1=pos, data2=len_bc, op=nl.less)

    notv = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=notv, data=valid, op0=nl.multiply, operand0=-1)
    nisa.tensor_scalar(dst=notv, data=notv, op0=nl.add, operand0=1)

    lhs = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=lhs, data1=valid, data2=inside, op=nl.multiply)
    rhs = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=rhs, data1=notv, data2=outside, op=nl.multiply)
    index = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=index, data1=lhs, data2=rhs, op=nl.add)
    return index


def _broadcast_scalar(hbm, row: int, rows: int):
    """One int32 from a ``[n, 1]`` HBM tensor, replicated into every partition of a ``(rows, 1)`` tile.

    A zero partition stride does the replication: the access pattern advances 0 elements per
    partition step, so all ``rows`` partitions read the same address.

    ``rows`` is a parameter because the replication width must be the partition count of the
    tile that consumes the result, and the last position tile of a sequence is short whenever
    ``max_len`` is not a multiple of ``pmax``. That is why callers invoke this inside the tile
    loop rather than once per sequence: one hoisted width would be wrong for the short tile.
    """
    tile = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=tile, src=hbm.ap(pattern=[[0, rows], [1, 1]], offset=row))
    return tile


@nki.jit
def _ragged_pack_nki(padded_hbm, pos_hbm, lengths_hbm, offsets_hbm, packed_len):
    """Move each sequence's valid rows down into a dense buffer.

    Args:
        padded_hbm: ``[batch * max_len, width]`` -- the padded batch, flattened so a sequence
            and a position together address one row.
        pos_hbm: ``[max_len, 1]`` int32 -- the position iota, handed in because this image has
            no device iota primitive.
        lengths_hbm: ``[batch, 1]`` int32 -- valid rows per sequence.
        offsets_hbm: ``[batch, 1]`` int32 -- each sequence's exclusive prefix-sum destination.
        packed_len: rows in the dense output. A compile-time constant.

    Returns:
        ``[packed_len, width]`` in ``padded_hbm``'s dtype.

    Padding rows go to a unique trash row above the dense region: position ``s`` of sequence
    ``b`` escapes to ``packed_len + b * max_len + s``. Uniqueness matters because the vendor
    requires scatter indices within a 128-row tile to be distinct; this formula never collides,
    and its in-range rows cover the dense region exactly once.
    """
    n_rows, width = padded_hbm.shape
    max_len = pos_hbm.shape[0]
    batch = lengths_hbm.shape[0]
    pmax = nl.tile_size.pmax

    # Over-allocated by the whole padded row count, the largest the trash region can need.
    staging = nl.ndarray((packed_len + n_rows, width), dtype=padded_hbm.dtype, buffer=nl.private_hbm)
    out_hbm = nl.ndarray((packed_len, width), dtype=padded_hbm.dtype, buffer=nl.shared_hbm)

    # Zero the dense region before anything is scattered into it. Every one of its rows is
    # written exactly once by construction, so this is not needed for correctness -- it makes an
    # indexing bug show up as a zero row instead of as whatever the buffer happened to hold and
    # might coincidentally match. The trash region is deliberately not zeroed: it is never read,
    # and zeroing it would cost a pass over as much memory again.
    n_dense_tiles = (packed_len + pmax - 1) // pmax
    for t in range(n_dense_tiles):
        rows = min(pmax, packed_len - t * pmax)
        z = nl.ndarray((rows, width), dtype=padded_hbm.dtype, buffer=nl.sbuf)
        nisa.memset(dst=z, value=0)
        nisa.dma_copy(
            dst=staging.ap(pattern=[[width, rows], [1, width]], offset=t * pmax * width), src=z
        )

    n_pos_tiles = (max_len + pmax - 1) // pmax
    for b in range(batch):
        for t in range(n_pos_tiles):
            # A short final tile is narrowed rather than padded, so no padded row can reach
            # the output.
            rows = min(pmax, max_len - t * pmax)
            # Broadcast-loaded inside the tile loop, because the replication width is
            # ``rows`` and the short final tile has a different one. Two extra small DMA reads
            # per tile, against a hoisted form that does not compile at all.
            len_bc = _broadcast_scalar(lengths_hbm, b, rows)
            off_bc = _broadcast_scalar(offsets_hbm, b, rows)
            pos = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=pos, src=pos_hbm.ap(pattern=[[1, rows], [1, 1]], offset=t * pmax)
            )
            # A unique trash row per (sequence, position): ``pos`` holds absolute positions,
            # so this is ``packed_len + b*max_len + s``.
            dst_index = _row_index(
                rows,
                pos,
                len_bc,
                _add_tile(rows, pos, off_bc),
                _add_const(rows, pos, packed_len + b * max_len),
            )
            payload = nl.ndarray((rows, width), dtype=padded_hbm.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=payload,
                src=padded_hbm.ap(
                    pattern=[[width, rows], [1, width]],
                    offset=(b * max_len + t * pmax) * width,
                ),
            )
            nisa.dma_copy(
                dst=staging.ap(
                    pattern=[[width, rows], [1, width]],
                    offset=0,
                    vector_offset=dst_index,
                    indirect_dim=0,
                ),
                src=payload,
            )

    for t in range(n_dense_tiles):
        rows = min(pmax, packed_len - t * pmax)
        tile = nl.ndarray((rows, width), dtype=padded_hbm.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=tile,
            src=staging.ap(pattern=[[width, rows], [1, width]], offset=t * pmax * width),
        )
        nl.store(
            out_hbm.ap(pattern=[[width, rows], [1, width]], offset=t * pmax * width), value=tile
        )
    return out_hbm


@nki.jit
def _ragged_unpack_nki(packed_hbm, pos_hbm, lengths_hbm, offsets_hbm, max_len):
    """Put every packed row back where it came from, and zero every padding position.

    Args:
        packed_hbm: ``[packed_len, width]`` -- the dense buffer.
        pos_hbm: ``[max_len, 1]`` int32 -- the position iota.
        lengths_hbm: ``[batch, 1]`` int32 -- valid rows per sequence.
        offsets_hbm: ``[batch, 1]`` int32 -- each sequence's exclusive prefix-sum source.
        max_len: padded rows per sequence. A compile-time constant.

    Returns:
        ``[batch * max_len, width]`` in ``packed_hbm``'s dtype, padding positions exactly
        ``+0.0``.

    Padding rows read a sentinel row that was memset to zero, which is why the padding comes
    back as positive zero and not as negative zero. Many rows reading one sentinel is a
    duplicate gather source, a read rather than a write, so the vendor's distinct-index
    requirement on scatters does not apply.
    """
    packed_len, width = packed_hbm.shape
    batch = lengths_hbm.shape[0]
    pmax = nl.tile_size.pmax

    # The packed rows, plus one zeroed sentinel row at index ``packed_len``.
    staging = nl.ndarray((packed_len + 1, width), dtype=packed_hbm.dtype, buffer=nl.private_hbm)
    n_dense_tiles = (packed_len + pmax - 1) // pmax
    for t in range(n_dense_tiles):
        rows = min(pmax, packed_len - t * pmax)
        tile = nl.ndarray((rows, width), dtype=packed_hbm.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=tile,
            src=packed_hbm.ap(pattern=[[width, rows], [1, width]], offset=t * pmax * width),
        )
        nisa.dma_copy(
            dst=staging.ap(pattern=[[width, rows], [1, width]], offset=t * pmax * width), src=tile
        )
    sentinel = nl.ndarray((1, width), dtype=packed_hbm.dtype, buffer=nl.sbuf)
    nisa.memset(dst=sentinel, value=0)
    nisa.dma_copy(
        dst=staging.ap(pattern=[[width, 1], [1, width]], offset=packed_len * width), src=sentinel
    )

    out_hbm = nl.ndarray((batch * max_len, width), dtype=packed_hbm.dtype, buffer=nl.shared_hbm)
    n_pos_tiles = (max_len + pmax - 1) // pmax
    for b in range(batch):
        for t in range(n_pos_tiles):
            rows = min(pmax, max_len - t * pmax)
            # As in the pack kernel: the replication width is ``rows``, so this cannot be hoisted.
            len_bc = _broadcast_scalar(lengths_hbm, b, rows)
            off_bc = _broadcast_scalar(offsets_hbm, b, rows)
            pos = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=pos, src=pos_hbm.ap(pattern=[[1, rows], [1, 1]], offset=t * pmax)
            )
            # One shared zeroed sentinel row for every padding position, at ``packed_len``.
            src_index = _row_index(
                rows, pos, len_bc, _add_tile(rows, pos, off_bc), _fill(rows, packed_len)
            )
            got = nl.ndarray((rows, width), dtype=packed_hbm.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=got,
                src=staging.ap(
                    pattern=[[width, rows], [1, width]],
                    offset=0,
                    vector_offset=src_index,
                    indirect_dim=0,
                ),
            )
            nl.store(
                out_hbm.ap(
                    pattern=[[width, rows], [1, width]],
                    offset=(b * max_len + t * pmax) * width,
                ),
                value=got,
            )
    return out_hbm


@torch._dynamo.assume_constant_result
def _record_nki_dispatch(
    direction: str, batch: int, max_len: int, width: int, packed_len: int
) -> None:
    """Record which kernel a seam dispatched, and log it, off the compiled graph.

    Both dispatch branches are traced under ``fullgraph=True``, so a host call Dynamo refuses
    would break them. A folded helper may take ints, strings and dtypes only: Dynamo converts
    every non-tensor argument into a python constant at trace time, and an ``@nki.jit`` kernel
    is an ``nki.framework.kernel.Kernel``, a frozen dataclass it cannot reconstruct. So the
    direction arrives as a ``str`` and both kernels are read as module globals. One helper
    serves both directions so there is only one fold to keep in step.
    """
    kernel = _ragged_pack_nki if direction == "pack" else _ragged_unpack_nki
    _COUNTERS.last_kernel = _kernel_identity_of(kernel)
    logger.info(
        "[dsa-ragged-pack] kernel=nki direction=%s batch=%d max_len=%d width=%d packed_len=%d",
        direction,
        batch,
        max_len,
        width,
        packed_len,
    )


def _validate_lengths(lengths: Sequence[int], max_len: int) -> tuple[int, ...]:
    """Host-side validation of the ragged shape. Python ints only, so no tensor is read."""
    if not isinstance(lengths, Sequence) or isinstance(lengths, (str, bytes)):
        raise DsaRaggedPackError(f"lengths must be a sequence of ints; got {type(lengths)!r}")
    if len(lengths) == 0:
        raise DsaRaggedPackError("lengths is empty; there is nothing to pack")
    out = []
    for i, length in enumerate(lengths):
        if isinstance(length, bool) or not isinstance(length, int):
            raise DsaRaggedPackError(f"lengths[{i}] must be an int; got {type(length)!r}")
        if length < 0 or length > max_len:
            raise DsaRaggedPackError(
                f"lengths[{i}] must lie in [0, {max_len}]; got {length}"
            )
        out.append(length)
    return tuple(out)


def _exclusive_offsets(lengths: Sequence[int]) -> tuple[int, ...]:
    """The exclusive prefix sum -- where each sequence starts in the dense buffer.

    Host python over at most a handful of ints; the module docstring says why ``lengths`` is not
    a tensor.
    """
    offsets = []
    acc = 0
    for length in lengths:
        offsets.append(acc)
        acc += length
    return tuple(offsets)


def _column(values: Sequence[int], device: torch.device) -> Tensor:
    """A ``[n, 1]`` int32 column built from python integers, in a form that traces.

    ``torch.tensor(<python list>)`` cannot be used here: inside a traced region it materialises
    a real tensor under ``FakeTensorMode`` and the next operation on it asserts with "Please
    convert all Tensors to FakeTensors first", whatever that operation is. ``torch.full`` and
    ``torch.cat`` are aten operations, so they produce fake tensors and trace cleanly.

    ``values`` is never empty on an admitted geometry -- ``can_run_dsa_ragged_pack`` requires
    ``sum(lengths) > 0`` -- so no guard is added for a case the module does not admit.
    """
    return torch.cat(
        [torch.full((1, 1), int(v), dtype=torch.int32, device=device) for v in values]
    )


def _metadata(
    lengths: Sequence[int], max_len: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    """The three int32 metadata tensors the kernels read: the iota, the lengths, the offsets."""
    pos = torch.arange(max_len, dtype=torch.int32, device=device).reshape(-1, 1).contiguous()
    len_t = _column(lengths, device)
    off_t = _column(_exclusive_offsets(lengths), device)
    return pos, len_t.contiguous(), off_t.contiguous()


def can_run_dsa_ragged_pack(padded: Tensor, lengths: Sequence[int]) -> bool:
    """True when the NKI route is available and this module admits this pack geometry.

    The conditions are the runtime gate, the rank of the padded batch, the admitted dtype, a
    batch size matching the lengths, and at least one row to move.
    """
    if not can_run_kernel(padded):
        return False
    if padded.ndim != 3:
        return False
    if padded.dtype not in _SUPPORTED_DTYPES:
        return False
    if len(lengths) != int(padded.shape[0]):
        return False
    if sum(lengths) <= 0:
        return False
    return True


def can_run_dsa_ragged_unpack(packed: Tensor, lengths: Sequence[int], max_len: int) -> bool:
    """True when the NKI route is available and this module admits this unpack geometry."""
    if not can_run_kernel(packed):
        return False
    if packed.ndim != 2:
        return False
    if packed.dtype not in _SUPPORTED_DTYPES:
        return False
    if int(packed.shape[0]) != sum(lengths):
        return False
    if max_len <= 0 or sum(lengths) <= 0:
        return False
    return True


def dsa_ragged_pack(padded: Tensor, lengths: Sequence[int]) -> Tensor:
    """Move each sequence's valid rows into a dense buffer.

    Args:
        padded: ``[batch, max_len, width]`` -- the padded batch. ``bfloat16`` takes the NKI
            route; any other dtype is served by the torch path.
        lengths: valid rows per sequence, as python ints. The module docstring gives the reason
            this is not a tensor.

    Returns:
        ``[sum(lengths), width]`` in ``padded``'s dtype: the sequences' valid rows, in order,
        with no padding row among them.

    Raises:
        DsaRaggedPackError: for a malformed call -- a non-3D ``padded``, a lengths sequence that
            does not describe its batch, or a length outside ``[0, max_len]``.
    """
    if padded.ndim != 3:
        raise DsaRaggedPackError(
            f"padded must be 3-D [batch, max_len, width]; got shape {tuple(padded.shape)}"
        )
    batch, max_len, width = (int(padded.shape[0]), int(padded.shape[1]), int(padded.shape[2]))
    checked = _validate_lengths(lengths, max_len)
    if len(checked) != batch:
        raise DsaRaggedPackError(
            f"lengths must describe every sequence; got {len(checked)} for batch {batch}"
        )
    packed_len = sum(checked)
    if packed_len == 0:
        raise DsaRaggedPackError("every sequence is empty; there is nothing to pack")

    if not can_run_dsa_ragged_pack(padded, checked):
        return _dsa_ragged_pack_torch(padded, checked)

    pos, len_t, off_t = _metadata(checked, max_len, padded.device)
    flat = padded.reshape(batch * max_len, width).contiguous()

    _count_nki_dispatch()
    # The counter, the log and the identity read are folded off the traced graph: a counter
    # store inside the trace becomes a value guard that fails on the first call after warmup.
    _record_nki_dispatch("pack", batch, max_len, width, packed_len)
    return wrap_nki(_ragged_pack_nki)(flat, pos, len_t, off_t, packed_len)


def dsa_ragged_unpack(packed: Tensor, lengths: Sequence[int], max_len: int) -> Tensor:
    """Put packed rows back where they came from, and zero the padding.

    Args:
        packed: ``[sum(lengths), width]`` -- the dense buffer.
        lengths: valid rows per sequence, as python ints.
        max_len: padded rows per sequence in the result.

    Returns:
        ``[batch, max_len, width]`` in ``packed``'s dtype, every padding position exactly
        ``+0.0``.

    Raises:
        DsaRaggedPackError: for a malformed call -- a non-2D ``packed``, a non-positive
            ``max_len``, a length outside ``[0, max_len]``, or a row count that the lengths do
            not sum to.
    """
    if packed.ndim != 2:
        raise DsaRaggedPackError(
            f"packed must be 2-D [packed_len, width]; got shape {tuple(packed.shape)}"
        )
    if not isinstance(max_len, int) or isinstance(max_len, bool) or max_len <= 0:
        raise DsaRaggedPackError(f"max_len must be a positive int; got {max_len!r}")
    checked = _validate_lengths(lengths, max_len)
    packed_len = sum(checked)
    if int(packed.shape[0]) != packed_len:
        raise DsaRaggedPackError(
            f"packed has {int(packed.shape[0])} rows but the lengths sum to {packed_len}"
        )
    if packed_len == 0:
        raise DsaRaggedPackError("every sequence is empty; there is nothing to unpack")

    batch = len(checked)
    width = int(packed.shape[1])
    if not can_run_dsa_ragged_unpack(packed, checked, max_len):
        return _dsa_ragged_unpack_torch(packed, checked, max_len)

    pos, len_t, off_t = _metadata(checked, max_len, packed.device)

    _count_nki_dispatch()
    _record_nki_dispatch("unpack", batch, max_len, width, packed_len)
    flat = wrap_nki(_ragged_unpack_nki)(packed.contiguous(), pos, len_t, off_t, max_len)
    return flat.reshape(batch, max_len, width)


def _dsa_ragged_pack_torch(padded: Tensor, lengths: Sequence[int]) -> Tensor:
    """CPU reference and fallback: reached without the simulator, with kernels off, or on an unadmitted dtype."""
    _count_torch_fallback()
    logger.info(
        "[dsa-ragged-pack] kernel=torch direction=pack batch=%d max_len=%d width=%d "
        "packed_len=%d reason=nki-route-unavailable",
        int(padded.shape[0]),
        int(padded.shape[1]),
        int(padded.shape[2]),
        sum(lengths),
    )
    return torch.cat([padded[b, : lengths[b], :] for b in range(len(lengths))], dim=0)


def _dsa_ragged_unpack_torch(packed: Tensor, lengths: Sequence[int], max_len: int) -> Tensor:
    """CPU reference and fallback for the unpack direction.

    Zeros first and then writes the valid rows, so a padding position is a true ``+0.0`` here
    too and this agrees with the kernel bit for bit rather than only numerically.
    """
    _count_torch_fallback()
    logger.info(
        "[dsa-ragged-pack] kernel=torch direction=unpack batch=%d max_len=%d width=%d "
        "packed_len=%d reason=nki-route-unavailable",
        len(lengths),
        max_len,
        int(packed.shape[1]),
        sum(lengths),
    )
    out = torch.zeros(
        (len(lengths), max_len, int(packed.shape[1])), dtype=packed.dtype, device=packed.device
    )
    for b, offset in enumerate(_exclusive_offsets(lengths)):
        out[b, : lengths[b], :] = packed[offset : offset + lengths[b], :]
    return out
