# SPDX-License-Identifier: Apache-2.0
"""DSA ragged pack and unpack -- an AUTHORED bf16 NKI kernel pair (`inc-glm53f-045`).

WHAT THIS DOES. A ragged batch arrives padded: ``[batch, max_len, width]``, where sequence ``b``
occupies only its first ``lengths[b]`` rows and the rest is padding. ``dsa_ragged_pack`` moves the
valid rows down into a dense ``[sum(lengths), width]`` buffer, in sequence order, with no padding
row in it. ``dsa_ragged_unpack`` is its exact inverse: it puts every packed row back where it came
from and fills every padding position with POSITIVE ZERO. Packing then unpacking reproduces the
input bit for bit, which is what this increment's acceptance measures.

WHY IT IS AUTHORED AND NOT A WRAP OF A VENDOR KERNEL. Measured before a line of this file existed,
and recorded in the increment plan at revision 151 as the lead's substrate ruling. The vendor
library on this pin has 507 python modules; the pre-authoring inventory imported the 112 that speak
any pack vocabulary and parsed every callable's signature, and NOT ONE is named or documented as a
sequence pack or unpack -- every name-level "pack" hit is scale packing for quantised weights. The
one structural candidate, ``nkilib.core.subkernels.indexed_flatten``, is refused on five grounds
read off its own assertions: it packs EQUAL-length blocks so raggedness can only enter through its
offsets and never through its lengths; no unpack member exists anywhere to invert it; its
``output_len`` must be divisible by 128, which forbids the one declared pattern whose packed length
deliberately is not; it requires LNC2; and its padding is an integer fill over index-typed data
rather than a bit-identical bf16 round trip. Evidence:
``artifacts/campaigns/glm-5.3-flash-port/increments/probe-045-substrate-reading.md``.

THE MECHANISM WAS MEASURED BEFORE IT WAS USED. Every device-side construct below was proved on this
image, at this dtype and this width, by ``probe-045-mechanism.py`` in a CPU-mode simulator round
before this file was authored -- because authoring against an unmeasured API and discovering it at
the round of record costs a whole repair round. Six arms, all held, at 128 rows and at a 37-row
tail: the per-row indirect SCATTER destination, the per-row indirect GATHER source, the index
vector computed on device from device scalars, the memset sentinel row reading back as positive
zero, duplicate gather sources, and every one of those again on a short tail.

THE IDIOM IS THE VENDOR'S OWN, not invented here. Per-row indirect access is
``.ap(..., vector_offset=<device int32 tile>, indirect_dim=0)``, read off
``nkilib/experimental/misc/scatter_add.py:114-137`` (gather on ``src=``, scatter on ``dst=``). A
per-sequence value is loaded into a ``(1, 1)`` int32 SBUF tile and used as a device operand, read
off ``nkilib/core/subkernels/indexed_flatten.py:177-179``. An over-allocated ``private_hbm``
staging buffer whose real region alone is returned is read off
``nkilib/core/subkernels/indexed_flatten.py:128-130``.

WHY THERE IS NO ``oob_mode.skip`` ANYWHERE IN THIS FILE, and this is a correctness decision rather
than a preference. ``nkilib/core/subkernels/indexed_flatten.py:122-127`` records, in the vendor's
own words, that an out-of-bounds dynamic-offset store under LNC2 drops the WHOLE TILE and not just
the out-of-bounds lanes. Masking a padding row by pointing it out of bounds would therefore have
semantics this campaign has not measured. So no row is ever masked by going out of bounds. Instead:

  - the PACK sends every padding row to a REAL in-range trash row, unique to that ``(sequence,
    position)`` pair, in a staging buffer over-allocated for exactly that purpose. Only the dense
    region is copied out, so the trash is never read;
  - the UNPACK sends every padding row to a REAL in-range SENTINEL row that was memset to zero.

WHY THE UNPACK USES A SENTINEL ROW AND NEVER MULTIPLIES BY A MASK. The obvious way to blank a
padding row is to multiply the gathered payload by a 0/1 mask. That is WRONG for a bit-identity
criterion, and the mechanism probe measured how wrong: in IEEE arithmetic ``-3.0 * 0.0`` is
``-0.0``, whose int16 bit pattern is ``0x8000`` while ``+0.0`` is ``0x0000``. The two compare EQUAL
numerically, so an assertion on ``max abs diff == 0.0`` would pass while the bytes differed. On a
128-row bf16 tile the probe measured 32,652 of 65,536 elements landing on the wrong bit pattern
under the multiply -- roughly every negative element. The sentinel row is gathered instead, and the
probe measured its int16 view as exactly zero in every element at both row counts.

WHY ``lengths`` IS A SEQUENCE OF PYTHON INTS AND NOT A TENSOR. Two reasons, and the second is the
one that matters. First, it is the fork's house style for exactly this job:
``vllm_neuron/model/qwen3_vl/utils/vision_block_packing.py`` takes ``tokens_per_image: list[int]``
and does its bin packing in host python before anything is traced. Second, the packed length is
DERIVED here from the lengths, and deriving it from tensor DATA would be a host read of a tensor
inside a region the runner compiles with ``fullgraph=True``
(``vllm_neuron/vllm/worker/neuron_model_runner.py:1457-1462``) -- a graph break. With python ints
the sum is a compile-time constant and nothing breaks. The acceptance then has teeth: the caller
never states the packed length, so ``packed.shape[0]`` versus the closed form is a real measurement
rather than an echo of an argument.

WHY THE BOUNDS TEST IS STILL DONE ON DEVICE when the lengths are known at trace time. Because it
keeps the kernel's trace INDEPENDENT of the length pattern: one compiled kernel serves every batch
composition that shares a ``(batch, max_len, packed_len)`` bucket, where host-side slicing would
force a fresh trace per composition. This is the same compile-stability argument
``vision_block_packing.py`` makes in its own docstring. Stated as design rationale, not as a
measured claim -- this increment's declared case set has no two patterns sharing a bucket, so it
does not measure trace reuse and this file does not claim it.

WHY THE POSITION IOTA COMES IN AS A TENSOR. This NKI image has no ``nl.arange``, no ``nl.mgrid``
and no ``nl.iota`` -- a fact `-044` measured on this pin. So a per-row position vector cannot be
generated on device and is handed in. It is metadata, not payload.

WHY ``bfloat16`` ONLY. It is the dtype this increment's substrate declaration names and the only
one the campaign has measured on this pin. Every other dtype is served correctly by the torch path,
and the test drives that path deliberately rather than leaving it unmeasured.

SUBSTRATE (P13). KERNEL-CLASS, and the movement is in NKI. What is left in torch is orchestration
and glue, named exactly so a reviewer can check it: shape validation, the exclusive prefix sum of
at most a handful of python ints, and building the three int32 metadata tensors. No torch-level
fallback is on the shipped path; ``_dsa_ragged_pack_torch`` is the CPU oracle and the
constraint-violation route (D6), and it counts itself so a test can say which route ran.

DECLARED ACCEPTANCE (D1 Tier N)::

    PYTHONDONTWRITEBYTECODE=1 VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \\
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \\
    python -m pytest test/vllm_neuron/functional/dsa/test_ragged_pack.py \\
        --timeout 60 -v -s -p no:cacheprovider
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

# The dtypes this module admits to the kernel. bfloat16 only; the module docstring gives the reason.
_SUPPORTED_DTYPES = (torch.bfloat16,)


class DsaRaggedPackError(ValueError):
    """Raised for an input this module will not hand to the kernel or to the oracle."""


@dataclass
class _RaggedPackDispatchCounters:
    """How the two seams below were reached, per process.

    ONE ``nki_dispatch`` counter for BOTH directions, which is what makes the increment's route
    predicate read ``2`` per declared pattern -- one for the pack and one for the unpack -- rather
    than two separate ones. A single-direction reading would leave the inverse unmeasured, and the
    round trip is what the acceptance asserts. ``last_kernel`` records the kernel a seam actually
    dispatched, so an identity can be derived THROUGH the seam rather than read off an import
    (D13.1).
    """

    nki_dispatch: int = 0
    torch_fallback: int = 0
    last_kernel: tuple[str, str] | None = None


_COUNTERS = _RaggedPackDispatchCounters()


def reset_ragged_pack_dispatch_counters() -> None:
    """Zero both seams' counters. Call immediately before a case's first call (§4b)."""
    _COUNTERS.nki_dispatch = 0
    _COUNTERS.torch_fallback = 0
    _COUNTERS.last_kernel = None


def ragged_pack_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last reset, summed over both directions."""
    return (_COUNTERS.nki_dispatch, _COUNTERS.torch_fallback)


def ragged_pack_kernel_identity() -> tuple[str, str] | None:
    """``(module, qualname)`` of the kernel a seam LAST dispatched, or ``None``.

    Derived through the seam, not from this module's import list, so it certifies what ran rather
    than what was defined (D13.1). ``None`` before any dispatch, which is the reading that
    distinguishes "no kernel ran" from "some kernel ran".
    """
    return _COUNTERS.last_kernel


def _kernel_identity_of(kernel) -> tuple[str, str]:
    """``(module, qualname)`` of the function a ``@nki.jit`` object actually wraps.

    MEASURED, not assumed, for the reason ``vllm_neuron/functional/dsa/paged_gather.py:177-189``
    records at its own equivalent: ``@nki.jit`` returns an ``nki.framework.kernel.Kernel`` whose
    own ``__module__`` is ``"nki.framework.kernel"`` and whose ``__qualname__`` is ``None``, so
    reading those two attributes off the decorated object would record the DECORATOR's identity and
    certify nothing about which kernel was wrapped.
    """
    inner = getattr(kernel, "__wrapped__", None) or getattr(kernel, "func", None) or kernel
    return (inner.__module__, inner.__qualname__)


def _add_tile(rows: int, pos, operand):
    """``pos + operand`` as a fresh ``(rows, 1)`` int32 tile, for a ``(rows, 1)`` DEVICE operand.

    ``nisa.tensor_tensor`` and deliberately NOT ``nisa.tensor_scalar``. When ``tensor_scalar``'s
    ``operand0`` is a tile it is a PER-PARTITION scalar and must carry one entry per partition of
    ``dst``, so a ``(1, 1)`` tile is refused outright. The MLIR verifier says so in as many words --
    ``'nisa.tensor_scalar_arith' op 'operand0' partition total elements 1 != 'dst' partition total
    elements 128`` at ``nki/isa/_tensor_ops.py:401`` -- and the operation dump names both shapes,
    ``1, 1`` against ``128, 1``.

    THE SIMULATOR CANNOT CATCH THIS. ``NKI_SIMULATOR=1`` does not run the MLIR verifier, so this
    file's own mechanism round read the rejected form as correct. A simulator pass is evidence of
    VALUES and never of COMPILABILITY. Compilability is settled by
    ``probe-045-broadcast-r3-host.out``, where this form compiles with one captured graph and holds
    the index exactly at 128 rows and at a 37-row tail, and the ``tensor_scalar`` form does not
    compile at all while computing identical values.
    """
    out = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=out, data1=pos, data2=operand, op=nl.add)
    return out


def _add_const(rows: int, pos, value: int):
    """``pos + value`` as a fresh ``(rows, 1)`` int32 tile, for a TRACE-TIME INTEGER.

    ``tensor_scalar`` is right here and stays: a python int is a true scalar and implies no partition
    count at all. Kept separate from ``_add_tile`` so that which rule applies is visible at the call
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
    """The per-row indirect index, computed ENTIRELY ON DEVICE. Shared by both kernels.

    ``index[i] = inside[i]`` while ``pos[i] < len_bc``, and ``index[i] = outside[i]`` otherwise --
    so a position inside its sequence addresses a real row of the dense buffer, and a padding
    position addresses an escape row.

    THE TWO CANDIDATE INDEX TILES ARE BUILT BY THE CALLER, AND THAT IS DELIBERATE, because the two
    directions need escape rows of a different SHAPE and folding them into one argument here was a
    defect I caught in this file before it shipped. The pack must escape to a row that is unique per
    ``(sequence, position)``, so its ``outside`` VARIES WITH THE ROW (``pos + packed_len +
    b*max_len``); a single trace-time constant would have sent every padding row in a tile to ONE
    address, colliding, in breach of the vendor's own per-tile uniqueness constraint at
    ``nkilib/experimental/misc/scatter_add.py:62``. The unpack escapes to one shared zeroed
    sentinel, where a constant is exactly right because a duplicate GATHER source is a read.

    ``len_bc`` is a ``(rows, 1)`` int32 SBUF tile carrying the sequence length in EVERY partition,
    as ``_broadcast_scalar`` builds it. It was a ``(1, 1)`` tile passed as ``tensor_scalar``'s
    ``operand0``, and that form does not compile -- ``_add_tile`` records the verifier's own words.
    The mechanism probe measured this arithmetic exact at 128 rows and at a 37-row tail, by both an
    ``nl.less`` route and a comparison-free clamp route; this is the ``nl.less`` route, chosen for
    having fewer ops, and ``probe-045-broadcast-r3-host.out`` re-measured it exact through the
    broadcast form at both sizes.

    ``1 - valid`` is built as ``valid * -1 + 1`` rather than with a reverse-subtract op, because
    ``nl.multiply`` and ``nl.add`` are both proven on this image by `-044`'s own kernel and a
    reverse-subtract is not. Two cheap proven ops beat one unproven one.
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
    """One int32 from a ``[n, 1]`` HBM tensor, replicated into EVERY partition of a ``(rows, 1)`` tile.

    A ZERO PARTITION STRIDE does the replication: the access pattern advances 0 elements per
    partition step, so all ``rows`` partitions read the same address. This is the vendor's own idiom
    for this exact job. ``nkilib/experimental/collectives/a2av_train/permute_a2av.py:139-150``
    broadcast-loads a per-destination displacement this way and then combines it with
    ``tensor_tensor``, reserving ``tensor_scalar``'s ``operand0`` for a python int -- which is the
    rule this file previously broke.

    ``rows`` is a parameter because the replication width must be the partition count of the tile
    that consumes the result, and the last position tile of a sequence is short whenever ``max_len``
    is not a multiple of ``pmax``. That is why callers invoke this inside the tile loop rather than
    once per sequence: one hoisted width would be wrong for the short tile.
    """
    tile = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=tile, src=hbm.ap(pattern=[[0, rows], [1, 1]], offset=row))
    return tile


@nki.jit
def _ragged_pack_nki(padded_hbm, pos_hbm, lengths_hbm, offsets_hbm, packed_len):
    """Move each sequence's valid rows down into a dense buffer.

    Args:
        padded_hbm: ``[batch * max_len, width]`` -- the padded batch, flattened so a sequence and
            a position together address one row.
        pos_hbm: ``[max_len, 1]`` int32 -- the position iota, handed in because this image has no
            device iota primitive.
        lengths_hbm: ``[batch, 1]`` int32 -- valid rows per sequence.
        offsets_hbm: ``[batch, 1]`` int32 -- each sequence's exclusive prefix-sum destination.
        packed_len: rows in the dense output. A compile-time constant.

    Returns:
        ``[packed_len, width]`` in ``padded_hbm``'s dtype.

    Padding rows go to a UNIQUE trash row above the dense region: position ``s`` of sequence ``b``
    escapes to ``packed_len + b * max_len + s``. Unique matters because
    ``nkilib/experimental/misc/scatter_add.py:62`` states, as a vendor constraint, that indices
    within a 128-row tile should be unique for correctness -- and the mechanism probe checked, as
    arithmetic over all four declared patterns, that no two rows this formula emits ever collide
    and that the in-range ones cover the dense region exactly once.
    """
    n_rows, width = padded_hbm.shape
    max_len = pos_hbm.shape[0]
    batch = lengths_hbm.shape[0]
    pmax = nl.tile_size.pmax

    # Over-allocated by the whole padded row count, which is the largest the trash region can ever
    # need. The vendor's own shape at ``nkilib/core/subkernels/indexed_flatten.py:128-130``.
    staging = nl.ndarray((packed_len + n_rows, width), dtype=padded_hbm.dtype, buffer=nl.private_hbm)
    out_hbm = nl.ndarray((packed_len, width), dtype=padded_hbm.dtype, buffer=nl.shared_hbm)

    # Zero the DENSE region before anything is scattered into it. Every one of its rows is written
    # exactly once by construction, so this is not needed for correctness -- it is needed so that
    # an indexing bug shows up as a zero row rather than as whatever the buffer happened to hold
    # and might coincidentally match. `-044`'s kernel zeroes for the same reason. The trash region
    # is deliberately NOT zeroed: it is never read, and zeroing it would cost a pass over as much
    # memory again for no reading.
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
            # The final tile is short whenever max_len is not a multiple of pmax. Handled by
            # narrowing the tile rather than by padding it, so no masked lane can contribute a row.
            rows = min(pmax, max_len - t * pmax)
            # Broadcast-loaded INSIDE the tile loop, because the replication width is ``rows`` and
            # the short final tile has a different one. Two extra small DMA reads per tile, against
            # a hoisted form that does not compile at all.
            len_bc = _broadcast_scalar(lengths_hbm, b, rows)
            off_bc = _broadcast_scalar(offsets_hbm, b, rows)
            pos = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=pos, src=pos_hbm.ap(pattern=[[1, rows], [1, 1]], offset=t * pmax)
            )
            # A UNIQUE trash row per (sequence, position): ``pos`` holds absolute positions, so
            # this is ``packed_len + b*max_len + s``. Unique by construction, and the mechanism
            # probe checked it as arithmetic over all four declared patterns.
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
        ``[batch * max_len, width]`` in ``packed_hbm``'s dtype, padding positions exactly ``+0.0``.

    Padding rows read a SENTINEL row that was memset to zero, which is why the padding comes back
    as positive zero and not as negative zero. The module docstring gives the measurement behind
    that choice. Many rows reading one sentinel is a duplicate GATHER source, which is a read and
    not a write, so the vendor's per-tile uniqueness constraint does not apply -- and the mechanism
    probe measured a 128-row all-duplicate gather correct anyway.
    """
    packed_len, width = packed_hbm.shape
    batch = lengths_hbm.shape[0]
    pmax = nl.tile_size.pmax

    # The packed rows, plus ONE zeroed sentinel row at index ``packed_len``.
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
            # ONE shared zeroed sentinel row for every padding position, at index ``packed_len``.
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
    """Record WHICH kernel a seam dispatched, and log it, OFF the compiled graph.

    THE TEMPLATE THIS FOLLOWS IS LANDED AND MEASURED, not invented here:
    ``vllm_neuron/functional/dsa/paged_gather.py:248-288``, itself the second repair of B58-M1 at
    ``vllm_neuron/functional/dsa/topk_select.py:225-276``. Both branches below are traced on the
    shipped path, so a host call Dynamo refuses would break them. The runner compiles with
    ``fullgraph=True`` unconditionally
    (``vllm_neuron/vllm/worker/neuron_model_runner.py:1457-1462``); CPU mode disables only the
    graph-capture backend (``vllm_neuron/vllm/worker/neuron_model_runner.py:1446-1448``) and only
    ``enforce_eager`` forces eager
    (``vllm_neuron/vllm/worker/neuron_model_runner.py:1246-1247``).

    ARGUMENT DISCIPLINE, WHICH IS THE WHOLE POINT: a folded helper takes ints, strings and dtypes
    ONLY, never an object. Dynamo runs a folded call at trace time and first converts every
    non-tensor argument into a python constant. An ``@nki.jit`` kernel is an
    ``nki.framework.kernel.Kernel``, a FROZEN DATACLASS, and Dynamo refuses to reconstruct one --
    ``NotImplementedError: currently can't reconstruct arbitrary frozen dataclass instances``,
    raised in the installed ``torch/_dynamo/variables/user_defined.py`` at line 2096 and measured
    on this image by `-044`'s capture probe, which saw it wrapped in
    ``torch._dynamo.exc.InternalTorchDynamoError``. So neither kernel is a parameter here: the
    direction arrives as a ``str`` and the kernel is read as a module global, the same object the
    call site hands to ``wrap_nki`` on the line after this call. The fork's own fold obeys the same
    discipline in its signature, ``vllm_neuron/functional/topk.py:90-92`` declaring ``kernel: str``.

    ONE HELPER FOR BOTH DIRECTIONS, because the template rule this campaign runs under asks for one
    folded helper per module rather than one per call site: a second fold is a second thing to keep
    in step, and the direction is exactly the kind of ``str`` a fold accepts.

    D13.1 STILL HOLDS. Because this body runs only when a dispatch branch runs, the recorded
    identity is derived by TAKING the branch rather than read off an import.
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

    Host python over at most a handful of ints: orchestration, named as such by the substrate
    declaration, and the reason ``lengths`` is not a tensor is in the module docstring.
    """
    offsets = []
    acc = 0
    for length in lengths:
        offsets.append(acc)
        acc += length
    return tuple(offsets)


def _column(values: Sequence[int], device: torch.device) -> Tensor:
    """A ``[n, 1]`` int32 column built from python integers, in a form that TRACES.

    ``torch.tensor(<python list>)`` CANNOT be used here. Inside a traced region it materialises a
    real tensor under ``FakeTensorMode``, and the next operation on it asserts with "Please convert
    all Tensors to FakeTensors first". That is categorical rather than shape dependent: it fails on
    ``.sum()`` exactly as it fails on ``.reshape()``. ``torch.full`` and ``torch.cat`` are aten
    operations, so they produce fake tensors and trace cleanly.

    Measured as arm F2 of ``probe-045-metadata-fix-host.out``. Arms F4 and F5 of the same round are
    what make it a proof about the live path rather than about a helper: with this swap in place the
    real pack and unpack seams both ran through the shared constant fold to their kernel calls and
    both kernel identities resolved statically.

    ``values`` is never empty on any admitted geometry -- ``can_run_dsa_ragged_pack`` requires
    ``sum(lengths) > 0`` -- so no guard is added here for a case the module does not admit.
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
    """True when the NKI route is available AND this module admits this pack geometry.

    The conditions are the runtime gate, the rank of the padded batch, the admitted dtype, a batch
    size matching the lengths, and at least one row to move. There is no factory dry-run to defer
    to, because this kernel is authored in this file and has no config factories -- so the envelope
    is stated here and the module docstring says why each part of it exists.
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
    """True when the NKI route is available AND this module admits this unpack geometry."""
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
    """THE COUNTED SEAM, pack direction. Move valid rows into a dense buffer.

    Args:
        padded: ``[batch, max_len, width]`` -- the padded batch. ``bfloat16`` takes the NKI route;
            any other dtype is served by the torch path.
        lengths: valid rows per sequence, as python ints. The module docstring gives the reason
            this is not a tensor.

    Returns:
        ``[sum(lengths), width]`` in ``padded``'s dtype: the sequences' valid rows, in order, with
        no padding row among them.

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

    _COUNTERS.nki_dispatch += 1
    # The log and the identity read are FOLDED off the traced graph. The helper takes a str and
    # ints ONLY and reads the kernel as a module global: passing the kernel object is the measured
    # defect that pattern exists to avoid. NO bare logging or other host call Dynamo refuses
    # belongs on this branch. The counter increment stays: it is a plain int attribute store.
    _record_nki_dispatch("pack", batch, max_len, width, packed_len)
    return wrap_nki(_ragged_pack_nki)(flat, pos, len_t, off_t, packed_len)


def dsa_ragged_unpack(packed: Tensor, lengths: Sequence[int], max_len: int) -> Tensor:
    """THE COUNTED SEAM, unpack direction. Put packed rows back and zero the padding.

    Args:
        packed: ``[sum(lengths), width]`` -- the dense buffer.
        lengths: valid rows per sequence, as python ints.
        max_len: padded rows per sequence in the result.

    Returns:
        ``[batch, max_len, width]`` in ``packed``'s dtype, every padding position exactly ``+0.0``.

    Raises:
        DsaRaggedPackError: for a malformed call -- a non-2D ``packed``, a non-positive ``max_len``,
            a length outside ``[0, max_len]``, or a row count that the lengths do not sum to.
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

    _COUNTERS.nki_dispatch += 1
    _record_nki_dispatch("unpack", batch, max_len, width, packed_len)
    flat = wrap_nki(_ragged_unpack_nki)(packed.contiguous(), pos, len_t, off_t, max_len)
    return flat.reshape(batch, max_len, width)


def _dsa_ragged_pack_torch(padded: Tensor, lengths: Sequence[int]) -> Tensor:
    """The CPU oracle and the constraint-violation fallback -- never the shipped path (D6).

    Reached on CPU without the simulator, with NKI kernels disabled, or on a dtype the gate does
    not admit. It increments its own counter so a test can state which route ran instead of
    assuming it.
    """
    _COUNTERS.torch_fallback += 1
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
    """The CPU oracle and the constraint-violation fallback for the unpack direction (D6).

    Zeros first and then writes the valid rows, so a padding position is a true ``+0.0`` here too
    and the oracle agrees with the kernel bit for bit rather than only numerically.
    """
    _COUNTERS.torch_fallback += 1
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
