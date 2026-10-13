# SPDX-License-Identifier: Apache-2.0
"""The runner's side of the bank-form carriers: when a step takes it, and its slot tensors.

At ``B > 1`` the carrier builder hands every recurrent (KDA) layer its two whole state
banks and every sparse (DSA) layer its whole pooled store and ring bank, each beside one
``[B]`` int64 slot tensor, instead of ``B`` per-request views of each bank. At ``B = 64``
those views are most of the decode graph's inputs (34 KDA layers x 64 x 2 states +
11 DSA layers x 64 x 2 side caches); in the bank form a layer's state costs two inputs
whatever ``B`` is, and the NRT submit, whose cost is per input, shrinks with it. The
layers gather the rows they step and write them back onto the bank itself
(:mod:`vllm_neuron.functional.state_banks`), which the backend keeps as whole-bank
aliased outputs.

Which steps take the form (:func:`bank_form`):

* a decode of two or more requests, only. One request keeps the per-request view form:
  the one-request graph is a device-bound step, and a gather/scatter per layer would
  add device work to it. A prefill serves one sequence and its state
  writes are the layer's own;
* the kill switch ``VLLM_NEURON_GLM5NEXT_STATE_BANKS=0`` keeps the view form at every
  ``B`` (today's graphs, unchanged);
* the recurrent family's bank form rides on the fused KDA decode launch, so
  ``VLLM_NEURON_KDA_FUSED_DECODE=0`` keeps the view form too: the per-request fallback
  writes each request's state through a view, which the bank form cannot express.

Padding rows. A decode step padded to its batch bucket carries rows no request holds.
On the recurrent family they keep the idle slots the runner already chose (the masked
step is the identity, so the row is handed back unchanged). On the sparse family every
padding row names the **scratch slot**: the last store of the side caches, one past the
engine's concurrency bound (``SCRATCH_SLOTS`` extra slots are allocated for it). Its ring
write lands where no request reads, so a request that holds a slot but is not scheduled
this step keeps its ring. The two families therefore get two slot tensors, built as two
uploads even when equal, so the graph's signature does not depend on whether the step
is padded.

Emptying a slot (:func:`empty_slot`). A side-cache slot outlives its owner, so the
runner empties it when it hands the slot to a new request and when a request opens at
position 0. The write is one slot, in place, as a ``copy_`` from a host tensor of zeros:
on the Neuron device ``copy_`` is the backend's native ``_copy_from`` (an NRT write into
the slot's contiguous slice of the bank), while ``zero_``, ``fill_``, ``index_fill_`` and
``index_put_`` take the backend's CPU fallback, whose write-back refuses a tensor with
shared storage ("Can't call ReserveSpace on shared storage"). It runs on the host
before the step's carriers are built, outside any captured graph, on the same tensor the
carriers hand the layers (a view of it or, in the bank form, the bank itself).

The backend's host-to-device ``_copy_from`` does not wait for an execution still writing
the bank; its device-to-host copy does (``NeuronTensorImpl::Await`` on the storage's
pending execution). A slot freed by a request whose last, asynchronously scheduled step
is still running is written by that step, so the write is preceded by a one-element read
of the slot, which orders it after that step as the whole-bank read of a rebuild did.

The KDA checkpoint store. With prefix caching on, vLLM can hand a new request a cached
prefix of whole blocks, and block ``b`` of a request's KDA group stands for the state
that group's layers reach at the block's last token. The banks above hold one latest
state per request slot, so the checkpoint of a block is kept in the block's own page
instead: vLLM gives every KV cache group its blocks from one pool, and each pool tensor
is shared by one layer of every group, so page ``b`` of the tensor a KDA layer shares
belongs to that layer while its group holds block ``b``. The runner allocates each pool
tensor whole (for the one latent layer on it), so these pages exist, and while the KDA
group holds block ``b`` no layer of another group writes page ``b``. A checkpoint is one committed state row of every carrier, at fixed byte
offsets of the page (:func:`kda_checkpoint_layout`); a speculative step's draft rows are
never stored.

The page view and the write form (:func:`kda_checkpoint_pages`,
:func:`write_kda_checkpoints`). On the device a write is one NKI kernel launch
(:mod:`vllm_neuron.functional.kda.checkpoint_write`): indirect DMAs move each carrier's
bytes into their pages, and the kernel returns the pages, so the backend makes the whole
tensor an aliased output. The operand is the page view or any other contiguous view of
the same pool tensor's whole pages, such as the latent layer's cache view itself, so a
graph that writes both has one written input per storage. The backend's aliasing pass
hands the kernel's result to every later operation on the same graph input, as it moves
the later uses of an in-place op's own first argument onto that op's result, so the
writes to one input run in order, each on the one before. The KDA layers that share one
pool tensor are handed one page view, so their writes, to their own groups' blocks,
chain on that one input whether or not a layer uses the pages a write returns. Another
view of the same buffer is another graph input, outside that chain (the latent cache
write in ``model/glm5_next/model_fp8.py``, "The persisting write"). On CPU tensors the
torch path writes the same bytes, eager.
Block id 0 is vLLM's null block: the pool never hands it out as a real block and it is
never the last entry of a hit, so no restore reads it and a padding row may name it.
Reads (:func:`read_kda_checkpoints`, :func:`restore_kda_slot`) gather whole pages and
cut the carriers out of them with dtype views. They are eager only on this backend: the
backend refuses an in-graph dtype view when it lowers a graph to HLO, so a traced caller
is refused there.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from vllm_neuron.functional.kda.checkpoint_write import write_page_rows

__all__ = [
    "SCRATCH_SLOTS",
    "STATE_BANKS_ENV",
    "bank_form",
    "empty_slot",
    "KdaCheckpointLayout",
    "kda_checkpoint_layout",
    "kda_checkpoint_pages",
    "read_kda_checkpoints",
    "restore_kda_slot",
    "scratch_slot",
    "slot_tensor",
    "sparse_bank_slots",
    "state_bank_regions",
    "state_banks_enabled",
    "strided_bank_problem",
    "write_kda_checkpoints",
]

#: Set to ``0`` to keep the per-request view carriers at every batch size.
STATE_BANKS_ENV = "VLLM_NEURON_GLM5NEXT_STATE_BANKS"

#: Extra side-cache slots past the engine's concurrency bound; the last is the scratch
#: store every padding row of a sparse layer names.
SCRATCH_SLOTS = 1


def state_banks_enabled() -> bool:
    """Whether the bank form is allowed at all (the kill switch is off)."""
    return os.environ.get(STATE_BANKS_ENV, "1") != "0"


def bank_form(requests: int, *, is_prefill: bool) -> bool:
    """Whether a step of ``requests`` rows hands the layers whole banks and slot tensors."""
    if bool(is_prefill) or int(requests) < 2:
        return False
    if not state_banks_enabled():
        return False
    from vllm_neuron.functional.kda.fused_decode import fused_decode_enabled

    return fused_decode_enabled()


def state_bank_regions(
    raw, shapes, dtypes, *, slot_bytes: int, dtype_view=None, checkpoints: int = 1
) -> list[torch.Tensor]:
    """One contiguous ``[slots, *shape]`` bank per state over ``raw``, a recurrent layer's buffer.

    ``raw`` is the layer's one raw byte buffer: ``slots`` request slots of ``slot_bytes``
    (:func:`~vllm_neuron.vllm.patches.kv_spec_patch.recurrent_state_slot_bytes`) each. The
    states are laid out as whole banks one after another, ``[conv bank | recurrent bank |
    slack]``, not slot by slot: the Neuron executor accepts a graph input only as a
    contiguous slice of its storage ("Detected non-contiguous slicing for requested Device
    Tensor"), and the bank form hands the WHOLE bank. A slot-strided view of the buffer
    (both states side by side inside every slot, the allocation before this) has
    contiguous rows but is not itself contiguous, so the first bank-form decode failed on
    device while the per-request view form ran. Here every bank is contiguous and so is
    every row ``bank[slot]`` (the view form's carrier, and what a row ``copy_`` writes).

    The buffer is not resized: ``slot_bytes`` is the states' byte sum rounded up, so the
    banks always fit, and the rounding's slack sits after the last bank. Each bank's byte
    offset must be a multiple of its own element size, which it is for every state whose
    predecessors' rows are (the KDA conv row is a multiple of 4 bytes); the refusal is by
    name otherwise. ``dtype_view(raw, dtype)`` reinterprets ``raw`` as ``dtype`` (the runner
    caches one per buffer and dtype, so the banks share one ``._base``); by default
    ``raw.view(dtype)``.

    ``checkpoints`` is the state rows one slot holds, ``1 + num_speculative_blocks`` of
    the layer's ``MambaSpec`` (one, the plain decode's, by default). Above one every bank
    is ``[slots, checkpoints, *shape]``: row ``j`` of a slot is the state after ``j + 1``
    tokens of the request's last speculative step, and row 0 also where a prefill or a
    one-token step writes (``functional.kda.fused_decode``, "Checkpoint banks"). A slot
    stays one contiguous ``bank[slot]`` view, so both carrier forms keep their shape of
    access; ``slot_bytes`` is the whole slot, every checkpoint included
    (``recurrent_state_slot_bytes``).
    """
    checkpoints = int(checkpoints)
    if checkpoints < 1:
        raise ValueError(
            f"a slot holds at least one state row; got checkpoints={checkpoints!r}"
        )
    if raw.dim() != 1 or raw.element_size() != 1:
        raise ValueError(
            f"a recurrent layer's raw buffer is a flat byte tensor; got shape "
            f"{tuple(raw.shape)} of {raw.dtype}"
        )
    slot_bytes = int(slot_bytes)
    total_bytes = int(raw.numel())
    if slot_bytes <= 0 or total_bytes % slot_bytes:
        raise ValueError(
            f"a recurrent layer's raw buffer holds whole slots of {slot_bytes} byte(s); "
            f"got {total_bytes} byte(s)"
        )
    slots = total_bytes // slot_bytes
    banks: list[torch.Tensor] = []
    offset_bytes = 0
    for index, (shape, dtype) in enumerate(zip(shapes, dtypes, strict=True)):
        shape = tuple(int(extent) for extent in shape)
        itemsize = int(dtype.itemsize)
        bank_bytes = slots * checkpoints * math.prod(shape) * itemsize
        if offset_bytes % itemsize:
            raise ValueError(
                f"state bank {index} ({shape}, {dtype}) would start at byte "
                f"{offset_bytes}, which is not a multiple of its {itemsize}-byte element"
            )
        if offset_bytes + bank_bytes > total_bytes:
            raise ValueError(
                f"state bank {index} ({shape}, {dtype}) needs bytes "
                f"[{offset_bytes}, {offset_bytes + bank_bytes}) of a {total_bytes}-byte "
                f"buffer of {slots} slot(s) x {slot_bytes} byte(s); the slot does not hold "
                f"the states it was sized for"
            )
        view = dtype_view(raw, dtype) if dtype_view is not None else raw.view(dtype)
        start = offset_bytes // itemsize
        rows = (slots, checkpoints) if checkpoints > 1 else (slots,)
        banks.append(view[start : start + bank_bytes // itemsize].view(*rows, *shape))
        offset_bytes += bank_bytes
    return banks


def strided_bank_problem(bank: torch.Tensor, *, name: str) -> str | None:
    """Return why ``bank`` cannot be handed to the graph whole, or ``None``.

    The Neuron executor takes a device tensor input only as a contiguous slice of its
    storage; a strided view is refused at execution ("Detected non-contiguous slicing for
    requested Device Tensor"), after the compile. Read where the carrier is built, so the
    refusal names the layer and the cure instead.
    """
    if bank.is_contiguous():
        return None
    return (
        f"{name} is a strided view (shape {tuple(bank.shape)}, stride {tuple(bank.stride())}) "
        f"and the bank form hands the whole bank to the decode graph, whose executor refuses "
        f"a non-contiguous device tensor input; allocate the bank contiguous "
        f"(glm5next_state_banks.state_bank_regions) or keep the per-request view form "
        f"({STATE_BANKS_ENV}=0)"
    )


def slot_tensor(slots, device) -> torch.Tensor:
    """``[B]`` int64 slots, built on the host and moved once.

    int64 because ``index_copy_`` takes a LongTensor index, so no cast is needed in the
    traced region; the sparse kernels cast to the int32 they read themselves.
    """
    return torch.tensor([int(slot) for slot in slots], dtype=torch.int64).to(device)


def scratch_slot(side: dict) -> int:
    """The scratch store of one sparse layer's side caches: the last slot."""
    return int(side["pool_cache"].shape[0]) - 1


def sparse_bank_slots(state_slots, *, real_requests: int, scratch: int) -> list[int]:
    """The sparse family's slots for a step: the real requests' own, then the scratch slot."""
    slots = [int(slot) for slot in state_slots]
    return slots[: int(real_requests)] + [int(scratch)] * (len(slots) - int(real_requests))


#: One host tensor of zeros per (slot shape, dtype), read by every :func:`empty_slot`.
_HOST_ZEROS: dict[tuple, torch.Tensor] = {}


def empty_slot(bank: torch.Tensor, slot: int) -> None:
    """Zero ``bank[slot]`` in place; every other slot of ``bank`` is left as it is.

    The fresh-slot contract is ``torch.zeros``' bytes (``+0.0``) in every element of the
    slot. Only that slot's bytes move (``bank[0].nbytes``), where a rebuild of the bank
    would read and write all of it.
    """
    row = bank[int(slot)]
    if not row.is_meta:
        # The ordering read (module docstring): 2 bytes, after any step writing this
        # bank. A capture on ``meta`` has no values and no step to wait for.
        row.reshape(-1)[:1].to("cpu", copy=True)
    key = (tuple(row.shape), row.dtype)
    zeros = _HOST_ZEROS.get(key)
    if zeros is None:
        zeros = _HOST_ZEROS[key] = torch.zeros(key[0], dtype=row.dtype)
    row.copy_(zeros)


# ── the KDA checkpoint store ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class KdaCheckpointLayout:
    """Where one block's checkpoint sits in a KV page of ``page_bytes`` bytes.

    Carrier ``c`` (``shapes[c]``, ``dtypes[c]``, in the ``MambaSpec``'s carrier order)
    takes ``nbytes[c]`` bytes from byte ``offsets[c]``; the rest of the page is unused.
    """

    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[torch.dtype, ...]
    offsets: tuple[int, ...]
    page_bytes: int

    @property
    def nbytes(self) -> tuple[int, ...]:
        return tuple(
            math.prod(shape) * dtype.itemsize
            for shape, dtype in zip(self.shapes, self.dtypes, strict=True)
        )

    @property
    def used_bytes(self) -> int:
        return self.offsets[-1] + self.nbytes[-1]


def kda_checkpoint_layout(spec) -> KdaCheckpointLayout:
    """The checkpoint layout of a recurrent layer of ``spec`` (its ``MambaSpec``).

    One committed state row of every carrier, laid out one after another from byte 0,
    each from a multiple of its element size, in the page vLLM's pool gives each block
    (``spec.page_size_bytes``, which ``kv_spec_patch`` pads up to the attention page).
    A speculative server's slots hold ``1 + num_speculative_blocks`` rows, one per
    verify token; a checkpoint is the state after a block's last committed token, so
    the layout is the same whatever ``num_speculative_blocks`` is. Refused by name
    when the checkpoint does not fit the page.
    """
    shapes = tuple(tuple(int(extent) for extent in shape) for shape in spec.shapes)
    dtypes = tuple(spec.dtypes)
    page_bytes = int(spec.page_size_bytes)
    offsets, offset = [], 0
    for shape, dtype in zip(shapes, dtypes, strict=True):
        itemsize = int(dtype.itemsize)
        offset = -(-offset // itemsize) * itemsize
        offsets.append(offset)
        offset += math.prod(shape) * itemsize
    layout = KdaCheckpointLayout(shapes, dtypes, tuple(offsets), page_bytes)
    if layout.used_bytes > page_bytes:
        raise ValueError(
            f"a KDA checkpoint ({', '.join(f'{s} {d}' for s, d in zip(shapes, dtypes))}) "
            f"takes {layout.used_bytes} bytes and a KV page holds {page_bytes}; the "
            f"checkpoint store keeps one checkpoint per page"
        )
    return layout


def kda_checkpoint_pages(raw, layout) -> torch.Tensor:
    """The ``[num_blocks, page_bytes]`` int8 view of a pool tensor: row ``b`` is page ``b``.

    ``raw`` is the flat byte buffer the runner allocated for one pool tensor (the one a
    KDA layer shares with its latent layer). No copy: the view is ``raw``'s storage,
    contiguous, so it can be a graph input and an aliased output.
    """
    page_bytes = int(layout.page_bytes)
    if raw.dim() != 1 or raw.element_size() != 1:
        raise ValueError(
            f"a pool tensor's raw buffer is a flat byte tensor; got shape "
            f"{tuple(raw.shape)} of {raw.dtype}"
        )
    if page_bytes <= 0 or int(raw.numel()) % page_bytes:
        raise ValueError(
            f"a pool tensor holds whole pages of {page_bytes} byte(s); got "
            f"{int(raw.numel())} byte(s)"
        )
    return raw.view(torch.int8).view(-1, page_bytes)


def _block_ids_problem(block_ids) -> str | None:
    if not torch.is_tensor(block_ids) or block_ids.dim() != 1:
        return (
            "block ids are a [n] tensor, one id per checkpoint; got "
            f"{block_ids if not torch.is_tensor(block_ids) else tuple(block_ids.shape)}"
        )
    if block_ids.dtype not in (torch.int64, torch.int32):
        return f"block ids must be int64 or int32; got {block_ids.dtype}"
    return None


def _pages_problem(pages, layout) -> str | None:
    if pages.dim() < 2 or int(pages[0].nbytes) != int(layout.page_bytes):
        return (
            f"the checkpoint pages are a [num_blocks, ...] tensor of {layout.page_bytes}-"
            f"byte rows (kda_checkpoint_pages); got {tuple(pages.shape)} {pages.dtype}"
        )
    return None


def _carrier_rows(rows: Sequence[torch.Tensor], layout, count: int) -> list[torch.Tensor]:
    """``rows`` checked against the layout, each as its carrier's dtype."""
    if len(rows) != len(layout.shapes):
        raise ValueError(
            f"a checkpoint has one row tensor per state carrier: the layout has "
            f"{len(layout.shapes)} carrier(s) and {len(rows)} row tensor(s) were given"
        )
    checked = []
    for carrier, (carrier_rows, shape, carrier_dtype) in enumerate(
        zip(rows, layout.shapes, layout.dtypes)
    ):
        if tuple(carrier_rows.shape) != (count, *shape):
            raise ValueError(
                f"carrier {carrier} of a KDA checkpoint is {shape} per block; for "
                f"{count} block id(s) the write takes {(count, *shape)}, got "
                f"{tuple(carrier_rows.shape)}"
            )
        checked.append(carrier_rows.to(carrier_dtype))
    return checked


def _unpack(pages: torch.Tensor, layout) -> list[torch.Tensor]:
    """The carriers of ``pages`` (``[n, ...]`` whole pages), one ``[n, *shape]`` each."""
    count = int(pages.shape[0])
    raw = pages.reshape(count, -1).view(torch.int8)
    return [
        raw[:, offset : offset + nbytes].contiguous().view(dtype).view(count, *shape)
        for shape, dtype, offset, nbytes in zip(
            layout.shapes, layout.dtypes, layout.offsets, layout.nbytes
        )
    ]


def write_kda_checkpoints(pages, block_ids, rows, layout) -> torch.Tensor:
    """Write one checkpoint per block: ``rows[c][i]`` into page ``block_ids[i]`` of ``pages``.

    ``pages`` is the page view of the pool tensor the layer shares
    (:func:`kda_checkpoint_pages`), or any other contiguous view of that tensor's whole
    pages, such as the latent layer's cache view; ``block_ids`` a ``[n]`` int64 or int32
    tensor of the layer's own group's blocks (int32 ids built on the host take no
    conversion in a graph); ``rows`` one ``[n, *shape]`` tensor per carrier, in the
    layout's carrier order. Each carrier's bytes go to its offset of the page, and the
    page's other bytes are left as they are. Rows naming one block more than once leave
    one of them; only padding rows, which name the null block 0, may repeat. The write
    finds page ``b`` at byte ``b * layout.page_bytes`` of ``pages``, whatever the tensor's
    row size: a ``[2 * n, page_bytes // 2]`` int8 tensor is taken as ``n`` pages. The
    read and the restore take ``[num_blocks, ...]`` tensors of whole-page rows only.

    Returns the written pages. On the device the write is one NKI kernel launch inside a
    compiled graph (there is no eager device form), and the backend hands this tensor to
    every later operation on the same graph input (module docstring). On CPU tensors,
    eager, the write is in place and returns ``pages``. The checks read shapes and dtypes
    only.
    """
    problem = _block_ids_problem(block_ids)
    if problem is not None:
        raise ValueError(problem)
    count = int(block_ids.shape[0])
    carriers = _carrier_rows(rows, layout, count)
    return write_page_rows(pages, block_ids, carriers, layout.offsets, layout.page_bytes)


def read_kda_checkpoints(pages, block_ids, layout) -> list[torch.Tensor]:
    """The checkpoints of ``block_ids``: one ``[n, *shape]`` tensor per carrier, in order.

    One gather of whole pages, then each carrier cut out of them with dtype views. Eager
    only on this backend: a traced caller is refused when the backend lowers its graph to
    HLO, which has no lowering for an in-graph dtype view.
    """
    problem = _block_ids_problem(block_ids) or _pages_problem(pages, layout)
    if problem is not None:
        raise ValueError(problem)
    return _unpack(pages.index_select(0, block_ids), layout)


def restore_kda_slot(banks, slot, pages, block_id, layout) -> None:
    """Copy block ``block_id``'s checkpoint into request slot ``slot`` of every bank.

    ``banks`` is the layer's slot banks (:func:`state_bank_regions`), carrier for
    carrier with the layout. The checkpoint goes to the slot's entering row:
    ``bank[slot]``, or ``bank[slot, 0]`` for a speculative server's checkpoint banks
    (row 0 is the row a slot with no resume row reads). Every other slot, and rows 1
    and up of the slot, are left as they are. Eager, on the host: the page is read to
    the host, which waits for any execution still writing it, and each bank row gets
    the ordering read of :func:`empty_slot` before its ``copy_``.
    """
    problem = _pages_problem(pages, layout)
    if problem is not None:
        raise ValueError(problem)
    if len(banks) != len(layout.shapes):
        raise ValueError(
            f"a restore copies one checkpoint row per state carrier: {len(banks)} bank(s) "
            f"and {len(layout.shapes)} carrier(s) in the layout"
        )
    targets = []
    for carrier, (bank, shape, dtype) in enumerate(zip(banks, layout.shapes, layout.dtypes)):
        target = bank[int(slot), 0] if bank.dim() == len(shape) + 2 else bank[int(slot)]
        if tuple(target.shape) != shape or bank.dtype != dtype:
            raise ValueError(
                f"carrier {carrier}: a checkpoint row is {shape} {dtype} and the bank "
                f"{tuple(bank.shape)} {bank.dtype} has no slot row of that form"
            )
        targets.append(target)
    page = pages[int(block_id)].unsqueeze(0).to("cpu", copy=True)
    for target, checkpoint in zip(targets, _unpack(page, layout)):
        if not target.is_meta:
            target.reshape(-1)[:1].to("cpu", copy=True)
        target.copy_(checkpoint[0])
