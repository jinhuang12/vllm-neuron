# SPDX-License-Identifier: Apache-2.0
"""The runner's side of the bank-form carriers: when a step takes it, and its slot tensors.

At ``B > 1`` the carrier builder hands every recurrent (KDA) layer its two whole state
banks and every sparse (DSA) layer its whole pooled store and ring bank, each beside one
``[B]`` int64 slot tensor, instead of ``B`` per-request views of each bank. The served
bs=64 graph had 7298 inputs, 5760 of them such views (34 KDA layers x 64 x 2 states +
11 DSA layers x 64 x 2 side caches); in the bank form a layer's state costs two inputs
whatever ``B`` is, and the NRT submit, whose cost is per input, shrinks with it. The
layers gather the rows they step and write them back onto the bank itself
(:mod:`vllm_neuron.functional.state_banks`), which the backend keeps as whole-bank
aliased outputs.

Which steps take the form (:func:`bank_form`):

* a decode of two or more requests, only. One request keeps the per-request view form:
  the bs=1 graph is the measured production line and a gather/scatter per layer would
  add device work to a device-bound step. A prefill serves one sequence and its state
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
"""

from __future__ import annotations

import math
import os

import torch

__all__ = [
    "SCRATCH_SLOTS",
    "STATE_BANKS_ENV",
    "bank_form",
    "scratch_slot",
    "slot_tensor",
    "sparse_bank_slots",
    "state_bank_regions",
    "state_banks_enabled",
    "strided_bank_problem",
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


def state_bank_regions(raw, shapes, dtypes, *, slot_bytes: int, dtype_view=None) -> list[torch.Tensor]:
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
    """
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
        bank_bytes = slots * math.prod(shape) * itemsize
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
        banks.append(view[start : start + bank_bytes // itemsize].view(slots, *shape))
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
