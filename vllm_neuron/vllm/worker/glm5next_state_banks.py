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

import os

import torch

__all__ = [
    "SCRATCH_SLOTS",
    "STATE_BANKS_ENV",
    "bank_form",
    "scratch_slot",
    "slot_tensor",
    "sparse_bank_slots",
    "state_banks_enabled",
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
