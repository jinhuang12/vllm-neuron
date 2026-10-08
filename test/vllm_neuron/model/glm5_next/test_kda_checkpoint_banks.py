# SPDX-License-Identifier: Apache-2.0
"""A recurrent layer's banks carved with ``checkpoints`` rows per slot.

A speculative server keeps ``1 + k`` state rows per request slot
(``MambaSpec(num_speculative_blocks=k)``), so ``state_bank_regions`` carves each
state bank as ``[slots, 1 + k, *shape]``: a slot stays one contiguous
``bank[slot]`` view (the view form's carrier), the whole bank stays contiguous
(the bank form's input), and the flat row ``slot * (1 + k) + j`` is the row
``fused_decode.kda_checkpoint_rows`` names for the bank form's gather. With one
checkpoint the carving is byte for byte the plain server's.
"""

from __future__ import annotations

import math

import pytest
import torch

from vllm_neuron.functional.kda import fused_decode as fused
from vllm_neuron.vllm.patches.kv_spec_patch import recurrent_state_slot_bytes
from vllm_neuron.vllm.worker.glm5next_state_banks import state_bank_regions

#: The TP=64 KDA slot: a bf16 ``[3, 384]`` conv row and a fp32 ``[1, 128, 128]`` recurrent row.
SHAPES = ((3, 384), (1, 128, 128))
DTYPES = (torch.bfloat16, torch.float32)
ALIGN = 256
SLOTS = 6


def _slot_bytes(checkpoints: int) -> int:
    state = sum(math.prod(s) * d.itemsize for s, d in zip(SHAPES, DTYPES, strict=True))
    return -(-(state * checkpoints) // ALIGN) * ALIGN


def _carve(checkpoints: int | None):
    slot_bytes = _slot_bytes(checkpoints or 1)
    raw = torch.zeros(SLOTS * slot_bytes, dtype=torch.uint8)
    kwargs = {} if checkpoints is None else {"checkpoints": checkpoints}
    return raw, state_bank_regions(raw, SHAPES, DTYPES, slot_bytes=slot_bytes, **kwargs)


@pytest.mark.parametrize("checkpoints", [2, 4, 6])
def test_checkpoint_banks_keep_whole_slots_contiguous(checkpoints):
    raw, banks = _carve(checkpoints)
    offset = 0
    for bank, shape, dtype in zip(banks, SHAPES, DTYPES, strict=True):
        assert tuple(bank.shape) == (SLOTS, checkpoints, *shape)
        assert bank.dtype == dtype
        assert bank.is_contiguous(), "the bank form hands the whole bank to the graph"
        assert bank.data_ptr() == raw.data_ptr() + offset, "banks follow one another"
        for slot in range(SLOTS):
            assert bank[slot].is_contiguous(), "a slot is one carrier view"
            assert bank[slot].data_ptr() == bank.data_ptr() + slot * bank[slot].nbytes
        offset += bank.nbytes
    assert offset <= raw.numel()
    assert offset > (SLOTS - 1) * _slot_bytes(checkpoints), (
        "the banks fill the buffer up to the alignment slack")


def test_one_checkpoint_is_the_plain_carving():
    raw_plain, plain = _carve(None)
    raw_one, one = _carve(1)
    for a, b in zip(plain, one, strict=True):
        assert a.shape == b.shape and a.stride() == b.stride() and a.dtype == b.dtype
        assert a.data_ptr() - raw_plain.data_ptr() == b.data_ptr() - raw_one.data_ptr()
    with pytest.raises(ValueError, match="checkpoints"):
        _carve(0)


def test_flat_checkpoint_rows_address_the_carved_banks():
    """The bank form reads row ``slot * T + accepted`` of the flattened bank, and the
    kernel's ``[B, T, ...]`` checkpoints are written back as whole slots."""
    checkpoints = 4
    _, banks = _carve(checkpoints)
    conv_bank, rec_bank = banks
    gen = torch.Generator().manual_seed(8500)
    rec_bank.copy_(torch.randn(rec_bank.shape, generator=gen))
    conv_bank.copy_(torch.randn(conv_bank.shape, generator=gen).to(conv_bank.dtype))
    slots = torch.tensor([4, 0, 2], dtype=torch.int64)
    kept = torch.tensor([4, 1, 2], dtype=torch.int32)  # tokens kept per request
    # The commit is a pointer: it validates and returns the rows the next step reads.
    committed = fused.commit_kda_checkpoints(
        (conv_bank, rec_bank), slots, kept, state_checkpoints=checkpoints
    )
    assert torch.equal(committed, kept - 1)
    rows = fused.kda_checkpoint_rows(slots, committed, checkpoints)
    for bank in (conv_bank, rec_bank):
        gathered = bank.flatten(0, 1).index_select(0, rows)
        expected = torch.stack([bank[s, a] for s, a in zip(slots.tolist(), committed.tolist())])
        assert torch.equal(gathered, expected)
    # A whole-slot write-back of ``[B, T, ...]`` checkpoints lands on the slots named.
    before = rec_bank.clone()
    new = torch.randn((3, checkpoints, *SHAPES[1]), generator=gen)
    rec_bank.index_copy_(0, slots, new)
    assert torch.equal(rec_bank[slots], new)
    untouched = [s for s in range(SLOTS) if s not in slots.tolist()]
    assert torch.equal(rec_bank[untouched], before[untouched])


@pytest.mark.parametrize("drafts", [0, 1, 3, 5])
def test_the_slot_bytes_hold_every_checkpoint_row(drafts):
    """``recurrent_state_slot_bytes`` prices ``1 + num_speculative_blocks`` rows, so the
    runner's ``slots x slot_bytes`` buffer carves exactly ``1 + k`` rows per slot; at
    ``k = 0`` (no speculative config) it is the plain server's slot."""
    from vllm.v1.kv_cache_interface import MambaSpec

    def spec(k):
        return MambaSpec(block_size=128, shapes=SHAPES, dtypes=DTYPES, num_speculative_blocks=k)

    plain = recurrent_state_slot_bytes(spec(0))
    assert plain == _slot_bytes(1)
    slot_bytes = recurrent_state_slot_bytes(spec(drafts))
    assert slot_bytes == _slot_bytes(1 + drafts)
    assert slot_bytes == (1 + drafts) * plain
    raw = torch.zeros(SLOTS * slot_bytes, dtype=torch.uint8)
    banks = state_bank_regions(raw, SHAPES, DTYPES, slot_bytes=slot_bytes,
                               checkpoints=1 + drafts)
    for bank, shape in zip(banks, SHAPES, strict=True):
        assert tuple(bank.shape) == ((SLOTS, 1 + drafts, *shape) if drafts else (SLOTS, *shape))
