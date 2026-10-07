# SPDX-License-Identifier: Apache-2.0
"""A new request's DSA side-cache slot is emptied in place, one slot, nothing else.

Every sparse (DSA) layer keeps two runner-allocated side caches with one store per
request slot: ``pool_cache`` ``[slots, max_seq_len // index_kpool + 1, index_head_dim]``
and the ring ``tail`` ``[slots, 2, index_kpool, index_head_dim]``. A slot outlives its
owner, so it is emptied when it is handed to a new request (both caches) and when a
request opens at position 0 (the ring). The fresh-slot contract is the one 0a08ff4
gives: every element of the slot's rows is ``+0.0`` (the bytes ``torch.zeros`` writes),
and no other slot's bytes, the scratch slot's or ``pad_tail``'s change.

Until this change the reset rebuilt every layer's whole cache with ``torch.where``
(``65 x 2049 x 128`` bf16 per layer at the bs=64 line, 11 layers, three rebuilds per new
request). Read here, on a synthetic stack with the runner's own side-cache allocator at
B = 64 (64 request slots plus the scratch slot):

1. the hand-out zeroes the new request's slot in both caches, leaves every other slot
   bitwise unchanged, and keeps the same tensors (the storage the bank-form graph takes
   as an aliased input) rather than rebuilding them;
2. a finished request's slot handed to the next request is emptied again, every time;
3. a request that re-opens at position 0 in the slot it holds (a preempted request
   resumed from computed length 0) gets its ring emptied in place; its pooled store is
   left as 0a08ff4 leaves it (rows above the new bound are unreachable);
4. the device rule: the only write the reset makes on a side cache is ``copy_`` of one
   slot from a host tensor. ``zero_``/``fill_``/``index_put_`` and friends reach the
   Neuron backend's CPU fallback, whose write-back (``_copy_from_and_resize``) refuses
   a tensor with shared storage ("Can't call ReserveSpace on shared storage"); ``copy_``
   is the backend's native ``_copy_from`` (an NRT write into the slice).

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/worker/test_glm5next_slot_reset.py
"""

from __future__ import annotations

import hashlib

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

pytestmark = [pytest.mark.fast, pytest.mark.forked]

BATCH = 64
LINEAR, SPARSE = 4, 11
POOL = 4
WIDTH = 8
MAX_SEQ_LEN = 64
DTYPE = torch.bfloat16
SEED = 20261007


def _digest(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8) if tensor.dtype != torch.bool else tensor
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


def _banks(slots: int = BATCH) -> list[dict]:
    banks = []
    for index in range(LINEAR):
        banks.append({
            "name": f"linear.{index}", "family": "linear_attn", "state_slots": slots,
        })
    for index in range(SPARSE):
        banks.append({
            "name": f"sparse.{index}", "family": "self_attn", "block_size": POOL,
            "latent_cache": torch.zeros((4, 1, POOL, WIDTH), dtype=DTYPE),
        })
    return banks


def _runner(slots: int = BATCH):
    """A runner shell holding the live side caches the runner itself allocates."""
    from vllm_neuron.vllm.worker.glm5next_state_banks import SCRATCH_SLOTS

    banks = _banks(slots)
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.max_num_reqs = slots
    runner.max_model_len = MAX_SEQ_LEN
    side = NeuronModelRunner._glm5next_side_caches(
        banks, index_kpool=POOL, index_head_dim=WIDTH, max_seq_len=MAX_SEQ_LEN,
        request_slots=slots + SCRATCH_SLOTS,
    )
    runner._glm5next_side_cache_set = side
    runner._glm5next_request_slot_table = {}
    runner._glm5next_side_cache_positions = {}
    # A previous owner's leftovers in every slot: distinct, non-zero values, so a reset
    # is visible and a slot that should not move can be told apart from its neighbours.
    gen = torch.Generator().manual_seed(SEED)
    for entry in side:
        for key in ("pool_cache", "tail", "pad_tail"):
            if key in entry:
                values = torch.rand(entry[key].shape, generator=gen) + 0.5
                entry[key].copy_(values.to(DTYPE))
    return runner, banks, side


def _occupy(runner, owners: dict[str, int], *, cursor: int = 9) -> None:
    runner._glm5next_request_slot_table = dict(owners)
    runner._glm5next_side_cache_positions = {slot: cursor for slot in owners.values()}


def _scattered_owners(free: int, slots: int = BATCH) -> dict[str, int]:
    """Every slot but ``free`` owned, in a shuffled, non-contiguous request order."""
    order = torch.randperm(slots, generator=torch.Generator().manual_seed(SEED + 1)).tolist()
    return {f"req-{slot}": slot for slot in order if slot != free}


def _slot_digests(side) -> dict:
    out = {}
    for index, entry in enumerate(side):
        for key in ("pool_cache", "tail", "pad_tail"):
            if key not in entry:
                continue
            for slot in range(int(entry[key].shape[0])):
                out[(index, key, slot)] = _digest(entry[key][slot])
    return out


def _identities(side) -> dict:
    return {
        (index, key): (id(entry[key]), entry[key].untyped_storage().data_ptr())
        for index, entry in enumerate(side) for key in entry
    }


def _assert_fresh(side, slot: int, keys=("pool_cache", "tail")) -> None:
    for index, entry in enumerate(side):
        for key in keys:
            if key not in entry:
                continue
            row = entry[key][slot]
            fresh = torch.zeros_like(row)
            # Bitwise, so a -0.0 or a stale NaN cannot pass as a clear.
            assert _digest(row) == _digest(fresh), (
                f"layer {index} {key}: slot {slot} is not the fresh-slot contract (+0.0)"
            )


def _assert_only_moved(before: dict, after: dict, moved: set) -> None:
    changed = {key for key in before if before[key] != after[key]}
    assert changed <= moved, f"slots outside {sorted(moved)} changed: {sorted(changed - moved)[:6]}"


# ── 1. hand-out at B = 64 over scattered owners ──────────────────────────────


def test_handing_a_slot_to_a_new_request_empties_that_slot_in_place_and_nothing_else():
    runner, banks, side = _runner()
    free = 37
    owners = _scattered_owners(free)
    _occupy(runner, owners)
    step_ids = list(owners)[:40] + ["new-request"] + list(owners)[40:]
    before, identity = _slot_digests(side), _identities(side)

    slots = runner._glm5next_request_slots(banks, step_ids, synthetic=False, side_caches=side)

    assert slots[40] == free
    assert [slots[i] for i in range(len(step_ids)) if i != 40] == [
        owners[rid] for rid in step_ids if rid != "new-request"
    ]
    assert _identities(side) == identity, (
        "the hand-out replaced a side-cache tensor; the reset must write the slot in "
        "place on the tensor the bank-form graph aliases"
    )
    _assert_fresh(side, free)
    moved = {(i, k, free) for i, entry in enumerate(side) for k in ("pool_cache", "tail") if k in entry}
    assert len(moved) == 2 * SPARSE
    after = _slot_digests(side)
    _assert_only_moved(before, after, moved)
    # The scratch slot and the padding rings are never a request's, so never emptied.
    assert all(after[(i, k, BATCH)] == before[(i, k, BATCH)]
               for i, entry in enumerate(side) for k in ("pool_cache", "tail") if k in entry)
    assert runner._glm5next_request_slot_table["new-request"] == free
    assert free not in runner._glm5next_side_cache_positions


def test_two_new_requests_in_one_step_empty_exactly_their_two_slots():
    runner, banks, side = _runner()
    owners = {rid: slot for rid, slot in _scattered_owners(3).items() if slot != 50}
    _occupy(runner, owners)
    before = _slot_digests(side)
    slots = runner._glm5next_request_slots(
        banks, ["a", *list(owners)[:5], "b"], synthetic=False, side_caches=side
    )
    assert (slots[0], slots[-1]) == (3, 50)
    _assert_fresh(side, 3)
    _assert_fresh(side, 50)
    moved = {(i, k, s) for s in (3, 50) for i, entry in enumerate(side)
             for k in ("pool_cache", "tail") if k in entry}
    _assert_only_moved(before, _slot_digests(side), moved)


def test_a_continuing_request_is_not_emptied():
    runner, banks, side = _runner()
    owners = _scattered_owners(12)
    _occupy(runner, owners)
    before, identity = _slot_digests(side), _identities(side)
    runner._glm5next_request_slots(banks, list(owners), synthetic=False, side_caches=side)
    assert _slot_digests(side) == before
    assert _identities(side) == identity


# ── 2. removal, then re-admission of the same slot ───────────────────────────


def test_a_released_slot_is_emptied_again_for_every_new_owner():
    runner, banks, side = _runner()
    free = 21
    owners = _scattered_owners(free)
    _occupy(runner, owners)
    gen = torch.Generator().manual_seed(SEED + 2)
    previous = None
    for generation in range(3):
        rid = f"gen-{generation}"
        if previous is not None:
            runner._glm5next_note_finished_requests({previous})
        before = _slot_digests(side)
        slots = runner._glm5next_request_slots(banks, [rid], synthetic=False, side_caches=side)
        assert slots == [free], (generation, slots)
        _assert_fresh(side, free)
        moved = {(i, k, free) for i, entry in enumerate(side)
                 for k in ("pool_cache", "tail") if k in entry}
        _assert_only_moved(before, _slot_digests(side), moved)
        # The owner runs: its slot fills with its own state before it finishes.
        for entry in side:
            for key in ("pool_cache", "tail"):
                if key in entry:
                    entry[key][free].copy_((torch.rand(entry[key][free].shape, generator=gen) + 1).to(DTYPE))
        runner._glm5next_side_cache_positions[free] = 11
        previous = rid
    assert runner._glm5next_request_slot_table[previous] == free


# ── 3. a request re-opening at position 0 in the slot it holds ───────────────


def test_a_request_resumed_from_position_zero_gets_its_ring_emptied_in_place():
    runner, banks, side = _runner()
    owners = _scattered_owners(BATCH)  # all 64 slots owned
    _occupy(runner, owners)
    resumed = "req-29"
    slot = owners[resumed]
    before, identity = _slot_digests(side), _identities(side)

    # The converter's order for an opening step: slot, then the position arm.
    slots = runner._glm5next_request_slots(banks, [resumed], synthetic=False, side_caches=side)
    assert slots == [slot]
    runner._glm5next_position_arm(slot, 0, side_caches=side, is_prefill=True)

    assert _identities(side) == identity
    _assert_fresh(side, slot, keys=("tail",))
    moved = {(i, "tail", slot) for i, entry in enumerate(side) if "tail" in entry}
    after = _slot_digests(side)
    _assert_only_moved(before, after, moved)
    # The pooled store is 0a08ff4's: left alone on a re-open (unreachable above the bound).
    assert all(after[(i, "pool_cache", slot)] == before[(i, "pool_cache", slot)]
               for i, entry in enumerate(side) if "pool_cache" in entry)
    assert slot not in runner._glm5next_side_cache_positions


# ── 4. the device rule: wait on the bank, then one host copy_ of the slot ──


class _Writes(TorchDispatchMode):
    """Every aten op that writes or reads a side-cache storage during the reset."""

    def __init__(self, storages: set[int]):
        super().__init__()
        self.storages = storages
        self.calls: list[tuple[str, list[int], int]] = []

    def _side(self, value):
        return (isinstance(value, torch.Tensor)
                and value.untyped_storage().data_ptr() in self.storages)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        touched = [i for i, a in enumerate(list(args) + list(kwargs.values())) if self._side(a)]
        if touched and func.overloadpacket.__name__ not in {"select", "slice", "view", "alias",
                                                          "as_strided", "detach"}:
            dst = args[0]
            self.calls.append((str(func), touched, int(dst.numel()) if self._side(dst) else -1))
        return func(*args, **kwargs)


def test_the_reset_reads_one_element_then_writes_the_slot_with_one_host_copy():
    """Per side cache: a one-element device-to-host read, then ``copy_`` of the slot.

    The read is the ordering. The backend's host-to-device ``_copy_from`` does not wait
    for an execution still writing the bank (a finished request's last, asynchronously
    scheduled step can still be writing its slot), while its device-to-host copy waits
    on the storage's pending execution (``NeuronTensorImpl::Await``). 0a08ff4's rebuild
    read the whole bank and so waited; one element waits the same and moves 2 bytes.
    """
    runner, banks, side = _runner()
    free = 44
    _occupy(runner, _scattered_owners(free))
    storages = {entry[k].untyped_storage().data_ptr() for entry in side for k in entry}
    slot_numel = {k: int(side[-1][k][0].numel()) for k in ("pool_cache", "tail")}

    with _Writes(storages) as seen:
        runner._glm5next_request_slots(banks, ["new"], synthetic=False, side_caches=side)
        runner._glm5next_position_arm(free, 0, side_caches=side, is_prefill=True)

    assert seen.calls, "the reset touched no side cache at all"
    kinds = {name for name, _, _ in seen.calls}
    assert kinds == {"aten._to_copy.default", "aten.copy_.default"}, (
        f"the reset reached {sorted(kinds)}; on the Neuron device only copy_ is the native "
        f"_copy_from write, every other in-place op takes the CPU fallback that refuses "
        f"shared storage, and a whole-tensor op touches every slot"
    )
    for name, touched, numel in seen.calls:
        # The side cache is argument 0 (the read's source, the write's destination).
        assert touched == [0], (name, touched)
    # Hand-out: pool_cache and tail per layer; the position arm: the tail again. Each
    # is a one-element read immediately followed by the slot's write.
    pairs = [seen.calls[i:i + 2] for i in range(0, len(seen.calls), 2)]
    assert len(seen.calls) == 2 * 3 * SPARSE
    for (read, read_touched, read_numel), (write, _, write_numel) in pairs:
        assert (read, read_numel) == ("aten._to_copy.default", 1), (read, read_numel)
        assert write == "aten.copy_.default"
        assert write_numel in slot_numel.values(), f"wrote {write_numel} elements, not one slot"
