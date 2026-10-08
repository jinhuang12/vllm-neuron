# SPDX-License-Identifier: Apache-2.0
"""The served converter empties a new request's DSA slot in place, before its first step.

``test/vllm_neuron/worker/test_glm5next_slot_reset.py`` reads the hand-out on a
synthetic stack. Here the tiny root is driven through ``_glm5next_model_kwargs``, the
converter a served step goes through, with 64 request slots (plus the scratch slot):

1. a new request admitted while 63 others hold scattered slots: after the converter and
   before the forward, its slot's pooled store and ring are the fresh-slot contract
   (``+0.0`` bytes), every other slot is bitwise unchanged, and the carriers the
   forward reads are views of the very tensors the reset wrote (same storage), so the
   reset lands where the step reads it;
2. the step that follows is the one a never-used slot gives: the logits and the slot's
   side-cache rows after the forward equal, bitwise, those of the same prompt served
   from all-zero caches, so a previous owner's leftovers cannot leak into a new sequence;
3. removal then re-admission, and a preempted request re-prefilled from position 0,
   through the converter: each re-open empties again, in place.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/model/glm5_next/tiny/test_tiny_glm5next_slot_reset.py
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest
import torch

from vllm_neuron.vllm.worker import glm5next_state_banks as runner_side
from vllm_neuron.vllm.worker.neuron_model_runner import NULL_BLOCK_ID, NeuronModelRunner

from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_batch_decode as dsa
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_e2e as e2e
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_forward as tiny

pytestmark = [pytest.mark.fast, pytest.mark.forked]

CAPACITY = 64
SEED = 20261007
PROMPT = 7


@pytest.fixture(autouse=True)
def _bank_form_on(monkeypatch):
    monkeypatch.delenv(runner_side.STATE_BANKS_ENV, raising=False)


def _digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _world(*, leftovers: bool):
    """The tiny root, 64 request slots, the side caches allocated by the runner itself.

    With ``leftovers`` every slot of every side cache holds distinct non-zero values, as
    a process that has served earlier sequences leaves them.
    """
    e2e._require_cpu_mode()
    max_model_len = tiny.STACK_TOKENS + 8
    window_blocks = -(-max_model_len // dsa.PAGE)
    root = e2e._fixture()["root"]
    head = torch.randn(tiny.STACK_VOCAB_SIZE, int(root.text_config.hidden_size),
                       generator=torch.Generator().manual_seed(dsa.SEED))
    root.lm_head_weight = torch.nn.Parameter(head.to(torch.bfloat16), requires_grad=False)
    requests = 4
    per_request = -(-max_model_len // dsa.PAGE)
    caches = {}
    for spec in root.get_kv_spec().layers:
        shape = (1 + requests * per_request, int(spec.num_kv_heads), dsa.PAGE, int(spec.head_size))
        caches[spec.name] = [torch.zeros(shape, dtype=spec.dtype)
                             for _ in range(1 if spec.latent_kv else 2)]
    root.bind_kv_cache(caches)
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.input_batch = SimpleNamespace(req_ids=[])
    runner.model = root
    runner.max_model_len = int(max_model_len)
    runner.max_num_reqs = CAPACITY
    tables = [[1 + r * per_request + k for k in range(per_request)]
              + [NULL_BLOCK_ID] * (window_blocks - per_request) for r in range(requests)]
    world = SimpleNamespace(root=root, runner=runner, caches=caches, tables=tables,
                            window_blocks=window_blocks,
                            req_ids=[f"new-{r}" for r in range(requests)])
    sides = runner._glm5next_live_side_caches(root.glm5next_layer_banks)
    assert sum(1 for side in sides if side) == tiny.STACK_LAYERS
    assert all(int(side["pool_cache"].shape[0]) == CAPACITY + runner_side.SCRATCH_SLOTS
               for side in sides if side)
    if leftovers:
        gen = torch.Generator().manual_seed(SEED)
        for side in sides:
            for key, value in side.items():
                value.copy_((torch.rand(value.shape, generator=gen) + 0.5).to(value.dtype))
    return world


def _hold(world, owners: dict[str, int], cursor: int = 5) -> None:
    world.runner._glm5next_request_slot_table = dict(owners)
    world.runner._glm5next_side_cache_positions = {slot: cursor for slot in owners.values()}


def _scattered_owners(free: set[int]) -> dict[str, int]:
    order = torch.randperm(CAPACITY, generator=torch.Generator().manual_seed(SEED + 1)).tolist()
    return {f"held-{slot}": slot for slot in order if slot not in free}


def _slot_digests(world) -> dict:
    out = {}
    for index, side in enumerate(world.runner._glm5next_side_cache_set):
        for key, value in side.items():
            for slot in range(int(value.shape[0])):
                out[(index, key, slot)] = _digest(value[slot])
    return out


def _storages(world) -> dict:
    return {(index, key): value.untyped_storage().data_ptr()
            for index, side in enumerate(world.runner._glm5next_side_cache_set)
            for key, value in side.items()}


def _opening(world, r: int, prompt: torch.Tensor) -> dict:
    """The converter's kwargs for request ``r``'s prefill at position 0 (no forward)."""
    runner = world.runner
    runner.input_batch.req_ids = [world.req_ids[r]]
    runner._glm5next_request_tokens = None
    return runner._glm5next_model_kwargs({
        "input_ids": prompt,
        "positions": None,
        "attn_metadata": dsa._metadata(world, [r], cached=[0], tokens=int(prompt.shape[0])),
        "sampling_positions": torch.tensor([int(prompt.shape[0]) - 1], dtype=torch.long),
        "sampling_params": None,
        "spec_decode_metadata": None,
        "rank": None,
        "logit_mask": None,
    })


def _prompt(seed: int = SEED + 3) -> torch.Tensor:
    return torch.randint(0, tiny.STACK_VOCAB_SIZE, (PROMPT,),
                         generator=torch.Generator().manual_seed(seed))


def _fresh(world, slot: int, keys=("pool_cache", "tail")) -> list:
    stale = []
    for index, side in enumerate(world.runner._glm5next_side_cache_set):
        for key in keys:
            if key in side and _digest(side[key][slot]) != _digest(torch.zeros_like(side[key][slot])):
                stale.append((index, key))
    return stale


def _changed(before: dict, after: dict) -> set:
    return {key for key in before if before[key] != after[key]}


def test_a_new_request_among_63_held_slots_opens_on_an_emptied_slot_of_the_same_storage():
    world = _world(leftovers=True)
    free = 41
    _hold(world, _scattered_owners({free}))
    before, storages = _slot_digests(world), _storages(world)

    converted = _opening(world, 0, _prompt())

    slot = world.runner._glm5next_request_slot_table[world.req_ids[0]]
    assert slot == free
    assert _storages(world) == storages, "the reset rebuilt a side cache instead of writing it"
    assert _fresh(world, free) == [], "the new request's slot still holds the last owner's rows"
    moved = {(i, k, free) for i, side in enumerate(world.runner._glm5next_side_cache_set)
             for k in ("pool_cache", "tail") if k in side}
    assert _changed(before, _slot_digests(world)) == moved
    # The carriers the forward reads are views of the storage the reset wrote.
    sides = world.runner._glm5next_side_cache_set
    read = 0
    for index, carrier in enumerate(converted["layer_carriers"]):
        for key in ("pool_cache", "tail", "prefill_tail"):
            if key in carrier and torch.is_tensor(carrier[key]):
                source = "tail" if key == "prefill_tail" else key
                assert (carrier[key].untyped_storage().data_ptr()
                        == sides[index][source].untyped_storage().data_ptr()), (index, key)
                read += 1
    assert read >= 2 * tiny.STACK_LAYERS, read

    # The forward writes the new request's slot and no other.
    before_forward = _slot_digests(world)
    world.root.forward(**converted)
    touched = _changed(before_forward, _slot_digests(world))
    assert touched and all(slot_ == free for _, _, slot_ in touched), sorted(touched)[:6]


def test_an_opening_on_a_reused_slot_is_bitwise_an_opening_on_a_never_used_one():
    prompt = _prompt()
    free = 17
    results = {}
    for leftovers in (True, False):
        world = _world(leftovers=leftovers)
        _hold(world, _scattered_owners({free}))
        converted = _opening(world, 0, prompt)
        logits = world.root.forward(**converted).float()
        sides = world.runner._glm5next_side_cache_set
        results[leftovers] = {
            "logits": _digest(logits),
            **{(i, k): _digest(side[k][free]) for i, side in enumerate(sides)
               for k in ("pool_cache", "tail") if k in side},
        }
    assert results[True] == results[False], (
        "a reused slot's opening differs from a never-used slot's: "
        f"{[k for k in results[False] if results[True][k] != results[False][k]]}"
    )


def test_removal_then_readmission_through_the_converter_empties_the_slot_each_time():
    world = _world(leftovers=True)
    free = 63
    _hold(world, _scattered_owners({free}))
    previous = None
    for r in range(3):
        if previous is not None:
            world.runner._glm5next_note_finished_requests({previous})
        before = _slot_digests(world)
        converted = _opening(world, r, _prompt(SEED + 10 + r))
        rid = world.req_ids[r]
        assert world.runner._glm5next_request_slot_table[rid] == free
        assert _fresh(world, free) == [], (r, _fresh(world, free))
        assert all(slot == free for _, _, slot in _changed(before, _slot_digests(world)))
        world.root.forward(**converted)
        # The owner's prefill leaves its own rows behind for the next owner to inherit.
        assert _fresh(world, free), "the prefill wrote nothing, so re-admission reads nothing"
        previous = rid


def test_a_preempted_request_reprefilled_from_position_zero_gets_its_ring_emptied_in_place():
    world = _world(leftovers=True)
    owners = _scattered_owners({9})
    owners[world.req_ids[0]] = 9  # the resumed request keeps the slot it holds
    _hold(world, owners, cursor=PROMPT + 3)
    before, storages = _slot_digests(world), _storages(world)

    _opening(world, 0, _prompt())

    assert world.runner._glm5next_request_slot_table[world.req_ids[0]] == 9
    assert _storages(world) == storages
    assert _fresh(world, 9, keys=("tail",)) == []
    changed = _changed(before, _slot_digests(world))
    assert changed == {(i, "tail", 9) for i, side in enumerate(world.runner._glm5next_side_cache_set)
                       if "tail" in side}, sorted(changed)[:6]
