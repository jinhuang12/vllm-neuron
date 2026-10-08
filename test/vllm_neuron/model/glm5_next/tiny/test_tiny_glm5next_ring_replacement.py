"""A new sequence's ring is emptied in place, in its own slot, by a host copy; never by zero_().

Opening a sequence discards the decode ring's rows. Emptying them with an eager
``zero_()`` is refused on the device ("Can't call ReserveSpace on shared storage") inside
the input builder: ``zero_`` (like ``fill_``, ``index_fill_``, ``masked_fill_`` and
``index_put_``) reaches the Neuron backend's CPU fallback, whose write-back
(``_copy_from_and_resize``) refuses a tensor whose storage is shared. A whole-ring rebuild
avoids that op but reads and writes every slot of every layer for each new sequence. The
ring is therefore emptied by ``copy_`` of the opening request's slot from a host tensor of
zeros (the backend's native ``_copy_from``), in place on the tensor the carriers bind.

Read here, through the converter on the tiny root: the entry keeps its object, the
opening request's slot is zero, every other slot keeps its planted bytes, and no
fallback-routed in-place op touches a ring.

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 python -m pytest \\
        test/vllm_neuron/model/glm5_next/tiny/test_tiny_glm5next_ring_replacement.py
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_e2e as e2e
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_forward as tiny

pytestmark = [pytest.mark.fast, pytest.mark.forked]

# Any non-zero value does. This one is exact in every dtype a bank can carry, so a stored
# value cannot round to zero and read as a clear that never happened.
PLANTED = 3.0

# The converter keys a sequence's state by request id and serves an id-less step as
# synthetic, which opens no ring at all.
REQUEST = "ring-replacement-request"

#: The in-place ops the Neuron backend serves through its CPU fallback (refused on a
#: shared storage); none may write a ring.
REFUSED = {"zero_", "fill_", "index_fill_", "masked_fill_", "index_put_", "_index_put_impl_"}


class _RefusedOps(TorchDispatchMode):
    def __init__(self, storages: set[int]):
        super().__init__()
        self.storages = storages
        self.hits: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        target = args[0] if args else None
        if (func.overloadpacket.__name__ in REFUSED and isinstance(target, torch.Tensor)
                and target.untyped_storage().data_ptr() in self.storages):
            self.hits.append(str(func))
        return func(*args, **(kwargs or {}))


def test_a_fresh_sequence_empties_its_ring_slot_in_place_and_never_zeroes_the_ring():
    """The entry keeps its object; the opening slot is zero; the other slots keep their bytes.

    The ring is filled with a recognisable value first: a ring that was already zero
    could not show whether anything wrote to it.
    """
    e2e._require_cpu_mode()
    root = e2e._fixture()["root"]
    root.bind_kv_cache(e2e._runner_shaped_caches(root))
    banks = root.glm5next_layer_banks
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.model = root
    runner.max_model_len = e2e.E2E_MAX_SEQ_LEN
    runner.max_num_reqs = e2e.E2E_MAX_NUM_SEQS
    runner.input_batch = SimpleNamespace(req_ids=[REQUEST])

    rings = [side for side in runner._glm5next_live_side_caches(banks) if "tail" in side]
    if not rings:
        raise tiny.VacuousControlError(
            "no bank in this stack carries a ring, so there is nothing to read here"
        )
    for side in rings:
        side["tail"].fill_(PLANTED)
    before = [side["tail"] for side in rings]
    planted = [buffer.clone() for buffer in before]
    storages = {buffer.untyped_storage().data_ptr() for buffer in before}

    with _RefusedOps(storages) as refused:
        e2e._model_kwargs(
            runner,
            input_ids=torch.zeros(tiny.STACK_TOKENS, dtype=torch.long),
            cached=0,
            sampling_row=tiny.STACK_TOKENS - 1,
        )

    slot = e2e._own_slot(runner, REQUEST)
    after = [side["tail"] for side in rings]
    kept = sum(1 for old, new in zip(before, after) if old is new)
    assert kept == len(rings), (
        f"{len(rings) - kept} ring(s) were replaced by a new buffer on the opening step; "
        f"the reset is one slot written in place, not a rebuild of the ring"
    )
    assert not refused.hits, (
        f"the opening step reached {sorted(set(refused.hits))} on a ring; on the device "
        f"those take the CPU fallback that refuses shared storage"
    )
    for index, (ring, own) in enumerate(zip(after, planted)):
        assert torch.equal(ring[slot], torch.zeros_like(ring[slot])), (
            f"ring {index}: the opening request's slot {slot} still holds the planted rows"
        )
        others = [s for s in range(int(ring.shape[0])) if s != slot]
        assert others, "a ring of one slot cannot show that its other slots are spared"
        assert torch.equal(ring[others], own[others]), (
            f"ring {index}: a slot other than the opening request's changed"
        )
