"""A new sequence gets a fresh ring; the ring it replaces is never written in place.

Opening a sequence discards the decode ring's rows. Emptying them with an eager
``zero_()`` is refused on the device ("Can't call ReserveSpace on shared storage") inside
the input builder, so the ring is replaced by a fresh allocation of the same shape, dtype
and device and the carriers built afterwards bind the new buffer. The two properties that
tell a replacement from a clear are read here: the entry's object changes, and the buffer
it replaced keeps its own bytes.

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 python -m pytest \\
        test/vllm_neuron/model/glm5_next/tiny/test_tiny_glm5next_ring_replacement.py
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

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


def test_a_fresh_sequence_replaces_the_ring_and_never_writes_the_old_one():
    """The entry's object changes on reset, and the buffer it replaced keeps its bytes.

    The ring is filled with a recognisable value first: an old buffer that was already
    zero could not show whether anything wrote to it. A clear in place fails both
    readings, because the entry then still holds the same object and that object's bytes
    are zero rather than the planted value.
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

    e2e._model_kwargs(
        runner,
        input_ids=torch.zeros(tiny.STACK_TOKENS, dtype=torch.long),
        cached=0,
        sampling_row=tiny.STACK_TOKENS - 1,
    )

    after = [side["tail"] for side in rings]
    replaced = sum(1 for old, new in zip(before, after) if old is not new)
    untouched = sum(
        1 for old, own in zip(before, planted) if torch.equal(old, own)
    )
    assert replaced == len(rings), (
        f"{len(rings) - replaced} ring(s) still hold the same buffer after the opening "
        f"step; the reset emptied a device-resident buffer in place instead of replacing it"
    )
    assert untouched == len(rings), (
        f"{len(rings) - untouched} replaced buffer(s) lost their own bytes; something wrote "
        f"the old ring in place, which is the write that was refused on the device"
    )
