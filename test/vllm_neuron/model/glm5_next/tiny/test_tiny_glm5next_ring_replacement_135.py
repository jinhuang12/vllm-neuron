"""A new sequence gets a FRESH ring, and the ring it replaces is never written in place.

WHAT THIS FILE MEASURES. The converter opens a sequence by discarding the decode ring's rows.
It used to empty them with an eager `zero_()`, which the runtime refuses on a device-resident
buffer -- "Can't call ReserveSpace on shared storage" -- inside the input builder, before any
forward runs. The ring is now REPLACED by a fresh allocation of the same shape, dtype and
device, and the carriers built after that loop bind the new buffer. This file reads the two
properties that distinguishes a replacement from a clear: the entry's object changes, and the
buffer it replaced keeps its own bytes.

WHAT IT DOES NOT MEASURE, stated so the gap is not read as coverage. That the new ring is
EMPTY, and that a decode step keeps the ring it was handed: item 10 of
``test_tiny_glm5next_e2e.py`` owns both and is graded beside this file. That the runtime
accepts the replacement on a device: no CPU-mode run can read that.

HOW TO RUN IT, both variables from the process environment:

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 python -m pytest -s -rA \\
        test/vllm_neuron/model/glm5_next/tiny/test_tiny_glm5next_ring_replacement_135.py
"""

from __future__ import annotations

import pytest
import torch

from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_e2e as e2e
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_forward as item

pytestmark = [pytest.mark.fast, pytest.mark.forked]

#: The value planted in the ring before the opening step. Any non-zero value does, and this
#: one is exact in every dtype a bank can carry, so a stored value cannot round to zero and
#: read as a clear that never happened.
PLANTED = 3.0


def test_a_fresh_sequence_replaces_the_ring_and_never_writes_the_old_one():
    """The entry's object changes on reset, and the buffer it replaced keeps its bytes.

    THE PLANTED VALUE IS WHAT MAKES THE SECOND READING POSSIBLE. An old buffer that was
    already zero cannot show whether anything wrote to it, so the ring is filled first and
    the same buffer is compared against its own clone afterwards.

    A CLEAR IN PLACE FAILS BOTH READINGS: the entry still holds the same object, and that
    object's bytes are now zero rather than the planted value.
    """
    e2e._require_cpu_mode()
    root = e2e._fixture()["root"]
    root.bind_kv_cache(e2e._runner_shaped_caches(root))
    banks = root.glm5next_layer_banks
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.model = root
    runner.max_model_len = e2e.E2E_MAX_SEQ_LEN

    rings = [side for side in runner._glm5next_live_side_caches(banks) if "tail" in side]
    if not rings:
        raise item.VacuousControlError(
            "no bank in this stack carries a ring, so this item measures nothing"
        )
    for side in rings:
        side["tail"].fill_(PLANTED)
    before = [side["tail"] for side in rings]
    planted = [buffer.clone() for buffer in before]

    e2e._model_kwargs(
        runner,
        input_ids=torch.zeros(item.STACK_TOKENS, dtype=torch.long),
        cached=0,
        sampling_row=item.STACK_TOKENS - 1,
    )

    after = [side["tail"] for side in rings]
    replaced = sum(1 for old, new in zip(before, after) if old is not new)
    untouched = sum(
        1 for old, own in zip(before, planted) if torch.equal(old, own)
    )
    print(
        f"INC135|reset|rings={len(rings)}|entries_replaced={replaced}"
        f"|old_buffers_that_kept_their_bytes={untouched}"
        f"|planted={PLANTED}|new_ring_max={max(float(new.abs().max()) for new in after)}"
    )
    assert replaced == len(rings), (
        f"{len(rings) - replaced} ring(s) still hold the SAME buffer after the opening "
        f"step; the reset emptied a device-resident buffer in place instead of replacing it"
    )
    assert untouched == len(rings), (
        f"{len(rings) - untouched} replaced buffer(s) lost their own bytes; something wrote "
        f"the old ring in place, which is the write the runtime refuses on a device"
    )
