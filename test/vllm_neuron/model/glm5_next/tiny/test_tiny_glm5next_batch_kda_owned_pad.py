# SPDX-License-Identifier: Apache-2.0
"""A padded KDA decode whose only idle slot belongs to a request the step does not schedule.

``_glm5next_idle_slots`` hands a batch bucket's padding row a slot no request in the step
holds. It takes slots no live request owns first. When every slot is owned, it falls back
to a slot owned by a live request that this step does not schedule. That request's
recurrence must come back exactly as it was, because the padding row carries no token
(real length 0, row mask 0, start position 1, so the kernel neither opens nor advances the
state).

``test_tiny_glm5next_batch_kda.py`` reads the first case (eight slots, three requests). This
file reads the fallback. The stack keys four slots (``max_num_seqs`` 4) and four requests
are prefilled, so all four slots are owned. A step then schedules three of them in the
bucket of four:

1. the fourth request's ``conv_state`` and ``recurrent_state`` rows are byte-equal before
   and after the step, in every layer;
2. the padding row was served from that slot, and the slot table is unchanged;
3. the three scheduled requests' rows equal three one-request steps from the same snapshot,
   so the padding row did not share a scheduled request's slot.

The fixtures are the ones of ``test_tiny_glm5next_batch_kda.py``, at a slot count of four:
real ``Glm5NextKDAAttention`` modules at the TP=64 geometry, driven through the runner's own
carrier builder, with random bank rows so a stray write shows.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/model/glm5_next/tiny/test_tiny_glm5next_batch_kda_owned_pad.py
"""

from __future__ import annotations

import pytest
import torch

from test.vllm_neuron.model.glm5_next import test_kda_layer as layer_half
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_batch_kda as kda

pytestmark = [pytest.mark.fast, pytest.mark.forked]

#: The batch bucket, and the stack's slot count: every slot is owned once all are prefilled.
BUCKET = 4
SCHEDULED = 3


def test_a_padding_row_on_an_owned_unscheduled_slot_leaves_that_slots_kda_state_unchanged(
    monkeypatch,
):
    monkeypatch.setattr(kda, "MAX_NUM_SEQS", BUCKET)
    world = kda._world(BUCKET)
    table = world.runner._glm5next_request_slot_table
    assert sorted(table.values()) == list(range(BUCKET)), table
    scheduled, idle = world.req_ids[:SCHEDULED], world.req_ids[SCHEDULED]
    idle_slot = table[idle]
    busy = {table[r] for r in scheduled}
    for index, bank in enumerate(world.banks):
        assert int(bank["state_slots"]) == BUCKET, (index, bank["state_slots"])
    snapshot = kda._snapshot(world)
    before_table = dict(table)
    lengths = list(kda.PROMPTS[:SCHEDULED])

    layer_half._reset_counters()
    rows = torch.cat([world.decodes[0][:SCHEDULED],
                      torch.zeros(BUCKET - SCHEDULED, world.hidden)])
    out, carriers = kda._step(
        world.runner, world.banks, world.layers, req_ids=scheduled, rows=rows,
        cached=lengths, max_query_len=1, real=[1] * SCHEDULED,
    )
    counts = layer_half._read_counters()
    assert counts["fused"] == (kda.LAYERS, 0), counts

    # ---- 1. The unscheduled request's slot is byte-equal before and after.
    for index, (after, saved) in enumerate(zip(world.banks, snapshot[0])):
        for key in ("conv_state", "recurrent_state"):
            assert torch.equal(kda._bytes(after[key][idle_slot]),
                               kda._bytes(saved[key][idle_slot])), (
                f"layer {index} {key}: slot {idle_slot}, owned by the unscheduled "
                f"{idle!r}, was changed by the padding row"
            )

    # ---- 2. The padding row is served from the unscheduled request's slot.
    assert dict(table) == before_table, (table, before_table)
    for index, (carrier, bank) in enumerate(zip(carriers, world.banks)):
        assert len(carrier["conv_state"]) == BUCKET
        assert carrier["real_tokens"].reshape(-1).tolist() == [1] * SCHEDULED + [0]
        assert carrier["row_mask"].reshape(-1).tolist() == [1.0] * SCHEDULED + [0.0]
        assert int(carrier["start_position"].reshape(-1)[SCHEDULED]) != 0
        for key in ("conv_state", "recurrent_state"):
            pad_view = carrier[key][SCHEDULED]
            pad_slot = next(
                slot for slot in range(BUCKET)
                if pad_view.data_ptr() == bank[key][slot].data_ptr()
            )
            assert pad_slot == idle_slot and pad_slot not in busy, (
                f"layer {index} {key}: the padding row took slot {pad_slot}; the only "
                f"slot outside the step is {idle_slot}, owned by {idle!r}"
            )

    # ---- 3. The scheduled requests advance as one-request steps advance them.
    padded_banks = [{k: v.clone() for k, v in bank.items() if torch.is_tensor(v)}
                    for bank in world.banks]
    kda._restore(world, snapshot)
    single = []
    for r, rid in enumerate(scheduled):
        row, _ = kda._step(world.runner, world.banks, world.layers, req_ids=[rid],
                           rows=world.decodes[0][r : r + 1], cached=[lengths[r]],
                           max_query_len=1)
        single.append(row)
    kda._assert_banks_equal(padded_banks, world.banks, "padded 3 in 4, all slots owned")
    torch.testing.assert_close(out[:, :SCHEDULED], torch.cat(single, dim=1),
                               rtol=kda.OUT_RTOL, atol=kda.OUT_ATOL)
