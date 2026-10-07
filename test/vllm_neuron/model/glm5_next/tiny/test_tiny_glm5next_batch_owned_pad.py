# SPDX-License-Identifier: Apache-2.0
"""A padded decode whose only idle slot belongs to a request the step does not schedule.

``_glm5next_idle_slots`` hands a batch bucket's padding row a slot no request in the step
holds. It takes slots no live request owns first. When every slot is owned, it falls back
to a slot owned by a live request that this step does not schedule. One slot number
addresses that request's recurrent (KDA) state and its indexer (DSA) side caches, so both
must come back as they were. The padding row carries no token: on the KDA family real
length 0, row mask 0 and start position 1, so the kernel neither opens nor advances the
state; on the DSA family position 0 in the null block, the scratch ring ``pad_tail``, and a
pooled write that lands on its view's trash row, which no candidate gather reads.

``test_tiny_glm5next_batch_kda.py`` and ``test_tiny_glm5next_batch_decode.py`` read the
first case (more slots than requests). This file reads the fallback. The stack keys four
slots (``max_num_seqs`` 4) and four requests are prefilled, so all four slots are owned.
A step then schedules three of them in the bucket of four.

KDA (the fixtures of ``test_tiny_glm5next_batch_kda.py`` at four slots: real
``Glm5NextKDAAttention`` modules at the TP=64 geometry, random bank rows so a stray write
shows):

1. the fourth request's ``conv_state`` and ``recurrent_state`` rows are byte-equal before
   and after the step, in every layer;
2. the padding row was served from that slot, and the slot table is unchanged;
3. the three scheduled requests' rows equal three one-request steps from the same snapshot,
   so the padding row did not share a scheduled request's slot.

DSA (the tiny root of ``test_tiny_glm5next_batch_decode.py`` at four slots):

1. the fourth request's ring and every pooled-store row of its slot but the trash row are
   byte-equal before and after the step, in every layer, and so are its latent pages;
2. the padding row's pooled-store view is that slot, and its ring is ``pad_tail``;
3. the fourth request's next decode gives bit-equal logits to the same decode from the
   snapshot, so the trash-row write changes nothing the request reads.

Both arms drive the runner's own carrier builder, ``_glm5next_model_kwargs``.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/model/glm5_next/tiny/test_tiny_glm5next_batch_owned_pad.py
"""

from __future__ import annotations

import pytest
import torch

from vllm_neuron.vllm.worker.neuron_model_runner import NULL_BLOCK_ID

from test.vllm_neuron.model.glm5_next import test_kda_layer as layer_half
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_batch_decode as batch
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_batch_kda as kda
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_forward as tiny

pytestmark = [pytest.mark.fast, pytest.mark.forked]

#: The batch bucket, and the stack's slot count: every slot is owned once all are prefilled.
BUCKET = 4
SCHEDULED = 3


def test_a_padding_row_on_an_owned_unscheduled_slot_leaves_that_slots_kda_state_unchanged(
    monkeypatch,
):
    monkeypatch.setattr(kda, "MAX_NUM_SEQS", BUCKET)
    monkeypatch.setenv("VLLM_NEURON_GLM5NEXT_STATE_BANKS", "0")  # this test pins the per-request VIEW form
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


def test_a_padding_row_on_an_owned_unscheduled_slot_leaves_that_requests_dsa_state_unchanged(monkeypatch):
    monkeypatch.setenv("VLLM_NEURON_GLM5NEXT_STATE_BANKS", "0")  # this test pins the per-request VIEW form
    max_model_len = tiny.STACK_TOKENS + 8
    world = batch._world(BUCKET, max_model_len=max_model_len, prompts=[6, 9, 11, 7],
                         window_blocks=-(-max_model_len // batch.PAGE), slots=BUCKET)
    table = world.runner._glm5next_request_slot_table
    assert sorted(table.values()) == list(range(BUCKET)), table
    idle = SCHEDULED
    idle_slot = table[world.req_ids[idle]]
    pages = sorted(set(world.tables[idle]) - {NULL_BLOCK_ID})
    snapshot = batch._snapshot(world)
    before_table = dict(table)

    # The idle request's next decode, from the snapshot, with no padded step before it.
    alone = batch._step(world, [idle], torch.tensor([world.first[idle]]),
                        cached=[world.lengths[idle]], sampling=[0])

    batch._restore(world, snapshot)
    carriers: list = []
    batch._step(world, list(range(SCHEDULED)),
                torch.tensor(world.first[:SCHEDULED] + [0] * (BUCKET - SCHEDULED)),
                cached=world.lengths[:SCHEDULED], sampling=list(range(SCHEDULED)),
                real=[1] * SCHEDULED, out=carriers)
    assert dict(table) == before_table, (table, before_table)

    sides = world.runner._glm5next_side_cache_set
    _, saved_sides, _ = snapshot
    checked = 0
    for index, (carrier, bank) in enumerate(zip(carriers[0], world.root.glm5next_layer_banks)):
        side, saved = sides[index], saved_sides[index]
        if not side:
            continue
        checked += 1
        # ---- 1. The idle request's ring, pooled rows and latent pages are unchanged.
        assert torch.equal(side["tail"][idle_slot], saved["tail"][idle_slot]), (
            f"layer {index}: the ring of slot {idle_slot} was changed by the padding row"
        )
        assert torch.equal(side["pool_cache"][idle_slot, :-1],
                           saved["pool_cache"][idle_slot, :-1]), (
            f"layer {index}: a pooled row of slot {idle_slot} other than its trash row "
            f"was changed by the padding row"
        )
        latent = world.caches[bank["name"]][0]
        saved_latent = snapshot[0][bank["name"]][0]
        assert torch.equal(latent[pages], saved_latent[pages]), (
            f"layer {index}: the idle request's latent pages {pages} changed"
        )
        # ---- 2. The padding row's views: the idle request's pooled store, the scratch ring.
        assert int(carrier["block_table_row"].shape[1]) == BUCKET
        stride = side["pool_cache"][0].numel() * side["pool_cache"].element_size()
        pad_slot = (
            carrier["pool_cache"][SCHEDULED].data_ptr() - side["pool_cache"].data_ptr()
        ) // stride
        assert pad_slot == idle_slot, (index, pad_slot, idle_slot)
        assert carrier["tail"][SCHEDULED].data_ptr() == side["pad_tail"][0].data_ptr()
    assert checked == tiny.STACK_LAYERS, checked

    # ---- 3. The idle request's next decode reads nothing the padding row wrote.
    after = batch._step(world, [idle], torch.tensor([world.first[idle]]),
                        cached=[world.lengths[idle]], sampling=[0])
    assert torch.equal(after, alone), float((after - alone).abs().max())
