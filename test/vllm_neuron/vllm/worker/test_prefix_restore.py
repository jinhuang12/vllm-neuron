# SPDX-License-Identifier: Apache-2.0
"""A prefix-cache hit is restored into its slot before the step that continues it.

With prefix caching on, vLLM can admit a request at a cached length ``P`` (a whole
number of blocks). The runner keeps a request's KDA state and DSA indexer state per
request slot, and the slot it hands such a request holds no sequence. The runner's
stores have room for the state a cold prefill leaves after ``P`` tokens, per block: each
recurrent layer's checkpoint in the page of its KDA group's block ``P // block_size - 1``
(``glm5next_state_banks``, "The KDA checkpoint store") and each sparse layer's pool rows
in ``NeuronModelRunner.dsa_pool_store`` under the latent group's blocks. Before the
position check, the step builder (``NeuronModelRunner._glm5next_model_kwargs``) copies
that state into the slot: the KDA checkpoint into the slot's entering row, pool rows
``0 .. P // index_kpool - 1`` into its ``pool_cache``, an empty ring, no resume row, and
the cursor at ``P``. The step then runs as the second segment of a cold prefill does.

A hit restores only from blocks a prefill has recorded writing
(``NeuronModelRunner._glm5next_record_stored_blocks``) since vLLM last handed them out;
a hit that names any other block is refused by name, and nothing is written.

No code in the tree writes the stores, so these tests write them by hand, record their
blocks, and step the runner's own step builder over a one-sparse-layer, one-KDA-layer
world whose KV cache config, checkpoint pages and pool store are built by the runner's
own functions.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_neuron.vllm.patches.kv_spec_patch import recurrent_state_slot_bytes
from vllm_neuron.vllm.worker import neuron_model_runner
from vllm_neuron.vllm.worker.glm5next_state_banks import (
    SCRATCH_SLOTS,
    kda_checkpoint_layout,
    read_kda_checkpoints,
    state_bank_regions,
    write_kda_checkpoints,
)
from vllm_neuron.vllm.worker.neuron_model_runner import (
    NULL_BLOCK_ID,
    NeuronModelRunner,
    dsa_pool_store_shapes,
    kda_checkpoint_page_views,
)

pytestmark = [pytest.mark.fast]

#: vLLM's KV block for this model with prefix caching, and the hit granularity: a hit is
#: a whole number of blocks.
BLOCK = 128
#: The indexer pools ``INDEX_KPOOL`` positions into one key row, so a block holds
#: ``BLOCK // INDEX_KPOOL`` pool rows.
INDEX_KPOOL = 4
POOLS_PER_BLOCK = BLOCK // INDEX_KPOOL
INDEX_HEAD_DIM = 16
#: The latent head width; one latent page is ``BLOCK * LATENT_WIDTH`` bf16 values, which
#: is also the page a KDA checkpoint is kept in.
LATENT_WIDTH = 16
MAX_MODEL_LEN = 1024
NUM_BLOCKS = 16
MAX_NUM_SEQS = 3
CONV_SHAPE = (3, 8)
RECURRENT_SHAPE = (1, 4, 4)
SPARSE = "model.layers.0.self_attn"
RECURRENT = "model.layers.1.attention"
#: The smallest hit that names a KDA block past the first entry: two whole blocks.
HIT = 2 * BLOCK
#: The tokens of the step that continues the hit: one more block, a prefill leg.
TOKENS = BLOCK
#: A block-table row is as wide as ``MAX_MODEL_LEN`` needs.
ROW_WIDTH = MAX_MODEL_LEN // BLOCK
DECODE_THRESHOLD = 1


def _row(*entries: int) -> list[int]:
    """A block-table row: the named entries, then the null block, ``ROW_WIDTH`` wide."""
    return [*entries] + [NULL_BLOCK_ID] * (ROW_WIDTH - len(entries))


#: Block-table rows of a request that hits ``HIT`` and prefills one more block. The
#: latent group holds a block per 128 positions. The KDA group, in vLLM's "align" mode,
#: holds the null block before the hit's last block, which is the block whose
#: checkpoint the hit restores, and a new block for the step's own end.
LATENT_ROW = _row(3, 4, 6)
KDA_ROW = _row(NULL_BLOCK_ID, 5, 7)
#: A second prefix, cached under other blocks.
OTHER_LATENT_ROW = _row(10, 11, 12)
OTHER_KDA_ROW = _row(NULL_BLOCK_ID, 13, 14)


# ---------------------------------------------------------------------------
# The world
# ---------------------------------------------------------------------------


def _text_config() -> SimpleNamespace:
    return SimpleNamespace(
        index_kpool=INDEX_KPOOL,
        index_head_dim=INDEX_HEAD_DIM,
        qk_nope_head_dim=LATENT_WIDTH,
        qk_rope_head_dim=0,
    )


def _kv_cache_config(draft_rows: int = 0):
    """One KV cache tensor shared by the latent layer and the KDA layer, as vLLM pairs them."""
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
        MambaSpec,
        MLAAttentionSpec,
    )

    latent = MLAAttentionSpec(
        block_size=BLOCK, num_kv_heads=1, head_size=LATENT_WIDTH, dtype=torch.bfloat16
    )
    recurrent = MambaSpec(
        block_size=BLOCK,
        shapes=(CONV_SHAPE, RECURRENT_SHAPE),
        dtypes=(torch.bfloat16, torch.float32),
        page_size_padded=latent.page_size_bytes,
        mamba_cache_mode="align",
        num_speculative_blocks=draft_rows,
    )
    return KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[
            KVCacheTensor(
                size=NUM_BLOCKS * latent.page_size_bytes, shared_by=[SPARSE, RECURRENT]
            )
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=[RECURRENT], kv_cache_spec=recurrent),
            KVCacheGroupSpec(layer_names=[SPARSE], kv_cache_spec=latent),
        ],
    )


def _world(*, prefix_caching: bool = True, draft_rows: int = 0):
    """A runner and its banks, with the stores ``initialize_kv_cache`` builds."""
    config = _kv_cache_config(draft_rows)
    latent_spec = config.kv_cache_groups[1].kv_cache_spec
    recurrent_spec = config.kv_cache_groups[0].kv_cache_spec
    raw = torch.zeros(config.kv_cache_tensors[0].size, dtype=torch.int8)
    latent_bank = raw.view(torch.bfloat16).view(NUM_BLOCKS, 1, BLOCK, LATENT_WIDTH)
    state_slots = MAX_NUM_SEQS + SCRATCH_SLOTS
    slot_bytes = recurrent_state_slot_bytes(recurrent_spec)
    conv, recurrent = state_bank_regions(
        torch.zeros(state_slots * slot_bytes, dtype=torch.int8),
        recurrent_spec.shapes,
        recurrent_spec.dtypes,
        slot_bytes=slot_bytes,
        checkpoints=1 + draft_rows,
    )
    banks = [
        {
            "name": SPARSE,
            "layer_index": 0,
            "family": "self_attn",
            "latent_bank": latent_bank,
            "latent_cache": latent_bank.view(NUM_BLOCKS * BLOCK, 1, LATENT_WIDTH),
            "blocks": NUM_BLOCKS,
            "block_size": BLOCK,
            "slots": NUM_BLOCKS * BLOCK,
            "head_size": LATENT_WIDTH,
        },
        {
            "name": RECURRENT,
            "layer_index": 1,
            "family": "linear_attn",
            "state_slots": state_slots,
            "conv_state": conv,
            "recurrent_state": recurrent,
        },
    ]
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.model = SimpleNamespace(
        text_config=_text_config(), glm5next_layer_banks=tuple(banks)
    )
    runner.max_model_len = MAX_MODEL_LEN
    runner.max_num_reqs = MAX_NUM_SEQS
    runner.input_batch = SimpleNamespace(req_ids=[])
    runner._glm5next_shadow_rank_cache = 0
    runner.kv_cache_config = config
    runner.kda_checkpoint_pages = (
        kda_checkpoint_page_views(config, {SPARSE: raw}) if prefix_caching else {}
    )
    runner.dsa_pool_store = (
        {
            name: torch.zeros(shape, dtype=dtype)
            for name, (shape, dtype) in dsa_pool_store_shapes(
                config, _text_config()
            ).items()
        }
        if prefix_caching
        else {}
    )
    # The live side-cache set, as the first (warmup) step allocates it.
    runner._glm5next_live_side_caches(banks)
    return runner, banks


def _entry(*, row, tokens: int, start: int, block_size: int = BLOCK) -> dict:
    """One KV-cache group's attention-metadata entry for a one-request step."""
    table = torch.tensor([row], dtype=torch.int32)
    return {
        "block_table_tensor": table,
        "full_block_table_tensor": table,
        "slot_mapping": torch.tensor(
            [row[(start + offset) // BLOCK] * BLOCK + (start + offset) % BLOCK
             for offset in range(tokens)],
            dtype=torch.int32,
        ),
        "max_query_len": int(tokens),
        "block_size": int(block_size),
        "max_blocks_per_seq": len(row),
        "decode_token_threshold": DECODE_THRESHOLD,
        "cached_seq_len": torch.tensor([start], dtype=torch.int32),
        "host_block_table": [list(row)],
        "host_num_computed_tokens": [int(start)],
        "kv_segment_size": len(row) * BLOCK,
    }


def _step(runner, banks, request_id: str, *, start: int, tokens: int = TOKENS,
          latent_row=LATENT_ROW, kda_row=KDA_ROW, block_size: int = BLOCK):
    """One request's step through the runner's step builder; returns its carriers."""
    runner.input_batch = SimpleNamespace(req_ids=[request_id])
    latent = _entry(row=latent_row, tokens=tokens, start=start, block_size=block_size)
    state = _entry(row=kda_row, tokens=tokens, start=start, block_size=block_size)
    converted = runner._glm5next_model_kwargs({
        "input_ids": torch.zeros(tokens, dtype=torch.long),
        "positions": torch.arange(tokens, dtype=torch.long) + start,
        "attn_metadata": {
            bank["name"]: latent if bank["family"] == "self_attn" else state
            for bank in banks
        },
        "sampling_positions": torch.tensor([tokens - 1], dtype=torch.long),
        "sampling_params": None,
        "spec_decode_metadata": None,
        "rank": None,
        "logit_mask": None,
    })
    return converted["layer_carriers"]


def _slot(runner, request_id: str) -> int:
    return int(runner._glm5next_request_slot_table[request_id])


def _side(runner, banks) -> dict:
    """The sparse layer's live side caches."""
    (side,) = [entry for entry in runner._glm5next_live_side_caches(banks) if entry]
    return side


def _recurrent(banks) -> dict:
    (bank,) = [bank for bank in banks if bank["family"] == "linear_attn"]
    return bank


def _layout(runner):
    return kda_checkpoint_layout(runner.kv_cache_config.kv_cache_groups[0].kv_cache_spec)


def _random_state(seed: int):
    """A checkpoint (conv, recurrent) and ``HIT // INDEX_KPOOL`` pool rows."""
    generator = torch.Generator().manual_seed(seed)
    conv = torch.randn(CONV_SHAPE, generator=generator).to(torch.bfloat16)
    recurrent = torch.randn(RECURRENT_SHAPE, generator=generator)
    pools = torch.randn((HIT // INDEX_KPOOL, INDEX_HEAD_DIM), generator=generator)
    return conv, recurrent, pools.to(torch.bfloat16)


def _store(runner, *, conv, recurrent, pools, kda_row=KDA_ROW, latent_row=LATENT_ROW,
           hit: int = HIT, record: bool = True) -> None:
    """Write a prefix's state to the stores by hand, and record it.

    The checkpoint goes to the page of the KDA block the prefix ends in, and pool row
    ``j`` of latent block index ``b`` to row ``j`` of that block's store page. With
    ``record``, the written blocks are recorded (``_glm5next_record_stored_blocks``).
    """
    block = kda_row[hit // BLOCK - 1]
    write_kda_checkpoints(
        runner.kda_checkpoint_pages[RECURRENT],
        torch.tensor([block], dtype=torch.int64),
        [conv.unsqueeze(0), recurrent.unsqueeze(0)],
        _layout(runner),
    )
    blocks = latent_row[: hit // BLOCK]
    runner.dsa_pool_store[SPARSE][blocks] = pools.view(len(blocks), POOLS_PER_BLOCK, -1)
    if record:
        runner._glm5next_record_stored_blocks([block, *blocks])


def _scheduler_output(*, new=(), running=(), resumed=()):
    """vLLM's scheduler output for one step, with only the fields block hand-out sets.

    ``new`` and ``resumed`` are ``(request id, block ids per group, computed tokens)``;
    ``running`` is ``(request id, new block ids per group or None, computed tokens)``.
    """
    from vllm.v1.core.sched.output import (
        CachedRequestData,
        NewRequestData,
        SchedulerOutput,
    )

    def groups(ids):
        return None if ids is None else tuple(list(group) for group in ids)

    cached = [*running, *resumed]
    return SchedulerOutput(
        scheduled_new_reqs=[
            NewRequestData(
                req_id=request_id, prompt_token_ids=[], mm_features=[],
                sampling_params=None, pooling_params=None, block_ids=groups(ids),
                num_computed_tokens=computed, lora_request=None,
            )
            for request_id, ids, computed in new
        ],
        scheduled_cached_reqs=CachedRequestData(
            req_ids=[request_id for request_id, _, _ in cached],
            resumed_req_ids={request_id for request_id, _, _ in resumed},
            new_token_ids=[[] for _ in cached],
            all_token_ids={},
            new_block_ids=[groups(ids) for _, ids, _ in cached],
            num_computed_tokens=[computed for _, _, computed in cached],
            num_output_tokens=[0 for _ in cached],
        ),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


class _Stop(Exception):
    """Ends ``_update_states`` once it has read the scheduled request's blocks."""


class _NewRequestLookup(dict):
    """The runner's request states; asking whether a new request is one ends the step."""

    def __contains__(self, key):
        raise _Stop


class _BatchRowLookup(dict):
    """The persistent batch's rows; looking up a cached request's row ends the step."""

    def get(self, key, default=None):
        raise _Stop


def _hand_out(runner, scheduler_output) -> None:
    """Run ``_update_states`` on a step of one scheduled request up to its first look-up.

    A new request's is its look-up in the request states; a running or resumed
    request's is its row in the persistent batch, after its blocks are read. Only the
    blocks are handed out: the request's state is not built.
    """
    (request_id,) = [
        *(new.req_id for new in scheduler_output.scheduled_new_reqs),
        *scheduler_output.scheduled_cached_reqs.req_ids,
    ]
    runner.requests = _NewRequestLookup({request_id: None})
    runner.input_batch = SimpleNamespace(req_ids=[], req_id_to_index=_BatchRowLookup())
    runner.speculative_config = None
    last_rank = SimpleNamespace(is_last_rank=True)
    with pytest.MonkeyPatch.context() as patch, pytest.raises(_Stop):
        patch.setattr(neuron_model_runner, "get_pp_group", lambda: last_rank)
        runner._update_states(scheduler_output)


def _entering(bank: dict, slot: int):
    """The slot's entering KDA state: row 0 of a speculative bank, the row otherwise."""
    conv, recurrent = bank["conv_state"], bank["recurrent_state"]
    if conv.dim() == len(CONV_SHAPE) + 2:
        return conv[slot, 0], recurrent[slot, 0]
    return conv[slot], recurrent[slot]


def _plant(runner, banks, slot: int, value: float) -> None:
    """Fill every per-slot state of ``slot`` so a later read of it is not vacuous."""
    bank = _recurrent(banks)
    bank["conv_state"][slot].fill_(value)
    bank["recurrent_state"][slot].fill_(value)
    side = _side(runner, banks)
    side["pool_cache"][slot].fill_(value)
    side["tail"][slot].fill_(value)


def _snapshot(runner, banks, slot: int) -> dict:
    bank = _recurrent(banks)
    side = _side(runner, banks)
    return {
        "conv_state": bank["conv_state"][slot].clone(),
        "recurrent_state": bank["recurrent_state"][slot].clone(),
        "pool_cache": side["pool_cache"][slot].clone(),
        "tail": side["tail"][slot].clone(),
        "pad_tail": side["pad_tail"][slot].clone(),
    }


def _assert_restored(runner, banks, slot: int, *, conv, recurrent, pools) -> None:
    entering_conv, entering_recurrent = _entering(_recurrent(banks), slot)
    side = _side(runner, banks)
    assert torch.equal(entering_conv, conv)
    assert torch.equal(entering_recurrent, recurrent)
    assert torch.equal(side["pool_cache"][slot, : HIT // INDEX_KPOOL], pools)
    assert not side["tail"][slot].any(), "the restored slot's ring is not empty"
    assert slot not in runner._glm5next_checkpoint_rows


# ---------------------------------------------------------------------------
# Cold prefix: nothing is restored
# ---------------------------------------------------------------------------


def test_a_cold_prefill_at_position_zero_reads_no_store() -> None:
    """A request at position 0 opens its slot as before: the stores are not read into it."""
    runner, banks = _world()
    conv, recurrent, pools = _random_state(seed=1)
    _store(runner, conv=conv, recurrent=recurrent, pools=pools)

    _step(runner, banks, "cold", start=0, tokens=HIT)

    slot = _slot(runner, "cold")
    entering_conv, entering_recurrent = _entering(_recurrent(banks), slot)
    assert not entering_conv.any() and not entering_recurrent.any()
    assert not _side(runner, banks)["pool_cache"][slot].any()
    assert runner._glm5next_side_cache_positions[slot] == HIT


def test_a_step_at_the_slots_own_cursor_is_not_restored() -> None:
    """A second prefill segment at the slot's own cursor keeps the slot's own state."""
    runner, banks = _world()
    _step(runner, banks, "segmented", start=0, tokens=HIT)
    slot = _slot(runner, "segmented")
    _plant(runner, banks, slot, 2.0)
    conv, recurrent, pools = _random_state(seed=2)
    _store(runner, conv=conv, recurrent=recurrent, pools=pools)
    before = _snapshot(runner, banks, slot)

    _step(runner, banks, "segmented", start=HIT)

    after = _snapshot(runner, banks, slot)
    assert all(torch.equal(after[key], before[key]) for key in before)
    assert runner._glm5next_side_cache_positions[slot] == HIT + TOKENS


# ---------------------------------------------------------------------------
# Reused prefix: a hit on a fresh slot is restored and armed
# ---------------------------------------------------------------------------


def test_a_hit_on_a_fresh_slot_is_restored_and_continued() -> None:
    """A hit of two blocks with stored state: the slot gets it and the step is served."""
    runner, banks = _world()
    conv, recurrent, pools = _random_state(seed=3)
    _store(runner, conv=conv, recurrent=recurrent, pools=pools)

    carriers = _step(runner, banks, "hit", start=HIT)

    slot = _slot(runner, "hit")
    _assert_restored(runner, banks, slot, conv=conv, recurrent=recurrent, pools=pools)
    # The rows past the hit are the fresh slot's zeros.
    assert not _side(runner, banks)["pool_cache"][slot, HIT // INDEX_KPOOL :].any()
    assert runner._glm5next_side_cache_positions[slot] == HIT + TOKENS
    # The step's recurrent carrier enters with the restored state.
    (linear,) = [
        carrier for carrier, bank in zip(carriers, banks)
        if bank["family"] == "linear_attn"
    ]
    assert torch.equal(linear["conv_state"][0], conv)
    assert torch.equal(linear["recurrent_state"][0], recurrent)


def test_a_restored_slot_equals_the_slot_a_cold_prefill_leaves_at_the_same_length() -> None:
    """A cold prefill of ``HIT`` tokens, stored; a hit of the same prefix on another slot.

    After each request's next step from ``HIT`` the two slots hold the same KDA rows,
    pool rows ``0 .. HIT // INDEX_KPOOL - 1``, ring, cursor and resume row.
    """
    runner, banks = _world()
    _step(runner, banks, "cold", start=0, tokens=HIT)
    cold = _slot(runner, "cold")
    # What the prefill's forward leaves in the slot: the KDA state after HIT tokens and
    # the pool rows of every whole pool; HIT completes every pool, so the ring is empty.
    conv, recurrent, pools = _random_state(seed=4)
    bank = _recurrent(banks)
    bank["conv_state"][cold].copy_(conv)
    bank["recurrent_state"][cold].copy_(recurrent)
    _side(runner, banks)["pool_cache"][cold, : HIT // INDEX_KPOOL].copy_(pools)
    _store(runner, conv=bank["conv_state"][cold], recurrent=bank["recurrent_state"][cold],
           pools=_side(runner, banks)["pool_cache"][cold, : HIT // INDEX_KPOOL])
    _step(runner, banks, "cold", start=HIT)

    _step(runner, banks, "hit", start=HIT)

    hit = _slot(runner, "hit")
    assert hit != cold
    side = _side(runner, banks)
    rows = HIT // INDEX_KPOOL
    assert torch.equal(bank["conv_state"][hit], bank["conv_state"][cold])
    assert torch.equal(bank["recurrent_state"][hit], bank["recurrent_state"][cold])
    assert torch.equal(side["pool_cache"][hit, :rows], side["pool_cache"][cold, :rows])
    assert torch.equal(side["tail"][hit], side["tail"][cold])
    assert (
        runner._glm5next_side_cache_positions[hit]
        == runner._glm5next_side_cache_positions[cold]
        == HIT + TOKENS
    )
    assert hit not in runner._glm5next_checkpoint_rows
    assert cold not in runner._glm5next_checkpoint_rows


# ---------------------------------------------------------------------------
# Slot reuse: a restore writes only its own slot
# ---------------------------------------------------------------------------


def test_a_restore_writes_only_its_own_slot() -> None:
    """Every other slot's KDA rows, pool_cache, ring and cursor stay bitwise unchanged."""
    runner, banks = _world()
    _step(runner, banks, "live", start=0, tokens=HIT,
          latent_row=OTHER_LATENT_ROW, kda_row=OTHER_KDA_ROW)
    live = _slot(runner, "live")
    _plant(runner, banks, live, 7.0)
    runner._glm5next_checkpoint_rows[live] = 1
    idle = [slot for slot in range(MAX_NUM_SEQS + SCRATCH_SLOTS) if slot != live]
    for slot in idle:
        _plant(runner, banks, slot, 5.0)
    before = {slot: _snapshot(runner, banks, slot) for slot in range(MAX_NUM_SEQS + SCRATCH_SLOTS)}
    conv, recurrent, pools = _random_state(seed=5)
    _store(runner, conv=conv, recurrent=recurrent, pools=pools)

    _step(runner, banks, "hit", start=HIT)

    hit = _slot(runner, "hit")
    _assert_restored(runner, banks, hit, conv=conv, recurrent=recurrent, pools=pools)
    for slot, old in before.items():
        if slot == hit:
            continue
        now = _snapshot(runner, banks, slot)
        assert all(torch.equal(now[key], old[key]) for key in old), slot
    assert runner._glm5next_side_cache_positions[live] == HIT
    assert runner._glm5next_checkpoint_rows[live] == 1


@pytest.mark.parametrize("draft_rows", [0, 3], ids=["plain", "speculative"])
def test_a_restore_writes_only_the_slots_entering_kda_row(draft_rows) -> None:
    """On a speculative server the slot's draft rows 1.. stay as they were."""
    runner, banks = _world(draft_rows=draft_rows)
    bank = _recurrent(banks)
    for slot in range(MAX_NUM_SEQS + SCRATCH_SLOTS):
        bank["conv_state"][slot].fill_(3.0)
        bank["recurrent_state"][slot].fill_(3.0)
    before = (bank["conv_state"].clone(), bank["recurrent_state"].clone())
    conv, recurrent, pools = _random_state(seed=6)
    _store(runner, conv=conv, recurrent=recurrent, pools=pools)

    _step(runner, banks, "hit", start=HIT)

    hit = _slot(runner, "hit")
    entering_conv, entering_recurrent = _entering(bank, hit)
    assert torch.equal(entering_conv, conv)
    assert torch.equal(entering_recurrent, recurrent)
    for now, old in zip((bank["conv_state"], bank["recurrent_state"]), before):
        others = [slot for slot in range(int(now.shape[0])) if slot != hit]
        assert torch.equal(now[others], old[others])
        if draft_rows:
            assert torch.equal(now[hit, 1:], old[hit, 1:])


def test_a_restored_slot_resumes_from_its_entering_row_whatever_it_held() -> None:
    """A held slot can lose its cursor and keep a resume row: an opening the carrier
    builder refuses empties the ring and drops the cursor, and the resume row the last
    verify step recorded stays. A hit restored on that slot, after the pending step is
    drained, resumes from row 0, the row the checkpoint went to."""
    runner, banks = _world()
    _step(runner, banks, "r", start=0, tokens=HIT)
    slot = _slot(runner, "r")
    # The state hook's record after a verify step that kept two rows.
    runner._glm5next_checkpoint_rows[slot] = 1
    with pytest.raises(ValueError) as refused:
        _step(runner, banks, "r", start=0, tokens=HIT, block_size=2 * BLOCK)
    assert "sequence cursor" not in str(refused.value)
    assert slot not in runner._glm5next_side_cache_positions
    assert runner._glm5next_checkpoint_rows[slot] == 1
    conv, recurrent, pools = _random_state(seed=15)
    _store(runner, conv=conv, recurrent=recurrent, pools=pools)
    # The slot is held, so no hand-out drains the pending step in this one; the restore
    # drains it before it writes the slot.
    drained = []

    class _Pending:
        def get_output(self):
            side = _side(runner, banks)
            drained.append(torch.equal(side["pool_cache"][slot, : HIT // INDEX_KPOOL], pools))

    runner.async_execution_buffer = {"async_output": _Pending()}

    _step(runner, banks, "r", start=HIT)

    assert drained == [False], "the pending step was not drained once before the restore"
    _assert_restored(runner, banks, slot, conv=conv, recurrent=recurrent, pools=pools)


# ---------------------------------------------------------------------------
# Cancellation: a cancelled request's cached blocks restore into its freed slot
# ---------------------------------------------------------------------------


def test_a_cancelled_requests_blocks_restore_into_its_freed_slot_without_its_leftovers() -> None:
    """The prefix outlives the request: its blocks stay hits after it is cancelled.

    The next request takes the freed slot and gets the stored prefix, and none of
    what the cancelled request left in the slot past it.
    """
    runner, banks = _world(prefix_caching=True)
    _step(runner, banks, "cancelled", start=0, tokens=HIT)
    slot = _slot(runner, "cancelled")
    conv, recurrent, pools = _random_state(seed=7)
    _store(runner, conv=conv, recurrent=recurrent, pools=pools)
    # The cancelled request ran on past the prefix before it was cancelled.
    _plant(runner, banks, slot, 9.0)
    runner._glm5next_checkpoint_rows[slot] = 2
    runner._glm5next_note_finished_requests(["cancelled"])

    _step(runner, banks, "next", start=HIT)

    assert _slot(runner, "next") == slot
    _assert_restored(runner, banks, slot, conv=conv, recurrent=recurrent, pools=pools)
    assert not _side(runner, banks)["pool_cache"][slot, HIT // INDEX_KPOOL :].any()
    assert runner._glm5next_side_cache_positions[slot] == HIT + TOKENS


# ---------------------------------------------------------------------------
# Eviction: a hit restores the blocks its table names when it arrives
# ---------------------------------------------------------------------------


def test_a_hit_restores_the_blocks_its_table_names_when_it_arrives() -> None:
    """The restore is addressed by block id, not by request or by the last write.

    Two prefixes are cached under different blocks. A hit on the second gets the
    second's state. Once vLLM evicts the second's blocks and a new prefill rewrites
    them, a hit that names them gets what the new prefill wrote.
    """
    runner, banks = _world()
    first = _random_state(seed=8)
    second = _random_state(seed=9)
    rewritten = _random_state(seed=10)
    _store(runner, conv=first[0], recurrent=first[1], pools=first[2])
    _store(runner, conv=second[0], recurrent=second[1], pools=second[2],
           kda_row=OTHER_KDA_ROW, latent_row=OTHER_LATENT_ROW)

    _step(runner, banks, "second", start=HIT,
          latent_row=OTHER_LATENT_ROW, kda_row=OTHER_KDA_ROW)
    _assert_restored(runner, banks, _slot(runner, "second"),
                     conv=second[0], recurrent=second[1], pools=second[2])

    # vLLM evicts the second's blocks and hands them to a new request from position 0:
    # until its prefill writes them, a hit on them has no state to restore.
    _hand_out(runner, _scheduler_output(
        new=[("rewriter", (OTHER_KDA_ROW[:3], OTHER_LATENT_ROW[:3]), 0)]))
    with pytest.raises(ValueError, match="no prefill has recorded"):
        _step(runner, banks, "before-rewrite", start=HIT,
              latent_row=OTHER_LATENT_ROW, kda_row=OTHER_KDA_ROW)
    _store(runner, conv=rewritten[0], recurrent=rewritten[1], pools=rewritten[2],
           kda_row=OTHER_KDA_ROW, latent_row=OTHER_LATENT_ROW)
    _step(runner, banks, "after-eviction", start=HIT,
          latent_row=OTHER_LATENT_ROW, kda_row=OTHER_KDA_ROW)
    _assert_restored(runner, banks, _slot(runner, "after-eviction"),
                     conv=rewritten[0], recurrent=rewritten[1], pools=rewritten[2])
    # The first prefix's blocks were never read or written by either restore.
    (stored_conv, stored_recurrent) = read_kda_checkpoints(
        runner.kda_checkpoint_pages[RECURRENT],
        torch.tensor([KDA_ROW[HIT // BLOCK - 1]], dtype=torch.int64),
        _layout(runner),
    )
    assert torch.equal(stored_conv[0], first[0])
    assert torch.equal(stored_recurrent[0], first[1])


# ---------------------------------------------------------------------------
# A hit restores only from blocks a prefill recorded writing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("unrecorded", "layer"),
    [("every block", SPARSE), ("the kda block", RECURRENT), ("a latent block", SPARSE)],
)
def test_a_hit_on_a_block_no_prefill_recorded_is_refused_by_name_and_writes_nothing(
    unrecorded, layer
) -> None:
    """The stores hold rows for every block; only the recorded ones may be restored.

    With nothing recorded, which is the runner's state until a prefill records the
    blocks it writes, every hit is refused.
    """
    runner, banks = _world()
    conv, recurrent, pools = _random_state(seed=16)
    _store(runner, conv=conv, recurrent=recurrent, pools=pools, record=False)
    kda_block, latent = KDA_ROW[HIT // BLOCK - 1], LATENT_ROW[: HIT // BLOCK]
    recorded = {
        "every block": [],
        "the kda block": list(latent),
        "a latent block": [kda_block, latent[0]],
    }[unrecorded]
    runner._glm5next_record_stored_blocks(recorded)

    with pytest.raises(ValueError, match=rf"{layer}.*no prefill has recorded"):
        _step(runner, banks, "hit", start=HIT)

    slot = _slot(runner, "hit")
    assert slot not in runner._glm5next_side_cache_positions
    assert not _entering(_recurrent(banks), slot)[0].any()
    assert not _side(runner, banks)["pool_cache"][slot].any()


def test_the_null_block_is_refused_as_a_stored_block_by_name() -> None:
    """The null block holds no state, so no prefill write records it; nothing is recorded."""
    runner, _ = _world()

    with pytest.raises(ValueError, match="null block"):
        runner._glm5next_record_stored_blocks([3, NULL_BLOCK_ID])

    assert runner._glm5next_stored_blocks == set()


#: Every block the hand-out test's rows name, and block 1, which none of them names.
RECORDED = {5, 7, 3, 4, 6, 9, 15, 13, 14, 10, 11, 12, 1}


@pytest.mark.parametrize(
    ("scheduled", "fresh"),
    [
        # KDA entries 0-1 and latent entries 0-1 are the hit; entry 2 of each is fresh.
        pytest.param(
            {"new": [("hit", (KDA_ROW[:3], LATENT_ROW[:3]), HIT)]},
            {KDA_ROW[2], LATENT_ROW[2]},
            id="new request past its hit",
        ),
        pytest.param(
            {"resumed": [("resumed", (OTHER_KDA_ROW[:3], OTHER_LATENT_ROW[:3]), HIT)]},
            {OTHER_KDA_ROW[2], OTHER_LATENT_ROW[2]},
            id="resumed request past its hit",
        ),
        # A running request's computed tokens are its own, not a hit on these blocks.
        pytest.param(
            {"running": [("running", ([9], [15]), 3 * BLOCK)]},
            {9, 15},
            id="running request's new blocks",
        ),
        pytest.param(
            {"running": [("idle", None, 2 * BLOCK)]},
            set(),
            id="running request with no new blocks",
        ),
    ],
)
def test_the_blocks_vllm_hands_out_again_lose_their_record(scheduled, fresh) -> None:
    """Every block vLLM hands a request past its cache hit is fresh, and loses its record.

    A new or resumed request's first ``computed // block_size`` blocks of a group are its
    cache hit and keep theirs; a running request's new blocks are all fresh.
    """
    runner, _ = _world()
    runner._glm5next_record_stored_blocks(RECORDED)

    _hand_out(runner, _scheduler_output(**scheduled))

    assert runner._glm5next_stored_blocks == RECORDED - fresh


# ---------------------------------------------------------------------------
# Refusals that stay
# ---------------------------------------------------------------------------


def test_with_prefix_caching_off_a_hit_is_refused_as_before() -> None:
    """No stores (prefix caching off): no restore runs and the position check refuses."""
    runner, banks = _world(prefix_caching=False)
    assert runner.kda_checkpoint_pages == {} and runner.dsa_pool_store == {}

    with pytest.raises(ValueError, match="holds no sequence cursor"):
        _step(runner, banks, "hit", start=HIT)

    slot = _slot(runner, "hit")
    assert slot not in runner._glm5next_side_cache_positions
    entering_conv, entering_recurrent = _entering(_recurrent(banks), slot)
    assert not entering_conv.any() and not entering_recurrent.any()


def test_a_cached_length_that_is_not_a_whole_number_of_blocks_is_refused_as_before() -> None:
    """vLLM's hits are whole blocks, so any other length on a fresh slot is not one."""
    runner, banks = _world()
    conv, recurrent, pools = _random_state(seed=11)
    _store(runner, conv=conv, recurrent=recurrent, pools=pools)

    with pytest.raises(ValueError, match="holds no sequence cursor"):
        _step(runner, banks, "unaligned", start=HIT - INDEX_KPOOL)

    slot = _slot(runner, "unaligned")
    assert not _entering(_recurrent(banks), slot)[0].any()
    assert not _side(runner, banks)["pool_cache"][slot].any()


@pytest.mark.parametrize("layer", [RECURRENT, SPARSE], ids=["kda", "latent"])
def test_a_hit_that_names_the_null_block_is_refused_by_name_and_writes_nothing(layer) -> None:
    """The null block holds no stored state, so a hit naming it has none to restore."""
    runner, banks = _world()
    conv, recurrent, pools = _random_state(seed=12)
    _store(runner, conv=conv, recurrent=recurrent, pools=pools)
    rows = (
        {"kda_row": _row(NULL_BLOCK_ID, NULL_BLOCK_ID, 7)}
        if layer == RECURRENT
        else {"latent_row": _row(3, NULL_BLOCK_ID, 6)}
    )

    with pytest.raises(ValueError, match=rf"{layer}.*null block"):
        _step(runner, banks, "hit", start=HIT, **rows)

    slot = _slot(runner, "hit")
    assert slot not in runner._glm5next_side_cache_positions
    assert not _entering(_recurrent(banks), slot)[0].any()
    assert not _side(runner, banks)["pool_cache"][slot].any()


@pytest.mark.parametrize("missing", ["checkpoint pages", "pool store"])
def test_a_layer_without_its_store_is_refused_by_name_and_writes_nothing(missing) -> None:
    """With prefix caching on every recurrent and sparse layer has its store."""
    runner, banks = _world()
    conv, recurrent, pools = _random_state(seed=13)
    _store(runner, conv=conv, recurrent=recurrent, pools=pools)
    if missing == "checkpoint pages":
        del runner.kda_checkpoint_pages[RECURRENT]
        name = RECURRENT
    else:
        del runner.dsa_pool_store[SPARSE]
        name = SPARSE

    with pytest.raises(ValueError, match=rf"{name}.*{missing}"):
        _step(runner, banks, "hit", start=HIT)

    slot = _slot(runner, "hit")
    assert slot not in runner._glm5next_side_cache_positions
    assert not _entering(_recurrent(banks), slot)[0].any()
    assert not _side(runner, banks)["pool_cache"][slot].any()


@pytest.mark.parametrize("form", ["dtype", "pools per block", "width"])
def test_a_pool_store_of_another_form_than_the_pool_cache_is_refused_by_name(form) -> None:
    """The rows are copied as they are: a store whose dtype or row width is not the
    pool_cache's, or whose pages do not hold one block's pools, is refused, not cast
    or cut."""
    runner, banks = _world()
    conv, recurrent, pools = _random_state(seed=14)
    _store(runner, conv=conv, recurrent=recurrent, pools=pools)
    store = runner.dsa_pool_store[SPARSE]
    runner.dsa_pool_store[SPARSE] = {
        "dtype": lambda: store.float(),
        "pools per block": lambda: store[:, : POOLS_PER_BLOCK // 2].contiguous(),
        "width": lambda: store[..., : INDEX_HEAD_DIM // 2].contiguous(),
    }[form]()

    with pytest.raises(ValueError, match=rf"{SPARSE}.*pool store.*does not hold"):
        _step(runner, banks, "hit", start=HIT)

    slot = _slot(runner, "hit")
    assert slot not in runner._glm5next_side_cache_positions
    assert not _entering(_recurrent(banks), slot)[0].any()


def test_a_hit_under_decode_context_parallelism_is_refused_by_name() -> None:
    """Under DCP a table entry covers more than one block of positions; no restore runs."""
    runner, banks = _world()
    _step(runner, banks, "live", start=0, tokens=HIT)
    runner._dcp_size = 2
    geometries = [
        {"page_size": BLOCK, "request_block_ids": [row[:3]]}
        for row in (LATENT_ROW, KDA_ROW)
    ]

    with pytest.raises(ValueError, match="decode_context_parallel_size=2"):
        runner._glm5next_restore_prefix(
            banks,
            runner._glm5next_live_side_caches(banks),
            geometries,
            request=0,
            request_id="hit",
            slot=1,
            start_position=HIT,
        )
    assert 1 not in runner._glm5next_side_cache_positions
