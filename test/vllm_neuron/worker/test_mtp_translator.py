# SPDX-License-Identifier: Apache-2.0
"""The GLM-5.3-Flash translator on a speculative verify step: ``T = 1 + k`` rows per request.

``NeuronModelRunner._glm5next_model_kwargs`` / ``_glm5next_layer_carriers`` build one
carrier per layer from the host arrays. On the decode leg they used to insist on one
token per request; a verify step carries ``T`` rows per request (row ``b * T + t`` is
request ``b``'s token ``t``, at position ``start_b + t``), padded to the batch bucket
by whole requests. Covered here, on the tiny root with the stack never run (the
kernels that consume ``T`` rows are another worker's): the one-request and the
request-major two-request layouts of every per-row and per-request operand, the
padding request's masks and null-block slots, the refusals (a mixed batch, a width
that does not divide into whole requests), the step record the per-step state hook
reads (``_glm5next_step_record``), the ``spec_decode_metadata`` and ``draft_k``
keywords handed to the root, the recurrent layers' checkpoint operands under method
"mtp" (``state_checkpoints``, ``checkpoint_rows``: the next step starts from the row
the host learned was accepted), and ``_glm5next_host_only_metadata`` staying on under
the mtp spec config.

Every expectation is derived from the world's own tables and positions.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/worker/test_mtp_translator.py
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_neuron import envs
from vllm_neuron.model.glm5_next import mtp as head_module
from vllm_neuron.vllm.worker.glm5next_state_banks import bank_form
from vllm_neuron.vllm.worker.neuron_model_runner import NULL_BLOCK_ID, NeuronModelRunner

from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_batch_decode as batch
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_batch_kda as kda
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_forward as tiny
from test.vllm_neuron.model.glm5_next import test_shadow_draft_e2e as shadow

pytestmark = [pytest.mark.forked]

PAGE = tiny.MLA_PAGE_SIZE
#: Draft length under test and the verify width it makes.
K = 3
T = 1 + K
#: Each world's requests are prefilled to these lengths: one past a page boundary,
#: one short of the next, so the two requests stand in different pages.
PROMPTS = [PAGE + 3, 2 * PAGE + 2]
#: The batch-decode file's selecting regime: the indexer's top-k wants this much room.
MAX_MODEL_LEN = tiny.STACK_TOKENS + 8
WINDOW_BLOCKS = -(-MAX_MODEL_LEN // PAGE)


def _world(batch_size: int, *, slots: int | None = None):
    """The batch-decode file's sparse (DSA) world: ``batch_size`` prefilled requests."""
    return batch._world(
        batch_size, max_model_len=MAX_MODEL_LEN, prompts=PROMPTS[:batch_size],
        window_blocks=WINDOW_BLOCKS, slots=slots,
    )


def _verify_metadata(world, rows: list[int], *, cached: list[int], tokens: int, width: int) -> dict:
    """The batch-decode file's metadata, read as a verify step of ``width`` rows per request."""
    entry = next(iter(batch._metadata(world, rows, cached=cached, tokens=tokens).values()))
    entry = dict(entry, max_query_len=width, decode_token_threshold=width)
    return {bank["name"]: entry for bank in world.root.glm5next_layer_banks}


def _convert(
    world, rows: list[int], *, cached: list[int], tokens: int, real: list[int], width: int = T,
    spec_decode_metadata=None,
):
    runner = world.runner
    runner.input_batch.req_ids = [world.req_ids[r] for r in rows]
    runner._glm5next_request_tokens = np.array(real, np.int32)
    return runner._glm5next_model_kwargs({
        "input_ids": torch.zeros(tokens, dtype=torch.int32),
        "positions": None,
        "attn_metadata": _verify_metadata(world, rows, cached=cached, tokens=tokens, width=width),
        "sampling_positions": torch.arange(tokens, dtype=torch.long),
        "sampling_params": None,
        "spec_decode_metadata": spec_decode_metadata,
        "rank": None,
        "logit_mask": None,
    })


def _sparse(world, carriers) -> list[dict]:
    banks = world.root.glm5next_layer_banks
    sparse = [c for bank, c in zip(banks, carriers) if bank["family"] == "self_attn"]
    assert sparse and len(sparse) == len(carriers), "the tiny root's stack is sparse throughout"
    return sparse


def _kda_convert(
    world, rows: list[int], *, cached: list[int], tokens: int, real: list[int], width: int = T,
) -> list[dict]:
    """The KDA batch file's recurrent-only world, read as a decode step of ``width`` rows per
    request (a verify step by default); returns the carriers."""
    runner = world.runner
    runner.input_batch.req_ids = [world.req_ids[r] for r in rows]
    runner._glm5next_request_tokens = np.array(real, np.int32)
    metadata = kda._metadata(world.banks, rows=tokens, max_query_len=width, cached=cached)
    for entry in metadata.values():
        entry["decode_token_threshold"] = T
    converted = runner._glm5next_model_kwargs({
        "input_ids": torch.zeros(tokens, dtype=torch.long),
        "attn_metadata": metadata,
        "sampling_positions": torch.arange(tokens, dtype=torch.long),
    })
    return converted["layer_carriers"]


def _slot(world, request: int, position: int) -> int:
    return world.tables[request][position // PAGE] * PAGE + position % PAGE


def _per_request(carrier: dict, key: str, requests: int) -> None:
    """``key`` is handed per request: a tuple of ``requests`` views, or (bank form) the
    whole bank beside a ``[requests]`` slot tensor."""
    if bank_form(requests, is_prefill=False):
        assert torch.is_tensor(carrier[key]) and carrier["state_slots"].tolist().__len__() == requests
    else:
        assert isinstance(carrier[key], tuple) and len(carrier[key]) == requests


def test_a_one_request_verify_step_carries_t_rows_of_one_sequence():
    world = _world(1)
    start = world.lengths[0]
    converted = _convert(world, [0], cached=[start], tokens=T, real=[T])
    for carrier in _sparse(world, converted["layer_carriers"]):
        assert carrier["seq_lens"].tolist() == [start + 1 + t for t in range(T)]
        assert carrier["latent_slots"].tolist() == [_slot(world, 0, start + t) for t in range(T)]
        assert int(carrier["position"]) == start and int(carrier["start_position"]) == start
        assert torch.is_tensor(carrier["tail"]), "one request: its own ring view"
        assert tuple(carrier["block_table_row"].shape) == (WINDOW_BLOCKS, 1)
    slot = world.runner._glm5next_step_record["slots"][0]
    assert world.runner._glm5next_step_record == {
        "slots": [slot], "starts": [start], "counts": [T], "is_prefill": False, "width": T,
    }
    # The cursor stands past every verify row; the per-step hook pulls it back to the
    # accepted count once the host knows it.
    assert world.runner._glm5next_side_cache_positions[slot] == start + T


def test_a_two_request_verify_step_is_request_major():
    world = _world(2)
    starts = list(world.lengths)
    converted = _convert(world, [0, 1], cached=starts, tokens=2 * T, real=[T, T])
    for carrier in _sparse(world, converted["layer_carriers"]):
        assert carrier["seq_lens"].tolist() == [s + 1 + t for s in starts for t in range(T)]
        assert carrier["latent_slots"].tolist() == [
            _slot(world, b, starts[b] + t) for b in range(2) for t in range(T)
        ]
        assert carrier["position"].tolist() == starts
        assert carrier["start_position"].tolist() == starts
        assert tuple(carrier["block_table_row"].shape) == (WINDOW_BLOCKS, 2)
        _per_request(carrier, "pool_cache", 2)
    assert world.runner._glm5next_step_record["starts"] == starts
    assert world.runner._glm5next_step_record["counts"] == [T, T]


def test_a_bucket_padded_verify_step_masks_the_padding_request():
    bucket = 3
    world = _world(2, slots=bucket)
    starts = list(world.lengths)
    converted = _convert(world, [0, 1], cached=starts, tokens=bucket * T, real=[T, T])
    for carrier in _sparse(world, converted["layer_carriers"]):
        assert len(carrier["seq_lens"]) == bucket * T
        # The padding request's rows write the null block, which no request holds.
        assert carrier["latent_slots"].tolist()[2 * T:] == [NULL_BLOCK_ID * PAGE] * T
        assert carrier["position"].tolist() == starts + [0]
        _per_request(carrier, "tail", bucket)
    # Only the real requests are recorded for the hook.
    assert world.runner._glm5next_step_record["counts"] == [T, T]


def test_a_verify_step_hands_the_recurrent_layers_each_requests_rows():
    world = kda._world(2)
    starts = list(kda.PROMPTS[:2])
    for carrier in _kda_convert(world, [0, 1], cached=starts, tokens=2 * T, real=[T, T]):
        assert carrier["start_position"].tolist() == starts
        assert carrier["real_tokens"].tolist() == [[T], [T]]
        assert tuple(carrier["row_mask"].shape) == (2, T, 1)
        assert carrier["row_mask"].reshape(-1).tolist() == [1.0] * (2 * T)
        _per_request(carrier, "conv_state", 2)
    assert world.runner._glm5next_step_record["starts"] == starts


def test_a_bucket_padded_verify_step_masks_the_recurrent_padding_rows():
    bucket = 3
    world = kda._world(2)
    starts = list(kda.PROMPTS[:2])
    for carrier in _kda_convert(world, [0, 1], cached=starts, tokens=bucket * T, real=[T, T]):
        assert carrier["real_tokens"].tolist() == [[T], [T], [0]]
        assert tuple(carrier["row_mask"].shape) == (bucket, T, 1)
        assert carrier["row_mask"][2].reshape(-1).tolist() == [0.0] * T
        assert carrier["row_mask"][:2].reshape(-1).tolist() == [1.0] * (2 * T)
        # The padding request keeps its idle slot's state at position 1, as before.
        assert carrier["start_position"].tolist() == starts + [1]


def test_a_verify_step_hands_the_root_its_spec_decode_metadata():
    """The root runs the rejection sampler, so the step's metadata crosses under its own
    name; a one-row decode (no drafts scheduled) carries none and the key stays absent."""
    world = _world(1)
    start = world.lengths[0]
    metadata = object()
    converted = _convert(world, [0], cached=[start], tokens=T, real=[T], spec_decode_metadata=metadata)
    assert converted["spec_decode_metadata"] is metadata
    # A fresh world: the verify step above advanced slot 0's ring past ``start``.
    world = _world(1)
    plain = _convert(world, [0], cached=[start], tokens=1, real=[1], width=1)
    assert "spec_decode_metadata" not in plain


def test_a_one_row_decode_and_a_prefill_are_recorded_for_the_hook_too():
    """The hook zeroes the accepted count on these steps, so they are recorded as well."""
    world = _world(1)
    start = world.lengths[0]
    _convert(world, [0], cached=[start], tokens=1, real=[1], width=1)
    record = world.runner._glm5next_step_record
    assert (record["is_prefill"], record["width"], record["counts"]) == (False, 1, [1])
    plain = shadow._world()
    _prefill_conversion(plain)
    record = plain.runner._glm5next_step_record
    assert (record["is_prefill"], record["counts"]) == (True, [len(shadow.PROMPT)])


def _mtp_server(runner) -> None:
    runner.is_mtp_spec = True
    runner.drafter = SimpleNamespace(num_speculative_tokens=K)


def test_the_recurrent_carriers_start_the_next_step_from_the_accepted_row():
    """Under method "mtp" every decode step's recurrent carrier names the checkpoint
    geometry (``state_checkpoints = T``) and each request's accepted draft count from
    the verify step before it (0 after a prefill), which the per-step state hook
    learned from the sampled ids; a bucket padding row starts from row 0."""
    world = kda._world(2)
    runner = world.runner
    _mtp_server(runner)
    starts = list(kda.PROMPTS[:2])
    for carrier in _kda_convert(world, [0, 1], cached=starts, tokens=2 * T, real=[T, T]):
        assert carrier["state_checkpoints"] == T
        assert carrier["checkpoint_rows"].tolist() == [0, 0], "after a prefill"
    # The host learns the kept ids: request 0 keeps three rows, request 1 one.
    runner._update_states_after_model_execute([[5, 6, 7], [9]], None)
    kept = [3, 1]
    nexts = [s + n for s, n in zip(starts, kept)]
    for carrier in _kda_convert(world, [0, 1], cached=nexts, tokens=2 * T, real=[T, T]):
        assert carrier["start_position"].tolist() == nexts
        assert carrier["checkpoint_rows"].tolist() == [n - 1 for n in kept]
        assert carrier["checkpoint_rows"].dtype == torch.int32
    runner._update_states_after_model_execute([[1, 2], [3, 4, 5, 6]], None)
    kept = [2, T]
    nexts = [s + n for s, n in zip(nexts, kept)]
    bucket = 3
    for carrier in _kda_convert(world, [0, 1], cached=nexts, tokens=bucket * T, real=[T, T]):
        assert carrier["checkpoint_rows"].tolist() == [n - 1 for n in kept] + [0]
    # A one-row decode on the same server (no drafts scheduled near max_model_len) keeps
    # the checkpoint geometry: the banks have not changed shape.
    runner._update_states_after_model_execute([[1], [2, 3]], None)
    kept = [1, 2]
    nexts = [s + n for s, n in zip(nexts, kept)]
    for carrier in _kda_convert(world, [0, 1], cached=nexts, tokens=2, real=[1, 1], width=1):
        assert carrier["state_checkpoints"] == T
        assert carrier["checkpoint_rows"].tolist() == [n - 1 for n in kept]
    runner._update_states_after_model_execute([[4], [5]], None)
    nexts = [s + 1 for s in nexts]
    for carrier in _kda_convert(world, [0, 1], cached=nexts, tokens=2, real=[1, 1], width=1):
        assert carrier["checkpoint_rows"].tolist() == [0, 0], "a one-row step keeps its row 0"


def test_without_the_mtp_spec_the_recurrent_carriers_carry_no_checkpoint_operands():
    world = kda._world(1)
    for carrier in _kda_convert(world, [0], cached=[kda.PROMPTS[0]], tokens=1, real=[1], width=1):
        assert "state_checkpoints" not in carrier and "checkpoint_rows" not in carrier


def test_a_padding_row_resumes_from_its_slots_own_row():
    """A padding row names a slot no scheduled request holds; when that slot is owned by a
    request this step does not schedule, its row is the owner's live one, so the layer
    rewrites the slot with its own bytes (worker-57's checkpoint hazard)."""
    runner = SimpleNamespace(
        is_mtp_spec=True, drafter=SimpleNamespace(num_speculative_tokens=K),
        _glm5next_checkpoint_rows={1: 2, 5: 3},
    )
    checkpoints = NeuronModelRunner._glm5next_recurrent_checkpoints(runner, [1, 7, 5], is_prefill=False)
    assert checkpoints == (T, [2, 0, 3])
    assert NeuronModelRunner._glm5next_recurrent_checkpoints(runner, [1], is_prefill=True) is None
    runner.is_mtp_spec = False
    assert NeuronModelRunner._glm5next_recurrent_checkpoints(runner, [1], is_prefill=False) is None


def test_a_mixed_batch_is_refused_by_name():
    """One request with k drafts beside one with none has no one width; refused rather
    than laid out until the prefill leg drafts too."""
    world = _world(2)
    starts = list(world.lengths)
    with pytest.raises(ValueError, match=r"\[4, 1\]"):
        _convert(world, [0, 1], cached=starts, tokens=T + 1, real=[T, 1])


def test_rows_that_do_not_divide_into_whole_requests_are_refused():
    world = _world(2)
    banks = world.root.glm5next_layer_banks
    side = world.runner._glm5next_live_side_caches(banks)
    geometries = []
    for bank in banks:
        geometries.append({
            "block_ids": world.tables[0][:1], "request_block_ids": [world.tables[0][:1], world.tables[1][:1]],
            "state_slot": 0, "state_slots": [0, 1], "page_size": PAGE, "window_blocks": WINDOW_BLOCKS,
        })
    with pytest.raises(ValueError, match="5 token"):
        NeuronModelRunner._glm5next_layer_carriers(
            banks, side, geometries=geometries, is_prefill=False, tokens=5,
            start_position=1, softmax_scale=1.0, max_seq_len=MAX_MODEL_LEN,
            index_kpool=int(world.root.text_config.index_kpool), requests=2,
            request_starts=[1, 1], real_tokens=5, request_real_tokens=[3, 2],
        )


def _prefill_conversion(world) -> dict:
    """The shadow e2e world's prompt, converted (not run) as its opening prefill."""
    runner = world.runner
    runner._glm5next_request_tokens = None
    return runner._glm5next_model_kwargs({
        "input_ids": torch.tensor(shadow.PROMPT),
        "positions": None,
        "attn_metadata": batch._metadata(world, [0], cached=[0], tokens=len(shadow.PROMPT)),
        "sampling_positions": torch.tensor([len(shadow.PROMPT) - 1], dtype=torch.long),
        "sampling_params": torch.tensor([shadow.GREEDY_ROW], dtype=torch.float32),
        "spec_decode_metadata": None,
        "rank": None,
        "logit_mask": None,
    })


def test_the_root_is_handed_the_draft_k_of_the_proposer(monkeypatch):
    """Under method "mtp" the runner names the proposer's k; the root never re-reads a knob."""
    monkeypatch.setenv(head_module.SHADOW_DRAFT_ENV, str(K))
    world = shadow._world()
    assert world.root.mtp is not None
    world.runner.is_mtp_spec = True
    world.runner.drafter = SimpleNamespace(num_speculative_tokens=K)
    monkeypatch.setattr(head_module, "shadow_draft_k", lambda: 0)
    assert _prefill_conversion(world)["draft_k"] == K
    # Without a head there is nothing to draft with, and the keyword stays absent so the
    # traced signature is the one every non-drafting serve traced.
    monkeypatch.undo()
    monkeypatch.delenv(head_module.SHADOW_DRAFT_ENV, raising=False)
    plain = shadow._world()
    assert plain.root.mtp is None
    assert "draft_k" not in _prefill_conversion(plain)


def test_host_only_metadata_stays_on_for_the_mtp_spec(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_NEURON_GLM5NEXT_HOST_ONLY_METADATA", True)
    world = _world(1)
    runner = world.runner
    runner.vllm_config = SimpleNamespace(kv_transfer_config=None)
    runner.speculative_config = SimpleNamespace(method="mtp", num_speculative_tokens=K)
    runner.is_mtp_spec = True
    assert runner._glm5next_host_only_metadata() is True
    runner.speculative_config = SimpleNamespace(method="eagle3", num_speculative_tokens=K)
    runner.is_mtp_spec = False
    assert runner._glm5next_host_only_metadata() is False
