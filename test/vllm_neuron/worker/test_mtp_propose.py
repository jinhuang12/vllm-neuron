# SPDX-License-Identifier: Apache-2.0
"""The runner's draft proposal under method "mtp": ``_glm5next_propose_drafts``.

The root drafts inside the target graph (the traced head), so proposing is bookkeeping:
after a decode step the ``[B, k]`` draft ids the root returned are the next step's
drafts; after a prefill the request gets ``k`` placeholder drafts (the model's pad id,
as the DI bootstrap does) so its first decode is a verify step of the same width as
everyone else's -- a partial prefill chunk, with no sampled token yet, gets none; near
``max_model_len`` nothing is proposed, by the same limit the eagle path uses, so the
scheduler never trims a draft; a synthetic step proposes nothing. Covered here on a
bare runner, together with the dispatch from ``_propose_draft_token_ids`` and the
shared ``_spec_decode_limit``.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/worker/test_mtp_propose.py
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

K = 3
T = 1 + K
MAX_MODEL_LEN = 64
PAD = 7
NUM_REQS = 2
BUCKET = 4


def _runner(*, record, drafts, async_scheduling: bool = False, pad_token_id=PAD):
    runner = SimpleNamespace(
        is_mtp_spec=True,
        drafter=SimpleNamespace(num_speculative_tokens=K),
        input_batch=SimpleNamespace(num_reqs=NUM_REQS),
        max_model_len=MAX_MODEL_LEN,
        use_async_scheduling=async_scheduling,
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(hf_config=SimpleNamespace(pad_token_id=pad_token_id))
        ),
        _glm5next_step_record=record,
        _glm5next_shadow_last_drafts=drafts,
    )
    # The two helpers the propose path shares with the eagle site, bound to the stub.
    runner._spec_decode_limit = lambda: NeuronModelRunner._spec_decode_limit(runner)
    runner._placeholder_drafts = lambda: NeuronModelRunner._placeholder_drafts(runner)
    runner._glm5next_propose_drafts = (
        lambda sampled, scheduler_output=None: NeuronModelRunner._glm5next_propose_drafts(
            runner, sampled, scheduler_output
        )
    )
    # A synchronous drafter: the real question, answering False for a fake proposer.
    runner._glm5next_async_drafter = (
        lambda: NeuronModelRunner._glm5next_async_drafter(runner)
    )
    return runner


def _record(starts, *, width=T, is_prefill=False):
    return {
        "slots": list(range(len(starts))), "starts": list(starts), "counts": [width] * len(starts),
        "is_prefill": is_prefill, "width": width,
    }


def _propose(runner, sampled):
    return NeuronModelRunner._glm5next_propose_drafts(runner, sampled)


def test_after_a_decode_step_the_roots_drafts_are_the_next_steps_drafts():
    drafts = torch.arange(BUCKET * K, dtype=torch.int32).reshape(BUCKET, K)
    runner = _runner(record=_record([10, 20]), drafts=drafts)
    assert _propose(runner, [[1], [2]]) == drafts[:NUM_REQS].tolist()


def test_a_one_row_decode_step_proposes_the_roots_drafts_too():
    drafts = torch.full((BUCKET, K), 3, dtype=torch.int32)
    runner = _runner(record=_record([10, 20], width=1), drafts=drafts)
    assert _propose(runner, [[1], [2]]) == [[3] * K] * NUM_REQS


def test_after_a_prefill_each_sampled_request_gets_placeholder_drafts():
    runner = _runner(record=_record([0, 0], width=9, is_prefill=True), drafts=torch.full((2, K), -1))
    assert _propose(runner, [[11], []]) == [[PAD] * K, []]


def test_the_placeholder_falls_back_to_token_zero_without_a_pad_id():
    runner = _runner(record=_record([0], width=9, is_prefill=True), drafts=None, pad_token_id=None)
    runner.input_batch.num_reqs = 1
    assert _propose(runner, [[11]]) == [[0] * K]


def test_nothing_is_proposed_near_max_model_len():
    limit = NeuronModelRunner._spec_decode_limit(_runner(record=None, drafts=None))
    assert limit == MAX_MODEL_LEN - K - 2
    drafts = torch.ones((BUCKET, K), dtype=torch.int32)
    # The last row of request 1 stands at the limit: no request gets drafts.
    runner = _runner(record=_record([10, limit - T + 1]), drafts=drafts)
    assert _propose(runner, [[1], [2]]) == [[], []]
    # One row short of it, drafts flow.
    runner = _runner(record=_record([10, limit - T]), drafts=drafts)
    assert _propose(runner, [[1], [2]]) == [[1] * K] * NUM_REQS


def test_async_scheduling_pulls_the_limit_one_step_earlier():
    runner = _runner(record=None, drafts=None, async_scheduling=True)
    assert NeuronModelRunner._spec_decode_limit(runner) == MAX_MODEL_LEN - K - 2 - T


def test_a_synthetic_step_proposes_nothing():
    runner = _runner(record=None, drafts=None)
    assert _propose(runner, [[1], [2]]) == [[], []]


def test_a_decode_step_whose_root_returned_no_drafts_is_refused_by_name():
    runner = _runner(record=_record([10, 20]), drafts=None)
    with pytest.raises(ValueError, match="draft"):
        _propose(runner, [[1], [2]])


def test_propose_draft_token_ids_dispatches_to_the_mtp_path():
    drafts = torch.arange(BUCKET * K, dtype=torch.int32).reshape(BUCKET, K)
    runner = _runner(record=_record([10, 20]), drafts=drafts)
    out = NeuronModelRunner._propose_draft_token_ids(
        runner, scheduler_output=None, sampled_token_ids=[[1], [2]], aux_hidden_states=None,
        spec_decode_metadata=None, input_ids=torch.zeros(1), positions=torch.zeros(1), attn_metadata={},
    )
    assert out == drafts[:NUM_REQS].tolist()
