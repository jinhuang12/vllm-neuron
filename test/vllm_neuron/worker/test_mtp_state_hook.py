# SPDX-License-Identifier: Apache-2.0
"""The per-step state hook under method "mtp": ``_update_states_after_model_execute``.

The GLM-5.3-Flash translator runs before the step knows how many drafts the trunk
will accept, so a verify step advances every request's indexer-ring cursor past all
``T = 1 + k`` of its rows and records the step (``_glm5next_step_record``). Once the
host holds the rejection sampler's rows, the hook pulls each request's cursor back to
``start + kept``, commits the kept counts to the KDA checkpoint banks
(``fused_decode.commit_kda_checkpoints``, worker-57's pointer commit: no bytes move)
and records ``kept - 1`` -- the checkpoint row the request resumes from -- for the
request's slot, which the next step's recurrent carriers read as ``checkpoint_rows``.
A one-row decode or a prefill records 0 for its slots, commits nothing and leaves the
cursor the translator set. Covered here on a bare runner: the bookkeeping writes, the
commit call and its arguments, the zeroing steps, the refusals (a row kept outside
``1 .. T``, fewer sampler rows than requests) and the no-op on a synthetic step. The
commit function is worker-57's and is monkeypatched onto its module here.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/worker/test_mtp_state_hook.py
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_neuron.functional.kda import fused_decode
from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

K = 3
T = 1 + K
SLOTS = [4, 1]
STARTS = [37, 12]
BANK_SLOTS = 8


def _recurrent_banks() -> list[dict]:
    return [
        {
            "family": "linear_attn",
            "name": "layers.0",
            "conv_state": torch.zeros((BANK_SLOTS, T, 2, 3)),
            "recurrent_state": torch.zeros((BANK_SLOTS, T, 2, 4, 4)),
        },
        {"family": "self_attn", "name": "layers.1"},
    ]


def _runner(*, record, banks=None):
    """A runner holding only what the hook reads and writes."""
    return SimpleNamespace(
        _glm5next_step_record=record,
        _glm5next_side_cache_positions={slot: start + T for slot, start in zip(SLOTS, STARTS)},
        _glm5next_checkpoint_rows={slot: 2 for slot in SLOTS},
        model=SimpleNamespace(glm5next_layer_banks=[] if banks is None else banks),
        drafter=SimpleNamespace(num_speculative_tokens=K),
    )


def _verify_record(width: int = T) -> dict:
    return {
        "slots": list(SLOTS), "starts": list(STARTS), "counts": [width] * len(SLOTS),
        "is_prefill": False, "width": width,
    }


def _hook(runner, rows):
    NeuronModelRunner._update_states_after_model_execute(runner, rows, None)


@pytest.fixture
def commit(monkeypatch):
    """The KDA commit, called through and recorded (the hook imports it by name at call
    time, so the recording wrapper is what it reaches)."""
    calls = []
    real = fused_decode.commit_kda_checkpoints

    def recording(banks, slot_ids, accepted_counts, *, state_checkpoints):
        calls.append((list(banks), list(slot_ids), list(accepted_counts), int(state_checkpoints)))
        return real(banks, slot_ids, accepted_counts, state_checkpoints=state_checkpoints)

    monkeypatch.setattr(fused_decode, "commit_kda_checkpoints", recording)
    return calls


def test_a_verify_step_pulls_each_cursor_back_to_the_kept_rows_and_records_the_resume_row(commit):
    runner = _runner(record=_verify_record())
    _hook(runner, [[5, 6, 7], [9]])
    assert runner._glm5next_side_cache_positions == {SLOTS[0]: STARTS[0] + 3, SLOTS[1]: STARTS[1] + 1}
    assert runner._glm5next_checkpoint_rows == {SLOTS[0]: 2, SLOTS[1]: 0}
    assert commit == [], "no recurrent bank, nothing to commit"


def test_a_verify_step_commits_the_kept_counts_to_the_recurrent_banks(commit):
    banks = _recurrent_banks()
    runner = _runner(record=_verify_record(), banks=banks)
    _hook(runner, [[5, 6, 7], [9]])
    assert len(commit) == 1
    tensors, slots, kept, checkpoints = commit[0]
    assert [t is banks[0]["conv_state"] or t is banks[0]["recurrent_state"] for t in tensors] == [True, True]
    assert (slots, kept, checkpoints) == (SLOTS, [3, 1], T)
    assert runner._glm5next_checkpoint_rows == {SLOTS[0]: 2, SLOTS[1]: 0}


def test_an_all_accepted_row_keeps_every_row_including_the_bonus(commit):
    runner = _runner(record=_verify_record())
    _hook(runner, [[1, 2, 3, 4], [1, 2, 3, 4]])
    assert runner._glm5next_side_cache_positions == {s: p + T for s, p in zip(SLOTS, STARTS)}
    assert runner._glm5next_checkpoint_rows == {s: K for s in SLOTS}


def test_padding_rows_past_the_real_requests_are_ignored(commit):
    runner = _runner(record=_verify_record(), banks=_recurrent_banks())
    _hook(runner, [[5, 6], [9, 8, 7], [0], [0]])
    assert runner._glm5next_checkpoint_rows == {SLOTS[0]: 1, SLOTS[1]: 2}
    assert commit[0][1:3] == (SLOTS, [2, 3])


def test_a_one_row_decode_zeroes_the_resume_row_leaves_the_cursor_and_commits_nothing(commit):
    runner = _runner(record=_verify_record(width=1), banks=_recurrent_banks())
    runner._glm5next_side_cache_positions = {s: p + 1 for s, p in zip(SLOTS, STARTS)}
    _hook(runner, [[5], [9]])
    assert runner._glm5next_side_cache_positions == {s: p + 1 for s, p in zip(SLOTS, STARTS)}
    assert runner._glm5next_checkpoint_rows == {s: 0 for s in SLOTS}
    assert commit == []


def test_a_prefill_zeroes_the_resume_row_even_for_a_partial_chunk(commit):
    record = {"slots": [SLOTS[0]], "starts": [0], "counts": [9], "is_prefill": True, "width": None}
    runner = _runner(record=record, banks=_recurrent_banks())
    runner._glm5next_side_cache_positions = {SLOTS[0]: 9}
    _hook(runner, [[]])
    assert runner._glm5next_side_cache_positions == {SLOTS[0]: 9}
    assert runner._glm5next_checkpoint_rows[SLOTS[0]] == 0
    assert commit == []


@pytest.mark.parametrize("rows", [[[], [9]], [[1, 2, 3, 4, 5], [9]]], ids=["none-kept", "past-T"])
def test_a_verify_row_kept_outside_one_to_t_is_refused_by_name(commit, rows):
    runner = _runner(record=_verify_record())
    with pytest.raises(ValueError, match=rf"kept {len(rows[0])} id\(s\).*{T}"):
        _hook(runner, rows)


def test_fewer_sampler_rows_than_requests_is_refused_by_name(commit):
    runner = _runner(record=_verify_record())
    with pytest.raises(ValueError, match=r"2 request\(s\).*1 row\(s\)"):
        _hook(runner, [[5, 6, 7]])


def test_a_synthetic_step_records_nothing_and_the_hook_does_nothing(commit):
    runner = _runner(record=None, banks=_recurrent_banks())
    before = (dict(runner._glm5next_side_cache_positions), dict(runner._glm5next_checkpoint_rows))
    _hook(runner, [[5]])
    assert (runner._glm5next_side_cache_positions, runner._glm5next_checkpoint_rows) == before
    assert commit == []
