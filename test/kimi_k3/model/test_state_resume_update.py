# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 state pages under speculation: the count the runner records after a step.

After each step the runner stores, for each request, the count its state layers resume
from on the next step: the step's valid sampled ids less one, never below 0. The sampler
gives the ids as a 1-D tensor with one id per request (a prefill), a 2-D tensor with one
row of verify tokens per request, or a list of each request's accepted ids. A stand-in
holds only what the update reads, so no model loads. Run with ``VLLM_NEURON_CPU_MODE=1``.
"""

import types

import numpy as np
import torch

from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

VOCAB, STALE, PAGES = 100, 9, 8
KEYS = (3, 1, 6, 4)


def _runner(num_reqs: int) -> types.SimpleNamespace:
    """A stand-in runner: the count table, the batch size, the vocabulary and the count keys."""
    runner = types.SimpleNamespace(
        _state_resume_table=torch.full((PAGES,), STALE, dtype=torch.int32),
        input_batch=types.SimpleNamespace(num_reqs=num_reqs),
        vocab_size=VOCAB,
        key_calls=[],
    )

    def state_page_keys(num_rows: int) -> np.ndarray:
        runner.key_calls.append(num_rows)
        return np.array(KEYS[:num_rows], dtype=np.int64)

    runner._state_page_keys = state_page_keys
    return runner


def _update(runner: types.SimpleNamespace, sampled_token_ids) -> torch.Tensor:
    """Run the update on the stand-in and return its count table."""
    NeuronModelRunner._update_states_after_model_execute(runner, sampled_token_ids, None)
    return runner._state_resume_table


def _table(counts: dict[int, int]) -> torch.Tensor:
    """The count table with ``counts`` at their keys and the stale value elsewhere."""
    table = torch.full((PAGES,), STALE, dtype=torch.int32)
    for key, count in counts.items():
        table[key] = count
    return table


def test_one_id_per_request_gives_count_zero():
    """A 1-D tensor holds one id per request, as after a prefill: each count is 0."""
    ids = torch.tensor([5, -1, VOCAB + 3, 7])  # the last id is a padding row
    assert torch.equal(_update(_runner(3), ids), _table({3: 0, 1: 0, 6: 0}))


def test_verify_rows_give_valid_count_less_one():
    """A 2-D tensor holds one row of verify tokens per request: its valid ids less one, at least 0."""
    ids = torch.tensor(
        [
            [4, 8, 2, -1, -1],
            [-1, -1, -1, -1, -1],
            [4, 8, VOCAB, VOCAB + 1, -1],
            [1, 2, 3, 4, 5],
            [1, 1, 1, 1, 1],  # a padding row
        ]
    )
    assert torch.equal(_update(_runner(4), ids), _table({3: 2, 1: 0, 6: 1, 4: 4}))


def test_accepted_id_lists_give_their_length_less_one():
    """A list holds each request's accepted ids: its length less one, at least 0."""
    ids = [[4, 8, 2], [], [7], [1, 2, 3, 4, 5], [9, 9]]  # the last list is a padding row
    assert torch.equal(_update(_runner(4), ids), _table({3: 2, 1: 0, 6: 0, 4: 4}))


def test_empty_batch_leaves_the_table():
    """With no request in the batch the update reads no key and writes nothing."""
    runner = _runner(0)
    assert torch.equal(_update(runner, torch.tensor([5, 7])), _table({}))
    assert runner.key_calls == []
