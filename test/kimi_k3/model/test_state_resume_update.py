# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 state pages under speculation: the count the runner records after a step.

After each step the runner records, for each request, the count its state layers resume
from on the next step: the step's valid sampled ids less one, never below 0. The sampler
gives the ids as a 1-D tensor with one id per request (a prefill), a 2-D tensor with one
row of verify tokens per request, or a list of each request's accepted ids. The record
waits in the runner until the next flush, which the record, the state slot zeroing and the
resume-row read each call first. The first items run the update on a stand-in that holds
only what it reads. The later items run on a bare runner with one attention group and one
state group, as test/vllm/worker/test_state_slot_zeroing.py builds one, where each request
holds 5 state pages, and test_load_time_layouts.py's torch function mode stands in for the
device's rule on the sampled ids. No model loads. Run with ``VLLM_NEURON_CPU_MODE=1``.
"""

import ast
import itertools
import types
from pathlib import Path

import numpy as np
import torch
from test_load_time_layouts import DeviceRule
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)

from vllm_neuron.vllm.worker import neuron_model_runner
from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

VOCAB, STALE, PAGES = 100, 9, 8
KEYS = (3, 1, 6, 4)
# The bare runner: its blocks and page width, the draft count, and the padded batch size.
BLOCKS, PAGE, DRAFTS, PADDED = 32, 24, 4, 4
WIDTH = 1 + DRAFTS
# The first state pages of three requests, two more pages, and 8 steps of mixed counts.
FIRSTS, P, Q = [1, 6, 11], 16, 21
STEPS = [[0, 1, 2], [4, 0, 3], [2, 4, 1], [1, 3, 0], [3, 2, 4], [1, 1, 2], [3, 0, 4], [0, 2, 1]]
INDEX_OPS = {"index_copy_", "index_select", "index_fill_", "index_put_"}
METHODS = ("_flush_state_resume", "_update_states_after_model_execute", "_zero_new_state_slots",
           "_state_resume_rows")


def _runner(num_reqs: int) -> types.SimpleNamespace:
    """A stand-in runner: the count table and record, batch size, vocabulary, keys and flush."""
    runner = types.SimpleNamespace(
        _state_resume_table=torch.full((PAGES,), STALE, dtype=torch.int32),
        _state_resume_pending=None,
        input_batch=types.SimpleNamespace(num_reqs=num_reqs),
        vocab_size=VOCAB,
        key_calls=[],
    )

    def state_page_keys(num_rows: int) -> np.ndarray:
        runner.key_calls.append(num_rows)
        return np.array(KEYS[:num_rows], dtype=np.int64)

    runner._state_page_keys = state_page_keys
    runner._flush_state_resume = types.MethodType(NeuronModelRunner._flush_state_resume, runner)
    return runner


def _update(runner: types.SimpleNamespace, sampled_token_ids) -> torch.Tensor:
    """Run the update on the stand-in, flush it as every reader does, and return its count table."""
    NeuronModelRunner._update_states_after_model_execute(runner, sampled_token_ids, None)
    runner._flush_state_resume()
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


def _bare(firsts: list[int]) -> NeuronModelRunner:
    """A bare runner with the count table, a batch, the speculative config and the vocabulary."""
    attention = FullAttentionSpec(block_size=8, num_kv_heads=1, head_size=8, dtype=torch.int8)
    state = MambaSpec(shapes=((3, 8),), dtypes=(torch.int8,), block_size=8, page_size_padded=PAGE)
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.kv_cache_config = KVCacheConfig(
        num_blocks=BLOCKS, kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["attention.0"], attention),
                         KVCacheGroupSpec(["state.0"], state)])
    runner._state_pages = {"state.0": torch.ones(BLOCKS, PAGE, dtype=torch.int8)}
    runner._state_zero_rows = {}
    runner.device = torch.device("cpu")
    runner._state_resume_table = torch.full((BLOCKS,), STALE, dtype=torch.int32)
    runner._state_resume_pending = None
    runner.speculative_config = types.SimpleNamespace(num_speculative_tokens=DRAFTS)
    runner.vocab_size = VOCAB
    _schedule(runner, firsts)
    return runner


def _schedule(runner: NeuronModelRunner, firsts: list[int]) -> None:
    """Make the batch one request a first state page, its 5 state pages from there."""
    state = np.zeros((PADDED, WIDTH), dtype=np.int32)
    for row, first in enumerate(firsts):
        state[row] = np.arange(first, first + WIDTH)
    tables = [np.ones((PADDED, 1), dtype=np.int32), state]
    runner.input_batch = types.SimpleNamespace(
        num_reqs=len(firsts),
        block_table=[types.SimpleNamespace(block_table=types.SimpleNamespace(np=table))
                     for table in tables],
    )


def _new_request(first: int) -> types.SimpleNamespace:
    """A request that starts from its first token, its state pages from ``first``."""
    return types.SimpleNamespace(block_ids=([1], list(range(first, first + WIDTH))),
                                 num_computed_tokens=0)


def _verify_ids(counts: list[int]) -> torch.Tensor:
    """Padded verify rows: row i holds ``counts[i] + 1`` valid ids, then -1."""
    ids = torch.full((PADDED, WIDTH), -1, dtype=torch.int64)
    for row, count in enumerate(counts):
        ids[row, : count + 1] = torch.arange(1, count + 2)
    return ids


class _Logged(DeviceRule):
    """The device rule, logging each placement and each host read of a marked tensor by phase."""

    def __init__(self, marked):
        super().__init__(marked)
        self.phase, self.log = None, []

    def __torch_function__(self, func, types_, args=(), kwargs=None):
        held = len(self.held)
        read = bool(args) and getattr(func, "__name__", None) in ("cpu", "to") and (
            self.is_marked(args[0]))
        out = super().__torch_function__(func, types_, args, kwargs)
        if read:
            self.log.append((self.phase, "read", args[0].untyped_storage().data_ptr()))
        if len(self.held) > held:
            self.log.append((self.phase, "place", out.dtype, tuple(out.shape)))
        return out


def _phased_flush(runner: NeuronModelRunner, rule: _Logged) -> None:
    """Make each flush tag the rule's log with its own phase while it runs."""
    flush, count = runner._flush_state_resume, itertools.count()

    def flushed() -> None:
        outer, rule.phase = rule.phase, ("flush", next(count))
        try:
            flush()
        finally:
            rule.phase = outer

    runner._flush_state_resume = flushed


def test_record_reset_and_read_place_one_row_tensor_a_read_and_read_each_id_tensor_once():
    """Under the device rule, over mixed calls, only a read places, and each flush reads once."""
    runner = _bare(FIRSTS)
    ids = [_verify_ids([4, 0, 2]), torch.tensor([5, 7, -1, 3]), _verify_ids([1, 3, 0])]
    calls = [("record", ids[0]), ("read",), ("record", ids[1]), ("reset",),
             ("record", [[1, 2], [3], []]), ("read",), ("record", ids[2]), ("read",), ("read",)]
    rule, errors = _Logged(ids), []
    _phased_flush(runner, rule)
    with rule:
        for i, (kind, *arg) in enumerate(calls):
            rule.phase = (kind, i)
            try:
                if kind == "record":
                    runner._update_states_after_model_execute(arg[0], None)
                elif kind == "reset":
                    runner._zero_new_state_slots([_new_request(P)])
                else:
                    runner._state_resume_rows(PADDED)
            except Exception as e:  # a refusal, or any other error, fails the item below
                errors.append(repr(e))
    assert not rule.refused and not errors, f"refused {rule.refused}, errors {errors}"
    own = [[entry[1:] for entry in rule.log if entry[0] == (kind, i)]
           for i, (kind, *_) in enumerate(calls)]
    want = [[("place", torch.int32, (PADDED,))] if kind == "read" else []
            for kind, *_ in calls]
    assert own == want, f"each call's own placements and host reads: {own}"
    flushed = [entry for entry in rule.log if entry[0][0] == "flush"]
    assert [entry[1] for entry in flushed] == ["read"] * len(ids), flushed
    assert len({entry[0] for entry in flushed}) == len(ids)
    assert [entry[2] for entry in flushed] == [t.untyped_storage().data_ptr() for t in ids]


def test_a_new_request_on_a_recycled_page_starts_at_count_0():
    """A record of 3 under page P, then a new request's slot zeroing on P: the read gives 0."""
    runner = _bare([P])
    runner._update_states_after_model_execute(_verify_ids([3]), None)
    runner._zero_new_state_slots([_new_request(P)])
    assert runner._state_resume_rows(PADDED).tolist() == [0, STALE, STALE, STALE]


def test_a_prefill_record_between_decode_records_drops_no_record():
    """A decode record of 2 under P, then a prefill record under Q: the read gives P 2, Q 0."""
    runner = _bare([P])
    runner._update_states_after_model_execute(_verify_ids([2]), None)
    _schedule(runner, [Q])
    runner._update_states_after_model_execute(torch.tensor([5, -1, -1, -1]), None)
    _schedule(runner, [P, Q])
    assert runner._state_resume_rows(PADDED).tolist() == [2, 0, STALE, STALE]


def test_eight_mixed_count_steps_read_back_each_steps_counts():
    """After each of 8 steps, as tensors and lists in turn, the read gives that step's counts."""
    runner, got = _bare(FIRSTS), []
    for step, counts in enumerate(STEPS):
        ids = _verify_ids(counts) if step % 2 == 0 else [[1] * (c + 1) for c in counts]
        runner._update_states_after_model_execute(ids, None)
        got.append(runner._state_resume_rows(PADDED).tolist()[: len(FIRSTS)])
    assert got == STEPS


def test_count_methods_call_no_index_op_and_flush_first():
    """The four count methods call no index op; the other three call the flush first."""
    tree = ast.parse(Path(neuron_model_runner.__file__).read_text())
    runner = next(node for node in tree.body
                  if isinstance(node, ast.ClassDef) and node.name == "NeuronModelRunner")
    methods = {f.name: f for f in runner.body
               if isinstance(f, ast.FunctionDef) and f.name in METHODS}
    assert set(methods) == set(METHODS)
    hits = [(name, node.lineno, node.func.attr) for name, f in methods.items()
            for node in ast.walk(f) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and node.func.attr in INDEX_OPS]
    assert hits == []
    first = {name: ast.unparse(f.body[1] if ast.get_docstring(f) else f.body[0])
             for name, f in methods.items() if name != METHODS[0]}
    assert first == {name: "self._flush_state_resume()" for name in METHODS[1:]}
