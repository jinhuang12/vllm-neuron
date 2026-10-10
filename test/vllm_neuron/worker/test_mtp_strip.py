# SPDX-License-Identifier: Apache-2.0
"""A mixed verify step under method "mtp" is stripped to the one-row graph.

The GLM-5.3-Flash translator serves one row width per step, so a decode step whose
requests carry different draft counts cannot be laid out. Two steps produce one: a
request's first decode after prefill (0 drafts) scheduled beside requests carrying
``k``, and a request near ``max_model_len`` whose drafts vLLM's scheduler truncated.
The runner's existing disaggregated-inference guard (``_local_step_forces_nonspec``
-> ``_maybe_strip_spec_for_nonspec_step``) already strips a step to one token per
request; the vote is extended by ``_local_force_nonspec`` to a mixed mtp step,
narrowed to steps whose per-request draft counts
differ: a uniform step keeps its drafts. Covered on bare stubs.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/worker/test_mtp_strip.py
"""

from __future__ import annotations

from types import SimpleNamespace

from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

K = 3
REQS = ["a", "b", "c"]


def _step(drafts: dict[str, list[int]]):
    rows = {req: 1 + len(drafts.get(req, [])) for req in REQS}
    return SimpleNamespace(
        scheduled_spec_decode_tokens={req: list(ids) for req, ids in drafts.items() if ids},
        num_scheduled_tokens=dict(rows),
        num_scheduled_tokens_padded=dict(rows),
        total_num_scheduled_tokens=sum(rows.values()),
    )


def _runner(*, mtp: bool = True, consumer: bool = False):
    runner = SimpleNamespace(
        is_mtp_spec=mtp,
        speculative_config=SimpleNamespace(num_speculative_tokens=K, method="mtp" if mtp else "eagle3"),
        vllm_config=SimpleNamespace(
            kv_transfer_config=SimpleNamespace(is_kv_consumer=True) if consumer else None
        ),
        input_batch=SimpleNamespace(num_reqs=len(REQS), req_ids=list(REQS)),
    )
    runner._local_step_forces_nonspec = (
        lambda step: NeuronModelRunner._local_step_forces_nonspec(runner, step)
    )
    runner._glm5next_step_is_mixed = (
        lambda step: NeuronModelRunner._glm5next_step_is_mixed(runner, step)
    )
    return runner


def _mixed(runner, step) -> bool:
    return NeuronModelRunner._glm5next_step_is_mixed(runner, step)


def _force(runner, step) -> bool:
    return NeuronModelRunner._local_force_nonspec(runner, step)


FULL = [11, 12, 13]


def test_a_fresh_requests_first_decode_beside_drafted_requests_is_mixed():
    step = _step({"a": FULL, "b": FULL})  # "c" just prefilled: no drafts
    assert _mixed(_runner(), step) is True
    assert _force(_runner(), step) is True


def test_a_request_whose_drafts_the_scheduler_truncated_is_mixed():
    step = _step({"a": FULL, "b": FULL, "c": FULL[:1]})  # "c" one token before max_model_len
    assert _mixed(_runner(), step) is True
    assert _force(_runner(), step) is True


def test_a_uniform_verify_step_keeps_its_drafts():
    step = _step({req: FULL for req in REQS})
    assert _mixed(_runner(), step) is False
    assert _force(_runner(), step) is False


def test_a_step_with_no_drafts_at_all_is_not_mixed():
    assert _mixed(_runner(), _step({})) is False
    assert _force(_runner(), _step({})) is False


def test_without_the_mtp_spec_the_mixed_vote_is_not_cast():
    step = _step({"a": FULL, "b": FULL})
    assert _force(_runner(mtp=False), step) is False


def test_the_disaggregated_consumer_vote_is_unchanged():
    runner = _runner(mtp=False, consumer=True)
    assert _force(runner, _step({})) is True, "no drafts after a KV transfer: the non-spec graph"
    assert _force(runner, _step({req: FULL for req in REQS})) is False


def test_the_vote_strips_the_whole_step_to_one_row_per_request():
    runner = _runner()
    for drafts in ({"a": FULL, "b": FULL}, {"a": FULL, "b": FULL, "c": FULL[:1]}):
        step = _step(drafts)
        vote = _force(runner, step)
        NeuronModelRunner._maybe_strip_spec_for_nonspec_step(runner, step, vote, False)
        assert step.scheduled_spec_decode_tokens == {}
        assert step.num_scheduled_tokens == {req: 1 for req in REQS}
        assert step.num_scheduled_tokens_padded == {req: 1 for req in REQS}
        assert step.total_num_scheduled_tokens == len(REQS)
