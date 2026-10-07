# SPDX-License-Identifier: Apache-2.0
"""Shadow-draft scoring: drafts are scored against the tokens sampled k steps later.

Stage A of MTP for GLM-5.3-Flash runs the layer-45 draft beside the decode graph and
never uses its output; the runner buffers the k drafts a step emits and scores them
against the tokens the trunk samples at the following k steps. The scorer here is the
pure-Python core (``Glm5NextShadowDraftScorer``) and the runner glue around it
(``_glm5next_shadow_*``): the per-step hand-off of sampled ids and drafts, the one-step
delay async scheduling imposes on host-visible ids, request completion with drafts still
pending, the reuse of a batch row by a new request, and the JSONL log.

Alignment (the classic MTP bug): a decode step at position s consumed x_s and sampled
x_{s+1}; its drafts d_1..d_k predict x_{s+2}..x_{s+k+1}. Draft d_j is accepted iff
d_1..d_{j-1} were accepted and d_j equals the token sampled j steps later.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/worker/test_shadow_draft_scoring.py
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from vllm_neuron.vllm.worker.neuron_model_runner import (
    Glm5NextShadowDraftScorer,
    NeuronModelRunner,
)

pytestmark = [pytest.mark.fast]

K = 5
VOCAB = 1000


# ── helpers ──────────────────────────────────────────────────────────────────


def _tokens(req: int, count: int) -> list[int]:
    """A request's generated tokens x_T, x_{T+1}, ...: distinct, request-specific."""
    return [(req * 100 + 7 * i + 3) % VOCAB for i in range(count)]


def _planted_drafts(future: list[int], accept: int) -> list[int]:
    """k drafts over the true ``future`` tokens: the first ``accept`` right, the rest wrong."""
    drafts = []
    for j in range(K):
        truth = future[j] if j < len(future) else 0
        drafts.append(truth if j < accept else (truth + 1) % VOCAB)
    return drafts


def _alphas(records: list[dict], k: int) -> tuple[list[float], list[float]]:
    """Conditional and cumulative acceptance per position from scored records."""
    conditional = []
    for j in range(1, k + 1):
        eligible = [r for r in records if r["accepted_prefix_len"] >= j - 1 and r["scored"] >= j]
        hits = [r for r in eligible if r["accepted_prefix_len"] >= j]
        conditional.append(len(hits) / len(eligible) if eligible else float("nan"))
    cumulative, run = [], 1.0
    for a in conditional:
        run *= a
        cumulative.append(run)
    return conditional, cumulative


def _runner_shell(*, k: int = K, req_ids=None, rank: int = 0, head: bool = True):
    """A runner with the state the shadow hooks read; ``head`` = the served root built ``mtp``."""
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.input_batch = SimpleNamespace(req_ids=list(req_ids or []))
    runner.requests = {}
    runner.rank_tensor = torch.tensor(rank, dtype=torch.int32)
    runner.model = SimpleNamespace(mtp=object() if head else None)
    return runner


# ── the scorer core ─────────────────────────────────────────────────────────


def test_a_known_per_position_acceptance_pattern_is_recovered_exactly():
    """One request, 40 decode steps; step t's drafts accept exactly ``plan[t]`` positions."""
    plan = [5, 3, 0, 1, 5, 2, 4, 0, 5, 1] * 4
    steps = len(plan)
    tokens = _tokens(1, steps + K + 1)          # x_T .. x_{T+steps+K}
    records: list[dict] = []
    scorer = Glm5NextShadowDraftScorer(K, records.append)
    # The prefill step samples x_T and emits no drafts.
    scorer.observe("r1", step=0, position=9, sampled=tokens[0], drafts=None)
    for t in range(steps):
        # Decode step t+1 consumed x_{T+t}, sampled x_{T+t+1}; drafts predict x_{T+t+2}..
        future = tokens[t + 2 : t + 2 + K]
        scorer.observe(
            "r1", step=t + 1, position=10 + t, sampled=tokens[t + 1],
            drafts=_planted_drafts(future, plan[t]),
        )
    scorer.retire("r1")
    # The last k records retire with fewer than k tokens after them: their acceptance is
    # capped by what was scored.
    assert [r["accepted_prefix_len"] for r in records] == [
        min(a, r["scored"]) for a, r in zip(plan, records)
    ]
    assert all(r["scored"] == K for r in records[: steps - K])
    assert [r["step"] for r in records] == list(range(1, steps + 1))
    assert [r["position"] for r in records] == [10 + t for t in range(steps)]
    assert records[0]["drafts"] == _planted_drafts(tokens[2 : 2 + K], plan[0])
    assert records[0]["actual"] == tokens[2 : 2 + K]
    full = [r for r in records if r["scored"] == K]
    conditional, cumulative = _alphas(full, K)
    expect_cond = []
    for j in range(1, K + 1):
        eligible = [a for a in plan[: len(full)] if a >= j - 1]
        expect_cond.append(sum(1 for a in eligible if a >= j) / len(eligible))
    assert conditional == pytest.approx(expect_cond)
    run, expect_cum = 1.0, []
    for a in expect_cond:
        run *= a
        expect_cum.append(run)
    assert cumulative == pytest.approx(expect_cum)


def test_a_record_is_emitted_only_once_its_k_following_tokens_arrived():
    records: list[dict] = []
    scorer = Glm5NextShadowDraftScorer(3, records.append)
    tokens = _tokens(2, 10)
    scorer.observe("r2", step=0, position=0, sampled=tokens[0], drafts=None)
    scorer.observe("r2", step=1, position=1, sampled=tokens[1], drafts=[tokens[2], tokens[3], 0])
    assert records == []
    scorer.observe("r2", step=2, position=2, sampled=tokens[2], drafts=[1, 2, 3])
    scorer.observe("r2", step=3, position=3, sampled=tokens[3], drafts=[1, 2, 3])
    assert records == []
    scorer.observe("r2", step=4, position=4, sampled=tokens[4], drafts=[1, 2, 3])
    assert len(records) == 1
    assert records[0]["step"] == 1
    assert records[0]["accepted_prefix_len"] == 2
    assert records[0]["scored"] == 3
    assert records[0]["actual"] == tokens[2:5]


def test_a_request_that_finishes_with_pending_drafts_scores_what_arrived():
    records: list[dict] = []
    scorer = Glm5NextShadowDraftScorer(K, records.append)
    tokens = _tokens(3, 10)
    scorer.observe("r3", step=0, position=0, sampled=tokens[0], drafts=None)
    scorer.observe("r3", step=1, position=1, sampled=tokens[1],
                   drafts=[tokens[2], tokens[3], tokens[4], tokens[5], tokens[6]])
    scorer.observe("r3", step=2, position=2, sampled=tokens[2],
                   drafts=[tokens[3], 0, 0, 0, 0])
    scorer.observe("r3", step=3, position=3, sampled=tokens[3], drafts=[0, 0, 0, 0, 0])
    assert records == []
    scorer.retire("r3")
    assert [r["step"] for r in records] == [1, 2, 3]
    first, second, third = records
    # Two tokens arrived after step 1 (x at steps 2 and 3): both drafts right, 3 unscored.
    assert (first["scored"], first["accepted_prefix_len"]) == (2, 2)
    assert first["actual"] == tokens[2:4]
    # One token arrived after step 2: right.
    assert (second["scored"], second["accepted_prefix_len"]) == (1, 1)
    # Nothing arrived after step 3: nothing scored, nothing accepted.
    assert (third["scored"], third["accepted_prefix_len"]) == (0, 0)
    assert third["actual"] == []
    assert "r3" not in scorer.tracked


def test_prefill_step_drafts_are_scored_against_the_first_generated_tokens():
    """A graph that drafts at the prefill step predicts x_{T+1}..x_{T+k}: scored the same way."""
    records: list[dict] = []
    scorer = Glm5NextShadowDraftScorer(3, records.append)
    tokens = _tokens(4, 8)
    scorer.observe("r4", step=0, position=6, sampled=tokens[0],
                   drafts=[tokens[1], tokens[2], 0])
    for t in range(1, 4):
        scorer.observe("r4", step=t, position=6 + t, sampled=tokens[t], drafts=[0, 0, 0])
    assert records and records[0]["step"] == 0
    assert records[0]["position"] == 6
    assert records[0]["actual"] == tokens[1:4]
    assert records[0]["accepted_prefix_len"] == 2


def test_an_accepted_draft_after_a_rejected_one_does_not_count():
    records: list[dict] = []
    scorer = Glm5NextShadowDraftScorer(3, records.append)
    tokens = _tokens(5, 8)
    scorer.observe("r5", step=0, position=0, sampled=tokens[0], drafts=None)
    scorer.observe("r5", step=1, position=1, sampled=tokens[1],
                   drafts=[(tokens[2] + 1) % VOCAB, tokens[3], tokens[4]])
    for t in range(2, 5):
        scorer.observe("r5", step=t, position=t, sampled=tokens[t], drafts=[0, 0, 0])
    assert records[0]["accepted_prefix_len"] == 0
    assert records[0]["scored"] == 3


def test_retire_absent_flushes_only_the_requests_that_left():
    records: list[dict] = []
    scorer = Glm5NextShadowDraftScorer(2, records.append)
    for req in ("a", "b"):
        scorer.observe(req, step=0, position=0, sampled=1, drafts=None)
        scorer.observe(req, step=1, position=1, sampled=2, drafts=[3, 4])
    scorer.retire_absent(["b"])
    assert [r["req_id"] for r in records] == ["a"]
    assert scorer.tracked == {"b"}


# ── the runner glue ─────────────────────────────────────────────────────────


def _observe_step(runner, req_ids, *, sampled, drafts, is_prefill=False, final=None,
                  starts=None, counts=None):
    """One step as the runner sees it: kwargs hook (stashes the step), then the unpack."""
    runner.input_batch.req_ids = list(req_ids)
    n = len(req_ids)
    starts = list(starts) if starts is not None else [0] * n
    counts = list(counts) if counts is not None else [1] * n
    runner._glm5next_shadow_kwargs(
        is_prefill=is_prefill, request_ids=list(req_ids), request_starts=starts,
        request_tokens=counts, synthetic=False, device=torch.device("cpu"), sampling_rows=n,
    )
    sampled_t = torch.tensor(sampled, dtype=torch.int32)
    drafts_t = None if drafts is None else torch.tensor(drafts, dtype=torch.int32)
    runner._glm5next_shadow_observe(sampled_t, drafts_t, is_prefill=is_prefill)


def _read_log(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_async_one_step_late_ids_are_resolved_at_the_next_step(monkeypatch, tmp_path):
    """The glue resolves a step's device outputs when the next step is observed, so the
    host never blocks on the step it just dispatched; the log is complete at retirement."""
    log = tmp_path / "shadow.jsonl"
    monkeypatch.setenv("VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT", "3")
    monkeypatch.setenv("VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT_LOG", str(log))
    runner = _runner_shell(k=3)
    tokens = _tokens(6, 10)
    # Prefill: one request, final chunk, position 4 (prompt of 5 tokens).
    _observe_step(runner, ["q"], sampled=[tokens[0]], drafts=[[-1, -1, -1]], is_prefill=True,
                  starts=[0], counts=[5])
    # Decode steps 1..4 at positions 5..8.
    for t in range(1, 5):
        future = tokens[t + 1 : t + 4]
        _observe_step(runner, ["q"], sampled=[tokens[t]],
                      drafts=[_planted_drafts(future, 2)[:3]], starts=[4 + t], counts=[1])
    # The newest step is still pending (its tensors are futures on device); every older
    # step has been resolved. Step 1's record needs steps 2..4: it is complete now.
    pending = runner._glm5next_shadow_pending
    assert len(pending) == 1
    # Step 1's record needs the tokens of steps 2, 3 and 4; step 4 is the one still in
    # flight, so nothing is written yet: the lag is exactly one step.
    assert not log.exists() or _read_log(log) == []
    # The request leaves the batch: the pending step is resolved and the rest flushed.
    _observe_step(runner, [], sampled=[], drafts=None)
    written = _read_log(log)
    assert [r["step"] for r in written] == [1, 2, 3, 4]
    assert [r["position"] for r in written] == [5, 6, 7, 8]
    assert all(r["req_id"] == "q" for r in written)
    assert [r["scored"] for r in written] == [3, 2, 1, 0]
    assert [r["accepted_prefix_len"] for r in written] == [2, 2, 1, 0]
    assert runner._glm5next_shadow_scorer_instance.tracked == set()


def test_slot_reuse_by_a_new_request_starts_a_fresh_history(monkeypatch, tmp_path):
    log = tmp_path / "shadow.jsonl"
    monkeypatch.setenv("VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT", "2")
    monkeypatch.setenv("VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT_LOG", str(log))
    runner = _runner_shell(k=2)
    a, b = _tokens(7, 6), _tokens(8, 6)
    _observe_step(runner, ["A"], sampled=[a[0]], drafts=[[-1, -1]], is_prefill=True, counts=[3])
    _observe_step(runner, ["A"], sampled=[a[1]], drafts=[[a[2], b[1]]], starts=[2])
    _observe_step(runner, ["A"], sampled=[a[2]], drafts=[[b[0], b[1]]], starts=[3])
    # A finishes; B takes row 0 in the very next step (its prefill).
    _observe_step(runner, ["B"], sampled=[b[0]], drafts=[[-1, -1]], is_prefill=True, counts=[3])
    _observe_step(runner, ["B"], sampled=[b[1]], drafts=[[b[2], b[3]]], starts=[2])
    _observe_step(runner, ["B"], sampled=[b[2]], drafts=[[0, 0]], starts=[3])
    _observe_step(runner, ["B"], sampled=[b[3]], drafts=[[0, 0]], starts=[4])
    _observe_step(runner, [], sampled=[], drafts=None)
    written = _read_log(log)
    by_req = {}
    for r in written:
        by_req.setdefault(r["req_id"], []).append(r)
    # A's second record predicted b[0], b[1]: B's tokens must not have scored it.
    assert [(r["scored"], r["accepted_prefix_len"]) for r in by_req["A"]] == [(1, 1), (0, 0)]
    assert by_req["A"][1]["actual"] == []
    # B's history starts at its own prefill token.
    assert [(r["scored"], r["accepted_prefix_len"]) for r in by_req["B"]] == [(2, 2), (1, 0), (0, 0)]
    assert by_req["B"][0]["actual"] == [b[2], b[3]]
    assert all(r["row"] == 0 for r in written)


def test_intermediate_prefill_chunks_and_sentinel_drafts_leave_no_trace(monkeypatch, tmp_path):
    log = tmp_path / "shadow.jsonl"
    monkeypatch.setenv("VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT", "2")
    monkeypatch.setenv("VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT_LOG", str(log))
    runner = _runner_shell(k=2)
    runner.requests = {"c": SimpleNamespace(prompt_token_ids=list(range(10)), num_prompt_tokens=10)}
    toks = _tokens(9, 6)
    # Chunk 1 of 2: positions 0..5, not final. Its sampled id is garbage and is not a token.
    _observe_step(runner, ["c"], sampled=[999], drafts=[[-1, -1]], is_prefill=True,
                  starts=[0], counts=[6])
    # Chunk 2: positions 6..9, final: samples x_T.
    _observe_step(runner, ["c"], sampled=[toks[0]], drafts=[[-1, -1]], is_prefill=True,
                  starts=[6], counts=[4])
    _observe_step(runner, ["c"], sampled=[toks[1]], drafts=[[toks[2], toks[3]]], starts=[10])
    _observe_step(runner, ["c"], sampled=[toks[2]], drafts=[[0, 0]], starts=[11])
    _observe_step(runner, ["c"], sampled=[toks[3]], drafts=[[0, 0]], starts=[12])
    _observe_step(runner, [], sampled=[], drafts=None)
    written = _read_log(log)
    # The record at step 3 (first decode) is scored 2/2: the history is [x_T, x_T+1, ...]
    # with no garbage token from chunk 1 in it.
    assert written[0]["step"] == 2 and written[0]["position"] == 10
    assert (written[0]["scored"], written[0]["accepted_prefix_len"]) == (2, 2)


def test_the_log_is_written_by_rank_zero_only(monkeypatch, tmp_path):
    log = tmp_path / "shadow.jsonl"
    monkeypatch.setenv("VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT", "2")
    monkeypatch.setenv("VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT_LOG", str(log))
    runner = _runner_shell(k=2, rank=3)
    toks = _tokens(10, 6)
    _observe_step(runner, ["z"], sampled=[toks[0]], drafts=[[-1, -1]], is_prefill=True, counts=[2])
    _observe_step(runner, ["z"], sampled=[toks[1]], drafts=[[toks[2], toks[3]]], starts=[2])
    _observe_step(runner, ["z"], sampled=[toks[2]], drafts=[[0, 0]], starts=[3])
    _observe_step(runner, [], sampled=[], drafts=None)
    assert not log.exists()


def test_without_a_log_path_nothing_is_scored(monkeypatch, tmp_path):
    monkeypatch.setenv("VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT", "2")
    monkeypatch.delenv("VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT_LOG", raising=False)
    runner = _runner_shell(k=2)
    _observe_step(runner, ["n"], sampled=[1], drafts=[[-1, -1]], is_prefill=True, counts=[2])
    _observe_step(runner, ["n"], sampled=[2], drafts=[[3, 4]], starts=[2])
    assert runner._glm5next_shadow_pending == []
    assert getattr(runner, "_glm5next_shadow_scorer_instance", None) is None


# ── the kwargs hook: the boundary id the prefill leg's last row takes ────────


def test_the_prefill_boundary_id_is_the_next_prompt_token_or_the_sampled_sentinel(monkeypatch):
    monkeypatch.setenv("VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT", "3")
    runner = _runner_shell(k=3)
    runner.requests = {"p": SimpleNamespace(prompt_token_ids=[11, 12, 13, 14, 15, 16, 17],
                                            num_prompt_tokens=7)}
    runner.input_batch.req_ids = ["p"]
    # Chunk 1 covers prompt[0:4]; the row at position 3 pairs with prompt[4] = 15.
    out = runner._glm5next_shadow_kwargs(
        is_prefill=True, request_ids=["p"], request_starts=[0], request_tokens=[4],
        synthetic=False, device=torch.device("cpu"), sampling_rows=1,
    )
    assert out["shadow_boundary_ids"].tolist() == [15]
    assert out["shadow_boundary_ids"].dtype == torch.int32
    # Chunk 2 covers prompt[4:7]: final, so the last row takes the in-graph sampled id.
    out = runner._glm5next_shadow_kwargs(
        is_prefill=True, request_ids=["p"], request_starts=[4], request_tokens=[3],
        synthetic=False, device=torch.device("cpu"), sampling_rows=1,
    )
    assert out["shadow_boundary_ids"].tolist() == [-1]
    # A warmup/capture step has no request: the sentinel, and no step is stashed.
    out = runner._glm5next_shadow_kwargs(
        is_prefill=True, request_ids=[None], request_starts=[0], request_tokens=[8],
        synthetic=True, device=torch.device("cpu"), sampling_rows=1,
    )
    assert out["shadow_boundary_ids"].tolist() == [-1]
    assert runner._glm5next_shadow_step is None
    # The decode leg carries no boundary.
    out = runner._glm5next_shadow_kwargs(
        is_prefill=False, request_ids=["p"], request_starts=[7], request_tokens=[1],
        synthetic=False, device=torch.device("cpu"), sampling_rows=1,
    )
    assert out == {}
    assert runner._glm5next_shadow_step is not None


def test_with_the_knob_off_the_hook_adds_nothing_and_the_output_passes_through(monkeypatch):
    monkeypatch.delenv("VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT", raising=False)
    runner = _runner_shell()
    out = runner._glm5next_shadow_kwargs(
        is_prefill=True, request_ids=["p"], request_starts=[0], request_tokens=[4],
        synthetic=False, device=torch.device("cpu"), sampling_rows=1,
    )
    assert out == {}
    ids = torch.tensor([1, 2, 3])
    assert runner._glm5next_shadow_take_output(ids) is ids
    pair = (ids, torch.tensor([[1.0]]))
    taken, drafts = runner._glm5next_shadow_take_output(pair), None
    assert taken is pair


def test_the_hook_engages_only_when_the_served_model_built_a_draft_head(monkeypatch):
    """The knob is GLM-specific; a model without ``mtp`` (another model, or the GLM root with
    the knob off at construction) must see its output untouched and no keyword added."""
    monkeypatch.setenv("VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT", "2")
    runner = _runner_shell(k=2, head=False)
    assert runner._glm5next_shadow_k() == 0
    pair = (torch.tensor([1, 2]), torch.tensor([[3, 4], [5, 6]]))
    assert runner._glm5next_shadow_take_output(pair) is pair
    out = runner._glm5next_shadow_kwargs(
        is_prefill=True, request_ids=["p"], request_starts=[0], request_tokens=[4],
        synthetic=False, device=torch.device("cpu"), sampling_rows=1,
    )
    assert out == {}
    assert not runner._glm5next_shadow_active()


def test_the_boundary_ids_cover_every_sampling_row_of_a_padded_prefill(monkeypatch):
    """The input builder pads ``sampling_positions`` to the request bucket by repeating the
    last real index; the boundary tensor is one entry per sampling row, padding rows taking
    the last real request's value, so duplicate indices write one value."""
    monkeypatch.setenv("VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT", "3")
    runner = _runner_shell(k=3)
    runner.requests = {"p": SimpleNamespace(prompt_token_ids=[11, 12, 13, 14, 15, 16, 17],
                                            num_prompt_tokens=7)}
    runner.input_batch.req_ids = ["p"]
    out = runner._glm5next_shadow_kwargs(
        is_prefill=True, request_ids=["p"], request_starts=[0], request_tokens=[4],
        synthetic=False, device=torch.device("cpu"), sampling_rows=3,
    )
    assert out["shadow_boundary_ids"].tolist() == [15, 15, 15]
    # Fewer sampling rows than requests is a geometry the translator never produces.
    with pytest.raises(ValueError, match="sampling row"):
        runner._glm5next_shadow_kwargs(
            is_prefill=True, request_ids=["p", "q"], request_starts=[0, 0],
            request_tokens=[4, 4], synthetic=False, device=torch.device("cpu"),
            sampling_rows=1,
        )


def test_take_output_peels_the_draft_ids_off_the_graph_output(monkeypatch):
    monkeypatch.setenv("VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT", "2")
    runner = _runner_shell(k=2)
    ids = torch.tensor([5, 6], dtype=torch.int32)
    drafts = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    out = runner._glm5next_shadow_take_output((ids, drafts))
    assert out is ids
    assert torch.equal(runner._glm5next_shadow_last_drafts, drafts)
    stream = torch.zeros(2, 3)
    out = runner._glm5next_shadow_take_output((ids, drafts, stream))
    assert isinstance(out, tuple) and out[0] is ids and out[1] is stream


# ── the unpack site: _execute_model_forward peels the drafts and observes the step ──


def test_execute_model_forward_peels_the_drafts_and_observes_the_step(monkeypatch, tmp_path):
    """With the knob on the GLM root returns ``(sampled_ids, draft_ids)``; the unpack hands the
    sampled ids on unchanged and queues the step for scoring, so the next step's observe can
    resolve it. Everything around the model call is a shell: this reads the two hunks only."""
    import contextlib

    from vllm_neuron.vllm.worker import neuron_model_runner as module

    log = tmp_path / "shadow.jsonl"
    monkeypatch.setenv("VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT", "2")
    monkeypatch.setenv("VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT_LOG", str(log))
    runner = _runner_shell(k=2, req_ids=["u"])
    runner.use_async_scheduling = False
    runner.device = torch.device("cpu")
    runner.on_device_sampling = True
    runner.is_eagle3_spec = False
    runner.drafter = None
    runner._debug_logits_dir = None
    runner._layer_stream_dump_dir = None
    runner._target_tensor_capture = None
    runner._tensor_replacer = None
    runner.enable_prompt_embeds = False
    runner.supports_mm_inputs = False
    runner.input_batch.num_reqs = 1
    runner.input_batch.sampling_metadata = object()
    runner.vllm_config = SimpleNamespace(model_config=SimpleNamespace(model="tiny"))
    monkeypatch.setattr(module, "model_forward_context", lambda config: contextlib.nullcontext())
    monkeypatch.setattr(module, "build_sampling_params_tensor",
                        lambda metadata, num_reqs, device: torch.zeros(num_reqs, 3))
    monkeypatch.setattr(NeuronModelRunner, "_snapshot_capture_context",
                        lambda self, positions, spec: contextlib.nullcontext())
    monkeypatch.setattr(NeuronModelRunner, "_model_is_async_spec_decoded", lambda self: False)
    monkeypatch.setattr(NeuronModelRunner, "_maybe_replicate_for_spec_decode",
                        lambda self, params, spec: params)
    seen_kwargs: list[dict] = []

    def converter(kwargs):
        seen_kwargs.append(kwargs)
        # The real converter stashes the step through _glm5next_shadow_kwargs; do the same.
        runner._glm5next_shadow_kwargs(is_prefill=False, request_ids=["u"], request_starts=[7],
                                       request_tokens=[1], synthetic=False, device=torch.device("cpu"),
                                       sampling_rows=1)
        return kwargs

    monkeypatch.setattr(runner, "_glm5next_model_kwargs", converter)
    sampled = torch.tensor([42], dtype=torch.int32)
    drafts = torch.tensor([[43, 44]], dtype=torch.int32)
    runner.model = lambda **kwargs: (sampled, drafts)
    runner.model.mtp = object()          # the served root built its draft head
    metadata = {"layer": {"max_query_len": 1, "decode_token_threshold": 1,
                          "block_table_tensor": torch.zeros(1, 4, dtype=torch.int32)}}

    out, aux, last = runner._execute_model_forward(
        torch.tensor([41]), torch.tensor([7]), torch.tensor([0]), metadata, None,
    )
    assert aux is None and last is None
    assert torch.equal(out, sampled), "the sampled ids pass through unchanged"
    assert len(runner._glm5next_shadow_pending) == 1
    step_no, step, queued_sampled, queued_drafts = runner._glm5next_shadow_pending[0]
    assert step["request_ids"] == ["u"] and step["positions"] == [7]
    assert torch.equal(queued_sampled, sampled) and torch.equal(queued_drafts, drafts)
    # The next step resolves it; with 'u' gone, the record is flushed with nothing scored.
    runner.input_batch.req_ids = []
    runner._glm5next_shadow_observe(torch.tensor([], dtype=torch.int32), None, is_prefill=False)
    written = _read_log(log)
    assert [(r["req_id"], r["step"], r["position"], r["drafts"]) for r in written] == [("u", 0, 7, [43, 44])]
