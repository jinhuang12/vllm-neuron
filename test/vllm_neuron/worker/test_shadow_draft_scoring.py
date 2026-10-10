# SPDX-License-Identifier: Apache-2.0
"""Shadow-draft scoring: drafts are scored against the tokens sampled k steps later.

The traced MTP head of GLM-5.3-Flash runs the layer-45 draft beside the decode graph and
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

import inspect
import json
import re
import traceback
from types import SimpleNamespace

import pytest
import torch

from vllm.distributed import parallel_state as dist_state

from vllm_neuron.vllm.worker.neuron_model_runner import (
    AsyncNeuronModelRunnerOutput,
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


class _DeviceTensor(torch.Tensor):
    """A tensor on the device: a host read (``.cpu()``, ``.tolist()``, ``.item()``, ``.to("cpu")``)
    raises while ``armed``, and every read is recorded with the function names on its stack
    (``_DeviceTensor.reads``). The glue must never read one on the worker's main thread; a
    test disarms a future once "the output thread" is the one reading it back.

    Tensor operations on it return plain tensors (the next step consumes the sampled ids as
    its device input), so only the host reads are watched.
    """

    __torch_function__ = torch._C._disabled_torch_function_impl
    reads: list[tuple[str, list[str]]] = []

    @staticmethod
    def of(tensor: torch.Tensor, *, armed: bool = True, label: str = "") -> "_DeviceTensor":
        device = tensor.as_subclass(_DeviceTensor)
        device.armed = armed
        device.label = label
        return device

    def _read(self) -> torch.Tensor:
        frames = [frame.name for frame in traceback.extract_stack()]
        _DeviceTensor.reads.append((getattr(self, "label", ""), frames))
        if self.armed:
            raise RuntimeError("device read on the host")
        return self.as_subclass(torch.Tensor)

    def cpu(self):
        return self._read()

    def tolist(self):
        return self._read().tolist()

    def item(self):
        return self._read().item()

    def to(self, *args, **kwargs):
        return self._read().to(*args, **kwargs)


def _runner_shell(*, k: int = K, req_ids=None, rank: int | None = 0, head: bool = True,
                  async_scheduling: bool = False):
    """A runner with the state the shadow hooks read.

    ``head`` = the served root built ``mtp``. ``rank`` is the host-side tensor-parallel rank a
    built runner reads once from its group; ``None`` leaves it to be read. ``rank_tensor`` is
    a device tensor that raises on any host read, as the production one is a device tensor.
    """
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.input_batch = SimpleNamespace(req_ids=list(req_ids or []))
    runner.requests = {}
    runner.rank_tensor = _DeviceTensor.of(
        torch.tensor(0 if rank is None else rank, dtype=torch.int32), label="rank_tensor"
    )
    if rank is not None:
        runner._glm5next_shadow_rank_cache = rank
    # The root records the k it built the head for; the runner reads it there.
    runner.model = SimpleNamespace(mtp=object() if head else None, draft_k=k if head else 0)
    runner.use_async_scheduling = async_scheduling
    # get_output() writes the ids back into the batch; the batch is not under test here.
    runner._update_batch_state_with_samples = lambda *args, **kwargs: None
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
    """One step as the runner sees it: kwargs hook (stashes the step), then the unpack.

    ``sampled`` and ``drafts`` are lists (the host copy a synchronous runner made) or tensors
    handed through as they are (a device future under async scheduling).
    """
    runner.input_batch.req_ids = list(req_ids)
    n = len(req_ids)
    starts = list(starts) if starts is not None else [0] * n
    counts = list(counts) if counts is not None else [1] * n
    runner._glm5next_shadow_kwargs(
        is_prefill=is_prefill, request_ids=list(req_ids), request_starts=starts,
        request_tokens=counts, synthetic=False, device=torch.device("cpu"), sampling_rows=n,
    )
    sampled_t = sampled if torch.is_tensor(sampled) else torch.tensor(sampled, dtype=torch.int32)
    if drafts is None or torch.is_tensor(drafts):
        drafts_t = drafts
    else:
        drafts_t = torch.tensor(drafts, dtype=torch.int32)
    runner._glm5next_shadow_observe(sampled_t, drafts_t, is_prefill=is_prefill)


def _async_output(runner, req_ids, sampled) -> AsyncNeuronModelRunnerOutput:
    """The step's async output the way ``sample_tokens`` builds it: one per step, right after
    the unpack, so it claims the step the glue just stashed."""
    return AsyncNeuronModelRunnerOutput(
        model_runner_output=SimpleNamespace(req_ids=list(req_ids), sampled_token_ids=sampled),
        model_runner=runner,
    )


def _device_step(runner, req_ids, *, sampled, drafts, **kwargs):
    """One async step: futures the main thread must not read, observed, then its output."""
    sampled_t = _DeviceTensor.of(torch.tensor(sampled, dtype=torch.int32))
    drafts_t = None if drafts is None else _DeviceTensor.of(torch.tensor(drafts, dtype=torch.int32))
    _observe_step(runner, req_ids, sampled=sampled_t, drafts=drafts_t, **kwargs)
    return _async_output(runner, req_ids, sampled_t), sampled_t, drafts_t


def _materialize(output, sampled_t, drafts_t):
    """The output thread's turn: the device has finished, the ids are read back."""
    sampled_t.armed = False
    if drafts_t is not None:
        drafts_t.armed = False
    return output.get_output().sampled_token_ids


def _read_log(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _records(path) -> list[dict]:
    """The records written so far; the scorer opens the file before its first record."""
    return _read_log(path) if path.exists() else []


def test_the_rank_is_read_on_the_host_from_the_tensor_parallel_group(monkeypatch):
    """``rank_tensor`` lives on the device; reading it is a device-to-host copy on the worker's
    main thread, so the rank comes from the tensor-parallel group instead, once. A process
    with no model-parallel group is rank 0."""
    runner = _runner_shell(k=2, rank=None)
    monkeypatch.setattr(dist_state, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(dist_state, "get_tp_group", lambda: SimpleNamespace(rank_in_group=3))
    assert runner._glm5next_shadow_rank() == 3
    monkeypatch.setattr(dist_state, "get_tp_group", lambda: SimpleNamespace(rank_in_group=0))
    assert runner._glm5next_shadow_rank() == 3, "read once, then held"
    alone = _runner_shell(k=2, rank=None)
    monkeypatch.setattr(dist_state, "model_parallel_is_initialized", lambda: False)
    assert alone._glm5next_shadow_rank() == 0


def test_under_async_scheduling_the_main_thread_reads_no_future_and_the_output_scores_the_step(
    monkeypatch, tmp_path
):
    """A knob-5 server hung at the first decode: the glue read the previous step's
    sampled ids back from the device on the worker's main thread while the output thread was
    reading the same future inside ``get_output()``. Now the main thread stashes the step
    (its bookkeeping and its draft future, unread); the step's ``AsyncNeuronModelRunnerOutput``
    claims it; ``get_output()`` scores it from the ids it read back, on whichever thread
    materializes, and reads the draft future there, once, after the sampled ids of the same
    execution. The log fills as outputs are materialized, not as steps are dispatched."""
    log = tmp_path / "shadow.jsonl"
    monkeypatch.setenv("VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT", "2")
    monkeypatch.setenv("VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT_LOG", str(log))
    runner = _runner_shell(k=2, req_ids=["r"], async_scheduling=True)
    toks = _tokens(3, 8)
    # Prefill (final chunk of a 5-token prompt), then three decodes at positions 5, 6, 7.
    plan = [
        dict(sampled=[toks[0]], drafts=[[-1, -1]], is_prefill=True, starts=[0], counts=[5]),
        dict(sampled=[toks[1]], drafts=[[toks[2], toks[3]]], starts=[5]),
        dict(sampled=[toks[2]], drafts=[[toks[3], 0]], starts=[6]),
        dict(sampled=[toks[3]], drafts=[[0, 0]], starts=[7]),
    ]
    steps = [_device_step(runner, ["r"], **spec) for spec in plan]
    # Every step was dispatched; no future was read; each step's record left with its output.
    assert _records(log) == []
    assert runner._glm5next_shadow_inflight is None
    assert all(sampled.armed and drafts.armed for _out, sampled, drafts in steps)
    # The output thread materializes the steps in order.
    ids = [_materialize(*step) for step in steps]
    assert ids == [[[toks[0]]], [[toks[1]]], [[toks[2]]], [[toks[3]]]]
    written = _read_log(log)
    # Step 1's two drafts were scored against steps 2 and 3; steps 2 and 3 still wait.
    assert [(r["step"], r["position"], r["scored"], r["accepted_prefix_len"]) for r in written] == [
        (1, 5, 2, 2)
    ]
    runner.ensure_kv_transfer_shutdown()
    written = _read_log(log)
    assert [(r["step"], r["scored"], r["accepted_prefix_len"]) for r in written] == [
        (1, 2, 2), (2, 1, 1), (3, 0, 0)
    ]
    assert runner._glm5next_shadow_log_handle.closed


def test_every_device_read_of_the_glue_happens_inside_get_output(monkeypatch, tmp_path):
    """The regression guard for the hang's class: with every tensor readable (nothing armed),
    three async steps are driven and every host read of a device tensor is traced. Each one
    sits under ``get_output()`` (the runner's own read of the sampled ids, and the glue's read
    of the drafts through ``_glm5next_shadow_commit``), and ``rank_tensor`` is never read."""
    log = tmp_path / "shadow.jsonl"
    monkeypatch.setenv("VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT", "2")
    monkeypatch.setenv("VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT_LOG", str(log))
    runner = _runner_shell(k=2, req_ids=["r"], rank=None, async_scheduling=True)
    monkeypatch.setattr(dist_state, "model_parallel_is_initialized", lambda: False)
    _DeviceTensor.reads.clear()
    outputs = []
    for n, (sampled, drafts) in enumerate([([10], [[11, 12]]), ([11], [[12, 13]]), ([12], [[13, 0]])]):
        sampled_t = _DeviceTensor.of(torch.tensor(sampled, dtype=torch.int32), armed=False, label="sampled")
        drafts_t = _DeviceTensor.of(torch.tensor(drafts, dtype=torch.int32), armed=False, label="drafts")
        _observe_step(runner, ["r"], sampled=sampled_t, drafts=drafts_t, starts=[4 + n])
        outputs.append(_async_output(runner, ["r"], sampled_t))
    assert _DeviceTensor.reads == [], "the main thread read nothing while dispatching"
    for output in outputs:
        output.get_output()
    runner.ensure_kv_transfer_shutdown()
    labels = sorted({label for label, _frames in _DeviceTensor.reads})
    assert labels == ["drafts", "sampled"], labels
    for label, frames in _DeviceTensor.reads:
        assert "get_output" in frames, (label, frames)
        if label == "drafts":
            assert "_glm5next_shadow_commit" in frames, frames
    assert [r["step"] for r in _read_log(log)] == [0, 1, 2]


def test_the_shadow_glue_reads_the_device_only_in_its_commit_path():
    """Grep-level guard: of every ``_glm5next_shadow_*`` method, only the commit path that
    ``get_output()`` calls (or a synchronous runner calls on its host copy) contains a host read
    (``.cpu()``, ``.item()``, ``.tolist()``, ``.numpy()``)."""
    allowed = {"_glm5next_shadow_commit", "_glm5next_shadow_ids"}
    pattern = re.compile(r"\.(cpu|item|tolist|numpy)\(")
    offenders = {}
    for name in dir(NeuronModelRunner):
        if not name.startswith("_glm5next_shadow_"):
            continue
        source = inspect.getsource(getattr(NeuronModelRunner, name))
        body = source.split('"""', 2)[-1] if source.count('"""') >= 2 else source
        hits = pattern.findall(body)
        if hits and name not in allowed:
            offenders[name] = hits
    assert offenders == {}, offenders
    assert pattern.findall(inspect.getsource(NeuronModelRunner._glm5next_shadow_commit)), "the drafts are read in the commit"


def test_a_second_get_output_on_the_same_step_scores_nothing_twice(monkeypatch, tmp_path):
    """Both the output thread and the worker's fallback path may call ``get_output()`` on one
    step; the step is scored once."""
    log = tmp_path / "shadow.jsonl"
    monkeypatch.setenv("VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT", "1")
    monkeypatch.setenv("VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT_LOG", str(log))
    runner = _runner_shell(k=1, req_ids=["r"], async_scheduling=True)
    first = _device_step(runner, ["r"], sampled=[10], drafts=[[11]], starts=[4])
    second = _device_step(runner, ["r"], sampled=[11], drafts=[[12]], starts=[5])
    _materialize(*first)
    assert first[0].get_output().sampled_token_ids == [[10]], "the list, read back once"
    _materialize(*second)
    _materialize(*second)
    assert [(r["step"], r["actual"]) for r in _read_log(log)] == [(0, [11])]
    assert runner._glm5next_shadow_scorer_instance.tracked == {"r"}


def test_outputs_materialized_out_of_dispatch_order_are_scored_in_dispatch_order(monkeypatch, tmp_path):
    """The output thread works through the steps in order, but the worker's fallback path may
    materialize the newest step first; the scorer sees steps in dispatch order regardless."""
    log = tmp_path / "shadow.jsonl"
    monkeypatch.setenv("VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT", "1")
    monkeypatch.setenv("VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT_LOG", str(log))
    runner = _runner_shell(k=1, req_ids=["r"], async_scheduling=True)
    steps = [
        _device_step(runner, ["r"], sampled=[20], drafts=[[21]], starts=[4]),
        _device_step(runner, ["r"], sampled=[21], drafts=[[22]], starts=[5]),
        _device_step(runner, ["r"], sampled=[22], drafts=[[0]], starts=[6]),
    ]
    _materialize(*steps[1])
    assert _records(log) == [], "step 1 waits for step 0"
    _materialize(*steps[2])
    assert _records(log) == []
    _materialize(*steps[0])
    written = _read_log(log)
    assert [(r["step"], r["position"], r["actual"], r["accepted_prefix_len"]) for r in written] == [
        (0, 4, [21], 1), (1, 5, [22], 1)
    ]


def test_an_intermediate_prefill_chunk_output_advances_the_order_and_scores_nothing(monkeypatch, tmp_path):
    """An all-partial prefill output is drained in ``get_output()`` without ids; the step takes
    its turn in the order and leaves no record, so the steps behind it are not held."""
    log = tmp_path / "shadow.jsonl"
    monkeypatch.setenv("VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT", "1")
    monkeypatch.setenv("VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT_LOG", str(log))
    runner = _runner_shell(k=1, req_ids=["c"], async_scheduling=True)
    runner.requests = {"c": SimpleNamespace(prompt_token_ids=list(range(100, 108)), num_prompt_tokens=8)}
    # Chunk 1 of 2: rows 0..3 of an 8-token prompt. Its output is all-partial.
    chunk_sampled = _DeviceTensor.of(torch.tensor([5], dtype=torch.int32))
    chunk_drafts = _DeviceTensor.of(torch.tensor([[-1]], dtype=torch.int32))
    _observe_step(runner, ["c"], sampled=chunk_sampled, drafts=chunk_drafts, is_prefill=True,
                  starts=[0], counts=[4])
    chunk_out = AsyncNeuronModelRunnerOutput(
        model_runner_output=SimpleNamespace(req_ids=["c"], sampled_token_ids=chunk_sampled),
        model_runner=runner, partial_prefill_req_ids={"c"},
    )
    final = _device_step(runner, ["c"], sampled=[107], drafts=[[-1]], is_prefill=True, starts=[4], counts=[4])
    decode = _device_step(runner, ["c"], sampled=[30], drafts=[[31]], starts=[8])
    _materialize(*final)
    _materialize(*decode)
    assert _records(log) == [], "the chunk's turn has not come"
    chunk_sampled.armed = False
    assert chunk_out.get_output().sampled_token_ids == [[]]
    runner.ensure_kv_transfer_shutdown()
    assert [(r["step"], r["position"], r["actual"]) for r in _read_log(log)] == [(2, 8, [])]


def test_without_async_scheduling_each_step_is_scored_as_it_is_observed(monkeypatch, tmp_path):
    """A synchronous runner moved the sampled ids to the host before the glue runs; the step is
    scored at once, the drafts read back once, and a request that leaves is retired."""
    log = tmp_path / "shadow.jsonl"
    monkeypatch.setenv("VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT", "3")
    monkeypatch.setenv("VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT_LOG", str(log))
    runner = _runner_shell(k=3)
    tokens = _tokens(6, 10)
    _observe_step(runner, ["q"], sampled=[tokens[0]], drafts=[[-1, -1, -1]], is_prefill=True,
                  starts=[0], counts=[5])
    for t in range(1, 5):
        future = tokens[t + 1 : t + 4]
        _observe_step(runner, ["q"], sampled=[tokens[t]],
                      drafts=[_planted_drafts(future, 2)[:3]], starts=[4 + t], counts=[1])
        if t == 4:
            # Step 1's record needs the tokens of steps 2, 3 and 4: complete as step 4 is observed.
            assert [r["step"] for r in _read_log(log)] == [1]
    assert runner._glm5next_shadow_inflight is None
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
    a, b = _tokens(1, 6), _tokens(2, 6)
    _observe_step(runner, ["a"], sampled=[a[0]], drafts=[[-1, -1]], is_prefill=True, counts=[3])
    _observe_step(runner, ["a"], sampled=[a[1]], drafts=[[a[2], a[3]]], starts=[3])
    # 'a' finishes; 'b' takes row 0. b's prefill id must not score a's drafts.
    _observe_step(runner, ["b"], sampled=[b[0]], drafts=[[-1, -1]], is_prefill=True, counts=[3])
    _observe_step(runner, ["b"], sampled=[b[1]], drafts=[[b[2], b[3]]], starts=[3])
    _observe_step(runner, ["b"], sampled=[b[2]], drafts=[[0, 0]], starts=[4])
    _observe_step(runner, [], sampled=[], drafts=None)
    written = _read_log(log)
    by_req = {r["req_id"]: r for r in written if r["drafts"] != [0, 0]}
    assert by_req["a"]["scored"] == 0 and by_req["a"]["actual"] == []
    assert by_req["b"]["scored"] == 1 and by_req["b"]["actual"] == [b[2]]
    assert by_req["b"]["accepted_prefix_len"] == 1


def test_intermediate_prefill_chunks_and_sentinel_drafts_leave_no_trace(monkeypatch, tmp_path):
    """A chunk that is not the prompt's last samples an id that is not a token of the sequence;
    the prefill leg's -1 drafts are not drafts. Neither reaches the scorer."""
    log = tmp_path / "shadow.jsonl"
    monkeypatch.setenv("VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT", "2")
    monkeypatch.setenv("VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT_LOG", str(log))
    runner = _runner_shell(k=2)
    runner.requests = {"c": SimpleNamespace(prompt_token_ids=list(range(100, 108)), num_prompt_tokens=8)}
    # Chunk 1 of 2: rows 0..3 of an 8-token prompt; its sampled id is noise.
    _observe_step(runner, ["c"], sampled=[5], drafts=[[-1, -1]], is_prefill=True, starts=[0], counts=[4])
    # Chunk 2: rows 4..7, the last; its sampled id is x_8.
    _observe_step(runner, ["c"], sampled=[107], drafts=[[-1, -1]], is_prefill=True, starts=[4], counts=[4])
    _observe_step(runner, ["c"], sampled=[30], drafts=[[31, 32]], starts=[8])
    _observe_step(runner, ["c"], sampled=[31], drafts=[[0, 0]], starts=[9])
    _observe_step(runner, [], sampled=[], drafts=None)
    written = _read_log(log)
    history = [31]  # what step 2's drafts were scored against
    assert [r["step"] for r in written] == [2, 3]
    assert written[0]["actual"] == history and written[0]["accepted_prefix_len"] == 1
    assert written[0]["position"] == 8


def test_shutdown_scores_the_rest_and_closes_the_log(monkeypatch, tmp_path):
    """The worker's shutdown reaches the runner through ``ensure_kv_transfer_shutdown``: every
    tracked request is retired with its pending drafts scored over what arrived, and the log
    handle is closed."""
    log = tmp_path / "shadow.jsonl"
    monkeypatch.setenv("VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT", "2")
    monkeypatch.setenv("VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT_LOG", str(log))
    runner = _runner_shell(k=2, req_ids=["r"])
    # Three decode steps. Step 0's two drafts need step 2's token: complete as step 2 is
    # observed; steps 1 and 2 wait for tokens that never come.
    _observe_step(runner, ["r"], sampled=[10], drafts=[[11, 12]], starts=[8], counts=[1])
    _observe_step(runner, ["r"], sampled=[11], drafts=[[12, 13]], starts=[9], counts=[1])
    _observe_step(runner, ["r"], sampled=[12], drafts=[[13, 0]], starts=[10], counts=[1])
    assert [r["step"] for r in _read_log(log)] == [0]
    runner.ensure_kv_transfer_shutdown()
    records = _read_log(log)
    assert [r["step"] for r in records] == [0, 1, 2]
    assert [r["scored"] for r in records] == [2, 1, 0]
    assert [r["accepted_prefix_len"] for r in records] == [2, 1, 0]
    assert runner._glm5next_shadow_log_handle.closed
    assert runner._glm5next_shadow_scorer_instance is None
    # Idempotent: a second shutdown neither writes nor fails.
    runner.ensure_kv_transfer_shutdown()
    assert len(_read_log(log)) == 3


def test_shutdown_drops_a_step_whose_output_was_never_read_back_and_closes_the_log(
    monkeypatch, tmp_path, caplog
):
    """An abort mid-step reaches shutdown with a dispatched step whose output no thread
    materialized. Shutdown does not read the device (the runtime may be gone, and a read
    here is the hang this file pins): the step is dropped with a warning naming it, every
    request retires over what did arrive, the handle closes and the scorer is dropped."""
    log = tmp_path / "shadow.jsonl"
    monkeypatch.setenv("VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT", "2")
    monkeypatch.setenv("VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT_LOG", str(log))
    runner = _runner_shell(k=2, req_ids=["r"], async_scheduling=True)
    steps = [
        _device_step(runner, ["r"], sampled=[10], drafts=[[11, 12]], starts=[8]),
        _device_step(runner, ["r"], sampled=[11], drafts=[[12, 13]], starts=[9]),
        _device_step(runner, ["r"], sampled=[12], drafts=[[13, 0]], starts=[10]),
    ]
    _materialize(*steps[0])
    _materialize(*steps[1])
    with caplog.at_level("WARNING"):
        runner.ensure_kv_transfer_shutdown()
    assert steps[2][1].armed and steps[2][2].armed, "shutdown read nothing from the device"
    records = _read_log(log)
    # Step 2's token never arrived: step 0 is scored over step 1's token only, step 1 over none.
    assert [r["step"] for r in records] == [0, 1]
    assert [r["scored"] for r in records] == [1, 0]
    assert [r["accepted_prefix_len"] for r in records] == [1, 0]
    assert runner._glm5next_shadow_log_handle.closed
    assert runner._glm5next_shadow_scorer_instance is None
    assert any("shadow draft" in rec.message and "step 2" in rec.message for rec in caplog.records)


def test_shutdown_scores_nothing_across_a_lost_step(monkeypatch, tmp_path, caplog):
    """A step lost in the middle (its output never materialized) leaves the histories without
    its token; a record that waited for it must not be scored against the token of the step
    after the gap as if it were the next one. Shutdown scores what arrived before the gap,
    starts every request afresh behind it, and names the lost step."""
    log = tmp_path / "shadow.jsonl"
    monkeypatch.setenv("VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT", "1")
    monkeypatch.setenv("VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT_LOG", str(log))
    runner = _runner_shell(k=1, req_ids=["r"], async_scheduling=True)
    # Step 0 drafts step 2's token: right only if step 1's token were skipped, which is the
    # mis-scoring a lost step 1 must not produce.
    steps = [
        _device_step(runner, ["r"], sampled=[10], drafts=[[12]], starts=[4]),
        _device_step(runner, ["r"], sampled=[11], drafts=[[12]], starts=[5]),
        _device_step(runner, ["r"], sampled=[12], drafts=[[13]], starts=[6]),
    ]
    _materialize(*steps[0])
    _materialize(*steps[2])
    with caplog.at_level("WARNING"):
        runner.ensure_kv_transfer_shutdown()
    records = _read_log(log)
    assert [(r["step"], r["scored"], r["actual"], r["accepted_prefix_len"]) for r in records] == [
        (0, 0, [], 0), (2, 0, [], 0)
    ]
    assert any("step 1" in rec.message for rec in caplog.records)


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
    runner = _runner_shell(k=2, async_scheduling=True)
    _observe_step(runner, ["n"], sampled=[1], drafts=[[-1, -1]], is_prefill=True, counts=[2])
    _observe_step(runner, ["n"], sampled=[2], drafts=[[3, 4]], starts=[2])
    assert runner._glm5next_shadow_inflight is None
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
    """The knob off at construction is a root with no head and ``draft_k = 0``; the runner
    reads that record (never the environment at step time), so nothing is added."""
    monkeypatch.delenv("VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT", raising=False)
    runner = _runner_shell(head=False)
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
    sampled ids on unchanged and observes the step for scoring. Everything around the model
    call is a shell: this reads the two hunks only."""
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
    runner.model.draft_k = 2             # ... for the knob's k, recorded at construction
    metadata = {"layer": {"max_query_len": 1, "decode_token_threshold": 1,
                          "block_table_tensor": torch.zeros(1, 4, dtype=torch.int32)}}

    out, aux, last = runner._execute_model_forward(
        torch.tensor([41]), torch.tensor([7]), torch.tensor([0]), metadata, None,
    )
    assert aux is None and last is None
    assert torch.equal(out, sampled), "the sampled ids pass through unchanged"
    # A synchronous runner: the step was scored as it was observed and its drafts wait.
    scorer = runner._glm5next_shadow_scorer_instance
    assert scorer.tracked == {"u"} and runner._glm5next_shadow_inflight is None
    assert _records(log) == []
    # The next step has 'u' gone: the record is flushed with nothing scored.
    _observe_step(runner, [], sampled=[], drafts=None)
    written = _read_log(log)
    assert [(r["req_id"], r["step"], r["position"], r["drafts"]) for r in written] == [("u", 0, 7, [43, 44])]
