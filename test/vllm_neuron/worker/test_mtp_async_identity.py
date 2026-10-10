# SPDX-License-Identifier: Apache-2.0
"""The async drafter's greedy output is the synchronous drafter's, token for token.

The control is the synchronous drafter: the same
tiny root, the same prompts, the same ``k``, served once with ``--no-async-scheduling``
(the shipped opt-in path, ``test_mtp_e2e_spec``'s ``_generate``) and once with
``--async-scheduling`` and ``VLLM_NEURON_GLM5NEXT_MTP_ASYNC=1``, driven the way the async
engine drives the worker: the scheduler hands each step the position it would stand at
had the previous step kept every row, reserves ``k`` placeholder drafts, and applies a
step's rejections to its count only after the next step was scheduled; the worker's
outputs are device futures materialised one step late. Both runs must return the plain
run's ids (greedy rejection sampling is lossless per step by id equality) with the same
rows kept at every step; the async run schedules one step more per prompt -- the step in
flight when the finishing output is applied.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/worker/test_mtp_async_identity.py
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm.engine.arg_utils import EngineArgs

from vllm_neuron.vllm.worker.neuron_model_runner import (
    AsyncNeuronModelRunnerOutput,
    NeuronModelRunner,
)

from vllm_neuron.vllm.worker.neuron_worker import NeuronWorker

from test.vllm_neuron.model.glm5_next import test_mtp_e2e_spec as spec
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_e2e as e2e
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_first_request as fr
from test.vllm_neuron.worker import test_mtp_proposer as proposer_tests

pytestmark = [pytest.mark.forked]

K = 3
T = 1 + K


def _async_config(k: int):
    """``test_mtp_e2e_spec._config(k)`` with async scheduling on (the proposer's knob admits it)."""
    neuron_config = {
        "num_batched_tokens_buckets": [spec.PREFILL_BUCKET, spec.MAX_MODEL_LEN],
        "num_seqs_buckets": [1],
        "on_device_sampling_config": {},
    }
    return EngineArgs(
        model=str(fr.FIXTURE),
        skip_tokenizer_init=True,
        max_model_len=spec.MAX_MODEL_LEN,
        max_num_seqs=1,
        max_num_batched_tokens=spec.MAX_MODEL_LEN,
        block_size=spec.PAGE,
        enforce_eager=True,
        enable_prefix_caching=False,
        async_scheduling=True,
        speculative_config={"method": "mtp", "num_speculative_tokens": k},
        additional_config={"neuron_config": neuron_config},
    ).create_engine_config()


def _materialize(output) -> list[int]:
    """One step's ids, read back the way the engine's output thread reads them."""
    assert isinstance(output, AsyncNeuronModelRunnerOutput), type(output)
    rows = output.get_output().sampled_token_ids
    assert len(rows) == 1, rows
    return [int(value) for value in rows[0]]


def _scheduler_drafts(runner: NeuronModelRunner) -> list[int]:
    """What the async scheduler reserves next step: ``k`` placeholders for a tensor proposal,
    the lists otherwise (``take_draft_token_ids``)."""
    proposed = runner.take_draft_token_ids()
    return [] if proposed is None else [int(value) for value in proposed.draft_token_ids[0]]


def _generate_async(runner, req: str, prompt: list[int], *, tokens: int, finished: set, prefill=None):
    """Prefill, then decode until ``tokens`` are generated, as the async engine drives it.

    ``handed`` is the scheduler's count: advanced by the whole step when it is scheduled,
    pulled back by a step's rejections only once that step's output is applied, which
    happens after the following step was scheduled (``step_with_batch_queue``).
    Returns ``(ids, decode steps scheduled, async steps, sync fallbacks)``. The steps
    scheduled are one more than the steps whose output reached ``tokens``: when that
    output is applied, the next step is already in flight -- the async scheduler
    dispatched it before it could know the request was finished.
    ``prefill(runner, req, prompt, groups, blocks, finished)`` drives the prefill and returns
    its (pending) output; the default is the one-chunk prefill of ``spec._prefill``.
    """
    groups = fr._groups(runner)
    blocks = list(range(spec.FIRST_BLOCK, spec.FIRST_BLOCK + -(-len(prompt) // spec.PAGE)))
    if prefill is None:
        _, pending = fr._step(runner, spec._prefill(req, prompt, groups, blocks, finished))
    else:
        pending = prefill(runner, req, prompt, groups, blocks, finished)
    pending_rows = 1
    handed = len(prompt)
    generated: list[int] = []
    drafts = _scheduler_drafts(runner)
    steps = 0
    async_before, sync_before = runner._async_steps, runner._sync_fallback_steps
    while True:
        rows = 1 + len(drafts)
        needed = -(-(handed + rows) // spec.PAGE)
        new = list(range(spec.FIRST_BLOCK + len(blocks), spec.FIRST_BLOCK + needed))
        blocks += new
        _, output = fr._step(runner, spec._decode(req, handed, len(generated), groups, drafts, new))
        steps += 1
        handed += rows
        # Apply the previous step's output, one step late.
        kept = _materialize(pending)
        assert 1 <= len(kept) <= pending_rows, (kept, pending_rows)
        generated += kept
        handed -= pending_rows - len(kept)
        pending, pending_rows = output, rows
        drafts = _scheduler_drafts(runner)
        if len(generated) >= tokens:
            break
    kept = _materialize(pending)
    generated += kept
    assert all(len(ids) == K for ids in [drafts] if ids), drafts
    return (
        generated[:tokens], steps,
        runner._async_steps - async_before, runner._sync_fallback_steps - sync_before,
    )


def test_the_async_drafters_greedy_output_is_the_synchronous_drafters(tmp_path, monkeypatch):
    e2e._require_cpu_mode()
    fr._declaring_a_sampler(monkeypatch)
    prompts = spec._prompts()
    monkeypatch.delenv(spec.KNOB, raising=False)
    monkeypatch.delenv(proposer_tests.ASYNC_KNOB, raising=False)
    references = spec._plain_references(tmp_path, prompts)

    # The control: the synchronous drafter, the shipped opt-in path.
    monkeypatch.setenv(spec.KNOB, str(K))
    sync_config = spec._config(K)
    (tmp_path / "sync").mkdir()
    control = []
    with fr._parallel_state(tmp_path / "sync", sync_config):
        root, _ = spec._root()
        runner = spec._runner(sync_config, root)
        assert runner.drafter.async_steps is False
        finished: set = set()
        for i, prompt in enumerate(prompts):
            ids, steps = spec._generate(runner, f"sync-{i}", prompt, tokens=spec.GENERATED, finished=finished)
            finished = {f"sync-{i}"}
            assert ids == references[i][:spec.GENERATED], i
            control.append((ids, steps))

    # The async drafter.
    monkeypatch.setenv(proposer_tests.ASYNC_KNOB, "1")
    async_config = _async_config(K)
    assert async_config.scheduler_config.async_scheduling is True
    (tmp_path / "async").mkdir()
    with fr._parallel_state(tmp_path / "async", async_config):
        root, _ = spec._root()
        runner = spec._runner(async_config, root)
        assert runner.use_async_scheduling and runner.drafter.async_steps is True
        finished = set()
        counts: list[tuple[int, int, int]] = []
        for i, prompt in enumerate(prompts):
            ids, steps, async_steps, fallbacks = _generate_async(
                runner, f"async-{i}", prompt, tokens=spec.GENERATED, finished=finished
            )
            finished = {f"async-{i}"}
            assert ids == control[i][0], (i, ids, control[i][0])
            # The same ids from the same kept rows per step reach ``tokens`` at the
            # synchronous drafter's step; the async scheduler has one more step in
            # flight when that step's output is applied (``_generate_async``).
            assert steps == control[i][1] + 1, (i, steps, control[i][1])
            counts.append((i, control[i][1], steps))
            # Every decode step after the first consumed the previous step's device
            # future as its input ids; the first decode after a prefill too (the
            # prefill's take carries the placeholders).
            assert fallbacks == 0, (i, async_steps, fallbacks)
            assert async_steps == steps, (i, async_steps, steps)
    # Per prompt: (prompt, the synchronous drafter's decode steps, the async drafter's);
    # ``GENERATED - 1`` synchronous steps means no draft was accepted on that prompt.
    print("identity steps (prompt, sync, async):", counts)


# ── the context limit: the served scheduler ends a request in one-row steps ─────


LIMIT_PROMPT = 5  # the longest prompt (44 tokens): the limit comes soonest


def _generate_async_to_the_limit(runner, req: str, prompt: list[int], *, finished: set):
    """Prefill, then decode as the served async scheduler drives it until ``max_model_len``.

    ``NeuronAsyncScheduler._update_after_schedule`` (the plugin's ``vllm/core/scheduler.py``) re-arms ``k``
    placeholder drafts after every step whatever the runner proposed, until the request's
    (optimistic) count passes ``max_model_len - 3 - 2 k``; from then on, stickily, it
    schedules one-row steps, so vLLM's own trim (``num_new = min(1 + k, max_model_len - 1 -
    num_computed)``) never shortens a step. The request is held while no row fits until its
    output lands (vLLM's ``scheduler.py``); the handed count is pulled back one step late as in
    ``_generate_async``. Returns ``(ids, steps, widths, async steps, sync fallbacks, the
    indices of the steps the runner counted as fallbacks)``.
    """
    groups = fr._groups(runner)
    blocks = list(range(spec.FIRST_BLOCK, spec.FIRST_BLOCK + -(-len(prompt) // spec.PAGE)))
    _, pending = fr._step(runner, spec._prefill(req, prompt, groups, blocks, finished))
    pending_rows = 1
    handed = len(prompt)
    safe = spec.MAX_MODEL_LEN - 3 - 2 * K
    disabled = handed > safe
    generated: list[int] = []
    widths: list[int] = []
    fallback_steps: list[int] = []
    steps = 0
    async_before, sync_before = runner._async_steps, runner._sync_fallback_steps
    while True:
        drafts = [] if disabled else [-1] * K
        rows = min(1 + len(drafts), spec.MAX_MODEL_LEN - handed - 1)
        if rows <= 0:
            assert pending is not None, "the request stands at the limit with nothing in flight"
            kept = _materialize(pending)
            assert 1 <= len(kept) <= pending_rows, (kept, pending_rows)
            generated += kept
            handed -= pending_rows - len(kept)
            pending = None
            if handed >= spec.MAX_MODEL_LEN - 1:
                break
            continue
        assert rows == 1 + len(drafts), "the served scheduler never trims a step"
        needed = -(-(handed + rows) // spec.PAGE)
        new = list(range(spec.FIRST_BLOCK + len(blocks), spec.FIRST_BLOCK + needed))
        blocks += new
        fallbacks_before = runner._sync_fallback_steps
        _, output = fr._step(runner, spec._decode(req, handed, len(generated), groups, drafts, new))
        if runner._sync_fallback_steps != fallbacks_before:
            fallback_steps.append(steps)
        steps += 1
        widths.append(rows)
        handed += rows
        # ``_update_after_schedule``: the sticky check on the count after this step.
        disabled = disabled or handed > safe
        if pending is not None:
            kept = _materialize(pending)
            assert 1 <= len(kept) <= pending_rows, (kept, pending_rows)
            generated += kept
            handed -= pending_rows - len(kept)
        pending, pending_rows = output, rows
    return (
        generated, steps, widths,
        runner._async_steps - async_before, runner._sync_fallback_steps - sync_before,
        fallback_steps,
    )


def test_the_async_drafter_reaches_the_context_limit_with_the_synchronous_drafters_ids(tmp_path, monkeypatch):
    """The last steps of a request are the scheduler's, not the drafter's: under async
    scheduling the proposal is never consulted and the served scheduler switches the request
    to one-row steps near ``max_model_len`` on its own. The first one-row step is the generic
    spec-to-non-spec transition (``execute_model``): the device-side last accepted
    token is its input and the previous step's output is read on the host, which the
    generic accounting counts as one sync fallback -- exactly one, at that step; every other
    step stays on the async path, and the correction pulls the handed start back at the
    carry's own width throughout. The ids to the last position are the synchronous
    drafter's, which stops proposing early (``_spec_decode_limit``) and ends in one-row
    steps too."""
    e2e._require_cpu_mode()
    fr._declaring_a_sampler(monkeypatch)
    prompt = spec._prompts()[LIMIT_PROMPT]
    tokens = spec.MAX_MODEL_LEN - len(prompt)  # every position to the last one
    monkeypatch.delenv(proposer_tests.ASYNC_KNOB, raising=False)
    monkeypatch.setenv(spec.KNOB, str(K))
    (tmp_path / "sync").mkdir()
    with fr._parallel_state(tmp_path / "sync", spec._config(K)):
        root, _ = spec._root()
        runner = spec._runner(spec._config(K), root)
        control, control_steps = spec._generate(runner, "sync-limit", prompt, tokens=tokens, finished=set())
    assert len(control) == tokens

    monkeypatch.setenv(proposer_tests.ASYNC_KNOB, "1")
    (tmp_path / "async").mkdir()
    with fr._parallel_state(tmp_path / "async", _async_config(K)):
        root, _ = spec._root()
        runner = spec._runner(_async_config(K), root)
        ids, steps, widths, async_steps, fallbacks, fallback_steps = _generate_async_to_the_limit(
            runner, "async-limit", prompt, finished=set()
        )
    assert ids == control, (ids, control)
    # Verify steps, then the sticky transition to one-row steps, nothing in between.
    assert widths[0] == T and widths[-1] == 1 and set(widths) == {1, T}, widths
    assert widths == sorted(widths, reverse=True), widths
    # One generic transition step, the first one-row one; the rest on the async path.
    transition = widths.index(1)
    assert fallback_steps == [transition], (fallback_steps, transition, widths)
    assert fallbacks == 1 and async_steps == steps - 1, (async_steps, fallbacks, steps)
    print("limit steps (sync, async, widths, transition):", control_steps, steps, widths, transition)


# ── every served step runs in a graph the warmup compiled ───────────────────


def _graph_signature(kwargs: dict) -> tuple:
    """What the compiled root's guards read of one call's kwargs: every tensor's path, shape,
    dtype and strides; every number; and, among the carriers, the paths that hold one tensor
    object (the guards keep inputs compiled as distinct tensors distinct objects, and one tensor
    compiled under two paths one object). The carriers are the translator's operands, the ones a
    drafter form changes; the generic kwargs are the generic runner's, read by the root under
    their own names (``spec_decode_metadata`` by three fields the rejection sampler names), so
    they are compared by shape, dtype and value. Hashable, so a served step's signature can be
    looked up among the warmup's."""
    leaves: list[tuple[str, object]] = []

    def walk(value, path: str) -> None:
        if torch.is_tensor(value):
            leaves.append((path, value))
        elif isinstance(value, dict):
            for key in value:
                walk(value[key], f"{path}[{key!r}]")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")
        elif value is None or isinstance(value, (bool, int, float, str)):
            leaves.append((path, value))
        else:
            for key, item in vars(value).items():
                walk(item, f"{path}.{key}")

    for key in sorted(kwargs):
        walk(kwargs[key], key)
    shapes = tuple(
        (path, (tuple(value.shape), value.dtype, tuple(value.stride())) if torch.is_tensor(value) else value)
        for path, value in leaves
    )
    holders: dict[int, list[str]] = {}
    for path, value in leaves:
        if torch.is_tensor(value) and path.startswith("layer_carriers["):
            holders.setdefault(id(value), []).append(path)
    shared = tuple(sorted(tuple(paths) for paths in holders.values() if len(paths) > 1))
    return shapes, shared


#: ``warmup_decode``'s two synthetic builds per target: the verify form, then the one-row form.
WARMUP_FORMS = (
    ("verify", dict(spec_decode_enabled=True)),
    ("one-row", dict(spec_decode_enabled=False, decode_token_threshold=1)),
)


def _warmup_signatures(runner: NeuronModelRunner) -> dict[str, tuple]:
    """The decode graphs the warmup compiles for this runner: ``warmup_decode``'s builds
    (:data:`WARMUP_FORMS`) per target of ``NeuronWorker._decode_compile_targets`` (batch bucket
    by context bucket), each through the translator, as graph signatures keyed by form and
    target."""
    targets = NeuronWorker._decode_compile_targets(SimpleNamespace(model_runner=runner))
    assert targets, "the runner's config names no batch bucket, so no decode graph is warmed"
    return {
        f"{form} batch {batch_size} context {ctx_bucket}": _graph_signature(
            runner._glm5next_model_kwargs(
                runner._build_decode_synthetic_inputs(
                    batch_size, ctx_bucket=ctx_bucket, compiled_graph_input=True, **build
                )
            )
        )
        for batch_size, ctx_bucket in targets
        for form, build in WARMUP_FORMS
    }


def _recording_every_model_call(runner: NeuronModelRunner, monkeypatch) -> list[tuple]:
    """Record the graph signature of every call the runner makes into the root (each goes
    through ``_glm5next_model_kwargs``), leaving the call in place."""
    recorded: list[tuple] = []
    original = runner._glm5next_model_kwargs

    def recording(kwargs: dict) -> dict:
        converted = original(kwargs)
        recorded.append(_graph_signature(converted))
        return converted

    monkeypatch.setattr(runner, "_glm5next_model_kwargs", recording)
    return recorded


def _is_prefill(signature: tuple) -> bool:
    """A prefill's carriers carry ``prefill_tail`` (sparse) or ``is_prefill`` True (recurrent)."""
    shapes, _ = signature
    return any(
        path.endswith("['prefill_tail']") or (path.endswith("['is_prefill']") and value is True)
        for path, value in shapes
    )


def _differences(warmed: tuple, step: tuple) -> dict:
    """Per path, ``(warmed, step)`` where they differ; under ``shared objects`` the object
    groups only one side has."""
    warmed_shapes, step_shapes = dict(warmed[0]), dict(step[0])
    out = {
        path: (warmed_shapes.get(path), step_shapes.get(path))
        for path in sorted(set(warmed_shapes) | set(step_shapes))
        if warmed_shapes.get(path) != step_shapes.get(path)
    }
    if set(warmed[1]) != set(step[1]):
        out["shared objects"] = (sorted(set(warmed[1]) - set(step[1])), sorted(set(step[1]) - set(warmed[1])))
    return out


def _assert_every_decode_call_is_warmed(drafter: str, recorded: list[tuple], warmed: dict[str, tuple]) -> None:
    """Every decode call's signature is one of the warmed graphs'; a failure names the call, the
    nearest warmed graph and what differs from it."""
    decode_calls = [signature for signature in recorded if not _is_prefill(signature)]
    assert decode_calls, (drafter, len(recorded))
    for index, signature in enumerate(decode_calls):
        if signature in warmed.values():
            continue
        nearest = min(warmed, key=lambda name: len(_differences(warmed[name], signature)))
        raise AssertionError(
            f"{drafter} drafter, decode call {index} of {len(decode_calls)}: no warmed graph takes "
            f"this step; against '{nearest}' it differs in {_differences(warmed[nearest], signature)}"
        )


def test_every_decode_step_to_the_limit_runs_in_a_graph_the_warmup_compiled(tmp_path, monkeypatch):
    """The serve refuses a recompile (``fail_on_recompile``), so a decode step whose graph
    signature no warmed graph takes fails the request. Both drafters are driven to
    ``max_model_len`` -- the verify steps, the switch to one-row steps, the one-row steps to the
    last position -- and every decode call's signature (shapes, dtypes, numbers, and which
    operands are one object) is one ``warmup_decode`` compiles: the verify form or the one-row
    form of a target."""
    e2e._require_cpu_mode()
    fr._declaring_a_sampler(monkeypatch)
    prompt = spec._prompts()[LIMIT_PROMPT]
    tokens = spec.MAX_MODEL_LEN - len(prompt)
    monkeypatch.delenv(proposer_tests.ASYNC_KNOB, raising=False)
    monkeypatch.setenv(spec.KNOB, str(K))
    (tmp_path / "sync").mkdir()
    with fr._parallel_state(tmp_path / "sync", spec._config(K)):
        root, _ = spec._root()
        runner = spec._runner(spec._config(K), root)
        warmed = _warmup_signatures(runner)
        recorded = _recording_every_model_call(runner, monkeypatch)
        ids, _ = spec._generate(runner, "sync-limit", prompt, tokens=tokens, finished=set())
    assert len(ids) == tokens
    _assert_every_decode_call_is_warmed("synchronous", recorded, warmed)

    monkeypatch.setenv(proposer_tests.ASYNC_KNOB, "1")
    (tmp_path / "async").mkdir()
    with fr._parallel_state(tmp_path / "async", _async_config(K)):
        root, _ = spec._root()
        runner = spec._runner(_async_config(K), root)
        warmed = _warmup_signatures(runner)
        recorded = _recording_every_model_call(runner, monkeypatch)
        ids, *_ = _generate_async_to_the_limit(runner, "async-limit", prompt, finished=set())
    assert len(ids) == tokens
    _assert_every_decode_call_is_warmed("async", recorded, warmed)
