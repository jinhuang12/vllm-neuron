"""The runner's first request on the tiny root: warmup, one prefill, one decode, integer token ids.

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \\
        python -m pytest -s -rA test/vllm_neuron/model/glm5_next/tiny/test_tiny_glm5next_first_request.py

The serving path the item tests never drove: a real ``NeuronModelRunner`` built from an engine
config with no sampling or scheduling knob set, its two warmups, then ``execute_model`` on a
scheduler output, the way the worker calls it. Readings: the platform resolves on-device sampling
and async scheduling off from the model class, by name; warmup leaves no async-execution state;
the first request returns integer token ids from the vLLM Sampler over the full vocabulary, each
the strict argmax of the model's logits under a per-vocabulary bias the item adds outside the model; the spec-to-non-spec transition is read from the recorded
fact, never taken without a speculative config and taken when the fact and the config agree; a
sampled-token future that holds logits is refused by name, where it used to reach ``numpy``. The controls re-create the class as it stood,
declaring a sampler it does not have.
"""

from __future__ import annotations

import contextlib
import logging
import pathlib
import traceback
from types import SimpleNamespace

import pytest
import torch
from vllm.config import set_current_vllm_config
from vllm.distributed import parallel_state as dist_state
from vllm.engine.arg_utils import EngineArgs
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheTensor

import vllm_neuron.vllm.worker.neuron_model_runner as runner_module
from vllm_neuron.model.glm5_next import Glm5NextForConditionalGeneration
from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_e2e as landed
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_forward as item

pytestmark = [pytest.mark.fast, pytest.mark.forked]

#: The fixture directory the engine config is built from: the checkpoint's config, no weights.
FIXTURE = pathlib.Path(__file__).resolve().parents[1] / "fixtures"
REQUEST = "req-0"
#: The prefill bucket and the decode batch the warmups compile, the capture harness's own.
PREFILL_BUCKET = item.STACK_TOKENS
DECODE_BATCH = 1
#: vLLM keeps block 0 as the null block, so a scheduler hands out blocks from 1.
FIRST_BLOCK = 1
#: The tolerance the end-to-end file compares the root's logits at.
LOGITS_RTOL, LOGITS_ATOL = 1e-2, 1e-5
#: The per-vocabulary bias the first-request item adds to the root's logits, in steps exact in bf16: the
#: seeded head scores two ids equally, and the ramp sets them apart so the argmax is one id with a margin.
HEAD_BIAS_STEP = 1 / 64
ARCH = "Glm5NextForConditionalGeneration"
PLATFORM_LOGGER = "vllm_neuron.vllm.platform"


@contextlib.contextmanager
def _parallel_state(tmp_path, vllm_config):
    """One rank over gloo with a file rendezvous, under the config vLLM's parallel state and op layer read."""
    with set_current_vllm_config(vllm_config, check_compile=False):
        dist_state.init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{tmp_path / 'rendezvous'}",
            local_rank=0,
            backend="gloo",
        )
        dist_state.ensure_model_parallel_initialized(1, 1)
        try:
            yield
        finally:
            dist_state.destroy_model_parallel()
            dist_state.destroy_distributed_environment()


def _engine_config(*, async_scheduling: bool | None = None, on_device_sampling: bool | None = None):
    """The engine config the worker builds for this root; ``None`` leaves a knob unset, as a serve does."""
    # The last prefill bucket must equal max_num_batched_tokens; the prompt picks the first.
    neuron_config: dict = {
        "num_batched_tokens_buckets": [PREFILL_BUCKET, landed.E2E_MAX_SEQ_LEN],
        "num_seqs_buckets": [DECODE_BATCH],
    }
    if on_device_sampling is not None:
        neuron_config["on_device_sampling_config"] = {} if on_device_sampling else None
    return EngineArgs(
        model=str(FIXTURE),
        skip_tokenizer_init=True,
        max_model_len=landed.E2E_MAX_SEQ_LEN,
        max_num_seqs=landed.E2E_MAX_NUM_SEQS,
        max_num_batched_tokens=landed.E2E_MAX_SEQ_LEN,
        block_size=item.MLA_PAGE_SIZE,
        enforce_eager=True,
        enable_prefix_caching=False,
        async_scheduling=async_scheduling,
        additional_config={"neuron_config": neuron_config},
    ).create_engine_config()


def _declaring_a_sampler(monkeypatch) -> None:
    """The class as it stood: declaring an on-device sampler it does not have, so both knobs stay on."""
    monkeypatch.setattr(Glm5NextForConditionalGeneration, "supports_on_device_sampling", True)


@contextlib.contextmanager
def _platform_rows():
    """Collect the platform's INFO rows while an engine config is built; yields the message list."""
    rows: list[str] = []

    class _Rows(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            rows.append(record.getMessage())

    handler = _Rows(level=logging.INFO)
    platform_logger = logging.getLogger(PLATFORM_LOGGER)
    level = platform_logger.level
    platform_logger.addHandler(handler)
    platform_logger.setLevel(logging.INFO)
    try:
        yield rows
    finally:
        platform_logger.removeHandler(handler)
        platform_logger.setLevel(level)


def _resolution(config) -> dict:
    """What the engine config resolved the two knobs and the scheduler to."""
    return {
        "on_device_sampling_config": config.additional_config["neuron_config"].get(
            "on_device_sampling_config", "absent"
        ),
        "async_scheduling": config.scheduler_config.async_scheduling,
        "scheduler_cls": str(config.scheduler_config.scheduler_cls).rsplit(".", 1)[-1],
    }


def _kv_cache_config(runner: NeuronModelRunner) -> KVCacheConfig:
    """One group per distinct spec, one tensor per layer, the generation's blocks plus the null block."""
    specs = runner.get_kv_cache_spec()
    num_blocks = landed.E2E_BLOCKS + 1
    groups: list[KVCacheGroupSpec] = []
    for name, spec in specs.items():
        for group in groups:
            if group.kv_cache_spec == spec:
                group.layer_names.append(name)
                break
        else:
            groups.append(KVCacheGroupSpec([name], spec))
    tensors = [
        KVCacheTensor(size=num_blocks * spec.page_size_bytes, shared_by=[name])
        for name, spec in specs.items()
    ]
    return KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=tensors, kv_cache_groups=groups)


def _runner(vllm_config, root) -> NeuronModelRunner:
    """The real runner on the tiny root, the model bound in place of ``load_model``."""
    runner = NeuronModelRunner(vllm_config, device=torch.device("cpu"))
    runner.model = root
    runner.vocab_size = item.STACK_VOCAB_SIZE
    runner.initialize_kv_cache(_kv_cache_config(runner))
    return runner


def _head_bias(root) -> torch.Tensor:
    """Add a per-vocabulary ramp to the root's logits outside the model; returns the ramp the reference adds too."""
    ramp = torch.arange(item.STACK_VOCAB_SIZE, dtype=torch.float32) * HEAD_BIAS_STEP
    forward = root.forward

    def biased(*args, **kwargs):
        logits = forward(*args, **kwargs)
        return logits + ramp.to(logits.dtype)

    root.forward = biased
    return ramp


def _warm(runner: NeuronModelRunner) -> None:
    """The worker's two warmups at the compile targets this root serves."""
    runner.warmup_prefill(PREFILL_BUCKET, 0)
    runner.warmup_decode(DECODE_BATCH, ctx_bucket=runner.max_model_len)


def _prompt() -> list[int]:
    """The end-to-end file's seeded prompt."""
    return torch.randint(
        0,
        item.STACK_VOCAB_SIZE,
        (item.STACK_TOKENS,),
        generator=torch.Generator().manual_seed(item.SEED_STACK_IDS),
        dtype=torch.int64,
    ).tolist()


def _prefill_step(prompt: list[int], groups: int) -> SchedulerOutput:
    """A new request's whole prompt, padded to its bucket the way the scheduler pads it."""
    blocks = list(range(FIRST_BLOCK, FIRST_BLOCK + landed.PROMPT_BLOCKS))
    new = NewRequestData(
        req_id=REQUEST,
        prompt_token_ids=prompt,
        mm_features=[],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=landed.GENERATED_TOKENS),
        pooling_params=None,
        block_ids=tuple(list(blocks) for _ in range(groups)),
        num_computed_tokens=0,
        lora_request=None,
    )
    step = SchedulerOutput(
        scheduled_new_reqs=[new],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={REQUEST: len(prompt)},
        total_num_scheduled_tokens=len(prompt),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[0],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    step.num_scheduled_tokens_padded = {REQUEST: PREFILL_BUCKET}
    return step


def _decode_step(position: int, generated: int, groups: int) -> SchedulerOutput:
    """The request's next token at ``position``, with the block that position opens."""
    opens_block = position % item.MLA_PAGE_SIZE == 0
    new_block = FIRST_BLOCK + position // item.MLA_PAGE_SIZE
    cached = CachedRequestData(
        req_ids=[REQUEST],
        resumed_req_ids=set(),
        new_token_ids=[],
        all_token_ids={},
        new_block_ids=[tuple([new_block] for _ in range(groups)) if opens_block else None],
        num_computed_tokens=[position],
        num_output_tokens=[generated],
    )
    step = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached,
        num_scheduled_tokens={REQUEST: 1},
        total_num_scheduled_tokens=1,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[0],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    step.num_scheduled_tokens_padded = {REQUEST: 1}
    return step


def _groups(runner: NeuronModelRunner) -> int:
    """How many KV-cache groups the runner allocated, one block list per group in every step."""
    return len(runner.kv_cache_config.kv_cache_groups)


def _step(runner: NeuronModelRunner, step: SchedulerOutput):
    """The worker's two calls for one step: the logits the sampler saw and the output."""
    returned = runner.execute_model(step)
    logits = runner.execute_model_state.logits if runner.execute_model_state is not None else None
    output = runner.sample_tokens(None) if returned is None else returned
    return logits, output


def _recording(runner: NeuronModelRunner, monkeypatch) -> list[str]:
    """Record every reason the runner breaks the async flow for, leaving the break itself in place."""
    reasons: list[str] = []
    original = runner._materialize_pending_async_output

    def recorded(reason: str) -> bool:
        reasons.append(reason)
        return original(reason)

    monkeypatch.setattr(runner, "_materialize_pending_async_output", recorded)
    return reasons


def _frames(raised) -> list[str]:
    """The function names on the traceback, innermost last."""
    return [frame.name for frame in traceback.extract_tb(raised.tb)]


def test_sampling_and_scheduling_resolve_from_the_model_class():
    """No knob set: the platform reads the class, turns both off, and says so by name."""
    landed._require_cpu_mode()
    with _platform_rows() as rows:
        config = _engine_config()
    resolved = _resolution(config)
    named = [row for row in rows if ARCH in row and "no on-device sampler" in row]
    print(f"FIRSTREQ|resolution|declared={Glm5NextForConditionalGeneration.supports_on_device_sampling}"
          f"|{resolved}|rows={named}")
    assert Glm5NextForConditionalGeneration.supports_on_device_sampling is False
    assert resolved == {"on_device_sampling_config": None, "async_scheduling": False,
                        "scheduler_cls": "NeuronScheduler"}, resolved
    kinds = sorted({("On-device sampling is off" in row, "Async scheduling is off" in row) for row in named})
    assert kinds == [(False, True), (True, False)], named


def test_an_explicit_sampler_config_for_the_class_is_refused_by_name():
    """An on-device sampling config for a class without a sampler is refused, naming both."""
    landed._require_cpu_mode()
    with pytest.raises(ValueError) as raised:
        _engine_config(on_device_sampling=True)
    message = " ".join(str(raised.value).split())
    print(f"FIRSTREQ|explicit_sampler|message={message[:300]}")
    assert ARCH in message and "on_device_sampling_config" in message and "no on-device sampler" in message


def test_an_explicit_async_request_is_turned_off_by_name():
    """vLLM resolves the async default before the platform hook, so an explicit request is turned off the same way."""
    landed._require_cpu_mode()
    with _platform_rows() as rows:
        config = _engine_config(async_scheduling=True)
    resolved = _resolution(config)
    named = [row for row in rows if ARCH in row and "Async scheduling is off" in row]
    print(f"FIRSTREQ|explicit_async|{resolved}|rows={named}")
    assert resolved["async_scheduling"] is False and resolved["scheduler_cls"] == "NeuronScheduler", resolved
    assert named, rows


def test_the_runner_refuses_async_without_a_sampler_by_name(tmp_path, monkeypatch):
    """Backstop: the class declaring a sampler, sampling off and async on, the runner refuses naming the class."""
    landed._require_cpu_mode()
    _declaring_a_sampler(monkeypatch)
    config = _engine_config(async_scheduling=True, on_device_sampling=False)
    with _parallel_state(tmp_path, config), pytest.raises(RuntimeError) as raised:
        NeuronModelRunner(config, device=torch.device("cpu"))
    message = str(raised.value)
    print(f"FIRSTREQ|runner_backstop|{_resolution(config)}|message={message[:200]}")
    assert config.scheduler_config.async_scheduling is True
    assert ARCH in message and "synchronous scheduling" in message


def _warmed_state(tmp_path, config, root) -> tuple[bool, dict, bool]:
    """Build the runner, run both warmups, and read the async flag, the buffer, and whether a step is pending.

    A synchronous runner never creates the buffer; an absent buffer reads as empty.
    """
    with _parallel_state(tmp_path, config):
        runner = _runner(config, root)
        _warm(runner)
        buffer = dict(getattr(runner, "async_execution_buffer", {}))
        return runner.use_async_scheduling, buffer, runner.execute_model_state is not None


def test_warmup_leaves_no_async_execution_state(tmp_path):
    """No knob set: both warmups leave the buffer empty and no step pending."""
    landed._require_cpu_mode()
    is_async, buffer, pending = _warmed_state(tmp_path, _engine_config(), landed._fixture()["root"])
    print(f"FIRSTREQ|warmup|resolved|async={is_async}|buffer_keys={sorted(buffer)}|pending={pending}")
    assert is_async is False and buffer == {} and pending is False


def test_warmup_left_no_async_execution_state_as_it_stood(tmp_path, monkeypatch):
    """The class as it stood, async on: both warmups still leave the buffer empty, so there was nothing to clear."""
    landed._require_cpu_mode()
    _declaring_a_sampler(monkeypatch)
    is_async, buffer, pending = _warmed_state(
        tmp_path, _engine_config(async_scheduling=True), landed._fixture()["root"]
    )
    print(f"FIRSTREQ|warmup|as_it_stood|async={is_async}|buffer_keys={sorted(buffer)}|pending={pending}")
    assert is_async is True and buffer == {} and pending is False


def test_the_first_request_returns_integer_token_ids(tmp_path):
    """No knob set: prefill then decode return one int each, the strict argmax of the biased full-vocabulary logits."""
    landed._require_cpu_mode()
    fixture = landed._fixture()
    root = fixture["root"]
    ramp = _head_bias(root)
    prompt = _prompt()
    config = _engine_config()
    with _parallel_state(tmp_path, config):
        runner = _runner(config, root)
        _warm(runner)
        prefill_logits, prefill = _step(runner, _prefill_step(prompt, _groups(runner)))
        first = prefill.sampled_token_ids
        decode_logits, decode = _step(
            runner, _decode_step(position=len(prompt), generated=1, groups=_groups(runner))
        )
        second = decode.sampled_token_ids
    print(f"FIRSTREQ|tokens|async={runner.use_async_scheduling}|ods={runner.on_device_sampling}"
          f"|prefill={first}|decode={second}|logits={tuple(prefill_logits.shape)},{prefill_logits.dtype}"
          f"|vocab={runner.vocab_size}")
    assert runner.use_async_scheduling is False and runner.on_device_sampling is False
    assert type(prefill).__name__ == "ModelRunnerOutput" and type(decode).__name__ == "ModelRunnerOutput"
    for label, ids, logits, sequence in (
        ("prefill", first, prefill_logits, prompt),
        ("decode", second, decode_logits, prompt + first[0]),
    ):
        assert ids == [[int(ids[0][0])]] and type(ids[0][0]) is int, (label, ids)
        assert tuple(logits.shape) == (1, runner.vocab_size) == (1, item.STACK_VOCAB_SIZE), (label, logits.shape)
        assert ids[0][0] == int(logits[0].float().argmax()), (label, ids)
        want = landed._reference_logits(fixture, torch.tensor(sequence, dtype=torch.int64))[0].float() + ramp
        spread = float((logits[0].float() - want).abs().max())
        top = logits[0].float().topk(2)
        margin = float(top.values[0] - top.values[1])
        ties = int((logits[0].float() == logits[0].float().max()).sum())
        print(f"FIRSTREQ|logits|{label}|tokens={len(sequence)}|max_abs_delta={spread:.6g}"
              f"|reference_argmax={int(want.argmax())}|top2={top.indices.tolist()}"
              f"|margin={margin:.6g}|ties_at_the_max={ties}|bias_step={HEAD_BIAS_STEP}")
        assert ties == 1 and margin > 0, (label, top.indices.tolist(), margin)
        torch.testing.assert_close(logits[0].float(), want, rtol=LOGITS_RTOL, atol=LOGITS_ATOL)


def test_the_transition_is_read_from_the_recorded_fact(tmp_path, monkeypatch):
    """After a real step the fact is recorded False; a planted True is not read without a speculative config."""
    landed._require_cpu_mode()
    _declaring_a_sampler(monkeypatch)
    root = landed._fixture()["root"]
    prompt = _prompt()
    config = _engine_config(async_scheduling=True)
    with _parallel_state(tmp_path, config):
        runner = _runner(config, root)
        _warm(runner)
        _, prefill = _step(runner, _prefill_step(prompt, _groups(runner)))
        recorded = runner.async_execution_buffer.get("prev_step_was_spec", "absent")
        runner.async_execution_buffer["prev_step_was_spec"] = True
        reasons = _recording(runner, monkeypatch)
        with pytest.raises(TypeError):
            runner.execute_model(_decode_step(position=len(prompt), generated=1, groups=_groups(runner)))
    print(f"FIRSTREQ|transition|output={type(prefill).__name__}|recorded={recorded}"
          f"|speculative_config={runner.speculative_config}|reasons={reasons}")
    assert type(prefill).__name__ == "AsyncNeuronModelRunnerOutput"
    assert recorded is False
    assert runner.speculative_config is None
    assert reasons and all("spec" not in reason for reason in reasons), reasons


def test_a_future_that_holds_logits_is_refused_by_name(tmp_path, monkeypatch):
    """Control 1 with the refusal: the class as it stood, async on, the first decode is refused naming what the future holds."""
    landed._require_cpu_mode()
    _declaring_a_sampler(monkeypatch)
    root = landed._fixture()["root"]
    prompt = _prompt()
    config = _engine_config(async_scheduling=True)
    with _parallel_state(tmp_path, config):
        runner = _runner(config, root)
        assert runner.use_async_scheduling and runner.on_device_sampling
        _warm(runner)
        _step(runner, _prefill_step(prompt, _groups(runner)))
        future = runner.async_execution_buffer["futures_sampled_token_ids"]
        with pytest.raises(TypeError) as raised:
            runner.execute_model(_decode_step(position=len(prompt), generated=1, groups=_groups(runner)))
    message = str(raised.value)
    frames = _frames(raised)
    print(f"FIRSTREQ|refusal|future={tuple(future.shape)},{future.dtype}"
          f"|frames={frames[-4:]}|message={message[:200]}")
    assert future.dtype == torch.bfloat16 and future.ndim == 2
    assert str(future.dtype) in message and str(tuple(future.shape)) in message and REQUEST in message
    assert "logits" in message and "on_device_sampling_config" in message
    assert frames[-4:] == [
        "execute_model", "_materialize_pending_async_output", "get_output", "_refuse_non_integer_future",
    ], frames
    assert "_parse_rejection_sampling_output" not in frames


def test_without_the_refusal_the_same_future_reaches_numpy(tmp_path, monkeypatch):
    """Control 1 as it stood: with the refusal removed, the same step raises the bf16 TypeError from the rejection parser."""
    landed._require_cpu_mode()
    _declaring_a_sampler(monkeypatch)
    monkeypatch.setattr(runner_module, "_refuse_non_integer_future", lambda *args: None)
    root = landed._fixture()["root"]
    prompt = _prompt()
    config = _engine_config(async_scheduling=True)
    with _parallel_state(tmp_path, config):
        runner = _runner(config, root)
        _warm(runner)
        _step(runner, _prefill_step(prompt, _groups(runner)))
        with pytest.raises(TypeError) as raised:
            runner.execute_model(_decode_step(position=len(prompt), generated=1, groups=_groups(runner)))
    message = str(raised.value)
    frames = _frames(raised)
    print(f"FIRSTREQ|control|frames={frames[-4:]}|message={message[:120]}")
    assert "BFloat16" in message, message
    assert frames[-3:] == ["_materialize_pending_async_output", "get_output", "_parse_rejection_sampling_output"], frames


def test_the_transition_is_taken_when_the_fact_and_the_config_agree(tmp_path, monkeypatch):
    """A recorded spec step under a speculative config breaks the async flow at the transition, handing on the last accepted token."""
    landed._require_cpu_mode()
    _declaring_a_sampler(monkeypatch)
    root = landed._fixture()["root"]
    prompt = _prompt()
    config = _engine_config(async_scheduling=True)
    with _parallel_state(tmp_path, config):
        runner = _runner(config, root)
        _warm(runner)
        _step(runner, _prefill_step(prompt, _groups(runner)))
        accepted = torch.tensor([prompt[-1]], dtype=torch.int64)
        runner.speculative_config = SimpleNamespace(num_speculative_tokens=1)
        runner.async_execution_buffer["prev_step_was_spec"] = True
        runner.async_execution_buffer["futures_last_accepted_token"] = accepted
        reasons = _recording(runner, monkeypatch)
        with pytest.raises(TypeError) as raised:
            runner.execute_model(_decode_step(position=len(prompt), generated=1, groups=_groups(runner)))
    frames = _frames(raised)
    print(f"FIRSTREQ|transition_taken|reasons={reasons}"
          f"|bonus_is_the_planted_tensor={runner._transition_bonus_tensor is accepted}"
          f"|composition_changed={runner._batch_composition_changed}|frames={frames[-3:]}")
    assert reasons[:1] == ["spec\u2192non-spec transition"], reasons
    assert runner._transition_bonus_tensor is accepted
    assert runner._batch_composition_changed is True
    assert frames[-3:] == ["_materialize_pending_async_output", "get_output", "_refuse_non_integer_future"], frames
