"""The runner's first request on the tiny root: warmup, one prefill, one decode, integer token ids.

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \\
        python -m pytest -s -rA test/vllm_neuron/model/glm5_next/tiny/test_tiny_glm5next_first_request.py

The serving path the item tests never drove: a real ``NeuronModelRunner`` built from an engine
config, its two warmups, then ``execute_model`` on a scheduler output, the way the worker calls
it. Readings: warmup leaves no async-execution state; the first request returns integer token ids
on the sampler path, each the argmax of the model's own logits; the spec-to-non-spec transition is
read from the recorded fact and never taken without a speculative config; a sampled-token future
that holds logits is refused by name, where it used to reach ``numpy``.
"""

from __future__ import annotations

import contextlib
import pathlib
import traceback

import pytest
import torch
from vllm.config import set_current_vllm_config
from vllm.distributed import parallel_state as dist_state
from vllm.engine.arg_utils import EngineArgs
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, KVCacheTensor

import vllm_neuron.vllm.worker.neuron_model_runner as runner_module
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


def _engine_config(*, async_scheduling: bool, on_device_sampling: bool):
    """The engine config the worker builds for this root, with the two knobs under test."""
    # The last prefill bucket must equal max_num_batched_tokens; the prompt picks the first.
    neuron_config: dict = {
        "num_batched_tokens_buckets": [PREFILL_BUCKET, landed.E2E_MAX_SEQ_LEN],
        "num_seqs_buckets": [DECODE_BATCH],
    }
    if not on_device_sampling:
        neuron_config["on_device_sampling_config"] = None
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


def test_warmup_leaves_no_async_execution_state(tmp_path):
    """Under async scheduling, both warmups leave the buffer empty and no step pending."""
    landed._require_cpu_mode()
    root = landed._fixture()["root"]
    config = _engine_config(async_scheduling=True, on_device_sampling=True)
    with _parallel_state(tmp_path, config):
        runner = _runner(config, root)
        _warm(runner)
        buffer = dict(runner.async_execution_buffer)
    print(f"FIRSTREQ|warmup|async={runner.use_async_scheduling}|buffer_keys={sorted(buffer)}"
          f"|pending={runner.execute_model_state is not None}")
    assert runner.use_async_scheduling
    assert buffer == {}, buffer
    assert runner.execute_model_state is None


def test_the_first_request_returns_integer_token_ids(tmp_path):
    """Synchronous scheduling, no on-device sampler: prefill then decode return one int each, the model's argmax."""
    landed._require_cpu_mode()
    fixture = landed._fixture()
    root = fixture["root"]
    prompt = _prompt()
    config = _engine_config(async_scheduling=False, on_device_sampling=False)
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
          f"|prefill={first}|decode={second}|logits={tuple(prefill_logits.shape)},{prefill_logits.dtype}")
    assert not runner.use_async_scheduling and not runner.on_device_sampling
    for label, ids, logits, sequence in (
        ("prefill", first, prefill_logits, prompt),
        ("decode", second, decode_logits, prompt + first[0]),
    ):
        assert ids == [[int(ids[0][0])]] and type(ids[0][0]) is int, (label, ids)
        assert tuple(logits.shape) == (1, item.STACK_VOCAB_SIZE), (label, logits.shape)
        assert ids[0][0] == int(logits[0].float().argmax()), (label, ids)
        want = landed._reference_logits(fixture, torch.tensor(sequence, dtype=torch.int64))[0].float()
        spread = float((logits[0].float() - want).abs().max())
        print(f"FIRSTREQ|logits|{label}|tokens={len(sequence)}|max_abs_delta={spread:.6g}"
              f"|reference_argmax={int(want.argmax())}")
        torch.testing.assert_close(logits[0].float(), want, rtol=LOGITS_RTOL, atol=LOGITS_ATOL)


def test_the_transition_is_read_from_the_recorded_fact(tmp_path, monkeypatch):
    """After a real step the fact is recorded False; a planted True is not read without a speculative config."""
    landed._require_cpu_mode()
    root = landed._fixture()["root"]
    prompt = _prompt()
    config = _engine_config(async_scheduling=True, on_device_sampling=True)
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


def test_a_future_that_holds_logits_is_refused_by_name(tmp_path):
    """Async scheduling with the default on-device sampler on a root that returns logits: refused, naming what it holds."""
    landed._require_cpu_mode()
    root = landed._fixture()["root"]
    prompt = _prompt()
    config = _engine_config(async_scheduling=True, on_device_sampling=True)
    with _parallel_state(tmp_path, config):
        runner = _runner(config, root)
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
    """Control: with the refusal removed, the same step raises the bf16 TypeError from the rejection parser."""
    landed._require_cpu_mode()
    monkeypatch.setattr(runner_module, "_refuse_non_integer_future", lambda *args: None)
    root = landed._fixture()["root"]
    prompt = _prompt()
    config = _engine_config(async_scheduling=True, on_device_sampling=True)
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
