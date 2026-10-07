# SPDX-License-Identifier: Apache-2.0
"""Request admission on the GLM-5.3-Flash server: what is refused, what is not.

Each test builds the engine config the way ``vllm serve`` does (``EngineArgs`` ->
``create_engine_config``, which runs ``NeuronPlatform.check_and_update_config``), then
hands a request to ``NeuronPlatform.validate_request``, the hook vLLM's
``InputProcessor.process_inputs`` calls before a request becomes an engine request.

Three request classes are refused, each with a message that names the problem:

* a prompt longer than the prefill window (the runner raises at
  ``neuron_model_runner.py`` "a request longer than its bucket cannot be served" and the
  engine dies);
* a sampling knob the ``all_greedy`` on-device sampler cannot apply (today it returns
  greedy tokens in silence);
* logprobs / prompt_logprobs under on-device sampling (today HTTP 500, IndexError in
  ``_create_completion_logprobs``).

Everything else is accepted: greedy requests, the gate's requests, an OpenAI request that
leaves every sampling knob at the server default, an 8000-token prompt on the bs=64 line,
and every one of the refused requests on a server that samples on the host.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 PYTHONPATH=$PWD python -m pytest \\
        test/vllm_neuron/vllm/test_admission.py
"""

from __future__ import annotations

import json
import logging
import pathlib
import shutil
import tempfile

import pytest
import torch
from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.sampling_params import SamplingParams

from vllm_neuron.vllm import admission
from vllm_neuron.vllm.platform import NeuronPlatform
from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_first_request as fr

pytestmark = [pytest.mark.fast]

KNOB = "VLLM_NEURON_GLM5NEXT_ON_DEVICE_SAMPLING"
# The served checkpoint's generation_config.json (GLM-5.3-Flash-04c4e9e9): an OpenAI
# request that leaves temperature and top_p unset arrives with these values.
SERVED_GENERATION_CONFIG = {
    "eos_token_id": [154820, 154827, 154829],
    "pad_token_id": 154820,
    "temperature": 1.0,
    "top_p": 0.95,
}
SERVED_BLOCK = 128
# The gate's standard line (gate/serve.sh, GATE_RECIPE=host) and its bs=64 @ 8k line.
STANDARD_LINE = dict(
    max_model_len=4096,
    max_num_seqs=1,
    max_num_batched_tokens=1024,
    neuron_config={
        "ep_degree": 16,
        "kv_segment_size_buckets": [1024],
        "num_batched_tokens_buckets": [1024],
        "decode_context_length_buckets": [2048],
        "hybrid_kv_block_size": SERVED_BLOCK,
    },
)
BS64_LINE = dict(
    max_model_len=8192,
    max_num_seqs=64,
    max_num_batched_tokens=1024,
    neuron_config={
        "ep_degree": 16,
        "kv_segment_size_buckets": [8192],
        "num_batched_tokens_buckets": [1024],
        "num_seqs_buckets": [1, 2, 4, 8, 16, 32, 64],
        "decode_context_length_buckets": [2048],
        "hybrid_kv_block_size": SERVED_BLOCK,
    },
)
ALL_GREEDY = {"all_greedy": True}
# The standard line's window: one 1024-token KV segment plus one 1024-token query bucket.
STANDARD_WINDOW = 2048


@pytest.fixture(scope="module")
def served_model_dir():
    """The fixture checkpoint config beside the served checkpoint's generation config."""
    root = pathlib.Path(tempfile.mkdtemp(prefix="glm53f-admission-"))
    shutil.copy(fr.FIXTURE / "config.json", root / "config.json")
    (root / "generation_config.json").write_text(json.dumps(SERVED_GENERATION_CONFIG))
    yield root
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture(autouse=True)
def _restore_platform_policy():
    """Each config build replaces the class-level policy; give the next test the old one."""
    saved = NeuronPlatform._admission
    yield
    NeuronPlatform._admission = saved


ABSENT = object()


def _serve(model_dir, line: dict, *, sampler, async_scheduling: bool | None = None):
    """Build the engine config a serve of ``line`` builds; ``sampler`` is the
    on_device_sampling_config value (a dict, None for an explicit null, or ABSENT)."""
    neuron_config = dict(line["neuron_config"])
    if sampler is not ABSENT:
        neuron_config["on_device_sampling_config"] = sampler
    return EngineArgs(
        model=str(model_dir),
        skip_tokenizer_init=True,
        max_model_len=line["max_model_len"],
        max_num_seqs=line["max_num_seqs"],
        max_num_batched_tokens=line["max_num_batched_tokens"],
        block_size=SERVED_BLOCK,
        enforce_eager=True,
        enable_prefix_caching=False,
        async_scheduling=async_scheduling,
        additional_config={"neuron_config": neuron_config},
    ).create_engine_config()


@pytest.fixture
def greedy_server(served_model_dir, monkeypatch):
    """The gate's fast recipe: knob on, all_greedy, async scheduling, standard line."""
    monkeypatch.setenv(KNOB, "1")
    return _serve(served_model_dir, STANDARD_LINE, sampler=ALL_GREEDY, async_scheduling=True)


@pytest.fixture
def host_server(served_model_dir, monkeypatch):
    """The as-built recipe: host vLLM Sampler (explicit null), synchronous scheduling."""
    monkeypatch.setenv(KNOB, "1")
    return _serve(served_model_dir, STANDARD_LINE, sampler=None, async_scheduling=False)


def _prompt(tokens: int) -> dict:
    """A processed token prompt, the shape the renderer hands ``process_inputs``."""
    return {"type": "token", "prompt_token_ids": [11] * tokens}


def _openai(config, **body) -> SamplingParams:
    """The SamplingParams vLLM's completions endpoint builds for ``body``: knobs the client
    left out take the server's defaults (the model's generation config, then vLLM's)."""
    request = CompletionRequest(model="m", prompt=[11, 12], **body)
    return request.to_sampling_params(
        body.get("max_tokens", 16), config.model_config.get_diff_sampling_param()
    )


def _validate(prompt_tokens: int, params) -> None:
    NeuronPlatform.validate_request(_prompt(prompt_tokens), params)


# ── prompt length: the prefill window ───────────────────────────────────────────────


def test_standard_line_window_is_one_segment_plus_one_query_bucket(greedy_server):
    window = NeuronPlatform._admission.window
    assert (window.tokens, window.kv_segment, window.query_bucket, window.block_size) == (
        STANDARD_WINDOW, 1024, 1024, SERVED_BLOCK
    ), window


def test_prompt_one_token_past_the_window_is_refused_naming_both_lengths(greedy_server):
    with pytest.raises(ValueError) as refused:
        _validate(STANDARD_WINDOW + 1, SamplingParams(temperature=0.0, max_tokens=8))
    message = str(refused.value)
    assert str(STANDARD_WINDOW + 1) in message and str(STANDARD_WINDOW) in message, message
    assert "kv_segment_size_buckets" in message, message


def test_prompt_of_exactly_the_window_is_accepted(greedy_server):
    _validate(STANDARD_WINDOW, SamplingParams(temperature=0.0, max_tokens=8))


def test_the_window_holds_with_host_sampling_too(host_server):
    """The window is the runner's, not the sampler's: a host-sampling serve has it too."""
    with pytest.raises(ValueError, match=str(STANDARD_WINDOW)):
        _validate(STANDARD_WINDOW + 1, SamplingParams(temperature=0.0, max_tokens=8))
    _validate(STANDARD_WINDOW, SamplingParams(temperature=0.0, max_tokens=8))


def test_bs64_line_accepts_an_8000_token_prompt(served_model_dir, monkeypatch):
    """kv_segment 8192 + query 1024 covers max_model_len 8192: nothing to add to vLLM's
    own max_model_len check, so no window is enforced."""
    monkeypatch.setenv(KNOB, "1")
    _serve(served_model_dir, BS64_LINE, sampler=ALL_GREEDY, async_scheduling=True)
    assert NeuronPlatform._admission.window is None
    _validate(8000, SamplingParams(temperature=0.0, max_tokens=64))
    _validate(8191, SamplingParams(temperature=0.0, max_tokens=1))


def test_a_refused_prompt_leaves_the_next_request_admitted(greedy_server):
    """Admission keeps no state: a valid request right after a refused one is admitted."""
    greedy = SamplingParams(temperature=0.0, max_tokens=8)
    with pytest.raises(ValueError):
        _validate(STANDARD_WINDOW + 1, greedy)
    _validate(1000, greedy)


@pytest.mark.parametrize(
    "line, expected",
    [
        # Explicit buckets, the runner's [0] segment and largest query bucket.
        ({"kv_segment_size_buckets": [2048], "num_batched_tokens_buckets": [1024]}, 3072),
        # Segment only: the runner sets the query buckets to the segment buckets.
        ({"kv_segment_size_buckets": [1024]}, 2048),
        # Neither: max_num_batched_tokens 1024 < max_model_len auto-enables [1024], [1024].
        ({}, 2048),
    ],
)
def test_window_follows_the_resolved_buckets(served_model_dir, monkeypatch, line, expected):
    monkeypatch.setenv(KNOB, "1")
    _serve(
        served_model_dir,
        dict(STANDARD_LINE, max_model_len=8192, neuron_config=dict(line, ep_degree=16)),
        sampler=ALL_GREEDY,
    )
    assert NeuronPlatform._admission.window.tokens == expected


def test_single_shot_prefill_has_no_window(served_model_dir, monkeypatch):
    """max_num_batched_tokens == max_model_len: segmented prefill is off, the block table
    spans max_model_len, and only vLLM's own length check applies."""
    monkeypatch.setenv(KNOB, "1")
    _serve(
        served_model_dir,
        dict(STANDARD_LINE, max_num_batched_tokens=4096, neuron_config={"ep_degree": 16}),
        sampler=ALL_GREEDY,
    )
    assert NeuronPlatform._admission.window is None


def test_other_architectures_have_no_window():
    """Only the GLM-5.3-Flash runner path reads a fixed window; segmented prefill for the
    other families walks prior KV segment by segment."""
    assert admission.prefill_window(
        {"kv_segment_size_buckets": [1024], "num_batched_tokens_buckets": [1024]},
        architectures=("LlamaForCausalLM",),
        max_model_len=4096,
        max_num_batched_tokens=1024,
        block_size=SERVED_BLOCK,
    ) is None


@pytest.mark.parametrize(
    "line",
    [STANDARD_LINE, BS64_LINE,
     dict(STANDARD_LINE, neuron_config={"kv_segment_size_buckets": [2048]}),
     dict(STANDARD_LINE, neuron_config={}),
     dict(STANDARD_LINE, max_num_batched_tokens=4096, neuron_config={})],
    ids=["standard", "bs64", "segment-only", "auto", "single-shot"],
)
def test_admission_resolves_the_buckets_the_runner_resolves(served_model_dir, monkeypatch,
                                                             tmp_path, line):
    """Drift guard: the window is computed in the API server from the same config the
    worker's runner resolves its buckets from. A runner change to that resolution must
    show up here, not as a refused-but-valid or admitted-but-fatal prompt."""
    monkeypatch.setenv(KNOB, "1")
    config = _serve(served_model_dir, line, sampler=ALL_GREEDY)
    with fr._parallel_state(tmp_path, config):
        runner = NeuronModelRunner(config, device=torch.device("cpu"))
    resolved = admission.resolve_prefill_buckets(
        config.additional_config["neuron_config"],
        max_num_batched_tokens=config.scheduler_config.max_num_batched_tokens,
        max_model_len=config.model_config.max_model_len,
        block_size=SERVED_BLOCK,
    )
    assert resolved == (
        runner.neuron_config.kv_segment_size_buckets,
        runner.neuron_config.num_batched_tokens_buckets,
    )


# ── sampling under the all_greedy on-device sampler ─────────────────────────────────


def test_top_k_is_refused_under_all_greedy_naming_the_knob(greedy_server):
    with pytest.raises(ValueError) as refused:
        _validate(100, _openai(greedy_server, top_k=50))
    message = str(refused.value)
    assert "top_k=50" in message and "all_greedy" in message, message


def test_temperature_zero_is_accepted(greedy_server):
    _validate(100, _openai(greedy_server, temperature=0.0))


@pytest.mark.parametrize(
    "body, knob",
    [
        ({"temperature": 0.7}, "temperature=0.7"),
        ({"top_p": 0.5}, "top_p=0.5"),
        ({"min_p": 0.1}, "min_p=0.1"),
        ({"seed": 1234}, "seed=1234"),
        ({"n": 2}, "n=2"),
    ],
)
def test_other_sampling_knobs_are_refused_under_all_greedy(greedy_server, body, knob):
    with pytest.raises(ValueError, match=knob):
        _validate(100, _openai(greedy_server, **body))


def test_every_offending_knob_is_named_in_one_refusal(greedy_server):
    with pytest.raises(ValueError) as refused:
        _validate(100, _openai(greedy_server, temperature=0.7, top_k=50))
    assert "temperature=0.7" in str(refused.value) and "top_k=50" in str(refused.value)


def test_an_openai_request_left_at_the_server_defaults_is_served_greedy_with_one_warning(
    greedy_server, caplog
):
    """The client sent no temperature and no top_p, so vLLM filled in the model's
    generation config (temperature 1.0, top_p 0.95). That is indistinguishable from an
    explicit request for those values, so it is accepted and served greedy, with a warning
    logged once rather than on every request."""
    params = _openai(greedy_server, max_tokens=32)
    assert (params.temperature, params.top_p) == (1.0, 0.95)
    admission._WARNED.clear()
    with caplog.at_level(logging.WARNING, logger=admission.__name__):
        _validate(100, params)
        _validate(100, _openai(greedy_server, max_tokens=32))
    rows = [r.getMessage() for r in caplog.records if r.name == admission.__name__]
    assert len(rows) == 1 and "greedy" in rows[0] and "temperature" in rows[0], rows


@pytest.mark.parametrize(
    "body",
    [
        # The gate (smoke.py, measure.py): greedy, fixed length, streamed.
        {"temperature": 0.0, "max_tokens": 64, "min_tokens": 64, "ignore_eos": True,
         "stream": True, "return_token_ids": True},
        # lm-eval local-completions (eval_gsm8k.py): greedy with a seed and stop strings.
        {"temperature": 0.0, "seed": 1234, "max_tokens": 1024, "stop": ["Question:"]},
        # top_k=1 is argmax whatever the temperature.
        {"temperature": 0.7, "top_k": 1},
        # vllm bench serve without --temperature: every knob at the server default.
        {"max_tokens": 128, "repetition_penalty": 1.0, "stream": True},
    ],
    ids=["gate", "lm-eval", "top-k-1", "bench-serve"],
)
def test_greedy_requests_are_accepted_under_all_greedy(greedy_server, body):
    _validate(1000, _openai(greedy_server, **body))


# ── logprobs under on-device sampling ───────────────────────────────────────────────


def test_logprobs_are_refused_under_all_greedy_naming_logprobs(greedy_server):
    with pytest.raises(ValueError) as refused:
        _validate(100, _openai(greedy_server, temperature=0.0, logprobs=5))
    message = str(refused.value)
    assert "logprobs=5" in message and "on_device_sampling_config" in message, message


def test_prompt_logprobs_are_refused_under_on_device_sampling(greedy_server):
    with pytest.raises(ValueError, match="prompt_logprobs=1"):
        _validate(100, SamplingParams(temperature=0.0, prompt_logprobs=1))


def test_logprobs_are_refused_under_the_full_on_device_sampler(served_model_dir, monkeypatch):
    """The full-vocabulary sampler returns token ids only, synchronous scheduling or not."""
    monkeypatch.setenv(KNOB, "1")
    _serve(served_model_dir, STANDARD_LINE, sampler={}, async_scheduling=False)
    with pytest.raises(ValueError, match="logprobs=5"):
        _validate(100, SamplingParams(temperature=0.0, logprobs=5))
    # That sampler applies top_k itself, so top_k is not refused there.
    _validate(100, SamplingParams(temperature=0.8, top_k=50))


# ── the same requests on a host-sampling server ─────────────────────────────────────


@pytest.mark.parametrize(
    "body",
    [{"top_k": 50}, {"temperature": 0.7}, {"top_p": 0.5}, {"min_p": 0.1},
     {"seed": 1234}, {"n": 2}, {"temperature": 0.0, "logprobs": 5},
     {"temperature": 0.0, "prompt_logprobs": 1}],
)
def test_host_sampling_accepts_what_all_greedy_refuses(host_server, body):
    _validate(100, _openai(host_server, **body))


def test_host_sampling_with_the_knob_unset_accepts_them_too(served_model_dir, monkeypatch):
    """The as-built default: knob unset, the platform turns on-device sampling off."""
    monkeypatch.delenv(KNOB, raising=False)
    config = _serve(served_model_dir, STANDARD_LINE, sampler=ABSENT)
    assert config.additional_config["neuron_config"]["on_device_sampling_config"] is None
    _validate(100, _openai(config, top_k=50, logprobs=5))
