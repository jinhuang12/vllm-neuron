# SPDX-License-Identifier: Apache-2.0
"""Request admission on the GLM-5.3-Flash server: what is refused, what is not.

Each test builds the engine config the way ``vllm serve`` does (``EngineArgs`` ->
``create_engine_config``, which runs ``NeuronPlatform.check_and_update_config``), then
hands a request to ``NeuronPlatform.validate_request``, the hook vLLM's
``InputProcessor.process_inputs`` calls before a request becomes an engine request.

Three request classes are refused, each with a message that names the problem:

* a prompt that leaves no room for a generated token (``prompt + 1 > max_model_len``);
  the prefill window itself is ``max_model_len`` since the runner picks a KV segment per
  request from ``kv_segment_size_buckets`` and refuses at startup a list whose largest
  segment does not cover ``max_model_len``, so admission has nothing shorter to enforce;
* a sampling knob the ``all_greedy`` on-device sampler cannot apply (today it returns
  greedy tokens in silence);
* logprobs / prompt_logprobs under on-device sampling (today HTTP 500, IndexError in
  ``_create_completion_logprobs``).

Everything else is accepted: greedy requests, the benchmark's requests, an OpenAI request that
leaves every sampling knob at the server default, a 3000-token prompt on the standard line
(refused with HTTP 400 before this change, when the window was 2048), an 8000-token
prompt on the bs=64 line, and every one of the refused requests on a server that samples
on the host.

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

from vllm.pooling_params import PoolingParams

from vllm_neuron.utils.bucket_utils import prefill_window_tokens
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
# The served standard line (with the three segments that
# cover max_model_len 4096: one 1024 segment alone served a 2048-token window) and its
# bs=64 @ 8k line.
STANDARD_LINE = dict(
    max_model_len=4096,
    max_num_seqs=1,
    max_num_batched_tokens=1024,
    neuron_config={
        "ep_degree": 16,
        "kv_segment_size_buckets": [1024, 2048, 4096],
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
# The standard line's longest prompt: max_model_len less one generated token.
STANDARD_MAX = STANDARD_LINE["max_model_len"]
# The window the old standard line ([1024] alone) served; prompts past it are admitted now.
OLD_WINDOW = 2048


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
    """The fast recipe: knob on, all_greedy, async scheduling, standard line."""
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


# ── prompt length: max_model_len ────────────────────────────────────────────────────


def test_a_prompt_of_max_model_len_tokens_is_refused_naming_max_model_len(greedy_server):
    """4096 prompt tokens leave no room for the one token every request generates."""
    with pytest.raises(ValueError) as refused:
        _validate(STANDARD_MAX, SamplingParams(temperature=0.0, max_tokens=1))
    message = str(refused.value)
    assert "max_model_len" in message and str(STANDARD_MAX) in message, message
    assert str(STANDARD_MAX - 1) in message, message


def test_a_prompt_of_max_model_len_minus_one_is_accepted(greedy_server):
    _validate(STANDARD_MAX - 1, SamplingParams(temperature=0.0, max_tokens=1))


@pytest.mark.parametrize("tokens", [OLD_WINDOW, OLD_WINDOW + 1, 3000, 3500, STANDARD_MAX - 1])
def test_prompts_past_the_old_window_are_admitted(greedy_server, tokens):
    """The 2048-token cap is gone: the runner serves every prompt vLLM admits."""
    _validate(tokens, SamplingParams(temperature=0.0, max_tokens=8))


def test_the_length_rule_holds_with_host_sampling_too(host_server):
    """The rule is the model length's, not the sampler's: a host-sampling serve has it too."""
    with pytest.raises(ValueError, match="max_model_len"):
        _validate(STANDARD_MAX, SamplingParams(temperature=0.0, max_tokens=8))
    _validate(STANDARD_MAX - 1, SamplingParams(temperature=0.0, max_tokens=8))


def test_bs64_line_accepts_an_8000_token_prompt_and_refuses_8192(served_model_dir,
                                                                monkeypatch):
    monkeypatch.setenv(KNOB, "1")
    _serve(served_model_dir, BS64_LINE, sampler=ALL_GREEDY, async_scheduling=True)
    _validate(8000, SamplingParams(temperature=0.0, max_tokens=64))
    _validate(8191, SamplingParams(temperature=0.0, max_tokens=1))
    with pytest.raises(ValueError, match="8192"):
        _validate(8192, SamplingParams(temperature=0.0, max_tokens=1))


def test_a_pooling_request_generates_nothing_so_it_may_fill_max_model_len(greedy_server):
    NeuronPlatform.validate_request(_prompt(STANDARD_MAX), PoolingParams())


def test_a_refused_prompt_leaves_the_next_request_admitted(greedy_server):
    """Admission keeps no state: a valid request right after a refused one is admitted."""
    greedy = SamplingParams(temperature=0.0, max_tokens=8)
    with pytest.raises(ValueError):
        _validate(STANDARD_MAX, greedy)
    _validate(3000, greedy)


def test_the_policy_names_max_model_len_in_its_startup_line(served_model_dir, monkeypatch,
                                                           caplog):
    monkeypatch.setenv(KNOB, "1")
    with caplog.at_level(logging.INFO, logger=admission.__name__):
        _serve(served_model_dir, STANDARD_LINE, sampler=ALL_GREEDY, async_scheduling=True)
    assert any(
        "max_model_len" in record.getMessage() and str(STANDARD_MAX) in record.getMessage()
        for record in caplog.records
    ), caplog.text


@pytest.mark.parametrize(
    "line",
    [STANDARD_LINE, BS64_LINE,
     dict(STANDARD_LINE, neuron_config={"kv_segment_size_buckets": [2048]}),
     dict(STANDARD_LINE, neuron_config={}),
     dict(STANDARD_LINE, max_num_batched_tokens=4096, neuron_config={})],
    ids=["standard", "bs64", "segment-only", "auto", "single-shot"],
)
def test_every_line_that_starts_serves_a_prefill_window_of_max_model_len(
    served_model_dir, monkeypatch, tmp_path, line
):
    """Drift guard: admission refuses only prompts past max_model_len, which holds because
    the worker's runner refuses at startup any segment list whose window is shorter. A
    runner change to that rule must show up here, not as an admitted-but-fatal prompt."""
    monkeypatch.setenv(KNOB, "1")
    config = _serve(served_model_dir, line, sampler=ALL_GREEDY)
    with fr._parallel_state(tmp_path, config):
        runner = NeuronModelRunner(config, device=torch.device("cpu"))
    segments = runner.neuron_config.kv_segment_size_buckets
    if segments is None:
        # Single-shot prefill: the block table spans max_model_len.
        assert config.scheduler_config.max_num_batched_tokens >= config.model_config.max_model_len
        return
    window = prefill_window_tokens(
        segments, runner.neuron_config.num_batched_tokens_buckets, SERVED_BLOCK
    )
    assert window >= config.model_config.max_model_len, (segments, window)


def test_the_old_standard_line_does_not_start(served_model_dir, monkeypatch, tmp_path):
    """[1024] alone at max_model_len 4096 is refused by the runner, naming the 2048-token
    window it would serve: the HTTP 400 guard for prompts past it has no config to guard."""
    monkeypatch.setenv(KNOB, "1")
    old = dict(STANDARD_LINE, neuron_config=dict(STANDARD_LINE["neuron_config"],
                                                 kv_segment_size_buckets=[1024]))
    config = _serve(served_model_dir, old, sampler=ALL_GREEDY)
    with fr._parallel_state(tmp_path, config), pytest.raises(ValueError) as refused:
        NeuronModelRunner(config, device=torch.device("cpu"))
    message = str(refused.value)
    assert "prefill window" in message and str(OLD_WINDOW) in message, message
    assert str(STANDARD_MAX) in message, message


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
