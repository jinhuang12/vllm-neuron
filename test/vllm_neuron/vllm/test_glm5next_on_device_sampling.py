# SPDX-License-Identifier: Apache-2.0
"""On-device sampling and async scheduling for GLM-5.3-Flash, opted into by one knob.

Today the platform refuses an on-device sampling config for this class and turns async
scheduling off, because the root returns logits. ``VLLM_NEURON_GLM5NEXT_ON_DEVICE_SAMPLING``
lifts that gate: the root then hands its full-vocabulary logits to
``sample_full_vocab`` and returns int32 token ids, which is what the async runner feeds
back as the next step's ``input_ids``. With the knob unset nothing changes; the locked
tests in ``test/vllm_neuron/model/glm5_next/tiny/test_tiny_glm5next_first_request.py``
keep covering that default.

Every token here is compared with the host vLLM Sampler's token on the same steps.

    PYTHONPATH=$PWD python -m pytest test/vllm_neuron/vllm/test_glm5next_on_device_sampling.py
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from vllm_neuron.model.glm5_next import model_fp8
from vllm_neuron.model.neuron_config import OnDeviceSamplingConfig

from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_e2e as e2e
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_first_request as fr
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_forward as tiny

pytestmark = [pytest.mark.fast]

KNOB = "VLLM_NEURON_GLM5NEXT_ON_DEVICE_SAMPLING"
DECODES = 2


@pytest.fixture
def knob_on(monkeypatch):
    monkeypatch.setenv(KNOB, "1")


def _neuron(config) -> dict:
    return config.additional_config["neuron_config"]


# ── platform gate ────────────────────────────────────────────────────────────


def test_knob_on_keeps_an_explicit_sampler_config_and_async(knob_on):
    config = fr._engine_config(async_scheduling=True, on_device_sampling=True)
    resolved = fr._resolution(config)
    assert resolved == {"on_device_sampling_config": {}, "async_scheduling": True,
                        "scheduler_cls": "NeuronAsyncScheduler"}, resolved


def test_knob_on_with_the_config_absent_keeps_the_neuron_default(knob_on):
    """Absent means NeuronConfig's default, which samples on device."""
    config = fr._engine_config(async_scheduling=False)
    assert "on_device_sampling_config" not in _neuron(config)
    assert config.scheduler_config.async_scheduling is False


def test_knob_on_with_an_explicit_null_keeps_the_host_sampler(knob_on):
    """``null`` is the as-built serve: host sampler, and async is turned off as before."""
    with fr._platform_rows() as rows:
        config = fr._engine_config(async_scheduling=True, on_device_sampling=False)
    resolved = fr._resolution(config)
    assert resolved == {"on_device_sampling_config": None, "async_scheduling": False,
                        "scheduler_cls": "NeuronScheduler"}, resolved
    assert any("Async scheduling is off" in row for row in rows), rows


def test_knob_on_refuses_data_parallel_sampling(knob_on):
    with pytest.raises(ValueError, match="data-parallel sampling cannot apply"):
        fr.EngineArgs(
            model=str(fr.FIXTURE), skip_tokenizer_init=True, max_model_len=e2e.E2E_MAX_SEQ_LEN,
            max_num_seqs=1, max_num_batched_tokens=e2e.E2E_MAX_SEQ_LEN,
            block_size=tiny.MLA_PAGE_SIZE, enforce_eager=True, enable_prefix_caching=False,
            additional_config={"neuron_config": {
                "num_batched_tokens_buckets": [fr.PREFILL_BUCKET, e2e.E2E_MAX_SEQ_LEN],
                "num_seqs_buckets": [1],
                "on_device_sampling_config": {"sampling_dp_degree": 2},
            }},
        ).create_engine_config()


def _with_sampler(sampler: dict):
    return fr.EngineArgs(
        model=str(fr.FIXTURE), skip_tokenizer_init=True, max_model_len=e2e.E2E_MAX_SEQ_LEN,
        max_num_seqs=1, max_num_batched_tokens=e2e.E2E_MAX_SEQ_LEN,
        block_size=tiny.MLA_PAGE_SIZE, enforce_eager=True, enable_prefix_caching=False,
        additional_config={"neuron_config": {
            "num_batched_tokens_buckets": [fr.PREFILL_BUCKET, e2e.E2E_MAX_SEQ_LEN],
            "num_seqs_buckets": [1],
            "on_device_sampling_config": sampler,
        }},
    ).create_engine_config()


SLOW_ROW = "top-k over the full vocabulary"


@pytest.mark.parametrize("sampler", [{}, {"all_greedy": False}, None])
def test_knob_on_warns_that_the_full_sampler_is_slow(knob_on, sampler):
    """trn2 measured ``torch.topk`` over all 154,880 entries at ~16 ms per call at any B
    (test/hardware/benchmark_sampler_decode.py); argmax alone is ~0.14 ms at B=1. An
    absent key (``None`` here) means NeuronConfig's default, which is the full sampler."""
    with fr._platform_rows() as rows:
        if sampler is None:
            fr._engine_config(async_scheduling=False)
        else:
            _with_sampler(sampler)
    assert any(SLOW_ROW in row and "all_greedy" in row for row in rows), rows


@pytest.mark.parametrize("sampler", [{"all_greedy": True}, {"all_greedy": "true"}])
def test_knob_on_with_a_greedy_only_sampler_does_not_warn(knob_on, sampler):
    with fr._platform_rows() as rows:
        _with_sampler(sampler)
    assert not any(SLOW_ROW in row for row in rows), rows
    assert any("On-device sampling is on" in row for row in rows), rows


def test_knob_off_still_refuses_an_explicit_sampler_config(monkeypatch):
    monkeypatch.delenv(KNOB, raising=False)
    with pytest.raises(ValueError, match="no on-device sampler"):
        fr._engine_config(on_device_sampling=True)


# ── runner + root, end to end ────────────────────────────────────────────────


def _recording(root) -> list[dict]:
    """Record the keywords each root call receives and the type of what it returns."""
    calls: list[dict] = []
    forward = root.forward

    def recorded(*args, **kwargs):
        out = forward(*args, **kwargs)
        calls.append({"keys": set(kwargs), "kwargs": kwargs, "args": args,
                      "dtype": getattr(out, "dtype", None)})
        return out

    root.forward = recorded
    return calls


def _ids(output) -> list[int]:
    if hasattr(output, "get_output"):
        output = output.get_output()
    return [int(row[0]) for row in output.sampled_token_ids]


def _generate(tmp_path, config, *, record: bool = False):
    """Prefill then ``DECODES`` greedy decodes; returns ids, the runner, and recorded calls."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    root = e2e._fixture()["root"]
    prompt = fr._prompt()
    with fr._parallel_state(tmp_path, config):
        runner = fr._runner(config, root)
        # load_model builds the root from the runner's own neuron config.
        root.text_config.neuron_config = runner.neuron_config
        fr._warm(runner)
        calls = _recording(root) if record else []
        groups = fr._groups(runner)
        ids = []
        out = fr._step(runner, fr._prefill_step(prompt, groups))[1]
        ids.append(_ids(out))
        for index in range(DECODES):
            position = len(prompt) + index
            out = fr._step(runner, fr._decode_step(position=position, generated=index + 1,
                                                   groups=groups))[1]
            ids.append(_ids(out))
    return ids, runner, calls


@pytest.fixture(scope="module")
def host_run(tmp_path_factory):
    """The as-built path once per module: host vLLM Sampler, synchronous scheduling."""
    saved = os.environ.pop(KNOB, None)
    try:
        ids, runner, calls = _generate(tmp_path_factory.mktemp("host"), fr._engine_config(),
                                       record=True)
    finally:
        if saved is not None:
            os.environ[KNOB] = saved
    assert runner.on_device_sampling is False and runner.use_async_scheduling is False
    assert all("device_sampling_params" not in call["keys"] for call in calls)
    return ids, calls


def test_sync_on_device_tokens_equal_the_host_sampler(tmp_path, knob_on, host_run):
    host_ids, _ = host_run
    config = fr._engine_config(async_scheduling=False, on_device_sampling=True)
    device_ids, runner, calls = _generate(tmp_path, config, record=True)
    assert runner.on_device_sampling is True and runner.use_async_scheduling is False
    assert device_ids == host_ids, (device_ids, host_ids)
    assert len(calls) == 1 + DECODES
    for call in calls:
        assert "device_sampling_params" in call["keys"] and "sampling_params" not in call["keys"]
        assert call["dtype"] == torch.int32


def test_async_on_device_tokens_equal_the_host_sampler(tmp_path, knob_on, host_run):
    host_ids, _ = host_run
    config = fr._engine_config(async_scheduling=True, on_device_sampling=True)
    device_ids, runner, _ = _generate(tmp_path, config)
    assert runner.use_async_scheduling is True and runner.on_device_sampling is True
    assert device_ids == host_ids, (device_ids, host_ids)
    # Every decode after the first fed the previous step's device tokens back as input_ids.
    assert runner._async_steps >= DECODES - 1, (runner._async_steps, runner._sync_fallback_steps)


def test_a_root_without_a_sampler_config_refuses_before_its_stack_runs(host_run, monkeypatch):
    """A recorded real call, replayed with sampling parameters on a root built without them."""
    _, calls = host_run
    call = calls[0]
    root = e2e._fixture()["root"]
    ran: list = []
    monkeypatch.setattr(root.model, "forward", lambda *a, **k: ran.append(1))
    with pytest.raises(ValueError, match="on_device_sampling_config"):
        root.forward(*call["args"], **call["kwargs"], device_sampling_params=torch.zeros(1, 3))
    assert ran == [], "the stack ran before the refusal"


@pytest.mark.parametrize("rows", [1, 4])
def test_the_root_samples_every_row_it_projects(host_run, monkeypatch, rows):
    """No B=1 assumption in the hand-off: a call projecting ``rows`` positions hands all
    ``rows`` logits rows to the sampler and returns ``rows`` tokens, each that row's argmax."""
    _, calls = host_run
    call = calls[0]
    kwargs = dict(call["kwargs"])
    kwargs["sampling_positions"] = torch.arange(rows, dtype=torch.long) * 7
    root = e2e._fixture()["root"]
    root.text_config.neuron_config = SimpleNamespace(
        on_device_sampling_config=OnDeviceSamplingConfig())
    seen: list = []
    real = model_fp8.sample_full_vocab

    def recorded(logits, *args, **kw):
        seen.append(logits.detach().clone())
        return real(logits, *args, **kw)

    monkeypatch.setattr(model_fp8, "sample_full_vocab", recorded)
    params = torch.tensor([[-1, 1.0, 0.0]] * rows, dtype=torch.float32)
    tokens = root.forward(*call["args"], **kwargs, device_sampling_params=params)
    assert len(seen) == 1 and tuple(seen[0].shape) == (rows, tiny.STACK_VOCAB_SIZE)
    assert tokens.dtype == torch.int32 and tuple(tokens.shape) == (rows,)
    assert torch.equal(tokens, torch.argmax(seen[0], dim=-1).to(torch.int32))
