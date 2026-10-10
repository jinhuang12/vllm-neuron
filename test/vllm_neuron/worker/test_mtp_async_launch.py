# SPDX-License-Identifier: Apache-2.0
"""The async drafter's two kernels launch inside compiled graphs on a device.

On a device an NKI launch outside a traced graph has no kernel to dispatch to
(``nki_kernel_wrapper`` raises ``NotImplementedError: could not find kernel for
HigherOrderOperator nki_kernel_wrapper at dispatch key DispatchKey.PrivateUse1``); the
simulator answers the same launch eagerly on the host, so a CPU run of the real runner
cannot see that rule by itself. These tests apply it on the host: the module's two
launchers (``async_step._TAKE_KERNEL``, ``async_step._CORRECT_KERNEL``) refuse an eager
call with the device's error and answer a traced one, every operand counts as off-host,
and the runner's launch accessor hands the :class:`DeviceLaunch` a device runner builds,
with the eager backend (the host's ``torch.compile`` backend) in place of the model's.
Those three stand-ins replace the device itself; nothing else is faked: the real runner
takes the worker's two calls per step (``execute_model`` then ``sample_tokens``, the order
the engine's executor makes them in) through the real scheduler shapes of
``test_mtp_async_identity`` with the async knob, async scheduling and on-device greedy
sampling. On a tree without the fix the first test fails at the prefill's
``sample_tokens`` with the device's error.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/worker/test_mtp_async_launch.py
"""

from __future__ import annotations

import pytest
import torch
from vllm.engine.arg_utils import EngineArgs
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput

from vllm_neuron.functional.mtp import async_step
from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

from test.vllm_neuron.functional import test_mtp_async_step as step_tests
from test.vllm_neuron.model.glm5_next import test_mtp_e2e_spec as spec
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_e2e as e2e
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_first_request as fr
from test.vllm_neuron.worker import test_mtp_async_identity as identity
from test.vllm_neuron.worker import test_mtp_proposer as proposer_tests

pytestmark = [pytest.mark.forked]

K = identity.K
T = identity.T
#: Tokens generated per run: the prefill, the first verify step after it (the correction
#: at ``prev_width = 1``) and verify steps after verify steps (``prev_width = 1 + k``).
TOKENS = 12
PROMPT = 0


def _emulated_device(monkeypatch, build_launch) -> dict:
    """Apply the device's dispatch rule on the host.

    Returns the per-launcher eager-call counts (an eager launch raises the device's error
    and is counted; a launch that returned was traced) and ``built``, the launch the runner
    handed. ``build_launch`` builds it once, on the first step that asks for one -- a tree
    before the fix never asks, and the two attributes it lacks are left absent
    (``raising=False``) so such a tree fails for the device's reason.
    """
    seen = {
        "take": step_tests.device_dispatch_rule(monkeypatch, "_TAKE_KERNEL"),
        "correct": step_tests.device_dispatch_rule(monkeypatch, "_CORRECT_KERNEL"),
    }
    monkeypatch.setattr(async_step, "_launches_in_a_graph", lambda tensor: True, raising=False)
    built: dict = {}

    def launch(self):
        if "launch" not in built:
            built["launch"] = build_launch()
        return built["launch"]

    monkeypatch.setattr(NeuronModelRunner, "_glm5next_async_launch", launch, raising=False)
    seen["built"] = built
    return seen


def _async_runner(tmp_path, monkeypatch):
    """The async drafter's config and a prompt; the caller opens the parallel state."""
    e2e._require_cpu_mode()
    fr._declaring_a_sampler(monkeypatch)
    monkeypatch.delenv(spec.KNOB, raising=False)
    monkeypatch.setenv(proposer_tests.ASYNC_KNOB, "1")
    config = identity._async_config(K)
    assert config.scheduler_config.async_scheduling is True
    (tmp_path / "async").mkdir()
    return config, spec._prompts()[PROMPT]


def test_the_prefill_and_the_first_decode_launch_the_kernels_inside_compiled_graphs(tmp_path, monkeypatch):
    """Prefill, first decode and the verify steps after it serve the plain run's ids with
    every kernel launch traced: the prefill's take at ``sample_tokens``, each decode's
    correction at ``execute_model`` and its take at ``sample_tokens``."""
    config, prompt = _async_runner(tmp_path, monkeypatch)
    monkeypatch.delenv(proposer_tests.ASYNC_KNOB)
    reference = spec._plain_references(tmp_path, [prompt])[0]
    monkeypatch.setenv(proposer_tests.ASYNC_KNOB, "1")
    seen = _emulated_device(monkeypatch, lambda: async_step.DeviceLaunch("eager"))
    with fr._parallel_state(tmp_path / "async", config):
        root, _ = spec._root()
        runner = spec._runner(config, root)
        assert runner.use_async_scheduling and runner.drafter.async_steps is True
        ids, steps, async_steps, fallbacks = identity._generate_async(
            runner, "async-launch", prompt, tokens=TOKENS, finished=set()
        )
    assert ids == reference[:TOKENS], (ids, reference[:TOKENS])
    assert fallbacks == 0 and async_steps == steps, (steps, async_steps, fallbacks)
    # One take per step (the prefill's and each decode's), one correction per decode;
    # all of them through the launch, none eager.
    assert seen["take"] == {"eager": 0} and seen["correct"] == {"eager": 0}
    assert seen["built"]["launch"].calls == {"take": steps + 1, "correct": steps}


def test_a_device_operand_without_the_runners_launch_is_refused_at_the_prefills_sample_tokens(tmp_path, monkeypatch):
    """A runner that hands no launch refuses by name at the first take, before any eager
    launch reaches the device's dispatcher."""
    config, prompt = _async_runner(tmp_path, monkeypatch)
    seen = _emulated_device(monkeypatch, lambda: None)
    with fr._parallel_state(tmp_path / "async", config):
        root, _ = spec._root()
        runner = spec._runner(config, root)
        groups = fr._groups(runner)
        blocks = list(range(spec.FIRST_BLOCK, spec.FIRST_BLOCK + -(-len(prompt) // spec.PAGE)))
        assert runner.execute_model(spec._prefill("refused", prompt, groups, blocks, set())) is None
        with pytest.raises(async_step.MtpAsyncStepError, match="no DeviceLaunch"):
            runner.sample_tokens(None)
    assert seen["take"] == {"eager": 0} and seen["correct"] == {"eager": 0}


def test_the_warmup_compiles_every_launch_signature_a_served_sequence_meets(tmp_path, monkeypatch):
    """The worker's warmups (each prefill bucket, the decode batch) compile the take at
    the one-row and the verify width and the correction at its four width pairs; a served
    sequence after them compiles nothing."""
    config, prompt = _async_runner(tmp_path, monkeypatch)
    compiles: list = []
    seen = _emulated_device(
        monkeypatch, lambda: async_step.DeviceLaunch(step_tests.counting_backend(compiles))
    )
    with fr._parallel_state(tmp_path / "async", config):
        root, _ = spec._root()
        runner = spec._runner(config, root)
        for bucket in (spec.PREFILL_BUCKET, spec.MAX_MODEL_LEN):
            runner.warmup_prefill(bucket, 0)
        runner.warmup_decode(fr.DECODE_BATCH, ctx_bucket=runner.max_model_len)
        assert runner._glm5next_async_carry is None and runner.execute_model_state is None
        pages = runner._glm5next_async_column_pages(spec.PAGE)
        int32 = torch.int32
        takes = sorted(entry[1:] for entry in compiles if entry[0] == 2)
        corrections = [entry[1:] for entry in compiles if entry[0] == 3]
        # The sampler's own shapes: ``[1]`` after a prefill or a one-row decode, ``[1, 1 + k]``
        # after a verify step; the carry's ``[1]`` rows, the ``[1]`` start, the fixed table.
        assert takes == sorted([
            (((1,), (1, K)), (int32, int32)),
            (((1, T), (1, K)), (int32, int32)),
        ]), takes
        assert corrections == [(((1,), (1,), (pages, 1)), (int32, int32, int32))] * 4, corrections
        warmed = len(compiles)
        launch = seen["built"]["launch"]
        assert launch.calls == {"take": 4, "correct": 8}, launch.calls
        ids, steps, _, _ = identity._generate_async(
            runner, "async-warmed", prompt, tokens=TOKENS, finished=set()
        )
    assert len(ids) == TOKENS and steps >= 2
    assert len(compiles) == warmed, compiles[warmed:]


# ── a two-chunk prefill: the served line's prefill shape ─────────────────────────────

#: The served line schedules a prompt longer than ``max_num_batched_tokens`` in chunks of
#: that many tokens; here the chunk is the spec file's prefill bucket and the prompt one
#: chunk and a quarter.
CHUNK = spec.PREFILL_BUCKET
LONG_PROMPT_LENGTH = CHUNK + CHUNK // 4


def _long_prompt() -> list[int]:
    gen = torch.Generator().manual_seed(spec.SEED_PROMPTS + 1)
    return torch.randint(0, spec.VOCAB, (LONG_PROMPT_LENGTH,), generator=gen).tolist()


def _chunked_config(k: int | None):
    """``identity._async_config`` with ``max_num_batched_tokens = CHUNK`` (the one prefill
    bucket), so a longer prompt prefills in chunks; ``k`` None is the plain run (no drafter,
    synchronous scheduling), the reference."""
    neuron_config = {
        "num_batched_tokens_buckets": [CHUNK],
        "num_seqs_buckets": [1],
        "on_device_sampling_config": {},
    }
    return EngineArgs(
        model=str(fr.FIXTURE),
        skip_tokenizer_init=True,
        max_model_len=spec.MAX_MODEL_LEN,
        max_num_seqs=1,
        max_num_batched_tokens=CHUNK,
        block_size=spec.PAGE,
        enforce_eager=True,
        enable_prefix_caching=False,
        async_scheduling=k is not None,
        speculative_config=(
            {"method": "mtp", "num_speculative_tokens": k} if k is not None else None
        ),
        additional_config={"neuron_config": neuron_config},
    ).create_engine_config()


def _prefill_in_chunks(record: dict):
    """A ``prefill`` driver for ``spec._generate`` / ``identity._generate_async``: the prompt
    in two chunks of at most ``CHUNK`` tokens, scheduled as the engine schedules them (the
    whole prompt and its blocks with the new request, the rest as a cached request at
    ``num_computed_tokens = CHUNK``). Records the intermediate chunk's output and the
    runner's carry after each chunk."""

    def prefill(runner, req, prompt, groups, blocks, finished):
        assert CHUNK < len(prompt) <= 2 * CHUNK, len(prompt)
        first = spec._prefill(req, prompt, groups, blocks, finished)
        first.num_scheduled_tokens = {req: CHUNK}
        first.total_num_scheduled_tokens = CHUNK
        first.num_scheduled_tokens_padded = {req: CHUNK}
        _, record["partial"] = fr._step(runner, first)
        record["carry_after_partial"] = runner._glm5next_async_carry
        rest = len(prompt) - CHUNK
        cached = CachedRequestData(
            req_ids=[req], resumed_req_ids=set(), new_token_ids=[], all_token_ids={},
            new_block_ids=[None], num_computed_tokens=[CHUNK], num_output_tokens=[0],
        )
        second = SchedulerOutput(
            scheduled_new_reqs=[], scheduled_cached_reqs=cached,
            num_scheduled_tokens={req: rest}, total_num_scheduled_tokens=rest,
            scheduled_spec_decode_tokens={}, scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0], finished_req_ids=set(), free_encoder_mm_hashes=[],
        )
        second.num_scheduled_tokens_padded = {req: CHUNK}
        _, out = fr._step(runner, second)
        record["carry"] = runner._glm5next_async_carry
        return out

    return prefill


def test_a_two_chunk_prefill_launches_each_chunks_take_in_a_graph_and_the_first_verify_step_follows(
    tmp_path, monkeypatch
):
    """Both prefill chunks take their rows through the launch (the intermediate chunk's
    rows are discarded by the output, its launch is a graph launch like any other), the
    carry after the last chunk is a prefill's (``prev_width`` 1), and the first verify step
    corrects from it; the ids are the plain run's, prefilled in the same chunks."""
    e2e._require_cpu_mode()
    fr._declaring_a_sampler(monkeypatch)
    monkeypatch.delenv(spec.KNOB, raising=False)
    monkeypatch.delenv(proposer_tests.ASYNC_KNOB, raising=False)
    prompt = _long_prompt()
    plain_record: dict = {}
    plain_config = _chunked_config(None)
    (tmp_path / "plain").mkdir()
    with fr._parallel_state(tmp_path / "plain", plain_config):
        runner = spec._runner(plain_config, spec._root()[0])
        assert runner.drafter is None
        reference, _ = spec._generate(
            runner, "plain-chunked", prompt, tokens=TOKENS, finished=set(),
            prefill=_prefill_in_chunks(plain_record),
        )
    assert plain_record["partial"].sampled_token_ids == [[]], "the intermediate chunk emits nothing"
    assert plain_record["carry"] is None

    monkeypatch.setenv(proposer_tests.ASYNC_KNOB, "1")
    config = _chunked_config(K)
    assert config.scheduler_config.async_scheduling is True
    assert int(config.scheduler_config.max_num_batched_tokens) == CHUNK < len(prompt)
    record: dict = {}
    seen = _emulated_device(monkeypatch, lambda: async_step.DeviceLaunch("eager"))
    (tmp_path / "async").mkdir()
    with fr._parallel_state(tmp_path / "async", config):
        root, _ = spec._root()
        runner = spec._runner(config, root)
        assert runner.use_async_scheduling and runner.drafter.async_steps is True
        ids, steps, async_steps, fallbacks = identity._generate_async(
            runner, "async-chunked", prompt, tokens=TOKENS, finished=set(),
            prefill=_prefill_in_chunks(record),
        )
    assert identity._materialize(record["partial"]) == [], "the intermediate chunk emits nothing"
    for carry in (record["carry_after_partial"], record["carry"]):
        assert carry is not None and carry.req_ids == ("async-chunked",)
        assert int(carry.prev_width) == 1, "a prefill chunk's carry is one row wide"
    assert ids == reference[:TOKENS], (ids, reference[:TOKENS])
    assert fallbacks == 0 and async_steps == steps, (steps, async_steps, fallbacks)
    # Two prefill chunks and each decode step take once; each decode corrects once; no
    # launch was eager.
    assert seen["take"] == {"eager": 0} and seen["correct"] == {"eager": 0}
    assert seen["built"]["launch"].calls == {"take": steps + 2, "correct": steps}
