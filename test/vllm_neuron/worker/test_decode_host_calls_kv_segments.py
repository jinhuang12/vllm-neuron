# SPDX-License-Identifier: Apache-2.0
"""A cached bs=1 decode step makes the same host calls whatever ``kv_segment_size_buckets`` holds.

The runner picks a request's KV segment once per prefill chunk
(``NeuronModelRunner._build_attention_metadata``: ``_prefill_request_tokens`` ->
``_prefill_kv_segment_size`` -> ``bucket_utils.select_kv_segment_size``, and one INFO line);
a decode step keeps the first segment and does no per-request work. A served config lists
several segments (``[1024, 2048, 4096]`` at max_model_len 4096) where one used to do, so this
file pins the host side of that claim: ``sys.setprofile`` over one cached step, the model's own
frames excluded, counts every call the step makes on the host.

1. a cached decode step's host calls (every Python call and C call outside the model, made by
   the worker's two calls per step, ``execute_model`` and ``sample_tokens``) are identical,
   call for call, with one, two and three segments of the same largest segment;
2. the per-request pick runs once in a prefill chunk and never in a decode step;
3. the metadata builder itself makes no host sync, list conversion or logging call in a
   decode step.

A real ``NeuronModelRunner`` on the tiny GLM-5.3-Flash root, through the first-request
harness, at the served KV page of 128 tokens and scaled down: max_model_len 512, one query
bucket of 128 tokens, segments [384] / [128, 384] / [128, 256, 384] (each window,
largest segment + query bucket = 512 tokens, covers the model), the worker's warmups (one
prefill per segment, one decode), then one request: a prompt of one chunk and decode steps
inside its first page.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 PYTHONPATH=$PWD python -m pytest \\
        test/vllm_neuron/worker/test_decode_host_calls_kv_segments.py
"""

from __future__ import annotations

import collections
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from vllm.engine.arg_utils import EngineArgs

from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_e2e as e2e
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_first_request as fr

pytestmark = [pytest.mark.forked]

#: The served KV page (``hybrid_kv_block_size``).
PAGE = 128
#: The one query bucket, which is also the token budget per step.
QUERY = 128
MAX_MODEL_LEN = 4 * PAGE
#: The smallest single segment whose window (segment + query bucket) covers the model.
LARGEST = MAX_MODEL_LEN - QUERY
SEGMENT_LISTS = ([LARGEST], [QUERY, LARGEST], [QUERY, 2 * QUERY, LARGEST])
DECODE_STEPS = 4
#: A prompt of one prefill chunk (at most ``QUERY`` tokens) whose decode steps stay inside its
#: first page, so no step opens a block: positions ``PROMPT .. PROMPT + DECODE_STEPS - 1``.
PROMPT = PAGE - DECODE_STEPS
#: The step profiled. The first decode step after a prefill carries the prefill-to-decode
#: transition; the ones after it are the steady state a served request spends its time in.
CACHED_STEP = 2
#: The host-only metadata knob, set as a served GLM-5.3-Flash step sets it.
HOST_ONLY_KNOB = "VLLM_NEURON_GLM5NEXT_HOST_ONLY_METADATA"
#: The name of the wrapper around the model's forward; frames below it are the model (on
#: Neuron, the graph), and are not counted.
MODEL_MARK = "_model_frames_below"
#: The runner's file, as :func:`_where` names it.
RUNNER_FILE = "worker/neuron_model_runner.py"
#: The per-request segment pick, by (file, function).
PER_REQUEST_PICK = (
    (RUNNER_FILE, "_prefill_request_tokens"),
    (RUNNER_FILE, "_prefill_kv_segment_size"),
    ("utils/bucket_utils.py", "select_kv_segment_size"),
)
METADATA_BUILDER = "_build_attention_metadata"
#: Tensor and array methods that copy device data to the host or build a host list.
HOST_SYNC_METHODS = ("item", "tolist", "cpu", "numpy")
#: The file the logging calls run in, as :func:`_where` names it, and the calls themselves.
LOGGING_FILE = "logging/__init__.py"
LOGGING_CALLS = ("debug", "info", "warning", "error", "log")


def _where(filename: str) -> str:
    """The file a frame runs in, by its package directory and name (``worker/foo.py``), so the
    key does not depend on where the tree or the interpreter is installed."""
    return "/".join(Path(filename).parts[-2:])


@dataclass
class HostCalls:
    """Calls made outside the model during one profiled call.

    A key is ``("py", file, function)`` for a Python call (``file`` as :func:`_where` names
    it) or ``("c", module, qualname)`` for a C call (a builtin, or a method of a C type such
    as ``Tensor.item``; ``module`` is empty for methods). ``calls`` counts every call;
    ``from_metadata`` counts the calls the metadata builder makes itself (its direct callees).
    """

    calls: collections.Counter = field(default_factory=collections.Counter)
    from_metadata: collections.Counter = field(default_factory=collections.Counter)
    _inside: int = 0

    def hook(self, frame, event, arg) -> None:
        code = frame.f_code
        if code.co_name == MODEL_MARK:
            if event == "call":
                self._inside += 1
            elif event == "return":
                self._inside -= 1
            return
        if self._inside:
            return
        if event == "call":
            key = ("py", _where(code.co_filename), code.co_name)
            caller = frame.f_back
        elif event == "c_call":
            module = getattr(arg, "__module__", None) or ""
            key = ("c", module, getattr(arg, "__qualname__", repr(arg)))
            caller = frame
        else:
            return
        self.calls[key] += 1
        if caller is not None and caller.f_code.co_name == METADATA_BUILDER:
            self.from_metadata[key] += 1


def _profiled(fn) -> HostCalls:
    """Run ``fn()`` under ``sys.setprofile`` and return the calls it made outside the model."""
    recorded = HostCalls()
    sys.setprofile(recorded.hook)
    try:
        fn()
    finally:
        sys.setprofile(None)
    return recorded


def _marked(forward):
    """Wrap the model's forward in a frame named ``MODEL_MARK``, which the profile skips below."""

    def _model_frames_below(*args, **kwargs):
        return forward(*args, **kwargs)

    return _model_frames_below


def _engine_config(segments: list[int]):
    """The engine config of this file's scaled-down serve, with ``segments`` as the segment list."""
    return EngineArgs(
        model=str(fr.FIXTURE),
        skip_tokenizer_init=True,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=1,
        max_num_batched_tokens=QUERY,
        block_size=PAGE,
        enforce_eager=True,
        enable_prefix_caching=False,
        additional_config={
            "neuron_config": {
                "kv_segment_size_buckets": list(segments),
                "num_batched_tokens_buckets": [QUERY],
                "num_seqs_buckets": [1],
            }
        },
    ).create_engine_config()


@dataclass
class Served:
    """One served request: the runner's resolved segment list and its two profiled steps."""

    segments: list[int]
    prefill: HostCalls
    decode: HostCalls


def _serve(segments: list[int]) -> Served:
    """Warm the runner as the worker does, prefill one chunk, decode; profile the prefill
    chunk and the cached decode step."""
    config = _engine_config(segments)
    root = e2e._fixture()["root"]
    root.forward = _marked(root.forward)
    pages = tuple(range(fr.FIRST_BLOCK, fr.FIRST_BLOCK + MAX_MODEL_LEN // PAGE))
    with tempfile.TemporaryDirectory() as tmp, fr._parallel_state(Path(tmp), config):
        runner = fr._runner(config, root, num_blocks=fr.FIRST_BLOCK + len(pages))
        resolved = list(runner.neuron_config.kv_segment_size_buckets)
        for segment in resolved:
            runner.warmup_prefill(QUERY, segment)
        runner.warmup_decode(1, ctx_bucket=MAX_MODEL_LEN)
        groups = fr._groups(runner)
        prompt = fr._prompt()[:PROMPT]
        assert len(prompt) == PROMPT, f"the harness prompt holds {len(prompt)} < {PROMPT} tokens"
        chunk = fr._prefill_step(prompt, groups, table=pages[: -(-PROMPT // PAGE)])
        chunk.num_scheduled_tokens_padded = {fr.REQUEST: QUERY}
        prefill = _profiled(lambda: fr._step(runner, chunk))
        decode = None
        for k in range(DECODE_STEPS):
            step = fr._decode_step(
                position=PROMPT + k, generated=1 + k, groups=groups, page=PAGE, table=pages
            )
            if k == CACHED_STEP:
                decode = _profiled(lambda: fr._step(runner, step))
            else:
                fr._step(runner, step)
    return Served(segments=resolved, prefill=prefill, decode=decode)


@pytest.fixture(scope="module")
def served() -> list[Served]:
    """One served request per entry of ``SEGMENT_LISTS``, in that order."""
    e2e._require_cpu_mode()
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(HOST_ONLY_KNOB, "1")
        return [_serve(list(segments)) for segments in SEGMENT_LISTS]


def _named(calls: collections.Counter, where: str, name: str) -> int:
    return calls[("py", where, name)]


def test_a_cached_decode_step_makes_the_same_host_calls_for_one_two_and_three_segments(served):
    assert [run.segments for run in served] == [list(s) for s in SEGMENT_LISTS]
    reference = served[0].decode.calls
    # The step went through the runner and its metadata builder, outside the model.
    assert _named(reference, RUNNER_FILE, "execute_model") == 1, reference
    assert _named(reference, RUNNER_FILE, METADATA_BUILDER) == 1, reference
    for run in served[1:]:
        differing = {
            key: (reference[key], run.decode.calls[key])
            for key in set(reference) | set(run.decode.calls)
            if reference[key] != run.decode.calls[key]
        }
        assert not differing, (
            f"a cached decode step with kv_segment_size_buckets={run.segments} makes host "
            f"calls that {served[0].segments} does not (one segment: n, these: m): {differing}"
        )


def test_the_segment_pick_runs_once_per_prefill_chunk_and_never_in_a_decode_step(served):
    run = served[-1]
    for where, name in PER_REQUEST_PICK:
        assert _named(run.prefill.calls, where, name) == 1, (where, name, run.prefill.calls)
        assert _named(run.decode.calls, where, name) == 0, (
            f"{where}:{name} runs in a cached decode step; the segment is picked once per "
            f"prefill chunk and a decode step keeps kv_segment_size_buckets[0]"
        )


def test_the_metadata_builder_makes_no_host_sync_list_or_log_call_in_a_decode_step(served):
    run = served[-1]
    direct = run.decode.from_metadata
    assert direct, "the profile saw no call made by the metadata builder"
    synced = {
        key: count
        for key, count in direct.items()
        if key[0] == "c" and key[-1].rsplit(".", 1)[-1] in HOST_SYNC_METHODS
    }
    logged = {
        key: count
        for key, count in direct.items()
        if key[:2] == ("py", LOGGING_FILE) and key[2] in LOGGING_CALLS
    }
    assert not synced, f"{METADATA_BUILDER} copies to the host in a decode step: {synced}"
    assert not logged, f"{METADATA_BUILDER} logs in a decode step: {logged}"
