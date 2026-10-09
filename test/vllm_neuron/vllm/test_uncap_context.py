# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash long-context lines: no software cap below ``max_model_len``.

The GLM-5.3-Flash runner reads a prefill chunk's prior KV through a gathered block-table
window (``window_blocks = min(table_width, ceil((segment + query) / page))``), not the
segmented attention kernel, so that kernel's segment sizes {512, ..., 8192}, its chunk
sizes and its 16k single-shot bound do not apply to it. Before this change the largest
segment was 8192, so the longest prompt was 8192 + the largest query bucket (16,384 at
chunk 8192) at any ``max_model_len``.

Checked here, each from the real config path (``EngineArgs.create_engine_config``, which
runs ``NeuronPlatform.check_and_update_config``) and the real runner constructor:

* the bucket rules accept a 65,536 and a 262,144 segment for a windowed-prefill model
  and still refuse them for the other families;
* a server with ``max_model_len`` 65,536 or 262,144, chunk 8192 and one segment of
  ``max_model_len`` refuses no prompt below ``max_model_len`` (admission's only length
  rule is prompt + 1 <= max_model_len);
* the runner serves that one segment while the decode candidate row (``max_model_len``
  / ``index_kpool``) fits the decode select kernel, and refuses a longer line at start
  (``DecodeSelectError``, naming ``MAX_DECODE_INDEX_CANDIDATES``);
* the old cap line (segment 8192 + chunk 8192 at ``max_model_len`` 65,536, a 16,384-token
  window) does not start: the runner names the window and the segment to add.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import tempfile

import pytest
import torch
from vllm.engine.arg_utils import EngineArgs
from vllm.sampling_params import SamplingParams

from vllm_neuron.functional.dsa import decode_select as DS
from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig
from vllm_neuron.utils import bucket_utils as BU
from vllm_neuron.vllm.platform import NeuronPlatform
from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_first_request as fr

pytestmark = [pytest.mark.fast]

KNOB = "VLLM_NEURON_GLM5NEXT_ON_DEVICE_SAMPLING"
SERVED_BLOCK = 128
#: The largest segment the segmented attention kernel takes, the old cap.
OLD_SEGMENT_MAX = max(BU.SUPPORTED_KV_SEGMENT_SIZES)
#: A chunk as wide as the old cap, so ``2 * CHUNK + 1`` tokens is past the old window.
CHUNK = OLD_SEGMENT_MAX
LONG_LINES = (65536, 262144)
#: Tokens per decode candidate pool, GLM-5.3-Flash's.
POOL = Glm5NextTextConfig().index_kpool
#: The long lines whose decode candidate row fits the decode select kernel, and the rest.
STARTED_LINES = tuple(n for n in LONG_LINES if n // POOL <= DS.MAX_DECODE_INDEX_CANDIDATES)
REFUSED_LINES = tuple(n for n in LONG_LINES if n not in STARTED_LINES)


def _long_line(max_model_len: int) -> dict:
    """A long line: bs=1, chunk ``CHUNK``, one segment of ``max_model_len``.

    The GLM-5.3-Flash long-context lines were served at TP 64 in this form: max_model_len
    65,536 at chunk 2048, and max_model_len 262,144 at chunk 512. Each had one KV segment
    and a mamba block size of max_model_len."""
    return dict(
        max_model_len=max_model_len,
        max_num_seqs=1,
        max_num_batched_tokens=CHUNK,
        neuron_config={
            "ep_degree": 16,
            "kv_segment_size_buckets": [max_model_len],
            "num_batched_tokens_buckets": [CHUNK],
            "decode_context_length_buckets": [2048],
            "hybrid_kv_block_size": SERVED_BLOCK,
            "on_device_sampling_config": {"all_greedy": True},
        },
    )


@pytest.fixture(scope="module")
def served_model_dir():
    root = pathlib.Path(tempfile.mkdtemp(prefix="glm53f-uncap-"))
    shutil.copy(fr.FIXTURE / "config.json", root / "config.json")
    (root / "generation_config.json").write_text(json.dumps({"temperature": 1.0}))
    yield root
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture(autouse=True)
def _restore_platform_policy():
    saved = NeuronPlatform._admission
    yield
    NeuronPlatform._admission = saved


def _serve(model_dir, line: dict):
    return EngineArgs(
        model=str(model_dir),
        skip_tokenizer_init=True,
        max_model_len=line["max_model_len"],
        max_num_seqs=line["max_num_seqs"],
        max_num_batched_tokens=line["max_num_batched_tokens"],
        block_size=SERVED_BLOCK,
        enforce_eager=True,
        enable_prefix_caching=False,
        mamba_block_size=line["max_model_len"],
        async_scheduling=True,
        additional_config={"neuron_config": dict(line["neuron_config"])},
    ).create_engine_config()


# ── bucket rules ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("segment", [16384, 65536, 262144, 1048576])
def test_a_windowed_prefill_model_takes_any_segment(segment):
    assert BU.validate_kv_segment_size_buckets(
        [segment], [CHUNK], allow_independent_query_buckets=True, windowed_prefill=True,
    ) == [segment]


@pytest.mark.parametrize("segment", [16384, 65536])
def test_the_segmented_kernel_families_keep_their_segment_sizes(segment):
    with pytest.raises(ValueError, match="not a supported segment size"):
        BU.validate_kv_segment_size_buckets([segment], None)


@pytest.mark.parametrize("chunk", [12288, 16384, 32768])
def test_a_windowed_prefill_model_chunks_at_any_size(chunk):
    assert BU.resolve_segmented_prefill_config(
        chunk, 65536, windowed_prefill=True) == ([chunk], [chunk])


@pytest.mark.parametrize("max_model_len", [32768, 65536])
def test_a_windowed_prefill_model_runs_single_shot_at_any_length(max_model_len):
    assert BU.resolve_segmented_prefill_config(
        max_model_len, max_model_len, windowed_prefill=True) == (None, None)


def test_the_segmented_kernel_families_keep_their_chunk_and_single_shot_bounds():
    with pytest.raises(ValueError, match="not a supported chunked prefill size"):
        BU.resolve_segmented_prefill_config(16384, 65536)
    with pytest.raises(ValueError, match="Single-shot prefill"):
        BU.resolve_segmented_prefill_config(32768, 32768)


# ── served lines ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("max_model_len", LONG_LINES)
def test_a_long_line_refuses_no_prompt_below_max_model_len(served_model_dir, monkeypatch,
                                                            max_model_len):
    monkeypatch.setenv(KNOB, "1")
    config = _serve(served_model_dir, _long_line(max_model_len))
    assert config.additional_config["neuron_config"][
        "_model_supports_windowed_prefill"] is True
    assert NeuronPlatform._admission.prompt.tokens == max_model_len - 1
    greedy = SamplingParams(temperature=0.0, max_tokens=1)
    for tokens in (2 * CHUNK + 1, max_model_len // 2, max_model_len - 1):
        NeuronPlatform.validate_request(
            {"type": "token", "prompt_token_ids": [11] * tokens}, greedy)
    with pytest.raises(ValueError, match=f"max_model_len is {max_model_len}"):
        NeuronPlatform.validate_request(
            {"type": "token", "prompt_token_ids": [11] * max_model_len}, greedy)


@pytest.mark.parametrize("max_model_len", STARTED_LINES)
def test_the_runner_builds_a_long_line_with_one_segment_of_max_model_len(
        served_model_dir, monkeypatch, tmp_path, max_model_len):
    monkeypatch.setenv(KNOB, "1")
    config = _serve(served_model_dir, _long_line(max_model_len))
    with fr._parallel_state(tmp_path, config):
        runner = NeuronModelRunner(config, device=torch.device("cpu"))
    assert runner.neuron_config._model_supports_windowed_prefill is True
    assert runner.neuron_config.kv_segment_size_buckets == [max_model_len]
    assert runner.neuron_config.num_batched_tokens_buckets == [CHUNK]
    assert BU.prefill_window_tokens(
        [max_model_len], [CHUNK], SERVED_BLOCK) >= max_model_len


@pytest.mark.parametrize("max_model_len", REFUSED_LINES)
def test_the_runner_refuses_a_line_past_the_decode_select_kernel(
        served_model_dir, monkeypatch, tmp_path, max_model_len):
    """Admission accepts the line (its only length rule is prompt + 1 <= max_model_len),
    but its decode candidate row is wider than the decode select kernel takes, so the
    runner refuses it at start and names the bound."""
    monkeypatch.setenv(KNOB, "1")
    config = _serve(served_model_dir, _long_line(max_model_len))
    assert NeuronPlatform._admission.prompt.tokens == max_model_len - 1
    with fr._parallel_state(tmp_path, config), pytest.raises(
            DS.DecodeSelectError, match="MAX_DECODE_INDEX_CANDIDATES"):
        NeuronModelRunner(config, device=torch.device("cpu"))


def test_the_old_cap_line_does_not_start(served_model_dir, monkeypatch, tmp_path):
    """The old largest segment + one chunk at max_model_len 65,536: its window of
    segment + chunk tokens is below max_model_len, so the runner refuses the line at
    startup and names the segment to add. No server serves a window below max_model_len."""
    monkeypatch.setenv(KNOB, "1")
    max_model_len = LONG_LINES[0]
    line = _long_line(max_model_len)
    line["neuron_config"] = dict(line["neuron_config"],
                                 kv_segment_size_buckets=[OLD_SEGMENT_MAX])
    config = _serve(served_model_dir, line)
    with fr._parallel_state(tmp_path, config), pytest.raises(ValueError) as refused:
        NeuronModelRunner(config, device=torch.device("cpu"))
    message = str(refused.value)
    # The window rule: largest segment + largest query bucket, rounded up to whole pages.
    window = SERVED_BLOCK * -(-(OLD_SEGMENT_MAX + CHUNK) // SERVED_BLOCK)
    assert window < max_model_len
    assert f"prefill window of {window} tokens" in message, message
    assert f"max_model_len={max_model_len}" in message, message
    assert f"at least {max_model_len - CHUNK} tokens" in message, message
