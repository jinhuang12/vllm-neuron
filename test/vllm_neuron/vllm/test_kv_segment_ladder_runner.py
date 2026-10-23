# SPDX-License-Identifier: Apache-2.0
"""The runner's KV segment list for a windowed-prefill model that sets none.

GLM-5.3-Flash with no ``kv_segment_size_buckets`` gets a segment every
``VLLM_NEURON_KV_SEGMENT_STEP`` tokens up to the first whose window covers
``max_model_len``, at most ``VLLM_NEURON_KV_SEGMENT_MAX_SEGMENTS`` of them
(``bucket_utils.kv_segment_step_ladder``). Step 0 keeps the resolver's list, completed
with a covering segment. An explicit list is kept as set. Each case builds the real
config (``EngineArgs.create_engine_config``) and the real runner constructor on CPU,
as ``test_uncap_context.py`` does, at ``max_model_len`` 131,072 and one 1,024-token
chunk: a length whose decode candidate row the runner serves at DCP 1.
"""

from __future__ import annotations

import pytest
import torch

from vllm_neuron import envs
from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_first_request as fr
from test.vllm_neuron.vllm import test_uncap_context as uc
from test.vllm_neuron.vllm.test_uncap_context import (  # noqa: F401  (fixtures)
    _restore_platform_policy,
    served_model_dir,
)

pytestmark = [pytest.mark.fast]

MAX_MODEL_LEN = 131072
CHUNK = 1024
STEP_KNOB = "VLLM_NEURON_KV_SEGMENT_STEP"
BOUND_KNOB = "VLLM_NEURON_KV_SEGMENT_MAX_SEGMENTS"


def _line(segments: list[int] | None = None) -> dict:
    """bs=1, one 1,024-token chunk, ``max_model_len`` 131,072; no segment list unless given."""
    neuron_config = {
        "ep_degree": 16,
        "num_batched_tokens_buckets": [CHUNK],
        "decode_context_length_buckets": [2048],
        "hybrid_kv_block_size": uc.SERVED_BLOCK,
        "on_device_sampling_config": {"all_greedy": True},
    }
    if segments is not None:
        neuron_config["kv_segment_size_buckets"] = segments
    return dict(max_model_len=MAX_MODEL_LEN, max_num_seqs=1, max_num_batched_tokens=CHUNK,
                neuron_config=neuron_config)


def _segments(model_dir, tmp_path, line: dict) -> list[int]:
    config = uc._serve(model_dir, line)
    with fr._parallel_state(tmp_path, config):
        runner = NeuronModelRunner(config, device=torch.device("cpu"))
    return list(runner.neuron_config.kv_segment_size_buckets)


@pytest.fixture(autouse=True)
def _served_sampling(monkeypatch):
    monkeypatch.setenv(uc.KNOB, "1")
    monkeypatch.delenv(STEP_KNOB, raising=False)
    monkeypatch.delenv(BOUND_KNOB, raising=False)


def test_the_knobs_default_to_a_16384_step_and_16_segments():
    assert envs.VLLM_NEURON_KV_SEGMENT_STEP == 16384
    assert envs.VLLM_NEURON_KV_SEGMENT_MAX_SEGMENTS == 16


@pytest.mark.parametrize("value, read", [("0", 0), ("4", 4), ("-1", -1)])
def test_the_bound_knob_reads_its_value(value, read, monkeypatch):
    monkeypatch.setenv(BOUND_KNOB, value)
    assert envs.VLLM_NEURON_KV_SEGMENT_MAX_SEGMENTS == read


def test_no_list_gets_a_segment_every_step(served_model_dir, tmp_path):
    assert _segments(served_model_dir, tmp_path, _line()) == [
        16384 * (index + 1) for index in range(8)]


def test_the_bound_scales_the_step(served_model_dir, tmp_path, monkeypatch):
    """Bound 4 at eight 16,384-token segments: the step becomes 32,768."""
    monkeypatch.setenv(BOUND_KNOB, "4")
    assert _segments(served_model_dir, tmp_path, _line()) == [32768, 65536, 98304, 131072]


def test_a_negative_bound_does_not_start(served_model_dir, tmp_path, monkeypatch):
    monkeypatch.setenv(BOUND_KNOB, "-1")
    with pytest.raises(ValueError, match="KV segment bound must be 0 or a positive"):
        _segments(served_model_dir, tmp_path, _line())


def test_step_zero_keeps_the_completed_resolver_list(served_model_dir, tmp_path, monkeypatch):
    """The resolver's one segment of the token budget, and one of max_model_len (no
    segmented-kernel size covers 131,072)."""
    monkeypatch.setenv(STEP_KNOB, "0")
    assert _segments(served_model_dir, tmp_path, _line()) == [CHUNK, MAX_MODEL_LEN]


def test_an_explicit_list_is_kept(served_model_dir, tmp_path, monkeypatch):
    monkeypatch.setenv(BOUND_KNOB, "1")
    explicit = [32768, 65536, 131072]
    assert _segments(served_model_dir, tmp_path, _line(explicit)) == explicit
