# SPDX-License-Identifier: Apache-2.0
"""The decode-context bucket configuration accepts any list up to ``max_model_len``.

At e3f38f8 ``validate_decode_context_length_buckets`` refused a bucket equal to
``max_model_len`` ("must be strictly less than"), so ``[2048, 8192, 32768]`` at
``max_model_len`` 32768 did not start. That bucket is the runner's implicit fallback
(``neuron_worker._decode_compile_targets`` and the runner's picker both append
``max_model_len``), so the validator now folds it into the fallback instead of refusing
it. The consistency rules stay: ascending, positive, whole 128-row tiles, nothing past
``max_model_len``.

The runner half reads ``NeuronModelRunner._decode_ctx_bucket_from_max_decode_ctx_len`` on a
runner shell (no model, no device): a decode step goes to the smallest bucket that holds
its context plus this step's token, and the window it gets is that bucket's pages.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_neuron.utils.bucket_utils import validate_decode_context_length_buckets
from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

MAX_MODEL_LEN = 32768
BUCKETS = [2048, 8192, 32768]
PAGE = 128


def test_buckets_up_to_max_model_len_validate():
    # The last bucket is max_model_len: it is the fallback, compiled once.
    assert validate_decode_context_length_buckets(list(BUCKETS), MAX_MODEL_LEN) == [2048, 8192]
    assert validate_decode_context_length_buckets([2048, 8192], MAX_MODEL_LEN) == [2048, 8192]
    assert validate_decode_context_length_buckets([MAX_MODEL_LEN], MAX_MODEL_LEN) == []
    # Any context up to max_model_len, not only the model's 2048-token top-k.
    many = [128 * k for k in range(1, 257)]
    assert validate_decode_context_length_buckets(many, MAX_MODEL_LEN) == many[:-1]


@pytest.mark.parametrize("buckets, match", [
    ([2048, 65536], "must not exceed max_model_len"),
    # Equal to max_model_len and then past it: still refused, not folded.
    ([2048, MAX_MODEL_LEN, 65536], "must not exceed max_model_len"),
    ([2048, 2000], "strictly ascending"),
    ([2048, 2048], "strictly ascending"),
    ([2048, 3000], "divisible by 128"),
    ([0, 2048], "must be positive"),
    ([], "non-empty"),
])
def test_the_consistency_rules_stay(buckets, match):
    with pytest.raises(ValueError, match=match):
        validate_decode_context_length_buckets(buckets, MAX_MODEL_LEN)


def _runner(buckets):
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.max_model_len = MAX_MODEL_LEN
    runner._dcp_size = 1
    runner.neuron_config = SimpleNamespace(
        decode_context_length_buckets=validate_decode_context_length_buckets(
            list(buckets), MAX_MODEL_LEN))
    return runner


@pytest.mark.parametrize("computed, bucket", [
    (0, 2048), (1000, 2048), (2047, 2048),       # this step's token is token 2048
    (2048, 8192), (5000, 8192), (8191, 8192),
    (8192, 32768), (20000, 32768), (32767, 32768),
])
def test_the_runner_picks_the_smallest_bucket_holding_the_context(computed, bucket):
    runner = _runner(BUCKETS)
    assert runner._decode_ctx_bucket_from_max_decode_ctx_len(computed, 0) == bucket
    assert runner._decode_ctx_blocks_from_max_decode_ctx_len(computed, PAGE, 0) \
        == bucket // PAGE


def test_draft_tokens_count_toward_the_context():
    runner = _runner(BUCKETS)
    assert runner._decode_ctx_bucket_from_max_decode_ctx_len(2045, 2) == 2048
    assert runner._decode_ctx_bucket_from_max_decode_ctx_len(2046, 2) == 8192


def test_a_list_of_max_model_len_alone_is_the_fallback_alone():
    runner = _runner([MAX_MODEL_LEN])
    for computed in (0, 2047, 32767):
        assert runner._decode_ctx_bucket_from_max_decode_ctx_len(computed, 0) == MAX_MODEL_LEN
