# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash serves up to 64 concurrent decode requests: the one-request refusals are gone.

Wave 1 left four places that served one request per decode step. Each is read here at
source level, by the exact text of its refusal, so a revert that restores one of them
fails this file before any model runs:

1. the runner's carrier builder refused a sparse (DSA) layer handed more than one state
   slot (``len(state_slots) != 1``);
2. ``Glm5NextMLAAttention.attend`` refused ``batch_size != 1``;
3. ``Glm5NextKDAAttention.forward`` served a concurrent decode by calling itself once per
   request (one fused kernel dispatch per request per layer);
4. the compact MoE decode kernel asserted one token per block (``assert q == 1``).

The second half reads the configuration the bs=64 @ 8k gate serves with: 64 sequences,
the power-of-two decode buckets the warmup compiles, and the decode context buckets.
"""

from __future__ import annotations

import pathlib

import pytest

from vllm_neuron.utils.bucket_utils import (
    get_decode_padded_batch_size,
    get_default_num_seqs_buckets,
    validate_decode_context_length_buckets,
    validate_num_seqs_buckets,
)

pytestmark = [pytest.mark.fast]

ROOT = pathlib.Path(__file__).resolve().parents[3]
RUNNER = ROOT / "vllm_neuron/vllm/worker/neuron_model_runner.py"
MODEL = ROOT / "vllm_neuron/model/glm5_next/model_fp8.py"
MOE_DECODE = ROOT / "vllm_neuron/functional/moe/moe_fused_fp8_decode.py"

#: The refusal text of each site that declines a batch, verbatim.
REFUSALS = {
    RUNNER: (
        "if len(state_slots) != 1:",
        "is a sparse-attention layer and one ",
        "per forward; this call carries",
    ),
    MODEL: (
        "if int(batch_size) != 1:",
        "the MLA decode path serves one sequence at a time",
        "One carrier per request, and the loop is the dispatch",
    ),
    MOE_DECODE: ("assert q == 1",),
}

#: The served bs=64 point.
MAX_NUM_SEQS = 64
NUM_SEQS_BUCKETS = [1, 2, 4, 8, 16, 32, 64]
MAX_MODEL_LEN = 8192
#: ``max_model_len`` is the implicit last decode context bucket, so only 2048 is listed.
DECODE_CONTEXT_BUCKETS = [2048]


@pytest.mark.parametrize("path", sorted(REFUSALS), ids=lambda p: p.name)
def test_the_one_request_refusals_are_gone(path):
    source = path.read_text()
    present = [text for text in REFUSALS[path] if text in source]
    assert not present, f"{path.name} still carries the one-request refusal(s) {present}"


def test_the_kda_decode_no_longer_recurses_per_request():
    """The concurrent decode is one stacked call; only the switched-off stage path loops."""
    source = MODEL.read_text()
    start = source.index("class Glm5NextKDAAttention")
    end = source.index("\nclass ", start + 1)
    body = source[start:end]
    assert body.count("self.forward(") == 1, (
        "Glm5NextKDAAttention must call itself at most once (the stage path behind "
        "VLLM_NEURON_KDA_FUSED_DECODE=0)"
    )
    # The stacked call is taken before the per-request fallback is reached.
    assert "self._fused_decode_requests(" in body
    assert body.index("self._fused_decode_requests(") < body.index("self.forward(")
    stacked = body[body.index("def _fused_decode_requests"):]
    assert "kda_fused_decode(" in stacked and "torch.stack(convs)" in stacked


def test_sixty_four_sequences_pass_the_bucket_validator():
    assert get_default_num_seqs_buckets(MAX_NUM_SEQS) == NUM_SEQS_BUCKETS
    assert validate_num_seqs_buckets(list(NUM_SEQS_BUCKETS), MAX_NUM_SEQS) == NUM_SEQS_BUCKETS
    # The validator requires the last bucket to equal max_num_seqs.
    with pytest.raises(ValueError, match="must equal max_num_seqs"):
        validate_num_seqs_buckets([1, 2, 4, 8, 16, 32], MAX_NUM_SEQS)


@pytest.mark.parametrize(
    "num_reqs, bucket", [(1, 1), (2, 2), (3, 4), (5, 8), (33, 64), (63, 64), (64, 64)]
)
def test_a_decode_batch_pads_to_the_next_bucket(num_reqs, bucket):
    assert get_decode_padded_batch_size(num_reqs, 1, NUM_SEQS_BUCKETS) == bucket


def test_the_gate_context_buckets_validate_and_max_model_len_is_the_fallback():
    assert (
        validate_decode_context_length_buckets(list(DECODE_CONTEXT_BUCKETS), MAX_MODEL_LEN)
        == DECODE_CONTEXT_BUCKETS
    )
    # Listing max_model_len itself is accepted and folded: it is the implicit last bucket,
    # so the returned list leaves it out and that graph compiles once.
    assert validate_decode_context_length_buckets(
        [2048, MAX_MODEL_LEN], MAX_MODEL_LEN) == [2048]
    # A bucket above max_model_len is still refused.
    with pytest.raises(ValueError, match="must not exceed max_model_len"):
        validate_decode_context_length_buckets([2048, 2 * MAX_MODEL_LEN], MAX_MODEL_LEN)
