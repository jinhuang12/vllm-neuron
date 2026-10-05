# SPDX-License-Identifier: Apache-2.0
"""The decode router (RMSNorm + router GEMM + noaux_tc top-8 in one launch).

Old side: the 5938748 ``route_tokens`` snapshot, which pads the token axis to
256 rows and runs the nkilib RMSNorm, the nkilib router and the noaux_tc stage.
New side: ``router_decode.noaux_tc_router_decode`` on the real token count.

Declared tolerances. Index sets must be equal on every row whose 8th/9th
corrected-score gap is at least ``TIE_MARGIN`` (the "ties aside" rows are
counted and bounded). Gate weights: ``rtol=1e-4, atol=1e-6`` on those rows.
Logits: ``rtol=2e-3, atol=2e-3`` everywhere (bf16 activations, fp32 sums).
"""

from __future__ import annotations

import pytest
import torch

from vllm_neuron.functional.moe import router_decode
from vllm_neuron.functional.moe.router_decode import (
    ROUTER_DECODE_MAX_TOKENS,
    noaux_tc_router_decode,
    reset_router_decode_counters,
    router_decode_dispatch_counters,
)

from .decode_fixtures import (
    EPS,
    SCALING,
    TOP_K,
    SimulatorCounter,
    baseline,
    index_sets_equal,
    random_router_inputs,
    realistic_router_inputs,
    tie_rows,
)

TIE_MARGIN = 1e-4
W_RTOL, W_ATOL = 1e-4, 1e-6
L_RTOL, L_ATOL = 2e-3, 2e-3
NEW_KERNEL = "noaux_router_decode_kernel"


def _new(x, gamma, weights, bias):
    reset_router_decode_counters()
    with SimulatorCounter() as sim:
        out = noaux_tc_router_decode(
            x, gamma, weights, bias, top_k=TOP_K, eps=EPS,
            norm_topk_prob=True, routed_scaling_factor=SCALING,
        )
    assert router_decode_dispatch_counters() == (1, 0), "the NKI route was not taken"
    assert sim.kernels == [NEW_KERNEL], f"simulated {sim.kernels}"
    return out


def _old(x, gamma, weights, bias):
    return baseline().route_tokens(
        x.unsqueeze(0), gamma, weights, bias, top_k=TOP_K, eps=EPS,
        norm_topk_prob=True, routed_scaling_factor=SCALING,
    )


def _compare(old, new, bias):
    (old_logits, old_index, old_aff), (logits, index, aff) = old, new
    tokens = old_logits.shape[0]
    assert logits.shape == old_logits.shape and logits.dtype == torch.float32
    assert index.shape == (tokens, TOP_K) and index.dtype == torch.int32
    assert aff.shape == old_aff.shape and aff.dtype == torch.float32
    torch.testing.assert_close(logits, old_logits, rtol=L_RTOL, atol=L_ATOL)
    ties = tie_rows(old_logits, bias, TIE_MARGIN)
    same = index_sets_equal(index, old_index)
    assert bool(same[~ties].all()), (
        f"index sets differ on non-tie rows {(~same & ~ties).nonzero().flatten().tolist()}"
    )
    torch.testing.assert_close(aff[~ties], old_aff[~ties], rtol=W_RTOL, atol=W_ATOL)
    # The support of the scattered weights is the index set, row by row.
    assert torch.equal((aff != 0).sum(-1), torch.full((tokens,), TOP_K))
    assert torch.equal(
        torch.gather(aff, 1, index.long()) > 0,
        torch.ones(tokens, TOP_K, dtype=torch.bool),
    )
    return int(ties.sum())


@pytest.mark.parametrize("tokens", [1, 4])
def test_decode_router_matches_5938748_on_random_inputs(tokens):
    x, gamma, weights, bias = random_router_inputs(tokens)
    old = _old(x, gamma, weights, bias)
    new = _new(x, gamma, weights, bias)
    assert _compare(old, new, bias) == 0


@pytest.mark.parametrize("tokens", [1, 4])
def test_decode_router_matches_5938748_on_checkpoint_router(tokens):
    x, gamma, weights, bias = realistic_router_inputs(tokens)
    old = _old(x, gamma, weights, bias)
    new = _new(x, gamma, weights, bias)
    assert _compare(old, new, bias) == 0


def test_decode_router_token_axis_at_64_rows():
    """The token axis is real: 64 rows, one launch, each row its own selection."""
    x, gamma, weights, bias = random_router_inputs(64, seed=3)
    old = _old(x, gamma, weights, bias)
    new = _new(x, gamma, weights, bias)
    assert _compare(old, new, bias) <= 1


def test_decode_router_rows_are_independent():
    """Row 0 of a 4-row call is the same row routed alone.

    Exact on the index set; fp32 round-off only on the floats, because the CPU
    simulator's matmul may sum a [128, 1] and a [128, 4] stationary in different
    orders (the device's PE columns do not interact).
    """
    x, gamma, weights, bias = random_router_inputs(4, seed=5)
    alone = _new(x[:1], gamma, weights, bias)
    batch = _new(x, gamma, weights, bias)
    assert torch.equal(alone[1][0].sort().values, batch[1][0].sort().values)
    torch.testing.assert_close(alone[0][0], batch[0][0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(alone[2][0], batch[2][0], rtol=1e-5, atol=1e-7)


def test_decode_router_takes_the_torch_oracle_without_a_device(monkeypatch):
    """No device: the torch oracle runs and still matches the old kernel."""
    x, gamma, weights, bias = random_router_inputs(2)
    old = _old(x, gamma, weights, bias)  # the 5938748 kernel, simulated
    monkeypatch.setenv("NKI_SIMULATOR", "0")
    reset_router_decode_counters()
    logits, index, aff = noaux_tc_router_decode(
        x, gamma, weights, bias, top_k=TOP_K, eps=EPS,
        norm_topk_prob=True, routed_scaling_factor=SCALING,
    )
    assert router_decode_dispatch_counters() == (0, 1)
    _compare(old, (logits, index, aff), bias)


def test_decode_router_admits_only_its_token_envelope():
    x, gamma, weights, bias = random_router_inputs(1)
    assert router_decode.can_run_router_decode(x, weights, TOP_K)
    wide = torch.zeros(ROUTER_DECODE_MAX_TOKENS + 1, x.shape[1], dtype=x.dtype)
    assert not router_decode.can_run_router_decode(wide, weights, TOP_K)
    assert not router_decode.can_run_router_decode(x, weights, 4)
