# SPDX-License-Identifier: Apache-2.0
"""The prefill router: RMSNorm in torch, router GEMM + noaux_tc top-8 in one launch.

Old side: 8aa22fa's prefill route, ``router.noaux_tc_rmsnorm_router_topk`` on the
pre-norm activations (nkilib RMSNorm, nkilib router, noaux_tc stage, one kernel).
New side: ``router_prefill.noaux_tc_router_prefill``: the RMSNorm's scale and its first
bf16 rounding, ``bf16(x * rstd)``, written in torch so it traces into XLA, then a kernel
that applies the gain the way the fused kernel's norm stage does (its second rounding)
and runs the same nkilib router call and the same noaux_tc stage.

Declared tolerance. The only arithmetic that moved is the RMSNorm's fp32 sum of
``x**2`` (torch's order instead of the kernel's ``activation_reduce``) and its
``rsqrt``. When a row's ``rstd`` lands one fp32 ulp away, the elements of that row
whose ``x * rstd`` sits on a bf16 rounding boundary move by one bf16 step. Measured
on the checkpoint routers of layers 3, 4, 10, 20, 40 at 1024 rows (10,240 rows): 4
rows moved (18-40 elements each), logits moved at most 1.05e-3, no index set changed.
So, per comparison:

* at most ``MOVED_ROW_FRACTION`` (1%) of rows may differ at all ("moved rows");
* every other row is bitwise equal: logits, the index row and the scattered weights
  (the router GEMM and the noaux_tc stage are the fused kernel's, unchanged);
* on moved rows: logits ``atol=MOVED_L_ATOL`` (5e-3); index sets equal on every row
  whose 8th/9th corrected-score gap is at least ``TIE_MARGIN`` (1e-4, the decode
  router's); gate weights ``atol=MOVED_W_ATOL`` (5e-3, 0.2% of the 2.5 scale);
* tie rows (gap below ``TIE_MARGIN``) are counted and at most 5% of rows.
"""

from __future__ import annotations

import pytest
import torch

from vllm_neuron.functional.moe import router_prefill
from vllm_neuron.functional.moe.router import (
    noaux_tc_dispatch_counters,
    noaux_tc_rmsnorm_router_topk,
    reset_noaux_tc_counters,
)
from vllm_neuron.functional.moe.router_decode import DECODE_ROUTE_MAX_TOKENS

from .decode_fixtures import (
    EPS,
    SCALING,
    TOP_K,
    SimulatorCounter,
    index_sets_equal,
    random_router_inputs,
    realistic_router_inputs,
    tie_rows,
)

TIE_MARGIN = 1e-4
MOVED_ROW_FRACTION = 0.01
MOVED_L_ATOL = 5e-3
MOVED_W_ATOL = 5e-3
TIE_ROW_FRACTION = 0.05
NEW_KERNEL = "noaux_tc_router_prefill_kernel"
OLD_KERNEL = "_noaux_tc_rmsnorm_router_topk_nki"


def _new(x, gamma, weights, bias):
    """The prefill router, asserting its NKI route ran the new kernel once."""
    reset_noaux_tc_counters()
    with SimulatorCounter() as sim:
        out = router_prefill.noaux_tc_router_prefill(
            x, gamma, weights, bias, top_k=TOP_K, eps=EPS,
            norm_topk_prob=True, routed_scaling_factor=SCALING,
        )
    assert noaux_tc_dispatch_counters() == (1, 0), "the NKI route was not taken"
    assert sim.kernels == [NEW_KERNEL], f"simulated {sim.kernels}"
    return out


def _old(x, gamma, weights, bias):
    """8aa22fa's prefill route on the same rows (``route_tokens`` above 64 tokens)."""
    with SimulatorCounter() as sim:
        logits, index, aff, _sub = noaux_tc_rmsnorm_router_topk(
            hidden_states=x.unsqueeze(0), gamma=gamma, router_weights=weights,
            correction_bias=bias, top_k=TOP_K, eps=EPS, norm_topk_prob=True,
            routed_scaling_factor=SCALING,
        )
    assert sim.kernels == [OLD_KERNEL], f"simulated {sim.kernels}"
    return logits, index, aff


def _compare(old, new, bias) -> dict:
    """Assert the declared tolerance; return what was measured."""
    (old_logits, old_index, old_aff), (logits, index, aff) = old, new
    tokens = old_logits.shape[0]
    assert logits.shape == old_logits.shape and logits.dtype == torch.float32
    assert index.shape == (tokens, TOP_K) and index.dtype == torch.int32
    assert aff.shape == old_aff.shape and aff.dtype == torch.float32
    moved = (logits != old_logits).any(dim=-1)
    kept = ~moved
    assert int(moved.sum()) <= max(1, int(tokens * MOVED_ROW_FRACTION)), (
        f"{int(moved.sum())} of {tokens} rows have other logits"
    )
    assert torch.equal(index[kept], old_index[kept]), "index differs on a bitwise row"
    assert torch.equal(aff[kept], old_aff[kept]), "weights differ on a bitwise row"
    torch.testing.assert_close(logits[moved], old_logits[moved], rtol=0.0,
                               atol=MOVED_L_ATOL)
    ties = tie_rows(old_logits, bias, TIE_MARGIN)
    same = index_sets_equal(index, old_index)
    assert bool(same[~ties].all()), (
        f"index sets differ on non-tie rows {(~same & ~ties).nonzero().flatten().tolist()}"
    )
    checked = moved & ~ties
    torch.testing.assert_close(aff[checked], old_aff[checked], rtol=0.0,
                               atol=MOVED_W_ATOL)
    assert int(ties.sum()) <= max(1, int(tokens * TIE_ROW_FRACTION))
    # The support of the scattered weights is the index set, row by row.
    assert torch.equal((aff != 0).sum(-1), torch.full((tokens,), TOP_K))
    assert torch.equal(
        torch.gather(aff, 1, index.long()) > 0,
        torch.ones(tokens, TOP_K, dtype=torch.bool),
    )
    return {
        "tokens": tokens,
        "moved_rows": moved.nonzero().flatten().tolist(),
        "tie_rows": int(ties.sum()),
        "rows_with_other_index_set": int((~same).sum()),
        "logits_max_abs_diff": float((logits - old_logits).abs().max()),
        "weights_max_abs_diff": float((aff - old_aff).abs().max()),
    }


@pytest.mark.parametrize("tokens", [65, 1024])
def test_prefill_router_matches_8aa22fa_on_checkpoint_router(tokens):
    """Checkpoint layer-3 router, gain and correction bias; residual-like activations."""
    x, gamma, weights, bias = realistic_router_inputs(tokens, layer=3, seed=41 + tokens)
    report = _compare(_old(x, gamma, weights, bias), _new(x, gamma, weights, bias), bias)
    print(f"T={tokens} checkpoint router: {report}")


def test_prefill_router_matches_8aa22fa_on_random_inputs_with_pad_rows():
    """300 rows: not a multiple of the 256-row launch unit, so 212 pad rows ride along."""
    x, gamma, weights, bias = random_router_inputs(300, seed=13)
    report = _compare(_old(x, gamma, weights, bias), _new(x, gamma, weights, bias), bias)
    print(f"T=300 random router: {report}")


def test_prefill_router_rows_are_independent():
    """Rows 0..127 of a 300-row call are the same rows routed alone (pad rows inert)."""
    x, gamma, weights, bias = random_router_inputs(300, seed=17)
    whole = _new(x, gamma, weights, bias)
    part = _new(x[:128].contiguous(), gamma, weights, bias)
    assert torch.equal(whole[1][:128], part[1])
    assert torch.equal(whole[2][:128], part[2])
    assert torch.equal(whole[0][:128], part[0])


def test_router_rms_scale_is_the_first_rounding_only():
    """The XLA side stops at ``bf16(x * rstd)``; the gamma multiply is the kernel's.

    On the device neuronx-cc folds a ``bf16 -> fp32`` convert that follows an
    ``fp32 -> bf16`` one, so a torch ``bf16(bf16(x * rstd) * gamma)`` runs there as one
    rounding (measured: 99.99% of elements equal ``bf16(x * rstd *
    gamma)``, 78% equal the two-rounding value). With nothing after the first rounding in
    XLA there is no pair to fold, and the kernel's gamma multiply rounds the way the
    fused kernel's norm stage does (same ``tensor_tensor`` on the same SBUF tile).
    """
    x, _gamma, _weights, _bias = realistic_router_inputs(256, layer=3, seed=5)
    got = router_prefill.router_rms_scale(x, EPS)
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + EPS)
    assert got.dtype == torch.bfloat16 and got.shape == x.shape
    assert torch.equal(got, (xf * rstd).to(torch.bfloat16))


def test_round_to_bf16_grid_is_round_to_nearest_even():
    """Veltkamp's split equals ``fp32 -> bf16 -> fp32`` on random values and exact ties.

    The device folds that convert pair; the arithmetic form is what the router's scale
    reads its rows through, so it must be the same rounding everywhere it is honoured.
    """
    gen = torch.Generator().manual_seed(3)
    x = torch.randn(1 << 20, generator=gen) * torch.exp(3 * torch.randn(1 << 20, generator=gen))
    ties = (torch.randn(1 << 18, generator=gen).to(torch.bfloat16).float()
            .view(torch.int32) + 0x8000).view(torch.float32)
    for values in (x, ties):
        assert torch.equal(router_prefill.round_to_bf16_grid(values),
                           values.to(torch.bfloat16).float())
    on_grid = x.to(torch.bfloat16).float()
    assert torch.equal(router_prefill.round_to_bf16_grid(on_grid), on_grid)


@pytest.mark.parametrize("tokens", [1, DECODE_ROUTE_MAX_TOKENS])
def test_prefill_route_declines_decode_token_counts(tokens):
    """Up to 64 rows the call site keeps its decode route, untouched."""
    x, _gamma, weights, _bias = random_router_inputs(tokens)
    assert not router_prefill.prefill_route_admits(x, weights, TOP_K)


def test_prefill_route_admits_prefill_token_counts():
    x, _gamma, weights, _bias = random_router_inputs(DECODE_ROUTE_MAX_TOKENS + 1)
    assert router_prefill.prefill_route_admits(x, weights, TOP_K)
    assert router_prefill.prefill_route_admits(x.unsqueeze(0), weights, TOP_K)


def test_prefill_route_kill_switch_declines(monkeypatch):
    x, _gamma, weights, _bias = random_router_inputs(128)
    monkeypatch.setenv(router_prefill.PREFILL_ROUTER_ENV, "0")
    assert not router_prefill.prefill_route_admits(x, weights, TOP_K)
    monkeypatch.setenv(router_prefill.PREFILL_ROUTER_ENV, "1")
    assert router_prefill.prefill_route_admits(x, weights, TOP_K)


def test_prefill_route_declines_without_a_kernel_route(monkeypatch):
    """No NKI device and no simulator: the call site keeps 8aa22fa's route."""
    x, _gamma, weights, _bias = random_router_inputs(128)
    monkeypatch.setenv("NKI_SIMULATOR", "0")
    assert not router_prefill.prefill_route_admits(x, weights, TOP_K)


@pytest.mark.parametrize(
    "case",
    ["top_k", "fp32_activations", "experts_over_512", "hidden_not_128_blocks"],
)
def test_prefill_route_declines_what_the_kernel_cannot_serve(case):
    x, _gamma, weights, _bias = random_router_inputs(128)
    top_k = TOP_K
    if case == "top_k":
        top_k = 6
    elif case == "fp32_activations":
        x = x.float()
    elif case == "experts_over_512":
        weights = torch.zeros(x.shape[1], 520, dtype=torch.bfloat16)
    elif case == "hidden_not_128_blocks":
        x = x[:, :4000].contiguous()
        weights = weights[:4000].contiguous()
    assert not router_prefill.prefill_route_admits(x, weights, top_k)
