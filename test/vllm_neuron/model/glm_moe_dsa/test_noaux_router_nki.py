# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from torch.fx.experimental.proxy_tensor import make_fx

from vllm_neuron.model.glm_moe_dsa.moe import GlmMoeDsaNoAuxRouter
from vllm_neuron.model.glm_moe_dsa.noaux_router import (
    NOAUX_TC_DENOM_EPS,
    NOAUX_TC_K,
    exact_noaux_tc,
    exact_noaux_tc_torch_reference,
)

_EXACT_ENV = "GLM_ENABLE_EXACT_NOAUX_ROUTER"


def _reference_router(
    router: GlmMoeDsaNoAuxRouter, hidden: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.sigmoid(F.linear(hidden, router.gate.weight).float())
    selected = torch.topk(
        scores + router.correction_bias,
        router.top_k,
        dim=-1,
        sorted=False,
    ).indices
    weights = torch.gather(scores, 1, selected)
    if router.norm_topk_prob:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + NOAUX_TC_DENOM_EPS)
    weights = weights * router.routed_scaling_factor
    affinities = torch.zeros_like(scores).scatter(1, selected, weights)
    return affinities, selected.to(torch.int32)


def test_feature_flag_is_fixed_when_router_is_constructed(monkeypatch) -> None:
    monkeypatch.delenv(_EXACT_ENV, raising=False)
    default_router = GlmMoeDsaNoAuxRouter(16, 256, 8)
    monkeypatch.setenv(_EXACT_ENV, "1")
    exact_router = GlmMoeDsaNoAuxRouter(16, 256, 8)
    monkeypatch.delenv(_EXACT_ENV)

    assert default_router.use_exact_noaux_router is False
    assert exact_router.use_exact_noaux_router is True


def test_default_path_is_unchanged_when_flag_is_off(monkeypatch) -> None:
    monkeypatch.delenv(_EXACT_ENV, raising=False)
    torch.manual_seed(1701)
    router = GlmMoeDsaNoAuxRouter(32, 16, 4, routed_scaling_factor=2.5)
    hidden = torch.randn(7, 32)
    expected_affinities, expected_selected = _reference_router(router, hidden)

    with (
        patch("vllm_neuron.model.glm_moe_dsa.moe.can_run_kernel", return_value=False),
        patch(
            "vllm_neuron.model.glm_moe_dsa.moe.exact_noaux_tc",
            side_effect=AssertionError("disabled path called the candidate"),
        ),
    ):
        actual_affinities, actual_selected = router(hidden)

    assert torch.equal(actual_selected, expected_selected)
    torch.testing.assert_close(actual_affinities, expected_affinities)


def test_static_graph_removes_torch_router_postprocessing(monkeypatch) -> None:
    hidden = torch.randn(32, 64)
    monkeypatch.delenv(_EXACT_ENV, raising=False)
    default_router = GlmMoeDsaNoAuxRouter(64, 256, 8)
    monkeypatch.setenv(_EXACT_ENV, "1")
    exact_router = GlmMoeDsaNoAuxRouter(64, 256, 8)
    exact_router.load_state_dict(default_router.state_dict())

    with patch(
        "vllm_neuron.model.glm_moe_dsa.moe.can_run_kernel",
        return_value=False,
    ):
        default_graph = make_fx(default_router)(hidden)

    def fake_exact(
        router_logits: torch.Tensor,
        correction_bias: torch.Tensor,
        **_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del correction_bias
        selected = torch.zeros(
            router_logits.shape[0], 8, dtype=torch.int32, device=router_logits.device
        )
        affinities = torch.zeros_like(router_logits)
        return selected, affinities

    with patch(
        "vllm_neuron.model.glm_moe_dsa.moe.exact_noaux_tc",
        side_effect=fake_exact,
    ):
        exact_graph = make_fx(exact_router)(hidden)

    default_targets = {str(node.target) for node in default_graph.graph.nodes}
    exact_targets = {str(node.target) for node in exact_graph.graph.nodes}
    removed = {
        "aten.sigmoid.default",
        "aten.topk.default",
        "aten.gather.default",
        "aten.sum.dim_IntList",
        "aten.div.Tensor",
        "aten.scatter.src",
    }
    assert removed <= default_targets
    assert removed.isdisjoint(exact_targets)
    assert len(list(exact_graph.graph.nodes)) < len(list(default_graph.graph.nodes))


@pytest.mark.parametrize("num_tokens", [1, 32, 128])
def test_cpu_candidate_matches_independent_reference(
    monkeypatch, num_tokens: int
) -> None:
    monkeypatch.setenv(_EXACT_ENV, "1")
    torch.manual_seed(1900 + num_tokens)
    router = GlmMoeDsaNoAuxRouter(64, 256, 8, routed_scaling_factor=2.5)
    hidden = torch.randn(num_tokens, 64)
    expected_affinities, expected_selected = _reference_router(router, hidden)

    with patch(
        "vllm_neuron.model.glm_moe_dsa.noaux_router.can_run_kernel",
        return_value=False,
    ):
        actual_affinities, actual_selected = router(hidden)

    assert torch.equal(
        torch.sort(actual_selected, dim=-1).values,
        torch.sort(expected_selected, dim=-1).values,
    )
    torch.testing.assert_close(actual_affinities, expected_affinities)
    assert actual_selected.dtype == torch.int32
    assert actual_affinities.dtype == torch.float32
    assert actual_selected.shape == (num_tokens, 8)
    assert actual_affinities.shape == (num_tokens, 256)


def test_correction_bias_selects_global_ids_but_does_not_weight_them() -> None:
    logits = torch.zeros(2, 256)
    correction_bias = torch.full((256,), -1.0)
    correction_bias[248:] = torch.arange(8, dtype=torch.float32)

    selected, affinities = exact_noaux_tc_torch_reference(
        logits,
        correction_bias,
        routed_scaling_factor=2.5,
    )

    assert set(selected[0].tolist()) == set(range(248, 256))
    torch.testing.assert_close(
        affinities[:, 248:],
        torch.full((2, 8), 2.5 / 8),
        rtol=0,
        atol=0,
    )
    assert torch.count_nonzero(affinities[:, :248]) == 0


def test_tied_scores_select_eight_distinct_experts_with_equal_weights() -> None:
    logits = torch.zeros(32, 256)
    selected, affinities = exact_noaux_tc_torch_reference(
        logits,
        torch.zeros(256),
        routed_scaling_factor=2.5,
    )

    for row in selected:
        assert len(set(row.tolist())) == NOAUX_TC_K
    assert torch.all(torch.count_nonzero(affinities, dim=-1) == NOAUX_TC_K)
    nonzero = affinities[affinities != 0]
    torch.testing.assert_close(
        nonzero,
        torch.full_like(nonzero, 2.5 / NOAUX_TC_K),
        rtol=0,
        atol=0,
    )


def test_zero_denominator_is_finite_and_stays_zero() -> None:
    logits = torch.full((32, 256), float("-inf"))
    selected, affinities = exact_noaux_tc_torch_reference(
        logits,
        torch.zeros(256),
        routed_scaling_factor=2.5,
    )

    assert selected.shape == (32, 8)
    assert torch.isfinite(affinities).all()
    assert torch.count_nonzero(affinities) == 0


@pytest.mark.skipif(
    os.environ.get("NKI_SIMULATOR") != "1",
    reason="requires the NKI CPU simulator",
)
@pytest.mark.parametrize("num_tokens", [1, 32, 128])
def test_nki_simulator_matches_reference(num_tokens: int) -> None:
    torch.manual_seed(2100 + num_tokens)
    logits = torch.randn(num_tokens, 256, dtype=torch.float32)
    bias = torch.randn(256, dtype=torch.float32) * 0.05
    expected_selected, expected_affinities = exact_noaux_tc_torch_reference(
        logits,
        bias,
        routed_scaling_factor=2.5,
    )

    with patch(
        "vllm_neuron.model.glm_moe_dsa.noaux_router.can_run_kernel",
        return_value=True,
    ):
        actual_selected, actual_affinities = exact_noaux_tc(
            logits,
            bias,
            routed_scaling_factor=2.5,
        )

    assert torch.equal(
        torch.sort(actual_selected, dim=-1).values,
        torch.sort(expected_selected, dim=-1).values,
    )
    torch.testing.assert_close(
        actual_affinities,
        expected_affinities,
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.skipif(
    os.environ.get("NKI_SIMULATOR") != "1",
    reason="requires the NKI CPU simulator",
)
@pytest.mark.parametrize("case", ["ties", "zero_denominator"])
def test_nki_simulator_edge_cases(case: str) -> None:
    logits = torch.zeros(32, 256)
    if case == "zero_denominator":
        logits.fill_(float("-inf"))

    with patch(
        "vllm_neuron.model.glm_moe_dsa.noaux_router.can_run_kernel",
        return_value=True,
    ):
        selected, affinities = exact_noaux_tc(
            logits,
            torch.zeros(256),
            routed_scaling_factor=2.5,
        )

    assert selected.shape == (32, 8)
    assert torch.isfinite(affinities).all()
    for row in selected:
        assert len(set(row.tolist())) == NOAUX_TC_K
    if case == "ties":
        assert torch.all(torch.count_nonzero(affinities, dim=-1) == NOAUX_TC_K)
        torch.testing.assert_close(
            affinities.sum(dim=-1),
            torch.full((32,), 2.5),
            rtol=1e-5,
            atol=1e-6,
        )
    else:
        assert torch.count_nonzero(affinities) == 0
