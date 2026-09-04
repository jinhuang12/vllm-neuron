# SPDX-License-Identifier: Apache-2.0
"""Exact GLM-5.2 ``noaux_tc`` routing in one NKI dispatch.

The checkpoint selects experts with ``sigmoid(logits) + correction_bias``.
It weights the selected experts with the unbiased sigmoid scores.  The vendor
router cannot express that split.  This kernel performs the split, top-8
selection, normalization, routed scaling, and dense affinity scatter together.
"""

from __future__ import annotations

import nki
import nki.isa as nisa
import nki.language as nl
import torch
from torch import Tensor

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron.utils.neuron_utils import can_run_kernel

NOAUX_TC_TILE = 128
NOAUX_TC_K = 8
NOAUX_TC_DENOM_EPS = 1e-20
_NOAUX_TC_MAX_EXPERTS = 512


class NoauxTcRouterError(ValueError):
    """The exact router cannot serve the requested geometry."""


def _require_extents(num_experts: int, top_k: int) -> None:
    if top_k != NOAUX_TC_K:
        raise NoauxTcRouterError(f"top_k must be exactly {NOAUX_TC_K}, got {top_k}")
    if not NOAUX_TC_K <= num_experts <= _NOAUX_TC_MAX_EXPERTS:
        raise NoauxTcRouterError(
            f"num_experts must be in [{NOAUX_TC_K}, "
            f"{_NOAUX_TC_MAX_EXPERTS}], got {num_experts}"
        )


def _legalize_bias(correction_bias: Tensor, num_experts: int) -> Tensor:
    if correction_bias.ndim == 1:
        correction_bias = correction_bias.unsqueeze(0)
    if correction_bias.shape != (1, num_experts):
        raise NoauxTcRouterError(
            "correction_bias must have shape [E] or [1, E], "
            f"got {tuple(correction_bias.shape)} for E={num_experts}"
        )
    return correction_bias.to(torch.float32).contiguous()


def _pad_target(num_tokens: int) -> int:
    if num_tokens < 1:
        raise NoauxTcRouterError("the router requires at least one token")
    return -(-num_tokens // NOAUX_TC_TILE) * NOAUX_TC_TILE


def _pad_tokens(x: Tensor, padded_tokens: int) -> Tensor:
    """Repeat the last row so padding does not introduce synthetic ties."""
    num_tokens = x.shape[0]
    if padded_tokens == num_tokens:
        return x.contiguous()
    return torch.cat(
        (x, x[-1:].expand(padded_tokens - num_tokens, -1)), dim=0
    ).contiguous()


def _noaux_tc_stage(
    router_logits_hbm,
    correction_bias_hbm,
    expert_index_hbm,
    expert_affinities_hbm,
    num_tokens: int,
    num_experts: int,
    norm_topk_prob: bool,
    routed_scaling_factor: float,
):
    """Apply the GLM-5.2 selection/weight split to fp32 router logits."""
    bias_sb = nl.load(correction_bias_hbm)

    for token_tile in range(num_tokens // NOAUX_TC_TILE):
        token_start = token_tile * NOAUX_TC_TILE
        rows = NOAUX_TC_TILE
        logits = nl.load(
            router_logits_hbm[token_start : token_start + rows, :],
            dtype=nl.float32,
        )
        scores = nl.sigmoid(logits, dtype=nl.float32)

        choice = nl.ndarray((rows, num_experts), dtype=nl.float32, buffer=nl.sbuf)
        bias = nl.broadcast_to(bias_sb, (rows, num_experts))
        nisa.tensor_tensor(dst=choice, data1=scores, data2=bias, op=nl.add)

        top_values = nl.ndarray((rows, NOAUX_TC_K), dtype=nl.float32, buffer=nl.sbuf)
        nisa.max8(dst=top_values, src=choice)
        top_indices = nl.ndarray((rows, NOAUX_TC_K), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.nc_find_index8(dst=top_indices, data=choice, vals=top_values)

        indices_fp32 = nl.ndarray((rows, NOAUX_TC_K), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=indices_fp32, src=top_indices)
        columns = nl.ndarray((rows, num_experts), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(
            dst=columns,
            pattern=[[1, num_experts]],
            offset=0,
            channel_multiplier=0,
        )

        mask = nl.ndarray((rows, num_experts), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=mask, value=0.0)
        hit = nl.ndarray((rows, num_experts), dtype=nl.float32, buffer=nl.sbuf)
        for k_index in range(NOAUX_TC_K):
            nisa.tensor_scalar(
                dst=hit,
                data=columns,
                op0=nl.equal,
                operand0=indices_fp32[:, k_index : k_index + 1],
            )
            nisa.tensor_tensor(dst=mask, data1=mask, data2=hit, op=nl.add)

        selected_scores = nl.ndarray(
            (rows, num_experts), dtype=nl.float32, buffer=nl.sbuf
        )
        nisa.tensor_tensor(
            dst=selected_scores,
            data1=scores,
            data2=mask,
            op=nl.multiply,
        )

        output = nl.ndarray((rows, num_experts), dtype=nl.float32, buffer=nl.sbuf)
        if norm_topk_prob:
            row_sum = nl.sum(selected_scores, axis=1, keepdims=True, dtype=nl.float32)
            denominator = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=denominator,
                data=row_sum,
                op0=nl.add,
                operand0=NOAUX_TC_DENOM_EPS,
            )
            reciprocal = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.reciprocal(dst=reciprocal, data=denominator)
            nisa.tensor_scalar(
                dst=output,
                data=selected_scores,
                op0=nl.multiply,
                operand0=reciprocal,
                op1=nl.multiply,
                operand1=float(routed_scaling_factor),
            )
        else:
            nisa.tensor_scalar(
                dst=output,
                data=selected_scores,
                op0=nl.multiply,
                operand0=float(routed_scaling_factor),
            )

        nl.store(
            expert_affinities_hbm[token_start : token_start + rows, :],
            value=output,
        )
        nl.store(
            expert_index_hbm[token_start : token_start + rows, :],
            value=top_indices,
        )


@nki.jit
def _exact_noaux_tc_nki(
    router_logits,
    correction_bias,
    norm_topk_prob: bool = True,
    routed_scaling_factor: float = 1.0,
):
    num_tokens, num_experts = router_logits.shape
    expert_index = nl.ndarray(
        (num_tokens, NOAUX_TC_K), dtype=nl.uint32, buffer=nl.shared_hbm
    )
    expert_affinities = nl.ndarray(
        (num_tokens, num_experts), dtype=nl.float32, buffer=nl.shared_hbm
    )
    _noaux_tc_stage(
        router_logits,
        correction_bias,
        expert_index,
        expert_affinities,
        num_tokens,
        num_experts,
        norm_topk_prob,
        routed_scaling_factor,
    )
    return expert_index, expert_affinities


def exact_noaux_tc_torch_reference(
    router_logits: Tensor,
    correction_bias: Tensor,
    *,
    top_k: int = NOAUX_TC_K,
    norm_topk_prob: bool = True,
    routed_scaling_factor: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Independent CPU reference for tests and CPU-mode model execution."""
    if router_logits.ndim != 2:
        raise NoauxTcRouterError(
            f"router_logits must have shape [T, E], got {tuple(router_logits.shape)}"
        )
    _, num_experts = router_logits.shape
    _require_extents(num_experts, top_k)
    bias = _legalize_bias(correction_bias, num_experts).reshape(-1)
    scores = torch.sigmoid(router_logits.to(torch.float32))
    selected_experts = torch.topk(scores + bias, top_k, dim=-1, sorted=False).indices
    selected_weights = torch.gather(scores, 1, selected_experts)
    if norm_topk_prob:
        selected_weights = selected_weights / (
            selected_weights.sum(dim=-1, keepdim=True) + NOAUX_TC_DENOM_EPS
        )
    selected_weights = selected_weights * routed_scaling_factor
    affinities = torch.zeros_like(scores).scatter(1, selected_experts, selected_weights)
    return selected_experts.to(torch.int32), affinities


def exact_noaux_tc(
    router_logits: Tensor,
    correction_bias: Tensor,
    *,
    top_k: int = NOAUX_TC_K,
    norm_topk_prob: bool = True,
    routed_scaling_factor: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Return global expert indices and dense affinities for GLM-5.2."""
    if router_logits.ndim != 2:
        raise NoauxTcRouterError(
            f"router_logits must have shape [T, E], got {tuple(router_logits.shape)}"
        )
    num_tokens, num_experts = router_logits.shape
    _require_extents(num_experts, top_k)
    bias = _legalize_bias(correction_bias, num_experts)

    if not can_run_kernel(router_logits):
        return exact_noaux_tc_torch_reference(
            router_logits,
            bias,
            top_k=top_k,
            norm_topk_prob=norm_topk_prob,
            routed_scaling_factor=routed_scaling_factor,
        )

    padded_tokens = _pad_target(num_tokens)
    expert_index, affinities = wrap_nki(_exact_noaux_tc_nki)(
        router_logits=_pad_tokens(router_logits.to(torch.float32), padded_tokens),
        correction_bias=bias,
        norm_topk_prob=norm_topk_prob,
        routed_scaling_factor=float(routed_scaling_factor),
    )
    return expert_index[:num_tokens].to(torch.int32), affinities[:num_tokens]
