# SPDX-License-Identifier: Apache-2.0
"""The 5938748 MoE decode path, router to combined routed output, as functions.

Copied from ``vllm_neuron/model/glm5_next/model_fp8.py`` at 5938748 and kept to
the operations that run on device: ``Glm5NextRoutedExperts.route_tokens``
(lines 1264-1314), the packed branch of ``block_quant_expert_mm``
(lines 1543-1684) and ``_token_gather_combine`` (lines 1113-1162). The refusal
checks of those methods are dropped; every tensor operation is kept in its
original order. Kernels and helpers come from the sibling snapshot modules, so
later edits to ``vllm_neuron/functional/moe`` cannot change this baseline.
"""

from __future__ import annotations

import torch

from vllm_neuron.functional import get_local_expert_affinities
from vllm_neuron.functional.moe.moe_blockwise_fp8 import _swiglu_bound_operand

from .fused_fp8 import PackedExperts, fused_fp8_experts
from .moe_blockwise import _cumsum_matmul, build_blockwise_mapping
from .router import noaux_tc_rmsnorm_router_topk
from .token_gather_combine import token_gather_combine

#: ``BLOCK_QUANT_SIZE`` at 5938748 (``blockwise_fp8_retile.py``).
BLOCK = 128


def route_tokens(hidden_states, gamma, router_weights, correction_bias, *,
                 top_k=8, eps=1e-5, norm_topk_prob=True, routed_scaling_factor=2.5):
    """``route_tokens`` at 5938748: ``[B, S, H]`` in, three router outputs out."""
    logits, expert_index, expert_affinities, _substrate_index = (
        noaux_tc_rmsnorm_router_topk(
            hidden_states=hidden_states,
            gamma=gamma,
            router_weights=router_weights,
            correction_bias=correction_bias,
            top_k=top_k,
            eps=eps,
            norm_topk_prob=norm_topk_prob,
            routed_scaling_factor=routed_scaling_factor,
        )
    )
    return logits, expert_index, expert_affinities


def _token_gather_combine(contribution, expert_affinities, block, rows, top_k):
    """``_token_gather_combine`` at 5938748, verbatim."""
    tokens, experts = expert_affinities.shape
    device = expert_affinities.device
    mask = (expert_affinities != 0).to(torch.float32)  # [T, E]
    place = _cumsum_matmul(mask) - 1.0
    blocks = torch.ceil(mask.sum(dim=0) / block)
    first_block = _cumsum_matmul(blocks.unsqueeze(1)).squeeze(1) - blocks
    row = first_block.unsqueeze(0) * rows + place  # [T, E], exact in fp32
    upper = torch.triu(
        torch.ones(experts, experts, dtype=torch.float32, device=device)
    )
    rank = torch.matmul(mask, upper) - 1.0  # [T, E]
    slots = min(top_k, experts)
    slot_ids = torch.arange(slots, dtype=torch.float32, device=device)
    pick = mask.unsqueeze(2) * (rank.unsqueeze(2) == slot_ids).to(torch.float32)
    valid = pick.sum(dim=1)  # [T, k], 0/1
    index = (pick * row.unsqueeze(2)).sum(dim=1).to(torch.int32)  # [T, k]
    return token_gather_combine(contribution, index, valid)


def routed_experts(hidden_states, expert_affinities, packed_weights, packed_scales,
                   expert_parallel_rank, *, top_k=8, swiglu_limit=10.0,
                   tp_degree=1, moe_group=None, out_dtype=None):
    """The packed branch of ``block_quant_expert_mm`` at 5938748.

    ``hidden_states`` is ``[T, H]``, ``expert_affinities`` the global ``[T, E]``
    router output; returns ``[T, H]`` in ``hidden_states.dtype``, as the model
    did. ``out_dtype=torch.float32`` returns the fp32 combine before that cast
    (a test-side tap; the model never asked for it).
    """
    tokens, hidden = (int(extent) for extent in hidden_states.shape)
    num_experts = int(packed_weights.shape[0])
    routed = int(expert_affinities.shape[1])
    if routed != num_experts:
        local_indices = (
            torch.arange(0, num_experts, dtype=torch.int64,
                         device=expert_affinities.device)
            + expert_parallel_rank * num_experts
        )
        expert_affinities = get_local_expert_affinities(
            expert_affinities, local_indices
        )
    block = BLOCK
    (
        expert_affinities_masked,
        token_position_to_id,
        block_to_expert,
        _conditions,
    ) = build_blockwise_mapping(
        expert_affinities=expert_affinities,
        num_local_experts=num_experts,
        num_experts_per_token=int(top_k),
        block_size=block,
        moe_group=moe_group,
        tp_degree=tp_degree,
    )
    pad_hidden = torch.zeros(
        1, hidden, dtype=hidden_states.dtype, device=hidden_states.device
    )
    padded_hidden = torch.cat([hidden_states, pad_hidden], dim=0)
    pad_masked = torch.zeros(
        num_experts, 1, dtype=expert_affinities_masked.dtype,
        device=expert_affinities_masked.device,
    )
    expert_affinities_masked = torch.cat(
        [expert_affinities_masked, pad_masked], dim=0
    )
    rows = min(tokens, block)
    kernel_row_ids = token_position_to_id.reshape(-1, block)[:, :rows].contiguous()
    contribution = fused_fp8_experts(
        padded_hidden.to(torch.bfloat16),
        PackedExperts(packed_weights, packed_scales),
        kernel_row_ids,
        block_to_expert.reshape(-1, 1),
        expert_affinities_masked.reshape(tokens + 1, num_experts),
        _swiglu_bound_operand(swiglu_limit, swiglu_limit, hidden_states.device),
    )
    combined = _token_gather_combine(
        contribution, expert_affinities, block, rows, int(top_k),
    )
    return combined.to(hidden_states.dtype if out_dtype is None else out_dtype)
