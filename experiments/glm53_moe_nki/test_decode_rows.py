"""Compact routed rows must preserve the mapping and FP32 combine.

Run in CPU mode with the Neuron SDK installed. Only the kernel boundary is
stubbed in the model tests. The mapping tests use the production mapping.
"""

from collections import Counter

import pytest
import torch

from benchmarks.glm53_moe.reference import make_fixture
from experiments.glm53_moe_nki.test_model_integration import checkpoint_operands, quant_config
from experiments.glm53_moe_nki.verify_decode_rows import routing_affinities
from vllm_neuron import functional
from vllm_neuron.functional.moe import fused_fp8, moe_blockwise, moe_blockwise_fp8
from vllm_neuron.model.glm5_next import model_fp8
from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig


BLOCK = 256
EXPERTS = 18
TOP_K = 8


def mapping(scores):
    return moe_blockwise.build_blockwise_mapping(
        scores, EXPERTS, TOP_K, BLOCK, moe_group=None, tp_degree=1,
    )


def pairs(rows, experts):
    return Counter((int(expert), int(token))
                   for block, expert in zip(rows, experts.reshape(-1))
                   for token in block if int(token) >= 0)


@pytest.mark.parametrize("tokens", [1, 3, 127, 256, 257, 511, 512])
@pytest.mark.parametrize("mode", ["mixed", "concentrated", "empty"])
def test_mapping_prefix_retains_every_route(tokens, mode):
    scores = routing_affinities(tokens, EXPERTS, TOP_K, mode)[:, :EXPERTS].contiguous()
    masked, flat, experts, conditions = mapping(scores)
    full = flat.reshape(-1, BLOCK)
    width = min(tokens, BLOCK)
    compact = full[:, :width]
    expected = Counter((int(expert), int(token))
                       for token, expert in torch.nonzero(scores, as_tuple=False))
    assert pairs(full, experts) == expected
    assert pairs(compact, experts) == expected
    assert bool((full[:, width:] == -1).all())
    assert torch.equal(masked.reshape_as(scores), scores)
    assert torch.equal(conditions.bool(), (compact >= 0).any(dim=1))
    if tokens >= BLOCK:
        assert torch.equal(compact, full)
    if mode == "empty":
        assert not bool(conditions.any())
    if mode == "concentrated" and tokens > BLOCK:
        # Each selected expert has a second block, including a one-row tail at T257.
        for expert in range(TOP_K):
            active = full[experts == expert]
            lengths = (active >= 0).sum(dim=1).tolist()
            assert lengths == [BLOCK, tokens - BLOCK]


def small_bank():
    case = make_fixture(q=1, hidden=128, intermediate=128, experts=EXPERTS)
    config = Glm5NextTextConfig(
        hidden_size=128, moe_intermediate_size=128,
        n_routed_experts=2 * EXPERTS, num_experts_per_tok=TOP_K,
        swiglu_limit=10.0,
    )
    bank = model_fp8.Glm5NextRoutedExperts(config, world_size=2, ep_degree=2)
    bank.prepare_scale_operands(**checkpoint_operands(case))
    return bank, case


def boundary_contributions(hidden, rows, experts, affinity):
    safe = torch.where(rows < 0, hidden.shape[0] - 1, rows).long()
    selected = experts.long().reshape(-1, 1).expand_as(safe)
    scale = affinity[safe, selected] * (selected + 1)
    return (hidden[safe].float() * scale[..., None]).reshape(-1, hidden.shape[1])


def direct_reference(hidden, scores, rank):
    local = scores[:, rank * EXPERTS:(rank + 1) * EXPERTS]
    out = torch.zeros_like(hidden, dtype=torch.float32)
    for expert in range(EXPERTS):
        out += hidden.float() * ((expert + 1) * local[:, expert, None])
    return out.to(hidden.dtype)


@pytest.mark.parametrize("tokens", [1, 3, 127, 256, 257, 511])
def test_packed_forward_uses_the_same_compact_ids_for_kernel_and_combine(monkeypatch, tokens):
    bank, _ = small_bank()
    prepared = bank._prepared_kernel_operands
    expected_mapping = []
    seen = []
    real_mapping = moe_blockwise.build_blockwise_mapping

    def record_mapping(*args, **kwargs):
        assert kwargs["block_size"] == BLOCK
        built = real_mapping(*args, **kwargs)
        expected_mapping.append((built[1].reshape(-1, BLOCK)[:, :min(tokens, BLOCK)].clone(),
                                 built[2].reshape(-1, 1).clone()))
        return built

    def fused_boundary(hidden, packed, rows, experts, affinity, bounds):
        assert packed.weights is prepared["packed_weights"]
        assert packed.scales is prepared["packed_scales"]
        assert rows.shape[1] == min(tokens, BLOCK)
        assert rows.is_contiguous()
        assert torch.equal(rows, expected_mapping[-1][0])
        assert torch.equal(experts, expected_mapping[-1][1])
        assert rows.dtype == experts.dtype == torch.int32
        seen.append(rows.clone())
        return boundary_contributions(hidden, rows, experts, affinity)

    monkeypatch.setattr(functional, "build_blockwise_mapping", record_mapping)
    monkeypatch.setattr(fused_fp8, "fused_fp8_experts", fused_boundary)
    hidden = ((torch.arange(tokens * 128).reshape(tokens, 128) % 15) - 7).to(torch.bfloat16)
    original = routing_affinities(tokens, EXPERTS, TOP_K, "mixed")
    for scores, rank in ((original, 0), (original.roll(7, 1), 0), (original, 1),
                         (torch.zeros_like(original), 1), (original, 0)):
        got = bank(hidden, scores, quant_config(), block_size=BLOCK,
                   expert_parallel_rank=torch.tensor([rank], dtype=torch.int64))
        torch.testing.assert_close(got, direct_reference(hidden, scores, rank), rtol=0, atol=0)
    assert len(seen) == 5
    assert bool((seen[3] == -1).all())


@pytest.mark.parametrize("tokens", [1, 3, 257])
def test_legacy_forward_keeps_full_mapping_blocks(monkeypatch, tokens):
    bank, case = small_bank()
    held = {}

    def gate(hidden, weights, scales, rows, experts, block):
        assert block == BLOCK
        assert rows.numel() == experts.numel() * BLOCK
        held.update(hidden=hidden, rows=rows.reshape(-1, BLOCK), experts=experts)
        return torch.zeros(rows.numel(), 256)

    def down(activated, weights, scales, affinity, rows, experts, block, count):
        assert block == BLOCK and count == tokens
        assert torch.equal(rows.reshape(-1, BLOCK), held["rows"])
        return boundary_contributions(held["hidden"], held["rows"], experts,
                                      affinity.reshape(tokens + 1, EXPERTS))

    monkeypatch.setattr(moe_blockwise_fp8, "moe_gate_up_blockwise_fp8", gate)
    monkeypatch.setattr(moe_blockwise_fp8, "moe_swiglu_transposed", lambda x, *args: x)
    monkeypatch.setattr(moe_blockwise_fp8, "moe_down_blockwise_fp8", down)
    hidden = torch.ones(tokens, 128, dtype=torch.bfloat16)
    scores = routing_affinities(tokens, EXPERTS, TOP_K, "concentrated")
    got = bank.block_quant_expert_mm(
        hidden, scores, case.gate_weight, case.down_weight,
        torch.zeros(EXPERTS, 128, 2), torch.zeros(EXPERTS, 128, 1), quant_config(),
        block_size=BLOCK, expert_parallel_rank=torch.tensor([0], dtype=torch.int64),
    )
    torch.testing.assert_close(got, direct_reference(hidden, scores, 0), rtol=0, atol=0)
