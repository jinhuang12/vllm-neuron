"""Check the model's ordered-combine gate with the real CPU mapping.

The expert device boundary is stubbed. Native equality and dispatch have
separate component and routed-expert verifiers.
"""

import pytest
import torch

from benchmarks.glm53_moe.reference import make_fixture
from experiments.glm53_moe_nki.test_model_integration import (
    bank_for,
    checkpoint_operands,
    quant_config,
)
from vllm_neuron.functional.moe import fused_fp8, ordered_combine


@pytest.mark.parametrize("tokens,capturing,expected_calls", [
    (4, True, 1), (4, False, 0), (1, True, 0), (1, False, 0),
])
def test_only_captured_packed_prefill_uses_ordered_combine(
    tokens, capturing, expected_calls, monkeypatch,
):
    case = make_fixture(q=tokens, hidden=256, intermediate=512, experts=2)
    bank = bank_for(case, ep_degree=2)
    bank.prepare_scale_operands(**checkpoint_operands(case))
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: capturing)
    emitted, combined = [], []

    def expert_boundary(hidden, packed, row_ids, expert_ids, affinity, bounds):
        assert packed.weights is bank._prepared_kernel_operands["packed_weights"]
        safe = row_ids.clamp_min(0).long()
        expert = expert_ids.long().expand_as(safe)
        values = hidden[safe].float() * (expert + 1)[..., None]
        values *= affinity[safe, expert][..., None]
        # Discarded expert rows must not affect any real token.
        values[row_ids < 0] = float("nan")
        flat = values.reshape(-1, hidden.shape[1])
        emitted.append((flat, row_ids))
        return flat

    real_combine = ordered_combine.ordered_combine

    def record_combine(contribution, row_ids, total_tokens, **kwargs):
        assert contribution is emitted[-1][0]
        assert row_ids is emitted[-1][1]
        assert contribution.dtype == torch.float32
        assert total_tokens == tokens
        result = real_combine(contribution, row_ids, total_tokens, **kwargs)
        assert result.dtype == torch.float32
        combined.append(result)
        return result

    monkeypatch.setattr(fused_fp8, "fused_fp8_experts", expert_boundary)
    monkeypatch.setattr(ordered_combine, "ordered_combine", record_combine)
    pattern = torch.tensor([[0.2, 0.3, 0.7, 0.1], [0.0, 0.9, 0.4, 0.0],
                            [0.6, 0.0, 0.0, 0.8], [0.1, 0.2, 0.3, 0.4]])
    affinity = pattern[:tokens]
    hidden = case.hidden[:-1]
    actual = bank(hidden, affinity, quant_config(),
                  expert_parallel_rank=torch.tensor([1]))
    local = affinity[:, 2:4]
    expected = (hidden.float() * local[:, :1]
                + hidden.float() * 2 * local[:, 1:2]).to(hidden.dtype)
    assert len(emitted) == 1
    assert len(combined) == expected_calls
    assert actual.shape == hidden.shape and actual.dtype == hidden.dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
