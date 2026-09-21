"""Load and routing contracts for the fused routed-expert model path.

Run with VLLM_NEURON_CPU_MODE=1. These checks stub only the device kernel;
the native verifier exercises the real kernel through the model forward.
"""

import pytest
import torch

from benchmarks.glm53_moe.reference import make_fixture
from vllm_neuron.functional.moe import fused_fp8
from vllm_neuron.functional.moe import fused_fp8_pack
from vllm_neuron.model.glm5_next import model_fp8
from vllm_neuron.model.glm5_next.config import Glm5NextConfig, Glm5NextTextConfig


def quant_config():
    return model_fp8.Glm5NextQuantConfig.from_model_config(Glm5NextConfig())


def bank_for(case, ep_degree=1):
    experts = case.gate_weight.shape[0]
    config = Glm5NextTextConfig(
        hidden_size=case.hidden.shape[1],
        moe_intermediate_size=case.down_weight.shape[1],
        n_routed_experts=experts * ep_degree,
        num_experts_per_tok=experts,
        swiglu_limit=10.0,
    )
    return model_fp8.Glm5NextRoutedExperts(
        config, world_size=ep_degree, ep_degree=ep_degree
    )


def checkpoint_operands(case, device="cpu"):
    return {
        "gate_proj_weight": case.gate_weight[:, :, 0].transpose(1, 2).contiguous().to(device),
        "up_proj_weight": case.gate_weight[:, :, 1].transpose(1, 2).contiguous().to(device),
        "down_proj_weight": case.down_weight.transpose(1, 2).contiguous().to(device),
        "gate_proj_scale": case.gate_scales[:, :, 0].transpose(1, 2).contiguous().to(device),
        "up_proj_scale": case.gate_scales[:, :, 1].transpose(1, 2).contiguous().to(device),
        "down_proj_scale": case.down_scales.transpose(1, 2).contiguous().to(device),
    }


def assert_prepared_tiles(case, prepared):
    """Read logical tiles directly, without the packer's inverse."""
    hidden = case.hidden.shape[1]
    intermediate = case.down_weight.shape[1]
    nh, ni = hidden // 128, intermediate // 128
    weights, scales = prepared["packed_weights"], prepared["packed_scales"]
    for expert in range(case.gate_weight.shape[0]):
        for panel in range(3 * ni):
            for hb in range(nh):
                h = slice(hb * 128, (hb + 1) * 128)
                if panel < 2 * ni:
                    half, ib = divmod(panel, ni)
                    i = slice(ib * 128, (ib + 1) * 128)
                    expected = case.gate_weight[expert, h, half, i]
                    scale = case.gate_scales[expert, hb, half, ib]
                else:
                    ib = panel - 2 * ni
                    i = slice(ib * 128, (ib + 1) * 128)
                    expected = case.down_weight[expert, i, h]
                    scale = case.down_scales[expert, ib, hb]
                assert torch.equal(weights[expert, panel, :, hb].view(torch.uint8),
                                   expected.view(torch.uint8))
                assert scales[expert, panel, hb] == scale


@pytest.mark.parametrize("hidden,intermediate", [(128, 128), (384, 640), (4096, 512)])
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_preparation_keeps_only_two_exact_banks(hidden, intermediate, device):
    case = make_fixture(q=2, hidden=hidden, intermediate=intermediate, experts=2)
    bank = bank_for(case)
    built = bank.prepare_scale_operands(**checkpoint_operands(case, device))
    assert built == 2
    prepared = bank._prepared_kernel_operands
    assert set(prepared) == {"packed_weights", "packed_scales"}
    assert prepared["packed_weights"].shape == (2, 3 * (intermediate // 128), 128, hidden // 128, 128)
    assert prepared["packed_scales"].shape == (2, 3 * (intermediate // 128), hidden // 128)
    assert all(t.device.type == device and t.is_contiguous() for t in prepared.values())
    assert bank._retile_health == {name: (0, 0, 0) for name in ("gate", "up", "down")}
    if device == "cpu":
        assert_prepared_tiles(case, prepared)


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_production_load_hook_releases_the_replaced_banks(device):
    case = make_fixture(q=2, hidden=256, intermediate=512, experts=2)
    bank = bank_for(case)
    for name, tensor in checkpoint_operands(case, device).items():
        if name.endswith("_weight"):
            setattr(bank, name, torch.nn.Parameter(tensor, requires_grad=False))
        else:
            setattr(bank, name.replace("_scale", "_weight_scale_inv"), tensor)
    root = model_fp8.Glm5NextForConditionalGeneration.__new__(
        model_fp8.Glm5NextForConditionalGeneration
    )
    torch.nn.Module.__init__(root)
    root.bank = bank
    assert root._run_load_time_preps(torch.device(device)) == (0, 1)
    assert all(getattr(bank, name) is None for name in bank.RELEASED_AFTER_PREP)
    assert set(bank._prepared_kernel_operands) == {"packed_weights", "packed_scales"}
    assert len(root.released_parameters()) == 6
    packed_bytes = sum(t.numel() * t.element_size() for t in bank._prepared_kernel_operands.values())
    original_bytes = sum(t.numel() * t.element_size() for t in checkpoint_operands(case).values())
    assert packed_bytes == original_bytes


def test_forward_packs_once_and_preserves_device_rank_routing(monkeypatch):
    case = make_fixture(q=4, hidden=256, intermediate=512, experts=2)
    bank = bank_for(case, ep_degree=2)
    pack_calls = []
    real_pack = fused_fp8_pack.pack_experts

    def record_pack(*args):
        pack_calls.append(1)
        return real_pack(*args)

    monkeypatch.setattr(fused_fp8_pack, "pack_experts", record_pack)
    bank.prepare_scale_operands(**checkpoint_operands(case))
    prepared = bank._prepared_kernel_operands
    calls = []

    def kernel_boundary(hidden, packed, row_ids, expert_ids, affinity, bounds):
        assert packed.weights is prepared["packed_weights"]
        assert packed.scales is prepared["packed_scales"]
        assert row_ids.dtype == expert_ids.dtype == torch.int32
        assert torch.equal(bounds, torch.tensor([10.0, -10.0, 10.0]).expand(128, 3))
        safe = torch.where(row_ids < 0, hidden.shape[0] - 1, row_ids).long()
        experts = expert_ids.long().expand_as(safe)
        # Distinct expert coefficients make column/rank mistakes observable.
        scale = affinity[safe, experts] * (experts + 1)
        calls.append((row_ids.clone(), expert_ids.clone()))
        return (hidden[safe].float() * scale[..., None]).reshape(-1, hidden.shape[1])

    monkeypatch.setattr(fused_fp8, "fused_fp8_experts", kernel_boundary)
    quant = quant_config()
    affinity = torch.tensor([[0.2, 0.3, 0.7, 0.1], [0.0, 0.9, 0.4, 0.0],
                             [0.6, 0.0, 0.0, 0.8], [0.1, 0.2, 0.3, 0.4]])
    hidden = case.hidden[:-1]
    for rank in (0, 1, 0):
        got = bank(hidden, affinity, quant, expert_parallel_rank=torch.tensor([rank]))
        local = affinity[:, rank * 2:(rank + 1) * 2]
        expected = (hidden.float() * local[:, 0, None]
                    + hidden.float() * (2 * local[:, 1, None]))
        torch.testing.assert_close(got, expected.to(hidden.dtype), rtol=0, atol=0)
    assert len(pack_calls) == 1
    assert len(calls) == 3
    assert any(bool((rows == -1).any()) for rows, _ in calls)


def test_missing_packed_bank_fails_before_mapping():
    case = make_fixture(q=2, hidden=256, intermediate=512, experts=2)
    bank = bank_for(case)
    bank.prepare_scale_operands(**checkpoint_operands(case))
    quant = quant_config()
    with pytest.raises(model_fp8.Glm5NextBlockQuantRouteError, match="both packed"):
        bank.block_quant_expert_mm(case.hidden[:-1], case.affinity[:-1], None, None, None, None,
                                   quant, packed_weights=bank._prepared_kernel_operands["packed_weights"])
