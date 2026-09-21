"""Lossless packing and independent coordinate checks."""
import importlib.util
from pathlib import Path
import sys

import torch
import pytest

_path = Path(__file__).resolve().parents[2] / "vllm_neuron/functional/moe/fused_fp8_pack.py"
_spec = importlib.util.spec_from_file_location("fused_pack", _path)
pack = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = pack
_spec.loader.exec_module(pack)


def test_weight_bits_and_scales_round_trip():
    # Every legal encoding, including both signed zeros and all subnormals.
    codes = torch.tensor(list(range(120)) + list(range(128, 248)), dtype=torch.uint8)
    gate = codes.repeat((2 * 4096 * 1024 + 239) // 240)[:2 * 4096 * 1024].reshape(2, 4096, 1024)
    down = codes.repeat((2 * 512 * 4096 + 239) // 240)[:2 * 512 * 4096].reshape(2, 512, 4096)
    gs = torch.arange(2 * 32 * 2 * 4, dtype=torch.float32).reshape(2, 32, 2, 4)
    ds = torch.arange(2 * 4 * 32, dtype=torch.float32).reshape(2, 4, 32)
    packed = pack.pack_experts(gate.view(torch.float8_e4m3fn), down.view(torch.float8_e4m3fn), gs, ds)
    unpacked = pack.unpack_experts(packed)
    assert torch.equal(unpacked[0].view(torch.uint8), gate)
    assert torch.equal(unpacked[1].view(torch.uint8), down)
    assert torch.equal(unpacked[2], gs)
    assert torch.equal(unpacked[3], ds)
    # Check coordinates independently of unpack_experts.
    for expert in (0, 1):
        for k, col in ((0, 0), (127, 129), (129, 511), (4095, 1023)):
            assert packed.weights.view(torch.uint8)[expert, col // 128, k % 128, k // 128, col % 128] == gate[expert, k, col]
            assert packed.scales[expert, col // 128, k // 128] == gs.reshape(2, 32, 8)[expert, k // 128, col // 128]
        for k, col in ((0, 0), (127, 129), (129, 511), (511, 4095)):
            assert packed.weights.view(torch.uint8)[expert, 8 + k // 128, k % 128, col // 128, col % 128] == down[expert, k, col]
            assert packed.scales[expert, 8 + k // 128, col // 128] == ds[expert, k // 128, col // 128]


@pytest.mark.parametrize("raw", list(range(120, 128)) + list(range(248, 256)))
@pytest.mark.parametrize("projection", ["gate", "down"])
def test_rejects_unprepared_range_and_nan(raw, projection):
    gate = torch.zeros((1, 4096, 1024), dtype=torch.uint8)
    down = torch.zeros((1, 512, 4096), dtype=torch.uint8)
    (gate if projection == "gate" else down)[0, 0, 0] = raw
    with pytest.raises(ValueError, match="model loader prepared weights"):
        pack.pack_experts(gate.view(torch.float8_e4m3fn), down.view(torch.float8_e4m3fn),
                          torch.ones(1, 32, 2, 4), torch.ones(1, 4, 32))
