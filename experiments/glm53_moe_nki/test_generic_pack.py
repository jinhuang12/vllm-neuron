"""Shape-derived packing, scale coordinates, and meta preparation."""
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'vllm_neuron/functional/moe' / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pack = load('generic_pack_test', 'fused_fp8_pack.py')
config = load('generic_config_test', 'fused_fp8_config.py')


@pytest.mark.parametrize('e,h,i', [(1, 128, 128), (3, 384, 640), (2, 1024, 256)])
def test_generic_coordinates_and_roundtrip(e, h, i):
    gate = (torch.arange(e * h * 2 * i) % 120).to(torch.uint8).reshape(e, h, 2 * i)
    down = (torch.arange(e * i * h) % 120 + 128).to(torch.uint8).reshape(e, i, h)
    gs = torch.arange(e * (h // 128) * 2 * (i // 128), dtype=torch.float32).reshape(e, h // 128, 2, i // 128)
    ds = torch.arange(e * (i // 128) * (h // 128), dtype=torch.float32).reshape(e, i // 128, h // 128)
    packed = pack.pack_experts(gate.view(torch.float8_e4m3fn), down.view(torch.float8_e4m3fn), gs, ds)
    assert packed.weights.shape == (e, 3 * (i // 128), 128, h // 128, 128)
    recovered = pack.unpack_experts(packed)
    for actual, expected in zip(recovered, (gate.view(torch.float8_e4m3fn), down.view(torch.float8_e4m3fn), gs, ds)):
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    for expert in range(e):
        for hb in range(h // 128):
            for ib in range(i // 128):
                for half in range(2):
                    panel = half * (i // 128) + ib
                    want = gate[expert, hb * 128:(hb + 1) * 128, panel * 128:(panel + 1) * 128]
                    assert torch.equal(packed.weights.view(torch.uint8)[expert, panel, :, hb], want)
                    assert packed.scales[expert, panel, hb] == gs[expert, hb, half, ib]
                panel = 2 * (i // 128) + ib
                assert torch.equal(packed.weights.view(torch.uint8)[expert, panel, :, hb],
                                   down[expert, ib * 128:(ib + 1) * 128, hb * 128:(hb + 1) * 128])
                assert packed.scales[expert, panel, hb] == ds[expert, ib, hb]


def test_meta_preparation_has_no_value_read():
    prepared = pack.pack_experts(torch.empty(2, 384, 1280, device='meta', dtype=torch.float8_e4m3fn),
                                torch.empty(2, 640, 384, device='meta', dtype=torch.float8_e4m3fn),
                                torch.empty(2, 3, 2, 5, device='meta'),
                                torch.empty(2, 5, 3, device='meta'))
    assert prepared.weights.shape == (2, 15, 128, 3, 128)
    assert prepared.scales.shape == (2, 15, 3)
    assert prepared.weights.device.type == prepared.scales.device.type == 'meta'


@pytest.mark.parametrize('kwargs', [dict(block_m=513), dict(block_n=64), dict(block_k=64)])
def test_invalid_tile_configuration_is_refused(kwargs):
    with pytest.raises(ValueError):
        config.select_tiles(384, 640, 7, **kwargs)


def test_explicit_nondividing_tiles_remain_compile_time_parameters():
    assert config.select_tiles(384, 640, 769, 37, 256, 256) == (37, 256, 256)
