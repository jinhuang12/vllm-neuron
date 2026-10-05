# SPDX-License-Identifier: Apache-2.0
"""Exact-shape dispatch and compact decode output/zero ownership."""

import pytest
import torch
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.moe import fused_fp8 as api
from vllm_neuron.functional.moe.fused_fp8_pack import PackedExperts, pack_experts
from vllm_neuron.functional.moe.moe_fused_fp8 import moe_fused_fp8_kernel
from vllm_neuron.functional.moe.moe_fused_fp8_decode import compact_decode_kernel


def _meta_case(hidden=4096, intermediate=512, tokens=1, blocks=8, rows=1, experts=18):
    weights = torch.empty(
        (experts, 3 * (intermediate // 128), 128, hidden // 128, 128),
        dtype=torch.float8_e4m3fn, device="meta",
    )
    scales = torch.empty(
        (experts, 3 * (intermediate // 128), hidden // 128),
        dtype=torch.float32, device="meta",
    )
    return (
        torch.empty((tokens + 1, hidden), dtype=torch.bfloat16, device="meta"),
        PackedExperts(weights, scales),
        torch.empty((blocks, rows), dtype=torch.int32, device="meta"),
        torch.empty((blocks, 1), dtype=torch.int32, device="meta"),
        torch.empty((tokens + 1, experts), dtype=torch.float32, device="meta"),
        torch.empty((128, 3), dtype=torch.float32, device="meta"),
    )


@pytest.mark.parametrize(
    "geometry,overrides,lnc,expected",
    [
        ({}, {}, "2", "compact"),
        ({}, {}, None, "original"),
        ({}, {}, "1", "original"),
        ({}, {}, "invalid", "original"),
        ({"hidden": 2048}, {}, "2", "original"),
        ({"intermediate": 256}, {}, "2", "original"),
        ({"tokens": 2}, {}, "2", "original"),
        ({"blocks": 7}, {}, "2", "original"),
        ({"rows": 2}, {}, "2", "original"),
        ({"experts": 17}, {}, "2", "original"),
        ({}, {"block_m": 2}, "2", "original"),
        ({}, {"block_n": 1024}, "2", "original"),
        ({}, {"block_k": 2048}, "2", "original"),
        ({}, {"block_m": 1, "block_n": 4096, "block_k": 4096}, "2", "compact"),
    ],
)
def test_public_dispatch_selects_only_measured_geometry(
    geometry, overrides, lnc, expected, monkeypatch
):
    if lnc is None:
        monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    else:
        monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", lnc)
    calls = []

    def call(label):
        def run(hidden, weights, scales, rows, experts, affinity, bounds, **tiles):
            calls.append((label, tiles))
            return torch.empty(
                (*rows.shape, hidden.shape[1]), dtype=torch.float32,
                device=hidden.device,
            )
        return run

    monkeypatch.setattr(api, "_FUSED_EXPERTS", call("original"))
    monkeypatch.setattr(api, "_FUSED_DECODE_EXPERTS", call("compact"))
    api.reset_fused_dispatch_counters()
    args = _meta_case(**geometry)
    output = api.fused_fp8_experts(*args, **overrides)
    assert calls[0][0] == expected and len(calls) == 1
    assert ("BLOCK_M" in calls[0][1]) == (expected == "original")
    assert output.shape == (args[2].numel(), args[0].shape[1])
    assert output.dtype == torch.float32
    assert api.fused_dispatch_counters() == (1, 0)


def test_public_validation_precedes_compact_dispatch(monkeypatch):
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    calls = []
    monkeypatch.setattr(api, "_FUSED_DECODE_EXPERTS", lambda *a, **k: calls.append(a))
    args = list(_meta_case())
    args[2] = args[2].to(torch.int64)
    with pytest.raises(ValueError, match="row_ids must use int32"):
        api.fused_fp8_experts(*args)
    assert calls == []


def _small_case(active, duplicate=False):
    generator = torch.Generator().manual_seed(20261003)
    experts, hidden, intermediate = 8, 128, 128
    x = torch.randn((2, hidden), generator=generator).to(torch.bfloat16)
    x[-1].zero_()
    gate_up = (torch.randn((experts, hidden, 2 * intermediate), generator=generator) / 16).to(torch.float8_e4m3fn)
    down = (torch.randn((experts, intermediate, hidden), generator=generator) / 16).to(torch.float8_e4m3fn)
    gu_scale = torch.rand((experts, 1, 2, 1), generator=generator) + 0.5
    down_scale = torch.rand((experts, 1, 1), generator=generator) + 0.5
    packed = pack_experts(gate_up, down, gu_scale, down_scale)
    rows = torch.full((8, 1), -1, dtype=torch.int32)
    rows[active] = 0
    expert_ids = torch.arange(experts, dtype=torch.int32).reshape(-1, 1)
    if duplicate:
        expert_ids[active] = experts - 1
    affinity = torch.zeros((2, experts), dtype=torch.float32)
    if active:
        affinity[0, expert_ids[active, 0].long()] = 1.0 / len(active)
    bounds = torch.tensor([0.6, -0.4, 0.5], dtype=torch.float32).repeat(128, 1)
    return (x, packed.weights, packed.scales, rows, expert_ids,
            affinity.reshape(-1, 1), bounds)


@pytest.mark.parametrize("programs", [1, 2])
@pytest.mark.parametrize(
    "active,duplicate",
    [([], False), ([0], False), ([7], False), ([0, 4, 7], False),
     ([0, 1, 2, 3, 4], False), (list(range(8)), False), ([0, 2, 7], True)],
)
def test_compact_math_keeps_original_output_slots(active, duplicate, programs, monkeypatch):
    monkeypatch.setenv("NKI_SIMULATOR", "1")
    monkeypatch.setenv("NEURON_LIBTORCH_CPU_MODE", "1")
    args = _small_case(active, duplicate)
    original = wrap_nki(moe_fused_fp8_kernel)[programs](
        *args, BLOCK_M=1, BLOCK_N=256, BLOCK_K=256, SKIP_PADDING=False,
    )
    actual = wrap_nki(compact_decode_kernel)[programs](
        *args, BLOCK_N=256, BLOCK_K=256,
    )
    assert torch.equal(actual.contiguous().view(torch.uint32),
                       original.contiguous().view(torch.uint32))
    assert torch.count_nonzero(actual[args[3] < 0]) == 0


def test_empty_decode_skips_invalid_expert_operands(monkeypatch):
    monkeypatch.setenv("NKI_SIMULATOR", "1")
    monkeypatch.setenv("NEURON_LIBTORCH_CPU_MODE", "1")
    args = _small_case([])
    args[4].fill_(99999)
    actual = wrap_nki(compact_decode_kernel)[2](
        *args, BLOCK_N=256, BLOCK_K=256,
    )
    assert torch.equal(actual.view(torch.uint32),
                       torch.zeros_like(actual).view(torch.uint32))
