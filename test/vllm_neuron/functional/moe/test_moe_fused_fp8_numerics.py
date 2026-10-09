# SPDX-License-Identifier: Apache-2.0
"""Numerics and Tensor Engine operand forms of the fused FP8 expert kernel.

The kernel output is compared bit for bit with the f3a833f kernel
(test/hardware/baselines/moe_5938748, byte-identical to f3a833f) and with a
blockwise FP8 reference built from the unpacked weights, at tilings that split
the gate/up and down loads and use every row class. The simulator accepts PE
operand access patterns that the hardware does not: a stationary with two free
dimensions passes here and gives wrong products on Trn2 (records of the
5debd95 device run). A separate test therefore checks that every stationary
the kernel passes to ``nc_matmul`` has one free dimension.
"""

import importlib.util
from pathlib import Path

import pytest
import torch
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.moe import moe_fused_fp8 as _module
from vllm_neuron.functional.moe.fused_fp8_pack import pack_experts
from vllm_neuron.functional.moe.moe_fused_fp8 import moe_fused_fp8_kernel

_BASELINE = (Path(__file__).resolve().parents[3]
             / "hardware/baselines/moe_5938748/moe_fused_fp8.py")


def _baseline_kernel():
    spec = importlib.util.spec_from_file_location("moe_fused_fp8_f3a833f", _BASELINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.moe_fused_fp8_kernel


def _stationary_shapes(monkeypatch):
    """Record the shape of every ``nc_matmul`` stationary the kernel issues."""
    shapes = []
    real = _module.nisa.nc_matmul

    def recording(*args, **kwargs):
        shapes.append(tuple(kwargs["stationary"].shape))
        return real(*args, **kwargs)

    monkeypatch.setattr(_module.nisa, "nc_matmul", recording)
    return shapes


@pytest.mark.parametrize("skip_padding", [True, False])
def test_every_stationary_has_one_free_dimension(skip_padding, monkeypatch):
    monkeypatch.setenv("NKI_SIMULATOR", "1")
    monkeypatch.setenv("NEURON_LIBTORCH_CPU_MODE", "1")
    shapes = _stationary_shapes(monkeypatch)
    args, _ = _case(blocks=4, q=64, hidden=512, intermediate=512)
    wrap_nki(moe_fused_fp8_kernel)[2](*args, BLOCK_M=64, BLOCK_N=256, BLOCK_K=256,
                                      SKIP_PADDING=skip_padding)
    # Per routed slice: 2I/128 * H/128 gate/up and I/128 * H/128 down products.
    assert len(shapes) >= 3 * (512 // 128) * (512 // 128)
    assert {len(shape) for shape in shapes} == {2}
    assert all(shape[0] == 128 and shape[1] <= 128 for shape in shapes)


def _case(blocks, q, hidden, intermediate, experts=3, tokens=40, seed=20261008):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(tokens + 1, hidden, generator=generator).to(torch.bfloat16)
    x[-1].zero_()
    gate_up = (torch.randn(experts, hidden, 2 * intermediate, generator=generator) / 16
               ).to(torch.float8_e4m3fn)
    down = (torch.randn(experts, intermediate, hidden, generator=generator) / 16
            ).to(torch.float8_e4m3fn)
    nh, ni = hidden // 128, intermediate // 128
    gu_scale = torch.rand(experts, nh, 2, ni, generator=generator) + 0.5
    down_scale = torch.rand(experts, ni, nh, generator=generator) + 0.5
    packed = pack_experts(gate_up, down, gu_scale, down_scale)
    affinity = torch.rand(tokens + 1, experts, generator=generator)
    affinity[-1].zero_()
    bounds = torch.tensor([0.6, -0.4, 0.5]).repeat(128, 1)
    # Block b holds a b-dependent number of rows with holes; block 1 is empty.
    ids = torch.full((blocks, q), -1, dtype=torch.int32)
    for block in range(blocks):
        count = 0 if block == 1 else (7 * block + 5) % q + 1
        rows = torch.randperm(q, generator=generator)[:count]
        ids[block, rows] = torch.randint(0, tokens, (count,), generator=generator,
                                         dtype=torch.int32)
    expert_ids = torch.tensor([[b % experts] for b in range(blocks)], dtype=torch.int32)
    args = (x, packed.weights, packed.scales, ids, expert_ids, affinity.reshape(-1, 1), bounds)
    return args, (gate_up.float(), down.float(), gu_scale, down_scale)


def _reference(args, weights):
    """Blockwise FP8 experts: each 128x128 product times its scale, then SwiGLU."""
    x, _, row_ids, expert_ids, affinity, bounds = args[0], args[1], args[3], args[4], args[5], args[6]
    gate_up, down, gu_scale, down_scale = weights
    experts = gate_up.shape[0]
    hidden, intermediate = x.shape[1], down.shape[1]
    affinity = affinity.reshape(-1, experts)
    output = torch.zeros(*row_ids.shape, hidden)
    for block in range(row_ids.shape[0]):
        expert = int(expert_ids[block, 0])
        valid = row_ids[block] >= 0
        rows = row_ids[block, valid].long()
        if rows.numel() == 0:
            continue
        selected = x[rows].float()
        projection = torch.zeros(rows.numel(), 2 * intermediate)
        for k in range(hidden // 128):
            product = selected[:, k * 128:(k + 1) * 128] @ gate_up[expert, k * 128:(k + 1) * 128]
            projection += product * gu_scale[expert, k].reshape(-1).repeat_interleave(128)
        gate, up = projection.split(intermediate, dim=1)
        gate = gate.clamp(max=float(bounds[0, 0]))
        up = up.clamp(min=float(bounds[0, 1]), max=float(bounds[0, 2]))
        activated = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16).float()
        result = torch.zeros(rows.numel(), hidden)
        for k in range(intermediate // 128):
            product = activated[:, k * 128:(k + 1) * 128] @ down[expert, k * 128:(k + 1) * 128]
            result += product * down_scale[expert, k].repeat_interleave(128)
        output[block, valid] = result * affinity[rows, expert, None]
    return output


@pytest.mark.parametrize("programs", [1, 2])
@pytest.mark.parametrize("q,block_m,hidden,intermediate,block_n,block_k", [
    (64, 64, 512, 512, 512, 512),     # one weight load per projection
    (64, 64, 512, 512, 256, 256),     # two loads per projection
    (64, 64, 768, 256, 256, 256),     # three loads per projection, I / 128 = 2
    (256, 256, 512, 512, 512, 512),   # every row class, up to 256-row products
])
def test_output_matches_f3a833f_and_the_blockwise_reference(
        q, block_m, hidden, intermediate, block_n, block_k, programs, monkeypatch):
    monkeypatch.setenv("NKI_SIMULATOR", "1")
    monkeypatch.setenv("NEURON_LIBTORCH_CPU_MODE", "1")
    args, weights = _case(blocks=4, q=q, hidden=hidden, intermediate=intermediate)
    tiles = dict(BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k)
    actual = wrap_nki(moe_fused_fp8_kernel)[programs](*args, **tiles, SKIP_PADDING=True)
    dense = wrap_nki(moe_fused_fp8_kernel)[programs](*args, **tiles, SKIP_PADDING=False)
    before = wrap_nki(_baseline_kernel())[programs](*args, **tiles, SKIP_PADDING=True)
    row_ids = args[3]
    # Every element keeps f3a833f's arithmetic sequence, so the bits match.
    assert torch.equal(actual.view(torch.int32), before.view(torch.int32))
    assert torch.equal(dense.view(torch.int32), before.view(torch.int32))
    torch.testing.assert_close(actual, _reference(args, weights), rtol=3e-2, atol=1e-5)
    assert torch.count_nonzero(actual[row_ids < 0]) == 0
