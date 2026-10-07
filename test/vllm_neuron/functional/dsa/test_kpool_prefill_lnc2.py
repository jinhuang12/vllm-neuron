# SPDX-License-Identifier: Apache-2.0
"""The prefill key pooling ``dsa_kpool_hadamard`` against 342e93e's kernel, bit for bit.

342e93e pools 128 pools per tile, one pool a partition, on one core: per tile it issues
``pool_size`` strided loads of each operand, the per-slot softmax, and the 254-instruction
one-block-at-a-time butterfly. This tree splits the pools over both programs of an LNC2
launch and puts several pools on each partition, so each softmax step and each butterfly
stage is one instruction over every pool of a tile. The per-element arithmetic is the
same fp32 sequence on the same operands in the same order (``score + ape``, the running
maximum, ``exp(total - max)``, the denominator and the weighted sum accumulated from zero
in slot order, ``reciprocal`` then multiply, the butterfly, one scale, one cast), so the
contract is equality, not a tolerance: any differing bit is a defect.

The reference is the 342e93e snapshot (``test/hardware/baselines/prefill_cores_342e93e``),
run in the same simulator. Pool counts: the prefill token counts 1, 64 and 1024
(``pool_window`` makes every position a candidate, so ``n_pools`` is the token count)
and ragged counts that leave a short tile on one or both programs.
"""

from __future__ import annotations

import pytest
import torch

from vllm_neuron.functional.dsa import kpool_hadamard as KH
from vllm_neuron.utils.neuron_utils import can_run_kernel

from test.hardware.baselines.prefill_cores_342e93e import load as load_342e93e

TOKEN_COUNTS = (1, 64, 1024)
RAGGED = (37, 130, 515, 1027)


@pytest.fixture(autouse=True)
def _kernels_on():
    assert can_run_kernel(), "run with VLLM_NEURON_CPU_MODE=1 and NKI_SIMULATOR=1"
    KH.reset_kpool_hadamard_dispatch_counters()


def _inputs(n_pools: int, pool_size: int, score_dtype, seed: int):
    gen = torch.Generator().manual_seed(seed)
    shape = (n_pools, pool_size, KH.INDEX_HEAD_DIM)
    slot_k = (torch.randn(shape, generator=gen) * 2.0).to(torch.bfloat16)
    slot_score = (torch.randn(shape, generator=gen) * 3.0).to(score_dtype)
    ape = torch.randn((pool_size, KH.INDEX_HEAD_DIM), generator=gen) * 0.5
    return slot_k, slot_score, ape


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.view(torch.int16) if x.dtype == torch.bfloat16 else x.view(torch.int32)


def _assert_equals_342e93e(n_pools: int, pool_size: int, score_dtype, seed: int) -> None:
    base = load_342e93e().kpool_hadamard
    slot_k, slot_score, ape = _inputs(n_pools, pool_size, score_dtype, seed)
    want = base.dsa_kpool_hadamard(slot_k, slot_score, ape)
    KH.reset_kpool_hadamard_dispatch_counters()
    got = KH.dsa_kpool_hadamard(slot_k, slot_score, ape)
    assert KH.kpool_hadamard_dispatch_counters() == (1, 0)
    assert got.dtype == torch.bfloat16
    assert tuple(got.shape) == (n_pools, KH.INDEX_HEAD_DIM)
    assert torch.equal(_bits(got), _bits(want)), (
        f"n_pools={n_pools} pool_size={pool_size}: "
        f"{int((_bits(got) != _bits(want)).sum())} elements differ from 342e93e"
    )


@pytest.mark.parametrize("n_pools", TOKEN_COUNTS + RAGGED)
@pytest.mark.parametrize("score_dtype", [torch.bfloat16, torch.float32])
def test_one_program_equals_342e93e_bit_for_bit(n_pools, score_dtype):
    _assert_equals_342e93e(n_pools, KH.DEFAULT_POOL_SIZE, score_dtype, seed=n_pools)


@pytest.mark.parametrize("n_pools", TOKEN_COUNTS + RAGGED)
def test_two_programs_equal_342e93e_bit_for_bit(monkeypatch, n_pools):
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    _assert_equals_342e93e(n_pools, KH.DEFAULT_POOL_SIZE, torch.bfloat16, seed=3 + n_pools)


@pytest.mark.parametrize("pool_size", [1, 3, 8])
def test_other_pool_sizes_equal_342e93e_bit_for_bit(monkeypatch, pool_size):
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    _assert_equals_342e93e(300, pool_size, torch.bfloat16, seed=11 * pool_size)


def test_pooling_runs_on_both_programs_from_two_pools_on(monkeypatch):
    monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    assert KH.kpool_hadamard_programs(1024) == 1
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    assert KH.kpool_hadamard_programs(1) == 1
    assert KH.kpool_hadamard_programs(2) == 2
    assert KH.kpool_hadamard_programs(1024) == 2


def _launches(monkeypatch, n_pools: int) -> list:
    """``(grid, args)`` of every launch of the pooling kernel during one call."""
    seen = []
    real = KH.wrap_nki

    def spy(kernel):
        call = real(kernel)

        class _Spy:
            def __getitem__(self, grid):
                inner = call[grid]
                return lambda *a: (seen.append((grid, a)), inner(*a))[1]

            def __call__(self, *a):
                seen.append((1, a))
                return call(*a)
        return _Spy()

    monkeypatch.setattr(KH, "wrap_nki", spy)
    KH.dsa_kpool_hadamard(*_inputs(n_pools, KH.DEFAULT_POOL_SIZE, torch.bfloat16, seed=1))
    return seen


def test_an_lnc2_launch_is_a_two_program_grid_keyed_by_the_file_digest(monkeypatch):
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    seen = _launches(monkeypatch, 64)
    assert [grid for grid, _ in seen] == [2]
    assert seen[-1][1][-1] == KH.SOURCE_DIGEST


def test_one_pool_is_one_program(monkeypatch):
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    seen = _launches(monkeypatch, 1)
    assert [grid for grid, _ in seen] == [1]


def test_the_fused_identity_is_still_the_pooling_kernel():
    KH.dsa_kpool_hadamard(*_inputs(4, KH.DEFAULT_POOL_SIZE, torch.bfloat16, seed=2))
    assert KH.kpool_hadamard_kernel_identity() == (
        "vllm_neuron.functional.dsa.kpool_hadamard", "_kpool_hadamard_nki")
