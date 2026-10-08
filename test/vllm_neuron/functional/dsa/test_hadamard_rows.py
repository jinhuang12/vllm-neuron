# SPDX-License-Identifier: Apache-2.0
"""The query rotation ``dsa_hadamard128`` against 0a08ff4's kernel, bit for bit.

0a08ff4 rotates 128 rows per tile and issues the butterfly one 128-channel block at a
time: 254 vector instructions per tile, on one core. This tree issues each butterfly
stage as two instructions over every block of every row of a tile at once, many rows per
partition, and splits the rows over both cores. The arithmetic is the same fp32 add and
subtract on the same operands in the same order, so the contract is equality, not a
tolerance: any differing bit is a defect.

The reference is the 0a08ff4 snapshot (``test/hardware/baselines/dsa_four_kernel_select``), run in
the same simulator.
"""

from __future__ import annotations

import pytest
import torch

from vllm_neuron.functional.dsa import kpool_hadamard as KH
from vllm_neuron.utils.neuron_utils import can_run_kernel

from test.hardware.baselines.dsa_four_kernel_select import load as load_0a08ff4

#: Row counts: one row, a ragged tile, one whole tile, more than one tile, the decode
#: query at B=4 and B=64 (32 heads per request), and a count that leaves a short final
#: tile on each of two programs.
ROWS = (1, 37, 128, 260, 4 * 32, 64 * 32, 2 * 1024 + 2 * 128 + 6)


@pytest.fixture(autouse=True)
def _kernels_on():
    assert can_run_kernel(), "run with VLLM_NEURON_CPU_MODE=1 and NKI_SIMULATOR=1"
    KH.reset_kpool_hadamard_dispatch_counters()


def _rows(n: int, dtype, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return (torch.randn(n, KH.INDEX_HEAD_DIM, generator=gen) * 3.0).to(dtype)


@pytest.mark.parametrize("n", ROWS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_rotation_equals_0a08ff4_bit_for_bit(n, dtype):
    base = load_0a08ff4()
    x = _rows(n, dtype, seed=n)
    want = base.kpool_hadamard.dsa_hadamard128(x)
    got = KH.dsa_hadamard128(x)
    assert KH.kpool_hadamard_dispatch_counters() == (1, 0)
    assert got.dtype == dtype and tuple(got.shape) == (n, KH.INDEX_HEAD_DIM)
    assert torch.equal(got.view(torch.int16 if dtype == torch.bfloat16 else torch.int32),
                       want.view(torch.int16 if dtype == torch.bfloat16 else torch.int32))


@pytest.mark.parametrize("n", [2, 37, 64 * 32, 2 * 1024 + 2 * 128 + 6])
def test_two_programs_split_the_rows_and_equal_one(monkeypatch, n):
    x = _rows(n, torch.bfloat16, seed=7 + n)
    one = KH.dsa_hadamard128(x)
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    two = KH.dsa_hadamard128(x)
    assert KH.hadamard128_programs(n) == 2
    assert torch.equal(one, two)


def test_one_row_runs_on_one_program(monkeypatch):
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    assert KH.hadamard128_programs(1) == 1
    assert KH.hadamard128_programs(2) == 2


def test_the_kernel_is_keyed_by_this_files_digest():
    """The compiled-kernel cache keys on a kernel's own source and arguments, so the
    rotation kernel takes the module digest as an argument: an edit to a helper it calls
    then changes the key too."""
    seen = []
    real = KH.wrap_nki

    def spy(kernel):
        call = real(kernel)

        class _Spy:
            def __getitem__(self, grid):
                inner = call[grid]
                return lambda *a: (seen.append(a), inner(*a))[1]

            def __call__(self, *a):
                seen.append(a)
                return call(*a)
        return _Spy()

    KH.wrap_nki = spy
    try:
        KH.dsa_hadamard128(_rows(5, torch.bfloat16, seed=1))
    finally:
        KH.wrap_nki = real
    assert seen and seen[-1][-1] == KH.SOURCE_DIGEST
