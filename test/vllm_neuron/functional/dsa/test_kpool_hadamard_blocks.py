# SPDX-License-Identifier: Apache-2.0
"""``_fwht128_blocks`` against the butterfly the decode kernels ran at f3a833f.

``_fwht128_blocks(buf_a, buf_b, parts, rows)`` is the name and signature the batched decode
ring step calls for its butterfly. f3a833f has no such function; its decode kernels ran
``_fwht128_inplace(buf_a, buf_b, head_dim)`` on one 128-channel row per partition, one block
per instruction. So the reference here is that function, read from f3a833f by the hardware
benchmark's pinned snapshot, applied row by row; the result must be equal bit for bit,
because both issue the same fp32 add and subtract on the same operands in the same stage
order, and the decode kernels' outputs are compared bit for bit on the device against
f3a833f's.

The inputs span 2^-20 to 2^20 so that a different addition order would show; seven stages
grow a value by at most 2^7, far from fp32 overflow.
"""

from __future__ import annotations

import pytest
import torch

import nki
import nki.isa as nisa
import nki.language as nl
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from test.hardware.benchmark_dsa_hadamard import load_baseline
from vllm_neuron.functional.dsa import kpool_hadamard
from vllm_neuron.functional.dsa.kpool_hadamard import INDEX_HEAD_DIM

#: f3a833f's module, loaded once; the reference kernel reads its butterfly through this name.
_BASELINE = None

#: Partitions of an SBUF tile.
PARTITIONS = nl.tile_size.pmax

#: Exponent range of the random inputs (see the module docstring).
EXPONENT_RANGE = 20


@pytest.fixture(scope="module", autouse=True)
def baseline(tmp_path_factory):
    global _BASELINE
    _BASELINE = load_baseline(tmp_path_factory.mktemp("f3a833f"))
    return _BASELINE


@nki.jit
def _blocks_kernel(x_hbm, rows):
    """``_fwht128_blocks`` over a ``(parts, rows * 128)`` fp32 tile, unscaled."""
    parts, width = x_hbm.shape
    buf_a = nl.ndarray((parts, width), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=buf_a, src=x_hbm)
    buf_b = nl.ndarray((parts, width), dtype=nl.float32, buffer=nl.sbuf)
    result = kpool_hadamard._fwht128_blocks(buf_a, buf_b, parts, rows)
    out = nl.ndarray((parts, width), dtype=nl.float32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=result)
    return out


@nki.jit
def _baseline_rows_kernel(x_hbm, rows):
    """f3a833f's ``_fwht128_inplace`` on each 128-channel row of the tile, one at a time."""
    parts, width = x_hbm.shape
    out = nl.ndarray((parts, width), dtype=nl.float32, buffer=nl.shared_hbm)
    for row in range(rows):
        columns = slice(row * INDEX_HEAD_DIM, (row + 1) * INDEX_HEAD_DIM)
        buf_a = nl.ndarray((parts, INDEX_HEAD_DIM), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=buf_a, src=x_hbm[:, columns])
        buf_b = nl.ndarray((parts, INDEX_HEAD_DIM), dtype=nl.float32, buffer=nl.sbuf)
        result = _BASELINE.kpool_hadamard._fwht128_inplace(buf_a, buf_b, INDEX_HEAD_DIM)
        nisa.dma_copy(dst=out[:, columns], src=result)
    return out


def _blocks(parts: int, rows: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    mantissa = torch.randn((parts, rows * INDEX_HEAD_DIM), generator=gen)
    exponent = torch.randint(-EXPONENT_RANGE, EXPONENT_RANGE + 1, mantissa.shape, generator=gen)
    return torch.ldexp(mantissa, exponent)


@pytest.mark.parametrize("parts", [1, 37, PARTITIONS])
@pytest.mark.parametrize("rows", [1, kpool_hadamard.DEFAULT_POOL_SIZE])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_blocks_equal_f3a833f_butterfly_bit_for_bit(parts: int, rows: int, seed: int) -> None:
    x = _blocks(parts, rows, seed)
    got = wrap_nki(_blocks_kernel)(x, rows)
    want = wrap_nki(_baseline_rows_kernel)(x, rows)
    assert torch.equal(got, want)
    # And both are the transform: H_128 on each row, unscaled, within the error of a
    # 7-level tree of fp32 sums, gamma(7) * sum_i |x_i|.
    rows_in = x.double().reshape(parts, rows, INDEX_HEAD_DIM)
    exact = rows_in @ kpool_hadamard.hadamard_matrix(INDEX_HEAD_DIM, torch.float64).t()
    eps = torch.finfo(torch.float32).eps / 2
    stages = len(kpool_hadamard.HADAMARD_STRIDES)
    gamma = stages * eps / (1 - stages * eps)
    error = (got.double().reshape(parts, rows, INDEX_HEAD_DIM) - exact).abs()
    assert bool((error <= gamma * rows_in.abs().sum(dim=-1, keepdim=True)).all())
