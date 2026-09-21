# SPDX-License-Identifier: Apache-2.0
"""Tests for the blockwise-fp8 GEMM in ``vllm_neuron.functional.blockwise_fp8_mm``.

The NKI simulator runs the kernel; nothing here touches hardware. Two fixtures:

* The tolerance fixture holds power-of-two block scales, distinct per block, so every
  cast in the fixture is exact and a transposed scale index shows up as a numeric
  disagreement. The kernel is compared per output tile against the module's torch
  dequantise-then-matmul oracle at ``rtol=3e-2, atol=1e-5``, the fp8 pair from
  ``vllm_neuron.accuracy.testing.FP8_DTYPE_TOLERANCE``. With every block scale 1.0 the
  dequantisation is a no-op, so that case is held to 1e-5 on both terms.
* The exact-grid fixture holds the scales 1.25, 1.75, 2.5 and 3.5 (not powers of two,
  not all power-of-two related) over small positive integers, so every partial sum is
  exactly representable in fp32 and the kernel is compared with ``torch.equal`` against
  the model's dequantise-first reference.
"""

from __future__ import annotations

import math
import os

import pytest
import torch

import nki
import nki.isa as nisa
import nki.language as nl
import nki.simulator

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.blockwise_fp8_mm import (
    K_TILES_PER_BLOCK,
    SCALE_BLOCK_SIZE,
    TILE_SIZE,
    BlockwiseFp8MmError,
    blockwise_fp8_mm,
    blockwise_fp8_mm_torch_oracle,
    can_run_blockwise_fp8_mm,
    dispatch_counters,
    flat_scale_index,
    kernel_identity,
    kernel_scale_shape,
    reset_dispatch_counters,
    scale_grid_shape,
    to_kernel_scale_layout,
)
from vllm_neuron.functional.moe.blockwise_fp8_retile import (
    BLOCK_QUANT_SIZE as _VENDOR_BLOCK_SIZE,
    is_pow2_exact,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

M = 256  # tokens; a whole number of TILE_SIZE partition tiles
K = 512  # contraction; a whole number of SCALE_BLOCK_SIZE scale blocks
N = 512  # output width; a whole number of SCALE_BLOCK_SIZE scale blocks

M_TILES = M // TILE_SIZE  # 2
K_BLOCKS = K // SCALE_BLOCK_SIZE  # 4
N_BLOCKS = N // SCALE_BLOCK_SIZE  # 4
OUTPUT_TILES = M_TILES * N_BLOCKS  # 8 tiles of (TILE_SIZE, SCALE_BLOCK_SIZE)

# 320: derived so it stays a non-multiple of SCALE_BLOCK_SIZE. 384 would be admissible
# (384 % 128 == 0), so it cannot serve as the refusal case.
INADMISSIBLE_K = SCALE_BLOCK_SIZE * 2 + SCALE_BLOCK_SIZE // 2

# The MoE vendor matmul quantises on 256-wide blocks. The power-of-two fixture below is
# conditioned on that grid: one base exponent per vendor block, one offset per tile.
VENDOR_BLOCK_SIZE = _VENDOR_BLOCK_SIZE
VENDOR_K_BLOCKS = K // VENDOR_BLOCK_SIZE  # 2
VENDOR_N_BLOCKS = N // VENDOR_BLOCK_SIZE  # 2
TILES_PER_VENDOR_BLOCK = VENDOR_BLOCK_SIZE // TILE_SIZE  # 2

# The fp8 pair of ``vllm_neuron.accuracy.testing.FP8_DTYPE_TOLERANCE``, (rtol, atol).
RTOL = 3e-2
ATOL = 1e-5
# With every block scale 1.0 the dequantisation is a no-op, so that case is held to
# single-op tolerance on both terms.
SINGLE_OP_TOL = ATOL

# Exact-grid scales: two mantissa families, ``2.5 == 2 * 1.25`` and ``3.5 == 2 * 1.75``.
# A 2x2 quad drawn from one family would coarsen to 256 losslessly, so the lossy-retile
# test checks that every quad mixes both families.
_FAMILY_A = (1.25, 2.5)
_FAMILY_B = (1.75, 3.5)
_SCALE_VALUES = (1.25, 1.75, 2.5, 3.5)

_FP8 = torch.float8_e4m3fn
_MODULE = "vllm_neuron.functional.blockwise_fp8_mm"


class RouteInstrumentError(AssertionError):
    """The dispatch route was not the expected one."""


class ScaleMappingError(AssertionError):
    """The observed block-to-scale mapping is not the one the module declares."""


class VacuousControlError(AssertionError):
    """A comparison whose input could not have made it fail."""


class _SimulatorCounter:
    """Counts real ``nki.simulator.simulate_kernel`` calls for the duration."""

    def __init__(self) -> None:
        self.calls = 0
        self._real = None

    def __enter__(self) -> "_SimulatorCounter":
        self._real = nki.simulator.simulate_kernel
        real = self._real

        def counting(*args, **kwargs):
            self.calls += 1
            return real(*args, **kwargs)

        nki.simulator.simulate_kernel = counting
        return self

    def __exit__(self, *exc_info) -> None:
        nki.simulator.simulate_kernel = self._real


def _assert_route(sim: _SimulatorCounter, expected_dispatches: int, label: str) -> None:
    """The seam counters, ``can_run_kernel`` and the simulator call count agree."""
    nki_dispatch, torch_fallback = dispatch_counters()
    gate = can_run_kernel(torch.zeros(1))
    if nki_dispatch != expected_dispatches:
        raise RouteInstrumentError(
            f"{label}: seam dispatch counter read {nki_dispatch}, expected "
            f"{expected_dispatches}"
        )
    if torch_fallback != 0:
        raise RouteInstrumentError(
            f"{label}: torch-fallback counter read {torch_fallback}, expected 0; a "
            f"fallback would compare torch against torch"
        )
    if gate is not True:
        raise RouteInstrumentError(
            f"{label}: can_run_kernel() read {gate!r}, expected True"
        )
    if sim.calls != expected_dispatches:
        raise RouteInstrumentError(
            f"{label}: nki.simulator.simulate_kernel ran {sim.calls} times, expected "
            f"{expected_dispatches}; a numeric pass without a simulator call says "
            f"nothing about the kernel"
        )


# tolerance fixture
_BLOCK_EXPONENTS = (-3, 1, 2, -1)
# Per-tile offsets inside one vendor block, bounded at one exponent so a weight m/8
# scaled by any neighbour ratio stays inside e4m3's normal range (1/16 against 2**-6).
_RATIO_OFFSETS = ((0, 1), (-1, 1))


def _pow2_checkpoint_scales(uniform_one: bool = False) -> torch.Tensor:
    """``(1, K//128, N//128)`` fp32 scales, every entry an exact power of two.

    Each vendor block takes a base exponent from ``_BLOCK_EXPONENTS`` and the tiles
    inside it take ``base + offset``, so the scales are distinct and asymmetric.
    ``uniform_one=True`` returns all ones.
    """
    grid = (1, K // TILE_SIZE, N // TILE_SIZE)
    if uniform_one:
        return torch.ones(grid, dtype=torch.float32)

    exponents = torch.zeros(grid[1:], dtype=torch.int64)
    for k_block in range(VENDOR_K_BLOCKS):
        for n_block in range(VENDOR_N_BLOCKS):
            base = _BLOCK_EXPONENTS[
                (k_block * VENDOR_N_BLOCKS + n_block) % len(_BLOCK_EXPONENTS)
            ]
            for d_k in range(TILES_PER_VENDOR_BLOCK):
                for d_n in range(TILES_PER_VENDOR_BLOCK):
                    exponents[
                        k_block * TILES_PER_VENDOR_BLOCK + d_k,
                        n_block * TILES_PER_VENDOR_BLOCK + d_n,
                    ] = base + _RATIO_OFFSETS[d_k][d_n]
    return torch.ldexp(torch.ones(grid, dtype=torch.float32), exponents.unsqueeze(0))


def _fp8_grid(seed: int, *shape: int, signed: bool = False) -> torch.Tensor:
    """Multiples of 1/8 in ``[1/8, 7/8]`` (``[-7/8, 7/8]`` if signed): exact in e4m3.

    Positive by default: over a 512-wide contraction a signed fixture cancels, and a
    pointwise relative tolerance then measures cancellation rather than the kernel.
    """
    generator = torch.Generator().manual_seed(seed)
    low = -7 if signed else 1
    return torch.randint(low, 8, shape, generator=generator).to(torch.float32) / 8.0


def _build_case(uniform_one: bool = False, signed: bool = False) -> dict:
    """The checkpoint's own ``(weight, weight_scale)`` pair plus a bf16 activation."""
    checkpoint = _pow2_checkpoint_scales(uniform_one=uniform_one)
    weights = _fp8_grid(21, 1, K, N, signed=signed)
    x = _fp8_grid(31, M, K, signed=signed).to(torch.bfloat16)
    return {
        "x": x,
        "weight": weights[0].to(_FP8).contiguous(),
        "weight_scale": checkpoint[0].contiguous(),
        "checkpoint": checkpoint,
        "raw_weights": weights,
    }


def _tile(index: int) -> tuple[slice, slice]:
    """Output tile ``index`` as ``(row slice, column slice)``."""
    m_tile, n_block = divmod(index, N_BLOCKS)
    return (
        slice(m_tile * TILE_SIZE, (m_tile + 1) * TILE_SIZE),
        slice(n_block * SCALE_BLOCK_SIZE, (n_block + 1) * SCALE_BLOCK_SIZE),
    )


def _compare_per_output_tile(
    got: torch.Tensor,
    expected: torch.Tensor,
    label: str,
    *,
    rtol: float = RTOL,
    atol: float = ATOL,
) -> None:
    """``assert_close`` per output tile, refusing an all-zero reference."""
    nonzero = int((expected.abs().sum(-1) > 0).sum())
    if nonzero == 0:
        raise VacuousControlError(
            f"{label}: the oracle produced an all-zero reference, so the comparison "
            f"would pass over empty input"
        )
    passed = 0
    for index in range(OUTPUT_TILES):
        rows, cols = _tile(index)
        torch.testing.assert_close(
            got[rows, cols], expected[rows, cols], rtol=rtol, atol=atol
        )
        passed += 1
    assert passed == OUTPUT_TILES, f"{passed}/{OUTPUT_TILES} tiles passed"


# exact-grid fixture
def _exact_grid_scales() -> torch.Tensor:
    """``[K_BLOCKS, N_BLOCKS]`` fp32 scales cycling through ``_SCALE_VALUES``."""
    grid = torch.empty(K_BLOCKS, N_BLOCKS, dtype=torch.float32)
    for k_block in range(K_BLOCKS):
        for n_block in range(N_BLOCKS):
            grid[k_block, n_block] = _SCALE_VALUES[
                (k_block * N_BLOCKS + n_block) % len(_SCALE_VALUES)
            ]
    return grid


def _integer_fp8(seed: int, *shape: int) -> torch.Tensor:
    """Small positive integers, exact in fp8-e4m3 and in fp32."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(1, 8, shape, generator=generator).to(torch.float32)


def _build_exact_grid_case() -> dict:
    weight = _integer_fp8(112, K, N)
    x = _integer_fp8(113, M, K)
    return {
        "x": x.to(torch.bfloat16),
        "x_fp32": x,
        "weight": weight.to(_FP8),
        "weight_fp32": weight,
        "scale": _exact_grid_scales(),
    }


def _model_reference(case: dict) -> torch.Tensor:
    """Dequantise first, then one fp32 matmul, as the model's dense call sites do."""
    dequantised = case["weight_fp32"] * case["scale"].repeat_interleave(
        SCALE_BLOCK_SIZE, 0
    ).repeat_interleave(SCALE_BLOCK_SIZE, 1)
    return case["x_fp32"] @ dequantised


def _fp64_reference_matches_fp32(case: dict) -> bool:
    """The fixture is exact in fp32: an fp64 reference agrees with it bit for bit."""
    scale64 = case["scale"].to(torch.float64)
    dequantised64 = case["weight_fp32"].to(torch.float64) * scale64.repeat_interleave(
        SCALE_BLOCK_SIZE, 0
    ).repeat_interleave(SCALE_BLOCK_SIZE, 1)
    reference64 = case["x_fp32"].to(torch.float64) @ dequantised64
    return bool(torch.equal(reference64, _model_reference(case).to(torch.float64)))


def _run_kernel(kernel, case: dict) -> torch.Tensor:
    """Run a kernel through ``wrap_nki``, the same seam the shipped path uses."""
    scale_t = to_kernel_scale_layout(case["scale"], K, N)
    return wrap_nki(kernel)(x=case["x"], weight=case["weight"], weight_scale_t=scale_t)


def _lossy_256_retile(
    weight_fp32: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Coarsen the 128 grid to 256: keep each quad's largest scale, rescale the bytes.

    Weights move with the scale (``w * s_tile / s_kept``, stored back as fp8). The ratio
    is not a power of two on this fixture, so the fp8 cast rounds; that rounding is the
    retile's loss. Keeping the quad maximum keeps every ratio ``<= 1``, so nothing
    overflows e4m3. Returns the rescaled fp8 weight, the coarsened scales expressed on
    the 128 grid the kernel indexes, and how many tiles the cast could not express.
    """
    retained_grid = scale.clone()
    rescaled = weight_fp32.clone()
    inexact_tiles = 0
    for k_quad in range(K_BLOCKS // 2):
        for n_quad in range(N_BLOCKS // 2):
            k0, n0 = k_quad * 2, n_quad * 2
            keep = float(scale[k0 : k0 + 2, n0 : n0 + 2].max())
            for d_k in range(2):
                for d_n in range(2):
                    k_tile, n_tile = k0 + d_k, n0 + d_n
                    ratio = float(scale[k_tile, n_tile]) / keep
                    rows = slice(k_tile * TILE_SIZE, (k_tile + 1) * TILE_SIZE)
                    cols = slice(n_tile * TILE_SIZE, (n_tile + 1) * TILE_SIZE)
                    exact = weight_fp32[rows, cols] * ratio
                    as_fp8 = exact.to(_FP8)
                    if not torch.equal(as_fp8.to(torch.float32), exact):
                        inexact_tiles += 1
                    rescaled[rows, cols] = as_fp8.to(torch.float32)
                    retained_grid[k_tile, n_tile] = keep
    return rescaled.to(_FP8), retained_grid, inexact_tiles


# tolerance fixture: the kernel against the torch oracle
def test_output_matches_torch_oracle_per_output_tile() -> None:
    """Simulated kernel output matches the torch oracle per tile at ``(RTOL, ATOL)``."""
    case = _build_case()
    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        got = blockwise_fp8_mm(case["x"], case["weight"], case["weight_scale"])
    _assert_route(sim, 1, "tolerance")

    expected = blockwise_fp8_mm_torch_oracle(
        case["x"], case["weight"], case["weight_scale"]
    )
    _compare_per_output_tile(got.to(torch.float32), expected, "tolerance")


def test_output_is_exact_when_every_block_scale_is_one() -> None:
    """All block scales 1.0: the kernel matches the oracle at single-op tolerance."""
    case = _build_case(uniform_one=True)
    scale = case["weight_scale"]
    assert torch.equal(scale, torch.ones_like(scale)), (
        f"the block scales are not all 1.0; got unique values {scale.unique().tolist()}"
    )
    assert is_pow2_exact(1.0)

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        got = blockwise_fp8_mm(case["x"], case["weight"], scale)
    _assert_route(sim, 1, "unit-scale")

    expected = blockwise_fp8_mm_torch_oracle(case["x"], case["weight"], scale)
    _compare_per_output_tile(
        got.to(torch.float32),
        expected,
        "unit-scale",
        rtol=SINGLE_OP_TOL,
        atol=SINGLE_OP_TOL,
    )


def test_flat_scale_index_maps_block_to_predicted_output_columns() -> None:
    """Raising one block's scale moves only the output columns that block owns."""
    case = _build_case(uniform_one=True)
    unit = case["weight_scale"]
    probed_scale = unit.clone()
    target_k, target_n = 1, 1
    probed_scale[target_k, target_n] = 2.0
    if torch.equal(probed_scale, unit):
        raise VacuousControlError("the probe changed no scale")

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        baseline = blockwise_fp8_mm(case["x"], case["weight"], unit).to(torch.float32)
        probed = blockwise_fp8_mm(case["x"], case["weight"], probed_scale).to(
            torch.float32
        )
    # Two kernel runs, so two dispatches.
    _assert_route(sim, 2, "flat-index")

    delta = (probed - baseline).abs()
    per_block = [
        float(delta[:, n * SCALE_BLOCK_SIZE : (n + 1) * SCALE_BLOCK_SIZE].max())
        for n in range(N_BLOCKS)
    ]
    if max(per_block) == 0.0:
        raise VacuousControlError(
            "raising a block scale changed no output at all, so this probe identifies "
            "nothing"
        )
    for n in range(N_BLOCKS):
        moved = per_block[n] > 0.0
        if moved != (n == target_n):
            raise ScaleMappingError(
                f"block (k={target_k}, n={target_n}) moved output columns of n_block "
                f"{n} = {moved}, expected {n == target_n}; observed deltas "
                f"{per_block}; flat_scale_index({target_k}, {target_n}, {N_BLOCKS}) = "
                f"{flat_scale_index(target_k, target_n, N_BLOCKS)}"
            )


def test_signed_fixture_agrees_in_norm_under_cancellation() -> None:
    """Signed weights and activations agree with the oracle in per-tile relative L2."""
    case = _build_case(signed=True)
    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        got = blockwise_fp8_mm(case["x"], case["weight"], case["weight_scale"]).to(
            torch.float32
        )
    _assert_route(sim, 1, "signed")
    expected = blockwise_fp8_mm_torch_oracle(
        case["x"], case["weight"], case["weight_scale"]
    )

    # Over a 512-wide contraction a signed fixture cancels, so RTOL bounds a per-tile
    # relative L2 norm here instead of the pointwise pair; the norm is not distorted.
    for index in range(OUTPUT_TILES):
        rows, cols = _tile(index)
        residual = float((got[rows, cols] - expected[rows, cols]).pow(2).sum().sqrt())
        reference = float(expected[rows, cols].pow(2).sum().sqrt())
        if reference == 0.0:
            raise VacuousControlError(
                f"tile {index}: the reference has zero norm, so a relative comparison "
                f"against it is vacuous"
            )
        rel_l2 = residual / reference
        assert rel_l2 <= RTOL, (
            f"tile {index}: relative L2 {rel_l2:.6e} exceeds rtol {RTOL}; that is a "
            f"structural disagreement, not cancellation"
        )


# exact-grid fixture: bit equality against the model's reference
def test_the_kernel_indexes_the_checkpoint_scale_grid_exactly() -> None:
    """On an fp32-exact fixture the kernel equals the model's reference bit for bit."""
    case = _build_exact_grid_case()
    if not _fp64_reference_matches_fp32(case):
        raise VacuousControlError(
            "the fp64 and fp32 references disagree on this fixture, so an equality "
            "taken here would be a coincidence either way"
        )

    assert scale_grid_shape(K, N) == (K_BLOCKS, N_BLOCKS)
    assert K_TILES_PER_BLOCK == 1, (
        f"K_TILES_PER_BLOCK={K_TILES_PER_BLOCK}: a scale block must hold exactly one "
        f"contraction tile for each product to carry exactly one scale"
    )

    reset_dispatch_counters()
    got = blockwise_fp8_mm(case["x"], case["weight"], case["scale"])
    expected = _model_reference(case)
    max_abs_diff = float((got - expected).abs().max())
    nki_dispatch, torch_fallback = dispatch_counters()
    assert torch.equal(got, expected), (
        f"kernel and the model's dequantisation disagree: max_abs_diff={max_abs_diff}"
    )
    assert max_abs_diff == 0
    assert (nki_dispatch, torch_fallback) == (1, 0)


def test_a_lossy_256_retile_does_not_reach_exactness() -> None:
    """Coarsening the checkpoint grid to 256 breaks the exact equality above."""
    case = _build_exact_grid_case()
    scale = case["scale"]
    families = []
    for k_quad in range(K_BLOCKS // 2):
        for n_quad in range(N_BLOCKS // 2):
            quad = scale[k_quad * 2 : k_quad * 2 + 2, n_quad * 2 : n_quad * 2 + 2]
            values = {float(v) for v in quad.flatten()}
            families.append(
                bool(values & set(_FAMILY_A)) and bool(values & set(_FAMILY_B))
            )
    if not all(families):
        raise VacuousControlError(
            "a quad drawn from one mantissa family retiles losslessly, so this "
            "comparison would read max_abs_diff=0 for the wrong reason"
        )

    lossy_weight, lossy_scale, inexact_tiles = _lossy_256_retile(
        case["weight_fp32"], scale
    )
    assert not torch.equal(lossy_scale, scale), "the retile changed no scale"
    assert not torch.equal(
        lossy_weight.to(torch.float32), case["weight_fp32"]
    ), "the retile changed no weight byte, so it is not a retile"
    if inexact_tiles == 0:
        raise VacuousControlError(
            "every rescaled tile stayed exactly on the fp8 grid, so this retile is "
            "lossless and cannot show that the 128 grid buys anything"
        )

    got = blockwise_fp8_mm(case["x"], lossy_weight, lossy_scale)
    expected = _model_reference(case)
    max_abs_diff = float((got - expected).abs().max())
    assert max_abs_diff != 0, (
        "a lossy 256 retile reached exact equality, so the exactness of the 128 grid "
        "is not a statement about the granularity"
    )


@nki.jit
def _variant_accumulate_true(x, weight, weight_scale_t):
    """The kernel with one PSUM tile per output tile, accumulating across k blocks."""
    m_extent, k_extent = x.shape
    _, n_extent = weight.shape
    n_n_blocks = n_extent // SCALE_BLOCK_SIZE
    n_k_blocks = k_extent // SCALE_BLOCK_SIZE
    out = nl.ndarray((m_extent, n_extent), dtype=nl.float32, buffer=nl.shared_hbm)
    scale_sb = nl.load(weight_scale_t)
    for m_tile in range(m_extent // TILE_SIZE):
        m0 = m_tile * TILE_SIZE
        for n_block in range(n_n_blocks):
            n0 = n_block * SCALE_BLOCK_SIZE
            acc = nl.ndarray(
                (TILE_SIZE, SCALE_BLOCK_SIZE), dtype=nl.float32, buffer=nl.sbuf
            )
            # One PSUM tile for every k_block. The first product defines it and each
            # later one accumulates onto it, so every scale after the first multiplies
            # a running sum of raw products instead of its own block's product.
            psum = nl.ndarray(
                (TILE_SIZE, SCALE_BLOCK_SIZE), dtype=nl.float32, buffer=nl.psum
            )
            for k_block in range(n_k_blocks):
                k0 = k_block * SCALE_BLOCK_SIZE
                x_t = nl.load_transpose2d(x[m0 : m0 + TILE_SIZE, k0 : k0 + TILE_SIZE])
                w_tile = nl.load(
                    weight[k0 : k0 + TILE_SIZE, n0 : n0 + SCALE_BLOCK_SIZE],
                    dtype=nl.bfloat16,
                )
                nisa.nc_matmul(
                    dst=psum[0:TILE_SIZE, 0:SCALE_BLOCK_SIZE],
                    stationary=x_t,
                    moving=w_tile,
                    accumulate=(k_block > 0),
                )
                flat = k_block * n_n_blocks + n_block
                if k_block == 0:
                    nisa.tensor_scalar(
                        dst=acc[0:TILE_SIZE, 0:SCALE_BLOCK_SIZE],
                        data=psum[0:TILE_SIZE, 0:SCALE_BLOCK_SIZE],
                        op0=nl.multiply,
                        operand0=scale_sb[0:TILE_SIZE, flat : flat + 1],
                    )
                else:
                    nisa.scalar_tensor_tensor(
                        dst=acc[0:TILE_SIZE, 0:SCALE_BLOCK_SIZE],
                        data=psum[0:TILE_SIZE, 0:SCALE_BLOCK_SIZE],
                        op0=nl.multiply,
                        operand0=scale_sb[0:TILE_SIZE, flat : flat + 1],
                        op1=nl.add,
                        operand1=acc[0:TILE_SIZE, 0:SCALE_BLOCK_SIZE],
                    )
            nl.store(
                out[m0 : m0 + TILE_SIZE, n0 : n0 + SCALE_BLOCK_SIZE],
                value=acc[0:TILE_SIZE, 0:SCALE_BLOCK_SIZE],
            )
    return out


@nki.jit
def _variant_no_transpose(x, weight, weight_scale_t):
    """The kernel with a plain activation load in place of ``load_transpose2d``."""
    m_extent, k_extent = x.shape
    _, n_extent = weight.shape
    n_n_blocks = n_extent // SCALE_BLOCK_SIZE
    n_k_blocks = k_extent // SCALE_BLOCK_SIZE
    out = nl.ndarray((m_extent, n_extent), dtype=nl.float32, buffer=nl.shared_hbm)
    scale_sb = nl.load(weight_scale_t)
    for m_tile in range(m_extent // TILE_SIZE):
        m0 = m_tile * TILE_SIZE
        for n_block in range(n_n_blocks):
            n0 = n_block * SCALE_BLOCK_SIZE
            acc = nl.ndarray(
                (TILE_SIZE, SCALE_BLOCK_SIZE), dtype=nl.float32, buffer=nl.sbuf
            )
            for k_block in range(n_k_blocks):
                psum = nl.ndarray(
                    (TILE_SIZE, SCALE_BLOCK_SIZE), dtype=nl.float32, buffer=nl.psum
                )
                k0 = k_block * SCALE_BLOCK_SIZE
                # The activation tile is square, so nc_matmul accepts the untransposed
                # operand and contracts the tile's transpose against the weight.
                x_plain = nl.load(x[m0 : m0 + TILE_SIZE, k0 : k0 + TILE_SIZE])
                w_tile = nl.load(
                    weight[k0 : k0 + TILE_SIZE, n0 : n0 + SCALE_BLOCK_SIZE],
                    dtype=nl.bfloat16,
                )
                nisa.nc_matmul(
                    dst=psum[0:TILE_SIZE, 0:SCALE_BLOCK_SIZE],
                    stationary=x_plain,
                    moving=w_tile,
                    accumulate=False,
                )
                flat = k_block * n_n_blocks + n_block
                if k_block == 0:
                    nisa.tensor_scalar(
                        dst=acc[0:TILE_SIZE, 0:SCALE_BLOCK_SIZE],
                        data=psum[0:TILE_SIZE, 0:SCALE_BLOCK_SIZE],
                        op0=nl.multiply,
                        operand0=scale_sb[0:TILE_SIZE, flat : flat + 1],
                    )
                else:
                    nisa.scalar_tensor_tensor(
                        dst=acc[0:TILE_SIZE, 0:SCALE_BLOCK_SIZE],
                        data=psum[0:TILE_SIZE, 0:SCALE_BLOCK_SIZE],
                        op0=nl.multiply,
                        operand0=scale_sb[0:TILE_SIZE, flat : flat + 1],
                        op1=nl.add,
                        operand1=acc[0:TILE_SIZE, 0:SCALE_BLOCK_SIZE],
                    )
            nl.store(
                out[m0 : m0 + TILE_SIZE, n0 : n0 + SCALE_BLOCK_SIZE],
                value=acc[0:TILE_SIZE, 0:SCALE_BLOCK_SIZE],
            )
    return out


@nki.jit
def _variant_no_block_scale(x, weight, weight_scale_t):
    """The kernel with the block scale never applied."""
    m_extent, k_extent = x.shape
    _, n_extent = weight.shape
    n_n_blocks = n_extent // SCALE_BLOCK_SIZE
    n_k_blocks = k_extent // SCALE_BLOCK_SIZE
    out = nl.ndarray((m_extent, n_extent), dtype=nl.float32, buffer=nl.shared_hbm)
    for m_tile in range(m_extent // TILE_SIZE):
        m0 = m_tile * TILE_SIZE
        for n_block in range(n_n_blocks):
            n0 = n_block * SCALE_BLOCK_SIZE
            acc = nl.ndarray(
                (TILE_SIZE, SCALE_BLOCK_SIZE), dtype=nl.float32, buffer=nl.sbuf
            )
            for k_block in range(n_k_blocks):
                psum = nl.ndarray(
                    (TILE_SIZE, SCALE_BLOCK_SIZE), dtype=nl.float32, buffer=nl.psum
                )
                k0 = k_block * SCALE_BLOCK_SIZE
                x_t = nl.load_transpose2d(x[m0 : m0 + TILE_SIZE, k0 : k0 + TILE_SIZE])
                w_tile = nl.load(
                    weight[k0 : k0 + TILE_SIZE, n0 : n0 + SCALE_BLOCK_SIZE],
                    dtype=nl.bfloat16,
                )
                nisa.nc_matmul(
                    dst=psum[0:TILE_SIZE, 0:SCALE_BLOCK_SIZE],
                    stationary=x_t,
                    moving=w_tile,
                    accumulate=False,
                )
                if k_block == 0:
                    nisa.tensor_copy(
                        dst=acc[0:TILE_SIZE, 0:SCALE_BLOCK_SIZE],
                        src=psum[0:TILE_SIZE, 0:SCALE_BLOCK_SIZE],
                    )
                else:
                    nisa.tensor_tensor(
                        dst=acc[0:TILE_SIZE, 0:SCALE_BLOCK_SIZE],
                        data1=psum[0:TILE_SIZE, 0:SCALE_BLOCK_SIZE],
                        data2=acc[0:TILE_SIZE, 0:SCALE_BLOCK_SIZE],
                        op=nl.add,
                    )
            nl.store(
                out[m0 : m0 + TILE_SIZE, n0 : n0 + SCALE_BLOCK_SIZE],
                value=acc[0:TILE_SIZE, 0:SCALE_BLOCK_SIZE],
            )
    return out


@pytest.mark.parametrize(
    "default_name, variant",
    [
        ("accumulate_false_on_every_matmul", _variant_accumulate_true),
        ("load_transpose2d_for_the_activation", _variant_no_transpose),
        ("per_block_scale_applied", _variant_no_block_scale),
    ],
)
def test_each_kernel_default_changes_the_result(default_name, variant) -> None:
    """A kernel that drops one default gives a different finite number, or refuses."""
    case = _build_exact_grid_case()
    expected = _model_reference(case)
    reached_zero = False
    refusal = ""
    max_abs_diff = float("nan")
    try:
        got = _run_kernel(variant, case)
        if got.shape == expected.shape:
            max_abs_diff = float((got - expected).abs().max())
            reached_zero = max_abs_diff == 0
        else:
            refusal = f"shape {tuple(got.shape)} != {tuple(expected.shape)}"
    except Exception as exc:  # noqa: BLE001 -- a refusal is a valid outcome
        refusal = f"{type(exc).__name__}: {str(exc)[:120]}"
    if refusal:
        return
    assert math.isfinite(max_abs_diff), (
        f"the variant without '{default_name}' read {max_abs_diff}: it compared "
        f"against memory nothing had written (an undefined PSUM tile or an unstored "
        f"output region), a defect in the variant that says nothing about the default"
    )
    assert not reached_zero, (
        f"the variant without '{default_name}' still reached exact equality, so that "
        f"default does not change the result"
    )


# dispatch route
def test_the_seam_dispatches_to_the_kernel_once_per_call() -> None:
    """Each call adds one kernel dispatch and no torch fallback."""
    case = _build_exact_grid_case()
    reset_dispatch_counters()
    blockwise_fp8_mm(case["x"], case["weight"], case["scale"])
    one = dispatch_counters()
    blockwise_fp8_mm(case["x"], case["weight"], case["scale"])
    two = dispatch_counters()
    gate = can_run_kernel(torch.zeros(1))
    assert one == (1, 0), f"expected one dispatch and no fallback, got {one}"
    assert two == (2, 0), f"the counter is not per call: {two}"
    assert gate is True, f"can_run_kernel() read {gate!r}, expected True"


def test_the_seam_falls_back_to_torch_when_the_kernel_cannot_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the simulator disabled the seam takes the torch path and counts it."""
    case = _build_case()
    monkeypatch.setitem(os.environ, "NKI_SIMULATOR", "0")
    assert can_run_kernel(torch.zeros(1)) is False, (
        "the gate did not flip with NKI_SIMULATOR=0"
    )

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        out = blockwise_fp8_mm(case["x"], case["weight"], case["weight_scale"])
    nki_dispatch, torch_fallback = dispatch_counters()
    assert nki_dispatch == 0, f"expected 0 NKI dispatches, got {nki_dispatch}"
    assert torch_fallback == 1, f"expected 1 torch fallback, got {torch_fallback}"
    assert sim.calls == 0, f"the simulator ran {sim.calls} times with it disabled"
    assert tuple(out.shape) == (M, N)


def test_kernel_identity_names_this_modules_kernel() -> None:
    """The seam dispatches to this module's own kernel, not a vendor one."""
    module, qualname = kernel_identity()
    assert module == _MODULE, module
    assert qualname == "blockwise_fp8_mm_kernel", qualname
    assert not module.startswith("nkilib"), (
        f"the seam dispatches to a vendor kernel: {module}.{qualname}"
    )


# geometry and scale layout
@pytest.mark.parametrize(
    "tokens,rows,cols,needle",
    [
        (200, 512, 512, "M=200 is not a positive multiple of TILE_SIZE"),
        (256, INADMISSIBLE_K, 512, f"K={INADMISSIBLE_K} is not a positive multiple of"),
        (256, 512, 300, "N=300 is not a positive multiple of"),
        (0, 512, 512, "M=0 is not a positive multiple of TILE_SIZE"),
    ],
)
def test_refuses_inadmissible_geometry_by_name(
    tokens: int, rows: int, cols: int, needle: str
) -> None:
    """Every refusal is a named error carrying the offending extent."""
    with pytest.raises(BlockwiseFp8MmError) as excinfo:
        can_run_blockwise_fp8_mm(torch.zeros(1), rows, cols, tokens)
    message = str(excinfo.value)
    assert needle in message, f"[M={tokens},K={rows},N={cols}] message was: {message}"


def test_to_kernel_scale_layout_refuses_and_conserves() -> None:
    """One operand column per block scale; mis-sized or bf16 grids are refused."""
    good = torch.ones(scale_grid_shape(K, N), dtype=torch.float32)
    bridged = to_kernel_scale_layout(good, K, N)
    assert tuple(bridged.shape) == kernel_scale_shape(K, N)
    assert bridged.shape[0] == TILE_SIZE
    assert bridged.shape[1] == K_BLOCKS * N_BLOCKS

    # Each column is one block's scale, replicated down the partition axis.
    distinct = torch.arange(
        K_BLOCKS * N_BLOCKS, dtype=torch.float32
    ).reshape(K_BLOCKS, N_BLOCKS) + 1.0
    operand = to_kernel_scale_layout(distinct, K, N)
    for k_block in range(K_BLOCKS):
        for n_block in range(N_BLOCKS):
            column = operand[:, flat_scale_index(k_block, n_block, N_BLOCKS)]
            assert torch.equal(column, column[0].expand(TILE_SIZE)), (
                "the operand column is not a constant replication, so tensor_scalar "
                "would broadcast a non-uniform scale"
            )
            assert float(column[0]) == float(distinct[k_block, n_block])

    with pytest.raises(BlockwiseFp8MmError) as excinfo:
        to_kernel_scale_layout(torch.ones((K_BLOCKS, N_BLOCKS + 1)), K, N)
    assert "mis-sized" in str(excinfo.value)

    with pytest.raises(BlockwiseFp8MmError):
        to_kernel_scale_layout(good.to(torch.bfloat16), K, N)
