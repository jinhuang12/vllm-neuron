"""`inc-glm53f-112` acceptance: the dense GEMM consumes the checkpoint's `[128,128]` grid.

WHAT THIS FILE MEASURES, and why it is exact rather than toleranced. The kernel used to index its
scales by ``256`` blocks, so the checkpoint's own ``128`` scales had to be retiled up to ``256``
first -- four scales replaced by one, which is arithmetic ON a scale. This file measures that the
kernel now reads the stored grid directly, and it measures it as EXACT BIT EQUALITY because at this
granularity nothing rescales anything: each ``nc_matmul`` result carries exactly one scale.

THE FIXTURE IS PART OF THE CRITERION. The block scales are ``1.25``, ``1.75``, ``2.5`` and ``3.5``:
not powers of two, not mutually power-of-two related, and short in the mantissa. Over bounded
integer-valued activations and weights every partial sum and every scaled product is exactly
representable in fp32, so ``torch.equal`` is a statement about the kernel rather than about
rounding luck. A precondition row PRINTS that exactness before any equality is read, and the item
REFUSES rather than comparing if the row fails.

THE REFERENCE IS THE MODEL'S, NEVER THE KERNEL'S. It is the dequantisation statement the model
makes at its own call sites (``model_fp8.py``, the dense shared-expert and dense-MLP routes):
dequantise first, then one fp32 matmul. It never consults ``flat_scale_index``, so a transposed
flattening inside the bridge shows up as a numeric disagreement rather than as agreement with
itself.

THREE KERNEL DEFAULTS EACH OWN A FAILING CONTROL. A default that is never falsified is a claim, not
a reading, so each of the three is re-run through a TEST-LOCAL variant kernel that differs in
exactly that one choice, and each variant must produce a FINITE, strictly positive difference, or a
refusal. A ``nan`` or an ``inf`` is a THIRD outcome and fails the item: it means the variant read
memory nothing had written, which is a defect in the variant and says nothing about the default.

THE FOURTH DEFAULT IS RECORDED, NOT FALSIFIED. No number on this route can falsify the fp8-to-bf16
upcast on the weight DMA. The module states that conversion is bit-exact
(``blockwise_fp8_mm.py:47-50``), so dropping it cannot change a result -- the counted run measured
exactly that, ``max_abs_diff=0.0`` with no refusal -- and what the default really buys, that the
Tensor Engine is handed an operand dtype it accepts, is a device-side reading a CPU-mode run cannot
take. So the default is PRINTED as a non-control row with that reason instead of being falsified
here, and the device-side claim is carried as debt ``D-112-UPCAST-DEVICE-ACCEPTANCE``.

Nothing here runs on hardware: the NKI simulator executes the kernel, as ``-026``'s landed
acceptance does.
"""

from __future__ import annotations

import math
import os

import pytest
import torch

import nki
import nki.isa as nisa
import nki.language as nl
import nki.simulator  # noqa: F401  -- imported for its side effect, as the landed file does

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.blockwise_fp8_mm import (
    K_TILES_PER_BLOCK,
    SCALE_BLOCK_SIZE,
    TILE_SIZE,
    blockwise_fp8_mm,
    blockwise_fp8_mm_kernel,
    dispatch_counters,
    reset_dispatch_counters,
    scale_grid_shape,
    to_kernel_scale_layout,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

_FP8 = torch.float8_e4m3fn

M = 256          # tokens; two TILE_SIZE PSUM partition tiles
K = 512          # contraction; four SCALE_BLOCK_SIZE scale blocks
N = 512          # output width; four SCALE_BLOCK_SIZE scale blocks
K_BLOCKS = K // SCALE_BLOCK_SIZE
N_BLOCKS = N // SCALE_BLOCK_SIZE

#: The mantissa families the control must mix. ``2.5 == 2 * 1.25`` and ``3.5 == 2 * 1.75``, so a
#: quad drawn from ONE family retiles losslessly and a control built from it would read 0 for the
#: wrong reason.
_FAMILY_A = (1.25, 2.5)
_FAMILY_B = (1.75, 3.5)
_SCALE_VALUES = (1.25, 1.75, 2.5, 3.5)

#: The kernel default that LEFT the falsification family, with the reason and the debt that carries
#: the reading this route cannot take. Printed by item 1 so the family is four defaults on the
#: transcript even though only three of them own a control.
_NON_CONTROL_DEFAULT = (
    "fp8_to_bf16_upcast_on_the_weight_dma",
    "the module states the conversion is bit-exact (blockwise_fp8_mm.py:47-50), so no fixture can "
    "make it change a number; its value is device-side operand-dtype acceptance, which a CPU-mode "
    "run cannot read",
    "D-112-UPCAST-DEVICE-ACCEPTANCE",
)


class VacuousReadingError(AssertionError):
    """A reading that cannot discriminate, raised instead of being reported as a pass."""


def _scale_grid() -> torch.Tensor:
    """``[K_BLOCKS, N_BLOCKS]`` fp32 scales, cycling the four declared values."""
    grid = torch.empty(K_BLOCKS, N_BLOCKS, dtype=torch.float32)
    for k_block in range(K_BLOCKS):
        for n_block in range(N_BLOCKS):
            grid[k_block, n_block] = _SCALE_VALUES[
                (k_block * N_BLOCKS + n_block) % len(_SCALE_VALUES)
            ]
    return grid


def _integer_fp8(seed: int, *shape: int) -> torch.Tensor:
    """Small positive integers, which are exactly representable in fp8-e4m3 and in fp32."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(1, 8, shape, generator=generator).to(torch.float32)


def _case() -> dict:
    weight = _integer_fp8(112, K, N)
    x = _integer_fp8(113, M, K)
    return {
        "x": x.to(torch.bfloat16),
        "x_fp32": x,
        "weight": weight.to(_FP8),
        "weight_fp32": weight,
        "scale": _scale_grid(),
    }


def _model_reference(case: dict) -> torch.Tensor:
    """The MODEL's dequantisation statement: dequantise first, then one fp32 matmul.

    ``dequantise(weight)[k, n] = weight[k, n] * scale[k // 128, n // 128]``, which is the statement
    the dense routes make at their own call sites. Written with ``repeat_interleave`` so the block
    expansion is the grid's own shape and not an index this file computes.
    """
    dequantised = case["weight_fp32"] * case["scale"].repeat_interleave(
        SCALE_BLOCK_SIZE, 0
    ).repeat_interleave(SCALE_BLOCK_SIZE, 1)
    return case["x_fp32"] @ dequantised


def _exactness_precondition(case: dict) -> tuple[bool, float]:
    """fp64 and fp32 references agree bit-for-bit on this fixture. Printed BEFORE any equality."""
    scale64 = case["scale"].to(torch.float64)
    dequantised64 = case["weight_fp32"].to(torch.float64) * scale64.repeat_interleave(
        SCALE_BLOCK_SIZE, 0
    ).repeat_interleave(SCALE_BLOCK_SIZE, 1)
    reference64 = case["x_fp32"].to(torch.float64) @ dequantised64
    reference32 = _model_reference(case)
    gap = float((reference64 - reference32.to(torch.float64)).abs().max())
    return bool(torch.equal(reference64, reference32.to(torch.float64))), gap


def _emit(item: str, body: str) -> None:
    print(f"E112|{item}|{body}", flush=True)


def _run_kernel(kernel, case: dict) -> torch.Tensor:
    """Run a variant through the SAME seam the real kernel goes through.

    ``wrap_nki`` is the route the shipped path uses, and under ``NKI_SIMULATOR=1`` it is the
    simulator that executes underneath -- the landed acceptance relies on exactly that. Calling the
    simulator directly here would measure a different path than the one item 1 measures, which
    would make the controls answer about the wrong thing.
    """
    scale_t = to_kernel_scale_layout(case["scale"], K, N)
    return wrap_nki(kernel)(x=case["x"], weight=case["weight"], weight_scale_t=scale_t)


# --------------------------------------------------------------------------- #
# Item 1: the grid the kernel indexes IS the grid the checkpoint stores.        #
# --------------------------------------------------------------------------- #
def test_the_kernel_indexes_the_checkpoints_own_grid_exactly() -> None:
    case = _case()
    exact, gap = _exactness_precondition(case)
    _emit(
        "I1_PRECONDITION",
        f"fp64_vs_fp32_bit_equal={int(exact)} max_gap={gap:.6e} "
        f"scales={sorted(set(_SCALE_VALUES))} blocks={K_BLOCKS * N_BLOCKS}",
    )
    if not exact:
        raise VacuousReadingError(
            "the fp64 and fp32 references disagree on this fixture, so an equality taken here "
            "would be a coincidence either way; refusing to compare"
        )

    assert scale_grid_shape(K, N) == (K_BLOCKS, N_BLOCKS)
    assert K_TILES_PER_BLOCK == 1, (
        f"K_TILES_PER_BLOCK={K_TILES_PER_BLOCK}: at the checkpoint's granularity a scale block "
        f"holds exactly one contraction tile, which is what makes one scale per product true"
    )

    reset_dispatch_counters()
    got = blockwise_fp8_mm(case["x"], case["weight"], case["scale"])
    want = _model_reference(case)
    max_abs_diff = float((got - want).abs().max())
    nki_dispatch, torch_fallback = dispatch_counters()
    _emit(
        "I1_EXACT",
        f"grid={K_BLOCKS}x{N_BLOCKS} blocks={K_BLOCKS * N_BLOCKS} "
        f"bit_equal={int(bool(torch.equal(got, want)))} max_abs_diff={max_abs_diff} "
        f"nki_dispatch={nki_dispatch} torch_fallback={torch_fallback}",
    )
    name, reason, debt = _NON_CONTROL_DEFAULT
    _emit(
        "I1_NON_CONTROL",
        f"default={name} falsifiable_on_this_route=0 debt={debt} reason={reason}",
    )
    assert torch.equal(got, want), (
        f"the kernel and the model's own dequantisation statement disagree: "
        f"max_abs_diff={max_abs_diff}"
    )
    assert max_abs_diff == 0
    assert (nki_dispatch, torch_fallback) == (1, 0)


# --------------------------------------------------------------------------- #
# The must-fail control: a lossy 256 retile MUST NOT reach exactness.           #
# --------------------------------------------------------------------------- #
def _test_local_lossy_256(
    weight_fp32: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """A REAL ``256`` retile written HERE, never the producer's: bytes move with the scale.

    THE WHOLE POINT IS THAT BOTH HALVES RUN. A retile keeps ONE scale per quad and re-expresses
    the other three tiles' WEIGHT BYTES against it, ``w * s_tile / s_retained``, then stores them
    back as fp8. Rewriting only the scale would corrupt the product by whole factors and the
    control would pass on gross scale damage rather than on retile loss -- which is what it did
    before this repair, and is the review finding it answers.

    Where the loss lives: the ratio ``s_tile / s_retained`` is not a power of two on this fixture
    (that is what the mixed mantissa families buy), so ``w * ratio`` leaves the fp8 grid and the
    cast rounds. That rounding IS the retile's loss, and it is the only difference between this
    pair and the checkpoint's.

    The quad MAXIMUM is retained, so every ratio is ``<= 1`` and no rescaled byte can overflow
    e4m3 -- a refusal or an ``inf`` would end the item before a number printed.

    Returns:
        The rescaled fp8 weight, the coarsened scale grid expressed on the ``128`` grid the
        kernel indexes, and how many ``128`` tiles the fp8 cast could not express exactly.
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


def test_a_lossy_256_retile_must_not_reach_exactness() -> None:
    case = _case()
    scale = case["scale"]
    families = []
    for k_quad in range(K_BLOCKS // 2):
        for n_quad in range(N_BLOCKS // 2):
            quad = scale[k_quad * 2 : k_quad * 2 + 2, n_quad * 2 : n_quad * 2 + 2]
            values = {float(v) for v in quad.flatten()}
            families.append(
                bool(values & set(_FAMILY_A)) and bool(values & set(_FAMILY_B))
            )
    _emit(
        "I2_CONTROL_QUADS",
        f"quads={len(families)} mixing_both_mantissa_families={sum(families)}",
    )
    if not all(families):
        raise VacuousReadingError(
            "a quad drawn from one mantissa family retiles losslessly, so this control would "
            "read max_abs_diff=0 for the wrong reason"
        )

    lossy_weight, lossy_scale, inexact_tiles = _test_local_lossy_256(
        case["weight_fp32"], scale
    )
    assert not torch.equal(lossy_scale, scale), "the test-local mapping changed no scale"
    assert not torch.equal(
        lossy_weight.to(torch.float32), case["weight_fp32"]
    ), "the test-local mapping changed no weight byte, so it is not a retile"
    if inexact_tiles == 0:
        raise VacuousReadingError(
            "every rescaled tile stayed exactly on the fp8 grid, so this mapping is a LOSSLESS "
            "retile and cannot show that the 128 grid buys anything"
        )

    got = blockwise_fp8_mm(case["x"], lossy_weight, lossy_scale)
    want = _model_reference(case)
    max_abs_diff = float((got - want).abs().max())
    _emit(
        "I2_CONTROL_LOSSY_256",
        f"retained_per_quad=max inexact_fp8_tiles={inexact_tiles}/{K_BLOCKS * N_BLOCKS} "
        f"max_abs_diff={max_abs_diff} reached_zero={int(max_abs_diff == 0)}",
    )
    assert max_abs_diff != 0, (
        "a lossy 256 retile reached exact equality, so the exactness in item 1 is not a "
        "statement about the granularity"
    )


# --------------------------------------------------------------------------- #
# The four kept kernel defaults, each with its own failing control.             #
# --------------------------------------------------------------------------- #
@nki.jit
def _variant_accumulate_true(x, weight, weight_scale_t):
    """DEFAULT (a) inverted, WITH A REAL DIFFERENT-NUMBER PATH.

    The naive inversion -- ``accumulate=True`` on a PSUM tile allocated fresh for each matmul --
    computes the shipped kernel's arithmetic exactly, because there is nothing in the tile to
    accumulate onto. It would read ``max_abs_diff == 0`` and fail this item on correct code. That
    was the review finding, and this is the repair.

    THE FAITHFUL INVERSION HOISTS THE PSUM TILE out of the ``k_block`` loop, which is what
    ``accumulate=True`` is FOR: block ``k`` then adds its raw product onto block ``k-1``'s before
    either is scaled, so every block after the first multiplies a running sum by its own scale.
    That is exactly the defect ``accumulate=False`` prevents once ``K_TILES_PER_BLOCK`` is 1 and
    each product carries exactly one scale.

    THE FIRST MATMUL MUST DEFINE THE HOISTED TILE. The counted run read ``max_abs_diff=nan`` here,
    because ``accumulate=True`` on the FIRST matmul adds a product onto a PSUM tile nothing has
    written yet, so this file was comparing against uninitialised bytes -- a pass on garbage rather
    than on the arithmetic below. ``accumulate=(k_block > 0)`` is how this codebase starts a hoisted
    accumulator: three landed call sites read ``accumulate=(li > 0)``
    (``vllm_neuron/functional/attention/mla_sparse.py:387``, ``:730``, ``:1111``) and no landed
    kernel memsets a PSUM tile. The inversion survives intact, because what the shipped kernel does
    is give every ``k`` block its OWN tile and pass ``accumulate=False`` on every matmul; this
    variant keeps one tile and accumulates across blocks.

    So the difference is arithmetic. With per-block products ``P0..P3`` and scales ``s0..s3`` this
    computes ``P0*s0 + (P0+P1)*s1 + (P0+P1+P2)*s2 + (P0+P1+P2+P3)*s3`` where the kernel computes
    ``P0*s0 + P1*s1 + P2*s2 + P3*s3``, an excess of ``P0*(s1+s2+s3) + P1*(s2+s3) + P2*s3`` -- and on
    this fixture's bounded positive integers and positive scales that excess is strictly positive
    and far inside fp32, so it is finite.
    """
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
            # ONE PSUM TILE FOR EVERY k_block, which is the inversion.
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
                    # The first product DEFINES the hoisted tile; every later one accumulates onto
                    # it. That is the inversion, and it reads no byte nothing wrote.
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
    """DEFAULT (c) inverted: the activation is loaded WITHOUT the DMA-side transpose.

    ONE CHANGE from the shipped body, and it is this load: ``nl.load`` where the kernel writes
    ``nl.load_transpose2d``. The activation tile is square, so ``nc_matmul`` accepts the operand and
    nothing refuses; what it then contracts is the tile's TRANSPOSE against the weight, which is a
    different number on any tile that is not symmetric.

    EVERY OUTPUT TILE IS STORED, which is the second half of this repair. The counted run read
    ``max_abs_diff=nan`` here because the earlier body computed the single tile
    ``out[0:TILE_SIZE, 0:SCALE_BLOCK_SIZE]`` and left the rest of a ``256x512`` ``shared_hbm``
    output never written, so 114,688 of its 131,072 elements were uninitialised bytes at comparison
    time. Covering the whole output is also what makes this a ONE-change inversion of the shipped
    kernel rather than a second, smaller kernel.
    """
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
                # THE INVERSION: a plain load where the kernel transposes on the DMA.
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
    """DEFAULT (d) inverted: the block scale is never applied."""
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
def test_each_kept_kernel_default_is_load_bearing(default_name, variant) -> None:
    """Each kept default, falsified: the variant that drops it must produce a DIFFERENT number.

    THREE OUTCOMES, NOT TWO. A finite, strictly positive ``max_abs_diff`` falsifies the default. A
    REFUSAL falsifies it too -- an operand the Tensor Engine will not take is as good a reading as a
    wrong number and better than a silent one. Anything else FAILS: exact equality says the default
    is not load-bearing, and a ``nan`` or an ``inf`` says the variant read memory nothing had
    written, which is a defect in the variant and no evidence about the default.

    The fourth default of the shipped kernel is not in this family. It is printed by item 1 as a
    non-control row, with its reason and its debt id; see this module's docstring.
    """
    case = _case()
    want = _model_reference(case)
    reached_zero = False
    refusal = ""
    max_abs_diff = float("nan")
    try:
        got = _run_kernel(variant, case)
        if got.shape == want.shape:
            max_abs_diff = float((got - want).abs().max())
            reached_zero = max_abs_diff == 0
        else:
            refusal = f"shape {tuple(got.shape)} != {tuple(want.shape)}"
    except Exception as exc:  # noqa: BLE001 -- the refusal itself is the reading
        refusal = f"{type(exc).__name__}: {str(exc)[:120]}"
    finite = math.isfinite(max_abs_diff)
    _emit(
        "I3_DEFAULT_CONTROL",
        f"default={default_name} max_abs_diff={max_abs_diff} finite={int(finite)} "
        f"refused={int(bool(refusal))} reached_zero={int(reached_zero)} detail={refusal or 'none'}",
    )
    if refusal:
        return  # A refusal IS the falsification; there is no number to read.
    assert finite, (
        f"the variant without '{default_name}' read {max_abs_diff}, which is not a number: it "
        f"compared against memory nothing had written -- an undefined PSUM tile, or an output region "
        f"the variant never stored. That is a defect in THIS variant and says nothing about the "
        f"default, so the item fails rather than passing on garbage"
    )
    assert not reached_zero, (
        f"the variant without '{default_name}' still reached exact equality, so that default is "
        f"not load-bearing and item 1's exactness does not depend on it"
    )


# --------------------------------------------------------------------------- #
# The route predicate, form R-1, with its firing control.                      #
# --------------------------------------------------------------------------- #
def test_the_route_is_the_nki_seam_once_per_call() -> None:
    """Every number above came through the NKI seam, once per call, with no torch fallback.

    Form R-1, `-026`'s, unmoved: the counters are the module's own (incremented at the seam), not a
    test spy, and ``can_run_kernel()`` is read from the real gate.
    """
    case = _case()
    reset_dispatch_counters()
    blockwise_fp8_mm(case["x"], case["weight"], case["scale"])
    one = dispatch_counters()
    blockwise_fp8_mm(case["x"], case["weight"], case["scale"])
    two = dispatch_counters()
    gate = can_run_kernel(torch.zeros(1))
    _emit(
        "I4_ROUTE",
        f"after_first={one} after_second={two} can_run_kernel={gate} kernel="
        f"{blockwise_fp8_mm_kernel.__module__}/{blockwise_fp8_mm_kernel.__name__}",
    )
    assert one == (1, 0), f"expected one dispatch and no fallback, got {one}"
    assert two == (2, 0), f"the counter is not per call: {two}"
    assert gate is True, f"can_run_kernel() read {gate!r}, declared True"


def test_route_control_the_fallback_counter_discriminates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The firing control: with the simulator disabled the seam falls back, and it is COUNTED.

    This is the arm that makes ``torch_fallback == 0`` above a reading rather than an assumption.
    The gate is flipped through the real environment variable
    (``can_run_kernel`` reads ``NKI_SIMULATOR`` at call time under ``VLLM_NEURON_CPU_MODE``), never
    by stubbing the module's own function, so what flips is the shipped condition.
    """
    case = _case()
    monkeypatch.setitem(os.environ, "NKI_SIMULATOR", "0")
    armed = can_run_kernel(torch.zeros(1))
    if armed is not False:
        raise VacuousReadingError(
            f"the gate did not flip with NKI_SIMULATOR=0 (read {armed!r}), so this control is "
            f"unarmed and the zero fallback above would mean nothing"
        )

    reset_dispatch_counters()
    out = blockwise_fp8_mm(case["x"], case["weight"], case["scale"])
    counters = dispatch_counters()
    _emit(
        "I5_ROUTE_CONTROL",
        f"gate={armed} counters={counters} out_shape={tuple(out.shape)}",
    )
    assert counters == (0, 1), f"expected no dispatch and one counted fallback, got {counters}"
