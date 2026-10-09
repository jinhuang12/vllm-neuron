# SPDX-License-Identifier: Apache-2.0
"""Tests for fused kpool compression + Hadamard-128 and for the rotation stage alone.

The fused kernel is compared against an unfused composition of the same two stages,
built here from ``torch.softmax`` and a multiply by the Sylvester ``H_128``. A matrix
multiply and the shipped 7-stage butterfly compute the same linear map, so the
composition is an independent way of writing the same thing, and it dispatches no
kernel of its own.

``dsa_hadamard128``, the rotation stage on its own, is read three ways: against the
same matrix multiply on non-identity rows, on ``I_128`` where its output is the
transform matrix itself, and against its own involution ``h(h(x)) == x``, which needs
no reference matrix at all.
"""

from __future__ import annotations

import math

import pytest
import torch

from vllm_neuron.functional.dsa import kpool_hadamard
from vllm_neuron.functional.dsa.kpool_hadamard import (
    HADAMARD_SCALE,
    HADAMARD_STAGES,
    INDEX_HEAD_DIM,
    KpoolHadamardError,
    can_run_dsa_hadamard128,
    can_run_dsa_kpool_hadamard,
    dsa_hadamard128,
    dsa_kpool_hadamard,
    hadamard_matrix,
    kpool_hadamard_dispatch_counters,
    kpool_hadamard_kernel_identity,
    reset_kpool_hadamard_dispatch_counters,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

# ``index_kpool`` for the target checkpoint.
POOL_SIZE = 4

# bf16 inputs carry about three decimal digits, so 1e-2 relative is the usable bound
# for the kernel-versus-reference comparison; the 1e-5 absolute floor covers the
# elements that sit near zero, where a relative bound says nothing.
RTOL = 1e-2
ATOL = 1e-5


def _inputs(
    n_pools: int,
    seed: int,
    score_dtype: torch.dtype = torch.float32,
    pool_size: int = POOL_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One case's three input tensors.

    ``slot_k`` is bf16 because that is the only dtype the fused gate admits; ``ape`` is
    fp32. ``slot_score`` is fp32 unless a case asks for the bf16 score the indexer
    serves; the kernel reads either.
    """
    gen = torch.Generator().manual_seed(seed)
    slot_k = torch.randn(
        (n_pools, pool_size, INDEX_HEAD_DIM), generator=gen, dtype=torch.float32
    ).to(torch.bfloat16)
    slot_score = torch.randn(
        (n_pools, pool_size, INDEX_HEAD_DIM), generator=gen, dtype=torch.float32
    ).to(score_dtype)
    ape = torch.randn((pool_size, INDEX_HEAD_DIM), generator=gen, dtype=torch.float32)
    return slot_k, slot_score, ape


def _rows(n_rows: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
    """``[n_rows, 128]`` of non-identity random values, for the rotation stage alone.

    Non-identity is the point: on ``I`` the output is the transform matrix, which cannot
    separate the transform from any other map with the same image of ``I``.
    """
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn((n_rows, INDEX_HEAD_DIM), generator=gen, dtype=torch.float32)
    return x.to(dtype)


def _reference_rotation(x: torch.Tensor, out_dtype: torch.dtype | None = None) -> torch.Tensor:
    """``x @ H_128.T * (1 / sqrt(128))`` as a plain matrix multiply, computed in fp32."""
    rotated = x.float() @ hadamard_matrix(INDEX_HEAD_DIM).t()
    return (rotated * HADAMARD_SCALE).to(out_dtype or x.dtype)


def _unfused_reference(
    slot_k: torch.Tensor, slot_score: torch.Tensor, ape: torch.Tensor
) -> torch.Tensor:
    """Pool, then rotate, as three separate whole-tensor steps. Dispatches no kernel.

    The softmax is over ``dim=1``, the slot axis, so the weights are per
    ``(pool, channel)`` and not per pool.
    """
    weights = torch.softmax(slot_score.float() + ape.float().unsqueeze(0), dim=1)
    pooled = (weights * slot_k.float()).sum(dim=1)
    return _reference_rotation(pooled, slot_k.dtype)


def _assert_fused_matches_unfused(
    n_pools: int,
    seed: int,
    score_dtype: torch.dtype = torch.float32,
    pool_size: int = POOL_SIZE,
) -> None:
    """Run the fused kernel for one pool count and compare it with the unfused form."""
    slot_k, slot_score, ape = _inputs(n_pools, seed, score_dtype, pool_size)
    # The reference is built before the reset, so the counters read below cover the
    # kernel call alone.
    expected = _unfused_reference(slot_k, slot_score, ape)
    reset_kpool_hadamard_dispatch_counters()
    gate = can_run_dsa_kpool_hadamard(slot_k, slot_score, ape)
    got = dsa_kpool_hadamard(slot_k, slot_score, ape)
    nki, fallback = kpool_hadamard_dispatch_counters()

    assert gate is True
    assert (nki, fallback) == (1, 0), (
        f"n_pools={n_pools} must take the kernel once and never the torch path; "
        f"got {nki} dispatch(es) and {fallback} fallback(s)"
    )
    assert tuple(got.shape) == (n_pools, INDEX_HEAD_DIM)
    torch.testing.assert_close(got.float(), expected.float(), rtol=RTOL, atol=ATOL)


def _assert_rotation_matches_reference(n_rows: int, dtype: torch.dtype, seed: int) -> None:
    """Run the rotation stage for one row count and dtype against the matrix multiply."""
    x = _rows(n_rows, dtype, seed)
    expected = _reference_rotation(x)
    reset_kpool_hadamard_dispatch_counters()
    gate = can_run_dsa_hadamard128(x)
    got = dsa_hadamard128(x)
    nki, fallback = kpool_hadamard_dispatch_counters()

    assert gate is True
    # One dispatch and not one per tile: the tile loop is inside the kernel.
    assert (nki, fallback) == (1, 0), (
        f"n_rows={n_rows} must take the kernel once and never the torch path; "
        f"got {nki} dispatch(es) and {fallback} fallback(s)"
    )
    assert tuple(got.shape) == (n_rows, INDEX_HEAD_DIM)
    assert got.dtype == dtype
    torch.testing.assert_close(got.float(), expected.float(), rtol=RTOL, atol=ATOL)


def test_gate_can_run_kernel_is_true() -> None:
    """``can_run_kernel()`` is True, so every case below takes the kernel route."""
    assert can_run_kernel() is True


def test_gate_admits_the_fused_input() -> None:
    """The fused gate admits the shapes and dtypes the cases below use."""
    slot_k, slot_score, ape = _inputs(33, seed=1)
    assert can_run_dsa_kpool_hadamard(slot_k, slot_score, ape) is True


def test_gate_admits_128_rows() -> None:
    """The rotation gate admits a ``[128, 128]`` tensor."""
    assert can_run_dsa_hadamard128(torch.eye(INDEX_HEAD_DIM, dtype=torch.bfloat16)) is True


def test_gate_admits_37_rows_fp32() -> None:
    """The rotation gate has no row-count and no dtype clause, so a partial tile is admitted."""
    assert can_run_dsa_hadamard128(_rows(37, torch.float32, seed=2)) is True


def test_gate_admits_260_rows_bf16() -> None:
    """The rotation gate admits a multi-tile row count."""
    assert can_run_dsa_hadamard128(_rows(260, torch.bfloat16, seed=3)) is True


def test_fused_matches_unfused_for_1_pool() -> None:
    """A single pool: the smallest tile the kernel can be handed."""
    _assert_fused_matches_unfused(1, seed=101)


def test_fused_matches_unfused_for_33_pools() -> None:
    """33 pools: a partial 128-partition tile, so the tile is narrowed."""
    _assert_fused_matches_unfused(33, seed=102)


def test_fused_matches_unfused_for_130_pools() -> None:
    """130 pools: one full tile plus two rows, so a short second tile follows."""
    _assert_fused_matches_unfused(130, seed=103)


def test_fused_matches_unfused_for_512_pools() -> None:
    """512 pools: four full tiles and no remainder."""
    _assert_fused_matches_unfused(512, seed=104)


def test_128_rows_bf16_matches_the_reference() -> None:
    """128 non-identity bf16 rows: exactly one full tile."""
    _assert_rotation_matches_reference(128, torch.bfloat16, seed=101)


def test_37_rows_fp32_matches_the_reference() -> None:
    """37 non-identity fp32 rows: a partial tile, and not a multiple of 128."""
    _assert_rotation_matches_reference(37, torch.float32, seed=102)


def test_260_rows_bf16_matches_the_reference() -> None:
    """260 non-identity bf16 rows: the tile loop runs three times, the last tile carrying four."""
    _assert_rotation_matches_reference(260, torch.bfloat16, seed=103)


def test_identity_input_reproduces_the_transform_matrix() -> None:
    """On ``I_128`` the output is ``H_128 / sqrt(128)``.

    Every one of the seven stages and the single final scale contribute to every
    element, so a wrong stage order, a wrong stride or a missing scale all show up
    here. fp32 input, because the reading is about the transform and not about bf16
    rounding.
    """
    reset_kpool_hadamard_dispatch_counters()
    got = dsa_hadamard128(torch.eye(INDEX_HEAD_DIM, dtype=torch.float32))
    nki, fallback = kpool_hadamard_dispatch_counters()

    assert (nki, fallback) == (1, 0)
    torch.testing.assert_close(
        got.float(),
        hadamard_matrix(INDEX_HEAD_DIM) * HADAMARD_SCALE,
        rtol=0.0,
        atol=ATOL,
    )


def test_the_transform_applied_twice_returns_the_input() -> None:
    """``h(h(x)) == x``, with no division by 128.

    ``H H == 128 I`` and the transform already carries ``1/sqrt(128)``, so
    ``(x H / sqrt(128)) H / sqrt(128) == x``; dividing by 128 a second time would give
    ``x / 128``. fp32 because bf16 carries about three decimal digits and no bf16
    reading could clear a 1e-5 bound. 37 rows reuses a row count above, and ``n_rows``
    is a compile-time constant in the kernel, so no extra compile is paid.
    """
    x = _rows(37, torch.float32, seed=104)
    reset_kpool_hadamard_dispatch_counters()
    twice = dsa_hadamard128(dsa_hadamard128(x))
    nki, fallback = kpool_hadamard_dispatch_counters()

    # Two dispatches and not one: the second application went through the seam as well,
    # rather than the first result being compared against itself.
    assert (nki, fallback) == (2, 0)
    assert tuple(twice.shape) == tuple(x.shape)
    torch.testing.assert_close(twice.float(), x.float(), rtol=0.0, atol=ATOL)


def test_unadmitted_fused_dtype_is_refused_by_the_gate_and_served_by_torch() -> None:
    """An fp32 ``slot_k`` is outside the supported dtypes: the torch path runs, and is correct."""
    slot_k, slot_score, ape = _inputs(8, seed=202)
    slot_k = slot_k.float()
    expected = _unfused_reference(slot_k, slot_score, ape)
    reset_kpool_hadamard_dispatch_counters()
    gate = can_run_dsa_kpool_hadamard(slot_k, slot_score, ape)
    got = dsa_kpool_hadamard(slot_k, slot_score, ape)
    nki, fallback = kpool_hadamard_dispatch_counters()

    assert gate is False
    assert (nki, fallback) == (0, 1)
    assert got.dtype == torch.float32
    torch.testing.assert_close(got.float(), expected.float(), rtol=RTOL, atol=ATOL)


def test_kernel_identity_is_none_before_any_dispatch() -> None:
    """With no dispatch behind it the identity is ``None``, which is what separates "none ran"."""
    reset_kpool_hadamard_dispatch_counters()
    assert kpool_hadamard_kernel_identity() is None


def test_kernel_identity_names_the_fused_kernel_after_a_fused_dispatch() -> None:
    """After a fused call the identity is this module's fused kernel, read through the seam."""
    slot_k, slot_score, ape = _inputs(33, seed=203)
    reset_kpool_hadamard_dispatch_counters()
    dsa_kpool_hadamard(slot_k, slot_score, ape)
    module, qualname = kpool_hadamard_kernel_identity()
    assert module.endswith("kpool_hadamard")
    assert qualname == "_kpool_hadamard_nki"


def test_kernel_identity_names_the_rotation_kernel_after_a_rotation_dispatch() -> None:
    """After a rotation-only call the identity names the other kernel, not the fused one."""
    reset_kpool_hadamard_dispatch_counters()
    dsa_hadamard128(torch.eye(INDEX_HEAD_DIM, dtype=torch.float32))
    module, qualname = kpool_hadamard_kernel_identity()
    assert module.endswith("kpool_hadamard")
    assert qualname == "_hadamard128_nki"


def test_hadamard_scale_is_the_correctly_rounded_one_over_sqrt_128() -> None:
    """The shipped literal is ``128 ** -0.5``, bit for bit.

    ``128 ** -0.5`` and ``1.0 / math.sqrt(128)`` are not the same double: they differ by
    exactly one ULP, because the division form rounds down. The literal is the first of
    those, which ``math.sqrt(1 / 128)`` agrees with. The 1.4e-17 difference cannot reach
    any tolerance in this file, but asserting the other spelling would fail a correct
    kernel, so which one is shipped is written down here.
    """
    assert HADAMARD_SCALE == INDEX_HEAD_DIM**-0.5
    assert HADAMARD_SCALE == math.sqrt(1.0 / INDEX_HEAD_DIM)
    assert HADAMARD_SCALE != 1.0 / math.sqrt(INDEX_HEAD_DIM)
    assert abs(HADAMARD_SCALE - 1.0 / math.sqrt(INDEX_HEAD_DIM)) < 1e-16


def test_hadamard_matrix_is_orthogonal_after_scaling() -> None:
    """``(H/sqrt(n)) @ (H/sqrt(n)).T == I``, so the matrix really is a Hadamard matrix."""
    h = hadamard_matrix(INDEX_HEAD_DIM) * HADAMARD_SCALE
    torch.testing.assert_close(
        h @ h.t(), torch.eye(INDEX_HEAD_DIM), rtol=0.0, atol=ATOL
    )


def test_stage_sequence_covers_every_channel_exactly_once() -> None:
    """The seven ``(groups, stride)`` pairs, asserted as a sequence so a reordering fails here."""
    assert HADAMARD_STAGES == ((64, 1), (32, 2), (16, 4), (8, 8), (4, 16), (2, 32), (1, 64))
    for groups, stride in HADAMARD_STAGES:
        assert groups * 2 * stride == INDEX_HEAD_DIM


def test_mismatched_score_shape_is_refused() -> None:
    """A ``slot_score`` that does not match ``slot_k`` raises instead of broadcasting."""
    slot_k, slot_score, ape = _inputs(8, seed=205)
    with pytest.raises(KpoolHadamardError):
        dsa_kpool_hadamard(slot_k, slot_score[:, :2, :], ape)


def test_wrong_head_dim_is_refused() -> None:
    """A head dimension other than 128 raises: the Hadamard path is a 128-point transform."""
    with pytest.raises(KpoolHadamardError):
        dsa_hadamard128(torch.zeros((4, 64), dtype=torch.bfloat16))


# ---------------------------------------------------------------------------------------------
# Tiling branches
#
# Both kernels split their rows (pools) over the programs of the launch and tile each program's
# share in pieces whose sizes come from the module's tiling constants. The cases below derive
# their row and pool counts from those constants, so every branch of the tiling is reached at
# grid 1 and under LNC2 (``NEURON_LOGICAL_NC_CONFIG=2``, two programs) whatever the constants are.
# ---------------------------------------------------------------------------------------------

PARTITIONS = 128
"""SBUF partitions, the row count of one tile on every NeuronCore generation."""

LNC2 = "2"


def _rows_reaching_every_rotation_branch(programs: int) -> int:
    """Per program: one whole chunk, a short chunk of three tiles, and a 5-row partial tile.

    The short chunk has an odd tile count, so its last PSUM group is narrower than
    ``_GROUP_TILES`` whenever that constant is above three.
    """
    per_program = PARTITIONS * kpool_hadamard._CHUNK_TILES + PARTITIONS * 3 + 5
    return programs * per_program


def _pools_reaching_every_pool_branch(programs: int) -> int:
    """Per program: one whole tile of ``_POOL_TILE_COLUMNS`` pools per partition, a narrower
    tile of one pool per partition, and a 3-partition tail."""
    per_program = PARTITIONS * (kpool_hadamard._POOL_TILE_COLUMNS + 1) + 3
    return programs * per_program


def test_launch_uses_both_cores_under_lnc2(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two programs under LNC2 once there are two rows to split, one otherwise."""
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", LNC2)
    assert kpool_hadamard._programs(2) == 2
    assert kpool_hadamard._programs(1) == 1
    monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG")
    assert kpool_hadamard._programs(2) == 1


@pytest.mark.parametrize("lnc", [None, LNC2], ids=["grid1", "lnc2"])
def test_rotation_reaches_every_tiling_branch(monkeypatch: pytest.MonkeyPatch, lnc) -> None:
    """Whole chunk, short chunk, short PSUM group and partial tile, on one core and on two."""
    if lnc is None:
        monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    else:
        monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", lnc)
    programs = kpool_hadamard._programs(2)
    _assert_rotation_matches_reference(
        _rows_reaching_every_rotation_branch(programs), torch.bfloat16, seed=301
    )


@pytest.mark.parametrize("n_rows", [1, 3, 32, 128, 512, 2048])
def test_rotation_at_decode_row_counts_under_lnc2(
    monkeypatch: pytest.MonkeyPatch, n_rows: int
) -> None:
    """The served decode calls rotate ``batch * index_n_heads`` rows; 1 and 3 split unevenly."""
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", LNC2)
    _assert_rotation_matches_reference(n_rows, torch.bfloat16, seed=400 + n_rows)


def test_rotation_of_fp32_rows_under_lnc2_is_an_involution(monkeypatch: pytest.MonkeyPatch) -> None:
    """fp32 rows take the same path under LNC2 and still satisfy ``h(h(x)) == x`` to 1e-5."""
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", LNC2)
    x = _rows(_rows_reaching_every_rotation_branch(2), torch.float32, seed=302)
    torch.testing.assert_close(dsa_hadamard128(dsa_hadamard128(x)), x, rtol=0.0, atol=ATOL)


@pytest.mark.parametrize("lnc", [None, LNC2], ids=["grid1", "lnc2"])
@pytest.mark.parametrize("score_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_pooling_reaches_every_tiling_branch(
    monkeypatch: pytest.MonkeyPatch, lnc, score_dtype: torch.dtype
) -> None:
    """Whole pool tile, narrow tile and partition tail, with the served bf16 score and fp32."""
    if lnc is None:
        monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    else:
        monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", lnc)
    programs = kpool_hadamard._programs(2)
    _assert_fused_matches_unfused(
        _pools_reaching_every_pool_branch(programs), seed=303, score_dtype=score_dtype
    )


@pytest.mark.parametrize("pool_size", [1, 3])
def test_pooling_with_other_pool_sizes_under_lnc2(
    monkeypatch: pytest.MonkeyPatch, pool_size: int
) -> None:
    """One slot (no max or sum chain at all) and an odd slot count."""
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", LNC2)
    _assert_fused_matches_unfused(260, seed=304 + pool_size, pool_size=pool_size)
