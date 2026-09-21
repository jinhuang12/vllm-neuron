# SPDX-License-Identifier: Apache-2.0
"""Tests for the mHC Sinkhorn normalisation kernels, square and batched.

Every numeric case is read two ways.

The first compares the kernel against a torch Sinkhorn reference authored here,
which is written as the classical Sinkhorn-Knopp form -- accumulate per-axis
scaling vectors and apply them to the original matrix -- rather than the module
oracle's in-place rescaling, so the two are independent statements of the same
algorithm.

The second is the stronger one, because it depends on no reference at all: every
row sum must be within ``1e-3`` of ``row_target()`` and every column sum within
``1e-3`` of ``column_target(M, N)``. Those two numbers come from the algorithm's
definition. The column target is what makes the reading real across row tiles: a
kernel that scaled each tile by its own column sums would leave ``tile_rows / N``
in each column instead of ``M / N``, and the reference comparison would nearly
pass.

Comparing simulated output against a torch reference would measure nothing if the
module took its torch path, because both sides would then be torch. So the numeric
cases also read the module's dispatch counters and count real
``nki.simulator.simulate_kernel`` calls, which is the vendor entry point and so
independent of the module's own counter. One dispatch per call is the reading that
matters: the iterations and the row tiling both live inside the kernel, so a
host-driven loop would read one per iteration or one per tile.

The batched form normalises one square block per token instead of the
``block_diag`` matrix of them. Its equivalence to the square form is checked block
for block at small token counts, because a block-diagonal matrix's row and column
sums are its blocks' own sums and a zero stays zero under any rescaling. At the
serving token counts that comparison does not exist -- the square form refuses
``T = 129``, where ``N = T * S`` passes the Tensor Engine's moving free bound, and
``T = 2048`` would need a 256 MB matrix -- which is the reason the batched form
exists.
"""

from __future__ import annotations

import ast
import importlib
import os
import pathlib

import pytest
import torch

import nki.simulator

from vllm_neuron.functional.mhc.sinkhorn import (
    MHC_STREAMS,
    MOVING_FMAX,
    PARTITION_MAX,
    SINKHORN_DENOM_EPS,
    SINKHORN_ITERS,
    SinkhornError,
    blocks_kernel_identity,
    can_run_sinkhorn,
    can_run_sinkhorn_blocks,
    column_target,
    dispatch_counters,
    kernel_identity,
    reset_dispatch_counters,
    row_target,
    row_tile_extent,
    row_tiles,
    sinkhorn_normalise,
    sinkhorn_normalise_blocks,
    sinkhorn_torch_oracle,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

#: The tiny case. ``N`` is read off ``MHC_STREAMS`` so the fixture and the model's
#: ``hc_mult`` cannot drift apart.
M = 64
N = MHC_STREAMS  # 4

#: Row extents past one partition tile. ``129`` is the first that neither fits one
#: tile nor is a whole number of blocks -- 32 blocks and one row -- so a ragged last
#: tile is covered. The two largest are written as ``tokens * MHC_STREAMS`` because
#: that is what they are.
TILED_ROWS = (129, 256, 512 * MHC_STREAMS, 2048 * MHC_STREAMS)

#: Token counts for the batched form's equivalence cases. Small, because each also
#: runs the square kernel on ``block_diag`` of the same blocks, which costs
#: ``(T*S)^2``. ``33`` is the first that makes the square side tile as well
#: (``33 * 4 = 132`` rows), so the equivalence crosses a tile seam too.
BLOCK_EQUIV_TOKENS = (1, 3, 33)

#: Token counts where no square comparison exists: ``129`` is the first ``T`` the
#: square form refuses, and ``2048`` is the serving extent this form exists for.
BLOCK_SCALE_TOKENS = (129, 2048)

RTOL = 1e-2
ATOL = 1e-5

#: The bound on every row and column sum's distance from its target.
STOCHASTIC_TOL = 1e-3

_MODULE = "vllm_neuron.functional.mhc.sinkhorn"


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


def _assert_nki_ran(sim: _SimulatorCounter, expected: int, label: str) -> None:
    """Require that the NKI path, and only the NKI path, served the calls."""
    nki_dispatch, torch_fallback = dispatch_counters()
    assert nki_dispatch == expected, (
        f"{label}: the dispatch counter read {nki_dispatch}, expected {expected}; "
        f"one per call, not one per iteration and not one per tile"
    )
    assert torch_fallback == 0, (
        f"{label}: the torch-fallback counter read {torch_fallback}; a fallback "
        f"would compare torch against torch"
    )
    assert can_run_kernel(torch.zeros(1)) is True, f"{label}: no device or simulator"
    assert sim.calls == expected, (
        f"{label}: nki.simulator.simulate_kernel ran {sim.calls} times, expected "
        f"{expected}; a numeric pass without a simulator call measures no kernel"
    )


def _affinity_rows(rows: int, seed: int = 21) -> torch.Tensor:
    """A strictly positive ``[rows, N]`` affinity matrix, fp32.

    ``exp`` of a bounded uniform draw, which is what an affinity matrix is in the
    model -- the exponential of a score -- and which gives strict positivity
    without a clamp, so :data:`SINKHORN_DENOM_EPS` is never what keeps the
    arithmetic finite. The ``[-1, 1]`` exponent range holds the dynamic range at
    ``e**2``, so no term dominates its row and a relative tolerance over the
    reduction measures the kernel rather than cancellation.

    One construction for every row extent, so a tiled case cannot pass on an easier
    input than the single-tile case.
    """
    generator = torch.Generator().manual_seed(seed)
    logits = (
        torch.rand((rows, N), generator=generator, dtype=torch.float32) * 2.0 - 1.0
    )
    return torch.exp(logits)


def _affinity(seed: int = 21) -> torch.Tensor:
    """The tiny case's ``[64, 4]`` affinity matrix."""
    return _affinity_rows(M, seed)


def _affinity_blocks(tokens: int, seed: int = 21) -> torch.Tensor:
    """``[T, S, S]`` affinities, one square block per token.

    Built by reshaping :func:`_affinity_rows`, so the batched fixture is the same
    draw as every other case here rather than a second construction that could be
    easier.
    """
    return _affinity_rows(tokens * N, seed).reshape(tokens, N, N)


def _reference(affinity: torch.Tensor, iters: int = SINKHORN_ITERS) -> torch.Tensor:
    """Sinkhorn-Knopp in float64: accumulate ``u`` and ``v``, apply to the original.

    A different formulation from the module's :func:`sinkhorn_torch_oracle`, which
    rescales a working matrix in place, so comparing the two says something. The
    ``eps`` guard is applied to both denominators exactly as the kernel and the
    module oracle do, so it cannot manufacture a disagreement either way.
    """
    rows, cols = int(affinity.shape[0]), int(affinity.shape[1])
    base = affinity.to(torch.float64)
    u = torch.ones((rows, 1), dtype=torch.float64)
    v = torch.ones((1, cols), dtype=torch.float64)
    row_goal = row_target()
    col_goal = column_target(rows, cols)

    for _ in range(iters):
        scaled = base * u * v
        u = u * (row_goal / (scaled.sum(dim=1, keepdim=True) + SINKHORN_DENOM_EPS))
        scaled = base * u * v
        v = v * (col_goal / (scaled.sum(dim=0, keepdim=True) + SINKHORN_DENOM_EPS))

    return (base * u * v).to(torch.float32)


def _blocks_reference(
    blocks: torch.Tensor, iters: int = SINKHORN_ITERS
) -> torch.Tensor:
    """:func:`_reference` with one extra leading axis, reducing over axes 2 and 1."""
    base = blocks.to(torch.float64)
    tokens, rows, cols = (int(v) for v in base.shape)
    u = torch.ones((tokens, rows, 1), dtype=torch.float64)
    v = torch.ones((tokens, 1, cols), dtype=torch.float64)
    row_goal = row_target()
    col_goal = column_target(rows, cols)

    for _ in range(iters):
        scaled = base * u * v
        u = u * (row_goal / (scaled.sum(dim=2, keepdim=True) + SINKHORN_DENOM_EPS))
        scaled = base * u * v
        v = v * (col_goal / (scaled.sum(dim=1, keepdim=True) + SINKHORN_DENOM_EPS))

    return (base * u * v).to(torch.float32)


def _deviations(result: torch.Tensor) -> tuple[float, float]:
    """``(worst_row_deviation, worst_column_deviation)`` for a matrix.

    Each axis against its own target: rows against :func:`row_target`, columns
    against :func:`column_target`.
    """
    rows, cols = int(result.shape[0]), int(result.shape[1])
    row_dev = float((result.sum(dim=1) - row_target()).abs().max())
    col_dev = float((result.sum(dim=0) - column_target(rows, cols)).abs().max())
    return row_dev, col_dev


def _block_deviations(result: torch.Tensor) -> tuple[float, float]:
    """The same two deviations over all ``T`` blocks, worst case rather than mean.

    Worst case so one bad block cannot hide behind the others.
    """
    _tokens, rows, cols = (int(v) for v in result.shape)
    row_dev = float((result.sum(dim=2) - row_target()).abs().max())
    col_dev = float((result.sum(dim=1) - column_target(rows, cols)).abs().max())
    return row_dev, col_dev


def _assert_doubly_stochastic(result: torch.Tensor, label: str, *, blocks=False):
    """Every row and column sum within :data:`STOCHASTIC_TOL` of its target."""
    assert torch.isfinite(result).all(), f"{label}: the kernel returned non-finite values"
    row_dev, col_dev = (
        _block_deviations(result) if blocks else _deviations(result)
    )
    assert row_dev <= STOCHASTIC_TOL, (
        f"{label}: worst row deviation {row_dev:.6e} exceeds {STOCHASTIC_TOL} "
        f"against row target {row_target()}"
    )
    assert col_dev <= STOCHASTIC_TOL, (
        f"{label}: worst column deviation {col_dev:.6e} exceeds {STOCHASTIC_TOL} "
        f"against the column target"
    )


def _assert_input_is_not_already_normalised(affinity, label: str, *, blocks=False):
    """The input must fail the bar its output passes.

    Without this, the doubly-stochastic reading would be satisfied by a kernel that
    returned its input untouched.
    """
    row_dev, col_dev = (
        _block_deviations(affinity) if blocks else _deviations(affinity)
    )
    assert row_dev > STOCHASTIC_TOL or col_dev > STOCHASTIC_TOL, (
        f"{label}: the fixture already satisfies the doubly-stochastic bar, so a "
        f"kernel that did nothing would pass"
    )


def test_output_matches_the_torch_reference_after_the_declared_iterations() -> None:
    """The ``[64, 4]`` case agrees with the torch reference at the declared count."""
    affinity = _affinity()
    assert tuple(affinity.shape) == (M, N), tuple(affinity.shape)
    assert SINKHORN_ITERS == 20, SINKHORN_ITERS

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        got = sinkhorn_normalise(affinity)
    _assert_nki_ran(sim, 1, "reference-case")

    expected = _reference(affinity)
    assert float(expected.abs().max()) > 0.0, "the reference is all zero"
    torch.testing.assert_close(
        got.to(torch.float32), expected, rtol=RTOL, atol=ATOL
    )


def test_the_output_is_doubly_stochastic() -> None:
    """Every row sum and column sum lands within 1e-3 of its target.

    This consults no reference: the two targets come from the algorithm's
    definition. The input is shown to fail the same bar, so a kernel returning its
    argument would not pass.
    """
    affinity = _affinity()
    _assert_input_is_not_already_normalised(affinity, "the [64, 4] fixture")
    assert float(affinity.min()) > 0.0, "the fixture is not strictly positive"

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        got = sinkhorn_normalise(affinity).to(torch.float32)
    _assert_nki_ran(sim, 1, "stochastic-case")

    _assert_doubly_stochastic(got, "the [64, 4] case")


def test_convergence_needs_the_declared_iteration_count() -> None:
    """One iteration misses the bar; the declared count reaches it.

    A single iteration ends on a column pass, so its columns are exact and its rows
    are not yet converged. That makes it the sharpest probe that the bar can fail at
    all, and it is why the iteration count is not arbitrary.
    """
    affinity = _affinity()

    one_pass = sinkhorn_normalise(affinity, iters=1).to(torch.float32)
    row_dev, _ = _deviations(one_pass)
    assert row_dev > STOCHASTIC_TOL, (
        f"a single iteration already lands inside {STOCHASTIC_TOL} (worst row "
        f"deviation {row_dev:.6e}), so the bar cannot tell a converged result from "
        f"a truncated one"
    )

    converged = sinkhorn_normalise(affinity, iters=SINKHORN_ITERS).to(torch.float32)
    _assert_doubly_stochastic(converged, f"iters={SINKHORN_ITERS}")


def test_the_two_torch_formulations_agree_and_both_reach_the_bar() -> None:
    """The module's oracle and the reference here agree, and both converge.

    Agreement between two different formulations is evidence about the reference;
    agreement between one formulation and itself would be nothing. Both are also
    required to reach the bar independently, or the agreement would only mean they
    are wrong together.
    """
    affinity = _affinity()
    module_side = sinkhorn_torch_oracle(affinity)
    local_side = _reference(affinity)
    torch.testing.assert_close(module_side, local_side, rtol=RTOL, atol=ATOL)

    _assert_doubly_stochastic(module_side, "the module oracle")
    _assert_doubly_stochastic(local_side, "the reference here")


def test_the_row_and_column_targets_describe_one_total_mass() -> None:
    """``M`` rows at the row target and ``N`` columns at the column target agree.

    If they did not, no matrix could satisfy both and the doubly-stochastic reading
    would be unsatisfiable by construction rather than by defect.
    """
    assert M * row_target() == N * column_target(M, N) == float(M)
    assert column_target(64, 4) == 16.0


def test_the_torch_fallback_is_taken_and_counted_without_a_simulator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no device or simulator the square form takes its torch path, counted.

    This is what makes ``torch_fallback == 0`` in the numeric cases meaningful.
    """
    affinity = _affinity()
    monkeypatch.setitem(os.environ, "NKI_SIMULATOR", "0")
    assert can_run_kernel(torch.zeros(1)) is False, (
        "the gate did not flip with NKI_SIMULATOR=0, so this case is vacuous"
    )

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        out = sinkhorn_normalise(affinity)

    assert dispatch_counters() == (0, 1), (
        f"expected 0 NKI dispatches and 1 torch fallback, got {dispatch_counters()}"
    )
    assert sim.calls == 0, f"the simulator ran {sim.calls} times with it disabled"
    assert tuple(out.shape) == (M, N)


def test_dispatch_counters_accumulate_across_calls() -> None:
    """The counters are module-level state: they accumulate until reset.

    The layer forward reads a per-layer-call total, which a counter saturating at
    one could not supply.
    """
    affinity = _affinity()
    reset_dispatch_counters()
    assert dispatch_counters() == (0, 0)

    sinkhorn_normalise(affinity)
    assert dispatch_counters() == (1, 0)

    sinkhorn_normalise(affinity)
    assert dispatch_counters() == (2, 0), (
        f"expected (2, 0) after two dispatches, got {dispatch_counters()}"
    )

    reset_dispatch_counters()
    assert dispatch_counters() == (0, 0)


def test_kernel_identity_names_this_modules_kernel() -> None:
    """``kernel_identity`` names the square kernel in this module."""
    assert kernel_identity() == (_MODULE, "sinkhorn_kernel")


@pytest.mark.parametrize("rows", TILED_ROWS)
def test_row_extents_past_one_tile_converge_and_match_the_reference(
    rows: int,
) -> None:
    """Both readings at a row extent that needs more than one tile, one dispatch.

    The doubly-stochastic reading comes first, because it does not depend on the
    reference being right.
    """
    tiles = row_tiles(rows, MHC_STREAMS)
    assert len(tiles) > 1, f"M={rows} did not tile"
    affinity = _affinity_rows(rows)
    assert tuple(affinity.shape) == (rows, N), tuple(affinity.shape)

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        got = sinkhorn_normalise(affinity).to(torch.float32)
    _assert_nki_ran(sim, 1, f"M={rows} tiles={len(tiles)}")

    _assert_doubly_stochastic(got, f"M={rows}")
    _assert_input_is_not_already_normalised(affinity, f"M={rows}")

    torch.testing.assert_close(got, _reference(affinity), rtol=RTOL, atol=ATOL)


def test_row_tiles_are_cut_on_block_boundaries() -> None:
    """No token's ``S x S`` block is split, at every row extent above.

    This reads the same :func:`row_tiles` arithmetic the kernel runs. Four things
    per extent: every tile starts on a multiple of the block, no tile is taller than
    one partition tile, the heights sum to ``M`` so no row is dropped or served
    twice, and the extent genuinely tiles.
    """
    extent = row_tile_extent(MHC_STREAMS)
    assert extent % MHC_STREAMS == 0, extent
    assert 0 < extent <= PARTITION_MAX, extent

    for rows in TILED_ROWS:
        tiles = row_tiles(rows, MHC_STREAMS)
        starts = [start for start, _ in tiles]
        heights = [height for _, height in tiles]
        assert len(tiles) > 1, f"M={rows} did not tile"
        assert sum(heights) == rows, (rows, sum(heights))
        assert all(start % MHC_STREAMS == 0 for start in starts), starts
        assert all(0 < height <= PARTITION_MAX for height in heights), heights


def test_the_kernels_tile_count_equals_the_tile_lists_length() -> None:
    """The loop bound the kernels count with equals the tile list's own length.

    The kernels cannot walk the tile list, because the NKI tracer refuses a ``for``
    whose loop variable is a tuple, so they count with ``range(tile_count)`` and
    ``tile_count`` comes from arithmetic rather than from ``len``. Arithmetic that
    disagreed by one would silently drop the last token tile or read past the end.
    """
    module = importlib.import_module(_MODULE)
    tile_list = module._row_tiles_unchecked
    tile_count = module._row_tile_count_unchecked
    extent = module._row_tile_extent_unchecked

    blocks = [b for b in range(1, PARTITION_MAX + 1) if extent(b) >= 1]
    rows_grid = list(range(1, 301)) + list(TILED_ROWS) + [PARTITION_MAX + 1, 16384]
    assert blocks and rows_grid
    for block in blocks:
        for rows in rows_grid:
            assert tile_count(rows, block) == len(tile_list(rows, block)), (block, rows)


def test_large_row_extents_are_admissible() -> None:
    """The geometry check raises for no row extent the tiling serves."""
    for rows in (PARTITION_MAX + 1,) + TILED_ROWS:
        assert isinstance(can_run_sinkhorn(torch.zeros(1), rows, N), bool)


def test_a_block_taller_than_one_tile_is_refused_by_name() -> None:
    """A stream count no tile height can align to is refused rather than split.

    It cannot arise at ``hc_mult = 4``; the refusal is what keeps the
    block-alignment property true rather than merely intended.
    """
    with pytest.raises(SinkhornError) as excinfo:
        can_run_sinkhorn(torch.zeros(1), 512, N, block=PARTITION_MAX + 1)
    message = str(excinfo.value)
    assert f"block={PARTITION_MAX + 1}" in message, message
    assert f"PARTITION_MAX={PARTITION_MAX}" in message, message

    with pytest.raises(SinkhornError) as excinfo:
        row_tile_extent(0)
    assert "block=0 must be positive" in str(excinfo.value)


@pytest.mark.parametrize(
    ("rows", "cols", "needle"),
    [
        (0, 4, "M=0 must be positive"),
        (64, 0, "N=0 must be positive"),
        (64, 513, "exceeds the Tensor Engine moving free bound"),
    ],
)
def test_refuses_inadmissible_geometry_by_name(
    rows: int, cols: int, needle: str
) -> None:
    """An extent this kernel cannot serve raises and names it, rather than coercing."""
    with pytest.raises(SinkhornError) as excinfo:
        can_run_sinkhorn(torch.zeros(1), rows, cols)
    assert needle in str(excinfo.value), f"[M={rows},N={cols}]: {excinfo.value}"


def test_refuses_non_2d_input_and_non_positive_iters() -> None:
    """The square form's own argument refusals, named rather than coerced."""
    with pytest.raises(SinkhornError) as excinfo:
        sinkhorn_normalise(torch.ones((2, 3, 4), dtype=torch.float32))
    assert "must be 2-D" in str(excinfo.value)

    for bad in (0, -1):
        with pytest.raises(SinkhornError) as excinfo:
            sinkhorn_normalise(_affinity(), iters=bad)
        assert f"iters={bad} must be positive" in str(excinfo.value)


@pytest.mark.parametrize("tokens", BLOCK_EQUIV_TOKENS)
def test_the_batched_form_equals_the_square_form_on_the_block_diagonal(
    tokens: int,
) -> None:
    """Normalising ``T`` blocks independently equals normalising ``block_diag`` of them.

    Both kernels run on the same blocks, which is why the route reading is two
    dispatches rather than one. The square result's off-diagonal is required to be
    exactly zero first, because that is the premise of the equivalence: if the
    square kernel leaked mass off the blocks, the premise and not the batched kernel
    would be what failed.
    """
    blocks = _affinity_blocks(tokens)
    matrix = torch.block_diag(*blocks.unbind(0))
    assert tuple(blocks.shape) == (tokens, N, N), tuple(blocks.shape)
    assert tuple(matrix.shape) == (tokens * N, tokens * N), tuple(matrix.shape)

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        got = sinkhorn_normalise_blocks(blocks).to(torch.float32)
        square = sinkhorn_normalise(matrix).to(torch.float32)
    _assert_nki_ran(sim, 2, f"T={tokens}")

    mask = torch.ones_like(square, dtype=torch.bool)
    for t in range(tokens):
        mask[t * N : (t + 1) * N, t * N : (t + 1) * N] = False
    off_diagonal_max = float(square[mask].abs().max()) if bool(mask.any()) else 0.0
    assert off_diagonal_max == 0.0, off_diagonal_max

    expected = torch.stack(
        [square[t * N : (t + 1) * N, t * N : (t + 1) * N] for t in range(tokens)]
    )
    assert float((expected - blocks).abs().max()) > ATOL, (
        f"T={tokens}: the normalised blocks are within atol of the raw input, so "
        f"this comparison would pass on a kernel that returned its input"
    )
    torch.testing.assert_close(got, expected, rtol=RTOL, atol=ATOL)

    _assert_doubly_stochastic(got, f"T={tokens}", blocks=True)


@pytest.mark.parametrize("tokens", BLOCK_SCALE_TOKENS)
def test_the_batched_form_converges_at_the_serving_token_counts(
    tokens: int,
) -> None:
    """The extents no square comparison can reach, against the batched reference.

    One dispatch each, however many token tiles the extent needs. The
    doubly-stochastic reading comes first, because it does not depend on the
    reference being right.
    """
    blocks = _affinity_blocks(tokens)

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        got = sinkhorn_normalise_blocks(blocks).to(torch.float32)
    _assert_nki_ran(sim, 1, f"T={tokens}")
    assert tuple(got.shape) == (tokens, N, N), tuple(got.shape)

    _assert_doubly_stochastic(got, f"T={tokens}", blocks=True)
    _assert_input_is_not_already_normalised(blocks, f"T={tokens}", blocks=True)

    torch.testing.assert_close(got, _blocks_reference(blocks), rtol=RTOL, atol=ATOL)


def test_the_square_form_refuses_the_token_counts_the_batched_form_serves() -> None:
    """``N = T * S`` past the moving free bound is refused, and the batched form is not.

    This is why the batched form exists: the square matrix rides the Tensor Engine's
    moving free axis, so ``T`` above ``MOVING_FMAX // S`` cannot be served at all.
    """
    for tokens in BLOCK_SCALE_TOKENS:
        square_side = tokens * N
        with pytest.raises(SinkhornError) as excinfo:
            can_run_sinkhorn(torch.zeros(1), square_side, square_side)
        message = str(excinfo.value)
        assert f"N={square_side}" in message, message
        assert f"moving free bound {MOVING_FMAX}" in message, message

        assert isinstance(
            can_run_sinkhorn_blocks(torch.zeros(1), tokens, N, N), bool
        )


def test_the_batched_reference_agrees_with_the_per_block_one() -> None:
    """Batching the reference over the leading axis did not change the answer.

    This is what lets the serving-scale cases rest on the batched reference, where a
    per-block python loop would dominate their runtime. The tolerance is tighter
    than a kernel comparison's on purpose: both sides are float64 torch running one
    formulation, so only reduction order can differ.
    """
    blocks = _affinity_blocks(3)
    batched = _blocks_reference(blocks)
    per_block = torch.stack([_reference(block) for block in blocks.unbind(0)])
    torch.testing.assert_close(batched, per_block, rtol=1e-6, atol=1e-7)


def test_the_two_kernel_identities_are_distinct() -> None:
    """Each entry point names its own kernel, so a reading says which one ran.

    For ``T <= 128`` a seam quietly wired to the other kernel would return
    plausible numbers, which is why the two are told apart by name.
    """
    assert blocks_kernel_identity() == (_MODULE, "sinkhorn_blocks_kernel")
    assert blocks_kernel_identity() != kernel_identity()


def test_the_explicit_vector_engine_is_only_the_single_token_branch() -> None:
    """``block_scalar_engine`` picks the Vector engine at ``T == 1`` and defaults above.

    Larger token batches must keep the SDK's own engine choice, so prefill and
    boundary shapes do not inherit a decode-only decision. Read off the source
    because simulator numerics cannot observe engine placement.
    """
    module = importlib.import_module(_MODULE)
    tree = ast.parse(pathlib.Path(module.__file__).read_text())
    [fn] = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "sinkhorn_blocks_kernel"
    ]
    assignments = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "block_scalar_engine"
            for target in node.targets
        )
    ]
    assert len(assignments) == 1
    value = assignments[0].value
    assert isinstance(value, ast.IfExp)
    assert ast.unparse(value.test) == "int(t_extent) == 1"
    assert ast.unparse(value.body) == "nisa.vector_engine"
    assert ast.unparse(value.orelse) == "nisa.unknown_engine"

    engine_sites = []
    for node in ast.walk(fn):
        if not (
            isinstance(node, ast.Call)
            and ast.unparse(node.func) == "nisa.tensor_scalar"
        ):
            continue
        for keyword in node.keywords:
            if keyword.arg == "engine":
                engine_sites.append(ast.unparse(keyword.value))
    assert engine_sites == ["block_scalar_engine"] * 5


@pytest.mark.parametrize(
    ("tokens", "streams", "iters"),
    [
        (1, MHC_STREAMS, SINKHORN_ITERS),
        (1, 2, 1),
        (2, MHC_STREAMS, SINKHORN_ITERS),
        (127, 2, 1),
        (128, 2, 1),
        (129, 2, 1),
        (1024, 2, 1),
    ],
)
def test_the_batched_form_matches_the_reference_at_the_token_tile_boundaries(
    tokens: int, streams: int, iters: int
) -> None:
    """The single-token branch and the tile boundaries around 128 all agree.

    ``T == 1`` takes the explicit-engine branch; the others take the default one, so
    the set covers both sides of that choice as well as the tile seam.
    """
    rng = torch.Generator().manual_seed(530919 + tokens * 17 + streams * 31 + iters)
    logits = torch.randn(tokens, streams, streams, generator=rng)
    blocks = torch.softmax(logits, dim=-1) + SINKHORN_DENOM_EPS

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        got = sinkhorn_normalise_blocks(blocks, iters=iters).to(torch.float32)
    _assert_nki_ran(sim, 1, f"T={tokens} S={streams} iters={iters}")

    torch.testing.assert_close(
        got, _blocks_reference(blocks, iters=iters), rtol=RTOL, atol=ATOL
    )


def test_the_batched_form_has_no_torch_path_at_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the route unavailable the batched form raises; it never computes torch.

    The opposite outcome from the square form on purpose: this one ships no torch
    path, so it has none to count and both counters must stay at zero. A layer that
    silently normalised 2048 tokens in torch would be slow in a way no numeric
    comparison could see.
    """
    blocks = _affinity_blocks(2)
    monkeypatch.setitem(os.environ, "NKI_SIMULATOR", "0")
    assert can_run_kernel(torch.zeros(1)) is False, (
        "the gate did not flip with NKI_SIMULATOR=0, so this case is vacuous"
    )

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        with pytest.raises(SinkhornError) as excinfo:
            sinkhorn_normalise_blocks(blocks)

    assert "no torch path" in str(excinfo.value), str(excinfo.value)
    assert dispatch_counters() == (0, 0), dispatch_counters()
    assert sim.calls == 0, sim.calls


@pytest.mark.parametrize(
    ("tokens", "rows", "cols", "needle"),
    [
        (0, N, N, "T=0 must be positive"),
        (4, 0, N, "S=0 must be positive"),
        (4, 3, 4, "must be square"),
    ],
)
def test_the_batched_form_refuses_inadmissible_geometry_by_name(
    tokens: int, rows: int, cols: int, needle: str
) -> None:
    """Each batched refusal names the offending extent rather than coercing it."""
    with pytest.raises(SinkhornError) as excinfo:
        can_run_sinkhorn_blocks(torch.zeros(1), tokens, rows, cols)
    assert needle in str(excinfo.value), (
        f"[T={tokens},block={rows}x{cols}]: {excinfo.value}"
    )


def test_the_batched_form_refuses_non_3d_input_and_non_positive_iters() -> None:
    """The batched form's own argument refusals, named rather than coerced."""
    with pytest.raises(SinkhornError) as excinfo:
        sinkhorn_normalise_blocks(_affinity())
    assert "must be 3-D" in str(excinfo.value)

    for bad in (0, -1):
        with pytest.raises(SinkhornError) as excinfo:
            sinkhorn_normalise_blocks(_affinity_blocks(2), iters=bad)
        assert f"iters={bad} must be positive" in str(excinfo.value)


def test_the_batched_form_admits_any_token_count() -> None:
    """No ``T`` is refused for being large, and its tiles cover every token.

    The absence of a ceiling is the property this form exists for, so it is read
    rather than assumed, including one extent an order of magnitude past serving.
    """
    for tokens in BLOCK_SCALE_TOKENS + (MOVING_FMAX // N, 32768):
        assert isinstance(
            can_run_sinkhorn_blocks(torch.zeros(1), tokens, N, N), bool
        )
        tiles = row_tiles(tokens, 1)
        assert sum(height for _start, height in tiles) == tokens


#: Python forms the NKI tracer refuses, or that no kernel in this repository uses.
#: A body the simulator runs happily can still be one the compiler will not
#: specialise, and the tracer follows every plain Python call it can resolve, so the
#: whole traced closure has to stay clean rather than just the decorated function.
_TRACE_HOSTILE_COMPREHENSIONS = (
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)

_TRACE_HOSTILE_CLASSES = (
    "raise",
    "comprehension",
    "min_call",
    "tuple_for_target",
    "container_loop",
    "computed_loop_bound",
)


def _module_functions(source: str) -> dict[str, ast.FunctionDef]:
    """Every function in ``source``, by name."""
    return {
        node.name: node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef)
    }


def _traced_callees(functions: dict[str, ast.FunctionDef], entry: str) -> set[str]:
    """The functions NKI would trace into from ``entry``, transitively."""
    seen: set[str] = set()
    frontier = {entry}
    while frontier:
        name = frontier.pop()
        if name in seen or name not in functions:
            continue
        seen.add(name)
        for node in ast.walk(functions[name]):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in functions
            ):
                frontier.add(node.func.id)
    seen.discard(entry)
    return seen


def _trace_hostile_sites(
    functions: dict[str, ast.FunctionDef], names: set[str]
) -> dict[str, list[str]]:
    """Every refused construct found in ``names``, by class, as ``function:line``.

    One form is deliberately not refused: a subscript of a subscript,
    ``work[idx][i]``. The batched kernel reaches a tensor in a list of lists that
    way and the compiler has never named it, so refusing it would report correct
    work as a defect.
    """
    found: dict[str, list[str]] = {name: [] for name in _TRACE_HOSTILE_CLASSES}
    for name in sorted(names):
        for node in ast.walk(functions[name]):
            where = f"{name}:{getattr(node, 'lineno', 0)}"
            if isinstance(node, ast.Raise):
                found["raise"].append(where)
            elif isinstance(node, _TRACE_HOSTILE_COMPREHENSIONS):
                found["comprehension"].append(where)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "min"
            ):
                found["min_call"].append(where)
            elif isinstance(node, ast.For):
                if not isinstance(node.target, ast.Name):
                    found["tuple_for_target"].append(where)
                if isinstance(node.iter, ast.Call):
                    for arg in node.iter.args:
                        if isinstance(arg, (ast.Call, ast.Attribute, ast.Subscript)):
                            found["computed_loop_bound"].append(where)
                else:
                    found["container_loop"].append(where)
    return found


def test_neither_kernel_traces_into_a_form_the_compiler_refuses() -> None:
    """The traced closure of both kernels is free of every refused construct.

    The entry is scanned as well as its callees: ``raise`` statements live in
    helpers, but a tuple loop variable can sit in the kernel body itself.

    Each class is here because the compiler named it, or because no NKI kernel in
    this repository uses the form: every loop in them is over ``range``,
    ``nl.sequential_range`` or ``nl.affine_range``, and every bound is a plain name,
    a constant or an expression.
    """
    module = importlib.import_module(_MODULE)
    functions = _module_functions(pathlib.Path(module.__file__).read_text())

    for entry in ("sinkhorn_kernel", "sinkhorn_blocks_kernel"):
        assert entry in functions, entry
        callees = _traced_callees(functions, entry)
        assert callees, f"{entry} resolved no traced callees, so this covers nothing"
        found = _trace_hostile_sites(functions, callees | {entry})
        for cls in _TRACE_HOSTILE_CLASSES:
            assert found[cls] == [], (
                f"{entry}'s traced program carries {cls} at {found[cls]}; the "
                f"compiler refuses that form, or no kernel here uses it"
            )
