# SPDX-License-Identifier: Apache-2.0
"""Tests for ``dsa_causal_fill``: exact causal index rows, compared bit for bit.

The fill is integer index arithmetic, so there is no tolerance to spend: every
comparison is ``torch.equal`` on int32. The reference is
``dsa_causal_fill_torch_oracle``, which is upstream's write-then-mask spelling,
while the kernel is a closed form over ``maximum``/``minimum`` that never writes a
value it takes back -- two different mechanisms reaching the same bytes, rather than
one mechanism agreeing with itself.

A query row occupies a hardware partition and the partition axis serves 128 rows, so
the kernel walks the row axis in tiles. The row counts below cover one tile (5, 128),
a ragged last tile (132), whole tiles with no remainder (256) and the per-request
envelope (2048).
"""

import json
from pathlib import Path

import pytest
import torch

from vllm_neuron.functional.dsa import causal_fill as mod
from vllm_neuron.functional.dsa.causal_fill import (
    SENTINEL,
    DsaCausalFillError,
    can_run_dsa_causal_fill,
    causal_fill_dispatch_counters,
    causal_fill_kernel_identity,
    dsa_causal_fill,
    dsa_causal_fill_torch_oracle,
    reset_causal_fill_dispatch_counters,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

ROWS = 5
"""Query rows in the small shape."""

WIDTH = 16
"""Columns in the small shape.

Small on purpose and not a multiple of ``KEY_CHUNK``: the kernel admits any positive
width, and the admissibility ceiling belongs to the caller
(``index_expand.index_expand_width``)."""

POSITIONS = [0, 3, 7, 15, 15]
"""The positions of the small shape, each earning its place.

``0`` is the first decode step, where exactly one column is causal and a bare
``c * keep`` closed form would be indistinguishable from a masked column 0. ``3`` and
``7`` are interior. The two rows at ``15 == WIDTH - 1`` are the saturated case, where
no column is masked and the sentinel count must read exactly zero -- and there are
two of them so a per-row reading cannot be confused with a whole-tensor one."""

PARTITION_MAX = 128
"""The partition-axis extent the hardware enforces, which is the kernel's tile height.

The kernel's per-row operand is sized per tile -- this tile's own height, never the
first tile's -- so the short last tile receives an operand of a matching shape."""

BYPASS_WIDTH = 2176
"""The width the short-sequence bypass asks for in production.

It is the value of the DSA indexer's ``bypass_width()``. Reaching that method needs
the model and its config, which a ``functional/dsa`` kernel test does not carry, so
the number is spelled here with its source named."""

ROWS_WITHIN_ONE_TILE = (5, PARTITION_MAX)
"""Row counts that fit a single tile."""

ROWS_ACROSS_TILES = (132, 256, 2048)
"""Row counts that need more than one tile. ``132`` is not a multiple of 128, so the
short last tile is exercised; ``256`` is two whole tiles with no remainder; ``2048``
is the per-request envelope."""

SEQ_LENS = [2048, 2049, 2051, 2052]
"""The sequence lengths the two bypass bounds are read at.

2048 is upstream's own bound; 2052 is this fork's first selecting length; 2049 and
2051 are the gap between them, where upstream would select and the fork still
bypasses, and where both routes attend every token so the two answers agree."""

POSITION_PERIOD = 17
"""The period of the generated position pattern, chosen so it does not divide the
128-row tile stride (``128 % 17 == 9``).

A period that divided the stride would give every tile the same positions as the
first, and a kernel that loaded its per-row operand from the first tile's offset in
every tile would then produce the correct answer at any row count."""


def _positions(values: list[int]) -> torch.Tensor:
    """The positions of an explicit case, at the shape the seam declares: ``[rows]`` int32."""
    return torch.tensor(values, dtype=torch.int32)


def _periodic_positions(rows: int) -> torch.Tensor:
    """``[rows]`` int32 positions whose period does not divide the 128-row tile stride."""
    return torch.tensor([i % POSITION_PERIOD for i in range(rows)], dtype=torch.int32)


def _max_abs_diff(got: torch.Tensor, expected: torch.Tensor) -> int:
    return int((got.to(torch.int64) - expected.to(torch.int64)).abs().max())


def _checkpoint_dials() -> tuple[int, int]:
    """``(index_topk, index_kpool)`` read from the pinned checkpoint config, never typed here."""
    path = (
        Path(__file__).resolve().parents[2]
        / "model" / "glm5_next" / "fixtures" / "hf-config.json"
    )
    assert path.is_file(), f"the pinned fixture config must exist at {path}"
    text_config = json.loads(path.read_text())["text_config"]
    return int(text_config["index_topk"]), int(text_config["index_kpool"])


def test_the_kernel_rows_equal_the_oracle_exactly() -> None:
    """The kernel's output equals the torch oracle element for element, in int32."""
    positions = _positions(POSITIONS)

    reset_causal_fill_dispatch_counters()
    admitted = can_run_dsa_causal_fill(positions, WIDTH)
    assert admitted is True, "the small shape must take the NKI route, or the reading is not one"
    got = dsa_causal_fill(positions, WIDTH)
    expected = dsa_causal_fill_torch_oracle(positions, WIDTH)
    nki_n, fallback_n = causal_fill_dispatch_counters()

    assert tuple(got.shape) == (ROWS, WIDTH), tuple(got.shape)
    assert got.dtype is torch.int32, got.dtype
    assert expected.dtype is torch.int32, expected.dtype
    diff = _max_abs_diff(got, expected)
    assert diff == 0, f"integer index arithmetic admits no tolerance; max abs diff {diff}"
    assert torch.equal(got, expected), "the kernel and the oracle must agree element for element"

    # Every tile in the kernel's arithmetic chain is float32, with one int32 cast at
    # the store, so the equality above holds only while every quantity flowing
    # through that chain is a whole number float32 holds exactly -- an integer of
    # magnitude below 2**24. Each magnitude is read off this run's own input and
    # output and round-tripped through float32 rather than assumed.
    exact_ceiling = 2 ** 24
    magnitudes = {
        "position_max": int(positions.max()),
        "room_max": int(positions.max()) + 1,      # `positions[i] - c + 1`, at column 0
        "column_plus_one_max": WIDTH,              # `cols1`, the ramp's largest entry plus one
        "result_max": int(got.max()),
        "result_min_magnitude": abs(int(got.min())),
    }
    for name, value in magnitudes.items():
        assert value < exact_ceiling, (name, value, exact_ceiling)
        roundtrip = torch.tensor([value], dtype=torch.float32)[0].item()
        assert roundtrip == float(value), (name, value, roundtrip)

    # 2**24 + 1 is the first integer float32 cannot represent: it round-trips to
    # 2**24, which is what makes the ceiling above the right one.
    beyond = exact_ceiling + 1
    beyond_roundtrip = torch.tensor([beyond], dtype=torch.float32)[0].item()
    assert beyond_roundtrip != float(beyond), (beyond, beyond_roundtrip)
    assert int(beyond_roundtrip) == exact_ceiling, beyond_roundtrip

    # And at the production magnitude, taken from the checkpoint's dials rather than
    # from this file's deliberately small shape, so the exactness is not exact merely
    # because five rows and sixteen columns are tiny. A position in the bypass regime
    # is at most ``select_k * index_kpool + index_kpool - 2``. The largest column
    # magnitude is the caller's admissible width, which is ``index_expand_width``'s
    # own invariant and so is deliberately not asserted here.
    index_topk, index_kpool = _checkpoint_dials()
    production_position_max = (index_topk // index_kpool) * index_kpool + index_kpool - 2
    assert production_position_max < exact_ceiling, production_position_max
    production_roundtrip = torch.tensor(
        [production_position_max], dtype=torch.float32
    )[0].item()
    assert production_roundtrip == float(production_position_max), production_position_max

    # The oracle itself, against a literal written out by hand. Row 1's position is 3,
    # so columns 0 to 3 carry themselves and the remaining twelve carry the sentinel.
    literal = torch.tensor(
        [0, 1, 2, 3] + [SENTINEL] * (WIDTH - 4), dtype=torch.int32
    )
    assert torch.equal(expected[1], literal), (expected[1].tolist(), literal.tolist())

    identity = causal_fill_kernel_identity()
    assert identity is not None, "the identity must be derived by taking the dispatch branch"
    assert identity[1] == "_causal_fill_nki", identity
    assert nki_n == 1 and fallback_n == 0, (nki_n, fallback_n)


def test_the_sentinel_count_per_row_is_the_width_less_the_position_less_one() -> None:
    """Per row, the number of ``-1`` columns is exactly ``width - position - 1``.

    This compares the kernel against arithmetic on the positions rather than against
    the oracle, so it fails on an off-by-one the two share.
    """
    positions = _positions(POSITIONS)

    reset_causal_fill_dispatch_counters()
    got = dsa_causal_fill(positions, WIDTH)
    nki_n, fallback_n = causal_fill_dispatch_counters()

    per_row = (got == SENTINEL).sum(dim=1).to(torch.int64)
    expected = torch.tensor([WIDTH - p - 1 for p in POSITIONS], dtype=torch.int64)
    assert torch.equal(per_row, expected), (per_row.tolist(), expected.tolist())
    assert int(per_row.numel()) == ROWS, per_row.numel()

    # The saturated rows carry zero sentinels, and every column of such a row is its
    # own index with nothing masked.
    saturated = [r for r, p in enumerate(POSITIONS) if p >= WIDTH - 1]
    assert len(saturated) == 2, saturated
    for row in saturated:
        assert int(per_row[row]) == 0, (row, int(per_row[row]))
        assert torch.equal(
            got[row], torch.arange(WIDTH, dtype=torch.int32)
        ), got[row].tolist()

    masked_rows = [r for r, p in enumerate(POSITIONS) if p < WIDTH - 1]
    assert len(masked_rows) == 3, masked_rows
    assert all(int(per_row[r]) > 0 for r in masked_rows), per_row.tolist()

    assert nki_n == 1 and fallback_n == 0, (nki_n, fallback_n)


def test_the_bypass_bound_is_the_pool_count_and_not_the_token_budget() -> None:
    """The fork's bypass predicate differs from upstream's, and they agree where it matters.

    The fork bounds on the candidate width (``seq_len // index_kpool <= select_k``)
    because that is what the selector refuses; upstream bounds on the token budget
    (``max_seq_len <= index_topk``). They disagree on 2,049 to 2,051 -- and on exactly
    that span both routes attend every token anyway, because the complete-pool count
    equals ``select_k`` there, so upstream selecting 512 of 512 and the fork filling
    causal rows are the same attention set.

    The dials come from the pinned checkpoint config and the minimum selecting length
    is found by scanning upward, so a checkpoint change moves this reading rather than
    leaving it accidentally true.
    """
    index_topk, index_kpool = _checkpoint_dials()
    select_k = index_topk // index_kpool

    fork = [seq // index_kpool <= select_k for seq in SEQ_LENS]
    upstream = [seq <= index_topk for seq in SEQ_LENS]
    assert fork == [True, True, True, False], (SEQ_LENS, fork)
    assert upstream == [True, False, False, False], (SEQ_LENS, upstream)
    assert fork != upstream, "the two bounds must differ, or the fork's predicate is upstream's"

    # Where they disagree the attention set is the same: the complete-pool count is
    # exactly ``select_k``, so a selection would take every candidate.
    disagreeing = [seq for seq, f, u in zip(SEQ_LENS, fork, upstream) if f != u]
    assert disagreeing == [2049, 2051], disagreeing
    for seq in disagreeing:
        assert seq // index_kpool == select_k, (seq, seq // index_kpool, select_k)

    # The minimum selecting length, found rather than typed: the first length the
    # fork's predicate does not let through. The scan bound comes from the dials.
    minimum = next(
        seq for seq in range(1, index_topk * 2 + index_kpool * 2)
        if not (seq // index_kpool <= select_k)
    )
    assert minimum == 2052, minimum
    assert minimum == (select_k + 1) * index_kpool, (minimum, select_k, index_kpool)
    assert minimum != index_topk, (
        f"the fork's bound is not the token budget: minimum selecting length {minimum} against "
        f"index_topk {index_topk}"
    )


def test_a_width_below_one_and_a_float_positions_tensor_are_refused() -> None:
    """Both malformed calls raise ``DsaCausalFillError`` naming what is wrong.

    Serving a zero width or a float position tensor through the torch oracle would
    hand the caller a correct-looking answer for a call that cannot be right, so these
    raise instead of falling back.
    """
    positions = _positions(POSITIONS)

    reset_causal_fill_dispatch_counters()
    with pytest.raises(DsaCausalFillError) as caught_width:
        dsa_causal_fill(positions, 0)
    width_message = str(caught_width.value)
    assert "width" in width_message, width_message
    assert "width=0" in width_message, width_message
    assert "at least 1 column" in width_message, width_message

    with pytest.raises(DsaCausalFillError) as caught_dtype:
        dsa_causal_fill(positions.to(torch.float32), WIDTH)
    dtype_message = str(caught_dtype.value)
    assert "int32" in dtype_message, dtype_message
    assert "torch.float32" in dtype_message, dtype_message
    assert "positions" in dtype_message, dtype_message

    refused_nki, refused_fallback = causal_fill_dispatch_counters()
    assert (refused_nki, refused_fallback) == (0, 0), (
        f"a refusal that dispatched first has already produced the wrong answer; got "
        f"{(refused_nki, refused_fallback)}"
    )


def test_the_torch_fallback_answers_the_same_as_the_kernel() -> None:
    """Where the kernel is unavailable the gate declines and the torch route still answers."""
    for positions in (_positions(POSITIONS), _periodic_positions(132)):
        reset_causal_fill_dispatch_counters()
        saved = mod.can_run_kernel
        try:
            # The gate reads ``can_run_kernel`` as a module global, so replacing it on
            # the module under test is what a call in a process without NKI sees.
            mod.can_run_kernel = lambda: False
            assert can_run_dsa_causal_fill(positions, WIDTH) is False
            served = dsa_causal_fill(positions, WIDTH)
        finally:
            mod.can_run_kernel = saved
        nki_n, fallback_n = causal_fill_dispatch_counters()
        assert (nki_n, fallback_n) == (0, 1), (nki_n, fallback_n)
        assert torch.equal(served, dsa_causal_fill_torch_oracle(positions, WIDTH))

        assert mod.can_run_kernel is can_run_kernel
        assert can_run_dsa_causal_fill(positions, WIDTH) is True


def test_row_counts_above_the_partition_limit_are_served_in_tiles() -> None:
    """132, 256 and 2048 rows each return the declared shape in one dispatch.

    132 is not a multiple of 128, so the short last tile is exercised here. One case
    at the production width shows the tiling is width-blind.
    """
    for rows in ROWS_ACROSS_TILES:
        for width in (WIDTH, BYPASS_WIDTH) if rows == 132 else (WIDTH,):
            positions = _periodic_positions(rows)
            reset_causal_fill_dispatch_counters()
            assert can_run_dsa_causal_fill(positions, width) is True
            got = dsa_causal_fill(positions, width)
            nki_n, fallback_n = causal_fill_dispatch_counters()
            assert tuple(got.shape) == (rows, width), tuple(got.shape)
            assert got.dtype is torch.int32, got.dtype
            assert (nki_n, fallback_n) == (1, 0), (nki_n, fallback_n)


def test_the_kernel_equals_the_torch_oracle_at_every_row_count() -> None:
    """One tile, a ragged last tile and sixteen whole tiles all match the oracle exactly."""
    for rows in ROWS_WITHIN_ONE_TILE + ROWS_ACROSS_TILES:
        positions = _periodic_positions(rows)
        got = dsa_causal_fill(positions, WIDTH)
        expected = dsa_causal_fill_torch_oracle(positions, WIDTH)
        diff = _max_abs_diff(got, expected)
        assert diff == 0 and torch.equal(got, expected), f"rows={rows} max_abs_diff={diff}"


def test_one_row_moves_only_its_own_output_across_a_tile_boundary() -> None:
    """At 2048 rows, perturbing one row leaves every other row bit-identical.

    The perturbed row sits inside the last tile, so a walk that leaked an earlier
    tile's operands forward would show.
    """
    rows = 2048
    positions = _periodic_positions(rows)
    base = dsa_causal_fill(positions, WIDTH)

    # The perturbation stays within the same position pattern, and it must really
    # change this row's output or "exactly one row changed" would assert nothing. At
    # width 16 the oracle fills the whole ramp for any position of 15 or more, so two
    # different positions can share an output row; the precondition is therefore read
    # off the oracle rather than assumed.
    target = rows - 3
    before_pos = int(positions[target])
    moved = positions.clone()
    moved[target] = (before_pos + 1) % POSITION_PERIOD
    row_before = dsa_causal_fill_torch_oracle(positions[target:target + 1], WIDTH)
    row_after = dsa_causal_fill_torch_oracle(moved[target:target + 1], WIDTH)
    assert not torch.equal(row_before, row_after), (
        f"positions {before_pos} and {int(moved[target])} produce the same output row at width "
        f"{WIDTH}, so this test could not detect a leak"
    )
    after = dsa_causal_fill(moved, WIDTH)

    changed = (after != base).any(dim=1)
    changed_rows = int(changed.sum())
    assert changed_rows == 1 and bool(changed[target]), changed_rows
    others = torch.cat([base[:target], base[target + 1:]])
    others_after = torch.cat([after[:target], after[target + 1:]])
    assert torch.equal(others, others_after), "a row outside the perturbed one moved"
