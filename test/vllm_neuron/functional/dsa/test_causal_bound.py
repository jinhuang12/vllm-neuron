# SPDX-License-Identifier: Apache-2.0
"""Tests for the causal bound and the sentinel writer.

The bound fills every pool a query row has not completed with ``BOUND_FILL`` and
leaves every other column untouched. The sentinel writes ``-1`` where a selected
value is a fill or a selected index lies past the real width, and leaves every
other index alone. Neither computes a new number, so every comparison is exact,
on raw int32 bits where the output is float because ``==`` on floats cannot tell
``-0.0`` from ``+0.0``.
"""

import pytest
import torch

from vllm_neuron.functional.attention.mla_sparse import (
    can_run_mla_sparse_attention,
    mla_sparse_attention,
)
from vllm_neuron.functional.dsa import causal_bound as mod
from vllm_neuron.functional.dsa.causal_bound import (
    BOUND_FILL,
    BOUND_FILL_MARK,
    PARTITION_MAX,
    SENTINEL,
    DsaCausalBoundError,
    can_run_dsa_causal_bound,
    can_run_dsa_causal_sentinel,
    causal_bound_dispatch_counters,
    causal_bound_kernel_identity,
    causal_sentinel_dispatch_counters,
    causal_sentinel_kernel_identity,
    dsa_causal_bound,
    dsa_causal_bound_torch_oracle,
    dsa_causal_sentinel,
    dsa_causal_sentinel_torch_oracle,
    reset_causal_bound_dispatch_counters,
    reset_causal_sentinel_dispatch_counters,
    row_tile_count,
)
from vllm_neuron.functional.dsa.index_expand import (
    can_run_dsa_index_expand,
    dsa_index_expand,
    index_expand_dispatch_counters,
    index_expand_width,
    reset_index_expand_dispatch_counters,
)
from vllm_neuron.functional.dsa.topk_select import (
    _nki_config,
    _nki_dtype_of,
    can_run_dsa_topk_select,
    dsa_topk_select,
    reset_topk_select_dispatch_counters,
    topk_select_dispatch_counters,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

from test.vllm_neuron.functional.attention.test_mla_sparse import (
    ATOL,
    RTOL,
    make_case,
    sentinel_reference,
)
from test.vllm_neuron.functional.dsa.test_index_expand import (
    _precondition_violations,
)

# Small shape: 5 query rows over 16 candidate pools of 4 tokens each.
ROWS = 5
POOL_COLUMNS = 16
POOL_SIZE = 4

# 1 is shorter than one pool, so the whole row is bounded; 4 is exactly one
# pool; 9 reaches one token into a third pool, which must still be bounded; 33
# is interior; 64 = POOL_COLUMNS * POOL_SIZE saturates the row, so nothing is
# bounded and the fill count must read zero.
CAUSAL_LENS = [1, 4, 9, 33, 64]

SELECT_K = 2

# k = 16 at width 32 puts the selector on its rotational path with two stages
# and a per-stage k of 8, the branch that strikes each taken value to -inf in
# its own input buffer. Four of the five rows complete fewer than k pools, so
# the selector fills their remaining slots from an all-BOUND_FILL buffer.
MULTIFOLD_SELECT_K = 16
MULTIFOLD_POOL_COLUMNS = 32

# Complete 16, 9, 5, 3 and 1 pools: exactly k on the first row, fewer after.
# Descending so the rows with the most real candidates sit in the selector's
# first (full) tile rather than its ragged last one.
MULTIFOLD_CAUSAL_LENS = [64, 36, 20, 12, 4]

# An odd width folds unevenly across two stages, so the selector's loader pads
# the last fold with one finite column at index 33, one past the last real pool.
PAD_POOL_COLUMNS = 33

# topk = index_expand_width(SELECT_K, POOL_SIZE); s_kv only has to exceed
# max(CAUSAL_LENS) - 1.
MLA_CASE = dict(seq=ROWS, heads=4, latent=128, topk=128, s_kv=256, rope=0)


def _scores(seed: int) -> torch.Tensor:
    """``[ROWS, POOL_COLUMNS]`` float32, scaled small because it feeds a softmax downstream."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(ROWS, POOL_COLUMNS, generator=gen, dtype=torch.float32) * 0.05


def _causal_len() -> torch.Tensor:
    return torch.tensor(CAUSAL_LENS, dtype=torch.int32).reshape(ROWS, 1)


def _complete_pools() -> list[int]:
    return [c // POOL_SIZE for c in CAUSAL_LENS]


def test_bound_fills_exactly_the_incomplete_pools_and_leaves_kept_bits_alone() -> None:
    """``BOUND_FILL`` at every pool the row does not complete; every kept column is bit-identical."""
    scores = _scores(103)
    causal_len = _causal_len()

    # Row 4 is saturated, so column 0 there is kept, and -0.0 is the one value an
    # additive mask would rewrite while every float == still passed.
    scores[4, 0] = -0.0
    planted_bits = int(scores[4, 0].view(torch.int32))

    reset_causal_bound_dispatch_counters()
    assert can_run_dsa_causal_bound(scores, causal_len, POOL_SIZE) is True
    got = dsa_causal_bound(scores, causal_len, POOL_SIZE)
    expected = dsa_causal_bound_torch_oracle(scores, causal_len, POOL_SIZE)

    assert tuple(got.shape) == (ROWS, POOL_COLUMNS), tuple(got.shape)
    assert got.dtype is torch.float32, got.dtype
    assert torch.equal(got.view(torch.int32), expected.view(torch.int32)), (
        "the kernel and the oracle must agree bit for bit; first differing entry "
        f"{(got.view(torch.int32) != expected.view(torch.int32)).nonzero()[:1].tolist()}"
    )

    complete = _complete_pools()
    per_row = (got == BOUND_FILL).sum(dim=1).to(torch.int64)
    expected_counts = torch.tensor(
        [POOL_COLUMNS - min(POOL_COLUMNS, c) for c in complete], dtype=torch.int64
    )
    assert torch.equal(per_row, expected_counts), (per_row.tolist(), expected_counts.tolist())
    assert per_row.tolist() == [16, 15, 14, 8, 0], per_row.tolist()

    kept_mask = torch.zeros(ROWS, POOL_COLUMNS, dtype=torch.bool)
    for r, c in enumerate(complete):
        kept_mask[r, : min(POOL_COLUMNS, c)] = True
    assert torch.equal(
        got.view(torch.int32)[kept_mask], scores.view(torch.int32)[kept_mask]
    ), "a kept column was rewritten; the mask is not a select"
    assert int(got[4, 0].view(torch.int32)) == planted_bits

    identity = causal_bound_kernel_identity()
    assert identity is not None and identity[1] == "_causal_bound_nki", identity
    assert causal_bound_dispatch_counters() == (1, 0), causal_bound_dispatch_counters()


def test_sentinel_marks_every_bounded_selection_and_no_other_index() -> None:
    """``-1`` exactly where the selected value is a fill; every other index is the selector's own."""
    scores = _scores(203)
    causal_len = _causal_len()
    complete = _complete_pools()

    reset_causal_bound_dispatch_counters()
    reset_causal_sentinel_dispatch_counters()
    reset_topk_select_dispatch_counters()

    bounded = dsa_causal_bound(scores, causal_len, POOL_SIZE)
    assert can_run_dsa_topk_select(bounded, SELECT_K) is True
    values, indices = dsa_topk_select(bounded, SELECT_K)
    assert tuple(values.shape) == (ROWS, SELECT_K), tuple(values.shape)

    # A bounded slot is one the row had no complete pool for, and the selector
    # must hand it back at or below the mark.
    expected_counts = [max(0, SELECT_K - c) for c in complete]
    is_filled = values <= BOUND_FILL_MARK
    assert is_filled.sum(dim=1).tolist() == expected_counts, (
        f"the selector did not return a filled value for every bounded slot: "
        f"{is_filled.sum(dim=1).tolist()} against {expected_counts}"
    )

    # dsa_topk_select returns int64 indices; dsa_index_expand reads int32, so the
    # dispatch site casts and this test casts the same way.
    idx32 = indices.to(torch.int32)
    assert can_run_dsa_causal_sentinel(values, idx32, POOL_COLUMNS) is True
    got = dsa_causal_sentinel(values, idx32, POOL_COLUMNS)
    expected = dsa_causal_sentinel_torch_oracle(values, idx32, POOL_COLUMNS)

    assert got.dtype is torch.int32, got.dtype
    assert torch.equal(got, expected), (
        f"the kernel and the oracle must agree element for element; first differing "
        f"{(got != expected).nonzero()[:1].tolist()}"
    )
    per_row = (got == SENTINEL).sum(dim=1).to(torch.int64)
    assert per_row.tolist() == expected_counts, (per_row.tolist(), expected_counts)
    assert torch.equal(got == SENTINEL, is_filled), (
        "a sentinel was written at a position whose value was a real score, or withheld "
        "at one whose value was a fill"
    )
    kept = ~is_filled
    assert torch.equal(got[kept], idx32[kept]), "a kept index was rewritten"

    # Row 0 completes no pool, so both of its selections are sentinels.
    assert bool((got[0] == SENTINEL).all()), got[0].tolist()

    identity = causal_sentinel_kernel_identity()
    assert identity is not None and identity[1] == "_causal_sentinel_nki", identity
    assert causal_bound_dispatch_counters() == (1, 0), causal_bound_dispatch_counters()
    assert causal_sentinel_dispatch_counters() == (1, 0), causal_sentinel_dispatch_counters()
    assert topk_select_dispatch_counters() == (1, 0), topk_select_dispatch_counters()


def test_sentinel_marks_bounded_slots_when_the_selector_strikes_its_own_input() -> None:
    """At k = 16 the selector overwrites taken values in place; every row still gets
    ``max(0, k - complete)`` sentinels and no legal pool id appears twice."""
    gen = torch.Generator().manual_seed(20316)
    scores = torch.randn(
        ROWS, MULTIFOLD_POOL_COLUMNS, generator=gen, dtype=torch.float32
    ) * 0.05
    causal_len = torch.tensor(MULTIFOLD_CAUSAL_LENS, dtype=torch.int32).reshape(ROWS, 1)
    complete = [min(c // POOL_SIZE, MULTIFOLD_POOL_COLUMNS) for c in MULTIFOLD_CAUSAL_LENS]
    expected_counts = [max(0, MULTIFOLD_SELECT_K - c) for c in complete]

    reset_causal_bound_dispatch_counters()
    reset_causal_sentinel_dispatch_counters()
    reset_topk_select_dispatch_counters()

    bounded = dsa_causal_bound(scores, causal_len, POOL_SIZE)
    assert can_run_dsa_topk_select(bounded, MULTIFOLD_SELECT_K) is True, (
        f"the selector must serve rows={ROWS} width={MULTIFOLD_POOL_COLUMNS} "
        f"k={MULTIFOLD_SELECT_K}; torch.topk never strikes its input"
    )
    values, indices = dsa_topk_select(bounded, MULTIFOLD_SELECT_K)
    assert tuple(values.shape) == (ROWS, MULTIFOLD_SELECT_K), tuple(values.shape)
    idx32 = indices.to(torch.int32)
    got = dsa_causal_sentinel(values, idx32, MULTIFOLD_POOL_COLUMNS)

    assert causal_bound_dispatch_counters() == (1, 0), causal_bound_dispatch_counters()
    assert causal_sentinel_dispatch_counters() == (1, 0), causal_sentinel_dispatch_counters()
    assert topk_select_dispatch_counters() == (1, 0), topk_select_dispatch_counters()

    # The branch is read off the selector's own config: at least two stages, a
    # per-stage k that is a multiple of topk_per_stage, and no pad columns.
    from vllm_neuron.functional.vendored_kernels.rotational_topk.rotational_topk_utils import (
        HW_PARAMS,
    )

    cfg = _nki_config(ROWS, MULTIFOLD_POOL_COLUMNS, MULTIFOLD_SELECT_K, _nki_dtype_of(bounded))
    per_stage = int(HW_PARAMS.topk_per_stage)
    assert int(cfg.n_stages) >= 2, int(cfg.n_stages)
    assert int(cfg.local_top_k_per_stage) % per_stage == 0, (
        int(cfg.local_top_k_per_stage), per_stage
    )
    assert int(cfg.padded_vocab_size) == int(cfg.vocab_size) == MULTIFOLD_POOL_COLUMNS, (
        int(cfg.padded_vocab_size), int(cfg.vocab_size)
    )

    # The oracle's value arm is le(values, mark) | isnan(values), so NaN counts as
    # filled here too.
    filled = (values <= BOUND_FILL_MARK) | torch.isnan(values)
    per_row = (got == SENTINEL).sum(dim=1).to(torch.int64)
    assert per_row.tolist() == expected_counts, (per_row.tolist(), expected_counts)
    assert torch.equal(got == SENTINEL, filled), (
        "a sentinel was written at a position whose value was a real score, or withheld "
        "at one whose value was a fill"
    )
    kept = ~filled
    assert torch.equal(got[kept], idx32[kept]), "a kept index was rewritten"

    for r in range(ROWS):
        legal = [int(v) for v in got[r].tolist() if v >= 0]
        assert len(legal) == min(complete[r], MULTIFOLD_SELECT_K), (r, legal)
        assert all(0 <= v < complete[r] for v in legal), (r, legal, complete[r])
        assert len(set(legal)) == len(legal), (r, legal)


def test_sentinel_marks_the_selector_pad_index_past_the_real_width() -> None:
    """An index at or past ``width`` is a pad the selector invented; it comes back as ``-1``
    even though its value is an ordinary finite number."""
    gen = torch.Generator().manual_seed(20333)
    scores = torch.randn(ROWS, PAD_POOL_COLUMNS, generator=gen, dtype=torch.float32) * 0.05
    causal_len = torch.tensor(MULTIFOLD_CAUSAL_LENS, dtype=torch.int32).reshape(ROWS, 1)
    complete = [min(c // POOL_SIZE, PAD_POOL_COLUMNS) for c in MULTIFOLD_CAUSAL_LENS]
    expected_counts = [max(0, MULTIFOLD_SELECT_K - c) for c in complete]

    reset_causal_bound_dispatch_counters()
    reset_causal_sentinel_dispatch_counters()
    reset_topk_select_dispatch_counters()

    bounded = dsa_causal_bound(scores, causal_len, POOL_SIZE)
    assert can_run_dsa_topk_select(bounded, MULTIFOLD_SELECT_K) is True, (
        f"the selector must serve rows={ROWS} width={PAD_POOL_COLUMNS} "
        f"k={MULTIFOLD_SELECT_K}; torch.topk pads nothing"
    )

    # The pad count is the selector's own padded_vocab_size - vocab_size, and the
    # pads occupy global positions [vocab_size, padded_vocab_size).
    cfg = _nki_config(ROWS, PAD_POOL_COLUMNS, MULTIFOLD_SELECT_K, _nki_dtype_of(bounded))
    pad_columns = int(cfg.padded_vocab_size) - int(cfg.vocab_size)
    assert int(cfg.vocab_size) == PAD_POOL_COLUMNS, int(cfg.vocab_size)
    assert int(cfg.n_stages) >= 2, int(cfg.n_stages)
    assert pad_columns >= 1, (
        f"width {PAD_POOL_COLUMNS} folded into {int(cfg.n_stages)} stages with no pad column"
    )

    values, indices = dsa_topk_select(bounded, MULTIFOLD_SELECT_K)
    idx32 = indices.to(torch.int32)

    # The finite pad outranks every fill, so a row with free slots spends them on
    # pads first and hands back an index at or past the real width.
    out_of_range = idx32 >= PAD_POOL_COLUMNS
    assert int(out_of_range.sum()) >= 1, (
        f"the selector returned no index at or past width {PAD_POOL_COLUMNS} on these inputs"
    )
    assert int(idx32.max()) < int(cfg.padded_vocab_size), int(idx32.max())

    assert can_run_dsa_causal_sentinel(values, idx32, PAD_POOL_COLUMNS) is True
    got = dsa_causal_sentinel(values, idx32, PAD_POOL_COLUMNS)
    expected = dsa_causal_sentinel_torch_oracle(values, idx32, PAD_POOL_COLUMNS)
    assert torch.equal(got, expected), (
        f"the kernel and the oracle must agree element for element; first differing "
        f"{(got != expected).nonzero()[:1].tolist()}"
    )
    assert bool((got[out_of_range] == SENTINEL).all()), "a pad index survived the sentinel"

    assert causal_bound_dispatch_counters() == (1, 0), causal_bound_dispatch_counters()
    assert causal_sentinel_dispatch_counters() == (1, 0), causal_sentinel_dispatch_counters()
    assert topk_select_dispatch_counters() == (1, 0), topk_select_dispatch_counters()

    # Every surviving id is a pool the row completes, and no id appears twice.
    per_row = (got == SENTINEL).sum(dim=1).to(torch.int64)
    assert per_row.tolist() == expected_counts, (per_row.tolist(), expected_counts)
    for r in range(ROWS):
        legal = [int(v) for v in got[r].tolist() if v >= 0]
        assert len(legal) == min(complete[r], MULTIFOLD_SELECT_K), (r, legal)
        assert all(0 <= v < complete[r] for v in legal), (r, legal, complete[r])
        assert len(set(legal)) == len(legal), (r, legal)


def test_bounded_chain_emits_legal_pool_ids_and_attention_matches_the_reference() -> None:
    """Every non-negative pool id is below ``seq_len // pool_size``, the expanded token
    indices stay inside their own rows, and sparse attention on them matches its reference."""
    scores = _scores(303)
    causal_len = _causal_len()
    seq_lens = torch.tensor(CAUSAL_LENS, dtype=torch.int32)

    reset_causal_bound_dispatch_counters()
    reset_causal_sentinel_dispatch_counters()
    reset_index_expand_dispatch_counters()

    bounded = dsa_causal_bound(scores, causal_len, POOL_SIZE)
    values, indices = dsa_topk_select(bounded, SELECT_K)
    pool_ids = dsa_causal_sentinel(values, indices.to(torch.int32), POOL_COLUMNS)

    violations = _precondition_violations(pool_ids.tolist(), CAUSAL_LENS, POOL_SIZE)
    assert violations == [], (
        f"the bounded chain breaks the expansion's precondition at {violations}"
    )

    width = index_expand_width(SELECT_K, POOL_SIZE)
    assert width == MLA_CASE["topk"], (width, MLA_CASE["topk"])

    assert can_run_dsa_index_expand(pool_ids, seq_lens, POOL_SIZE) is True
    token_idx = dsa_index_expand(pool_ids, seq_lens, POOL_SIZE)
    assert tuple(token_idx.shape) == (ROWS, width), tuple(token_idx.shape)
    assert index_expand_dispatch_counters() == (1, 0), index_expand_dispatch_counters()

    live = token_idx >= 0
    assert int(live.sum()) > 0, "the expansion must emit at least one live token index"
    limit = seq_lens.to(torch.int64).reshape(ROWS, 1).expand_as(token_idx)
    out_of_row = int((live & (token_idx.to(torch.int64) >= limit)).sum())
    assert out_of_row == 0, (
        f"{out_of_row} live token indices point past their own row's causal length"
    )
    assert int(token_idx.max()) < MLA_CASE["s_kv"], int(token_idx.max())

    q_lift, c_kv, _discarded_idx, q_pe, k_pe = make_case(**MLA_CASE, seed=403)
    assert q_pe is None and k_pe is None, "this geometry is at rope == 0"
    scale = float(MLA_CASE["latent"]) ** -0.5
    assert can_run_mla_sparse_attention(
        q_lift, MLA_CASE["seq"], MLA_CASE["heads"], MLA_CASE["latent"], MLA_CASE["rope"],
        MLA_CASE["topk"], MLA_CASE["s_kv"], scale,
    ) is True, "the sparse kernel must admit the emitted width"
    out = mla_sparse_attention(q_lift, c_kv, token_idx, scale)
    ref = sentinel_reference(q_lift, c_kv, token_idx, scale)
    assert bool(torch.isfinite(out).all()), (
        "the chain produced a non-finite value, which is what a fill reaching the softmax "
        "unmasked looks like"
    )
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)

    assert causal_bound_dispatch_counters() == (1, 0), causal_bound_dispatch_counters()
    assert causal_sentinel_dispatch_counters() == (1, 0), causal_sentinel_dispatch_counters()


def test_malformed_calls_are_refused_by_name_before_any_dispatch() -> None:
    """Each bad argument raises ``DsaCausalBoundError`` naming it, and neither path runs."""
    scores = _scores(403)
    causal_len = _causal_len()

    reset_causal_bound_dispatch_counters()
    reset_causal_sentinel_dispatch_counters()

    with pytest.raises(DsaCausalBoundError) as caught:
        dsa_causal_bound(scores, causal_len, 6)
    message = str(caught.value)
    assert "pool_size=6" in message and "power of two" in message, message

    with pytest.raises(DsaCausalBoundError) as caught:
        dsa_causal_bound(scores, causal_len.to(torch.float32), POOL_SIZE)
    message = str(caught.value)
    assert "causal_len" in message and "int32" in message, message
    assert "torch.float32" in message, message

    with pytest.raises(DsaCausalBoundError) as caught:
        dsa_causal_bound(scores, causal_len[: ROWS - 1], POOL_SIZE)
    message = str(caught.value)
    assert "one length per score row" in message, message
    assert f"{ROWS - 1} lengths" in message and f"{ROWS} score rows" in message, message

    good_values = torch.zeros(ROWS, SELECT_K, dtype=torch.float32)
    good_idx = torch.zeros(ROWS, SELECT_K, dtype=torch.int32)

    with pytest.raises(DsaCausalBoundError) as caught:
        dsa_causal_sentinel(good_values, good_idx.to(torch.int64), POOL_COLUMNS)
    message = str(caught.value)
    assert "int32" in message and "torch.int64" in message, message

    with pytest.raises(DsaCausalBoundError) as caught:
        dsa_causal_sentinel(good_values, good_idx, torch.tensor(POOL_COLUMNS))
    message = str(caught.value)
    assert "width must be a python int" in message and "Tensor" in message, message

    with pytest.raises(DsaCausalBoundError) as caught:
        dsa_causal_sentinel(good_values, good_idx, 0)
    message = str(caught.value)
    assert "width=0" in message and "positive number of real pool columns" in message, message

    assert causal_bound_dispatch_counters() == (0, 0), causal_bound_dispatch_counters()
    assert causal_sentinel_dispatch_counters() == (0, 0), causal_sentinel_dispatch_counters()


def test_torch_path_serves_both_entry_points_when_the_kernel_is_unavailable() -> None:
    """With ``can_run_kernel`` False both gates decline and both oracles answer correctly."""
    scores = _scores(403)
    causal_len = _causal_len()
    values = torch.zeros(ROWS, SELECT_K, dtype=torch.float32)
    values[0, 0] = BOUND_FILL
    idx32 = torch.zeros(ROWS, SELECT_K, dtype=torch.int32)

    reset_causal_bound_dispatch_counters()
    reset_causal_sentinel_dispatch_counters()

    # Both gates read can_run_kernel as a module global, so replacing it on the
    # module is what a call in a process without NKI sees.
    saved = mod.can_run_kernel
    try:
        mod.can_run_kernel = lambda: False
        assert can_run_dsa_causal_bound(scores, causal_len, POOL_SIZE) is False
        assert can_run_dsa_causal_sentinel(values, idx32, POOL_COLUMNS) is False
        served_bound = dsa_causal_bound(scores, causal_len, POOL_SIZE)
        served_sentinel = dsa_causal_sentinel(values, idx32, POOL_COLUMNS)
    finally:
        mod.can_run_kernel = saved

    assert causal_bound_dispatch_counters() == (0, 1), causal_bound_dispatch_counters()
    assert causal_sentinel_dispatch_counters() == (0, 1), causal_sentinel_dispatch_counters()
    assert torch.equal(
        served_bound.view(torch.int32),
        dsa_causal_bound_torch_oracle(scores, causal_len, POOL_SIZE).view(torch.int32),
    )
    assert torch.equal(
        served_sentinel, dsa_causal_sentinel_torch_oracle(values, idx32, POOL_COLUMNS)
    )
    assert int((served_sentinel == SENTINEL).sum()) == 1, served_sentinel.tolist()

    assert mod.can_run_kernel is can_run_kernel
    assert can_run_dsa_causal_bound(scores, causal_len, POOL_SIZE) is True
    assert can_run_dsa_causal_sentinel(values, idx32, POOL_COLUMNS) is True


# --------------------------------------------------------------------------- #
# Many-tile shapes. PARTITION_MAX is the tile height on the query-token axis.
# --------------------------------------------------------------------------- #

# The largest call: 2,048 query tokens per request.
TILED_ROWS = 2048

# Tile edges: one row, the last single tile, the first two-tile call (second
# tile one row high), a second tile of four rows, a whole number of tiles, a
# remainder tile of 80 rows, and the largest call.
TILED_LADDER = [
    1,
    PARTITION_MAX - 1,
    PARTITION_MAX,
    PARTITION_MAX + 1,
    PARTITION_MAX + 4,
    2 * PARTITION_MAX,
    2000,
    TILED_ROWS,
]

# Row i completes i % 17 pools, so every count from 0 (wholly bounded) to 16
# (saturated) appears, and because 17 does not divide the 128-row tile height
# no two tiles hold the same set of lengths.
TILED_LENGTH_PERIOD = POOL_COLUMNS + 1

# 127 is coprime with the tile height, so rows planted at this stride land at a
# different offset inside every tile.
TILED_PAD_STRIDE = PARTITION_MAX - 1


def _tiled_scores(rows: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(rows, POOL_COLUMNS, generator=gen, dtype=torch.float32) * 0.05


def _tiled_causal_len(rows: int) -> torch.Tensor:
    """``[rows, 1]`` int32 lengths, row ``i`` completing ``i % TILED_LENGTH_PERIOD`` pools."""
    lens = [(i % TILED_LENGTH_PERIOD) * POOL_SIZE for i in range(rows)]
    return torch.tensor(lens, dtype=torch.int32).reshape(rows, 1)


def _tiled_complete_pools(rows: int) -> torch.Tensor:
    return torch.tensor([i % TILED_LENGTH_PERIOD for i in range(rows)], dtype=torch.int64)


def _selection_of(bounded: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """A ``(values, int32 indices)`` pair from ``torch.topk``.

    The sentinel reads only whether a value is a fill and whether an index is
    past the real width, so the host selection is a fine source for it.
    """
    values, indices = torch.topk(bounded, k, dim=1)
    return values.contiguous(), indices.to(torch.int32).contiguous()


def test_bound_and_sentinel_match_the_oracles_at_every_tile_edge() -> None:
    """Bit equality with both oracles, and the fill count from the closed form, at each ladder extent."""
    for rows in TILED_LADDER:
        scores = _tiled_scores(rows, 3000 + rows)
        clen = _tiled_causal_len(rows)

        reset_causal_bound_dispatch_counters()
        reset_causal_sentinel_dispatch_counters()
        got = dsa_causal_bound(scores, clen, POOL_SIZE)
        expected = dsa_causal_bound_torch_oracle(scores, clen, POOL_SIZE)
        assert torch.equal(got.view(torch.int32), expected.view(torch.int32)), (
            f"rows={rows}: the bound's bits differ from the oracle's"
        )

        values, idx32 = _selection_of(got, SELECT_K)
        got_marked = dsa_causal_sentinel(values, idx32, POOL_COLUMNS)
        expected_marked = dsa_causal_sentinel_torch_oracle(values, idx32, POOL_COLUMNS)
        assert torch.equal(got_marked, expected_marked), f"rows={rows}: the sentinel differs"

        assert causal_bound_dispatch_counters() == (1, 0), (rows, causal_bound_dispatch_counters())
        assert causal_sentinel_dispatch_counters() == (1, 0), (
            rows, causal_sentinel_dispatch_counters()
        )

        complete = _tiled_complete_pools(rows)
        expected_bounded = (POOL_COLUMNS - complete).clamp(min=0).to(torch.int64)
        per_row = (got == BOUND_FILL).sum(dim=1).to(torch.int64)
        assert torch.equal(per_row, expected_bounded), (
            f"rows={rows}: filled-column counts disagree with the closed form"
        )


def test_perturbing_one_row_moves_only_that_row_across_tiles() -> None:
    """Changing one row's score and length moves that row's output and no other row's bits."""
    rows = TILED_ROWS
    scores = _tiled_scores(rows, 1051)
    clen = _tiled_causal_len(rows)
    base = dsa_causal_bound(scores, clen, POOL_SIZE)
    complete = _tiled_complete_pools(rows)

    # The first row of the second tile, and a row near the end of the last tile.
    # Both complete strictly between zero and every pool (9 and 5), so both move
    # when their own score and length do.
    for probe in (PARTITION_MAX, rows - 3):
        moved_scores = scores.clone()
        moved_scores[probe] = moved_scores[probe] + 1.0
        moved_clen = clen.clone()
        new_complete = (int(complete[probe]) + 1) % TILED_LENGTH_PERIOD
        moved_clen[probe, 0] = new_complete * POOL_SIZE

        got = dsa_causal_bound(moved_scores, moved_clen, POOL_SIZE)
        assert not torch.equal(got[probe], base[probe]), (
            f"probe row {probe} did not move when its own score and length did"
        )

        keep = torch.ones(rows, dtype=torch.bool)
        keep[probe] = False
        assert torch.equal(got[keep].view(torch.int32), base[keep].view(torch.int32)), (
            f"perturbing row {probe} moved another row's bits"
        )
        expected = dsa_causal_bound_torch_oracle(moved_scores, moved_clen, POOL_SIZE)
        assert torch.equal(got.view(torch.int32), expected.view(torch.int32))


def test_sentinel_marks_bounded_slots_and_planted_pads_across_tiles() -> None:
    """At 2,048 rows the ``-1`` count per row is ``max(0, k - complete)``, and a pad
    index planted in many tiles is marked without moving any other slot."""
    rows = TILED_ROWS
    scores = _tiled_scores(rows, 1061)
    clen = _tiled_causal_len(rows)
    bounded = dsa_causal_bound(scores, clen, POOL_SIZE)
    values, idx32 = _selection_of(bounded, SELECT_K)

    complete = _tiled_complete_pools(rows)
    expected_counts = (SELECT_K - complete).clamp(min=0).to(torch.int64)

    reset_causal_sentinel_dispatch_counters()
    marked = dsa_causal_sentinel(values, idx32, POOL_COLUMNS)
    assert causal_sentinel_dispatch_counters() == (1, 0), causal_sentinel_dispatch_counters()
    per_row = (marked == SENTINEL).sum(dim=1).to(torch.int64)
    assert torch.equal(per_row, expected_counts), (
        "the per-row sentinel count disagrees with max(0, k - complete pools)"
    )
    expected = dsa_causal_sentinel_torch_oracle(values, idx32, POOL_COLUMNS)
    assert torch.equal(marked, expected), "the sentinel differs from the oracle"

    # Plant a pad on every TILED_PAD_STRIDE-th row whose slots are all real
    # scores, so only the index arm can mark it.
    stride_rows = torch.arange(0, rows, TILED_PAD_STRIDE)
    planted = stride_rows[expected_counts[stride_rows] == 0]
    padded_idx = idx32.clone()
    padded_idx[planted, 0] = POOL_COLUMNS  # one past the last real pool
    reset_causal_sentinel_dispatch_counters()
    marked_pad = dsa_causal_sentinel(values, padded_idx, POOL_COLUMNS)
    assert causal_sentinel_dispatch_counters() == (1, 0), causal_sentinel_dispatch_counters()
    assert bool((marked_pad[planted, 0] == SENTINEL).all()), "a planted pad was kept"

    expected_pad = marked.clone()
    expected_pad[planted, 0] = SENTINEL
    assert torch.equal(marked_pad, expected_pad), (
        "planting a pad changed a slot it was not planted on"
    )
    assert torch.equal(
        marked_pad, dsa_causal_sentinel_torch_oracle(values, padded_idx, POOL_COLUMNS)
    )


def test_multi_tile_call_takes_one_kernel_dispatch_per_entry_point() -> None:
    """2,048 rows span many tiles, yet each entry point dispatches its own kernel once per call."""
    rows = TILED_ROWS
    scores = _tiled_scores(rows, 1071)
    clen = _tiled_causal_len(rows)
    tiles = row_tile_count(rows)
    assert tiles > 1, tiles

    reset_causal_bound_dispatch_counters()
    reset_causal_sentinel_dispatch_counters()
    assert causal_bound_kernel_identity() is None
    assert causal_sentinel_kernel_identity() is None

    assert can_run_dsa_causal_bound(scores, clen, POOL_SIZE) is True
    bounded = dsa_causal_bound(scores, clen, POOL_SIZE)
    assert tuple(bounded.shape) == (rows, POOL_COLUMNS), tuple(bounded.shape)
    assert bounded.dtype is torch.float32, bounded.dtype

    values, idx32 = _selection_of(bounded, SELECT_K)
    assert can_run_dsa_causal_sentinel(values, idx32, POOL_COLUMNS) is True
    marked = dsa_causal_sentinel(values, idx32, POOL_COLUMNS)
    assert tuple(marked.shape) == (rows, SELECT_K), tuple(marked.shape)
    assert marked.dtype is torch.int32, marked.dtype

    # The tile loop is inside the kernel: a host-side loop over 128-row slices
    # would read `tiles` dispatches here instead of 1.
    assert causal_bound_dispatch_counters() == (1, 0), (causal_bound_dispatch_counters(), tiles)
    assert causal_sentinel_dispatch_counters() == (1, 0), (
        causal_sentinel_dispatch_counters(), tiles
    )

    bound_id = causal_bound_kernel_identity()
    sentinel_id = causal_sentinel_kernel_identity()
    assert bound_id is not None and sentinel_id is not None
    assert bound_id[1] == "_causal_bound_nki", bound_id
    assert sentinel_id[1] == "_causal_sentinel_nki", sentinel_id
    assert bound_id[0].endswith("dsa.causal_bound"), bound_id
    assert sentinel_id[0].endswith("dsa.causal_bound"), sentinel_id

    # A second call dispatches a second time, so 1 is a per-call count and not a
    # saturated flag.
    dsa_causal_bound(scores, clen, POOL_SIZE)
    assert causal_bound_dispatch_counters() == (2, 0), causal_bound_dispatch_counters()
