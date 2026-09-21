# SPDX-License-Identifier: Apache-2.0
"""Tests for ``dsa_index_expand``: pool ids expanded to token indices.

The expansion is integer index arithmetic, so there is no tolerance to spend -- the kernel
is either bit-identical to the torch route or it is wrong. Every entry is either the ``-1``
sentinel, meaning "this column selects no token", or an index inside its own row's sequence.

Callers must keep every non-negative pool id below ``seq_lens[row] // pool_size``: upstream's
kernel gates only on ``pool_ids >= 0``, so a pool id past a row's last pool expands past the
end of the sequence, and both routes here reproduce that.

The emitted width is the raw width -- history columns plus the forced tail -- rounded up to
``KEY_CHUNK`` and padded with the same sentinel, because ``mla_sparse_attention`` refuses a
selected-row count that is not a whole number of chunks. Row counts above ``PARTITION_MAX``
are walked in tiles, since SBUF's partition axis holds no more rows than that.
"""

from __future__ import annotations

import ast
import pathlib

import pytest
import torch

from vllm_neuron.functional.attention import mla_sparse as ms
from vllm_neuron.functional.attention.mla_sparse import (
    KEY_CHUNK,
    MlaSparseAttentionError,
)
from vllm_neuron.functional.dsa.causal_bound import (
    can_run_dsa_causal_bound,
    can_run_dsa_causal_sentinel,
    causal_bound_dispatch_counters,
    causal_sentinel_dispatch_counters,
    dsa_causal_bound,
    dsa_causal_sentinel,
    reset_causal_bound_dispatch_counters,
    reset_causal_sentinel_dispatch_counters,
)
from vllm_neuron.functional.dsa.index_expand import (
    INDEX_KPOOL,
    PARTITION_MAX,
    IndexExpandError,
    _dsa_index_expand_torch,
    can_run_dsa_index_expand,
    dsa_index_expand,
    index_expand_dispatch_counters,
    index_expand_kernel_identity,
    index_expand_raw_width,
    index_expand_width,
    is_power_of_two,
    reset_index_expand_dispatch_counters,
    row_tile_count,
    row_tiles,
)
from vllm_neuron.functional.dsa.topk_select import (
    can_run_dsa_topk_select,
    dsa_topk_select,
    reset_topk_select_dispatch_counters,
    topk_select_dispatch_counters,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

# Tokens per pool: this checkpoint's compress ratio, and a power of two.
POOL_SIZE = 4

# Selected pools per row. The history loop is pool_size iterations however wide the
# selection is, so a large group count costs simulator time without a distinct reading;
# N_GROUPS_PRODUCTION is dsa_topk_select's select_k, the width that actually ships.
N_GROUPS_SMALL = 8
N_GROUPS_PRODUCTION = 512
N_GROUPS_TINY = 2

# The tiny geometry's two rows. At 12 tokens the tail count is 0, so every column past the
# history width is -1 and a tail sentinel cannot be told from a padded one; at 15 the tail
# count is 3, so the last raw column is a real token and the first padded column is -1.
SEQ_TINY = 12
SEQS_TINY = [SEQ_TINY, SEQ_TINY + 3]

# Every pool valid and both tail counts 0, so every tail column is the sentinel.
EVEN_TAIL_PIDS = [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]]
EVEN_TAIL_SEQS = [32, 40]

# Tail counts 1 and 3, so both ends of [1, pool_size) are covered.
SHORT_TAIL_PIDS = [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]]
SHORT_TAIL_SEQS = [33, 35]

# Both sentinel sources live at once: a negative pool id, and a tail column past the tail
# count. Tail counts are 1, 2 and 3.
SENTINEL_PIDS = [[0, 1, 2, 3, 4, 5, 6, -1], [-1, 1, 2, -1, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]]
SENTINEL_SEQS = [33, 34, 39]

# The width that ships: 2048 history columns, 2051 raw, 2176 emitted, tails 2 and 3.
PRODUCTION_PIDS = [list(range(N_GROUPS_PRODUCTION)), list(range(N_GROUPS_PRODUCTION))]
PRODUCTION_SEQS = [2050, 2051]

# Row counts for the tiling tests. ROWS_RAGGED is one tile plus four rows; ROWS_MAX is the
# longest extent served, 16 whole tiles, so it has no short last tile; ROWS_SHORT_LAST_TILE
# is the smallest extent that does have one -- 17 tiles, the last four rows tall -- which is
# where a remainder bug lives. Both are written as arithmetic on PARTITION_MAX so they cannot
# drift from the tile height.
ROWS_SMALL = 4
ROWS_ONE_TILE = PARTITION_MAX
ROWS_RAGGED = PARTITION_MAX + 4
ROWS_TWO_TILES = 2 * PARTITION_MAX
ROWS_MAX = 16 * PARTITION_MAX
ROWS_SHORT_LAST_TILE = 16 * PARTITION_MAX + 4

ROW_LADDER = [ROWS_SMALL, ROWS_ONE_TILE, ROWS_RAGGED, ROWS_TWO_TILES, ROWS_MAX]
MULTI_TILE_ROWS = [ROWS_RAGGED, ROWS_TWO_TILES, ROWS_MAX]

# Where the whole chain runs above one tile. Two row counts rather than ROWS_MAX: the reading
# is that the chain runs above the tile height, and the selector's cost at 2048 rows buys no
# new fact. Its scores are CHAIN_WIDTH wide because the selector needs 0 < k < width.
CHAIN_ROWS = [ROWS_RAGGED, ROWS_TWO_TILES]
CHAIN_WIDTH = 32

# Row i of a tiling fixture is SEQ_FLOOR + i % SEQ_PERIOD tokens long. SEQ_FLOOR is just
# enough complete pools that every planted pool id is legal. 17 does not divide the tile
# height, so every tile past the first holds different lengths than the first and the tail
# remainder cycles through every value. A -1 pool id is planted on every fifth (row + group),
# a stride coprime with both the group count and the tile height, so the negative-id branch
# is live in every tile at a different offset.
SEQ_FLOOR = N_GROUPS_SMALL * POOL_SIZE
SEQ_PERIOD = 17
PID_SENTINEL_STRIDE = 5


def _tensors(pids: list[list[int]], seqs: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """One case's two inputs at the shapes the seam declares: ``[rows, n_groups]``, ``[rows]``."""
    return (torch.tensor(pids, dtype=torch.int32), torch.tensor(seqs, dtype=torch.int32))


def _seq_lens(rows: int) -> torch.Tensor:
    """``[rows]`` int32 sequence lengths for the tiling tests, one per selected row."""
    return torch.tensor(
        [SEQ_FLOOR + (i % SEQ_PERIOD) for i in range(rows)], dtype=torch.int32
    )


def _pool_ids(rows: int) -> torch.Tensor:
    """``[rows, N_GROUPS_SMALL]`` int32 pool ids, ``-1`` planted on a coprime stride."""
    return torch.tensor(
        [
            [-1 if (i + g) % PID_SENTINEL_STRIDE == 0 else (g + i) % N_GROUPS_SMALL
             for g in range(N_GROUPS_SMALL)]
            for i in range(rows)
        ],
        dtype=torch.int32,
    )


def _diff(got: torch.Tensor, expected: torch.Tensor) -> int:
    """Max abs difference in int64, so nothing wraps."""
    return int((got.to(torch.int64) - expected.to(torch.int64)).abs().max().item())


def _range_counts(got: torch.Tensor, seqs: list[int]) -> tuple[int, int, int, int]:
    """``(in_range, out_of_bounds, sentinels, population)``, the sentinel excluded by name.

    ``in_range`` and ``out_of_bounds`` are counted over non-sentinel entries only, per row
    against that row's own sequence length.
    """
    in_range = out_of_bounds = sentinels = 0
    for r, limit in enumerate(seqs):
        for value in got[r].tolist():
            value = int(value)
            if value == -1:
                sentinels += 1
            elif 0 <= value < limit:
                in_range += 1
            else:
                out_of_bounds += 1
    return (in_range, out_of_bounds, sentinels, int(got.numel()))


def _precondition_violations(
    pids: list[list[int]], seqs: list[int], pool_size: int
) -> list[tuple[int, int]]:
    """Every ``(row, pool_id)`` breaking the caller precondition. Empty means the ids are legal."""
    return [
        (r, int(p))
        for r in range(len(pids))
        for p in pids[r]
        if int(p) != -1 and not (0 <= int(p) < seqs[r] // pool_size)
    ]


def _python_oracle(pids: list[list[int]], seqs: list[int], pool_size: int) -> list[list[int]]:
    """A third spelling of the expansion: plain python loops, no torch, no vectorisation.

    The module's ``_dsa_index_expand_torch`` is upstream's ``where`` form and the kernel is a
    closed form in max and min; this is neither. The width comes from the module, and the
    padded columns need no branch of their own: past the raw width the tail offset is at least
    ``pool_size - 1`` while the tail count is at most that, so the ``else`` already appends -1.
    """
    rows = len(pids)
    n_groups = len(pids[0])
    topk = n_groups * pool_size
    rowsout = []
    for r in range(rows):
        seq = seqs[r]
        tail_start = (seq // pool_size) * pool_size
        tail_count = seq - tail_start
        row = []
        for col in range(index_expand_width(n_groups, pool_size)):
            if col < topk:
                pid = pids[r][col // pool_size]
                row.append(pid * pool_size + col % pool_size if pid >= 0 else -1)
            else:
                offset = col - topk
                row.append(tail_start + offset if offset < tail_count else -1)
        rowsout.append(row)
    return rowsout


# ---------------------------------------------------------------------------------------------
# Even tail: every pool valid, both tail counts zero
# ---------------------------------------------------------------------------------------------


def test_even_tail_expansion_is_bit_identical():
    """An even tail expands bit-identically to the torch route, at the emitted width."""
    reset_index_expand_dispatch_counters()
    pids, seqs = _tensors(EVEN_TAIL_PIDS, EVEN_TAIL_SEQS)
    got = dsa_index_expand(pids, seqs, POOL_SIZE)
    expected = _dsa_index_expand_torch(pids, seqs, POOL_SIZE)
    assert got.shape == (len(EVEN_TAIL_SEQS), index_expand_width(N_GROUPS_SMALL, POOL_SIZE))
    assert got.dtype is torch.int32
    assert _diff(got, expected) == 0


def test_even_tail_entries_are_sentinel_or_in_range():
    """No entry is out of its own row's range, with the sentinel excluded by name."""
    reset_index_expand_dispatch_counters()
    pids, seqs = _tensors(EVEN_TAIL_PIDS, EVEN_TAIL_SEQS)
    got = dsa_index_expand(pids, seqs, POOL_SIZE)
    in_range, oob, sentinels, population = _range_counts(got, EVEN_TAIL_SEQS)
    assert oob == 0
    assert in_range + oob + sentinels == population


def test_even_tail_columns_are_all_sentinel():
    """Both tail counts are 0 here, so every tail column is ``-1``."""
    pids, seqs = _tensors(EVEN_TAIL_PIDS, EVEN_TAIL_SEQS)
    got = dsa_index_expand(pids, seqs, POOL_SIZE)
    tail = got[:, N_GROUPS_SMALL * POOL_SIZE:]
    assert [s % POOL_SIZE for s in EVEN_TAIL_SEQS] == [0, 0]
    assert bool((tail == -1).all().item())


def test_even_tail_takes_the_kernel_in_one_dispatch():
    """The gate admits the even-tail shape and the kernel runs once, with no torch fallback."""
    reset_index_expand_dispatch_counters()
    pids, seqs = _tensors(EVEN_TAIL_PIDS, EVEN_TAIL_SEQS)
    admitted = can_run_dsa_index_expand(pids, seqs, POOL_SIZE)
    dsa_index_expand(pids, seqs, POOL_SIZE)
    nki_n, fallback_n = index_expand_dispatch_counters()
    assert admitted is True
    assert nki_n == 1
    assert fallback_n == 0


# ---------------------------------------------------------------------------------------------
# Short tail: a sequence length that is not a multiple of the pool size
# ---------------------------------------------------------------------------------------------


def test_short_tail_expansion_is_bit_identical():
    """A non-multiple tail expands bit-identically to the torch route."""
    reset_index_expand_dispatch_counters()
    pids, seqs = _tensors(SHORT_TAIL_PIDS, SHORT_TAIL_SEQS)
    got = dsa_index_expand(pids, seqs, POOL_SIZE)
    expected = _dsa_index_expand_torch(pids, seqs, POOL_SIZE)
    assert _diff(got, expected) == 0


def test_short_tail_entries_are_sentinel_or_in_range():
    """No entry is out of its own row's range when the tail is a partial pool."""
    reset_index_expand_dispatch_counters()
    pids, seqs = _tensors(SHORT_TAIL_PIDS, SHORT_TAIL_SEQS)
    got = dsa_index_expand(pids, seqs, POOL_SIZE)
    in_range, oob, sentinels, population = _range_counts(got, SHORT_TAIL_SEQS)
    assert oob == 0
    assert in_range + oob + sentinels == population


def test_short_tail_carries_one_and_three_tokens():
    """The tail carries 1 token on row 0 and 3 on row 1, both ends of ``[1, pool_size)``."""
    pids, seqs = _tensors(SHORT_TAIL_PIDS, SHORT_TAIL_SEQS)
    got = dsa_index_expand(pids, seqs, POOL_SIZE)
    tail = got[:, N_GROUPS_SMALL * POOL_SIZE:]
    carried = [int((row != -1).sum().item()) for row in tail]
    assert carried == [1, 3]
    assert carried == [s % POOL_SIZE for s in SHORT_TAIL_SEQS]


def test_short_tail_takes_the_kernel_in_one_dispatch():
    """The gate admits the short-tail shape and the kernel runs once, with no torch fallback."""
    reset_index_expand_dispatch_counters()
    pids, seqs = _tensors(SHORT_TAIL_PIDS, SHORT_TAIL_SEQS)
    admitted = can_run_dsa_index_expand(pids, seqs, POOL_SIZE)
    dsa_index_expand(pids, seqs, POOL_SIZE)
    nki_n, fallback_n = index_expand_dispatch_counters()
    assert admitted is True
    assert nki_n == 1
    assert fallback_n == 0


# ---------------------------------------------------------------------------------------------
# Negative pool ids, alongside tail sentinels
# ---------------------------------------------------------------------------------------------


def test_negative_pool_ids_expand_bit_identically():
    """Negative pool ids expand bit-identically to the torch route."""
    reset_index_expand_dispatch_counters()
    pids, seqs = _tensors(SENTINEL_PIDS, SENTINEL_SEQS)
    got = dsa_index_expand(pids, seqs, POOL_SIZE)
    expected = _dsa_index_expand_torch(pids, seqs, POOL_SIZE)
    assert _diff(got, expected) == 0


def test_negative_pool_id_entries_are_sentinel_or_in_range():
    """No entry is out of range with sentinels arriving from both sources at once."""
    reset_index_expand_dispatch_counters()
    pids, seqs = _tensors(SENTINEL_PIDS, SENTINEL_SEQS)
    got = dsa_index_expand(pids, seqs, POOL_SIZE)
    in_range, oob, sentinels, population = _range_counts(got, SENTINEL_SEQS)
    assert oob == 0
    assert in_range + oob + sentinels == population


def test_negative_pool_ids_become_exactly_minus_one():
    """Every column of a ``-1`` pool id reads exactly ``-1``.

    This is the closed form's whole claim: ``max(pid * pool_size + o, -1)`` pins a negative
    pool id to the sentinel without a compare and without a select. A leak would show up here
    as a value like ``-4`` or ``-3``.
    """
    pids, seqs = _tensors(SENTINEL_PIDS, SENTINEL_SEQS)
    got = dsa_index_expand(pids, seqs, POOL_SIZE)
    leaked: list[int] = []
    sentinel_columns = 0
    for r, row in enumerate(SENTINEL_PIDS):
        for g, pid in enumerate(row):
            if pid >= 0:
                continue
            sentinel_columns += POOL_SIZE
            block = got[r, g * POOL_SIZE:(g + 1) * POOL_SIZE].tolist()
            leaked.extend(int(v) for v in block if int(v) != -1)
    assert sentinel_columns == 12
    assert leaked == []


def test_negative_pool_ids_take_the_kernel_in_one_dispatch():
    """The gate admits negative pool ids and the kernel runs once, with no torch fallback."""
    reset_index_expand_dispatch_counters()
    pids, seqs = _tensors(SENTINEL_PIDS, SENTINEL_SEQS)
    admitted = can_run_dsa_index_expand(pids, seqs, POOL_SIZE)
    dsa_index_expand(pids, seqs, POOL_SIZE)
    nki_n, fallback_n = index_expand_dispatch_counters()
    assert admitted is True
    assert nki_n == 1
    assert fallback_n == 0


# ---------------------------------------------------------------------------------------------
# The production width
# ---------------------------------------------------------------------------------------------


def test_production_width_expansion_is_bit_identical():
    """512 selected pools: 2051 raw columns emitted as 2176, bit-identical to the torch route.

    The 2051 raw columns carry every meaningful value; the 125 after them are the ``-1``
    padding that makes the width a whole number of ``KEY_CHUNK`` columns. Both the derived and
    the literal width are asserted: the derived one catches a change in the rule, the literal
    one catches a change that happens to leave the derivation self-consistent.
    """
    reset_index_expand_dispatch_counters()
    pids, seqs = _tensors(PRODUCTION_PIDS, PRODUCTION_SEQS)
    got = dsa_index_expand(pids, seqs, POOL_SIZE)
    expected = _dsa_index_expand_torch(pids, seqs, POOL_SIZE)
    assert got.shape == (2, index_expand_width(N_GROUPS_PRODUCTION, POOL_SIZE))
    assert int(got.shape[1]) == 2176
    assert index_expand_raw_width(N_GROUPS_PRODUCTION, POOL_SIZE) == 2051
    assert _diff(got, expected) == 0


def test_production_width_entries_are_sentinel_or_in_range():
    """No entry is out of range over two rows of the padded production width."""
    reset_index_expand_dispatch_counters()
    pids, seqs = _tensors(PRODUCTION_PIDS, PRODUCTION_SEQS)
    got = dsa_index_expand(pids, seqs, POOL_SIZE)
    in_range, oob, sentinels, population = _range_counts(got, PRODUCTION_SEQS)
    assert oob == 0
    assert in_range + oob + sentinels == population
    assert population == 2 * index_expand_width(N_GROUPS_PRODUCTION, POOL_SIZE)


def test_production_width_tail_carries_two_and_three_tokens():
    """The tail carries 2 tokens on row 0 and 3 on row 1."""
    pids, seqs = _tensors(PRODUCTION_PIDS, PRODUCTION_SEQS)
    got = dsa_index_expand(pids, seqs, POOL_SIZE)
    tail = got[:, N_GROUPS_PRODUCTION * POOL_SIZE:]
    carried = [int((row != -1).sum().item()) for row in tail]
    assert carried == [2, 3]
    assert carried == [s % POOL_SIZE for s in PRODUCTION_SEQS]


def test_production_width_takes_the_kernel_in_one_dispatch():
    """The gate admits the production width and the kernel runs once, with no torch fallback."""
    reset_index_expand_dispatch_counters()
    pids, seqs = _tensors(PRODUCTION_PIDS, PRODUCTION_SEQS)
    admitted = can_run_dsa_index_expand(pids, seqs, POOL_SIZE)
    dsa_index_expand(pids, seqs, POOL_SIZE)
    nki_n, fallback_n = index_expand_dispatch_counters()
    assert admitted is True
    assert nki_n == 1
    assert fallback_n == 0


# ---------------------------------------------------------------------------------------------
# The edge of the caller precondition, and the reference the routes are read against
# ---------------------------------------------------------------------------------------------


def test_pool_id_past_the_last_pool_expands_past_the_sequence_end():
    """A pool id past its row's last pool expands out of range, on the kernel and the reference.

    Upstream gates only on ``pool_ids >= 0``, so this is the behaviour to reproduce rather
    than an error to raise: four entries land past the end of the sequence and the two routes
    agree on all of them.
    """
    reset_index_expand_dispatch_counters()
    pids_raw = [[0, 1, 2, 3, 4, 5, 6, 20]]
    seqs_raw = [33]
    pids, seqs = _tensors(pids_raw, seqs_raw)
    got = dsa_index_expand(pids, seqs, POOL_SIZE)
    expected = _dsa_index_expand_torch(pids, seqs, POOL_SIZE)
    in_range, oob, sentinels, population = _range_counts(got, seqs_raw)
    nki_n, fallback_n = index_expand_dispatch_counters()
    assert _diff(got, expected) == 0
    assert oob == 4
    assert in_range + oob + sentinels == population
    assert nki_n == 1
    assert fallback_n == 0


def test_torch_route_matches_a_plain_python_expansion():
    """The torch route agrees with a third spelling of the index arithmetic."""
    pids, seqs = _tensors(SENTINEL_PIDS, SENTINEL_SEQS)
    expected = _dsa_index_expand_torch(pids, seqs, POOL_SIZE)
    assert expected.tolist() == _python_oracle(SENTINEL_PIDS, SENTINEL_SEQS, POOL_SIZE)


# ---------------------------------------------------------------------------------------------
# The gate, the route it selects, and the power-of-two rule it enforces
# ---------------------------------------------------------------------------------------------


def test_non_power_of_two_pool_size_is_refused_and_served_by_torch():
    """A ``pool_size`` of 3 is refused by the gate, served by torch, and still correct.

    3 rather than a bad dtype because it is the refusal this module exists to make: the tail
    derivation ``seq & (pool_size - 1)`` is exact only for a power of two.
    """
    reset_index_expand_dispatch_counters()
    pids, seqs = _tensors([[0, 1, 2]], [9])
    admitted = can_run_dsa_index_expand(pids, seqs, 3)
    got = dsa_index_expand(pids, seqs, 3)
    expected = _dsa_index_expand_torch(pids, seqs, 3)
    nki_n, fallback_n = index_expand_dispatch_counters()
    assert admitted is False
    assert nki_n == 0
    assert fallback_n == 1
    assert _diff(got, expected) == 0


def test_kernel_identity_is_none_before_any_dispatch():
    """Before any dispatch the identity is ``None``, so "no kernel ran" is distinguishable."""
    reset_index_expand_dispatch_counters()
    assert index_expand_kernel_identity() is None


def test_kernel_identity_after_dispatch_names_the_nki_kernel():
    """After a dispatch the identity names this module's own kernel, read back through the seam."""
    reset_index_expand_dispatch_counters()
    pids, seqs = _tensors(EVEN_TAIL_PIDS, EVEN_TAIL_SEQS)
    dsa_index_expand(pids, seqs, POOL_SIZE)
    identity = index_expand_kernel_identity()
    assert identity is not None
    module_name, qualname = identity
    assert module_name.endswith("index_expand")
    assert qualname == "_index_expand_nki"


def test_gate_follows_the_house_kernel_predicate():
    """The gate consults the house predicate, so a host with no kernels is served by torch."""
    pids, seqs = _tensors(EVEN_TAIL_PIDS, EVEN_TAIL_SEQS)
    house = bool(can_run_kernel())
    assert can_run_dsa_index_expand(pids, seqs, POOL_SIZE) is house


def test_gate_admits_every_fixture_shape():
    """The gate admits the shape and dtype of all four geometries above."""
    verdicts = []
    for pids_raw, seqs_raw in (
        (EVEN_TAIL_PIDS, EVEN_TAIL_SEQS),
        (SHORT_TAIL_PIDS, SHORT_TAIL_SEQS),
        (SENTINEL_PIDS, SENTINEL_SEQS),
        (PRODUCTION_PIDS, PRODUCTION_SEQS),
    ):
        pids, seqs = _tensors(pids_raw, seqs_raw)
        verdicts.append(can_run_dsa_index_expand(pids, seqs, POOL_SIZE))
    assert verdicts == [True, True, True, True]


def test_gate_refuses_a_non_power_of_two_pool_size():
    """The gate refuses ``pool_size`` 6, because ``seq & (pool_size - 1)`` is not its remainder."""
    pids, seqs = _tensors([[0, 1]], [12])
    assert can_run_dsa_index_expand(pids, seqs, 6) is False
    assert is_power_of_two(6) is False


def test_gate_refuses_an_unadmitted_index_dtype():
    """The gate refuses int64 indices; the torch route serves them."""
    pids = torch.tensor(EVEN_TAIL_PIDS, dtype=torch.int64)
    seqs = torch.tensor(EVEN_TAIL_SEQS, dtype=torch.int32)
    assert can_run_dsa_index_expand(pids, seqs, POOL_SIZE) is False


def test_is_power_of_two_accepts_powers_of_two_only():
    """``is_power_of_two`` accepts 1, 2, 4, 8, 512 and refuses 0, 3, 6, 12, -4."""
    accepted = [n for n in (1, 2, 4, 8, 512) if is_power_of_two(n)]
    refused = [n for n in (0, 3, 6, 12, -4) if not is_power_of_two(n)]
    assert accepted == [1, 2, 4, 8, 512]
    assert refused == [0, 3, 6, 12, -4]


def test_index_kpool_is_the_checkpoint_compress_ratio():
    """``INDEX_KPOOL`` is the checkpoint value 4, and it is a power of two."""
    assert INDEX_KPOOL == 4
    assert is_power_of_two(INDEX_KPOOL)


# ---------------------------------------------------------------------------------------------
# Malformed calls
# ---------------------------------------------------------------------------------------------


def test_refuses_a_non_2d_pool_ids():
    """A 1-D ``pool_ids`` is a malformed call, not a shape to guess at."""
    with pytest.raises(IndexExpandError, match="2-D"):
        dsa_index_expand(torch.tensor([0, 1], dtype=torch.int32),
                         torch.tensor([8], dtype=torch.int32), POOL_SIZE)


def test_refuses_a_non_1d_seq_lens():
    """A 2-D ``seq_lens`` is refused: upstream's shape is ``[rows]``, and the seam reshapes it."""
    with pytest.raises(IndexExpandError, match="1-D"):
        dsa_index_expand(torch.tensor([[0, 1]], dtype=torch.int32),
                         torch.tensor([[8]], dtype=torch.int32), POOL_SIZE)


def test_refuses_a_seq_lens_that_does_not_match_the_row_count():
    """One sequence length per row, or the call is malformed."""
    with pytest.raises(IndexExpandError, match="one length per row"):
        dsa_index_expand(torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
                         torch.tensor([8], dtype=torch.int32), POOL_SIZE)


def test_refuses_a_non_positive_pool_size():
    """A ``pool_size`` of 0 is malformed rather than merely unadmitted."""
    with pytest.raises(IndexExpandError, match="pool_size must be positive"):
        dsa_index_expand(torch.tensor([[0, 1]], dtype=torch.int32),
                         torch.tensor([8], dtype=torch.int32), 0)


# ---------------------------------------------------------------------------------------------
# The emitted width, and the sparse attention kernel that consumes it
# ---------------------------------------------------------------------------------------------


def _admit(topk: int) -> None:
    """Ask the sparse kernel's own gate whether it serves a selected-row count of ``topk``.

    Every other axis is held at a value the gate admits -- this checkpoint's latent rank and
    RoPE width, one query, one cache row, one head -- so a raise can only be about ``topk``.
    The gate is called rather than re-implemented, or the test would assert that one copy of
    the rule agrees with another.
    """
    ms._require_admissible(
        seq=1,
        heads=1,
        latent=ms.TARGET_LATENT_RANK,
        rope=ms.TARGET_ROPE_WIDTH,
        topk=topk,
        s_kv=1,
        softmax_scale=1.0,
    )


def test_emitted_width_is_admitted_by_the_sparse_attention_gate():
    """The emitted width is a positive multiple of ``KEY_CHUNK`` and the sparse gate admits it.

    The raw width is refused by that same gate, which is what the rounding exists for: the raw
    width is ``pool_size - 1`` mod ``pool_size``, so it is never a whole number of chunks.
    """
    reset_index_expand_dispatch_counters()
    for n_groups, seqs in (
        (N_GROUPS_TINY, SEQS_TINY),
        (N_GROUPS_PRODUCTION, PRODUCTION_SEQS),
    ):
        pids = [list(range(n_groups)) for _ in seqs]
        got = dsa_index_expand(*_tensors(pids, seqs), POOL_SIZE)
        emitted = int(got.shape[1])
        raw = index_expand_raw_width(n_groups, POOL_SIZE)

        assert emitted == index_expand_width(n_groups, POOL_SIZE)
        assert emitted > 0 and emitted % KEY_CHUNK == 0
        _admit(emitted)

        assert raw % KEY_CHUNK != 0
        with pytest.raises(MlaSparseAttentionError, match="positive multiple"):
            _admit(raw)


def test_padding_is_sentinel_and_both_routes_agree():
    """The first ``width_raw`` columns carry the expansion, every column beyond is ``-1``.

    Read on both routes, and the routes are compared with each other. The boundary is read on
    row 1 of each geometry, whose tail count fills the tail region completely, so the last raw
    column is a real token index rather than a tail sentinel: an off-by-one in the memset start
    clobbers that token, and one the other way leaves a column unwritten.
    """
    reset_index_expand_dispatch_counters()
    for n_groups, seqs in (
        (N_GROUPS_TINY, SEQS_TINY),
        (N_GROUPS_PRODUCTION, PRODUCTION_SEQS),
    ):
        pids = [list(range(n_groups)) for _ in seqs]
        tensors = _tensors(pids, seqs)
        kernel = dsa_index_expand(*tensors, POOL_SIZE)
        torch_route = _dsa_index_expand_torch(*tensors, POOL_SIZE)
        oracle = torch.tensor(_python_oracle(pids, seqs, POOL_SIZE), dtype=torch.int32)
        raw = index_expand_raw_width(n_groups, POOL_SIZE)

        pad_kernel = kernel[:, raw:]
        pad_torch = torch_route[:, raw:]
        last_raw = int(kernel[1, raw - 1].item())

        assert _diff(kernel[:, :raw], oracle[:, :raw]) == 0
        assert _diff(torch_route[:, :raw], oracle[:, :raw]) == 0
        assert int(pad_kernel.shape[1]) == int(kernel.shape[1]) - raw
        assert bool((pad_kernel == -1).all().item())
        assert bool((pad_torch == -1).all().item())
        assert last_raw != -1
        assert int(torch_route[1, raw - 1].item()) == last_raw
        assert int(kernel[1, raw].item()) == -1
        assert _diff(kernel, torch_route) == 0
        assert kernel.shape == torch_route.shape


_SPARSE_TEST = pathlib.Path(__file__).resolve().parents[1] / "attention" / "test_mla_sparse.py"


def _sparse_test_definitions(*names: str) -> dict:
    """The named top-level definitions of the sparse attention tests, read from disk.

    The tolerance pair, the float64 reference, the case builder and the geometry all arrive as
    that file's own bytes rather than retyped here, so the two files cannot drift apart -- and
    a retyped tolerance is a tolerance that can be nudged to reach green. Parsing rather than
    importing keeps that file's module-level ``nki`` imports out of this one. The namespace
    holds ``torch`` and nothing else, so an extracted definition that quietly depended on
    something else in its home module fails loudly instead of picking up a stand-in.
    """
    src = _SPARSE_TEST.read_text(encoding="utf-8")
    ns: dict = {"torch": torch}
    for node in ast.parse(src).body:
        declared = None
        if isinstance(node, ast.FunctionDef):
            declared = node.name
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            declared = node.targets[0].id
        if declared in names:
            exec(ast.get_source_segment(src, node), ns)  # noqa: S102 -- see the docstring
    missing = [n for n in names if n not in ns]
    assert not missing, f"not declared at top level in {_SPARSE_TEST.name}: {missing}"
    return ns


def test_padded_rows_through_the_sparse_kernel_match_a_float64_reference():
    """The padded rows, fed through ``mla_sparse_attention``, match a float64 reference.

    The tiny geometry's emitted width is 128, which is exactly the selected-row count the
    sparse kernel's own sentinel case runs at, so this module's padded output is a drop-in for
    the index tensor that case already exercises -- the only difference is that the ``-1``
    columns come from padding a real expansion. The reference reads only the live columns; a
    second reference that attends the padded columns as cache row 0 must disagree, or the
    comparison says nothing about whether the padding was masked at all.

    No row here is wholly sentinel, so the all-sentinel rule is not exercised; it is read in
    the sparse kernel's own tests.
    """
    cited = _sparse_test_definitions("RTOL", "ATOL", "SENTINEL_UNTILED", "case_scale", "make_case",
                                     "sparse_mla_torch_reference", "sentinel_reference",
                                     "attending_reference")
    case = dict(cited["SENTINEL_UNTILED"])
    rtol, atol = cited["RTOL"], cited["ATOL"]

    emitted = index_expand_width(N_GROUPS_TINY, POOL_SIZE)
    assert emitted == case["topk"]

    # The kernel's own index tensor is discarded: the point is that the index rows come from
    # this module's padded expansion instead.
    q_lift, c_kv, _discarded_idx, q_pe, k_pe = cited["make_case"](**case, seed=102)
    assert q_pe is None and k_pe is None, "the sentinel geometry is at rope width 0"

    rows = case["seq"]
    pids = [list(range(N_GROUPS_TINY)) for _ in range(rows)]
    seqs = [SEQS_TINY[r % len(SEQS_TINY)] for r in range(rows)]
    reset_index_expand_dispatch_counters()
    ms.reset_mla_sparse_dispatch_counters()
    ms.reset_mla_sparse_tiled_dispatch_counters()
    idx = dsa_index_expand(*_tensors(pids, seqs), POOL_SIZE)
    assert idx.shape == (rows, emitted)

    live_per_row = [int((idx[r] >= 0).sum().item()) for r in range(rows)]
    scale = cited["case_scale"](case)
    assert ms.can_run_mla_sparse_attention(
        q_lift, case["seq"], case["heads"], case["latent"], case["rope"],
        case["topk"], case["s_kv"], scale,
    ), "the sparse gate refused the padded width, so nothing below would mean anything"

    got = ms.mla_sparse_attention(q_lift, c_kv, idx, scale)
    ref = cited["sentinel_reference"](q_lift, c_kv, idx, scale)
    attending = cited["attending_reference"](q_lift, c_kv, idx, scale)
    wrong = float((got - attending).abs().max())

    assert int((idx < 0).sum().item()) == idx.numel() - sum(live_per_row)
    assert min(live_per_row) > 0
    assert sum(live_per_row) < idx.numel()
    torch.testing.assert_close(got, ref, rtol=rtol, atol=atol)
    assert wrong > atol, (
        f"the reference that attends the padded columns agrees with the kernel to {wrong:.3e}, so "
        f"this comparison is blind. Report it; never widen the tolerance to absorb it"
    )
    assert index_expand_dispatch_counters() == (1, 0)
    assert ms.mla_sparse_dispatch_counters() == (1, 0)
    assert ms.mla_sparse_tiled_dispatch_counters() == (0, 0)
    assert got.shape == (case["seq"], case["heads"], case["latent"])


def test_emitted_width_equals_upstream_buffer_width():
    """The emitted width equals upstream's allocated buffer width, by upstream's own expression.

    Upstream, at ``878631b6``, ``models/glm5next/nvidia/model.py:594-599``::

        buffer_width = topk_tokens + (kpool - 1 if kpool > 1 else 0)
        buffer_width = ceil(buffer_width / 128) * 128

    written out in that form rather than by calling this module's function, because the reading
    is that two independently spelled expressions land on the same number. The raw widths
    agreeing is the exact comparison -- an off-by-one in either breaks it -- while the emitted
    widths agreeing is weaker, since rounding to ``KEY_CHUNK`` absorbs a one-column error.
    """

    def upstream_raw_width(topk_tokens: int, kpool: int) -> int:
        return topk_tokens + (kpool - 1 if kpool > 1 else 0)

    def upstream_buffer_width(topk_tokens: int, kpool: int, block: int = 128) -> int:
        return ((upstream_raw_width(topk_tokens, kpool) + block - 1) // block) * block

    for n_groups in (N_GROUPS_TINY, N_GROUPS_PRODUCTION):
        topk_tokens = n_groups * POOL_SIZE
        fork_raw = index_expand_raw_width(n_groups, POOL_SIZE)
        up_raw = upstream_raw_width(topk_tokens, POOL_SIZE)
        assert fork_raw == up_raw
        assert fork_raw == topk_tokens + POOL_SIZE - 1
        assert index_expand_width(n_groups, POOL_SIZE) == upstream_buffer_width(
            topk_tokens, POOL_SIZE
        )


# ---------------------------------------------------------------------------------------------
# Row tiling: SBUF's partition axis holds PARTITION_MAX rows, so more rows are walked in tiles
# ---------------------------------------------------------------------------------------------


def test_multi_tile_row_counts_are_served_in_one_dispatch() -> None:
    """Row counts that need more than one tile are each served by the kernel in one dispatch.

    A host-side loop over 128-row slices would serve the same shapes and read ``len(tiles)``
    dispatches, which is the number that tells the two designs apart.
    """
    out_cols = index_expand_width(N_GROUPS_SMALL, POOL_SIZE)
    assert out_cols % KEY_CHUNK == 0, (out_cols, KEY_CHUNK)

    for rows in MULTI_TILE_ROWS:
        pool_ids = _pool_ids(rows)
        seq_lens = _seq_lens(rows)
        tiles = row_tiles(rows)
        assert row_tile_count(rows) == len(tiles), (row_tile_count(rows), len(tiles))
        assert len(tiles) > 1, f"rows={rows} must need more than one tile"

        reset_index_expand_dispatch_counters()
        assert can_run_dsa_index_expand(pool_ids, seq_lens, POOL_SIZE) is True
        got = dsa_index_expand(pool_ids, seq_lens, POOL_SIZE)
        assert tuple(got.shape) == (rows, out_cols), tuple(got.shape)
        assert got.dtype is torch.int32, got.dtype
        assert index_expand_dispatch_counters() == (1, 0), (rows, len(tiles))


def test_tiled_expansion_matches_the_torch_route_at_every_row_count() -> None:
    """The kernel and the torch route agree bit for bit from one tile up to the longest extent.

    The padded columns are read separately as all ``-1``, so the pad region cannot pass by
    being compared only with itself.
    """
    raw_cols = index_expand_raw_width(N_GROUPS_SMALL, POOL_SIZE)
    out_cols = index_expand_width(N_GROUPS_SMALL, POOL_SIZE)
    assert out_cols > raw_cols, (out_cols, raw_cols)

    for rows in ROW_LADDER:
        pool_ids = _pool_ids(rows)
        seq_lens = _seq_lens(rows)

        reset_index_expand_dispatch_counters()
        got = dsa_index_expand(pool_ids, seq_lens, POOL_SIZE)
        expected = _dsa_index_expand_torch(pool_ids, seq_lens, POOL_SIZE)
        assert torch.equal(got, expected), (
            f"rows={rows}: the kernel and the torch route must agree bit for bit; first "
            f"differing entry {(got != expected).nonzero()[:1].tolist()}"
        )
        assert index_expand_dispatch_counters() == (1, 0)

        pad = got[:, raw_cols:out_cols]
        assert int((pad != -1).sum()) == 0, (
            f"rows={rows}: {int((pad != -1).sum())} padded columns are not the sentinel"
        )


def test_rows_stay_independent_across_a_tile_boundary() -> None:
    """Perturbing one row's ids and length moves that row's output row and nothing else.

    Two row counts, one probe each, because one cannot carry both readings: 2048 rows is 16
    whole tiles and is probed at the first row of the second tile, where a boundary bug shows
    first; 2052 rows has a 4-row last tile and is probed inside it, where a remainder bug shows.
    The probe row is derived from the tile list rather than typed, so the claim about which tile
    it lands in cannot go stale. Both inputs are perturbed because they enter by different
    routes -- the ids as the tile itself, the length as the per-row column operand.
    """
    for rows, short_last_tile in ((ROWS_MAX, False), (ROWS_SHORT_LAST_TILE, True)):
        tiles = row_tiles(rows)
        if short_last_tile:
            probe = tiles[-1][0] + 2
            assert tiles[-1][1] < PARTITION_MAX, (rows, tiles[-1])
            assert tiles[-1][1] == rows % PARTITION_MAX, (rows, tiles[-1])
            assert tiles[-1][0] <= probe < rows, (probe, tiles[-1])
        else:
            probe = tiles[1][0]
            assert tiles[-1][1] == PARTITION_MAX, (rows, tiles[-1])
            assert rows % PARTITION_MAX == 0, rows
        probe_tile = probe // PARTITION_MAX
        assert (probe_tile == len(tiles) - 1) is short_last_tile, (probe, probe_tile, len(tiles))

        pool_ids = _pool_ids(rows)
        seq_lens = _seq_lens(rows)
        base = dsa_index_expand(pool_ids, seq_lens, POOL_SIZE)

        # Neither perturbation can pass vacuously: the probe row selects a live pool and has a
        # non-empty tail, so both the ids and the length reach its output row.
        assert int((pool_ids[probe] >= 0).sum()) > 0, f"probe row {probe} selects no pool at all"
        assert int(seq_lens[probe]) % POOL_SIZE > 0, f"probe row {probe} has an empty tail"

        moved_ids = pool_ids.clone()
        moved_ids[probe, 0] = (int(pool_ids[probe, 0]) + 1) % N_GROUPS_SMALL
        moved_lens = seq_lens.clone()
        moved_lens[probe] = SEQ_FLOOR + ((int(seq_lens[probe]) - SEQ_FLOOR + 1) % SEQ_PERIOD)
        assert int(moved_lens[probe]) != int(seq_lens[probe])

        got = dsa_index_expand(moved_ids, moved_lens, POOL_SIZE)
        assert not torch.equal(got[probe], base[probe]), \
            f"probe row {probe} did not move when its own ids and length did"
        keep = torch.ones(rows, dtype=torch.bool)
        keep[probe] = False
        assert torch.equal(got[keep], base[keep]), \
            f"perturbing row {probe} moved another row's bits"
        assert torch.equal(got, _dsa_index_expand_torch(moved_ids, moved_lens, POOL_SIZE))


def test_tiled_route_is_one_dispatch_and_the_chain_runs_on_device() -> None:
    """One dispatch per call at a tiled extent, and the whole selection chain above one tile.

    The four stages run in order -- bound, select, sentinel, expand -- at 132 and 256 rows, and
    each is asserted to have taken its own kernel with no fallback. Nothing substitutes a torch
    reference for a kernel that refused: a stage that cannot serve these row counts is a finding
    about that kernel rather than a reason to fall back.
    """
    rows = ROWS_MAX
    tiles = row_tile_count(rows)
    assert tiles > 1, tiles

    reset_index_expand_dispatch_counters()
    assert index_expand_kernel_identity() is None
    pool_ids = _pool_ids(rows)
    seq_lens = _seq_lens(rows)
    assert can_run_dsa_index_expand(pool_ids, seq_lens, POOL_SIZE) is True
    dsa_index_expand(pool_ids, seq_lens, POOL_SIZE)
    assert index_expand_dispatch_counters() == (1, 0), tiles
    identity = index_expand_kernel_identity()
    assert identity is not None
    assert identity[1] == "_index_expand_nki", identity
    assert identity[0].endswith("dsa.index_expand"), identity

    # One dispatch per call, not per tile.
    dsa_index_expand(pool_ids, seq_lens, POOL_SIZE)
    assert index_expand_dispatch_counters() == (2, 0)

    for chain_rows in CHAIN_ROWS:
        assert row_tile_count(chain_rows) > 1, chain_rows
        seq = _seq_lens(chain_rows)
        clen = seq.reshape(chain_rows, 1).contiguous()
        gen = torch.Generator().manual_seed(1030 + chain_rows)
        scores = torch.randn(chain_rows, CHAIN_WIDTH, generator=gen, dtype=torch.float32) * 0.05

        reset_causal_bound_dispatch_counters()
        reset_topk_select_dispatch_counters()
        reset_causal_sentinel_dispatch_counters()
        reset_index_expand_dispatch_counters()

        # The bound fills every pool a row's own length does not complete, so the selector
        # cannot pick one and the expansion's caller precondition holds by construction.
        assert can_run_dsa_causal_bound(scores, clen, POOL_SIZE) is True
        bounded = dsa_causal_bound(scores, clen, POOL_SIZE)
        assert causal_bound_dispatch_counters() == (1, 0), causal_bound_dispatch_counters()

        assert can_run_dsa_topk_select(bounded, N_GROUPS_SMALL) is True
        values, indices = dsa_topk_select(bounded, N_GROUPS_SMALL)
        assert topk_select_dispatch_counters() == (1, 0), topk_select_dispatch_counters()

        # Anything the selector returned at or above the real pool width is a pad it invented;
        # the sentinel stage turns that into a -1 the expansion understands.
        idx32 = indices.to(torch.int32).contiguous()
        vals = values.contiguous()
        assert can_run_dsa_causal_sentinel(vals, idx32, CHAIN_WIDTH) is True
        marked = dsa_causal_sentinel(vals, idx32, CHAIN_WIDTH)
        assert causal_sentinel_dispatch_counters() == (1, 0), causal_sentinel_dispatch_counters()

        assert can_run_dsa_index_expand(marked, seq, POOL_SIZE) is True
        expanded = dsa_index_expand(marked, seq, POOL_SIZE)
        assert index_expand_dispatch_counters() == (1, 0), index_expand_dispatch_counters()
        assert tuple(expanded.shape) == (chain_rows, index_expand_width(N_GROUPS_SMALL, POOL_SIZE))
        assert expanded.dtype is torch.int32, expanded.dtype

        # The ids the chain really produced satisfy the caller precondition.
        violations = _precondition_violations(marked.tolist(), seq.tolist(), POOL_SIZE)
        assert violations == [], violations[:8]
        expected = _dsa_index_expand_torch(marked, seq, POOL_SIZE)
        assert torch.equal(expanded, expected), \
            f"rows={chain_rows}: the expansion of the chain's own ids differs from the torch route"
