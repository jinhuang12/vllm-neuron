# SPDX-License-Identifier: Apache-2.0
"""The indexer ring's depth on a speculative server, and the seams that read it.

One helper derives the depth from the model constant and the draft count; the prefill
leg's ``seed_tail`` writes the open pool's rows at ``position % depth``; a ring seeded at
the deep depth pools the same rows as today's ring once the decode step closes the pool.
"""

from __future__ import annotations

import os

import pytest
import torch

from test.vllm_neuron.model.glm5_next import test_dsa_layer as layer_half
from vllm_neuron.functional.dsa.decode_tail_update import (
    dsa_decode_ring_rows,
    dsa_decode_tail_update,
    max_rows_for,
)
from vllm_neuron.functional.dsa.decode_trow import indexer_ring_depth

#: Slots per pool: the model constant the depth is derived from.
POOL = layer_half.POOL_SIZE
#: The depth :func:`indexer_ring_depth` answers for the gate's draft count, and the one
#: every deep-ring case below runs at.
GATE_DRAFTS = 3
#: Chunk ends that leave 1, 2 and 3 rows in the open pool, straddling the deep ring's
#: wrap at ``2 * POOL``; and a long one, to show the modulus is the depth, not the pool.
CHUNK_ENDS = (5, 6, 7, 9, 10, 11, 13, 15, 2101, 2103)


def _require_cpu_mode() -> None:
    assert os.environ.get("VLLM_NEURON_CPU_MODE") == "1", (
        "the declared acceptance runs under VLLM_NEURON_CPU_MODE=1 and this process "
        "does not carry it"
    )


def _indexer():
    """A bare indexer with its per-slot bias materialised: enough for both ring seams."""
    indexer = layer_half._bare_indexer()
    gen = torch.Generator().manual_seed(58_001)
    indexer.index_kpool_compress_ape = torch.nn.Parameter(
        torch.randn(POOL, int(indexer.index_head_dim), generator=gen).to(torch.bfloat16),
        requires_grad=False,
    )
    return indexer


def _rows(count: int, head_dim: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    key = torch.randn(count, head_dim, generator=gen).to(torch.bfloat16)
    gate = torch.randn(count, head_dim, generator=gen).to(torch.bfloat16)
    return key, gate


# --------------------------------------------------------------------------- #
# the helper
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("drafts", [0, None])
def test_the_depth_is_todays_pool_when_nothing_drafts(drafts) -> None:
    assert indexer_ring_depth(POOL, drafts) == POOL


@pytest.mark.parametrize("drafts", [1, 2, GATE_DRAFTS, 4, 5, 6])
def test_the_depth_holds_every_verify_row_and_the_rollback_window(drafts: int) -> None:
    """The ruling's closed form: the next power of two at or above
    ``max(index_kpool, drafts + 3)``; a verify step of ``1 + drafts`` rows fits, and
    half the depth would not."""
    depth = indexer_ring_depth(POOL, drafts)
    assert depth % POOL == 0 and depth & (depth - 1) == 0, depth
    assert depth >= max(POOL, drafts + 3), depth
    assert max_rows_for(depth, POOL) >= 1 + drafts, (depth, drafts)
    assert depth == POOL or max_rows_for(depth // 2, POOL) < 1 + drafts, (depth, drafts)


def test_the_gate_point_is_a_ring_of_eight() -> None:
    assert indexer_ring_depth(POOL, GATE_DRAFTS) == 2 * POOL


def test_a_negative_draft_count_is_refused() -> None:
    with pytest.raises(ValueError, match="num_speculative_tokens"):
        indexer_ring_depth(POOL, -1)


# --------------------------------------------------------------------------- #
# seed_tail at the deep depth
# --------------------------------------------------------------------------- #
def _seed_oracle(depth: int, pool: int, key, gate, end: int, start: int | None):
    """The rows a seed writes: the open pool's ``r = end % pool`` positions sit at ring
    rows ``position % depth``; a chunk writes only the ones it holds."""
    real = int(key.shape[0]) if start is None else end - start
    open_rows = end % pool
    take = min(open_rows, real)
    want = torch.full((2, depth, key.shape[1]), -1.0, dtype=torch.bfloat16)
    for s in range(take):
        position = end - take + s
        want[0, position % depth] = key[real - take + s]
        want[1, position % depth] = gate[real - take + s]
    return want, take


@pytest.mark.parametrize("end", CHUNK_ENDS)
@pytest.mark.parametrize("chunked", [False, True])
def test_seed_tail_writes_the_open_pool_at_position_mod_depth(end: int, chunked: bool) -> None:
    _require_cpu_mode()
    indexer = _indexer()
    head_dim = int(indexer.index_head_dim)
    depth = indexer_ring_depth(POOL, GATE_DRAFTS)
    tokens = POOL  # a chunk no longer than the shortest end in CHUNK_ENDS
    # A chunked prefill ends the open pool with a key the chunk does not hold when the
    # chunk is shorter than the remainder: start two rows before the end.
    start = end - 2 if chunked else None
    key, gate = _rows(tokens, head_dim, seed=58_000 + end)

    want, want_take = _seed_oracle(depth, POOL, key, gate, end, start)

    by_int = torch.full((2, depth, head_dim), -1.0, dtype=torch.bfloat16)
    took_int = indexer.seed_tail(by_int, key, gate, end, start)
    by_tensor = torch.full((2, depth, head_dim), -1.0, dtype=torch.bfloat16)
    took_tensor = indexer.seed_tail(
        by_tensor,
        key,
        gate,
        torch.tensor(end, dtype=torch.int32),
        None if start is None else torch.tensor(start, dtype=torch.int32),
    )

    assert int(took_int) == want_take and int(took_tensor) == want_take
    assert torch.equal(by_int, want), f"int route wrote other rows at end={end}"
    assert torch.equal(by_tensor, want), f"tensor route wrote other rows at end={end}"


def test_seed_tail_still_refuses_a_ring_that_is_not_a_power_of_two_of_the_pool() -> None:
    from vllm_neuron.model.glm5_next.model_fp8 import Glm5NextDSAIndexerError

    indexer = _indexer()
    head_dim = int(indexer.index_head_dim)
    key, gate = _rows(2, head_dim, seed=1)
    for depth in (POOL + 1, 3 * POOL):
        ring = torch.zeros(2, depth, head_dim, dtype=torch.bfloat16)
        with pytest.raises(Glm5NextDSAIndexerError, match="index_kpool"):
            indexer.seed_tail(ring, key, gate, 6)


# --------------------------------------------------------------------------- #
# the seam: a deep seeded ring pools what today's ring pools
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("end", CHUNK_ENDS)
def test_a_deep_seeded_ring_pools_the_same_rows_as_todays_ring(end: int) -> None:
    """Seed at depth ``POOL`` and at the deep depth from one prefill remainder, then
    close the open pool: today's one-token kernel step by step against the T-row ring
    kernel in one step. The pooled rows are bit-equal."""
    _require_cpu_mode()
    indexer = _indexer()
    head_dim = int(indexer.index_head_dim)
    ape = indexer.index_kpool_compress_ape.detach().to(torch.float32)
    deep = indexer_ring_depth(POOL, GATE_DRAFTS)
    key, gate = _rows(POOL, head_dim, seed=58_100 + end)

    today = torch.zeros(2, POOL, head_dim, dtype=torch.bfloat16)
    deep_ring = torch.zeros(2, deep, head_dim, dtype=torch.bfloat16)
    indexer.seed_tail(today, key, gate, end)
    indexer.seed_tail(deep_ring, key, gate, end)

    # The tokens that close the open pool: from ``end`` to the next multiple of POOL.
    rows = POOL - end % POOL
    assert 1 <= rows <= max_rows_for(deep, POOL)
    step_key, step_gate = _rows(rows, head_dim, seed=58_200 + end)

    pooled_today = None
    for t in range(rows):
        pooled_t, today = dsa_decode_tail_update(
            today, step_key[t : t + 1], step_gate[t : t + 1], ape, end + t
        )
        if pooled_t is not None:
            pooled_today = pooled_t
    assert pooled_today is not None, "the last step closes the pool by construction"

    pooled_deep, rings = dsa_decode_ring_rows(
        deep_ring.reshape(1, 2, deep, head_dim),
        torch.zeros(1, dtype=torch.int64),
        step_key,
        step_gate,
        ape,
        torch.tensor([end], dtype=torch.int32),
    )
    assert torch.equal(pooled_deep[rows - 1 : rows], pooled_today), (
        f"the deep ring pooled other values at end={end}: a seeded row sat at a row the "
        f"step did not read"
    )
    # Every stashed position agrees too, at its own row of each ring.
    new_deep = rings.reshape(2, deep, head_dim)
    for t in range(rows):
        position = end + t
        assert torch.equal(new_deep[:, position % deep], today[:, position % POOL])


# --------------------------------------------------------------------------- #
# the helper refuses malformed inputs by name instead of looping or answering
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pool", [0, 1, 3, 6, -4])
def test_ring_depth_for_refuses_a_pool_size_that_is_not_a_power_of_two_of_at_least_two(pool: int) -> None:
    from vllm_neuron.functional.dsa.decode_tail_update import (
        DecodeTailUpdateError,
        ring_depth_for,
    )

    with pytest.raises(DecodeTailUpdateError, match="pool_size"):
        ring_depth_for(pool, 3)


@pytest.mark.parametrize("max_rows", [0, -1])
def test_ring_depth_for_refuses_a_step_of_no_rows(max_rows: int) -> None:
    from vllm_neuron.functional.dsa.decode_tail_update import (
        DecodeTailUpdateError,
        ring_depth_for,
    )

    with pytest.raises(DecodeTailUpdateError, match="max_rows"):
        ring_depth_for(POOL, max_rows)


def test_indexer_ring_depth_refuses_a_malformed_pool_by_name() -> None:
    from vllm_neuron.functional.dsa.decode_tail_update import DecodeTailUpdateError

    with pytest.raises(DecodeTailUpdateError, match="pool_size"):
        indexer_ring_depth(0, GATE_DRAFTS)
