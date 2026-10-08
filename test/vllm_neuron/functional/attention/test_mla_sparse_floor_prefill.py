# SPDX-License-Identifier: Apache-2.0
"""Per-query invariants of the low-precision sparse MLA prefill body, in the simulator.

The body serves the one-head geometry at more than one query (``_lowp_serves``). It
gathers each query's selected rows once, with the latent on partitions; a sentinel
column's gather offset is a row its own query selects (the index in the same partition
of the first 128-column chunk), or row 0 where that index is a sentinel too. The
softmax of a group of queries runs on one partition per query. These tests pin what
that must not change:

* a softmax group whose queries' first chunks select only cache rows reads no other
  row: poisoning row 0, which no index selects, leaves the group's outputs bit for bit
  what they are with that row finite. (A non-finite value in any row a group gathers
  reaches every query of the group: MM1's stationary is zero outside the query's own
  column, and ``0 * nan`` is nan.);
* a query's output depends only on its own query row and index row, not on the queries
  that share its group or ran before it in the same buffer: reversing the query order
  reverses the output bit for bit;
* the grouped softmax computes the one-query path's values: a call whose query count
  is not a whole query block runs one query per group, and its rows match the grouped
  call's to within one re-association of MM1's fp32 sums (see
  :data:`REASSOCIATION_REL_L2`). On the device the two are equal bit for bit.

The index rows follow the selector's layout at the start of a prompt (the chunk-1
case): query ``t`` names ``t + 1`` distinct rows in a random order, then -1 to the end
of the row, so most columns are the sentinel. Some rows repeat an index and some are
wholly sentinel, which the body must zero.
"""

from __future__ import annotations

import torch

from vllm_neuron.functional.attention import mla_sparse as MS

LATENT = MS.TARGET_LATENT_RANK
#: The selector's index row: 512 pools of 4 rows, the 3-row tail, -1 to whole chunks.
WIDTH = 17 * MS.KEY_CHUNK
SCALE = 0.0625
#: Query rows of one query block (``_queries_per_block`` at one head).
BLOCK = MS.DGE_TRANSPOSE_ROWS
#: The simulator computes a matmul with numpy's sgemm, whose summation order depends on
#: the stationary's width: a one-query group and a four-query group sum MM1's
#: ``LATENT``-term dot products in different orders. One re-association of such a sum
#: moves it by about ``sqrt(LATENT)`` fp32 rounding units (random-walk rounding error),
#: and the softmax and MM2 carry that relative change into the output. The device's PE
#: sums every column in one fixed order, so there the grouping changes no bit
#: (``reports/mla_sparse.md``).
REASSOCIATION_REL_L2 = LATENT ** 0.5 * 2.0 ** -24


def _empty_row(seq: int) -> int:
    """The row :func:`_operands` makes wholly sentinel."""
    return seq // 3


def _operands(seq: int, rows: int, seed: int, first_row: int = 0):
    """``(q, cache, indices)``: chunk-1 index rows over cache rows ``first_row ..``."""
    gen = torch.Generator().manual_seed(seed)
    q = (torch.randn(seq, 1, LATENT, generator=gen) * 0.5).to(torch.bfloat16)
    cache = torch.randn(rows, LATENT, generator=gen).to(torch.bfloat16)
    indices = torch.full((seq, WIDTH), MS.SENTINEL_INDEX, dtype=torch.int32)
    for t in range(seq):
        seen = min(WIDTH, 8 * (t + 1))
        chosen = first_row + torch.randperm(rows - first_row, generator=gen)[:seen]
        indices[t, :seen] = chosen.to(torch.int32)
    indices[_empty_row(seq)] = MS.SENTINEL_INDEX           # attends nothing: zeros
    indices[seq // 2, 1::2] = indices[seq // 2, 0]         # one row repeated
    indices[seq - 1, :WIDTH // 2] = indices[seq - 1, WIDTH // 2 - 1]
    return q, cache, indices


def _attend(q, cache, indices):
    out = MS.mla_sparse_attention(q, cache, indices, SCALE)
    assert out.dtype == torch.float32 and out.shape == q.shape
    return out


def test_a_sentinel_column_reads_a_row_its_query_selects(monkeypatch):
    monkeypatch.delenv("VLLM_NEURON_MLA_SPARSE_FP32", raising=False)
    monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    q, cache, indices = _operands(2 * BLOCK, 2600, seed=5, first_row=1)
    assert int(indices.min()) == MS.SENTINEL_INDEX and not bool((indices == 0).any())
    group = MS._softmax_rows(MS._queries_per_block(2 * BLOCK, 1))
    full_first_chunk = (indices[:, :MS.KEY_CHUNK] >= 0).all(dim=1)
    full_group = full_first_chunk.reshape(-1, group).all(dim=1).repeat_interleave(group)
    has_sentinel = (indices == MS.SENTINEL_INDEX).any(dim=1)
    assert bool((full_group & has_sentinel).any()) and bool((~full_first_chunk).any())
    clean = _attend(q, cache, indices)
    poisoned_cache = cache.clone()
    poisoned_cache[0] = float("nan")
    poisoned = _attend(q, poisoned_cache, indices)
    assert torch.isfinite(clean).all()
    assert torch.count_nonzero(clean[_empty_row(2 * BLOCK)]) == 0
    assert torch.equal(poisoned[full_group], clean[full_group])


def test_a_query_row_does_not_depend_on_its_neighbours(monkeypatch):
    monkeypatch.delenv("VLLM_NEURON_MLA_SPARSE_FP32", raising=False)
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    q, cache, indices = _operands(2 * BLOCK, 2600, seed=7)
    forward = _attend(q, cache, indices)
    backward = _attend(q.flip(0), cache, indices.flip(0))
    assert torch.isfinite(forward).all()
    assert torch.equal(backward.flip(0), forward)


def test_the_grouped_softmax_computes_the_one_query_values(monkeypatch):
    monkeypatch.delenv("VLLM_NEURON_MLA_SPARSE_FP32", raising=False)
    monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    q, cache, indices = _operands(2 * BLOCK + 1, 2600, seed=11)
    assert MS._softmax_rows(MS._queries_per_block(2 * BLOCK + 1, 1)) == 1
    assert MS._softmax_rows(MS._queries_per_block(2 * BLOCK, 1)) == MS.SOFTMAX_ROWS > 1
    single = _attend(q, cache, indices)[:2 * BLOCK]
    grouped = _attend(q[:2 * BLOCK], cache, indices[:2 * BLOCK])
    rel = float((grouped.double() - single.double()).norm() / single.double().norm())
    assert rel <= REASSOCIATION_REL_L2, rel
    empty = _empty_row(2 * BLOCK + 1)
    assert torch.count_nonzero(grouped[empty]) == 0 and torch.count_nonzero(single[empty]) == 0
