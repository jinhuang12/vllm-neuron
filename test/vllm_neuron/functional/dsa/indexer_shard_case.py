# SPDX-License-Identifier: Apache-2.0
"""Shared operands and per-rank work for the query-sharded DSA prefill selection tests.

The sharded selection is per query row: rank ``r`` of ``d`` scores, bounds, selects and
orders rows ``[r * R, (r + 1) * R)`` of the chunk over every candidate pool, and an
all-gather of the selected pool ids gives every rank the whole ``[T, select_k]``. The
tests simulate the ``d`` ranks as separate calls. Each call is a whole kernel chain on the
CPU NKI simulator, so the calls are spread over worker processes; the functions here are
module-level so a ``spawn`` worker can import them by name.

Every operand is rebuilt from ``(tokens, cands, seed)`` inside each worker, so a worker
receives three ints and not a tensor, and the parent and every worker see the same data.
"""

from __future__ import annotations

import os

import torch

#: The production indexer dials: 32 heads of 128 features, pools of 4 tokens, a
#: 2048-token budget, so ``select_k`` is 512. ``Glm5NextTextConfig()``'s own defaults.
PRODUCTION_TOPK = 2048

#: At ``cands == 512`` the production ``select_k`` equals the candidate count, which is the
#: bypass regime (``_require_serviceable``: ``selects = cands > select_k``), so no selection
#: runs there in production. The chain-level case at that width runs a 128-pool budget
#: instead, so the sharded selection itself is exercised at the narrowest width too.
NARROW_TOPK = 512

#: Weight scale for the synthetic gate: the order of ``projection_scale()`` at production
#: dials (``128 ** -0.5 * 32 ** -0.5 = 0.0156``) times a unit projection.
WEIGHT_SCALE = 0.05


def index_topk_for(cands: int) -> int:
    """The token budget a case runs at: production's, unless that is the bypass regime."""
    return PRODUCTION_TOPK if cands > PRODUCTION_TOPK // 4 else NARROW_TOPK


def make_indexer(cands: int):
    """A ``Glm5NextDSAIndexer`` at production dials; selection needs no weights."""
    from dataclasses import replace

    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig
    from vllm_neuron.model.glm5_next.model_fp8 import Glm5NextDSAIndexer

    config = replace(Glm5NextTextConfig(), index_topk=index_topk_for(cands))
    return Glm5NextDSAIndexer(config)


def chunk_start(tokens: int, cands: int, select_k: int, pool: int = 4) -> int:
    """Where the chunk begins in its sequence: straddling the ``select_k`` boundary.

    Rows before position ``select_k * pool`` see fewer than ``select_k`` complete pools and
    carry sentinels; rows after it select fully. Clamped so the chunk starts at 0 or later
    and ends inside the ``cands * pool``-token context the candidate axis spans.
    """
    want = select_k * pool - tokens // 2
    return max(0, min(want, cands * pool - tokens))


def case_operands(tokens: int, cands: int, seed: int):
    """``(query, keys, weights, seq_lens)`` for one case, rebuilt identically anywhere."""
    gen = torch.Generator().manual_seed(int(seed) * 1_000_003 + tokens * 7919 + cands)
    query = torch.randn(tokens, 32, 128, generator=gen).to(torch.bfloat16)
    keys = torch.randn(cands, 128, generator=gen).to(torch.bfloat16)
    weights = torch.randn(tokens, 32, generator=gen) * WEIGHT_SCALE
    indexer = make_indexer(cands)
    start = chunk_start(tokens, cands, indexer.select_k(), indexer.index_kpool)
    seq_lens = torch.arange(start + 1, start + tokens + 1, dtype=torch.int32)
    return query, keys, weights, seq_lens


def _counters() -> dict[str, tuple[int, int]]:
    from vllm_neuron.functional.dsa import causal_bound, score_gemm, sentinel_order, topk_select

    return {
        "score_gemm": score_gemm.score_gemm_dispatch_counters(),
        "causal_bound": causal_bound.causal_bound_dispatch_counters(),
        "causal_sentinel": causal_bound.causal_sentinel_dispatch_counters(),
        "topk_select": topk_select.topk_select_dispatch_counters(),
        "sentinel_order": sentinel_order.sentinel_order_dispatch_counters(),
    }


def _reset_counters() -> None:
    from vllm_neuron.functional.dsa import causal_bound, score_gemm, sentinel_order, topk_select

    score_gemm.reset_score_gemm_dispatch_counters()
    causal_bound.reset_causal_bound_dispatch_counters()
    causal_bound.reset_causal_sentinel_dispatch_counters()
    topk_select.reset_topk_select_dispatch_counters()
    sentinel_order.reset_sentinel_order_dispatch_counters()


def replicated_task(tokens: int, cands: int, seed: int):
    """The as-built path on every row: ``(pool_ids [T, k], bounded scores [T, C], counters)``.

    The bounded scores are returned for the tie census and for the torch top-k check; they
    are computed by the same two kernels ``select_bounded_pools`` runs, in the same order.
    """
    from vllm_neuron.functional.dsa.causal_bound import dsa_causal_bound

    torch.set_num_threads(1)
    query, keys, weights, seq_lens = case_operands(tokens, cands, seed)
    indexer = make_indexer(cands)
    _reset_counters()
    pool_ids = indexer.select_bounded_pools(
        indexer.score_pools(query, keys, weights), seq_lens
    )
    counters = _counters()
    bounded = dsa_causal_bound(
        indexer.score_pools(query, keys, weights),
        seq_lens.reshape(-1, 1),
        indexer.index_kpool,
    )
    return pool_ids, bounded, counters


def rank_task(tokens: int, cands: int, degree: int, rank: int, seed: int):
    """Rank ``rank`` of ``degree``: its ``[R, k]`` pool ids from the production local path."""
    from vllm_neuron.functional.dsa.indexer_shard import row_shard

    torch.set_num_threads(1)
    query, keys, weights, seq_lens = case_operands(tokens, cands, seed)
    indexer = make_indexer(cands)
    shard = row_shard(tokens, degree)
    _reset_counters()
    local = indexer.local_pool_ids(query, keys, weights, seq_lens, shard, rank)
    return local, _counters()


def worker_environment() -> dict[str, str]:
    """Thread caps for a simulator worker. Everything else it inherits from the parent's
    environment, which the root conftest has already pinned (CPU mode, the platform
    target) and this directory's conftest has set ``NKI_SIMULATOR`` in."""
    return {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
