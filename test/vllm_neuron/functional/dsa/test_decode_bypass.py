# SPDX-License-Identifier: Apache-2.0
"""The DSA decode bypass: below 2052 tokens selection keeps every token, so the layer
attends the causal prefix densely and skips scoring and selection.

Every case runs one decode step of one DSA/MLA layer (per-rank TP=64 geometry, hidden
and query-latent widths narrowed for the simulator) through
``Glm5NextMLAAttention.forward`` three ways:

* **bypass** -- this tree, ``max_seq_len = context``: the bound proves selection is a
  no-op, so ``mla_decode_attention`` attends the prefix and the indexer's query side,
  scores and top-k never run.
* **forced selection** -- this tree, ``max_seq_len = 4096`` over a 4096-row window: the
  indexer scores and selects exactly as before; its selected set is read and must be the
  whole causal prefix, and the layer output must equal the bypass output.
* **5938748** -- the snapshot model, also selecting, with its fp32 projections: the
  bypass output must agree within the stated fp8-vs-fp32 tolerance.

At 2052 tokens the bound no longer proves anything, the indexer selects, and one pool
of four tokens is dropped. At 4096 tokens, where half the pools are dropped, the
selection shares at least 99% of its tokens with 5938748's.
"""

from __future__ import annotations

import pytest
import torch

from test.hardware.baselines.dsa_5938748 import load as load_5938748
from test.vllm_neuron.functional.dsa import dsa_decode_case as case
from vllm_neuron.functional.attention import mla_decode, mla_projections, mla_sparse
from vllm_neuron.functional.dsa import decode_batch, kpool_hadamard, score_gemm, topk_select
from vllm_neuron.functional.dsa.decode_bypass import (
    bypass_max_context,
    selection_bound,
    selection_is_a_no_op,
)
from vllm_neuron.model.glm5_next import model_fp8

#: Narrowed widths: both only scale projection cost in the simulator; the bound reads
#: neither.
HIDDEN, Q_LORA = 256, 256
#: The window of the forced-selection run: the runner's model length, 32 pages of 128.
FULL_WINDOW_PAGES, FULL_MAX_SEQ_LEN = 32, 4096

#: fp8 weights read as stored against the 5938748 fp32 upcasts. The layer output is bf16
#: and passes through two fp8 projections whose stationary operand is rounded to bf16
#: (``mla_projection_lowp``), so it agrees to a few bf16 units of its own peak.
LAYER_REL_L2 = 1e-2
LAYER_ATOL_PEAK_FRACTION = 2.0 ** -5


@pytest.fixture(scope="module")
def layers():
    cfg = case.decode_config(hidden_size=HIDDEN, q_lora_rank=Q_LORA)
    live = case.build_attention(model_fp8, cfg)
    base = case.build_attention(load_5938748().model_fp8, cfg)
    return cfg, live, base


def _reset():
    mla_decode.reset_mla_decode_dispatch_counters()
    mla_sparse.reset_mla_sparse_dispatch_counters()
    score_gemm.reset_score_gemm_dispatch_counters()
    decode_batch.reset_decode_batch_dispatch_counters()
    topk_select.reset_topk_select_dispatch_counters()
    kpool_hadamard.reset_kpool_hadamard_dispatch_counters()
    mla_projections.reset_mla_projection_lowp_counts()


def _counts():
    topk = topk_select.topk_select_dispatch_counters()
    return {
        "dense": mla_decode.mla_decode_route_counts()[0],
        "sparse": mla_sparse.mla_sparse_dispatch_counters()[0],
        "score_gemm": score_gemm.score_gemm_dispatch_counters()[0],
        # A one-request decode step scores through the batched kernel, as a batch of one.
        "batch_scores": decode_batch.decode_batch_route_counts()[1],
        "topk": topk[0] + topk[1],
        # The decode step selects in one kernel (top-k, sentinel, order and expand),
        # counted in the batched decode family.
        "select": decode_batch.decode_batch_route_counts()[3],
        "hadamard": kpool_hadamard.kpool_hadamard_dispatch_counters()[0],
        "lowp": mla_projections.mla_projection_lowp_counts()[0],
    }


def _run(module, operands: dict, **override):
    ops = case.cloned(operands)
    ops.update(override)
    collector: list[torch.Tensor] = []
    _reset()
    out = module.forward(**ops, collector=collector)
    return out, collector, _counts(), ops


def _selected_set(indices: torch.Tensor) -> set[int]:
    return {int(v) for v in indices.reshape(-1) if int(v) >= 0}


def _agree(label: str, got: torch.Tensor, want: torch.Tensor) -> float:
    got, want = got.to(torch.float32), want.to(torch.float32)
    rel = float((got - want).norm() / want.norm())
    peak = float(want.abs().max())
    assert rel <= LAYER_REL_L2, f"{label}: relative L2 {rel:.3e} > {LAYER_REL_L2}"
    worst = float((got - want).abs().max()) / peak
    print(f"{label}: relative L2 {rel:.3e}, max |diff| / peak {worst:.3e}")
    torch.testing.assert_close(got, want, rtol=0, atol=LAYER_ATOL_PEAK_FRACTION * peak)
    return rel


def test_the_threshold_is_the_dials_own_and_matches_the_indexer():
    """2051 = select_k * kpool + kpool - 1, read from the config, and the indexer agrees."""
    cfg = case.decode_config(hidden_size=HIDDEN, q_lora_rank=Q_LORA)
    topk, pool = int(cfg.index_topk), int(cfg.index_kpool)
    assert (topk, pool) == (2048, 4)
    assert bypass_max_context(topk, pool) == (topk // pool) * pool + pool - 1 == 2051
    indexer = model_fp8.Glm5NextDSAIndexer(cfg)
    pool_cache = torch.zeros(1152, int(cfg.index_head_dim), dtype=torch.bfloat16)
    for length in (64, 1024, 2051, 2052, 4096):
        _, _, selects = indexer._require_serviceable(length, case.PAGE, pool_cache)
        assert selects == (not selection_is_a_no_op(length, topk, pool)), length


def test_the_bound_is_the_smaller_of_the_model_length_and_the_window():
    assert selection_bound(4096, 2048) == 2048
    assert selection_bound(2051, 2176) == 2051
    assert selection_bound(4096, None) == 4096
    assert not selection_is_a_no_op(selection_bound(4096, 4096), 2048, 4)
    assert selection_is_a_no_op(selection_bound(4096, 2048), 2048, 4)


@pytest.mark.parametrize("context", [64, 1024, 2051])
def test_the_bypass_is_exact_and_skips_selection(layers, context):
    cfg, live, base = layers
    pages = -(-context // case.PAGE)
    operands = case.decode_operands(cfg, context, window_pages=FULL_WINDOW_PAGES,
                                    max_seq_len=FULL_MAX_SEQ_LEN)
    short = {"block_table_row": operands["block_table_row"][:pages]}

    dense, dense_col, dense_counts, dense_ops = _run(live, operands, max_seq_len=context,
                                                     **short)
    assert dense_counts["dense"] == 1, dense_counts
    assert dense_counts["sparse"] == 0, dense_counts
    assert dense_counts["score_gemm"] == 0 and dense_counts["batch_scores"] == 0, dense_counts
    assert dense_counts["topk"] == 0 and dense_counts["select"] == 0, dense_counts
    # The query rotation is skipped with the query; the four sites the step still reads
    # (q_a twice, q_b, kv_a, o_proj, wk, gate) all took the fp8/bf16 route.
    assert dense_counts["hadamard"] == 0, dense_counts
    assert dense_counts["lowp"] == 7, dense_counts

    chosen, chosen_col, chosen_counts, chosen_ops = _run(live, operands)
    assert chosen_counts["dense"] == 0 and chosen_counts["sparse"] == 1, chosen_counts
    assert chosen_counts["batch_scores"] == 1 and chosen_counts["score_gemm"] == 0, chosen_counts
    assert chosen_counts["select"] == 1 and chosen_counts["topk"] == 0, chosen_counts
    # Selection is a no-op here: the set it picks is the whole causal prefix.
    assert _selected_set(chosen_col[0][0]) == set(range(context))
    # The dump's index entry is the same tensor either way.
    assert torch.equal(dense_col[0], live.indexer._bypass_indices(operands["seq_lens"])[:5])
    # Same rows attended, so the same output; and the write stage is untouched.
    torch.testing.assert_close(dense.float(), chosen.float(), rtol=0,
                               atol=2.0 ** -8 * float(chosen.float().abs().max()))
    for name in ("latent_cache", "pool_cache", "tail"):
        assert torch.equal(dense_ops[name], chosen_ops[name]), name

    baseline, _, _, _ = _run(base, operands)
    _agree(f"bypass vs 5938748 at {context}", dense, baseline)


def test_2052_tokens_still_select_and_drop_one_pool(layers):
    cfg, live, base = layers
    context = bypass_max_context(int(cfg.index_topk), int(cfg.index_kpool)) + 1
    pages = -(-context // case.PAGE)
    operands = case.decode_operands(cfg, context, window_pages=pages, max_seq_len=context)
    assert selection_bound(context, pages * case.PAGE) == context == 2052

    out, col, counts, _ = _run(live, operands)
    assert counts["dense"] == 0 and counts["sparse"] == 1, counts
    assert counts["batch_scores"] == 1 and counts["score_gemm"] == 0, counts
    assert counts["select"] == 1 and counts["topk"] == 0, counts
    assert counts["hadamard"] == 1, counts
    picked = _selected_set(col[0][0])
    # 513 complete pools, 512 kept, no open tail at 2052 = 513 * 4: four tokens dropped.
    assert len(picked) == context - int(cfg.index_kpool) == 2048
    assert picked < set(range(context))

    baseline, base_col, _, _ = _run(base, operands)
    assert _selected_set(base_col[0][0]) == picked
    _agree("selected vs 5938748 at 2052", out, baseline)


#: At 4096 tokens selection keeps half the pools. The index query now rides fp8 weights
#: read as stored, so a pool whose score sits at the cut can cross it: measured 2 of 512
#: pools (8 of 2048 tokens) against 5938748 on this draw.
SELECTED_OVERLAP = 0.99


def test_selected_decode_keeps_5938748s_selection(layers):
    cfg, live, base = layers
    context = FULL_MAX_SEQ_LEN
    operands = case.decode_operands(cfg, context, window_pages=FULL_WINDOW_PAGES,
                                    max_seq_len=FULL_MAX_SEQ_LEN)
    out, col, counts, _ = _run(live, operands)
    assert counts["dense"] == 0 and counts["sparse"] == 1 and counts["select"] == 1, counts
    baseline, base_col, _, _ = _run(base, operands)
    picked, base_picked = _selected_set(col[0][0]), _selected_set(base_col[0][0])
    assert len(picked) == len(base_picked) == int(cfg.index_topk)
    common = len(picked & base_picked)
    print(f"selected at {context}: {common} of {len(base_picked)} tokens shared with 5938748")
    assert common >= SELECTED_OVERLAP * len(base_picked), common
    assert torch.isfinite(out.float()).all()
