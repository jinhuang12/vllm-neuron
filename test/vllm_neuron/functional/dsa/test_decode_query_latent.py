# SPDX-License-Identifier: Apache-2.0
"""A selecting decode step projects its query latent once.

``Glm5NextMLAAttention._forward_requests`` computes the normalised query latent
(:meth:`project_query_latent`, the ``q_a_proj`` projection and its norm) for the
indexer's ``wq_b``. A selecting step hands that same tensor to the attention
(``q_latent=``), whose ``q_b_proj`` reads it, so the step projects it once. Two steps
keep the attention's own projection:

* the bypass step (the selection keeps every token): its indexer reads no query side,
  so the indexer's latent has no reader there and the compiled graph drops it;
* a later draft iteration that reuses the first iteration's selection
  (``index_share``): it runs no indexer, so no latent exists to hand over.

The reference is the two-projection form: the same chain with the attention projecting
its own latent from the same rows. The latent is one projection of the same rows either
way, so the step equals the reference bit for bit, output and every bank.

Shapes are this checkpoint's per-rank decode shape at TP=64 (``decode_config()``:
hidden 4096, q_lora_rank 1536, one MLA head, the indexer's 32 heads of 128) on the two
decode windows either side of the bypass bound: the widest whole-page window the bypass
serves and the next one, twice as wide, where the step selects. Rows per request are the
one-token step and a verify step of ``1 + 3`` speculative tokens; two requests take the
batched attention at one row each.
"""

from __future__ import annotations

import contextlib
import functools
from unittest import mock

import pytest
import torch

import vllm_neuron.model.glm5_next.model_fp8 as model_fp8
from test.vllm_neuron.functional.dsa.dsa_batch_case import pool_rows
from test.vllm_neuron.functional.dsa.dsa_decode_case import (
    PAGE,
    build_attention,
    decode_config,
)
from vllm_neuron.functional.dsa import decode_tail_update as TU
from vllm_neuron.functional.dsa.decode_bypass import (
    bypass_max_context,
    selection_bound,
    selection_is_a_no_op,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

#: Rows per request: the one-token step and a verify step of 1 + 3 speculative tokens.
ROWS = (1, 4)


@functools.lru_cache(maxsize=1)
def _module():
    assert can_run_kernel(), "run with VLLM_NEURON_CPU_MODE=1 and NKI_SIMULATOR=1"
    return build_attention(model_fp8, decode_config())


def _windows() -> dict[str, int]:
    """The decode windows either side of the bypass bound, in whole pages."""
    indexer = _module().indexer
    bypass = bypass_max_context(indexer.index_topk, indexer.index_kpool) // PAGE * PAGE
    return {"bypass": bypass, "selecting": 2 * bypass}


def _dense(window: int) -> bool:
    indexer = _module().indexer
    return selection_is_a_no_op(
        selection_bound(window, window), indexer.index_topk, indexer.index_kpool)


def _starts(regime: str, rows: int) -> tuple[int, ...]:
    """Row 0's position per request: the window's last rows, and its first rows past
    the bypass bound (the selecting window) or its middle (the bypass window)."""
    window = _windows()[regime]
    indexer = _module().indexer
    edge = (bypass_max_context(indexer.index_topk, indexer.index_kpool)
            if regime == "selecting" else window // 2)
    return (window - rows, edge)


def _operands(regime: str, batch: int, rows: int, *, seed: int,
              starts: tuple[int, ...] | None = None) -> dict:
    """One step of ``batch`` requests x ``rows`` tokens on the runner's layout: a paged
    latent bank, and a pooled store and a ring per state slot, each request on pages and
    a slot no other request holds. ``starts`` defaults to :func:`_starts`."""
    module = _module()
    window = _windows()[regime]
    pool, head_dim = int(module.indexer.index_kpool), int(module.indexer.index_head_dim)
    gen = torch.Generator().manual_seed(seed)
    starts = _starts(regime, rows) if starts is None else starts
    start = torch.tensor(starts[:batch], dtype=torch.int64)
    used = [-(-(int(s) + rows) // PAGE) for s in start]
    bank_pages = sum(used) + 3
    order = torch.randperm(bank_pages, generator=gen).to(torch.int32)
    table = torch.full((batch, window // PAGE), -1, dtype=torch.int32)
    first = 0
    for b, n in enumerate(used):
        table[b, :n] = order[first:first + n]
        first += n
    positions = (start[:, None] + torch.arange(rows)[None, :]).reshape(-1)
    request = torch.arange(batch).repeat_interleave(rows)
    slots = batch + 3
    return {
        "hidden": (torch.randn(batch * rows, int(module.hidden_size), generator=gen) * 0.5
                   ).to(torch.bfloat16),
        "latent_cache": (torch.randn(bank_pages * PAGE, 1, int(module.kv_lora_rank),
                                     generator=gen) * 0.5).to(torch.bfloat16),
        "pool_bank": (torch.randn(slots, pool_rows(window, pool), head_dim, generator=gen)
                      * 0.5).to(torch.bfloat16),
        "tail_bank": (torch.randn(slots, 2, TU.ring_depth_for(pool, max(ROWS)), head_dim,
                                  generator=gen) * 0.5).to(torch.bfloat16),
        "state_slots": torch.randperm(slots, generator=gen)[:batch].to(torch.int32),
        "start": start,
        "seq_lens": (positions + 1).to(torch.int32),
        "block_table": table,
        "latent_slots": (table[request, positions // PAGE].to(torch.int64) * PAGE
                         + positions % PAGE),
        "softmax_scale": float(int(module.qk_nope_head_dim)) ** -0.5,
        "window": window,
        "batch": batch,
        "rows": rows,
    }


def _cloned(ops: dict) -> dict:
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in ops.items()}


def _step(ops: dict, *, index_share=None) -> torch.Tensor:
    """The layer's decode step as served, banks written in place."""
    return _module()._forward_requests(
        ops["hidden"],
        latent_cache=ops["latent_cache"],
        pool_cache=ops["pool_bank"],
        seq_lens=ops["seq_lens"],
        start_position=ops["start"],
        softmax_scale=ops["softmax_scale"],
        max_seq_len=ops["window"],
        page_size=PAGE,
        block_table_row=ops["block_table"].t().contiguous(),
        latent_slots=ops["latent_slots"],
        tail=ops["tail_bank"],
        position=ops["start"],
        collector=None,
        state_slots=ops["state_slots"],
        index_share=index_share,
    )


def _two_projection_step(ops: dict) -> torch.Tensor:
    """The same chain with the attention projecting its own latent from the same rows.

    The selection takes the step's own form: a pool mask where the step attends through
    ``mla_decode_masked`` (one request of several rows that selects), token indices
    elsewhere.
    """
    module = _module()
    indexer = module.indexer
    hidden = ops["hidden"]
    bound = selection_bound(ops["window"], ops["window"])
    dense = _dense(ops["window"])
    table_row = ops["block_table"].t().contiguous()
    masked = (not dense and ops["batch"] == 1 and ops["rows"] > 1
              and module._masked_attention_serves(ops["rows"], table_row, PAGE))
    q_latent = module.project_query_latent(hidden)
    projected = indexer.project_stage(hidden, q_latent, query_side=not dense)
    selection = indexer.forward_requests(
        hidden, q_latent, ops["pool_bank"], ops["tail_bank"], ops["state_slots"],
        ops["seq_lens"], ops["start"], max_seq_len=bound, indices_wanted=not dense,
        projected=projected, pool_mask=masked)
    common = {
        "block_table_row": table_row,
        "latent_slots": ops["latent_slots"],
        "page_size": PAGE,
        "dense": dense,
    }
    if ops["rows"] > 1:
        return module._attend_requests(
            hidden, ops["latent_cache"], ops["start"], None if masked else selection,
            ops["softmax_scale"], ops["batch"], prefill_end_position=None, collector=None,
            active_mla_query_rows=None, pool_mask=selection if masked else None, **common)
    return module.attend(hidden, ops["latent_cache"], ops["start"], selection,
                         ops["softmax_scale"], batch_size=ops["batch"], collector=None,
                         **common)


@contextlib.contextmanager
def _recorded():
    """Record the latents the step computes and what the attention projects from.

    Yields ``(latents, handed)``: every :meth:`project_query_latent` result, in call
    order, and for each :meth:`project_query_and_latent` call the ``q_latent`` it was
    handed (None when it projects its own).
    """
    module = _module()
    latents, handed = [], []
    own_latent = module.project_query_latent
    own_query = module.project_query_and_latent

    def latent(hidden_states):
        out = own_latent(hidden_states)
        latents.append(out)
        return out

    def query(hidden_states, **kwargs):
        handed.append(kwargs.get("q_latent"))
        return own_query(hidden_states, **kwargs)

    with mock.patch.object(module, "project_query_latent", latent), \
            mock.patch.object(module, "project_query_and_latent", query):
        yield latents, handed


def _assert_banks_equal(got: dict, want: dict) -> None:
    assert torch.equal(got["latent_cache"], want["latent_cache"])
    assert torch.equal(got["pool_bank"], want["pool_bank"])
    assert torch.equal(got["tail_bank"], want["tail_bank"])


CASES = [(regime, batch, rows) for regime in ("bypass", "selecting")
         for batch, rows in ((1, 1), (1, 4), (2, 1))]


def test_the_windows_sit_either_side_of_the_bypass_bound():
    windows = _windows()
    assert _dense(windows["bypass"]) and not _dense(windows["selecting"])
    for regime in windows:
        for rows in ROWS:
            assert all(s + rows <= windows[regime] for s in _starts(regime, rows))


@pytest.mark.parametrize("regime, batch, rows", CASES)
def test_the_step_equals_the_two_projection_form_bit_for_bit(regime, batch, rows):
    ops = _operands(regime, batch, rows, seed=31 + 10 * batch + rows)
    mine, ref = _cloned(ops), _cloned(ops)
    out = _step(mine)
    want = _two_projection_step(ref)
    assert torch.equal(out, want)
    _assert_banks_equal(mine, ref)


@pytest.mark.parametrize("batch, rows", [(1, 1), (1, 4), (2, 1)])
def test_a_selecting_step_hands_the_indexer_s_latent_to_the_attention(batch, rows):
    ops = _operands("selecting", batch, rows, seed=47 + 10 * batch + rows)
    with _recorded() as (latents, handed):
        _step(ops)
    assert len(latents) == 1
    assert len(handed) == 1 and handed[0] is latents[0]


@pytest.mark.parametrize("batch, rows", [(1, 1), (1, 4), (2, 1)])
def test_a_bypass_step_keeps_the_attention_s_own_projection(batch, rows):
    # The indexer's latent has no reader here (no query side); the graph drops it.
    ops = _operands("bypass", batch, rows, seed=53 + 10 * batch + rows)
    with _recorded() as (latents, handed):
        _step(ops)
    assert handed == [None]
    assert len(latents) == 2


def test_a_shared_draft_iteration_keeps_the_attention_s_own_projection():
    # Iteration 0 selects and publishes; iteration 1, one position on (the same page),
    # reuses its selection, runs no indexer and so has no latent to hand over.
    edge = _starts("selecting", 1)[1]
    first = _operands("selecting", 1, 1, seed=61, starts=(edge,))
    assert (edge + 1) // PAGE == edge // PAGE
    share = model_fp8.IndexShare()
    _step(first, index_share=share)
    assert share.topk_indices is not None
    second = _operands("selecting", 1, 1, seed=62, starts=(edge,))
    second["start"] = first["start"] + 1
    second["seq_lens"] = first["seq_lens"] + 1
    second["latent_slots"] = first["latent_slots"] + 1
    for name in ("latent_cache", "pool_bank", "tail_bank", "block_table", "state_slots"):
        second[name] = first[name]
    with _recorded() as (latents, handed):
        _step(second, index_share=share)
    assert handed == [None]
    assert len(latents) == 1


@pytest.mark.parametrize("rows", ROWS)
def test_the_query_and_latent_from_a_handed_latent_are_its_own(rows):
    module = _module()
    hidden = _operands("selecting", 1, rows, seed=71)["hidden"]
    query, kv_latent = module.project_query_and_latent(hidden)
    handed = module.project_query_and_latent(
        hidden, q_latent=module.project_query_latent(hidden))
    assert torch.equal(handed[0], query)
    assert torch.equal(handed[1], kv_latent)


def test_a_latent_that_does_not_pair_with_the_rows_is_refused_by_name():
    module = _module()
    hidden = _operands("selecting", 1, 4, seed=73)["hidden"]
    latent = module.project_query_latent(hidden)
    for wrong in (latent[:3], latent[:, :-PAGE], latent.to(torch.bfloat16)):
        with pytest.raises(model_fp8.Glm5NextMLADecodeError, match="q_latent"):
            module.project_query_and_latent(hidden, q_latent=wrong)
