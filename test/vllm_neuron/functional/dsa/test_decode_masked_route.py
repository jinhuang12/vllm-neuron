# SPDX-License-Identifier: Apache-2.0
"""Which decode steps attend through ``mla_decode_masked``, and what they keep.

A selecting step of one request and several rows (the speculative verify step at
``B = 1``) takes its selection as a pool mask (``dsa_decode_select_mask``) and attends
through ``mla_decode_masked_attention``, which reads the window once for all the rows.
Every other step keeps its route: one row, several requests, the bypass window, and a
draft iteration that shares the first iteration's token indices.

The reference is the same step through the index form (the selection as token indices,
``mla_decode``'s selected kernel). The token set is the same, so the banks are the same
bit for bit and the output agrees to the two kernels' rounding through the bf16 cast
before ``W_UV`` (``test_mla_decode_masked.py`` holds the kernels to 2e-5 relative).

Shapes are ``test_decode_query_latent.py``'s: this checkpoint's per-rank decode shape on
the windows either side of the bypass bound.
"""

from __future__ import annotations

from unittest import mock

import pytest
import torch

import vllm_neuron.model.glm5_next.model_fp8 as model_fp8
from test.vllm_neuron.functional.dsa.test_decode_query_latent import (
    _assert_banks_equal,
    _cloned,
    _module,
    _operands,
    _step,
)
from vllm_neuron.functional.attention import mla_decode as MD
from vllm_neuron.functional.attention import mla_decode_masked as MM
from vllm_neuron.functional.dsa import decode_batch as DB


def _reset():
    MM.reset_mla_decode_masked_dispatch_counters()
    MD.reset_mla_decode_dispatch_counters()
    DB.reset_decode_batch_dispatch_counters()


def _index_form():
    """The same step with the selection as token indices (the masked kernel declined)."""
    return mock.patch.object(_module(), "_masked_attention_serves", lambda *a: False)


def test_a_one_request_verify_step_that_selects_attends_by_pool_mask():
    ops = _operands("selecting", 1, 4, seed=83)
    mine, ref = _cloned(ops), _cloned(ops)
    _reset()
    out = _step(mine)
    assert MM.mla_decode_masked_dispatch_counters() == (1, 0)
    assert MD.mla_decode_dispatch_counters() == (0, 0)
    # One selection launch, the mask entry, in decode_batch's family.
    assert DB.decode_batch_route_counts()[3] == 1
    _reset()
    with _index_form():
        want = _step(ref)
    assert MM.mla_decode_masked_dispatch_counters() == (0, 0)
    assert MD.mla_decode_route_counts()[:2] == (0, 1)
    _assert_banks_equal(mine, ref)
    torch.testing.assert_close(out.float(), want.float(), rtol=2e-2, atol=2e-3)


@pytest.mark.parametrize("regime, batch, rows", [
    ("selecting", 1, 1), ("selecting", 2, 1), ("selecting", 2, 4), ("bypass", 1, 4),
])
def test_every_other_step_keeps_its_route(regime, batch, rows):
    ops = _operands(regime, batch, rows, seed=89 + 10 * batch + rows)
    _reset()
    _step(ops)
    assert MM.mla_decode_masked_dispatch_counters() == (0, 0)


def test_a_draft_head_carrier_keeps_token_indices():
    # The carrier hands its selection to later iterations as token indices, whatever
    # the rows.
    ops = _operands("selecting", 1, 4, seed=97)
    share = model_fp8.IndexShare()
    _reset()
    _step(ops, index_share=share)
    assert MM.mla_decode_masked_dispatch_counters() == (0, 0)
    assert MD.mla_decode_route_counts()[:2] == (0, 1)
    assert share.topk_indices is not None and share.topk_indices.dtype == torch.int32


def test_a_pool_mask_of_a_step_that_does_not_select_is_refused_before_any_write():
    module = _module()
    ops = _operands("bypass", 1, 4, seed=101)
    before = _cloned(ops)
    hidden = ops["hidden"]
    q_latent = module.project_query_latent(hidden)
    with pytest.raises(model_fp8.Glm5NextDSAIndexerError, match="pool mask"):
        module.indexer.forward_requests(
            hidden, q_latent, ops["pool_bank"], ops["tail_bank"], ops["state_slots"],
            ops["seq_lens"], ops["start"], max_seq_len=ops["window"],
            indices_wanted=False, pool_mask=True)
    _assert_banks_equal(ops, before)
