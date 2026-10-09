# SPDX-License-Identifier: Apache-2.0
"""One DSA/MLA layer's decode step with ``T`` tokens per request: the verify step's legs.

``Glm5NextMLAAttention._forward_requests`` takes ``[B * T, hidden]`` rows, request-major,
with ``position``/``start_position`` ``[B]`` (row 0's position per request) and
``seq_lens``/``latent_slots`` ``[B * T]`` (row ``b * T + t``'s causal length ``position[b]
+ t + 1`` and bank row). The indexer advances each ring by ``T`` tokens, writes every
pool a row closes, scores and selects each row at its own causal length, and the attention
attends each row's own prefix (dense) or selection; the step's own rows are stood in from
its operands, never read back from a bank written in the same step.

The reference is the path the step must equal: ``T`` sequential one-token steps of the
same layer, each writing its banks before the next. Outputs, the latent cache, the pooled
stores (but the trash row, nobody's value) and the rings are compared bit for bit, in the
bypass regime (a 2048-token window attends densely) and the selecting one (8192), at the
served ring depth for ``T <= 2`` and at the deeper ring a speculative config allocates.
The rollback test proposes ``T`` rows, keeps ``a`` of them, steps on, and matches the
path that never saw the rejected tokens -- output and every bank.

Two one-row attention kernels exist (``mla_decode._route``): at the served per-rank shape
(one head, latent 512, 128-row pages) a one-row call runs the key-split kernel, every
request on both programs; a ``T``-row call always runs the general kernel. The bit-for-bit
reference therefore takes its one-token steps through the general kernel
(:func:`_general_kernel_only`); the same sequence as dispatched -- the key-split kernel at
every attention launch -- is matched to the two kernels' rounding through the bf16 cast
before ``W_UV`` (the bound ``test_mla_decode_split.py`` holds the kernels to is 2e-5
relative; the cast turns that into single bf16 ulps), and its banks bit for bit (no bank
row depends on the attention output within a step).
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
from vllm_neuron.functional.attention import mla_decode as MD
from vllm_neuron.functional.dsa import decode_batch as DB
from vllm_neuron.functional.dsa import decode_tail_update as TU
from vllm_neuron.functional.dsa import decode_trow as TR
from vllm_neuron.utils.neuron_utils import can_run_kernel

ROWS = (1, 2, 4, 6)
BATCHES = (1, 4)
#: (max_seq_len, row 0's positions): the bypass window and the selecting one, every
#: residue mod 4, one request against the window's last pool.
REGIMES = {"bypass": (2048, (2040, 100, 1023, 511)), "selecting": (8192, (7990, 2100, 2101, 4094))}


@functools.lru_cache(maxsize=1)
def _module():
    assert can_run_kernel(), "run with VLLM_NEURON_CPU_MODE=1 and NKI_SIMULATOR=1"
    return build_attention(model_fp8, decode_config(hidden_size=512, q_lora_rank=256))


def _operands(batch, rows, max_seq_len, starts, *, depth, seed, spare_slots=3,
              spare_pages=3):
    """One step of ``batch`` requests x ``rows`` tokens at ``starts[b] .. starts[b] + rows
    - 1``, on the runner's layout: a paged latent bank, a pooled store and a ring per state
    slot, each request on pages and a slot no other request holds."""
    cfg = _module().indexer
    gen = torch.Generator().manual_seed(seed)
    pool, head_dim = int(cfg.index_kpool), int(cfg.index_head_dim)
    latent = int(_module().kv_lora_rank)
    pages = int(max_seq_len) // PAGE
    start = torch.tensor(list(starts)[:batch], dtype=torch.int64)
    used = [-(-(int(s) + rows) // PAGE) for s in start]
    bank_pages = sum(used) + spare_pages
    order = torch.randperm(bank_pages, generator=gen).to(torch.int32)
    table = torch.full((batch, pages), -1, dtype=torch.int32)
    first = 0
    for b, n in enumerate(used):
        table[b, :n] = order[first:first + n]
        first += n
    positions = (start[:, None] + torch.arange(rows)[None, :]).reshape(-1)
    request = torch.arange(batch).repeat_interleave(rows)
    latent_slots = (table[request, positions // PAGE].to(torch.int64) * PAGE
                    + positions % PAGE)
    slots_total = batch + spare_slots
    hidden = int(_module().hidden_size)
    return {
        "hidden": (torch.randn(batch * rows, hidden, generator=gen) * 0.5).to(torch.bfloat16),
        "latent_cache": (torch.randn(bank_pages * PAGE, 1, latent, generator=gen) * 0.5
                         ).to(torch.bfloat16),
        "pool_bank": (torch.randn(slots_total, pool_rows(max_seq_len, pool), head_dim,
                                  generator=gen) * 0.5).to(torch.bfloat16),
        "tail_bank": (torch.randn(slots_total, 2, depth, head_dim, generator=gen) * 0.5
                      ).to(torch.bfloat16),
        "state_slots": torch.randperm(slots_total, generator=gen)[:batch].to(torch.int32),
        "start": start,
        "block_table": table,
        "latent_slots": latent_slots,
        "softmax_scale": float(int(_module().qk_nope_head_dim)) ** -0.5,
        "max_seq_len": int(max_seq_len),
        "rows": rows,
    }


def _cloned(ops):
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in ops.items()}


def _step(ops, hidden, start, latent_slots, rows, *, index_share=None):
    """The layer's batched decode leg on ``rows`` tokens per request, banks written in
    place. ``[B * rows, hidden]``."""
    seq_lens = (start[:, None] + torch.arange(rows)[None, :] + 1).reshape(-1)
    return _module()._forward_requests(
        hidden,
        latent_cache=ops["latent_cache"],
        pool_cache=ops["pool_bank"],
        seq_lens=seq_lens.to(torch.int32),
        start_position=start,
        softmax_scale=ops["softmax_scale"],
        max_seq_len=ops["max_seq_len"],
        page_size=PAGE,
        block_table_row=ops["block_table"].t().contiguous(),
        latent_slots=latent_slots,
        tail=ops["tail_bank"],
        position=start,
        collector=None,
        state_slots=ops["state_slots"],
        index_share=index_share,
    )


def _rows_step(ops):
    return _step(ops, ops["hidden"], ops["start"], ops["latent_slots"], ops["rows"])


def _one_token_step_batched_attention(ops, hidden, start, latent_slots):
    """One token per request through the leg's own pieces, the attention batched.

    At ``B = 1`` the production one-token step attends through the one-request sparse
    kernel (``attend`` at ``batch_size`` 1), whose rounding is not the batched decode
    kernel's; a ``T``-row step attends batched whatever ``B``. This runs
    :meth:`_forward_requests`'s chain with :meth:`_attend_requests` in place of
    :meth:`attend`, so a ``B = 1`` reference meets the rows step kernel for kernel.
    """
    from vllm_neuron.functional.dsa.decode_bypass import selection_bound, selection_is_a_no_op

    module = _module()
    indexer = module.indexer
    window = int(ops["block_table"].shape[1]) * PAGE
    bound = selection_bound(int(ops["max_seq_len"]), window)
    dense = selection_is_a_no_op(bound, indexer.index_topk, indexer.index_kpool)
    q_latent = module.project_query_latent(hidden)
    projected = indexer.project_stage(hidden, q_latent, query_side=not dense)
    topk = indexer.forward_requests(
        hidden, q_latent, ops["pool_bank"], ops["tail_bank"], ops["state_slots"],
        (start + 1).to(torch.int32), start,
        max_seq_len=bound if dense else int(ops["max_seq_len"]), indices_wanted=not dense,
        projected=projected)
    return module._attend_requests(
        hidden, ops["latent_cache"], start, topk, ops["softmax_scale"], int(start.shape[0]),
        prefill_end_position=None, collector=None, active_mla_query_rows=None,
        block_table_row=ops["block_table"].t().contiguous(), latent_slots=latent_slots,
        page_size=PAGE, dense=dense)


@contextlib.contextmanager
def _general_kernel_only():
    """Route every one-row attention call through the general kernel.

    As dispatched (``mla_decode._route``), a one-row call at the served per-rank shape runs
    the key-split kernel, whose rounding is not the general kernel's; the ``T``-row step
    runs the general kernel, so the bit-for-bit reference is the general kernel's sequence.
    """
    with mock.patch.object(MD, "_split_serves", lambda *_: False):
        yield


def _sequential(ops, hidden, start, latent_slots, rows, *, production=False, general=True):
    """``rows`` one-token steps per request on the same banks, in position order.

    The production leg at ``B > 1``; at ``B = 1`` the same chain with the batched
    attention (see :func:`_one_token_step_batched_attention`) unless ``production``.
    ``general`` takes every attention launch through the general kernel (the bit-for-bit
    reference); ``general=False`` dispatches as served (the key-split kernel at this shape).
    """
    batch = int(start.shape[0])
    out = torch.empty(batch * rows, int(hidden.shape[1]), dtype=hidden.dtype)
    with _general_kernel_only() if general else contextlib.nullcontext():
        for t in range(rows):
            picked = torch.arange(batch) * rows + t
            if batch > 1 or production:
                out[picked] = _step(ops, hidden[picked], start + t, latent_slots[picked], 1)
            else:
                out[picked] = _one_token_step_batched_attention(
                    ops, hidden[picked], start + t, latent_slots[picked])
    return out


def _assert_served_sequence_agrees(out, mine, ops, rows, *, hidden=None, start=None,
                                   latent_slots=None, before=None):
    """The same sequence as dispatched: every attention launch the key-split kernel;
    output within the two kernels' rounding, banks bit for bit. ``before`` replays earlier
    steps on the clone first (the rollback's accepted rows)."""
    served = _cloned(ops)
    if before is not None:
        before(served)
    _reset_counters()
    want = _sequential(served, ops["hidden"] if hidden is None else hidden,
                       ops["start"] if start is None else start,
                       ops["latent_slots"] if latent_slots is None else latent_slots, rows,
                       general=False)
    launches = MD.mla_decode_dispatch_counters()[0]
    assert MD.mla_decode_split_counts()[0] == launches, (MD.mla_decode_split_counts(), launches)
    torch.testing.assert_close(out.float(), want.float(), rtol=2e-2, atol=2e-3)
    _assert_banks_equal(mine, served)


def _assert_banks_equal(got, want):
    assert torch.equal(got["latent_cache"], want["latent_cache"])
    assert torch.equal(got["pool_bank"][:, :-1], want["pool_bank"][:, :-1])
    assert torch.equal(got["tail_bank"], want["tail_bank"])


def _reset_counters():
    DB.reset_decode_batch_dispatch_counters()
    TU.reset_decode_tail_dispatch_counters()
    TR.reset_decode_trow_dispatch_counters()
    MD.reset_mla_decode_dispatch_counters()


@pytest.mark.parametrize("regime", sorted(REGIMES))
@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("rows", ROWS)
def test_the_rows_step_equals_sequential_one_token_steps_bit_for_bit(rows, batch, regime):
    max_seq_len, starts = REGIMES[regime]
    depth = TU.ring_depth_for(int(_module().indexer.index_kpool), max(ROWS))
    ops = _operands(batch, rows, max_seq_len, starts, depth=depth, seed=11 + rows + batch)
    mine, ref = _cloned(ops), _cloned(ops)
    _reset_counters()
    out = _rows_step(mine)
    # One launch per stage, every one a kernel: the ring (decode_tail_update); selecting
    # only, the scores (decode_trow) and the selection; then the attention (mla_decode).
    # The selection is one kernel, decode_select.dsa_decode_select (top-k, causal
    # sentinel, order and expansion together), and it counts in decode_batch's family,
    # so that family reads one launch here, its select entry, and no ring or scores.
    selecting = regime == "selecting"
    assert TU.decode_tail_dispatch_counters() == (1, 0)
    assert TR.decode_trow_dispatch_counters() == ((1, 0) if selecting else (0, 0))
    assert DB.decode_batch_dispatch_counters() == ((1, 0) if selecting else (0, 0))
    ring, scores, _, select = DB.decode_batch_route_counts()
    assert (ring, scores, select) == (0, 0, int(selecting))
    if batch == 1 and rows == 1 and regime == "selecting":
        # The one-request one-token step keeps its one-request sparse attention.
        assert MD.mla_decode_dispatch_counters() == (0, 0)
    else:
        assert MD.mla_decode_dispatch_counters() == (1, 0)
        assert MD.mla_decode_route_counts()[:2] == ((1, 0) if regime == "bypass" else (0, 1))
    # A T-row step runs the general kernel, so its bit-for-bit reference is the general
    # kernel's sequence; a one-row step dispatches as served (the key-split kernel here),
    # and so does its reference.
    want = _sequential(ref, ops["hidden"], ops["start"], ops["latent_slots"], rows,
                       general=rows > 1)
    assert out.shape == (batch * rows, int(_module().hidden_size))
    if batch == 1 and regime == "selecting":
        # Kernel for kernel the rows step equals the sequence; the production
        # one-token path at B = 1 attends through the one-request sparse kernel, so
        # against it the output agrees to the two kernels' rounding, and every bank is
        # the same bit for bit (the indexer chain is one and the same).
        if rows == 1:
            torch.testing.assert_close(out.float(), want.float(), rtol=2e-2, atol=2e-3)
        else:
            assert torch.equal(out, want)
        prod = _cloned(ops)
        want_prod = _sequential(prod, ops["hidden"], ops["start"], ops["latent_slots"], rows,
                                production=True)
        torch.testing.assert_close(out.float(), want_prod.float(), rtol=2e-2, atol=2e-3)
        _assert_banks_equal(mine, prod)
    else:
        assert torch.equal(out, want)
    _assert_banks_equal(mine, ref)
    assert not torch.equal(mine["tail_bank"], ops["tail_bank"])
    _assert_served_sequence_agrees(out, mine, ops, rows)


@pytest.mark.parametrize("regime", sorted(REGIMES))
@pytest.mark.parametrize("rows", (1, 2))
def test_the_served_ring_depth_takes_one_and_two_rows(rows, regime):
    """At depth index_kpool the one-token step keeps decode_batch's kernels (unchanged),
    and two rows equal two of them."""
    max_seq_len, starts = REGIMES[regime]
    pool = int(_module().indexer.index_kpool)
    ops = _operands(4, rows, max_seq_len, starts, depth=pool, seed=31 + rows)
    mine, ref = _cloned(ops), _cloned(ops)
    _reset_counters()
    out = _rows_step(mine)
    # One launch per stage, every one a kernel. One row: decode_batch's ring step, and,
    # selecting, its scores and the selection kernel (decode_select, same family). Two
    # rows: the T-row ring and scores, and the same selection kernel.
    selecting = regime == "selecting"
    ring, scores, _, select = DB.decode_batch_route_counts()
    if rows == 1:
        assert DB.decode_batch_dispatch_counters() == ((3, 0) if selecting else (1, 0))
        assert (ring, scores, select) == (1, int(selecting), int(selecting))
        assert TU.decode_tail_dispatch_counters() == (0, 0)
        assert TR.decode_trow_dispatch_counters() == (0, 0)
    else:
        assert TU.decode_tail_dispatch_counters() == (1, 0)
        assert TR.decode_trow_dispatch_counters() == ((1, 0) if selecting else (0, 0))
        assert DB.decode_batch_dispatch_counters() == ((1, 0) if selecting else (0, 0))
        assert (ring, scores, select) == (0, 0, int(selecting))
    want = _sequential(ref, ops["hidden"], ops["start"], ops["latent_slots"], rows,
                       general=rows > 1)
    assert torch.equal(out, want)
    _assert_banks_equal(mine, ref)
    _assert_served_sequence_agrees(out, mine, ops, rows)


@pytest.mark.parametrize("accepted", (1, 2, 3))
def test_rollback_leaves_no_rejected_state_the_next_step_reads(accepted):
    rows = 4
    max_seq_len, starts = REGIMES["selecting"]
    depth = TU.ring_depth_for(int(_module().indexer.index_kpool), max(ROWS))
    ops = _operands(4, rows, max_seq_len, starts, depth=depth, seed=51 + accepted)
    gen = torch.Generator().manual_seed(61 + accepted)
    hidden2 = (torch.randn(4 * rows, int(_module().hidden_size), generator=gen) * 0.5
               ).to(torch.bfloat16)
    start2 = ops["start"] + accepted
    positions2 = (start2[:, None] + torch.arange(rows)[None, :]).reshape(-1)
    request = torch.arange(4).repeat_interleave(rows)
    slots2 = (ops["block_table"][request, positions2 // PAGE].to(torch.int64) * PAGE
              + positions2 % PAGE)
    # The speculative path: every proposed row, then the next step from the first
    # rejected position.
    spec = _cloned(ops)
    _rows_step(spec)
    out2 = _step(spec, hidden2, start2, slots2, rows)
    # The honest path: the accepted rows one at a time, then the next step's rows one at
    # a time.
    picked = (torch.arange(4)[:, None] * rows + torch.arange(accepted)[None, :]).reshape(-1)
    honest = _cloned(ops)
    _sequential(honest, ops["hidden"][picked], ops["start"], ops["latent_slots"][picked],
                accepted)
    want2 = _sequential(honest, hidden2, start2, slots2, rows)
    assert torch.equal(out2, want2)
    _assert_banks_equal(spec, honest)
    _assert_served_sequence_agrees(
        out2, spec, ops, rows, hidden=hidden2, start=start2, latent_slots=slots2,
        before=lambda clone: _sequential(clone, ops["hidden"][picked], ops["start"],
                                         ops["latent_slots"][picked], accepted, general=False))


def test_a_shared_selection_must_match_the_rows():
    """A T-row step publishes [B * T, width]; an iteration with another row count refuses."""
    max_seq_len, starts = REGIMES["selecting"]
    depth = TU.ring_depth_for(int(_module().indexer.index_kpool), max(ROWS))
    ops = _operands(2, 4, max_seq_len, starts, depth=depth, seed=71)
    share = model_fp8.IndexShare()
    _step(ops, ops["hidden"], ops["start"], ops["latent_slots"], 4, index_share=share)
    assert tuple(share.topk_indices.shape)[0] == 8
    with pytest.raises(model_fp8.Glm5NextMLADecodeError, match="index_share"):
        _step(ops, ops["hidden"][:2], ops["start"] + 4, ops["latent_slots"][:2], 1,
              index_share=share)


def test_refusals_name_the_rows_and_the_depth():
    max_seq_len, starts = REGIMES["bypass"]
    pool = int(_module().indexer.index_kpool)
    ops = _operands(4, 3, max_seq_len, starts, depth=pool, seed=81)
    with pytest.raises(model_fp8.Glm5NextDSAIndexerError, match="rows"):
        _rows_step(_cloned(ops))
    ops = _operands(4, 2, max_seq_len, starts, depth=pool, seed=82)
    with pytest.raises(model_fp8.Glm5NextMLADecodeError, match="whole number of rows"):
        _step(_cloned(ops), ops["hidden"][:7], ops["start"], ops["latent_slots"][:7], 2)
