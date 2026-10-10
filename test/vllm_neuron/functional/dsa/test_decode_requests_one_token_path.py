# SPDX-License-Identifier: Apache-2.0
"""``Glm5NextDSAIndexer.forward_requests`` at one token per request traces no row
bookkeeping.

Request ``b``'s row ``t`` sits at ``position[b] + t`` and writes its pool through slot
``slots[b]``, so a step of ``T`` rows per request spells out ``[B * T]`` row positions
(``torch.arange(T)`` added to ``position``) and ``[B * T]`` row slots
(``slots.repeat_interleave(T)``). At ``T = 1`` both are the request's own values, and
the decode step of one token per request -- every request of a non-speculative decode
and every draft iteration -- must not trace them: in the compiled graph each would be a
few nodes per DSA layer that compute an identity.

Pinned here, on the leg the attention layer compiles (``test_decode_host_path._leg``,
``fullgraph=True``) in both store forms and both regimes:

* (a) at ``T = 1`` the graph holds no ``repeat_interleave`` and no ``arange`` but the
  carrier form's slot numbering;
* (b) at ``T > 1`` it still holds both, so (a) is a property of the one-token step and
  not of the probe;
* (c) eagerly, at every ``T``, the pool-write destinations are bit for bit the
  ``[B * T]`` row formula's, and a wrong ``seq_lens`` at ``T = 1`` is still refused by
  name.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/functional/dsa/test_decode_requests_one_token_path.py
"""

from __future__ import annotations

import collections

import pytest
import torch

import vllm_neuron.model.glm5_next.model_fp8 as model_fp8
from test.vllm_neuron.functional.dsa.dsa_batch_case import pool_rows
from test.vllm_neuron.functional.dsa.dsa_decode_case import decode_config
from test.vllm_neuron.functional.dsa.test_decode_host_path import (
    SERVED_LNC,
    _indexer,
    _leg,
    _Recorder,
)
from vllm_neuron.functional.dsa import decode_batch as DB
from vllm_neuron.functional.dsa import decode_tail_update as TU
from vllm_neuron.utils.neuron_utils import can_run_kernel

pytestmark = [pytest.mark.fast]

#: The selecting regime and the one inside the selection bound (the host-path test's).
CONTEXTS = {"selecting": 4096, "inside-the-bound": 2048}
FORMS = ("bank", "carrier")


@pytest.fixture(autouse=True)
def _kernels_on(monkeypatch):
    assert can_run_kernel(), "run with VLLM_NEURON_CPU_MODE=1 and NKI_SIMULATOR=1"
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", SERVED_LNC)


def _config():
    return decode_config(hidden_size=512, q_lora_rank=256)


def _operands(cfg, ctx: int, batch: int, rows: int, form: str, seed: int) -> dict:
    """One decode step of ``batch`` requests x ``rows`` tokens, request-major, ending at
    ``ctx - 1 - b % 4``; the ring at the depth a ``rows``-token step needs."""
    gen = torch.Generator().manual_seed(seed)
    heads, dim, pool = int(cfg.index_n_heads), int(cfg.index_head_dim), int(cfg.index_kpool)
    total = batch * rows
    starts = torch.tensor([ctx - rows - b % 4 for b in range(batch)], dtype=torch.int64)
    seq_lens = (starts[:, None] + torch.arange(rows)[None, :] + 1).reshape(-1)
    slots_total = batch + 3
    pool_bank = (torch.randn(slots_total, pool_rows(ctx, pool), dim, generator=gen) * 0.5
                 ).to(torch.bfloat16)
    tail_bank = (torch.randn(slots_total, 2, TU.ring_depth_for(pool, rows), dim,
                             generator=gen) * 0.5).to(torch.bfloat16)
    slots = torch.randperm(slots_total, generator=gen)[:batch].to(torch.int32)
    ops = {
        "query_rows": torch.randn(total * heads, dim, generator=gen).to(torch.bfloat16),
        "key": torch.randn(total, dim, generator=gen).to(torch.bfloat16),
        "weights": torch.randn(total, heads, generator=gen) * (dim ** -0.5) * (heads ** -0.5),
        "gate": torch.randn(total, dim, generator=gen).to(torch.bfloat16),
        "pool_bank": pool_bank,
        "tail_bank": tail_bank,
        "slots": slots,
        "seq_lens": seq_lens.to(torch.int32),
        "position": starts,
    }
    if form == "carrier":
        # One view per request, the runner's per-request carrier; no slots.
        ops["pool_bank"] = tuple(pool_bank[int(s)] for s in slots)
        ops["tail_bank"] = tuple(tail_bank[int(s)] for s in slots)
        ops["slots"] = None
    return ops


def _traced(ctx: int, batch: int, rows: int, form: str):
    """The leg's one graph at this shape, and its call targets by name."""
    cfg = _config()
    recorder = _Recorder()
    torch._dynamo.reset()
    compiled = torch.compile(_leg(_indexer(cfg), ctx), backend=recorder, fullgraph=True)
    compiled(**_operands(cfg, ctx, batch, rows, form, seed=ctx + batch + rows))
    assert len(recorder.graphs) == 1, f"{len(recorder.graphs)} graphs, not one"
    graph = recorder.graphs[0].graph
    calls = [node for node in graph.nodes if node.op in ("call_function", "call_method")]
    targets = collections.Counter(
        node.target if isinstance(node.target, str) else node.target.__name__
        for node in calls)
    return calls, targets


@pytest.mark.parametrize("ctx", CONTEXTS.values(), ids=CONTEXTS.keys())
@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("form", FORMS)
def test_a_one_token_step_traces_no_row_bookkeeping(ctx, batch, form):
    """(a) One token per request: the rows are the requests, so no row offsets and no
    per-row slots. The carrier form numbers its views ``arange(B)`` as their slots; that
    is the only ``arange`` left."""
    calls, targets = _traced(ctx, batch, 1, form)
    aranges = [node for node in calls if getattr(node.target, "__name__", "") == "arange"]
    slot_numbering = [node for node in aranges if form == "carrier"
                      and node.args == (batch,)
                      and node.kwargs.get("dtype") == torch.int32]
    stray = [node.format_node() for node in aranges if node not in slot_numbering]
    assert (targets["repeat_interleave"], stray) == (0, []), (
        f"the one-token step traces row bookkeeping: {targets['repeat_interleave']} "
        f"repeat_interleave node(s) and the aranges {stray}")
    assert len(slot_numbering) == (1 if form == "carrier" else 0)


@pytest.mark.parametrize("rows", [2, 4])
def test_a_rows_step_keeps_its_row_bookkeeping(rows):
    """(b) ``T > 1`` rows per request: the row offsets and the per-row slots are traced,
    so the probe above sees them where they exist."""
    calls, targets = _traced(CONTEXTS["selecting"], 4, rows, "bank")
    assert targets["repeat_interleave"] >= 1
    offsets = [node for node in calls if getattr(node.target, "__name__", "") == "arange"
               and node.args == (rows,)]
    assert offsets, f"no arange({rows}) row offsets in the {rows}-row step's graph"


def _row_formula(slots, position, *, rows: int, store_rows: int, pool: int):
    """The destinations of a ``rows``-row step: request ``b``'s row ``t`` at slot
    ``slots[b]`` and position ``position[b] + t``."""
    offsets = torch.arange(rows, dtype=torch.int64)
    row_positions = (position.to(torch.int64)[:, None] + offsets[None, :]).reshape(-1)
    return DB.decode_pool_destinations(slots.repeat_interleave(rows), row_positions,
                                       rows=store_rows, pool_size=pool)


@pytest.mark.parametrize("ctx", CONTEXTS.values(), ids=CONTEXTS.keys())
@pytest.mark.parametrize("rows", [1, 2, 4])
@pytest.mark.parametrize("form", FORMS)
def test_the_destinations_are_the_row_formula_bit_for_bit(monkeypatch, ctx, rows, form):
    """(c) Eagerly, every ``T``: the ``(slot, row)`` each pooled row is written to is the
    row formula's, dtype and value."""
    cfg = _config()
    batch = 4
    ops = _operands(cfg, ctx, batch, rows, form, seed=ctx + 31 * rows)
    seen = []
    real = DB.decode_pool_destinations

    def spy(*args, **kwargs):
        seen.append(real(*args, **kwargs))
        return seen[-1]

    monkeypatch.setattr(DB, "decode_pool_destinations", spy)
    _leg(_indexer(cfg), ctx)(**ops)
    assert len(seen) == 1
    slots = (torch.arange(batch, dtype=torch.int32) if form == "carrier"
             else ops["slots"])
    store_rows = pool_rows(ctx, int(cfg.index_kpool))
    want = _row_formula(slots, ops["position"], rows=rows, store_rows=store_rows,
                        pool=int(cfg.index_kpool))
    for got_part, want_part in zip(seen[0], want):
        assert got_part.dtype == want_part.dtype == torch.int64
        assert torch.equal(got_part, want_part)


@pytest.mark.parametrize("form", FORMS)
def test_a_one_token_step_refuses_seq_lens_that_are_not_position_plus_one(form):
    """(c) The eager check still reads every row: one request's length off by one is
    refused by name, before anything is written."""
    cfg = _config()
    ctx = CONTEXTS["selecting"]
    ops = _operands(cfg, ctx, 4, 1, form, seed=11)
    ops["seq_lens"] = ops["seq_lens"].clone()
    ops["seq_lens"][2] += 1
    with pytest.raises(model_fp8.Glm5NextDSAIndexerError, match="seq_lens - 1"):
        _leg(_indexer(cfg), ctx)(**ops)
