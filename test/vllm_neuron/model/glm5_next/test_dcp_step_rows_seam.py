# SPDX-License-Identifier: Apache-2.0
"""The three per-row DCP builders of a ``T``-row step agree, row for row.

A speculative verify step under decode context parallelism hands every rank ``T = 1 + k``
query rows per request; row ``t`` of request ``b`` sits at ``start[b] + t``. Three call
sites build the per-row pair ``(position, owner)`` from that rule: ``position`` is the
row's :func:`~vllm_neuron.utils.dcp_ownership.owned_length` on the rank and ``owner`` is
whether the rank owns the row's block.

* ``mla_decode.dcp_step_rows``, the helper the DCP partial-attention entry publishes;
* ``Glm5NextMLAAttention._dcp_decode``, which builds the pair inline and hands it to
  ``mla_decode_partial_attention`` (recorded here at that entry, which then stops);
* ``Glm5NextDSAIndexer.forward_requests``, which builds the row positions inline and
  hands them to ``_dcp_owned_position`` (the owner gate of the pool store) and
  ``_dcp_owned_lengths`` (the rows' bound), recorded there; the step stops after it.

For DCP 2 and 4, ``T`` 1, 2 and 4 and every rank of the group, with steps that cross a
128-token block (the two blocks' owners each hold part of the step), a step that crosses
block 0 into block 1, and a short request that leaves every rank but 0 an empty window,
the three pairs are equal, and equal to a plain-integer count of the interleave. The
indexer runs at both ring depths a step of ``T`` rows takes (the ``index_kpool`` ring
at ``T = 1``, the speculative ring at every ``T``).

No test builds an engine or touches a device; the indexer's kernels run on the CPU
simulator up to the point the pair is recorded.
"""

from __future__ import annotations

import pytest
import torch

import vllm_neuron.model.glm5_next.model_fp8 as model_fp8
from test.vllm_neuron.model.glm5_next import test_indexer_dcp_trow as indexer_side
from test.vllm_neuron.model.glm5_next import test_mtp_under_dcp as attention_side
from vllm_neuron.functional.attention import mla_decode
from vllm_neuron.functional.dsa import decode_trow
from vllm_neuron.utils.neuron_utils import can_run_kernel

BLOCK = 128
CPS = (2, 4)
ROWS = (1, 2, 4)
#: The indexer's window: the bypass regime of the indexer's own T-row tests.
WINDOW = 2048


def _starts(rows: int) -> list[int]:
    """Each request's first position: a step that ends just past the block boundary at
    1024 (rows on both sides at ``rows >= 2``), one that crosses block 0 into block 1, and
    a short request inside block 0, which leaves every rank but 0 an empty window."""
    back = rows // 2 if rows > 1 else 1
    return [8 * BLOCK - back, BLOCK - back, 5]


def _oracle(starts: list[int], rows: int, rank: int, cp: int) -> tuple[list[int], list[bool]]:
    """The pair counted from the interleave: block ``q // 128`` is rank ``(q // 128) % cp``'s,
    and a row's position on ``rank`` is the count of positions before it the rank holds."""
    positions, owners = [], []
    for start in starts:
        for t in range(rows):
            at = start + t
            positions.append(sum(1 for q in range(at) if (q // BLOCK) % cp == rank))
            owners.append((at // BLOCK) % cp == rank)
    return positions, owners


def _pair(position: torch.Tensor, owner: torch.Tensor) -> tuple[list[int], list[bool]]:
    return position.to(torch.int64).reshape(-1).tolist(), owner.to(torch.bool).reshape(-1).tolist()


def _from_helper(starts: list[int], rows: int, rank: int, cp: int):
    position, owner = mla_decode.dcp_step_rows(
        torch.tensor(starts, dtype=torch.int64), rank, rows=rows, block_size=BLOCK, dcp_size=cp
    )
    assert position.dtype == torch.int32 and owner.dtype == torch.int32
    return _pair(position, owner)


def _from_attention(monkeypatch, starts: list[int], rows: int, rank: int, cp: int, *,
                    selected: bool):
    seen = attention_side._record_partial_entry(monkeypatch)
    attention = attention_side._attention(cp, rank)
    q_lift, c_kv, table, start, written, topk = attention_side._decode_operands(
        len(starts), rows, starts, selected=selected
    )
    with pytest.raises(attention_side._Stop):
        attention._dcp_decode(q_lift, c_kv, table, start, written, 0.125, BLOCK,
                              topk_indices=topk)
    (call,) = seen
    return _pair(call["position"], call["owner"])


class _Recorded(Exception):
    pass


def _from_indexer(monkeypatch, starts: list[int], rows: int, rank: int, cp: int, *,
                  depth: int):
    """The real ``forward_requests`` step on ``rank``: the owner gate handed to the pool
    store and the owned lengths of the row positions, recorded as they are computed. The
    step stops at the second owned-length call (the rows' own; the first is
    ``seq_lens``'), before the scores."""
    indexer = indexer_side._module().indexer
    cls = type(indexer)
    recorded: dict = {}

    def owned_position(self, position, block_size):
        local, owned = cls._dcp_owned_position(self, position, block_size)
        recorded["gate_positions"] = position.to(torch.int64).reshape(-1).tolist()
        recorded["owner"] = owned
        return local, owned

    def owned_lengths(self, lengths, block_size):
        out = cls._dcp_owned_lengths(self, lengths, block_size)
        recorded.setdefault("lengths", []).append(
            (lengths.to(torch.int64).reshape(-1).tolist(), out)
        )
        if len(recorded["lengths"]) == 2:
            raise _Recorded
        return out

    monkeypatch.setattr(indexer, "_dcp_owned_position",
                        owned_position.__get__(indexer, cls), raising=False)
    monkeypatch.setattr(indexer, "_dcp_owned_lengths",
                        owned_lengths.__get__(indexer, cls), raising=False)
    case = indexer_side._case(rows, depth, WINDOW, seed=100 * cp + 10 * rows + rank)
    case["start"] = torch.tensor(starts, dtype=torch.int64)
    group = indexer_side._ReplayGroup(cp)
    local_window = indexer_side.OWN.local_context_capacity(WINDOW, block_size=BLOCK,
                                                           dcp_size=cp)
    with indexer_side._as_rank(indexer, rank, cp, group, monkeypatch):
        with pytest.raises(_Recorded):
            indexer_side._step(case, case["pool_bank"].clone(), case["tail_bank"].clone(),
                               local_window, page_size=BLOCK)
    row_positions = [s + t for s in starts for t in range(rows)]
    seq_lens, _ = recorded["lengths"][0]
    handed, position = recorded["lengths"][1]
    # What the indexer handed both calls is the step's row positions.
    assert recorded["gate_positions"] == row_positions
    assert handed == row_positions
    assert seq_lens == [p + 1 for p in row_positions]
    return _pair(position, recorded["owner"])


@pytest.fixture(autouse=True)
def _simulator(monkeypatch):
    monkeypatch.setenv("NKI_SIMULATOR", "1")
    assert can_run_kernel(), "run with VLLM_NEURON_CPU_MODE=1"


def _forms():
    for cp in CPS:
        for rows in ROWS:
            for rank in range(cp):
                yield cp, rows, rank


@pytest.mark.parametrize("cp, rows, rank", list(_forms()))
def test_the_three_builders_hand_the_same_pair(monkeypatch, cp, rows, rank) -> None:
    starts = _starts(rows)
    want = _oracle(starts, rows, rank, cp)
    assert _from_helper(starts, rows, rank, cp) == want, "mla_decode.dcp_step_rows"
    for selected in (False, True):
        assert _from_attention(monkeypatch, starts, rows, rank, cp, selected=selected) == want, (
            "Glm5NextMLAAttention._dcp_decode", selected)
    pool = int(indexer_side._module().indexer.index_kpool)
    depths = [decode_trow.indexer_ring_depth(pool, 3)] + ([pool] if rows == 1 else [])
    for depth in depths:
        assert _from_indexer(monkeypatch, starts, rows, rank, cp, depth=depth) == want, (
            "Glm5NextDSAIndexer.forward_requests", depth)


@pytest.mark.parametrize("cp", CPS)
def test_the_grid_holds_block_crossings_and_empty_windows(cp) -> None:
    """The cases above are the ones named: at every ``T >= 2`` the first two requests
    have rows on two owners, and the short request leaves every rank but 0 nothing."""
    for rows in ROWS:
        starts = _starts(rows)
        for b, start in enumerate(starts[:2]):
            owners = {((start + t) // BLOCK) % cp for t in range(rows)}
            assert len(owners) == (2 if rows >= 2 else 1), (rows, b, owners)
        for rank in range(1, cp):
            positions, owners = _oracle(starts[2:], rows, rank, cp)
            assert positions == [0] * rows and owners == [False] * rows, (rows, rank)
