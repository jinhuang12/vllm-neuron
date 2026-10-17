# SPDX-License-Identifier: Apache-2.0
"""The GLM-5.3-Flash draft head and the DCP decode call site under decode context parallelism.

Under decode context parallelism (DCP) the ``dcp_size`` ranks of a group split every
sequence by 128-token block (``vllm_neuron/utils/dcp_ownership.py``): block ``b`` lives on
rank ``b % dcp_size`` as its local block ``b // dcp_size``, and a request's block-table row
on a rank lists that rank's blocks only. These tests pin:

1. ``dcp_ownership.owned_latent_slots``, the one rule for the latent slot a rank writes a
   position to: its local row through the rank's table where it owns the position, the null
   slot where another rank does or the table names no block. It equals an oracle written
   from the ownership rule, and the runner's ``_dcp_owned_latent_slots`` returns the same.
2. ``Glm5NextMultiTokenPredictor._latent_slots_at`` at DCP 2 and 4, positions spanning two
   blocks and an unallocated next page: owned positions take the runner's slot, the others
   the null slot. At DCP 1 it is today's formula, bit for bit.
3. ``_iteration_carrier`` on a head whose layer is bound to DCP 2: draft iterations past the
   first carry the ownership slots, and the position clamp is the group's last position
   (the carrier's ``max_seq_len`` is the rank's share there).
4. ``Glm5NextMLAAttention._dcp_decode`` hands the DCP partial entry per-row operands for a
   step of ``T`` rows per request (``owned_length`` and the owner gate at ``p_b + t``) and a
   selection padded on all ``B * T`` rows; one row per request keeps today's operands.
5. The masked decode dispatch is not taken under DCP: a selecting step of one request and
   ``T > 1`` rows hands the indexer ``pool_scores=False`` at ``dcp_size > 1``.

No test runs a kernel, builds an engine or touches a device.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from vllm_neuron.utils import dcp_ownership

pytestmark = [pytest.mark.fast]

BLOCK = 128
NULL_SLOT = 0
#: The served form's per-rank share at DCP 2 (``local_context_capacity(131072, 128, 2)``).
SERVED_SHARE = 65536


def _oracle_slot(row: list[int], position: int, rank: int, *, block_size: int, dcp_size: int) -> int:
    """The slot rank ``rank`` writes ``position`` to, from the ownership rule in plain ints.

    Block ``b = position // block_size`` is rank ``b % dcp_size``'s local block
    ``b // dcp_size``; a rank that does not own the position, or whose table names no block
    there (``-1`` or past the row), writes the null slot."""
    block = position // block_size
    if block % dcp_size != rank:
        return NULL_SLOT
    entry = block // dcp_size
    if entry >= len(row) or row[entry] < 0:
        return NULL_SLOT
    return row[entry] * block_size + position % block_size


def _today_latent_slots_at(block_table_row, position, page_size, rows):
    """``_latent_slots_at`` as it reads on b5b516b3 (mtp.py :427-:444), the DCP 1 oracle."""
    device = block_table_row.device
    if torch.is_tensor(position):
        pos = position.to(device=device, dtype=torch.int64).reshape(-1)
    else:
        pos = torch.full((rows,), int(position), dtype=torch.int64, device=device)
    if int(pos.shape[0]) == 1 and rows > 1:
        pos = pos.expand(rows)
    table = block_table_row.to(torch.int64)
    pages, page = int(table.shape[0]), int(page_size)
    page_index = torch.div(pos, page, rounding_mode="floor").clamp(min=0, max=pages - 1)
    blocks = torch.gather(table, 0, page_index.reshape(1, rows)).reshape(rows)
    inside = (pos < pages * page) & (blocks >= 0)
    return torch.where(inside, blocks * page + pos % page, torch.zeros_like(pos))


def _head_class():
    from vllm_neuron.model.glm5_next.mtp import Glm5NextMultiTokenPredictor

    return Glm5NextMultiTokenPredictor


# --------------------------------------------------------------------------- #
# 1. the shared rule.


@pytest.mark.parametrize("dcp", [2, 4, 8])
def test_owned_latent_slots_follow_the_ownership_rule_on_every_rank(dcp: int) -> None:
    # Two requests: one with holes in its block numbers, one whose last entry is unallocated.
    rows = ([7, 3, 11], [5, 9, -1])
    entries = len(rows[0])
    table = torch.tensor(rows, dtype=torch.int32).t().contiguous()  # [entries, R]
    for rank in range(dcp):
        for position in range(0, entries * BLOCK * dcp + BLOCK, 13):
            got = dcp_ownership.owned_latent_slots(
                table,
                torch.tensor([position, position], dtype=torch.int64),
                rank,
                block_size=BLOCK,
                dcp_size=dcp,
                null_block=0,
            )
            want = [
                _oracle_slot(list(row), position, rank, block_size=BLOCK, dcp_size=dcp)
                for row in rows
            ]
            assert got.dtype == torch.int64
            assert got.tolist() == want, (dcp, rank, position)


def test_owned_latent_slots_take_the_rank_as_a_device_tensor() -> None:
    table = torch.tensor([[4], [6]], dtype=torch.int32)
    positions = torch.tensor([130], dtype=torch.int64)
    as_int = dcp_ownership.owned_latent_slots(
        table, positions, 1, block_size=BLOCK, dcp_size=2, null_block=0
    )
    as_tensor = dcp_ownership.owned_latent_slots(
        table, positions, torch.tensor([1]), block_size=BLOCK, dcp_size=2, null_block=0
    )
    assert as_int.tolist() == as_tensor.tolist() == [4 * BLOCK + 2]


def test_owned_latent_slots_at_dcp_1_is_the_page_index_rule() -> None:
    row = [3, 8, -1]
    table = torch.tensor([row], dtype=torch.int32).t().contiguous()
    for position in range(0, 4 * BLOCK):
        got = dcp_ownership.owned_latent_slots(
            table, torch.tensor([position]), 0, block_size=BLOCK, dcp_size=1, null_block=0
        )
        assert got.tolist() == [_oracle_slot(row, position, 0, block_size=BLOCK, dcp_size=1)]


@pytest.mark.parametrize("dcp", [2, 4])
def test_the_runners_dcp_slots_are_the_shared_rule(dcp: int) -> None:
    from vllm_neuron.vllm.worker.neuron_model_runner import _dcp_owned_latent_slots

    row = [7, 3, 11, 2]
    positions = torch.arange(0, len(row) * BLOCK * dcp, 5, dtype=torch.int64)
    for rank in range(dcp):
        got = _dcp_owned_latent_slots(
            row, positions, block_size=BLOCK, dcp_size=dcp, cp_rank=rank
        )
        want = [
            _oracle_slot(row, int(p), rank, block_size=BLOCK, dcp_size=dcp) for p in positions
        ]
        assert got == want, (dcp, rank)


def test_the_runner_still_refuses_a_position_past_its_row() -> None:
    from vllm_neuron.vllm.worker.neuron_model_runner import _dcp_owned_latent_slots

    with pytest.raises(ValueError, match="has no slot"):
        _dcp_owned_latent_slots(
            [7], torch.tensor([2 * BLOCK]), block_size=BLOCK, dcp_size=2, cp_rank=0
        )


# --------------------------------------------------------------------------- #
# 2. the draft head's slot rule.


def _two_request_table() -> tuple[tuple[list[int], ...], torch.Tensor]:
    """Two requests' local tables, ``[pages, 2]``: request 0 holds two blocks, its third page
    unallocated; request 1 holds one block, its other pages unallocated."""
    rows = ([7, 3, -1, -1], [5, -1, -1, -1])
    return rows, torch.tensor(rows, dtype=torch.int32).t().contiguous()


@pytest.mark.parametrize("dcp", [2, 4])
def test_latent_slots_at_follow_the_ownership_map_under_dcp(dcp: int) -> None:
    rows, table = _two_request_table()
    head = _head_class()
    # Request 0's positions span its second and third owned blocks on rank 0, and so blocks
    # held by every other rank between them; its third local page is not allocated.
    first = (2 * dcp - 1) * BLOCK - 3
    for rank in range(dcp):
        for position in range(first, first + 2 * dcp * BLOCK + 8, 7):
            positions = torch.tensor([position, position - BLOCK * dcp], dtype=torch.int64)
            got = head._latent_slots_at(
                table, positions, BLOCK, 2, dcp_size=dcp, dcp_rank=torch.tensor([rank])
            )
            want = [
                _oracle_slot(list(rows[r]), int(positions[r]), rank, block_size=BLOCK, dcp_size=dcp)
                for r in range(2)
            ]
            assert got.tolist() == want, (dcp, rank, position)


def test_latent_slots_at_name_both_owners_of_a_block_crossing() -> None:
    """A draft that crosses a block boundary at DCP 2: positions 254 .. 257 lie in block 1
    (rank 1's local block 0) and block 2 (rank 0's local block 1)."""
    table = torch.tensor([[7], [3]], dtype=torch.int32)
    head = _head_class()
    positions = [254, 255, 256, 257]
    on = {
        rank: [
            int(
                head._latent_slots_at(
                    table, torch.tensor([p]), BLOCK, 1, dcp_size=2, dcp_rank=torch.tensor([rank])
                )[0]
            )
            for p in positions
        ]
        for rank in (0, 1)
    }
    assert on[1] == [7 * BLOCK + 126, 7 * BLOCK + 127, NULL_SLOT, NULL_SLOT]
    assert on[0] == [NULL_SLOT, NULL_SLOT, 3 * BLOCK + 0, 3 * BLOCK + 1]


def test_latent_slots_at_dcp_1_is_todays_formula_bit_for_bit() -> None:
    head = _head_class()
    rows = ([2, 9, -1, 4], [6, -1, -1, -1])
    table = torch.tensor(rows, dtype=torch.int32).t().contiguous()
    page = 4
    for position in range(0, 4 * page + 6):
        for form in (torch.tensor([position, position + 1]), torch.tensor([position]), position):
            got = head._latent_slots_at(table, form, page, 2)
            want = _today_latent_slots_at(table, form, page, 2)
            assert got.dtype == want.dtype
            assert torch.equal(got, want), (position, form)


# --------------------------------------------------------------------------- #
# 3. the draft carrier on a head bound to DCP 2.


def _tiny_head(dcp: int, rank: int):
    """A head over the draft-test geometry with its layer bound to ``dcp`` ranks at ``rank``.

    Nothing is materialised: ``_iteration_carrier`` reads the carrier and the layer's DCP
    binding only."""
    from test.vllm_neuron.model.glm5_next.test_mtp_draft import _tiny_text_config

    head = _head_class()(
        _tiny_text_config(),
        embed_tokens=lambda: None,
        lm_head=lambda: None,
        world_size=1,
        tp_group=lambda: None,
    )
    attention = head.block.self_attn
    attention.dcp_size = dcp
    attention.indexer.dcp_size = dcp
    setattr(attention.indexer, attention.indexer.SHARD_RANK_ATTR, torch.tensor([rank]))
    return head


def _carrier(table: torch.Tensor, position: int, *, max_seq_len: int) -> dict:
    return {
        "position": torch.tensor([position], dtype=torch.int64),
        "seq_lens": torch.tensor([position + 1], dtype=torch.int32),
        "start_position": torch.tensor([position], dtype=torch.int64),
        "max_seq_len": int(max_seq_len),
        "page_size": BLOCK,
        "block_table_row": table,
        "latent_slots": torch.tensor([-7], dtype=torch.int64),
        "tail": "real ring",
    }


@pytest.mark.parametrize("rank", [0, 1])
def test_draft_iterations_past_the_first_take_ownership_slots(rank: int) -> None:
    # A local table of 8 entries. From 1022, iteration 1 lands at 1023 in block 7 (rank 1's
    # local block 3) and iterations 2 and 3 at 1024 and 1025 in block 8 (rank 0's local
    # block 4), so each rank owns one side of the crossing.
    table = torch.tensor([[10 + e] for e in range(8)], dtype=torch.int32)
    head = _tiny_head(2, rank)
    position = 1022
    carriers = [
        head._iteration_carrier(
            _carrier(table, position, max_seq_len=8 * BLOCK), i, 1, "scratch ring"
        )
        for i in range(4)
    ]
    assert carriers[0]["latent_slots"].tolist() == [-7], "iteration 0 keeps the runner's slot"
    row = [10 + e for e in range(8)]
    for i in (1, 2, 3):
        want = _oracle_slot(row, position + i, rank, block_size=BLOCK, dcp_size=2)
        assert carriers[i]["latent_slots"].tolist() == [want], (rank, i)
        assert carriers[i]["position"].tolist() == [position + i]
        assert carriers[i]["tail"] == "scratch ring"


def test_the_draft_clamp_is_the_groups_last_position_under_dcp() -> None:
    """At the served DCP 2 geometry the carrier's ``max_seq_len`` is the rank's share,
    65536, and a sequence reaches 131071. A draft at 70000 stays at 70000 + i."""
    entries = SERVED_SHARE // BLOCK
    table = torch.arange(1, entries + 1, dtype=torch.int32).reshape(entries, 1)
    head = _tiny_head(2, 0)
    position = 70_000
    carrier = head._iteration_carrier(
        _carrier(table, position, max_seq_len=SERVED_SHARE), 3, 1, "scratch ring"
    )
    assert carrier["position"].tolist() == [position + 3]
    assert carrier["start_position"].tolist() == [position + 3]
    assert carrier["seq_lens"].tolist() == [position + 4]
    want = _oracle_slot(
        list(range(1, entries + 1)), position + 3, 0, block_size=BLOCK, dcp_size=2
    )
    assert carrier["latent_slots"].tolist() == [want]
    # The group's last position is still the clamp.
    last = 2 * SERVED_SHARE - 1
    edge = head._iteration_carrier(
        _carrier(table, last - 1, max_seq_len=SERVED_SHARE), 3, 1, "scratch ring"
    )
    assert edge["position"].tolist() == [last]
    assert edge["seq_lens"].tolist() == [last + 1]


def test_the_draft_clamp_at_dcp_1_is_unchanged() -> None:
    head = _tiny_head(1, 0)
    table = torch.arange(4, dtype=torch.int32).reshape(4, 1)
    carrier = head._iteration_carrier(
        _carrier(table, 4 * BLOCK - 2, max_seq_len=4 * BLOCK), 3, 1, "scratch ring"
    )
    assert carrier["position"].tolist() == [4 * BLOCK - 1]
    assert carrier["latent_slots"].tolist() == _today_latent_slots_at(
        table, torch.tensor([4 * BLOCK - 1]), BLOCK, 1
    ).tolist()


# --------------------------------------------------------------------------- #
# 4. the DCP decode call site: T rows per request.


class _Group:
    """A two-rank DCP KV group stand-in: ``all_gather`` stacks this rank's tensor twice."""

    world_size = 2

    def all_gather(self, tensor, dim):
        return torch.cat((tensor, tensor), dim=dim)


class _Stop(Exception):
    pass


def _attention(dcp: int, rank: int):
    from test.vllm_neuron.model.glm5_next.test_mtp_draft import _tiny_text_config
    from vllm_neuron.model.glm5_next.model_fp8 import Glm5NextMLAAttention

    attention = Glm5NextMLAAttention(_tiny_text_config())
    attention.dcp_size = dcp
    attention.indexer.dcp_size = dcp
    setattr(attention.indexer, attention.indexer.SHARD_RANK_ATTR, torch.tensor([rank]))
    return attention


def _record_partial_entry(monkeypatch) -> list[dict]:
    from vllm_neuron.functional.attention import mla_decode
    from vllm_neuron.model.glm5_next import model_fp8

    seen: list[dict] = []

    def entry(q_lift, bank, block_table, position, written, owner, softmax_scale,
              page_size, topk_indices=None):
        seen.append({
            "q_lift": q_lift, "block_table": block_table, "position": position,
            "written": written, "owner": owner, "topk_indices": topk_indices,
        })
        raise _Stop

    monkeypatch.setattr(mla_decode, "mla_decode_partial_attention", entry)
    monkeypatch.setattr(model_fp8, "_resolve_dcp_kv_group", lambda: (_Group(), 0))
    return seen


def _decode_operands(batch: int, rows: int, starts: list[int], *, selected: bool):
    heads, latent = 1, 128
    q_lift = torch.zeros(batch * rows, heads, latent, dtype=torch.bfloat16)
    written = torch.zeros(batch * rows, latent, dtype=torch.bfloat16)
    c_kv = torch.zeros(8 * BLOCK, latent, dtype=torch.bfloat16)
    table = torch.arange(batch * 4, dtype=torch.int32).reshape(batch, 4)
    topk = (torch.zeros(batch * rows, 2176, dtype=torch.int32) if selected else None)
    return q_lift, c_kv, table, torch.tensor(starts, dtype=torch.int64), written, topk


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("selected", [False, True])
def test_dcp_decode_hands_per_row_operands_for_t_rows(monkeypatch, rank, selected) -> None:
    seen = _record_partial_entry(monkeypatch)
    attention = _attention(2, rank)
    starts = [254, 400]  # request 0 crosses a block boundary at 256; request 1 does not
    rows = 4
    q_lift, c_kv, table, start, written, topk = _decode_operands(2, rows, starts, selected=selected)
    with pytest.raises(_Stop):
        attention._dcp_decode(q_lift, c_kv, table, start, written, 0.125, BLOCK,
                              topk_indices=topk)
    (call,) = seen
    at = torch.tensor([s + t for s in starts for t in range(rows)], dtype=torch.int64)
    want_owner = dcp_ownership.token_owner(at, block_size=BLOCK, dcp_size=2) == rank
    want_position = dcp_ownership.owned_length(at, rank, block_size=BLOCK, dcp_size=2)
    assert call["owner"].tolist() == want_owner.tolist()
    assert call["position"].to(torch.int64).tolist() == want_position.tolist()
    assert tuple(call["q_lift"].shape) == (2 * rows, 2, 128), "every head of the group, every row"
    assert call["written"] is written
    if selected:
        assert tuple(call["topk_indices"].shape) == (2 * rows, 2304), "padded on every row"
        assert bool((call["topk_indices"][:, 2176:] == -1).all())
    else:
        assert call["topk_indices"] is None
        assert tuple(call["block_table"].shape) == (2, 4)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("selected", [False, True])
def test_dcp_decode_keeps_todays_operands_at_one_row(monkeypatch, rank, selected) -> None:
    seen = _record_partial_entry(monkeypatch)
    attention = _attention(2, rank)
    starts = [255, 256]
    q_lift, c_kv, table, start, written, topk = _decode_operands(2, 1, starts, selected=selected)
    with pytest.raises(_Stop):
        attention._dcp_decode(q_lift, c_kv, table, start, written, 0.125, BLOCK,
                              topk_indices=topk)
    (call,) = seen
    want_owner = dcp_ownership.token_owner(start, block_size=BLOCK, dcp_size=2) == rank
    want_position = dcp_ownership.owned_length(start, rank, block_size=BLOCK, dcp_size=2)
    assert call["owner"].tolist() == want_owner.tolist()
    assert call["position"].to(torch.int64).tolist() == want_position.tolist()
    if selected:
        assert tuple(call["topk_indices"].shape) == (2, 2304)


# --------------------------------------------------------------------------- #
# 5. the masked dispatch.


@pytest.mark.parametrize("dcp, want", [(1, True), (2, False)])
def test_the_masked_dispatch_is_not_taken_under_dcp(monkeypatch, dcp, want) -> None:
    """A selecting step of one request and four rows, with the masked kernel serving the
    geometry: at DCP 1 the indexer is asked for pool scores, under DCP for indices (the
    DCP attention takes indices; pool scores would reach it as a dense step)."""
    attention = _attention(dcp, 0)
    seen: dict = {}

    def forward_requests(*args, **kwargs):
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(attention, "_masked_attention_serves", lambda *a, **k: True)
    monkeypatch.setattr(attention, "project_query_latent", lambda hidden: hidden)
    monkeypatch.setattr(attention.indexer, "project_stage", lambda *a, **k: None)
    monkeypatch.setattr(attention.indexer, "forward_requests", forward_requests)
    rows, pages, page = 4, 16, 4
    hidden = torch.zeros(rows, int(attention.hidden_size), dtype=torch.bfloat16)
    with pytest.raises(_Stop):
        attention._forward_requests(
            hidden,
            latent_cache=torch.zeros(pages * page, 1, 128, dtype=torch.bfloat16),
            pool_cache=(torch.zeros(20, 128, dtype=torch.bfloat16),),
            seq_lens=torch.arange(41, 41 + rows, dtype=torch.int32),
            start_position=torch.tensor([40]),
            softmax_scale=0.125,
            max_seq_len=pages * page,
            page_size=page,
            block_table_row=torch.arange(pages, dtype=torch.int32).reshape(pages, 1),
            latent_slots=torch.arange(40, 40 + rows, dtype=torch.int64),
            tail=(torch.zeros(2, 4, 128, dtype=torch.bfloat16),),
            position=torch.tensor([40]),
            collector=None,
        )
    assert seen["pool_scores"] is want
