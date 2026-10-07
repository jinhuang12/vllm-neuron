# SPDX-License-Identifier: Apache-2.0
"""The bank form of the GLM-5.3-Flash decode carriers: whole state banks plus slot tensors.

At ``B > 1`` the runner hands every recurrent (KDA) layer its two whole banks and one
``[B]`` int64 slot tensor, and every sparse (DSA) layer its whole pooled store and ring
bank plus the same kind of slot tensor, instead of ``B`` per-request views of each. The
graph then has one input per bank rather than one per request per bank (5760 of the
served bs=64 graph's 7298 inputs were such views), and the write-back is an in-place op
on the bank placeholder itself, which the backend's aliasing pass maps to a whole-bank
aliased output; a write through a view it does not keep.

Read here, on a synthetic 45-layer stack (no model):

1. the form switch: one request, a prefill, the kill switch and a switched-off KDA fused
   decode all keep the per-request view form; a decode of several requests takes banks;
2. the bank carriers: the bank tensors themselves (identity), one int64 slot tensor per
   family shared by the layers of that family, padding rows on the sparse family naming
   the scratch slot (the last slot of the side caches) and on the recurrent family the
   idle slots the runner chose;
3. the page-independent and page-dependent sparse operands of one step are built once
   and shared by the layers of one KV-cache group (identity), never stacked on device;
4. the model-side helpers: gather/scatter by slot on a bank, and the shared-store rule
   that lets several padding rows name the scratch slot while real rows stay distinct.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/worker/test_glm5next_state_banks.py
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from vllm_neuron.functional import state_banks as model_side
from vllm_neuron.functional.kda.fused_decode import FUSED_DECODE_ENV
from vllm_neuron.vllm.worker import glm5next_state_banks as runner_side
from vllm_neuron.vllm.worker.neuron_model_runner import NULL_BLOCK_ID, NeuronModelRunner

pytestmark = [pytest.mark.fast, pytest.mark.forked]

LINEAR, SPARSE = 34, 11
SLOTS = 8
PAGE = 4
WINDOW = 4
POOL = 4
DIM = 2


def _stack(linear=LINEAR, sparse=SPARSE, *, slots=SLOTS, scratch=True):
    """Banks, side caches and one decode geometry per bank, the runner's own shapes."""
    banks, side, geometries = [], [], []
    for index in range(linear):
        banks.append({
            "name": f"linear.{index}",
            "family": "linear_attn",
            "state_slots": slots,
            "conv_state": torch.full((slots, 2, 3), float(index)),
            "recurrent_state": torch.full((slots, 1, 2, 2), float(index)),
        })
        side.append({})
        geometries.append({"state_slot": 0, "state_slots": [0]})
    stores = slots + (1 if scratch else 0)
    for index in range(sparse):
        banks.append({
            "name": f"sparse.{index}",
            "family": "self_attn",
            "block_size": PAGE,
            "latent_cache": torch.full((16, 1, PAGE, DIM), float(index)),
        })
        side.append({
            "pool_cache": torch.full((stores, 8, DIM), float(index)),
            "tail": torch.full((stores, 2, POOL, DIM), float(index)),
            "pad_tail": torch.zeros((slots, 2, POOL, DIM)),
        })
        geometries.append({"state_slot": 0, "state_slots": [0], "page_size": PAGE,
                           "window_blocks": WINDOW})
    return banks, side, geometries


def _decode(banks, side, geometries, *, slots, starts, padding=0, request_block_ids=None):
    """One decode step of ``len(slots)`` rows through the carrier builder."""
    requests = len(slots)
    real = requests - padding
    # Three pages per request: every position used here is below 12.
    rows = request_block_ids or [[1 + 3 * r, 2 + 3 * r, 3 + 3 * r] for r in range(real)]
    for geometry in geometries:
        geometry["state_slot"] = slots[0]
        geometry["state_slots"] = list(slots)
        if "page_size" in geometry:
            geometry["request_block_ids"] = rows
            geometry["block_ids"] = rows[0]
    return NeuronModelRunner._glm5next_layer_carriers(
        banks, side,
        geometries=geometries,
        is_prefill=False,
        tokens=requests,
        start_position=starts[0],
        softmax_scale=0.25,
        max_seq_len=16,
        index_kpool=POOL,
        requests=requests,
        request_starts=list(starts) + [0] * padding,
        real_tokens=1,
        request_real_tokens=[1] * real + [0] * padding,
        padded_requests=padding,
    )


# ── 1. the form switch ───────────────────────────────────────────────────────


def test_the_bank_form_is_on_by_default_and_the_kill_switch_turns_it_off(monkeypatch):
    monkeypatch.delenv(runner_side.STATE_BANKS_ENV, raising=False)
    monkeypatch.delenv(FUSED_DECODE_ENV, raising=False)
    assert runner_side.state_banks_enabled()
    assert runner_side.bank_form(2, is_prefill=False)
    monkeypatch.setenv(runner_side.STATE_BANKS_ENV, "0")
    assert not runner_side.state_banks_enabled()
    assert not runner_side.bank_form(64, is_prefill=False)


def test_one_request_a_prefill_and_a_switched_off_fused_decode_keep_the_view_form(monkeypatch):
    monkeypatch.delenv(runner_side.STATE_BANKS_ENV, raising=False)
    monkeypatch.delenv(FUSED_DECODE_ENV, raising=False)
    # One request: the bs=1 graph is the measured production line and stays as it is.
    assert not runner_side.bank_form(1, is_prefill=False)
    # A prefill serves one sequence per forward and its state writes are the layer's own.
    assert not runner_side.bank_form(4, is_prefill=True)
    # The recurrent layers' bank form rides on the fused decode launch; without it the
    # per-request loop writes through views, which the bank form cannot express.
    monkeypatch.setenv(FUSED_DECODE_ENV, "0")
    assert not runner_side.bank_form(4, is_prefill=False)


def test_a_one_request_decode_builds_the_view_carriers_unchanged(monkeypatch):
    monkeypatch.delenv(runner_side.STATE_BANKS_ENV, raising=False)
    banks, side, geometries = _stack(linear=2, sparse=2)
    carriers = _decode(banks, side, geometries, slots=[3], starts=[5])
    assert isinstance(carriers[0]["conv_state"], tuple) and len(carriers[0]["conv_state"]) == 1
    assert "state_slots" not in carriers[0]
    assert torch.is_tensor(carriers[2]["tail"]) and carriers[2]["tail"].dim() == 3
    assert "state_slots" not in carriers[2]


def test_the_kill_switch_builds_the_view_carriers_at_b_four(monkeypatch):
    monkeypatch.setenv(runner_side.STATE_BANKS_ENV, "0")
    banks, side, geometries = _stack(linear=2, sparse=2)
    carriers = _decode(banks, side, geometries, slots=[3, 0, 5, 1], starts=[5, 6, 7, 8])
    assert isinstance(carriers[0]["conv_state"], tuple) and len(carriers[0]["conv_state"]) == 4
    assert isinstance(carriers[2]["tail"], tuple) and len(carriers[2]["tail"]) == 4
    assert "state_slots" not in carriers[0] and "state_slots" not in carriers[2]


# ── 2. the bank carriers ─────────────────────────────────────────────────────


def test_a_decode_of_several_requests_hands_every_layer_its_whole_banks(monkeypatch):
    monkeypatch.delenv(runner_side.STATE_BANKS_ENV, raising=False)
    monkeypatch.delenv(FUSED_DECODE_ENV, raising=False)
    banks, side, geometries = _stack()
    slots = [3, 0, 5, 1]
    carriers = _decode(banks, side, geometries, slots=slots, starts=[5, 6, 7, 8])
    for index in range(LINEAR):
        carrier = carriers[index]
        assert carrier["conv_state"] is banks[index]["conv_state"]
        assert carrier["recurrent_state"] is banks[index]["recurrent_state"]
        assert carrier["state_slots"].dtype == torch.int64
        assert carrier["state_slots"].tolist() == slots
        # One upload for the whole family.
        assert carrier["state_slots"] is carriers[0]["state_slots"]
        assert carrier["is_prefill"] is False
        for key in ("start_position", "real_tokens", "row_mask"):
            assert carrier[key] is carriers[0][key]
    for index in range(LINEAR, LINEAR + SPARSE):
        carrier = carriers[index]
        assert carrier["pool_cache"] is side[index]["pool_cache"]
        assert carrier["tail"] is side[index]["tail"]
        assert carrier["state_slots"].dtype == torch.int64
        assert carrier["state_slots"].tolist() == slots
        assert carrier["state_slots"] is carriers[LINEAR]["state_slots"]
        assert carrier["latent_cache"] is banks[index]["latent_cache"]
        assert tuple(carrier["block_table_row"].shape) == (WINDOW, 4)
        assert tuple(carrier["latent_slots"].shape) == (4,)
        for key in ("seq_lens", "start_position", "position"):
            assert carrier[key] is carriers[LINEAR][key]
    # The two families' slot tensors are distinct uploads even when equal, so the
    # graph signature does not depend on whether a step is padded.
    assert carriers[0]["state_slots"] is not carriers[LINEAR]["state_slots"]
    # The carrier contracts: the layers take these as keywords, so an extra key raises
    # and a missing one is served as a default.
    assert set(carriers[0]) == {
        "conv_state", "recurrent_state", "state_slots", "is_prefill",
        "start_position", "real_tokens", "row_mask",
    }
    assert set(carriers[LINEAR]) == {
        "latent_cache", "block_table_row", "latent_slots", "seq_lens", "start_position",
        "position", "softmax_scale", "max_seq_len", "page_size",
        "pool_cache", "tail", "state_slots",
    }


def test_padding_rows_name_the_scratch_slot_on_the_sparse_family_and_idle_slots_on_the_recurrent(monkeypatch):
    monkeypatch.delenv(runner_side.STATE_BANKS_ENV, raising=False)
    monkeypatch.delenv(FUSED_DECODE_ENV, raising=False)
    banks, side, geometries = _stack(linear=2, sparse=2)
    # Three real requests and one padding row; the runner chose idle slot 7 for it.
    slots = [3, 0, 5, 7]
    carriers = _decode(banks, side, geometries, slots=slots, starts=[5, 6, 7], padding=1)
    assert carriers[0]["state_slots"].tolist() == [3, 0, 5, 7]
    assert carriers[0]["real_tokens"].reshape(-1).tolist() == [1, 1, 1, 0]
    assert carriers[0]["start_position"].tolist()[-1] != 0
    scratch = SLOTS  # the slot past the engine's concurrency bound
    assert int(side[2]["pool_cache"].shape[0]) == scratch + 1
    assert carriers[2]["state_slots"].tolist() == [3, 0, 5, scratch]
    assert carriers[2]["position"].tolist()[-1] == 0 and carriers[2]["seq_lens"].tolist()[-1] == 1
    # The padding row is a one-token sequence in the null block: its table column names
    # that block and nothing else, and its latent write lands in it.
    assert carriers[2]["block_table_row"][:, -1].tolist() == [NULL_BLOCK_ID] + [-1] * (WINDOW - 1)
    assert int(carriers[2]["latent_slots"][-1]) == NULL_BLOCK_ID * PAGE
    assert carriers[0]["row_mask"].reshape(-1).tolist() == [1.0, 1.0, 1.0, 0.0]


def test_a_sparse_bank_without_a_scratch_slot_is_refused_by_name(monkeypatch):
    monkeypatch.delenv(runner_side.STATE_BANKS_ENV, raising=False)
    monkeypatch.delenv(FUSED_DECODE_ENV, raising=False)
    banks, side, geometries = _stack(linear=0, sparse=1, scratch=False)
    # Every slot of the store is a request's; the last one is owned here, so a
    # padding row that named it would write a live ring.
    with pytest.raises(ValueError, match="scratch"):
        _decode(banks, side, geometries, slots=[SLOTS - 1, 0, 1, 2], starts=[5, 6, 7],
                padding=1)


# ── 3. the step operands are built once per group ────────────────────────────


def test_sparse_step_operands_are_built_once_per_group_and_never_stacked(monkeypatch):
    monkeypatch.delenv(runner_side.STATE_BANKS_ENV, raising=False)
    monkeypatch.delenv(FUSED_DECODE_ENV, raising=False)
    banks, side, geometries = _stack()
    moves = []
    original_to = torch.Tensor.to

    def record_move(tensor, *args, **kwargs):
        moves.append((tuple(tensor.shape), tensor.dtype))
        return original_to(tensor, *args, **kwargs)

    rows = [[1 + 3 * r, 2 + 3 * r, 3 + 3 * r] for r in range(4)]
    with patch.object(torch.Tensor, "to", record_move), patch.object(
        torch, "stack", side_effect=AssertionError("request axis must be built on host")
    ):
        carriers = _decode(banks, side, geometries, slots=[3, 0, 5, 1], starts=[5, 6, 7, 8],
                           request_block_ids=rows)
    # 3 KDA step operands + 1 KDA slot tensor; 2 DSA step operands (seq lens, and one
    # positions tensor read as both start_position and position) + 1 DSA slot tensor +
    # the 2 page-dependent operands of the one group. Nothing per layer.
    assert len(moves) == 4 + 2 + 1 + 2, moves
    for carrier in carriers[LINEAR + 1:]:
        assert carrier["block_table_row"] is carriers[LINEAR]["block_table_row"]
        assert carrier["latent_slots"] is carriers[LINEAR]["latent_slots"]


def test_layers_over_different_pages_keep_their_own_page_operands(monkeypatch):
    monkeypatch.delenv(runner_side.STATE_BANKS_ENV, raising=False)
    monkeypatch.delenv(FUSED_DECODE_ENV, raising=False)
    banks, side, geometries = _stack(linear=0, sparse=2)
    slots = [3, 0]
    for geometry in geometries:
        geometry["state_slot"] = slots[0]
        geometry["state_slots"] = list(slots)
    geometries[0]["request_block_ids"] = [[1], [2]]
    geometries[1]["request_block_ids"] = [[3], [4]]
    for geometry in geometries:
        geometry["block_ids"] = geometry["request_block_ids"][0]
    carriers = NeuronModelRunner._glm5next_layer_carriers(
        banks, side, geometries=geometries, is_prefill=False, tokens=2, start_position=1,
        softmax_scale=0.25, max_seq_len=16, index_kpool=POOL, requests=2,
        request_starts=[1, 2], real_tokens=1, request_real_tokens=[1, 1],
    )
    assert carriers[0]["block_table_row"][0].tolist() == [1, 2]
    assert carriers[1]["block_table_row"][0].tolist() == [3, 4]
    assert carriers[0]["latent_slots"].tolist() == [1 * PAGE + 1, 2 * PAGE + 2]
    assert carriers[1]["latent_slots"].tolist() == [3 * PAGE + 1, 4 * PAGE + 2]


# ── 4. the model-side helpers ────────────────────────────────────────────────


def test_gather_and_scatter_move_exactly_the_named_rows():
    bank = torch.arange(8 * 6, dtype=torch.float32).reshape(8, 2, 3)
    before = bank.clone()
    slots = torch.tensor([5, 0, 7], dtype=torch.int64)
    rows = model_side.gather_bank_rows(bank, slots)
    assert torch.equal(rows, torch.stack([bank[5], bank[0], bank[7]]))
    model_side.scatter_bank_rows(bank, slots, -rows.to(torch.float64))
    for slot in range(8):
        if slot in (5, 0, 7):
            assert torch.equal(bank[slot], -before[slot])
        else:
            assert torch.equal(bank[slot], before[slot])
    assert bank.dtype == torch.float32


def test_bank_rows_problem_names_a_bank_of_the_wrong_row_shape():
    bank = torch.zeros(8, 2, 3)
    slots = torch.tensor([1, 2], dtype=torch.int64)
    assert model_side.bank_rows_problem(bank, slots, (2, 3), name="conv_state") is None
    problem = model_side.bank_rows_problem(bank, slots, (3, 3), name="conv_state")
    assert problem and "conv_state" in problem and "(3, 3)" in problem
    assert model_side.bank_rows_problem(bank, torch.zeros(2, 2), (2, 3), name="x")
    assert model_side.bank_rows_problem(bank, torch.zeros(0, dtype=torch.int64), (2, 3), name="x")


def test_shared_store_rule_allows_the_scratch_slot_only():
    scratch = 8
    ok = torch.tensor([3, 0, 5, scratch, scratch], dtype=torch.int64)
    assert model_side.shared_store_problem(ok, 5, scratch=scratch) is None
    twice = torch.tensor([3, 0, 3, scratch], dtype=torch.int64)
    problem = model_side.shared_store_problem(twice, 4, scratch=scratch)
    assert problem and "distinct" in problem
    assert model_side.shared_store_problem(ok, 4, scratch=scratch)
