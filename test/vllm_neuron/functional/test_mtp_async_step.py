# SPDX-License-Identifier: Apache-2.0
"""``functional/mtp/async_step.py``: the async drafter's two per-step device kernels.

Under ``--async-scheduling`` the GLM-5.3-Flash verify step's accepted rows are a device
future when the host builds the next step, so the three host corrections of the sync
state hook (ring cursor, KDA checkpoint commit, resume row) and the position-derived
operands the translator builds from host ints have to be computed on device from the
previous step's output. Two launches, one at each end of a step:

* ``mtp_async_take(accepted [B, W], drafts [B, k])`` right after the step: the kept
  count per request (the non-placeholder prefix of the rejection sampler's row), the
  resume row ``kept - 1`` (the KDA commit's carry), the last kept id and the next
  step's ``[B, 1 + k]`` input ids (last kept id, then the drafts);
* ``mtp_async_correct(prev_checkpoint_rows [B], optimistic_starts [B], block_table_row
  [pages, R])`` when the next step is built: each request's position pulled back by the
  rows the previous step rejected (``prev_width - 1 - prev_checkpoint_rows``), and from
  it the per-row causal lengths and physical latent slots the sparse carrier hands the
  layers, the recurrent carrier's start positions and the checkpoint rows, padding rows
  (the bucket's, past the ``B`` real requests) laid out as the host builder lays them.

The torch route is the contract; every expectation below is the HOST formula the runner
builds the same operand from today (``_glm5next_latent_slot_mapping``,
``_glm5next_batch_row_seq_lens``, ``_glm5next_start_positions``) evaluated at the
corrected positions, so the kernel is held to the sync translator, not to itself. The
kernel is checked bit for bit against the torch route on the simulator.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/functional/test_mtp_async_step.py
"""

from __future__ import annotations

import pytest
import torch

from vllm_neuron.functional.mtp import async_step
from vllm_neuron.nn.rejection_sampler import PLACEHOLDER_TOKEN_ID
from vllm_neuron.vllm.worker.neuron_model_runner import NULL_BLOCK_ID, NeuronModelRunner

K = 3
T = 1 + K
PAGE = 128
VOCAB = 154_880


def _accepted(seed: int, batch: int, width: int) -> torch.Tensor:
    """Rejection-sampler rows: a kept prefix of ``1 .. width`` ids, placeholders after."""
    gen = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, VOCAB, (batch, width), generator=gen, dtype=torch.int32)
    kept = torch.randint(1, width + 1, (batch,), generator=gen)
    keep = torch.arange(width).unsqueeze(0) < kept.unsqueeze(1)
    return torch.where(keep, ids, torch.full_like(ids, PLACEHOLDER_TOKEN_ID)), kept


def _drafts(seed: int, batch: int) -> torch.Tensor:
    return torch.randint(0, VOCAB, (batch, K), generator=torch.Generator().manual_seed(seed),
                         dtype=torch.int32)


# ── take: the step's output, read on device ───────────────────────────────────────────


@pytest.mark.parametrize("batch, width", [(1, T), (3, T), (4, 1), (1, 1), (64, T)])
def test_take_reads_the_kept_prefix_the_last_kept_id_and_assembles_the_next_input_ids(batch, width):
    accepted, kept = _accepted(11 + batch + width, batch, width)
    drafts = _drafts(23 + batch, batch)
    for route in ("torch", "kernel"):
        async_step.reset_dispatch_counters()
        if route == "torch":
            take = async_step.mtp_async_take_torch(accepted, drafts)
        else:
            take = async_step.mtp_async_take(accepted, drafts)
            assert async_step.dispatch_counters() == (1, 0), async_step.dispatch_counters()
        assert take.valid_count.dtype == torch.int32 and take.valid_count.tolist() == kept.tolist()
        assert take.checkpoint_rows.tolist() == (kept - 1).tolist()
        last = accepted[torch.arange(batch), kept - 1]
        assert take.last_accepted.dtype == torch.int32 and take.last_accepted.tolist() == last.tolist()
        assert take.next_input_ids.dtype == torch.int32 and tuple(take.next_input_ids.shape) == (batch, T)
        assert torch.equal(take.next_input_ids, torch.cat([last.reshape(-1, 1), drafts], dim=1))


def test_take_accepts_a_one_row_steps_sampled_ids_as_a_column_or_a_vector():
    """A prefill or a one-row decode samples ``[B]``; the next step's row is that id and the drafts."""
    sampled = torch.tensor([5, 9], dtype=torch.int32)
    drafts = _drafts(31, 2)
    for form in (sampled, sampled.reshape(2, 1)):
        take = async_step.mtp_async_take(form, drafts)
        assert take.valid_count.tolist() == [1, 1] and take.checkpoint_rows.tolist() == [0, 0]
        assert take.last_accepted.tolist() == [5, 9]
        assert take.next_input_ids.tolist() == [[5] + drafts[0].tolist(), [9] + drafts[1].tolist()]


def test_the_kernel_take_is_the_torch_take_bit_for_bit():
    accepted, _ = _accepted(41, 5, T)
    drafts = _drafts(43, 5)
    want = async_step.mtp_async_take_torch(accepted, drafts)
    got = async_step.mtp_async_take(accepted, drafts)
    for name in want._fields:
        assert torch.equal(getattr(got, name), getattr(want, name)), name


@pytest.mark.parametrize(
    "accepted, drafts, match",
    [
        (torch.zeros((2, T), dtype=torch.int64), _drafts(1, 2), "accepted must be"),
        (torch.zeros((2, T), dtype=torch.int32), torch.zeros((3, K), dtype=torch.int32), "one row per request"),
        (torch.zeros((2, T), dtype=torch.int32), torch.zeros((2, K), dtype=torch.int64), "drafts must be"),
        (torch.full((2, T), PLACEHOLDER_TOKEN_ID, dtype=torch.int32), _drafts(1, 2), "keeps at least"),
    ],
)
def test_take_refuses_a_geometry_or_a_row_that_is_not_a_verify_steps_by_name(accepted, drafts, match):
    with pytest.raises(async_step.MtpAsyncStepError, match=match):
        async_step.mtp_async_take(accepted, drafts)


# ── correct: the next step's operands from the previous step's output ─────────────────


def _tables(seed: int, batch: int, pages: int) -> list[list[int]]:
    """One block-table row per request: distinct pages anywhere in the bank, ``pages`` wide."""
    gen = torch.Generator().manual_seed(seed)
    blocks = torch.randperm(4096, generator=gen)[1:1 + batch * pages].reshape(batch, pages)
    return [[int(value) for value in row] for row in blocks]


def _column(tables: list[list[int]], window: int, padding: int) -> torch.Tensor:
    """The sparse carrier's ``[pages, R]`` int32 column form, ``-1`` past a row's pages,
    a padding request's column naming the null block only."""
    rows = [list(row) for row in tables] + [[NULL_BLOCK_ID]] * padding
    return torch.tensor(
        [[row[page] if page < len(row) else -1 for row in rows] for page in range(window)],
        dtype=torch.int32,
    )


def _host_operands(tables, starts, *, width: int, padding: int, prev_width: int, prev_rows):
    """What the sync translator builds from host ints at the TRUE positions."""
    rejected = [prev_width - 1 - int(row) for row in prev_rows]
    true_starts = [int(start) - rej for start, rej in zip(starts, rejected)]
    sparse_starts = true_starts + [0] * padding
    linear_starts = true_starts + [1] * padding
    rows = [list(row) for row in tables] + [[NULL_BLOCK_ID]] * padding
    reals = [width] * len(tables) + [1] * padding
    return {
        "start_position": torch.tensor(sparse_starts, dtype=torch.int32),
        "linear_start": torch.tensor(linear_starts, dtype=torch.int32),
        "seq_lens": NeuronModelRunner._glm5next_batch_row_seq_lens(
            [(width, start) for start in sparse_starts], device=torch.device("cpu")),
        "latent_slots": NeuronModelRunner._glm5next_latent_slot_mapping(
            rows=rows, starts=sparse_starts, tokens=width, block_size=PAGE, reals=reals,
            device=torch.device("cpu")),
        "checkpoint_rows": torch.tensor([int(row) for row in prev_rows] + [0] * padding, dtype=torch.int32),
    }


@pytest.mark.parametrize(
    "batch, padding, prev_width, width, starts, prev_rows",
    [
        # bs = 1: a verify step after a verify step that kept 1, 2, 3 and 4 rows; the kept
        # counts place the true position on both sides of a page boundary.
        (1, 0, T, T, [PAGE + 3], [0]),
        (1, 0, T, T, [PAGE + 3], [1]),
        (1, 0, T, T, [PAGE + 3], [2]),
        (1, 0, T, T, [PAGE + 3], [3]),
        (1, 0, T, T, [PAGE], [0]),
        # the first verify step after a prefill (previous width 1: nothing to pull back)
        (1, 0, 1, T, [37], [0]),
        # a one-row decode after a verify step (the spec-to-non-spec transition)
        (1, 0, T, 1, [2 * PAGE + 1], [1]),
        # several requests, each with its own kept count, two bucket padding rows
        (2, 2, T, T, [PAGE + 3, 2 * PAGE + 2], [3, 0]),
        (4, 0, T, T, [5, PAGE - 1, PAGE, 3 * PAGE - 2], [0, 1, 2, 3]),
        (64, 0, T, T, [PAGE + 4] * 64, [0, 1, 2, 3] * 16),
    ],
)
def test_correct_builds_what_the_sync_translator_builds_at_the_true_positions(
    batch, padding, prev_width, width, starts, prev_rows,
):
    pages = 4
    tables = _tables(7 + batch, batch, pages)
    column = _column(tables, pages, padding)
    want = _host_operands(tables, starts, width=width, padding=padding, prev_width=prev_width,
                          prev_rows=prev_rows)
    prev = torch.tensor(prev_rows, dtype=torch.int32)
    optimistic = torch.tensor(starts, dtype=torch.int32)
    for route in ("torch", "kernel"):
        async_step.reset_dispatch_counters()
        if route == "torch":
            got = async_step.mtp_async_correct_torch(
                prev, optimistic, column, prev_width=prev_width, width=width, page_size=PAGE,
                padding=padding)
        else:
            got = async_step.mtp_async_correct(
                prev, optimistic, column, prev_width=prev_width, width=width, page_size=PAGE,
                padding=padding)
            assert async_step.dispatch_counters() == (1, 0), async_step.dispatch_counters()
        for name, expected in want.items():
            value = getattr(got, name)
            assert value.dtype == expected.dtype, (route, name, value.dtype)
            assert value.tolist() == expected.tolist(), (route, name)
        assert got.latent_slots.dtype == torch.int64


def test_the_kernel_correction_is_the_torch_correction_bit_for_bit():
    tables = _tables(51, 3, 4)
    column = _column(tables, 4, 1)
    prev = torch.tensor([0, 3, 1], dtype=torch.int32)
    optimistic = torch.tensor([PAGE + 3, 2 * PAGE, 9], dtype=torch.int32)
    want = async_step.mtp_async_correct_torch(prev, optimistic, column, prev_width=T, width=T,
                                              page_size=PAGE, padding=1)
    got = async_step.mtp_async_correct(prev, optimistic, column, prev_width=T, width=T,
                                       page_size=PAGE, padding=1)
    for name in want._fields:
        assert torch.equal(getattr(got, name), getattr(want, name)), name


def test_correct_works_on_the_torch_route_when_kernels_cannot_run(monkeypatch):
    monkeypatch.setattr(async_step, "can_run_kernel", lambda *_: False)
    async_step.reset_dispatch_counters()
    column = _column(_tables(61, 1, 2), 2, 0)
    got = async_step.mtp_async_correct(torch.tensor([2], dtype=torch.int32),
                                       torch.tensor([PAGE + 1], dtype=torch.int32), column,
                                       prev_width=T, width=T, page_size=PAGE, padding=0)
    assert async_step.dispatch_counters() == (0, 1)
    assert got.start_position.tolist() == [PAGE + 1 - 1]


@pytest.mark.parametrize(
    "mutation, match",
    [
        (dict(page_size=96), "power of two"),
        (dict(prev=torch.tensor([T], dtype=torch.int32)), "resume row"),
        (dict(prev=torch.tensor([-1], dtype=torch.int32)), "resume row"),
        # prev 1 of a step T wide rejected T - 2 = 2 rows; an optimistic 1 pulls back to -1
        (dict(optimistic=torch.tensor([T - 3], dtype=torch.int32)), "below zero"),
        (dict(column=torch.zeros((2, 3), dtype=torch.int32)), "one column per request"),
        (dict(optimistic=torch.tensor([PAGE + 3], dtype=torch.int64)), "optimistic_starts must be"),
        (dict(prev_width=0), "prev_width"),
        (dict(width=0), "width"),
    ],
)
def test_correct_refuses_an_operand_that_is_not_the_steps_by_name(mutation, match):
    column = _column(_tables(71, 1, 2), 2, 0)
    args = dict(prev=torch.tensor([1], dtype=torch.int32),
                optimistic=torch.tensor([PAGE + 3], dtype=torch.int32), column=column,
                prev_width=T, width=T, page_size=PAGE, padding=0)
    args.update(mutation)
    with pytest.raises(async_step.MtpAsyncStepError, match=match):
        async_step.mtp_async_correct(args["prev"], args["optimistic"], args["column"],
                                     prev_width=args["prev_width"], width=args["width"],
                                     page_size=args["page_size"], padding=args["padding"])


def test_without_a_page_table_the_correction_builds_no_slots_on_either_route():
    """A recurrent-only stack has no latent bank: the correction takes no table and hands
    back ``latent_slots`` None, the position operands as before, on both routes."""
    prev = torch.tensor([1, 3], dtype=torch.int32)
    optimistic = torch.tensor([PAGE + 3, 2 * PAGE], dtype=torch.int32)
    want = async_step.mtp_async_correct_torch(prev, optimistic, _column(_tables(5, 2, 3), 3, 1),
                                              prev_width=T, width=T, page_size=PAGE, padding=1)
    for route in (async_step.mtp_async_correct_torch, async_step.mtp_async_correct):
        async_step.reset_dispatch_counters()
        got = route(prev, optimistic, None, prev_width=T, width=T, page_size=PAGE, padding=1)
        assert got.latent_slots is None
        for name in ("start_position", "linear_start", "seq_lens", "checkpoint_rows"):
            assert getattr(got, name).tolist() == getattr(want, name).tolist(), (route, name)
            assert getattr(got, name).dtype == torch.int32
    assert async_step.dispatch_counters() == (1, 0)


def test_a_position_whose_page_the_table_does_not_name_is_refused_by_name():
    """The host builder refuses it ("a position outside the row has no slot"); so does this."""
    column = _column(_tables(81, 1, 1), 1, 0)
    with pytest.raises(async_step.MtpAsyncStepError, match="names 1 page"):
        async_step.mtp_async_correct_torch(torch.tensor([3], dtype=torch.int32),
                                           torch.tensor([PAGE - 1], dtype=torch.int32), column,
                                           prev_width=T, width=T, page_size=PAGE, padding=0)
