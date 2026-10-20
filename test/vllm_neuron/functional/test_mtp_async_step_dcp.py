# SPDX-License-Identifier: Apache-2.0
"""``mtp_async_correct`` under decode context parallelism: the trunk's latent slots.

Under DCP (``dcp_size > 1``) a request's block-table entries are this rank's blocks, each
holding the rank's share of ``block_size * dcp_size`` positions
(:mod:`vllm_neuron.utils.dcp_ownership`). The synchronous translator writes a position this
rank owns to its local row and a position another rank owns to the null slot
(``neuron_model_runner._dcp_owned_latent_slots``). The async drafter builds the same slots on
device from the corrected positions, so both routes of the correction -- the torch route and
the NKI kernel on the simulator -- are held to that function here, at every rank of DCP 1, 2
and 4, on verify steps that cross a 128-token block (and an ownership stride) with the page
after the step's last one unallocated. The other fields do not depend on DCP.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/functional/test_mtp_async_step_dcp.py
"""

from __future__ import annotations

import pytest
import torch

from vllm_neuron.functional.mtp import async_step
from vllm_neuron.vllm.worker.neuron_model_runner import NULL_BLOCK_ID, _dcp_owned_latent_slots

K = 3
T = 1 + K
PAGE = 128
#: (dcp, rank) for every rank of DCP 1, 2 and 4.
RANKS = [(dcp, rank) for dcp in (1, 2, 4) for rank in range(dcp)]
#: (prev_width, width): verify after verify, the first verify after a prefill, the one-row
#: step after a verify step.
WIDTHS = [(T, T), (1, T), (T, 1)]


def _true_starts(dcp: int) -> list[int]:
    """Verify steps whose rows cross a block boundary: inside one table entry (block 0 -> 1),
    across an entry (the last block of entry 0 -> the first of entry 1), and further on."""
    return [PAGE - 2, dcp * PAGE - 1, (dcp + 1) * PAGE - 3, 3 * dcp * PAGE - 2]


def _table(request: int, entries: int) -> list[int]:
    """Distinct non-null block ids, ``entries`` of them, apart from every other request's."""
    return [1 + 97 * request + 5 * entry for entry in range(entries)]


def _column(tables: list[list[int]], window: int, padding: int) -> torch.Tensor:
    """The ``[window, R]`` int32 column form, ``-1`` past a row's entries, a padding column
    naming the null block only (``NeuronModelRunner._glm5next_block_table_column``)."""
    rows = [list(row) for row in tables] + [[NULL_BLOCK_ID]] * padding
    return torch.tensor(
        [[row[page] if page < len(row) else -1 for row in rows] for page in range(window)],
        dtype=torch.int32,
    )


def _step(true_starts: list[int], *, dcp: int, prev_width: int, width: int, padding: int):
    """One step's operands and its tables: each request holds just the entries its rows reach
    (vLLM's per-rank rule, ``ceil(end / (block_size * dcp))``), and the window has two more,
    unallocated (-1)."""
    stride = PAGE * dcp
    tables = [_table(r, -(-(start + width) // stride)) for r, start in enumerate(true_starts)]
    window = max(len(row) for row in tables) + 2
    prev_rows = [(r + 1) % prev_width for r in range(len(true_starts))]
    optimistic = [start + (prev_width - 1 - row) for start, row in zip(true_starts, prev_rows)]
    return (
        tables,
        torch.tensor(prev_rows, dtype=torch.int32),
        torch.tensor(optimistic, dtype=torch.int32),
        _column(tables, window, padding),
    )


def _runner_slots(tables, true_starts, *, width: int, padding: int, dcp: int, rank: int) -> list[int]:
    """The runner's own DCP slot function per request, padding rows as the synchronous batch
    carrier lays them (``rows=[NULL_BLOCK_ID]``, every row at position 0)."""
    rows = [list(row) for row in tables] + [[NULL_BLOCK_ID]] * padding
    starts = list(true_starts) + [0] * padding
    steps = [width] * len(tables) + [1] * padding
    slots: list[int] = []
    for row, start, real in zip(rows, starts, steps):
        positions = torch.arange(width, dtype=torch.int64).clamp(max=real - 1) + start
        slots += _dcp_owned_latent_slots(row, positions, block_size=PAGE, dcp_size=dcp, cp_rank=rank)
    return slots


def _correct(route: str, prev, optimistic, column, **kwargs):
    if route == "torch":
        return async_step.mtp_async_correct_torch(prev, optimistic, column, page_size=PAGE, **kwargs)
    async_step.reset_dispatch_counters()
    got = async_step.mtp_async_correct(prev, optimistic, column, page_size=PAGE, **kwargs)
    assert async_step.dispatch_counters() == (1, 0), async_step.dispatch_counters()
    return got


@pytest.mark.parametrize("prev_width, width", WIDTHS)
@pytest.mark.parametrize("dcp, rank", RANKS)
def test_both_routes_write_the_runners_dcp_slots(dcp, rank, prev_width, width):
    """One request per step, every start of :func:`_true_starts`, both routes."""
    for start in _true_starts(dcp):
        tables, prev, optimistic, column = _step([start], dcp=dcp, prev_width=prev_width,
                                                 width=width, padding=0)
        want = _runner_slots(tables, [start], width=width, padding=0, dcp=dcp, rank=rank)
        for route in ("torch", "kernel"):
            got = _correct(route, prev, optimistic, column, prev_width=prev_width, width=width,
                           padding=0, dcp_size=dcp, cp_rank=rank)
            assert got.latent_slots.dtype == torch.int64
            assert got.latent_slots.tolist() == want, (route, start)


@pytest.mark.parametrize("dcp, rank", [(2, 0), (2, 1), (4, 3)])
def test_several_requests_and_padding_rows_write_the_runners_dcp_slots(dcp, rank):
    """Two requests at different points of their sequences and one padding row: the padding
    row is position 0 of the null block, rank 0's, so every other rank writes the null slot."""
    starts = _true_starts(dcp)[1:3]
    tables, prev, optimistic, column = _step(starts, dcp=dcp, prev_width=T, width=T, padding=1)
    want = _runner_slots(tables, starts, width=T, padding=1, dcp=dcp, rank=rank)
    for route in ("torch", "kernel"):
        got = _correct(route, prev, optimistic, column, prev_width=T, width=T, padding=1,
                       dcp_size=dcp, cp_rank=rank)
        assert got.latent_slots.tolist() == want, route


@pytest.mark.parametrize("dcp, rank", [(2, 0), (2, 1), (4, 2)])
def test_the_other_fields_do_not_depend_on_dcp(dcp, rank):
    """The positions, causal lengths and resume rows are the single-rank correction's (built
    here without a table: the DCP tables are too short for the single-rank page math)."""
    starts = _true_starts(dcp)[1:3]
    _, prev, optimistic, column = _step(starts, dcp=dcp, prev_width=T, width=T, padding=1)
    for route in ("torch", "kernel"):
        plain = _correct(route, prev, optimistic, None, prev_width=T, width=T, padding=1)
        sharded = _correct(route, prev, optimistic, column, prev_width=T, width=T, padding=1,
                           dcp_size=dcp, cp_rank=rank)
        for name in ("start_position", "linear_start", "seq_lens", "checkpoint_rows"):
            assert torch.equal(getattr(sharded, name), getattr(plain, name)), (route, name)


def test_a_rank_writes_only_its_own_rows_and_the_null_slot_elsewhere():
    """DCP 2, a step from the last two rows of block 1 (rank 1's, entry 0) into block 2
    (rank 0's, entry 1): each rank writes its two rows, the other two land on the null slot.
    The one-rank page math would read entry 2 (unallocated, -1) for the last two."""
    tables = [[11, 23]]
    column = _column(tables, 4, 0)
    prev = torch.tensor([3], dtype=torch.int32)  # the previous step kept every row
    optimistic = torch.tensor([2 * PAGE - 2], dtype=torch.int32)
    null = NULL_BLOCK_ID * PAGE
    want = {0: [null, null, 23 * PAGE + 0, 23 * PAGE + 1], 1: [11 * PAGE + 126, 11 * PAGE + 127, null, null]}
    for rank, slots in want.items():
        for route in ("torch", "kernel"):
            got = _correct(route, prev, optimistic, column, prev_width=T, width=T, padding=0,
                           dcp_size=2, cp_rank=rank)
            assert got.latent_slots.tolist() == slots, (route, rank)


def test_a_position_whose_entry_the_column_does_not_name_is_refused_by_name():
    """The runner's function refuses a position past the row's entries, owned or not
    ("a position outside the row has no slot"); so does the correction, on the local entry."""
    column = _column([[11]], 3, 0)  # one entry: positions 0 .. 255 at DCP 2
    for rank in (0, 1):
        with pytest.raises(ValueError, match="no slot"):
            _dcp_owned_latent_slots([11], torch.arange(254, 258), block_size=PAGE, dcp_size=2, cp_rank=rank)
        with pytest.raises(async_step.MtpAsyncStepError, match="decode_context_parallel_size=2"):
            async_step.mtp_async_correct(
                torch.tensor([3], dtype=torch.int32), torch.tensor([254], dtype=torch.int32), column,
                prev_width=T, width=T, page_size=PAGE, padding=0, dcp_size=2, cp_rank=rank,
            )


@pytest.mark.parametrize(
    "dcp_size, cp_rank, match",
    [(0, 0, "dcp_size"), (2, 2, "cp_rank"), (2, -1, "cp_rank"), (1, 1, "cp_rank")],
)
def test_a_dcp_geometry_outside_the_group_is_refused_by_name(dcp_size, cp_rank, match):
    tables, prev, optimistic, column = _step([PAGE - 2], dcp=2, prev_width=T, width=T, padding=0)
    for route in (async_step.mtp_async_correct_torch, async_step.mtp_async_correct):
        with pytest.raises(async_step.MtpAsyncStepError, match=match):
            route(prev, optimistic, column, prev_width=T, width=T, page_size=PAGE, padding=0,
                  dcp_size=dcp_size, cp_rank=cp_rank)


def test_the_null_block_is_the_runners():
    """``functional`` restates the null block rather than import the worker; the two agree."""
    assert async_step.NULL_BLOCK_ID == NULL_BLOCK_ID


def test_a_queue_full_step_served_on_the_host_keeps_the_dcp_slots(monkeypatch):
    """A launch the runtime refuses with a full queue is served by the torch route on the host
    (``_correct_on_the_host``): that route must take the same DCP geometry."""

    class _Refused:
        calls = {"take": 0, "correct": 0}

        def correct(self, *args, **kwargs):
            raise RuntimeError("Failed to schedule neff execution. status=7 message=Execution Queue Full")

    starts = [3 * PAGE - 1]
    tables, prev, optimistic, column = _step(starts, dcp=2, prev_width=T, width=T, padding=0)
    monkeypatch.setattr(async_step, "_launches_in_a_graph", lambda tensor: True)
    async_step.reset_dispatch_counters()
    got = async_step.mtp_async_correct(prev, optimistic, column, prev_width=T, width=T,
                                       page_size=PAGE, padding=0, dcp_size=2, cp_rank=1,
                                       launch=_Refused())
    assert async_step.queue_full_fallbacks() == 1
    assert got.latent_slots.tolist() == _runner_slots(tables, starts, width=T, padding=0, dcp=2, rank=1)


@pytest.fixture
def fresh_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def test_a_device_launch_compiles_the_dcp_geometry_into_its_signature(fresh_dynamo, monkeypatch):
    """The DCP size and rank are the kernel's compile-time ints: a repeated geometry compiles
    nothing, another one compiles its own graph, and each graph writes that rank's slots."""
    compiles: list = []

    def backend(graph_module, example_inputs):
        compiles.append(len(example_inputs))
        return graph_module.forward

    monkeypatch.setattr(async_step, "_launches_in_a_graph", lambda tensor: True)
    launch = async_step.DeviceLaunch(backend)
    starts = [2 * PAGE - 2]
    tables, prev, optimistic, column = _step(starts, dcp=2, prev_width=T, width=T, padding=0)
    for rank in (0, 0, 1, 1):
        got = async_step.mtp_async_correct(prev, optimistic, column, prev_width=T, width=T,
                                           page_size=PAGE, padding=0, dcp_size=2, cp_rank=rank,
                                           launch=launch)
        assert got.latent_slots.tolist() == _runner_slots(tables, starts, width=T, padding=0, dcp=2, rank=rank)
    assert compiles == [3, 3], compiles
    assert launch.calls == {"take": 0, "correct": 4}


def test_the_capability_flag_vouches_for_the_dcp_slots_on_both_routes():
    """``DCP_LATENT_SLOTS`` is what the platform's speculative-decoding refusal under DCP reads
    (worker-116's conjunction, needed when ``VLLM_NEURON_GLM5NEXT_MTP_ASYNC`` is set): True only
    beside the path it names, so it is pinned to that path's behaviour here. DCP 2, the step of
    :func:`test_a_rank_writes_only_its_own_rows_and_the_null_slot_elsewhere` at rank 1: two
    local rows, then two null slots, on the torch route and the kernel."""
    assert async_step.DCP_LATENT_SLOTS is True
    column = _column([[11, 23]], 4, 0)
    prev, optimistic = torch.tensor([3], dtype=torch.int32), torch.tensor([2 * PAGE - 2], dtype=torch.int32)
    null = NULL_BLOCK_ID * PAGE
    for route in ("torch", "kernel"):
        got = _correct(route, prev, optimistic, column, prev_width=T, width=T, padding=0,
                       dcp_size=2, cp_rank=1)
        assert got.latent_slots.tolist() == [11 * PAGE + 126, 11 * PAGE + 127, null, null], route
