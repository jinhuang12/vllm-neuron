# SPDX-License-Identifier: Apache-2.0
"""The DCP candidate merge's tie step: a device branch only where a program runs one group.

``dsa_dcp_merge_select_kernel`` reselects a group's ties on a device branch, an
``nl.fori_loop`` of 0 or 1 trips on the pair's joint flag (``_any_excess``). neuronx-cc lowers
each such branch to a dynamic loop. A program of more than one group (more than
``_TILES_IN_FLIGHT`` tiles of rows, the 1024-row prefill chunk) reselects every group's ties
with no branch: the tie step leaves a row without an excess tie as the threshold selected it,
so the selection is the same and the launch adds no dynamic loop. These tests pin, on the
NKI simulator under the LNC2 grid, at ``CP = 2`` and ``k = 512``:

1. a launch of two groups a program traces no branch and no joint flag, and selects as the
   torch oracle does, for both ranks, with no tie, with ties in one tile of the first core,
   in the second group of the first core, and in the second core's rows only;
2. a launch of one group a program keeps a branch on the joint flag for its group: one row
   (a decode step), and 512 rows (two tiles a program), where it selects as the torch oracle
   does with no tie, with ties in one of a group's two tiles, and in the second core's rows
   only.
"""

from __future__ import annotations

import nki.language as nl
import pytest
import torch

from vllm_neuron.functional.dsa import dcp_candidate_merge as merge

CP = 2
K = 512
PPB = 32
PREFILL_ROWS = 1024
#: Rows of the excess-tie block in each case: the first core's second tile (one tile of its
#: first group), the first core's second group, and a tile of the second core.
TIE_ROWS = {"no_tie": None, "first_group_one_tile": (128, 256),
            "second_group": (256, 384), "second_core": (640, 768)}
#: Rows that give each program one group of two 128-row tiles.
ONE_GROUP_ROWS = 512
#: Rows of the excess-tie block at :data:`ONE_GROUP_ROWS`: the first core's second tile, and
#: the second core's first tile.
ONE_GROUP_TIE_ROWS = {"no_tie": None, "one_tile": (128, 256), "second_core": (256, 384)}


def _case(rows: int, tie_rows: tuple[int, int] | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Every rank's ``k`` candidates a row: distinct fp32 values and distinct local ids. In
    ``tie_rows`` every candidate has one value, so ``CP * k`` candidates tie for ``k`` places."""
    g = torch.Generator().manual_seed(rows)
    values = torch.randn(CP, rows, K, generator=g, dtype=torch.float32)
    ids = torch.stack([torch.stack([torch.randperm(16384, generator=g)[:K]
                                    for _ in range(rows)]) for _ in range(CP)]).to(torch.int32)
    if tie_rows is not None:
        values[:, tie_rows[0]:tie_rows[1], :] = 0.5
    return values.contiguous(), ids.contiguous()


@pytest.fixture
def branches(monkeypatch):
    """Count the kernel's tie branches and joint flags as it is traced."""
    counts = {"fori_loop": 0, "any_excess": 0}
    fori_loop = nl.fori_loop
    any_excess = merge._any_excess

    def counted_fori_loop(*args, **kwargs):
        counts["fori_loop"] += 1
        return fori_loop(*args, **kwargs)

    def counted_any_excess(*args, **kwargs):
        counts["any_excess"] += 1
        return any_excess(*args, **kwargs)

    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    monkeypatch.setattr(nl, "fori_loop", counted_fori_loop)
    monkeypatch.setattr(merge, "_any_excess", counted_any_excess)
    assert merge.merge_programs() == 2
    return counts


def _merge(values: torch.Tensor, ids: torch.Tensor, rank: int) -> torch.Tensor:
    merge.reset_dcp_merge_select_dispatch_counters()
    got = merge.dsa_dcp_merge_select(values, ids, cp_rank=rank, pools_per_block=PPB)
    assert merge.dcp_merge_select_dispatch_counters() == (1, 0)
    return got


@pytest.mark.parametrize("case", sorted(TIE_ROWS))
def test_a_two_group_launch_has_no_tie_branch_and_selects_as_the_oracle(branches, case):
    assert len(merge.program_groups(PREFILL_ROWS, 2)[0]) == 2
    values, ids = _case(PREFILL_ROWS, TIE_ROWS[case])
    for rank in range(CP):
        got = _merge(values, ids, rank)
        want = merge.dsa_dcp_merge_select_torch_oracle(values, ids, cp_rank=rank,
                                                       pools_per_block=PPB)
        assert torch.equal(got, want), (case, rank)
    assert branches == {"fori_loop": 0, "any_excess": 0}


def test_a_one_group_launch_keeps_its_tie_branch(branches):
    assert len(merge.program_groups(1, 2)[0]) == 1
    values, ids = _case(1, None)
    got = _merge(values, ids, 0)
    want = merge.dsa_dcp_merge_select_torch_oracle(values, ids, cp_rank=0, pools_per_block=PPB)
    assert torch.equal(got, want)
    assert branches["fori_loop"] >= 1
    assert branches["fori_loop"] == branches["any_excess"]


@pytest.mark.parametrize("case", sorted(ONE_GROUP_TIE_ROWS))
def test_a_one_group_launch_of_two_tiles_branches_and_selects_as_the_oracle(branches, case):
    groups = merge.program_groups(ONE_GROUP_ROWS, 2)
    assert [len(program) for program in groups] == [1, 1]
    assert [len(program[0][1]) for program in groups] == [2, 2]
    values, ids = _case(ONE_GROUP_ROWS, ONE_GROUP_TIE_ROWS[case])
    for rank in range(CP):
        got = _merge(values, ids, rank)
        want = merge.dsa_dcp_merge_select_torch_oracle(values, ids, cp_rank=rank,
                                                       pools_per_block=PPB)
        assert torch.equal(got, want), (case, rank)
    # One branch on the joint flag a group: 2 ranks x 2 programs x 1 group.
    assert branches == {"fori_loop": 4, "any_excess": 4}
