# SPDX-License-Identifier: Apache-2.0
"""The ``T``-row DSA decode scores above 16384 candidates: ``decode_trow_wide``.

``decode_trow``'s kernel moves a row's scores out with one Tensor Engine transpose of the
whole ``[128, n_tiles]`` score tile, and a transpose takes ``[128, 128]`` at most: it
serves up to 16384 candidates (128 tiles). ``decode_trow_wide``'s kernel is the same body
with that transpose, copy and store done per group of 128 tiles; the model takes it for a
static candidate axis above 16384 (``decode_scores_rows_for``). Pinned here, on the CPU
simulator:

* (1) at 128 and 16384 candidates (one output group) the wide kernel gives the old
  kernel's bits;
* (2) above 16384, at a ragged last tile in the second group (16400, 32740), a whole
  second group (16512) and the ceiling (32768), it equals ``T`` sequential one-row
  ``decode_batch.dsa_decode_scores`` calls bit for bit (``test_decode_scores_trow``'s
  reference), with rows in both output groups and at the window's end;
* (3) against the torch oracle: ``BOUND_FILL`` at exactly the same places and the other
  scores within ``SCORE_RTOL`` / ``SCORE_ATOL`` (``test_decode_batch``'s form);
* the per-row position form (a DCP rank's operands), two programs (an LNC2 pair) and the
  on-chip rotation, as the old kernel's tests pin them;
* the route, the ceiling and its refusal, and the start check of the candidate axis;
* the drift guard: the two kernel bodies are one syntax tree but for the output block;
* the kernel key covers this module, ``decode_trow.py`` and ``kpool_hadamard.py``.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/functional/dsa/test_decode_scores_trow_wide.py
"""

from __future__ import annotations

import ast
import pathlib

import pytest
import torch

from vllm_neuron.functional.dsa import decode_trow as TR
from vllm_neuron.functional.dsa import decode_trow_wide as TW
from vllm_neuron.functional.dsa import kpool_hadamard as KH
from vllm_neuron.functional.dsa.causal_bound import BOUND_FILL
from vllm_neuron.utils.neuron_utils import can_run_kernel

from test.vllm_neuron.functional.dsa.test_decode_batch import SCORE_ATOL, SCORE_RTOL
from test.vllm_neuron.functional.dsa.test_decode_scores_trow import _rotated, _sequential

POOL = 4
HEAD_DIM = 128
HEADS = 32
PARTITIONS = 128
#: Row 0's position of a step at output group 1's first pool: rows 0-2 keep candidates
#: 0-16383 (group 0), and the last of 4 rows closes candidate 16384, stands it in from
#: ``pooled`` and keeps it (group 1's first tile).
GROUP_EDGE = PARTITIONS * PARTITIONS * POOL
#: One output group (128, 16384), a ragged last tile in group 1 (16400, 32740), a whole
#: group-1 tile (16512) and the ceiling (32768, the 1M / DCP 8 axis).
ONE_GROUP = (128, 16384)
TWO_GROUPS = (16400, 16512, 32740, 32768)


def _bf16(gen, *shape, scale=1.0):
    return (torch.randn(shape, generator=gen) * scale).to(torch.bfloat16)


def _positions(candidates, rows, batch):
    """Row 0's position per request: the window's last step (its last row at the window's
    last token), then the step at the group edge (mid-window when the axis has one
    group)."""
    window = candidates * POOL
    second = GROUP_EDGE if window >= GROUP_EDGE + rows else window // 3
    return torch.tensor([window - rows, second][:batch], dtype=torch.int32)


def _case(batch, rows, candidates, *, seed, slots_total=3):
    gen = torch.Generator().manual_seed(seed)
    bank = _bf16(gen, slots_total, candidates + 1, HEAD_DIM, scale=0.5)
    slots = torch.randperm(slots_total, generator=gen)[:batch].to(torch.int32)
    query = _bf16(gen, batch * rows, HEADS, HEAD_DIM)
    weights = torch.randn(batch * rows, HEADS, generator=gen) * HEADS ** -0.5
    pooled = _bf16(gen, batch * rows, HEAD_DIM, scale=0.5)
    return query, weights, bank, slots, _positions(candidates, rows, batch), pooled


def _set_lnc(monkeypatch, lnc):
    if lnc is None:
        monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    else:
        monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", lnc)


def _wide(case, candidates, monkeypatch, lnc=None, **options):
    _set_lnc(monkeypatch, lnc)
    TW.reset_decode_trow_wide_dispatch_counters()
    out = TW.dsa_decode_scores_rows_wide(*case, candidates=candidates, pool_size=POOL,
                                         **options)
    assert TW.decode_trow_wide_dispatch_counters() == (1, 0)
    return out


def _old(case, candidates, monkeypatch, **options):
    _set_lnc(monkeypatch, None)
    TR.reset_decode_trow_dispatch_counters()
    out = TR.dsa_decode_scores_rows(*case, candidates=candidates, pool_size=POOL, **options)
    assert TR.decode_trow_dispatch_counters() == (1, 0)
    return out


def _assert_scores_agree(got, want):
    filled = want == BOUND_FILL
    assert torch.equal(got == BOUND_FILL, filled)
    torch.testing.assert_close(got[~filled], want[~filled], rtol=SCORE_RTOL, atol=SCORE_ATOL)


@pytest.fixture(autouse=True)
def _kernels_on():
    assert can_run_kernel(), "run with VLLM_NEURON_CPU_MODE=1 and NKI_SIMULATOR=1"


# ------------------------------------------------------------------ the scores


@pytest.mark.parametrize("candidates", ONE_GROUP)
@pytest.mark.parametrize("batch, rows", [(1, 1), (1, 4), (2, 4)])
def test_one_output_group_gives_the_old_kernels_bits(batch, rows, candidates, monkeypatch):
    case = _case(batch, rows, candidates, seed=rows + 10 * batch + candidates)
    assert torch.equal(_wide(case, candidates, monkeypatch), _old(case, candidates, monkeypatch))


@pytest.mark.parametrize("candidates", TWO_GROUPS)
@pytest.mark.parametrize("batch, rows", [(1, 1), (1, 4)])
def test_two_output_groups_equal_sequential_one_row_scores_bit_for_bit(batch, rows,
                                                                      candidates,
                                                                      monkeypatch):
    case = _case(batch, rows, candidates, seed=100 + rows + 10 * batch + candidates)
    got = _wide(case, candidates, monkeypatch)
    assert got.shape == (batch * rows, candidates) and got.dtype == torch.float32
    assert torch.equal(got, _sequential(*case, candidates))


def test_both_requests_of_a_batch_at_the_ceiling_equal_sequential_scores(monkeypatch):
    """Two requests at 32768 candidates: the window's last step and the step at the group
    edge, whose rows end their bounds on both sides of it."""
    candidates = TW.MAX_DECODE_SCORES_ROWS_WIDE_CANDIDATES
    case = _case(2, 4, candidates, seed=201)
    got = _wide(case, candidates, monkeypatch)
    assert torch.equal(got, _sequential(*case, candidates))
    kept = candidates - (got[4:] == BOUND_FILL).sum(dim=1)
    assert kept.tolist() == [PARTITIONS * PARTITIONS] * 3 + [PARTITIONS * PARTITIONS + 1]


@pytest.mark.parametrize("candidates", (16400, TW.MAX_DECODE_SCORES_ROWS_WIDE_CANDIDATES))
def test_two_output_groups_agree_with_the_torch_oracle(candidates, monkeypatch):
    case = _case(2, 4, candidates, seed=300 + candidates)
    want = TR.dsa_decode_scores_rows_torch_oracle(*case, candidates=candidates,
                                                  pool_size=POOL)
    _assert_scores_agree(_wide(case, candidates, monkeypatch), want)


def test_the_per_row_position_form_is_the_per_request_form(monkeypatch):
    """A DCP rank hands each row's own position ([B * rows]); fed ``p_b + t`` it is the
    per-request form bit for bit, here across the group edge."""
    candidates = TW.MAX_DECODE_SCORES_ROWS_WIDE_CANDIDATES
    case = _case(2, 4, candidates, seed=401)
    position = case[4]
    per_row = (position.reshape(2, 1) + torch.arange(4, dtype=torch.int32)).reshape(8)
    want = _wide(case, candidates, monkeypatch)
    got = _wide(case[:4] + (per_row,) + case[5:], candidates, monkeypatch)
    assert torch.equal(got, want)


def test_two_programs_agree_with_one(monkeypatch):
    candidates = TW.MAX_DECODE_SCORES_ROWS_WIDE_CANDIDATES
    case = _case(2, 4, candidates, seed=501)
    one = _wide(case, candidates, monkeypatch)
    two = _wide(case, candidates, monkeypatch, lnc="2")
    assert TW.decode_trow_wide_route_counts() == (0, 1)
    assert torch.equal(one, two)


def test_a_projected_query_is_rotated_on_chip_as_the_rotation_kernel_rotates_it(monkeypatch):
    candidates = TW.MAX_DECODE_SCORES_ROWS_WIDE_CANDIDATES
    case = _case(1, 4, candidates, seed=601)
    want = _wide((_rotated(case[0]),) + case[1:], candidates, monkeypatch)
    KH.reset_kpool_hadamard_dispatch_counters()
    got = _wide(case, candidates, monkeypatch, rotate_query=True)
    assert KH.kpool_hadamard_dispatch_counters() == (0, 0)
    assert torch.equal(got, want)


# ------------------------------------------------------------------ the route


def test_the_route_takes_the_old_kernel_up_to_its_bound_and_the_wide_one_above():
    bound = TW.MAX_DECODE_SCORES_ROWS_CANDIDATES
    assert bound == PARTITIONS * PARTITIONS == 16384
    assert TW.decode_scores_rows_for(bound) is TR.dsa_decode_scores_rows
    assert TW.decode_scores_rows_for(bound + 1) is TW.dsa_decode_scores_rows_wide
    assert TW.decode_scores_rows_for(1) is TR.dsa_decode_scores_rows


@pytest.mark.parametrize("candidates, old, wide", [(16384, 1, 0), (16512, 0, 1)])
def test_the_routed_call_dispatches_one_kernel(candidates, old, wide, monkeypatch):
    case = _case(1, 2, candidates, seed=701 + candidates)
    _set_lnc(monkeypatch, None)
    TR.reset_decode_trow_dispatch_counters()
    TW.reset_decode_trow_wide_dispatch_counters()
    TW.decode_scores_rows_for(candidates)(*case, candidates=candidates, pool_size=POOL)
    assert TR.decode_trow_dispatch_counters() == (old, 0)
    assert TW.decode_trow_wide_dispatch_counters() == (wide, 0)


# ------------------------------------------------------------------ the ceiling


def test_the_ceiling_is_two_output_groups():
    assert TW.OUTPUT_GROUP_TILES == PARTITIONS
    assert TW.MAX_DECODE_SCORES_ROWS_WIDE_CANDIDATES == 2 * 128 * 128 == 32768


def test_the_wrapper_refuses_an_axis_past_the_ceiling_naming_the_bound_and_the_fix():
    candidates = TW.MAX_DECODE_SCORES_ROWS_WIDE_CANDIDATES + 1
    case = list(_case(1, 2, 128, seed=801))
    case[2] = torch.zeros(1, candidates + 1, HEAD_DIM, dtype=torch.bfloat16)
    case[3] = torch.zeros(1, dtype=torch.int32)
    with pytest.raises(TR.DecodeTrowError,
                       match=r"MAX_DECODE_SCORES_ROWS_WIDE_CANDIDATES=32768.*decode_trow_wide\.py"
                             r".*max_model_len.*decode_context_parallel_size"):
        TW.dsa_decode_scores_rows_wide(*case, candidates=candidates, pool_size=POOL)


def test_the_wrapper_runs_the_old_operand_checks():
    query, weights, bank, slots, position, pooled = _case(2, 2, 16400, seed=802)
    with pytest.raises(TR.DecodeTrowError, match="pooled"):
        TW.dsa_decode_scores_rows_wide(query, weights, bank, slots, position, pooled[:2],
                                       candidates=16400, pool_size=POOL)
    with pytest.raises(TR.DecodeTrowError, match="window"):
        TW.dsa_decode_scores_rows_wide(query, weights, bank, slots, position + 8, pooled,
                                       candidates=16400, pool_size=POOL)


@pytest.mark.parametrize("rank_context, pool_size, drafts", [
    (131072, 4, 3),          # 1M at DCP 8 with MTP k = 3: 32768 candidates
    (131072, 4, 1),
    (262144, 4, 0),          # no drafts: the one-token kernel serves, no ceiling
    (1 << 40, None, 3),      # no indexer
])
def test_the_start_check_admits(rank_context, pool_size, drafts):
    TW.check_decode_scores_rows_context(rank_context, pool_size, drafts)


@pytest.mark.parametrize("rank_context, drafts", [(262144, 3), (131076, 1)])
def test_the_start_check_refuses_an_axis_past_the_ceiling_when_mtp_drafts(rank_context,
                                                                          drafts):
    with pytest.raises(TR.DecodeTrowError,
                       match=rf"{rank_context}.*{rank_context // POOL} candidates"
                             r".*MAX_DECODE_SCORES_ROWS_WIDE_CANDIDATES=32768"
                             r".*decode_trow_wide\.py.*max_model_len"):
        TW.check_decode_scores_rows_context(rank_context, POOL, drafts)


# ------------------------------------------------------------------ the copy's guards


def _starts_the_output_block(statement: ast.stmt) -> bool:
    """The old block's first statement (the ``o_ps`` tile) or the output-group loop."""
    if isinstance(statement, ast.Assign):
        return any(isinstance(t, ast.Name) and t.id == "o_ps" for t in statement.targets)
    return (isinstance(statement, ast.For) and isinstance(statement.target, ast.Name)
            and statement.target.id == "o0")


def _kernel_without_output_block(path: pathlib.Path, name: str) -> str:
    """The kernel's syntax tree with its docstring, its name and the output block (from
    its first statement to the end of the row loop that holds it) removed."""
    tree = ast.parse(path.read_text())
    kernel = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == name)
    kernel.name = "kernel"
    if isinstance(kernel.body[0], ast.Expr) and isinstance(kernel.body[0].value, ast.Constant):
        kernel.body = kernel.body[1:]
    # Breadth first: the row loop holding the block is reached before any loop inside it.
    loop = next(node for node in ast.walk(kernel) if isinstance(node, ast.For)
                and any(_starts_the_output_block(s) for s in node.body))
    loop.body = loop.body[:next(i for i, s in enumerate(loop.body) if _starts_the_output_block(s))]
    left = [node for node in ast.walk(kernel)
            if isinstance(node, ast.Name) and node.id in ("o_ps", "o0")]
    assert not left, f"{path.name}: output-block names outside the cut block"
    return ast.dump(kernel)


def test_the_wide_kernel_is_the_old_body_but_for_the_output_block():
    """An edit to one kernel body fails this until the other follows."""
    old = _kernel_without_output_block(pathlib.Path(TR.__file__), "dsa_decode_scores_rows_kernel")
    wide = _kernel_without_output_block(pathlib.Path(TW.__file__),
                                        "dsa_decode_scores_rows_wide_kernel")
    assert wide == old


def test_the_kernel_key_covers_both_kernel_files_and_the_rotation_constants():
    """The wide kernel reads ``decode_trow``'s helpers and ``kpool_hadamard``'s rotation
    size and scale: an edit to any of the three files re-keys it."""
    from vllm_neuron.source_digest import source_digest

    assert TW.SOURCE_DIGEST == source_digest(TW.__file__, TR.__file__, KH.__file__)
