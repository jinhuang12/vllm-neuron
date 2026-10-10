# SPDX-License-Identifier: Apache-2.0
"""``functional/mtp/tail_out.py``: the draft iteration's output tail as one NKI kernel.

The kernel replaces the traced tail of the head's draft loop (``mtp.py``)::

    mixed = attended + ffn                                    # bf16 add
    hidden = rms(mixed, shared_head_norm)                     # fp32 math, bf16 out
    logits = linear(hidden, head_rows)                        # bf16 x bf16 -> bf16
    pair = (logits.max(-1), logits.argmax(-1))                # fp32 [B, 2]

over this rank's vocab shard ``head_rows [Vs, H]``; the pair is what the draft-token
gather (``functional/draft_token.py``) consumes. The bf16 add is a single rounding of
the exact sum on both sides, so ``mixed`` is bit-equal to the reference's and the
checks start at the norm:

* ``hidden`` equals the exact norm's bf16 rounding except, by one bf16 step, at the
  elements the derived term in ``test_mtp_tail_bounds.py`` makes eligible;
* the exact logits are taken from the kernel's OWN ``hidden`` (so no flip term), and
  the kernel's reported max is the bf16 logit at the reported index: it is within the
  GEMV bound of the exact logit there, that logit is within two bounds of the exact
  maximum, and on every DECISIVE row -- exact top-1 minus top-2 margin above the two
  candidates' bounds -- the reported index is the exact argmax. Most rows are decisive
  (asserted from ``DECISIVE_SAMPLE`` rows up), so the check reads the argmax and not
  only its neighbourhood;
* ties: two identical head rows produce identical logits, and the reported index is
  the lower one, the convention of ``torch.argmax`` the traced route had.

Shapes: the tiny fixture's H=512 with 64 head rows and the served H=4096 with the
TP=64 shard of 2420 rows; B in {1, 4, 64} and 130 (two partition tiles); one program
and two (LNC2, the shard split in two with the pair combined across the programs).
"""

from __future__ import annotations

import pytest
import torch

from test.vllm_neuron.functional.test_mtp_tail_bounds import (
    assert_normed_within_one_flip,
    gemv_bound_rounded_rows,
    rms64,
)
from vllm_neuron.functional.mtp import tail_out

EPS = 1e-5
TINY_HIDDEN, TINY_ROWS = 512, 64
#: Rows from which most are required to be decisive (a single row may be a close race).
DECISIVE_SAMPLE = 4
SERVED_HIDDEN, SERVED_SHARD_ROWS = 4096, 154880 // 64


def _operands(seed: int, batch: int, hidden: int, rows: int) -> dict:
    gen = torch.Generator().manual_seed(seed)
    attended = torch.randn(batch, hidden, generator=gen).to(torch.bfloat16)
    ffn = torch.randn(batch, hidden, generator=gen).to(torch.bfloat16)
    gain = (1.0 + torch.randn(hidden, generator=gen) * 0.05).to(torch.bfloat16)
    head_rows = (torch.randn(rows, hidden, generator=gen) * hidden ** -0.5).to(torch.bfloat16)
    return dict(attended=attended, ffn=ffn, gain=gain, head_rows=head_rows)


def _run(ops: dict) -> tuple[torch.Tensor, torch.Tensor]:
    tail_out.reset_dispatch_counters()
    hidden, pair = tail_out.mtp_tail_out(**ops, eps=EPS)
    assert tail_out.dispatch_counters()[0] >= 1 and tail_out.dispatch_counters()[1] == 0, (
        f"the kernel must serve and the torch route not: {tail_out.dispatch_counters()}")
    return hidden, pair


def _exact_hidden(ops: dict) -> torch.Tensor:
    mixed = (ops["attended"].double() + ops["ffn"].double()).to(torch.bfloat16)
    return rms64(mixed, ops["gain"], EPS)


def _check(hidden: torch.Tensor, pair: torch.Tensor, ops: dict, *,
           require_decisive: bool = True) -> None:
    """The module doc's checks; ``require_decisive`` is off for the planted-tie cases."""
    batch, width = ops["attended"].shape
    rows = ops["head_rows"].shape[0]
    assert hidden.dtype == torch.bfloat16 and tuple(hidden.shape) == (batch, width)
    assert pair.dtype == torch.float32 and tuple(pair.shape) == (batch, 2)
    assert_normed_within_one_flip(hidden, _exact_hidden(ops), width, "hidden")

    exact = hidden.double() @ ops["head_rows"].double().t()               # [B, Vs]
    reported = pair[:, 1].to(torch.int64)
    assert bool((pair[:, 1] == reported.to(torch.float32)).all()), "indices are whole numbers"
    assert bool((reported >= 0).all()) and bool((reported < rows).all()), reported.tolist()
    at = exact.gather(1, reported.reshape(-1, 1)).reshape(-1)
    bound = gemv_bound_rounded_rows(exact, hidden, ops["head_rows"])          # [B, Vs]
    bound_at = bound.gather(1, reported.reshape(-1, 1)).reshape(-1)
    assert bool(((pair[:, 0].double() - at).abs() <= bound_at).all()), (
        "the reported max is not the bf16 logit at the reported index")
    top2 = exact.topk(2, dim=-1)
    best, runner = top2.values[:, 0], top2.values[:, 1]
    best_idx = top2.indices[:, 0]
    bound_best = bound.gather(1, best_idx.reshape(-1, 1)).reshape(-1)
    assert bool((at >= best - bound_best - bound_at).all()), "the reported logit is not near-maximal"
    bound_runner = bound.gather(1, top2.indices[:, 1:2]).reshape(-1)
    decisive = (best - runner) > (bound_best + bound_runner)
    if require_decisive and batch >= DECISIVE_SAMPLE:
        share = float(decisive.double().mean())
        assert share >= 0.5, f"only {share:.0%} rows decisive"
    assert torch.equal(reported[decisive], best_idx[decisive]), (
        f"decisive rows disagree: got {reported[decisive].tolist()}, exact {best_idx[decisive].tolist()}")


@pytest.mark.parametrize(
    "batch, hidden, rows",
    [(1, TINY_HIDDEN, TINY_ROWS), (4, TINY_HIDDEN, TINY_ROWS), (64, TINY_HIDDEN, TINY_ROWS),
     (130, TINY_HIDDEN, TINY_ROWS), (1, SERVED_HIDDEN, SERVED_SHARD_ROWS),
     (4, SERVED_HIDDEN, SERVED_SHARD_ROWS)],
    ids=["b1_h512_v64", "b4_h512_v64", "b64_h512_v64", "b130_h512_two_tiles",
         "b1_h4096_shard2420", "b4_h4096_shard2420"],
)
def test_hidden_and_the_pair_are_within_the_derived_bounds(batch, hidden, rows) -> None:
    ops = _operands(7_501 + batch, batch, hidden, rows)
    got_hidden, pair = _run(ops)
    _check(got_hidden, pair, ops)


def test_a_tie_between_two_head_rows_reports_the_lower_index() -> None:
    """Identical rows in two different 512-column chunks: identical logits, lower index wins."""
    ops = _operands(7_601, 3, TINY_HIDDEN, 1100)
    hidden_ref = _exact_hidden(ops).to(torch.bfloat16).float()
    low, high = 17, 900
    ops["head_rows"][low] = (hidden_ref[0] / hidden_ref[0].norm() * 8.0).to(torch.bfloat16)
    ops["head_rows"][high] = ops["head_rows"][low]
    hidden, pair = _run(ops)
    logits = torch.nn.functional.linear(hidden, ops["head_rows"]).float()
    assert torch.equal(logits[0, low], logits[0, high]) and int(logits[0].argmax()) == low
    assert int(pair[0, 1]) == low, f"the tie resolved to {int(pair[0, 1])}, not {low}"
    _check(hidden, pair, ops, require_decisive=False)


def test_two_programs_agree_with_one(monkeypatch) -> None:
    """LNC2 splits the shard rows over two programs and combines the pairs; a tie
    across the two programs resolves to program 0's (lower) index."""
    ops = _operands(7_701, 4, TINY_HIDDEN, 1000)
    hidden_ref = _exact_hidden(ops).to(torch.bfloat16).float()
    low, high = 3, 997                      # one row per program
    ops["head_rows"][low] = (hidden_ref[1] / hidden_ref[1].norm() * 8.0).to(torch.bfloat16)
    ops["head_rows"][high] = ops["head_rows"][low]
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "1")
    hidden_one, pair_one = _run(ops)
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    assert tail_out.launch_programs() == 2
    hidden_two, pair_two = _run(ops)
    assert tail_out.dispatch_counters() == (1, 0), "one launch, two programs"
    assert torch.equal(hidden_two, hidden_one)
    assert torch.equal(pair_two, pair_one), (pair_two.tolist(), pair_one.tolist())
    assert int(pair_two[1, 1]) == low
    _check(hidden_two, pair_two, ops, require_decisive=False)


def test_two_partition_tiles_equal_the_tiles_run_apart() -> None:
    ops = _operands(7_801, 130, TINY_HIDDEN, TINY_ROWS)
    hidden, pair = _run(ops)
    parts = []
    for lo, hi in ((0, 128), (128, 130)):
        sub = dict(ops, attended=ops["attended"][lo:hi], ffn=ops["ffn"][lo:hi])
        parts.append(_run(sub))
    assert torch.equal(hidden, torch.cat([p[0] for p in parts]))
    assert torch.equal(pair, torch.cat([p[1] for p in parts]))


def test_the_torch_route_is_stage_a_arithmetic_bit_for_bit() -> None:
    """``mtp_tail_out_torch`` is the draft loop's tail as the traced head wrote it."""
    ops = _operands(7_901, 5, TINY_HIDDEN, TINY_ROWS)
    hidden, pair = tail_out.mtp_tail_out_torch(**ops, eps=EPS)
    mixed = ops["attended"] + ops["ffn"]
    x = mixed.to(torch.float32)
    want_hidden = (x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
                   * ops["gain"].to(torch.float32)).to(mixed.dtype)
    logits = torch.nn.functional.linear(want_hidden, ops["head_rows"]).to(torch.float32)
    local_max, local_arg = logits.max(dim=-1)
    want_pair = torch.stack([local_max, local_arg.to(torch.float32)], dim=-1)
    assert torch.equal(hidden, want_hidden) and torch.equal(pair, want_pair)
    assert torch.equal(tail_out.shard_pair(logits), want_pair)


def test_the_torch_route_is_inside_the_bounds_as_well() -> None:
    ops = _operands(8_001, 4, TINY_HIDDEN, TINY_ROWS)
    hidden, pair = tail_out.mtp_tail_out_torch(**ops, eps=EPS)
    _check(hidden, pair, ops)


def test_shard_pair_takes_the_first_of_equal_maxima() -> None:
    logits = torch.tensor([[1.0, 5.0, 5.0, 2.0], [7.0, 7.0, 7.0, 7.0]])
    pair = tail_out.shard_pair(logits)
    assert pair.dtype == torch.float32 and pair.tolist() == [[5.0, 1.0], [7.0, 0.0]]


@pytest.mark.parametrize(
    "mutation, match",
    [
        (lambda o: o.update(ffn=o["ffn"][:, :256]), "ffn"),
        (lambda o: o.update(attended=o["attended"].float()), "attended"),
        (lambda o: o.update(gain=o["gain"][:500]), "gain"),
        (lambda o: o.update(head_rows=o["head_rows"][:, :500]), "head_rows"),
        (lambda o: o.update(head_rows=o["head_rows"][:7]), "8"),
        (lambda o: o.update(head_rows=o["head_rows"].float()), "head_rows"),
    ],
    ids=["ffn_width", "attended_dtype", "gain_length", "head_columns", "too_few_rows", "head_dtype"],
)
def test_a_geometry_that_is_not_the_tails_is_refused_by_name(mutation, match) -> None:
    ops = _operands(8_101, 3, TINY_HIDDEN, TINY_ROWS)
    mutation(ops)
    with pytest.raises(tail_out.MtpTailError, match=match):
        tail_out.mtp_tail_out(**ops, eps=EPS)


def test_more_shard_rows_than_one_program_can_scan_are_refused() -> None:
    """``max8`` scans at most 16384 elements per partition; a shard past that per program
    is refused (the served shard is 2420)."""
    ops = _operands(8_201, 1, 128, tail_out.MAX_ROWS_PER_PROGRAM + 1)
    with pytest.raises(tail_out.MtpTailError, match="16384"):
        tail_out.mtp_tail_out(**ops, eps=EPS)
    ops = _operands(8_202, 1, 128, tail_out.MAX_ROWS_PER_PROGRAM)
    _check(*_run(ops), ops)


def test_a_hidden_size_that_is_not_a_multiple_of_the_partition_count_is_refused() -> None:
    ops = _operands(8_301, 2, 384 + 64, TINY_ROWS)
    with pytest.raises(tail_out.MtpTailError, match="128"):
        tail_out.mtp_tail_out(**ops, eps=EPS)
