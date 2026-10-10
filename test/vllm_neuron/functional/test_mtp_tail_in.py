# SPDX-License-Identifier: Apache-2.0
"""``functional/mtp/tail_in.py``: the draft iteration's input tail as one NKI kernel.

The kernel replaces the head's traced ``_layer_input`` (``mtp.py``)::

    embeds = table[token_ids]; embeds[positions == 0] = 0
    e = rms(embeds, enorm); h = rms(previous, hnorm)          # fp32 math, bf16 out
    out = linear(cat([e, h]), eh_proj_rows)                   # bf16 x bf16 -> bf16

with ``eh_proj_rows`` this rank's row shard ``[H / world, 2H]`` (the whole ``[H, 2H]``
at one rank). The reference is the same arithmetic in float64 from the same bf16
operands, and the comparison is to the bound DERIVED from the kernel's arithmetic in
``test_mtp_tail_bounds.py`` (``gemv_bound``: the output's own rounding, the norms'
eligible one-step flips carried through the weights, the ``2H``-term accumulation
order), not a chosen tolerance. Two mutant references (the halves concatenated in the
wrong order; the position-0 mask dropped) violate the bound, so it reads the arithmetic
rather than passing anything.

Shapes walked: the tiny fixture's H=512 (vocab 64) and the served H=4096; one rank
(the whole ``eh_proj``) and a 64-way row shard; B in {1, 4, 64} and 130 (two partition
tiles). The simulator runs one program.
"""

from __future__ import annotations

import pytest
import torch

from test.vllm_neuron.functional.test_mtp_tail_bounds import gemv_bound, rms64
from vllm_neuron.functional.mtp import tail_in

EPS = 1e-5
TINY_HIDDEN, TINY_VOCAB = 512, 64
SERVED_HIDDEN, SERVED_WORLD = 4096, 64
#: Positions whose embedding half the head masks (upstream: no previous token).
MASKED_POSITION = 0


def exact_reference(ids, table, positions, previous, enorm, hnorm, rows, eps, *,
                    mask: bool = True, swap_halves: bool = False):
    """``(exact [B, R] fp64, joined_exact [B, 2H] fp64)`` of the tail from bf16 operands.

    The GEMV consumes the bf16-ROUNDED normed halves, as the kernel does; ``mask`` and
    ``swap_halves`` build the two mutants.
    """
    embeds = table[ids.to(torch.int64)].double()
    if mask:
        embeds = torch.where((positions == MASKED_POSITION).unsqueeze(-1),
                             torch.zeros_like(embeds), embeds)
    halves = [rms64(embeds, enorm, eps), rms64(previous, hnorm, eps)]
    if swap_halves:
        halves.reverse()
    joined = torch.cat(halves, dim=-1)
    joined_bf = joined.to(torch.bfloat16).double()
    return joined_bf @ rows.double().t(), joined


def _operands(seed: int, batch: int, hidden: int, vocab: int, world: int, rank: int,
              zero_positions: tuple[int, ...] = ()):
    gen = torch.Generator().manual_seed(seed)
    table = torch.randn(vocab, hidden, generator=gen).to(torch.bfloat16)
    ids = torch.randint(0, vocab, (batch,), generator=gen, dtype=torch.int32)
    positions = torch.randint(1, 4096, (batch,), generator=gen, dtype=torch.int32)
    for row in zero_positions:
        positions[row] = MASKED_POSITION
    previous = torch.randn(batch, hidden, generator=gen).to(torch.bfloat16)
    enorm = (1.0 + torch.randn(hidden, generator=gen) * 0.05).to(torch.bfloat16)
    hnorm = (1.0 + torch.randn(hidden, generator=gen) * 0.05).to(torch.bfloat16)
    eh_proj = (torch.randn(hidden, 2 * hidden, generator=gen) * (2 * hidden) ** -0.5).to(torch.bfloat16)
    shard = tail_in.eh_proj_shard_rows(hidden, world)
    rows = eh_proj[rank * shard:(rank + 1) * shard].contiguous()
    return dict(token_ids=ids, table=table, positions=positions, previous=previous,
                enorm=enorm, hnorm=hnorm, eh_proj_rows=rows)


def _run(ops: dict) -> torch.Tensor:
    tail_in.reset_dispatch_counters()
    out = tail_in.mtp_tail_in(**ops, eps=EPS)
    assert tail_in.dispatch_counters() == (1, 0), (
        f"the kernel must dispatch once and the torch route not at all: "
        f"{tail_in.dispatch_counters()}")
    return out


def _check(out: torch.Tensor, ops: dict, hidden: int) -> torch.Tensor:
    """Asserts the bound; returns the exact reference for the callers' controls."""
    batch, rows = ops["token_ids"].shape[0], ops["eh_proj_rows"].shape[0]
    assert out.dtype == torch.bfloat16 and tuple(out.shape) == (batch, rows), (out.dtype, out.shape)
    exact, joined = exact_reference(ops["token_ids"], ops["table"], ops["positions"],
                                    ops["previous"], ops["enorm"], ops["hnorm"],
                                    ops["eh_proj_rows"], EPS)
    bound = gemv_bound(out, joined, ops["eh_proj_rows"], hidden)
    diff = (out.double() - exact).abs()
    assert bool((diff <= bound).all()), (
        f"max excess over the derived bound {float((diff - bound).max()):.3e}; "
        f"max |diff| {float(diff.max()):.3e}, max bound {float(bound.max()):.3e}")
    return exact


@pytest.mark.parametrize(
    "batch, hidden, world, rank",
    [(1, TINY_HIDDEN, 1, 0), (4, TINY_HIDDEN, SERVED_WORLD, 37), (64, TINY_HIDDEN, 1, 0),
     (130, TINY_HIDDEN, 1, 0), (1, SERVED_HIDDEN, SERVED_WORLD, 63), (2, SERVED_HIDDEN, SERVED_WORLD, 0)],
    ids=["b1_h512_whole", "b4_h512_shard37of64", "b64_h512_whole", "b130_h512_two_tiles",
         "b1_h4096_shard63of64", "b2_h4096_shard0of64"],
)
def test_the_kernel_is_within_the_derived_bound_of_the_exact_tail(batch, hidden, world, rank) -> None:
    ops = _operands(5_501 + batch, batch, hidden, TINY_VOCAB, world, rank)
    out = _run(ops)
    _check(out, ops, hidden)
    # Control: the halves in the wrong order are far outside the bound.
    swapped, joined = exact_reference(ops["token_ids"], ops["table"], ops["positions"],
                                      ops["previous"], ops["enorm"], ops["hnorm"],
                                      ops["eh_proj_rows"], EPS, swap_halves=True)
    bound = gemv_bound(out, joined, ops["eh_proj_rows"], hidden)
    violating = float(((out.double() - swapped).abs() > bound).double().mean())
    assert violating > 0.9, f"only {violating:.0%} of the swapped mutant's elements leave the bound"


def test_the_whole_served_width_at_one_rank_matches_too() -> None:
    """``[4096, 8192]`` at one rank: eight 512-column chunks and a 64-block contraction."""
    ops = _operands(5_601, 1, SERVED_HIDDEN, TINY_VOCAB, 1, 0)
    _check(_run(ops), ops, SERVED_HIDDEN)


def test_position_zero_zeroes_the_embedding_half_and_nothing_else() -> None:
    """Rows at position 0 draft from the hidden half alone; a reference without the
    mask leaves the bound on exactly those rows."""
    ops = _operands(5_701, 6, TINY_HIDDEN, TINY_VOCAB, 1, 0, zero_positions=(0, 3))
    out = _run(ops)
    _check(out, ops, TINY_HIDDEN)
    unmasked, joined = exact_reference(ops["token_ids"], ops["table"], ops["positions"],
                                       ops["previous"], ops["enorm"], ops["hnorm"],
                                       ops["eh_proj_rows"], EPS, mask=False)
    bound = gemv_bound(out, joined, ops["eh_proj_rows"], TINY_HIDDEN)
    outside = ((out.double() - unmasked).abs() > bound).any(dim=-1)
    assert outside.tolist() == [True, False, False, True, False, False], outside.tolist()


def test_two_partition_tiles_equal_the_tiles_run_apart() -> None:
    ops = _operands(5_801, 130, TINY_HIDDEN, TINY_VOCAB, 1, 0)
    whole = _run(ops)
    first = {k: (v[:128] if k in ("token_ids", "positions", "previous") else v) for k, v in ops.items()}
    second = {k: (v[128:] if k in ("token_ids", "positions", "previous") else v) for k, v in ops.items()}
    apart = torch.cat([_run(first), _run(second)], dim=0)
    assert torch.equal(whole, apart), "a 130-row call must equal a 128-row and a 2-row call"


def test_the_torch_route_is_stage_a_arithmetic_bit_for_bit() -> None:
    """``mtp_tail_in_torch`` is the head's traced ``_layer_input`` expression, restricted
    to the shard rows: the CPU route and the reference the kernel is held to."""
    ops = _operands(5_901, 5, TINY_HIDDEN, TINY_VOCAB, 1, 0, zero_positions=(2,))
    got = tail_in.mtp_tail_in_torch(**ops, eps=EPS)

    def rms(x, gain):
        x32 = x.to(torch.float32)
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + EPS) * gain.to(torch.float32)).to(x.dtype)

    embeds = ops["table"][ops["token_ids"].to(torch.int64)]
    embeds = torch.where(ops["positions"].reshape(-1, 1) == 0, torch.zeros((), dtype=embeds.dtype), embeds)
    joined = torch.cat([rms(embeds, ops["enorm"]), rms(ops["previous"], ops["hnorm"])], dim=-1)
    want = torch.nn.functional.linear(joined, ops["eh_proj_rows"])
    assert got.dtype == torch.bfloat16 and torch.equal(got, want)


def test_the_torch_route_is_inside_the_bound_as_well() -> None:
    """The bound covers torch's own bf16 GEMV too (its summation order is also not the exact one)."""
    ops = _operands(6_001, 4, TINY_HIDDEN, TINY_VOCAB, 1, 0)
    got = tail_in.mtp_tail_in_torch(**ops, eps=EPS)
    exact, joined = exact_reference(ops["token_ids"], ops["table"], ops["positions"],
                                    ops["previous"], ops["enorm"], ops["hnorm"],
                                    ops["eh_proj_rows"], EPS)
    bound = gemv_bound(got, joined, ops["eh_proj_rows"], TINY_HIDDEN)
    assert bool(((got.double() - exact).abs() <= bound).all())


def test_shard_rows_divide_the_hidden_size_or_are_refused() -> None:
    assert tail_in.eh_proj_shard_rows(SERVED_HIDDEN, SERVED_WORLD) == 64
    assert tail_in.eh_proj_shard_rows(SERVED_HIDDEN, 1) == SERVED_HIDDEN
    with pytest.raises(tail_in.MtpTailError, match="divide"):
        tail_in.eh_proj_shard_rows(SERVED_HIDDEN, 48)
    with pytest.raises(tail_in.MtpTailError, match="positive"):
        tail_in.eh_proj_shard_rows(SERVED_HIDDEN, 0)


@pytest.mark.parametrize(
    "mutation, match",
    [
        (lambda o: o.update(previous=o["previous"][:, :256]), "previous"),
        (lambda o: o.update(eh_proj_rows=o["eh_proj_rows"][:, :1000]), "2H"),
        (lambda o: o.update(table=o["table"][:, :500]), "table"),
        (lambda o: o.update(positions=o["positions"][:2]), "positions"),
        (lambda o: o.update(eh_proj_rows=o["eh_proj_rows"][:0]), "row"),
        (lambda o: o.update(token_ids=o["token_ids"].to(torch.int64)), "int32"),
    ],
    ids=["previous_width", "eh_proj_columns", "table_width", "positions_length", "no_rows", "ids_dtype"],
)
def test_a_geometry_that_is_not_the_tails_is_refused_by_name(mutation, match) -> None:
    ops = _operands(6_101, 3, TINY_HIDDEN, TINY_VOCAB, 1, 0)
    mutation(ops)
    with pytest.raises(tail_in.MtpTailError, match=match):
        tail_in.mtp_tail_in(**ops, eps=EPS)


def test_a_hidden_size_that_is_not_a_multiple_of_the_partition_count_is_refused() -> None:
    ops = _operands(6_201, 2, 384 + 64, 16, 1, 0)
    with pytest.raises(tail_in.MtpTailError, match="128"):
        tail_in.mtp_tail_in(**ops, eps=EPS)
