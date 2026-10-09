# SPDX-License-Identifier: Apache-2.0
"""The sparse MLA kernel's decode-context-parallel partials, in the simulator.

Under DCP the CP ranks of a group interleave the latent cache in 128-token blocks (block
``b`` lives on rank ``b % CP``). Every rank attends with the gathered query of the whole
group, ``Hq = CP * H`` heads, over the selected columns it owns; a column it does not own
reaches it as ``-1``, the selector's sentinel. Per (head, row) it returns
(``mla_sparse.mla_sparse_attention_partial``):

* ``partial [Hq, S, L]`` float32, the attention normalised over its own valid columns;
* ``lse [Hq, S]`` float32, ``softmax_scale * m + ln(l)`` over the same columns;
* for a row with no valid column, ``partial`` exactly 0 and ``lse = dcp_merge.EMPTY_LSE``.

The all-to-all hands rank r the CP partials of its own H heads and ``dcp_merge`` merges
them. The tests here hold the partials to float64 attention and the merged partials to the
CP = 1 kernel, each within a bound derived from the body's arithmetic
(:func:`body_bounds`, :func:`merged_tolerance`), and pin the empty rows exactly. Each body
in partial mode is also built to a NEFF by neuronx-cc on the CPU, because the simulator never
runs the backend verifier (``test_neuronx_cc_builds_the_partial_entry``).

The dense-window prefill kernel has the same partial mode
(``mla_dense_window.mla_dense_window_attention_partial``): rank c's block table names its own
blocks in order, so a query's causal set is a prefix of that window, ``own_c(seq_len)`` rows
long, and 0 rows for a non-owner rank in a request's first block. Its tests are in the last
section and hold it to the same contract and the same derived bounds, with the term its MM2
adds (a bf16 hi/lo split of the weights).
"""

from __future__ import annotations

import math
import pathlib
import sys
from dataclasses import dataclass

import pytest
import torch

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.attention import dcp_merge as DM
from vllm_neuron.functional.attention import mla_dense_window as DW
from vllm_neuron.functional.attention import mla_sparse as MS
from test.vllm_neuron.functional.attention.test_dcp_merge import (EXP_U, TINY, U16, U32,
                                                                   compile_in_a_child,
                                                                   compile_to_neff, gamma,
                                                                   merge_bound)

LATENT = MS.TARGET_LATENT_RANK
#: Tokens per DCP interleave block: block b of the latent cache is rank (b % CP)'s.
DCP_BLOCK = 128
#: Error of the simulator's ``log`` (numpy float32), in U32 relative to the result: numpy
#: documents at most 3.83 ulp, 7.66 U32; measured 4.18 U32 on (0, 1e6]
#: (``reports/dcp_item4-logs/exp_log_ulp.txt``).
LOG_U = 8.0
#: The served selected-row width: 512 pools of 4 rows, the 3-row tail, whole chunks.
SERVED_TOPK = 17 * MS.KEY_CHUNK
#: Headroom for the products of first-order terms that the bounds below drop: every term
#: is below 1e-3, so their products are below 1 % of the sum.
SECOND_ORDER = 1.01
#: The partial entry's compile cases, ``(body, rows, Hq, latent, K)``: the served row-tiled
#: call (paged bf16 window of 17 pages, K = 2176) at Hq 2 and 8, and the untiled and the
#: latent-tiled bodies at the widths that select them.
PARTIAL_COMPILES = (("row_tiled", 256, 2, LATENT, SERVED_TOPK),
                    ("row_tiled", 256, 8, LATENT, SERVED_TOPK),
                    ("untiled", 64, 8, LATENT, MS.MOVING_MAX),
                    ("latent_tiled", 64, 8, 640, MS.MOVING_MAX))
COMPILE_PAGE = 128


# --------------------------------------------------------------------------- #
# Operands
# --------------------------------------------------------------------------- #
def make_operands(seq: int, hq: int, topk: int, s_kv: int, seed: int, latent: int = LATENT):
    """bf16 query and cache, and distinct selected rows per query.

    The query is scaled so the scaled scores spread over about ten units: the softmax is
    neither uniform nor one-hot, and the max shift and the tile merge both matter.
    """
    gen = torch.Generator().manual_seed(seed)
    q = (torch.randn(seq, hq, latent, generator=gen) * 2.0).to(torch.bfloat16)
    c = torch.randn(s_kv, latent, generator=gen).to(torch.bfloat16)
    idx = torch.stack([torch.randperm(s_kv, generator=gen)[:topk] for _ in range(seq)])
    return q, c, idx.to(torch.int32)


def owned(idx: torch.Tensor, cp: int, rank: int) -> torch.Tensor:
    """``idx`` with every column rank ``rank`` does not own replaced by the sentinel."""
    mine = (torch.div(idx, DCP_BLOCK, rounding_mode="floor") % cp == rank) & (idx >= 0)
    return torch.where(mine, idx, torch.full_like(idx, MS.SENTINEL_INDEX))


def rows_of_rank(s_kv: int, cp: int, rank: int) -> torch.Tensor:
    """The cache rows rank ``rank`` owns."""
    rows = torch.arange(s_kv)
    return rows[torch.div(rows, DCP_BLOCK, rounding_mode="floor") % cp == rank]


def scale_of(latent: int) -> float:
    return float(latent) ** -0.5


# --------------------------------------------------------------------------- #
# float64 reference and the derived bounds
# --------------------------------------------------------------------------- #
def partial_reference(q, c, idx, scale):
    """float64 ``(partial [Hq, S, L], lse [Hq, S])`` over each row's valid columns."""
    qd, cd = q.double(), c.double()
    seq, hq, latent = qd.shape
    partial = torch.zeros(hq, seq, latent, dtype=torch.float64)
    lse = torch.full((hq, seq), DM.EMPTY_LSE, dtype=torch.float64)
    for s in range(seq):
        keep = idx[s] >= 0
        if not bool(keep.any()):
            continue
        rows = cd[idx[s][keep].long()]                              # [k, L]
        x = torch.einsum("hl,kl->hk", qd[s], rows) * scale          # [Hq, k]
        lse[:, s] = torch.logsumexp(x, dim=1)
        partial[:, s] = torch.einsum("hk,kl->hl", torch.exp(x - lse[:, s:s + 1]), rows)
    return partial, lse


@dataclass
class BodyBounds:
    """:func:`body_bounds`' result: ``|partial - P*| <= eps * s_abs + floor`` element-wise and
    ``|lse - L*| <= eta``."""

    eps: torch.Tensor        # [Hq, S], relative to s_abs
    eta: torch.Tensor        # [Hq, S], absolute
    s_abs: torch.Tensor      # [Hq, S, L], sum_k a_k |c_k| with a the exact weights
    floor: torch.Tensor      # [Hq, S, L], absolute: the weights that underflow

    def partial(self) -> torch.Tensor:
        return self.eps.unsqueeze(-1) * self.s_abs + self.floor


def body_bounds(q, c, idx, scale, tiles: int, split: bool = False) -> BodyBounds:
    """Per (head, row) bounds on the fp32 body's partial and lse, from its arithmetic.

    ``tiles`` is the body's score-tile count (1 below 512 columns).

    The steps, with u the fp32 unit roundoff, K the selected-row width:

    * MM1 sums L exact bf16 products: each scaled score is off by at most ``zeta = scale *
      gamma(L) * max_k sum_l |q_l c_kl|``. The softmax and the lse move by at most that.
    * The exp's argument ``scale * s + bias`` is two roundings of values at most ``lam =
      scale * max |s|`` and ``A = scale * (max s - min s)``; the exp is ``EXP_U`` u. The bias
      is common to every column and cancels in the weights and in ``lse = ln(sum) - bias``.
      Each weight's exp is off by ``theta = zeta + u (lam + A) + EXP_U u``.
    * Each of the ``tiles - 1`` merges rescales the running sum and accumulator by one exp
      of a difference of tile maxima (``u A`` and ``EXP_U u``) and one multiply and add:
      ``rho = (tiles - 1) (EXP_U + A + 2) u``.
    * A relative error ``t`` of every weight moves the normalised weights by ``2t / (1 -
      t)``; the softmax sum and MM2's sum are ``gamma(K)`` each, the reciprocal and the
      normalising multiply ``u`` each.
    * The lse: ``ln`` of a sum off by ``theta' + gamma(K) + rho`` relative (``theta'`` is
      ``theta`` without ``zeta``, which moves the lse by ``zeta`` directly), the log's own
      ``LOG_U`` u of ``|ln l|``, and the final add's u of ``|lse|``.
    * A weight whose exp underflows is off by less than TINY absolute, against a maximum
      weight of 1 and a sum of at least 1: ``floor = TINY * sum_k |c_k|`` on the partial and
      ``k * TINY`` on the lse (k valid columns). Every relative term above is for normal
      results.
    * ``split`` (the dense-window body): MM2 contracts a bf16 hi/lo split of the normalised
      weights over ``2K`` exact products. ``p_hi`` is p rounded to bf16 (off by at most
      ``U16 p``), ``p - p_hi`` is exact in fp32 (Sterbenz), and ``p_lo`` is it rounded to
      bf16, so ``p_hi + p_lo = p (1 + d)`` with ``|d| <= U16**2 = 2^-16``. MM2's ``gamma(K)``
      becomes ``gamma(2K) + 2^-16``.

    A row with no valid column gets 0 everywhere: its partial and lse are exact.
    """
    qd, cd = q.double(), c.double()
    seq, hq, latent = qd.shape
    topk = int(idx.shape[1])
    eps = torch.zeros(hq, seq, dtype=torch.float64)
    eta = torch.zeros(hq, seq, dtype=torch.float64)
    s_abs = torch.zeros(hq, seq, latent, dtype=torch.float64)
    floor = torch.zeros(hq, seq, latent, dtype=torch.float64)
    for s in range(seq):
        keep = idx[s] >= 0
        if not bool(keep.any()):
            continue
        rows = cd[idx[s][keep].long()]
        raw = torch.einsum("hl,kl->hk", qd[s], rows)
        m1 = torch.einsum("hl,kl->hk", qd[s].abs(), rows.abs())
        x = raw * scale
        zeta = scale * gamma(latent) * m1.max(dim=1).values
        lam = x.abs().max(dim=1).values
        spread = x.max(dim=1).values - x.min(dim=1).values
        theta_exp = U32 * (lam + spread) + EXP_U * U32
        theta = zeta + theta_exp
        rho = (tiles - 1) * (EXP_U + spread + 2.0) * U32
        t = theta + rho
        mm2 = gamma(2 * topk) + U16 * U16 if split else gamma(topk)
        eps[:, s] = SECOND_ORDER * (2 * t / (1 - t) + gamma(topk) + mm2 + 2 * U32)
        lse = torch.logsumexp(x, dim=1)
        ln_l = lse - x.max(dim=1).values
        rel = theta_exp + gamma(topk) + rho
        eta[:, s] = SECOND_ORDER * (zeta + rel / (1 - rel) + LOG_U * U32 * ln_l.abs()
                                    + U32 * lse.abs()) + rows.shape[0] * TINY
        weights = torch.exp(x - lse.unsqueeze(1))
        s_abs[:, s] = torch.einsum("hk,kl->hl", weights, rows.abs())
        floor[:, s] = TINY * rows.abs().sum(dim=0)
    return BodyBounds(eps, eta, s_abs, floor)


def tiles_of(topk: int) -> int:
    """The body's score tiles: one below a moving tile, else whole 512-column tiles."""
    return 1 if topk <= MS.MOVING_MAX else len(MS._score_tiles(topk, MS.MOVING_MAX))


def merge64(partials: torch.Tensor, lses: torch.Tensor) -> torch.Tensor:
    """float64 llama3 merge of ``[CP, Hq, S, L]`` partials and ``[CP, Hq, S]`` lses, head-major."""
    p, l = partials.double(), lses.double()
    return (p * torch.exp(l - torch.logsumexp(l, dim=0)).unsqueeze(-1)).sum(dim=0)


def merged_tolerance(full_bounds, rank_bounds, live) -> torch.Tensor:
    """``[Hq, S, L]`` bound on |merge of the CP kernel partials - the CP = 1 kernel output|.

    With ``|P_c - P*_c| <= eps_c S_c``, ``|L_c - L*_c| <= eta_c`` and the exact identity
    ``sum_c w*_c P*_c = y*`` (``w*_c = exp(L*_c - logsumexp L*)``, every valid column on
    exactly one rank): each weight is off by at most ``exp(2 eta') - 1`` relative, eta' the
    largest live rank's, ``sum_c w*_c S_c = S`` and ``sum_c w*_c |P*_c| <= S``. So the merge
    is within ``(eps' exp(2 eta') + exp(2 eta') - 1) S`` of y*, eps' the largest live rank's,
    and the CP = 1 output within ``eps S``. The underflow floors add as they are, a rank's
    scaled by its weight (at most ``exp(2 eta')``). ``live`` [CP, Hq, S] marks the ranks that
    own a column of the row; an empty rank's weight is exactly 0 in both.
    """
    eps_r = torch.stack([b.eps for b in rank_bounds])
    eta_r = torch.stack([b.eta for b in rank_bounds])
    floor_r = torch.stack([b.floor for b in rank_bounds])
    zero = torch.zeros_like(eps_r)
    eps_p = torch.where(live, eps_r, zero).max(dim=0).values
    eta_p = torch.where(live, eta_r, zero).max(dim=0).values
    floor_p = torch.where(live.unsqueeze(-1), floor_r, torch.zeros_like(floor_r)).max(dim=0).values
    grow = torch.exp(2 * eta_p)
    per_row = SECOND_ORDER * (full_bounds.eps + eps_p * grow + (grow - 1))
    return (per_row.unsqueeze(-1) * full_bounds.s_abs + full_bounds.floor
            + grow.unsqueeze(-1) * floor_p)


# --------------------------------------------------------------------------- #
# Calls
# --------------------------------------------------------------------------- #
def run_ranks(q, c, idx, cp, scale):
    """Every rank's ``(partial, lse)`` and the [CP, Hq, S] live mask."""
    partials, lses, live = [], [], []
    for rank in range(cp):
        mine = owned(idx, cp, rank)
        partial, lse = MS.mla_sparse_attention_partial(q, c, mine, scale)
        partials.append(partial)
        lses.append(lse)
        live.append((mine >= 0).any(dim=1).unsqueeze(0).expand(q.shape[1], -1))
    return torch.stack(partials), torch.stack(lses), torch.stack(live)


def assert_partial_is_float64_attention(partial, lse, q, c, idx, scale, tiles):
    want_p, want_l = partial_reference(q, c, idx, scale)
    bounds = body_bounds(q, c, idx, scale, tiles)
    eta = bounds.eta
    assert partial.dtype == torch.float32 and lse.dtype == torch.float32
    assert tuple(partial.shape) == (q.shape[1], q.shape[0], q.shape[2])
    assert tuple(lse.shape) == (q.shape[1], q.shape[0])
    err_p = (partial.double() - want_p).abs()
    bound_p = bounds.partial()
    assert bool((err_p <= bound_p).all()), \
        f"partial: max err/bound {float((err_p / bound_p.clamp_min(1e-300)).max()):.3g}"
    empty = ~(idx >= 0).any(dim=1)                                  # [S]
    live_l = lse[:, ~empty].double()
    err_l = (live_l - want_l[:, ~empty]).abs()
    assert bool((err_l <= eta[:, ~empty]).all()), \
        f"lse: max err/bound {float((err_l / eta[:, ~empty].clamp_min(1e-300)).max()):.3g}"
    assert_empty_rows_are_exact(partial, lse, empty)


def assert_empty_rows_are_exact(partial, lse, empty):
    """A row with no valid column: partial exactly 0, lse exactly EMPTY_LSE."""
    if bool(empty.any()):
        assert torch.count_nonzero(partial[:, empty]) == 0
        assert bool((lse[:, empty] == torch.tensor(DM.EMPTY_LSE, dtype=torch.float32)).all())


# --------------------------------------------------------------------------- #
# Geometry cases: (seq, Hq, topk, s_kv, latent), one per body
# --------------------------------------------------------------------------- #
UNTILED = dict(seq=4, topk=256, s_kv=2048, latent=LATENT)
ROW_TILED = dict(seq=4, topk=640, s_kv=2048, latent=LATENT)
SERVED = dict(seq=2, topk=SERVED_TOPK, s_kv=4096, latent=LATENT)
LATENT_TILED = dict(seq=4, topk=128, s_kv=1024, latent=131)
BODIES = {"untiled": UNTILED, "row_tiled": ROW_TILED, "served": SERVED,
          "latent_tiled": LATENT_TILED}


def operands_for(body: str, hq: int, seed: int, s_kv: int | None = None):
    g = BODIES[body]
    return make_operands(g["seq"], hq, g["topk"], s_kv or g["s_kv"], seed, latent=g["latent"])


# --------------------------------------------------------------------------- #
# The partial contract
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("hq", (1, 2, 4, 8))
@pytest.mark.parametrize("body", tuple(BODIES))
def test_one_rank_partial_is_float64_attention_within_the_derived_bound(body, hq):
    """One rank's partial and lse, against float64 attention over its own columns."""
    q, c, idx = operands_for(body, hq, seed=10 * hq + len(body))
    cp = 4
    mine = owned(idx, cp, 1)
    scale = scale_of(q.shape[2])
    partial, lse = MS.mla_sparse_attention_partial(q, c, mine, scale)
    assert_partial_is_float64_attention(partial, lse, q, c, mine, scale, tiles_of(idx.shape[1]))


@pytest.mark.parametrize("hq", (1, 2, 8))
@pytest.mark.parametrize("body", tuple(BODIES))
def test_with_no_owner_mask_the_partial_is_the_attention_output_bit_for_bit(body, hq,
                                                                           monkeypatch):
    """CP = 1: the partial is the existing seam's fp32 output, transposed, every bit.

    The fp32 arithmetic is shared, so only the store's layout differs. The row-tiled seam is
    held to its fp32 path (the kill switch); the low-precision one-head body is not served
    in partial mode (``reports/dcp_item4.md``).
    """
    monkeypatch.setenv(MS.FP32_ENV, "1")
    q, c, idx = operands_for(body, hq, seed=20 * hq + len(body))
    scale = scale_of(q.shape[2])
    partial, lse = MS.mla_sparse_attention_partial(q, c, idx, scale)
    full = MS.mla_sparse_attention(q, c, idx, scale)
    torch.testing.assert_close(partial, full.permute(1, 0, 2), rtol=0.0, atol=0.0)
    _want_p, want_l = partial_reference(q, c, idx, scale)
    eta = body_bounds(q, c, idx, scale, tiles_of(idx.shape[1])).eta
    assert bool(((lse.double() - want_l).abs() <= eta).all())


@pytest.mark.parametrize("cp,heads", ((2, 1), (4, 1), (8, 1), (2, 2), (4, 2)))
@pytest.mark.parametrize("body", ("untiled", "row_tiled", "served"))
def test_merged_partials_are_the_cp1_kernel_within_the_derived_merge_tolerance(body, cp,
                                                                              heads):
    """The float64 merge of the CP ranks' partials against the CP = 1 kernel's output."""
    hq = cp * heads
    q, c, idx = operands_for(body, hq, seed=100 * cp + heads)
    scale = scale_of(q.shape[2])
    tiles = tiles_of(idx.shape[1])
    partials, lses, live = run_ranks(q, c, idx, cp, scale)
    full = MS.mla_sparse_attention_partial(q, c, idx, scale)[0]       # [Hq, S, L]
    got = merge64(partials, lses)
    full_bounds = body_bounds(q, c, idx, scale, tiles)
    rank_bounds = [body_bounds(q, c, owned(idx, cp, r), scale, tiles) for r in range(cp)]
    tol = merged_tolerance(full_bounds, rank_bounds, live)
    err = (got - full.double()).abs()
    assert bool((err <= tol).all()), f"max err/tol {float((err / tol.clamp_min(1e-300)).max()):.3g}"


def test_the_merge_tolerance_sees_a_wrong_lse_and_a_dropped_rank():
    """The tolerance is not vacuous: an lse off by ln 2 on one rank breaks it, as does a
    dropped rank."""
    cp, hq = 4, 4
    q, c, idx = operands_for("row_tiled", hq, seed=7)
    scale = scale_of(q.shape[2])
    tiles = tiles_of(idx.shape[1])
    partials, lses, live = run_ranks(q, c, idx, cp, scale)
    full = MS.mla_sparse_attention_partial(q, c, idx, scale)[0].double()
    tol = merged_tolerance(body_bounds(q, c, idx, scale, tiles),
                           [body_bounds(q, c, owned(idx, cp, r), scale, tiles) for r in range(cp)],
                           live)
    assert bool(((merge64(partials, lses) - full).abs() <= tol).all())
    shifted = lses.clone()
    shifted[1] += math.log(2.0)
    assert bool(((merge64(partials, shifted) - full).abs() > tol).any())
    dropped = lses.clone()
    dropped[2] = DM.EMPTY_LSE
    assert bool(((merge64(partials, dropped) - full).abs() > tol).any())


# --------------------------------------------------------------------------- #
# Empty rows
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("body", ("untiled", "row_tiled", "served", "latent_tiled"))
def test_a_rank_that_owns_no_column_of_a_row_returns_the_empty_partial(body):
    """Row 1 selects only rank 0's rows: ranks 1..3 return (0, EMPTY_LSE) for it, exactly."""
    cp, hq = 4, 4
    g = BODIES[body]
    # Enough cache that rank 0 alone owns topk distinct rows.
    s_kv = max(g["s_kv"], -(-cp * g["topk"] // (cp * DCP_BLOCK)) * cp * DCP_BLOCK)
    q, c, idx = operands_for(body, hq, seed=31, s_kv=s_kv)
    gen = torch.Generator().manual_seed(32)
    rank0 = rows_of_rank(s_kv, cp, 0)
    idx[1] = rank0[torch.randperm(rank0.numel(), generator=gen)[:g["topk"]]].to(torch.int32)
    scale = scale_of(q.shape[2])
    for rank in range(cp):
        mine = owned(idx, cp, rank)
        partial, lse = MS.mla_sparse_attention_partial(q, c, mine, scale)
        assert_partial_is_float64_attention(partial, lse, q, c, mine, scale,
                                            tiles_of(idx.shape[1]))
        if rank > 0:
            assert torch.count_nonzero(partial[:, 1]) == 0
            assert bool((lse[:, 1] == torch.tensor(DM.EMPTY_LSE)).all())
        else:
            assert bool((lse[:, 1] > DM.EMPTY_LSE / 2).all())


@pytest.mark.parametrize("body", ("untiled", "row_tiled", "served", "latent_tiled"))
def test_an_all_sentinel_row_is_empty_on_every_rank_and_merges_to_exact_zero(body):
    """A row of -1 (the selector's empty row) is (0, EMPTY_LSE) on every rank; merged, 0."""
    cp, heads = 2, 2
    q, c, idx = operands_for(body, cp * heads, seed=41)
    row = q.shape[0] - 1
    idx[row] = MS.SENTINEL_INDEX
    scale = scale_of(q.shape[2])
    partials, lses, _live = run_ranks(q, c, idx, cp, scale)
    for rank in range(cp):
        assert torch.count_nonzero(partials[rank, :, row]) == 0
        assert bool((lses[rank, :, row] == torch.tensor(DM.EMPTY_LSE)).all())
    mine = partials[:, 0:heads].contiguous()                      # rank 0's heads, every source
    merged = DM.dcp_lse_merge(mine, lses[:, 0:heads].contiguous())
    assert not torch.isnan(merged.float()).any()
    assert torch.count_nonzero(merged[row]) == 0


def test_the_old_row_sum_floor_does_not_leak_into_an_empty_lse():
    """Before this change a wholly-sentinel row's sum was at least 1, so ln(sum) - bias was
    a finite, plausible lse. The partial must report the row empty, not that number."""
    q, c, idx = operands_for("untiled", 2, seed=51)
    idx[:] = MS.SENTINEL_INDEX
    partial, lse = MS.mla_sparse_attention_partial(q, c, idx, scale_of(LATENT))
    assert torch.count_nonzero(partial) == 0
    assert bool((lse == torch.tensor(DM.EMPTY_LSE)).all())


# --------------------------------------------------------------------------- #
# End to end through the NKI merge
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cp,heads", ((2, 1), (8, 1), (4, 2)))
def test_dcp_through_the_merge_kernel_is_the_cp1_output_rounded_once(cp, heads):
    """Every rank's merged bf16 heads against bf16 of the CP = 1 kernel, at the served K.

    ``|merge_kernel - bf16(Y1)| <= merge_bound(partials) + tau + U16 |Y1|``: the merge
    kernel's own bound around the exact merge of the kernel partials, the merge tolerance
    from there to the CP = 1 output Y1, and Y1's own bf16 rounding.
    """
    hq = cp * heads
    q, c, idx = operands_for("served", hq, seed=60 + cp)
    idx[1, : SERVED_TOPK // 2] = MS.SENTINEL_INDEX                  # a half-empty row
    scale = scale_of(LATENT)
    tiles = tiles_of(SERVED_TOPK)
    partials, lses, live = run_ranks(q, c, idx, cp, scale)
    full = MS.mla_sparse_attention_partial(q, c, idx, scale)[0]       # [Hq, S, L]
    tol = merged_tolerance(body_bounds(q, c, idx, scale, tiles),
                           [body_bounds(q, c, owned(idx, cp, r), scale, tiles) for r in range(cp)],
                           live)
    for rank in range(cp):
        heads_of_rank = slice(rank * heads, (rank + 1) * heads)
        mine_p = partials[:, heads_of_rank].contiguous()             # [CP, H, S, L]
        mine_l = lses[:, heads_of_rank].contiguous()
        got = DM.dcp_lse_merge(mine_p, mine_l)                        # [S, H, L] bf16
        y1 = full[heads_of_rank].permute(1, 0, 2).double()            # [S, H, L]
        bound = (merge_bound(mine_p, mine_l) + tol[heads_of_rank].permute(1, 0, 2)
                 + U16 * y1.abs())
        err = (got.double() - y1.to(torch.bfloat16).double()).abs()
        assert bool((err <= bound).all()), \
            f"rank {rank}: max err/bound {float((err / bound.clamp_min(1e-300)).max()):.3g}"


# --------------------------------------------------------------------------- #
# Paged operands, the launch grid and the refusals
# --------------------------------------------------------------------------- #
def test_a_paged_partial_is_the_unpaged_partial_of_its_window_bit_for_bit():
    """The served call is paged: the partial of a block-table window is that of the window."""
    page, pages, cp, hq = 128, 16, 2, 4
    q, bank, _ = make_operands(4, hq, SERVED_TOPK, 32 * page, seed=71)
    gen = torch.Generator().manual_seed(72)
    table = torch.randperm(32, generator=gen)[:pages].to(torch.int32).unsqueeze(1)
    window = bank.view(32, page, LATENT)[table[:, 0].long()].reshape(pages * page, LATENT)
    idx = torch.stack([torch.randperm(pages * page, generator=gen)[:SERVED_TOPK]
                       for _ in range(4)]).to(torch.int32)
    mine = owned(idx, cp, 1)
    scale = scale_of(LATENT)
    paged = MS.mla_sparse_attention_partial(q, bank, mine, scale, block_table_row=table,
                                            page_size=page)
    flat = MS.mla_sparse_attention_partial(q, window.contiguous(), mine, scale)
    torch.testing.assert_close(paged[0], flat[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(paged[1], flat[1], rtol=0.0, atol=0.0)


class RecordingCall:
    """Stands in for ``wrap_nki(entry)``: records the grid it was indexed with and the call."""

    def __init__(self, entry):
        self.entry = entry
        self.grid = None
        self.inner = wrap_nki(entry)

    def __getitem__(self, grid):
        self.grid = grid
        self.inner = self.inner[grid]
        return self

    def __call__(self, *args):
        return self.inner(*args)


@pytest.mark.parametrize("hq,seq", ((2, 32), (8, 32), (2, 24)))
def test_two_programs_are_bitwise_one_program_in_partial_mode(hq, seq, monkeypatch):
    """At LNC2 the partial's row-tiled launch splits the query blocks over both cores.

    The one-head rule of the DCP-off seam's grid does not apply here: Hq is CP * H >= 2, and
    the heads ride the partitions. (The DCP-off seam keeps it: ``test_mla_sparse_spmd.py``
    pins a two-head call to one program.) 24 queries at Hq 2 are three query blocks: the
    first program takes two and the second one, an uneven split fixed at trace time
    (``reports/dcp_item4.md`` section 14).
    """
    q, c, idx = make_operands(seq, hq, SERVED_TOPK, 4096, seed=48 + hq + seq)
    mine = owned(idx, 2, 0)
    scale = scale_of(LATENT)
    monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    assert MS.partial_programs(seq, hq, SERVED_TOPK) == 1
    one = MS.mla_sparse_attention_partial(q, c, mine, scale)
    calls = []

    def record(entry):
        calls.append(RecordingCall(entry))
        return calls[-1]

    monkeypatch.setattr(MS, "wrap_nki", record)
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    assert MS.partial_programs(seq, hq, SERVED_TOPK) == 2
    two = MS.mla_sparse_attention_partial(q, c, mine, scale)
    assert [call.grid for call in calls] == [2]
    torch.testing.assert_close(two[0], one[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(two[1], one[1], rtol=0.0, atol=0.0)
    # One query block has nothing to split; an untiled call has one program.
    assert MS.partial_programs(16 // hq, hq, SERVED_TOPK) == 1
    assert MS.partial_programs(32, hq, 256) == 1


def test_the_partial_seam_refuses_a_rope_limb_by_name():
    """Partial mode serves this checkpoint's NoPE geometry only (``qk_rope_head_dim`` 0)."""
    q, c, idx = make_operands(2, 2, 256, 1024, seed=99)
    with pytest.raises(MS.MlaSparseAttentionError, match="RoPE"):
        MS.mla_sparse_attention_partial(q, c, idx, scale_of(LATENT), q_pe=torch.zeros(2, 2, 8),
                                        k_pe=torch.zeros(1024, 8))


@pytest.mark.parametrize("named", ("q_lift", "multiple", "heads"))
def test_the_partial_seam_refuses_a_malformed_call_by_name(named):
    """The partial seam runs the attention seam's own checks."""
    hq = 129 if named == "heads" else 2
    q, c, idx = make_operands(2, hq, 256, 1024, seed=98)
    if named == "q_lift":
        q = q[0]
    if named == "multiple":
        idx = idx[:, :200].contiguous()
    with pytest.raises(MS.MlaSparseAttentionError, match=named):
        MS.mla_sparse_attention_partial(q, c, idx, scale_of(LATENT))


def test_the_partial_entry_is_reported_with_the_module_s_kernels():
    names = {qualname for _module, qualname in MS.mla_sparse_kernel_identity()}
    assert "mla_sparse_attention_nope_partial_kernel" in names
    assert MS.mla_sparse_attention_nope_partial_kernel.func.__module__ == MS.__name__


# --------------------------------------------------------------------------- #
# The dense window in partial mode
# --------------------------------------------------------------------------- #
#: Bank pages of the dense cases: more than any window, so a window is a scattered subset.
DENSE_BANK_PAGES = 48
#: The dense partial entry's compile cases, ``(queries, Hq, window pages, active queries)``
#: at page 128 and latent 512, one head per rank (TP 64), so CP = Hq: a 1024-query chunk at
#: CP 2 over rank 0's 9 of the 17 blocks the identity bound of 2051 tokens spans; at CP 8
#: over its 3, padded to 1000 active; and at CP 2 over the widest window the kernel stages,
#: 20 pages. The overlay is the rank's share of the chunk's rows, ``queries / CP``.
DENSE_COMPILES = ((1024, 2, 9, 1024), (1024, 8, 3, 1000), (1024, 2, 20, 1000))


def dense_case(seq: int, start: int, blocks: int, hq: int, seed: int):
    """``(q [seq, hq, L], bank, table [blocks, 1], lens [seq])`` for one request's chunk.

    The chunk's queries sit at tokens ``start .. start + seq - 1``, so query s attends the
    first ``start + s + 1`` tokens. The request's blocks are random pages of the bank. The
    query is scaled as :func:`make_operands` scales it.
    """
    gen = torch.Generator().manual_seed(seed)
    q = (torch.randn(seq, hq, LATENT, generator=gen) * 2.0).to(torch.bfloat16)
    bank = torch.randn(DENSE_BANK_PAGES * DCP_BLOCK, LATENT, generator=gen).to(torch.bfloat16)
    table = torch.randperm(DENSE_BANK_PAGES, generator=gen)[:blocks].to(torch.int32)
    lens = torch.arange(seq, dtype=torch.int32) + start + 1
    return q, bank, table.unsqueeze(1), lens


def rank_table(table: torch.Tensor, cp: int, rank: int) -> torch.Tensor:
    """Rank ``rank``'s block table: its own blocks in order; one ``-1`` pad if it has none."""
    mine = table[torch.arange(table.shape[0]) % cp == rank]
    return mine if mine.shape[0] else torch.full((1, 1), -1, dtype=torch.int32)


def own_lens(lens: torch.Tensor, cp: int, rank: int, blocks: int) -> torch.Tensor:
    """``own_c(n) = sum over blocks b, b % CP == c, of min(128, max(0, n - 128 b))``."""
    b = torch.arange(blocks)
    b = b[b % cp == rank]
    own = (lens.long().unsqueeze(1) - DCP_BLOCK * b).clamp(0, DCP_BLOCK).sum(dim=1)
    return own.to(torch.int32)


def window_of(bank: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """The rows a block table names, in order (a ``-1`` pad reads as page 0: never attended)."""
    pages = table[:, 0].long().clamp_min(0)
    return bank.view(-1, DCP_BLOCK, LATENT)[pages].reshape(-1, LATENT)


def dense_columns(lens: torch.Tensor, width: int) -> torch.Tensor:
    """The dense body's columns as an index matrix: row s keeps ``0 .. lens[s] - 1``, then -1.

    ``width`` is the staged window: the body's softmax sums and MM2 contracts all of it (the
    masked columns as exact zeros), so it is the K of :func:`body_bounds`.
    """
    cols = torch.arange(width).unsqueeze(0)
    keep = cols < lens.long().unsqueeze(1)
    return torch.where(keep, cols, torch.full_like(cols, MS.SENTINEL_INDEX)).to(torch.int32)


def dense_bounds(q, window, lens) -> BodyBounds:
    return body_bounds(q, window, dense_columns(lens, window.shape[0]), scale_of(LATENT), 1,
                       split=True)


def dense_ranks(q, bank, table, lens, cp):
    """Every rank's dense ``(partial, lse)``, its window and owned lengths, and [CP, Hq, S] live."""
    partials, lses, windows, owns, live = [], [], [], [], []
    for rank in range(cp):
        mine = rank_table(table, cp, rank)
        own = own_lens(lens, cp, rank, table.shape[0])
        partial, lse = DW.mla_dense_window_attention_partial(q, bank, own, scale_of(LATENT),
                                                             mine, page_size=DCP_BLOCK)
        partials.append(partial)
        lses.append(lse)
        windows.append(window_of(bank, mine))
        owns.append(own)
        live.append((own > 0).unsqueeze(0).expand(q.shape[1], -1))
    return torch.stack(partials), torch.stack(lses), windows, owns, torch.stack(live)


@pytest.mark.parametrize("hq", (1, 2, 8))
def test_a_dense_window_partial_is_float64_attention_within_the_derived_bound(hq):
    """Rank 1 of CP 2 over its own blocks (1 and 3): a 200-query chunk at tokens 300 .. 499."""
    q, bank, table, lens = dense_case(200, 300, 4, hq, seed=110 + hq)
    mine, own = rank_table(table, 2, 1), own_lens(lens, 2, 1, 4)
    assert int(own.min()) == 128 and int(own.max()) == 244
    partial, lse = DW.mla_dense_window_attention_partial(q, bank, own, scale_of(LATENT), mine,
                                                         page_size=DCP_BLOCK)
    window = window_of(bank, mine)
    assert partial.dtype == torch.float32 and tuple(partial.shape) == (hq, 200, LATENT)
    assert lse.dtype == torch.float32 and tuple(lse.shape) == (hq, 200)
    want_p, want_l = partial_reference(q, window, dense_columns(own, window.shape[0]),
                                       scale_of(LATENT))
    bounds = dense_bounds(q, window, own)
    err_p = (partial.double() - want_p).abs()
    assert bool((err_p <= bounds.partial()).all()), \
        f"partial: max err/bound {float((err_p / bounds.partial()).max()):.3g}"
    err_l = (lse.double() - want_l).abs()
    assert bool((err_l <= bounds.eta).all()), \
        f"lse: max err/bound {float((err_l / bounds.eta).max()):.3g}"


@pytest.mark.parametrize("hq", (1, 2, 8))
def test_with_whole_lengths_the_dense_window_partial_is_the_dcp_off_output_bit_for_bit(hq):
    """CP = 1: the partial is the dense seam's output, transposed, every bit, with this step's
    rows overlaid and a padded chunk (the rows from ``active_rows`` on are 0 and EMPTY_LSE).

    The per-row arithmetic is shared; only the query DMA (one head's rows ``Hq * L`` apart)
    and the head-major store differ.
    """
    seq, active, start, blocks = 144, 140, 200, 3
    q, bank, table, lens = dense_case(seq, start, blocks, hq, seed=120 + hq)
    gen = torch.Generator().manual_seed(121)
    written = torch.randn(seq, LATENT, generator=gen).to(torch.bfloat16)
    offset = torch.tensor([[start]], dtype=torch.int32)
    call = dict(written=written, write_offset=offset, page_size=DCP_BLOCK, active_rows=active)
    partial, lse = DW.mla_dense_window_attention_partial(q, bank, lens, scale_of(LATENT), table,
                                                         **call)
    full = DW.mla_dense_window_attention(q, bank, lens, scale_of(LATENT), table, **call)
    torch.testing.assert_close(partial.permute(1, 0, 2), full, rtol=0.0, atol=0.0)
    assert torch.count_nonzero(partial[:, active:]) == 0
    assert bool((lse[:, active:] == torch.tensor(DM.EMPTY_LSE)).all())
    window = window_of(bank, table).clone()
    window[start:start + seq] = written
    cols = dense_columns(lens[:active], window.shape[0])
    _want_p, want_l = partial_reference(q[:active], window, cols, scale_of(LATENT))
    eta = body_bounds(q[:active], window, cols, scale_of(LATENT), 1, split=True).eta
    assert bool(((lse[:, :active].double() - want_l).abs() <= eta).all())


@pytest.mark.parametrize("cp,heads", ((2, 1), (4, 1), (4, 2), (8, 1)))
def test_merged_dense_window_partials_are_the_cp1_output_within_the_derived_merge_tolerance(
        cp, heads):
    """The float64 merge of the CP ranks' dense partials against the CP = 1 dense partial.

    A 150-query chunk at tokens 600 .. 749 over 6 blocks: at CP 8 ranks 6 and 7 own no block
    (a ``-1`` table, every length 0), and rank 5's block starts at token 640, so it is empty
    for the queries before it.
    """
    hq = cp * heads
    q, bank, table, lens = dense_case(150, 600, 6, hq, seed=130 + 10 * cp + heads)
    partials, lses, windows, owns, live = dense_ranks(q, bank, table, lens, cp)
    full = DW.mla_dense_window_attention_partial(q, bank, lens, scale_of(LATENT), table,
                                                 page_size=DCP_BLOCK)[0]
    for rank in range(cp):
        assert_empty_rows_are_exact(partials[rank], lses[rank], owns[rank] == 0)
    tol = merged_tolerance(dense_bounds(q, window_of(bank, table), lens),
                           [dense_bounds(q, w, n) for w, n in zip(windows, owns)], live)
    err = (merge64(partials, lses) - full.double()).abs()
    assert bool((err <= tol).all()), f"max err/tol {float((err / tol).max()):.3g}"


def test_a_non_owner_rank_s_dense_window_partial_of_a_first_block_query_is_empty():
    """Every query of a request's first block: ranks 1 .. 3 of CP 4 hold none of its tokens.

    Their length is 0, which the partial seam serves (the attention seam still refuses it by
    name), and they return 0 and EMPTY_LSE exactly. Merged by the NKI merge, rank 0's one head
    is its own partial rounded once: its weight is exp(0) / 1 and the others' exactly 0.
    """
    cp = 4
    q, bank, table, lens = dense_case(100, 0, 1, cp, seed=140)
    partials, lses, _windows, owns, _live = dense_ranks(q, bank, table, lens, cp)
    assert torch.equal(owns[0], lens) and all(int(n.max()) == 0 for n in owns[1:])
    for rank in range(1, cp):
        assert torch.count_nonzero(partials[rank]) == 0
        assert bool((lses[rank] == torch.tensor(DM.EMPTY_LSE)).all())
    merged = DM.dcp_lse_merge(partials[:, 0:1].contiguous(), lses[:, 0:1].contiguous())
    torch.testing.assert_close(merged, partials[0, 0:1].permute(1, 0, 2).to(torch.bfloat16),
                               rtol=0.0, atol=0.0)
    with pytest.raises(DW.MlaDenseWindowError, match="seq_lens"):
        DW.mla_dense_window_attention(q, bank, owns[1], scale_of(LATENT), rank_table(table, cp, 1),
                                      page_size=DCP_BLOCK)


@pytest.mark.parametrize("cp,heads", ((2, 1), (4, 2)))
def test_dense_window_dcp_through_the_merge_kernel_is_the_cp1_output_rounded_once(cp, heads):
    """Every rank's merged bf16 heads against bf16 of the CP = 1 dense partial (as the sparse
    test above: the merge kernel's bound, the merge tolerance and Y1's own rounding)."""
    hq = cp * heads
    q, bank, table, lens = dense_case(130, 250, 4, hq, seed=150 + cp)
    partials, lses, windows, owns, live = dense_ranks(q, bank, table, lens, cp)
    full = DW.mla_dense_window_attention_partial(q, bank, lens, scale_of(LATENT), table,
                                                 page_size=DCP_BLOCK)[0]
    tol = merged_tolerance(dense_bounds(q, window_of(bank, table), lens),
                           [dense_bounds(q, w, n) for w, n in zip(windows, owns)], live)
    for rank in range(cp):
        heads_of_rank = slice(rank * heads, (rank + 1) * heads)
        mine_p = partials[:, heads_of_rank].contiguous()
        mine_l = lses[:, heads_of_rank].contiguous()
        got = DM.dcp_lse_merge(mine_p, mine_l)
        y1 = full[heads_of_rank].permute(1, 0, 2).double()
        bound = (merge_bound(mine_p, mine_l) + tol[heads_of_rank].permute(1, 0, 2)
                 + U16 * y1.abs())
        err = (got.double() - y1.to(torch.bfloat16).double()).abs()
        assert bool((err <= bound).all()), \
            f"rank {rank}: max err/bound {float((err / bound.clamp_min(1e-300)).max()):.3g}"


@pytest.mark.parametrize("hq", (1, 2, 8))
def test_two_programs_are_bitwise_one_program_for_the_dense_window_partial(hq, monkeypatch):
    """At LNC2 the dense partial deals its jobs over both cores: each head's whole tiles, then
    each head's partial tile, then each head's zero-row chunk (150 of 200 queries active).

    At Hq 1 those are three jobs: the first program takes two and the second one, an uneven
    split fixed at trace time (``reports/dcp_item4.md`` section 14).
    """
    q, bank, table, lens = dense_case(200, 100, 3, hq, seed=160 + hq)
    own = own_lens(lens, 2, 0, 3)
    mine = rank_table(table, 2, 0)
    call = dict(page_size=DCP_BLOCK, active_rows=150)
    monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    assert DW._programs(200 * hq) == 1
    one = DW.mla_dense_window_attention_partial(q, bank, own, scale_of(LATENT), mine, **call)
    calls = []

    def record(entry):
        calls.append(RecordingCall(entry))
        return calls[-1]

    monkeypatch.setattr(DW, "wrap_nki", record)
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    two = DW.mla_dense_window_attention_partial(q, bank, own, scale_of(LATENT), mine, **call)
    assert [call.grid for call in calls] == [2]
    torch.testing.assert_close(two[0], one[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(two[1], one[1], rtol=0.0, atol=0.0)


@pytest.mark.parametrize("named", ("int32", "negative", "past", "shape"))
def test_the_dense_window_partial_seam_refuses_a_malformed_length_by_name(named):
    """``owned_lens`` must be int32 (no cast is traced), [S], and in [0, staged rows]."""
    q, bank, table, lens = dense_case(4, 0, 2, 2, seed=170)
    if named == "int32":
        lens = lens.long()
    if named == "negative":
        lens[1] = -1
    if named == "past":
        lens[2] = 2 * DCP_BLOCK + 1
    if named == "shape":
        lens = lens[:3]
    match = "int32" if named == "int32" else "owned_lens"
    with pytest.raises(DW.MlaDenseWindowError, match=match):
        DW.mla_dense_window_attention_partial(q, bank, lens, scale_of(LATENT), table,
                                              page_size=DCP_BLOCK)


def _dense_child(rows: int, hq: int, pages: int, active: int) -> None:
    meta = torch.device("meta")
    bf, i32 = torch.bfloat16, torch.int32

    def fn(q, bank, table, lens, written, offset):
        return DW.mla_dense_window_attention_partial(q, bank, lens, scale_of(LATENT), table,
                                                     written=written, write_offset=offset,
                                                     page_size=COMPILE_PAGE, active_rows=active)

    args = (torch.empty((rows, hq, LATENT), dtype=bf, device=meta),
            torch.empty((2 * pages * COMPILE_PAGE, LATENT), dtype=bf, device=meta),
            torch.empty((pages, 1), dtype=i32, device=meta),
            torch.empty((rows,), dtype=i32, device=meta),
            torch.empty((rows // hq, LATENT), dtype=bf, device=meta),
            torch.empty((1, 1), dtype=i32, device=meta))
    compile_to_neff(fn, args, DW._programs(rows * hq))


@pytest.mark.parametrize("rows,hq,pages,active", DENSE_COMPILES,
                         ids=[f"q{r}-hq{h}-w{p * COMPILE_PAGE}-a{a}"
                              for r, h, p, a in DENSE_COMPILES])
def test_neuronx_cc_builds_the_dense_window_partial_entry(rows, hq, pages, active):
    """neuronx-cc builds the dense partial through its seam, on both cores of an LNC2 pair."""
    fields = compile_in_a_child(str(pathlib.Path(__file__).resolve()), "dense", str(rows),
                                str(hq), str(pages), str(active))
    assert fields["programs"] == "2", fields


def _child(body: str, rows: int, hq: int, latent: int, topk: int) -> None:
    meta = torch.device("meta")
    bf, i32 = torch.bfloat16, torch.int32
    q = torch.empty((rows, hq, latent), dtype=bf, device=meta)
    idx = torch.empty((rows, topk), dtype=i32, device=meta)
    if body == "row_tiled":
        pages = topk // COMPILE_PAGE
        bank = torch.empty((2 * pages * COMPILE_PAGE, latent), dtype=bf, device=meta)
        table = torch.empty((pages, 1), dtype=i32, device=meta)
        written = torch.empty((rows, latent), dtype=bf, device=meta)
        offset = torch.empty((1, 1), dtype=i32, device=meta)

        def fn(q, bank, idx, table, written, offset):
            return MS.mla_sparse_attention_partial(q, bank, idx, scale_of(latent),
                                                   block_table_row=table, written=written,
                                                   write_offset=offset, page_size=COMPILE_PAGE)

        args = (q, bank, idx, table, written, offset)
    else:
        def fn(q, c, idx):
            return MS.mla_sparse_attention_partial(q, c, idx, scale_of(latent))

        args = (q, torch.empty((2048, latent), dtype=bf, device=meta), idx)
    compile_to_neff(fn, args, MS.partial_programs(rows, hq, topk, latent))


@pytest.mark.parametrize("body,rows,hq,latent,topk", PARTIAL_COMPILES,
                         ids=[f"{b}-r{r}-hq{h}-l{lat}-k{k}"
                              for b, r, h, lat, k in PARTIAL_COMPILES])
def test_neuronx_cc_builds_the_partial_entry(body, rows, hq, latent, topk):
    """neuronx-cc builds every body in partial mode through the partial seam.

    The served row-tiled call runs on both cores of an LNC2 pair; the other two bodies on one.
    """
    fields = compile_in_a_child(str(pathlib.Path(__file__).resolve()), body, str(rows), str(hq),
                                str(latent), str(topk))
    assert fields["programs"] == ("2" if body == "row_tiled" else "1"), fields


if __name__ == "__main__" and sys.argv[1:2] == ["child"]:
    if sys.argv[2] == "dense":
        _dense_child(*(int(v) for v in sys.argv[3:7]))
    else:
        _child(sys.argv[2], *(int(v) for v in sys.argv[3:7]))
