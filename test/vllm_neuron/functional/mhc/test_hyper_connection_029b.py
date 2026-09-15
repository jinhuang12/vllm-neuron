# SPDX-License-Identifier: Apache-2.0
"""Tier N acceptance for `inc-glm53f-029b` -- tiling the mHC combine's token axis.

Acceptance command (the same harness `-029` declares, this file substituted)::

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
    python -m pytest test/vllm_neuron/functional/mhc/test_hyper_connection_029b.py \
      --timeout 60 -p no:cacheprovider -s -rA

What `-029b` changed, and what therefore has to be measured
-----------------------------------------------------------
`-029` allocated every sbuf tile in the kernel body at the FULL token count, so a
token extent above the partition limit could not be expressed: measured then,
``T = 128`` ran and ``T = 129`` trapped inside NKI with ``dma_copy dst partition
dimension 129 exceeds maximum 128``. `-029b` walks the token axis in tiles of
``nl.tile_size.pmax`` instead, so the extent is served.

Six items, NO ``parametrize``, so the collected count is derivable from the file
before it runs:

1. a FULL tile -- ``T = PARTITION_MAX`` exactly;
2. a genuinely SHORT last tile -- ``T = PARTITION_MAX + 1``, one row in the last
   tile, plus the pass-through identity pattern at that extent, which is
   BIT-exact against a reference this file did not author;
3. TWO OR MORE tiles -- ``T = 300`` (2 whole tiles and a 44-row remainder) and
   ``T = 2048`` (16 whole tiles, the extent the WP11 rungs need);
4. BIT-EXACTNESS below the old ceiling against an UNTILED reference kernel -- the
   pre-`-029b` body, copied in verbatim because the tree no longer holds it;
5. THE TRAP CONTROL, both directions: the untiled reference must trap at
   ``PARTITION_MAX + 1`` with the extent and the maximum EXTRACTED from the
   vendor's message, and the tiled kernel must serve the same shape. Without the
   first half, item 1 could be satisfied by a kernel that tiles nothing;
6. THE ROUTE CONTROL: one NKI dispatch per call and zero torch fallbacks, with
   the fallback zero shown able to move.

Why item 4 needs a copy of the old body. The untiled kernel is gone from the tree,
so "bit-identical to the code it replaced" can only be a measurement if the
replaced code is here. The copy is TEST-ONLY, is imported by nothing shipped, and
is deliberately left untiled so item 5 can drive it into the vendor's partition
check.

Tolerances are the pair `-029` already registered (``test_hyper_connection.py``
``:102-103``): ``rtol=1e-2`` with ``atol=1e-5``. No tolerance is authored, widened
or narrowed here, and the tiled-versus-untiled arm carries no tolerance at all --
tiling reorders nothing WITHIN a row, so the comparison is ``torch.equal``.
"""

from __future__ import annotations

import re
import time

import pytest
import torch

import nki
import nki.isa as nisa
import nki.language as nl
import nki.simulator
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.mhc import hyper_connection as mod
from vllm_neuron.functional.mhc.hyper_connection import (
    MHC_STREAMS,
    PARTITION_MAX,
    can_run_hyper_connection,
    dispatch_counters,
    hyper_connection_combine,
    hyper_connection_torch_oracle,
    reset_dispatch_counters,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

S = MHC_STREAMS  # 4, the target's `hc_mult`, read off the module rather than typed

#: The pair `-029` registered. Cited, not re-authored (P9).
RTOL = 1e-2
ATOL = 1e-5

#: Token extents. Each one is a DIFFERENT relationship to the tile height, which
#: is the whole point of the set: exactly one tile; one tile plus a single row;
#: two tiles plus a short remainder; and many whole tiles.
FULL_TILE = PARTITION_MAX  # 128
SHORT_LAST_TILE = PARTITION_MAX + 1  # 129 -- the extent `-029` trapped on
MULTI_TILE_REMAINDER = 300  # 2 * 128 + 44
MULTI_TILE_WHOLE = 2048  # 16 * 128, the WP11 prefill extent

#: Extents where the UNTILED parent still runs, so item 4 can compare against it.
ROWS_PARENT_RUNS = (5, PARTITION_MAX)


def _emit(tag: str, **values: object) -> None:
    """One row per reading, pipe-separated, prefix first.

    The prefix is an exact literal a transcript reader anchors on, and every value
    is a ``key=value`` field, so a row can be read without knowing the order.
    """
    fields = "|".join(f"{k}={v}" for k, v in values.items())
    print(f"D029B|{tag}|{fields}", flush=True)


class _SimulatorCounter:
    """Counts real ``nki.simulator.simulate_kernel`` calls for the duration.

    The second route instrument, in the shape `-029`'s acceptance already uses. It
    counts the VENDOR entry point, so a bug in the seam's own counter cannot fake
    it -- which is what makes "a kernel ran" a reading rather than an inference
    (F1). Under a torch fallback both sides of a numeric comparison would be
    torch, and this instrument reads 0.
    """

    def __init__(self) -> None:
        self.calls = 0
        self._real = None

    def __enter__(self) -> "_SimulatorCounter":
        self._real = nki.simulator.simulate_kernel
        real = self._real

        def counting(*args, **kwargs):
            self.calls += 1
            return real(*args, **kwargs)

        nki.simulator.simulate_kernel = counting
        return self

    def __exit__(self, *exc_info) -> None:
        nki.simulator.simulate_kernel = self._real


# --------------------------------------------------------------------------- #
# The UNTILED reference kernel -- `-029`'s body, copied verbatim.                #
# TEST-ONLY. Nothing shipped imports this, and it is left untiled ON PURPOSE so   #
# item 5 can drive it into the vendor's partition check.                         #
# --------------------------------------------------------------------------- #
@nki.jit
def _untiled_reference_kernel(x, residual, post_layer_mix, comb_res_mix):
    """`-029`'s mHC combine body, before `-029b` tiled the token axis.

    Every sbuf tile here is allocated at the FULL token count, which is exactly
    the limitation `-029b` removed.
    """
    t_extent, s_extent, h_extent = residual.shape

    out = nl.ndarray(
        (t_extent, s_extent, h_extent), dtype=nl.float32, buffer=nl.shared_hbm
    )

    x_tile = nl.load(x, dtype=nl.float32)

    streams = [
        nl.load(residual[0:t_extent, i, 0:h_extent], dtype=nl.float32)
        for i in range(s_extent)
    ]

    acc = nl.ndarray((t_extent, h_extent), dtype=nl.float32, buffer=nl.sbuf)
    term = nl.ndarray((t_extent, h_extent), dtype=nl.float32, buffer=nl.sbuf)

    for j in range(s_extent):
        post_j = nl.load(post_layer_mix[0:t_extent, j, 0:1], dtype=nl.float32)
        nisa.tensor_scalar(dst=acc, data=x_tile, op0=nl.multiply, operand0=post_j)

        for i in range(s_extent):
            w_ij = nl.load(comb_res_mix[0:t_extent, i, j : j + 1], dtype=nl.float32)
            nisa.tensor_scalar(
                dst=term, data=streams[i], op0=nl.multiply, operand0=w_ij
            )
            nisa.tensor_tensor(dst=acc, data1=acc, data2=term, op=nl.add)

        nl.store(out[0:t_extent, j, 0:h_extent], value=acc)

    return out


# --------------------------------------------------------------------------- #
# Fixtures and references.                                                      #
# --------------------------------------------------------------------------- #
def _inputs(rows: int, hidden: int, seed: int = 29):
    """The four tensors, fp32, deterministic by seed.

    Same shape and construction as `-029`'s fixture: ``comb_res_mix`` is
    row-stochastic (what a Sinkhorn stage hands the combine) and asymmetric (so an
    ``i``/``j`` transpose is visible), and ``x``/``residual`` are signed so a sign
    error shows.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn((rows, hidden), generator=g, dtype=torch.float32)
    residual = torch.randn((rows, S, hidden), generator=g, dtype=torch.float32)
    post_layer_mix = torch.rand((rows, S, 1), generator=g, dtype=torch.float32)
    comb_res_mix = torch.softmax(
        torch.randn((rows, S, S), generator=g, dtype=torch.float32), dim=-1
    )
    return x, residual, post_layer_mix, comb_res_mix


def _reference(x, residual, post_layer_mix, comb_res_mix):
    """The base's ``bmm(comb.mT, residual)`` spelling, authored here.

    The module's own oracle uses the ``einsum`` spelling, so this is a SECOND,
    independent statement of the ``i``/``j`` convention rather than a restatement
    of the module's.
    """
    term2 = torch.bmm(comb_res_mix.mT.to(torch.float32), residual.to(torch.float32))
    return x.to(torch.float32).unsqueeze(-2) * post_layer_mix.to(torch.float32) + term2


def _pass_through_weights(rows: int):
    """``comb_res_mix = I_S`` and ``post_layer_mix = 0``, so ``out == residual``.

    In fp32 that is BIT-exact: multiplying by exactly ``1.0`` and adding exact
    ``0.0`` are both lossless. The expectation is the ``residual`` tensor itself,
    which nothing in this file authored -- so at a multi-tile extent it is also
    the strongest available reading that each output row got its OWN input row
    across a tile boundary.
    """
    ident = torch.eye(S, dtype=torch.float32).expand(rows, S, S).contiguous()
    zero_post = torch.zeros((rows, S, 1), dtype=torch.float32)
    return zero_post, ident


def _errors(got, want) -> tuple[float, float]:
    """``(max_abs, max_rel)`` -- numbers, not verdicts."""
    got = got.to(torch.float32)
    want = want.to(torch.float32)
    max_abs = float((got - want).abs().max())
    max_rel = float(((got - want).abs() / (want.abs() + ATOL)).max())
    return max_abs, max_rel


def _tiles(rows: int) -> tuple[int, int]:
    """``(tile_count, rows_in_the_last_tile)`` at the tile height."""
    count = -(-rows // PARTITION_MAX)
    return count, rows - (count - 1) * PARTITION_MAX


def _serve(rows: int, hidden: int, tag: str):
    """Run the seam at one extent, read the route, and compare at the pair.

    Returns the served tensor so a caller can make a further, stricter assertion.
    """
    x, residual, post_layer_mix, comb_res_mix = _inputs(rows, hidden)
    tile_count, last_rows = _tiles(rows)

    assert can_run_hyper_connection(
        x, residual, post_layer_mix, comb_res_mix
    ) is True, f"{tag}: the gate refused {rows} tokens"

    reset_dispatch_counters()
    started = time.perf_counter()
    with _SimulatorCounter() as sim:
        got = hyper_connection_combine(x, residual, post_layer_mix, comb_res_mix)
    elapsed = time.perf_counter() - started
    nki_n, fallback_n = dispatch_counters()

    want = _reference(x, residual, post_layer_mix, comb_res_mix)
    max_abs, max_rel = _errors(got, want)
    _emit(
        tag,
        rows=rows,
        hidden=hidden,
        streams=S,
        tile_height=PARTITION_MAX,
        tiles=tile_count,
        last_tile_rows=last_rows,
        nki_dispatch=nki_n,
        torch_fallback=fallback_n,
        simulate_kernel_calls=sim.calls,
        can_run_kernel=can_run_kernel(torch.zeros(1)),
        max_abs_error=f"{max_abs:.6e}",
        max_rel_error=f"{max_rel:.6e}",
        rtol=RTOL,
        atol=ATOL,
        want_absmax=f"{float(want.abs().max()):.6e}",
        seconds=f"{elapsed:.2f}",
    )

    assert (nki_n, fallback_n) == (1, 0), f"{tag}: route read {(nki_n, fallback_n)}"
    assert sim.calls == 1, (
        f"{tag}: nki.simulator.simulate_kernel ran {sim.calls} times, declared 1. "
        f"A numeric pass without a simulator call is the F1 false green"
    )
    assert can_run_kernel(torch.zeros(1)) is True, f"{tag}: the NKI route is absent"
    assert float(want.abs().max()) > 0.0, f"{tag}: the reference is all zero"
    assert tuple(got.shape) == (rows, S, hidden), tuple(got.shape)
    torch.testing.assert_close(got.to(torch.float32), want, rtol=RTOL, atol=ATOL)
    return got


# --------------------------------------------------------------------------- #
# ITEM 1 -- a full tile.                                                        #
# --------------------------------------------------------------------------- #
def test_a_full_tile_runs_and_matches_the_reference() -> None:
    """``T = PARTITION_MAX`` exactly: one tile, no remainder.

    This is the extent `-029` already served, and it is here because tiling must
    not break it: at exactly the tile height the loop runs once and the narrowing
    ``min`` must pick the full height rather than a short one.
    """
    tile_count, last_rows = _tiles(FULL_TILE)
    assert (tile_count, last_rows) == (1, PARTITION_MAX), (tile_count, last_rows)
    _serve(FULL_TILE, 256, "ITEM1_FULL_TILE")


# --------------------------------------------------------------------------- #
# ITEM 2 -- a genuinely short last tile, the extent `-029` trapped on.           #
# --------------------------------------------------------------------------- #
def test_a_short_last_tile_runs_and_is_bit_exact_under_the_identity_pattern() -> None:
    """``T = PARTITION_MAX + 1``: the last tile holds ONE row.

    A short last tile is where a padded tile or a mis-computed offset shows, so
    this extent carries a second, stricter reading on top of the tolerance
    comparison. Under the pass-through pattern the expected output IS the
    ``residual`` tensor, so the comparison is bit-exact and its reference is not
    authored here -- which makes it a direct reading that each output row, in both
    tiles, received its own input row.
    """
    tile_count, last_rows = _tiles(SHORT_LAST_TILE)
    assert (tile_count, last_rows) == (2, 1), (tile_count, last_rows)
    _serve(SHORT_LAST_TILE, 256, "ITEM2_SHORT_LAST_TILE")

    _, residual, _, _ = _inputs(SHORT_LAST_TILE, 256)
    zero_post, ident = _pass_through_weights(SHORT_LAST_TILE)
    x = torch.randn(
        (SHORT_LAST_TILE, 256),
        generator=torch.Generator().manual_seed(7),
        dtype=torch.float32,
    )

    reset_dispatch_counters()
    got = hyper_connection_combine(x, residual, zero_post, ident)
    nki_n, fallback_n = dispatch_counters()
    max_abs, _ = _errors(got, residual)
    differing = int((got.to(torch.float32) != residual).sum())
    _emit(
        "ITEM2_IDENTITY_IS_BIT_EXACT",
        rows=SHORT_LAST_TILE,
        tiles=tile_count,
        last_tile_rows=last_rows,
        nki_dispatch=nki_n,
        torch_fallback=fallback_n,
        max_abs_error=f"{max_abs:.6e}",
        differing_entries=differing,
        bit_exact=torch.equal(got.to(torch.float32), residual),
        residual_absmax=f"{float(residual.abs().max()):.6e}",
        x_absmax=f"{float(x.abs().max()):.6e}",
    )
    assert (nki_n, fallback_n) == (1, 0), (nki_n, fallback_n)
    assert float(residual.abs().max()) > 0.0, "the residual fixture is all zero"
    assert float(x.abs().max()) > 0.0, (
        "x is all zero, so the post term could not have contaminated the result "
        "even if `post_layer_mix = 0` were ignored"
    )
    assert torch.equal(got.to(torch.float32), residual), (
        f"{differing} entries differ from the residual tensor; max_abs={max_abs:.6e}"
    )


# --------------------------------------------------------------------------- #
# ITEM 3 -- two or more tiles.                                                  #
# --------------------------------------------------------------------------- #
def test_multi_tile_extents_run_and_match_the_reference() -> None:
    """``T = 300`` (a short remainder) and ``T = 2048`` (whole tiles).

    Two different remainder shapes at a token count no single tile could hold. The
    2,048-row extent is the one the WP11 rungs need, and it is the reading that
    says the ceiling is gone rather than merely raised by one row.
    """
    remainder_tiles, remainder_rows = _tiles(MULTI_TILE_REMAINDER)
    whole_tiles, whole_rows = _tiles(MULTI_TILE_WHOLE)
    assert (remainder_tiles, remainder_rows) == (3, 44), (
        remainder_tiles,
        remainder_rows,
    )
    assert (whole_tiles, whole_rows) == (16, PARTITION_MAX), (whole_tiles, whole_rows)

    _serve(MULTI_TILE_REMAINDER, 32, "ITEM3_MULTI_TILE_REMAINDER")
    _serve(MULTI_TILE_WHOLE, 8, "ITEM3_MULTI_TILE_WHOLE")


# --------------------------------------------------------------------------- #
# ITEM 4 -- bit-exactness against the untiled parent, where the parent runs.      #
# --------------------------------------------------------------------------- #
def test_tiled_is_bit_identical_to_the_untiled_parent_where_the_parent_runs() -> None:
    """Below the old ceiling the tiled kernel equals the code it replaced, exactly.

    Tiling changes which rows share a tile; it does not change any row's
    arithmetic or the order of it. So the claim is not "close" but IDENTICAL, and
    the comparison carries no tolerance. This is what separates a tiling change
    from a rewrite that happens to stay inside the tolerance.
    """
    for rows in ROWS_PARENT_RUNS:
        x, residual, post_layer_mix, comb_res_mix = _inputs(rows, 32)

        reset_dispatch_counters()
        with _SimulatorCounter() as sim:
            got = hyper_connection_combine(x, residual, post_layer_mix, comb_res_mix)
            parent = wrap_nki(_untiled_reference_kernel)(
                x=x,
                residual=residual,
                post_layer_mix=post_layer_mix,
                comb_res_mix=comb_res_mix,
            )
        nki_n, fallback_n = dispatch_counters()

        got32 = got.to(torch.float32)
        parent32 = parent.to(torch.float32)
        max_abs, _ = _errors(got32, parent32)
        differing = int((got32 != parent32).sum())
        _emit(
            "ITEM4_EQUALS_THE_UNTILED_PARENT",
            rows=rows,
            hidden=32,
            entries=got32.numel(),
            nki_dispatch=nki_n,
            torch_fallback=fallback_n,
            simulate_kernel_calls=sim.calls,
            max_abs_diff=f"{max_abs:.6e}",
            differing_entries=differing,
            bit_exact=torch.equal(got32, parent32),
            parent_absmax=f"{float(parent32.abs().max()):.6e}",
        )
        assert (nki_n, fallback_n) == (1, 0), (rows, nki_n, fallback_n)
        # TWO kernels ran under one counter: the seam's, which the seam counted,
        # and this file's untiled parent, which it did not. A reading of 1 would
        # mean the parent never reached the simulator, so the comparison would be
        # against nothing.
        assert sim.calls == 2, (
            f"rows={rows}: simulate_kernel ran {sim.calls} times, declared 2 "
            f"(the seam's kernel and this file's untiled parent)"
        )
        assert float(parent32.abs().max()) > 0.0, (
            f"rows={rows}: the parent produced an all-zero tensor, so the "
            f"comparison would pass over empty output"
        )
        assert torch.equal(got32, parent32), (
            f"rows={rows}: {differing} entries differ, max_abs_diff={max_abs:.6e}"
        )


# --------------------------------------------------------------------------- #
# ITEM 5 -- the trap control, both directions.                                  #
# --------------------------------------------------------------------------- #
def test_the_untiled_parent_traps_where_the_tiled_kernel_serves() -> None:
    """The parent must TRAP at ``PARTITION_MAX + 1``; the tiled kernel must serve.

    Both halves are needed. Without the first, every admission item above could be
    satisfied by a kernel that tiles nothing while the ceiling had quietly moved
    elsewhere -- the defect this control exists to exclude. Without the second, the
    trap alone would say nothing about the candidate.

    The extent and the maximum are EXTRACTED from the vendor's message and
    asserted as numbers, and the message is printed verbatim, so the reading is
    the vendor's own rather than this file's paraphrase of it.
    """
    x, residual, post_layer_mix, comb_res_mix = _inputs(SHORT_LAST_TILE, 8)

    with pytest.raises(Exception) as excinfo:  # noqa: PT011 - a bare vendor assert
        wrap_nki(_untiled_reference_kernel)(
            x=x,
            residual=residual,
            post_layer_mix=post_layer_mix,
            comb_res_mix=comb_res_mix,
        )
    message = str(excinfo.value)
    print(f"D029B|ITEM5_PARENT_RAISES_VERBATIM|{message}", flush=True)
    matched = re.search(r"partition dimension (\d+) exceeds maximum (\d+)", message)
    _emit(
        "ITEM5_PARENT_TRAPS",
        rows=SHORT_LAST_TILE,
        raised=1,
        matched=int(bool(matched)),
        dimension=matched.group(1) if matched else "none",
        maximum=matched.group(2) if matched else "none",
    )
    assert matched is not None, (
        f"expected the vendor partition assert naming the extent; got {message!r}"
    )
    assert matched.group(1) == str(SHORT_LAST_TILE), message
    assert matched.group(2) == str(PARTITION_MAX), message

    # The other direction, on the SAME shape: the tiled kernel serves it.
    served = _serve(SHORT_LAST_TILE, 8, "ITEM5_TILED_SERVES_THE_SAME_SHAPE")
    assert tuple(served.shape) == (SHORT_LAST_TILE, S, 8), tuple(served.shape)


# --------------------------------------------------------------------------- #
# ITEM 6 -- the route control, with the fallback zero shown able to move.        #
# --------------------------------------------------------------------------- #
def test_the_route_is_one_dispatch_per_call_and_no_torch_fallback() -> None:
    """One NKI dispatch per call above the old ceiling, and zero fallbacks.

    Tiling could have been done on the HOST -- a python loop over tiles calling
    the seam once per tile -- and a numeric comparison could not tell the
    difference. The dispatch counter can: a host loop over 16 tiles would read
    16. So this reads ``1`` at the 2,048-row extent, which is the counted
    statement that the tiling is INSIDE one dispatch.

    The zero would be worthless if nothing could make it nonzero, so the same seam
    is then forced down the oracle route and the counter read again.
    """
    x, residual, post_layer_mix, comb_res_mix = _inputs(MULTI_TILE_WHOLE, 8)
    tile_count, _ = _tiles(MULTI_TILE_WHOLE)

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        hyper_connection_combine(x, residual, post_layer_mix, comb_res_mix)
    nki_n, fallback_n = dispatch_counters()
    _emit(
        "ITEM6_ROUTE",
        rows=MULTI_TILE_WHOLE,
        tiles=tile_count,
        nki_dispatch=nki_n,
        torch_fallback=fallback_n,
        simulate_kernel_calls=sim.calls,
        a_host_loop_would_read=tile_count,
    )
    assert (nki_n, fallback_n) == (1, 0), (nki_n, fallback_n)
    assert sim.calls == 1, (
        f"simulate_kernel ran {sim.calls} times at {tile_count} tiles, declared 1; "
        f"a host loop over tiles would read {tile_count} on BOTH instruments"
    )

    # THE FIRING CONTROL, in the landed form this suite uses: the gate reads
    # `can_run_kernel` as a module global, so replacing it on the module under
    # test is what a call in a no-NKI process sees.
    reset_dispatch_counters()
    saved = mod.can_run_kernel
    try:
        mod.can_run_kernel = lambda *args, **kwargs: False
        with _SimulatorCounter() as sim_off:
            assert (
                can_run_hyper_connection(x, residual, post_layer_mix, comb_res_mix)
                is False
            )
            served = hyper_connection_combine(
                x, residual, post_layer_mix, comb_res_mix
            )
    finally:
        mod.can_run_kernel = saved
    nki_after, fallback_after = dispatch_counters()
    oracle = hyper_connection_torch_oracle(x, residual, post_layer_mix, comb_res_mix)
    _emit(
        "ITEM6_FALLBACK_ZERO_CAN_MOVE",
        nki_dispatch=nki_after,
        torch_fallback=fallback_after,
        simulate_kernel_calls=sim_off.calls,
        served_equals_oracle=torch.equal(served, oracle),
    )
    assert (nki_after, fallback_after) == (0, 1), (nki_after, fallback_after)
    assert sim_off.calls == 0, (
        f"the fallback path reached the simulator {sim_off.calls} times; both "
        f"instruments must move together, or one of them is not reading the route"
    )
    assert torch.equal(served, oracle), (
        "the fallback route must still answer correctly at a token count above "
        "the old ceiling"
    )
    assert mod.can_run_kernel is can_run_kernel, "the control leaked into the process"
    _emit(
        "ITEM6_GATE_RESTORED",
        can_run=can_run_hyper_connection(x, residual, post_layer_mix, comb_res_mix),
    )
    assert (
        can_run_hyper_connection(x, residual, post_layer_mix, comb_res_mix) is True
    ), "the gate must be restored"
