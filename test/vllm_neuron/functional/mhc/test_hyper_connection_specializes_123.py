# SPDX-License-Identifier: Apache-2.0
"""Tier N acceptance for the mHC combine kernel COMPILING, not merely simulating.

Acceptance command (the harness this suite already uses, this file substituted)::

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 \
    python -m pytest \
      test/vllm_neuron/functional/mhc/test_hyper_connection_specializes_123.py \
      --timeout 900 -p no:cacheprovider -s -rA

What this file measures that the landed acceptance could not
-----------------------------------------------------------
The landed acceptance calls this kernel directly on the NKI SIMULATOR, which
executes the body as ordinary python and asks no compiler anything. The server first captures a graph of the whole model and
hands every NKI seam in it to the NKI COMPILER to specialise. A body the
simulator runs happily can be one the compiler refuses, and this one was --
``failed to specialize NKI kernel: Collected 1 different diagnostics: - [x1]
error: unsupported expression``, at the first kernel call of every layer's
forward, on every rank, so no graph of the model could be captured at all.

Three items, NO ``parametrize``, so the collected count is derivable from this
file before it runs:

1. SPECIALISATION at the seam, under a META FORWARD of the tiny stack -- the
   venue that refused the pre-change body on this host, reported per capture leg.
   The -119 meta-forward item's own helpers are imported and used: they build the
   tiny root and its runner on ``meta`` and drive the two capture entry points
   with nothing stood down, so the seam dispatches for real and the vendor's
   frontend is asked to specialise this kernel.

   WHY NOT A DIRECT CALL ON THE WRAPPED KERNEL. That call is answered from the
   simulator, which interprets the body rather than compiling it, so both this
   body and the one it replaced come back with a shape and no complaint. What
   reaches the frontend is META tensors arriving through the seam. The pin set is
   unchanged for all three items.

   The candidate requirement is that no specialisation refusal appears; the base
   arm requires the refusal, its two vendor fragments and the seam as the site.
   Everything else the forward says is printed WHOLE and not judged: a second
   refused construct behind the first is a result to show, not a failure.
2. BIT-IDENTITY against that refused body on the simulator venue, and exact
   identity against the module's torch oracle under the pass-through pattern
   where both are lossless. No tolerance is used or authored here.
3. THE ROUTE: one NKI dispatch per call, no torch fallback, off the module's own
   counters -- a body that answered in torch would satisfy item 2.

THE BASE ARM. These bytes also run against the tree this change was made on,
where the refusal is the EXPECTED reading: under ``GLM53F_123_EXPECT_BASE=1``
item 1 requires the module's kernel to be refused and to say why. Item 2 holds
there because the refused body IS that tree's body.
"""

from __future__ import annotations

import os

import pytest
import torch

import nki
import nki.isa as nisa
import nki.language as nl
import nki.simulator

from vllm_neuron.functional.mhc.hyper_connection import (
    MHC_STREAMS,
    PARTITION_MAX,
    dispatch_counters,
    hyper_connection_combine,
    hyper_connection_kernel,
    hyper_connection_torch_oracle,
    reset_dispatch_counters,
)

# The landed meta-forward item, imported rather than re-implemented: its tiny root on
# ``meta``, its runner shell, its two capture entry points and its site reader are the
# venue that refused the body this change replaces.
from test.vllm_neuron.model.glm5_next import test_meta_forward_119 as meta119

S = MHC_STREAMS  # 4, the target's `hc_mult`, read off the module rather than typed

#: The vendor's own two fragments, required of the base arm and of the copy.
WANT_REFUSAL = "failed to specialize NKI kernel"
WANT_DIAGNOSTIC = "unsupported expression"

#: Which tree these bytes run against; the file cannot tell, so the env names it.
EXPECT_BASE = os.environ.get("GLM53F_123_EXPECT_BASE") == "1"

#: Identity extents: a short last tile, one whole tile, and many whole tiles.
IDENTITY_SHAPES = (7, PARTITION_MAX, 2048)
IDENTITY_HIDDEN = 512


def _emit(tag: str, **values: object) -> None:
    """One row per reading, pipe-separated, prefix first."""
    fields = "|".join(f"{k}={v}" for k, v in values.items())
    print(f"HC123|{tag}|{fields}", flush=True)


class _SimulatorCounter:
    """Counts real ``nki.simulator.simulate_kernel`` calls, the vendor's own entry."""

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


# THE REFUSED BODY, copied verbatim from the tree this change was made on.
# TEST-ONLY: nothing shipped imports it. It keeps the LIST COMPREHENSION over the
# loaded stream tiles ON PURPOSE -- that is the construct the compiler counts and
# refuses -- and everything else verbatim, or item 2 compares against something else.
@nki.jit
def _refused_reference_kernel(x, residual, post_layer_mix, comb_res_mix):
    """The mHC combine as it was written before it was made to compile."""
    t_extent, s_extent, h_extent = residual.shape
    pmax = nl.tile_size.pmax
    n_tiles = (t_extent + pmax - 1) // pmax

    out = nl.ndarray(
        (t_extent, s_extent, h_extent), dtype=nl.float32, buffer=nl.shared_hbm
    )

    for t in range(n_tiles):
        rows = min(pmax, t_extent - t * pmax)
        off = t * pmax

        x_tile = nl.load(x[off : off + rows, 0:h_extent], dtype=nl.float32)

        streams = [
            nl.load(residual[off : off + rows, i, 0:h_extent], dtype=nl.float32)
            for i in range(s_extent)
        ]

        acc = nl.ndarray((rows, h_extent), dtype=nl.float32, buffer=nl.sbuf)
        term = nl.ndarray((rows, h_extent), dtype=nl.float32, buffer=nl.sbuf)

        for j in range(s_extent):
            post_j = nl.load(
                post_layer_mix[off : off + rows, j, 0:1], dtype=nl.float32
            )
            nisa.tensor_scalar(dst=acc, data=x_tile, op0=nl.multiply, operand0=post_j)

            for i in range(s_extent):
                w_ij = nl.load(
                    comb_res_mix[off : off + rows, i, j : j + 1], dtype=nl.float32
                )
                nisa.tensor_scalar(
                    dst=term, data=streams[i], op0=nl.multiply, operand0=w_ij
                )
                nisa.tensor_tensor(dst=acc, data1=acc, data2=term, op=nl.add)

            nl.store(out[off : off + rows, j, 0:h_extent], value=acc)

    return out


def _inputs(rows: int, hidden: int, seed: int = 123):
    """The four tensors, fp32, deterministic by seed, as the landed suite builds them."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn((rows, hidden), generator=g, dtype=torch.float32)
    residual = torch.randn((rows, S, hidden), generator=g, dtype=torch.float32)
    post_layer_mix = torch.rand((rows, S, 1), generator=g, dtype=torch.float32)
    comb_res_mix = torch.softmax(
        torch.randn((rows, S, S), generator=g, dtype=torch.float32), dim=-1
    )
    return x, residual, post_layer_mix, comb_res_mix


def _pass_through_weights(rows: int):
    """``comb_res_mix = I_S`` and ``post_layer_mix = 0``, so ``out == residual``.

    Lossless on both sides in fp32 -- a multiply by exactly ``1.0`` and an add of
    exact ``0.0`` -- so kernel and oracle must agree to the bit here.
    """
    ident = torch.eye(S, dtype=torch.float32).expand(rows, S, S).contiguous()
    zero_post = torch.zeros((rows, S, 1), dtype=torch.float32)
    return zero_post, ident


#: The two capture legs the meta forward drives, and the seam the diagnostic must name.
LEGS = meta119.LEGS
SEAM_MODULE = "functional/mhc/hyper_connection.py"


# --------------------------------------------------------------------------- #
# ITEM 1 -- the compiler specialises this body where it refused the one before.  #
# --------------------------------------------------------------------------- #
@pytest.mark.forked
def test_the_kernel_specialises_under_the_capture_regime() -> None:
    """One row per leg, off the venue that refused the body this change replaced.

    THE VENUE IS THE SEAM UNDER A META FORWARD, and it is the landed one: the -119
    meta-forward item's own helpers build the tiny root and its runner on ``meta``
    and drive the two capture entry points, with nothing stood down, so the seam
    dispatches for real and the vendor's frontend is asked to specialise this
    kernel. That venue refused the pre-change body on this host, at
    ``functional/mhc/hyper_connection.py``, with the compiler naming the
    comprehension's own line. A direct call on the wrapped kernel does NOT reach
    it: the wrapper answers such a call from the simulator, which interprets the
    body instead of compiling it, and reports nothing.

    Under ``GLM53F_123_EXPECT_BASE=1`` the refusal is the expected reading and is
    required, with the vendor's own two fragments and the seam as the site. On the
    candidate the requirement is that no specialisation refusal appears. Anything
    else the forward says is PRINTED WHOLE and not judged: a second refused
    construct behind the first would be a result, and this item's job is to show
    it rather than to hide it behind a green.
    """
    for leg in LEGS:
        standing = meta119._standing_arm()
        _, runner = meta119._meta_root_and_runner()
        completed, error = True, None
        try:
            meta119._extract(runner, leg)
        except Exception as caught:  # noqa: BLE001 - the vendor's own failure is the reading
            completed, error = False, caught
        message = "" if error is None else " ".join(str(error).split())
        site = "none" if error is None else meta119._site_of(error)
        print(f"HC123|I1_META_VERBATIM|{leg}|{message}", flush=True)
        _emit(
            "I1_META_FORWARD",
            leg=leg, expect_base=int(EXPECT_BASE),
            completed="yes" if completed else "no",
            route="real" if not standing else "arm:" + ",".join(standing),
            venue="nki_simulator" if os.environ.get("NKI_SIMULATOR") == "1" else "device",
            site=site,
            names_the_refusal=int(WANT_REFUSAL in message),
            names_the_diagnostic=int(WANT_DIAGNOSTIC in message),
            error_type="none" if error is None else type(error).__name__,
        )
        assert not standing, f"leg={leg}: another item's stand-in is still bound: {standing}"
        if EXPECT_BASE:
            assert not completed, f"leg={leg}: the pre-change body completed the forward"
            assert WANT_REFUSAL in message, message
            assert WANT_DIAGNOSTIC in message, message
            assert site.startswith(SEAM_MODULE), site
        else:
            assert WANT_REFUSAL not in message, message
            assert WANT_DIAGNOSTIC not in message, message


# --------------------------------------------------------------------------- #
# ITEM 2 -- identical numbers, to the bit, against two independent references.   #
# --------------------------------------------------------------------------- #
def test_the_body_is_bit_identical_to_the_one_it_replaced_and_to_the_oracle() -> None:
    """Equality, not closeness, on the simulator venue at three tile shapes.

    Against the refused body the claim is BIT-IDENTITY: the rewrite changed which
    python forms express the loop, not the numbers nor their order. Against the
    oracle it is the pass-through pattern, because on a random fixture a scalar
    accumulation and an ``einsum`` contraction round differently by construction.
    """
    for rows in IDENTITY_SHAPES:
        x, residual, post_layer_mix, comb_res_mix = _inputs(rows, IDENTITY_HIDDEN)

        reset_dispatch_counters()
        got = hyper_connection_combine(x, residual, post_layer_mix, comb_res_mix)
        nki_n, fallback_n = dispatch_counters()
        replaced = wrap_nki(_refused_reference_kernel)(
            x=x,
            residual=residual,
            post_layer_mix=post_layer_mix,
            comb_res_mix=comb_res_mix,
        )

        got32 = got.to(torch.float32)
        replaced32 = replaced.to(torch.float32)
        differing = int((got32 != replaced32).sum())
        _emit(
            "I2_EQUALS_THE_BODY_IT_REPLACED",
            rows=rows, hidden=IDENTITY_HIDDEN, entries=got32.numel(),
            nki_dispatch=nki_n, torch_fallback=fallback_n,
            differing_entries=differing, bit_exact=torch.equal(got32, replaced32),
            replaced_absmax=f"{float(replaced32.abs().max()):.6e}",
        )
        assert (nki_n, fallback_n) == (1, 0), (rows, nki_n, fallback_n)
        assert float(replaced32.abs().max()) > 0.0, f"rows={rows}: reference all zero"
        assert torch.equal(got32, replaced32), (
            f"rows={rows}: {differing} entries differ from the body this replaced"
        )

        zero_post, ident = _pass_through_weights(rows)
        reset_dispatch_counters()
        served = hyper_connection_combine(x, residual, zero_post, ident)
        oracle = hyper_connection_torch_oracle(x, residual, zero_post, ident)
        served32 = served.to(torch.float32)
        _emit(
            "I2_EQUALS_THE_ORACLE_EXACTLY",
            rows=rows, hidden=IDENTITY_HIDDEN,
            differing_from_oracle=int((served32 != oracle).sum()),
            differing_from_residual=int((served32 != residual).sum()),
            bit_exact_oracle=torch.equal(served32, oracle),
            bit_exact_residual=torch.equal(served32, residual),
            residual_absmax=f"{float(residual.abs().max()):.6e}",
            x_absmax=f"{float(x.abs().max()):.6e}",
        )
        assert float(residual.abs().max()) > 0.0, "the residual fixture is all zero"
        assert float(x.abs().max()) > 0.0, (
            "x is all zero, so an ignored post term could not have shown here"
        )
        assert torch.equal(served32, oracle), f"rows={rows}: the oracle disagrees"
        assert torch.equal(served32, residual), f"rows={rows}: not the residual"


# --------------------------------------------------------------------------- #
# ITEM 3 -- one NKI dispatch per call, and no torch fallback.                   #
# --------------------------------------------------------------------------- #
def test_the_route_is_one_nki_dispatch_per_call() -> None:
    """Two calls, read after each: ``1`` then ``2``, with the fallback at zero.

    A body walking the token axis from the HOST would read one dispatch per tile;
    one answering in torch would read a fallback.
    """
    x, residual, post_layer_mix, comb_res_mix = _inputs(2048, 8)

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        hyper_connection_combine(x, residual, post_layer_mix, comb_res_mix)
        after_first = dispatch_counters()
        hyper_connection_combine(x, residual, post_layer_mix, comb_res_mix)
        after_second = dispatch_counters()
    tiles = -(-2048 // PARTITION_MAX)
    _emit(
        "I3_ROUTE",
        rows=2048, tiles=tiles, simulate_kernel_calls=sim.calls,
        after_first_call="/".join(str(v) for v in after_first),
        after_second_call="/".join(str(v) for v in after_second),
        a_host_tile_loop_would_read=tiles * 2,
    )
    assert after_first == (1, 0), after_first
    assert after_second == (2, 0), after_second
    assert sim.calls == 2, (
        f"nki.simulator.simulate_kernel ran {sim.calls} times for two calls, "
        f"declared 2; a host loop over tiles would read {tiles * 2}"
    )
