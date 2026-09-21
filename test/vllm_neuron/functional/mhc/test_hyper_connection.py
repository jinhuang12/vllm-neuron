# SPDX-License-Identifier: Apache-2.0
"""Tests for the mHC combine kernel at ``hc_mult = 4``.

Two kinds of numeric case, and they arm each other rather than repeating each
other.

The tolerance case compares the kernel against a torch reference authored here.
That reference could carry the same ``i``/``j`` mistake as the kernel, so it is
corroborated two ways: against the module's own ``einsum`` oracle, which is an
independent spelling of the same convention, and by requiring that a transposed
mix does *not* match.

The pass-through case has no authored reference at all: with ``comb_res_mix = I``
and ``post_layer_mix = 0`` the expected output is the ``residual`` tensor itself,
and in fp32 that is bit-exact, because multiplying by exactly ``1.0`` and adding
exact ``0.0`` are both lossless. It catches an indexing error a tolerance would
absorb, but being symmetric it cannot see a transpose, which is why the two cases
are both here.

The kernel walks the token axis in tiles of ``nl.tile_size.pmax``, so the extents
below are chosen for their relationship to that tile height: shorter than one
tile, exactly one tile, one tile plus a single row, two tiles plus a remainder,
and many whole tiles.

Comparing simulated output against a torch reference would measure nothing if the
module took its torch path, because both sides would then be torch. So the numeric
cases also read the module's dispatch counters and count real
``nki.simulator.simulate_kernel`` calls, which is the vendor entry point and so
independent of the module's own counter.
"""

from __future__ import annotations

import importlib
import os

import pytest
import torch

import nki.simulator

from vllm_neuron.functional.mhc.hyper_connection import (
    MHC_STREAMS,
    PARTITION_MAX,
    HyperConnectionError,
    can_run_hyper_connection,
    dispatch_counters,
    hyper_connection_combine,
    hyper_connection_torch_oracle,
    kernel_identity,
    reset_dispatch_counters,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

#: The tiny case. ``S`` is read off ``MHC_STREAMS`` so the fixture and the model's
#: ``hc_mult`` cannot drift apart; ``H`` is a multiple of 256.
T = 64
S = MHC_STREAMS  # 4
H = 256

RTOL = 1e-2
ATOL = 1e-5

#: The pass-through and permutation cases recover their input exactly, so they
#: carry no relative slack at all.
EXACT_RTOL = 0.0

#: Token extents and hidden widths, one per relationship to the tile height:
#: shorter than a tile, exactly one tile, one tile plus a row, two tiles plus a
#: 44-row remainder, and sixteen whole tiles. The wide extents use a narrow hidden
#: width to keep the simulator inside the test timeout.
TILE_EXTENTS = (
    (7, 32),
    (PARTITION_MAX, 256),
    (PARTITION_MAX + 1, 256),
    (300, 32),
    (2048, 8),
)

_MODULE = "vllm_neuron.functional.mhc.hyper_connection"
_SINKHORN_MODULE = "vllm_neuron.functional.mhc.sinkhorn"


class _SimulatorCounter:
    """Counts real ``nki.simulator.simulate_kernel`` calls for the duration."""

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


def _assert_nki_ran(sim: _SimulatorCounter, expected: int, label: str) -> None:
    """Require that the NKI path, and only the NKI path, served the calls."""
    nki_dispatch, torch_fallback = dispatch_counters()
    assert nki_dispatch == expected, (
        f"{label}: the dispatch counter read {nki_dispatch}, expected {expected}; "
        f"one per call, not one per tile"
    )
    assert torch_fallback == 0, (
        f"{label}: the torch-fallback counter read {torch_fallback}; a fallback "
        f"would compare torch against torch"
    )
    assert can_run_kernel(torch.zeros(1)) is True, f"{label}: no device or simulator"
    assert sim.calls == expected, (
        f"{label}: nki.simulator.simulate_kernel ran {sim.calls} times, expected "
        f"{expected}; a numeric pass without a simulator call measures no kernel"
    )


def _inputs(seed: int = 29, rows: int = T, hidden: int = H):
    """The four tensors, fp32, deterministic by seed.

    ``comb_res_mix`` is row-stochastic, which is what a Sinkhorn stage hands the
    combine, and asymmetric, without which an ``i``/``j`` transpose would be
    invisible. ``x`` and ``residual`` are signed, so a sign error shows.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn((rows, hidden), generator=g, dtype=torch.float32)
    residual = torch.randn((rows, S, hidden), generator=g, dtype=torch.float32)
    post_layer_mix = torch.rand((rows, S, 1), generator=g, dtype=torch.float32)
    comb_res_mix = torch.softmax(
        torch.randn((rows, S, S), generator=g, dtype=torch.float32), dim=-1
    )
    return x, residual, post_layer_mix, comb_res_mix


def _pass_through_weights(rows: int = T):
    """``comb_res_mix = I_S`` and ``post_layer_mix = 0``, so ``out == residual``.

    Defined here rather than taken from the module, so the code under test is not
    on both sides of the comparison.
    """
    ident = torch.eye(S, dtype=torch.float32).expand(rows, S, S).contiguous()
    zero_post = torch.zeros((rows, S, 1), dtype=torch.float32)
    return zero_post, ident


def _bmm_reference(x, residual, post_layer_mix, comb_res_mix):
    """The combine written as ``bmm(comb.mT, residual)``, in fp32.

    A second spelling of the ``i``/``j`` convention, independent of the module's
    ``einsum`` oracle. The ``.mT`` is where the convention lives.
    """
    term2 = torch.bmm(comb_res_mix.mT.to(torch.float32), residual.to(torch.float32))
    return x.to(torch.float32).unsqueeze(-2) * post_layer_mix.to(torch.float32) + term2


def _errors(got, expected) -> tuple[float, float]:
    """``(max_abs, max_rel)`` between two tensors."""
    got = got.to(torch.float32)
    expected = expected.to(torch.float32)
    max_abs = float((got - expected).abs().max())
    max_rel = float(((got - expected).abs() / (expected.abs() + ATOL)).max())
    return max_abs, max_rel


def test_output_matches_the_torch_reference_on_the_tiny_case() -> None:
    """The kernel agrees with the ``bmm`` reference at ``rtol=1e-2``, ``atol=1e-5``."""
    x, residual, post_layer_mix, comb_res_mix = _inputs()
    assert tuple(residual.shape) == (T, S, H), tuple(residual.shape)

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        got = hyper_connection_combine(x, residual, post_layer_mix, comb_res_mix)
    _assert_nki_ran(sim, 1, "tolerance-case")

    expected = _bmm_reference(x, residual, post_layer_mix, comb_res_mix)
    assert float(expected.abs().max()) > 0.0, "the reference is all zero"
    assert tuple(got.shape) == (T, S, H), tuple(got.shape)
    torch.testing.assert_close(got.to(torch.float32), expected, rtol=RTOL, atol=ATOL)


def test_the_pass_through_pattern_returns_the_residual_exactly() -> None:
    """``comb = I`` and ``post = 0`` give back the residual tensor, bit for bit."""
    x, residual, _, _ = _inputs()
    zero_post, ident = _pass_through_weights()

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        got = hyper_connection_combine(x, residual, zero_post, ident)
    _assert_nki_ran(sim, 1, "pass-through-case")

    assert float(residual.abs().max()) > 0.0, "the residual fixture is all zero"
    assert float(x.abs().max()) > 0.0, (
        "x is all zero, so an ignored post term could not have shown here"
    )
    assert tuple(got.shape) == (T, S, H), tuple(got.shape)
    torch.testing.assert_close(
        got.to(torch.float32), residual, rtol=EXACT_RTOL, atol=ATOL
    )


def test_the_kernel_reads_the_mix_in_the_i_j_convention() -> None:
    """A reference built from the transposed mix must not match the kernel.

    The pass-through case cannot see a transpose, because the identity is
    symmetric, so this is where an ``i``/``j`` swap is caught. It only works if the
    fixture's mix is genuinely asymmetric, which is asserted first.
    """
    x, residual, post_layer_mix, comb_res_mix = _inputs()
    transposed = comb_res_mix.transpose(-1, -2).contiguous()
    assert not torch.equal(transposed, comb_res_mix), (
        "the fixture's mix is symmetric, so transposing it changed nothing"
    )

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        got = hyper_connection_combine(x, residual, post_layer_mix, comb_res_mix)
    _assert_nki_ran(sim, 1, "transpose-case")

    wrong = _bmm_reference(x, residual, post_layer_mix, transposed)
    _, max_rel = _errors(got, wrong)
    assert max_rel > RTOL, (
        f"transposing the mix moved the comparison by only {max_rel:.6e}, inside "
        f"rtol {RTOL}, so an i/j swap would be invisible"
    )
    with pytest.raises(AssertionError):
        torch.testing.assert_close(got.to(torch.float32), wrong, rtol=RTOL, atol=ATOL)


def test_a_cyclic_permutation_routes_the_streams_exactly() -> None:
    """``comb[i, j] = 1 iff j == (i + 1) mod S`` rolls the stream axis, bit for bit.

    An exact case that is also transpose-sensitive, which the identity pattern is
    not: the expected output is a roll of ``residual``, so it is again
    reference-free, but the matrix is asymmetric.
    """
    x, residual, _, _ = _inputs()
    zero_post, _ = _pass_through_weights()
    perm = torch.zeros((S, S), dtype=torch.float32)
    for i in range(S):
        perm[i, (i + 1) % S] = 1.0
    perm_b = perm.expand(T, S, S).contiguous()
    expected = torch.roll(residual, shifts=1, dims=1)

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        got = hyper_connection_combine(x, residual, zero_post, perm_b)
    _assert_nki_ran(sim, 1, "permutation-case")

    against_untouched, _ = _errors(got, residual)
    assert against_untouched > ATOL, (
        "the permutation produced the untouched residual, so the stream axis was "
        "not routed at all"
    )
    torch.testing.assert_close(
        got.to(torch.float32), expected, rtol=EXACT_RTOL, atol=ATOL
    )


def test_the_two_torch_spellings_of_the_combine_agree() -> None:
    """The module's ``einsum`` oracle equals ``bmm(comb.mT, residual)``.

    If these two disagreed, every numeric comparison here would rest on an
    ``i``/``j`` convention nobody had checked.
    """
    x, residual, post_layer_mix, comb_res_mix = _inputs()
    einsum_side = hyper_connection_torch_oracle(
        x, residual, post_layer_mix, comb_res_mix
    )
    bmm_side = _bmm_reference(x, residual, post_layer_mix, comb_res_mix)
    torch.testing.assert_close(einsum_side, bmm_side, rtol=0.0, atol=ATOL)


@pytest.mark.parametrize(("rows", "hidden"), TILE_EXTENTS)
def test_token_extents_across_tile_boundaries_match_the_reference(
    rows: int, hidden: int
) -> None:
    """Each tile relationship serves, matches the reference, and passes through exactly.

    A short last tile is where a padded tile or a mis-computed offset shows, so each
    extent also runs the pass-through pattern, whose expected output is the
    ``residual`` tensor itself. That is the strongest available reading that every
    output row received its own input row across a tile boundary. It is checked
    against the module's torch oracle too, which is lossless under this pattern.

    The dispatch count stays at one however many tiles the extent needs: tiling on
    the host would read one per tile, which no numeric comparison could tell apart.
    """
    tiles = -(-rows // PARTITION_MAX)
    x, residual, post_layer_mix, comb_res_mix = _inputs(rows=rows, hidden=hidden)

    assert can_run_hyper_connection(
        x, residual, post_layer_mix, comb_res_mix
    ) is True, f"the gate refused {rows} tokens"

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        got = hyper_connection_combine(x, residual, post_layer_mix, comb_res_mix)
    _assert_nki_ran(sim, 1, f"rows={rows} tiles={tiles}")

    expected = _bmm_reference(x, residual, post_layer_mix, comb_res_mix)
    assert float(expected.abs().max()) > 0.0, f"rows={rows}: the reference is all zero"
    assert tuple(got.shape) == (rows, S, hidden), tuple(got.shape)
    torch.testing.assert_close(got.to(torch.float32), expected, rtol=RTOL, atol=ATOL)

    zero_post, ident = _pass_through_weights(rows)
    reset_dispatch_counters()
    served = hyper_connection_combine(x, residual, zero_post, ident).to(torch.float32)
    assert dispatch_counters() == (1, 0), dispatch_counters()
    oracle = hyper_connection_torch_oracle(x, residual, zero_post, ident)
    assert torch.equal(served, residual), (
        f"rows={rows}: {int((served != residual).sum())} entries differ from the "
        f"residual tensor"
    )
    assert torch.equal(served, oracle), f"rows={rows}: the torch oracle disagrees"


def test_the_torch_fallback_is_taken_counted_and_correct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no device or simulator the torch path runs, is counted, and is right.

    This is what makes ``torch_fallback == 0`` in the numeric cases meaningful. The
    extent is above one tile, so the fallback is also shown to answer correctly
    where the kernel would tile.
    """
    rows, hidden = 300, 8
    x, residual, post_layer_mix, comb_res_mix = _inputs(rows=rows, hidden=hidden)
    monkeypatch.setitem(os.environ, "NKI_SIMULATOR", "0")
    assert can_run_kernel(torch.zeros(1)) is False, (
        "the gate did not flip with NKI_SIMULATOR=0, so this case is vacuous"
    )
    assert (
        can_run_hyper_connection(x, residual, post_layer_mix, comb_res_mix) is False
    )

    reset_dispatch_counters()
    with _SimulatorCounter() as sim:
        served = hyper_connection_combine(x, residual, post_layer_mix, comb_res_mix)

    assert dispatch_counters() == (0, 1), (
        f"expected 0 NKI dispatches and 1 torch fallback, got {dispatch_counters()}"
    )
    assert sim.calls == 0, f"the simulator ran {sim.calls} times with it disabled"
    assert torch.equal(
        served, hyper_connection_torch_oracle(x, residual, post_layer_mix, comb_res_mix)
    )


def test_dispatch_counters_accumulate_across_calls() -> None:
    """The counters are module-level state: they accumulate until reset.

    The layer forward reads a per-layer-call total, which a counter saturating at
    one could not supply.
    """
    x, residual, post_layer_mix, comb_res_mix = _inputs(rows=8, hidden=32)
    reset_dispatch_counters()
    assert dispatch_counters() == (0, 0)

    hyper_connection_combine(x, residual, post_layer_mix, comb_res_mix)
    assert dispatch_counters() == (1, 0)

    hyper_connection_combine(x, residual, post_layer_mix, comb_res_mix)
    assert dispatch_counters() == (2, 0), (
        f"expected (2, 0) after two dispatches, got {dispatch_counters()}"
    )

    reset_dispatch_counters()
    assert dispatch_counters() == (0, 0)


def test_the_two_mhc_modules_count_separately() -> None:
    """The combine's counters and the Sinkhorn module's are two numbers, not one.

    The layer forward enters both per call and asserts each ran once. A shared
    counter, or a reset that cleared both, could not tell "both ran once" from "one
    ran twice".
    """
    sinkhorn = importlib.import_module(_SINKHORN_MODULE)
    combine = importlib.import_module(_MODULE)

    x, residual, post_layer_mix, comb_res_mix = _inputs(rows=8, hidden=32)
    g = torch.Generator().manual_seed(29)
    affinity = torch.exp(
        torch.rand((8, S), generator=g, dtype=torch.float32) * 2.0 - 1.0
    )

    combine.reset_dispatch_counters()
    sinkhorn.reset_dispatch_counters()

    combine.hyper_connection_combine(x, residual, post_layer_mix, comb_res_mix)
    assert (combine.dispatch_counters(), sinkhorn.dispatch_counters()) == (
        (1, 0),
        (0, 0),
    ), "driving the combine moved the Sinkhorn module's counter"

    sinkhorn.sinkhorn_normalise(affinity)
    assert (combine.dispatch_counters(), sinkhorn.dispatch_counters()) == (
        (1, 0),
        (1, 0),
    ), "driving Sinkhorn moved the combine's counter, or did not move its own"

    combine.reset_dispatch_counters()
    assert (combine.dispatch_counters(), sinkhorn.dispatch_counters()) == (
        (0, 0),
        (1, 0),
    ), "resetting the combine cleared the Sinkhorn module's counter too"
    sinkhorn.reset_dispatch_counters()


def test_kernel_identity_names_this_modules_kernel() -> None:
    """``kernel_identity`` names the kernel in this module, not a vendor one."""
    assert kernel_identity() == (_MODULE, "hyper_connection_kernel")


def test_the_stream_count_and_tile_height_are_shared_with_sinkhorn() -> None:
    """Both mHC modules are sized by one ``MHC_STREAMS`` and one ``PARTITION_MAX``."""
    sinkhorn = importlib.import_module(_SINKHORN_MODULE)
    assert MHC_STREAMS == sinkhorn.MHC_STREAMS == 4
    assert PARTITION_MAX == sinkhorn.PARTITION_MAX == 128


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        ("x_rows", "expected [T, H]"),
        ("x_hidden", "expected [T, H]"),
        ("x_rank", "x must be 2-D [T, H]"),
        ("post_shape", "expected [T, S, 1]"),
        ("comb_shape", "expected [T, S, S]"),
        ("residual_rank", "residual must be 3-D [T, S, H]"),
    ],
)
def test_refuses_inadmissible_geometry_by_name(mutate: str, needle: str) -> None:
    """A rank or extent mismatch between arguments raises and names the offender.

    Refused rather than coerced or routed to torch. On the bare kernel these same
    shapes trap inside NKI or numpy with messages that name no argument.
    """
    x, residual, post_layer_mix, comb_res_mix = _inputs(rows=8, hidden=32)
    if mutate == "x_rows":
        x = x[:-1]
    elif mutate == "x_hidden":
        x = x[:, :-1]
    elif mutate == "x_rank":
        x = x.unsqueeze(0)
    elif mutate == "post_shape":
        post_layer_mix = post_layer_mix.squeeze(-1)
    elif mutate == "comb_shape":
        comb_res_mix = comb_res_mix[:, :, :-1]
    elif mutate == "residual_rank":
        residual = residual.reshape(8, -1)

    with pytest.raises(HyperConnectionError) as excinfo:
        can_run_hyper_connection(x, residual, post_layer_mix, comb_res_mix)
    assert needle in str(excinfo.value), f"[{mutate}] message was: {excinfo.value}"
