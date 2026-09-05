# SPDX-License-Identifier: Apache-2.0
"""Tier N acceptance for ``inc-glm53f-050`` -- the Q-side FWHT, as COVERAGE on a landed seam.

WHAT THIS FILE IS, AND WHY IT AUTHORS NO KERNEL:

The Q-side transform of the origin's ``fwht128_quant_fp8`` is the SAME 128-point butterfly with the
SAME exact ``1/sqrt(128)`` scale as the K side. ``inc-glm53f-047`` already landed that transform as a
complete public seam -- ``dsa_hadamard128``, its gate, its ``@nki.jit`` kernel, its counters and its
torch oracle. So there is no kernel-class work left here: writing one would be a second copy of
identical math. This increment is NON-KERNEL-CLASS by design ruling (entry s), and P13 is untouched
because the kernel-class work already sits in NKI.

WHAT IS ACTUALLY UNCOVERED AT THE BASE, which is what this file measures:

Every call to the seam in the landed suite passes ``torch.eye(128)`` -- ONE row count, ONE dtype,
ONE input, plus one wrong-width refusal. Three things follow that nothing at this base reads:

* the seam has never been handed a NON-IDENTITY input, so no reading distinguishes the transform
  from anything else that maps ``I`` to ``H/sqrt(128)``;
* the seam has never been handed a row count other than 128, so the kernel's tile LOOP
  (``kpool_hadamard.py:407-408``, ``ceil(n_rows / pmax)`` with a narrowed final tile) has never run
  more than once, and its partial-tile branch has never run at all;
* the involution property -- the transform applied twice returns the input -- is read NOWHERE under
  ``functional/dsa``.

THE ORACLE IS BUILT HERE AND BORROWS NOTHING FROM THE MODULE UNDER TEST. The landed sibling
``test_kpool_hadamard.py`` imports ``hadamard_matrix`` for its oracle; this file deliberately does
not, on the design ruling's words -- "BUILT IN THE TEST (Sylvester construction, never the module's
own ``hadamard_matrix``)". The scale is recomputed from ``math.sqrt`` rather than imported for the
same reason. If the module's ``HADAMARD_SCALE`` literal were wrong, the direct cases below would
fail, which is exactly what an independent oracle is for.

A NOTE ON THE INVOLUTION'S NORMALISATION, because the obvious form is the wrong one. For the
UNSCALED transform ``x H H == 128 x``, so that convention needs a division by ``n``. The landed seam
and the origin both INCLUDE the ``1/sqrt(128)`` scale, and for the scaled transform the identity is
``seam(seam(x)) == x`` with NO division at all::

    (x H / sqrt(128)) H / sqrt(128) = x H^2 / 128 = x (128 I) / 128 = x

Dividing by ``n`` a second time would yield ``x / 128`` and could not pass. That was measured before
this file was written, in pure python with no torch and no device, at
``../increments/probe-050-involution-normalisation.out``: the scaled form with no division reads
4.4e-16 and the same form divided by ``n`` reads 3.544 against a 1e-5 bound.

ONE ITEM PER COUNTED CONJUNCT and no ``parametrize`` (plan section 6, rules 4b and 6), so a failure
names the reading that failed instead of a parameter id. Counters are reset at the START of each
declared case and read at its END (section 4b's per-case convention).

EVERY COUNTED ZERO HAS A CONTROL THAT FIRES (plan D1.5). This file counts two zeros -- the
torch-fallback counter on the NKI route, and the NKI-dispatch counter across the oracle -- and each
has a companion item that makes the same counter move. A zero from a counter that cannot move is not
a measurement.

THE ENVIRONMENT IS PINNED IN THE INVOCATION, NEVER IN A FIXTURE (plan D2): this file is run under
``VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 NEURON_PLATFORM_TARGET_OVERRIDE=trn2``.
Nothing here reads or sets an environment variable.

A NOTE ON WHAT THE SIMULATOR PROVES. ``NKI_SIMULATOR=1`` does not run the MLIR verifier, so a pass
here is evidence about VALUES and never about COMPILABILITY. This increment adds no ``@nki.jit``
kernel, so there is no new compilability question for it to answer.
"""

from __future__ import annotations

import math

import torch

from vllm_neuron.functional.dsa.kpool_hadamard import (
    INDEX_HEAD_DIM,
    _dsa_hadamard128_torch,
    can_run_dsa_hadamard128,
    dsa_hadamard128,
    kpool_hadamard_dispatch_counters,
    reset_kpool_hadamard_dispatch_counters,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

RTOL = 1e-2
ATOL = 1e-5
"""The Acceptance bullet's tolerance for the seam-versus-oracle equivalence, verbatim."""

INVOLUTION_ATOL = 1e-5
"""The Acceptance bullet's bound for ``seam(seam(x)) == x``, verbatim. No division: see the module
docstring."""

D1_ROWS = 128
"""CASE D1: exactly ONE full 128-partition tile."""

D2_ROWS = 37
"""CASE D2: a PARTIAL tile, and not a multiple of 128."""

D3_ROWS = 260
"""CASE D3: TWO full tiles plus a partial of 4, so the kernel's tile loop runs three times."""

# ``_dsa_hadamard128_torch`` is imported for ONE control item and nothing else. It is the only way
# to make the torch-fallback zero falsifiable from inside the test: the gate cannot be turned off
# here (the environment is pinned in the invocation), and the seam RAISES on a wrong width rather
# than falling back to torch. Reading a private name does not modify the module, and the module is
# not modified by this increment.


# ---------------------------------------------------------------------------------------------
# The oracle. Built here, from nothing the module under test provides.
# ---------------------------------------------------------------------------------------------


def _sylvester(n: int) -> torch.Tensor:
    """The UNNORMALISED Sylvester ``H_n``, built by doubling. Entries are all ``+1`` or ``-1``.

    Built rather than transcribed so that no 128x128 literal has to be trusted, and checked two
    independent ways by the items below: against orthogonality, and against the closed form
    ``H[i, j] == (-1) ** popcount(i & j)``.
    """
    if n <= 0 or n & (n - 1):
        raise ValueError(f"n must be a positive power of two; got {n}")
    h = torch.ones((1, 1), dtype=torch.float32)
    while h.shape[0] < n:
        h = torch.cat((torch.cat((h, h), dim=1), torch.cat((h, -h), dim=1)), dim=0)
    return h


def _closed_form_sylvester(n: int) -> torch.Tensor:
    """The same matrix from its closed form, ``H[i, j] = (-1) ** popcount(i & j)``.

    This is a genuinely independent second definition -- a per-element formula rather than a
    recursive doubling -- so agreement between the two is a reading about the construction and not a
    restatement of it.
    """
    idx = torch.arange(n, dtype=torch.int64)
    parity = torch.tensor(
        [[bin(int(i) & int(j)).count("1") & 1 for j in idx] for i in idx],
        dtype=torch.float32,
    )
    return 1.0 - 2.0 * parity


def _oracle(x: torch.Tensor) -> torch.Tensor:
    """``FWHT_128(row) / sqrt(128)`` for every row, as a naive matrix multiply. Dispatches NO kernel.

    The scale is recomputed from ``math.sqrt`` rather than imported, so the module's own
    ``HADAMARD_SCALE`` literal is something this file MEASURES rather than assumes. ``.t()`` is
    cosmetic -- the Sylvester matrix is symmetric -- and is kept because the origin writes it that
    way at ``kpool_compress.py:37-45``.
    """
    scale = 1.0 / math.sqrt(int(x.shape[1]))
    rotated = x.float() @ _sylvester(int(x.shape[1])).t()
    return (rotated * scale).to(x.dtype)


def _rows(n_rows: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
    """One case's input: ``[n_rows, 128]`` of NON-IDENTITY random values at the declared dtype.

    Non-identity is the point. On ``I`` the seam's output IS the transform matrix, so an identity
    reading cannot separate the transform from any other map with the same image of ``I``.
    """
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn((n_rows, INDEX_HEAD_DIM), generator=gen, dtype=torch.float32)
    return x.to(dtype)


def _emit(tag: str, **values: object) -> None:
    """Print one MACHINE-READABLE reading line, for the driver to re-check independently.

    The landed pattern at ``test_ragged_pack.py`` and ``test_kpool_hadamard.py``: the test asserts,
    and then PRINTS the value it asserted on, so the driver owning the transcript can check the same
    number without trusting the assertion. Requires ``-s``, which the declared invocation passes.
    """
    body = " ".join(f"{k}={v}" for k, v in values.items())
    print(f"[{tag}] {body}")


def _diff(got: torch.Tensor, expected: torch.Tensor) -> tuple[float, int]:
    """``(max_abs_diff, differing_elements)`` between two tensors, both as plain python numbers."""
    a, b = got.float(), expected.float()
    return (a - b).abs().max().item(), int((a != b).sum().item())


def _direct_case(n_rows: int, dtype: torch.dtype, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one declared direct case. Returns ``(got, expected)``.

    The oracle is evaluated BEFORE the reset, so the reset cannot be credited with hiding an oracle
    dispatch: the counters are zeroed after the oracle has already run, and the reading taken at the
    end therefore belongs to the seam call alone.
    """
    x = _rows(n_rows, dtype, seed)
    expected = _oracle(x)
    reset_kpool_hadamard_dispatch_counters()
    got = dsa_hadamard128(x)
    return got, expected


# ---------------------------------------------------------------------------------------------
# The gate. Nothing below means anything if the NKI route is not the route taken.
# ---------------------------------------------------------------------------------------------


def test_gate_can_run_kernel_is_true() -> None:
    """``can_run_kernel()`` is True, so every case below takes the NKI route and not the oracle."""
    assert can_run_kernel() is True


def test_gate_admits_case_d1_128_rows_bf16() -> None:
    """The seam's gate admits D1. It tests ``ndim`` and the head dimension only -- no dtype clause."""
    assert can_run_dsa_hadamard128(_rows(D1_ROWS, torch.bfloat16, seed=1)) is True


def test_gate_admits_case_d2_37_rows_fp32() -> None:
    """The seam's gate admits D2. There is no row-count clause, so a partial tile is admitted."""
    assert can_run_dsa_hadamard128(_rows(D2_ROWS, torch.float32, seed=2)) is True


def test_gate_admits_case_d3_260_rows_bf16() -> None:
    """The seam's gate admits D3, the multi-tile case."""
    assert can_run_dsa_hadamard128(_rows(D3_ROWS, torch.bfloat16, seed=3)) is True


# ---------------------------------------------------------------------------------------------
# The oracle's own matrix, checked two independent ways before anything is compared against it.
# ---------------------------------------------------------------------------------------------


def test_the_oracle_matrix_is_orthogonal() -> None:
    """``H @ H.T == 128 * I`` exactly. An oracle built on a wrong matrix would measure nothing."""
    h = _sylvester(INDEX_HEAD_DIM)
    product = h @ h.t()
    expected = float(INDEX_HEAD_DIM) * torch.eye(INDEX_HEAD_DIM, dtype=torch.float32)
    max_abs, differing = _diff(product, expected)
    _emit(
        "oracle-selfcheck",
        check="orthogonality",
        n=INDEX_HEAD_DIM,
        max_abs_diff=f"{max_abs:.6e}",
        differing_elements=differing,
    )
    assert differing == 0


def test_the_oracle_matrix_matches_the_closed_form() -> None:
    """The doubling construction equals ``(-1) ** popcount(i & j)``, element for element.

    Two independent definitions of the same matrix -- a recursion and a per-element formula. This is
    what lets the oracle stand without borrowing the module's ``hadamard_matrix``.
    """
    built = _sylvester(INDEX_HEAD_DIM)
    closed = _closed_form_sylvester(INDEX_HEAD_DIM)
    max_abs, differing = _diff(built, closed)
    _emit(
        "oracle-selfcheck",
        check="closed_form",
        n=INDEX_HEAD_DIM,
        max_abs_diff=f"{max_abs:.6e}",
        differing_elements=differing,
        entries_are_pm_one=int(bool(((built.abs() == 1.0).all()).item())),
    )
    assert differing == 0
    assert bool(((built.abs() == 1.0).all()).item()) is True


# ---------------------------------------------------------------------------------------------
# The direct arm: 3/3 declared cases over row count and dtype. Two counted conjuncts each.
# D1 = 128 rows bf16 (one full tile), D2 = 37 rows fp32 (a partial tile),
# D3 = 260 rows bf16 (two full tiles plus a partial of four).
# ---------------------------------------------------------------------------------------------


def test_case_d1_128_rows_bf16_matches_the_oracle() -> None:
    """CASE D1/3: 128 rows of non-identity bf16, one full tile, against the in-test oracle."""
    got, expected = _direct_case(D1_ROWS, torch.bfloat16, seed=101)
    max_abs, differing = _diff(got, expected)
    _emit(
        "acceptance",
        case="d1",
        n_rows=D1_ROWS,
        dtype="bfloat16",
        rows=got.shape[0],
        cols=got.shape[1],
        max_abs_diff=f"{max_abs:.6e}",
        differing_elements=differing,
        rtol=RTOL,
        atol=ATOL,
    )
    assert tuple(got.shape) == (D1_ROWS, INDEX_HEAD_DIM)
    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got.float(), expected.float(), rtol=RTOL, atol=ATOL)


def test_case_d1_route() -> None:
    """CASE D1/3 route: exactly one NKI dispatch and zero torch fallbacks."""
    x = _rows(D1_ROWS, torch.bfloat16, seed=101)
    reset_kpool_hadamard_dispatch_counters()
    gate = can_run_dsa_hadamard128(x)
    dsa_hadamard128(x)
    nki, fallback = kpool_hadamard_dispatch_counters()
    _emit("route-predicate", case="d1", nki_dispatch=nki, torch_fallback=fallback, gate=gate)
    assert (nki, fallback) == (1, 0)
    assert gate is True


def test_case_d2_37_rows_fp32_matches_the_oracle() -> None:
    """CASE D2/3: 37 rows of non-identity fp32 -- a PARTIAL tile, and not a multiple of 128."""
    got, expected = _direct_case(D2_ROWS, torch.float32, seed=102)
    max_abs, differing = _diff(got, expected)
    _emit(
        "acceptance",
        case="d2",
        n_rows=D2_ROWS,
        dtype="float32",
        rows=got.shape[0],
        cols=got.shape[1],
        max_abs_diff=f"{max_abs:.6e}",
        differing_elements=differing,
        rtol=RTOL,
        atol=ATOL,
    )
    assert tuple(got.shape) == (D2_ROWS, INDEX_HEAD_DIM)
    assert got.dtype == torch.float32
    torch.testing.assert_close(got.float(), expected.float(), rtol=RTOL, atol=ATOL)


def test_case_d2_route() -> None:
    """CASE D2/3 route: exactly one NKI dispatch and zero torch fallbacks."""
    x = _rows(D2_ROWS, torch.float32, seed=102)
    reset_kpool_hadamard_dispatch_counters()
    gate = can_run_dsa_hadamard128(x)
    dsa_hadamard128(x)
    nki, fallback = kpool_hadamard_dispatch_counters()
    _emit("route-predicate", case="d2", nki_dispatch=nki, torch_fallback=fallback, gate=gate)
    assert (nki, fallback) == (1, 0)
    assert gate is True


def test_case_d3_260_rows_bf16_matches_the_oracle() -> None:
    """CASE D3/3: 260 rows of non-identity bf16 -- the kernel's tile loop runs three times.

    This is the first reading anywhere of that loop running more than once, and of its narrowed
    final tile: ``kpool_hadamard.py:407-408`` iterates ``ceil(260 / 128) == 3`` times with
    ``rows = min(128, 260 - t * 128)``, so the last tile carries four rows.
    """
    got, expected = _direct_case(D3_ROWS, torch.bfloat16, seed=103)
    max_abs, differing = _diff(got, expected)
    _emit(
        "acceptance",
        case="d3",
        n_rows=D3_ROWS,
        dtype="bfloat16",
        rows=got.shape[0],
        cols=got.shape[1],
        tiles=(D3_ROWS + INDEX_HEAD_DIM - 1) // INDEX_HEAD_DIM,
        last_tile_rows=D3_ROWS - (D3_ROWS // INDEX_HEAD_DIM) * INDEX_HEAD_DIM,
        max_abs_diff=f"{max_abs:.6e}",
        differing_elements=differing,
        rtol=RTOL,
        atol=ATOL,
    )
    assert tuple(got.shape) == (D3_ROWS, INDEX_HEAD_DIM)
    torch.testing.assert_close(got.float(), expected.float(), rtol=RTOL, atol=ATOL)


def test_case_d3_route() -> None:
    """CASE D3/3 route: exactly one NKI dispatch for the whole multi-tile call, zero fallbacks.

    One dispatch and not three: the tile loop is INSIDE the kernel, so a per-tile count here would
    mean the seam had been restructured.
    """
    x = _rows(D3_ROWS, torch.bfloat16, seed=103)
    reset_kpool_hadamard_dispatch_counters()
    gate = can_run_dsa_hadamard128(x)
    dsa_hadamard128(x)
    nki, fallback = kpool_hadamard_dispatch_counters()
    _emit("route-predicate", case="d3", nki_dispatch=nki, torch_fallback=fallback, gate=gate)
    assert (nki, fallback) == (1, 0)
    assert gate is True


# ---------------------------------------------------------------------------------------------
# The involution arm: the one reading that needs no oracle at all.
# ---------------------------------------------------------------------------------------------


def test_involution_the_transform_applied_twice_returns_the_input() -> None:
    """``seam(seam(x)) == x`` within 1e-5, with NO division. fp32, 37 rows.

    This arm is ORACLE-INDEPENDENT: it compares the seam against its own input, so it would still
    hold if every Hadamard matrix in this file were wrong. It is the second arm of the acceptance
    and not a substitute for the first.

    fp32 because the bound is 1e-5 and bf16 carries about three decimal digits, so no bf16 reading
    could clear it. 37 rows because that is D2's row count, and ``n_rows`` is a compile-time
    constant in the kernel -- reusing it adds no third compile.
    """
    x = _rows(D2_ROWS, torch.float32, seed=104)
    reset_kpool_hadamard_dispatch_counters()
    once = dsa_hadamard128(x)
    twice = dsa_hadamard128(once)
    max_abs, differing = _diff(twice, x)
    _emit(
        "acceptance",
        case="involution",
        n_rows=D2_ROWS,
        dtype="float32",
        max_abs_diff=f"{max_abs:.6e}",
        differing_elements=differing,
        atol=INVOLUTION_ATOL,
        divided_by_n="no",
        input_max_abs=f"{x.abs().max().item():.6e}",
    )
    assert tuple(twice.shape) == tuple(x.shape)
    torch.testing.assert_close(twice.float(), x.float(), rtol=0.0, atol=INVOLUTION_ATOL)


def test_involution_route_is_exactly_two_dispatches() -> None:
    """The involution case applies the transform TWICE, so its window reads exactly two dispatches.

    Two and not one is the reading that proves the second application went through the seam as well,
    rather than the first result being compared against itself.
    """
    x = _rows(D2_ROWS, torch.float32, seed=104)
    reset_kpool_hadamard_dispatch_counters()
    dsa_hadamard128(dsa_hadamard128(x))
    nki, fallback = kpool_hadamard_dispatch_counters()
    _emit("route-predicate", case="involution", nki_dispatch=nki, torch_fallback=fallback)
    assert (nki, fallback) == (2, 0)


def test_dispatch_total_over_the_declared_case_set_is_five() -> None:
    """The declared total: three direct dispatches plus the involution's two, in ONE reset window.

    The per-case items above each read their own window; this item reads the SUM the route predicate
    declares, so the two readings cannot disagree without one of them failing.
    """
    reset_kpool_hadamard_dispatch_counters()
    dsa_hadamard128(_rows(D1_ROWS, torch.bfloat16, seed=101))
    dsa_hadamard128(_rows(D2_ROWS, torch.float32, seed=102))
    dsa_hadamard128(_rows(D3_ROWS, torch.bfloat16, seed=103))
    dsa_hadamard128(dsa_hadamard128(_rows(D2_ROWS, torch.float32, seed=104)))
    nki, fallback = kpool_hadamard_dispatch_counters()
    _emit("route-predicate", case="declared_total", nki_dispatch=nki, torch_fallback=fallback)
    assert (nki, fallback) == (5, 0)


# ---------------------------------------------------------------------------------------------
# The two counted zeros, each with a control that makes the same counter move.
# ---------------------------------------------------------------------------------------------


def test_the_test_oracle_dispatches_zero_nki_kernels() -> None:
    """The oracle this file compares against runs NO NKI kernel.

    Without this reading the equivalence could be the kernel agreeing with itself.
    """
    x = _rows(D2_ROWS, torch.float32, seed=201)
    reset_kpool_hadamard_dispatch_counters()
    _oracle(x)
    _sylvester(INDEX_HEAD_DIM)
    nki, fallback = kpool_hadamard_dispatch_counters()
    _emit("control", case="oracle_dispatches_zero", nki_dispatch=nki, torch_fallback=fallback)
    assert (nki, fallback) == (0, 0)


def test_control_the_nki_dispatch_counter_does_move() -> None:
    """THE CONTROL for the zero above: the same counter, same instrument, reads one after a seam call.

    A zero from a counter that cannot move is not a measurement. This item reads zero and then one
    inside a single window, so the zero above is a reading about the oracle and not about the meter.
    """
    x = _rows(D2_ROWS, torch.float32, seed=201)
    reset_kpool_hadamard_dispatch_counters()
    _oracle(x)
    before, _ = kpool_hadamard_dispatch_counters()
    dsa_hadamard128(x)
    after, fallback = kpool_hadamard_dispatch_counters()
    _emit(
        "control",
        case="nki_counter_moves",
        before=before,
        after=after,
        torch_fallback=fallback,
    )
    assert before == 0
    assert after == 1


def test_the_torch_fallback_counter_is_zero_across_the_declared_cases() -> None:
    """No declared case falls back to torch: the fallback counter is exactly zero over the whole set.

    A pure-torch implementation of this transform would read zero NKI dispatches and a non-zero
    fallback, so this item and the dispatch total together are what a torch fallback could not pass.
    """
    reset_kpool_hadamard_dispatch_counters()
    dsa_hadamard128(_rows(D1_ROWS, torch.bfloat16, seed=101))
    dsa_hadamard128(_rows(D2_ROWS, torch.float32, seed=102))
    dsa_hadamard128(_rows(D3_ROWS, torch.bfloat16, seed=103))
    dsa_hadamard128(dsa_hadamard128(_rows(D2_ROWS, torch.float32, seed=104)))
    nki, fallback = kpool_hadamard_dispatch_counters()
    _emit("control", case="fallback_is_zero", nki_dispatch=nki, torch_fallback=fallback)
    assert fallback == 0
    assert nki == 5


def test_control_the_torch_fallback_counter_does_move() -> None:
    """THE CONTROL for the zero above: the module's own torch path increments the fallback counter.

    The fallback cannot be reached through the seam from here -- the environment is pinned in the
    invocation so the gate stays True, and the seam RAISES on a wrong width instead of falling back.
    So the control calls the module's torch oracle directly, which is the one thing that moves this
    counter. It proves the counter is live; it makes no claim about the seam.
    """
    x = _rows(D2_ROWS, torch.float32, seed=202)
    reset_kpool_hadamard_dispatch_counters()
    before_nki, before_fallback = kpool_hadamard_dispatch_counters()
    _dsa_hadamard128_torch(x)
    after_nki, after_fallback = kpool_hadamard_dispatch_counters()
    _emit(
        "control",
        case="fallback_counter_moves",
        before_fallback=before_fallback,
        after_fallback=after_fallback,
        before_nki=before_nki,
        after_nki=after_nki,
    )
    assert (before_nki, before_fallback) == (0, 0)
    assert after_fallback == 1
    assert after_nki == 0
