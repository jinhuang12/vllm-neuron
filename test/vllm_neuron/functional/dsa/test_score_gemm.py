# SPDX-License-Identifier: Apache-2.0
"""Tier N acceptance for ``inc-glm53f-046`` -- the DSA indexer's score GEMM.

WHAT THIS FILE MEASURES, and the shape of the argument it makes.

The kernel is correct only if it agrees with an independent statement of the same function, so that
agreement IS the measurement. The independent statement lives HERE, in the test, written as three
whole-tensor torch steps (``einsum``, ``clamp``, weighted ``sum``) that touch no seam in the module
under test. That matters: the oracle must dispatch ZERO NKI kernels, or the comparison would be the
kernel against itself.

THE ONE THING THAT COULD BE SILENTLY WRONG IS THE ORDER OF OPERATIONS, so this file carries a control
for it. Rectifying AFTER the head reduction is a different function from rectifying before it, and
both are one line of torch. The control computes the wrong order on the SAME fixture and asserts the
two disagree by a wide margin. Without it, an agreeing kernel would only be evidence that the test and
the kernel share an assumption.

ONE ITEM PER COUNTED CONJUNCT and no ``parametrize`` (plan section 6, rules 4b and 6), so a failure
names the reading that failed instead of a parameter id. Counters are reset at the START of each
declared case and read at its END (section 4b's per-case convention).

THE DECLARED CASE SET IS EXACTLY TWO, per the lead's ruling recorded in ``scope-lap-100.md``: the
tolerance case and the orthonormal case, one dispatch each, for a declared total of 2. The two
tile-boundary items are SUPPLEMENTARY: each reads one dispatch in its own reset window, and neither is
counted toward the declared total. Their names begin with ``test_supplementary_`` so the distinction
is visible in the transcript and not only here.

EVERY COUNTED ZERO HAS A CONTROL THAT FIRES (plan D1.5). This file counts three zeros -- the NKI
dispatch counter across the torch oracle, the torch-fallback counter on the NKI route, and the count
of MX primitive calls in the module source -- and each has a companion item that makes the same reader
produce a nonzero. A zero from a reader that cannot move is not a measurement.

THE ENVIRONMENT IS PINNED IN THE INVOCATION, NEVER IN A FIXTURE (plan D2): this file is run under
``VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 NEURON_PLATFORM_TARGET_OVERRIDE=trn2``.
Nothing here reads or sets an environment variable.

WHAT THE SIMULATOR PROVES, AND WHAT IT DOES NOT. ``NKI_SIMULATOR=1`` does not run the MLIR verifier,
so a pass here is evidence about VALUES and never about COMPILABILITY. Compilability is settled
separately by this increment's capture leg with the verifier on
(``probe-046-mechanism-r2b-host.out``, which reads ``ARM_A2_ON_GRAPH_HLO_COUNT_WITH_VERIFIER|1
verifier=ON`` for the bf16 chain, with an int32-operand refusal as its firing control).
"""

from __future__ import annotations

import pathlib

import pytest
import torch

import nki.language as nl

from vllm_neuron.functional.dsa import score_gemm as mod
from vllm_neuron.functional.dsa.score_gemm import (
    CAND_TILE,
    CONTRACTION_TILE,
    INDEX_HEAD_DIM,
    TOKEN_TILE,
    ScoreGemmError,
    _dsa_score_gemm_torch,
    can_run_dsa_score_gemm,
    dsa_score_gemm,
    reset_score_gemm_dispatch_counters,
    score_gemm_dispatch_counters,
    score_gemm_kernel_identity,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

RTOL = 1e-2
ATOL = 1e-5
"""The Acceptance bullet's tolerance for the score-versus-reference reading, verbatim."""

ORTHO_ATOL = 1e-5
"""The Acceptance bullet's tolerance for the orthonormal-input reading, verbatim.

Taken with ``rtol=0`` so the reading is an ABSOLUTE one. That is the point of the case: on an
orthonormal basis the true off-diagonal score is exactly zero, and a relative tolerance around zero
would accept anything.
"""

BF16_GRAM_BOUND = 4e-3
"""How far the bf16 Gram matrix may sit from the identity, DERIVED rather than observed.

``c = bfloat16(1/sqrt(128))``. bf16 keeps 8 mantissa bits, so ``c`` carries a relative error up to
``2 ** -9``, and ``c * c`` up to twice that -- ``3.90625e-3``. The Gram diagonal is ``128 * c * c``,
so it may miss 1 by that much. This bound comes from the FORMAT and not from the measurement: the
pre-authoring probe happened to read ``2.136e-04``, which is well inside it, and if the bound had been
fitted to that reading it would have certified nothing.

THIS IS A CORRECTED PREDICTION AND IT IS WORTH SAYING SO. ``predictions-046-mechanism-r2.txt``
predicted this reading under ``1e-5``, and the probe falsified it. The prediction was wrong because it
assumed exact representation; the kernel was never implicated. The OFF-diagonal, which is what makes
the case non-vacuous, did read exactly zero.
"""

HEADS = 4
"""Heads per declared case. Small on purpose: the head axis is a plain loop bound in the kernel, so a
larger count would multiply simulator time without adding a distinct reading. The checkpoint's own
``index_n_heads`` is 32 and is asserted as a recorded constant, not exercised as a shape here."""


def _emit(tag: str, **values: object) -> None:
    """Print one MACHINE-READABLE reading line, for the driver to re-check independently.

    The pattern is the landed one at ``test_ragged_pack.py``: the item asserts, and then PRINTS the
    value it asserted on, so the driver that owns the transcript can check the same number without
    trusting this file's own verdict.
    """
    body = " ".join(f"{k}={v}" for k, v in values.items())
    print(f"S046|{tag}|{body}", flush=True)


def _diff(got: torch.Tensor, want: torch.Tensor) -> tuple[float, int]:
    """``(max_abs_diff, differing_element_count)`` in fp32. Reported whether or not the item passes."""
    d = (got.float() - want.float()).abs()
    return (float(d.max().item()), int((d > 0).sum().item()))


def _random_inputs(
    tokens: int, cands: int, heads: int = HEADS, seed: int = 7
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One case's three inputs at the reference's dtypes: bf16 query and key, fp32 weights.

    THE WEIGHTS DELIBERATELY STRADDLE ZERO. If every weight were positive, ``relu(w * s)`` and
    ``w * relu(s)`` would agree and the order control below could not fire. ``randn`` gives both
    signs, which is also what the upstream weights carry -- they are a learned projection, not a
    magnitude.
    """
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn((tokens, heads, INDEX_HEAD_DIM), generator=gen, dtype=torch.float32).to(
        torch.bfloat16
    )
    k = torch.randn((cands, INDEX_HEAD_DIM), generator=gen, dtype=torch.float32).to(torch.bfloat16)
    weights = torch.randn((tokens, heads), generator=gen, dtype=torch.float32)
    return q, k, weights


def _hadamard_rows(n: int) -> torch.Tensor:
    """``n`` rows of the Sylvester Hadamard basis, scaled to unit norm, in bf16.

    Built by doubling rather than by a library call so the basis is this file's own construction and
    not an import shared with the module under test.
    """
    h = torch.ones((1, 1), dtype=torch.float32)
    while h.shape[0] < INDEX_HEAD_DIM:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return (h[:n] * (INDEX_HEAD_DIM**-0.5)).to(torch.bfloat16)


def _orthonormal_inputs(
    tokens: int, cands: int, heads: int = HEADS, seed: int = 23
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Query and key drawn from the same orthonormal basis, so the true score is known exactly.

    Every off-diagonal per-head dot product sums 64 ``+c*c`` terms and 64 ``-c*c`` terms whose true
    value is exactly zero, and they cancel exactly in fp32 whatever order they are added in. That is
    what isolates accumulation order from the tolerance budget: a nonzero off-diagonal here is a real
    finding about the accumulation rather than a rounding allowance being used up.
    """
    basis = _hadamard_rows(INDEX_HEAD_DIM)
    q = basis[:tokens].unsqueeze(1).expand(tokens, heads, INDEX_HEAD_DIM).contiguous()
    k = basis[:cands].contiguous()
    gen = torch.Generator().manual_seed(seed)
    weights = torch.randn((tokens, heads), generator=gen, dtype=torch.float32)
    return q, k, weights


def _mismatched_head_dim_inputs(
    width: int = 64, seed: int = 41
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inputs whose head dimension is NOT 128, for the gate and the validator items.

    Built once and used by both, because two hand-built copies of the same malformed fixture can
    drift apart and then the two items would no longer be testing the same input.
    """
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn((4, HEADS, width), generator=gen).to(torch.bfloat16)
    k = torch.randn((4, width), generator=gen).to(torch.bfloat16)
    weights = torch.randn((4, HEADS), generator=gen, dtype=torch.float32)
    return q, k, weights


def _oracle(q: torch.Tensor, k: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """THE TEST'S OWN reference: dot, rectify, weight, sum. Dispatches NO NKI kernel.

    Transcribed from ``vllm/v1/attention/ops/rocm_aiter_mla_sparse.py:732-733`` at pin ``878631b6``,
    itself DeepGEMM's own torch reference, keeping its order exactly. The upstream per-key dequant
    scale is absent rather than dropped: a bf16 route quantises nothing, so there is no scale.
    """
    per_head = torch.einsum("mhd,nd->mhn", q.float(), k.float())
    return (per_head.clamp(min=0.0) * weights.float().unsqueeze(-1)).sum(dim=1)


def _oracle_wrong_order(q: torch.Tensor, k: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """THE CONTROL: rectify AFTER weighting and summing. A different function, used to prove the
    fixture can tell the two apart."""
    per_head = torch.einsum("mhd,nd->mhn", q.float(), k.float())
    return (per_head * weights.float().unsqueeze(-1)).sum(dim=1).clamp(min=0.0)


def _module_source() -> str:
    """The module's own committed bytes. Read from disk, so the reading is about what ships."""
    return pathlib.Path(mod.__file__).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# Declared case 1 of 2 -- the tolerance reading
# ---------------------------------------------------------------------------------------------


def test_declared_case_1_tolerance_acceptance():
    """DECLARED CASE 1: scores agree with the torch reference at rtol 1e-2, atol 1e-5."""
    reset_score_gemm_dispatch_counters()
    q, k, weights = _random_inputs(tokens=8, cands=12)
    got = dsa_score_gemm(q, k, weights)
    want = _oracle(q, k, weights)
    worst, differing = _diff(got, want)
    _emit(
        "CASE1_TOLERANCE",
        tokens=8,
        cands=12,
        heads=HEADS,
        worst_abs=f"{worst:.6e}",
        differing=differing,
        population=got.numel(),
        rtol=RTOL,
        atol=ATOL,
    )
    assert got.shape == (8, 12)
    assert got.dtype is torch.float32
    torch.testing.assert_close(got.float(), want, rtol=RTOL, atol=ATOL)


def test_declared_case_1_route_predicate_one_dispatch():
    """DECLARED CASE 1's route predicate, form R-1: gate True, 1 NKI dispatch, 0 torch fallbacks."""
    reset_score_gemm_dispatch_counters()
    q, k, weights = _random_inputs(tokens=8, cands=12)
    admitted = can_run_dsa_score_gemm(q, k, weights)
    dsa_score_gemm(q, k, weights)
    nki_n, fallback_n = score_gemm_dispatch_counters()
    _emit("CASE1_ROUTE", can_run=admitted, nki_dispatch=nki_n, torch_fallback=fallback_n)
    assert admitted is True
    assert nki_n == 1
    assert fallback_n == 0


# ---------------------------------------------------------------------------------------------
# Declared case 2 of 2 -- the orthonormal reading
# ---------------------------------------------------------------------------------------------


def test_declared_case_2_orthonormal_acceptance_at_atol_1e5():
    """DECLARED CASE 2: orthonormal inputs agree absolutely at atol 1e-5, rtol 0.

    The off-diagonal block is reported separately because it is the accumulation-order reading: its
    true value is exactly zero, so any nonzero there is a finding and not a tolerance being spent.
    """
    reset_score_gemm_dispatch_counters()
    tokens, cands = 6, 8
    q, k, weights = _orthonormal_inputs(tokens=tokens, cands=cands)
    got = dsa_score_gemm(q, k, weights)
    want = _oracle(q, k, weights)
    worst, differing = _diff(got, want)
    n = min(tokens, cands)
    block = got.float()[:n, :n]
    offdiag = (block - torch.diag(torch.diagonal(block))).abs().max().item()
    diag_first = float(torch.diagonal(block)[0].item())
    _emit(
        "CASE2_ORTHONORMAL",
        tokens=tokens,
        cands=cands,
        worst_abs=f"{worst:.6e}",
        differing=differing,
        population=got.numel(),
        offdiag_absmax=f"{offdiag:.6e}",
        diag_first=f"{diag_first:.6e}",
        atol=ORTHO_ATOL,
        rtol=0.0,
    )
    torch.testing.assert_close(got.float(), want, rtol=0.0, atol=ORTHO_ATOL)
    assert offdiag <= ORTHO_ATOL
    # The diagonal carries signal, which is what stops the off-diagonal zero above from being the
    # zero of an all-zero tensor.
    assert abs(diag_first) > 1e-6


def test_declared_case_2_route_predicate_one_dispatch():
    """DECLARED CASE 2's route predicate, form R-1."""
    reset_score_gemm_dispatch_counters()
    q, k, weights = _orthonormal_inputs(tokens=6, cands=8)
    admitted = can_run_dsa_score_gemm(q, k, weights)
    dsa_score_gemm(q, k, weights)
    nki_n, fallback_n = score_gemm_dispatch_counters()
    _emit("CASE2_ROUTE", can_run=admitted, nki_dispatch=nki_n, torch_fallback=fallback_n)
    assert admitted is True
    assert nki_n == 1
    assert fallback_n == 0


def test_declared_case_set_total_dispatches_is_two():
    """The DECLARED TOTAL over the declared case set: exactly 2, in ONE reset window.

    Read here rather than summed across items, because a total assembled by adding up other items'
    readings would not notice a dispatch that happened outside all of them.
    """
    reset_score_gemm_dispatch_counters()
    q1, k1, w1 = _random_inputs(tokens=8, cands=12)
    dsa_score_gemm(q1, k1, w1)
    q2, k2, w2 = _orthonormal_inputs(tokens=6, cands=8)
    dsa_score_gemm(q2, k2, w2)
    nki_n, fallback_n = score_gemm_dispatch_counters()
    _emit("DECLARED_TOTAL", declared_cases=2, nki_dispatch=nki_n, torch_fallback=fallback_n)
    assert nki_n == 2
    assert fallback_n == 0


# ---------------------------------------------------------------------------------------------
# SUPPLEMENTARY shape items -- EXCLUDED from the declared total above
# ---------------------------------------------------------------------------------------------


def test_supplementary_token_tile_boundary_129_tokens():
    """SUPPLEMENTARY, not a declared case: tokens crossing ``TOKEN_TILE`` by one.

    ``TOKEN_TILE + 1`` forces a second, PARTIAL token tile, which is where an off-by-one in the tile
    loop would show up. The other extents are kept minimal so the reading is about the boundary and
    not about simulator time.
    """
    reset_score_gemm_dispatch_counters()
    tokens = TOKEN_TILE + 1
    q, k, weights = _random_inputs(tokens=tokens, cands=4, heads=1, seed=31)
    got = dsa_score_gemm(q, k, weights)
    want = _oracle(q, k, weights)
    worst, differing = _diff(got, want)
    nki_n, fallback_n = score_gemm_dispatch_counters()
    _emit(
        "SUPPLEMENTARY_TOKEN_BOUNDARY",
        tokens=tokens,
        tile=TOKEN_TILE,
        worst_abs=f"{worst:.6e}",
        nki_dispatch=nki_n,
        torch_fallback=fallback_n,
    )
    assert got.shape == (tokens, 4)
    torch.testing.assert_close(got.float(), want, rtol=RTOL, atol=ATOL)
    assert (nki_n, fallback_n) == (1, 0)


def test_supplementary_cand_tile_boundary_513_cands():
    """SUPPLEMENTARY, not a declared case: candidates crossing ``CAND_TILE`` by one."""
    reset_score_gemm_dispatch_counters()
    cands = CAND_TILE + 1
    q, k, weights = _random_inputs(tokens=2, cands=cands, heads=1, seed=37)
    got = dsa_score_gemm(q, k, weights)
    want = _oracle(q, k, weights)
    worst, differing = _diff(got, want)
    nki_n, fallback_n = score_gemm_dispatch_counters()
    _emit(
        "SUPPLEMENTARY_CAND_BOUNDARY",
        cands=cands,
        tile=CAND_TILE,
        worst_abs=f"{worst:.6e}",
        nki_dispatch=nki_n,
        torch_fallback=fallback_n,
    )
    assert got.shape == (2, cands)
    torch.testing.assert_close(got.float(), want, rtol=RTOL, atol=ATOL)
    assert (nki_n, fallback_n) == (1, 0)


# ---------------------------------------------------------------------------------------------
# The counted zeros, each with a control that fires
# ---------------------------------------------------------------------------------------------


def test_torch_oracle_dispatches_zero_nki_kernels():
    """ZERO 1: a pure-torch computation of the same function dispatches no NKI kernel.

    This is what makes the comparison independent. If the oracle dispatched even once, the acceptance
    items would be comparing the kernel against itself.
    """
    reset_score_gemm_dispatch_counters()
    q, k, weights = _random_inputs(tokens=8, cands=12)
    out = _oracle(q, k, weights)
    nki_n, fallback_n = score_gemm_dispatch_counters()
    _emit("ORACLE_ZERO_DISPATCH", nki_dispatch=nki_n, torch_fallback=fallback_n, shape=tuple(out.shape))
    assert nki_n == 0
    assert fallback_n == 0


def test_control_the_dispatch_counter_can_move():
    """ZERO 1's FIRING CONTROL: the same counter, read the same way, moves when the seam runs."""
    reset_score_gemm_dispatch_counters()
    before, _ = score_gemm_dispatch_counters()
    q, k, weights = _random_inputs(tokens=4, cands=4)
    dsa_score_gemm(q, k, weights)
    after, _ = score_gemm_dispatch_counters()
    _emit("CONTROL_DISPATCH_COUNTER_MOVES", before=before, after=after)
    assert before == 0
    assert after == 1


def test_torch_fallback_counter_is_zero_on_the_nki_route():
    """ZERO 2: kernel-class work takes NKI and never a torch fallback (P13).

    A nonzero here would be a substrate defect, not a performance note: the plan declares this
    increment kernel-class, so a torch path serving an admitted call would be the thing P13 forbids.
    """
    reset_score_gemm_dispatch_counters()
    q, k, weights = _random_inputs(tokens=8, cands=12)
    dsa_score_gemm(q, k, weights)
    nki_n, fallback_n = score_gemm_dispatch_counters()
    _emit("FALLBACK_ZERO", nki_dispatch=nki_n, torch_fallback=fallback_n)
    assert fallback_n == 0
    assert nki_n == 1


def test_control_torch_fallback_counter_fires_on_unadmitted_dtype():
    """ZERO 2's FIRING CONTROL: an unadmitted dtype takes the torch path and MOVES the counter.

    fp32 is not in ``_SUPPORTED_Q_DTYPES``, so the gate refuses it and the oracle serves it. That is
    a legitimate fallback -- a malformed-input path, not kernel-class work taking torch.
    """
    reset_score_gemm_dispatch_counters()
    q, k, weights = _random_inputs(tokens=4, cands=4)
    q32 = q.float()
    admitted = can_run_dsa_score_gemm(q32, k, weights)
    got = dsa_score_gemm(q32, k, weights)
    nki_n, fallback_n = score_gemm_dispatch_counters()
    _emit(
        "CONTROL_FALLBACK_FIRES",
        q_dtype=str(q32.dtype),
        can_run=admitted,
        nki_dispatch=nki_n,
        torch_fallback=fallback_n,
    )
    assert admitted is False
    assert nki_n == 0
    assert fallback_n == 1
    torch.testing.assert_close(got, _oracle(q32, k, weights), rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------------------------


def test_gate_admits_the_declared_shape():
    """The gate admits the declared shape, and ``can_run_kernel()`` is True in this environment.

    The second half is a PRECONDITION, not a courtesy: if the image refused kernels outright, every
    route item above would read 0 dispatches and would look like a substrate failure.
    """
    q, k, weights = _random_inputs(tokens=8, cands=12)
    _emit("GATE_ADMITS", can_run_kernel=can_run_kernel(), can_run_module=can_run_dsa_score_gemm(q, k, weights))
    assert can_run_kernel() is True
    assert can_run_dsa_score_gemm(q, k, weights) is True


def test_gate_refuses_unadmitted_q_dtype():
    """The gate refuses an fp32 query rather than quietly widening the supported set."""
    q, k, weights = _random_inputs(tokens=4, cands=4)
    admitted = can_run_dsa_score_gemm(q.float(), k, weights)
    _emit("GATE_REFUSES_DTYPE", q_dtype="torch.float32", can_run=admitted)
    assert admitted is False


def test_gate_refuses_a_head_dim_the_kernel_does_not_tile():
    """The gate refuses a head dimension other than 128, which the kernel contracts in one tile."""
    q, k, weights = _mismatched_head_dim_inputs()
    admitted = can_run_dsa_score_gemm(q, k, weights)
    _emit("GATE_REFUSES_HEAD_DIM", head_dim=64, required=INDEX_HEAD_DIM, can_run=admitted)
    assert admitted is False


# ---------------------------------------------------------------------------------------------
# Kernel identity, derived THROUGH the seam (D13.1)
# ---------------------------------------------------------------------------------------------


def test_kernel_identity_is_none_before_any_dispatch():
    """``None`` before any dispatch: the reading that separates "no kernel ran" from "one ran"."""
    reset_score_gemm_dispatch_counters()
    identity = score_gemm_kernel_identity()
    _emit("IDENTITY_BEFORE", identity=identity)
    assert identity is None


def test_kernel_identity_after_dispatch_names_the_nki_kernel():
    """After a dispatch the identity names THIS module's kernel, not the ``@nki.jit`` decorator.

    Reading ``__module__`` off the decorated object would report ``nki.framework.kernel``, which is
    why the module unwraps it. This item is what makes that unwrap load-bearing rather than decorative.
    """
    reset_score_gemm_dispatch_counters()
    q, k, weights = _random_inputs(tokens=4, cands=4)
    dsa_score_gemm(q, k, weights)
    identity = score_gemm_kernel_identity()
    _emit("IDENTITY_AFTER", identity=identity)
    assert identity is not None
    module_name, qualname = identity
    assert module_name == "vllm_neuron.functional.dsa.score_gemm"
    assert qualname == "_score_gemm_nki"


# ---------------------------------------------------------------------------------------------
# Constants, asserted against the ISA rather than against themselves
# ---------------------------------------------------------------------------------------------


def test_tile_constants_match_the_isa_tile_extents():
    """The module's tile literals equal the ISA's own extents.

    The module declares them as literals instead of reading ``nl.tile_size`` at import, because
    ``nl.tile_size.psum_num_banks`` raises ``RuntimeError: No backend set`` outside an activated
    backend -- so a module that read its neighbours would import here and break elsewhere. The three
    extents read below are the safe ones, and reading them HERE is what stops the literals drifting.
    """
    pmax = int(nl.tile_size.pmax)
    stationary = int(nl.tile_size.gemm_stationary_fmax)
    moving = int(nl.tile_size.gemm_moving_fmax)
    _emit("TILE_EXTENTS", pmax=pmax, stationary_fmax=stationary, moving_fmax=moving,
          TOKEN_TILE=TOKEN_TILE, CAND_TILE=CAND_TILE, CONTRACTION_TILE=CONTRACTION_TILE)
    assert TOKEN_TILE == stationary
    assert CAND_TILE == moving
    assert CONTRACTION_TILE == pmax


def test_contraction_tile_equals_index_head_dim():
    """The contraction extent equals the head dimension, which is why there is no contraction loop."""
    _emit("CONTRACTION_FITS", CONTRACTION_TILE=CONTRACTION_TILE, INDEX_HEAD_DIM=INDEX_HEAD_DIM)
    assert CONTRACTION_TILE == INDEX_HEAD_DIM


# ---------------------------------------------------------------------------------------------
# Discriminating controls: the fixture can tell right from wrong
# ---------------------------------------------------------------------------------------------


def test_control_the_fixture_discriminates_the_relu_order():
    """CONTROL: the wrong rectify order disagrees WIDELY on this fixture.

    Without this, the acceptance items would only show that the kernel and the oracle share an
    assumption about the order. The gap is reported so a reviewer can see it is not marginal.
    """
    q, k, weights = _random_inputs(tokens=8, cands=12)
    right = _oracle(q, k, weights)
    wrong = _oracle_wrong_order(q, k, weights)
    gap, differing = _diff(wrong, right)
    _emit("CONTROL_RELU_ORDER_GAP", gap=f"{gap:.6e}", differing=differing, population=right.numel())
    assert gap > 1.0
    assert differing > 0


def test_orthonormal_fixture_really_is_orthonormal():
    """The orthonormal case's fixture is what it claims, so the case is not vacuous.

    Two readings, because bf16 splits them: the OFF-diagonal is exactly zero (the ``+c*c`` and
    ``-c*c`` terms cancel exactly whatever ``c`` rounds to), while the DIAGONAL misses 1 by up to the
    format's own error. The bound for the second is derived at ``BF16_GRAM_BOUND``.
    """
    basis = _hadamard_rows(INDEX_HEAD_DIM).float()
    gram = basis @ basis.t()
    offdiag = (gram - torch.diag(torch.diagonal(gram))).abs().max().item()
    diag_err = (torch.diagonal(gram) - 1.0).abs().max().item()
    _emit(
        "ORTHONORMAL_FIXTURE",
        rows=INDEX_HEAD_DIM,
        offdiag_absmax=f"{offdiag:.6e}",
        diag_err_max=f"{diag_err:.6e}",
        derived_bound=BF16_GRAM_BOUND,
    )
    assert offdiag == 0.0
    assert diag_err <= BF16_GRAM_BOUND


# ---------------------------------------------------------------------------------------------
# Refusals: a malformed call raises rather than computing something plausible
# ---------------------------------------------------------------------------------------------


def test_refuses_a_non_3d_q():
    """A 2-D query raises, naming the rank it wanted."""
    q, k, weights = _random_inputs(tokens=4, cands=4)
    with pytest.raises(ScoreGemmError, match="q must be 3-D"):
        dsa_score_gemm(q[:, 0, :], k, weights)
    _emit("REFUSES_NON_3D_Q", raised="ScoreGemmError")


def test_refuses_a_k_whose_feature_width_mismatches():
    """A key whose feature width does not match the head dimension raises."""
    q, _k, weights = _random_inputs(tokens=4, cands=4)
    gen = torch.Generator().manual_seed(43)
    bad_k = torch.randn((4, 64), generator=gen).to(torch.bfloat16)
    with pytest.raises(ScoreGemmError, match="feature width must match"):
        dsa_score_gemm(q, bad_k, weights)
    _emit("REFUSES_K_WIDTH", k_width=64, head_dim=INDEX_HEAD_DIM, raised="ScoreGemmError")


def test_refuses_weights_of_the_wrong_shape():
    """Weights that are not ``[tokens, heads]`` raise rather than broadcasting into something."""
    q, k, weights = _random_inputs(tokens=4, cands=4)
    with pytest.raises(ScoreGemmError, match="weights must be"):
        dsa_score_gemm(q, k, weights[:, :1])
    _emit("REFUSES_WEIGHTS_SHAPE", raised="ScoreGemmError")


def test_refuses_a_head_dim_that_is_not_128():
    """The seam RAISES on a head dimension of 64, distinct from the gate merely refusing it.

    The gate answers "can the kernel serve this"; the validator answers "is this call well formed".
    A 64-wide head is malformed for this increment, so it raises instead of silently taking torch.
    """
    q, k, weights = _mismatched_head_dim_inputs()
    with pytest.raises(ScoreGemmError, match="contracts exactly"):
        dsa_score_gemm(q, k, weights)
    _emit("REFUSES_HEAD_DIM_64", raised="ScoreGemmError")


# ---------------------------------------------------------------------------------------------
# MX abstention (kickoff contract rule 3), and the control for its zero
# ---------------------------------------------------------------------------------------------


def test_module_source_calls_no_mx_primitive():
    """ZERO 3: the module CALLS neither gen4 MX primitive.

    Both names are asserted, not just the matmul. The census behind that: on this image the only
    ``dir(nki.isa)`` names ending in ``_mx`` are ``nc_matmul_mx`` and ``quantize_mx``, and "ending
    with" and "containing" give the same two, so there is no third spelling to miss
    (``probe-046-mechanism-r2b-host.out``). ``predictions-046-sizing.txt`` named only the matmul; this
    is the corrected, wider claim.

    CALLS are counted, not mentions. The module docstring discusses ``nc_matmul_mx`` at length to say
    what it deliberately avoids and why, and a reader who deleted that discussion to satisfy a
    name-based scan would be deleting the reason. The mention count is emitted alongside so the
    distinction is visible rather than assumed.
    """
    src = _module_source()
    calls = {name: src.count(f"{name}(") for name in ("nc_matmul_mx", "quantize_mx")}
    mentions = {name: src.count(name) for name in ("nc_matmul_mx", "quantize_mx")}
    _emit("MX_ABSTENTION", calls=calls, mentions=mentions, population_chars=len(src))
    assert calls["nc_matmul_mx"] == 0
    assert calls["quantize_mx"] == 0


def test_control_the_mx_reader_fires_on_a_planted_call():
    """ZERO 3's FIRING CONTROL: the same reader counts a planted call of each name.

    Run against a string rather than a file, so the control cannot leave anything behind that a later
    scan would find.

    THE FIXTURE IS BUILT BY CONCATENATION, ON PURPOSE. Written as one literal it would put each MX
    name immediately followed by an open parenthesis into this file, and a call-scoped MX scan reading
    the diff would count that as a call -- the control would trip the very scanner it exists to
    validate. Joining each name to its parenthesis at runtime gives the reader the same bytes to count
    while leaving no call form in the source at all. The classification in ``scan-046-mx-gen4.out`` is
    what caught this, and it caught a second copy in the sentence that first explained it.
    """
    planted = "".join(("x = nisa.", "nc_matmul_mx", "(dst=d)\ny = nisa.", "quantize_mx", "(dst=d)\n"))
    calls = {name: planted.count(f"{name}(") for name in ("nc_matmul_mx", "quantize_mx")}
    _emit("CONTROL_MX_READER_FIRES", calls=calls)
    assert calls["nc_matmul_mx"] == 1
    assert calls["quantize_mx"] == 1
