# SPDX-License-Identifier: Apache-2.0
"""Tier N acceptance for ``inc-glm53f-047`` -- fused kpool compression + Hadamard-128.

WHAT THIS FILE MEASURES, and the shape of the argument it makes:

The fused kernel is correct only if it agrees with an UNFUSED composition of the same two stages,
so that equivalence IS the measurement. The unfused composition lives HERE, in the test, and it is
built from ``torch.softmax`` and a matrix multiply against a Sylvester Hadamard matrix -- neither of
which touches the module's seam. That is deliberate: the oracle must dispatch ZERO NKI kernels, or
the comparison would be the kernel against itself.

ONE ITEM PER COUNTED CONJUNCT and no ``parametrize`` (plan section 6, rules 4b and 6), so a failure
names the reading that failed instead of a parameter id. Counters are reset at the START of each
declared case and read at its END (section 4b's per-case convention).

EVERY COUNTED ZERO HAS A CONTROL THAT FIRES (plan D1.5). The two zeros this file counts are the
torch-fallback counter on the NKI route and the NKI-dispatch counter across the oracle, and each has
a companion item that makes the same counter move. A zero from a counter that cannot move is not a
measurement.

THE ENVIRONMENT IS PINNED IN THE INVOCATION, NEVER IN A FIXTURE (plan D2): this file is run under
``VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 NEURON_PLATFORM_TARGET_OVERRIDE=trn2``.
Nothing here reads or sets an environment variable.

A NOTE ON WHAT THE SIMULATOR PROVES. ``NKI_SIMULATOR=1`` does not run the MLIR verifier, so a pass
here is evidence about VALUES and never about COMPILABILITY. Compilability is settled separately, by
this increment's capture leg with the verifier on.
"""

from __future__ import annotations

import math

import pytest
import torch

from vllm_neuron.functional.dsa.kpool_hadamard import (
    HADAMARD_SCALE,
    HADAMARD_STAGES,
    INDEX_HEAD_DIM,
    KpoolHadamardError,
    can_run_dsa_hadamard128,
    can_run_dsa_kpool_hadamard,
    dsa_hadamard128,
    dsa_kpool_hadamard,
    hadamard_matrix,
    kpool_hadamard_dispatch_counters,
    kpool_hadamard_kernel_identity,
    reset_kpool_hadamard_dispatch_counters,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

POOL_SIZE = 4
"""``index_kpool`` on the target checkpoint (``fixtures/hf-config.json``)."""

RTOL = 1e-2
ATOL = 1e-5
"""The Acceptance bullet's tolerance for the fused-versus-unfused equivalence, verbatim."""

IDENTITY_ATOL = 1e-5
"""The Acceptance bullet's tolerance for the stage-alone identity reading, verbatim."""


def _inputs(n_pools: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One case's three input tensors, at the reference's dtypes.

    ``slot_k`` bf16 and ``ape`` fp32 are the reference's own asserts
    (``kpool_compress.py:283-286``). ``slot_score`` is fp32 here; the module casts in-kernel and a
    separate item covers the bf16-score route.
    """
    gen = torch.Generator().manual_seed(seed)
    slot_k = torch.randn(
        (n_pools, POOL_SIZE, INDEX_HEAD_DIM), generator=gen, dtype=torch.float32
    ).to(torch.bfloat16)
    slot_score = torch.randn(
        (n_pools, POOL_SIZE, INDEX_HEAD_DIM), generator=gen, dtype=torch.float32
    )
    ape = torch.randn((POOL_SIZE, INDEX_HEAD_DIM), generator=gen, dtype=torch.float32)
    return slot_k, slot_score, ape


def _unfused_reference(
    slot_k: torch.Tensor, slot_score: torch.Tensor, ape: torch.Tensor
) -> torch.Tensor:
    """THE TEST'S OWN unfused composition: pool, then rotate. Dispatches NO NKI kernel.

    Transcribed from ``vllm/models/glm5next/nvidia/ops/kpool_compress.py`` at origin head
    ``878631b6``, in the reference's order:

    * ``:164-165`` and ``:174-200`` -- ``softmax(slot_score + ape)``-weighted sum over the slots,
      with the softmax taken over ``dim=1``, THE SLOT AXIS, so it is per ``(pool, channel)``.
    * ``:37-45`` -- the 7-stage FWHT, expressed here as a multiply by the Sylvester ``H_128``
      because a matrix multiply and the butterfly compute the same linear map; the butterfly is the
      SHIPPED form and this is the independent way of writing it, which is what makes the
      comparison worth making.
    * ``:45`` -- one final multiply by ``1/sqrt(128)``.

    Deliberately UNFUSED: three separate whole-tensor steps, each materialised.
    """
    weights = torch.softmax(slot_score.float() + ape.float().unsqueeze(0), dim=1)
    pooled = (weights * slot_k.float()).sum(dim=1)
    rotated = pooled @ hadamard_matrix(INDEX_HEAD_DIM).t()
    return (rotated * HADAMARD_SCALE).to(slot_k.dtype)


def _emit(tag: str, **values: object) -> None:
    """Print one MACHINE-READABLE reading line, for the driver to re-check independently.

    The pattern is the landed one at ``test_ragged_pack.py``: the test asserts, and then PRINTS the
    value it asserted on, so the driver that owns the transcript can check the same number without
    trusting the assertion. That is D1.3's read-and-record -- the instrument produces the value.
    Requires ``-s``, which the declared invocation passes.
    """
    body = " ".join(f"{k}={v}" for k, v in values.items())
    print(f"[{tag}] {body}")


def _diff(got: torch.Tensor, expected: torch.Tensor) -> tuple[float, int]:
    """``(max_abs_diff, differing_elements)`` between two tensors, both as plain python numbers."""
    a, b = got.float(), expected.float()
    return (a - b).abs().max().item(), int((a != b).sum().item())


def _fused_case(n_pools: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one declared fused case. Returns ``(got, expected)``.

    The oracle is evaluated BEFORE the reset so that the reset cannot be credited with hiding an
    oracle dispatch: the counters are zeroed after the oracle has already run, and the reading taken
    at the end therefore belongs to the kernel call alone.
    """
    slot_k, slot_score, ape = _inputs(n_pools, seed)
    expected = _unfused_reference(slot_k, slot_score, ape)
    reset_kpool_hadamard_dispatch_counters()
    got = dsa_kpool_hadamard(slot_k, slot_score, ape)
    return got, expected


# ---------------------------------------------------------------------------------------------
# The gate. Nothing below means anything if the NKI route is not the route taken.
# ---------------------------------------------------------------------------------------------


def test_gate_can_run_kernel_is_true() -> None:
    """``can_run_kernel()`` is True, so every case below takes the NKI route and not the oracle."""
    assert can_run_kernel() is True


def test_gate_admits_the_declared_fused_input() -> None:
    """The fused gate admits the declared shapes and dtypes."""
    slot_k, slot_score, ape = _inputs(33, seed=1)
    assert can_run_dsa_kpool_hadamard(slot_k, slot_score, ape) is True


def test_gate_admits_the_declared_stage_alone_input() -> None:
    """The stage-alone gate admits a ``[n_rows, 128]`` tensor."""
    assert can_run_dsa_hadamard128(torch.eye(INDEX_HEAD_DIM, dtype=torch.bfloat16)) is True


# ---------------------------------------------------------------------------------------------
# The fused arm: 4/4 declared cases over n_pools. Two counted conjuncts each, one item apiece.
# n_pools = 1 (one pool), 33 (a partial 128-partition tile), 130 (one full tile plus two),
# 512 (the fixture scale, four full tiles).
# ---------------------------------------------------------------------------------------------


def test_fused_matches_unfused_n_pools_1() -> None:
    """CASE 1/4, ``n_pools = 1``: a single pool, the smallest tile the kernel can be handed."""
    got, expected = _fused_case(1, seed=101)
    max_abs, differing = _diff(got, expected)
    _emit("acceptance", n_pools=1, rows=got.shape[0], cols=got.shape[1],
          max_abs_diff=f"{max_abs:.6e}", differing_elements=differing,
          rtol=RTOL, atol=ATOL)
    assert tuple(got.shape) == (1, INDEX_HEAD_DIM)
    torch.testing.assert_close(got.float(), expected.float(), rtol=RTOL, atol=ATOL)


def test_fused_route_n_pools_1() -> None:
    """CASE 1/4 route: exactly one NKI dispatch and zero torch fallbacks."""
    slot_k, slot_score, ape = _inputs(1, seed=101)
    reset_kpool_hadamard_dispatch_counters()
    gate = can_run_dsa_kpool_hadamard(slot_k, slot_score, ape)
    dsa_kpool_hadamard(slot_k, slot_score, ape)
    nki, fallback = kpool_hadamard_dispatch_counters()
    _emit("route-predicate", n_pools=1, nki_dispatch=nki, torch_fallback=fallback,
          can_run_kernel=gate)
    assert (nki, fallback) == (1, 0)
    assert gate is True


def test_fused_matches_unfused_n_pools_33() -> None:
    """CASE 2/4, ``n_pools = 33``: a PARTIAL 128-partition tile, so the tile is narrowed."""
    got, expected = _fused_case(33, seed=102)
    max_abs, differing = _diff(got, expected)
    _emit("acceptance", n_pools=33, rows=got.shape[0], cols=got.shape[1],
          max_abs_diff=f"{max_abs:.6e}", differing_elements=differing,
          rtol=RTOL, atol=ATOL)
    assert tuple(got.shape) == (33, INDEX_HEAD_DIM)
    torch.testing.assert_close(got.float(), expected.float(), rtol=RTOL, atol=ATOL)


def test_fused_route_n_pools_33() -> None:
    """CASE 2/4 route: exactly one NKI dispatch and zero torch fallbacks."""
    slot_k, slot_score, ape = _inputs(33, seed=102)
    reset_kpool_hadamard_dispatch_counters()
    gate = can_run_dsa_kpool_hadamard(slot_k, slot_score, ape)
    dsa_kpool_hadamard(slot_k, slot_score, ape)
    nki, fallback = kpool_hadamard_dispatch_counters()
    _emit("route-predicate", n_pools=33, nki_dispatch=nki, torch_fallback=fallback,
          can_run_kernel=gate)
    assert (nki, fallback) == (1, 0)
    assert gate is True


def test_fused_matches_unfused_n_pools_130() -> None:
    """CASE 3/4, ``n_pools = 130``: one FULL tile plus two rows, so a short second tile follows."""
    got, expected = _fused_case(130, seed=103)
    max_abs, differing = _diff(got, expected)
    _emit("acceptance", n_pools=130, rows=got.shape[0], cols=got.shape[1],
          max_abs_diff=f"{max_abs:.6e}", differing_elements=differing,
          rtol=RTOL, atol=ATOL)
    assert tuple(got.shape) == (130, INDEX_HEAD_DIM)
    torch.testing.assert_close(got.float(), expected.float(), rtol=RTOL, atol=ATOL)


def test_fused_route_n_pools_130() -> None:
    """CASE 3/4 route: exactly one NKI dispatch and zero torch fallbacks."""
    slot_k, slot_score, ape = _inputs(130, seed=103)
    reset_kpool_hadamard_dispatch_counters()
    gate = can_run_dsa_kpool_hadamard(slot_k, slot_score, ape)
    dsa_kpool_hadamard(slot_k, slot_score, ape)
    nki, fallback = kpool_hadamard_dispatch_counters()
    _emit("route-predicate", n_pools=130, nki_dispatch=nki, torch_fallback=fallback,
          can_run_kernel=gate)
    assert (nki, fallback) == (1, 0)
    assert gate is True


def test_fused_matches_unfused_n_pools_512() -> None:
    """CASE 4/4, ``n_pools = 512``: the fixture scale, four full tiles and no remainder."""
    got, expected = _fused_case(512, seed=104)
    max_abs, differing = _diff(got, expected)
    _emit("acceptance", n_pools=512, rows=got.shape[0], cols=got.shape[1],
          max_abs_diff=f"{max_abs:.6e}", differing_elements=differing,
          rtol=RTOL, atol=ATOL)
    assert tuple(got.shape) == (512, INDEX_HEAD_DIM)
    torch.testing.assert_close(got.float(), expected.float(), rtol=RTOL, atol=ATOL)


def test_fused_route_n_pools_512() -> None:
    """CASE 4/4 route: exactly one NKI dispatch and zero torch fallbacks."""
    slot_k, slot_score, ape = _inputs(512, seed=104)
    reset_kpool_hadamard_dispatch_counters()
    gate = can_run_dsa_kpool_hadamard(slot_k, slot_score, ape)
    dsa_kpool_hadamard(slot_k, slot_score, ape)
    nki, fallback = kpool_hadamard_dispatch_counters()
    _emit("route-predicate", n_pools=512, nki_dispatch=nki, torch_fallback=fallback,
          can_run_kernel=gate)
    assert (nki, fallback) == (1, 0)
    assert gate is True


# ---------------------------------------------------------------------------------------------
# The stage-alone arm: the identity case reads all seven butterfly stages and the final scale.
# ---------------------------------------------------------------------------------------------


def test_stage_alone_identity_reproduces_the_transform_matrix() -> None:
    """On ``I_128`` the rotation's output IS ``H_128 / sqrt(128)``, at the declared atol.

    This is the cheapest complete reading of the transform: every one of the seven stages and the
    single final scale contribute to every element, so a wrong stage order, a wrong stride or a
    missing scale all show up here. fp32 input, because the reading is about the transform and not
    about bf16 rounding.
    """
    reset_kpool_hadamard_dispatch_counters()
    got = dsa_hadamard128(torch.eye(INDEX_HEAD_DIM, dtype=torch.float32))
    expected = hadamard_matrix(INDEX_HEAD_DIM) * HADAMARD_SCALE
    max_abs, differing = _diff(got, expected)
    _emit("acceptance", case="stage_alone_identity", rows=got.shape[0], cols=got.shape[1],
          max_abs_diff=f"{max_abs:.6e}", differing_elements=differing, atol=IDENTITY_ATOL)
    torch.testing.assert_close(got.float(), expected, rtol=0.0, atol=IDENTITY_ATOL)


def test_stage_alone_route() -> None:
    """Identity case route: exactly one NKI dispatch, through the same seam, and zero fallbacks."""
    reset_kpool_hadamard_dispatch_counters()
    dsa_hadamard128(torch.eye(INDEX_HEAD_DIM, dtype=torch.float32))
    nki, fallback = kpool_hadamard_dispatch_counters()
    _emit("route-predicate", case="stage_alone", nki_dispatch=nki, torch_fallback=fallback)
    assert (nki, fallback) == (1, 0)


def test_dispatch_total_over_the_declared_case_set_is_five() -> None:
    """The declared total: 4 fused dispatches plus 1 stage-alone, counted in ONE reset window.

    The per-case items above each read their own window; this item reads the SUM the route predicate
    declares, so the two readings cannot disagree without one of them failing.
    """
    reset_kpool_hadamard_dispatch_counters()
    for n_pools, seed in ((1, 101), (33, 102), (130, 103), (512, 104)):
        slot_k, slot_score, ape = _inputs(n_pools, seed)
        dsa_kpool_hadamard(slot_k, slot_score, ape)
    dsa_hadamard128(torch.eye(INDEX_HEAD_DIM, dtype=torch.float32))
    nki, fallback = kpool_hadamard_dispatch_counters()
    _emit("route-predicate", case="declared_total", nki_dispatch=nki, torch_fallback=fallback)
    assert (nki, fallback) == (5, 0)


# ---------------------------------------------------------------------------------------------
# The oracle dispatches zero -- and the control that proves the counter can move.
# ---------------------------------------------------------------------------------------------


def test_the_test_oracle_dispatches_zero_nki_kernels() -> None:
    """The unfused composition this file compares against runs NO NKI kernel.

    Without this reading the equivalence could be the kernel agreeing with itself.
    """
    slot_k, slot_score, ape = _inputs(33, seed=201)
    reset_kpool_hadamard_dispatch_counters()
    _unfused_reference(slot_k, slot_score, ape)
    nki, fallback = kpool_hadamard_dispatch_counters()
    _emit("control", case="test_oracle", nki_dispatch=nki, torch_fallback=fallback)
    assert (nki, fallback) == (0, 0)


def test_control_the_dispatch_counter_does_move_on_the_fused_seam() -> None:
    """CONTROL for the zero above: the same counter reads 1 after one fused call.

    A zero from a counter that cannot move measures nothing. This item makes it move.
    """
    slot_k, slot_score, ape = _inputs(33, seed=201)
    reset_kpool_hadamard_dispatch_counters()
    dsa_kpool_hadamard(slot_k, slot_score, ape)
    nki = kpool_hadamard_dispatch_counters()[0]
    _emit("control", case="dispatch_counter_moves", nki_dispatch=nki)
    assert nki == 1


def test_torch_fallback_counter_is_zero_on_the_nki_route() -> None:
    """The module's torch-fallback counter is EXACTLY zero across all five declared dispatches."""
    reset_kpool_hadamard_dispatch_counters()
    for n_pools, seed in ((1, 101), (33, 102), (130, 103), (512, 104)):
        slot_k, slot_score, ape = _inputs(n_pools, seed)
        dsa_kpool_hadamard(slot_k, slot_score, ape)
    dsa_hadamard128(torch.eye(INDEX_HEAD_DIM, dtype=torch.float32))
    assert kpool_hadamard_dispatch_counters()[1] == 0


def test_control_the_fallback_counter_does_move_on_an_unsupported_dtype() -> None:
    """CONTROL for the zero above: an fp32 ``slot_k`` takes the oracle and the counter reads 1.

    fp32 is outside the module's supported dtypes, so this exercises the real gate rather than a
    monkeypatched one.
    """
    slot_k, slot_score, ape = _inputs(8, seed=202)
    reset_kpool_hadamard_dispatch_counters()
    gate = can_run_dsa_kpool_hadamard(slot_k.float(), slot_score, ape)
    dsa_kpool_hadamard(slot_k.float(), slot_score, ape)
    nki, fallback = kpool_hadamard_dispatch_counters()
    _emit("control", case="unadmitted_dtype", dtype="torch.float32", can_run_kernel=gate,
          nki_dispatch=nki, torch_fallback=fallback)
    assert (nki, fallback) == (0, 1)
    assert gate is False


# ---------------------------------------------------------------------------------------------
# The certifying component: identity derived THROUGH the seam (D13.1).
# ---------------------------------------------------------------------------------------------


def test_kernel_identity_is_none_before_any_dispatch() -> None:
    """Before a dispatch the identity is ``None`` -- the reading that separates "none ran"."""
    reset_kpool_hadamard_dispatch_counters()
    assert kpool_hadamard_kernel_identity() is None


def test_kernel_identity_names_the_fused_kernel_after_a_fused_dispatch() -> None:
    """After a fused call the identity is THIS module's fused kernel, read through the seam."""
    slot_k, slot_score, ape = _inputs(33, seed=203)
    reset_kpool_hadamard_dispatch_counters()
    dsa_kpool_hadamard(slot_k, slot_score, ape)
    module, qualname = kpool_hadamard_kernel_identity()
    assert module.endswith("kpool_hadamard")
    assert qualname == "_kpool_hadamard_nki"


def test_kernel_identity_names_the_stage_kernel_after_a_stage_dispatch() -> None:
    """After a stage-alone call the identity names the OTHER kernel, so the read discriminates."""
    reset_kpool_hadamard_dispatch_counters()
    dsa_hadamard128(torch.eye(INDEX_HEAD_DIM, dtype=torch.float32))
    module, qualname = kpool_hadamard_kernel_identity()
    assert module.endswith("kpool_hadamard")
    assert qualname == "_hadamard128_nki"


# ---------------------------------------------------------------------------------------------
# The transform's own constants, checked against arithmetic rather than against themselves.
# ---------------------------------------------------------------------------------------------


def test_hadamard_scale_literal_is_the_correctly_rounded_one_over_sqrt_128() -> None:
    """The shipped literal is the CORRECTLY ROUNDED ``128 ** -0.5``, bit for bit.

    WHICH SPELLING, AND WHY IT MATTERS. ``128 ** -0.5`` and ``1.0 / math.sqrt(128)`` are not the
    same double: they differ by exactly ONE ULP, because the division form rounds down. Measured::

        128 ** -0.5        = 0x1.6a09e667f3bcdp-4   (the origin's literal, kpool_compress.py:45)
        1.0/math.sqrt(128) = 0x1.6a09e667f3bccp-4   (one ULP lower)

    The origin's literal is the first of those, and ``math.sqrt(1/128)``, ``2 ** -3.5`` and
    ``math.sqrt(2)/16`` all agree with it. So this item asserts against the power form. The
    difference is 1.4e-17 absolute and cannot affect any tolerance in this file, but asserting the
    WRONG spelling would have failed a correct kernel -- which is exactly what it did when this item
    was first written, before the run, and is why it is now written down.
    """
    assert HADAMARD_SCALE == INDEX_HEAD_DIM**-0.5
    assert HADAMARD_SCALE == math.sqrt(1.0 / INDEX_HEAD_DIM)
    # And the one-ULP disagreement is itself recorded, so nobody re-derives it and calls the shipped
    # constant wrong.
    assert HADAMARD_SCALE != 1.0 / math.sqrt(INDEX_HEAD_DIM)
    assert abs(HADAMARD_SCALE - 1.0 / math.sqrt(INDEX_HEAD_DIM)) < 1e-16


def test_hadamard_matrix_is_orthogonal_after_scaling() -> None:
    """``(H/sqrt(n)) @ (H/sqrt(n)).T == I``, so the oracle's matrix really is a Hadamard matrix."""
    h = hadamard_matrix(INDEX_HEAD_DIM) * HADAMARD_SCALE
    torch.testing.assert_close(
        h @ h.t(), torch.eye(INDEX_HEAD_DIM), rtol=0.0, atol=1e-5
    )


def test_stage_sequence_is_the_origin_sequence() -> None:
    """The seven ``(groups, stride)`` pairs are the origin's, and each covers all 128 channels.

    ``kpool_compress.py:38-44`` in order. Asserted as a sequence rather than as seven call sites so
    that a reordering is a failure here instead of a silent numerical difference.
    """
    assert HADAMARD_STAGES == ((64, 1), (32, 2), (16, 4), (8, 8), (4, 16), (2, 32), (1, 64))
    for groups, stride in HADAMARD_STAGES:
        assert groups * 2 * stride == INDEX_HEAD_DIM


# ---------------------------------------------------------------------------------------------
# The discriminating case: a whole-vector softmax must NOT pass.
# ---------------------------------------------------------------------------------------------


def test_per_pool_channel_softmax_is_distinguishable_from_a_whole_vector_softmax() -> None:
    """A softmax over the 128 channels instead of the 4 slots is a DIFFERENT kernel, and this
    file's inputs are able to tell them apart.

    Without this reading the equivalence items would still pass for a module that softmaxed the
    wrong axis if the inputs happened to make the two agree. This item measures that they do not:
    it builds the wrong-axis composition and asserts it is far outside the acceptance tolerance.
    """
    slot_k, slot_score, ape = _inputs(33, seed=204)
    correct = _unfused_reference(slot_k, slot_score, ape)
    biased = torch.softmax(slot_score.float() + ape.float().unsqueeze(0), dim=-1)
    wrong_pooled = (biased * slot_k.float()).sum(dim=1)
    wrong = (wrong_pooled @ hadamard_matrix(INDEX_HEAD_DIM).t() * HADAMARD_SCALE).to(slot_k.dtype)
    gap = (correct.float() - wrong.float()).abs().max().item()
    threshold = 10.0 * (ATOL + RTOL * correct.float().abs().max().item())
    _emit("control", case="wrong_axis_softmax", gap=f"{gap:.6e}",
          threshold=f"{threshold:.6e}")
    assert gap > threshold


# ---------------------------------------------------------------------------------------------
# Malformed calls are refused rather than silently reshaped.
# ---------------------------------------------------------------------------------------------


def test_mismatched_score_shape_is_refused() -> None:
    """A ``slot_score`` that does not match ``slot_k`` raises instead of broadcasting."""
    slot_k, slot_score, ape = _inputs(8, seed=205)
    with pytest.raises(KpoolHadamardError):
        dsa_kpool_hadamard(slot_k, slot_score[:, :2, :], ape)


def test_wrong_head_dim_is_refused() -> None:
    """A head dimension other than 128 raises: the Hadamard path is a 128-point transform."""
    with pytest.raises(KpoolHadamardError):
        dsa_hadamard128(torch.zeros((4, 64), dtype=torch.bfloat16))
