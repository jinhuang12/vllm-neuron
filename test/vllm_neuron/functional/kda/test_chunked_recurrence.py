# SPDX-License-Identifier: Apache-2.0
"""Acceptance for `inc-glm53f-035a` -- the KDA intra-chunk NKI kernel.

**Four items under the ``-k intra`` selection, one per declared conjunct, and no
``parametrize`` decorator in this file** (D1.2). Each item names the component
whose behaviour it certifies (D1.4). `-035b` extends this file under ``-k
inter``; the two selections are disjoint, so neither block's counted items can be
satisfied or broken by the other's.

Run on the Tier N harness -- the NKI simulator on the host CPU, no device and no
lease::

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
    python -m pytest test/vllm_neuron/functional/kda/test_chunked_recurrence.py \
        -k intra -q -s --timeout 60 -p no:cacheprovider

What each item is for, in one line each:

1. the kernel's four returned values match an independent torch intra-chunk
   reference at the frozen comparator, over all three declared chunk sizes;
2. the inverse the kernel returns really is an inverse -- ``(I + A) . T == I``;
3. **the discriminator** -- the stage-3 entry must respond to the inverse it is
   handed, which a token-walking implementation would ignore;
4. the route predicate -- one dispatch per declared case, no torch fallback.

Item 2 is deliberately weaker than it looks and the plan block says so: a kernel
that formed the inverse and then produced ``w`` / ``u`` by a token-sequential
recurrence would pass items 1 and 2 together, because with per-chunk zero state
that recurrence yields exactly the WY values. Item 3 is the one that tells the
two designs apart at rung 1, and the plan attaches a rung-2 obligation beside it
that only review can settle.
"""

from __future__ import annotations

import pytest
import torch

from vllm_neuron.accuracy.testing import assert_close
from vllm_neuron.functional.kda.chunked_recurrence import (
    ChunkConstants,
    chunk_constants,
    dispatch_counters,
    doubling_stages,
    inter_dispatch_counters,
    inter_kernel_identity,
    kda_inter_chunk,
    kda_intra_chunk,
    kda_intra_chunk_kernel,
    kda_intra_chunk_torch_oracle,
    kda_sequential_torch_oracle,
    kda_stage3_kernel,
    kernel_identity,
    rebuild_i_plus_a,
    reset_dispatch_counters,
    reset_inter_dispatch_counters,
    stage3_kernel_identity,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

#: The declared case set, as the plan block's Acceptance bullet writes it:
#: "chunk sizes ``{32, 64, 128}``, three cases". Cited rather than chosen here --
#: the block is the single place those numbers are declared, and `-035b` and
#: `-052` cite the same bullet instead of restating it.
CHUNK_SIZES = (32, 64, 128)

#: The frozen comparator pair, from the plan block's conjunct 1: "at
#: ``assert_close(rtol=1e-2, atol=1e-5)`` over the three chunk sizes the
#: Acceptance bullet above declares, **3/3**". Read from the block, never
#: retyped from memory and never widened (P9).
RTOL = 1e-2
ATOL = 1e-5

#: Two chunks, so the route predicate's reading is discriminating: a seam that
#: looped over chunks and dispatched per chunk would read 2 where the design
#: reads 1. With a single chunk the two designs would be indistinguishable.
N_CHUNKS = 2

#: Key and value widths. The block declares the CHUNK sizes and leaves these
#: free; 64 is a real head width that keeps the simulator inside the harness
#: timeout, and the kernel's admissibility limit covers the full 128.
KDIM = 64
VDIM = 64

#: Gate magnitude. KDA gates are log-space decays, so a negative draw is the
#: realistic sign; this range keeps the cumulative gate well inside the kernel's
#: declared ``GATE_CUMSUM_ABS_LIMIT``.
GATE_SCALE = 0.05


def _inputs(chunk: int, seed: int = 20260903):
    """Deterministic inputs for one declared chunk size."""
    gen = torch.Generator().manual_seed(seed + chunk)
    shape_k = (N_CHUNKS, chunk, KDIM)
    q = torch.randn(shape_k, generator=gen, dtype=torch.float32)
    k = torch.randn(shape_k, generator=gen, dtype=torch.float32)
    v = torch.randn((N_CHUNKS, chunk, VDIM), generator=gen, dtype=torch.float32)
    beta = torch.rand((N_CHUNKS, chunk), generator=gen, dtype=torch.float32) * 0.9 + 0.05
    gk = -torch.rand(shape_k, generator=gen, dtype=torch.float32) * GATE_SCALE
    return q, k, v, beta, gk


def _t(x) -> torch.Tensor:
    """Whatever the kernel returned, as a float32 torch tensor.

    The seam hands back torch tensors; a direct entry call under the simulator can
    hand back an array-like, so both are accepted.
    """
    if isinstance(x, torch.Tensor):
        return x.float()
    try:
        return torch.as_tensor(x).float()
    except Exception:  # pragma: no cover - array-like without __torch_function__
        import numpy as np

        return torch.as_tensor(np.asarray(x)).float()


def _report(item: str, certifies: str) -> None:
    print(f"\nINTRA|{item}|certifies={certifies}", flush=True)


def _worst(actual, expected) -> float:
    return (_t(actual) - _t(expected)).abs().max().item()


def test_intra_chunk_kernel_matches_the_torch_intra_chunk_reference():
    """Conjunct 1. Certifying component: the intra-chunk kernel this block authors.

    Compares the four values the block names -- ``w``, ``u``, ``(I + A)**-1`` and
    ``Aqk`` -- against a torch reference that computes stages 1 to 3 and nothing
    downstream of them, at the frozen comparator, over 3/3 declared chunk sizes.

    ``kg`` is asserted here as well although conjunct 1 names four values: `-035b`
    declares ``kg`` as a seam input, this is the only place it is measured, and
    an assertion can only fail if the kernel is wrong.
    """
    _report("conjunct1_numeric_agreement", "the intra-chunk kernel")
    print(f"INTRA|kernel_identity={kernel_identity()}", flush=True)
    matched = 0
    for chunk in CHUNK_SIZES:
        q, k, v, beta, gk = _inputs(chunk)
        got = kda_intra_chunk(q, k, v, beta, gk)
        want = kda_intra_chunk_torch_oracle(q, k, v, beta, gk)
        worst = {}
        for field in ("w", "u", "a_inv", "aqk", "kg"):
            actual, expected = getattr(got, field), getattr(want, field)
            worst[field] = _worst(actual, expected)
            assert_close(
                _t(actual), _t(expected), rtol=RTOL, atol=ATOL,
                name=f"intra[chunk={chunk}].{field}",
            )
        worst_field = max(worst, key=worst.get)
        print(
            f"INTRA|conjunct1|chunk={chunk}|stages={doubling_stages(chunk)}|"
            f"worst_field={worst_field}|worst_abs_error={worst[worst_field]:.3e}|"
            + "|".join(f"{f}={worst[f]:.3e}" for f in ("w", "u", "a_inv", "aqk", "kg")),
            flush=True,
        )
        matched += 1
    print(f"INTRA|conjunct1|matched={matched}/{len(CHUNK_SIZES)}", flush=True)
    assert matched == len(CHUNK_SIZES)


def test_intra_chunk_returned_inverse_times_i_plus_a_is_the_identity():
    """Conjunct 2. Certifying component: the ``(I + A)**-1`` tile the kernel returns.

    ``(I + A) . T == I`` at **atol 1e-5**, where ``T`` is the kernel's own
    returned inverse and ``(I + A)`` is rebuilt by the torch reference from the
    same inputs. In this reading the reference's own inverse plays no part at
    all, so the blocked 16x16 algorithm upstream uses is not mirrored anywhere.

    The block declares an absolute tolerance for this conjunct and no relative
    one, so ``rtol`` is pinned to ``0.0`` -- a purely absolute reading, stricter
    than the frozen pair and never wider -- and the max absolute deviation is
    printed so any other tolerance can be applied to the number afterwards.
    """
    _report("conjunct2_inverse_is_an_inverse", "the returned (I + A)**-1 tile")
    checked = 0
    for chunk in CHUNK_SIZES:
        q, k, v, beta, gk = _inputs(chunk)
        got = kda_intra_chunk(q, k, v, beta, gk)
        i_plus_a = rebuild_i_plus_a(k, beta, gk)
        product = i_plus_a @ _t(got.a_inv)
        identity = torch.eye(chunk).expand(N_CHUNKS, chunk, chunk)
        deviation = (product - identity).abs().max().item()
        print(
            f"INTRA|conjunct2|chunk={chunk}|max_abs_deviation_from_identity="
            f"{deviation:.3e}|declared_atol={ATOL}",
            flush=True,
        )
        assert_close(
            product, identity, rtol=0.0, atol=ATOL,
            name=f"intra[chunk={chunk}].(I+A)@inv",
        )
        checked += 1
    print(f"INTRA|conjunct2|checked={checked}/{len(CHUNK_SIZES)}", flush=True)
    assert checked == len(CHUNK_SIZES)


def test_intra_chunk_stage3_entry_responds_to_the_inverse_it_is_handed():
    """Conjunct 3, THE DISCRIMINATOR. Certifying component: the stage-3 kernel entry.

    The entry is called twice per declared chunk size: once with the kernel's
    returned inverse, once with the same inverse whose last row is scaled by
    ``2.0``. Because ``u = (I + A)**-1 (beta . V)`` is linear in the inverse,
    that perturbation scales ``u``'s last row by two, so the two results must
    **not** be within the frozen comparator of each other.

    An entry that re-derived ``w`` / ``u`` by walking tokens would never read the
    inverse it was handed, both calls would return identical ``u``, the
    comparison would not fail, and this item would not pass.

    The non-match is asserted with ``pytest.raises`` on the exception the fork's
    own ``assert_close`` raises, which is **``AssertionError``** -- read at
    ``vllm_neuron/accuracy/testing.py:402``, where the ``raise`` stands, rather
    than assumed.

    **Both kernel entries are called directly through ``wrap_nki`` here and the
    seam is never called**, so the seam's dispatch counter reads ``0`` across the
    whole item and this conjunct moves neither this block's route-predicate count
    nor any other block's. That is not a torch fallback, and the fallback counter
    stays ``0`` too.
    """
    _report("conjunct3_discriminator", "the stage-3 kernel entry")
    print(f"INTRA|stage3_kernel_identity={stage3_kernel_identity()}", flush=True)
    reset_dispatch_counters()
    non_matches = 0
    for chunk in CHUNK_SIZES:
        q, k, v, beta, gk = _inputs(chunk)
        consts: ChunkConstants = chunk_constants(chunk)
        beta_col = beta.unsqueeze(-1).contiguous()

        # Direct entry call, bypassing the seam, so no counter moves.
        _, _, _, a_inv, _ = wrap_nki(kda_intra_chunk_kernel)(
            q_hbm=q, k_hbm=k, v_hbm=v, beta_hbm=beta_col, gk_hbm=gk,
            triu_hbm=consts.triu_ones, eye_hbm=consts.eye,
            mask_lower_hbm=consts.mask_lower, last_row_hbm=consts.last_row,
        )
        a_inv = _t(a_inv)
        perturbed = a_inv.clone()
        perturbed[:, -1, :] *= 2.0

        def _stage3(inverse):
            return wrap_nki(kda_stage3_kernel)(
                k_hbm=k, v_hbm=v, beta_hbm=beta_col, gk_hbm=gk,
                a_inv_hbm=inverse.contiguous(),
                triu_hbm=consts.triu_ones, last_row_hbm=consts.last_row,
            )

        _, base_u, _ = _stage3(a_inv)
        _, alt_u, _ = _stage3(perturbed)
        difference = _worst(alt_u, base_u)
        print(
            f"INTRA|conjunct3|chunk={chunk}|u_max_abs_difference={difference:.3e}|"
            f"u_abs_max={_t(base_u).abs().max().item():.3e}",
            flush=True,
        )
        with pytest.raises(AssertionError):
            assert_close(
                _t(alt_u), _t(base_u), rtol=RTOL, atol=ATOL,
                name=f"intra[chunk={chunk}].u_perturbed_vs_base",
            )
        non_matches += 1

    counters = dispatch_counters()
    print(
        f"INTRA|conjunct3|declared_non_matches={non_matches}/{len(CHUNK_SIZES)}|"
        f"seam_counters={counters}",
        flush=True,
    )
    assert non_matches == len(CHUNK_SIZES)
    assert counters == (0, 0)


def test_intra_chunk_route_predicate_reads_one_dispatch_per_declared_case():
    """Conjunct 4, route predicate D13 form R-1. Certifying component: the ``wrap_nki`` seam.

    One dispatch per declared chunk-size case -- ``1``, not the chunk count --
    with the torch-fallback counter at exactly ``0`` and ``can_run_kernel()``
    true. The chunking is inside the kernel, so a per-chunk host loop would read
    :data:`N_CHUNKS` instead, and the two readings tell the two designs apart. A
    pure-torch implementation yields ``0`` and therefore cannot pass.
    """
    _report("conjunct4_route_predicate", "the wrap_nki seam this block authors")
    for chunk in CHUNK_SIZES:
        q, k, v, beta, gk = _inputs(chunk)
        assert can_run_kernel(q) is True
        reset_dispatch_counters()
        kda_intra_chunk(q, k, v, beta, gk)
        nki_dispatch, torch_fallback = dispatch_counters()
        print(
            f"INTRA|conjunct4|chunk={chunk}|n_chunks={N_CHUNKS}|"
            f"nki_dispatch={nki_dispatch}|torch_fallback={torch_fallback}|"
            f"can_run_kernel=True",
            flush=True,
        )
        assert nki_dispatch == 1, (
            f"expected exactly 1 dispatch for chunk={chunk}; {N_CHUNKS} would mean "
            f"a per-chunk host loop"
        )
        assert torch_fallback == 0


# =========================================================================== #
# Acceptance for `inc-glm53f-035b` -- the KDA inter-chunk state carry and output.
#
# **Three items under the ``-k inter`` selection, one per declared conjunct, and
# still no ``parametrize`` decorator in this file** (D1.2). Each item names the
# component whose behaviour it certifies (D1.4).
#
# THE TWO SELECTIONS ARE DISJOINT AND THAT IS CHECKED BY NAME, NOT BY HOPE: every
# name below contains ``inter`` and none contains ``intra``, and every name above
# contains ``intra`` and none contains ``inter``. So ``-k intra`` still selects
# exactly the four items above and ``-k inter`` selects exactly the three below.
#
# Everything below is PURELY ADDITIVE. The module docstring is left alone because
# it scopes itself to `-035a` in its own first sentence and already forecasts this
# extension, and because the intra-chunk increment's evidence record cites this
# file by line.
#
# Run on the same Tier N harness with the selection changed::
#
#     VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \
#     NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
#     python -m pytest test/vllm_neuron/functional/kda/test_chunked_recurrence.py \
#         -k inter -q -s --timeout 60 -p no:cacheprovider
#
# What each item is for, in one line each:
#
# 1. the chunked final state matches a SEQUENTIAL torch scan at the frozen
#    comparator, across 3/3 chunk sizes -- which is a claim about the chunking
#    being associativity-correct rather than tuned to one shape;
# 2. the chunked output ``o`` matches the same scan's ``o = H q``, the check the
#    retired increment never made at all;
# 3. **the wiring conjunct** -- the state must respond to the ``w`` / ``u`` it is
#    handed, which a kernel that re-derived the state from its other inputs would
#    fail.
#
# THE ROUTE PREDICATE IS NOT A FOURTH ITEM. The plan takes its reading **per
# call**, so every item below resets and reads this increment's own counter around
# every seam call it makes: 1 call in items 1 and 2, 2 calls per size in item 3.
# One consequence is worth stating because the intra-chunk block had to disclose
# its absence: a torch fallback inside items 1 or 2 would read ``0`` dispatches
# and fail the item that contains it, so no item here can pass through the
# fallback path.
# =========================================================================== #

#: Total tokens in the flat sequence every inter-chunk item scans. Chosen here
#: because the plan block declares the CHUNK sizes and leaves the sequence length
#: free. 256 divides all three declared sizes and leaves at least two chunks at
#: every one of them -- 8, 4 and 2 -- so the route predicate's reading stays
#: discriminating at each size: a per-chunk host loop would read 8, 4 or 2 where
#: the design reads 1.
TOKENS = 256

#: One sequential reference serves all three chunk sizes, because the flat inputs
#: do not depend on the chunking. That is the point of conjunct 1: the SAME
#: reference is compared against three different chunkings.
_SEQ_CACHE: dict[int, object] = {}

#: The intra-chunk seam's outputs per chunk size, computed once. Every item below
#: needs them as its inputs, and they are pure values, so caching them removes
#: two thirds of the intra-chunk dispatches without coupling any item's assertion
#: to any other's. No item reads the intra-chunk counter, so nothing observes the
#: cache; run any item alone and the first call fills it.
_INTRA_CACHE: dict[int, object] = {}


def _flat_inputs(seed: int = 20260903):
    """Deterministic FLAT inputs -- ``[T, *]``, no chunk axis.

    Flat rather than chunked because the sequential reference walks tokens and
    knows nothing about chunks; the chunked views are derived from these.
    """
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn((TOKENS, KDIM), generator=gen, dtype=torch.float32)
    k = torch.randn((TOKENS, KDIM), generator=gen, dtype=torch.float32)
    v = torch.randn((TOKENS, VDIM), generator=gen, dtype=torch.float32)
    beta = torch.rand(TOKENS, generator=gen, dtype=torch.float32) * 0.9 + 0.05
    gk = -torch.rand((TOKENS, KDIM), generator=gen, dtype=torch.float32) * GATE_SCALE
    return q, k, v, beta, gk


def _chunked(chunk: int):
    """The flat inputs regrouped into ``[NC, chunk, *]`` for one declared size."""
    q, k, v, beta, gk = _flat_inputs()
    return (
        q.reshape(-1, chunk, KDIM).contiguous(),
        k.reshape(-1, chunk, KDIM).contiguous(),
        v.reshape(-1, chunk, VDIM).contiguous(),
        beta.reshape(-1, chunk).contiguous(),
        gk.reshape(-1, chunk, KDIM).contiguous(),
    )


def _sequential_reference():
    """The sequential scan over the flat inputs, computed once."""
    if 0 not in _SEQ_CACHE:
        _SEQ_CACHE[0] = kda_sequential_torch_oracle(*_flat_inputs())
    return _SEQ_CACHE[0]


def _intra_for(chunk: int):
    """The intra-chunk seam's real kernel outputs for one declared chunk size.

    THE INTER-CHUNK KERNEL IS FED THE INTRA-CHUNK **KERNEL**'S OUTPUTS, NOT THE
    INTRA-CHUNK ORACLE'S. So what every item below measures is the composed
    chunked pipeline against the sequential scan, which is the stronger reading
    and the one the plan's phrase "over the same inputs" asks for.
    """
    if chunk not in _INTRA_CACHE:
        q, k, v, beta, gk = _chunked(chunk)
        _INTRA_CACHE[chunk] = kda_intra_chunk(q, k, v, beta, gk)
    return _INTRA_CACHE[chunk]


def _report_inter(item: str, certifies: str) -> None:
    print(f"\nINTER|{item}|certifies={certifies}", flush=True)


def _inter_call(chunk: int, w, u):
    """One counted inter-chunk seam call, with the route reading taken around it.

    Returns ``(outputs, counters)``. The reset happens immediately before the
    call and the read immediately after, so the reading belongs to THIS call and
    to no other -- which is what the plan's per-call convention requires.
    """
    q, _, _, _, gk = _chunked(chunk)
    intra = _intra_for(chunk)
    reset_inter_dispatch_counters()
    out = kda_inter_chunk(intra.kg, w, u, gk, q, intra.aqk)
    return out, inter_dispatch_counters()


def test_inter_chunk_final_state_matches_the_sequential_torch_scan():
    """Conjunct 1. Certifying component: the inter-chunk state carry this block authors.

    The chunked kernel's final state against a SEQUENTIAL torch delta-rule scan
    over the same inputs, at the frozen comparator, and the load-bearing part is
    that agreement holds across **3/3** chunk sizes -- so the chunking is proved
    associativity-correct rather than merely tuned to one shape.

    The reference shares no structure with the kernel: it materialises no ``A``,
    forms no inverse, computes no ``w`` / ``u`` / ``kg`` / ``Aqk``, and walks
    tokens one at a time.

    The plan records that this conjunct ALONE is the one a sequential-in-SBUF
    kernel was measured passing, which is exactly why conjunct 3 exists. This item
    carries rung-1 authority for the carry's numeric correctness and for nothing
    about how the carry is structured.
    """
    _report_inter("conjunct1_final_state_vs_sequential", "the inter-chunk state carry")
    print(f"INTER|inter_kernel_identity={inter_kernel_identity()}", flush=True)
    ref = _sequential_reference()
    print(
        f"INTER|reference|tokens={TOKENS}|state_abs_max="
        f"{ref.final_state.abs().max().item():.3e}",
        flush=True,
    )
    matched = 0
    for chunk in CHUNK_SIZES:
        q, _, _, _, gk = _chunked(chunk)
        assert can_run_kernel(q) is True
        intra = _intra_for(chunk)
        out, counters = _inter_call(chunk, intra.w, intra.u)
        worst = _worst(out.final_state, ref.final_state)
        print(
            f"INTER|conjunct1|chunk={chunk}|n_chunks={q.shape[0]}|"
            f"worst_abs_error={worst:.3e}|declared_rtol={RTOL}|declared_atol={ATOL}|"
            f"chunk_local_gate_abs_max={gk.cumsum(dim=1).abs().max().item():.4f}|"
            f"nki_dispatch={counters[0]}|torch_fallback={counters[1]}|"
            f"can_run_kernel=True",
            flush=True,
        )
        assert_close(
            _t(out.final_state), ref.final_state, rtol=RTOL, atol=ATOL,
            name=f"inter[chunk={chunk}].final_state",
        )
        assert counters == (1, 0), (
            f"route predicate for chunk={chunk}: expected exactly 1 dispatch and 0 "
            f"torch fallbacks, read {counters}; {q.shape[0]} would mean a per-chunk "
            f"host loop and 0 would mean the torch path served this item"
        )
        matched += 1
    print(f"INTER|conjunct1|matched={matched}/{len(CHUNK_SIZES)}", flush=True)
    assert matched == len(CHUNK_SIZES)


def test_inter_chunk_output_matches_the_sequential_scan_o_equals_h_q():
    """Conjunct 2. Certifying component: the stage-5 output path this block authors.

    The chunked ``o`` against the sequential scan's ``o = H q`` at the same frozen
    comparator over the same three sizes, 3/3, worst error printed as a number.

    THE TWO FORMULAS ARE NOT THE SAME EXPRESSION AND THAT IS THE WHOLE VALUE OF
    THIS ITEM. The kernel forms the gate-decayed inter-chunk part ``qg @ h_chunk``
    plus the intra-chunk part ``Aqk @ v_new``; the reference forms ``H q`` after
    every single-token update. Agreement is therefore a real reading about stage 5
    and about its consumption of the intra-chunk block's ``Aqk``, which the
    retired increment never checked at all.
    """
    _report_inter("conjunct2_output_vs_sequential", "the stage-5 output path")
    ref = _sequential_reference()
    print(
        f"INTER|reference|o_shape={tuple(ref.o.shape)}|o_abs_max="
        f"{ref.o.abs().max().item():.3e}",
        flush=True,
    )
    matched = 0
    for chunk in CHUNK_SIZES:
        q, _, _, _, _ = _chunked(chunk)
        assert can_run_kernel(q) is True
        intra = _intra_for(chunk)
        out, counters = _inter_call(chunk, intra.w, intra.u)
        flat_o = _t(out.o).reshape(TOKENS, VDIM)
        worst = _worst(flat_o, ref.o)
        print(
            f"INTER|conjunct2|chunk={chunk}|n_chunks={q.shape[0]}|"
            f"worst_abs_error={worst:.3e}|declared_rtol={RTOL}|declared_atol={ATOL}|"
            f"nki_dispatch={counters[0]}|torch_fallback={counters[1]}|"
            f"can_run_kernel=True",
            flush=True,
        )
        assert_close(
            flat_o, ref.o, rtol=RTOL, atol=ATOL, name=f"inter[chunk={chunk}].o",
        )
        assert counters == (1, 0), (
            f"route predicate for chunk={chunk}: expected exactly 1 dispatch and 0 "
            f"torch fallbacks, read {counters}"
        )
        matched += 1
    print(f"INTER|conjunct2|matched={matched}/{len(CHUNK_SIZES)}", flush=True)
    assert matched == len(CHUNK_SIZES)


def test_inter_chunk_state_responds_to_the_w_and_u_it_is_handed():
    """Conjunct 3, THE WIRING CONJUNCT. Certifying component: the ``w`` / ``u``
    consumption path in this block's seam.

    Two declared perturbations at each of the three sizes, 2/2 x 3: ``w`` and
    ``u`` both replaced by zeros, and again ``u``'s last row scaled by ``2.0``. In
    each case the final state must **NOT** be within the frozen comparator of the
    sequential reference.

    WHY IT DISCRIMINATES: a kernel that ignored ``w`` / ``u`` and re-derived the
    state from its other inputs would still match the reference under both
    perturbations, and would therefore FAIL this arm. That is the reading the
    phrase "consumes the intra-chunk block's ``w`` / ``u``" needed and did not
    have.

    No comparator value is introduced or moved: the frozen pair is used as a match
    in items 1 and 2 and as a declared non-match here, exactly as the intra-chunk
    block's conjunct 3 uses its own. The non-match is asserted with
    ``pytest.raises`` on ``AssertionError``, the exception the fork's own
    ``assert_close`` raises at ``vllm_neuron/accuracy/testing.py:402``.

    UNLIKE THE INTRA-CHUNK DISCRIMINATOR, THIS ITEM GOES THROUGH THE SEAM, so the
    route reading is live in all six calls and each must read ``(1, 0)``.
    """
    _report_inter(
        "conjunct3_wiring_discriminator", "the w/u consumption path in the seam"
    )
    ref = _sequential_reference()
    non_matches = 0
    for chunk in CHUNK_SIZES:
        q, _, _, _, _ = _chunked(chunk)
        assert can_run_kernel(q) is True
        intra = _intra_for(chunk)

        zero_u = torch.zeros_like(_t(intra.u))
        last_row_doubled = _t(intra.u).clone()
        last_row_doubled[:, -1, :] *= 2.0
        arms = (
            ("w_and_u_zeroed", torch.zeros_like(_t(intra.w)), zero_u),
            ("u_last_row_times_two", _t(intra.w), last_row_doubled),
        )

        for tag, pert_w, pert_u in arms:
            out, counters = _inter_call(chunk, pert_w.contiguous(), pert_u.contiguous())
            difference = _worst(out.final_state, ref.final_state)
            print(
                f"INTER|conjunct3|chunk={chunk}|perturbation={tag}|"
                f"state_max_abs_difference={difference:.3e}|"
                f"reference_state_abs_max={ref.final_state.abs().max().item():.3e}|"
                f"nki_dispatch={counters[0]}|torch_fallback={counters[1]}|"
                f"can_run_kernel=True",
                flush=True,
            )
            with pytest.raises(AssertionError):
                assert_close(
                    _t(out.final_state), ref.final_state, rtol=RTOL, atol=ATOL,
                    name=f"inter[chunk={chunk}].final_state_{tag}_vs_sequential",
                )
            assert counters == (1, 0), (
                f"route predicate for chunk={chunk} perturbation {tag}: expected "
                f"exactly 1 dispatch and 0 torch fallbacks, read {counters}"
            )
            non_matches += 1

    expected = 2 * len(CHUNK_SIZES)
    print(
        f"INTER|conjunct3|declared_non_matches={non_matches}/{expected}", flush=True
    )
    assert non_matches == expected
