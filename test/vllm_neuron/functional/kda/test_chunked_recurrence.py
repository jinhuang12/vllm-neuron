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


# =========================================================================== #
# Acceptance for `inc-glm53f-089` -- the geometry production actually resolves.
#
# **Four items under the ``-k production_geometry`` selection, one per declared
# conjunct, and still no ``parametrize`` decorator in this file** (D1.2). Each
# item names the component whose behaviour it certifies (D1.4).
#
# WHY THIS SECTION EXISTS. `-035a` measures at head width 64 over chunk sizes
# ``{32, 64, 128}``. The landed KDA layer enters this seam at ``kdim = vdim =
# 128`` and resolves chunk width **8**, so the one geometry production actually
# runs was measured by nothing. This section adds that reading and CHANGES NO
# SOURCE FILE, which is the whole finding: the kernel already admits both
# production values, so nothing was broken -- only unmeasured.
#
# THIS IS THE THIRD SECTION IN THIS FILE, AND THE SELECTION INVARIANT NOW HAS
# THREE PARTS RATHER THAN TWO. `-035b`'s banner above states the invariant for
# the two sections that existed when it was written -- "every name below
# contains ``inter``" -- and a third section appended after it makes that
# sentence read wider than it was scoped. Its LOAD-BEARING claim is untouched
# and is restated here for all three: no name in this section contains the
# substring ``intra`` or the substring ``inter``, so ``-k intra`` still collects
# exactly `-035a`'s four items, ``-k inter`` still collects exactly `-035b`'s
# three, and ``-k production_geometry`` collects exactly the four below. That is
# checked by the acceptance harness, which reads all three counts, rather than
# by this comment. `-035b`'s landed text is left alone deliberately: the
# disjointness it protects still holds, and rewriting another block's comment is
# a wider surface than adding a scoping paragraph to this one.
#
# Run on the same Tier N harness with the selection changed::
#
#     VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \
#     NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
#     python -m pytest test/vllm_neuron/functional/kda/test_chunked_recurrence.py \
#         -k production_geometry -v -s --timeout 60 -p no:cacheprovider
#
# What each item is for, in one line each:
#
# 1. the three production values are READ from the fork and match the declared
#    128 / 128 / 8, so nothing below runs at an assumed geometry;
# 2. the kernel's numbers at that geometry match the chunk-local torch reference
#    at the frozen comparator, with the route reading taken around the call;
# 3. the gate bound is MEASURED against the declared limit, and the derivation
#    that picks 8 is shown to be tight rather than restated;
# 4. the doubling-stage count at chunk 8 is 3 -- a count no graded run has read.
#
# NO COMPARATOR, TOLERANCE OR THRESHOLD IS INTRODUCED HERE (P9). Conjunct 2 uses
# ``RTOL`` / ``ATOL`` above, which are `-035a`'s own landed pair; conjuncts 1, 3
# and 4 are exact readings with no tolerance at all.
# =========================================================================== #

#: The production geometry, as the plan block declares it. These are the values
#: the READ must produce, not the values the tests run at -- what they run at is
#: whatever the fork says, and a disagreement fails loudly rather than being
#: absorbed. A disagreeing reading is ``evidence_contradicts_design`` and goes to
#: the lead; it is never a silent re-declaration here.
DECLARED_PRODUCTION_KDIM = 128
DECLARED_PRODUCTION_VDIM = 128
DECLARED_PRODUCTION_CHUNK = 8

#: The checkpoint's gate lower bound, and the limit the two chunked seams apply.
#: Declared here so the derivation below is checked against both ends.
DECLARED_GATE_LOWER_BOUND = -5.0

#: ``ceil(log2 8)``. The graded runs read 5, 6 and 7 for chunk 32, 64 and 128, so
#: 3 is a stage count no recorded run in this campaign has exercised.
DECLARED_DOUBLING_STAGES_AT_PRODUCTION_CHUNK = 3

_PRODUCTION_CACHE: dict[str, object] = {}


def _production_fixture_path():
    """The in-repo config fixture, as a SECOND root beside the config module.

    Read as well as ``config.py`` because the plan block cites ``config.py`` for
    ``head_dim`` while its conjunct-1 prose says "the in-repo config fixture" --
    two different files. Reading both costs nothing and turns that ambiguity into
    a measurement.

    A FUNCTION RATHER THAN A MODULE CONSTANT, and the reason is load-bearing: this
    module imports no ``pathlib`` and adding one to its header would shift every
    line below it, drifting the citations two other files pin into this one. A
    function-local import keeps the header byte-identical.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[4]
    return (
        root / "test" / "vllm_neuron" / "model" / "glm5_next"
        / "fixtures" / "config.json"
    )


def _production_geometry() -> dict:
    """The production geometry, read from the fork rather than assumed.

    Builds the real ``Glm5NextKDAAttention`` and asks it, because the chunk width
    is a LAYER decision (``_resolve_chunk_size``) and re-deriving it here would
    measure this file's arithmetic instead of the layer's. Nothing else in this
    repository constructs that layer, so this is the first place it is built.

    The imports are function-local on purpose. This module is imported whenever
    ``-k intra`` or ``-k inter`` runs, and those two selections must not acquire
    a vLLM dependency they never had -- ``model_fp8.py`` itself holds no vLLM
    import at module level for the same reason.

    ``world_size=1`` because none of the three values read here shards: the layer
    binds ``head_size = head_dim``, and the chunk derivation reads only
    ``gate_lower_bound``.
    """
    if "geometry" not in _PRODUCTION_CACHE:
        import json

        from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig
        from vllm_neuron.model.glm5_next.model_fp8 import Glm5NextKDAAttention

        text_config = Glm5NextTextConfig()
        module_root = text_config.linear_attn_config
        fixture_path = _production_fixture_path()
        fixture_root = json.loads(fixture_path.read_text())
        fixture_root = fixture_root["text_config"]["linear_attn_config"]
        layer = Glm5NextKDAAttention(text_config, world_size=1)
        _PRODUCTION_CACHE["geometry"] = {
            "kdim": int(layer.head_dim),
            "vdim": int(layer.head_size),
            "chunk": int(layer._resolve_chunk_size(None)),
            "gate_lower_bound": float(layer.gate_lower_bound),
            "cache_chunk_size": layer.cache_chunk_size,
            "module_head_dim": int(module_root["head_dim"]),
            "module_gate_lower_bound": float(module_root["gate_lower_bound"]),
            "fixture_head_dim": int(fixture_root["head_dim"]),
            "fixture_gate_lower_bound": float(fixture_root["gate_lower_bound"]),
        }
    return _PRODUCTION_CACHE["geometry"]


def _production_inputs(seed: int = 20260904):
    """Deterministic inputs at the geometry the fork reports, not at a literal.

    Same construction as :func:`_inputs` above -- the same distributions, the same
    ``GATE_SCALE`` -- with the widths and the chunk taken from the READ geometry,
    so this arm cannot drift away from the geometry conjunct 1 certifies.
    """
    geo = _production_geometry()
    chunk, kdim, vdim = geo["chunk"], geo["kdim"], geo["vdim"]
    gen = torch.Generator().manual_seed(seed + chunk)
    shape_k = (N_CHUNKS, chunk, kdim)
    q = torch.randn(shape_k, generator=gen, dtype=torch.float32)
    k = torch.randn(shape_k, generator=gen, dtype=torch.float32)
    v = torch.randn((N_CHUNKS, chunk, vdim), generator=gen, dtype=torch.float32)
    beta = torch.rand((N_CHUNKS, chunk), generator=gen, dtype=torch.float32) * 0.9 + 0.05
    gk = -torch.rand(shape_k, generator=gen, dtype=torch.float32) * GATE_SCALE
    return q, k, v, beta, gk


def _report_production(item: str, certifies: str) -> None:
    print(f"\nPRODUCTION|{item}|certifies={certifies}", flush=True)


def test_production_geometry_values_are_read_from_the_fork_not_assumed():
    """Conjunct 1. Certifying component: ``_resolve_chunk_size`` and the layer's width binding.

    Three values, 3/3: ``kdim`` and ``vdim`` from the layer's ``head_dim`` /
    ``head_size`` (both from ``linear_attn_config["head_dim"]``), and the chunk
    width from ``_resolve_chunk_size(None)`` on a layer built here.

    BOTH CONFIG ROOTS ARE READ AND COMPARED. The block cites ``config.py`` for
    ``head_dim`` and its prose says "the in-repo config fixture"; those are two
    files, so both are read and their agreement is a reading rather than an
    assumption. The fixture also carries a top-level ``text_config.head_dim`` of
    ``0`` -- a different field, the MLA one -- so reading the wrong key would give
    a geometry of zero width, and only ``linear_attn_config``'s entry is the KDA
    width.

    ``cache_chunk_size`` is printed because the derivation only runs when the dial
    is ``None``: a config that pinned the dial would return the pinned value and
    this reading would be about the dial rather than about the derivation.
    """
    _report_production(
        "conjunct1_geometry_is_read", "_resolve_chunk_size and the width binding"
    )
    geo = _production_geometry()
    print(
        f"PRODUCTION|conjunct1|kdim={geo['kdim']}|vdim={geo['vdim']}|"
        f"chunk={geo['chunk']}|gate_lower_bound={geo['gate_lower_bound']}|"
        f"cache_chunk_size={geo['cache_chunk_size']!r}",
        flush=True,
    )
    print(
        f"PRODUCTION|conjunct1|config_module_head_dim={geo['module_head_dim']}|"
        f"fixture_head_dim={geo['fixture_head_dim']}|"
        f"config_module_gate_lower_bound={geo['module_gate_lower_bound']}|"
        f"fixture_gate_lower_bound={geo['fixture_gate_lower_bound']}",
        flush=True,
    )
    read = (geo["kdim"], geo["vdim"], geo["chunk"])
    declared = (
        DECLARED_PRODUCTION_KDIM,
        DECLARED_PRODUCTION_VDIM,
        DECLARED_PRODUCTION_CHUNK,
    )
    agreed = sum(1 for r, d in zip(read, declared) if r == d)
    print(
        f"PRODUCTION|conjunct1|read={read}|declared={declared}|"
        f"agreed={agreed}/3|two_config_roots_agree="
        f"{geo['module_head_dim'] == geo['fixture_head_dim']}",
        flush=True,
    )
    assert agreed == 3, (
        f"the fork reports {read} where this plan block declares {declared}; "
        f"that is evidence_contradicts_design and goes to the lead, never a "
        f"silent re-declaration here"
    )
    assert geo["cache_chunk_size"] is None
    assert geo["module_head_dim"] == geo["fixture_head_dim"]
    assert geo["module_gate_lower_bound"] == geo["fixture_gate_lower_bound"]
    assert geo["gate_lower_bound"] == DECLARED_GATE_LOWER_BOUND


def test_production_geometry_numerics_match_the_chunk_local_reference():
    """Conjunct 2, and the route predicate (D13 form R-2). Certifying component: the ``wrap_nki`` seam `-035a` authors.

    The seam's five returned values at the READ production geometry against the
    torch chunk-local reference this file already carries, at `-035a`'s own frozen
    comparator. No new tolerance (P9).

    THE ROUTE READING IS TAKEN AROUND THE SEAM CALL AND NOWHERE ELSE, on `-035b`'s
    per-call convention: the reset happens immediately before the call and the read
    immediately after, before the reference is computed, so the reading belongs to
    this call. It must read exactly ``1`` dispatch -- :data:`N_CHUNKS` would mean a
    per-chunk host loop and ``0`` would mean the torch path served this item, which
    is why a reference-only run cannot satisfy this conjunct.
    """
    _report_production(
        "conjunct2_numeric_agreement_at_production_geometry", "the wrap_nki seam"
    )
    geo = _production_geometry()
    q, k, v, beta, gk = _production_inputs()
    assert can_run_kernel(q) is True
    print(
        f"PRODUCTION|conjunct2|shape_q={tuple(q.shape)}|shape_v={tuple(v.shape)}|"
        f"n_chunks={N_CHUNKS}|stages={doubling_stages(geo['chunk'])}|"
        f"kernel_identity={kernel_identity()}",
        flush=True,
    )

    reset_dispatch_counters()
    got = kda_intra_chunk(q, k, v, beta, gk)
    nki_dispatch, torch_fallback = dispatch_counters()

    want = kda_intra_chunk_torch_oracle(q, k, v, beta, gk)
    worst = {}
    for field in ("w", "u", "a_inv", "aqk", "kg"):
        actual, expected = getattr(got, field), getattr(want, field)
        worst[field] = _worst(actual, expected)
        assert_close(
            _t(actual), _t(expected), rtol=RTOL, atol=ATOL,
            name=f"production_geometry[chunk={geo['chunk']}].{field}",
        )
    worst_field = max(worst, key=worst.get)
    print(
        f"PRODUCTION|conjunct2|kdim={geo['kdim']}|chunk={geo['chunk']}|"
        f"worst_field={worst_field}|worst_abs_error={worst[worst_field]:.3e}|"
        + "|".join(f"{f}={worst[f]:.3e}" for f in ("w", "u", "a_inv", "aqk", "kg"))
        + f"|declared_rtol={RTOL}|declared_atol={ATOL}",
        flush=True,
    )
    print(
        f"PRODUCTION|conjunct2|nki_dispatch={nki_dispatch}|"
        f"torch_fallback={torch_fallback}|can_run_kernel=True",
        flush=True,
    )
    assert nki_dispatch == 1, (
        f"expected exactly 1 dispatch at the production geometry; {N_CHUNKS} "
        f"would mean a per-chunk host loop and 0 would mean the torch path "
        f"served this item"
    )
    assert torch_fallback == 0


def test_production_geometry_gate_bound_is_measured_against_the_declared_limit():
    """Conjunct 3. Certifying component: the seam's gate-range admissibility gate.

    The gate bound is MEASURED and printed beside ``GATE_CUMSUM_ABS_LIMIT``'s
    ``60.0`` and the derivation's ``5.0 x 8 = 40``, rather than restated. Two
    readings, and the second is the one that matters:

    * over conjunct 2's own inputs, computed the way the seam computes it
      (``gk.cumsum(dim=1).abs().max()``), which is small because those inputs use
      this file's small ``GATE_SCALE``;
    * over the exact WORST CASE the checkpoint can produce -- every gate entry at
      ``gate_lower_bound`` -- which is where the derivation's number comes from.

    Landed `-084` changed the gate to ``gate_lower_bound * sigmoid(...)`` and so
    raised the TYPICAL magnitude without moving the bound, which is exactly why
    this conjunct measures instead of assuming.

    THE DERIVATION IS ALSO SHOWN TO BE TIGHT, not merely satisfied: one power of
    two wider is ``5.0 x 16 = 80``, above the limit, which is what makes 8 the
    widest admissible chunk rather than an arbitrary choice. And the seam itself
    is asked whether it admits the worst case, so this is a reading about
    ``_require_admissible`` and not only about arithmetic.
    """
    _report_production(
        "conjunct3_measured_gate_bound", "the seam's gate-range admissibility gate"
    )
    from vllm_neuron.functional.kda.chunked_recurrence import (
        GATE_CUMSUM_ABS_LIMIT,
        can_run_intra_chunk,
    )

    geo = _production_geometry()
    chunk, kdim, vdim = geo["chunk"], geo["kdim"], geo["vdim"]
    bound = abs(geo["gate_lower_bound"])

    _, _, _, _, gk = _production_inputs()
    measured = float(gk.float().cumsum(dim=1).abs().max().item())

    worst_gk = torch.full((N_CHUNKS, chunk, kdim), geo["gate_lower_bound"],
                          dtype=torch.float32)
    worst_case = float(worst_gk.cumsum(dim=1).abs().max().item())
    derived = bound * chunk
    one_wider = bound * (chunk * 2)

    print(
        f"PRODUCTION|conjunct3|measured_gate_abs_max={measured:.4f}|"
        f"worst_case_gate_abs_max={worst_case}|derivation={bound} x {chunk} = "
        f"{derived}|GATE_CUMSUM_ABS_LIMIT={GATE_CUMSUM_ABS_LIMIT}|"
        f"one_chunk_wider={bound} x {chunk * 2} = {one_wider}",
        flush=True,
    )

    reference = torch.zeros((N_CHUNKS, chunk, kdim), dtype=torch.float32)
    admits_worst_case = can_run_intra_chunk(
        reference, N_CHUNKS, chunk, kdim, vdim, worst_case
    )
    print(
        f"PRODUCTION|conjunct3|seam_admits_worst_case={admits_worst_case}|"
        f"headroom={GATE_CUMSUM_ABS_LIMIT - worst_case}",
        flush=True,
    )

    # The worst case IS the derivation's number, exactly: eight additions of
    # -5.0 in fp32 are exact, both being powers of two times a small integer.
    assert worst_case == derived
    assert derived <= GATE_CUMSUM_ABS_LIMIT
    assert one_wider > GATE_CUMSUM_ABS_LIMIT
    assert measured <= worst_case
    assert admits_worst_case is True


def test_production_geometry_doubling_stage_count_is_three_at_chunk_eight():
    """Conjunct 4. Certifying component: ``doubling_stages``.

    ``ceil(log2 8) == 3``, read from the fork's own ``doubling_stages`` rather
    than recomputed here -- the point of the item is that the kernel and this
    reading agree on one number, and a second implementation of ``log2`` would
    only agree with itself.

    THE POPULATION IS PRINTED BESIDE IT so the claim "a stage count no recorded
    run has read" is measured rather than asserted: the three graded sizes read
    5, 6 and 7, and none of them is 3.
    """
    _report_production("conjunct4_doubling_stage_count", "doubling_stages")
    geo = _production_geometry()
    chunk = geo["chunk"]
    stages = doubling_stages(chunk)
    graded = {c: doubling_stages(c) for c in CHUNK_SIZES}
    print(
        f"PRODUCTION|conjunct4|chunk={chunk}|stages={stages}|"
        f"declared={DECLARED_DOUBLING_STAGES_AT_PRODUCTION_CHUNK}|"
        f"graded_sizes={graded}|graded_stage_counts={sorted(graded.values())}",
        flush=True,
    )
    assert stages == DECLARED_DOUBLING_STAGES_AT_PRODUCTION_CHUNK
    assert stages not in graded.values(), (
        f"chunk={chunk} reads {stages} stages and the graded sizes read "
        f"{sorted(graded.values())}; if one of them already read {stages} this "
        f"conjunct would be measuring nothing new"
    )
