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
    kda_intra_chunk,
    kda_intra_chunk_kernel,
    kda_intra_chunk_torch_oracle,
    kda_stage3_kernel,
    kernel_identity,
    rebuild_i_plus_a,
    reset_dispatch_counters,
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
