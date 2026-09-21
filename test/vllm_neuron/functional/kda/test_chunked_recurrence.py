# SPDX-License-Identifier: Apache-2.0
"""Tests for the KDA chunked recurrence kernels, both halves.

The intra-chunk kernel is compared against a torch reference for stages 1 to 3.
The inter-chunk kernel is compared against a sequential single-token delta-rule
scan, which shares no structure with the chunked path, so agreement across
several chunk widths says the chunking is associativity-correct rather than tuned
to one shape.

Two properties need a perturbation to be visible, because a kernel that ignored
an argument and re-derived its value would otherwise agree: the stage-3 entry
point must respond to the inverse it is handed, and the inter-chunk kernel must
respond to the ``w`` / ``u`` it is handed. Those cases assert a deliberate
mismatch.
"""

from __future__ import annotations

import pytest
import torch

from vllm_neuron.accuracy.testing import assert_close
from vllm_neuron.functional.kda.chunked_recurrence import (
    ChunkConstants,
    ChunkedRecurrenceError,
    chunk_constants,
    dispatch_counters,
    doubling_stages,
    inter_dispatch_counters,
    kda_inter_chunk,
    kda_intra_chunk,
    kda_intra_chunk_kernel,
    kda_intra_chunk_torch_oracle,
    kda_sequential_torch_oracle,
    kda_stage3_kernel,
    rebuild_i_plus_a,
    reset_dispatch_counters,
    reset_inter_dispatch_counters,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

#: The chunk widths every numeric case runs at.
CHUNK_SIZES = (32, 64, 128)

RTOL = 1e-2
ATOL = 1e-5

#: More than one chunk, so that a kernel dispatching once per chunk instead of
#: once per call reads a different number.
N_CHUNKS = 2

#: Key and value widths. 64 is a real head width that keeps the simulator inside
#: the test timeout; the kernel admits the full 128.
KDIM = 64
VDIM = 64

#: Gate magnitude. KDA gates are log-space decays, so a negative draw is the
#: realistic sign, and this range keeps the cumulative gate well inside
#: ``GATE_CUMSUM_ABS_LIMIT``.
GATE_SCALE = 0.05

#: Total tokens in the flat sequence the inter-chunk cases scan. 256 divides all
#: three chunk widths and leaves at least two chunks at each.
TOKENS = 256


def _inputs(chunk: int, seed: int = 20260903):
    """Deterministic chunked inputs for one chunk width."""
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

    The entry points hand back torch tensors; a direct kernel call under the
    simulator can hand back an array-like, so both are accepted.
    """
    if isinstance(x, torch.Tensor):
        return x.float()
    try:
        return torch.as_tensor(x).float()
    except Exception:  # pragma: no cover - array-like without __torch_function__
        import numpy as np

        return torch.as_tensor(np.asarray(x)).float()


def test_intra_chunk_kernel_matches_the_torch_intra_chunk_reference():
    """The five stage 1-3 outputs match the torch reference at every chunk width."""
    for chunk in CHUNK_SIZES:
        q, k, v, beta, gk = _inputs(chunk)
        got = kda_intra_chunk(q, k, v, beta, gk)
        expected = kda_intra_chunk_torch_oracle(q, k, v, beta, gk)
        for field in ("w", "u", "a_inv", "aqk", "kg"):
            assert_close(
                _t(getattr(got, field)), _t(getattr(expected, field)),
                rtol=RTOL, atol=ATOL, name=f"intra[chunk={chunk}].{field}",
            )


def test_intra_chunk_returned_inverse_times_i_plus_a_is_the_identity():
    """``(I + A) @ T == I`` for the inverse the kernel returns.

    ``(I + A)`` is rebuilt by the torch reference from the same inputs, so the
    reference's own inverse plays no part and the comparison is not circular.
    ``rtol`` is 0 because this is a purely absolute reading against the identity.
    """
    for chunk in CHUNK_SIZES:
        q, k, v, beta, gk = _inputs(chunk)
        got = kda_intra_chunk(q, k, v, beta, gk)
        product = rebuild_i_plus_a(k, beta, gk) @ _t(got.a_inv)
        identity = torch.eye(chunk).expand(N_CHUNKS, chunk, chunk)
        assert_close(
            product, identity, rtol=0.0, atol=ATOL,
            name=f"intra[chunk={chunk}].(I+A)@inv",
        )


def test_intra_chunk_stage3_entry_responds_to_the_inverse_it_is_handed():
    """Perturbing the supplied inverse must move stage 3's ``u``.

    ``u = (I + A)**-1 (beta . v)`` is linear in the inverse, so scaling the
    inverse's last row scales ``u``'s last row. A stage-3 implementation that
    re-derived ``w`` / ``u`` by walking tokens would ignore the argument, return
    identical ``u`` for both calls, and fail here.

    Both kernels are called directly rather than through the counted entry
    points, so the dispatch counters must stay at zero throughout.
    """
    reset_dispatch_counters()
    for chunk in CHUNK_SIZES:
        q, k, v, beta, gk = _inputs(chunk)
        consts: ChunkConstants = chunk_constants(chunk)
        beta_col = beta.unsqueeze(-1).contiguous()

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
        with pytest.raises(AssertionError):
            assert_close(
                _t(alt_u), _t(base_u), rtol=RTOL, atol=ATOL,
                name=f"intra[chunk={chunk}].u_perturbed_vs_base",
            )

    assert dispatch_counters() == (0, 0), (
        "a direct kernel call must not move the counted entry point's counters"
    )


def test_intra_chunk_makes_one_kernel_dispatch_per_call():
    """One call is one dispatch, not one per chunk, and never the torch path."""
    for chunk in CHUNK_SIZES:
        q, k, v, beta, gk = _inputs(chunk)
        assert can_run_kernel(q) is True
        reset_dispatch_counters()
        kda_intra_chunk(q, k, v, beta, gk)
        nki_dispatch, torch_fallback = dispatch_counters()
        assert nki_dispatch == 1, (
            f"expected exactly 1 dispatch for chunk={chunk}; {N_CHUNKS} would mean "
            f"a per-chunk host loop"
        )
        assert torch_fallback == 0


#: One sequential reference serves every chunk width, because the flat inputs do
#: not depend on the chunking.
_SEQ_CACHE: dict[int, object] = {}

#: The intra-chunk outputs per chunk width, computed once. They are pure values and
#: no case reads the intra-chunk counter, so nothing observes the cache.
_INTRA_CACHE: dict[int, object] = {}


def _flat_inputs(seed: int = 20260903):
    """Deterministic flat inputs, ``[T, *]`` with no chunk axis.

    The sequential reference walks tokens and knows nothing about chunks, so the
    chunked views are derived from these rather than the other way round.
    """
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn((TOKENS, KDIM), generator=gen, dtype=torch.float32)
    k = torch.randn((TOKENS, KDIM), generator=gen, dtype=torch.float32)
    v = torch.randn((TOKENS, VDIM), generator=gen, dtype=torch.float32)
    beta = torch.rand(TOKENS, generator=gen, dtype=torch.float32) * 0.9 + 0.05
    gk = -torch.rand((TOKENS, KDIM), generator=gen, dtype=torch.float32) * GATE_SCALE
    return q, k, v, beta, gk


def _chunked(chunk: int):
    """The flat inputs regrouped into ``[NC, chunk, *]``."""
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
    """The intra-chunk kernel's own outputs for one chunk width.

    The inter-chunk kernel is fed the intra-chunk kernel's outputs rather than the
    reference's, so what these cases measure is the composed chunked pipeline
    against the sequential scan.
    """
    if chunk not in _INTRA_CACHE:
        q, k, v, beta, gk = _chunked(chunk)
        _INTRA_CACHE[chunk] = kda_intra_chunk(q, k, v, beta, gk)
    return _INTRA_CACHE[chunk]


def _inter_call(chunk: int, w, u):
    """One inter-chunk call, with the dispatch reading taken around it.

    The reset happens immediately before the call and the read immediately after,
    so the reading belongs to this call and to no other.
    """
    q, _, _, _, gk = _chunked(chunk)
    intra = _intra_for(chunk)
    reset_inter_dispatch_counters()
    out = kda_inter_chunk(intra.kg, w, u, gk, q, intra.aqk)
    return out, inter_dispatch_counters()


def test_inter_chunk_final_state_matches_the_sequential_torch_scan():
    """The chunked final state matches the sequential scan at every chunk width."""
    ref = _sequential_reference()
    for chunk in CHUNK_SIZES:
        q, _, _, _, _ = _chunked(chunk)
        assert can_run_kernel(q) is True
        intra = _intra_for(chunk)
        out, counters = _inter_call(chunk, intra.w, intra.u)
        assert_close(
            _t(out.final_state), ref.final_state, rtol=RTOL, atol=ATOL,
            name=f"inter[chunk={chunk}].final_state",
        )
        assert counters == (1, 0), (
            f"chunk={chunk}: expected exactly 1 dispatch and 0 torch fallbacks, "
            f"read {counters}; {q.shape[0]} would mean a per-chunk host loop and 0 "
            f"would mean the torch path served this case"
        )


def test_inter_chunk_output_matches_the_sequential_scan_o_equals_h_q():
    """The chunked output matches the sequential scan's ``o = H q``.

    The two are not the same expression: the kernel forms the gate-decayed
    inter-chunk part ``qg @ h_chunk`` plus the intra-chunk part ``Aqk @ v_new``,
    while the reference forms ``H q`` after every single-token update.
    """
    ref = _sequential_reference()
    for chunk in CHUNK_SIZES:
        q, _, _, _, _ = _chunked(chunk)
        assert can_run_kernel(q) is True
        intra = _intra_for(chunk)
        out, counters = _inter_call(chunk, intra.w, intra.u)
        flat_o = _t(out.o).reshape(TOKENS, VDIM)
        assert_close(
            flat_o, ref.o, rtol=RTOL, atol=ATOL, name=f"inter[chunk={chunk}].o",
        )
        assert counters == (1, 0), (
            f"chunk={chunk}: expected exactly 1 dispatch and 0 torch fallbacks, "
            f"read {counters}"
        )


def test_inter_chunk_state_responds_to_the_w_and_u_it_is_handed():
    """Perturbing ``w`` / ``u`` must move the final state away from the reference.

    Two perturbations per chunk width: both replaced by zeros, then ``u``'s last
    row scaled by two. A kernel that ignored ``w`` / ``u`` and re-derived the state
    from its other inputs would still match the reference and fail here.
    """
    ref = _sequential_reference()
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
            with pytest.raises(AssertionError):
                assert_close(
                    _t(out.final_state), ref.final_state, rtol=RTOL, atol=ATOL,
                    name=f"inter[chunk={chunk}].final_state_{tag}_vs_sequential",
                )
            assert counters == (1, 0), (
                f"chunk={chunk} perturbation {tag}: expected exactly 1 dispatch "
                f"and 0 torch fallbacks, read {counters}"
            )


#: The split-sequence cases run at one chunk width, with two chunks per half so the
#: entering state is decayed through a second chunk inside the kernel.
SPLIT_CHUNK = 32
SPLIT_HALF_TOKENS = SPLIT_CHUNK * 2
SPLIT_TOKENS = SPLIT_HALF_TOKENS * 2


def _split_flat(start: int, stop: int):
    """The flat inputs sliced to one token range."""
    return tuple(x[start:stop].contiguous() for x in _flat_inputs())


def _split_pipeline(tokens, state):
    """One half through the real intra-chunk and inter-chunk kernels.

    Returns ``(o_flat, final_state, counters)``, the counters read straight after
    this call's own dispatch.
    """
    q, k, v, beta, gk = tokens
    n_chunks = q.shape[0] // SPLIT_CHUNK
    q_c = q.reshape(n_chunks, SPLIT_CHUNK, KDIM).contiguous()
    k_c = k.reshape(n_chunks, SPLIT_CHUNK, KDIM).contiguous()
    v_c = v.reshape(n_chunks, SPLIT_CHUNK, VDIM).contiguous()
    beta_c = beta.reshape(n_chunks, SPLIT_CHUNK).contiguous()
    gk_c = gk.reshape(n_chunks, SPLIT_CHUNK, KDIM).contiguous()
    intra = kda_intra_chunk(q_c, k_c, v_c, beta_c, gk_c)
    reset_inter_dispatch_counters()
    out = kda_inter_chunk(
        intra.kg, intra.w, intra.u, gk_c, q_c, intra.aqk, state=state
    )
    counters = inter_dispatch_counters()
    o_flat = _t(out.o).reshape(q.shape[0], VDIM)
    return o_flat, _t(out.final_state), counters


def _split_halves(carry: bool):
    """Both halves in order; the second entered with the first's state or not."""
    first = _split_pipeline(_split_flat(0, SPLIT_HALF_TOKENS), None)
    entering = first[1] if carry else None
    second = _split_pipeline(_split_flat(SPLIT_HALF_TOKENS, SPLIT_TOKENS), entering)
    return first, second


def test_inter_chunk_entering_state_carries_a_split_sequence():
    """A sequence run in two halves equals one run over the same flat tokens.

    The second half is entered with the first half's final state. The same halves
    run with the state dropped must not match, which is what shows the kernel
    consumes the operand rather than ignoring it.
    """
    ref = kda_sequential_torch_oracle(*_split_flat(0, SPLIT_TOKENS))

    (first_o, _, _), (second_o, second_state, _) = _split_halves(carry=True)
    joined = torch.cat((first_o, second_o), dim=0)
    assert_close(joined, ref.o, rtol=RTOL, atol=ATOL, name="split.o")
    assert_close(
        second_state, ref.final_state, rtol=RTOL, atol=ATOL, name="split.final_state"
    )

    (_, _, _), (bad_o, bad_state, _) = _split_halves(carry=False)
    with pytest.raises(AssertionError):
        assert_close(
            bad_o, ref.o[SPLIT_HALF_TOKENS:], rtol=RTOL, atol=ATOL, name="no_carry.o"
        )
    with pytest.raises(AssertionError):
        assert_close(
            bad_state, ref.final_state, rtol=RTOL, atol=ATOL,
            name="no_carry.final_state",
        )


def test_inter_chunk_refuses_an_entering_state_it_cannot_serve():
    """A wrong rank, shape or dtype for the entering state each raise by name."""
    tokens = _split_flat(0, SPLIT_HALF_TOKENS)
    refused = {
        "wrong_rank": torch.zeros(VDIM, dtype=torch.float32),
        "wrong_shape": torch.zeros(VDIM, KDIM + 1, dtype=torch.float32),
        "wrong_dtype": torch.zeros(VDIM, KDIM, dtype=torch.bfloat16),
    }
    for label, state in refused.items():
        with pytest.raises(ChunkedRecurrenceError) as raised:
            _split_pipeline(tokens, state)
        assert "state" in str(raised.value), (
            f"the {label} refusal does not name the operand"
        )

    good = torch.zeros(VDIM, KDIM, dtype=torch.float32)
    _, final_state, _ = _split_pipeline(tokens, good)
    assert tuple(final_state.shape) == (VDIM, KDIM), (
        f"returned {tuple(final_state.shape)} for a {(VDIM, KDIM)} entering state; "
        f"the operand and the return are one orientation"
    )


#: The geometry the KDA layer actually enters these kernels at, which is not one of
#: :data:`CHUNK_SIZES`: the layer binds width 128 and resolves chunk width 8.
EXPECTED_PRODUCTION_KDIM = 128
EXPECTED_PRODUCTION_VDIM = 128
EXPECTED_PRODUCTION_CHUNK = 8

#: The checkpoint's gate lower bound, which is what the chunk derivation reads.
EXPECTED_GATE_LOWER_BOUND = -5.0

_PRODUCTION_CACHE: dict[str, object] = {}


def _production_geometry() -> dict:
    """The geometry the KDA layer resolves, read from the layer rather than assumed.

    The chunk width is a layer decision (``_resolve_chunk_size``), so re-deriving
    it here would measure this file's arithmetic instead of the layer's.

    The imports are function-local so that the rest of this module does not
    acquire a vLLM dependency it otherwise has no need of. ``world_size=1``
    because none of the values read here shards.
    """
    if "geometry" not in _PRODUCTION_CACHE:
        from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig
        from vllm_neuron.model.glm5_next.model_fp8 import Glm5NextKDAAttention

        text_config = Glm5NextTextConfig()
        layer = Glm5NextKDAAttention(text_config, world_size=1)
        _PRODUCTION_CACHE["geometry"] = {
            "kdim": int(layer.head_dim),
            "vdim": int(layer.head_size),
            "chunk": int(layer._resolve_chunk_size(None)),
            "gate_lower_bound": float(layer.gate_lower_bound),
            "cache_chunk_size": layer.cache_chunk_size,
        }
    return _PRODUCTION_CACHE["geometry"]


def _production_inputs(seed: int = 20260904):
    """Deterministic inputs at the geometry the layer reports, not at a literal."""
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


def test_intra_chunk_matches_the_reference_at_the_production_geometry():
    """The kernel agrees with the reference at the geometry the KDA layer resolves.

    The chunk width the layer picks, 8, is narrower than any in
    :data:`CHUNK_SIZES`, and the head width is wider, so this covers a shape the
    cases above do not.
    """
    geo = _production_geometry()
    assert geo["cache_chunk_size"] is None, (
        "a config that pinned the chunk dial would return the pinned value, and "
        "this case would be about the dial rather than the layer's derivation"
    )
    assert (geo["kdim"], geo["vdim"], geo["chunk"]) == (
        EXPECTED_PRODUCTION_KDIM,
        EXPECTED_PRODUCTION_VDIM,
        EXPECTED_PRODUCTION_CHUNK,
    ), f"the layer resolved {geo}, which is not the geometry this case covers"

    q, k, v, beta, gk = _production_inputs()
    assert can_run_kernel(q) is True

    reset_dispatch_counters()
    got = kda_intra_chunk(q, k, v, beta, gk)
    nki_dispatch, torch_fallback = dispatch_counters()

    expected = kda_intra_chunk_torch_oracle(q, k, v, beta, gk)
    for field in ("w", "u", "a_inv", "aqk", "kg"):
        assert_close(
            _t(getattr(got, field)), _t(getattr(expected, field)),
            rtol=RTOL, atol=ATOL,
            name=f"production_geometry[chunk={geo['chunk']}].{field}",
        )

    assert nki_dispatch == 1, (
        f"expected exactly 1 dispatch at the production geometry; {N_CHUNKS} would "
        f"mean a per-chunk host loop and 0 would mean the torch path served this case"
    )
    assert torch_fallback == 0


def test_production_geometry_gate_bound_stays_inside_the_declared_limit():
    """The worst cumulative gate at chunk 8 is admissible, and chunk 16 would not be.

    The worst case the checkpoint can produce is every gate entry at
    ``gate_lower_bound``, so the chunk-local cumulative gate reaches
    ``|gate_lower_bound| * chunk``. That is what makes 8 the widest admissible
    chunk rather than an arbitrary choice, and the kernel's own admissibility check
    is asked rather than the arithmetic restated.
    """
    from vllm_neuron.functional.kda.chunked_recurrence import (
        GATE_CUMSUM_ABS_LIMIT,
        can_run_intra_chunk,
    )

    geo = _production_geometry()
    chunk, kdim, vdim = geo["chunk"], geo["kdim"], geo["vdim"]
    bound = abs(geo["gate_lower_bound"])
    assert geo["gate_lower_bound"] == EXPECTED_GATE_LOWER_BOUND

    worst_gk = torch.full(
        (N_CHUNKS, chunk, kdim), geo["gate_lower_bound"], dtype=torch.float32
    )
    worst_case = float(worst_gk.cumsum(dim=1).abs().max().item())

    # Eight additions of -5.0 are exact in fp32, both being a power of two times a
    # small integer, so the measured worst case is the derivation's number exactly.
    assert worst_case == bound * chunk
    assert bound * chunk <= GATE_CUMSUM_ABS_LIMIT
    assert bound * (chunk * 2) > GATE_CUMSUM_ABS_LIMIT

    _, _, _, _, gk = _production_inputs()
    assert float(gk.float().cumsum(dim=1).abs().max().item()) <= worst_case

    reference = torch.zeros((N_CHUNKS, chunk, kdim), dtype=torch.float32)
    assert can_run_intra_chunk(
        reference, N_CHUNKS, chunk, kdim, vdim, worst_case
    ) is True


def test_doubling_stages_is_ceil_log2_of_the_chunk_width():
    """``doubling_stages`` is the Neumann series' stage count, ``ceil(log2(chunk))``."""
    assert doubling_stages(EXPECTED_PRODUCTION_CHUNK) == 3
    assert {c: doubling_stages(c) for c in CHUNK_SIZES} == {32: 5, 64: 6, 128: 7}
