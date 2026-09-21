# SPDX-License-Identifier: Apache-2.0
"""Tests for the DSA indexer's score GEMM.

Two references are compared against. The first is written here in three whole-tensor torch steps
(``einsum``, ``clamp``, weighted ``sum``) and touches nothing in the module under test, so the
agreement is a real comparison rather than the kernel against itself. The second is a NKI kernel
that receives its operands already transposed on the host -- the layout the kernel's own on-chip
transport replaces. Both kernels see the same bf16 operands in the same order, so they must agree
bit for bit rather than within a tolerance.
"""

from __future__ import annotations

import pytest
import torch

import nki
import nki.isa as nisa
import nki.language as nl
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.dsa import score_gemm as mod
from vllm_neuron.functional.dsa.score_gemm import (
    CAND_TILE,
    CONTRACTION_TILE,
    INDEX_HEAD_DIM,
    TOKEN_TILE,
    ScoreGemmError,
    can_run_dsa_score_gemm,
    dsa_score_gemm,
    reset_score_gemm_dispatch_counters,
    score_gemm_dispatch_counters,
    score_gemm_kernel_identity,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

# Tolerance for the comparison against the torch reference. The operands are bf16 and the
# reduction is fp32, so the only expected difference is a rounding one.
RTOL = 1e-2
ATOL = 1e-5

# The orthonormal case is read absolutely, with rtol 0: the true off-diagonal score there is
# exactly zero, and a relative tolerance around zero would accept anything.
ORTHO_ATOL = 1e-5

# Heads per case. The head axis is a plain loop bound in the kernel, so a larger count multiplies
# simulator time without adding a distinct reading.
HEADS = 4

# The reference kernel below spells its tile extents itself instead of importing the module's, so
# it stays an independent statement of the same function.
_REF_TOKEN_TILE = 128
_REF_CAND_TILE = 512


def _random_inputs(
    tokens: int, cands: int, heads: int = HEADS, seed: int = 7
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One case's three operands at the supported dtypes: bf16 query and key, fp32 weights.

    The weights straddle zero on purpose. Were every weight positive, ``relu(w * s)`` and
    ``w * relu(s)`` would agree and the rectify order would go unmeasured; ``randn`` gives both
    signs, which is also what a learned projection carries.
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

    Built by doubling rather than by a library call, so the basis is this file's own construction
    and not an import shared with the module under test.
    """
    h = torch.ones((1, 1), dtype=torch.float32)
    while h.shape[0] < INDEX_HEAD_DIM:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return (h[:n] * (INDEX_HEAD_DIM**-0.5)).to(torch.bfloat16)


def _orthonormal_inputs(
    tokens: int, cands: int, heads: int = HEADS, seed: int = 23
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Query and key drawn from the same orthonormal basis, so the true score is known exactly.

    Every off-diagonal per-head dot product sums 64 ``+c*c`` terms and 64 ``-c*c`` terms, which
    cancel exactly in fp32 in any order. That isolates the accumulation from the tolerance budget:
    a nonzero off-diagonal is then a real finding rather than a rounding allowance being spent.
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
    """Inputs whose head dimension is not 128, shared by the gate and the validator tests."""
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn((4, HEADS, width), generator=gen).to(torch.bfloat16)
    k = torch.randn((4, width), generator=gen).to(torch.bfloat16)
    weights = torch.randn((4, HEADS), generator=gen, dtype=torch.float32)
    return q, k, weights


def _oracle(q: torch.Tensor, k: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """The torch reference: dot, rectify, weight, sum. Dispatches no NKI kernel.

    Transcribed from ``vllm/v1/attention/ops/rocm_aiter_mla_sparse.py``, keeping its order. The
    upstream per-key dequant scale is absent rather than dropped: a bf16 route quantises nothing.
    """
    per_head = torch.einsum("mhd,nd->mhn", q.float(), k.float())
    return (per_head.clamp(min=0.0) * weights.float().unsqueeze(-1)).sum(dim=1)


@nki.jit
def _reference_kernel(qt_hbm, kt_hbm, w_hbm):
    """The same GEMM over operands transposed on the host, instead of on chip."""
    heads = qt_hbm.shape[0]
    head_dim = qt_hbm.shape[1]
    tokens = qt_hbm.shape[2]
    cands = kt_hbm.shape[1]
    out = nl.ndarray((tokens, cands), dtype=nl.float32, buffer=nl.shared_hbm)
    for m0 in range(0, tokens, _REF_TOKEN_TILE):
        mw = min(_REF_TOKEN_TILE, tokens - m0)
        for n0 in range(0, cands, _REF_CAND_TILE):
            nw = min(_REF_CAND_TILE, cands - n0)
            acc = nl.ndarray((mw, nw), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=acc, value=0.0)
            for h in range(heads):
                q_tile = nl.ndarray((head_dim, mw), dtype=qt_hbm.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=q_tile, src=nl.load(qt_hbm[h, :, m0:m0 + mw]))
                k_tile = nl.ndarray((head_dim, nw), dtype=kt_hbm.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=k_tile, src=nl.load(kt_hbm[:, n0:n0 + nw]))
                ps = nl.ndarray((mw, nw), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=ps, stationary=q_tile, moving=k_tile, accumulate=False)
                raw = nl.ndarray((mw, nw), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=raw, src=ps)
                rect = nl.ndarray((mw, nw), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=rect, op=nl.relu, data=raw)
                wcol = nl.ndarray((mw, 1), dtype=nl.float32, buffer=nl.sbuf)
                wsrc = nl.load(w_hbm[m0:m0 + mw, h:h + 1], dtype=nl.float32)
                nisa.tensor_copy(dst=wcol, src=wsrc)
                scaled = nl.ndarray((mw, nw), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=scaled, data=rect, op0=nl.multiply, operand0=wcol)
                nisa.tensor_tensor(dst=acc, data1=acc, data2=scaled, op=nl.add)
            nl.store(out[m0:m0 + mw, n0:n0 + nw], value=acc)
    return out


def _reference_kernel_scores(
    q: torch.Tensor, k: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """The reference kernel on the host relayout it expects."""
    qt, kt = q.permute(1, 2, 0).contiguous(), k.t().contiguous()
    return wrap_nki(_reference_kernel)(qt, kt, weights.contiguous())


def _assert_matches_reference_kernel(tokens: int, heads: int, cands: int) -> None:
    """The seam equals the reference kernel exactly at one shape."""
    q, k, weights = _random_inputs(tokens=tokens, cands=cands, heads=heads)
    got = dsa_score_gemm(q, k, weights)
    expected = _reference_kernel_scores(q, k, weights)
    differing = int(torch.ne(got, expected).sum().item())
    assert got.dtype == torch.float32
    assert differing == 0, f"{differing} of {got.numel()} elements differ"
    assert torch.equal(got, expected)


def _tile_rows(index: int) -> slice:
    """The 128-element block that holds ``index``: the widest a non-finite element may spread."""
    start = index - index % 128
    return slice(start, start + 128)


def test_matches_the_torch_reference_at_8_tokens_12_candidates():
    """Scores agree with the torch reference at rtol 1e-2, atol 1e-5."""
    q, k, weights = _random_inputs(tokens=8, cands=12)
    got = dsa_score_gemm(q, k, weights)
    assert got.shape == (8, 12)
    assert got.dtype is torch.float32
    torch.testing.assert_close(got.float(), _oracle(q, k, weights), rtol=RTOL, atol=ATOL)


def test_matches_the_torch_reference_on_orthonormal_inputs():
    """Orthonormal query and key agree absolutely at atol 1e-5, rtol 0."""
    tokens, cands = 6, 8
    q, k, weights = _orthonormal_inputs(tokens=tokens, cands=cands)
    got = dsa_score_gemm(q, k, weights)
    torch.testing.assert_close(got.float(), _oracle(q, k, weights), rtol=0.0, atol=ORTHO_ATOL)

    n = min(tokens, cands)
    block = got.float()[:n, :n]
    offdiag = (block - torch.diag(torch.diagonal(block))).abs().max().item()
    assert offdiag <= ORTHO_ATOL
    # The diagonal carries signal, which is what stops the off-diagonal reading above from being
    # the zero of an all-zero tensor.
    assert abs(float(torch.diagonal(block)[0].item())) > 1e-6


def test_matches_the_torch_reference_one_token_past_the_token_tile():
    """``TOKEN_TILE + 1`` tokens forces a second, partial token tile."""
    reset_score_gemm_dispatch_counters()
    tokens = TOKEN_TILE + 1
    q, k, weights = _random_inputs(tokens=tokens, cands=4, heads=1, seed=31)
    got = dsa_score_gemm(q, k, weights)
    assert got.shape == (tokens, 4)
    torch.testing.assert_close(got.float(), _oracle(q, k, weights), rtol=RTOL, atol=ATOL)
    assert score_gemm_dispatch_counters() == (1, 0)


def test_matches_the_torch_reference_one_candidate_past_the_candidate_tile():
    """``CAND_TILE + 1`` candidates forces a second, partial candidate tile."""
    reset_score_gemm_dispatch_counters()
    cands = CAND_TILE + 1
    q, k, weights = _random_inputs(tokens=2, cands=cands, heads=1, seed=37)
    got = dsa_score_gemm(q, k, weights)
    assert got.shape == (2, cands)
    torch.testing.assert_close(got.float(), _oracle(q, k, weights), rtol=RTOL, atol=ATOL)
    assert score_gemm_dispatch_counters() == (1, 0)


def test_matches_the_reference_at_8_tokens_4_heads_12_candidates():
    """A shape well inside one tile of each axis."""
    _assert_matches_reference_kernel(8, 4, 12)


def test_matches_the_reference_at_129_tokens_1_head_4_candidates():
    """One token past the token tile: a ragged one-token stationary tile."""
    _assert_matches_reference_kernel(129, 1, 4)


def test_matches_the_reference_at_2_tokens_1_head_513_candidates():
    """One candidate past the candidate tile: a ragged one-column moving tile."""
    _assert_matches_reference_kernel(2, 1, 513)


def test_matches_the_reference_at_2048_tokens_32_heads_512_candidates():
    """The prefill bucket's shape at the checkpoint's head count."""
    _assert_matches_reference_kernel(2048, 32, 512)


def test_a_non_finite_element_stays_inside_its_tile():
    """A NaN or Inf in ``q`` or ``k`` changes nothing outside its own 128-row tile."""
    tokens, heads, cands = 200, 2, 300
    q_at, k_at = (5, 1, 7), (130, 3)
    for operand, kind in (("q", "nan"), ("q", "inf"), ("k", "nan"), ("k", "inf")):
        q, k, weights = _random_inputs(tokens=tokens, cands=cands, heads=heads)
        clean = _reference_kernel_scores(q, k, weights)
        if operand == "q":
            q[q_at] = float(kind)
        else:
            k[k_at] = float(kind)
        got = dsa_score_gemm(q, k, weights)
        expected = _reference_kernel_scores(q, k, weights)
        same = torch.eq(got.view(torch.int32), expected.view(torch.int32))
        if operand == "q":
            inside = torch.zeros(tokens, dtype=torch.bool)
            inside[_tile_rows(q_at[0])] = True
            outside_differing = int((~same[~inside]).sum().item())
            planted_row, clean_row = got[q_at[0]], clean[q_at[0]]
        else:
            inside = torch.zeros(cands, dtype=torch.bool)
            inside[_tile_rows(k_at[0])] = True
            outside_differing = int((~same[:, ~inside]).sum().item())
            planted_row, clean_row = got[:, k_at[0]], clean[:, k_at[0]]
        assert outside_differing == 0, f"{operand} {kind} changed {outside_differing} elements"
        assert not torch.equal(planted_row, clean_row)


def test_both_input_sets_take_the_kernel_and_not_the_torch_path():
    """Each input set is admitted by the gate, dispatches the kernel once, and never falls back."""
    for q, k, weights in (
        _random_inputs(tokens=8, cands=12),
        _orthonormal_inputs(tokens=6, cands=8),
    ):
        reset_score_gemm_dispatch_counters()
        admitted = can_run_dsa_score_gemm(q, k, weights)
        dsa_score_gemm(q, k, weights)
        assert admitted is True
        assert score_gemm_dispatch_counters() == (1, 0)


def test_the_kernel_receives_q_and_k_as_stored(monkeypatch):
    """At the kernel boundary the first two operands are ``q`` and ``k`` themselves, uncopied."""
    seen = []
    real_wrap = mod.wrap_nki

    def capture(kernel):
        run = real_wrap(kernel)

        def call(*operands):
            seen.append(operands)
            return run(*operands)

        return call

    monkeypatch.setattr(mod, "wrap_nki", capture)
    q, k, weights = _random_inputs(tokens=8, cands=12, heads=4)
    dsa_score_gemm(q, k, weights)
    assert len(seen) == 1
    q_in, k_in = seen[0][0], seen[0][1]
    assert tuple(q_in.shape) == (8, 4, INDEX_HEAD_DIM)
    assert tuple(k_in.shape) == (12, INDEX_HEAD_DIM)
    assert q_in.dtype == torch.bfloat16 and k_in.dtype == torch.bfloat16
    assert q_in.data_ptr() == q.data_ptr()
    assert k_in.data_ptr() == k.data_ptr()


def test_gate_admits_the_supported_shape():
    """The gate admits the supported shape, and kernels can run in this environment at all."""
    q, k, weights = _random_inputs(tokens=8, cands=12)
    assert can_run_kernel() is True
    assert can_run_dsa_score_gemm(q, k, weights) is True


def test_gate_refuses_an_fp32_query():
    """The gate refuses an fp32 query rather than quietly widening the supported dtypes."""
    q, k, weights = _random_inputs(tokens=4, cands=4)
    assert can_run_dsa_score_gemm(q.float(), k, weights) is False


def test_gate_refuses_a_head_dim_the_kernel_does_not_tile():
    """The gate refuses a head dimension other than 128, which the kernel contracts in one tile."""
    q, k, weights = _mismatched_head_dim_inputs()
    assert can_run_dsa_score_gemm(q, k, weights) is False


def test_an_fp32_query_takes_the_torch_path_and_still_computes_correctly():
    """The gate refuses fp32, the torch path serves it once, and the answer is still right."""
    reset_score_gemm_dispatch_counters()
    q, k, weights = _random_inputs(tokens=4, cands=4)
    q32 = q.float()
    admitted = can_run_dsa_score_gemm(q32, k, weights)
    got = dsa_score_gemm(q32, k, weights)
    assert admitted is False
    assert score_gemm_dispatch_counters() == (0, 1)
    torch.testing.assert_close(got, _oracle(q32, k, weights), rtol=RTOL, atol=ATOL)


def test_refuses_a_non_3d_q():
    """A 2-D query raises, naming the rank it expected."""
    q, k, weights = _random_inputs(tokens=4, cands=4)
    with pytest.raises(ScoreGemmError, match="q must be 3-D"):
        dsa_score_gemm(q[:, 0, :], k, weights)


def test_refuses_a_k_whose_feature_width_mismatches():
    """A key whose feature width does not match the head dimension raises."""
    q, _k, weights = _random_inputs(tokens=4, cands=4)
    gen = torch.Generator().manual_seed(43)
    bad_k = torch.randn((4, 64), generator=gen).to(torch.bfloat16)
    with pytest.raises(ScoreGemmError, match="feature width must match"):
        dsa_score_gemm(q, bad_k, weights)


def test_refuses_weights_of_the_wrong_shape():
    """Weights that are not ``[tokens, heads]`` raise rather than broadcasting into something."""
    q, k, weights = _random_inputs(tokens=4, cands=4)
    with pytest.raises(ScoreGemmError, match="weights must be"):
        dsa_score_gemm(q, k, weights[:, :1])


def test_refuses_a_head_dim_that_is_not_128():
    """A 64-wide head raises from the seam, which is distinct from the gate merely refusing it."""
    q, k, weights = _mismatched_head_dim_inputs()
    with pytest.raises(ScoreGemmError, match="contracts exactly"):
        dsa_score_gemm(q, k, weights)


def test_kernel_identity_is_none_before_any_dispatch():
    """``None`` before any dispatch: the reading that separates "no kernel ran" from "one ran"."""
    reset_score_gemm_dispatch_counters()
    assert score_gemm_kernel_identity() is None


def test_kernel_identity_after_dispatch_names_the_nki_kernel():
    """After a dispatch the identity names this module's kernel, not the ``@nki.jit`` decorator."""
    reset_score_gemm_dispatch_counters()
    q, k, weights = _random_inputs(tokens=4, cands=4)
    dsa_score_gemm(q, k, weights)
    identity = score_gemm_kernel_identity()
    assert identity is not None
    module_name, qualname = identity
    assert module_name == "vllm_neuron.functional.dsa.score_gemm"
    assert qualname == "_score_gemm_nki"


def test_tile_constants_match_the_isa_tile_extents():
    """The module's tile literals equal the ISA's own extents.

    The module spells them as literals instead of reading ``nl.tile_size`` at import, because
    ``nl.tile_size.psum_num_banks`` raises ``RuntimeError: No backend set`` outside an activated
    backend. Reading the three safe extents here is what stops the literals drifting.
    """
    assert TOKEN_TILE == int(nl.tile_size.gemm_stationary_fmax)
    assert CAND_TILE == int(nl.tile_size.gemm_moving_fmax)
    assert CONTRACTION_TILE == int(nl.tile_size.pmax)


def test_contraction_tile_equals_index_head_dim():
    """The contraction extent equals the head dimension, so there is no contraction loop."""
    assert CONTRACTION_TILE == INDEX_HEAD_DIM
