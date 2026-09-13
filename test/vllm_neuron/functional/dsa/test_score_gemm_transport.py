# SPDX-License-Identifier: Apache-2.0
"""The score GEMM's operands reach its NKI kernel as stored, with the scores unchanged bit for bit.

The reference is a frozen copy of the kernel as it stood before the transport moved on chip, fed by
the host-side relayout it expected. The seam must agree with it exactly, not within a tolerance: the
same bf16 operands meet the same matmul in the same order, so any differing element is a defect.
Run under ``VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1
NEURON_PLATFORM_TARGET_OVERRIDE=trn2``; nothing here reads or sets an environment variable.
"""

from __future__ import annotations

import pathlib
import re

import torch

import nki
import nki.isa as nisa
import nki.language as nl
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.dsa import score_gemm as mod
from vllm_neuron.functional.dsa.score_gemm import (
    INDEX_HEAD_DIM,
    dsa_score_gemm,
    reset_score_gemm_dispatch_counters,
    score_gemm_dispatch_counters,
)

_TOKEN_TILE = 128
_CAND_TILE = 512
_RELAYOUT_FORMS = (r"\.permute\([^)]*\)\.contiguous\(\)", r"\.t\(\)\.contiguous\(\)")


def _emit(tag: str, **values: object) -> None:
    """Print one machine-readable reading line for the transcript's reader."""
    body = " ".join(f"{k}={v}" for k, v in values.items())
    print(f"SGT|{tag}|{body}", flush=True)


@nki.jit
def _landed_kernel(qt_hbm, kt_hbm, w_hbm):
    """The kernel before the transport moved on chip: it takes the relaid-out operands."""
    heads = qt_hbm.shape[0]
    head_dim = qt_hbm.shape[1]
    tokens = qt_hbm.shape[2]
    cands = kt_hbm.shape[1]
    out = nl.ndarray((tokens, cands), dtype=nl.float32, buffer=nl.shared_hbm)
    for m0 in range(0, tokens, _TOKEN_TILE):
        mw = min(_TOKEN_TILE, tokens - m0)
        for n0 in range(0, cands, _CAND_TILE):
            nw = min(_CAND_TILE, cands - n0)
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


def _inputs(tokens: int, heads: int, cands: int, seed: int = 7):
    """One case's operands: bf16 query and key, fp32 weights of both signs."""
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn((tokens, heads, INDEX_HEAD_DIM), generator=gen).to(torch.bfloat16)
    k = torch.randn((cands, INDEX_HEAD_DIM), generator=gen).to(torch.bfloat16)
    weights = torch.randn((tokens, heads), generator=gen, dtype=torch.float32)
    return q, k, weights


def _landed_scores(q: torch.Tensor, k: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """The frozen kernel on the host relayout it expected."""
    qt, kt = q.permute(1, 2, 0).contiguous(), k.t().contiguous()
    return wrap_nki(_landed_kernel)(qt, kt, weights.contiguous())


def _assert_bit_identical(tokens: int, heads: int, cands: int) -> None:
    """The seam equals the frozen kernel exactly at one shape; differing elements are the reading."""
    q, k, weights = _inputs(tokens, heads, cands)
    got = dsa_score_gemm(q, k, weights)
    want = _landed_scores(q, k, weights)
    differing = int(torch.ne(got, want).sum().item())
    maxabs = float((got - want).abs().max().item()) if differing else 0.0
    _emit("BIT_IDENTITY", tokens=tokens, heads=heads, cands=cands, equal=torch.equal(got, want),
          differing=differing, maxabs=maxabs, dtype=got.dtype, shape=tuple(got.shape))
    assert got.dtype == torch.float32
    assert differing == 0
    assert torch.equal(got, want)


def _relayout_counts(text: str) -> tuple[int, ...]:
    """How many times each host relayout form occurs in a text."""
    return tuple(len(re.findall(form, text)) for form in _RELAYOUT_FORMS)


def test_bit_identical_to_the_landed_kernel_at_8_tokens_4_heads_12_cands():
    """A small shape well inside one tile of each axis."""
    _assert_bit_identical(8, 4, 12)


def test_bit_identical_to_the_landed_kernel_at_129_tokens_1_head_4_cands():
    """One token past the token tile: a ragged one-token stationary tile."""
    _assert_bit_identical(129, 1, 4)


def test_bit_identical_to_the_landed_kernel_at_2_tokens_1_head_513_cands():
    """One candidate past the candidate tile: a ragged one-column moving tile."""
    _assert_bit_identical(2, 1, 513)


def test_bit_identical_to_the_landed_kernel_at_2048_tokens_32_heads_512_cands():
    """The prefill bucket's shape at the checkpoint's head count."""
    _assert_bit_identical(2048, 32, 512)


def test_no_host_relayout_on_the_nki_route():
    """The module's own bytes carry neither host relayout form."""
    counts = _relayout_counts(pathlib.Path(mod.__file__).read_text(encoding="utf-8"))
    _emit("NO_HOST_RELAYOUT", permute_contiguous=counts[0], t_contiguous=counts[1])
    assert counts == (0, 0)


def test_control_the_relayout_reader_fires_on_a_planted_form():
    """The same reader counts one of each form in a planted text."""
    planted = "".join(
        ("qt = q", ".permute(1, 2, 0)", ".contiguous()\nkt = k", ".t()", ".contiguous()\n")
    )
    counts = _relayout_counts(planted)
    _emit("CONTROL_RELAYOUT_READER_FIRES", permute_contiguous=counts[0], t_contiguous=counts[1])
    assert counts == (1, 1)


def test_the_four_shapes_take_nki_with_zero_fallback():
    """One reset window over the four shapes: four NKI dispatches, no torch fallback."""
    reset_score_gemm_dispatch_counters()
    for tokens, heads, cands in ((8, 4, 12), (129, 1, 4), (2, 1, 513), (2048, 32, 512)):
        dsa_score_gemm(*_inputs(tokens, heads, cands))
    nki_n, fallback_n = score_gemm_dispatch_counters()
    _emit("DISPATCH", nki_dispatch=nki_n, torch_fallback=fallback_n)
    assert (nki_n, fallback_n) == (4, 0)


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
    q, k, weights = _inputs(8, 4, 12)
    dsa_score_gemm(q, k, weights)
    assert len(seen) == 1
    q_in, k_in = seen[0][0], seen[0][1]
    _emit("AS_STORED", q_shape=tuple(q_in.shape), k_shape=tuple(k_in.shape), q_dtype=q_in.dtype,
          k_dtype=k_in.dtype, q_same_storage=q_in.data_ptr() == q.data_ptr(),
          k_same_storage=k_in.data_ptr() == k.data_ptr())
    assert tuple(q_in.shape) == (8, 4, INDEX_HEAD_DIM)
    assert tuple(k_in.shape) == (12, INDEX_HEAD_DIM)
    assert q_in.dtype == torch.bfloat16 and k_in.dtype == torch.bfloat16
    assert q_in.data_ptr() == q.data_ptr()
    assert k_in.data_ptr() == k.data_ptr()
