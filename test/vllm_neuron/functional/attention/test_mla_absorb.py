# SPDX-License-Identifier: Apache-2.0
"""Tests for the MLA absorb kernel, a per-head batched matmul.

Two references are used. The module's float32 reference is compared inside the
tolerance below, because the kernel accumulates in a different order. The
predecessor kernel further down -- a copy of this kernel from before the transpose
moved on chip, fed the host permute it expected -- is compared exactly: the same
values meet the same matmul in the same order, so any differing element is a
defect. One test plants a non-finite element instead and reads how far it reaches,
because the on-chip transpose is exact only for finite values.
"""

from __future__ import annotations

import math

import pytest
import torch

import nki
import nki.isa as nisa
import nki.language as nl
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.attention import mla_absorb as MA

#: The head count this checkpoint declares for the MLA attention half. Both
#: production shapes carry it.
HEADS = 64

#: `128` is a whole sequence tile; `1` and `7` are ragged against it, and `1` is the
#: decode step, which is the length production spends almost all of its time at.
SEQ_CASES = (1, 7, 128)

#: absorb-in: `query [S, H, 256] @ W_UK [H, 256, 512]` -> `q_lift [S, H, 512]`.
ABSORB_IN_K, ABSORB_IN_N = 256, 512

#: absorb-out: the sparse seam's `[S, H, 512]` @ `W_UV [H, 512, 256]` -> the width
#: `project_output` consumes.
ABSORB_OUT_K, ABSORB_OUT_N = 512, 256

#: The kernel accumulates in PSUM in a different order from the reference's
#: `einsum`, so agreement is to a tolerance rather than exact.
RTOL = 1e-2
ATOL = 1e-5


def _ragged_cases() -> tuple[tuple[str, int, int, int, int], ...]:
    """Shapes that leave a short final tile, so the ``min`` bounds are load-bearing.

    Both production shapes divide exactly on the contraction and output axes -- `K`
    is 256 or 512 against a 128-wide contraction tile, `N` is 512 or 256 against a
    512-wide output tile -- so they cannot fail when a ragged tail is dropped and
    returns memory the kernel never wrote.

    The extents are derived from the kernel module's own tile constants and never
    retyped, so a tile width change keeps them ragged instead of silently making
    them exact. Each axis is covered both below its tile and past it, because the
    two fail differently: below the tile the loop runs once with a short bound, past
    it the loop runs again with a short final bound. The head count is varied down
    to 1 as well -- `H` is not tiled, but a head loop that ran the wrong number of
    times would be invisible at the single production `H = 64`.
    """
    k, n, s = MA.CONTRACTION_TILE, MA.OUTPUT_TILE, MA.SEQUENCE_TILE
    return (
        #  label                    S        H   K         N
        ("k_below_tile",            2,       3,  63,       n),
        ("k_one_past_tile",         2,       3,  k + 1,    n),
        ("k_ragged_wide",           2,       2,  k + 72,   n),
        ("n_below_tile",            2,       3,  k,        300),
        ("n_one_past_tile",         2,       2,  k,        n + 1),
        ("n_ragged_wide",           2,       2,  k,        n + 88),
        ("s_one_past_tile",         s + 1,   2,  k,        n),
        ("single_head",             5,       1,  k + 13,   n + 29),
        ("all_axes_ragged",         s + 7,   3,  k + 72,   n + 88),
        ("all_axes_below_tile",     7,       2,  13,       29),
    )


RAGGED_CASES = _ragged_cases()


def _operands(seq: int, heads: int, kdim: int, ndim: int, seed: int):
    """bf16 `x [S, H, K]` and `w [H, K, N]`, with `w` scaled by ``1/sqrt(K)``.

    The scale is how a projection weight is initialised, and it is what puts output
    elements at unit scale: with unit-variance operands an output element is a sum
    of `K` products, so it has standard deviation ``sqrt(K)`` -- about 16 at
    `K = 256` -- and near a cancellation zero the float32 summation-order difference
    then exceeds `atol` while `rtol` still has nothing to bite on.

    The seed is per case, so a failure names one reproducible case rather than
    depending on how many cases ran before it.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(seq, heads, kdim, generator=g).to(torch.bfloat16)
    w = (torch.randn(heads, kdim, ndim, generator=g) / math.sqrt(kdim)).to(
        torch.bfloat16
    )
    return x, w


def _assert_matches_the_float32_reference(
    label: str, seq: int, kdim: int, ndim: int, seed: int, heads: int = HEADS
) -> None:
    """Run one shape through the seam and compare it to the float32 reference.

    The reference is given the same bf16 tensors the kernel got, upcast, rather than
    a separately drawn float32 pair: otherwise the comparison would measure the
    difference between two random draws.
    """
    x, w = _operands(seq, heads, kdim, ndim, seed)
    got = MA.mla_absorb(x, w)
    expected = MA.mla_absorb_torch_oracle(x, w)

    assert tuple(got.shape) == (seq, heads, ndim), (
        f"{label}: seam returned {tuple(got.shape)}, expected {(seq, heads, ndim)}"
    )
    assert got.dtype == x.dtype, (
        f"{label}: the output dtype must be x.dtype ({x.dtype}); got {got.dtype}"
    )

    # Compared in float32. The seam's own output dtype is asserted just above, so
    # widening here loses no reading -- and comparing a bf16 result against a
    # bf16-rounded reference would hide the seam rounding badly, which is the one
    # thing the reference exists to catch.
    torch.testing.assert_close(
        got.to(torch.float32), expected, rtol=RTOL, atol=ATOL
    )


def test_absorb_in_matches_the_float32_reference():
    """absorb-in agrees with the float32 reference at every sequence length."""
    MA.reset_mla_absorb_dispatch_counters()
    for seq in SEQ_CASES:
        _assert_matches_the_float32_reference(
            "absorb_in", seq, ABSORB_IN_K, ABSORB_IN_N, seed=9700 + seq
        )

    nki_dispatch, torch_fallback = MA.mla_absorb_dispatch_counters()
    assert nki_dispatch == len(SEQ_CASES), (
        f"one NKI dispatch per call, so {len(SEQ_CASES)} calls read "
        f"{len(SEQ_CASES)}; got {nki_dispatch}"
    )
    assert torch_fallback == 0, (
        f"this module has no torch absorb route, so this counter can only read 0; "
        f"got {torch_fallback}"
    )


def test_absorb_out_matches_the_float32_reference():
    """absorb-out agrees with the float32 reference at every sequence length."""
    MA.reset_mla_absorb_dispatch_counters()
    for seq in SEQ_CASES:
        _assert_matches_the_float32_reference(
            "absorb_out", seq, ABSORB_OUT_K, ABSORB_OUT_N, seed=9800 + seq
        )

    nki_dispatch, torch_fallback = MA.mla_absorb_dispatch_counters()
    assert nki_dispatch == len(SEQ_CASES)
    assert torch_fallback == 0


def test_ragged_tiles_match_the_float32_reference():
    """The same comparison at shapes that leave a short final tile on each axis."""
    for index, (label, seq, heads, kdim, ndim) in enumerate(RAGGED_CASES):
        _assert_matches_the_float32_reference(
            f"ragged/{label}", seq, kdim, ndim, seed=9750 + index, heads=heads
        )


def test_kernel_identity_names_this_modules_own_kernel():
    """The dispatched kernel is defined here and not imported from the vendor tree."""
    module, qualname = MA.mla_absorb_kernel_identity()
    assert module == "vllm_neuron.functional.attention.mla_absorb", (
        f"the kernel under test is defined in {module}, not in this fork's "
        f"mla_absorb module -- an imported vendor kernel would read exactly this way"
    )
    assert qualname == "mla_absorb_kernel"


def test_inadmissible_geometries_are_refused_and_never_dispatch():
    """Four bad operand pairs raise ``MlaAbsorbError`` and leave the counter at 0."""
    MA.reset_mla_absorb_dispatch_counters()
    heads, kdim, ndim = HEADS, ABSORB_IN_K, ABSORB_IN_N
    good_x, good_w = _operands(4, heads, kdim, ndim, seed=9900)

    cases = (
        ("x_is_rank_2", good_x[:, 0, :], good_w),
        ("w_is_rank_2", good_x, good_w[0]),
        ("head_count_mismatch", good_x, good_w[: heads - 1]),
        ("contraction_mismatch", good_x, good_w[:, : kdim - 1, :]),
    )

    for label, bad_x, bad_w in cases:
        with pytest.raises(MA.MlaAbsorbError) as excinfo:
            MA.mla_absorb(bad_x, bad_w)
        assert str(excinfo.value), f"{label}: refused with an empty message"

    nki_dispatch, torch_fallback = MA.mla_absorb_dispatch_counters()
    assert nki_dispatch == 0, (
        f"every geometry check runs before the counter moves, so a refused call "
        f"must leave it at 0; got {nki_dispatch}"
    )
    assert torch_fallback == 0


#: The three tile extents as the predecessor kernel below saw them, retyped on
#: purpose: a reference that imported them would follow the live module if they
#: ever moved.
_CONTRACTION_TILE = 128
_SEQUENCE_TILE = 128
_OUTPUT_TILE = 512


@nki.jit
def _reference_kernel(xt_hbm, w_hbm):
    """This kernel's predecessor: it takes ``x`` already permuted on the host."""
    heads, kdim, seq = xt_hbm.shape
    odim = w_hbm.shape[2]
    out = nl.ndarray((seq, heads, odim), dtype=nl.float32, buffer=nl.shared_hbm)
    for h in range(heads):
        for s0 in range(0, seq, _SEQUENCE_TILE):
            sw = min(_SEQUENCE_TILE, seq - s0)
            for o0 in range(0, odim, _OUTPUT_TILE):
                ow = min(_OUTPUT_TILE, odim - o0)
                acc_ps = nl.ndarray((sw, ow), dtype=nl.float32, buffer=nl.psum)
                for k0 in range(0, kdim, _CONTRACTION_TILE):
                    kw = min(_CONTRACTION_TILE, kdim - k0)
                    x_tile = nl.ndarray((kw, sw), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(
                        dst=x_tile,
                        src=nl.load(
                            xt_hbm[h, k0:k0 + kw, s0:s0 + sw], dtype=nl.float32
                        ),
                    )
                    w_tile = nl.ndarray((kw, ow), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(
                        dst=w_tile,
                        src=nl.load(
                            w_hbm[h, k0:k0 + kw, o0:o0 + ow], dtype=nl.float32
                        ),
                    )
                    nisa.nc_matmul(
                        dst=acc_ps,
                        stationary=x_tile,
                        moving=w_tile,
                        accumulate=(k0 > 0),
                    )
                out_sb = nl.ndarray((sw, ow), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=out_sb, src=acc_ps)
                nl.store(out[s0:s0 + sw, h, o0:o0 + ow], value=out_sb)
    return out


def _exact_operands(seq: int, heads: int, kdim: int, ndim: int, seed: int = 7):
    """bf16 `x` as the chain carries it, float32 `w` as the caller prepares it.

    Unscaled, unlike ``_operands``: the comparison against the predecessor kernel is
    exact, so the operand magnitude has nothing to do with whether it passes.
    """
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn((seq, heads, kdim), generator=gen).to(torch.bfloat16)
    w = torch.randn((heads, kdim, ndim), generator=gen, dtype=torch.float32)
    return x, w


def _reference_absorb(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """The predecessor kernel on the host permute it expected, cast back as it did."""
    xt = x.permute(1, 2, 0).contiguous().to(torch.float32)
    return wrap_nki(_reference_kernel)(xt, w.contiguous().to(torch.float32)).to(x.dtype)


def _assert_bit_identical(seq: int, heads: int, kdim: int, ndim: int) -> None:
    """The seam equals the predecessor kernel element for element, with no tolerance."""
    x, w = _exact_operands(seq, heads, kdim, ndim)
    got, expected = MA.mla_absorb(x, w), _reference_absorb(x, w)
    differing = int(torch.ne(got, expected).sum().item())
    assert differing == 0, (
        f"S={seq} H={heads} K={kdim} N={ndim}: {differing} of {got.numel()} "
        f"elements differ from the predecessor kernel, max abs difference "
        f"{float((got - expected).abs().max().item()):.6f}"
    )


def test_matches_the_reference_at_1_seq_1_head_256_by_512():
    """The decode shape: a one-row sequence tile, two contraction tiles."""
    _assert_bit_identical(1, 1, 256, 512)


def test_matches_the_reference_at_129_seq_2_heads_512_by_256():
    """One row past the sequence tile, four contraction tiles, a ragged output tile."""
    _assert_bit_identical(129, 2, 512, 256)


def test_matches_the_reference_at_2048_seq_2_heads_256_by_512():
    """The prefill bucket's sequence length."""
    _assert_bit_identical(2048, 2, 256, 512)


def test_the_kernel_receives_x_as_stored(monkeypatch):
    """At the kernel boundary the first operand is ``x`` itself, uncopied."""
    seen = []
    real_wrap = MA.wrap_nki

    def capture(kernel):
        run = real_wrap(kernel)

        def call(*operands):
            seen.append(operands)
            return run(*operands)

        return call

    monkeypatch.setattr(MA, "wrap_nki", capture)
    x, w = _exact_operands(8, 2, 256, 512)
    MA.mla_absorb(x, w)
    assert len(seen) == 1
    x_in = seen[0][0]
    assert tuple(x_in.shape) == (8, 2, 256)
    assert x_in.dtype == x.dtype
    assert x_in.data_ptr() == x.data_ptr()


def _tile_rows(index: int) -> slice:
    """The 128-row sequence tile that holds ``index``."""
    start = index - index % _SEQUENCE_TILE
    return slice(start, start + _SEQUENCE_TILE)


def _bits(value: torch.Tensor) -> torch.Tensor:
    """The tensor's bit pattern, so a NaN compares equal to the same NaN."""
    assert value.dtype == torch.bfloat16
    return value.view(torch.int16)


def test_a_non_finite_element_stays_inside_its_sequence_tile():
    """A NaN or Inf in ``x`` changes nothing outside its own head and sequence tile."""
    seq, heads, kdim, ndim = 200, 2, 256, 64
    for at in ((5, 1, 7), (130, 0, 200)):
        for kind in ("nan", "inf"):
            x, w = _exact_operands(seq, heads, kdim, ndim)
            clean = _reference_absorb(x, w)
            x[at] = float(kind)
            got, expected = MA.mla_absorb(x, w), _reference_absorb(x, w)
            same = torch.eq(_bits(got), _bits(expected))
            inside = torch.zeros((seq, heads), dtype=torch.bool)
            inside[_tile_rows(at[0]), at[1]] = True
            outside_differing = int((~same[~inside]).sum().item())
            planted_row, clean_row = got[at[0], at[1]], clean[at[0], at[1]]
            assert outside_differing == 0, (
                f"{kind} at {at}: {outside_differing} elements outside its own head "
                f"and sequence tile changed"
            )
            assert not torch.equal(planted_row, clean_row), (
                f"{kind} at {at}: the row holding the planted element did not change"
            )
