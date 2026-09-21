# SPDX-License-Identifier: Apache-2.0
"""Tests for the MLA low-rank projection kernel against a torch reference.

The five projection widths are the published checkpoint's own. Two of them --
16,384 and 32,768 out_features against a 4,096 bound -- are wider than the vendor
QKV kernel admits, which is why this kernel exists.

All five widths divide exactly on the three tiled axes, so a kernel that dropped a
short final tile would still agree on every one of them. The ragged set below
leaves a short final tile on each axis instead; its extents are derived from the
module's own tile constants, so the cases stay ragged if a tile width moves.
"""

from __future__ import annotations

import torch

from vllm_neuron.functional.attention import mla_projections as MP

#: The five projection sites as ``(name, in_features, out_features)``, at the
#: sequence length prefill runs. The widths are the published checkpoint's own
#: (``hf-config.json`` and the weight index). ``SEQ_LEN`` is above the vendor
#: threshold of 96 that selects the contraction-bounded sub-kernel: below it the
#: vendor code routes to a token-generation sub-kernel that bounds nothing, where
#: these widths would measure nothing either.
SEQ_LEN = 128
PROJECTION_SITES = (
    ("q_a_proj", 4096, 1536),
    ("q_b_proj", 1536, 16384),
    ("kv_a_proj_with_mqa", 4096, 512),
    ("kv_b_proj", 512, 32768),
    ("o_proj", 16384, 4096),
)

#: The kernel accumulates its tiles in a different order from ``einsum``, so
#: agreement is to a tolerance rather than exact.
RTOL = 1e-2
ATOL = 1e-5


def _ragged_cases() -> tuple[tuple[str, int, int, int], ...]:
    """Shapes that leave a short final tile, so the ``min`` bounds are load-bearing.

    Extents are derived from the module's tile constants and never retyped, so a
    tile width change keeps them ragged instead of silently making them exact. Each
    axis is covered both below its tile and past it, because the two fail
    differently: below the tile the loop runs once with a short bound, past it the
    loop runs again with a short final bound.
    """
    s, i, o = MP.SEQUENCE_TILE, MP.CONTRACTION_TILE, MP.OUTPUT_TILE
    return (
        #  name                    S         I         O
        ("s_single_row",           1,        i,        o),
        ("s_below_tile",           63,       i,        o),
        ("s_one_past_tile",        s + 1,    i,        o),
        ("s_two_tiles_ragged",     s + 37,   i,        o),
        ("i_below_tile",           s,        63,       o),
        ("i_one_past_tile",        s,        i + 1,    o),
        ("i_ragged_wide",          s,        i + 72,   o),
        ("o_below_tile",           s,        i,        300),
        ("o_one_past_tile",        s,        i,        o + 1),
        ("o_ragged_wide",          s,        i,        o + 88),
        ("all_three_ragged",       s + 1,    i + 72,   o + 88),
        ("all_three_below_tile",   7,        13,       29),
    )


RAGGED_CASES = _ragged_cases()


def _operands(seq: int, idim: int, odim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Inputs for one site, scaled down so a 16,384-long contraction stays in range.

    The generator is seeded per shape, so a failure is reproducible from the shape
    alone and does not depend on test execution order.
    """
    gen = torch.Generator().manual_seed(1000 * idim + odim)
    x = torch.randn(seq, idim, generator=gen, dtype=torch.float32) * 0.05
    w = torch.randn(idim, odim, generator=gen, dtype=torch.float32) * 0.05
    return x, w


def _torch_reference(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """An independently written reference for the projection.

    It goes through ``einsum`` with the contraction named explicitly, a different
    torch entry point from the ``@`` the module's own reference uses, and it states
    the index pattern the kernel is supposed to implement. A per-column Python
    reduction would be more independent still, but at the widest site that is
    16,384 interpreted iterations, which does not fit the suite timeout.
    """
    return torch.einsum("si,io->so", x, w)


def test_matches_the_torch_reference_at_the_five_projection_widths() -> None:
    """Every projection width agrees with ``einsum`` inside the tolerance."""
    for name, idim, odim in PROJECTION_SITES:
        x, w = _operands(SEQ_LEN, idim, odim)
        got = MP.mla_projection(x, w)
        ref = _torch_reference(x, w)
        assert got.shape == (SEQ_LEN, odim), (
            f"{name}: kernel returned {tuple(got.shape)}, expected "
            f"{(SEQ_LEN, odim)}"
        )
        torch.testing.assert_close(got, ref, rtol=RTOL, atol=ATOL)


def test_matches_the_torch_reference_on_ragged_tiles() -> None:
    """The same comparison at shapes that leave a short final tile on each axis."""
    for name, seq, idim, odim in RAGGED_CASES:
        x, w = _operands(seq, idim, odim)
        got = MP.mla_projection(x, w)
        ref = _torch_reference(x, w)
        assert got.shape == (seq, odim), (
            f"{name}: kernel returned {tuple(got.shape)}, expected {(seq, odim)}"
        )
        # A tile the kernel never wrote reads as whatever the buffer held, which
        # surfaces here before the value comparison gets a chance to.
        assert bool(torch.isfinite(got).all()), (
            f"{name}: the result holds a non-finite value, which is what a tile "
            f"the kernel never wrote looks like"
        )
        torch.testing.assert_close(got, ref, rtol=RTOL, atol=ATOL)


def test_every_projection_width_takes_the_kernel_exactly_once() -> None:
    """Each width is admitted by the gate, dispatches NKI once, and never falls back."""
    for name, idim, odim in PROJECTION_SITES:
        x, w = _operands(SEQ_LEN, idim, odim)
        assert MP.can_run_mla_projection(x, SEQ_LEN, idim, odim) is True, (
            f"{name}: can_run_mla_projection is not True, so the NKI route is "
            f"unavailable"
        )

        MP.reset_mla_projection_dispatch_counters()
        MP.mla_projection(x, w)
        nki_dispatch, torch_fallback = MP.mla_projection_dispatch_counters()
        assert nki_dispatch == 1, (
            f"{name}: seam counted {nki_dispatch} NKI dispatches, expected exactly 1"
        )
        assert torch_fallback == 0, (
            f"{name}: seam counted {torch_fallback} torch fallbacks; this module "
            f"has no torch route, so the only correct reading is 0"
        )


def test_kernel_identity_names_this_modules_own_kernel() -> None:
    """The dispatched kernel is defined here and not imported from the vendor tree."""
    module, qualname = MP.mla_projection_kernel_identity()
    assert module == MP.__name__, (
        f"the kernel under test reports module {module}.{qualname}, not "
        f"{MP.__name__}: an imported vendor kernel would read exactly this way"
    )
