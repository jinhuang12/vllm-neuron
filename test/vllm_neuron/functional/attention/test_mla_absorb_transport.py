# SPDX-License-Identifier: Apache-2.0
"""The absorb seam's x reaches its NKI kernel as stored, with the product unchanged.

The reference is a frozen copy of the kernel as it stood before the transport moved
on chip, fed the host permute it expected. The seam must agree with it exactly and
not within a tolerance: the same values meet the same matmul in the same order, so
any differing element is a defect. One item plants a non-finite element instead and
reads how far it reaches, because the on-chip turn is exact only for finite values.

Run under ``VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1
NEURON_PLATFORM_TARGET_OVERRIDE=trn2``; nothing here reads or sets an environment
variable.
"""

from __future__ import annotations

import pathlib
import re

import torch

import nki
import nki.isa as nisa
import nki.language as nl
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.attention import mla_absorb as mod
from vllm_neuron.functional.attention.mla_absorb import (
    mla_absorb,
    mla_absorb_dispatch_counters,
    reset_mla_absorb_dispatch_counters,
)

#: The three tile extents AS THE FROZEN KERNEL SAW THEM, retyped on purpose: a frozen
#: reference that imported them would follow the live module if they ever moved.
_CONTRACTION_TILE = 128
_SEQUENCE_TILE = 128
_OUTPUT_TILE = 512

#: The declared shapes, ``(seq, heads, contraction, out_features)``.
_SHAPES = ((1, 1, 256, 512), (129, 2, 512, 256), (2048, 2, 256, 512))

#: The host relayout this block removes, and the form the control plants.
_RELAYOUT_FORM = r"\.permute\([^)]*\)\.contiguous\(\)"


def _emit(tag: str, **values: object) -> None:
    """Print one machine-readable reading line for the transcript's reader."""
    body = " ".join(f"{k}={v}" for k, v in values.items())
    print(f"MAT|{tag}|{body}", flush=True)


@nki.jit
def _landed_kernel(xt_hbm, w_hbm):
    """The kernel before the turn moved on chip: it takes x already permuted."""
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


def _inputs(seq: int, heads: int, kdim: int, ndim: int, seed: int = 7):
    """One case's operands: bf16 x as the chain carries it, float32 w as prepared."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn((seq, heads, kdim), generator=gen).to(torch.bfloat16)
    w = torch.randn((heads, kdim, ndim), generator=gen, dtype=torch.float32)
    return x, w


def _landed_absorb(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """The frozen kernel on the host permute it expected, cast back as that seam did."""
    xt = x.permute(1, 2, 0).contiguous().to(torch.float32)
    return wrap_nki(_landed_kernel)(xt, w.contiguous().to(torch.float32)).to(x.dtype)


def _assert_bit_identical(seq: int, heads: int, kdim: int, ndim: int) -> None:
    """The seam equals the frozen kernel exactly; differing elements are the reading."""
    x, w = _inputs(seq, heads, kdim, ndim)
    got, want = mla_absorb(x, w), _landed_absorb(x, w)
    differing = int(torch.ne(got, want).sum().item())
    maxabs = float((got - want).abs().max().item()) if differing else 0.0
    _emit("BIT_IDENTITY", seq=seq, heads=heads, contraction=kdim, out_features=ndim,
          equal=torch.equal(got, want), differing=differing, maxabs=maxabs,
          dtype=got.dtype, shape=tuple(got.shape))
    assert got.dtype == x.dtype
    assert differing == 0
    assert torch.equal(got, want)


def test_bit_identical_to_the_frozen_kernel_at_1_seq_1_head_256_by_512():
    """The decode shape: a one-row sequence tile, two contraction tiles."""
    _assert_bit_identical(*_SHAPES[0])


def test_bit_identical_to_the_frozen_kernel_at_129_seq_2_heads_512_by_256():
    """One row past the sequence tile, four contraction tiles, a ragged output tile."""
    _assert_bit_identical(*_SHAPES[1])


def test_bit_identical_to_the_frozen_kernel_at_2048_seq_2_heads_256_by_512():
    """The prefill bucket's sequence length."""
    _assert_bit_identical(*_SHAPES[2])


def test_no_host_relayout_on_the_seam():
    """The module's own bytes carry no permute-then-contiguous relayout."""
    text = pathlib.Path(mod.__file__).read_text(encoding="utf-8")
    count = len(re.findall(_RELAYOUT_FORM, text))
    _emit("NO_HOST_RELAYOUT", count=count, form=_RELAYOUT_FORM)
    assert count == 0


def test_control_the_relayout_reader_fires_on_a_planted_form():
    """The same reader counts one occurrence in a planted text."""
    planted = "".join(
        ("xt = x", ".permute(1, 2, 0)", ".contiguous()", ".to(torch.float32)")
    )
    count = len(re.findall(_RELAYOUT_FORM, planted))
    _emit("CONTROL_RELAYOUT_READER_FIRES", count=count, form=_RELAYOUT_FORM)
    assert count == 1


def test_the_three_shapes_take_nki_with_zero_fallback():
    """One reset window over the three shapes: three dispatches, no torch fallback."""
    reset_mla_absorb_dispatch_counters()
    for shape in _SHAPES:
        mla_absorb(*_inputs(*shape))
    nki_n, fallback_n = mla_absorb_dispatch_counters()
    _emit("DISPATCH", nki_dispatch=nki_n, torch_fallback=fallback_n)
    assert (nki_n, fallback_n) == (3, 0)


def test_the_kernel_receives_x_as_stored(monkeypatch):
    """At the kernel boundary the first operand is ``x`` itself, uncopied."""
    seen = []
    real_wrap = mod.wrap_nki

    def capture(kernel):
        run = real_wrap(kernel)

        def call(*operands):
            seen.append(operands)
            return run(*operands)

        return call

    monkeypatch.setattr(mod, "wrap_nki", capture)
    x, w = _inputs(8, 2, 256, 512)
    mla_absorb(x, w)
    assert len(seen) == 1
    x_in = seen[0][0]
    _emit("AS_STORED", x_shape=tuple(x_in.shape), x_dtype=x_in.dtype,
          same_storage=x_in.data_ptr() == x.data_ptr())
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
            x, w = _inputs(seq, heads, kdim, ndim)
            clean = _landed_absorb(x, w)
            x[at] = float(kind)
            got, want = mla_absorb(x, w), _landed_absorb(x, w)
            same = torch.eq(_bits(got), _bits(want))
            inside = torch.zeros((seq, heads), dtype=torch.bool)
            inside[_tile_rows(at[0]), at[1]] = True
            outside_differing = int((~same[~inside]).sum().item())
            inside_differing = int((~same[inside]).sum().item())
            planted_row, clean_row = got[at[0], at[1]], clean[at[0], at[1]]
            _emit("NONFINITE", at=at, kind=kind,
                  outside_tile_differing=outside_differing,
                  inside_tile_differing=inside_differing,
                  planted_row_changed=not torch.equal(planted_row, clean_row),
                  planted_nonfinite=int((~torch.isfinite(planted_row)).sum().item()),
                  planted_row_len=int(planted_row.numel()))
            assert outside_differing == 0
            assert not torch.equal(planted_row, clean_row)
