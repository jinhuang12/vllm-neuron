# SPDX-License-Identifier: Apache-2.0
"""MLA absorb: one head-batched matmul, authored in NKI.

`inc-glm53f-097`. This module computes one per-head batched matmul::

    out[s, h, :] = x[s, h, :] @ w[h]

for `x [S, H, K]` and `w [H, K, N]`, and it exists because nothing on disk bridges
two spaces this model's attention chain needs bridged. `-039b`'s `project_qkv`
returns `query [S, H, 256]` at the head width, `project_output` consumes the same
width, and the sparse seam `mla_sparse_attention` works entirely in the LATENT rank
512 -- it consumes `q_lift [S, H, 512]` and returns `[S, H, 512]`. The two
per-head matmuls that cross between them are what this module serves: **absorb-in**
`K = 256, N = 512` and **absorb-out** `K = 512, N = 256`, both at `H = 64`.

The authority for "nothing bridges them" is a read-only measurement, not a reading
of this file's author: `../../../artifacts/campaigns/glm-5.3-flash-port/increments/
invest-042-seam-gap.out`, `INVEST_042_SEAM_GAP=PASS` 12/12, every zero carrying a
control that is shown to fire. It is NOT restated here.

WHAT THIS MODULE DOES NOT DO, and the boundary is load-bearing. It does not split
`kv_b_proj` into `W_UK` and `W_UV`, it does not view a checkpoint weight per head,
and it knows nothing about where its operands come from. Preparing the two weights
once, off the per-forward path, is `inc-glm53f-042`'s work. This module is one seam
that multiplies what it is handed, which is why its gate checks geometry and never
provenance.

WHY THIS IS A KERNEL AND NOT TORCH GLUE (P13). A matmul on the per-forward device
path is kernel-class in this campaign, and the classification is the plan's rather
than this file's: every landed forward-path matmul here is an NKI seam, and torch
has been accepted only for normalisation and glue. Upstream computes this same
absorb with `torch.bmm`; that is upstream's substrate, not this fork's. So this
module carries NO torch absorb path. Section 4 of the plan permits a `functional/`
module a torch path that is (a) the CPU oracle for the test **or** (b) the
constraint-violation fallback the pin's own dispatchers carry. Only (a) is present,
by name: :func:`mla_absorb_torch_oracle`. Region (b) IS DELIBERATELY ABSENT -- an
inadmissible geometry raises, exactly as the landed `functional/kda/gate_clamp.py`
and `mla_projections.py` do, because a fallback for kernel-class work is the defect
and not the remedy.

WHY THIS IS AUTHORED AND NOT CALLED. A read-only inventory of this image's
substrate ran before a line of this file was written -- 507 `nkilib` modules, the
search shown able to find and able to return nothing -- and no runnable member
serves this seam. The four generic matmul candidates are inner-loop primitives
wanting NKI tile objects, sequence-batched helpers writing into a caller's buffer,
or MX-quantised paths; the library's DeepSeek-MLA members DO carry this absorb, but
each fuses it into a whole projection stage behind twelve or more weight, scale,
norm and RoPE operands, and all of them are MX-quantised where this seam is bf16.
The readings and each refusal's grounds are in `../../../artifacts/campaigns/
glm-5.3-flash-port/increments/probe-097-substrate-reading.md`. This module is
therefore an ADAPT of the landed `mla_projections.mla_projection_kernel` with a
head axis added, and it is deliberately the same shape as that kernel so the two
read as one technique applied twice.

THE OUTPUT LAYOUT WAS MEASURED, NOT ASSUMED. `-039a` loads and stores rank-2 tiles
only, so nothing landed said whether this image accepts a rank-3 HBM tensor or a
MIDDLE scalar index in a store -- and the answer decides whether this seam returns
`[S, H, N]` directly or has to permute on the host on every single call. Four
candidate forms were run in the simulator at a shape ragged on all three tiled
axes, against an oracle proven able to mismatch, with the landed rank-2 seam run in
the same process to separate "the image rejects the form" from "the environment is
broken", and an out-of-range kernel to prove a refusal is reported rather than
swallowed. All four were accepted, so the best one is used: the store below indexes
the SEQUENCE axis as a slice and the HEAD axis as a scalar, and the kernel returns
`[S, H, N]` with NO host permute on the output. The transcript is
``probe-097-store-form-host.out`` (`FORMS_ACCEPTED=A,B,C,D`, `CHOSEN_FORM=B`, 8/8).

`x` REACHES HBM IN THE CALLER'S DTYPE, and the cast is the kernel's. Every
`nl.load` below names ``dtype=nl.float32``, so a bf16 tensor is widened on device as
it lands in SBUF, which is the follow-up the earlier revision of this docstring
named and left. Upcasting bf16 to float32 is lossless either way, so the values are
the same; what changes is that `x` no longer crosses HBM at twice its width. `w` is
still handed over as float32 because it already IS float32 where it is prepared, so
its cast is a no-op and removing it would buy nothing.

WHERE THE TRANSPORT HAPPENS. A `nc_matmul` contracts the PARTITION axis, so both
operands must present the contraction extent there. The kernel makes that turn for
`x` itself: each slab is loaded exactly as the caller stores it, ``[sw, kw]`` with
the sequence on the partition axis, and turned on the PE by one `nisa.nc_transpose`
through a PSUM tile into the ``[kw, sw]`` stationary operand. The turn happens once
per (head, sequence tile) ABOVE the output loop and its PSUM tile is dead before the
accumulator is allocated, so the two never want a bank at the same time. For FINITE
operands the matmul sees exactly the values the caller holds, and the result is
bit-identical to the host-relayout kernel this replaces.

WHERE THAT IDENTITY STOPS. The PE turn is a matmul against an identity matrix, and
the vendor rules it not bit-accurate when the tile holds NaN or Inf (Kaena
``nki/isa/neuron_isa.py:221-224``, NeuronCore-v2): a non-finite element can reach
every output of its partition column, so one bad `x` element can touch its own
``[SEQUENCE_TILE, CONTRACTION_TILE]`` slab for that head, where the host relayout
kept it to the one element. Outside that slab's sequence tile nothing changes, and
the planted row itself changes in both kernels; the CPU items prove exactly that and
record what the simulator emits inside the tile. `w` is not turned, so a non-finite
weight behaves the same in both. No host-side finiteness scan stands on this path:
the callers hand this seam a projected query and the sparse seam's own output
(``model_fp8.py:7409``, ``:7422-7424``), neither of which removes a non-finite
value, and a scan per call would cost more than the transport it replaces.

WHY THE WEIGHT IS ``[H, K, N]`` AND NOT ``[H, N, K]``. Per head the weight is
contraction-major for the same partition-axis reason, which is `mla_projection`'s
``[in, out]`` convention with a head axis in front -- not torch's `nn.Linear`
``[out, in]``. The consequence is declared rather than absorbed: whoever prepares
`W_UK` and `W_UV` transposes ONCE when they are built, off the per-forward path,
and never per call.

EVERY LOOP BOUND IS A TRACE-TIME PYTHON INT, which is why the loops below are
`range` and not `nl.affine_range`. Each extent comes from a tensor shape, so it is
known when the kernel is traced, and that lets the ragged final tile on each axis
be an ordinary `min`. Under `nl.affine_range` the same expression would be a trace
value and the ragged tile would silently need the extent to divide exactly. BOTH
production shapes divide exactly on all three tiled axes at `S = 128`, so a broken
ragged path would not show up in them at all -- which is why the acceptance also
runs `S = 1` and `S = 7`, where the sequence tile is ragged and the decode path
lives.

Tier N harness -- the NKI simulator on a host CPU, no device and no lease::

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
    python -m pytest test/vllm_neuron/functional/attention/test_mla_absorb.py \
        -q -s --timeout 60 -p no:cacheprovider
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.attention.mla_projections import (
    CONTRACTION_TILE,
    OUTPUT_TILE,
    SEQUENCE_TILE,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

logger = logging.getLogger(__name__)

#: The three tile extents are IMPORTED from the sibling module rather than retyped
#: here, and the distinction from `mla_projections`' own choice not to import them
#: from `functional/kda/` is deliberate. That module declined a CROSS-PACKAGE reach
#: into another package's private module for numbers that merely happened to be
#: equal. These three are the SAME quantity -- this image's
#: ``nl.tile_size.{pmax, gemm_stationary_fmax, gemm_moving_fmax}``, documented as
#: such where they are declared -- read from a PUBLIC name in the SAME package,
#: whose kernel this one is an adaptation of. Retyping them would let the two
#: kernels' tiling drift apart silently, and a silent divergence between an
#: adaptation and its source is worse than a dependency that is stated out loud.
_TILE_SOURCE = "vllm_neuron.functional.attention.mla_projections"


class MlaAbsorbError(ValueError):
    """Raised for a geometry this kernel does not serve.

    Inadmissibility RAISES rather than falling back, because falling back would
    ship a torch path for kernel-class work (P13).
    """


@dataclass
class _MlaAbsorbDispatchCounters:
    """How the seam below was reached, per process.

    A counter object private to this module, so a test can attribute a dispatch to
    this seam and to no other -- the discipline the landed KDA modules, the sparse
    seam's three counter trios and `mla_projections` all use.
    """

    nki_dispatch: int = 0
    torch_fallback: int = 0


_MLA_ABSORB_COUNTERS = _MlaAbsorbDispatchCounters()


def reset_mla_absorb_dispatch_counters() -> None:
    """Zero this seam's counters. Call immediately before a case's first call."""
    _MLA_ABSORB_COUNTERS.nki_dispatch = 0
    _MLA_ABSORB_COUNTERS.torch_fallback = 0


def mla_absorb_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last reset.

    ``torch_fallback`` can only ever read ``0``, because this module has no torch
    absorb route to increment it: an inadmissible geometry raises instead (P13).
    The counter is kept so a test can STATE that reading rather than assume it,
    which is what makes the zero a measurement instead of an omission.
    """
    return (
        _MLA_ABSORB_COUNTERS.nki_dispatch,
        _MLA_ABSORB_COUNTERS.torch_fallback,
    )


def _sbuf(*shape: int):
    return nl.ndarray(tuple(shape), dtype=nl.float32, buffer=nl.sbuf)


def _psum(rows: int, cols: int):
    return nl.ndarray((rows, cols), dtype=nl.float32, buffer=nl.psum)


def _transpose_tile(src_sb, rows: int, cols: int):
    """``[rows, cols]`` SBUF -> ``[cols, rows]`` SBUF, turned on the PE.

    The destination is PSUM rather than SBUF because a PSUM destination routes the
    turn to the tensor engine and serves the full 128 partitions, while an SBUF one
    runs on Vector and refuses a tile above ``[32, 32]`` -- the rule the landed
    ``chunked_recurrence.py`` records with its own measurement. Every tile here is
    float32 because the load above casts, so the destination dtype matches the
    input's without a choice to make.
    """
    ps = _psum(cols, rows)
    nisa.nc_transpose(dst=ps, data=src_sb)
    tile = _sbuf(cols, rows)
    nisa.tensor_copy(dst=tile, src=ps)
    return tile


@nki.jit
def mla_absorb_kernel(x_hbm, w_hbm):
    """``out[s, h, :] = x[s, h, :] @ w[h]``, tiled on all three inner axes.

    ``x_hbm`` is x AS THE CALLER STORES IT, ``[S, H, K]``, and ``w_hbm`` is
    ``[H, K, N]``. The kernel puts the contraction extent on the partition axis of
    both operands itself: the weight already presents it, and each ``x`` slab is
    turned on the PE by :func:`_transpose_tile`. The module docstring says where
    that turn is exact and where it is not.

    THE TURN IS HOISTED ABOVE THE OUTPUT LOOP. One ``[CONTRACTION_TILE, ktiles, sw]``
    SBUF tile holds every turned slab for this (head, sequence tile), so a slab is
    loaded and turned once instead of once per output tile, and no transpose PSUM
    tile is live while the accumulator is. A whole contraction column cannot be one
    SBUF tile -- the partition axis stops at 128 -- so the tile carries the
    contraction-tile index on its free axis, and the matmul reads back the leading
    ``kw`` partitions of one index.

    The head axis is the OUTER loop and carries no arithmetic of its own -- each
    head is an independent matmul, which is exactly why this adapts a rank-2 kernel
    rather than needing a different technique. The accumulation is in PSUM across
    the contraction loop, ``accumulate`` False on the first contraction tile and
    True on every later one, so one output tile is written to HBM exactly once,
    fully summed.

    The store indexes the sequence axis as a SLICE and the head axis as a SCALAR,
    which is the form measured as accepted in ``probe-097-store-form-host.out``. It
    is what lets this kernel return ``[S, H, N]`` and spares the seam a host-side
    permute on every call.
    """
    seq, heads, kdim = x_hbm.shape
    odim = w_hbm.shape[2]
    ktiles = (kdim + CONTRACTION_TILE - 1) // CONTRACTION_TILE
    out = nl.ndarray((seq, heads, odim), dtype=nl.float32, buffer=nl.shared_hbm)

    for h in range(heads):
        for s0 in range(0, seq, SEQUENCE_TILE):
            sw = min(SEQUENCE_TILE, seq - s0)
            xt = _sbuf(CONTRACTION_TILE, ktiles, sw)
            for ki in range(ktiles):
                k0 = ki * CONTRACTION_TILE
                kw = min(CONTRACTION_TILE, kdim - k0)
                x_sb = _sbuf(sw, kw)
                nisa.tensor_copy(
                    dst=x_sb,
                    src=nl.load(
                        x_hbm[s0:s0 + sw, h, k0:k0 + kw], dtype=nl.float32
                    ),
                )
                nisa.tensor_copy(
                    dst=xt[0:kw, ki, :], src=_transpose_tile(x_sb, sw, kw)
                )
            for o0 in range(0, odim, OUTPUT_TILE):
                ow = min(OUTPUT_TILE, odim - o0)
                acc_ps = _psum(sw, ow)
                for ki in range(ktiles):
                    k0 = ki * CONTRACTION_TILE
                    kw = min(CONTRACTION_TILE, kdim - k0)
                    w_tile = _sbuf(kw, ow)
                    nisa.tensor_copy(
                        dst=w_tile,
                        src=nl.load(
                            w_hbm[h, k0:k0 + kw, o0:o0 + ow], dtype=nl.float32
                        ),
                    )
                    nisa.nc_matmul(
                        dst=acc_ps,
                        stationary=xt[0:kw, ki, :],
                        moving=w_tile,
                        accumulate=(ki > 0),
                    )
                out_sb = _sbuf(sw, ow)
                nisa.tensor_copy(dst=out_sb, src=acc_ps)
                nl.store(out[s0:s0 + sw, h, o0:o0 + ow], value=out_sb)
    return out


def _require_mla_absorb_admissible(seq: int, heads: int, kdim: int, ndim: int) -> None:
    """Raise unless this kernel serves the geometry, rather than fall back (P13).

    ONLY POSITIVITY IS CHECKED, and the absence of an upper bound is deliberate:
    the kernel walks the head axis and all three inner axes in loops, so no extent
    has a magnitude limit. That is the property the refused substrate members lack
    and the reason this module exists, so a geometry check that re-imposed one of
    their bounds would defeat its own purpose.
    """
    if seq < 1 or heads < 1 or kdim < 1 or ndim < 1:
        raise MlaAbsorbError(
            f"mla_absorb needs a positive extent on every axis; got seq={seq}, "
            f"heads={heads}, contraction={kdim}, out_features={ndim}"
        )


def can_run_mla_absorb(
    reference: Tensor, seq: int, heads: int, kdim: int, ndim: int
) -> bool:
    """True when the NKI route is available AND serves this geometry."""
    if not can_run_kernel(reference):
        return False
    try:
        _require_mla_absorb_admissible(seq, heads, kdim, ndim)
    except MlaAbsorbError:
        return False
    return True


def mla_absorb(x: Tensor, w: Tensor) -> Tensor:
    """The counted seam: ``x [S, H, K]``, ``w [H, K, N]``, out ``[S, H, N]``.

    The result carries ``x.dtype``: the accumulation is float32 inside the kernel,
    and the value is cast back on the way out so this seam does not silently widen
    the dtype of the chain it sits in.

    ``w`` is contraction-major PER HEAD -- ``[heads, contraction, out_features]``,
    the orientation the plan block uses when it writes each site as ``K -> N`` --
    and NOT ``[heads, out_features, contraction]``. The module docstring gives the
    reason and names who pays for it.

    EVERY CHECK HERE HAPPENS BEFORE THE COUNTER MOVES, so a refused call leaves
    ``nki_dispatch`` untouched. The acceptance reads that zero as a measurement of
    the gate, which it can only be if a refusal cannot increment.
    """
    if x.ndim != 3:
        raise MlaAbsorbError(
            f"x must be [seq, heads, contraction]; got shape {tuple(x.shape)}"
        )
    if w.ndim != 3:
        raise MlaAbsorbError(
            f"w must be [heads, contraction, out_features]; got shape "
            f"{tuple(w.shape)}"
        )
    seq, heads, kdim = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    if int(w.shape[0]) != heads:
        raise MlaAbsorbError(
            f"w must carry one matrix per head, so w.shape[0] must equal "
            f"x.shape[1]; got w {tuple(w.shape)} against x {tuple(x.shape)}"
        )
    if int(w.shape[1]) != kdim:
        raise MlaAbsorbError(
            f"w is contraction-major per head, so w.shape[1] must equal "
            f"x.shape[2]; got w {tuple(w.shape)} against x {tuple(x.shape)}. A "
            f"[heads, out_features, contraction] weight is the likely cause -- "
            f"this seam does not accept that orientation"
        )
    ndim = int(w.shape[2])
    _require_mla_absorb_admissible(seq, heads, kdim, ndim)

    _MLA_ABSORB_COUNTERS.nki_dispatch += 1
    out = wrap_nki(mla_absorb_kernel)(
        x.contiguous(),
        w.contiguous().to(torch.float32),
    )
    return out.to(x.dtype)


def mla_absorb_torch_oracle(x: Tensor, w: Tensor) -> Tensor:
    """CPU oracle -- section 4 clause (a), and the ONLY torch arithmetic here.

    This is the region the acceptance excludes BY NAME when it screens this module
    for a torch absorb path. It is not a fallback and nothing dispatches to it: the
    seam above never calls it, so no input can reach it except a test's. It returns
    float32 rather than ``x.dtype`` on purpose -- an oracle that rounded to the
    dtype under test could not detect the seam rounding badly.
    """
    return torch.einsum("shk,hkn->shn", x.to(torch.float32), w.to(torch.float32))


def mla_absorb_kernel_identity() -> tuple[str, str]:
    """``(module, qualname)`` of the absorb kernel this module authors.

    Read by the acceptance to prove the kernel under test is authored here rather
    than imported from the substrate. THE UNWRAP IS THE WHOLE READING: ``nki.jit``
    returns a wrapper whose own ``__module__`` is the substrate's, so reading the
    attribute off the decorated object reports the same answer for an authored
    kernel and an imported one alike. Unwrapping ``.func`` first is what makes the
    two cases differ, and it is the form the landed modules use.
    """
    func = getattr(mla_absorb_kernel, "func", None)
    target = func if func is not None else mla_absorb_kernel
    return target.__module__, target.__qualname__
