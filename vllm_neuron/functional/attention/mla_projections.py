# SPDX-License-Identifier: Apache-2.0
"""MLA low-rank projections: a tiled matmul, authored in NKI.

`inc-glm53f-039a`. This module computes one low-rank projection of the MLA
attention half::

    y[S, O] = x[S, I] @ w[I, O]

and it exists because the substrate members that would otherwise have served these
projections REFUSE this checkpoint's geometry. The authority for that refusal is
`../../../artifacts/campaigns/glm-5.3-flash-port/increments/evidence-072.md`, whose
verdict table reports BOTH rows REFUSE, 2/2 decided, 0 unknown, and the increment
plan's own `inc-glm53f-039a` block, which carries every deciding bound with its
file and line. NEITHER IS RESTATED HERE, and no symbol of either refused member is
named in this file -- the acceptance screens this module's source for exactly that,
so naming one would be a defect rather than a courtesy to the reader.

WHY THIS IS A KERNEL AND NOT TORCH GLUE (P13). A low-rank projection is per-token
device computation in the middle of the attention chain: its output feeds the
attention kernels directly. The plan block classifies this increment KERNEL-CLASS
on measurement rather than on preference -- the substrate refuses the extents, so
there is no member to call, and a torch implementation would be the fallback P13
forbids. This module therefore carries NO torch projection path at all. Section 4
of the plan permits a `functional/` module a torch path that is (a) the CPU oracle
for the test **or** (b) the constraint-violation fallback the pin's own dispatchers
carry. Only (a) is present below, by name: :func:`mla_projection_torch_oracle`.
Region (b) IS DELIBERATELY ABSENT -- an inadmissible geometry raises, exactly as
the landed `functional/kda/gate_clamp.py` does, because a fallback for
kernel-class work is the defect and not the remedy.

THE THREE TILE BOUNDS ARE THIS IMAGE'S, READ FROM IT AND THEN MEASURED. A
`nc_matmul` contracts the PARTITION axis, so both operands must present the
contraction extent on that axis. The three bounds are `nl.tile_size.pmax` = 128 on
the partition axis, `nl.tile_size.gemm_stationary_fmax` = 128 on the stationary
free axis and `nl.tile_size.gemm_moving_fmax` = 512 on the moving free axis. Each
was ALSO measured by refusal: one element past each bound raises, and the three
messages name `nc_version.gen3`, which is this campaign's trn2 target. The
readings are in ``probe-039a-matmul.out`` in this campaign's increments directory.

WHY THE WEIGHT ARRIVES AS ``[I, O]`` AND NOT AS torch's ``[O, I]``. The contraction
extent must sit on the partition axis for both operands, so the kernel needs the
weight contraction-major. Accepting torch's ``nn.Linear`` orientation instead would
force either a per-tile on-device transpose -- which caps the output tile at 128
rather than 512 and so quadruples the matmul count -- or a host-side copy of up to
64 MB on every call. Both are real costs paid per call, whereas a projection weight
is CONSTANT: transposing it once when it is loaded costs nothing per call. So the
seam declares the orientation it can serve for free and the caller supplies it.
THE CONSEQUENCE IS DECLARED RATHER THAN ABSORBED: `inc-glm53f-039b`, which wires
these five sites, transposes each weight ONCE at load time and never per call.

THERE IS NO BIAS PARAMETER, and the absence is deliberate. This checkpoint's
config does not model `attention_bias` -- it is one of the keys
`Glm5NextTextConfig` drops -- so a bias limb could not be reached by any
configuration this campaign supports, and an unreachable branch is untestable
code. The landed `gate_clamp.py` left the reference's softplus limb out for the
same reason.

EVERY LOOP BOUND IS A TRACE-TIME PYTHON INT, which is why the loops below are
`range` and not `nl.affine_range`. Each extent comes from a tensor shape, so it is
known when the kernel is traced; that lets the ragged final tile on each of the
three axes be an ordinary `min`. Under `nl.affine_range` the same expression would
be a trace value and the ragged tile would silently need the extent to divide
exactly. All five widths this checkpoint declares DO divide exactly, so the bug
would not have shown up in the declared cases at all -- which is precisely why the
ragged path is measured in its own right (``probe-039a-kernel.out``: ragged on all
three axes, a single row, a single column, and a sequence longer than one
stationary tile).

Tier N harness -- the NKI simulator on a host CPU, no device and no lease::

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
    python -m pytest test/vllm_neuron/functional/attention/test_mla_projections_kernel.py \
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

from vllm_neuron.utils.neuron_utils import can_run_kernel

logger = logging.getLogger(__name__)

#: Partition-axis extent, ``nl.tile_size.pmax``. The contraction axis is tiled to
#: this because a ``nc_matmul`` contracts the partition axis.
CONTRACTION_TILE = 128

#: Stationary free-axis extent, ``nl.tile_size.gemm_stationary_fmax``. The
#: sequence axis rides the stationary operand, so this bounds the sequence TILE --
#: not the sequence, which the loop below walks.
SEQUENCE_TILE = 128

#: Moving free-axis extent, ``nl.tile_size.gemm_moving_fmax``. The output axis
#: rides the moving operand, so output tiles are four times wider than sequence
#: tiles. That asymmetry is the image's, not a choice made here.
OUTPUT_TILE = 512

#: These three numbers are declared HERE and not imported from
#: ``vllm_neuron/functional/kda/``. Two of them equal a constant that package
#: already holds, but equal numbers are not the same quantity: reaching into
#: another package's private module to borrow a value is the private
#: cross-module-import pattern this campaign already carries as a review item, and
#: one more instance of it is not this increment's to add.
_TILE_PROVENANCE = "nl.tile_size.{pmax, gemm_stationary_fmax, gemm_moving_fmax}"


class MlaProjectionError(ValueError):
    """Raised for a geometry this kernel does not serve.

    Inadmissibility RAISES rather than falling back, because falling back would
    ship a torch path for kernel-class work (P13).
    """


@dataclass
class _MlaProjectionDispatchCounters:
    """How the seam below was reached, per process.

    A counter object private to this module, so a test can attribute a dispatch to
    this seam and to no other -- the same discipline the four landed KDA modules
    each use with their own counter objects.
    """

    nki_dispatch: int = 0
    torch_fallback: int = 0


_MLA_PROJECTION_COUNTERS = _MlaProjectionDispatchCounters()


def reset_mla_projection_dispatch_counters() -> None:
    """Zero this seam's counters. Call immediately before a case's first call."""
    _MLA_PROJECTION_COUNTERS.nki_dispatch = 0
    _MLA_PROJECTION_COUNTERS.torch_fallback = 0


def mla_projection_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last reset.

    ``torch_fallback`` can only ever read ``0``, because this module has no torch
    projection route to increment it: an inadmissible geometry raises instead
    (P13). The counter is kept so a test can STATE that reading rather than assume
    it, which is what makes the zero a measurement.
    """
    return (
        _MLA_PROJECTION_COUNTERS.nki_dispatch,
        _MLA_PROJECTION_COUNTERS.torch_fallback,
    )


def _sbuf(rows: int, cols: int):
    return nl.ndarray((rows, cols), dtype=nl.float32, buffer=nl.sbuf)


def _psum(rows: int, cols: int):
    return nl.ndarray((rows, cols), dtype=nl.float32, buffer=nl.psum)


@nki.jit
def mla_projection_kernel(xt_hbm, w_hbm):
    """``y[S, O] = x[S, I] @ w[I, O]``, tiled on all three axes.

    ``xt_hbm`` is x ALREADY transposed to ``[I, S]`` and ``w_hbm`` is ``[I, O]``:
    both present the contraction extent on the partition axis, which is what
    ``nc_matmul`` contracts. The module docstring says why the caller owns that
    orientation.

    The accumulation is in PSUM across the contraction loop -- ``accumulate`` is
    False on the first contraction tile and True on every later one -- so one
    output tile is written to HBM exactly once, fully summed.
    """
    idim, seq = xt_hbm.shape
    _, odim = w_hbm.shape
    out = nl.ndarray((seq, odim), dtype=nl.float32, buffer=nl.shared_hbm)

    for s0 in range(0, seq, SEQUENCE_TILE):
        sw = min(SEQUENCE_TILE, seq - s0)
        for o0 in range(0, odim, OUTPUT_TILE):
            ow = min(OUTPUT_TILE, odim - o0)
            acc_ps = _psum(sw, ow)
            for k0 in range(0, idim, CONTRACTION_TILE):
                kw = min(CONTRACTION_TILE, idim - k0)
                x_tile = _sbuf(kw, sw)
                nisa.tensor_copy(
                    dst=x_tile,
                    src=nl.load(xt_hbm[k0:k0 + kw, s0:s0 + sw], dtype=nl.float32),
                )
                w_tile = _sbuf(kw, ow)
                nisa.tensor_copy(
                    dst=w_tile,
                    src=nl.load(w_hbm[k0:k0 + kw, o0:o0 + ow], dtype=nl.float32),
                )
                nisa.nc_matmul(
                    dst=acc_ps,
                    stationary=x_tile,
                    moving=w_tile,
                    accumulate=(k0 > 0),
                )
            out_sb = _sbuf(sw, ow)
            nisa.tensor_copy(dst=out_sb, src=acc_ps)
            nl.store(out[s0:s0 + sw, o0:o0 + ow], value=out_sb)
    return out


def _require_mla_projection_admissible(seq: int, idim: int, odim: int) -> None:
    """Raise unless this kernel serves the geometry, rather than fall back (P13).

    ONLY POSITIVITY IS CHECKED, and the absence of an upper bound is the whole
    point of this module. The kernel walks all three axes in tiles, so no extent
    has a magnitude limit -- which is exactly the property the refused substrate
    members lack, and the reason this increment exists. A geometry check that
    re-imposed one of their bounds would defeat its own purpose.
    """
    if seq < 1 or idim < 1 or odim < 1:
        raise MlaProjectionError(
            f"mla_projection needs a positive extent on every axis; got "
            f"seq={seq}, in_features={idim}, out_features={odim}"
        )


def can_run_mla_projection(reference: Tensor, seq: int, idim: int, odim: int) -> bool:
    """True when the NKI route is available AND serves this geometry."""
    if not can_run_kernel(reference):
        return False
    try:
        _require_mla_projection_admissible(seq, idim, odim)
    except MlaProjectionError:
        return False
    return True


def mla_projection(x: Tensor, weight: Tensor) -> Tensor:
    """The counted seam. ``x`` is ``[S, I]``, ``weight`` is ``[I, O]``, out ``[S, O]``.

    The result is float32. ``weight`` is CONTRACTION-MAJOR -- ``[in_features,
    out_features]``, the orientation the plan block itself uses when it writes each
    site as ``I -> O`` -- and NOT torch's ``nn.Linear`` ``[O, I]``. The module
    docstring gives the measured reason and names who pays for it.
    """
    if x.ndim != 2:
        raise MlaProjectionError(
            f"x must be [seq, in_features]; got shape {tuple(x.shape)}"
        )
    if weight.ndim != 2:
        raise MlaProjectionError(
            f"weight must be [in_features, out_features]; got shape "
            f"{tuple(weight.shape)}"
        )
    seq, idim = int(x.shape[0]), int(x.shape[1])
    if int(weight.shape[0]) != idim:
        raise MlaProjectionError(
            f"weight is contraction-major, so weight.shape[0] must equal "
            f"x.shape[1]; got weight {tuple(weight.shape)} against x "
            f"{tuple(x.shape)}. A [out_features, in_features] weight is the "
            f"likely cause -- this seam does not accept that orientation"
        )
    odim = int(weight.shape[1])
    _require_mla_projection_admissible(seq, idim, odim)

    _MLA_PROJECTION_COUNTERS.nki_dispatch += 1
    return wrap_nki(mla_projection_kernel)(
        x.t().contiguous().to(torch.float32),
        weight.to(torch.float32),
    )


def mla_projection_torch_oracle(x: Tensor, weight: Tensor) -> Tensor:
    """CPU oracle -- section 4 clause (a), and the ONLY torch arithmetic here.

    This is the region the acceptance excludes BY NAME when it screens this module
    for a torch projection path. It is not a fallback and nothing dispatches to it:
    the seam above never calls it, so no input can reach it except a test's.
    """
    return x.to(torch.float32) @ weight.to(torch.float32)


def mla_projection_kernel_identity() -> tuple[str, str]:
    """``(module, qualname)`` of the projection kernel this module authors.

    Read by the acceptance driver to prove the kernel under test is authored here
    rather than imported from the substrate. THE UNWRAP IS THE WHOLE READING:
    ``nki.jit`` returns a wrapper whose own ``__module__`` is the substrate's, so
    reading the attribute off the decorated object reports the same answer for an
    authored kernel and an imported one alike. Unwrapping ``.func`` first is what
    makes the two cases differ, and it is the form the landed KDA modules use.
    """
    func = getattr(mla_projection_kernel, "func", None)
    target = func if func is not None else mla_projection_kernel
    return target.__module__, target.__qualname__
