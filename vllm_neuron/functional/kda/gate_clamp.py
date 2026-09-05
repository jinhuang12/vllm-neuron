# SPDX-License-Identifier: Apache-2.0
"""KDA gate: a bounded sigmoid, in NKI.

`inc-glm53f-084`. The KDA gate turns a projected pre-gate tensor into the
per-key-channel log-decay that the recurrence consumes. This module computes that
gate as one device op::

    gate = gate_lower_bound * sigmoid(exp(A_log) * (g + dt_bias))

``A_log`` is one head's learned decay exponent, ``g`` is the projected pre-gate
tensor, ``dt_bias`` is the per-key-channel gate bias, and ``gate_lower_bound`` is
the bound the checkpoint carries. THE AUTHORITY FOR THIS EXPRESSION IS THE SCREEN
NOTE, NOT THIS DOCSTRING: ``pin-feasibility-note-lap-0903b.md`` S2 holds the
reference function, every reference line it cites, and the closed-form comparison
against what this module used to compute. Nothing of it is restated here.

WHY THIS FILE WAS REWRITTEN. `inc-glm53f-037` landed a different function in this
file -- an unbounded softplus floored at the bound -- and the campaign's declared
correctness reference multiplies a saturating sigmoid BY the bound instead. The two
are different functions, not two spellings of one, and the screen note prices the
gap. `inc-glm53f-037`'s history is not re-opened; this block is a second writer
into the same file and the co-authorship is declared on both sides in the plan's
own register.

THE BOUND IS NOW A FACTOR AND NO LONGER A FLOOR, which is why no clamp op appears
below. ``sigmoid`` is bounded in ``(0, 1)`` at every input, so multiplying by a
negative ``gate_lower_bound`` puts the result in ``(gate_lower_bound, 0)`` by
construction. There is nothing left for a floor to do, and an op that can never
change a value is dead weight in a kernel-class chain.

THE FILE AND FUNCTION NAMES ARE HISTORY, NOT DESCRIPTION. ``gate_clamp`` and
``kda_gate_clamp`` are `inc-glm53f-037`'s names, kept on purpose so that the
rewrite is one diff in one file rather than a rename fanning out across call sites
and records. The function no longer clamps anything. A reader who trusts the name
over this paragraph will be wrong about the arithmetic.

THE REFERENCE'S SOFTPLUS LIMB IS NOT IMPLEMENTED. In the reference it is the
``else`` limb of a bound that is never absent for this checkpoint, so it cannot be
reached here (screen note S2). An unreachable branch is untestable code, so it is
not written.

WHY THIS IS A KERNEL AND NOT TORCH GLUE (P13). The gate is a per-token elementwise
op fused into the middle of the KDA chain: the recurrence reads its output directly
as ``exp(g)``. Computing it in torch would move the tensor off device and back
inside a kernel-class chain, which is the fallback P13 forbids. Small size does not
change the classification -- the block says so in its own Substrate bullet.

THE SIGMOID IS ONE ACTIVATION-ENGINE OP, NOT A COMPOSITION. ``nl.sigmoid`` is an
activation function this image's ``nisa.activation`` serves directly, measured
against torch over ``g`` from ``-200`` to ``+200``: worst absolute difference
``5.960464e-08``, which is one float32 step at ``0.5``, and no non-finite value at
any input. The landed softplus form needed nine ISA ops and a branch-free identity
to stay finite; this needs one, and the identity it replaced is gone with it.

THIS OP DECLARES NO MAGNITUDE LIMIT, and the absence is deliberate rather than
overlooked. The chunked and decode modules bound their gate inputs because they
call ``exp`` on a positive quantity, which overflows float32 near ``88``. A sigmoid
cannot overflow: it saturates toward ``0`` or ``1``, so the gate saturates toward
``0`` or toward ``gate_lower_bound``, and both are correct answers. Measured in the
acceptance transcript at ``|g| = 30`` in both signs with zero non-finite elements.

NEITHER IS THERE A CONSTRAINT ON ``gate_lower_bound`` ITSELF, and that is also
deliberate. The landed module rejected a non-positive softplus ``beta`` because its
identity divided by ``beta``; this function only multiplies by the bound, so no
value of it is arithmetically inadmissible and inventing a range check here would
add a rule the design never declared.

INTERNALLY THE TILE IS HELD TRANSPOSED, as ``[D, T]``. The bias is per key channel,
so with ``D`` on the partition axis it becomes a ``[D, 1]`` operand broadcast along
the FREE axis, which is exactly what ``nisa.tensor_scalar`` does; a ``[T, D]``
layout would need a partition-axis broadcast, which it does not do. The boundary
orientation stays ``[T, D]``, so two transposes are paid, one in and one out.

THIS KERNEL CONTAINS NO LOOP AT ALL. The op is elementwise over one tile, so one
dispatch serves one call and the route predicate is an equality: each declared case
makes exactly one call and must read exactly one dispatch.

Tier N harness -- the NKI simulator on a host CPU, no device and no lease::

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
    python -m pytest test/vllm_neuron/functional/kda/test_gate_clamp.py \
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

from vllm_neuron.functional.kda.chunked_recurrence import MAX_TILE, _psum, _sbuf
from vllm_neuron.utils.neuron_utils import can_run_kernel

logger = logging.getLogger(__name__)


class GateClampError(ValueError):
    """Raised for a geometry this kernel does not serve.

    Inadmissibility raises rather than falling back, because falling back would
    ship a torch path for kernel-class work (P13).
    """


@dataclass
class _GateClampDispatchCounters:
    """How the seam below was reached, per process.

    A THIRD counter object in this package, deliberately distinct from the chunked
    module's two and from the decode module's one, so a test can attribute a
    dispatch to this seam and to no other.
    """

    nki_dispatch: int = 0
    torch_fallback: int = 0


_GATE_CLAMP_COUNTERS = _GateClampDispatchCounters()


def reset_gate_clamp_dispatch_counters() -> None:
    """Zero this seam's counters. Call immediately before a case's first call."""
    _GATE_CLAMP_COUNTERS.nki_dispatch = 0
    _GATE_CLAMP_COUNTERS.torch_fallback = 0


def gate_clamp_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last reset.

    ``torch_fallback`` can only ever read ``0``, because this module has no torch
    route to increment it: an inadmissible input raises instead (P13). The counter
    is kept so that a test can state that reading rather than assume it.
    """
    return (
        _GATE_CLAMP_COUNTERS.nki_dispatch,
        _GATE_CLAMP_COUNTERS.torch_fallback,
    )


@nki.jit
def kda_gate_clamp_kernel(g_hbm, a_hbm, bias_hbm, lower):
    """``lower * sigmoid(exp(A_log) * (g + bias))`` for one tile.

    Shapes: ``g_hbm`` is ``[T, D]``, ``a_hbm`` is ``[1, 1]`` (one head's ``A_log``),
    ``bias_hbm`` is ``[D, 1]``. Returns ``[T, D]``. ``lower`` is a compile-time
    scalar.

    No loop, and no branch. Every step is a single elementwise ISA call over the
    transposed ``[D, T]`` tile.
    """
    tokens, kdim = g_hbm.shape

    # In: [T, D] -> [D, T], so the per-channel bias broadcasts along the free axis.
    g_in = _sbuf(tokens, kdim)
    nisa.tensor_copy(dst=g_in, src=nl.load(g_hbm, dtype=nl.float32))
    ps_in = _psum(kdim, tokens)
    nisa.nc_transpose(dst=ps_in, data=g_in)
    biased = _sbuf(kdim, tokens)
    nisa.tensor_copy(dst=biased, src=ps_in)

    bias_col = _sbuf(kdim, 1)
    nisa.tensor_copy(dst=bias_col, src=nl.load(bias_hbm, dtype=nl.float32))
    nisa.tensor_scalar(dst=biased, data=biased, op0=nl.add, operand0=bias_col)

    # z = exp(A_log) * (g + bias). The decay rate scales the pre-activation; it is
    # NOT a factor on the result, which is what makes this a different function
    # from the one this file used to hold.
    exp_a = _sbuf(1, 1)
    nisa.activation(dst=exp_a, data=nl.load(a_hbm, dtype=nl.float32), op=nl.exp)
    scaled = _sbuf(kdim, tokens)
    nisa.tensor_scalar(dst=scaled, data=biased, op0=nl.multiply, operand0=exp_a)

    # gate = lower * sigmoid(z). One activation op, then one scale. The bound is
    # the factor, so the result lands in (lower, 0) without a clamp.
    squashed = _sbuf(kdim, tokens)
    nisa.activation(dst=squashed, data=scaled, op=nl.sigmoid)
    bounded = _sbuf(kdim, tokens)
    nisa.tensor_scalar(dst=bounded, data=squashed, op0=nl.multiply, operand0=lower)

    # Out: [D, T] -> [T, D], back to the boundary orientation.
    ps_out = _psum(tokens, kdim)
    nisa.nc_transpose(dst=ps_out, data=bounded)
    out_sb = _sbuf(tokens, kdim)
    nisa.tensor_copy(dst=out_sb, src=ps_out)
    out_hbm = nl.ndarray((tokens, kdim), dtype=nl.float32, buffer=nl.shared_hbm)
    nl.store(out_hbm, value=out_sb)
    return out_hbm


def _require_gate_clamp_admissible(tokens: int, kdim: int) -> None:
    """Raise unless this kernel serves the geometry, rather than fall back (P13).

    BOTH axes are bounded by ``MAX_TILE``, and the reason is structural rather than
    conventional: both pass through an ``nc_transpose``, whose tensor-engine route
    serves ``128``. ``MAX_TILE`` is imported from the chunked module rather than
    redeclared because it is the SAME quantity there -- the partition-axis limit of
    this image -- unlike a bound that happens to share a number.

    GEOMETRY IS THE ONLY THING CHECKED HERE. The module docstring says why the gate
    bound and the input magnitude are each unconstrained.
    """
    if tokens < 1 or kdim < 1:
        raise GateClampError(
            f"kda_gate_clamp needs at least one token and one key channel; "
            f"got tokens={tokens}, kdim={kdim}"
        )
    if tokens > MAX_TILE or kdim > MAX_TILE:
        raise GateClampError(
            f"kda_gate_clamp cannot serve this input: tokens={tokens} and "
            f"kdim={kdim} must both be in [1, {MAX_TILE}]; both axes pass through "
            f"a transpose, which serves {MAX_TILE}"
        )


def can_run_gate_clamp(reference: Tensor, tokens: int, kdim: int) -> bool:
    """True when the NKI route is available AND serves this geometry."""
    if not can_run_kernel(reference):
        return False
    try:
        _require_gate_clamp_admissible(tokens, kdim)
    except GateClampError:
        return False
    return True


def kda_gate_clamp(
    g: Tensor,
    a_log: Tensor,
    *,
    lower: float,
    bias: Tensor | None = None,
) -> Tensor:
    """The counted seam. ``g`` is ``[T, D]``; the result is ``[T, D]``, float32.

    ``a_log`` is one head's ``A_log`` as a scalar-shaped tensor. ``bias`` is the
    per-key-channel gate bias, which this checkpoint's loader carries as a bare KDA
    leaf named ``dt_bias`` (``weight_loaders_fp8.py:470``); that mapping was
    confirmed against the published checkpoint index at ``evidence-038.md`` §6,
    which found one ``dt_bias`` key per KDA layer, so it is a checked wiring and no
    longer a provisional one. ``None`` means no bias, and it is passed as an exact
    zero column rather than by a second kernel path, because adding ``0.0`` is
    bit-exact -- measured -- so the two are the same computation.

    ``lower`` IS REQUIRED AND HAS NO DEFAULT. It is the checkpoint's
    ``gate_lower_bound``, which the caller reads from
    ``Glm5NextTextConfig.linear_attn_config`` (``model/glm5_next/config.py:170``,
    value ``-5.0``). This module does not read that config and does not carry a copy
    of the value, because ``functional/kda/`` imports nothing from
    ``model/glm5_next/`` -- so a stale second copy of a model constant cannot exist
    here.

    BOTH ``lower`` AND ``bias`` ARE KEYWORD-ONLY. The landed signature took ``bias``
    third positionally; making the new required argument positional in that slot
    would let an old three-argument call bind a bias tensor to ``lower`` and run
    silently. Keyword-only makes that call a ``TypeError`` instead.
    """
    if g.ndim != 2:
        raise GateClampError(f"g must be [tokens, kdim]; got shape {tuple(g.shape)}")
    tokens, kdim = int(g.shape[0]), int(g.shape[1])
    _require_gate_clamp_admissible(tokens, kdim)

    if a_log.numel() != 1:
        raise GateClampError(
            f"a_log must hold exactly one value for one head; got "
            f"{a_log.numel()} in shape {tuple(a_log.shape)}"
        )
    if bias is None:
        bias_col = torch.zeros((kdim, 1), dtype=torch.float32)
    else:
        if bias.numel() != kdim:
            raise GateClampError(
                f"bias must hold one value per key channel; got {bias.numel()} "
                f"for kdim={kdim}"
            )
        bias_col = bias.reshape(kdim, 1).to(torch.float32)

    _GATE_CLAMP_COUNTERS.nki_dispatch += 1
    return wrap_nki(kda_gate_clamp_kernel)(
        g.to(torch.float32),
        a_log.reshape(1, 1).to(torch.float32),
        bias_col,
        float(lower),
    )


def gate_clamp_kernel_identity() -> tuple[str, str]:
    """``(module, qualname)`` of the gate kernel this module authors.

    Read by the acceptance driver to prove the kernel under test is authored here
    rather than imported from the substrate.

    THE UNWRAP IS THE WHOLE READING. ``nki.jit`` returns a wrapper whose own
    ``__module__`` is the substrate's, so reading the attribute off the decorated
    object reports ``nki.framework.kernel`` for an authored kernel and for an
    imported one alike -- the same answer either way, which is no reading at all.
    Unwrapping ``.func`` first is what makes the two cases differ, and it is the
    form the three landed KDA modules already use. The sibling
    ``depthwise_conv1d.kernel_identity`` reads the same way to prove the opposite
    claim, that its seam dispatches to the SUBSTRATE's member.
    """
    func = getattr(kda_gate_clamp_kernel, "func", None)
    target = func if func is not None else kda_gate_clamp_kernel
    return target.__module__, target.__qualname__
