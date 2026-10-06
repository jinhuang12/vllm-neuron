# SPDX-License-Identifier: Apache-2.0
"""KDA gate in NKI: ``lower * sigmoid(exp(a_log) * (g + bias))`` over one tile.

The gate turns a projected pre-gate tensor into the per-key-channel log-decay
that the KDA recurrence consumes. ``a_log`` is one head's learned decay exponent,
``bias`` is the per-key-channel gate bias, and ``lower`` is the checkpoint's
negative ``gate_lower_bound``.

Despite the module name there is no clamp op. ``sigmoid`` is bounded in
``(0, 1)`` at every input, so multiplying by a negative ``lower`` puts the result
in ``(lower, 0)`` by construction and a floor could never change a value. For the
same reason the input needs no magnitude limit: the sibling chunked and decode
modules bound their gate inputs because they exponentiate a positive quantity,
which overflows float32 near 88, while a sigmoid only saturates.

Inside the kernel the tile is held transposed, as ``[D, T]``. The bias is per key
channel, so with ``D`` on the partition axis it becomes a ``[D, 1]`` operand
broadcast along the free axis, which is what ``nisa.tensor_scalar`` serves; a
``[T, D]`` layout would need a partition-axis broadcast, which it does not do.
The boundary orientation stays ``[T, D]``, so two transposes are paid.
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
    """Raised for a geometry this kernel does not serve. There is no torch fallback."""


@dataclass
class _GateClampDispatchCounters:
    """Per-process count of how this module's gate entry point was reached.

    Distinct from the chunked and decode modules' counters so that a test can
    attribute a dispatch to this entry point and to no other.
    """

    nki_dispatch: int = 0
    torch_fallback: int = 0


_GATE_CLAMP_COUNTERS = _GateClampDispatchCounters()


def reset_gate_clamp_dispatch_counters() -> None:
    """Zero this module's dispatch counters."""
    _GATE_CLAMP_COUNTERS.nki_dispatch = 0
    _GATE_CLAMP_COUNTERS.torch_fallback = 0


def gate_clamp_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last reset.

    ``torch_fallback`` always reads ``0``: this module has no torch route, and an
    inadmissible input raises. The counter exists so a test can assert that.
    """
    return (
        _GATE_CLAMP_COUNTERS.nki_dispatch,
        _GATE_CLAMP_COUNTERS.torch_fallback,
    )


@torch._dynamo.assume_constant_result
def _count_nki_dispatch() -> None:
    """Count one kernel dispatch, off the traced graph: a store Dynamo reads becomes a guard."""
    _GATE_CLAMP_COUNTERS.nki_dispatch += 1


@nki.jit
def kda_gate_clamp_kernel(g_hbm, a_hbm, bias_hbm, lower):
    """``lower * sigmoid(exp(a_log) * (g + bias))`` for one tile.

    ``g_hbm`` is ``[T, D]``, ``a_hbm`` is ``[1, 1]`` (one head's ``A_log``),
    ``bias_hbm`` is ``[D, 1]``; ``lower`` is a compile-time scalar. Returns
    ``[T, D]`` float32.
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

    # z = exp(a_log) * (g + bias). The decay rate scales the pre-activation, not
    # the result.
    exp_a = _sbuf(1, 1)
    nisa.activation(dst=exp_a, data=nl.load(a_hbm, dtype=nl.float32), op=nl.exp)
    # The decay scale has to reach kdim partitions before it can multiply this
    # tile. ``tensor_scalar``'s ``operand0`` is a per-partition scalar, broadcast
    # along the free axis only, so passing the ``[1, 1]`` tile straight in makes
    # the compiler refuse the graph: "'nisa.tensor_scalar_arith' op 'operand0'
    # partition total elements 1 != 'dst' partition total elements 16" at
    # kdim=16, and likewise at every kdim above one. ``nl.broadcast_to`` is the
    # member that broadcasts on the partition axis. A plain ``float`` operand0
    # like ``lower`` below stays legal: a compile-time scalar has no partition
    # axis to mismatch.
    exp_col = nl.broadcast_to(exp_a, (kdim, 1))
    scaled = _sbuf(kdim, tokens)
    nisa.tensor_scalar(dst=scaled, data=biased, op0=nl.multiply, operand0=exp_col)

    # gate = lower * sigmoid(z). The bound is the factor, so the result lands in
    # (lower, 0) without a clamp.
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
    """Raise unless this kernel serves ``[tokens, kdim]``.

    Both axes are bounded by ``MAX_TILE`` because both pass through an
    ``nc_transpose``, whose tensor-engine route serves 128 partitions.
    ``MAX_TILE`` is imported from the chunked module rather than redeclared
    because it is the same quantity there, this image's partition-axis limit.
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
    """True when the NKI route is available and serves this geometry."""
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
    """Apply the KDA gate to ``g`` ``[T, D]``; returns ``[T, D]`` float32.

    ``a_log`` is one head's ``A_log`` as a scalar-shaped tensor. ``bias`` is the
    per-key-channel gate bias, which this checkpoint carries as a bare KDA leaf
    named ``dt_bias``. ``None`` means no bias and is passed as an exact zero
    column rather than by a second kernel path, because adding ``0.0`` is
    bit-exact, so the two are the same computation.

    ``lower`` is the checkpoint's ``gate_lower_bound`` and has no default; the
    caller reads it from ``Glm5NextTextConfig.linear_attn_config``. This module
    imports nothing from ``model/glm5_next/``, so no second copy of that model
    constant can go stale here. Both ``lower`` and ``bias`` are keyword-only.
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
        bias_col = torch.zeros((kdim, 1), dtype=torch.float32, device=g.device)
    else:
        if bias.numel() != kdim:
            raise GateClampError(
                f"bias must hold one value per key channel; got {bias.numel()} "
                f"for kdim={kdim}"
            )
        bias_col = bias.reshape(kdim, 1).to(torch.float32)

    _count_nki_dispatch()
    return wrap_nki(kda_gate_clamp_kernel)(
        g.to(torch.float32),
        a_log.reshape(1, 1).to(torch.float32),
        bias_col,
        float(lower),
    )


def gate_clamp_kernel_identity() -> tuple[str, str]:
    """``(module, qualname)`` of the gate kernel this module authors.

    Lets a caller tell an in-tree kernel from an imported one. ``nki.jit`` returns
    a wrapper whose own ``__module__`` is ``nki.framework.kernel`` either way, so
    the attributes must be read off the unwrapped ``.func``.
    """
    func = getattr(kda_gate_clamp_kernel, "func", None)
    target = func if func is not None else kda_gate_clamp_kernel
    return target.__module__, target.__qualname__
