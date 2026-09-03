# SPDX-License-Identifier: Apache-2.0
"""KDA gate activation with its lower bound, fused, in NKI.

`inc-glm53f-037`. The KDA gate turns a projected pre-gate tensor into the
per-key-channel log-decay the recurrence consumes, and this checkpoint bounds that
decay from below at ``-5.0``. This module computes the activation and the bound as
one device op.

WHY THIS IS A KERNEL AND NOT TORCH GLUE (P13). The gate is a per-token elementwise
op fused into the middle of the KDA chain: the recurrence reads its output
directly as ``exp(g)``. Computing it in torch would move the tensor off device and
back inside a kernel-class chain, which is the fallback P13 forbids. Small size
does not change the classification -- the block says so in its own Substrate
bullet.

THE ACTIVATION IS PINNED UPSTREAM AND IS QUOTED, NOT INVENTED. It is
``kda_gate_fwd_kernel`` in the vLLM the fork targets, at
``vllm/model_executor/layers/fla/ops/kda.py:1541-1600`` (1647 lines, sha256
``84e0bfd395f6e739...``, the same file and digest ``invest-035`` §7.3 pinned), with
its python entry ``fused_kda_gate`` at ``:1603-1647``. In the Triton source's own
terms::

    b_a = -exp(A[h])                                          # :1558-1559
    b_g = g + g_bias                                          # :1584-1590
    sp  = where(beta*b_g > threshold, b_g,
                (1/beta) * log(1 + exp(beta*b_g)))            # :1595-1597
    b_y = b_a * sp                                            # :1598

``beta`` defaults to ``1.0`` and ``threshold`` to ``20.0`` (``:1608-1609``), and
the output is float32 (``:1627``). Since ``-exp(A) < 0`` and ``softplus >= 0``, the
gate is never positive, so a LOWER bound is the only bound that can bite.

THE BOUND IS THIS CAMPAIGN'S OWN LANDED VALUE, NOT AN UPSTREAM ONE. It is
``"gate_lower_bound": -5.0`` in ``vllm_neuron/model/glm5_next/config.py:115``,
inside ``linear_attn_config`` (``:110-117``). Upstream's KDA carries no ``-5.0``
anywhere -- searched, and the search came back empty -- so the composition this
module implements is the port's, and only the activation half is upstream's.

THIS KERNEL COMPUTES SOFTPLUS BY THE BRANCH-FREE IDENTITY, NOT BY UPSTREAM'S
THRESHOLD FORM::

    softplus_beta(z) = (1/beta) * ( max(beta*z, 0) + log(1 + exp(-|beta*z|)) )

The two are the same function. Upstream needs its ``threshold`` because
``log(1 + exp(y))`` overflows float32 near ``y = 88`` and the linear branch is what
saves it; this form never evaluates ``exp`` at a positive argument, so it needs no
branch and no threshold at all. Measured: at ``g = 100`` upstream's threshold
branch returns ``-100.0``, the same form WITHOUT the branch returns ``-inf``, and
this identity returns ``-100.0``. The two forms are therefore not bit-identical,
and the block's tolerance is what covers the difference -- measured at
``2.384e-07`` against a ``1e-5`` bound on the acceptance case.

INTERNALLY THE TILE IS HELD TRANSPOSED, as ``[D, T]``. The bias is per key channel,
so with ``D`` on the partition axis it becomes a ``[D, 1]`` operand broadcast along
the FREE axis, which is exactly what ``nisa.tensor_scalar`` does; a ``[T, D]``
layout would need a partition-axis broadcast, which it does not do. The boundary
orientation stays ``[T, D]``, the layout upstream's ``fused_kda_gate`` returns, so
two transposes are paid, one in and one out, and both are exact -- measured at
``0.000e+00``.

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


#: The gate's lower bound, read from this campaign's landed model config rather
#: than chosen here: ``vllm_neuron/model/glm5_next/config.py:115`` declares
#: ``"gate_lower_bound": -5.0`` inside ``linear_attn_config``. A LOWER bound is the
#: only bound that can bite, because the activation is never positive.
KDA_GATE_LOWER_BOUND = -5.0

#: ``fused_kda_gate``'s own defaults, at ``kda.py:1608-1609``. ``beta`` reaches the
#: kernel; ``threshold`` reaches only the torch oracle, because the kernel's
#: branch-free identity has no threshold to take. That asymmetry is the point of
#: the identity, not an omission.
GATE_SOFTPLUS_BETA = 1.0
GATE_SOFTPLUS_THRESHOLD = 20.0


class GateClampError(ValueError):
    """Raised for a geometry or a ``beta`` this kernel does not serve.

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
    """``(nki_dispatch, torch_fallback)`` since the last reset."""
    return (
        _GATE_CLAMP_COUNTERS.nki_dispatch,
        _GATE_CLAMP_COUNTERS.torch_fallback,
    )


@nki.jit
def kda_gate_clamp_kernel(g_hbm, a_hbm, bias_hbm, beta, lower):
    """``clamp(-exp(A) * softplus_beta(g + bias), min=lower)`` for one tile.

    Shapes: ``g_hbm`` is ``[T, D]``, ``a_hbm`` is ``[1, 1]`` (one head's ``A_log``),
    ``bias_hbm`` is ``[D, 1]``. Returns ``[T, D]``. ``beta`` and ``lower`` are
    compile-time scalars.

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

    # z = beta * (g + bias)
    scaled = _sbuf(kdim, tokens)
    nisa.tensor_scalar(dst=scaled, data=biased, op0=nl.multiply, operand0=beta)

    # softplus_beta(z) = (1/beta) * ( max(z, 0) + log(1 + exp(-|z|)) ).
    # -|z| is min(z, -z), so no absolute-value op is needed and exp never sees a
    # positive argument.
    hinge = _sbuf(kdim, tokens)
    nisa.tensor_scalar(dst=hinge, data=scaled, op0=nl.maximum, operand0=0.0)
    negated = _sbuf(kdim, tokens)
    nisa.tensor_scalar(dst=negated, data=scaled, op0=nl.multiply, operand0=-1.0)
    neg_abs = _sbuf(kdim, tokens)
    nisa.tensor_tensor(dst=neg_abs, data1=scaled, data2=negated, op=nl.minimum)
    decayed = _sbuf(kdim, tokens)
    nisa.activation(dst=decayed, data=neg_abs, op=nl.exp)
    shifted = _sbuf(kdim, tokens)
    nisa.tensor_scalar(dst=shifted, data=decayed, op0=nl.add, operand0=1.0)
    logged = _sbuf(kdim, tokens)
    nisa.activation(dst=logged, data=shifted, op=nl.log)
    softplus = _sbuf(kdim, tokens)
    nisa.tensor_tensor(dst=softplus, data1=hinge, data2=logged, op=nl.add)
    unbeta = _sbuf(kdim, tokens)
    nisa.tensor_scalar(dst=unbeta, data=softplus, op0=nl.multiply, operand0=1.0 / beta)

    # y = -exp(A) * softplus, then the lower bound.
    exp_a = _sbuf(1, 1)
    nisa.activation(dst=exp_a, data=nl.load(a_hbm, dtype=nl.float32), op=nl.exp)
    flipped = _sbuf(kdim, tokens)
    nisa.tensor_scalar(dst=flipped, data=unbeta, op0=nl.multiply, operand0=-1.0)
    gated = _sbuf(kdim, tokens)
    nisa.tensor_scalar(dst=gated, data=flipped, op0=nl.multiply, operand0=exp_a)
    bounded = _sbuf(kdim, tokens)
    nisa.tensor_scalar(dst=bounded, data=gated, op0=nl.maximum, operand0=lower)

    # Out: [D, T] -> [T, D], back to the layout upstream returns.
    ps_out = _psum(tokens, kdim)
    nisa.nc_transpose(dst=ps_out, data=bounded)
    out_sb = _sbuf(tokens, kdim)
    nisa.tensor_copy(dst=out_sb, src=ps_out)
    out_hbm = nl.ndarray((tokens, kdim), dtype=nl.float32, buffer=nl.shared_hbm)
    nl.store(out_hbm, value=out_sb)
    return out_hbm


def _require_gate_clamp_admissible(tokens: int, kdim: int, beta: float) -> None:
    """Raise unless this kernel serves the geometry, rather than fall back (P13).

    BOTH axes are bounded by ``MAX_TILE``, and the reason is structural rather than
    conventional: both pass through an ``nc_transpose``, whose tensor-engine route
    serves ``128``. ``MAX_TILE`` is imported from the chunked module rather than
    redeclared because it is the SAME quantity there -- the partition-axis limit of
    this image -- unlike a bound that happens to share a number.
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
    if not beta > 0.0:
        raise GateClampError(
            f"kda_gate_clamp needs a positive softplus beta; got beta={beta}. "
            f"The identity divides by beta, so zero or negative is not a value "
            f"this op has"
        )


def can_run_gate_clamp(
    reference: Tensor, tokens: int, kdim: int, beta: float = GATE_SOFTPLUS_BETA
) -> bool:
    """True when the NKI route is available AND serves this geometry.

    NOTE THAT THIS OP DECLARES NO MAGNITUDE LIMIT, and the absence is deliberate
    rather than overlooked. The chunked and decode modules bound their gate inputs
    because they call ``exp`` on a positive quantity, which overflows float32 near
    ``88``. This kernel calls ``exp`` only on ``-|z|``, which is never positive, so
    the composition cannot overflow: an enormous positive input saturates into the
    lower bound instead, which is the correct answer and is measured in the
    acceptance transcript.
    """
    if not can_run_kernel(reference):
        return False
    try:
        _require_gate_clamp_admissible(tokens, kdim, beta)
    except GateClampError:
        return False
    return True


def kda_gate_clamp(
    g: Tensor,
    a_log: Tensor,
    bias: Tensor | None = None,
    beta: float = GATE_SOFTPLUS_BETA,
    lower: float = KDA_GATE_LOWER_BOUND,
) -> Tensor:
    """The counted seam. ``g`` is ``[T, D]``; the result is ``[T, D]``, float32.

    ``a_log`` is one head's ``A_log`` as a scalar-shaped tensor. ``bias`` is the
    per-key-channel gate bias (``dt_bias``, mapped for this checkpoint at
    ``weight_loaders_fp8.py:417``); ``None`` means no bias, and it is passed as an
    exact zero column rather than by a second kernel path, because adding ``0.0``
    is bit-exact -- measured -- so the two are the same computation.
    """
    if g.ndim != 2:
        raise GateClampError(f"g must be [tokens, kdim]; got shape {tuple(g.shape)}")
    tokens, kdim = int(g.shape[0]), int(g.shape[1])
    _require_gate_clamp_admissible(tokens, kdim, beta)

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
        float(beta),
        float(lower),
    )


def kda_gate_clamp_torch_oracle(
    g: Tensor,
    a_log: Tensor,
    bias: Tensor | None = None,
    beta: float = GATE_SOFTPLUS_BETA,
    threshold: float = GATE_SOFTPLUS_THRESHOLD,
    lower: float = KDA_GATE_LOWER_BOUND,
) -> Tensor:
    """UPSTREAM'S form, in torch, as the reference. NOT a fallback for the kernel.

    This is ``kda_gate_fwd_kernel``'s arithmetic transcribed line for line
    (``kda.py:1558-1598``), including the ``threshold`` branch, composed with the
    port's lower bound. It exists because upstream's own implementation is Triton
    and is NOT callable on this host -- ``fused_kda_gate`` imports, then raises
    ``RuntimeError: 0 active drivers ([])`` -- which is the same negative
    ``invest-035`` §7.1 recorded for the CPU delta rule. The reference therefore
    has to be built from the quoted source, so the source is quoted exactly.

    P13 note: this function is the acceptance's REFERENCE, and no test item lets
    the seam reach it. It is not a torch route for kernel-class work.
    """
    b_a = -torch.exp(a_log.reshape(()).to(torch.float32))
    pre = g.to(torch.float32)
    if bias is not None:
        pre = pre + bias.reshape(1, -1).to(torch.float32)
    scaled = pre * beta
    softplus = torch.where(
        scaled > threshold,
        pre,
        (1.0 / beta) * torch.log(1.0 + torch.exp(scaled)),
    )
    return torch.clamp(b_a * softplus, min=lower)


def gate_clamp_kernel_identity() -> tuple[str, str]:
    """``(module, kernel name)`` of the jit entry this module authors."""
    return (kda_gate_clamp_kernel.__module__, kda_gate_clamp_kernel.__name__)
