# SPDX-License-Identifier: Apache-2.0
"""KDA decode state carry: advance the recurrent state by one token, in NKI.

Prefill groups tokens into chunks and is served by
:mod:`vllm_neuron.functional.kda.chunked_recurrence`. Decode has one token and no
chunk to group it with, so it gets its own entry point: take the state prefill
left, advance it by exactly one token, and return the advanced state together
with that token's output. The state keeps prefill's ``[V, K]`` orientation on the
way in and out, so a decode loop needs no reshape between steps.

For one token, with the state written ``[V, K]``::

    state = state * exp(gk)                  # per key channel, not one scalar
    delta = (v - state @ kn) * beta          # reads the decayed state
    state = state + outer(delta, kn)
    o     = state @ qn                       # reads the updated state

``kn`` and ``qn`` are L2-normalised with :data:`~.chunked_recurrence.L2_NORM_EPS`
**inside** the square root, and ``qn`` carries ``K ** -0.5``. The order is
load-bearing: the delta reads the decayed state and the output reads the state
after the update. Getting the last one wrong yields a correct state and a wrong
output.

Inside the kernel the state is held transposed, as ``[K, V]``, for the reason
prefill records for its own carry: the per-key-channel decay then becomes a
``[K, 1]`` operand broadcast along the free axis, which is what
``nisa.tensor_scalar`` serves, while a ``[V, K]`` layout would need a
partition-axis broadcast, which it does not do. Two transposes are paid per step,
both exact.

There is no loop in the kernel. One dispatch advances one token; the caller's
loop advances the sequence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.kda.chunked_recurrence import (
    L2_NORM_EPS,
    MAX_TILE,
    _emit_l2_normalise,
    _emit_transpose,
    _psum,
    _sbuf,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel, values_are_readable

logger = logging.getLogger(__name__)


#: Largest ``|gk|`` this kernel accepts for one token. ``exp`` of a gate near 88
#: overflows fp32. A separate constant from the chunked module's
#: ``GATE_CUMSUM_ABS_LIMIT`` even though the number matches, because the two bound
#: different quantities: there a chunk-local cumulative sum over tokens, here a
#: single token's gate, since a decode step accumulates nothing before ``exp``.
DECODE_GATE_ABS_LIMIT = 60.0


class DecodeStateError(ValueError):
    """Raised for a geometry or gate range this kernel does not serve."""


class DecodeStepOutputs(NamedTuple):
    """One decode step's output and its advanced ``[V, K]`` state."""

    o: Tensor
    state: Tensor


@dataclass
class _DecodeDispatchCounters:
    """Which path actually ran, counted rather than inferred.

    Separate from the chunked module's counters so that a prefill dispatch cannot
    pollute a decode reading. The count is per dispatch, and one decode step is
    one dispatch, so over ``k`` steps the counter reads exactly ``k``.
    """

    nki_dispatch: int = 0
    torch_fallback: int = 0


#: Module level so a caller outside this module can reset and read it.
_DECODE_COUNTERS = _DecodeDispatchCounters()


def reset_decode_dispatch_counters() -> None:
    """Zero both counters."""
    _DECODE_COUNTERS.nki_dispatch = 0
    _DECODE_COUNTERS.torch_fallback = 0


def decode_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last reset."""
    return _DECODE_COUNTERS.nki_dispatch, _DECODE_COUNTERS.torch_fallback


@torch._dynamo.assume_constant_result
def _count_torch_fallback() -> None:
    """Count one torch-path entry, off the traced graph."""
    _DECODE_COUNTERS.torch_fallback += 1


@torch._dynamo.assume_constant_result
def _count_nki_dispatch() -> None:
    """Count one kernel dispatch, off the traced graph: a store Dynamo reads becomes a guard."""
    _DECODE_COUNTERS.nki_dispatch += 1


@nki.jit
def kda_decode_step_kernel(state_hbm, q_hbm, k_hbm, v_hbm, beta_hbm, gk_hbm):
    """Advance the KDA recurrent state by exactly one token.

    ``state`` is ``[V, K]``; ``q``, ``k`` and ``gk`` are ``[1, K]``; ``v`` is
    ``[1, V]``; ``beta`` is ``[1, 1]``. Returns ``o`` as ``[1, V]`` and the
    advanced state as ``[V, K]``.

    Two operand shapes below are unusual on this image and both are exact: an
    ``nc_transpose`` whose source has partition extent 1, and an ``nc_matmul``
    whose two operands both have partition extent 1.
    """
    vdim, kdim = state_hbm.shape
    scale = float(kdim) ** -0.5

    o_hbm = nl.ndarray((1, vdim), dtype=nl.float32, buffer=nl.shared_hbm)
    state_out_hbm = nl.ndarray((vdim, kdim), dtype=nl.float32, buffer=nl.shared_hbm)

    # ---- the state, transposed once into the layout the decay wants ---------- #
    st_sb = _sbuf(vdim, kdim)
    nisa.tensor_copy(dst=st_sb, src=nl.load(state_hbm, dtype=nl.float32))
    s_sb = _sbuf(kdim, vdim)
    _emit_transpose(s_sb, st_sb, vdim, kdim)

    # ---- normalise k and q in the ROW layout, where the reduction is free-axis  #
    kn_sb = _sbuf(1, kdim)
    _emit_l2_normalise(kn_sb, nl.load(k_hbm, dtype=nl.float32), 1, kdim)
    qn_sb = _sbuf(1, kdim)
    _emit_l2_normalise(qn_sb, nl.load(q_hbm, dtype=nl.float32), 1, kdim)
    nisa.tensor_scalar(dst=qn_sb, data=qn_sb, op0=nl.multiply, operand0=scale)

    # ---- decay, per KEY channel: S[k, v] *= exp(gk[k]) ---------------------- #
    egk_row = _sbuf(1, kdim)
    nisa.activation(dst=egk_row, data=nl.load(gk_hbm, dtype=nl.float32), op=nl.exp)
    egk_col = _sbuf(kdim, 1)
    _emit_transpose(egk_col, egk_row, 1, kdim)
    nisa.tensor_scalar(dst=s_sb, data=s_sb, op0=nl.multiply, operand0=egk_col)

    # ---- delta, as a [1, V] row.  (state @ kn)^T == kn^T @ S ---------------- #
    kn_col = _sbuf(kdim, 1)
    _emit_transpose(kn_col, kn_sb, 1, kdim)
    ps_sk = _psum(1, vdim)
    nisa.nc_matmul(dst=ps_sk, stationary=kn_col, moving=s_sb, accumulate=False)
    sk_sb = _sbuf(1, vdim)
    nisa.tensor_copy(dst=sk_sb, src=ps_sk)
    d_sb = _sbuf(1, vdim)
    nisa.tensor_tensor(
        dst=d_sb, data1=nl.load(v_hbm, dtype=nl.float32), data2=sk_sb, op=nl.subtract
    )
    beta_sb = _sbuf(1, 1)
    nisa.tensor_copy(dst=beta_sb, src=nl.load(beta_hbm, dtype=nl.float32))
    nisa.tensor_scalar(dst=d_sb, data=d_sb, op0=nl.multiply, operand0=beta_sb)

    # ---- the rank-1 update: S[k, v] += kn[k] * delta[v] -------------------- #
    ps_up = _psum(kdim, vdim)
    nisa.nc_matmul(dst=ps_up, stationary=kn_sb, moving=d_sb, accumulate=False)
    up_sb = _sbuf(kdim, vdim)
    nisa.tensor_copy(dst=up_sb, src=ps_up)
    # A fresh destination, not an in-place add: no ``tensor_tensor`` in this
    # package writes back onto one of its own operands.
    new_sb = _sbuf(kdim, vdim)
    nisa.tensor_tensor(dst=new_sb, data1=s_sb, data2=up_sb, op=nl.add)

    # ---- the output, read from the UPDATED state.  (S^T @ qn)^T == qn^T @ S -- #
    qn_col = _sbuf(kdim, 1)
    _emit_transpose(qn_col, qn_sb, 1, kdim)
    ps_o = _psum(1, vdim)
    nisa.nc_matmul(dst=ps_o, stationary=qn_col, moving=new_sb, accumulate=False)
    o_sb = _sbuf(1, vdim)
    nisa.tensor_copy(dst=o_sb, src=ps_o)
    nl.store(o_hbm, value=o_sb)

    # ---- back to the [V, K] contract orientation --------------------------- #
    out_sb = _sbuf(vdim, kdim)
    _emit_transpose(out_sb, new_sb, kdim, vdim)
    nl.store(state_out_hbm, value=out_sb)

    return o_hbm, state_out_hbm


def _require_decode_admissible(kdim: int, vdim: int, gate_abs_max: float) -> None:
    """Raise unless the decode kernel serves this input.

    ``gate_abs_max`` is one token's largest absolute gate, not a cumulative sum
    over tokens, so it is the quantity ``exp`` sees. See
    :data:`DECODE_GATE_ABS_LIMIT`.
    """
    problems: list[str] = []
    if kdim < 1 or kdim > MAX_TILE:
        problems.append(
            f"kdim={kdim} must be in [1, {MAX_TILE}]; it is the transposed state's "
            f"partition extent and the kernel declares no tiling"
        )
    if vdim < 1 or vdim > MAX_TILE:
        problems.append(
            f"vdim={vdim} must be in [1, {MAX_TILE}]; it is the transposed state's "
            f"free extent"
        )
    if gate_abs_max > DECODE_GATE_ABS_LIMIT:
        problems.append(
            f"max|per-token gate|={gate_abs_max:.3f} exceeds "
            f"{DECODE_GATE_ABS_LIMIT}; the state decay is exp of that quantity, so "
            f"a single-token gate this far from zero would overflow fp32"
        )
    if problems:
        raise DecodeStateError(
            "kda_decode_step cannot serve this input: " + "; ".join(problems)
        )


def can_run_decode_step(
    reference: Tensor, kdim: int, vdim: int, gate_abs_max: float
) -> bool:
    """Is the NKI path available *and* admissible for this decode step?

    Two independent conditions: ``can_run_kernel`` answers whether a device or
    simulator exists, :func:`_require_decode_admissible` whether this kernel
    accepts these extents and this gate range.
    """
    _require_decode_admissible(kdim, vdim, gate_abs_max)
    return can_run_kernel(reference)


def kda_decode_step(
    state: Tensor, q: Tensor, k: Tensor, v: Tensor, beta: Tensor, gk: Tensor
) -> DecodeStepOutputs:
    """Advance the KDA state by one token: one call, one kernel dispatch.

    Args:
        state: ``[V, K]`` fp32, the incoming state, in the orientation a
            prefill's ``final_state`` is stored in.
        q: ``[1, K]`` fp32, raw. Normalised and scaled inside the kernel.
        k: ``[1, K]`` fp32, raw. Normalised inside the kernel.
        v: ``[1, V]`` fp32.
        beta: ``[1, 1]`` fp32, the delta-rule step size.
        gk: ``[1, K]`` fp32, this token's per-key-channel log gate, not
            accumulated.

    Returns:
        :class:`DecodeStepOutputs`, whose ``state`` is ``[V, K]``.

    Raises:
        DecodeStateError: on a rank mismatch, a shape disagreement, or an
            inadmissible geometry or gate range.

    Shapes are checked strictly rather than broadcast into place: a decode loop
    calls this once per generated token, so a silently reshaped argument would be
    a defect repeated every token.
    """
    if state.dim() != 2:
        raise DecodeStateError(
            f"state must be 2-D [V, K], got shape {tuple(state.shape)}"
        )
    vdim, kdim = (int(x) for x in state.shape)
    expected = (
        ("q", q, (1, kdim)),
        ("k", k, (1, kdim)),
        ("gk", gk, (1, kdim)),
        ("v", v, (1, vdim)),
        ("beta", beta, (1, 1)),
    )
    for name, tensor, shape in expected:
        if tuple(tensor.shape) != shape:
            raise DecodeStateError(
                f"{name} {tuple(tensor.shape)} must be {shape} for a state shaped "
                f"{(vdim, kdim)}"
            )

    # An eager call reads the gate range; a traced or meta-built one has no values
    # to read and passes 0.0, which is inside the limit and refuses nothing. Only
    # `can_run_kernel` decides the path, so a graph build loses the diagnostic
    # message and nothing else.
    gate_abs_max = (
        float(gk.float().abs().max().item()) if values_are_readable(gk) else 0.0
    )
    if not can_run_decode_step(state, kdim, vdim, gate_abs_max):
        _count_torch_fallback()
        logger.debug(
            "kda_decode_step: NKI route unavailable, using the torch path "
            "(oracle only, never the shipped path)"
        )
        return kda_decode_step_torch_oracle(state, q, k, v, beta, gk)

    _count_nki_dispatch()
    o, state_out = wrap_nki(kda_decode_step_kernel)(
        state_hbm=state,
        q_hbm=q,
        k_hbm=k,
        v_hbm=v,
        beta_hbm=beta,
        gk_hbm=gk,
    )
    return DecodeStepOutputs(o=o, state=state_out)


def kda_decode_step_torch_oracle(
    state: Tensor, q: Tensor, k: Tensor, v: Tensor, beta: Tensor, gk: Tensor
) -> DecodeStepOutputs:
    """One decode step in torch: the fallback path, never the shipped one.

    Present so a host with no device and no simulator can exercise this module's
    contract.
    """
    st = state.float()
    q32, k32, v32 = q.float(), k.float(), v.float()
    beta32, gk32 = beta.float(), gk.float()
    kdim = st.shape[1]

    kn = k32 / torch.sqrt((k32 * k32).sum(-1, keepdim=True) + L2_NORM_EPS)
    qn = q32 / torch.sqrt((q32 * q32).sum(-1, keepdim=True) + L2_NORM_EPS)
    qn = qn * (float(kdim) ** -0.5)

    st = st * torch.exp(gk32)
    delta = (v32 - (st @ kn.squeeze(0)).unsqueeze(0)) * beta32
    st = st + delta.reshape(-1, 1) @ kn
    o = (st @ qn.squeeze(0)).unsqueeze(0)
    return DecodeStepOutputs(o=o, state=st.contiguous())


def decode_kernel_identity() -> tuple[str, str]:
    """``(module, qualname)`` of the decode kernel this module authors.

    Read off the unwrapped ``.func``, since ``nki.jit``'s wrapper reports its own
    module either way.
    """
    func = getattr(kda_decode_step_kernel, "func", None)
    target = func if func is not None else kda_decode_step_kernel
    return target.__module__, target.__qualname__
