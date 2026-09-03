# SPDX-License-Identifier: Apache-2.0
"""KDA decode state carry -- one token, on device, in NKI.

`inc-glm53f-036`. Prefill groups tokens into chunks and is served by
:mod:`vllm_neuron.functional.kda.chunked_recurrence`. Decode has one token and no
chunk to group it with, so it needs its own entry: take the state the prefill
left, advance it by exactly one token, and return both the advanced state and
that token's output.

WHY THIS IS A KERNEL AND NOT TORCH GLUE (P13). The recurrent state is a
``[V, K]`` tile that lives on device between decode steps. A torch-level step
would move it off device and back on every token, and a decode loop runs one step
per generated token, so the substrate has to advance it in place. The substrate
provides no KDA state step, so this module writes one.

THE STATE CONTRACT IS THE ONE `-035b` LANDED, NOT A NEW ONE. Its
``final_state`` is stored ``[V, K]``, so that is what this module accepts and what
it returns. Feeding a prefill's ``final_state`` straight into
:func:`kda_decode_step` needs no reshape and no transpose on the caller's side.

THE RULE, AND WHICH PARTS OF IT ARE DECLARED VALUES RATHER THAN DEFAULTS. For one
token, with the state written ``[V, K]``::

    state = state * exp(gk)                  # per KEY channel, not one scalar
    delta = (v - state @ kn) * beta          # reads the DECAYED state
    state = state + outer(delta, kn)
    o     = state @ qn                       # reads the UPDATED state

``kn`` and ``qn`` are L2-normalised with :data:`~.chunked_recurrence.L2_NORM_EPS`
**inside** the square root, and ``qn`` carries ``K ** -0.5``. The four steps are
in a load-bearing order: the state is decayed first, the delta reads the decayed
state, and the output reads the state after the update. Getting the last one wrong
produces a state that matches and an output that does not, which is why the
acceptance measures both.

INTERNALLY THE STATE IS HELD TRANSPOSED, as ``[K, V]``, for the reason `-035b`
records for its own carry: the per-key-channel decay then becomes a ``[K, 1]``
operand broadcast along the FREE axis, which is exactly what
``nisa.tensor_scalar`` does, and a ``[V, K]`` layout would need a partition-axis
broadcast, which it does not do. Two transposes are paid per step, one on the way
in and one on the way out, and both are exact -- measured at ``0.000e+00``.

THIS KERNEL CONTAINS NO LOOP AT ALL. Not ``nl.affine_range`` and not
``nl.sequential_range``. One dispatch advances exactly one token, and the caller's
loop is what advances the sequence. That is what makes the route predicate an
equality: ``k`` decode steps make ``k`` dispatches, so the counter reads the step
count and a seam that batched tokens internally would read fewer.

Tier N harness -- the NKI simulator on a host CPU, no device and no lease::

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
    python -m pytest test/vllm_neuron/functional/kda/test_decode_state.py \
        -q -s --timeout 60 -p no:cacheprovider
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
from vllm_neuron.utils.neuron_utils import can_run_kernel

logger = logging.getLogger(__name__)


#: Largest ``|gk|`` this kernel accepts for ONE token. Deliberately a separate
#: constant from the chunked module's ``GATE_CUMSUM_ABS_LIMIT`` even though the
#: number is the same, because the two bound DIFFERENT quantities and reusing the
#: name would misdescribe the check: there the quantity is a chunk-local
#: cumulative sum over tokens, here it is a single token's gate, since a decode
#: step accumulates nothing before calling ``exp``. Stated and checked rather than
#: assumed, because ``exp`` of a gate near ``88`` overflows fp32 whatever the
#: state does.
DECODE_GATE_ABS_LIMIT = 60.0


class DecodeStateError(ValueError):
    """Raised for a geometry or a gate range this kernel does not serve.

    Inadmissibility raises rather than falling back, because falling back would
    ship a torch path for kernel-class work (P13).
    """


class DecodeStepOutputs(NamedTuple):
    """What one decode step returns.

    ``state`` is the ADVANCED state in the same ``[V, K]`` orientation it arrived
    in, so the caller's next step takes it unchanged and a decode loop needs no
    reshaping between steps.
    """

    o: Tensor
    state: Tensor


@dataclass
class _DecodeDispatchCounters:
    """What route actually ran, counted rather than inferred.

    A THIRD counter object, separate from the chunked module's two. Separate
    because this increment's route predicate is an equality against a step count,
    so a reading polluted by a prefill dispatch would be wrong rather than merely
    noisy. Nothing here reads or writes the chunked module's counters.

    The count is per **dispatch**. One decode step is one dispatch, so over ``k``
    steps the counter reads ``k`` exactly -- not at most ``k``.
    """

    nki_dispatch: int = 0
    torch_fallback: int = 0


#: MODULE-LEVEL so a test outside this module can reset and read it, on the
#: `inc-glm53f-028` precedent that `-035a` and `-035b` both follow.
_DECODE_COUNTERS = _DecodeDispatchCounters()


def reset_decode_dispatch_counters() -> None:
    """Zero both counters. Called at the start of each declared case."""
    _DECODE_COUNTERS.nki_dispatch = 0
    _DECODE_COUNTERS.torch_fallback = 0


def decode_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last reset."""
    return _DECODE_COUNTERS.nki_dispatch, _DECODE_COUNTERS.torch_fallback


@nki.jit
def kda_decode_step_kernel(state_hbm, q_hbm, k_hbm, v_hbm, beta_hbm, gk_hbm):
    """Advance the KDA recurrent state by exactly ONE token.

    Shapes: ``state`` is ``[V, K]``; ``q``, ``k`` and ``gk`` are ``[1, K]``; ``v``
    is ``[1, V]``; ``beta`` is ``[1, 1]``. Returns ``o`` as ``[1, V]`` and the
    advanced state as ``[V, K]``.

    Every operand orientation below was measured against torch before this body
    was written, including the two that are unusual on this image: a
    ``nc_transpose`` whose source has partition extent ``1``, and an
    ``nc_matmul`` whose two operands both have partition extent ``1``. Both are
    exact.

    THE ONE-TOKEN SHAPE IS NOT AN INEFFICIENCY TO BE BATCHED AWAY. It is the
    contract: the route predicate reads the dispatch count as an equality against
    the caller's step count, so batching tokens inside this kernel would break the
    reading the predicate exists to take.
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
    # A FRESH DESTINATION, not an in-place add. The landed chunked kernels never
    # write a ``tensor_tensor`` back onto one of its own operands, and this body
    # keeps to that so the update has one reviewed idiom rather than two.
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

    ``gate_abs_max`` is ONE TOKEN's largest absolute gate, not a cumulative sum
    over tokens. A decode step accumulates nothing before calling ``exp``, so the
    quantity checked here is the quantity ``exp`` sees. See
    :data:`DECODE_GATE_ABS_LIMIT` for why that is a separate constant.
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
    """Is the NKI route available *and* admissible for this decode step?

    Two independent conditions, kept separate for the reason the chunked module's
    version states: ``can_run_kernel`` answers "is there a device or a simulator",
    :func:`_require_decode_admissible` answers "does this kernel accept these
    extents and this gate range".
    """
    _require_decode_admissible(kdim, vdim, gate_abs_max)
    return can_run_kernel(reference)


def kda_decode_step(
    state: Tensor, q: Tensor, k: Tensor, v: Tensor, beta: Tensor, gk: Tensor
) -> DecodeStepOutputs:
    """The seam THIS increment's route predicate counts. ONE token, one dispatch.

    Args:
        state: ``[V, K]`` fp32 -- the incoming state, in the orientation
            `-035b`'s ``final_state`` is stored in.
        q: ``[1, K]`` fp32, raw. Normalised and scaled inside the kernel.
        k: ``[1, K]`` fp32, raw. Normalised inside the kernel.
        v: ``[1, V]`` fp32.
        beta: ``[1, 1]`` fp32, the delta-rule step size.
        gk: ``[1, K]`` fp32, the per-key-channel log gate for this one token.
            NOT accumulated -- there is nothing to accumulate over.

    Returns:
        :class:`DecodeStepOutputs`, whose ``state`` is ``[V, K]``.

    Raises:
        DecodeStateError: on a rank mismatch, a shape disagreement, or an
            inadmissible geometry or gate range.

    The shapes are checked strictly rather than broadcast into place. A decode
    loop calls this once per generated token, so a silently reshaped argument
    would be a defect repeated every token rather than once.
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

    gate_abs_max = float(gk.float().abs().max().item())
    if not can_run_decode_step(state, kdim, vdim, gate_abs_max):
        _DECODE_COUNTERS.torch_fallback += 1
        logger.debug(
            "kda_decode_step: NKI route unavailable, using the torch path "
            "(oracle only, never the shipped path)"
        )
        return kda_decode_step_torch_oracle(state, q, k, v, beta, gk)

    _DECODE_COUNTERS.nki_dispatch += 1
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
    """One decode step in torch. THE FALLBACK PATH ONLY, never the shipped path.

    Present for the same reason the chunked module's oracles are: so a host with
    no device and no simulator can exercise the seam's contract. It is NOT the
    reference this increment's acceptance compares against -- that reference is
    :func:`~.chunked_recurrence.kda_sequential_torch_oracle`, which this module
    does not call and which walks a whole prefill rather than one token.
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

    Read by the acceptance driver to prove the kernel under test is authored here
    rather than imported from the substrate.
    """
    func = getattr(kda_decode_step_kernel, "func", None)
    target = func if func is not None else kda_decode_step_kernel
    return target.__module__, target.__qualname__
