# SPDX-License-Identifier: Apache-2.0
"""mHC combine (the hyper-connection "post" block): a SCRATCH NKI kernel.

`inc-glm53f-029`, WP8's second half. `inc-glm53f-028` normalises the mixing
scores; this module spends them. Given the ``hc_mult`` residual streams and the
sub-block's single-stream output, it mixes the streams back into ``hc_mult``
streams -- one NKI dispatch, on device, per layer call.

It is **kernel-class** under P13, and it is **SCRATCH**: the plan's substrate
bullet gives the same rationale as `-028`'s -- a per-token device mixing
operation with no substrate member. Contract §5.2 declares WP8 new in-tree NKI
authorship with **ZERO precedent**, and the ``vendored_kernels/`` precedent
covers version-lag vendoring only. **No "vendoring precedent" claim is available
for this increment and none is made.** The torch code below is the CPU oracle --
one of the two roles the plan's substrate register admits -- and never the
shipped path.

What the kernel computes, and where the formula comes from
----------------------------------------------------------
The operation is the pinned base's own ``mhc_post``. Its one-line statement, at
``vllm/model_executor/layers/mhc.py`` in ``vllm==0.24.0``::

    out_j = post_layer_mix_j * x + sum_i comb_res_mix_ij * residual_i

with tensors, at ``hc_mult = 4``:

======================  =====================  ==============================
name                    shape                  what it is
======================  =====================  ==============================
``x``                   ``[T, H]``             the sub-block's single-stream
                                               output
``residual``            ``[T, S, H]``          the ``S = hc_mult`` residual
                                               streams
``post_layer_mix``      ``[T, S, 1]``          per token, per output stream
``comb_res_mix``        ``[T, S, S]``          per token, ``[i, j]`` = input
                                               stream ``i`` -> output ``j``
``out``                 ``[T, S, H]``          the re-mixed streams
======================  =====================  ==============================

**``i`` is the summed INPUT stream and ``j`` is the OUTPUT stream, and that is
the one thing in this file worth getting right.** The base states it in two
independent spellings that agree bit-for-bit: ``torch.einsum("...ij,...ih->...jh",
comb, residual)`` in the plain-torch backend, and ``torch.bmm(comb.mT, residual)``
in the TileLang reference. The ``.mT`` in the second is the whole content of the
convention: a kernel that read ``comb[j, i]`` would be transposed, and both
spellings are carried in the acceptance so the reading is corroborated rather
than remembered.

Why there is no matmul in this kernel
-------------------------------------
The obvious shape for "mix ``S`` streams by an ``S x S`` matrix" is a matmul, and
it is the wrong one here, because **the mixing matrix is PER TOKEN.** With tokens
on the partition axis the Tensor Engine has no shared stationary operand to hold:
expressing this as one matmul would need a block-diagonal ``[T*S, T*S]`` matrix
built out of ``T`` different ``4 x 4`` blocks. What the operation actually is, in
that layout, is ``S * S`` per-token scalar broadcasts along the free axis -- which
is precisely ``nisa.tensor_scalar`` with an ``[T, 1]`` ``operand0``, the member
``functional/moe/router.py:1159-1173`` uses and `-028` reuses at
``sinkhorn.py:276-281``. So the kernel is 16 scalar-engine multiplies, 16 adds
and 4 post terms, all on ``[T, H]`` tiles, and every primitive is attested at a
landed line.

Precision, and why fp32 rather than the base's bf16
---------------------------------------------------
The base's ``mhc_post`` takes bf16 ``x``/``residual`` and returns
``residual.dtype``; its own kernel test therefore compares at ``atol=5e-2``. This
module is **fp32 in and fp32 out**, deliberately, because the plan declares this
increment's acceptance at ``atol=1e-5`` -- three orders tighter -- and bf16's ~3
decimal digits cannot express that difference at all. This is `-028`'s landed
choice for the same reason (``sinkhorn.py`` "Precision, stated rather than
implied"). Casting the result is the caller's business; throwing precision away
inside a kernel whose acceptance measures precision is not. The base's looser
tolerance is an artefact of its output dtype and is recorded here, not adopted.

The extents this kernel serves, measured rather than assumed
------------------------------------------------------------
* ``T <= PARTITION_MAX``. Tokens occupy the partition axis in a single tile.
  Measured at the boundary: ``T = 128`` runs, ``T = 129`` traps inside NKI with
  ``dma_copy dst partition dimension 129 exceeds maximum 128``. This is the SAME
  bound `-028` carries on its own token axis, so `inc-glm53f-030`'s tiling
  question is one question for both WP8 kernels rather than two.
* ``H`` needs **no** tiling at the target's real hidden sizes. Measured:
  ``H = 4096`` and ``H = 7168`` -- the base's own test shapes -- both run in one
  tile, at ``T = 128``, in 0.22 s and 0.27 s of simulator time. Recorded because
  it is the question a reader coming from `-028`'s ``M > 128`` refusal will ask
  next, and the answer here is the reassuring one.

Both readings are in ``probe-029-shape-ceiling.out`` beside this increment's
evidence record; they are cited rather than restated, and the constants below are
the single place the bound is written.

Route
-----
Acceptance is Tier N: the NKI simulator, through this module's own
:func:`hyper_connection_combine` seam (``wrap_nki -> NKIHOPCaller -> HOP ->
DispatchKey.CPU -> nki.simulator.simulate_kernel``). The seam counts its
dispatches, and the counters are **module-level** state with module-level reset
and read functions **on purpose**: `inc-glm53f-030`'s route predicate is form R-2
over *this* seam together with `-028`'s, read per layer call, so a later
increment's own test must be able to zero and read these counters from another
module. The names match `-028`'s exactly -- :func:`reset_dispatch_counters` and
:func:`dispatch_counters` -- so the two seams `-030` reads present one shape.

Under F1 a numeric comparison alone cannot prove a kernel ran: a torch fallback
would put torch on both sides and pass green. The counters below are therefore
acceptance criteria, not diagnostics.
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

from vllm_neuron.functional.mhc.sinkhorn import MHC_STREAMS, PARTITION_MAX
from vllm_neuron.utils.neuron_utils import can_run_kernel

logger = logging.getLogger(__name__)

# `MHC_STREAMS` (the target's `hc_mult 4`) and `PARTITION_MAX` are IMPORTED from
# `-028`'s module rather than restated here, and that is what `-028` asked for:
# `sinkhorn.py:136-139` records `MHC_STREAMS` as "a named constant because
# `inc-glm53f-029`'s combine kernel and `inc-glm53f-030`'s layer wiring are sized
# by the same number". `PARTITION_MAX` is the same physical bound on the same
# axis -- tokens on the partition axis -- so a second copy would be a second
# thing that can drift. Neither is redefined below.

__all__ = [
    "MHC_STREAMS",
    "PARTITION_MAX",
    "HyperConnectionError",
    "can_run_hyper_connection",
    "dispatch_counters",
    "hyper_connection_combine",
    "hyper_connection_kernel",
    "hyper_connection_torch_oracle",
    "kernel_identity",
    "reset_dispatch_counters",
]


class HyperConnectionError(ValueError):
    """A geometry or rank this module refuses, named rather than coerced.

    Raised in preference to letting NKI or numpy trap, because those traps do not
    name the offending extent in the caller's vocabulary. Measured, on the bare
    kernel with no gate in front of it: a ``[T, 3, 3]`` mix against 4 streams
    gives ``Out-of-bound access for tensor `unnamed` on dimension 1``, a mismatched
    token count gives ``operands could not be broadcast together with shapes
    (7,32) (8,1)``, and a mismatched hidden extent gives a remapped-shape
    ValueError. None of those tells a caller which argument was wrong.

    Refusing is also what P13 requires here: a geometry this kernel cannot serve
    must NOT quietly route to the torch oracle, because that would ship a torch
    path for kernel-class work (D6).
    """


# --------------------------------------------------------------------------- #
# The NKI kernel. SCRATCH: nkilib provides no mHC member at any shape.          #
# --------------------------------------------------------------------------- #
@nki.jit
def hyper_connection_kernel(x, residual, post_layer_mix, comb_res_mix):
    """The mHC combine, in NKI. One dispatch per call.

    Args:
        x: ``[T, H]`` fp32 in HBM -- the sub-block's single-stream output. ``T``
            occupies the partition axis, so ``T <= PARTITION_MAX``.
        residual: ``[T, S, H]`` fp32 in HBM -- the ``S`` residual streams.
        post_layer_mix: ``[T, S, 1]`` fp32 -- per token, per OUTPUT stream.
        comb_res_mix: ``[T, S, S]`` fp32 -- per token, ``[i, j]`` weights input
            stream ``i`` into output stream ``j``.

    Returns:
        ``[T, S, H]`` fp32, ``out_j = post_layer_mix_j * x + sum_i comb_ij * res_i``.

    Both python loops are **trace-time** ``range`` over ``S``, so the whole
    ``S x S`` mix unrolls INSIDE this one dispatch. That is a counted property,
    not a stylistic one: the route predicate declares ``1`` dispatch per case, and
    a host-driven loop over output streams would read ``S``.
    """
    t_extent, s_extent, h_extent = residual.shape

    out = nl.ndarray(
        (t_extent, s_extent, h_extent), dtype=nl.float32, buffer=nl.shared_hbm
    )

    # The single-stream layer output, loaded once: every output stream reads it.
    x_tile = nl.load(x, dtype=nl.float32)

    # All S streams loaded once each, rather than S times each inside the j loop.
    # S * S loads of the same data would be S * (S - 1) redundant DMAs.
    streams = [
        nl.load(residual[0:t_extent, i, 0:h_extent], dtype=nl.float32)
        for i in range(s_extent)
    ]

    # Two scratch tiles, allocated ONCE and reused across all S output streams.
    acc = nl.ndarray((t_extent, h_extent), dtype=nl.float32, buffer=nl.sbuf)
    term = nl.ndarray((t_extent, h_extent), dtype=nl.float32, buffer=nl.sbuf)

    for j in range(s_extent):
        # The post term INITIALISES the accumulator, so no separate memset pass:
        # `tensor_scalar` writes `dst` rather than adding into it. One fewer op,
        # and it also means the identity case's `post_layer_mix = 0` produces an
        # exact zero start rather than a zeroed-then-added-to tile.
        post_j = nl.load(post_layer_mix[0:t_extent, j, 0:1], dtype=nl.float32)
        nisa.tensor_scalar(dst=acc, data=x_tile, op0=nl.multiply, operand0=post_j)

        for i in range(s_extent):
            # `comb_res_mix[t, i, j]` -- i is the INPUT stream being summed, j the
            # OUTPUT stream being written. The [T, 1] slice is a per-token scalar
            # that `tensor_scalar` broadcasts along the free (hidden) axis.
            w_ij = nl.load(
                comb_res_mix[0:t_extent, i, j : j + 1], dtype=nl.float32
            )
            nisa.tensor_scalar(
                dst=term, data=streams[i], op0=nl.multiply, operand0=w_ij
            )
            nisa.tensor_tensor(dst=acc, data1=acc, data2=term, op=nl.add)

        nl.store(out[0:t_extent, j, 0:h_extent], value=acc)

    return out


# --------------------------------------------------------------------------- #
# Geometry admission.                                                          #
# --------------------------------------------------------------------------- #
def _require_admissible(
    x: Tensor, residual: Tensor, post_layer_mix: Tensor, comb_res_mix: Tensor
) -> tuple[int, int, int]:
    """Every rank and extent condition the kernel imposes, checked in one place.

    Each condition names what in the kernel needs it, so a reader can check the
    refusal against the code rather than against prose.

    Returns:
        ``(T, S, H)`` once every condition holds.
    """
    problems: list[str] = []

    if residual.dim() != 3:
        raise HyperConnectionError(
            f"residual must be 3-D [T, S, H], got shape {tuple(residual.shape)}; "
            f"T maps onto the partition axis, S is the stream axis and H the free "
            f"axis"
        )
    rows, streams, hidden = (int(v) for v in residual.shape)

    if rows <= 0:
        problems.append(f"T={rows} must be positive")
    elif rows > PARTITION_MAX:
        problems.append(
            f"T={rows} exceeds PARTITION_MAX={PARTITION_MAX}; the token extent "
            f"occupies the partition axis in a SINGLE tile and this kernel does "
            f"not tile T. Multi-tile T is a change to the kernel's shape rather "
            f"than its parameters, so it is outside `inc-glm53f-029`'s declared "
            f"scope and routes to the lead, never to a silent pad and never to a "
            f"torch path. `inc-glm53f-028` carries the same bound on the same "
            f"axis, so this is one tiling question for WP8, not two"
        )
    if streams <= 0:
        problems.append(f"S={streams} must be positive")
    if hidden <= 0:
        problems.append(f"H={hidden} must be positive")

    if x.dim() != 2:
        problems.append(
            f"x must be 2-D [T, H], got shape {tuple(x.shape)}"
        )
    elif tuple(int(v) for v in x.shape) != (rows, hidden):
        problems.append(
            f"x has shape {tuple(x.shape)}, expected [T, H] = [{rows}, {hidden}] "
            f"to match residual"
        )

    if post_layer_mix.dim() != 3 or tuple(
        int(v) for v in post_layer_mix.shape
    ) != (rows, streams, 1):
        problems.append(
            f"post_layer_mix has shape {tuple(post_layer_mix.shape)}, expected "
            f"[T, S, 1] = [{rows}, {streams}, 1] -- one scalar per token per "
            f"OUTPUT stream"
        )

    if comb_res_mix.dim() != 3 or tuple(
        int(v) for v in comb_res_mix.shape
    ) != (rows, streams, streams):
        problems.append(
            f"comb_res_mix has shape {tuple(comb_res_mix.shape)}, expected "
            f"[T, S, S] = [{rows}, {streams}, {streams}] -- per token, [i, j] "
            f"weights input stream i into output stream j"
        )

    if problems:
        raise HyperConnectionError(
            "mHC combine refuses this geometry: " + "; ".join(problems)
        )
    return rows, streams, hidden


# --------------------------------------------------------------------------- #
# The route seam and its counters.                                             #
# --------------------------------------------------------------------------- #
@dataclass
class _DispatchCounters:
    """What route actually ran, counted rather than inferred.

    ``nki_dispatch`` counts entries into the ``wrap_nki`` seam; ``torch_fallback``
    counts entries into the torch path. Two counters rather than one flag, so
    "the kernel ran" and "the fallback did not run" are independent readings and
    a test can require both.
    """

    nki_dispatch: int = 0
    torch_fallback: int = 0


#: MODULE-LEVEL, and that is a contract rather than an implementation detail:
#: `inc-glm53f-030` counts this seam's dispatches from its OWN test module (form
#: R-2, together with `-028`'s seam), per layer call, so the counter must be
#: resettable and readable from outside this module and outside this increment's
#: test. `-026` and `-028` placed their counters this way for the same reason,
#: and this object is DISTINCT from `-028`'s: `-030` reads two numbers, so the
#: two seams must not share one counter.
_COUNTERS = _DispatchCounters()


def reset_dispatch_counters() -> None:
    """Zero both counters. Called at the start of each declared test case."""
    _COUNTERS.nki_dispatch = 0
    _COUNTERS.torch_fallback = 0


def dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last reset."""
    return _COUNTERS.nki_dispatch, _COUNTERS.torch_fallback


def can_run_hyper_connection(
    x: Tensor, residual: Tensor, post_layer_mix: Tensor, comb_res_mix: Tensor
) -> bool:
    """Is the NKI route available *and* admissible for these shapes?

    Two independent conditions, deliberately not merged: ``can_run_kernel``
    answers "is there a device or a simulator", :func:`_require_admissible`
    answers "does this kernel accept these extents". A geometry the kernel cannot
    serve raises rather than falling back, because falling back would ship a
    torch path for kernel-class work (P13, D6).

    Raises:
        HyperConnectionError: if any rank or extent is inadmissible.
    """
    _require_admissible(x, residual, post_layer_mix, comb_res_mix)
    return can_run_kernel(residual)


def hyper_connection_combine(
    x: Tensor,
    residual: Tensor,
    post_layer_mix: Tensor,
    comb_res_mix: Tensor,
) -> Tensor:
    """The mHC combine. The seam the route predicate counts.

    Argument names and order are the pinned base's ``mhc_post`` exactly, so
    `inc-glm53f-030` wires a call rather than a translation.

    Args:
        x: ``[T, H]`` -- the sub-block's single-stream output.
        residual: ``[T, S, H]`` -- the ``S = hc_mult`` residual streams.
        post_layer_mix: ``[T, S, 1]`` -- per token, per output stream.
        comb_res_mix: ``[T, S, S]`` -- ``[i, j]``: input stream ``i`` into output
            stream ``j``.

    Returns:
        ``[T, S, H]`` fp32.

    Raises:
        HyperConnectionError: on an inadmissible rank or extent.
    """
    if not can_run_hyper_connection(x, residual, post_layer_mix, comb_res_mix):
        _COUNTERS.torch_fallback += 1
        logger.debug(
            "hyper_connection_combine: NKI route unavailable, using the torch "
            "path (oracle only, not the shipped path)"
        )
        return hyper_connection_torch_oracle(
            x, residual, post_layer_mix, comb_res_mix
        )

    _COUNTERS.nki_dispatch += 1
    return wrap_nki(hyper_connection_kernel)(
        x=x,
        residual=residual,
        post_layer_mix=post_layer_mix,
        comb_res_mix=comb_res_mix,
    )


def hyper_connection_torch_oracle(
    x: Tensor,
    residual: Tensor,
    post_layer_mix: Tensor,
    comb_res_mix: Tensor,
) -> Tensor:
    """The same operation in torch, in fp32. The CPU oracle -- never shipped.

    The formula is the pinned base's, and this is deliberately the ``einsum``
    spelling (``vllm/model_executor/kernels/mhc/torch.py``'s ``mhc_post_torch``)
    rather than the ``bmm(comb.mT, residual)`` spelling the TileLang reference
    uses. The acceptance carries the OTHER spelling and cross-checks the two, so
    the ``i``/``j`` convention is corroborated by two independent statements of it
    instead of by this file agreeing with itself.

    It is independent of the kernel in the way that matters: ``einsum`` contracts
    the stream axis in one call, where the kernel accumulates ``S`` per-token
    scalar broadcasts in a fixed order, so the two round differently.

    Kept in **fp32** rather than cast to ``residual.dtype`` as the base does --
    the base's own combine test compares at ``atol=5e-2`` precisely because it
    returns bf16, and this increment's declared ``atol=1e-5`` needs the precision
    kept.

    This is **never** the shipped kernel-class path (P13, D6). It exists to be
    compared against, and as the return for a route the seam refuses.

    Returns:
        ``[T, S, H]`` fp32.
    """
    mixed_residual = torch.einsum(
        "...ij,...ih->...jh",
        comb_res_mix.to(torch.float32),
        residual.to(torch.float32),
    )
    post_term = post_layer_mix.to(torch.float32) * x.unsqueeze(-2).to(torch.float32)
    return mixed_residual + post_term


def kernel_identity() -> tuple[str, str]:
    """``(module, qualname)`` of the NKI kernel, read off the object.

    Exposed so a test can assert the seam dispatches to the kernel this module
    authors -- which is how SCRATCH is checkable rather than merely claimed -- and
    so a substitution shows up as a changed reading rather than as silence.
    """
    func = getattr(hyper_connection_kernel, "func", None)
    target = func if func is not None else hyper_connection_kernel
    return target.__module__, target.__qualname__
