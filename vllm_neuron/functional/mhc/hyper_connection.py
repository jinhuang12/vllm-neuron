# SPDX-License-Identifier: Apache-2.0
"""mHC combine, the hyper-connection "post" block, as an NKI kernel.

:mod:`vllm_neuron.functional.mhc.sinkhorn` normalises the mixing scores; this
module spends them. Given the ``hc_mult`` residual streams and the sub-block's
single-stream output, it mixes the streams back into ``hc_mult`` streams in one
dispatch per layer call. The torch code here is a CPU oracle, never the shipped
path.

The operation is upstream's ``mhc_post``::

    out_j = post_layer_mix_j * x + sum_i comb_res_mix_ij * residual_i

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

``i`` is the summed input stream and ``j`` is the output stream. Upstream states
this twice in spellings that agree bit-for-bit,
``torch.einsum("...ij,...ih->...jh", comb, residual)`` and
``torch.bmm(comb.mT, residual)``; the ``.mT`` in the second is the whole content
of the convention, and a kernel that read ``comb[j, i]`` would be transposed.

There is no matmul in this kernel, because the mixing matrix is per token. With
tokens on the partition axis the tensor engine has no shared stationary operand to
hold: one matmul would need a block-diagonal ``[T*S, T*S]`` matrix built from
``T`` different ``S x S`` blocks. In that layout the operation is ``S * S``
per-token scalar broadcasts along the free axis, which is ``nisa.tensor_scalar``
with an ``[T, 1]`` ``operand0``. So the kernel is ``S * S`` scalar-engine
multiplies and adds plus ``S`` post terms, all on ``[T, H]`` tiles.

This module is fp32 in and fp32 out where upstream's ``mhc_post`` takes bf16 and
returns ``residual.dtype``, which is why upstream's own test compares at
``atol=5e-2``. bf16's roughly three decimal digits cannot express a tighter
agreement than that. Casting the result is the caller's business.

``T`` is unbounded: tokens occupy the partition axis, which is capped at
``nl.tile_size.pmax``, so the kernel walks that axis in tiles of that size and
:data:`PARTITION_MAX` is the tile height rather than a token ceiling. ``H`` lands
on the free axis, which has no partition cap, and needs no tiling at the target's
real hidden sizes: ``H = 4096`` and ``H = 7168`` both run in one tile.
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

# `MHC_STREAMS` (the target's `hc_mult`) and `PARTITION_MAX` are imported from
# `sinkhorn.py` rather than restated: both name the same quantity for both halves
# of mHC, so a second copy would be a second thing that can drift.

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
    name the offending argument. Without this check a ``[T, 3, 3]`` mix against 4
    streams gives ``Out-of-bound access for tensor `unnamed` on dimension 1``, a
    mismatched token count gives ``operands could not be broadcast together with
    shapes (7,32) (8,1)``, and a mismatched hidden extent gives a remapped-shape
    ``ValueError``.
    """


@nki.jit
def hyper_connection_kernel(x, residual, post_layer_mix, comb_res_mix):
    """The mHC combine, in NKI. One dispatch per call.

    Args:
        x: ``[T, H]`` fp32 in HBM -- the sub-block's single-stream output. ``T``
            occupies the partition axis, so it is walked in tiles of
            :data:`PARTITION_MAX` rows and is not bounded by it.
        residual: ``[T, S, H]`` fp32 in HBM -- the ``S`` residual streams.
        post_layer_mix: ``[T, S, 1]`` fp32 -- per token, per OUTPUT stream.
        comb_res_mix: ``[T, S, S]`` fp32 -- per token, ``[i, j]`` weights input
            stream ``i`` into output stream ``j``.

    Returns:
        ``[T, S, H]`` fp32, ``out_j = post_layer_mix_j * x + sum_i comb_ij * res_i``.

    Every python loop here is a trace-time ``range``, so the whole ``S x S`` mix
    and the walk over row tiles unroll inside this one dispatch. A token extent
    above the partition limit therefore costs more tiles and never a second
    dispatch.

    Two constraints on the python in this body, both from the NKI compiler
    frontend rather than from the simulator, which executes this as ordinary
    python and never runs the frontend at all:

    * A comprehension inside a traced kernel does not specialise. The stream tiles
      below are collected with a ``for`` loop and ``append``; the identical list
      built by a list comprehension fails with ``failed to specialize NKI kernel:
      ... error: unsupported expression`` at the first call.
    * The short last tile comes from an ``if``/``else`` rather than from ``min``,
      and the tile height is the imported :data:`PARTITION_MAX` rather than an
      attribute read inside the traced body.
    """
    t_extent, s_extent, h_extent = residual.shape
    n_tiles = (t_extent + PARTITION_MAX - 1) // PARTITION_MAX

    # `out` is HBM, which has no partition cap, so it is allocated at the full
    # token extent; only the SBUF tiles below are per-tile.
    out = nl.ndarray(
        (t_extent, s_extent, h_extent), dtype=nl.float32, buffer=nl.shared_hbm
    )

    for t in range(n_tiles):
        off = t * PARTITION_MAX
        # The last tile is narrowed rather than padded: a padded tile would put
        # values the caller never sent into the arithmetic.
        remaining = t_extent - off
        if remaining < PARTITION_MAX:
            rows = remaining
        else:
            rows = PARTITION_MAX

        # The single-stream layer output, loaded once per tile: every output
        # stream of this tile reads it.
        x_tile = nl.load(x[off : off + rows, 0:h_extent], dtype=nl.float32)

        # All S streams loaded once each rather than once per output stream: S * S
        # loads of the same data would be S * (S - 1) redundant DMAs.
        streams = []
        for i in range(s_extent):
            stream = nl.load(
                residual[off : off + rows, i, 0:h_extent], dtype=nl.float32
            )
            streams.append(stream)

        # Two scratch tiles, allocated once per row tile and reused across that
        # tile's S output streams.
        acc = nl.ndarray((rows, h_extent), dtype=nl.float32, buffer=nl.sbuf)
        term = nl.ndarray((rows, h_extent), dtype=nl.float32, buffer=nl.sbuf)

        for j in range(s_extent):
            # The post term initialises the accumulator, so there is no separate
            # memset pass: `tensor_scalar` writes `dst` rather than adding into it.
            # It also makes `post_layer_mix = 0` an exact zero start.
            post_j = nl.load(
                post_layer_mix[off : off + rows, j, 0:1], dtype=nl.float32
            )
            nisa.tensor_scalar(dst=acc, data=x_tile, op0=nl.multiply, operand0=post_j)

            for i in range(s_extent):
                # In `comb_res_mix[t, i, j]`, i is the input stream being summed
                # and j the output stream being written. The [rows, 1] slice is a
                # per-token scalar that `tensor_scalar` broadcasts along the free
                # (hidden) axis.
                w_ij = nl.load(
                    comb_res_mix[off : off + rows, i, j : j + 1], dtype=nl.float32
                )
                nisa.tensor_scalar(
                    dst=term, data=streams[i], op0=nl.multiply, operand0=w_ij
                )
                nisa.tensor_tensor(dst=acc, data1=acc, data2=term, op=nl.add)

            nl.store(out[off : off + rows, j, 0:h_extent], value=acc)

    return out


def _require_admissible(
    x: Tensor, residual: Tensor, post_layer_mix: Tensor, comb_res_mix: Tensor
) -> tuple[int, int, int]:
    """Every rank and extent condition the kernel imposes, checked in one place.

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

    # There is no upper bound on T: the kernel walks the token axis in
    # `PARTITION_MAX` tiles, so that constant is the tile height and not a ceiling.
    if rows <= 0:
        problems.append(f"T={rows} must be positive")
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


@dataclass
class _DispatchCounters:
    """Which path actually ran, counted rather than inferred.

    ``nki_dispatch`` counts entries into the ``wrap_nki`` call, ``torch_fallback``
    entries into the torch path. Two counters rather than one flag, so "the kernel
    ran" and "the fallback did not" are independent readings.
    """

    nki_dispatch: int = 0
    torch_fallback: int = 0


#: Module level so a caller outside this module can reset and read it. A distinct
#: object from Sinkhorn's: a caller that wires both halves of mHC reads two
#: numbers, so the two must not share one counter.
_COUNTERS = _DispatchCounters()


def reset_dispatch_counters() -> None:
    """Zero both counters."""
    _COUNTERS.nki_dispatch = 0
    _COUNTERS.torch_fallback = 0


def dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last reset."""
    return _COUNTERS.nki_dispatch, _COUNTERS.torch_fallback


@torch._dynamo.assume_constant_result
def _count_torch_fallback() -> None:
    """Count one torch-path entry, off the traced graph."""
    _COUNTERS.torch_fallback += 1


@torch._dynamo.assume_constant_result
def _count_nki_dispatch() -> None:
    """Count one kernel dispatch, off the traced graph: a store Dynamo reads becomes a guard."""
    _COUNTERS.nki_dispatch += 1


def can_run_hyper_connection(
    x: Tensor, residual: Tensor, post_layer_mix: Tensor, comb_res_mix: Tensor
) -> bool:
    """Is the NKI path available *and* admissible for these shapes?

    Two independent conditions: ``can_run_kernel`` answers whether a device or
    simulator exists, :func:`_require_admissible` whether this kernel accepts these
    extents. A geometry the kernel cannot serve raises rather than falling back.

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
    """The mHC combine: one kernel dispatch per call.

    Argument names and order match upstream's ``mhc_post``, so a layer wires a
    call rather than a translation.

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
        _count_torch_fallback()
        logger.debug(
            "hyper_connection_combine: NKI route unavailable, using the torch "
            "path (oracle only, not the shipped path)"
        )
        return hyper_connection_torch_oracle(
            x, residual, post_layer_mix, comb_res_mix
        )

    _count_nki_dispatch()
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
    """The same operation in torch, in fp32. The CPU oracle, never shipped.

    Written in upstream's ``einsum`` spelling rather than its
    ``bmm(comb.mT, residual)`` one, so that the two independent statements of the
    ``i``/``j`` convention can be cross-checked instead of one file agreeing with
    itself.

    Independent of the kernel where it matters: ``einsum`` contracts the stream
    axis in one call, where the kernel accumulates ``S`` per-token scalar
    broadcasts in a fixed order, so the two round differently.

    Kept in fp32 rather than cast to ``residual.dtype`` as upstream does, since
    returning bf16 is why upstream's own test compares at ``atol=5e-2``.

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

    Lets a caller check that this module dispatches to the kernel it authors, so a
    substitution shows up as a changed reading.
    """
    func = getattr(hyper_connection_kernel, "func", None)
    target = func if func is not None else hyper_connection_kernel
    return target.__module__, target.__qualname__
