# SPDX-License-Identifier: Apache-2.0
"""Top-k selection for the DSA indexer.

The indexer scores every candidate context position for a query token and then keeps only
the highest-scoring ones; this module is that selection. It takes a score tensor whose last
axis is the candidate axis and returns the selected values and their indices, highest first.
The selection itself is done by ``rotational_topk``, the vendored ``@nki.jit`` kernel; this
module adds only a dispatch seam, a route gate and a torch reference.

Which geometries the kernel serves is decided by its own config factories and never by a
constant here. The kernel splits the candidate axis into stages and asserts a concatenated
SBUF free dimension, and that bound is not monotonic in ``k``, so no static width or ``k``
cap can express it; ``_config_builds`` dry-runs the factories instead.

A bfloat16 input is passed through rather than upcast, and that has a consequence a caller
must know. bfloat16 collapses a score distribution onto few distinct values, so many
candidates tie at the selection boundary, and the kernel and ``torch.topk`` break those ties
differently: the selected values stay bit-identical while the index sets need not agree. A
caller that needs a reproducible index set from bfloat16 scores must break the ties itself.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import Tensor

import nki.language as nl
from libtorch_neuronx_lite.nki.nki_dtype import torch_to_nki_dtype
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.vendored_kernels.rotational_topk import (
    create_rotational_topk_config,
    create_topk_config,
    rotational_topk,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

logger = logging.getLogger(__name__)

# LNC grid degree handed to the config factories. The feasibility dry-run
# (``_config_builds``) and the real config build (``_nki_config``) must use one value, or the
# gate would admit a config the run then fails to build. The kernel forces it to 1 internally
# when the row count is 1.
_NUM_PROGRAMS = 2

# The dtypes the kernel is tested for. Its pad sentinel has no float16 branch, so float16
# is refused here rather than handed to it.
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float32)


class DsaTopkSelectError(ValueError):
    """Raised for an input this module will not hand to the kernel or the torch path."""


@dataclass
class _TopkSelectDispatchCounters:
    """Per-process record of how the seam below was reached."""

    nki_dispatch: int = 0
    torch_fallback: int = 0
    last_kernel: tuple[str, str] | None = None


_COUNTERS = _TopkSelectDispatchCounters()


def reset_topk_select_dispatch_counters() -> None:
    """Zero this seam's counters."""
    _COUNTERS.nki_dispatch = 0
    _COUNTERS.torch_fallback = 0
    _COUNTERS.last_kernel = None


def topk_select_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last reset."""
    return (_COUNTERS.nki_dispatch, _COUNTERS.torch_fallback)


@torch._dynamo.assume_constant_result
def _count_nki_dispatch() -> None:
    """Count one kernel dispatch, off the traced graph: a store Dynamo traces becomes a guard."""
    _COUNTERS.nki_dispatch += 1


@torch._dynamo.assume_constant_result
def _count_torch_fallback() -> None:
    """Count one torch-path entry, off the traced graph."""
    _COUNTERS.torch_fallback += 1


def topk_select_kernel_identity() -> tuple[str, str] | None:
    """``(module, qualname)`` of the kernel the seam last dispatched, or ``None``."""
    return _COUNTERS.last_kernel


def _kernel_identity_of(kernel) -> tuple[str, str]:
    """``(module, qualname)`` of the function a ``@nki.jit`` object wraps.

    Reading those attributes off the decorated object would report the decorator instead:
    ``@nki.jit`` returns an ``nki.framework.kernel.Kernel`` whose ``__module__`` is
    ``"nki.framework.kernel"`` and whose ``__qualname__`` is ``None``. The wrapped function is
    reachable at ``__wrapped__`` (the ``functools.wraps`` convention) and at ``.func`` (this
    decorator's own name).
    """
    inner = getattr(kernel, "__wrapped__", None) or getattr(kernel, "func", None) or kernel
    return (inner.__module__, inner.__qualname__)


def _require_2d_or_more(scores: Tensor) -> None:
    if scores.ndim < 2:
        raise DsaTopkSelectError(
            f"scores must have a leading row axis and a trailing candidate axis; "
            f"got shape {tuple(scores.shape)}"
        )


def _nki_dtype_of(scores: Tensor):
    return getattr(nl, torch_to_nki_dtype(scores.dtype))


@torch._dynamo.assume_constant_result
def _nki_config(n_rows: int, width: int, k: int, nki_dtype):
    """Build the kernel's compile-time config once per distinct geometry.

    The factories are host-side Python that emits log records, which Dynamo cannot trace under
    ``fullgraph=True``; folding the result to a constant runs them eagerly, off the compiled
    graph.
    """
    topk_config = create_topk_config(
        inp_shape=(n_rows, width),
        inp_dtype=nki_dtype,
        k=k,
        num_programs=_NUM_PROGRAMS,
    )
    return create_rotational_topk_config(inp_shape=(n_rows, width), topk_config=topk_config)


@torch._dynamo.assume_constant_result
def _config_builds(n_rows: int, width: int, k: int, nki_dtype) -> bool:
    """True when the kernel's own factories accept this geometry.

    ``kernel_assert`` inside them raises ``AssertionError`` for an unsupported geometry, and
    that is the only exception treated as a "cannot run" answer. Any other exception -- a
    signature change, an import failure -- propagates, because swallowing it would silently
    disable the kernel for every geometry.

    Caveat inherited from the kernel: it signals infeasibility with a bare ``assert``, so under
    ``python -O`` the factories would not raise and this would wrongly return True. vLLM-Neuron
    is not run under ``-O``.
    """
    try:
        _nki_config(n_rows, width, k, nki_dtype)
        return True
    except AssertionError:
        return False


@torch._dynamo.assume_constant_result
def _record_nki_dispatch(n_rows: int, width: int, k: int) -> None:
    """Record which kernel the seam dispatched, and log it, off the compiled graph.

    The dispatch branch is traced under ``fullgraph=True``, so a host call Dynamo refuses would
    break it -- a bare ``logger.info`` here fails with ``Unsupported: logging.Logger method not
    supported for non-export cases``. A folded helper may take ints, strings and dtypes only:
    Dynamo turns every non-tensor argument into a Python constant at trace time, and an
    ``@nki.jit`` kernel is a frozen dataclass it cannot reconstruct (``NotImplementedError:
    currently can't reconstruct arbitrary frozen dataclass instances``). The kernel is therefore
    read as the module global ``rotational_topk`` instead of passed in.
    """
    _COUNTERS.last_kernel = _kernel_identity_of(rotational_topk)
    logger.info(
        "[dsa-topk] kernel=rotational-nki rows=%d width=%d k=%d", n_rows, width, k
    )


def can_run_dsa_topk_select(scores: Tensor, k: int) -> bool:
    """True when the NKI route is available and the kernel serves this geometry.

    Four necessary conditions, then the factory dry-run: the runtime gate, the rank, the dtype,
    and ``0 < k < width``. That last pre-filter is load-bearing -- the kernel refuses
    ``k == width`` for sorted output from an assert in its body rather than in the factories, so
    the dry-run cannot catch it.
    """
    if not can_run_kernel(scores):
        return False
    if scores.ndim < 2:
        return False
    if scores.dtype not in _SUPPORTED_DTYPES:
        return False
    width = int(scores.shape[-1])
    if not 0 < k < width:
        return False
    n_rows = 1
    for d in scores.shape[:-1]:
        n_rows *= int(d)
    return _config_builds(n_rows, width, k, _nki_dtype_of(scores))


def dsa_topk_select(scores: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """Select the ``k`` highest scores along the last axis.

    Args:
        scores: ``[..., width]``. The last axis is the candidate axis; every leading axis
            is flattened into rows. ``float32`` or ``bfloat16`` -- read the module
            docstring on why ``bfloat16`` makes an index-set comparison ill-defined.
        k: how many candidates to keep. ``0 < k < width``.

    Returns:
        ``(values, indices)`` shaped ``[..., k]``, highest first, indices ``int64`` to
        match ``torch.topk``; the kernel emits unsigned indices and they are cast once,
        here.
    """
    _require_2d_or_more(scores)
    width = int(scores.shape[-1])
    if k <= 0:
        raise DsaTopkSelectError(f"k must be positive; got k={k}")
    if k > width:
        raise DsaTopkSelectError(
            f"k must not exceed the candidate axis; got k={k} for width={width}"
        )
    if scores.dtype not in _SUPPORTED_DTYPES:
        raise DsaTopkSelectError(
            f"scores dtype must be one of {[str(d) for d in _SUPPORTED_DTYPES]}; "
            f"got {scores.dtype}. float16 is refused because the kernel's pad sentinel "
            f"has no float16 branch."
        )

    if not can_run_dsa_topk_select(scores, k):
        return _dsa_topk_select_torch(scores, k)

    leading = tuple(scores.shape[:-1])
    flat = scores.reshape(-1, width)
    n_rows = int(flat.shape[0])
    config = _nki_config(n_rows, width, k, _nki_dtype_of(scores))

    _count_nki_dispatch()
    _record_nki_dispatch(n_rows, width, k)
    values, indices = wrap_nki(rotational_topk)[config.n_prgs](flat, config)

    out_shape = (*leading, k)
    return values.reshape(out_shape), indices.reshape(out_shape).to(torch.int64)


def _dsa_topk_select_torch(scores: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """CPU reference and fallback: reached without the simulator, with kernels off, or on a geometry the kernel refuses."""
    _count_torch_fallback()
    logger.info(
        "[dsa-topk] kernel=torch rows=%d width=%d k=%d reason=nki-route-unavailable",
        scores.numel() // int(scores.shape[-1]),
        int(scores.shape[-1]),
        k,
    )
    return torch.topk(scores, k, dim=-1)
