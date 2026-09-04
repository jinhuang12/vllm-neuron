# SPDX-License-Identifier: Apache-2.0
"""DSA indexer top-k selection, as a WRAP of the tested rotational NKI top-k kernel.

`inc-glm53f-043`. The DSA indexer scores every candidate context position for a query
token and then keeps only the highest-scoring ones; this module is that selection. It
takes a score tensor whose last axis is the candidate axis and returns the selected
values and their indices, highest first::

    values, indices = dsa_topk_select(scores, k)

WHAT THIS MODULE AUTHORS, AND WHAT IT DELIBERATELY DOES NOT. It authors a seam, a gate
and a torch oracle. It authors NO kernel arithmetic at all: the selection is performed by
``rotational_topk``, the ``@nki.jit`` kernel the fork already vendors at
``vllm_neuron/functional/vendored_kernels/rotational_topk/``. That is what the increment's
substrate declaration means by WRAP, and it is why P13 is satisfied without a line of NKI
being written here -- the substrate already provides the primitive, so re-deriving it would
add a second implementation of a tested kernel rather than close a gap.

WHY A SECOND SEAM ONTO A KERNEL ``functional/topk.py`` ALREADY WRAPS, stated plainly
because a reader will ask. ``functional/topk.py`` is the SAMPLING top-k: it exists to do a
distributed top-k over a vocabulary sharded across tensor-parallel ranks, and its public
entry points take a process group, a gather dimension and a rank tensor
(``topk``, ``batch_sharded_topk``). The DSA indexer needs none of that -- one device, one
tensor, no collective -- and it needs one thing that module does not have: a per-call
dispatch count its own tests can read, which is the route predicate this increment's
acceptance is built on. Adding that counter to ``functional/topk.py`` would make this
increment a second writer on a file that ships at the pin, widening its declared surface
from one new file to a shared pin surface it carries no registration for. So the seam is
authored here and the KERNEL is shared, which is the part that matters: there is exactly
one top-k kernel in this tree and this module adds none.

WHAT IS SHARED WITH THAT MODULE IS THE KERNEL'S OWN CONFIG FACTORIES, NOT A COPY OF ITS
LOGIC. ``create_topk_config`` and ``create_rotational_topk_config`` are the kernel's own
public factories; calling them is how both modules stay correct when nkilib changes its
hardware parameters or its staging cost model. Neither the envelope rule nor the staging
arithmetic is restated here -- it is asked, not remembered.

THE ENVELOPE IS DECIDED BY THE KERNEL AND NEVER BY A CONSTANT IN THIS FILE. The kernel
splits the candidate axis into stages and asserts a concatenated SBUF free dimension; that
bound is NOT monotonic in ``k``, so no static width or ``k`` cap can express it. ``_config_builds``
below dry-runs the factories and reports whether they raise. Measured on this image at the
increment's own geometry: at 4 rows and 4,096 candidates BOTH declared ``k`` values build,
while ``k == 2048`` needs a candidate width of at least 3,072 -- 2,048 is refused because
``k == width`` is out of envelope, not because 2,048 is a special number.

ONE MEASURED FACT ABOUT DTYPE THAT A CALLER MUST KNOW, because it decides whether an index
comparison against ``torch.topk`` means anything. A top-k index SET is well defined only
when the k-th and (k+1)-th scores differ. In ``float32`` at this geometry all 16,384
sampled scores are distinct and the selected index sets agree with ``torch.topk`` on every
row. In ``bfloat16`` the SAME scores collapse to 1,025 distinct values out of 16,384, so
thousands of candidates tie at the selection boundary; the kernel and ``torch.topk`` then
break those ties differently and the index sets disagree (measured: 1 of 4 rows agree at
``k == 2048``, 0 of 4 at ``k == 512``) WHILE THE SELECTED VALUES REMAIN BIT-IDENTICAL, max
absolute difference exactly ``0.0``. That is a tie, not a selection error. This module
therefore does NOT upcast a bfloat16 input -- silently changing a caller's numerics and
memory would be worse than the tie -- and this increment's acceptance declares ``float32``,
where the comparison it makes is meaningful. A caller who needs a reproducible index set
from bfloat16 scores must break the ties itself.

THE TORCH PATH IS THE ORACLE AND THE CONSTRAINT-VIOLATION FALLBACK, NEVER THE SHIPPED
IMPLEMENTATION (D6). It runs when the NKI route is unavailable -- on CPU without the
simulator, with kernels disabled, or on a geometry the kernel refuses -- and it increments
its own counter when it does. Keeping the fallback REACHABLE is what makes the route
predicate's ``torch_fallback == 0`` a measurement instead of decoration (D1.5): the
acceptance drives an out-of-envelope input and reads that same counter NON-zero, so a zero
on the declared cases is known to be a zero the instrument could have moved.

Tier N harness -- the NKI simulator on a host CPU, no device and no lease::

    PYTHONDONTWRITEBYTECODE=1 VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \\
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \\
    python -m pytest test/vllm_neuron/functional/dsa/test_topk_select.py \\
        --timeout 60 -v -s -p no:cacheprovider
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

# LNC grid degree handed to the config factories. It is the same value
# ``functional/topk.py`` pins for the same reason: the feasibility dry-run
# (``_config_builds``) and the real-run config build (``_nki_config``) must use ONE value,
# or the gate would admit a config the run does not build. The kernel forces it to 1
# internally when the row count is 1.
_NUM_PROGRAMS = 2

# The dtypes the kernel is tested for. Its pad sentinel has no float16 branch, so float16
# is refused here rather than handed to it.
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float32)


class DsaTopkSelectError(ValueError):
    """Raised for an input this module will not hand to the kernel or to the oracle."""


@dataclass
class _TopkSelectDispatchCounters:
    """How the seam below was reached, per process.

    A counter object private to this module, so a test can attribute a dispatch to THIS
    seam and to no other. ``last_kernel`` records the kernel the seam actually dispatched,
    which is what lets ``topk_select_kernel_identity`` report an identity derived THROUGH
    the seam rather than one read off an import (D13.1).
    """

    nki_dispatch: int = 0
    torch_fallback: int = 0
    last_kernel: tuple[str, str] | None = None


_COUNTERS = _TopkSelectDispatchCounters()


def reset_topk_select_dispatch_counters() -> None:
    """Zero this seam's counters. Call immediately before a case's first call."""
    _COUNTERS.nki_dispatch = 0
    _COUNTERS.torch_fallback = 0
    _COUNTERS.last_kernel = None


def topk_select_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last reset."""
    return (_COUNTERS.nki_dispatch, _COUNTERS.torch_fallback)


def topk_select_kernel_identity() -> tuple[str, str] | None:
    """``(module, qualname)`` of the kernel the seam LAST dispatched, or ``None``.

    Derived through the seam, not from this module's import list, so it certifies what
    ran rather than what was imported (D13.1). ``None`` before any dispatch, which is the
    reading that distinguishes "no kernel ran" from "some kernel ran".
    """
    return _COUNTERS.last_kernel


def _kernel_identity_of(kernel) -> tuple[str, str]:
    """``(module, qualname)`` of the function a ``@nki.jit`` object actually wraps.

    MEASURED, not assumed. ``@nki.jit`` returns an ``nki.framework.kernel.Kernel``, whose
    own ``__module__`` is ``"nki.framework.kernel"`` and whose ``__qualname__`` is ``None``
    -- so reading those two attributes off the decorated object would record the DECORATOR's
    identity and certify nothing about which kernel was wrapped. On this image the wrapped
    function is reachable at both ``__wrapped__`` (the ``functools.wraps`` convention) and
    ``.func``, and both resolve to
    ``("vllm_neuron.functional.vendored_kernels.rotational_topk.rotational_topk",
    "rotational_topk")``. ``__wrapped__`` is preferred because it is the language-level
    convention rather than this decorator's private attribute name.
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

    ``assume_constant_result`` for the reason ``functional/topk.py`` gives at its own
    equivalent: the factories are host-side Python that emits log records, which Dynamo
    cannot trace under ``fullgraph=True``. Folding the result to a constant runs them
    eagerly, off the compiled graph, once per distinct geometry.
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
    """True when the kernel's OWN factories accept this geometry.

    This is the authoritative envelope test and there is no second copy of it: the
    factories are asked. ``kernel_assert`` inside them raises ``AssertionError`` for an
    out-of-envelope geometry, and that is the ONLY exception treated as a "cannot run"
    answer -- any other exception (a signature change, an import failure) is not a
    feasibility verdict and is allowed to propagate, because swallowing it would disable
    the kernel for every geometry with nothing failing to say so.

    CAVEAT, inherited from the kernel: it signals infeasibility with a bare ``assert``, so
    under ``python -O`` the factories would not raise and this would wrongly return True.
    vLLM-Neuron is not run under ``-O``.
    """
    try:
        _nki_config(n_rows, width, k, nki_dtype)
        return True
    except AssertionError:
        return False


def can_run_dsa_topk_select(scores: Tensor, k: int) -> bool:
    """True when the NKI route is available AND the kernel serves this geometry.

    Four necessary conditions then the authoritative dry-run: the runtime gate, the rank,
    the dtype, and ``0 < k < width`` -- the kernel refuses ``k == width`` for sorted
    output, and the dry-run cannot catch that because the assert lives in the kernel body
    rather than in the factories, so this pre-filter is load-bearing.
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
    """THE COUNTED SEAM. Select the ``k`` highest scores along the last axis.

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

    _COUNTERS.nki_dispatch += 1
    _COUNTERS.last_kernel = _kernel_identity_of(rotational_topk)
    logger.info(
        "[dsa-topk] kernel=rotational-nki rows=%d width=%d k=%d", n_rows, width, k
    )
    values, indices = wrap_nki(rotational_topk)[config.n_prgs](flat, config)

    out_shape = (*leading, k)
    return values.reshape(out_shape), indices.reshape(out_shape).to(torch.int64)


def _dsa_topk_select_torch(scores: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """The CPU oracle and the constraint-violation fallback -- never the shipped path (D6).

    Reached on CPU without the simulator, with NKI kernels disabled, or on a geometry the
    kernel refuses. It increments its own counter so a test can state which route ran
    instead of assuming it.
    """
    _COUNTERS.torch_fallback += 1
    logger.info(
        "[dsa-topk] kernel=torch rows=%d width=%d k=%d reason=nki-route-unavailable",
        scores.numel() // int(scores.shape[-1]),
        int(scores.shape[-1]),
        k,
    )
    return torch.topk(scores, k, dim=-1)
