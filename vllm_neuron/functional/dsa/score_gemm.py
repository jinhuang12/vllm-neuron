# SPDX-License-Identifier: Apache-2.0
"""The sparse-attention indexer's score GEMM, in NKI.

For every query token ``m`` and every key candidate ``n``::

    logits[m, n] = sum over h of  weights[m, h] * ReLU( dot( q[m, h, :], k[n, :] ) )

``q`` is the per-head indexer query, ``k`` the single key per candidate (the indexer
is multi-query attention, so ``k`` carries no head axis), and ``weights`` the
per-token per-head gate, already carrying every scale the caller folds in. The head
axis is reduced away; the output is one score per (token, candidate) pair, in fp32
because ``nc_matmul`` accumulates there anyway and because bf16 collapses distinct
scores badly enough to destroy the ranking the next stage computes.

The ReLU sits inside the head sum, and that order is the function -- rectifying after
the head reduction computes something else. It also rules out the free PSUM
accumulation a plain contraction would get: a nonlinearity and a per-token scale
stand between each head's matmul and the sum, so every head pays its own matmul,
rectify, scale and one add into an SBUF accumulator.

``nisa.nc_matmul`` contracts the partition axis of both operands, so the head
dimension has to arrive there. The kernel makes that turn itself, transposing each
slab on the PE through a PSUM tile of its own dtype. The turn is exact for finite
values, because the destination dtype matches the input and a bf16 value written to a
bf16 tile cannot round. It is not bit-accurate for NaN or Inf, where one bad element
can reach every output of its partition column inside that tile. Nothing scans for
finiteness here: neither the rotation nor the normalisation the callers apply removes
a non-finite value, and a per-call scan would cost more than the transport it
replaces.

Quantisation, masking, top-k and head padding all belong to neighbouring stages. In
particular there is no MX path: ``nc_matmul_mx`` needs NeuronCore-v4 and this target
is v3. Dropping the upstream Hadamard rotation along with the quantisation is exact
rather than merely cheaper, since ``q @ H @ H.T @ k.T == q @ k.T``.
"""

import logging

import torch
from dataclasses import dataclass
from torch import Tensor

import nki
import nki.isa as nisa
import nki.language as nl

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.utils.neuron_utils import can_run_kernel

logger = logging.getLogger(__name__)

INDEX_HEAD_DIM = 128
"""The indexer head dimension, and the length of every dot product this kernel contracts.

It is pinned rather than tiled, and the reason is a coincidence worth stating: 128 is also the
partition maximum, so the contraction axis fits in exactly one tile and there is no contraction loop
at all. A head dimension other than 128 would need that loop, so the gate refuses it and the torch
reference serves it instead.
"""

INDEX_N_HEADS = 32
"""``index_n_heads`` as the model configuration declares it, recorded for the reader; not a limit.

The head axis is an ordinary loop bound here, so any positive head count runs. Two upstream
comments disagree with this value (``attention.py:233`` says 64, ``attention.py:384`` says 16),
but both are prose rather than configuration; the value the model loads at run time is 32.
"""

TOKEN_TILE = 128
"""Query tokens per matmul, bounded by the stationary free size, which is 128.

Declared as a literal and asserted against ``nl.tile_size`` by the test rather than read from it at
import. Reading the tile extents at import is not safe on this image: ``nl.tile_size.psum_num_banks``
raises ``RuntimeError: No backend set`` outside an activated backend, so a module that read its
neighbours would import fine here and break somewhere else.
"""

CAND_TILE = 512
"""Key candidates per matmul, bounded by the moving free size, which is 512 on NeuronCore-v2 and v3."""

CONTRACTION_TILE = 128
"""The contraction extent, bounded by the partition maximum. Equal to ``INDEX_HEAD_DIM`` by design."""

_SUPPORTED_Q_DTYPES = (torch.bfloat16,)
"""Query and key dtypes that take the NKI route.

bf16 is the indexer path's dtype and it is also what the upstream reference computes in: it
casts both fp8 operands up to bf16 before the matmul (``rocm_aiter_mla_sparse.py:714-715``) and
accumulates in fp32. ``nc_matmul`` does the same natively, so the two paths agree numerically
rather than approximately.
"""

_SUPPORTED_W_DTYPES = (torch.float32,)
"""Weight dtypes that take the NKI route. Upstream declares ``weights`` fp32
(``rocm_aiter_mla_sparse.py:703``) and the per-token scale is applied in fp32 here, so a bf16
weight would quietly lose precision in the fold."""


class ScoreGemmError(ValueError):
    """A malformed call: wrong rank, a head or feature mismatch, or an unsupported head dimension."""


@dataclass
class _ScoreGemmDispatchCounters:
    """Per-process record of how this module's entry point was reached."""

    nki_dispatch: int = 0
    torch_fallback: int = 0
    last_kernel: tuple[str, str] | None = None


_COUNTERS = _ScoreGemmDispatchCounters()


def reset_score_gemm_dispatch_counters() -> None:
    """Zero this module's dispatch counters."""
    _COUNTERS.nki_dispatch = 0
    _COUNTERS.torch_fallback = 0
    _COUNTERS.last_kernel = None


def score_gemm_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last reset."""
    return (_COUNTERS.nki_dispatch, _COUNTERS.torch_fallback)


@torch._dynamo.assume_constant_result
def _count_torch_fallback() -> None:
    """Count one torch-path entry, off the traced graph."""
    _COUNTERS.torch_fallback += 1


@torch._dynamo.assume_constant_result
def _count_nki_dispatch() -> None:
    """Count one kernel dispatch, off the traced graph: a store Dynamo traces becomes a guard."""
    _COUNTERS.nki_dispatch += 1


def score_gemm_kernel_identity() -> tuple[str, str] | None:
    """``(module, qualname)`` of the kernel the seam last dispatched, or ``None``."""
    return _COUNTERS.last_kernel


def _kernel_identity_of(kernel) -> tuple[str, str]:
    """``(module, qualname)`` of the function a ``@nki.jit`` object wraps.

    Reading those attributes off the decorated object would report the decorator
    instead: ``@nki.jit`` returns an ``nki.framework.kernel.Kernel`` whose
    ``__module__`` is ``"nki.framework.kernel"`` and whose ``__qualname__`` is
    ``None``. The wrapped function is reachable at ``__wrapped__`` (the
    ``functools.wraps`` convention) and at ``.func`` (this decorator's own name).
    """
    inner = getattr(kernel, "__wrapped__", None) or getattr(kernel, "func", None) or kernel
    return (inner.__module__, inner.__qualname__)


# ---------------------------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------------------------


def _transpose_tile(src_sb, rows, cols):
    """``[rows, cols]`` SBUF -> ``[cols, rows]`` SBUF, turned on the PE through same-dtype PSUM."""
    ps = nl.ndarray((cols, rows), dtype=src_sb.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=ps, data=src_sb)
    tile = nl.ndarray((cols, rows), dtype=src_sb.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tile, src=ps)
    return tile


def _keys_transposed(k_hbm, head_dim, cands):
    """``k`` as stored, ``[cands, head_dim]`` -> ``[head_dim, cands]`` SBUF, one PE turn per 128 rows."""
    kt = nl.ndarray((head_dim, cands), dtype=k_hbm.dtype, buffer=nl.sbuf)
    for c0 in range(0, cands, CONTRACTION_TILE):
        cw = min(CONTRACTION_TILE, cands - c0)
        k_sb = nl.ndarray((cw, head_dim), dtype=k_hbm.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=k_sb, src=nl.load(k_hbm[c0:c0 + cw, :]))
        nisa.tensor_copy(dst=kt[:, c0:c0 + cw], src=_transpose_tile(k_sb, cw, head_dim))
    return kt


@nki.jit
def _score_gemm_nki(q_hbm, k_hbm, w_hbm):
    """Rectified per-head scores, weighted and summed over heads.

    Args:
        q_hbm: ``[tokens, heads, head_dim]`` bf16 -- the query as stored; the kernel puts the head
            dimension on the partition axis itself.
        k_hbm: ``[cands, head_dim]`` bf16 -- the key as stored, with no head axis (MQA).
        w_hbm: ``[tokens, heads]`` fp32 -- the per-token per-head gate, fully pre-scaled.

    Returns:
        ``[tokens, cands]`` fp32.

    All transport is on chip. Every operand slab is a ``[rows, head_dim]`` tile with
    ``rows <= 128``, loaded as stored and turned by one PE ``nc_transpose`` into a
    same-dtype PSUM tile, then copied to SBUF; a ragged ``rows`` needs no padding
    because the PE takes any tile up to (128, 128). ``k`` is turned once per call into
    ``[head_dim, cands]`` and each (token tile, candidate tile, head) reads a column
    range of it -- at ``cands`` 1024 that tile is 2,048 bytes per partition, 262,144
    bytes of SBUF in all.

    Each loop level answers to one bound: ``tokens`` walks in ``TOKEN_TILE`` steps
    because it becomes the stationary free size (max 128), ``cands`` walks in
    ``CAND_TILE`` steps because it becomes the moving free size (max 512), and
    ``head_dim`` does not walk at all because it is the contraction axis and equals
    the partition maximum exactly.

    The matmul tiles are loaded without a widening cast: ``nc_matmul`` admits bf16
    operands and accumulates in fp32 regardless, so casting up first would buy no
    precision and would cost four times the cycles by this ISA's own cost model.
    """
    tokens = q_hbm.shape[0]
    heads = q_hbm.shape[1]
    head_dim = q_hbm.shape[2]
    cands = k_hbm.shape[0]

    out = nl.ndarray((tokens, cands), dtype=nl.float32, buffer=nl.shared_hbm)
    kt = _keys_transposed(k_hbm, head_dim, cands)

    for m0 in range(0, tokens, TOKEN_TILE):
        mw = min(TOKEN_TILE, tokens - m0)
        for n0 in range(0, cands, CAND_TILE):
            nw = min(CAND_TILE, cands - n0)

            # The head sum lives in SBUF, not PSUM, because a rectify and a scale stand between each
            # head's matmul and this add. One memset per output tile reads more easily than
            # special-casing the first head inside the loop.
            acc = nl.ndarray((mw, nw), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=acc, value=0.0)

            for h in range(heads):
                # Stationary: this head's [mw, head_dim] rows as stored, turned to [head_dim, mw] so
                # head_dim is the partition axis that contracts and mw becomes PSUM's partition.
                q_sb = nl.ndarray((mw, head_dim), dtype=q_hbm.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=q_sb, src=nl.load(q_hbm[m0:m0 + mw, h, :]))
                q_tile = _transpose_tile(q_sb, mw, head_dim)

                # Moving: [head_dim, nw], a column range of the keys turned once above.
                k_tile = kt[:, n0:n0 + nw]

                # dst = stationary.T @ moving = [mw, nw]. PSUM and fp32 are both forced here.
                ps = nl.ndarray((mw, nw), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=ps, stationary=q_tile, moving=k_tile, accumulate=False)

                raw = nl.ndarray((mw, nw), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=raw, src=ps)

                # Rectify per head, before the weight and before the sum.
                rect = nl.ndarray((mw, nw), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=rect, op=nl.relu, data=raw)

                # The per-token weight has to be a (mw, 1) column: tensor_scalar's operand0 is a
                # per-partition scalar, and a (1, nw) row is refused by the MLIR verifier.
                wcol = nl.ndarray((mw, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=wcol, src=nl.load(w_hbm[m0:m0 + mw, h:h + 1], dtype=nl.float32))

                scaled = nl.ndarray((mw, nw), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=scaled, data=rect, op0=nl.multiply, operand0=wcol)

                nisa.tensor_tensor(dst=acc, data1=acc, data2=scaled, op=nl.add)

            nl.store(out[m0:m0 + mw, n0:n0 + nw], value=acc)

    return out


# ---------------------------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------------------------


@torch._dynamo.assume_constant_result
def _record_nki_dispatch(tokens: int, cands: int, heads: int, head_dim: int) -> None:
    """Record which kernel the seam dispatched, and log it, off the compiled graph.

    The dispatch branch is traced under ``fullgraph=True``, so a host call Dynamo
    refuses would break it. A folded helper may take ints, strings and dtypes only:
    Dynamo converts every non-tensor argument into a constant at trace time, and an
    ``@nki.jit`` object is a frozen dataclass it cannot reconstruct. The kernel is
    therefore read as a module global instead of passed in.
    """
    _COUNTERS.last_kernel = _kernel_identity_of(_score_gemm_nki)
    logger.info(
        "[dsa-score-gemm] kernel=nki tokens=%d cands=%d heads=%d head_dim=%d",
        tokens,
        cands,
        heads,
        head_dim,
    )


def _validate(q: Tensor, k: Tensor, weights: Tensor) -> tuple[int, int, int, int]:
    """Host-side shape validation. Returns ``(tokens, heads, head_dim, cands)``.

    Reads only ``.shape`` and ``.dtype``, never a tensor value, so nothing here forces
    a device-to-host synchronisation or a data-dependent trace.
    """
    if q.ndim != 3:
        raise ScoreGemmError(
            f"q must be 3-D [tokens, heads, head_dim]; got shape {tuple(q.shape)}"
        )
    if k.ndim != 2:
        raise ScoreGemmError(
            f"k must be 2-D [cands, head_dim] -- the indexer is MQA, so k carries no head axis; "
            f"got shape {tuple(k.shape)}"
        )
    tokens, heads, head_dim = (int(d) for d in q.shape)
    cands = int(k.shape[0])
    if int(k.shape[1]) != head_dim:
        raise ScoreGemmError(
            f"k's feature width must match q's head_dim {head_dim}; got {int(k.shape[1])}"
        )
    if weights.ndim != 2 or tuple(weights.shape) != (tokens, heads):
        raise ScoreGemmError(
            f"weights must be [tokens, heads] = {(tokens, heads)}; got {tuple(weights.shape)}"
        )
    if tokens <= 0 or cands <= 0 or heads <= 0:
        raise ScoreGemmError(
            f"tokens, cands and heads must all be positive; got {(tokens, cands, heads)}"
        )
    if head_dim != INDEX_HEAD_DIM:
        raise ScoreGemmError(
            f"the score GEMM contracts exactly {INDEX_HEAD_DIM} features in one tile; got head_dim "
            f"{head_dim}"
        )
    return tokens, heads, head_dim, cands


def can_run_dsa_score_gemm(q: Tensor, k: Tensor, weights: Tensor) -> bool:
    """Whether the NKI kernel serves this call. ``False`` sends it to the torch path."""
    if not can_run_kernel():
        return False
    if q.dtype not in _SUPPORTED_Q_DTYPES or k.dtype not in _SUPPORTED_Q_DTYPES:
        return False
    if weights.dtype not in _SUPPORTED_W_DTYPES:
        return False
    if q.ndim != 3 or k.ndim != 2 or weights.ndim != 2:
        return False
    if int(q.shape[2]) != INDEX_HEAD_DIM or int(k.shape[1]) != INDEX_HEAD_DIM:
        return False
    return tuple(weights.shape) == (int(q.shape[0]), int(q.shape[1]))


def dsa_score_gemm(q: Tensor, k: Tensor, weights: Tensor) -> Tensor:
    """Rectified per-head query-key scores, weighted and summed over heads.

    Args:
        q: ``[tokens, heads, head_dim]`` -- the indexer query, one vector per head.
            ``bfloat16`` takes the NKI route; any other dtype is served by the torch path.
        k: ``[cands, head_dim]`` -- one key per candidate, shared across heads (MQA). The candidate
            axis is pool-granular on this checkpoint rather than token-granular, which is why the
            next stage selects pools.
        weights: ``[tokens, heads]`` fp32 -- the per-token per-head gate, already carrying every
            scale the caller folds in, including the query scale, ``head_dim ** -0.5`` and
            ``n_head ** -0.5``. This seam applies no scale of its own.

    Returns:
        ``[tokens, cands]`` fp32.

    Raises:
        ScoreGemmError: for a malformed call -- a non-3D ``q``, a ``k`` whose feature width does not
            match, a ``weights`` of the wrong shape, or a head dimension that is not 128.
    """
    tokens, heads, head_dim, cands = _validate(q, k, weights)

    if not can_run_dsa_score_gemm(q, k, weights):
        _count_torch_fallback()
        return _dsa_score_gemm_torch(q, k, weights)

    _count_nki_dispatch()
    # The counter, the log and the identity read are folded off the traced graph: a counter store
    # inside the trace becomes a value guard that fails on the first call after warmup.
    _record_nki_dispatch(tokens, cands, heads, head_dim)
    # Same layout in, same layout out: a contiguous caller gets its own storage back, a strided
    # one gets a plain copy. Neither is a transposing relayout; the kernel turns the operands.
    return wrap_nki(_score_gemm_nki)(q.contiguous(), k.contiguous(), weights.contiguous())


# ---------------------------------------------------------------------------------------------
# Torch reference
# ---------------------------------------------------------------------------------------------


def _dsa_score_gemm_torch(q: Tensor, k: Tensor, weights: Tensor) -> Tensor:
    """The CPU reference the NKI route is measured against, and the path a refused call takes.

    Keeps the reference order -- dot, rectify, weight, sum -- and runs everything in
    fp32, which is what ``nc_matmul`` does with bf16 operands and the only basis on
    which the two can be compared. There is no per-key dequant scale because a bf16
    route has nothing quantised to rescale.
    """
    per_head = torch.einsum("mhd,nd->mhn", q.float(), k.float())
    return (per_head.clamp(min=0.0) * weights.float().unsqueeze(-1)).sum(dim=1)
