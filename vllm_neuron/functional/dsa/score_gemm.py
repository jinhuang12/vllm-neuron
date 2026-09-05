# SPDX-License-Identifier: Apache-2.0
"""The DSA indexer's score GEMM: per-head query-key dot products, rectified, then weighted and summed.

WHAT THIS COMPUTES, in one line. For every query token ``m`` and every key candidate ``n``::

    logits[m, n] = sum over h of  weights[m, h] * ReLU( dot( q[m, h, :], k[n, :] ) )

``q`` is the per-head indexer query, ``k`` is the single shared key per candidate (the indexer is
multi-query attention, so ``k`` carries no head axis), and ``weights`` is the per-token per-head
gate. The head axis is reduced away; the output is one fp32 score per (token, candidate) pair.

**THE ReLU SITS INSIDE THE HEAD SUM, AND THAT ORDER IS THE WHOLE FUNCTION.** Rectifying after the
head reduction is a different function, and it is the easiest thing here to get wrong, so the order
is taken verbatim from two independent world sources rather than from anyone's recollection:

  * ``vllm/v1/attention/ops/triton_fp8_mqa_logits.py:125-132`` at pin ``878631b6`` -- ``tl.dot`` then
    ``tl.maximum(scores, 0.0)`` then ``scores * w_block`` then ``tl.sum(scores, axis=0)``.
  * ``vllm/v1/attention/ops/rocm_aiter_mla_sparse.py:732-733`` at the same pin, which is DeepGEMM's
    own torch reference (its ``:687`` cites
    ``https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_attention.py#L84``)::

        score = torch.einsum("mhd,nd->hmn", q, k).float() * scale.reshape(-1)
        logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)

The test carries a discriminating control for this: on its own fixture the wrong order (weight before
rectify) differs from the right one by about ``6.7e+01``, so an agreeing kernel is agreeing about the
order and not merely about the arithmetic.

**THAT ORDER ALSO COSTS THE FREE PSUM ACCUMULATION, WHICH IS WHY THE HEAD LOOP LOOKS EXPENSIVE.**
``nisa.nc_matmul`` can accumulate successive matmuls into one PSUM tile for nothing, which is how
``vllm_neuron/functional/attention/mla_absorb.py:203-214`` walks its contraction axis. Here the head
axis cannot use it, because a nonlinearity and a per-token scale stand between each head's matmul and
the sum. So each head gets its own matmul, its own rectify, its own scale, and one add into an SBUF
accumulator. This is a property of the function, not a missed optimisation.

WHY THE HOST OWNS THE TRANSPORT. ``nisa.nc_matmul`` contracts the PARTITION axis of both operands, so
the head dimension has to be the partition axis of both tiles on the way in. The caller therefore
hands over ``q`` already permuted to ``[H, D, M]`` and ``k`` already transposed to ``[D, N]``. This
follows the reason ``mla_absorb.py:78-84`` records for taking its weight as ``[H, K, N]``: the
orientation belongs to whoever can pay for it once, and a device-side transpose would route through
``nc_matmul`` again for no gain.

WHAT THE OUTPUT DTYPE IS, AND WHY IT IS NOT NEGOTIABLE. fp32. Three independent reasons agree, so
this is not a preference: ``nc_matmul`` writes an fp32 PSUM destination on this generation and always
accumulates in fp32; the upstream contract returns fp32 (``rocm_aiter_mla_sparse.py:710``); and
``vllm_neuron/functional/dsa/topk_select.py``'s own record shows bf16 collapsing 16,384 distinct
scores to 1,025 distinct values, which would destroy the ranking the next stage exists to compute.

WHAT THE bf16 IN THE INCREMENT TITLE MEANS. It names this campaign's bf16 NKI route, which is the
same wording correction the ``-044`` and ``-045`` ledger rows already carry. It does NOT mean an MX
form was re-derived here. The design trace's own score stage is ALREADY bf16: its docstring reads
"Q/K/W projections still use ``nc_matmul_mx`` for speed, but the score matmul uses bf16 nc_matmul (no
Q/K quantization, no Hadamard rotation)", and its pseudocode names the stage
``score_against_cache_bf16`` (``kickoff/preflight/g1-probe-transcripts.md:883-899``). What is MX and
gen4-only in that trace is the Q/K/W PROJECTION stage, which belongs to other increments.

THE MX FORM THIS DOES REPLACE IS THE UPSTREAM ONE. On GPU the same scores come out of
``fp8_fp4_mqa_logits``, whose operands are fp8-e4m3 values carrying ``ue8m0`` power-of-two block
scales -- produced by ``fwht128_quant_fp8``'s Hadamard-128 rotation plus block-128 quantisation, and
by the key cache's block-32 path (``sparse_attn_indexer_kpool.py:46-47`` reads "MXFP4 layout: 2 values
packed per byte, ue8m0 (1-byte) scale per block of 32"). Dropping all of it is sound rather than
merely cheaper: the trace records that ``q_H @ (k_H).T = q @ H @ H.T @ k.T = q @ k.T``, so absent
quantisation the rotation is provably a no-op on the score, and the rotation existed only to make the
quantisation tolerable. No ``nc_matmul_mx`` and no MX-quantised nkilib kernel is reached from here;
that primitive needs NeuronCore-v4 and this target is v3.

WHAT THIS MODULE DELIBERATELY DOES NOT DO.
  * No quantisation, no dequantisation, and no scale tensors. A bf16 route has nothing to scale. The
    upstream per-query and per-key scales are folded into ``weights`` by the caller before the call,
    which is what ``vllm/models/glm5next/nvidia/attention.py:61-67`` already does on GPU.
  * No masking and no ``-inf`` fill. Upstream applies its causal and window mask AFTER this op
    (``rocm_aiter_mla_sparse.py:734``); the mask stage is a separate increment.
  * No top-k. The scores are pool-granular, and selecting over them belongs to the landed
    ``dsa_topk_select`` (the ``-043`` ledger row; see ``evidence-043.md``), which this feeds.
  * No head padding. Upstream pads the head count to 32 or 64 because the DeepGEMM kernels demand it
    (``attention.py:382-390``); a NKI head loop has no such constraint, so the padding does not port.

MEASURED REFUSALS THIS MODULE IS AUTHORED AROUND, each one read off a compiler rather than guessed.
The pre-authoring probe ``probe-046-mechanism.py`` and its bf16 round ``probe-046-mechanism-r2.py``
compiled the intended chain with the MLIR verifier ON and compared it against the torch oracle under
the simulator; the transcripts are ``probe-046-mechanism-capture-host.out`` and
``probe-046-mechanism-r2-host.out``.

  * A ``nisa.tensor_scalar`` per-token scale takes a ``(P, 1)`` COLUMN and refuses a ``(1, N)`` ROW.
    The probe's control arm was refused with ``Generated MLIR failed verification`` naming
    ``nisa.tensor_scalar_arith`` with ``operand0_static_tile_shape = array<i64: 1, 8>`` against
    ``dst_static_tile_shape = array<i64: 6, 8>``. This is the ``-045`` finding, re-read on the image
    that will compile this file. The column form below is the one that passed.
  * ``nc_matmul`` refuses an int32 operand (``nc_matmul stationary dtype int32 not supported``,
    recorded at ``vllm_neuron/functional/dsa/paged_gather.py:65-68``). The bf16 round used exactly
    that refusal as the firing control for its own pass, so "bf16 operands are accepted" is a
    reading and not a sweep that would have accepted anything.
  * A ``for`` target that unpacks a tuple is refused at lowering time with ``expecting simple
    variable`` (``kpool_hadamard.py:120-131``). Every loop here walks one plain variable.
  * A helper that builds a buffer from keyword arguments is refused, so buffers are declared inline.
  * A folded host helper may take ints, strings and dtypes ONLY; handing it an ``@nki.jit`` object
    fails because that object is a frozen dataclass Dynamo will not reconstruct
    (``kpool_hadamard.py:436-442``). ``_record_nki_dispatch`` below takes a ``str``.
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
at all. A head dimension other than 128 would need that loop, no declared case exercises one, so the
gate refuses it and the torch oracle serves it instead.
"""

INDEX_N_HEADS = 32
"""``index_n_heads`` on the target checkpoint's config, recorded for the reader; NOT a limit.

The head axis is an ordinary loop bound here, so any positive head count runs. Two comments upstream
disagree with this value (``attention.py:233`` says 64, ``attention.py:384`` says 16); the checkpoint
fixture is the authority and it reads 32.
"""

TOKEN_TILE = 128
"""Query tokens per matmul, bounded by the STATIONARY free size, which is 128.

Declared as a literal and asserted against ``nl.tile_size`` by the test rather than read from it at
import. Reading the tile extents at import is not safe on this image: ``nl.tile_size.psum_num_banks``
raises ``RuntimeError: No backend set`` outside an activated backend, so a module that read its
neighbours would import fine here and break somewhere else.
"""

CAND_TILE = 512
"""Key candidates per matmul, bounded by the MOVING free size, which is 512 on NeuronCore-v2 and v3."""

CONTRACTION_TILE = 128
"""The contraction extent, bounded by the partition maximum. Equal to ``INDEX_HEAD_DIM`` by design."""

_SUPPORTED_Q_DTYPES = (torch.bfloat16,)
"""Query and key dtypes that take the NKI route.

bf16 is the indexer path's dtype and it is also what the upstream reference itself computes in: it
casts both fp8 operands up to bf16 before the matmul (``rocm_aiter_mla_sparse.py:714-715``) and
accumulates in fp32. ``nc_matmul`` does the same thing natively, which is why this route is a
faithful re-derivation rather than an approximation of one.
"""

_SUPPORTED_W_DTYPES = (torch.float32,)
"""Weight dtypes that take the NKI route. Upstream declares ``weights`` fp32 (``:703``) and the
per-token scale is applied in fp32 here, so a bf16 weight would quietly lose the fold's precision."""


class ScoreGemmError(ValueError):
    """A malformed call: wrong rank, a head or feature mismatch, or an unsupported head dimension."""


@dataclass
class _ScoreGemmDispatchCounters:
    """Route-predicate counters for this module, form R-1 (``design/increment-plan.md`` D13).

    ``nki_dispatch`` counts dispatches THROUGH THIS MODULE'S SEAM, so the declared total over the
    declared case set is readable from one place. The declared set is the two cases the plan block
    names, one dispatch each, for a declared total of 2. Supplementary tile-boundary cases are
    counted in their own reset windows and are excluded from that total.
    """

    nki_dispatch: int = 0
    torch_fallback: int = 0
    last_kernel: tuple[str, str] | None = None


_COUNTERS = _ScoreGemmDispatchCounters()


def reset_score_gemm_dispatch_counters() -> None:
    """Zero the counters. Called at the START of each declared case (section 4b's convention)."""
    _COUNTERS.nki_dispatch = 0
    _COUNTERS.torch_fallback = 0
    _COUNTERS.last_kernel = None


def score_gemm_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last reset."""
    return (_COUNTERS.nki_dispatch, _COUNTERS.torch_fallback)


def score_gemm_kernel_identity() -> tuple[str, str] | None:
    """``(module, qualname)`` of the kernel the seam LAST dispatched, or ``None``.

    Derived THROUGH the seam rather than from this module's import list, so it certifies what ran
    instead of what was defined (D13.1). ``None`` before any dispatch, which is the reading that
    separates "no kernel ran" from "some kernel ran".
    """
    return _COUNTERS.last_kernel


def _kernel_identity_of(kernel) -> tuple[str, str]:
    """``(module, qualname)`` of the function a ``@nki.jit`` object actually wraps.

    Measured, not assumed, for the reason ``vllm_neuron/functional/dsa/ragged_pack.py:170-181``
    records: ``@nki.jit`` returns an ``nki.framework.kernel.Kernel`` whose ``__module__`` is
    ``"nki.framework.kernel"`` and whose ``__qualname__`` is ``None``, so reading those attributes off
    the decorated object would record the DECORATOR's identity and certify nothing.
    """
    inner = getattr(kernel, "__wrapped__", None) or getattr(kernel, "func", None) or kernel
    return (inner.__module__, inner.__qualname__)


# ---------------------------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------------------------


@nki.jit
def _score_gemm_nki(qt_hbm, kt_hbm, w_hbm):
    """Rectified per-head scores, weighted and summed over heads.

    Args:
        qt_hbm: ``[heads, head_dim, tokens]`` -- the query, ALREADY permuted so that the head
            dimension is the partition axis the matmul contracts.
        kt_hbm: ``[head_dim, cands]`` -- the key, ALREADY transposed, with no head axis (MQA).
        w_hbm: ``[tokens, heads]`` fp32 -- the per-token per-head gate, fully pre-scaled.

    Returns:
        ``[tokens, cands]`` fp32.

    THE TILE LOOP, and which bound each level answers to. ``tokens`` walks in ``TOKEN_TILE`` steps
    because it becomes the stationary free size (max 128); ``cands`` walks in ``CAND_TILE`` steps
    because it becomes the moving free size (max 512); ``head_dim`` does not walk at all, because it
    is the contraction axis and equals the partition maximum exactly.

    THE MATMUL TILES ARE LOADED WITHOUT A WIDENING CAST. ``nc_matmul`` admits bf16 operands and
    accumulates in fp32 regardless, so casting up first would buy no precision and would cost four
    times the cycles by this ISA's own cost model. The bf16 round of the pre-authoring probe compiled
    exactly this, with an int32-operand refusal as its control.
    """
    heads = qt_hbm.shape[0]
    head_dim = qt_hbm.shape[1]
    tokens = qt_hbm.shape[2]
    cands = kt_hbm.shape[1]

    out = nl.ndarray((tokens, cands), dtype=nl.float32, buffer=nl.shared_hbm)

    for m0 in range(0, tokens, TOKEN_TILE):
        mw = min(TOKEN_TILE, tokens - m0)
        for n0 in range(0, cands, CAND_TILE):
            nw = min(CAND_TILE, cands - n0)

            # The head sum lives in SBUF, not PSUM, because a rectify and a scale stand between each
            # head's matmul and this add. Zeroed rather than special-casing the first head: one
            # memset per output tile is cheaper to read than a branch inside the head loop.
            acc = nl.ndarray((mw, nw), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(dst=acc, value=0.0)

            for h in range(heads):
                # Stationary: [head_dim, mw]. Partition axis is head_dim, so it is what contracts;
                # the free axis mw becomes the PSUM tile's partition axis.
                q_tile = nl.ndarray((head_dim, mw), dtype=qt_hbm.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=q_tile, src=nl.load(qt_hbm[h, :, m0:m0 + mw]))

                # Moving: [head_dim, nw]. Same partition axis; its free axis becomes PSUM's free axis.
                k_tile = nl.ndarray((head_dim, nw), dtype=kt_hbm.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=k_tile, src=nl.load(kt_hbm[:, n0:n0 + nw]))

                # dst = stationary.T @ moving = [mw, nw]. PSUM and fp32 are both forced here.
                ps = nl.ndarray((mw, nw), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=ps, stationary=q_tile, moving=k_tile, accumulate=False)

                raw = nl.ndarray((mw, nw), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=raw, src=ps)

                # The rectify. It must happen HERE, per head, before the weight and before the sum.
                rect = nl.ndarray((mw, nw), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=rect, op=nl.relu, data=raw)

                # The per-token weight, as a (mw, 1) COLUMN. A (1, nw) row is refused by the MLIR
                # verifier -- see the module docstring's refusal census.
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
    """Record which kernel the seam dispatched, and log it, OFF the compiled graph.

    The template is landed and measured: ``kpool_hadamard.py:428-457``, itself following
    ``paged_gather.py:248-288``.

    ARGUMENT DISCIPLINE, WHICH IS THE POINT. A folded helper takes ints, strings and dtypes ONLY.
    Dynamo runs a folded call at trace time and converts every non-tensor argument to a constant
    first, and an ``@nki.jit`` object is a frozen dataclass it refuses to reconstruct
    (``NotImplementedError: currently can't reconstruct arbitrary frozen dataclass instances``). So
    the kernel is not a parameter: it is read as a module global, the same object the call site hands
    to ``wrap_nki`` on the next line.

    D13.1 still holds: this body runs only when the dispatch branch runs, so the recorded identity is
    derived by TAKING the branch rather than read off an import.
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

    Reads only ``.shape`` and ``.dtype``, never a tensor VALUE, so nothing here forces a
    device-to-host synchronisation or a data-dependent trace.
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
    """Whether the NKI kernel serves this call. ``False`` sends it to the torch oracle."""
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
    """THE COUNTED SEAM. Rectified per-head query-key scores, weighted and summed over heads.

    Args:
        q: ``[tokens, heads, head_dim]`` -- the indexer query, one vector per head.
            ``bfloat16`` takes the NKI route; any other dtype is served by the torch oracle.
        k: ``[cands, head_dim]`` -- one key per candidate, shared across heads (MQA). The candidate
            axis is POOL-GRANULAR on this checkpoint, not token-granular
            (``sparse_attn_indexer_kpool.py:551-554`` reads "logits are pool-granular
            (compress_ratio == index_kpool)"), which is why the next stage selects pools.
        weights: ``[tokens, heads]`` fp32 -- the per-token per-head gate, ALREADY carrying every
            scale the caller folds in. Upstream folds the query scale, ``head_dim ** -0.5`` and
            ``n_head ** -0.5`` into it before the call (``attention.py:61-67, 373-375``); this seam
            applies no scale of its own.

    Returns:
        ``[tokens, cands]`` fp32. fp32 is forced rather than chosen -- see the module docstring.

    Raises:
        ScoreGemmError: for a malformed call -- a non-3D ``q``, a ``k`` whose feature width does not
            match, a ``weights`` of the wrong shape, or a head dimension that is not 128.
    """
    tokens, heads, head_dim, cands = _validate(q, k, weights)

    if not can_run_dsa_score_gemm(q, k, weights):
        _COUNTERS.torch_fallback += 1
        return _dsa_score_gemm_torch(q, k, weights)

    # The transport the device cannot pay for: put the contracted axis on the partition axis of both
    # operands. `.contiguous()` is what makes each permute a real relayout rather than a stride view.
    qt = q.permute(1, 2, 0).contiguous()
    kt = k.t().contiguous()

    _COUNTERS.nki_dispatch += 1
    # The log and the identity read are FOLDED off the traced graph. The helper takes ints only and
    # reads the kernel as a module global; passing the kernel object is the measured defect that
    # pattern exists to avoid. The counter increment stays -- a plain int attribute store is a
    # recorded side effect, not a host call.
    _record_nki_dispatch(tokens, cands, heads, head_dim)
    return wrap_nki(_score_gemm_nki)(qt, kt, weights.contiguous())


# ---------------------------------------------------------------------------------------------
# Torch oracle
# ---------------------------------------------------------------------------------------------


def _dsa_score_gemm_torch(q: Tensor, k: Tensor, weights: Tensor) -> Tensor:
    """The reference the NKI route is measured against. NOT a fallback for kernel-class work (P13).

    This exists to be the oracle in the test and to serve a call the gate refuses, which is a
    malformed or unsupported-dtype call rather than kernel-class work taking a torch path. The
    counted ``torch_fallback`` reading is exactly 0 across the declared cases, and the test proves
    that zero can fire by handing the seam an unadmitted dtype on purpose.

    Transcribed from ``rocm_aiter_mla_sparse.py:732-733``, keeping its order: dot, rectify, weight,
    sum. The upstream per-key dequant scale is identically 1 on a bf16 route and so is absent, not
    dropped -- there is nothing quantised here to rescale.

    Everything runs in fp32. The inputs are upcast rather than accumulated in bf16, which matches
    what ``nc_matmul`` does with bf16 operands and is the only way the two can be compared at all.
    """
    per_head = torch.einsum("mhd,nd->mhn", q.float(), k.float())
    return (per_head.clamp(min=0.0) * weights.float().unsqueeze(-1)).sum(dim=1)
