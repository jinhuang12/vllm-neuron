# SPDX-License-Identifier: Apache-2.0
"""Fused kpool compression and Hadamard-128 rotation for the sparse-attention indexer.

``inc-glm53f-047``. This is WP4's key-pooling half: it takes ``index_kpool`` consecutive
tokens' indexer keys and compresses them into ONE key per pool, then rotates that key by the
Hadamard-128 transform. One kernel does both, which is the increment's whole purpose -- the
pooled vector never leaves the on-chip tile between the reduction and the rotation.

It is **kernel-class** under P13, and it is **SCRATCH with ZERO precedent**. That claim rests on
this increment's own tree-wide census, recorded in
``artifacts/campaigns/glm-5.3-flash-port/increments/predictions-047-prep-r2.txt``: a
case-insensitive search of ``vllm_neuron/`` and ``test/`` finds exactly ONE mention of "hadamard",
and it is a docstring in ``vllm_neuron/functional/mhc/sinkhorn.py:10`` stating that ``nkilib`` has
no Hadamard implementation; ``vllm_neuron/functional/vendored_kernels/`` carries ZERO Hadamard
files. There is no vendor kernel to wrap or adapt, so the arithmetic below is authored in NKI. The
torch code in this module is the CPU oracle -- one of the two roles the plan's substrate register
admits (``design/increment-plan.md`` section 4) -- and never the shipped implementation. **A
pooling reduction and a 128-point transform written in torch would be exactly the P13 fallback the
rule forbids.**

What the kernel computes
------------------------
The semantics are TRANSCRIBED from the upstream reference, not invented here, and every claim below
is cited by origin line in ``increments/probe-047-upstream-semantics.md``. The reference is
``vllm/models/glm5next/nvidia/ops/kpool_compress.py`` at ``vllm-project/vllm`` head ``878631b6``
(891 L, sha256 ``6e19b31c50de43f7dfa9c7d9b7a3d7b856939f49bd4a3c031d2b50d28fe04f68``).

Given ``slot_k[n_pools, 4, 128]``, ``slot_score[n_pools, 4, 128]`` and ``ape[4, 128]``::

    w[p, s, d] = softmax_over_s( slot_score[p, s, d] + ape[s, d] )
    pooled[p, d] = sum_s w[p, s, d] * slot_k[p, s, d]
    out[p, :] = FWHT_128( pooled[p, :] ) * (1 / sqrt(128))

**THE SOFTMAX IS PER (POOL, HEAD-DIM CHANNEL), NOT PER POOL.** The reference's own accumulators are
vectors over the head dimension (``kpool_compress.py:174-200``: ``max_score`` and ``denom`` are
``(BLOCK_D,)``), so there is one independent 4-way softmax for every ``(p, d)`` pair. A
whole-vector softmax over the 128 channels is a plausible-looking DIFFERENT kernel, and
``test_kpool_hadamard.py`` carries a case whose only job is to tell the two apart.

``ape`` is an ADDITIVE PER-SLOT bias applied INSIDE the softmax, not a projection and not a
post-pool step; the gate arrives already evaluated as ``slot_score``. The checkpoint tensors
``indexer.index_kpool_compress_ape`` and ``...index_kpool_compress_gate`` are what PRODUCE these
two inputs upstream of this kernel, and that production belongs to the indexer forward
(``inc-glm53f-051``), not here.

The rotation is the reference's **7-stage FWHT butterfly** (``kpool_compress.py:27-45``) with
``(groups, stride)`` running ``(64,1) (32,2) (16,4) (8,8) (4,16) (2,32) (1,64)``, followed by a
SINGLE multiply by ``0.08838834764831845``, whose in-source comment is ``# 1/sqrt(128)``. It is an
in-register butterfly and deliberately NOT a 128x128 matmul: no transform matrix is built, stored
or multiplied anywhere in this file.

What this kernel deliberately does NOT do
-----------------------------------------
Four things the upstream reference does in the same breath are OWNED ELSEWHERE, and leaving them
out is a design decision recorded in the plan rather than an omission:

* **fp8 quantisation and the ue8m0 scale** (``kpool_compress.py:112-125``). This kernel returns
  bf16 and no scale. ``inc-glm53f-053``'s adapter owns that half.
* **the cache write** at ``loc``. Nothing here touches a KV cache.
* **pool formation, the sliding window, ``write_mask`` and slot mapping**
  (``sparse_attn_indexer_kpool.py:54-99``). This kernel is handed COMPLETE pools; the reference
  asserts the same shape at ``kpool_compress.py:281-285``. ``inc-glm53f-051`` owns the integration.
* **the raw tail cache** for a request's incomplete trailing pool
  (``kpool_compress.py:411``). ``inc-glm53f-049`` owns it.

Consequently this kernel NEVER SEES A PARTIAL POOL, which is why its declared case set varies
``n_pools`` and not the token count.

Why ``pool_size`` and ``n_pools`` are python ints
-------------------------------------------------
Both are trace-time constants. ``pool_size`` selects the number of unrolled slot loads and
``n_pools`` the tile count, so a tensor would force a data-dependent trace. The fixture's
``index_kpool`` is ``4`` (``test/vllm_neuron/model/glm5_next/fixtures/hf-config.json``), and the
compiled graph is specialised on the exact ``(n_pools, pool_size, head_dim, dtype)`` tuple: one
trace per distinct tuple, not one per bucket.
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

from vllm_neuron.utils.neuron_utils import can_run_kernel

logger = logging.getLogger(__name__)

INDEX_HEAD_DIM = 128
"""The indexer head dimension. The Hadamard path is a 128-point transform and nothing else."""

DEFAULT_POOL_SIZE = 4
"""``index_kpool`` on the target checkpoint's config. Not a hardcoded limit -- see ``_validate``."""

HADAMARD_STAGES: tuple[tuple[int, int], ...] = (
    (64, 1),
    (32, 2),
    (16, 4),
    (8, 8),
    (4, 16),
    (2, 32),
    (1, 64),
)
"""``(groups, stride)`` per butterfly stage, in the origin's order (``kpool_compress.py:38-44``).

``groups * 2 * stride == 128`` for every entry, and the strides are ``2**0 .. 2**6``. Kept as data
rather than as seven call sites so that the sequence can be asserted by the test as a sequence.
"""

HADAMARD_STRIDES: tuple[int, ...] = tuple(stride for _groups, stride in HADAMARD_STAGES)
"""Just the strides, for the device loop to walk.

THE DEVICE LOOP CANNOT WALK ``HADAMARD_STAGES`` DIRECTLY. The NKI front end requires a ``for``
target that is a single variable and refuses one that unpacks a tuple, which
``for _groups, stride in HADAMARD_STAGES:`` does. It is a lowering-time refusal, so the simulator
never sees it and only a compile does::

    error: expecting simple variable
        for _groups, stride in HADAMARD_STAGES:
            ^

Derived here rather than retyped as seven literals so the strides cannot drift from
``HADAMARD_STAGES``, which the test asserts as a whole sequence. This comprehension runs on the host
at import, so NKI never traces it.
"""

HADAMARD_SCALE = 0.08838834764831845
"""``1 / sqrt(128)``, copied as a LITERAL from ``kpool_compress.py:45``.

Copied rather than computed so the shipped constant is bit-identical to the origin's.

**IT IS THE CORRECTLY ROUNDED VALUE, AND NOT EVERY SPELLING GIVES IT.** ``128 ** -0.5``,
``math.sqrt(1/128)``, ``2 ** -3.5`` and ``math.sqrt(2)/16`` all produce this exact double;
``1.0 / math.sqrt(128)`` produces one ULP LOWER (``0x1.6a09e667f3bccp-4`` against this value's
``0x1.6a09e667f3bcdp-4``), because the division rounds down. The difference is 1.4e-17 and cannot
move any tolerance here, but it is written down because asserting the division spelling would fail a
correct kernel -- which is what happened when this module's test was first drafted.
"""

_SUPPORTED_DTYPES = (torch.bfloat16,)
"""``slot_k`` dtypes that take the NKI route. bf16 is the indexer path's dtype (as ``-045``/``-046``)."""


class KpoolHadamardError(ValueError):
    """A malformed call: wrong rank, mismatched shapes, or a pool size that does not divide."""


@dataclass
class _KpoolHadamardDispatchCounters:
    """Route-predicate counters for this module, form R-1 (``design/increment-plan.md`` D13).

    ONE ``nki_dispatch`` counter for BOTH entry points -- the fused kernel and the stage-alone
    rotation -- because the route predicate counts dispatches THROUGH THIS MODULE'S SEAM and the
    declared total is 5 over the declared case set: 4 fused plus 1 stage-alone. Two counters would
    make that total un-readable from one place.
    """

    nki_dispatch: int = 0
    torch_fallback: int = 0
    last_kernel: tuple[str, str] | None = None


_COUNTERS = _KpoolHadamardDispatchCounters()


def reset_kpool_hadamard_dispatch_counters() -> None:
    """Zero the counters. Called at the START of each declared case (section 4b's convention)."""
    _COUNTERS.nki_dispatch = 0
    _COUNTERS.torch_fallback = 0
    _COUNTERS.last_kernel = None


def kpool_hadamard_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last reset, summed over both entry points."""
    return (_COUNTERS.nki_dispatch, _COUNTERS.torch_fallback)


def kpool_hadamard_kernel_identity() -> tuple[str, str] | None:
    """``(module, qualname)`` of the kernel a seam LAST dispatched, or ``None``.

    Derived THROUGH the seam rather than from this module's import list, so it certifies what ran
    instead of what was defined (D13.1). ``None`` before any dispatch, which is the reading that
    separates "no kernel ran" from "some kernel ran".
    """
    return _COUNTERS.last_kernel


def _kernel_identity_of(kernel) -> tuple[str, str]:
    """``(module, qualname)`` of the function a ``@nki.jit`` object actually wraps.

    MEASURED, not assumed, for the reason ``vllm_neuron/functional/dsa/ragged_pack.py:170-181``
    records at its own equivalent: ``@nki.jit`` returns an ``nki.framework.kernel.Kernel`` whose
    ``__module__`` is ``"nki.framework.kernel"`` and whose ``__qualname__`` is ``None``, so reading
    those attributes off the decorated object would record the DECORATOR's identity and certify
    nothing about which kernel was wrapped.
    """
    inner = getattr(kernel, "__wrapped__", None) or getattr(kernel, "func", None) or kernel
    return (inner.__module__, inner.__qualname__)


# ---------------------------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------------------------


def _fwht128_inplace(buf_a, buf_b, head_dim: int):
    """The 7-stage FWHT butterfly along the FREE axis. Returns whichever tile holds the result.

    Ping-pongs between two ``(rows, head_dim)`` tiles because a butterfly stage reads two slices
    and writes both of them: writing into the tile being read would let a later group in the same
    stage consume an already-rotated value. Two buffers cost one extra tile and remove the hazard
    outright, which is cheaper than reasoning about it.

    Each stage, for every block of ``2 * stride`` channels::

        out[lo] = in[lo] + in[hi]
        out[hi] = in[lo] - in[hi]

    which is exactly the origin's ``reshape -> trans -> split -> join(a+b, a-b) -> trans ->
    reshape`` (``kpool_compress.py:27-33``) written as the index arithmetic it performs. Seven
    stages is an ODD number, so the result ends up in the buffer this function returns rather than
    always in ``buf_a``; the caller must use the return value and never assume.

    THE SCALE IS NOT APPLIED HERE. The origin applies ``1/sqrt(128)`` once, after all seven stages
    (``:45``), and folding it into a stage would change the intermediate values that the
    stage-alone identity case reads.
    """
    src, dst = buf_a, buf_b
    for stride in HADAMARD_STRIDES:
        block = 2 * stride
        for start in range(0, head_dim, block):
            lo_in = src[:, start:start + stride]
            hi_in = src[:, start + stride:start + block]
            nisa.tensor_tensor(
                dst=dst[:, start:start + stride], data1=lo_in, data2=hi_in, op=nl.add
            )
            nisa.tensor_tensor(
                dst=dst[:, start + stride:start + block], data1=lo_in, data2=hi_in, op=nl.subtract
            )
        src, dst = dst, src
    return src


def _load_fp32(hbm, rows: int, head_dim: int, row_stride: int, offset: int):
    """A ``(rows, head_dim)`` fp32 tile from a 2-D HBM buffer, cast on the way in if needed.

    ``row_stride`` is in ELEMENTS, so a caller reading slot ``s`` of a flattened
    ``[n_pools * pool_size, head_dim]`` buffer passes ``pool_size * head_dim``.

    The DMA lands in a tile of the SOURCE dtype and a separate ``tensor_copy`` does the widening.
    Two reasons, and the second is the load-bearing one. First, the reference states the score is
    cast to fp32 in-kernel (``kpool_compress.py:181``) and says nothing about how, so this file does
    the conversion where it can be seen. Second, the staging is UNCONDITIONAL rather than guarded by
    a ``hbm.dtype == nl.float32`` test: a dtype comparison between a NKI tensor's dtype and a
    ``nki.language`` dtype object is a trace-time equality this file would have to be right about,
    and an always-stage form is correct for every input dtype at the cost of one extra copy when the
    source is already fp32. Cheap certainty beats a clever branch.
    """
    out = nl.ndarray((rows, head_dim), dtype=nl.float32, buffer=nl.sbuf)
    staged = nl.ndarray((rows, head_dim), dtype=hbm.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=staged, src=hbm.ap(pattern=[[row_stride, rows], [1, head_dim]], offset=offset))
    nisa.tensor_copy(dst=out, src=staged)
    return out


def _broadcast_row(hbm, rows: int, head_dim: int, row: int):
    """One row of a 2-D HBM buffer replicated across ``rows`` partitions, as an fp32 tile.

    A ZERO PARTITION STRIDE is what replicates: ``pattern=[[0, rows], ...]`` reads the same source
    row for every partition. This is the vendor's own broadcast idiom
    (``nkilib/experimental/misc/permute_a2av.py:139-150``) and the reason it is used here rather
    than ``nisa.tensor_scalar`` is recorded at ``ragged_pack.py:183-204``: when ``tensor_scalar``'s
    ``operand0`` is a tile it is a PER-PARTITION scalar and must carry one entry per partition of
    ``dst``, so a ``(1, head_dim)`` row is refused by the MLIR verifier outright.

    Stages through the source dtype unconditionally, for the reason ``_load_fp32`` records.
    """
    out = nl.ndarray((rows, head_dim), dtype=nl.float32, buffer=nl.sbuf)
    staged = nl.ndarray((rows, head_dim), dtype=hbm.dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=staged, src=hbm.ap(pattern=[[0, rows], [1, head_dim]], offset=row * head_dim)
    )
    nisa.tensor_copy(dst=out, src=staged)
    return out


# ---------------------------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------------------------


@nki.jit
def _kpool_hadamard_nki(slot_k_hbm, slot_score_hbm, ape_hbm, n_pools, pool_size):
    """Fused per-(pool, channel) softmax-weighted pooling and Hadamard-128 rotation.

    Args:
        slot_k_hbm: ``[n_pools * pool_size, head_dim]`` -- ``slot_k`` flattened so that a pool and
            a slot together address one row. bf16 on the shipped path.
        slot_score_hbm: ``[n_pools * pool_size, head_dim]`` -- the gate's per-token score,
            flattened the same way.
        ape_hbm: ``[pool_size, head_dim]`` fp32 -- the per-slot additive bias.
        n_pools: pools in the batch. A compile-time constant.
        pool_size: tokens per pool. A compile-time constant.

    Returns:
        ``[n_pools, head_dim]`` in ``slot_k_hbm``'s dtype.

    The pool axis is the PARTITION axis and the head dimension is the FREE axis, which is what
    makes the whole reduction elementwise: the softmax runs over the four SLOT TILES, so each of
    its steps is a plain ``tensor_tensor`` between two ``(rows, head_dim)`` tiles and no
    cross-partition or cross-free reduction is needed anywhere. The final tile is narrowed rather
    than masked when ``n_pools`` is not a multiple of the partition maximum, so no masked lane can
    contribute to a result -- the same choice ``ragged_pack.py:337-341`` records.
    """
    head_dim = slot_k_hbm.shape[1]
    out_hbm = nl.ndarray((n_pools, head_dim), dtype=slot_k_hbm.dtype, buffer=nl.shared_hbm)
    pmax = nl.tile_size.pmax
    row_stride = pool_size * head_dim

    for t in range((n_pools + pmax - 1) // pmax):
        rows = min(pmax, n_pools - t * pmax)
        base = t * pmax * row_stride

        # --- Pass 1: per-(pool, channel) max of slot_score + ape, for softmax stability. ---
        # The sums are KEPT rather than recomputed in pass 2. The reference recomputes them
        # (kpool_compress.py:176-185) because a Triton program holds one pool; here a tile holds up
        # to 128 pools and keeping four fp32 tiles is cheaper than four more strided DMA reads.
        totals = []
        running_max = nl.ndarray((rows, head_dim), dtype=nl.float32, buffer=nl.sbuf)
        for slot in range(pool_size):
            score = _load_fp32(slot_score_hbm, rows, head_dim, row_stride, base + slot * head_dim)
            bias = _broadcast_row(ape_hbm, rows, head_dim, slot)
            total = nl.ndarray((rows, head_dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=total, data1=score, data2=bias, op=nl.add)
            totals.append(total)
            if slot == 0:
                nisa.tensor_copy(dst=running_max, src=total)
            else:
                nisa.tensor_tensor(
                    dst=running_max, data1=running_max, data2=total, op=nl.maximum
                )

        # --- Pass 2: softmax-weighted sum of slot_k. ---
        acc = nl.ndarray((rows, head_dim), dtype=nl.float32, buffer=nl.sbuf)
        denom = nl.ndarray((rows, head_dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=acc, value=0.0)
        nisa.memset(dst=denom, value=0.0)
        for slot in range(pool_size):
            shifted = nl.ndarray((rows, head_dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=shifted, data1=totals[slot], data2=running_max, op=nl.subtract)
            weight = nl.ndarray((rows, head_dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=weight, op=nl.exp, data=shifted)
            nisa.tensor_tensor(dst=denom, data1=denom, data2=weight, op=nl.add)
            key = _load_fp32(slot_k_hbm, rows, head_dim, row_stride, base + slot * head_dim)
            weighted = nl.ndarray((rows, head_dim), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=weighted, data1=weight, data2=key, op=nl.multiply)
            nisa.tensor_tensor(dst=acc, data1=acc, data2=weighted, op=nl.add)

        # Reciprocal-then-multiply, not a divide: one op per tile either way, and the reciprocal
        # is the form the ISA exposes directly.
        inv = nl.ndarray((rows, head_dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=inv, data=denom)
        pooled = nl.ndarray((rows, head_dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=pooled, data1=acc, data2=inv, op=nl.multiply)

        # --- Rotate, scale once, cast, store. ---
        scratch = nl.ndarray((rows, head_dim), dtype=nl.float32, buffer=nl.sbuf)
        rotated = _fwht128_inplace(pooled, scratch, head_dim)
        scaled = nl.ndarray((rows, head_dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=scaled, data=rotated, op0=nl.multiply, operand0=HADAMARD_SCALE)
        result = nl.ndarray((rows, head_dim), dtype=slot_k_hbm.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=result, src=scaled)
        nl.store(
            out_hbm.ap(pattern=[[head_dim, rows], [1, head_dim]], offset=t * pmax * head_dim),
            value=result,
        )
    return out_hbm


@nki.jit
def _hadamard128_nki(x_hbm, n_rows):
    """The rotation stage ALONE: ``FWHT_128(row) * (1 / sqrt(128))`` for every row.

    Args:
        x_hbm: ``[n_rows, head_dim]`` -- the rows to rotate.
        n_rows: rows in the batch. A compile-time constant.

    Returns:
        ``[n_rows, head_dim]`` in ``x_hbm``'s dtype.

    This exists so the transform can be read on its own, which is what the declared identity case
    does: on ``I_128`` the output IS ``H_128 / sqrt(128)``, so one case reads all seven stages and
    the final scale at once. It shares ``_fwht128_inplace`` with the fused kernel, so the identity
    case certifies the same butterfly code the fused path runs -- a separate transcription here
    would let the two drift and would make the identity reading evidence about nothing.
    """
    head_dim = x_hbm.shape[1]
    out_hbm = nl.ndarray((n_rows, head_dim), dtype=x_hbm.dtype, buffer=nl.shared_hbm)
    pmax = nl.tile_size.pmax
    for t in range((n_rows + pmax - 1) // pmax):
        rows = min(pmax, n_rows - t * pmax)
        src = _load_fp32(x_hbm, rows, head_dim, head_dim, t * pmax * head_dim)
        scratch = nl.ndarray((rows, head_dim), dtype=nl.float32, buffer=nl.sbuf)
        rotated = _fwht128_inplace(src, scratch, head_dim)
        scaled = nl.ndarray((rows, head_dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=scaled, data=rotated, op0=nl.multiply, operand0=HADAMARD_SCALE)
        result = nl.ndarray((rows, head_dim), dtype=x_hbm.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=result, src=scaled)
        nl.store(
            out_hbm.ap(pattern=[[head_dim, rows], [1, head_dim]], offset=t * pmax * head_dim),
            value=result,
        )
    return out_hbm


# ---------------------------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------------------------


@torch._dynamo.assume_constant_result
def _record_nki_dispatch(entry: str, n_pools: int, pool_size: int, head_dim: int) -> None:
    """Record WHICH kernel a seam dispatched, and log it, OFF the compiled graph.

    THE TEMPLATE THIS FOLLOWS IS LANDED AND MEASURED, not invented here:
    ``vllm_neuron/functional/dsa/ragged_pack.py:471-517``, itself following
    ``vllm_neuron/functional/dsa/paged_gather.py:248-288``.

    ARGUMENT DISCIPLINE, WHICH IS THE WHOLE POINT: a folded helper takes ints, strings and dtypes
    ONLY, never an object. Dynamo runs a folded call at trace time and first converts every
    non-tensor argument into a python constant. An ``@nki.jit`` kernel is an
    ``nki.framework.kernel.Kernel``, a FROZEN DATACLASS, and Dynamo refuses to reconstruct one --
    ``NotImplementedError: currently can't reconstruct arbitrary frozen dataclass instances``. So
    neither kernel is a parameter here: the entry point arrives as a ``str`` and the kernel is read
    as a module global, the same object the call site hands to ``wrap_nki`` on the next line.

    ONE HELPER FOR BOTH ENTRY POINTS, matching this campaign's one-fold-per-module rule.

    D13.1 STILL HOLDS: this body runs only when a dispatch branch runs, so the recorded identity is
    derived by TAKING the branch rather than read off an import.
    """
    kernel = _kpool_hadamard_nki if entry == "fused" else _hadamard128_nki
    _COUNTERS.last_kernel = _kernel_identity_of(kernel)
    logger.info(
        "[dsa-kpool-hadamard] kernel=nki entry=%s n_pools=%d pool_size=%d head_dim=%d",
        entry,
        n_pools,
        pool_size,
        head_dim,
    )


def _validate(slot_k: Tensor, slot_score: Tensor, ape: Tensor) -> tuple[int, int, int]:
    """Host-side shape and dtype validation. Returns ``(n_pools, pool_size, head_dim)``.

    Reads only ``.shape`` and ``.dtype``, never a tensor VALUE, so nothing here forces a
    device-to-host synchronisation or a data-dependent trace.
    """
    if slot_k.ndim != 3:
        raise KpoolHadamardError(
            f"slot_k must be 3-D [n_pools, pool_size, head_dim]; got shape {tuple(slot_k.shape)}"
        )
    if tuple(slot_score.shape) != tuple(slot_k.shape):
        raise KpoolHadamardError(
            f"slot_score must match slot_k; got {tuple(slot_score.shape)} against "
            f"{tuple(slot_k.shape)}"
        )
    n_pools, pool_size, head_dim = (int(d) for d in slot_k.shape)
    if ape.ndim != 2 or tuple(ape.shape) != (pool_size, head_dim):
        raise KpoolHadamardError(
            f"ape must be [pool_size, head_dim] = {(pool_size, head_dim)}; got "
            f"{tuple(ape.shape)}"
        )
    if n_pools <= 0:
        raise KpoolHadamardError(f"n_pools must be positive; got {n_pools}")
    if pool_size <= 0:
        raise KpoolHadamardError(f"pool_size must be positive; got {pool_size}")
    if head_dim != INDEX_HEAD_DIM:
        raise KpoolHadamardError(
            f"the Hadamard path is a {INDEX_HEAD_DIM}-point transform; got head_dim {head_dim}"
        )
    return n_pools, pool_size, head_dim


def can_run_dsa_kpool_hadamard(slot_k: Tensor, slot_score: Tensor, ape: Tensor) -> bool:
    """Whether the fused NKI kernel serves this call. ``False`` sends it to the torch oracle."""
    if not can_run_kernel():
        return False
    if slot_k.dtype not in _SUPPORTED_DTYPES:
        return False
    if slot_k.ndim != 3 or tuple(slot_score.shape) != tuple(slot_k.shape):
        return False
    if int(slot_k.shape[2]) != INDEX_HEAD_DIM:
        return False
    return ape.ndim == 2 and tuple(ape.shape) == tuple(slot_k.shape[1:])


def can_run_dsa_hadamard128(x: Tensor) -> bool:
    """Whether the stage-alone NKI kernel serves this call."""
    if not can_run_kernel():
        return False
    return x.ndim == 2 and int(x.shape[1]) == INDEX_HEAD_DIM


def dsa_kpool_hadamard(slot_k: Tensor, slot_score: Tensor, ape: Tensor) -> Tensor:
    """THE COUNTED SEAM, fused entry. Pool ``pool_size`` keys into one and rotate it.

    Args:
        slot_k: ``[n_pools, pool_size, head_dim]`` -- raw per-token indexer keys, one complete pool
            per row of the first axis. ``bfloat16`` takes the NKI route; any other dtype is served
            by the torch oracle.
        slot_score: ``[n_pools, pool_size, head_dim]`` -- the gate's per-token score.
        ape: ``[pool_size, head_dim]`` -- the per-slot additive position bias.

    Returns:
        ``[n_pools, head_dim]`` in ``slot_k``'s dtype: the pooled, rotated key per pool. No fp8
        output and no scale -- ``inc-glm53f-053``'s adapter owns that half.

    Raises:
        KpoolHadamardError: for a malformed call -- a non-3D ``slot_k``, a ``slot_score`` that does
            not match it, an ``ape`` of the wrong shape, or a head dimension that is not 128.
    """
    n_pools, pool_size, head_dim = _validate(slot_k, slot_score, ape)

    if not can_run_dsa_kpool_hadamard(slot_k, slot_score, ape):
        return _dsa_kpool_hadamard_torch(slot_k, slot_score, ape)

    flat_k = slot_k.reshape(n_pools * pool_size, head_dim).contiguous()
    flat_score = slot_score.reshape(n_pools * pool_size, head_dim).contiguous()

    _COUNTERS.nki_dispatch += 1
    # The log and the identity read are FOLDED off the traced graph. The helper takes a str and
    # ints ONLY and reads the kernel as a module global: passing the kernel object is the measured
    # defect that pattern exists to avoid. The counter increment stays -- a plain int attribute
    # store is a recorded side effect, not a host call.
    _record_nki_dispatch("fused", n_pools, pool_size, head_dim)
    return wrap_nki(_kpool_hadamard_nki)(
        flat_k, flat_score, ape.contiguous(), n_pools, pool_size
    )


def dsa_hadamard128(x: Tensor) -> Tensor:
    """THE COUNTED SEAM, stage-alone entry. ``FWHT_128(row) / sqrt(128)`` for every row.

    Args:
        x: ``[n_rows, head_dim]`` with ``head_dim == 128``.

    Returns:
        ``[n_rows, head_dim]`` in ``x``'s dtype.

    Raises:
        KpoolHadamardError: if ``x`` is not 2-D or its head dimension is not 128.
    """
    if x.ndim != 2:
        raise KpoolHadamardError(f"x must be 2-D [n_rows, head_dim]; got shape {tuple(x.shape)}")
    n_rows, head_dim = int(x.shape[0]), int(x.shape[1])
    if head_dim != INDEX_HEAD_DIM:
        raise KpoolHadamardError(
            f"the Hadamard path is a {INDEX_HEAD_DIM}-point transform; got head_dim {head_dim}"
        )
    if n_rows <= 0:
        raise KpoolHadamardError(f"n_rows must be positive; got {n_rows}")

    if not can_run_dsa_hadamard128(x):
        return _dsa_hadamard128_torch(x)

    _COUNTERS.nki_dispatch += 1
    _record_nki_dispatch("stage", n_rows, 1, head_dim)
    return wrap_nki(_hadamard128_nki)(x.contiguous(), n_rows)


# ---------------------------------------------------------------------------------------------
# Torch oracles -- the CPU reference, never the shipped implementation (P13)
# ---------------------------------------------------------------------------------------------


def hadamard_matrix(head_dim: int = INDEX_HEAD_DIM, dtype=torch.float32) -> Tensor:
    """The UNNORMALISED Sylvester ``H_n``, built by doubling. ``H @ H.T == n * I``.

    Used by the oracle and by the identity case's expectation. Built rather than transcribed so
    that no 128x128 literal has to be trusted, and asserted orthogonal by the test.
    """
    if head_dim <= 0 or head_dim & (head_dim - 1):
        raise KpoolHadamardError(f"head_dim must be a positive power of two; got {head_dim}")
    h = torch.ones((1, 1), dtype=dtype)
    while h.shape[0] < head_dim:
        h = torch.cat((torch.cat((h, h), dim=1), torch.cat((h, -h), dim=1)), dim=0)
    return h


def _dsa_kpool_hadamard_torch(slot_k: Tensor, slot_score: Tensor, ape: Tensor) -> Tensor:
    """The UNFUSED torch composition: pool, then rotate. THE ORACLE, and the fallback path.

    It transcribes the reference's order (``kpool_compress.py:164-200`` then ``:37-45``) as three
    separate steps, deliberately unfused, because the acceptance measurement is the fused kernel's
    agreement with an unfused composition. ``dim=1`` is the SLOT axis, which is what makes the
    softmax per ``(pool, channel)``; a ``dim=-1`` here would be the whole-vector softmax this
    module is not.
    """
    _COUNTERS.torch_fallback += 1
    weights = torch.softmax(slot_score.float() + ape.float().unsqueeze(0), dim=1)
    pooled = (weights * slot_k.float()).sum(dim=1)
    rotated = pooled @ hadamard_matrix(int(slot_k.shape[2])).t()
    return (rotated * HADAMARD_SCALE).to(slot_k.dtype)


def _dsa_hadamard128_torch(x: Tensor) -> Tensor:
    """The rotation alone, in torch. THE ORACLE, and the fallback path."""
    _COUNTERS.torch_fallback += 1
    rotated = x.float() @ hadamard_matrix(int(x.shape[1])).t()
    return (rotated * HADAMARD_SCALE).to(x.dtype)
