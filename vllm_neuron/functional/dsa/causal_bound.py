# SPDX-License-Identifier: Apache-2.0
"""The DSA indexer's selecting-regime causal bound -- ``inc-glm53f-103``.

WHAT THIS MODULE IS FOR, in one sentence: a query row must not select a key pool that finishes
after the row's own position, so the scores of every such pool are pushed to ``-inf`` before the
selector runs, and any selection that comes back holding ``-inf`` is replaced by the ``-1``
sentinel.

WHY THE GAP EXISTED. The landed chain ``dsa_score_gemm`` -> ``dsa_topk_select`` ->
``dsa_index_expand`` carries no per-row bound anywhere. ``inc-glm53f-046`` said so in the code it
landed -- "No masking and no ``-inf`` fill. Upstream applies its causal and window mask AFTER this
op ... the mask stage is a separate increment" (``score_gemm.py:71-72``) -- and no increment was
ever minted for that stage. ``inc-glm53f-099`` owns the SHORT regime only: below the selection
bound every candidate would be selected anyway, so it fills exact causal rows and never selects.
Above that bound the selector runs, and until this module landed it could pick a pool whose tokens
the row must not see. That breaks ``inc-glm53f-048``'s declared caller precondition -- "every
non-negative pool id satisfies 0 <= pool_ids[row, g] < seq_len[row] // pool_size"
(``index_expand.py:48``, and ``:51`` "a pool id past the row's last pool expands to token indices
past ``seq_len``").

THE RULE, AND IT IS UPSTREAM'S. Pool ``p`` holds tokens ``p * pool_size .. (p + 1) * pool_size - 1``,
so it is complete for a row of length ``causal_len`` exactly when ``(p + 1) * pool_size <=
causal_len``, which is ``p < causal_len // pool_size``::

    scores[i, p] = -inf   where (p + 1) * pool_size >  causal_len[i]
    scores[i, p] = scores[i, p]  (untouched, bit for bit)  otherwise

Upstream counts the same complete pools, in position units rather than length units:
``num_compressed = (tl.load(positions + row) + 1) // COMPRESS_RATIO``
(``vllm/models/deepseek_v4/attention.py:82`` at ``878631b6``), and again as
``topk_len = tl.minimum((pos + 1) // COMPRESS_RATIO, TOP_K)``
(``common/ops/cache_utils.py:684``, whose own comment at ``:678-680`` calls this formula the one
"both the C4A indexer and the C128A metadata builder emit"). ``causal_len`` here IS ``pos + 1``:
the landed bypass passes ``seq_lens - 1`` as the position column
(``model_fp8.py:3912``), so ``causal_len == seq_len == pos + 1`` and the two formulas are the same
formula. THE READ-FIRST COMPARISON THIS BLOCK OWES IS RECORDED, NOT ASSERTED:
``increments/readfirst-103-upstream-r2.out``, ``READFIRST_103_R2=AGREE`` 10/10.

``-inf`` IS UPSTREAM'S VALUE AT THIS STAGE, AND THE STAGE MATTERS. Upstream bounds the indexer
LOGITS with a literal ``float("-inf")`` -- ``logits = logits.masked_fill(~mask, float("-inf"))``
(``vllm/v1/attention/ops/rocm_aiter_mla_sparse.py:734``), and bounds the sparse attention score the
same way (``backends/mla/rocm_aiter_mla_sparse.py:720``). The ``-FLT_MAX`` a reader may find at
``csrc/libtorch_stable/sampler.cu:407`` is the top-k selector's OUTPUT tail padding, downstream of
selection, and is not this stage; round 1 of this block's read-first gate compared against it by
mistake and the correction is kept beside it rather than deleted
(``readfirst-103-upstream.out``, ``EXIT=2``, superseded).

A WHOLLY-BOUNDED ROW IS LEGAL AND ITS MEANING IS ALREADY FIXED. A row with ``causal_len <
pool_size`` has zero complete pools, so every column is ``-inf``. The consumer settles what that
means: ``mla_sparse_attention``'s oracle records "A wholly-sentinel row is all -inf, and softmax of
that is NaN rather than zero -- so the zeros the kernels produce for it are written here too"
(``mla_sparse.py:1494-1496``). Nothing here masks, clamps or compacts; this module only writes.

WHY THIS IS KERNEL-CLASS AND LANDS IN NKI (P13). Both outputs are device tensors produced for the
device kernels that consume them, which is ``inc-glm53f-048``'s precedent exactly. A torch mask on
the host would be a fallback for kernel-class work AND one host round trip per step, on the
per-forward path.

CONSTRUCTS, AND THE SCREENING FOR EACH. Every construct below has a landed fork call site except
one, which is named as such.

  * NOTHING IS DIVIDED. ``nl.divide`` is silently wrong on int32 and ``nl.right_shift`` refuses as
    the second op of a chain (``index_expand.py:108-114``), so the bound is rearranged into a
    multiply and a compare: ``(p + 1) * pool_size > causal_len`` rather than
    ``p >= causal_len // pool_size``. The two are the same predicate over integers.
  * ``nl.arange``, ``mgrid``, ``nl.iota`` and ``nl.affine_select`` DO NOT EXIST on this image
    (``index_expand.py:115``). The column ramp is ``nisa.iota`` with ``channel_multiplier=0``, the
    landed form at ``moe/router.py:1286``, and it is float32 because both landed
    ``channel_multiplier=0`` sites use a float32 destination.
  * A PER-ROW VALUE REACHES ``tensor_scalar`` AS A ``(rows, 1)`` COLUMN OPERAND and is broadcast
    along the free axis (``causal_fill.py:220-224`` is the landed call, and its dtype note records
    why the operand tile must be float32: the ISA requires it and the MLIR verifier refuses an
    int32 operand there). A ``(1, N)`` row operand is refused in that position -- the ``-045``
    finding at ``score_gemm.py:84-88``.
  * THE COMPARE PRODUCES AN INTEGER PREDICATE, which is the landed shape at
    ``moe/topk_reduce.py:351-355``: ``tensor_scalar`` with ``op0=nl.greater`` into an integer
    destination. ``nl.less`` is NOT used, because it is screened only for ``tensor_tensor``; the
    arithmetic is arranged so ``greater`` is the only comparison needed.
  * THE MASK IS APPLIED BY ``nisa.tensor_copy_predicated``, the landed select at
    ``moe/topk_reduce.py:309`` and ``:360``. It writes ``src`` where the predicate is set and LEAVES
    ``dst`` ALONE elsewhere -- which is how the untouched columns stay bit-identical BY
    CONSTRUCTION rather than by an arithmetic identity. An additive or multiplicative mask cannot
    make that claim: ``x + 0.0`` turns ``-0.0`` into ``+0.0``, and ``0 * -inf`` is NaN.
  * THE ONE CONSTRUCT WITH NO FORK-AUTHORED SCREENING: ``nisa.memset`` with a NON-FINITE value.
    ``memset`` itself has 37 landed call sites, but every landed value is finite (``0.0`` eight
    times, ``0`` three times, ``1.0``, ``-2``, and ``SENTINEL_INDEX`` at ``mla_sparse.py:236``).
    The nearest on-image precedent for an ``inf`` immediate is vendored rather than fork-authored:
    ``nisa.nc_match_replace8(imm=float("-inf"))``
    (``vendored_kernels/rotational_topk/rotational_topk_utils.py:1065``, ``:1109``, ``:1116``).
    THE NKI SIMULATOR DOES NOT RUN THE MLIR VERIFIER STAGE -- ``causal_fill.py:189-196`` records
    that trap costing ``-099`` four green items -- so a green Tier N run does not clear this one.
    The capture leg is what clears it, and this comment is here so the next reader knows which line
    to look at first if capture refuses.
"""

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

SENTINEL = -1
"""What a selection that reaches no valid pool holds. ``inc-glm53f-098``'s value, written here and
masked there. Upstream writes the same ``-1`` for a candidate-less slot (``sampler.cu:405``)."""

NEG_INF = float("-inf")
"""The bounded-score value. Upstream's at this stage (``rocm_aiter_mla_sparse.py:734``)."""

_FLT_MAX = 3.4028234663852886e38
"""Largest finite float32. Used ONLY as the threshold that separates ``+inf`` from every finite
score in :func:`_causal_sentinel_nki`, never as a fill value."""

_SCORE_DTYPES = (torch.float32,)
"""Score dtypes that take the NKI route. The bound compares against a length and writes a float
sentinel; fp32 is what ``dsa_score_gemm`` hands the selector on this path."""

_INDEX_DTYPES = (torch.int32,)
"""Index dtypes that take the NKI route. ``dsa_index_expand`` admits int32, so the sentinel writer
keeps the selector's output in the dtype its consumer reads."""


class DsaCausalBoundError(ValueError):
    """A malformed call: a ``pool_size`` that is not a power of two, a ``causal_len`` of the wrong
    dtype, or a ``causal_len`` whose row count does not match ``scores``."""


@dataclass
class _DispatchCounters:
    """Route-predicate counters for ONE entry point, form R-1 (``design/increment-plan.md`` D13).

    ONE INSTANCE PER ENTRY POINT, WITH ITS OWN ACCESSOR AND ITS OWN RESET, which is what this
    block's surface line asks for: "``_COUNTERS.nki_dispatch``/``torch_fallback`` accessors and
    reset per entry point". A single summed pair would make the declared reading "exactly 1 per
    call, per entry point" unreadable -- ``(2, 0)`` after a bound call and a sentinel call cannot be
    told apart from two bound calls.

    ``torch_fallback`` is a declared 0 whose firing control is in the acceptance: with
    ``can_run_kernel`` forced False one call moves it to 1, so the zero is a reading and not
    decoration (D1.5).
    """

    nki_dispatch: int = 0
    torch_fallback: int = 0
    last_kernel: tuple[str, str] | None = None


_BOUND = _DispatchCounters()
_SENTINEL_COUNTERS = _DispatchCounters()


def reset_causal_bound_dispatch_counters() -> None:
    """Zero the BOUND entry point's counters. Called at the START of each declared case."""
    _BOUND.nki_dispatch = 0
    _BOUND.torch_fallback = 0
    _BOUND.last_kernel = None


def causal_bound_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` for :func:`dsa_causal_bound` since its last reset."""
    return (_BOUND.nki_dispatch, _BOUND.torch_fallback)


def causal_bound_kernel_identity() -> tuple[str, str] | None:
    """``(module, qualname)`` of the kernel the BOUND seam LAST dispatched, or ``None``.

    Derived THROUGH the seam rather than from this module's import list, so it certifies what ran
    instead of what was defined (D13.1). ``None`` before any dispatch, which is the reading that
    separates "no kernel ran" from "some kernel ran".
    """
    return _BOUND.last_kernel


def reset_causal_sentinel_dispatch_counters() -> None:
    """Zero the SENTINEL entry point's counters."""
    _SENTINEL_COUNTERS.nki_dispatch = 0
    _SENTINEL_COUNTERS.torch_fallback = 0
    _SENTINEL_COUNTERS.last_kernel = None


def causal_sentinel_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` for :func:`dsa_causal_sentinel` since its last reset."""
    return (_SENTINEL_COUNTERS.nki_dispatch, _SENTINEL_COUNTERS.torch_fallback)


def causal_sentinel_kernel_identity() -> tuple[str, str] | None:
    """``(module, qualname)`` of the kernel the SENTINEL seam LAST dispatched, or ``None``."""
    return _SENTINEL_COUNTERS.last_kernel


def _kernel_identity_of(kernel) -> tuple[str, str]:
    """``(module, qualname)`` of the function a ``@nki.jit`` object actually wraps.

    Measured, not assumed, for the reason ``ragged_pack.py:170-181`` records: reading those
    attributes off the decorated object records the DECORATOR's identity and certifies nothing.
    """
    inner = getattr(kernel, "__wrapped__", None) or getattr(kernel, "func", None) or kernel
    return (inner.__module__, inner.__qualname__)


# ---------------------------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------------------------


@nki.jit
def _causal_bound_nki(scores_hbm, causal_len_hbm, pool_size):
    """``-inf`` at every pool column the row's own length does not complete.

    Args:
        scores_hbm: ``[rows, width]`` float32 -- one score per candidate pool per query row.
        causal_len_hbm: ``[rows, 1]`` int32 -- each row's own causal length, ALREADY a column,
            because it reaches ``tensor_scalar`` as a per-row scalar operand.
        pool_size: python int, tokens per pool. A trace-time constant; ``wrap_nki`` passes an int
            through unchanged (``decode_tail_update.py:544-546``).

    Returns:
        ``[rows, width]`` float32. Column ``p`` of row ``i`` holds :data:`NEG_INF` when
        ``(p + 1) * pool_size > causal_len[i]`` and row ``i``'s original score, bit for bit,
        otherwise.

    THE KEPT COLUMNS ARE NEVER WRITTEN. The scores are copied into SBUF once and the bounded
    columns are overwritten in place by one predicated copy, so a kept column carries the loaded
    bits unchanged -- no add of 0.0, no multiply by 1.0, and therefore no ``-0.0`` to ``+0.0``
    rewrite and no ``0 * -inf`` NaN.
    """
    rows = scores_hbm.shape[0]
    width = scores_hbm.shape[1]

    out = nl.ndarray((rows, width), dtype=nl.float32, buffer=nl.shared_hbm)

    # The scores, loaded once. This tile IS the result: the mask writes into it.
    scores_sb = nl.ndarray((rows, width), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=scores_sb, src=nl.load(scores_hbm))

    # The per-row length as a float32 COLUMN operand. float32 because the ISA requires a
    # `tensor_scalar` operand tile to be float32 and the MLIR verifier refuses int32 there -- the
    # reading `-099` paid for, recorded at `causal_fill.py:189-196`. Exact: a causal length is a
    # whole number far below 2**24.
    clen = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=clen, src=nl.load(causal_len_hbm))

    # `-causal_len`, so the only tile-operand call below can be an ADD, which is the screened form
    # (`causal_fill.py:224`). A `subtract` with a tile operand has no landed call site.
    nclen = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=nclen, data=clen, op0=nl.multiply, operand0=-1.0)

    # The column ramp `p`, the same 0..width-1 on every partition.
    ramp = nl.ndarray((rows, width), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(dst=ramp, pattern=[[1, width]], offset=0, channel_multiplier=0)

    # `(p + 1) * pool_size`, the first token index past pool `p`, as one two-scalar chain.
    end = nl.ndarray((rows, width), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=end, data=ramp,
                       op0=nl.multiply, operand0=float(pool_size),
                       op1=nl.add, operand1=float(pool_size))

    # `(p + 1) * pool_size - causal_len[i]`, one tile operand broadcast along the free axis.
    room = nl.ndarray((rows, width), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=room, data=end, op0=nl.add, operand0=nclen)

    # 1 exactly where the pool is INCOMPLETE for this row, which is where the bound applies.
    # `greater` into an integer destination is the landed compare (`moe/topk_reduce.py:351-355`).
    bounded = nl.ndarray((rows, width), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=bounded, data=room, op0=nl.greater, operand0=0.0)

    # The fill source. THIS memset is the one construct in this module with no fork-authored
    # screening; see the module docstring's last bullet.
    fill = nl.ndarray((rows, width), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=fill, value=NEG_INF)

    # The mask. `scores_sb` is left alone wherever `bounded` is 0.
    nisa.tensor_copy_predicated(src=fill, predicate=bounded, dst=scores_sb)

    nl.store(out, value=scores_sb)
    return out


@nki.jit
def _causal_sentinel_nki(values_hbm, indices_hbm):
    """``-1`` at every selection whose value came back ``-inf``.

    Args:
        values_hbm: ``[rows, k]`` float32 -- the selector's returned values.
        indices_hbm: ``[rows, k]`` int32 -- the selector's returned pool ids.

    Returns:
        ``[rows, k]`` int32. :data:`SENTINEL` where the paired value is ``-inf``, and the original
        index, bit for bit, everywhere else.

    HOW ``-inf`` IS DETECTED WITHOUT A ``less``. Negating turns ``-inf`` into ``+inf`` and every
    finite score into a finite number, so one ``greater`` against the largest finite float32
    separates them: ``+inf > FLT_MAX`` is true and no finite value can be. That keeps the compare
    in the one form ``tensor_scalar`` has a landed call site for.
    """
    rows = values_hbm.shape[0]
    k = values_hbm.shape[1]

    out = nl.ndarray((rows, k), dtype=nl.int32, buffer=nl.shared_hbm)

    # The indices, loaded once. This tile IS the result: the sentinel writes into it.
    idx_sb = nl.ndarray((rows, k), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=idx_sb, src=nl.load(indices_hbm))

    vals = nl.ndarray((rows, k), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=vals, src=nl.load(values_hbm))

    # `-value`: `+inf` exactly where the bound fired, finite everywhere else.
    negv = nl.ndarray((rows, k), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=negv, data=vals, op0=nl.multiply, operand0=-1.0)

    masked = nl.ndarray((rows, k), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=masked, data=negv, op0=nl.greater, operand0=_FLT_MAX)

    # `memset` with the int32 sentinel is the landed form at `mla_sparse.py:236`.
    fill = nl.ndarray((rows, k), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=fill, value=SENTINEL)

    nisa.tensor_copy_predicated(src=fill, predicate=masked, dst=idx_sb)

    nl.store(out, value=idx_sb)
    return out


# ---------------------------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------------------------


@torch._dynamo.assume_constant_result
def _record_bound_dispatch(rows: int, width: int) -> None:
    """Record which kernel the bound seam dispatched, and log it, OFF the compiled graph.

    The template is landed and measured: ``kpool_hadamard.py:428-457`` by way of
    ``score_gemm.py:306-330``. A folded helper takes ints only -- Dynamo runs it at trace time and
    refuses to reconstruct an ``@nki.jit`` object -- so the kernel is read as a module global.
    """
    _BOUND.last_kernel = _kernel_identity_of(_causal_bound_nki)
    logger.info("[dsa-causal-bound] kernel=nki rows=%d width=%d", rows, width)


@torch._dynamo.assume_constant_result
def _record_sentinel_dispatch(rows: int, k: int) -> None:
    """Record which kernel the sentinel seam dispatched, and log it, OFF the compiled graph."""
    _SENTINEL_COUNTERS.last_kernel = _kernel_identity_of(_causal_sentinel_nki)
    logger.info("[dsa-causal-sentinel] kernel=nki rows=%d k=%d", rows, k)


def _validate_bound(scores: Tensor, causal_len: Tensor, pool_size: int) -> int:
    """Host-side validation for the bound. Returns ``rows``.

    Reads only ``.shape`` and ``.dtype``, never a VALUE, so nothing here forces a device-to-host
    synchronisation or a data-dependent trace. All three refusals RAISE rather than declining to
    the oracle: each is a caller bug, and serving one through the oracle would hide it behind a
    correct-looking answer.
    """
    if not isinstance(pool_size, int) or isinstance(pool_size, bool):
        raise DsaCausalBoundError(
            f"pool_size must be a python int, because it is a trace-time constant; got "
            f"{type(pool_size).__name__} {pool_size!r}"
        )
    if pool_size < 1 or (pool_size & (pool_size - 1)) != 0:
        raise DsaCausalBoundError(
            f"pool_size must be a power of two; got pool_size={pool_size}. The pooling geometry "
            f"this bound shares with dsa_index_expand and dsa_kpool_hadamard is a power of two "
            f"throughout, so any other value is a caller error rather than a shape this kernel "
            f"declines to serve"
        )
    if scores.ndim != 2:
        raise DsaCausalBoundError(
            f"scores must be 2-D [rows, width], one score per candidate pool per query row; got "
            f"shape {tuple(scores.shape)}"
        )
    if causal_len.dtype not in _INDEX_DTYPES:
        raise DsaCausalBoundError(
            f"causal_len must be int32, the dtype the indexer carries and the consumer reads; got "
            f"{causal_len.dtype}"
        )
    rows = int(scores.shape[0])
    if int(causal_len.shape[0]) != rows:
        raise DsaCausalBoundError(
            f"causal_len must carry one length per score row; got {int(causal_len.shape[0])} "
            f"lengths for {rows} score rows"
        )
    return rows


def can_run_dsa_causal_bound(scores: Tensor, causal_len: Tensor, pool_size: int) -> bool:
    """Whether the NKI kernel serves this bound call. ``False`` sends it to the torch oracle.

    Narrow on purpose. Every malformed call is already refused by :func:`_validate_bound`, so the
    only thing left is whether NKI is available at all -- which is what makes ``can_run_kernel``
    the firing control the acceptance uses to move the fallback zero off 0.
    """
    if not can_run_kernel():
        return False
    return scores.dtype in _SCORE_DTYPES and causal_len.dtype in _INDEX_DTYPES


def dsa_causal_bound(scores: Tensor, causal_len: Tensor, pool_size: int) -> Tensor:
    """THE COUNTED SEAM. ``-inf`` at every pool a query row's own length does not complete.

    Args:
        scores: ``[rows, width]`` float32, one score per candidate pool per query row, as
            ``dsa_score_gemm`` returns them.
        causal_len: ``[rows, 1]`` or ``[rows]`` int32 -- each row's own causal length. This is the
            SAME per-row length column ``dsa_index_expand``'s call already receives; it is not
            recomputed here and no second spelling of it exists.
        pool_size: tokens per pool, a power of two.

    Returns:
        ``[rows, width]`` float32, bounded as :func:`_causal_bound_nki` describes.

    Raises:
        DsaCausalBoundError: for a non-int or non-power-of-two ``pool_size``, a ``scores`` that is
            not 2-D, a ``causal_len`` that is not int32, or a row-count mismatch.
    """
    rows = _validate_bound(scores, causal_len, pool_size)

    if not can_run_dsa_causal_bound(scores, causal_len, pool_size):
        _BOUND.torch_fallback += 1
        return dsa_causal_bound_torch_oracle(scores, causal_len, pool_size)

    # The transport the device cannot pay for: the per-row length reaches `tensor_scalar` as a
    # COLUMN operand, so the reshape happens once here rather than per use on the device.
    clen_col = causal_len.reshape(rows, 1).contiguous()

    _BOUND.nki_dispatch += 1
    _record_bound_dispatch(rows, int(scores.shape[1]))
    return wrap_nki(_causal_bound_nki)(scores.contiguous(), clen_col, pool_size)


def _validate_sentinel(values: Tensor, indices: Tensor) -> int:
    """Host-side validation for the sentinel writer. Returns ``rows``."""
    if values.ndim != 2 or indices.ndim != 2:
        raise DsaCausalBoundError(
            f"values and indices must both be 2-D [rows, k]; got {tuple(values.shape)} and "
            f"{tuple(indices.shape)}"
        )
    if tuple(values.shape) != tuple(indices.shape):
        raise DsaCausalBoundError(
            f"values and indices must be the same shape, one value per selected index; got "
            f"{tuple(values.shape)} and {tuple(indices.shape)}"
        )
    if indices.dtype not in _INDEX_DTYPES:
        raise DsaCausalBoundError(
            f"indices must be int32, the dtype dsa_index_expand admits; got {indices.dtype}"
        )
    return int(values.shape[0])


def can_run_dsa_causal_sentinel(values: Tensor, indices: Tensor) -> bool:
    """Whether the NKI kernel serves this sentinel call. ``False`` sends it to the torch oracle."""
    if not can_run_kernel():
        return False
    return values.dtype in _SCORE_DTYPES and indices.dtype in _INDEX_DTYPES


def dsa_causal_sentinel(values: Tensor, indices: Tensor) -> Tensor:
    """THE COUNTED SEAM. :data:`SENTINEL` at every selection whose value came back ``-inf``.

    Args:
        values: ``[rows, k]`` float32, the selector's returned values.
        indices: ``[rows, k]`` int32, the selector's returned pool ids.

    Returns:
        ``[rows, k]`` int32, sentinelised as :func:`_causal_sentinel_nki` describes.

    Raises:
        DsaCausalBoundError: for a rank or shape mismatch, or a non-int32 ``indices``.
    """
    rows = _validate_sentinel(values, indices)

    if not can_run_dsa_causal_sentinel(values, indices):
        _SENTINEL_COUNTERS.torch_fallback += 1
        return dsa_causal_sentinel_torch_oracle(values, indices)

    _SENTINEL_COUNTERS.nki_dispatch += 1
    _record_sentinel_dispatch(rows, int(values.shape[1]))
    return wrap_nki(_causal_sentinel_nki)(values.contiguous(), indices.contiguous())


# ---------------------------------------------------------------------------------------------
# Torch oracles
# ---------------------------------------------------------------------------------------------


def dsa_causal_bound_torch_oracle(
    scores: Tensor, causal_len: Tensor, pool_size: int
) -> Tensor:
    """The reference the bound is measured against. NOT a fallback for kernel-class work (P13).

    Kept in UPSTREAM'S OWN ``masked_fill`` form and in UPSTREAM'S OWN floor-division spelling --
    ``p >= causal_len // pool_size`` -- rather than the kernel's rearranged
    ``(p + 1) * pool_size > causal_len``. That is the point: if the two spellings were the same
    spelling, agreeing with this would only prove the kernel agrees with itself. The kernel never
    divides; this oracle does nothing else.
    """
    rows, width = int(scores.shape[0]), int(scores.shape[1])
    complete = (causal_len.reshape(rows, 1).to(torch.int64) // pool_size)
    cols = torch.arange(width, device=scores.device, dtype=torch.int64)[None, :]
    return scores.masked_fill(cols >= complete, NEG_INF)


def dsa_causal_sentinel_torch_oracle(values: Tensor, indices: Tensor) -> Tensor:
    """The reference the sentinel writer is measured against. NOT a P13 fallback.

    Uses ``torch.isinf`` on the sign-checked value rather than the kernel's negate-and-compare, so
    the two spellings are independent. Upstream's equivalent is
    ``tl.where(offsets < num_compressed, offsets, -1)`` (``attention.py:85``) and
    ``outIndices[rowIt] = -1`` (``sampler.cu:405``).
    """
    masked = torch.isinf(values) & (values < 0)
    return torch.where(masked, torch.full_like(indices, SENTINEL), indices)
