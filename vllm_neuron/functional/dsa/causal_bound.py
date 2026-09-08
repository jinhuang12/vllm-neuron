# SPDX-License-Identifier: Apache-2.0
"""The DSA indexer's selecting-regime causal bound -- ``inc-glm53f-103``.

WHAT THIS MODULE IS FOR, in one sentence: a query row must not select a key pool that finishes
after the row's own position, so the scores of every such pool are pushed to a fill value below
every legal score before the selector runs, and any selection that comes back holding that fill --
or holding an index the selector's own padding invented -- is replaced by the ``-1`` sentinel.

THE FILL IS FINITE, AND THAT IS THE ``103r5`` REPAIR. The first three rounds of this module wrote
``-inf`` and read it back verbatim. The selector will not carry that promise: it pads its own input
with a finite constant at indices past the real width, and it moves selected values across
partitions with a permutation matmul where ``0 * -inf`` is NaN. Both faces are recorded with their
bytes in ``increments/contradiction-103-selector-pad-6874a0f5.md``; the repair the lead ruled
(``approvals/LEAD-LOG.md`` §752) is :data:`BOUND_FILL` plus the marker's second arm on the index.

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

    scores[i, p] = BOUND_FILL   where (p + 1) * pool_size >  causal_len[i]
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

UPSTREAM'S VALUE AT THIS STAGE IS ``-inf``, AND THIS MODULE DELIBERATELY DIFFERS. Upstream bounds the
indexer LOGITS with a literal ``float("-inf")`` -- ``logits = logits.masked_fill(~mask,
float("-inf"))`` (``vllm/v1/attention/ops/rocm_aiter_mla_sparse.py:734``), and bounds the sparse
attention score the same way (``backends/mla/rocm_aiter_mla_sparse.py:720``). Upstream can afford the
exact value because its selector is ``torch.topk``, which neither pads nor rotates. This fork's
selector does both, so this module writes :data:`BOUND_FILL` instead and the difference is a
DEVIATION WITH A REASON rather than a port mismatch: the equivalence upstream needs is "a pool the
row may not see is never selected", which a fill below every legal score gives exactly. The
``-FLT_MAX`` a reader may find at ``csrc/libtorch_stable/sampler.cu:407`` is the top-k selector's
OUTPUT tail padding, downstream of selection, and is not this stage; round 1 of this block's
read-first gate compared against it by mistake and the correction is kept beside it rather than
deleted (``readfirst-103-upstream.out``, ``EXIT=2``, superseded). That value is finite; what reason
upstream had for it is not read here and is not claimed.

A WHOLLY-BOUNDED ROW IS LEGAL AND ITS MEANING IS ALREADY FIXED. A row with ``causal_len <
pool_size`` has zero complete pools, so every column is filled. The consumer settles what that
means: ``mla_sparse_attention``'s oracle records "A wholly-sentinel row is all -inf, and softmax of
that is NaN rather than zero -- so the zeros the kernels produce for it are written here too"
(``mla_sparse.py:1494-1496``). Nothing here masks, clamps or compacts; this module only writes.

WHY THIS IS KERNEL-CLASS AND LANDS IN NKI (P13). Both outputs are device tensors produced for the
device kernels that consume them, which is ``inc-glm53f-048``'s precedent exactly. A torch mask on
the host would be a fallback for kernel-class work AND one host round trip per step, on the
per-forward path.

CONSTRUCTS, AND THE SCREENING FOR EACH. Since the ``103r5`` repair made the fill finite, EVERY
construct below has a landed fork call site -- the one exception this list used to carry is retired,
and its old text is kept as the last bullet because the reason it is gone is worth reading.

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
  * THE CONSTRUCT THAT USED TO HAVE NO FORK-AUTHORED SCREENING, AND NO LONGER EXISTS HERE:
    ``nisa.memset`` with a NON-FINITE value. ``memset`` has 37 landed call sites and every landed
    value is finite (``0.0`` eight times, ``0`` three times, ``1.0``, ``-2``, and ``SENTINEL_INDEX``
    at ``mla_sparse.py:236``); the nearest on-image precedent for an ``inf`` immediate is vendored
    rather than fork-authored, ``nisa.nc_match_replace8(imm=float("-inf"))``
    (``vendored_kernels/rotational_topk/rotational_topk_utils.py:1065``, ``:1109``, ``:1116``). Two
    risks came with that: the MLIR verifier stage, which the NKI simulator DOES NOT RUN
    (``causal_fill.py:189-196`` records that trap costing ``-099`` four green items, so a green
    Tier N run would not have cleared it), and the arithmetic risk this bullet's own last line
    already named -- ``0 * -inf`` is NaN -- which is what the selector's permutation matmul does to
    a bounded value. :data:`BOUND_FILL` is finite, so this memset is now one of the 37 and both
    risks are gone rather than deferred to the capture leg.
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

BOUND_FILL = -1.0e30
"""The bounded-score value: FINITE, and far below any score the indexer can produce.

WHY NOT ``-inf``, WHICH IS UPSTREAM'S VALUE HERE (``rocm_aiter_mla_sparse.py:734``). Repair
``103r5``, ruled by the lead at ``approvals/LEAD-LOG.md`` §752 on the evidence in
``increments/contradiction-103-selector-pad-6874a0f5.md``. The landed selector this bound feeds moves
selected VALUES between partitions with a 0/1 permutation matrix on the tensor engine
(``vendored_kernels/rotational_topk/rotational_topk.py:385`` into
``rotational_topk_utils.py:867-886``), and ``0 * -inf`` is NaN under IEEE-754 -- so an ``-inf`` handed
to that kernel need not come back as ``-inf``, and a marker that keys on the exact value would then
mark nothing. A finite fill crosses a permutation matmul unchanged: every output is one ``1 * value``
term plus zeros.

WHY THIS MAGNITUDE. Three constraints, all from bytes rather than taste. It must be BELOW every
legal indexer score, so no real candidate is ever mistaken for a fill -- indexer scores are softmax-
scale logits, tens at most. It must be far ABOVE ``FLOAT32_MIN`` (-3.4e38), so a matmul accumulation
cannot overflow to an infinity, which is the vendor's own reason for padding with a modest
``-9948.0`` rather than the type minimum (``rotational_topk_utils.py:32-40``). And it must survive a
bfloat16 round-trip with room to spare, which :data:`BOUND_FILL_MARK` provides. The fork already
prefers finite mask fills where a kernel consumes them: ``functional/sampling.py:263`` masks with
``-3000.0`` and ``functional/attention/attention_cte.py:145`` with ``torch.finfo(dtype).min``."""

BOUND_FILL_MARK = -1.0e29
"""The threshold the marker fires at or below. Ten times closer to zero than :data:`BOUND_FILL`.

The gap is what makes the marker independent of dtype rounding: a ``BOUND_FILL`` that has been
rounded to bfloat16 and back is still orders of magnitude below this, while no real score comes near
it. A ``-inf`` that somehow survives is below it too, so this threshold also catches the value the
vendored selector itself writes when it strikes a taken candidate
(``rotational_topk_utils.py:1065``)."""

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
    """:data:`BOUND_FILL` at every pool column the row's own length does not complete.

    Args:
        scores_hbm: ``[rows, width]`` float32 -- one score per candidate pool per query row.
        causal_len_hbm: ``[rows, 1]`` int32 -- each row's own causal length, ALREADY a column,
            because it reaches ``tensor_scalar`` as a per-row scalar operand.
        pool_size: python int, tokens per pool. A trace-time constant; ``wrap_nki`` passes an int
            through unchanged (``decode_tail_update.py:544-546``).

    Returns:
        ``[rows, width]`` float32. Column ``p`` of row ``i`` holds :data:`BOUND_FILL` when
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

    # The fill source. FINITE since repair `103r5`, which puts this memset in the same landed
    # family as the other 37 (every landed value is finite; see the module docstring).
    fill = nl.ndarray((rows, width), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=fill, value=BOUND_FILL)

    # The mask. `scores_sb` is left alone wherever `bounded` is 0.
    nisa.tensor_copy_predicated(src=fill, predicate=bounded, dst=scores_sb)

    nl.store(out, value=scores_sb)
    return out


@nki.jit
def _causal_sentinel_nki(values_hbm, indices_hbm, width):
    """``-1`` at every selection that reaches no pool the row may see -- by value OR by index.

    Args:
        values_hbm: ``[rows, k]`` float32 -- the selector's returned values.
        indices_hbm: ``[rows, k]`` int32 -- the selector's returned pool ids.
        width: python int, the number of REAL pool columns the selector was given. A trace-time
            constant, the way ``pool_size`` is in :func:`_causal_bound_nki`.

    Returns:
        ``[rows, k]`` int32. :data:`SENTINEL` at every slot whose value is at or below
        :data:`BOUND_FILL_MARK` and at every slot whose index is ``>= width``; the selector's own
        index, bit for bit, everywhere else.

    TWO ARMS, AND EACH HAS ITS OWN JOB. Repair ``103r5`` (``approvals/LEAD-LOG.md`` §752; the
    evidence is ``increments/contradiction-103-selector-pad-6874a0f5.md``).

      * THE VALUE ARM catches a column the bound filled. Those columns are real columns with legal
        indices, so only their value distinguishes them.
      * THE INDEX ARM catches a PAD column the selector invented. The vendored loader pads the last
        fold of an uneven fold with a FINITE ``-9948.0``
        (``vendored_kernels/rotational_topk/cascaded_max_utils.py:61-66``, ``:154-158``) and hands
        those columns positions that keep counting past the real width
        (``rotational_topk.py:203-207`` with ``rotational_topk_utils.py:826-856``, and the padded
        extent it names at ``:434``). A finite pad OUTRANKS every bound-filled column, so on the
        rows this bound acts on the pads win slots -- and no value test can see them, because their
        value is an ordinary finite number.

    HOW THE POLARITY WORKS WITHOUT A ``less`` OR AN ``or``. The result tile starts as all
    :data:`SENTINEL` and the selector's index is copied IN only where the value is a real score, so
    "mark" is the default and "keep" is the exception:

      1. the ids are copied to float32 (exact: a pool index is a whole number far below 2**24);
      2. ``greater`` against ``width - 1`` marks the pad ids, and one predicated copy writes
         :data:`SENTINEL` over them IN the id tile;
      3. ``greater`` against :data:`BOUND_FILL_MARK` -- applied to the value directly, no negation --
         is 1 exactly where the value is a real score. NaN fails that compare, so a NaN value is
         marked rather than silently kept, which is what makes this kernel independent of whether
         the tensor engine's ``0 * x`` follows IEEE-754;
      4. one predicated copy moves the screened ids into the all-``-1`` result where that keep mask
         is set.

    Every construct here is already landed in this module: ``greater`` into an integer destination
    (``moe/topk_reduce.py:351-355``), ``memset`` with the int32 sentinel (``mla_sparse.py:236``), and
    ``tensor_copy_predicated`` as the select (``moe/topk_reduce.py:309``, ``:360``).
    """
    rows = values_hbm.shape[0]
    k = values_hbm.shape[1]

    out = nl.ndarray((rows, k), dtype=nl.int32, buffer=nl.shared_hbm)

    # The indices, loaded once. The pad screen writes into this tile.
    idx_sb = nl.ndarray((rows, k), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=idx_sb, src=nl.load(indices_hbm))

    vals = nl.ndarray((rows, k), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=vals, src=nl.load(values_hbm))

    # The sentinel source, and the result that starts out entirely sentinel.
    fill = nl.ndarray((rows, k), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=fill, value=SENTINEL)
    marked = nl.ndarray((rows, k), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(dst=marked, value=SENTINEL)

    # THE INDEX ARM. `index >= width` written as `index > width - 1`, which needs only `greater`.
    idxf = nl.ndarray((rows, k), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=idxf, src=idx_sb)
    pad = nl.ndarray((rows, k), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=pad, data=idxf, op0=nl.greater, operand0=float(width) - 1.0)
    nisa.tensor_copy_predicated(src=fill, predicate=pad, dst=idx_sb)

    # THE VALUE ARM, as a KEEP mask: 1 where the value is a real score. A fill, a `-inf` and a NaN
    # all fail this compare, so all three are left at the sentinel the result already holds.
    keep = nl.ndarray((rows, k), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=keep, data=vals, op0=nl.greater, operand0=BOUND_FILL_MARK)
    nisa.tensor_copy_predicated(src=idx_sb, predicate=keep, dst=marked)

    nl.store(out, value=marked)
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
def _record_sentinel_dispatch(rows: int, k: int, width: int) -> None:
    """Record which kernel the sentinel seam dispatched, and log it, OFF the compiled graph."""
    _SENTINEL_COUNTERS.last_kernel = _kernel_identity_of(_causal_sentinel_nki)
    logger.info(
        "[dsa-causal-sentinel] kernel=nki rows=%d k=%d width=%d", rows, k, width
    )


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
    """THE COUNTED SEAM. :data:`BOUND_FILL` at every pool a query row's own length does not complete.

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


def _validate_sentinel(values: Tensor, indices: Tensor, width: int) -> int:
    """Host-side validation for the sentinel writer. Returns ``rows``.

    ``width`` is validated the way ``pool_size`` is in :func:`_validate_bound` and for the same
    reason: it is a trace-time constant, so a tensor or a bool arriving here would be baked into the
    graph as something the caller did not mean. A caller that cannot say how many real columns the
    selector was given cannot use this seam -- the pad arm has no meaning without it -- so this
    RAISES rather than declining to the oracle.
    """
    if not isinstance(width, int) or isinstance(width, bool):
        raise DsaCausalBoundError(
            f"width must be a python int, because it is a trace-time constant; got "
            f"{type(width).__name__} {width!r}"
        )
    if width < 1:
        raise DsaCausalBoundError(
            f"width must be the positive number of real pool columns the selector was given; got "
            f"width={width}"
        )
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


def can_run_dsa_causal_sentinel(values: Tensor, indices: Tensor, width: int) -> bool:
    """Whether the NKI kernel serves this sentinel call. ``False`` sends it to the torch oracle.

    ``width`` is accepted so the gate has the same signature as the seam it guards -- a caller that
    can ask the gate with fewer arguments than the call it is about would be asking about a different
    call. It is not read here: a malformed ``width`` RAISES in :func:`_validate_sentinel`, so the only
    thing left for this gate to decide is whether NKI is available at all.
    """
    if not can_run_kernel():
        return False
    return values.dtype in _SCORE_DTYPES and indices.dtype in _INDEX_DTYPES


def dsa_causal_sentinel(values: Tensor, indices: Tensor, width: int) -> Tensor:
    """THE COUNTED SEAM. :data:`SENTINEL` at every selection that reaches no pool the row may see.

    Args:
        values: ``[rows, k]`` float32, the selector's returned values.
        indices: ``[rows, k]`` int32, the selector's returned pool ids.
        width: the number of REAL pool columns the selector was given, which is
            ``bounded.shape[1]`` at the dispatch site. Anything the selector returns at or above it
            is a pad the selector invented, not a pool -- see :func:`_causal_sentinel_nki`.

    Returns:
        ``[rows, k]`` int32, sentinelised as :func:`_causal_sentinel_nki` describes.

    Raises:
        DsaCausalBoundError: for a non-int or non-positive ``width``, a rank or shape mismatch, or a
            non-int32 ``indices``.
    """
    rows = _validate_sentinel(values, indices, width)

    if not can_run_dsa_causal_sentinel(values, indices, width):
        _SENTINEL_COUNTERS.torch_fallback += 1
        return dsa_causal_sentinel_torch_oracle(values, indices, width)

    _SENTINEL_COUNTERS.nki_dispatch += 1
    _record_sentinel_dispatch(rows, int(values.shape[1]), width)
    return wrap_nki(_causal_sentinel_nki)(values.contiguous(), indices.contiguous(), width)


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
    return scores.masked_fill(cols >= complete, BOUND_FILL)


def dsa_causal_sentinel_torch_oracle(
    values: Tensor, indices: Tensor, width: int
) -> Tensor:
    """The reference the sentinel writer is measured against. NOT a P13 fallback.

    WRITTEN IN THE OPPOSITE POLARITY TO THE KERNEL, on purpose. The kernel starts from an all-``-1``
    tile and copies an index IN where a KEEP mask is set; this oracle builds the MARK mask directly
    out of ``le`` and ``ge``, and names NaN explicitly with ``torch.isnan`` where the kernel gets NaN
    for free from a failed ``greater``. Two spellings that disagree about NaN would be caught by the
    items that feed one in; one spelling written twice would only prove the module agrees with
    itself. Upstream's equivalent is ``tl.where(offsets < num_compressed, offsets, -1)``
    (``attention.py:85``) and ``outIndices[rowIt] = -1`` (``sampler.cu:405``).
    """
    filled = torch.le(values, BOUND_FILL_MARK) | torch.isnan(values)
    pad = torch.ge(indices, width)
    return torch.where(filled | pad, torch.full_like(indices, SENTINEL), indices)
