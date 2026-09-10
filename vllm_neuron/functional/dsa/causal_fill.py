# SPDX-License-Identifier: Apache-2.0
"""Exact causal index rows for the DSA indexer's short-sequence regime -- ``inc-glm53f-099``.

WHAT THIS KERNEL IS FOR, in one sentence: below the selection bound every candidate pool would be
selected anyway, so there is nothing to select and the right answer is each query row's own causal
prefix, written directly.

WHY THE REGIME EXISTS AT ALL. The landed selector ``dsa_topk_select`` admits ``0 < k < width``
STRICTLY: at ``k > width`` it raises (``topk_select.py:320-323``) and at ``k == width`` its route
gate returns False (``topk_select.py:294``) and the call takes the torch route SILENTLY. At the
target checkpoint's dials -- ``index_topk = 2048``, ``index_kpool = 4``, so ``select_k = 512`` --
the candidate width is the row's complete-pool count ``seq_len // 4``, which does not exceed 512
until ``seq_len`` reaches 2,052. So every request's first ~2,000 decode steps sit below the bound
and no landed seam serves them. Upstream serves exactly this regime by bypassing selection and
filling exact causal rows (``_fill_short_decode_causal_indices``,
``sparse_attn_indexer_kpool.py:203-217`` at ``878631b6``, taken when ``max_seq_len <=
topk_tokens``).

THE RULE, AND IT IS UPSTREAM'S SPELLED AS A CLOSED FORM::

    indices[i, c] = c   where c <= positions[i]
    indices[i, c] = -1  otherwise

Upstream writes the same thing in two torch statements -- ``rows[:] = causal_range[None, :]`` then
``rows[causal_range[None, :] > positions[:, None]] = -1`` (``sparse_attn_indexer_kpool.py:196-201``).

THE ``-1`` IS ``inc-glm53f-098``'s SENTINEL AND THIS MODULE ONLY WRITES IT. It means *"this column
selects no token"* and is a VALUE, not an out-of-bounds index. The masking is the consumer's:
``mla_sparse_attention`` masks ``-1`` columns before the row max and sum and admits ``lo >= -1``
(``mla_sparse.py:1262-1267``). Nothing here masks, clamps, compacts or fills, exactly as entry
``design-20260905-af`` route (a) ruled for the whole chain.

THE WIDTH IS THE CALLER'S, AND IT IS THE ADMISSIBLE ONE. ``width`` is the expanded width
``inc-glm53f-102`` emits -- a positive multiple of ``KEY_CHUNK``, from
``index_expand.index_expand_width`` -- never the raw ``pool_size * (n_groups + 1) - 1``. This
kernel fills ``-1`` beyond a row's causal columns exactly as that padding does, so the bypass path
and the selecting path hand ``mla_sparse_attention`` the same shape and the same sentinel
convention. The width is NOT recomputed here: a second spelling of that ceiling would be a second
thing to keep in step.

WHY THIS IS KERNEL-CLASS AND LANDS IN NKI (P13). The output is an index tensor produced on device
for the device kernel that consumes it -- ``inc-glm53f-048``'s precedent exactly. A torch fill on
the host would be a fallback for kernel-class work AND one host round trip per decode step, on the
step-by-step path that runs for the first two thousand steps of every request.

CONSTRUCTS, AND WHY EACH IS THE ONE USED. Every shape below is already screened on this image by a
landed kernel; nothing here is a new spelling.

  * ``nl.arange`` DOES NOT EXIST on this image, nor do ``mgrid``, ``nl.iota`` or
    ``nl.affine_select`` (``index_expand.py:115``, round-4 census ``NL_HAS_ARANGE=False``). The
    column index therefore comes from ``nisa.iota`` with ``channel_multiplier=0``, which gives
    every partition the same free-axis ramp -- the landed form at ``moe/router.py:1286`` and
    ``moe/topk_reduce.py:252``.
  * THE IOTA IS float32 BECAUSE THAT IS THE SCREENED DTYPE FOR THIS FORM. Both landed
    ``channel_multiplier=0`` call sites use a float32 destination; the only landed int32 iota takes
    ``channel_multiplier=1`` (``moe/permute_routed_tokens.py:675``), a different operation. One
    ``nisa.tensor_copy`` casts to int32 afterwards, which is the landed cross-dtype copy
    (``moe/router.py:1284``). The cast is exact: a column index is an integer far below float32's
    2**24 exactly-representable bound, and no width any sequence reaches comes close.
  * NO COMPARE AND NO SELECT. The rule is a closed form over ``maximum`` and ``minimum``, which is
    the landed tail-region idiom at ``index_expand.py:353-367`` generalised from one column to the
    whole width::

        keep = clamp(positions[i] - c + 1, 0, 1)     1 exactly while c <= positions[i]
        out  = (c + 1) * keep - 1                    c where keep is 1, -1 where keep is 0

  * A PER-ROW VALUE REACHES ``tensor_scalar`` AS A ``(rows, 1)`` COLUMN OPERAND, broadcast along
    the free axis (``moe/router.py:1291`` states the broadcast; ``moe/topk_reduce.py:262`` is the
    landed call). A ``(1, N)`` row operand is refused by the MLIR verifier for this position -- the
    ``-045`` finding, recorded at ``score_gemm.py:84-88`` -- so the reshape happens once on the
    host.
  * A CHAIN MIXING A TILE OPERAND AND AN INT OPERAND IN ONE ``tensor_scalar`` IS NOT USED, because
    no landed call site screens it. Two-int chains are screened (``index_expand.py:361``) and
    single-tile-operand calls are screened; this body uses only those two.
  * ``nl.divide`` is silently wrong on int32 and ``nl.right_shift`` refuses as the second op of a
    chain (``index_expand.py:108-114``). Neither is needed here and neither is used.
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
"""What a column that selects no token holds. ``inc-glm53f-098``'s value, written here, masked there."""

_SUPPORTED_DTYPES = (torch.int32,)
"""Position dtypes that take the NKI route. int32 is what the indexer carries and what the consumer
reads; a wider index type would double SBUF traffic for a range no sequence length reaches."""


class DsaCausalFillError(ValueError):
    """A malformed call: a non-positive ``width``, or ``positions`` of the wrong rank or dtype."""


@dataclass
class _CausalFillDispatchCounters:
    """Route-predicate counters for this module, form R-1 (``design/increment-plan.md`` D13).

    ``nki_dispatch`` counts dispatches THROUGH THIS MODULE'S SEAM, so the declared total over the
    declared cases is readable from one place. The declared total is 2 -- one per seam-calling
    acceptance item. ``torch_fallback`` is a declared 0 whose firing control is in the acceptance:
    with ``can_run_kernel`` forced False one call moves it to 1, so the zero is a reading rather
    than decoration (D1.5).
    """

    nki_dispatch: int = 0
    torch_fallback: int = 0
    last_kernel: tuple[str, str] | None = None


_COUNTERS = _CausalFillDispatchCounters()


def reset_causal_fill_dispatch_counters() -> None:
    """Zero the counters. Called at the START of each declared case (section 4b's convention)."""
    _COUNTERS.nki_dispatch = 0
    _COUNTERS.torch_fallback = 0
    _COUNTERS.last_kernel = None


def causal_fill_dispatch_counters() -> tuple[int, int]:
    """``(nki_dispatch, torch_fallback)`` since the last reset."""
    return (_COUNTERS.nki_dispatch, _COUNTERS.torch_fallback)


def causal_fill_kernel_identity() -> tuple[str, str] | None:
    """``(module, qualname)`` of the kernel the seam LAST dispatched, or ``None``.

    Derived THROUGH the seam rather than from this module's import list, so it certifies what ran
    instead of what was defined (D13.1). ``None`` before any dispatch, which is the reading that
    separates "no kernel ran" from "some kernel ran".
    """
    return _COUNTERS.last_kernel


def _kernel_identity_of(kernel) -> tuple[str, str]:
    """``(module, qualname)`` of the function a ``@nki.jit`` object actually wraps.

    Measured, not assumed, for the reason ``ragged_pack.py:170-181`` records: ``@nki.jit`` returns a
    kernel object whose ``__module__`` is ``"nki.framework.kernel"`` and whose ``__qualname__`` is
    ``None``, so reading those attributes off the decorated object would record the DECORATOR's
    identity and certify nothing.
    """
    inner = getattr(kernel, "__wrapped__", None) or getattr(kernel, "func", None) or kernel
    return (inner.__module__, inner.__qualname__)


# ---------------------------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------------------------


@nki.jit
def _causal_fill_nki(positions_hbm, width):
    """Each row's causal prefix, ``-1`` past its own position.

    Args:
        positions_hbm: ``[rows, 1]`` int32 -- each query row's own absolute position, ALREADY a
            column, because it reaches ``tensor_scalar`` as a per-row scalar operand.
        width: python int, the admissible expanded width. A trace-time constant, which it must be
            because it sizes the output tile; ``wrap_nki`` passes an int through unchanged
            (``decode_tail_update.py:544-546`` is the landed precedent).

    Returns:
        ``[rows, width]`` int32, holding ``c`` in column ``c`` of every row whose position is at
        least ``c`` and ``-1`` in every other column.

    EVERY TILE IS WRITTEN BY ONE EXPRESSION, so no region can be left undefined and none can be
    written twice -- the failure mode ``index_expand.py:297-303`` had to partition three regions to
    avoid. The column ramp is one iota and the rule is one closed form over it.

    THE ROW AXIS IS TILED AT ``nl.tile_size.pmax`` (``inc-glm53f-103d``). It used to say "there is
    no loop here at all", and that sentence was the defect: a row is a PARTITION, the partition axis
    serves 128, and this kernel bound every one of its nine tiles to the full row count. Any prefill
    above 128 tokens therefore died inside the vendor's own check --
    ``AssertionError: dma_copy dst partition dimension 132 exceeds maximum 128``
    (``nki/isa/_copy.py:152`` calling ``nki/isa/_validation.py:261``) -- and this kernel is the ONLY
    one the short-sequence bypass regime reaches, so no request above 128 tokens ran at all.

    THE TILING IS LAYOUT AND NOT ARITHMETIC, which is why the acceptance may ask for bit equality
    rather than a tolerance. Row ``i``'s output depends on ``positions[i]`` and the column index and
    on nothing else: ``ramp`` is an iota with ``channel_multiplier=0``, identical on every
    partition, every step is an elementwise ``tensor_scalar``/``tensor_tensor``, and there is no
    reduction across rows anywhere. A tile boundary may therefore fall at any row.

    THE LOOP FORM IS THE ONE LANDED IN THIS PACKAGE, not one invented here: ``paged_gather.py``
    tiles its token axis the same way (``:212-219`` for the bound and the short last tile,
    ``:243-244`` for the strided store), including the ``min`` for the ragged tile and the
    ``.ap(pattern=..., offset=...)`` access patterns. ``pos`` is loaded INSIDE the loop, once per
    tile, because it is the one PER-ROW operand -- hoisting it would bound every tile by the first
    tile's rows and would read identically to this kernel at any row count of 128 or fewer.
    """
    rows_total = positions_hbm.shape[0]
    pmax = nl.tile_size.pmax
    n_tiles = (rows_total + pmax - 1) // pmax

    out = nl.ndarray((rows_total, width), dtype=nl.int32, buffer=nl.shared_hbm)

    for t in range(n_tiles):
        # The last tile is short whenever the row count is not a multiple of `pmax`. It is narrowed
        # rather than padded, so no lane that holds no row can contribute one -- the same choice
        # `paged_gather.py:216-219` records, and the reason a non-multiple row count is one of the
        # acceptance's declared extents rather than an afterthought.
        rows = min(pmax, rows_total - t * pmax)
        off = t * pmax

        # WHY float32 AND NOT int32, WHICH IS WHAT THIS TILE HELD UNTIL THE COMPILER REFUSED IT.
        # `pos` is the only TILE this kernel passes as a `tensor_scalar` operand, and the ISA
        # requires a tile operand to be float32: "arithmetic operators impose no restriction on the
        # data types of input tensor ``data`` and output tensor ``dst``, but the operand0 and
        # operand1 (if used) must be float32" (`nki/isa/_tensor_ops.py:279`, with the conversion at
        # `:227`). An int32 tile here is refused by the MLIR verifier with
        # `'nisa.tensor_scalar_arith' op 'operand0' must be float32, got 'i32'` -- read at every
        # width in `capture-099-r1.out`. The NKI SIMULATOR does not run that stage, which is why
        # four green items never saw it.
        #
        # THE VALUE IS UNCHANGED, not merely close. The engine already casts `data` to float32 and
        # does the arithmetic in float32 math regardless, casting back to `dst.dtype` at no cost, so
        # this makes an existing float32 computation explicit for one operand rather than
        # introducing one. Every quantity involved is a whole number below 2**24 -- a position is at
        # most `select_k * pool + pool - 2` and a column at most `width - 1` -- so float32 holds
        # each one exactly. `test_causal_fill.py`'s exactness items are the reading, not this
        # comment.
        #
        # LOADED PER TILE, DELIBERATELY. `pos` is this kernel's only per-row operand, so a load
        # hoisted out of this loop would bound every tile by the FIRST tile's rows. That reads
        # identically to this kernel at any row count of 128 or fewer, which is why the acceptance
        # carries a control that hoists it and drives a row count above 128.
        pos = nl.ndarray((rows, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=pos, src=nl.load(positions_hbm.ap(pattern=[[1, rows], [1, 1]], offset=off))
        )

        # THE COLUMN RAMP. `channel_multiplier=0` gives every row the same 0..width-1, which is the
        # broadcast `causal_range[None, :]` upstream builds with a torch arange. It stays float32
        # and is used as the ramp directly: the int32 `cols` copy this used to make is gone, because
        # the whole closed form below is float32 now and that copy was the cast the chain no longer
        # needs. It is rebuilt per tile and is identical in every tile -- it carries no row state,
        # so a tile boundary cannot change it.
        ramp = nl.ndarray((rows, width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.iota(dst=ramp, pattern=[[1, width]], offset=0, channel_multiplier=0)

        # `1 - c`, as one two-scalar chain. Building it this way rather than as `-(c - 1)` keeps the
        # tile operand and the scalar operands in separate calls, which is the screening constraint.
        one_minus = nl.ndarray((rows, width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=one_minus, data=ramp,
                           op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1=1.0)

        # `room = positions[i] - c + 1`, at least 1 exactly while `c <= positions[i]` and at most 0
        # after it. One call, one tile operand, broadcast along the free axis. THIS is the call the
        # MLIR verifier refused while `pos` was int32; see the dtype note above `pos`.
        room = nl.ndarray((rows, width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=room, data=one_minus, op0=nl.add, operand0=pos)

        # `keep = clamp(room, 0, 1)`. The landed clamp shape, `index_expand.py:361`.
        keep = nl.ndarray((rows, width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=keep, data=room,
                           op0=nl.maximum, operand0=0.0, op1=nl.minimum, operand1=1.0)

        # `out = (c + 1) * keep - 1`: `c` where the column is causal, `-1` where it is not. The
        # `+1`/`-1` pair is what lets column 0 survive -- a bare `c * keep` would write 0 for a
        # masked column 0 and 0 is a real token index.
        cols1 = nl.ndarray((rows, width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=cols1, data=ramp, op0=nl.add, operand0=1.0)
        prod = nl.ndarray((rows, width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=prod, data1=cols1, data2=keep, op=nl.multiply)
        acc = nl.ndarray((rows, width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=acc, data=prod, op0=nl.subtract, operand0=1.0)

        # THE ONE CAST AT THE STORE. Every tile above is float32, so the closed form runs in one
        # dtype from the ramp to here and no intermediate is converted. The result is an index
        # tensor, so it is cast to int32 exactly once, by `nisa.tensor_copy`, which is this
        # repository's landed cross-dtype copy. Casting here rather than inside `nl.store` keeps the
        # conversion a named instruction a reader can see, instead of an implicit property of the
        # store.
        result = nl.ndarray((rows, width), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=result, src=acc)

        # The strided store puts this tile's rows at their own offset in the full output. The
        # pattern is `paged_gather.py:243-244`'s, which is the landed form for writing a row slice
        # of a `[rows, width]` HBM tensor.
        nl.store(
            out.ap(pattern=[[width, rows], [1, width]], offset=off * width), value=result
        )

    return out


# ---------------------------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------------------------


@torch._dynamo.assume_constant_result
def _record_nki_dispatch(rows: int, width: int) -> None:
    """Record which kernel the seam dispatched, and log it, OFF the compiled graph.

    The template is landed and measured: ``kpool_hadamard.py:428-457`` by way of
    ``score_gemm.py:306-330``. A folded helper takes ints only -- Dynamo runs it at trace time and
    refuses to reconstruct an ``@nki.jit`` object -- so the kernel is read as a module global rather
    than passed in. D13.1 still holds: this body runs only when the dispatch branch runs, so the
    recorded identity is derived by TAKING the branch.
    """
    _COUNTERS.last_kernel = _kernel_identity_of(_causal_fill_nki)
    logger.info("[dsa-causal-fill] kernel=nki rows=%d width=%d", rows, width)


def _validate(positions: Tensor, width: int) -> int:
    """Host-side validation. Returns ``rows``.

    Reads only ``.shape`` and ``.dtype``, never a tensor VALUE, so nothing here forces a
    device-to-host synchronisation or a data-dependent trace. Both refusals are RAISES rather than
    gate declines, which is the block's own ruling: a non-positive width and a non-integer position
    tensor are malformed calls, and serving either through the torch oracle would hide a caller bug
    behind a correct-looking answer.
    """
    if not isinstance(width, int) or isinstance(width, bool):
        raise DsaCausalFillError(
            f"width must be a python int, because it sizes the output tile at trace time; got "
            f"{type(width).__name__} {width!r}"
        )
    if width < 1:
        raise DsaCausalFillError(
            f"width must be at least 1 column; got width={width}. The admissible expanded width "
            f"comes from index_expand.index_expand_width and is a positive multiple of KEY_CHUNK, "
            f"so a width below 1 is a caller error rather than an empty batch"
        )
    if positions.dtype not in _SUPPORTED_DTYPES:
        raise DsaCausalFillError(
            f"positions must be int32, the dtype the indexer carries and the consumer reads; got "
            f"{positions.dtype}"
        )
    if positions.ndim != 1:
        raise DsaCausalFillError(
            f"positions must be 1-D [rows], one absolute position per query row; got shape "
            f"{tuple(positions.shape)}"
        )
    rows = int(positions.shape[0])
    if rows <= 0:
        raise DsaCausalFillError(f"positions must carry at least one row; got {rows}")
    return rows


def can_run_dsa_causal_fill(positions: Tensor, width: int) -> bool:
    """Whether the NKI kernel serves this call. ``False`` sends it to the torch oracle.

    Narrow on purpose. Every malformed call is already refused by :func:`_validate` before this is
    reached, so the only thing left for the gate to decide is whether NKI is available at all --
    which is what makes ``can_run_kernel`` the firing control the acceptance uses to move the
    fallback zero off 0.
    """
    if not can_run_kernel():
        return False
    return positions.dtype in _SUPPORTED_DTYPES and positions.ndim == 1 and width >= 1


def dsa_causal_fill(positions: Tensor, width: int) -> Tensor:
    """THE COUNTED SEAM. Each query row's causal prefix, ``-1`` past its own position.

    Args:
        positions: ``[rows]`` int32 -- each query row's own absolute position. For a decode step
            that is the new token's position; for a prefill row it is the row's index in its
            sequence.
        width: the ADMISSIBLE expanded width, a positive multiple of ``KEY_CHUNK``, obtained from
            ``index_expand.index_expand_width``. Never the raw expansion width, and never recomputed
            here.

    Returns:
        ``[rows, width]`` int32. Column ``c`` of row ``i`` holds ``c`` when ``c <= positions[i]``
        and :data:`SENTINEL` otherwise. The sentinel is a VALUE that the sparse attention kernel
        masks (``inc-glm53f-098``); this function never masks it.

    Raises:
        DsaCausalFillError: for a malformed call -- a non-int ``width``, a ``width`` below 1, a
            ``positions`` that is not int32 or not 1-D, or an empty ``positions``.
    """
    rows = _validate(positions, width)

    if not can_run_dsa_causal_fill(positions, width):
        _COUNTERS.torch_fallback += 1
        return dsa_causal_fill_torch_oracle(positions, width)

    # The transport the device cannot pay for: the per-row position reaches `tensor_scalar` as a
    # COLUMN operand, so the reshape happens once here rather than per use on the device.
    pos_col = positions.reshape(rows, 1).contiguous()

    _COUNTERS.nki_dispatch += 1
    # The log and the identity read are FOLDED off the traced graph; the counter increment stays,
    # because a plain int attribute store is a recorded side effect and not a host call.
    _record_nki_dispatch(rows, width)
    return wrap_nki(_causal_fill_nki)(pos_col, width)


# ---------------------------------------------------------------------------------------------
# Torch oracle
# ---------------------------------------------------------------------------------------------


def dsa_causal_fill_torch_oracle(positions: Tensor, width: int) -> Tensor:
    """The reference the NKI route is measured against. NOT a fallback for kernel-class work (P13).

    This exists to be the oracle in the acceptance and to serve a call the gate refuses -- which,
    after :func:`_validate`, means only a call made where NKI is unavailable. The counted
    ``torch_fallback`` reading is exactly 0 across the declared cases, and the acceptance proves
    that zero can fire by forcing ``can_run_kernel`` False on purpose.

    Transcribed from upstream's ``_fill_causal_indices``
    (``sparse_attn_indexer_kpool.py:196-201`` at ``878631b6``) and kept in UPSTREAM'S OWN
    write-then-mask form rather than rewritten into the kernel's closed form. That is the point: if
    the two spellings were the same spelling, agreeing with this would only prove the kernel agrees
    with itself. Upstream assigns the ramp and then overwrites the columns past each row's position;
    the kernel never writes a value it has to take back.
    """
    rows = int(positions.shape[0])
    causal_range = torch.arange(width, device=positions.device, dtype=torch.int32)
    out = causal_range[None, :].expand(rows, width).clone()
    out[causal_range[None, :] > positions.to(torch.int32)[:, None]] = SENTINEL
    return out.to(torch.int32)
