# SPDX-License-Identifier: Apache-2.0
"""Acceptance for the query-row tiling of the index expansion -- ``inc-glm53f-103c``.

WHAT IS BEING ASSERTED. ``_index_expand_nki`` used to bind the selected-row count to dim 0 of twelve
SBUF tiles, and SBUF's partition axis holds 128 rows at most, so a selecting-regime prefill above that
did not refuse by name -- nothing in the module counts rows -- it dispatched and died inside the
vendor's own assert. The kernel now walks that axis in tiles of at most 128 rows, and at 128 rows or
fewer it is one tile, which is the program that was there before.

FIVE ITEMS, ONE PER CONJUNCT, NO ``parametrize``, every name carrying ``tiled`` so ``pytest -k tiled``
selects exactly them and the declared count is derivable from this file. Two failing controls live
INSIDE the items whose readings they protect (``design-20260905`` §63: a strengthening under the same id
never moves a declared count). A NEW FILE rather than an extension, because the landed
``test_index_expand.py`` is ``-048``'s acceptance at 44 items in 1,237 L; re-running those 44 on the new
body is a second step of the acceptance run, not a reading this file takes.

NO TOLERANCE IS AUTHORED HERE, and none may be. Tiling moves rows and computes no new number; every
output is int32; the registered bar is EXACT bit equality (``increment-plan.md`` :1266). ``torch.equal``
is the only comparison, and ``max_abs_diff`` is printed beside it as a measured number.

THE PARENT BODY IS RE-SPELLED BELOW because commit 1 replaced it and item 2 needs the pre-tiling result
to still be measurable. A mis-copied replica cannot pass quietly: item 2 measures the replica against
the torch oracle too, so a drifted replica reddens instead of agreeing with a wrong candidate.
"""

import re

import torch

import nki
import nki.isa as nisa
import nki.language as nl

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.attention.mla_sparse import KEY_CHUNK
from vllm_neuron.functional.dsa import index_expand as mod
from vllm_neuron.functional.dsa.causal_bound import (
    can_run_dsa_causal_bound,
    can_run_dsa_causal_sentinel,
    causal_bound_dispatch_counters,
    causal_sentinel_dispatch_counters,
    dsa_causal_bound,
    dsa_causal_sentinel,
    reset_causal_bound_dispatch_counters,
    reset_causal_sentinel_dispatch_counters,
)
from vllm_neuron.functional.dsa.index_expand import (
    INDEX_KPOOL,
    PARTITION_MAX,
    can_run_dsa_index_expand,
    dsa_index_expand,
    index_expand_dispatch_counters,
    index_expand_kernel_identity,
    index_expand_raw_width,
    index_expand_width,
    reset_index_expand_dispatch_counters,
    row_tile_count,
    row_tiles,
)
from vllm_neuron.functional.dsa.topk_select import (
    can_run_dsa_topk_select,
    dsa_topk_select,
    reset_topk_select_dispatch_counters,
    topk_select_dispatch_counters,
)

# The two UNCHECKED tiling helpers, for the WRONG-WALK control kernel only, so it differs from the
# candidate in exactly one thing -- the hoist -- and not in its tile arithmetic.
from vllm_neuron.functional.dsa.index_expand import (
    _row_tile_count_unchecked,
    _row_tiles_unchecked,
)

# `-048`'s OWN precondition reader, imported and never re-implemented: re-spelling the rule here would
# let this file pass against its own idea of it.
from test.vllm_neuron.functional.dsa.test_index_expand import (
    _precondition_violations,
)

POOL_SIZE = INDEX_KPOOL
"""Tokens per pool, READ from the module rather than typed: it is the checkpoint's compress ratio."""

N_GROUPS = 8
"""Selected pools per row -- the landed acceptance's own small shape, and the ``k`` item 5's chain hands
the selector, so the chain's output is exactly this file's ``pool_ids`` shape."""

SMALL_ROWS = 4
"""The landed acceptance's row count. The extent item 2 compares against the parent at."""

TRAP_ROWS = 132
"""The row count the observed trap was reported at -- and a non-multiple of 128, so one reading."""

TILED_ROWS = 2048
"""The registered envelope in query tokens (``acceptance-preregistration.md`` A-5).

IT IS A WHOLE NUMBER OF TILES -- 16 x 128, remainder 0 -- so it has NO short last tile. Item 4 used to
claim it probed one here; see :data:`TAIL_TILE_ROWS`."""

TAIL_TILE_ROWS = 16 * PARTITION_MAX + 4
"""2,052 rows: the SMALLEST selecting-regime extent that has a short last tile, and item 4's second one.

The regime this kernel serves starts at a longest sequence of 2,052 tokens, and 2,052 is 16 x 128 + 4 --
seventeen tiles, the last of them 4 rows tall. That is where a remainder bug lives, and the envelope
cannot show it: the registered 2,048 divides by 128 exactly. Written as arithmetic on
:data:`PARTITION_MAX` so the two facts cannot drift apart."""

TILED_LADDER = [SMALL_ROWS, PARTITION_MAX, TRAP_ROWS, 2 * PARTITION_MAX, TILED_ROWS]
"""Every edge of the tiling: the landed shape, the last single tile, the observed trap and
non-multiple, two whole tiles, and the envelope -- derived from :data:`PARTITION_MAX`, not typed."""

ADMISSION_LADDER = [TRAP_ROWS, 2 * PARTITION_MAX, TILED_ROWS]
"""The three extents the PARENT cannot serve at all -- item 1's own set."""

CHAIN_LADDER = [TRAP_ROWS, 2 * PARTITION_MAX]
"""Where item 5 runs the whole chain above 128 rows. Two extents rather than the envelope: the reading
is that the chain RUNS above the ceiling, and the selector's cost at 2,048 rows buys no new fact."""

CHAIN_WIDTH = 32
"""Candidate columns for the chain's selector, which needs ``0 < k < width`` strictly for ``k`` 8."""

SEQ_FLOOR = N_GROUPS * POOL_SIZE
"""The shortest fixture sequence: just enough complete pools that every planted id is legal for -048."""

SEQ_PERIOD = 17
"""Row ``i``'s length is ``SEQ_FLOOR + i % SEQ_PERIOD``, and 17 does not divide 128 -- so every tile
past the first holds DIFFERENT lengths than the first and the hoist control can fail. It also cycles
the tail remainder through every value, so the tail region is live on every tile."""

PID_SENTINEL_STRIDE = 5
"""Where a ``-1`` "no pool selected" id is planted: every fifth ``(row + group)``. Coprime with both
:data:`N_GROUPS` and 128, so the negative-id branch is live in every tile at a different offset."""

VENDOR_PARTITION_ASSERT = "dma_copy dst partition dimension {rows} exceeds maximum {pmax}"
"""What the PARENT died of above the ceiling in the run that found it -- kept for the READER.

Nothing in the module counts rows, so the parent does not refuse by name; it dispatches, and the
vendor's assert fires in ``nki/isa/_copy.py:152`` by way of ``nki/isa/_validation.py:261``."""

VENDOR_PARTITION_NUMBERS = r"partition dimension (\d+) exceeds maximum (\d+)"
"""WHAT THE CONTROL ASSERTS ON. The claim is "132 rows exceeded the 128-row partition axis" and it lives
in the two NUMBERS; the words around them are the vendor's to change. Asserting the sentence would
redden a correct candidate on a rewording -- ``-103b`` commit 3's repair. The wording is still printed,
so a drift shows up in the transcript as a fact."""


def _emit(item: str, **values: object) -> None:
    """Print one machine-readable reading row. Grep it with ``grep -o``, never with a ``^`` anchor:
    pytest writes a progress marker with no trailing newline, so a row can arrive prefixed by it."""
    body = " ".join(f"{k}={v}" for k, v in values.items())
    print(f"E103|{item}|{body}", flush=True)


def _seq_lens(rows: int) -> torch.Tensor:
    """``[rows]`` int32 sequence lengths, upstream's shape, one per selected row."""
    return torch.tensor(
        [SEQ_FLOOR + (i % SEQ_PERIOD) for i in range(rows)], dtype=torch.int32
    )


def _pool_ids(rows: int) -> torch.Tensor:
    """``[rows, N_GROUPS]`` int32 pool ids, ``-1`` planted on a coprime stride. Every non-negative id
    is ``(g + i) % N_GROUPS``, below ``SEQ_FLOOR // POOL_SIZE``, so the fixture satisfies ``-048``'s
    caller precondition -- which item 3 reads with ``-048``'s own reader."""
    return torch.tensor(
        [
            [-1 if (i + g) % PID_SENTINEL_STRIDE == 0 else (g + i) % N_GROUPS
             for g in range(N_GROUPS)]
            for i in range(rows)
        ],
        dtype=torch.int32,
    )


def _max_abs_diff(got: torch.Tensor, want: torch.Tensor) -> int:
    """The largest absolute elementwise gap, as a number to print. NOT a tolerance: every reading
    asserts BIT equality and prints this beside it, so ``0`` is a measured value."""
    return int((got.to(torch.int64) - want.to(torch.int64)).abs().max())


def _partition_numbers(text: str) -> tuple[int, int]:
    """The ``(rows, maximum)`` pair out of a vendor partition-ceiling trap, or a failure saying so.

    Searched anywhere in the text, not anchored, because the trap arrives wrapped in whatever frames the
    dispatch path adds. The two numbers are what say the trap was the ceiling and not another assert.
    """
    found = re.search(VENDOR_PARTITION_NUMBERS, text)
    assert found is not None, (
        f"the trap did not report a partition dimension against a maximum, so it is not the ceiling "
        f"this control is about; the raw text was: {' '.join(text.split())[:400]}"
    )
    return (int(found.group(1)), int(found.group(2)))


def _trap_of(call) -> tuple[str, str]:
    """Run ``call`` expecting it to trap, and return ``(exception type name, text)``.

    ``BaseException`` rather than a named class: the assert caught here is the VENDOR's, and how the
    dispatch wrapper re-raises it is not this file's to declare. The control asserts on the NUMBERS and
    prints the type it saw, so a change in wrapping is visible instead of silent.
    """
    try:
        call()
    except BaseException as exc:  # noqa: BLE001 - the vendor's own trap is the reading
        return (type(exc).__name__, f"{exc}")
    raise AssertionError("the call was expected to trap above the partition ceiling and did not")


# THE PARENT BODY, RE-SPELLED, AND THE WRONG WALK. The replica is the kernel body at `b7b3d80e`
# (`index_expand.py:308-370`, the body's own first statement) with its comments dropped and not one
# call changed; this increment's checker compares it against that commit's bytes, statement by statement.


@nki.jit
def _parent_index_expand_untiled(pool_ids_hbm, seq_lens_hbm, pool_size, pool_mask):
    """The expansion kernel as it stood at ``b7b3d80e``: one tile, so 129 rows or more trap."""
    rows = pool_ids_hbm.shape[0]
    n_groups = pool_ids_hbm.shape[1]
    topk = n_groups * pool_size
    raw_cols = index_expand_raw_width(n_groups, pool_size)
    out_cols = index_expand_width(n_groups, pool_size)

    out = nl.ndarray((rows, out_cols), dtype=nl.int32, buffer=nl.shared_hbm)

    pid = nl.ndarray((rows, n_groups), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=pid, src=nl.load(pool_ids_hbm))
    seq = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=seq, src=nl.load(seq_lens_hbm))

    acc = nl.ndarray((rows, out_cols), dtype=nl.int32, buffer=nl.sbuf)

    if out_cols > raw_cols:
        nisa.memset(acc[:, raw_cols:out_cols], -1)

    for o in range(pool_size):
        vals = nl.ndarray((rows, n_groups), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=vals, data=pid,
                           op0=nl.multiply, operand0=pool_size, op1=nl.add, operand1=o)
        nisa.tensor_scalar(dst=vals, data=vals, op0=nl.maximum, operand0=-1)
        nisa.tensor_copy(dst=acc[:, o:topk:pool_size], src=vals)

    rem = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rem, data=seq, op0=nl.bitwise_and, operand0=pool_mask)
    tail_start = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=tail_start, data1=seq, data2=rem, op=nl.subtract)

    for t in range(pool_size - 1):
        pos = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=pos, data=tail_start, op0=nl.add, operand0=t)
        clipped = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=clipped, data1=pos, data2=seq, op=nl.minimum)
        room = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=room, data1=seq, data2=clipped, op=nl.subtract)
        mask = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=mask, data=room, op0=nl.maximum, operand0=0,
                           op1=nl.minimum, operand1=1)
        pos1 = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=pos1, data=pos, op0=nl.add, operand0=1)
        prod = nl.ndarray((rows, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=prod, data1=pos1, data2=mask, op=nl.multiply)
        col = topk + t
        nisa.tensor_scalar(dst=acc[:, col:col + 1], data=prod, op0=nl.subtract, operand0=1)

    nl.store(out, value=acc)
    return out


@nki.jit
def _hoisted_seq_index_expand(pool_ids_hbm, seq_lens_hbm, pool_size, pool_mask):
    """THE WRONG WALK, and the only thing wrong with it is the hoist.

    It tiles the row axis with the candidate's OWN tile arithmetic, then lifts the per-row length column
    out of the loop, so every tile's tail region is computed from the FIRST tile's lengths. That is the
    plausible bug: ``seq`` is the one per-row operand the tail depends on, hoisting it looks like a
    loop-invariant win, and at one tile the two walks produce identical bits. Item 2 reads both facts.

    DEFINED ONLY FOR A WHOLE NUMBER OF TILES, which :func:`_run_hoisted` asserts, so the hoisted column
    is used with no reslicing and this control introduces no construct the candidate does not use.
    """
    rows = pool_ids_hbm.shape[0]
    n_groups = pool_ids_hbm.shape[1]
    topk = n_groups * pool_size
    raw_cols = index_expand_raw_width(n_groups, pool_size)
    out_cols = index_expand_width(n_groups, pool_size)

    out = nl.ndarray((rows, out_cols), dtype=nl.int32, buffer=nl.shared_hbm)

    tiles = _row_tiles_unchecked(int(rows))
    tile_count = _row_tile_count_unchecked(int(rows))
    first_geom = tiles[0]
    first_height = first_geom[1]

    # THE BUG: read once, off the first tile's rows, and reused by every tile below.
    seq = nl.ndarray((first_height, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=seq, src=nl.load(seq_lens_hbm[0:first_height, 0:1]))

    for idx in range(tile_count):
        tile_geom = tiles[idx]
        start = tile_geom[0]
        height = tile_geom[1]

        pid = nl.ndarray((height, n_groups), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=pid, src=nl.load(pool_ids_hbm[start:start + height, 0:n_groups])
        )

        acc = nl.ndarray((height, out_cols), dtype=nl.int32, buffer=nl.sbuf)

        if out_cols > raw_cols:
            nisa.memset(acc[:, raw_cols:out_cols], -1)

        for o in range(pool_size):
            vals = nl.ndarray((height, n_groups), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=vals, data=pid,
                               op0=nl.multiply, operand0=pool_size, op1=nl.add, operand1=o)
            nisa.tensor_scalar(dst=vals, data=vals, op0=nl.maximum, operand0=-1)
            nisa.tensor_copy(dst=acc[:, o:topk:pool_size], src=vals)

        rem = nl.ndarray((height, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=rem, data=seq, op0=nl.bitwise_and, operand0=pool_mask)
        tail_start = nl.ndarray((height, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=tail_start, data1=seq, data2=rem, op=nl.subtract)

        for t in range(pool_size - 1):
            pos = nl.ndarray((height, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=pos, data=tail_start, op0=nl.add, operand0=t)
            clipped = nl.ndarray((height, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=clipped, data1=pos, data2=seq, op=nl.minimum)
            room = nl.ndarray((height, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=room, data1=seq, data2=clipped, op=nl.subtract)
            mask = nl.ndarray((height, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=mask, data=room, op0=nl.maximum, operand0=0,
                               op1=nl.minimum, operand1=1)
            pos1 = nl.ndarray((height, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=pos1, data=pos, op0=nl.add, operand0=1)
            prod = nl.ndarray((height, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=prod, data1=pos1, data2=mask, op=nl.multiply)
            col = topk + t
            nisa.tensor_scalar(dst=acc[:, col:col + 1], data=prod, op0=nl.subtract, operand0=1)

        nl.store(out[start:start + height, 0:out_cols], value=acc)
    return out


def _run_parent(pool_ids: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
    """Dispatch the parent replica the way the seam dispatches the real kernel."""
    rows = int(pool_ids.shape[0])
    seq_col = seq_lens.reshape(rows, 1).contiguous()
    return wrap_nki(_parent_index_expand_untiled)(
        pool_ids.contiguous(), seq_col, POOL_SIZE, POOL_SIZE - 1
    )


def _run_hoisted(pool_ids: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
    """Dispatch the wrong-walk control, which takes only a whole number of tiles."""
    rows = int(pool_ids.shape[0])
    assert rows % PARTITION_MAX == 0, (
        f"the wrong-walk control takes only a whole number of tiles; got rows={rows}"
    )
    seq_col = seq_lens.reshape(rows, 1).contiguous()
    return wrap_nki(_hoisted_seq_index_expand)(
        pool_ids.contiguous(), seq_col, POOL_SIZE, POOL_SIZE - 1
    )



def test_tiled_admits_the_extents_the_parent_could_not_reach() -> None:
    """Item 1. 132, 256 and 2,048 selected rows are each served in ONE dispatch, and the parent is not.

    THE CONTROL IS IN THIS ITEM because it is what makes the admission mean something: without it, a
    candidate that tiles nothing could pass by the extent never having been tried. It runs the PARENT
    replica at :data:`TRAP_ROWS`, EXTRACTS the two numbers out of the trap and asserts them against this
    file's dials; the wording is printed, not asserted.

    Certifying component (D1.4): ``index_expand._index_expand_nki`` through ``dsa_index_expand``.
    """
    out_cols = index_expand_width(N_GROUPS, POOL_SIZE)
    raw_cols = index_expand_raw_width(N_GROUPS, POOL_SIZE)
    assert out_cols % KEY_CHUNK == 0, (out_cols, KEY_CHUNK)
    _emit('I1_WIDTHS', n_groups=N_GROUPS, pool_size=POOL_SIZE, raw_cols=raw_cols, out_cols=out_cols,
          key_chunk=KEY_CHUNK)

    served = 0
    for rows in ADMISSION_LADDER:
        pool_ids = _pool_ids(rows)
        seq_lens = _seq_lens(rows)
        tiles = row_tiles(rows)
        assert row_tile_count(rows) == len(tiles), (row_tile_count(rows), len(tiles))
        assert len(tiles) > 1, f"rows={rows} must be a MULTI-tile call or this item reads nothing"

        reset_index_expand_dispatch_counters()
        assert can_run_dsa_index_expand(pool_ids, seq_lens, POOL_SIZE) is True
        got = dsa_index_expand(pool_ids, seq_lens, POOL_SIZE)
        assert tuple(got.shape) == (rows, out_cols), tuple(got.shape)
        assert got.dtype is torch.int32, got.dtype
        route = index_expand_dispatch_counters()
        assert route == (1, 0), (rows, route, len(tiles))
        served += 1
        _emit('I1_ADMITTED', rows=rows, tiles=len(tiles), last_height=tiles[-1][1],
              shape=tuple(got.shape), dtype=got.dtype, route=route)

    assert served == len(ADMISSION_LADDER), (served, len(ADMISSION_LADDER))
    _emit("I1_CASES", served=served, of=len(ADMISSION_LADDER), extents=ADMISSION_LADDER)

    # ---- CONTROL: the parent traps at the observed extent, and the NUMBERS are the reading ----
    quoted = VENDOR_PARTITION_ASSERT.format(rows=TRAP_ROWS, pmax=PARTITION_MAX)
    kind, text = _trap_of(lambda: _run_parent(_pool_ids(TRAP_ROWS), _seq_lens(TRAP_ROWS)))
    numbers = _partition_numbers(text)
    assert numbers == (TRAP_ROWS, PARTITION_MAX), (numbers, kind, text)
    _emit('I1_CONTROL_PARENT_TRAPS', rows=TRAP_ROWS, exception=kind, extracted_rows=numbers[0],
          extracted_maximum=numbers[1], wording_matches_the_recorded_run=int(quoted in text),
          raw=' '.join(text.split())[:240])



def test_tiled_is_bit_identical_to_the_parent_kernel_below_the_ceiling() -> None:
    """Item 2. At 128 rows or fewer the tiled kernel IS the parent kernel, bit for bit.

    Tiling is layout and not arithmetic, and this is the reading that says so. The parent's own result is
    compared and so is the torch oracle's, so a replica that had drifted from the parent reddens here
    rather than agreeing with a wrong candidate.

    THE CONTROL IS IN THIS ITEM, pointing the opposite way from item 1's: a walk that HOISTS the per-row
    length column must AGREE at one tile, where the hoist is invisible, and DISAGREE above the ceiling. A
    control that only disagreed would not show the bug is invisible exactly where every landed reading
    lives. Certifying component (D1.4): ``_index_expand_nki`` against ``_parent_index_expand_untiled``
    and ``index_expand._dsa_index_expand_torch``.
    """
    agreed = 0
    for rows in (SMALL_ROWS, PARTITION_MAX):
        assert row_tile_count(rows) == 1, (rows, row_tile_count(rows))
        pool_ids = _pool_ids(rows)
        seq_lens = _seq_lens(rows)

        reset_index_expand_dispatch_counters()
        tiled = dsa_index_expand(pool_ids, seq_lens, POOL_SIZE)
        assert index_expand_dispatch_counters() == (1, 0)
        parent = _run_parent(pool_ids, seq_lens)
        oracle = mod._dsa_index_expand_torch(pool_ids, seq_lens, POOL_SIZE)

        assert torch.equal(tiled, parent), (
            f"rows={rows}: the tiled kernel differs from the parent kernel's own bits"
        )
        assert torch.equal(tiled, oracle), (
            f"rows={rows}: differs from the oracle -- if the parent agreed here, the replica drifted too"
        )
        agreed += 1
        _emit('I2_BIT_IDENTICAL', rows=rows, tiles=row_tile_count(rows), entries=tiled.numel(),
              max_abs_diff_vs_parent=_max_abs_diff(tiled, parent),
              max_abs_diff_vs_oracle=_max_abs_diff(tiled, oracle),
              differing=int((tiled != parent).sum()))

    assert agreed == 2, agreed
    _emit("I2_CASES", agreed=agreed, of=2, extents=[SMALL_ROWS, PARTITION_MAX])

    # ---- CONTROL, part 1: at one tile the hoist is INVISIBLE ----
    rows = PARTITION_MAX
    tiled_one = dsa_index_expand(_pool_ids(rows), _seq_lens(rows), POOL_SIZE)
    hoisted_one = _run_hoisted(_pool_ids(rows), _seq_lens(rows))
    assert torch.equal(hoisted_one, tiled_one), (
        "the wrong walk must be INDISTINGUISHABLE at one tile, or it is not the bug this control is about"
    )
    _emit('I2_CONTROL_HOIST_INVISIBLE_AT_ONE_TILE', rows=rows,
          max_abs_diff=_max_abs_diff(hoisted_one, tiled_one))

    # ---- CONTROL, part 2: above the ceiling it must differ, by a number ----
    big = TILED_ROWS
    tiled_big = dsa_index_expand(_pool_ids(big), _seq_lens(big), POOL_SIZE)
    hoisted_big = _run_hoisted(_pool_ids(big), _seq_lens(big))
    diff = _max_abs_diff(hoisted_big, tiled_big)
    differing_rows = int((hoisted_big != tiled_big).any(dim=1).sum())
    assert diff > 0, "the wrong walk agreed above the ceiling; this control cannot fail"
    assert differing_rows > 0, differing_rows
    first_tile_rows = int(
        (hoisted_big[:PARTITION_MAX] != tiled_big[:PARTITION_MAX]).any(dim=1).sum()
    )
    assert first_tile_rows == 0, "the hoist must be harmless in the tile it reads from"
    _emit('I2_CONTROL_HOIST_DIFFERS_ABOVE_THE_CEILING', rows=big, tiles=row_tile_count(big),
          max_abs_diff=diff, differing_rows=differing_rows, differing_rows_in_first_tile=first_tile_rows)



def test_tiled_matches_the_torch_oracle_at_every_declared_extent() -> None:
    """Item 3. The oracle agrees at all five extents, including the three the parent cannot reach.

    The oracle is upstream's ``where`` form and the kernel is a closed form in max and min, so this is two
    mechanisms arriving at the same bytes. The SENTINEL PADDING past the raw width is read separately as
    all ``-1``, so the pad region cannot pass by being compared only with itself, and the fixture's
    legality is read with ``-048``'s own reader. Certifying component (D1.4): ``_index_expand_nki``
    against ``index_expand._dsa_index_expand_torch``.
    """
    raw_cols = index_expand_raw_width(N_GROUPS, POOL_SIZE)
    out_cols = index_expand_width(N_GROUPS, POOL_SIZE)
    assert out_cols > raw_cols, (out_cols, raw_cols)

    agreed = 0
    for rows in TILED_LADDER:
        pool_ids = _pool_ids(rows)
        seq_lens = _seq_lens(rows)

        violations = _precondition_violations(
            pool_ids.tolist(), seq_lens.tolist(), POOL_SIZE
        )
        assert violations == [], violations

        reset_index_expand_dispatch_counters()
        got = dsa_index_expand(pool_ids, seq_lens, POOL_SIZE)
        want = mod._dsa_index_expand_torch(pool_ids, seq_lens, POOL_SIZE)
        assert torch.equal(got, want), (
            f"rows={rows}: the kernel and the oracle must agree BIT FOR BIT; first differing entry "
            f"{(got != want).nonzero()[:1].tolist()}"
        )
        assert index_expand_dispatch_counters() == (1, 0)

        pad = got[:, raw_cols:out_cols]
        assert int((pad != -1).sum()) == 0, (
            f"rows={rows}: {int((pad != -1).sum())} padded columns are not the sentinel"
        )
        agreed += 1
        _emit('I3_ORACLE_AGREEMENT', rows=rows, tiles=row_tile_count(rows), entries=got.numel(),
              max_abs_diff=_max_abs_diff(got, want), precondition_violations=len(violations),
              pad_columns=pad.shape[1], pad_non_sentinel=0)

    assert agreed == len(TILED_LADDER), (agreed, len(TILED_LADDER))
    _emit("I3_CASES", agreed=agreed, of=len(TILED_LADDER), ladder=TILED_LADDER)



def test_tiled_rows_stay_independent_across_a_tile_boundary() -> None:
    """Item 4. Perturbing one row's ids and length moves that row's output row and NOTHING else.

    TWO EXTENTS, one probe each, because one extent cannot carry both readings:

      * T = 2,048, the registered envelope, probed at the FIRST ROW OF THE SECOND TILE (row 128), which
        is where a boundary bug shows first. This extent is 16 x 128 exactly, so it has NO short tile.
      * T = 2,052, probed at a row INSIDE the 4-row LAST TILE (row 2,050), which is where a remainder bug
        shows. 2,052 is the smallest selecting-regime extent with a short last tile.

    Earlier revisions of this item claimed a short-tile probe at 2,048 and did not have one -- the second
    probe sat in the last FULL tile. The tile a probe lands in is now ASSERTED from the tile list rather
    than described, so the claim and the reading cannot part company again.

    Both inputs are perturbed because they enter by different routes -- the ids as the tile itself, the
    length as the per-row column operand. Each probe row is asserted to have a non-empty tail and at
    least one live pool, so neither case can pass vacuously.
    Certifying component (D1.4): ``_index_expand_nki`` through ``dsa_index_expand``.
    """
    cases = 0
    for rows, want_short in ((TILED_ROWS, False), (TAIL_TILE_ROWS, True)):
        tiles = row_tiles(rows)
        # THE PROBE IS DERIVED FROM THE TILE LIST, never typed: the first row of the second tile, or a
        # row two into the last tile. A typed row number is what let the short-tile claim go stale.
        if want_short:
            probe = tiles[-1][0] + 2
            assert tiles[-1][1] < PARTITION_MAX, (rows, tiles[-1])
            assert tiles[-1][1] == rows % PARTITION_MAX, (rows, tiles[-1])
            assert tiles[-1][0] <= probe < rows, (probe, tiles[-1])
        else:
            probe = tiles[1][0]
            assert tiles[-1][1] == PARTITION_MAX, (rows, tiles[-1])
            assert rows % PARTITION_MAX == 0, rows
        probe_tile = probe // PARTITION_MAX
        assert (probe_tile == len(tiles) - 1) is want_short, (probe, probe_tile, len(tiles))

        pool_ids = _pool_ids(rows)
        seq_lens = _seq_lens(rows)
        base = dsa_index_expand(pool_ids, seq_lens, POOL_SIZE)

        live = int((pool_ids[probe] >= 0).sum())
        tail = int(seq_lens[probe]) % POOL_SIZE
        assert live > 0, f"probe row {probe} selects no pool at all"
        assert tail > 0, f"probe row {probe} has an empty tail, so a length change may not show"

        moved_ids = pool_ids.clone()
        moved_ids[probe, 0] = (int(pool_ids[probe, 0]) + 1) % N_GROUPS
        moved_lens = seq_lens.clone()
        moved_lens[probe] = SEQ_FLOOR + ((int(seq_lens[probe]) - SEQ_FLOOR + 1) % SEQ_PERIOD)
        assert int(moved_lens[probe]) != int(seq_lens[probe])

        got = dsa_index_expand(moved_ids, moved_lens, POOL_SIZE)
        assert not torch.equal(got[probe], base[probe]), \
            f"probe row {probe} did not move when its own ids and length did"
        keep = torch.ones(rows, dtype=torch.bool)
        keep[probe] = False
        assert torch.equal(got[keep], base[keep]), f"perturbing row {probe} moved another row's bits"
        want = mod._dsa_index_expand_torch(moved_ids, moved_lens, POOL_SIZE)
        assert torch.equal(got, want)
        cases += 1
        _emit("I4_ROW_INDEPENDENCE", rows=rows, probe=probe, tile=probe_tile, tiles=len(tiles),
              probe_tile_height=tiles[probe_tile][1], tile_is_short=int(want_short),
              live_pools=live, tail_tokens=tail,
              moved_rows=int((got != base).any(dim=1).sum()),
              other_rows_bit_identical=int(keep.sum()),
              max_abs_diff_vs_oracle=_max_abs_diff(got, want))

    assert cases == 2, cases
    _emit("I4_PROBES", cases=cases, extents=[TILED_ROWS, TAIL_TILE_ROWS],
          tiles=[row_tile_count(TILED_ROWS), row_tile_count(TAIL_TILE_ROWS)],
          remainders=[TILED_ROWS % PARTITION_MAX, TAIL_TILE_ROWS % PARTITION_MAX])



def test_tiled_route_is_one_dispatch_standalone_and_along_the_device_chain() -> None:
    """Item 5. Route form R-1 at a tiled extent, and the whole DSA chain on device above 128 rows.

    STANDALONE: many tiles, one dispatch, zero fallbacks, and the kernel identity read back through the
    seam. A host-side loop over 128-row slices would serve the same shapes and read ``len(tiles)``
    dispatches, which is the design P13 excludes and the number that tells the two apart.

    ALONG THE CHAIN IS WHERE ``-103b``'S RECORDED GAP CLOSES: it fed its sentinel from a host
    ``torch.topk`` on purpose and recorded that no reading had run the DEVICE selector above 128 rows.
    Here the four stages run in order -- bound, select, sentinel, expand -- at 132 and 256 rows, and
    EVERY ONE of them is asserted to have taken its kernel: one dispatch, no fallback, per stage per
    extent. That is only askable because commit 3 merged ``-103b``'s tiled bound and sentinel onto this
    tree; before it, the chain trapped in its first seam. Nothing here substitutes a torch reference for
    a kernel that refused -- a guarded chain would be a hollow chain, and a stage that cannot serve these
    extents is a finding about THAT kernel. Certifying component (D1.4): the four seams' own counters and
    ``index_expand_kernel_identity``.
    """
    rows = TILED_ROWS
    tiles = row_tile_count(rows)
    assert tiles > 1, tiles

    reset_index_expand_dispatch_counters()
    assert index_expand_kernel_identity() is None
    pool_ids = _pool_ids(rows)
    seq_lens = _seq_lens(rows)
    assert can_run_dsa_index_expand(pool_ids, seq_lens, POOL_SIZE) is True
    dsa_index_expand(pool_ids, seq_lens, POOL_SIZE)
    route = index_expand_dispatch_counters()
    assert route == (1, 0), (route, tiles)
    identity = index_expand_kernel_identity()
    assert identity is not None
    assert identity[1] == "_index_expand_nki", identity
    assert identity[0].endswith("dsa.index_expand"), identity
    _emit("I5_ROUTE_STANDALONE", rows=rows, tiles=tiles, nki_dispatch=route[0],
          torch_fallback=route[1], kernel="/".join(identity))

    dsa_index_expand(pool_ids, seq_lens, POOL_SIZE)
    assert index_expand_dispatch_counters() == (2, 0)
    _emit("I5_PER_CALL", second_call=index_expand_dispatch_counters()[0], tiles=tiles)

    # ---- THE CHAIN, ABOVE THE CEILING, EVERY STAGE ON DEVICE ----
    selector_on_device = 0
    for chain_rows in CHAIN_LADDER:
        assert row_tile_count(chain_rows) > 1, chain_rows
        seq = _seq_lens(chain_rows)
        clen = seq.reshape(chain_rows, 1).contiguous()
        gen = torch.Generator().manual_seed(1030 + chain_rows)
        scores = torch.randn(chain_rows, CHAIN_WIDTH, generator=gen, dtype=torch.float32) * 0.05

        reset_causal_bound_dispatch_counters()
        reset_topk_select_dispatch_counters()
        reset_causal_sentinel_dispatch_counters()
        reset_index_expand_dispatch_counters()

        # STAGE 1, the bound: it fills every pool a row's own length does not complete, so the selector
        # cannot pick one and the expansion's caller precondition holds by construction.
        assert can_run_dsa_causal_bound(scores, clen, POOL_SIZE) is True
        bounded = dsa_causal_bound(scores, clen, POOL_SIZE)
        assert causal_bound_dispatch_counters() == (1, 0), causal_bound_dispatch_counters()

        # STAGE 2, the vendored selector. Its gate is ASSERTED, not merely read: on this tree it must
        # serve this geometry, and a refusal is a finding about that kernel, not a reason to fall back.
        assert can_run_dsa_topk_select(bounded, N_GROUPS) is True
        values, indices = dsa_topk_select(bounded, N_GROUPS)
        assert topk_select_dispatch_counters() == (1, 0), topk_select_dispatch_counters()
        selector_on_device += 1

        # STAGE 3, the sentinel: anything the selector returned at or above the real pool width is a
        # pad it invented, and this turns that into a -1 the expansion understands.
        idx32 = indices.to(torch.int32).contiguous()
        vals = values.contiguous()
        assert can_run_dsa_causal_sentinel(vals, idx32, CHAIN_WIDTH) is True
        marked = dsa_causal_sentinel(vals, idx32, CHAIN_WIDTH)
        assert causal_sentinel_dispatch_counters() == (1, 0), causal_sentinel_dispatch_counters()

        # STAGE 4, THE INCREMENT UNDER TEST, on the ids the chain really produced.
        assert can_run_dsa_index_expand(marked, seq, POOL_SIZE) is True
        expanded = dsa_index_expand(marked, seq, POOL_SIZE)
        assert index_expand_dispatch_counters() == (1, 0), index_expand_dispatch_counters()
        assert tuple(expanded.shape) == (chain_rows, index_expand_width(N_GROUPS, POOL_SIZE))
        assert expanded.dtype is torch.int32, expanded.dtype

        # The chain's own ids must satisfy -048's precondition, read with -048's reader.
        violations = _precondition_violations(marked.tolist(), seq.tolist(), POOL_SIZE)
        assert violations == [], violations[:8]
        want = mod._dsa_index_expand_torch(marked, seq, POOL_SIZE)
        assert torch.equal(expanded, want), \
            f"rows={chain_rows}: the expansion of the chain's own ids differs from the oracle's"
        _emit("I5_CHAIN", rows=chain_rows, tiles=row_tile_count(chain_rows),
              bound=causal_bound_dispatch_counters(), select=topk_select_dispatch_counters(),
              sentinel=causal_sentinel_dispatch_counters(), expand=index_expand_dispatch_counters(),
              sentinel_ids=int((marked == -1).sum()), live_ids=int((marked >= 0).sum()),
              precondition_violations=len(violations),
              max_abs_diff_vs_oracle=_max_abs_diff(expanded, want))

    assert selector_on_device == len(CHAIN_LADDER), selector_on_device
    _emit("I5_CHAIN_CASES", extents=CHAIN_LADDER, stages=4, of=len(CHAIN_LADDER),
          device_selector_above_128=selector_on_device,
          gap="-103b's gap is closed: the device selector ran above 128 rows at every extent")
