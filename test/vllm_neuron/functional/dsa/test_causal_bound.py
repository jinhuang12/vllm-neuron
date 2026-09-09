# SPDX-License-Identifier: Apache-2.0
"""Acceptance for the selecting-regime causal bound -- ``inc-glm53f-103``.

WHAT IS BEING ASSERTED. A query row must select only key pools that are COMPLETE at or before its
own position, and a selection that reached no such pool must read the ``-1`` sentinel. The bound
writes ``-inf``; the sentinel writes ``-1``; nothing else in either output moves.

FOUR ITEMS, ONE PER COUNTED CONJUNCT, NO ``parametrize`` -- section 6 rule 6, so the declared count
is derivable before a line runs. Controls live INSIDE the item whose zero or whose comparison they
protect, on the ``design-20260905`` §63 precedent: a strengthening under the same id never moves a
declared item count.

WHAT EACH READING IS WORTH, AND WHERE A TOLERANCE IS AND IS NOT SPENT.
  * Items 1, 2 and 4 are BIT COMPARISONS. The bound moves a column to ``-inf`` or leaves it alone,
    and the sentinel moves an index to ``-1`` or leaves it alone; neither computes a new number, so
    there is nothing for a tolerance to absorb and none is authored. Item 1 reads the kept columns
    through ``.view(torch.int32)`` -- the RAW BITS -- because ``==`` on floats cannot tell ``-0.0``
    from ``+0.0`` and an arithmetic mask would silently rewrite exactly that.
  * Item 3's LAST reading is the only one with a tolerance, because it runs a float attention kernel.
    That pair is NOT authored here: ``RTOL`` and ``ATOL`` are IMPORTED from ``inc-glm53f-098``'s
    acceptance file, which is what "the pair ``-098``'s item (1) carries (cited, not restated)"
    asks for. A second spelling of a registered comparator value is a place for it to drift (P9).

TWO REFERENCES ARE IMPORTED RATHER THAN RE-IMPLEMENTED, and the block names both.
  * ``_precondition_violations`` from ``inc-glm53f-048``'s acceptance file, which is the reader that
    DEFINES the precondition this block exists to restore. Re-spelling it here would let this file
    pass against its own idea of ``-048``'s rule instead of against ``-048``'s rule.
  * ``sentinel_reference`` and the ``RTOL``/``ATOL`` pair from ``inc-glm53f-098``'s acceptance file.

THE ZEROS AND THE COMPARISONS OWN FIRING CONTROLS.
  * item 1's bit equality is shown able to FAIL by a doctored oracle off by one at each row's LAST
    COMPLETE pool -- the single boundary column a wrong inequality would move;
  * item 1 PLANTS a ``-0.0`` in a kept column, so the bit-identity reading is taken on the one value
    that separates "left alone" from "added to zero";
  * item 2 reads the selector's returned VALUE for a bounded slot before reading the sentinel, so a
    sentinel that fired for the wrong reason cannot pass (see that item on why this is not pedantry);
  * item 3's precondition zero is read beside the UNBOUNDED chain on the same inputs, which must
    violate;
  * item 4 moves BOTH entry points' ``torch_fallback`` off 0 by forcing ``can_run_kernel`` False,
    so the zeros items 1 to 3 read are readings and not decoration (D1.5).

WHY THE ORACLES ARE REAL ORACLES. The kernel never divides -- it compares
``(p + 1) * pool_size > causal_len`` -- and the bound oracle uses upstream's own
``masked_fill`` over ``p >= causal_len // pool_size``. The marker kernel starts from an all-``-1``
tile and copies an index IN where a KEEP mask holds; its oracle builds the MARK mask directly out of
``le``, ``isnan`` and ``ge``. Two different mechanisms arriving at the same bits is an agreement; one
mechanism written twice would only prove the module agrees with itself.

WHAT CHANGED IN ``103r5``, so a reader of this file is not surprised by the constants. The bound used
to write ``-inf`` and the marker used to key on it exactly. It now writes the FINITE
:data:`BOUND_FILL` and the marker fires on two arms -- a value at or below ``BOUND_FILL_MARK``, OR an
index at or past the real ``width``, which is how a pad column the selector invented is caught. The
reason is recorded with its bytes in
``increments/contradiction-103-selector-pad-6874a0f5.md``.
"""

import os
from pathlib import Path

import pytest
import torch

from vllm_neuron.functional.attention.mla_sparse import (
    can_run_mla_sparse_attention,
    mla_sparse_attention,
)
from vllm_neuron.functional.dsa import causal_bound as mod
from vllm_neuron.functional.dsa.causal_bound import (
    BOUND_FILL,
    BOUND_FILL_MARK,
    SENTINEL,
    DsaCausalBoundError,
    can_run_dsa_causal_bound,
    can_run_dsa_causal_sentinel,
    causal_bound_dispatch_counters,
    causal_bound_kernel_identity,
    causal_sentinel_dispatch_counters,
    causal_sentinel_kernel_identity,
    dsa_causal_bound,
    dsa_causal_bound_torch_oracle,
    dsa_causal_sentinel,
    dsa_causal_sentinel_torch_oracle,
    reset_causal_bound_dispatch_counters,
    reset_causal_sentinel_dispatch_counters,
)
from vllm_neuron.functional.dsa.index_expand import (
    can_run_dsa_index_expand,
    dsa_index_expand,
    index_expand_dispatch_counters,
    index_expand_width,
    reset_index_expand_dispatch_counters,
)
from vllm_neuron.functional.dsa.topk_select import (
    can_run_dsa_topk_select,
    dsa_topk_select,
    reset_topk_select_dispatch_counters,
    topk_select_dispatch_counters,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

# `-048`'s OWN precondition reader and `-098`'s OWN reference and tolerance pair. Imported, never
# re-implemented -- see this file's docstring.
from test.vllm_neuron.functional.attention.test_mla_sparse import (
    ATOL,
    RTOL,
    make_case,
    sentinel_reference,
)
from test.vllm_neuron.functional.dsa.test_index_expand import (
    _precondition_violations,
)

ROWS = 5
"""Query rows in the declared small shape -- the block's own figure."""

POOL_COLUMNS = 16
"""Candidate pool columns in the declared small shape -- the block's own figure."""

POOL_SIZE = 4
"""Tokens per pool in the declared small shape -- the block's own figure."""

CAUSAL_LENS = [1, 4, 9, 33, 64]
"""The declared causal lengths -- the block's own set, and every one earns its place.

``1`` is shorter than one pool, so NO pool is complete and the whole row is bounded -- the case a
closed form that assumed at least one live column would get wrong. ``4`` is exactly one pool, the
boundary where the inequality is tight. ``9`` is one token past two pools, so the third pool is
incomplete and must be bounded even though the row reaches into it. ``33`` is interior. ``64`` is
``POOL_COLUMNS * POOL_SIZE``, the saturated row where NOTHING is bounded and the count must read
exactly zero -- so a bound that fired unconditionally could not pass."""

SELECT_K = 2
"""The block's declared ``k`` for the sentinel and integration items."""

MULTIFOLD_SELECT_K = 16
"""The ``k`` that puts the landed selector on its STRIKING branch -- repair ``103r4``.

Review finding F1 (``reviews/glm-5.3-flash-port/bless-103-code-c81113a2-findings.md``) is about a
branch no reading at ``SELECT_K`` can reach. WHICH BRANCH THAT IS, CORRECTED IN ``103r5`` after the
fresh read of ``6874a0f5`` (record-only finding R2 in
``reviews/glm-5.3-flash-port/bless-103-code-6874a0f5-findings.md``): at rows 5, width 32, k 16 the
factory's small-``k`` heuristic sets ``n_stages = min(max_n_stages, div_ceil(min(k, vocab), 8)) = 2``
(``rotational_topk_utils.py:406-421``), so the call takes the ROTATIONAL path, not the scanning one,
and ``topk_core`` is called once per stage with ``local_top_k_per_stage = 8``
(``rotational_topk.py:354``). ``8 % 8 == 0``, so that single fold takes the ``else`` branch that
STRIKES each taken value to ``-inf`` IN THE INPUT BUFFER
(``rotational_topk_utils.py:1053-1066``), and the sorted finish strikes twice more in ``sort()``
(``:1096-1118``). Later passes then read struck buffers, which is what this case needs. The earlier
account here -- "two folds of ``topk_core(k=16)``" -- described the ``n_stages == 1`` scanning path
this geometry does not take, and the claim "16 is the smallest such ``k``" was wrong on its own
terms: ``k = 8`` gives ``n_stages = 1``, one fold, ``8 % 8 == 0`` and the same striking branch, so
only ``k`` in 1..7 never strikes. 16 is kept because it also leaves four rows with fewer complete
pools than ``k``. Production is ``index_topk // index_kpool`` = 2048 // 4 = 512, above the factory's
small-``k`` threshold (``rotational_topk_utils.py:403``, ``:517``), so production runs the rotational
path with ``n_stages >= 2`` and a per-stage ``k`` that is a multiple of 8 -- always striking."""

MULTIFOLD_POOL_COLUMNS = 32
"""Candidate columns for the multi-fold case. Two clauses fix it rather than taste:
``can_run_dsa_topk_select`` needs ``0 < k < width`` STRICTLY (``topk_select.py:294``) and the
kernel's own factory needs ``vocab_size >= k`` (``rotational_topk_utils.py:241-243``; ``:240`` is the
SEPARATE 2D-shape assert, so citing it here would name the wrong clause). 32 satisfies
both with room, and keeps every column reachable by a 128-token row."""

MULTIFOLD_CAUSAL_LENS = [64, 36, 20, 12, 4]
"""Lengths that complete 16, 9, 5, 3 and 1 pools -- exactly ``MULTIFOLD_SELECT_K`` on the FIRST
row and fewer on the other four.

A row completing fewer pools than ``k`` is the whole point: the selector must fill the remaining
slots from an all-fill buffer (``BOUND_FILL``, a finite ``-1e30`` since ``-103``; the earlier
wording here said ``-inf`` and was stale), which is where the strike substitution happens. The
first row completes exactly ``k`` pools and must show ZERO sentinels, so a case that sentinelised
unconditionally could not pass.

WHY THE ORDER IS DESCENDING, AND IT IS NOT COSMETIC (rev 268). The ascending order this list
carried until rev 268 put the two rows with more than ``topk_per_stage`` real candidates at rows
3 and 4 -- which are exactly the rows the seam's SECOND program holds, and that program's tile is
the ragged one. So "the selector breaks above the per-stage count" and "the selector breaks on a
ragged tile" predicted the same two failing rows, and the reading could not tell them apart. It
was read as the former and the mechanism was the latter (`design-20260909-bh`). Descending puts
the high-count rows in the FIRST program, whose tile is full, so the two explanations now predict
DIFFERENT rows and this case can no longer be read either way."""

PAD_POOL_COLUMNS = 33
"""Candidate columns for the PAD case -- an ODD width, which is what makes the fold uneven.

``103r5``. The vendored loader folds the candidate axis into ``n_stages`` partitions of
``n_folded = ceil(width / n_stages)`` columns and pads the LAST fold's tail with a finite
``-9948.0`` (``cascaded_max_utils.py:61-66``, ``:154-158``). Pads exist exactly when
``width != n_stages * n_folded``, which the kernel's own config reports as
``padded_vocab_size - vocab_size`` (``rotational_topk_utils.py:433-434``). At ``n_stages = 2`` any
odd width gives exactly one pad column, at global index ``width`` -- one past the last real pool.
The case READS that number off the config rather than assuming it, so a factory that folds evenly
here reddens with a dial finding instead of passing vacuously."""

MLA_CASE = dict(seq=ROWS, heads=4, latent=128, topk=128, s_kv=256, rope=0)
"""The geometry item 3's chain ends in, taken from ``-098``'s own declared sentinel family.

``topk=128`` is not a free choice: it is ``index_expand_width(SELECT_K, POOL_SIZE)``, which item 3
ASSERTS rather than assumes, so this dict cannot drift from what the chain actually emits. ``s_kv``
only has to exceed the largest token index the chain can produce, which is ``max(CAUSAL_LENS) - 1``;
item 3 reads that too."""


def _emit(tag: str, **values: object) -> None:
    """Print one MACHINE-READABLE reading line, for the driver to re-check independently.

    The pattern is the landed one at ``test_index_expand.py:110-118``: the item asserts, and then
    PRINTS the value it asserted on, so the driver that owns the transcript can check the same
    number without trusting this file's own verdict.

    Note for whoever greps the transcript: pytest writes a progress marker with NO trailing newline
    after each test, so the first line printed by every test after the first can be prefixed by it.
    Match with ``grep -o``, never with a ``^`` anchor.
    """
    body = " ".join(f"{k}={v}" for k, v in values.items())
    print(f"S103|{tag}|{body}", flush=True)


def _scores(seed: int) -> torch.Tensor:
    """One case's scores at the declared shape: ``[ROWS, POOL_COLUMNS]`` float32.

    Seeded from the caller so a failure reproduces from its own item. Scaled small for the same
    reason ``-098``'s fixture is: it feeds a softmax in item 3.
    """
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(ROWS, POOL_COLUMNS, generator=gen, dtype=torch.float32) * 0.05


def _causal_len() -> torch.Tensor:
    """The declared lengths at the shape the seam documents: ``[ROWS, 1]`` int32."""
    return torch.tensor(CAUSAL_LENS, dtype=torch.int32).reshape(ROWS, 1)


def _complete_pools() -> list[int]:
    """Complete pools per row, COMPUTED from this file's own dials and never typed.

    Every expected count in items 1 and 2 is derived from this, so changing a dial moves the reading
    instead of leaving it accidentally true.
    """
    return [c // POOL_SIZE for c in CAUSAL_LENS]


def _assert_module_under_test_is_the_candidate() -> str:
    """Assert the module being measured is the candidate tree, and return where it resolved.

    §78.1's obligation in the form ``inc-glm53f-056``'s repair settled (DECISIONS §88, §91 i): the
    declared-root arm binds when ``GLM53F_CANDIDATE_ROOT`` is set, and the root is DERIVED from this
    file's own tree when it is not. A test that requires the campaign harness's environment variable
    is red by construction on every plain ``pytest`` run of the fork, which is the landed defect
    that repair exists to remove.
    """
    resolved = Path(mod.__file__).resolve()
    declared = os.environ.get("GLM53F_CANDIDATE_ROOT")
    if declared:
        root = Path(declared).resolve()
        origin = "declared"
    else:
        root = Path(__file__).resolve().parents[4]
        origin = "derived-from-this-file"
    assert resolved.is_relative_to(root), (
        f"the module under test must come from the candidate tree {root}; it resolved to {resolved}"
    )
    assert (root / "vllm_neuron" / "__init__.py").is_file(), (
        f"the candidate root {root} must contain the vllm_neuron package being measured"
    )
    _emit("IMPORT_ORIGIN", origin=origin, root=root, module=resolved)
    return origin


# =========================================================================== #
# CONJUNCT 1 -- THE EXACT BOUND
# =========================================================================== #


def test_the_bound_fills_exactly_the_incomplete_pools_and_a_doctored_oracle_fails() -> None:
    """Conjunct 1. ``BOUND_FILL`` at exactly the pools the row does not complete, nothing else touched.

    FOUR READINGS IN ONE ITEM, because each alone leaves a hole. Bit equality against the oracle
    certifies the rearranged inequality against upstream's floor-division spelling. The per-row
    FILL COUNT, computed from this file's dials, fails on an off-by-one that both spellings could
    share. The KEPT-COLUMN bit identity certifies that the mask is a select and not arithmetic. The
    doctored oracle certifies the COMPARISON, so a pass means the comparison was able to fail.

    Certifying component (D1.4): ``causal_bound._causal_bound_nki`` through the
    ``causal_bound.dsa_causal_bound`` seam.
    """
    _assert_module_under_test_is_the_candidate()
    scores = _scores(103)
    causal_len = _causal_len()

    # THE PLANTED `-0.0`. Row 4 is the saturated row, so column 0 there is KEPT -- and `-0.0` is the
    # one value an additive mask would rewrite while every `==` comparison still passed. Planted
    # BEFORE the call so the kernel sees it, and read back through the raw bits below.
    scores[4, 0] = -0.0
    planted_bits = int(scores[4, 0].view(torch.int32))
    assert planted_bits == -2147483648, planted_bits  # 0x80000000, the sign bit alone
    _emit("C1_PLANTED_NEGATIVE_ZERO", row=4, column=0, bits=planted_bits)

    reset_causal_bound_dispatch_counters()
    admitted = can_run_dsa_causal_bound(scores, causal_len, POOL_SIZE)
    assert admitted is True, (
        "the declared case must take the NKI route, or no reading below is one. Under the Tier N "
        "harness a False here means NKI_SIMULATOR=1 is unset"
    )
    got = dsa_causal_bound(scores, causal_len, POOL_SIZE)
    want = dsa_causal_bound_torch_oracle(scores, causal_len, POOL_SIZE)
    nki_n, fallback_n = causal_bound_dispatch_counters()

    assert tuple(got.shape) == (ROWS, POOL_COLUMNS), tuple(got.shape)
    assert got.dtype is torch.float32, got.dtype
    assert want.dtype is torch.float32, want.dtype

    # BIT EQUALITY, not `assert_close`. `torch.equal` on the raw int32 view sees a `+0.0` where a
    # `-0.0` belongs, which `assert_close` at any tolerance would not.
    assert torch.equal(got.view(torch.int32), want.view(torch.int32)), (
        "the kernel and the oracle must agree BIT FOR BIT; the first differing entry is "
        f"{(got.view(torch.int32) != want.view(torch.int32)).nonzero()[:1].tolist()}"
    )
    _emit("C1_BIT_EQUALITY", rows=ROWS, columns=POOL_COLUMNS, entries=got.numel(),
          differing_bits=0)

    # THE PER-ROW FILL COUNT, computed from the dials. This is the reading a shared off-by-one
    # cannot survive: it compares the kernel against ARITHMETIC, not against another spelling.
    complete = _complete_pools()
    per_row = (got == BOUND_FILL).sum(dim=1).to(torch.int64)
    expected = torch.tensor(
        [POOL_COLUMNS - min(POOL_COLUMNS, c) for c in complete], dtype=torch.int64
    )
    assert torch.equal(per_row, expected), (per_row.tolist(), expected.tolist())
    _emit("C1_BOUNDED_COUNT", lengths=CAUSAL_LENS, complete_pools=complete,
          counts=per_row.tolist(), computed=expected.tolist())

    # AND AGAINST THE BLOCK'S OWN DECLARED FIGURE, which is a second, independent source for the
    # same five numbers. If the computation above drifted from the design, this disagrees.
    assert per_row.tolist() == [16, 15, 14, 8, 0], per_row.tolist()
    _emit("C1_MATCHES_BLOCK_FIGURE", declared=[16, 15, 14, 8, 0], read=per_row.tolist())

    # THE SATURATED ROW carries ZERO. §79.1: the subset is asserted NON-EMPTY as well as correct, so
    # a selector that matched nothing cannot pass as a vacuous truth.
    saturated = [r for r, c in enumerate(CAUSAL_LENS) if c >= POOL_COLUMNS * POOL_SIZE]
    assert len(saturated) == 1, saturated
    assert saturated == [4], saturated
    assert int(per_row[4]) == 0, int(per_row[4])
    assert CAUSAL_LENS[4] == POOL_COLUMNS * POOL_SIZE, CAUSAL_LENS[4]
    _emit("C1_SATURATED", row=4, causal_len=CAUSAL_LENS[4],
          width_times_pool=POOL_COLUMNS * POOL_SIZE, bounded=int(per_row[4]))

    # The complement is non-empty too, or the saturated reading would be the only one taken.
    bounded_rows = [r for r in range(ROWS) if r not in saturated]
    assert len(bounded_rows) == 4, bounded_rows
    assert all(int(per_row[r]) > 0 for r in bounded_rows), per_row.tolist()
    _emit("C1_BOUNDED_ROWS", rows=len(bounded_rows), which=bounded_rows)

    # THE KEPT COLUMNS ARE BIT-IDENTICAL TO THE INPUT. This is what `tensor_copy_predicated` buys
    # over an arithmetic mask, and it is read on RAW BITS so the planted `-0.0` is in scope.
    kept_mask = torch.zeros(ROWS, POOL_COLUMNS, dtype=torch.bool)
    for r, c in enumerate(complete):
        kept_mask[r, : min(POOL_COLUMNS, c)] = True
    assert int(kept_mask.sum()) == sum(min(POOL_COLUMNS, c) for c in complete)
    assert int(kept_mask.sum()) > 0, "the kept set must be non-empty for this reading to exist"
    assert torch.equal(
        got.view(torch.int32)[kept_mask], scores.view(torch.int32)[kept_mask]
    ), "a kept column was rewritten; the mask is not a select"
    read_back = int(got[4, 0].view(torch.int32))
    assert read_back == planted_bits, (read_back, planted_bits)
    _emit("C1_KEPT_BIT_IDENTICAL", kept=int(kept_mask.sum()), population=got.numel(),
          negative_zero_survived=int(read_back == planted_bits))

    # THE DOCTORED-ORACLE CONTROL. Off by one at each row's LAST COMPLETE pool: exactly the column a
    # `>=` written as a `>` would move. One entry per row that HAS a complete pool.
    doctored = want.clone()
    moved = 0
    for r, c in enumerate(complete):
        if c > 0:
            doctored[r, min(POOL_COLUMNS, c) - 1] = BOUND_FILL
            moved += 1
    assert moved == 4, moved  # every row but the wholly-bounded row 0
    differing = int((doctored != got).sum())
    assert not torch.equal(doctored, got), (
        "the doctored oracle must DISAGREE, or the equality assertion above proves nothing"
    )
    assert differing == moved, (differing, moved)
    _emit("C1_DOCTORED_CONTROL", doctored_rows=moved, differing=differing,
          population=got.numel())

    identity = causal_bound_kernel_identity()
    assert identity is not None, "the identity must be derived by TAKING the dispatch branch (D13.1)"
    assert identity[1] == "_causal_bound_nki", identity
    assert (nki_n, fallback_n) == (1, 0), (nki_n, fallback_n)
    _emit("C1_ROUTE", can_run=admitted, nki_dispatch=nki_n, torch_fallback=fallback_n,
          kernel=identity[1])


# =========================================================================== #
# CONJUNCT 2 -- THE SENTINEL
# =========================================================================== #


def test_the_sentinel_marks_every_bounded_selection_and_no_other_index() -> None:
    """Conjunct 2. ``-1`` where the selected value is ``-inf``, and the index untouched elsewhere.

    THE READING THIS ITEM TAKES FIRST, AND WHY IT IS NOT PEDANTRY. The marker's value arm keys on the
    selected VALUE, so this conjunct depends on a filled column coming back at or below
    ``BOUND_FILL_MARK``. That is a LOAD-BEARING fact about someone else's kernel, so it is READ HERE,
    before anything depends on it, and named in the transcript. If it fails, the finding is about the
    selector's contract and not about this module's arithmetic -- report it as such rather than
    widening anything.

    WHY THAT READING IS NOW ROBUST WHERE THE OLD ONE WAS NOT. Until ``103r5`` the fill was ``-inf``
    and this item read it back EXACTLY. The selector cannot promise that: it moves selected values
    across partitions with a 0/1 permutation matmul (``rotational_topk.py:385`` into
    ``rotational_topk_utils.py:867-886``) where ``0 * -inf`` is NaN, and it uses ``-inf`` itself as
    its own already-taken marker (``rotational_topk_utils.py:1065``). :data:`BOUND_FILL` is finite, so
    it crosses that matmul as itself, and the mark THRESHOLD is a decade closer to zero than the fill,
    so even a perturbed round trip is still caught. The record with every link at the bytes is
    ``increments/contradiction-103-selector-pad-6874a0f5.md``.

    THIS ITEM HAS THREE CASES AFTER REPAIR ``103r5``. The first is the declared small shape at
    ``SELECT_K``. The second runs the same readings at ``MULTIFOLD_SELECT_K``, where the selector
    strikes its own input -- the branch review finding F1 is about, and the one the retired
    ``bounded.gather`` form got wrong. The third runs them at ``PAD_POOL_COLUMNS``, an ODD width, so
    the selector pads its own input and hands back an index PAST the last real pool -- the defect
    ``103r5`` repairs, and the one no value test can see. It is one item and three cases rather than
    three items, because the conjunct is the same conjunct; the plan declares four items and there
    are still four.

    Certifying component (D1.4): ``causal_bound._causal_sentinel_nki`` through the
    ``causal_bound.dsa_causal_sentinel`` seam.
    """
    scores = _scores(203)
    causal_len = _causal_len()
    complete = _complete_pools()

    reset_causal_bound_dispatch_counters()
    reset_causal_sentinel_dispatch_counters()
    reset_topk_select_dispatch_counters()

    bounded = dsa_causal_bound(scores, causal_len, POOL_SIZE)
    assert can_run_dsa_topk_select(bounded, SELECT_K) is True, (
        "the landed selector must serve the declared shape, or this conjunct has no chain to read"
    )
    values, indices = dsa_topk_select(bounded, SELECT_K)
    assert tuple(values.shape) == (ROWS, SELECT_K), tuple(values.shape)

    # THE LOAD-BEARING FACT, read before anything depends on it. A bounded slot is one the row had
    # no complete pool for; its count per row is `max(0, k - complete)`, which is the block's own
    # formula and is computed here rather than typed.
    want_counts = [max(0, SELECT_K - c) for c in complete]
    assert want_counts == [2, 1, 0, 0, 0], want_counts  # the block's declared figure, cross-checked
    is_filled = values <= BOUND_FILL_MARK
    got_filled = is_filled.sum(dim=1).to(torch.int64)
    assert torch.equal(got_filled, torch.tensor(want_counts, dtype=torch.int64)), (
        f"the selector did not return a filled value for every bounded slot: got "
        f"{got_filled.tolist()} against the computed {want_counts}. This is a reading about "
        f"dsa_topk_select's returned VALUES, not about this module's mask -- see this item's "
        f"docstring before changing anything here"
    )
    _emit("C2_SELECTOR_RETURNS_THE_FILL_BELOW_THE_MARK", per_row=got_filled.tolist(),
          computed=want_counts, total=int(is_filled.sum()),
          mark=BOUND_FILL_MARK, fill=BOUND_FILL)

    # THE SENTINEL. `dsa_topk_select` returns int64 indices to match `torch.topk`; the consumer
    # `dsa_index_expand` reads int32, so the cast is the dispatch site's transport and is spelled
    # the same way here. It is dtype plumbing, not a torch implementation of anything.
    idx32 = indices.to(torch.int32)
    assert can_run_dsa_causal_sentinel(values, idx32, POOL_COLUMNS) is True
    got = dsa_causal_sentinel(values, idx32, POOL_COLUMNS)
    want = dsa_causal_sentinel_torch_oracle(values, idx32, POOL_COLUMNS)
    sent_nki, sent_fb = causal_sentinel_dispatch_counters()

    assert got.dtype is torch.int32, got.dtype
    assert torch.equal(got, want), (
        f"the kernel and the oracle must agree element for element; first differing "
        f"{(got != want).nonzero()[:1].tolist()}"
    )
    _emit("C2_ORACLE_AGREEMENT", entries=got.numel(), differing=0)

    per_row = (got == SENTINEL).sum(dim=1).to(torch.int64)
    assert torch.equal(per_row, torch.tensor(want_counts, dtype=torch.int64)), (
        per_row.tolist(), want_counts
    )
    assert int(per_row.numel()) == ROWS, per_row.numel()
    _emit("C2_SENTINEL_COUNT", rows=ROWS, of=ROWS, counts=per_row.tolist(),
          computed=want_counts)

    # "AND NOWHERE ELSE", as two halves. Every sentinel position is a `-inf` position, and every
    # non-sentinel index is the selector's own index unchanged.
    assert torch.equal(got == SENTINEL, is_filled), (
        "a sentinel was written at a position whose value was a real score, or withheld at one "
        "whose value was a fill"
    )
    kept = ~is_filled
    assert int(kept.sum()) > 0, "the kept set must be non-empty for this reading to exist"
    assert torch.equal(got[kept], idx32[kept]), "a kept index was rewritten"
    _emit("C2_NOWHERE_ELSE", sentinel_positions=int(is_filled.sum()),
          kept_positions=int(kept.sum()), kept_unchanged=1)

    # THE WHOLLY-BOUNDED ROW. Row 0 has no complete pool, so both of its selections are sentinels --
    # the case `-098`'s consumer settles as exact zeros, and the one a formula that assumed at least
    # one live column would get wrong.
    assert complete[0] == 0, complete[0]
    assert int(per_row[0]) == SELECT_K, int(per_row[0])
    assert bool((got[0] == SENTINEL).all()), got[0].tolist()
    _emit("C2_WHOLLY_BOUNDED_ROW", row=0, causal_len=CAUSAL_LENS[0], sentinels=int(per_row[0]),
          k=SELECT_K)

    identity = causal_sentinel_kernel_identity()
    assert identity is not None and identity[1] == "_causal_sentinel_nki", identity
    bound_nki, bound_fb = causal_bound_dispatch_counters()
    topk_nki, topk_fb = topk_select_dispatch_counters()
    assert (bound_nki, bound_fb) == (1, 0), (bound_nki, bound_fb)
    assert (sent_nki, sent_fb) == (1, 0), (sent_nki, sent_fb)
    assert (topk_nki, topk_fb) == (1, 0), (topk_nki, topk_fb)
    _emit("C2_ROUTE", bound=(bound_nki, bound_fb), sentinel=(sent_nki, sent_fb),
          topk_047=(topk_nki, topk_fb), kernel=identity[1])

    # ----------------------------------------------------------------------------------- #
    # THE MULTI-FOLD CASE. Repair `103r4`, review finding F1.
    # ----------------------------------------------------------------------------------- #
    # WHY THE READINGS ABOVE COULD NOT SEE THE DEFECT F1 NAMES. They run at `SELECT_K` = 2,
    # one fold, the non-striking path. The dispatch site used to read the selected score by
    # GATHERING the bounded scores at the returned index; on the striking branch a later
    # fold's `max8` matches a position an earlier fold overwrote with `-inf`, so the index
    # that comes back beside an `-inf` VALUE can be a column that was originally finite and
    # legal. The gather then read that original finite score, no `-1` was written, and the
    # row silently carried a legal pool id twice -- which the expansion expands twice.
    # THE TWO GATES THIS CASE ADDS ARE THE ONES THAT FORM WOULD HAVE FAILED: the per-row
    # sentinel count, and no legal pool id appearing twice in a row. The retired form is
    # computed here too and its own count is printed beside the current one, so the
    # transcript says whether the substitution actually fired at this geometry rather than
    # leaving a reader to assume it.
    reset_causal_bound_dispatch_counters()
    reset_causal_sentinel_dispatch_counters()
    reset_topk_select_dispatch_counters()

    mf_gen = torch.Generator().manual_seed(20316)
    mf_scores = torch.randn(
        ROWS, MULTIFOLD_POOL_COLUMNS, generator=mf_gen, dtype=torch.float32
    ) * 0.05
    mf_causal = torch.tensor(MULTIFOLD_CAUSAL_LENS, dtype=torch.int32).reshape(ROWS, 1)
    # Complete pools per row, computed from the dials with the block's own formula.
    mf_complete = [
        min(c // POOL_SIZE, MULTIFOLD_POOL_COLUMNS) for c in MULTIFOLD_CAUSAL_LENS
    ]
    mf_want = [max(0, MULTIFOLD_SELECT_K - c) for c in mf_complete]
    # The fixture's shape properties, DERIVED rather than typed (rev 268). The typed vectors
    # that stood here went stale the instant the order changed, and a stale vector is exactly
    # what makes the next reader trust the wrong premise.
    assert mf_complete == sorted(mf_complete, reverse=True), (
        f"the multifold rows must complete STRICTLY DECREASING pool counts, so the rows with "
        f"the most real candidates land in the seam's FIRST program; got {mf_complete}"
    )
    assert len(set(mf_complete)) == ROWS, (
        f"each row must complete a DIFFERENT number of pools, or two rows read as one case; "
        f"got {mf_complete}"
    )
    assert mf_want[0] == 0, (
        f"row 0 must complete all {MULTIFOLD_SELECT_K} pools and so owe NO sentinel, which is "
        f"what a case that sentinelised unconditionally would fail; got {mf_want}"
    )
    assert min(mf_want[1:]) > 0, (
        f"every row after the first must owe at least one sentinel; got {mf_want}"
    )
    # THE CONFOUND BREAK, read off the seam's own program count rather than described in prose.
    # Until rev 268 the high-count rows and the ragged tile's rows were the same rows, so the
    # per-stage count and the program split predicted one reading and could not be separated.
    from vllm_neuron.functional.dsa.topk_select import _NUM_PROGRAMS

    mf_rows_per_program = -(-ROWS // _NUM_PROGRAMS)
    assert min(mf_complete[:mf_rows_per_program]) > max(mf_complete[mf_rows_per_program:]), (
        f"every row in the FULL first tile must complete more pools than every row in the "
        f"RAGGED last tile, or the per-stage count and the program split coincide again and "
        f"this case cannot tell them apart; got {mf_complete} split at {mf_rows_per_program}"
    )

    mf_bounded = dsa_causal_bound(mf_scores, mf_causal, POOL_SIZE)
    assert can_run_dsa_topk_select(mf_bounded, MULTIFOLD_SELECT_K) is True, (
        f"the landed selector must serve rows={ROWS} width={MULTIFOLD_POOL_COLUMNS} "
        f"k={MULTIFOLD_SELECT_K}. A False here sends the selection to torch.topk, which "
        f"never strikes its input, and this case would then read a branch it is not about "
        f"-- report it as an envelope finding rather than widening anything"
    )
    mf_values, mf_indices = dsa_topk_select(mf_bounded, MULTIFOLD_SELECT_K)
    assert tuple(mf_values.shape) == (ROWS, MULTIFOLD_SELECT_K), tuple(mf_values.shape)
    mf_idx32 = mf_indices.to(torch.int32)
    mf_got = dsa_causal_sentinel(mf_values, mf_idx32, MULTIFOLD_POOL_COLUMNS)

    mf_bound_nki, mf_bound_fb = causal_bound_dispatch_counters()
    mf_sent_nki, mf_sent_fb = causal_sentinel_dispatch_counters()
    mf_topk_nki, mf_topk_fb = topk_select_dispatch_counters()
    assert (mf_bound_nki, mf_bound_fb) == (1, 0), (mf_bound_nki, mf_bound_fb)
    assert (mf_sent_nki, mf_sent_fb) == (1, 0), (mf_sent_nki, mf_sent_fb)
    assert (mf_topk_nki, mf_topk_fb) == (1, 0), (mf_topk_nki, mf_topk_fb)

    # THE BRANCH IS READ OFF THE KERNEL'S OWN DIAL, never typed here. By this line the NKI
    # route has already dispatched, so the vendored module is imported and its constant is
    # the one the kernel used.
    from vllm_neuron.functional.vendored_kernels.rotational_topk.rotational_topk_utils import (
        HW_PARAMS,
    )

    from vllm_neuron.functional.dsa.topk_select import _nki_config, _nki_dtype_of

    per_stage = int(HW_PARAMS.topk_per_stage)
    assert per_stage == 8, per_stage

    # THE PATH IS READ OFF THE CONFIG THE SEAM ITSELF BUILDS, not inferred here. `103r5` fixes
    # what this reading used to claim: at this geometry the call is ROTATIONAL with n_stages = 2,
    # so `topk_core` runs once per stage at `local_top_k_per_stage`, and that per-stage k is the
    # number whose remainder decides the branch.
    mf_cfg = _nki_config(
        ROWS, MULTIFOLD_POOL_COLUMNS, MULTIFOLD_SELECT_K, _nki_dtype_of(mf_bounded)
    )
    mf_stages = int(mf_cfg.n_stages)
    mf_local_k = int(mf_cfg.local_top_k_per_stage)
    mf_folds_per_stage = -(-mf_local_k // per_stage)
    assert mf_stages >= 2, (
        f"this case is about the rotational path; the factory chose n_stages={mf_stages}, which "
        f"is the scanning path -- a dial finding, not a module finding"
    )
    assert mf_local_k % per_stage == 0, (mf_local_k, per_stage)
    assert mf_folds_per_stage == 1, (mf_folds_per_stage, mf_local_k)
    assert SELECT_K % per_stage != 0 and -(-SELECT_K // per_stage) == 1, SELECT_K
    # No pads at this width, which is why the retired GATHER form below can even be evaluated.
    assert int(mf_cfg.padded_vocab_size) == int(mf_cfg.vocab_size) == MULTIFOLD_POOL_COLUMNS, (
        int(mf_cfg.padded_vocab_size), int(mf_cfg.vocab_size)
    )
    _emit("C2_MF_BRANCH", k=MULTIFOLD_SELECT_K, topk_per_stage=per_stage,
          n_stages=mf_stages, local_top_k_per_stage=mf_local_k,
          folds_per_stage=mf_folds_per_stage, striking=int(mf_local_k % per_stage == 0),
          pad_columns=int(mf_cfg.padded_vocab_size) - int(mf_cfg.vocab_size),
          k2_folds=1, k2_striking=0)

    # "EVERY FILLED SLOT AND NO OTHER", the same two halves as above, at the striking k.
    # THE ``isnan`` TERM MIRRORS THE ORACLE AND IS NOT OPTIONAL (rev 268, r2 audit 4). The marker's
    # own value arm is ``torch.le(values, BOUND_FILL_MARK) | torch.isnan(values)``
    # (``causal_bound.py:631``): NaN fails the ``<=`` compare, so without this term the two sides
    # disagree BY CONSTRUCTION the moment a NaN appears -- this expression would read False exactly
    # where the marker marks, and the assertion below would fail on a CORRECT marker. It went
    # unnoticed because an earlier assertion aborted the case first.
    mf_filled = (mf_values <= BOUND_FILL_MARK) | torch.isnan(mf_values)
    mf_per_row = (mf_got == SENTINEL).sum(dim=1).to(torch.int64)

    # THE PER-ROW READING, PRINTED BEFORE ANY ASSERTION CAN ABORT THE CASE. It began as `-103`'s r7
    # diagnostic and is kept as a regression reading, but its ORIGINAL INTERPRETATION IS WITHDRAWN
    # (rev 268). The r7 run read 16 marks holding NaN on rows 3 and 4 and this block called that a
    # cliff at the selector's per-stage cap, because those were the rows whose completed-pool counts
    # exceeded it. They were ALSO the rows the seam's second program held on a ragged tile, and the
    # ragged tile was the mechanism (`design-20260909-bh`): two explanations, one reading, and this
    # block asserted the wrong one. `MULTIFOLD_CAUSAL_LENS` is now descending precisely so the two
    # can no longer coincide, which is asserted above rather than described here.
    # What the rows still usefully read: with a NaN-free return every row's values are finite, its
    # indices distinct, and its real count exactly `mf_complete[row]`.
    # The block boundary is DERIVED from the config the kernel used, never typed.
    mf_block = MULTIFOLD_POOL_COLUMNS // mf_stages
    # EVERY row, not a chosen three (rev 268). The old subset was picked because rows 3 and 4 were
    # the failing ones and row 2 was their control -- a choice that only made sense under the
    # withdrawn reading, and one that would now print the three LOWEST counts and miss the rows the
    # reversal moved the high counts to. Five rows is cheap and needs no judgment about which
    # matter.
    for mf_row in range(ROWS):
        mf_vrow = mf_values[mf_row].to(torch.float64)
        mf_irow = mf_idx32[mf_row].to(torch.int64)
        _emit(
            "C2_MF_READ1_PAIRS",
            row=mf_row,
            complete=mf_complete[mf_row],
            pairs=";".join(
                f"{slot}:{int(i)}:{float(v):.9e}"
                for slot, (i, v) in enumerate(zip(mf_irow.tolist(), mf_vrow.tolist()))
            ),
        )
        _emit(
            "C2_MF_READ1_COUNTS",
            row=mf_row,
            complete=mf_complete[mf_row],
            marks=int((mf_got[mf_row] == SENTINEL).sum()),
            at_or_below_mark=int((mf_vrow <= BOUND_FILL_MARK).sum()),
            nan=int(torch.isnan(mf_vrow).sum()),
            equal_to_bound_fill=int((mf_vrow == BOUND_FILL).sum()),
            distinct_indices=len(set(mf_irow.tolist())),
            block_width=mf_block,
            in_block0=int((mf_irow < mf_block).sum()),
            in_block1=int((mf_irow >= mf_block).sum()),
            # READ THIS ONE ONLY BESIDE `distinct_indices` (rev 268). It counts returned indices
            # below the row's completed-pool count, so it is the number of REAL columns the row
            # got back -- but only when the indices are distinct. In the r7 reading every index
            # was 0, which is below every non-zero `complete`, so it reported 16 real columns for
            # a row that had received none. It is a real-count reading when
            # `distinct_indices == k`, and an artifact otherwise.
            real_columns_returned=int((mf_irow < mf_complete[mf_row]).sum()),
        )

    assert torch.equal(mf_per_row, torch.tensor(mf_want, dtype=torch.int64)), (
        f"per-row sentinel count {mf_per_row.tolist()} against the computed {mf_want}"
    )
    assert torch.equal(mf_got == SENTINEL, mf_filled), (
        "a sentinel was written at a position whose value was a real score, or withheld at "
        "one whose value was a fill"
    )
    mf_kept = ~mf_filled
    assert torch.equal(mf_got[mf_kept], mf_idx32[mf_kept]), "a kept index was rewritten"
    _emit("C2_MF_SENTINEL_COUNT", counts=mf_per_row.tolist(), computed=mf_want,
          filled_slots=int(mf_filled.sum()), kept=int(mf_kept.sum()))

    # NO LEGAL POOL ID TWICE IN A ROW, and every legal id is a pool the row completes. This
    # is the reading the retired gather form fails: it returned a struck-but-legal column
    # instead of a sentinel, so the row held that id twice.
    mf_dup_rows = []
    for r in range(ROWS):
        legal = [int(v) for v in mf_got[r].tolist() if v >= 0]
        if len(set(legal)) != len(legal):
            mf_dup_rows.append((r, legal))
        assert len(legal) == min(mf_complete[r], MULTIFOLD_SELECT_K), (r, legal)
        assert all(0 <= v < mf_complete[r] for v in legal), (r, legal, mf_complete[r])
    assert mf_dup_rows == [], (
        f"a row carries a legal pool id twice, which is exactly what the retired gather "
        f"form produced: {mf_dup_rows}"
    )
    _emit("C2_MF_NO_DUPLICATE_LEGAL_ID", rows=ROWS, duplicate_rows=len(mf_dup_rows),
          legal_per_row=[min(c, MULTIFOLD_SELECT_K) for c in mf_complete])

    # THE RETIRED FORM IS DISCLOSED HERE, AND NO LONGER ASSERTED TO DISAGREE. Repair `103r6`
    # under the lead's ruling at `approvals/LEAD-LOG.md` §766: §752's finite bound fill supersedes
    # §750's ruling (3), which asked this case to assert that the retired gather form MUST differ
    # from the value-keyed form. Under a finite fill that demand is false on correct code. At this
    # geometry the fill folds evenly, no pad column exists, and every pass holds sixteen finite
    # entries, so the selector's own strike value never appears and `bounded[index]` equals
    # `value` on every slot -- the two forms AGREE. The struck assertion was therefore a reading
    # about this case's DIALS, not about the module under test. What survives is the inequality
    # the gather form cannot violate, plus the printed comparison beside it. The agreement is not
    # asserted either: agreeing here is a fact about this geometry and not a property of the
    # retired form.
    # Computed through the ORACLE so no second kernel dispatch is charged to the counters above.
    mf_retired = dsa_causal_sentinel_torch_oracle(
        mf_bounded.gather(1, mf_indices.to(torch.int64)), mf_idx32, MULTIFOLD_POOL_COLUMNS
    )
    mf_retired_counts = (mf_retired == SENTINEL).sum(dim=1).to(torch.int64)
    assert bool((mf_retired_counts <= mf_per_row).all()), (
        f"the retired form reported MORE sentinels than the value-keyed form, which it "
        f"cannot: {mf_retired_counts.tolist()} against {mf_per_row.tolist()}"
    )
    _emit("C2_MF_RETIRED_GATHER_FORM", retired=mf_retired_counts.tolist(),
          value_keyed=mf_per_row.tolist(),
          disagreeing_rows=int(bool((mf_retired_counts != mf_per_row).any())))
    _emit("C2_MF_ROUTE", bound=(mf_bound_nki, mf_bound_fb),
          sentinel=(mf_sent_nki, mf_sent_fb), topk_047=(mf_topk_nki, mf_topk_fb))

    # ----------------------------------------------------------------------------------- #
    # THE PAD CASE. Repair `103r5`: the selector's OWN padding, at an index past the last
    # real pool. The record is `increments/contradiction-103-selector-pad-6874a0f5.md`.
    # ----------------------------------------------------------------------------------- #
    # WHY NEITHER CASE ABOVE CAN SEE THIS. Both run at an EVEN fold (width 16 and 32 with
    # n_stages 2), so the vendored loader's fast path overwrites every column and no pad
    # exists (`cascaded_max_utils.py:134-152`). At an ODD width the slow path runs, memsets
    # the last fold's tail with a FINITE `-9948.0` (`:154-158`), and gives those columns
    # positions that keep counting past the real width (`rotational_topk.py:203-207` with
    # `rotational_topk_utils.py:826-856`). A finite pad OUTRANKS every filled column, so a
    # row with fewer complete pools than `k` spends its free slots on pads FIRST -- and the
    # value arm cannot see them, because their value is an ordinary finite number. The index
    # arm is what catches them, and this case is that arm's reading.
    reset_causal_bound_dispatch_counters()
    reset_causal_sentinel_dispatch_counters()
    reset_topk_select_dispatch_counters()

    pad_gen = torch.Generator().manual_seed(20333)
    pad_scores = torch.randn(
        ROWS, PAD_POOL_COLUMNS, generator=pad_gen, dtype=torch.float32
    ) * 0.05
    pad_causal = torch.tensor(MULTIFOLD_CAUSAL_LENS, dtype=torch.int32).reshape(ROWS, 1)
    pad_complete = [
        min(c // POOL_SIZE, PAD_POOL_COLUMNS) for c in MULTIFOLD_CAUSAL_LENS
    ]
    pad_want = [max(0, MULTIFOLD_SELECT_K - c) for c in pad_complete]
    # DERIVED, never typed (rev 268), and this case is why the rule matters: it shares
    # `MULTIFOLD_CAUSAL_LENS` with the multifold case, so the typed vectors that stood here went
    # stale the moment that list was reversed, in a case whose subject is padding and not order.
    assert pad_complete == sorted(pad_complete, reverse=True), (
        f"this case inherits the multifold lengths, so it inherits their descending order; "
        f"got {pad_complete}"
    )
    assert pad_want[0] == 0 and min(pad_want[1:]) > 0, (
        f"row 0 completes all {MULTIFOLD_SELECT_K} pools and owes no sentinel, every later row "
        f"owes at least one; got {pad_want}"
    )

    pad_bounded = dsa_causal_bound(pad_scores, pad_causal, POOL_SIZE)
    assert can_run_dsa_topk_select(pad_bounded, MULTIFOLD_SELECT_K) is True, (
        f"the landed selector must serve rows={ROWS} width={PAD_POOL_COLUMNS} "
        f"k={MULTIFOLD_SELECT_K}; a False sends the selection to torch.topk, which pads "
        f"nothing, and this case would then read a path it is not about"
    )

    # THE PREMISE, READ OFF THE KERNEL'S OWN CONFIG. `padded_vocab_size - vocab_size` IS the
    # pad-column count (`rotational_topk_utils.py:433-434`), and those columns occupy global
    # positions `[vocab_size, padded_vocab_size)`. If the factory folds this width evenly the
    # premise fails and the message says so -- a dial finding, not a module finding.
    pad_cfg = _nki_config(
        ROWS, PAD_POOL_COLUMNS, MULTIFOLD_SELECT_K, _nki_dtype_of(pad_bounded)
    )
    pad_columns = int(pad_cfg.padded_vocab_size) - int(pad_cfg.vocab_size)
    assert int(pad_cfg.vocab_size) == PAD_POOL_COLUMNS, int(pad_cfg.vocab_size)
    assert int(pad_cfg.n_stages) >= 2, int(pad_cfg.n_stages)
    assert pad_columns >= 1, (
        f"the factory folded width {PAD_POOL_COLUMNS} into {int(pad_cfg.n_stages)} stages of "
        f"{int(pad_cfg.stage_free_size)} with NO pad column, so this case cannot read the pad "
        f"arm. Pick a width whose fold is uneven -- a dial finding, not a module finding"
    )
    _emit("C2_PAD_PREMISE", width=PAD_POOL_COLUMNS, n_stages=int(pad_cfg.n_stages),
          stage_free_size=int(pad_cfg.stage_free_size),
          padded_vocab_size=int(pad_cfg.padded_vocab_size), pad_columns=pad_columns,
          first_pad_index=PAD_POOL_COLUMNS)

    pad_values, pad_indices = dsa_topk_select(pad_bounded, MULTIFOLD_SELECT_K)
    pad_idx32 = pad_indices.to(torch.int32)

    # THE DEFECT'S OWN PRECONDITION, READ RATHER THAN ASSUMED: the selector really did hand
    # back an index at or past the real width. A zero here means the pads never won a slot,
    # and then this case proves nothing about the index arm.
    pad_out_of_range = pad_idx32 >= PAD_POOL_COLUMNS
    pad_per_row_oor = pad_out_of_range.sum(dim=1).to(torch.int64)
    assert int(pad_out_of_range.sum()) >= 1, (
        f"the selector returned no index at or past width {PAD_POOL_COLUMNS} on these "
        f"inputs, so the pad arm has nothing to catch here. The pads exist "
        f"({pad_columns} column(s) per row by the config above) but did not win a slot -- "
        f"a dial finding"
    )
    # A row cannot take more pads than exist, nor more than its free slots. The equality
    # (pads win before fills, because -9948.0 > BOUND_FILL) is DISCLOSED, not gated.
    pad_ceiling = [min(pad_columns, w) for w in pad_want]
    assert all(
        int(pad_per_row_oor[r]) <= pad_ceiling[r] for r in range(ROWS)
    ), (pad_per_row_oor.tolist(), pad_ceiling)
    assert int(pad_idx32.max()) < int(pad_cfg.padded_vocab_size), (
        f"an index past the PADDED extent {int(pad_cfg.padded_vocab_size)} came back: "
        f"{int(pad_idx32.max())}. That is outside anything this case can explain"
    )
    _emit("C2_PAD_WAS_SELECTED", per_row=pad_per_row_oor.tolist(),
          ceiling=pad_ceiling, total=int(pad_out_of_range.sum()),
          max_index=int(pad_idx32.max()), width=PAD_POOL_COLUMNS)

    assert can_run_dsa_causal_sentinel(pad_values, pad_idx32, PAD_POOL_COLUMNS) is True
    pad_got = dsa_causal_sentinel(pad_values, pad_idx32, PAD_POOL_COLUMNS)
    pad_oracle = dsa_causal_sentinel_torch_oracle(
        pad_values, pad_idx32, PAD_POOL_COLUMNS
    )
    assert torch.equal(pad_got, pad_oracle), (
        f"the kernel and the oracle must agree element for element; first differing "
        f"{(pad_got != pad_oracle).nonzero()[:1].tolist()}"
    )

    pad_bound_nki, pad_bound_fb = causal_bound_dispatch_counters()
    pad_sent_nki, pad_sent_fb = causal_sentinel_dispatch_counters()
    pad_topk_nki, pad_topk_fb = topk_select_dispatch_counters()
    assert (pad_bound_nki, pad_bound_fb) == (1, 0), (pad_bound_nki, pad_bound_fb)
    assert (pad_sent_nki, pad_sent_fb) == (1, 0), (pad_sent_nki, pad_sent_fb)
    assert (pad_topk_nki, pad_topk_fb) == (1, 0), (pad_topk_nki, pad_topk_fb)

    # EVERY SURVIVING ID IS A POOL THE ROW COMPLETES, which is the reading the one-arm form
    # fails: an out-of-range pad id would survive it and land in `dsa_index_expand`.
    pad_per_row = (pad_got == SENTINEL).sum(dim=1).to(torch.int64)
    assert torch.equal(pad_per_row, torch.tensor(pad_want, dtype=torch.int64)), (
        f"per-row sentinel count {pad_per_row.tolist()} against the computed {pad_want}"
    )
    for r in range(ROWS):
        legal = [int(v) for v in pad_got[r].tolist() if v >= 0]
        assert len(legal) == min(pad_complete[r], MULTIFOLD_SELECT_K), (r, legal)
        assert all(0 <= v < pad_complete[r] for v in legal), (r, legal, pad_complete[r])
        assert len(set(legal)) == len(legal), (r, legal)
    _emit("C2_PAD_SENTINEL_COUNT", counts=pad_per_row.tolist(), computed=pad_want,
          legal_per_row=[min(c, MULTIFOLD_SELECT_K) for c in pad_complete])

    # THE ONE-ARM FORM IS ASSERTED RED. Passing a width the selector's padded extent cannot
    # reach disables the index arm and leaves exactly the value-keyed marker this increment
    # shipped before `103r5`. It MUST disagree here, or the index arm is not load-bearing at
    # this geometry and the case proves nothing.
    value_only = dsa_causal_sentinel_torch_oracle(
        pad_values, pad_idx32, int(pad_cfg.padded_vocab_size) + 1
    )
    value_only_counts = (value_only == SENTINEL).sum(dim=1).to(torch.int64)
    value_only_illegal = [
        (r, [int(v) for v in value_only[r].tolist() if v >= pad_complete[r]])
        for r in range(ROWS)
        if any(int(v) >= pad_complete[r] for v in value_only[r].tolist())
    ]
    _emit("C2_PAD_ONE_ARM_FORM", value_only=value_only_counts.tolist(),
          two_arm=pad_per_row.tolist(), illegal_rows=len(value_only_illegal),
          illegal_detail=value_only_illegal)
    assert bool((value_only_counts != pad_per_row).any()), (
        f"the value-only marker AGREED with the two-arm marker, so the index arm caught "
        f"nothing here: {value_only_counts.tolist()} against {pad_per_row.tolist()}. Since "
        f"{int(pad_out_of_range.sum())} out-of-range index(es) were read above, an agreement "
        f"means the oracle's index arm is not doing what its name says"
    )
    assert len(value_only_illegal) >= 1, (
        "the value-only marker left NO illegal pool id, so this case cannot show what the "
        "index arm prevents"
    )
    _emit("C2_PAD_ROUTE", bound=(pad_bound_nki, pad_bound_fb),
          sentinel=(pad_sent_nki, pad_sent_fb), topk_047=(pad_topk_nki, pad_topk_fb))


# =========================================================================== #
# CONJUNCT 3 -- INTEGRATION, computed not typed
# =========================================================================== #


def test_the_sentinelised_ids_are_legal_for_048_and_the_chain_matches_the_reference() -> None:
    """Conjunct 3. The chain's pool ids satisfy ``-048``'s precondition, and attention agrees.

    THIS IS THE ITEM THE WHOLE BLOCK EXISTS FOR. ``-048``'s expansion declares a caller
    precondition -- every non-negative pool id is below ``seq_len // pool_size`` -- and the landed
    chain had nothing that enforced it. So the reading is taken with ``-048``'s OWN reader, imported
    rather than re-spelled, on the bounded chain and then on the UNBOUNDED chain as the control. If
    the control did not violate, this block would be enforcing a precondition nothing broke.

    Certifying component (D1.4): the two seams composed -- the bound before the selector and the
    sentinel after it -- as the dispatch site will compose them.
    """
    scores = _scores(303)
    causal_len = _causal_len()
    seq_lens = torch.tensor(CAUSAL_LENS, dtype=torch.int32)

    reset_causal_bound_dispatch_counters()
    reset_causal_sentinel_dispatch_counters()
    reset_index_expand_dispatch_counters()

    bounded = dsa_causal_bound(scores, causal_len, POOL_SIZE)
    values, indices = dsa_topk_select(bounded, SELECT_K)
    pool_ids = dsa_causal_sentinel(values, indices.to(torch.int32), POOL_COLUMNS)

    # `-048`'S OWN READER, on `-048`'s own argument shapes (python lists). ZERO violations.
    violations = _precondition_violations(pool_ids.tolist(), CAUSAL_LENS, POOL_SIZE)
    _emit("C3_PRECONDITION", violations=len(violations), detail=violations,
          rows=ROWS, population=pool_ids.numel())
    assert violations == [], (
        f"the bounded chain still breaks -048's precondition at {violations}"
    )
    legal_rows = ROWS - len({r for r, _ in violations})
    assert legal_rows == ROWS, legal_rows
    _emit("C3_LEGAL_ROWS", legal=legal_rows, of=ROWS)

    # THE CONTROL: the UNBOUNDED chain on the SAME scores. Row 0 completes no pool, so any
    # non-negative id it selects is a violation -- which is the gap this block closes.
    raw_values, raw_indices = dsa_topk_select(scores, SELECT_K)
    unbounded_ids = raw_indices.to(torch.int32)
    control = _precondition_violations(unbounded_ids.tolist(), CAUSAL_LENS, POOL_SIZE)
    _emit("C3_UNBOUNDED_CONTROL", violations=len(control), detail=control,
          fires=int(len(control) >= 1))
    assert len(control) >= 1, (
        "the UNBOUNDED chain satisfied -048's precondition on these inputs, so this item cannot "
        "see whether the bound did anything. Choose inputs the unbounded chain breaks"
    )
    assert any(r == 0 for r, _ in control), control
    assert bool((raw_values > BOUND_FILL_MARK).all()), "the control must run on unbounded scores"

    # THE EMITTED WIDTH COMES FROM `-048`'s OWN FUNCTION, never typed. This is what pins MLA_CASE.
    width = index_expand_width(SELECT_K, POOL_SIZE)
    assert width == MLA_CASE["topk"], (width, MLA_CASE["topk"])
    _emit("C3_WIDTH", derived=width, mla_case_topk=MLA_CASE["topk"], n_groups=SELECT_K,
          pool_size=POOL_SIZE)

    assert can_run_dsa_index_expand(pool_ids, seq_lens, POOL_SIZE) is True
    token_idx = dsa_index_expand(pool_ids, seq_lens, POOL_SIZE)
    assert tuple(token_idx.shape) == (ROWS, width), tuple(token_idx.shape)
    expand_nki, expand_fb = index_expand_dispatch_counters()
    assert (expand_nki, expand_fb) == (1, 0), (expand_nki, expand_fb)

    # EVERY EMITTED TOKEN INDEX IS INSIDE ITS OWN ROW, which is the precondition's whole purpose
    # restated at the token level rather than the pool level -- and the reading a reviewer can check
    # without following the pool arithmetic.
    live = token_idx >= 0
    assert int(live.sum()) > 0, "the expansion must emit at least one live token index"
    row_of = torch.arange(ROWS).reshape(ROWS, 1).expand_as(token_idx)
    limit = seq_lens.to(torch.int64).reshape(ROWS, 1).expand_as(token_idx)
    out_of_row = int((live & (token_idx.to(torch.int64) >= limit)).sum())
    _emit("C3_TOKENS_INSIDE_ROW", live=int(live.sum()), out_of_row=out_of_row,
          max_token=int(token_idx.max()), s_kv=MLA_CASE["s_kv"])
    assert out_of_row == 0, (
        f"{out_of_row} live token indices point past their own row's causal length"
    )
    assert int(token_idx.max()) < MLA_CASE["s_kv"], int(token_idx.max())
    assert int(row_of.max()) == ROWS - 1

    # AND THE CHAIN'S TAIL: `-098`'s consumer, against `-098`'s own float64 reference at `-098`'s
    # own tolerance pair, both IMPORTED. The scale is derived from the case's latent rank, the way
    # `-098`'s `case_scale` derives it, so it cannot drift from the width.
    q_lift, c_kv, _discarded_idx, q_pe, k_pe = make_case(**MLA_CASE, seed=403)
    assert q_pe is None and k_pe is None, "the declared geometry is at R == 0"
    scale = float(MLA_CASE["latent"]) ** -0.5
    assert can_run_mla_sparse_attention(
        q_lift, MLA_CASE["seq"], MLA_CASE["heads"], MLA_CASE["latent"], MLA_CASE["rope"],
        MLA_CASE["topk"], MLA_CASE["s_kv"], scale,
    ) is True, "the sparse kernel must admit the emitted width, or the chain has no tail to read"
    out = mla_sparse_attention(q_lift, c_kv, token_idx, scale)
    ref = sentinel_reference(q_lift, c_kv, token_idx, scale)
    err = float((out - ref).abs().max())
    _emit("C3_ATTENTION_MAXABS_VS_IMPORTED_REFERENCE", err=f"{err:.3e}", rtol=RTOL, atol=ATOL,
          pair_source="test_mla_sparse.py (inc-glm53f-098), imported not authored")
    assert bool(torch.isfinite(out).all()), (
        "the chain produced a non-finite value -- what a filled column reaching the softmax "
        "unmasked looks like"
    )
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)

    bound_nki, bound_fb = causal_bound_dispatch_counters()
    sent_nki, sent_fb = causal_sentinel_dispatch_counters()
    assert (bound_nki, bound_fb) == (1, 0), (bound_nki, bound_fb)
    assert (sent_nki, sent_fb) == (1, 0), (sent_nki, sent_fb)
    _emit("C3_ROUTE", bound=(bound_nki, bound_fb), sentinel=(sent_nki, sent_fb),
          index_expand_048=(expand_nki, expand_fb))


# =========================================================================== #
# CONJUNCT 4 -- REFUSAL, and the fallback firing controls
# =========================================================================== #


def test_three_malformed_bound_calls_are_refused_by_name_and_the_fallback_can_fire() -> None:
    """Conjunct 4. The three declared refusals raise by name; both fallback zeros are shown to move.

    THESE ARE RAISES, NOT GATE DECLINES, and that is a ruling rather than a style choice. Serving a
    non-power-of-two ``pool_size`` or a mismatched ``causal_len`` through the torch oracle would hand
    the caller a correct-looking answer for a call that cannot be right -- which is the silent
    fallback this increment exists to make unreachable.

    THE ITEM ALSO CARRIES THE FIRING CONTROL FOR BOTH FALLBACK ZEROS (D1.5). Because every malformed
    call RAISES, the only way either gate declines is NKI being unavailable -- so that is the
    control: force ``can_run_kernel`` False and watch each seam route to its oracle and its counter
    move off 0. Without it, the ``torch_fallback == 0`` of conjuncts 1 to 3 would be decoration.

    Certifying component (D1.4): ``causal_bound._validate_bound``,
    ``causal_bound.can_run_dsa_causal_bound`` and ``causal_bound.can_run_dsa_causal_sentinel``.
    """
    scores = _scores(403)
    causal_len = _causal_len()

    reset_causal_bound_dispatch_counters()
    reset_causal_sentinel_dispatch_counters()

    # (i) a non-power-of-two pool_size
    with pytest.raises(DsaCausalBoundError) as caught_pool:
        dsa_causal_bound(scores, causal_len, 6)
    pool_message = str(caught_pool.value)
    assert "pool_size" in pool_message, pool_message
    assert "power of two" in pool_message, pool_message
    assert "pool_size=6" in pool_message, pool_message
    _emit("C4_REFUSAL_POOL_SIZE", pool_size=6, message=pool_message.split(".")[0])

    # (ii) a causal_len that is not int32
    with pytest.raises(DsaCausalBoundError) as caught_dtype:
        dsa_causal_bound(scores, causal_len.to(torch.float32), POOL_SIZE)
    dtype_message = str(caught_dtype.value)
    assert "causal_len" in dtype_message, dtype_message
    assert "int32" in dtype_message, dtype_message
    assert "torch.float32" in dtype_message, dtype_message
    _emit("C4_REFUSAL_DTYPE", dtype="torch.float32", message=dtype_message.split(";")[0])

    # (iii) a causal_len whose row count differs from scores
    with pytest.raises(DsaCausalBoundError) as caught_rows:
        dsa_causal_bound(scores, causal_len[: ROWS - 1], POOL_SIZE)
    rows_message = str(caught_rows.value)
    assert "one length per score row" in rows_message, rows_message
    assert f"{ROWS - 1} lengths" in rows_message, rows_message
    assert f"{ROWS} score rows" in rows_message, rows_message
    _emit("C4_REFUSAL_ROW_COUNT", lengths=ROWS - 1, score_rows=ROWS,
          message=rows_message.split(";")[0])

    # AND THE SENTINEL'S OWN DTYPE REFUSAL, which protects the int64-to-int32 cast the dispatch site
    # performs. A strengthening inside an existing item under the same increment id, so the declared
    # count of four items stays four (§63 precedent, cited in this file's docstring).
    with pytest.raises(DsaCausalBoundError) as caught_idx:
        dsa_causal_sentinel(
            torch.zeros(ROWS, SELECT_K, dtype=torch.float32),
            torch.zeros(ROWS, SELECT_K, dtype=torch.int64),
            POOL_COLUMNS,
        )
    idx_message = str(caught_idx.value)
    assert "int32" in idx_message and "torch.int64" in idx_message, idx_message
    _emit("C4_REFUSAL_SENTINEL_DTYPE", dtype="torch.int64", message=idx_message.split(";")[0])

    # AND THE ``width`` REFUSALS, new in ``103r5``. ``width`` is the pad arm's whole basis, so a
    # caller that cannot say it must not be served: a tensor would be baked into the graph as
    # something nobody meant, and a zero or negative width would mark every slot. Both RAISE, for
    # the same reason the other three do, and both are read here so the validation is not decoration.
    good_values = torch.zeros(ROWS, SELECT_K, dtype=torch.float32)
    good_idx = torch.zeros(ROWS, SELECT_K, dtype=torch.int32)
    with pytest.raises(DsaCausalBoundError) as caught_wtype:
        dsa_causal_sentinel(good_values, good_idx, torch.tensor(POOL_COLUMNS))
    wtype_message = str(caught_wtype.value)
    assert "width must be a python int" in wtype_message, wtype_message
    assert "Tensor" in wtype_message, wtype_message
    _emit("C4_REFUSAL_WIDTH_TYPE", passed="torch.tensor", message=wtype_message.split(";")[0])

    with pytest.raises(DsaCausalBoundError) as caught_wzero:
        dsa_causal_sentinel(good_values, good_idx, 0)
    wzero_message = str(caught_wzero.value)
    assert "width=0" in wzero_message, wzero_message
    assert "positive number of real pool columns" in wzero_message, wzero_message
    _emit("C4_REFUSAL_WIDTH_ZERO", passed=0, message=wzero_message.split(";")[0])

    refused = (causal_bound_dispatch_counters(), causal_sentinel_dispatch_counters())
    assert refused == ((0, 0), (0, 0)), (
        f"a refusal that dispatched first has already produced the wrong answer; got {refused}"
    )
    _emit("C4_NOTHING_DISPATCHED", bound=refused[0], sentinel=refused[1])

    # THE FIRING CONTROL, both entry points. `can_run_kernel` is read as a module global by both
    # gates, so replacing it on the module under test is what a call in a no-NKI process sees.
    reset_causal_bound_dispatch_counters()
    reset_causal_sentinel_dispatch_counters()
    values = torch.zeros(ROWS, SELECT_K, dtype=torch.float32)
    values[0, 0] = BOUND_FILL
    idx32 = torch.zeros(ROWS, SELECT_K, dtype=torch.int32)
    saved = mod.can_run_kernel
    try:
        mod.can_run_kernel = lambda: False
        assert can_run_dsa_causal_bound(scores, causal_len, POOL_SIZE) is False
        assert can_run_dsa_causal_sentinel(values, idx32, POOL_COLUMNS) is False
        served_bound = dsa_causal_bound(scores, causal_len, POOL_SIZE)
        served_sentinel = dsa_causal_sentinel(values, idx32, POOL_COLUMNS)
    finally:
        mod.can_run_kernel = saved
    control = (causal_bound_dispatch_counters(), causal_sentinel_dispatch_counters())
    assert control == ((0, 1), (0, 1)), (
        f"both fallback zeros must be able to MOVE; got {control}"
    )
    assert torch.equal(
        served_bound.view(torch.int32),
        dsa_causal_bound_torch_oracle(scores, causal_len, POOL_SIZE).view(torch.int32),
    )
    assert torch.equal(
        served_sentinel, dsa_causal_sentinel_torch_oracle(values, idx32, POOL_COLUMNS)
    )
    assert int((served_sentinel == SENTINEL).sum()) == 1, served_sentinel.tolist()
    _emit("C4_FALLBACK_CONTROL", bound=control[0], sentinel=control[1],
          oracle_sentinels=int((served_sentinel == SENTINEL).sum()))

    # and both gates are live again afterwards, so the control did not leak into the process
    assert mod.can_run_kernel is can_run_kernel
    assert can_run_dsa_causal_bound(scores, causal_len, POOL_SIZE) is True
    assert can_run_dsa_causal_sentinel(values, idx32, POOL_COLUMNS) is True
    _emit("C4_GATES_RESTORED", bound=1, sentinel=1)
