# SPDX-License-Identifier: Apache-2.0
"""Acceptance for ``dsa_causal_fill``: exact causal index rows, bit-identically -- ``inc-glm53f-099``.

WHAT IS BEING ASSERTED, and what each reading is worth. The fill is INTEGER INDEX ARITHMETIC, so
there is no tolerance to spend and none is authored: the kernel is either exactly equal to the
reference, element for element, or it is wrong. Every comparison here is ``torch.equal`` on int32
tensors. The block registers no tolerance pair for this increment and this file introduces none.

FOUR ITEMS, ONE PER COUNTED CONJUNCT, NO ``parametrize`` -- section 6 rule 6, so the declared count
is derivable before a line runs. Controls live INSIDE the item whose zero or whose comparison they
protect, on the ``design-20260905`` §63 precedent: a strengthening under the same id never moves a
declared item count.

THE ZEROS HERE OWN FIRING CONTROLS.
  * the exact-equality comparison in item 1 is shown able to FAIL by a doctored oracle that is off
    by one at each row's last valid column;
  * the ``torch_fallback == 0`` of items 1 and 2 is shown able to MOVE in item 4, by forcing
    ``can_run_kernel`` False and watching the same seam route to the oracle;
  * item 3's predicate table is computed from the fixture's own dials, never typed, so a dial change
    moves the reading instead of silently keeping it true.

WHY THE ORACLE IS A REAL ORACLE. It is upstream's own write-then-mask spelling
(``sparse_attn_indexer_kpool.py:196-201`` at ``878631b6``); the kernel is a closed form over
``maximum``/``minimum`` that never writes a value it takes back. Two different mechanisms arriving
at the same bytes is an agreement; one mechanism written twice would only prove the module agrees
with itself.
"""

import json
import os
from pathlib import Path

import pytest
import torch

from vllm_neuron.functional.dsa import causal_fill as mod
from vllm_neuron.functional.dsa.causal_fill import (
    SENTINEL,
    DsaCausalFillError,
    can_run_dsa_causal_fill,
    causal_fill_dispatch_counters,
    causal_fill_kernel_identity,
    dsa_causal_fill,
    dsa_causal_fill_torch_oracle,
    reset_causal_fill_dispatch_counters,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

ROWS = 5
"""Query rows in the declared small shape -- the block's own figure."""

WIDTH = 16
"""Columns in the declared small shape -- the block's own figure.

Small on purpose and NOT a multiple of ``KEY_CHUNK``: the kernel admits any positive width, and the
admissibility ceiling belongs to the CALLER (``index_expand.index_expand_width``). Asserting the
ceiling here would be asserting ``inc-glm53f-102``'s claim in ``inc-glm53f-099``'s file."""

POSITIONS = [0, 3, 7, 15, 15]
"""The declared positions -- the block's own set, and every one of them earns its place.

``0`` is the first decode step, where exactly one column is causal and a bare ``c * keep`` closed
form would be indistinguishable from a masked column 0. ``3`` and ``7`` are interior. The two rows
at ``15 == WIDTH - 1`` are the saturated case, where NO column is masked and the sentinel count must
read exactly zero -- and there are TWO of them so a per-row reading cannot be confused with a
whole-tensor one."""

SEQ_LENS_ITEM3 = [2048, 2049, 2051, 2052]
"""The four sequence lengths item 3 reads the two predicates at -- the block's own set.

2048 is upstream's own bound; 2052 is the fork's first selecting length; 2049 and 2051 are the gap
between them, where upstream would select and the fork still bypasses, and where both routes attend
every token so the two answers agree."""


def _emit(tag: str, **values: object) -> None:
    """Print one MACHINE-READABLE reading line, for the driver to re-check independently.

    The pattern is the landed one at ``test_index_expand.py:110-118``: the item asserts, and then
    PRINTS the value it asserted on, so the driver that owns the transcript can check the same
    number without trusting this file's own verdict.
    """
    body = " ".join(f"{k}={v}" for k, v in values.items())
    print(f"S099|{tag}|{body}", flush=True)


def _positions(values: list[int]) -> torch.Tensor:
    """One case's input at the shape the seam declares: ``[rows]`` int32."""
    return torch.tensor(values, dtype=torch.int32)


def _assert_module_under_test_is_the_candidate() -> str:
    """Assert the module being measured is the candidate tree, and return where it resolved.

    §78.1's obligation, in the form ``inc-glm53f-056``'s repair settled (DECISIONS §88, §91 i): when
    ``GLM53F_CANDIDATE_ROOT`` is set the declared-root arm binds, and when it is UNSET the root is
    derived from this file's OWN tree instead of failing. A test that requires the campaign harness's
    environment variable is red by construction on every plain ``pytest`` run of the fork, which is
    exactly the landed defect that repair exists to remove -- so it is not repeated here.
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


def _fixture_dials() -> tuple[int, int]:
    """``(index_topk, index_kpool)`` READ FROM THE PINNED FIXTURE, never typed here.

    The dials are the target checkpoint's, so the bound item 3 computes moves when the checkpoint
    moves. Typing 2048 and 4 into this file would make item 3 agree with itself.
    """
    path = (
        Path(__file__).resolve().parents[2]
        / "model" / "glm5_next" / "fixtures" / "hf-config.json"
    )
    assert path.is_file(), f"the pinned fixture config must exist at {path}"
    text_config = json.loads(path.read_text())["text_config"]
    return int(text_config["index_topk"]), int(text_config["index_kpool"])


# =========================================================================== #
# CONJUNCT 1 -- EXACT ROWS
# =========================================================================== #


def test_the_kernel_rows_equal_the_oracle_exactly_and_a_doctored_oracle_fails() -> None:
    """Conjunct 1. The kernel's output equals the torch oracle element for element, int32.

    THREE READINGS IN ONE ITEM, because each alone would leave a hole. Exact equality against the
    oracle certifies the closed form against upstream's spelling. The hand-written literal for row 1
    certifies the ORACLE, so the agreement is not two wrong things matching. The doctored oracle
    certifies the COMPARISON, so a pass means the comparison was able to fail.

    Certifying component (D1.4): ``causal_fill._causal_fill_nki`` through the
    ``causal_fill.dsa_causal_fill`` seam.
    """
    _assert_module_under_test_is_the_candidate()
    positions = _positions(POSITIONS)

    reset_causal_fill_dispatch_counters()
    admitted = can_run_dsa_causal_fill(positions, WIDTH)
    assert admitted is True, "the declared case must take the NKI route, or the reading is not one"
    got = dsa_causal_fill(positions, WIDTH)
    want = dsa_causal_fill_torch_oracle(positions, WIDTH)
    nki_n, fallback_n = causal_fill_dispatch_counters()

    assert tuple(got.shape) == (ROWS, WIDTH), tuple(got.shape)
    assert got.dtype is torch.int32, got.dtype
    assert want.dtype is torch.int32, want.dtype
    diff = int((got.to(torch.int64) - want.to(torch.int64)).abs().max())
    assert diff == 0, f"integer index arithmetic admits no tolerance; max abs diff {diff}"
    assert torch.equal(got, want), "the kernel and the oracle must agree element for element"
    _emit("C1_EQUALITY", rows=ROWS, width=WIDTH, entries=got.numel(), max_abs_diff=diff)

    # THE EXACTNESS READING, and why conjunct 1 is where it belongs. Every tile in the kernel's
    # arithmetic chain is float32, with one int32 cast at the store, so the equality just read is only
    # trustworthy while every quantity flowing through that chain is a whole number float32 holds
    # EXACTLY -- an integer of magnitude below 2**24. That is not assumed here and it is not argued
    # in prose: each magnitude is read off THIS RUN's own input and output and round-tripped through
    # float32, so the claim is a measurement.
    #
    # THE COMPARATOR IS UNCHANGED. Still `torch.equal` on int32, still no tolerance, still the same
    # torch oracle. This is a strengthening inside an existing item under the same increment id, so
    # the declared count of four items stays four (§63 precedent, cited in this file's docstring).
    exact_ceiling = 2 ** 24
    magnitudes = {
        "position_max": int(positions.max()),
        "room_max": int(positions.max()) + 1,      # `positions[i] - c + 1`, at column 0
        "column_plus_one_max": WIDTH,              # `cols1`, the ramp's largest entry plus one
        "result_max": int(got.max()),
        "result_min_magnitude": abs(int(got.min())),
        "ceiling_minus_one": exact_ceiling - 1,    # the boundary itself, measured not asserted
    }
    for name, value in magnitudes.items():
        assert value < exact_ceiling, (name, value, exact_ceiling)
        roundtrip = torch.tensor([value], dtype=torch.float32)[0].item()
        assert roundtrip == float(value), (name, value, roundtrip)
    _emit("C1_FLOAT32_EXACTNESS", largest_magnitude=max(magnitudes.values()),
          ceiling=exact_ceiling, measured=len(magnitudes), all_round_trip=1)

    # THE FIRING CONTROL for that reading. The first integer float32 cannot hold is ``2**24 + 1``,
    # and the same round-trip reader must REJECT it. Without this, every round-trip above could be a
    # reader that returns True on whatever it is handed.
    beyond = exact_ceiling + 1
    beyond_roundtrip = torch.tensor([beyond], dtype=torch.float32)[0].item()
    assert beyond_roundtrip != float(beyond), (beyond, beyond_roundtrip)
    assert int(beyond_roundtrip) == exact_ceiling, beyond_roundtrip
    _emit("C1_EXACTNESS_CONTROL", probed=beyond, round_tripped_to=int(beyond_roundtrip),
          reader_rejected=1)

    # AND AT THE PRODUCTION MAGNITUDE, from the pinned checkpoint's dials rather than from this
    # file's deliberately small shape, so the exactness is not exact merely because five rows and
    # sixteen columns are tiny. A position in the bypass regime is at most
    # ``select_k * index_kpool + index_kpool - 2``. The largest COLUMN magnitude is the caller's
    # admissible width, which is ``index_expand_width``'s declared invariant and `inc-glm53f-102`'s
    # claim, so it is deliberately not asserted here -- the same scope line this file's WIDTH
    # docstring draws.
    index_topk, index_kpool = _fixture_dials()
    production_position_max = (index_topk // index_kpool) * index_kpool + index_kpool - 2
    assert production_position_max < exact_ceiling, production_position_max
    production_roundtrip = torch.tensor(
        [production_position_max], dtype=torch.float32
    )[0].item()
    assert production_roundtrip == float(production_position_max), production_position_max
    _emit("C1_PRODUCTION_MAGNITUDE", position_max=production_position_max,
          index_topk=index_topk, index_kpool=index_kpool, ceiling=exact_ceiling)

    # THE ORACLE ITSELF, against a literal written out by hand. Row 1's position is 3, so columns 0
    # to 3 carry themselves and the remaining twelve carry the sentinel.
    literal = torch.tensor(
        [0, 1, 2, 3] + [SENTINEL] * (WIDTH - 4), dtype=torch.int32
    )
    assert torch.equal(want[1], literal), (want[1].tolist(), literal.tolist())
    _emit("C1_LITERAL", row=1, position=POSITIONS[1], values=want[1].tolist())

    # THE CONTROL. Off by one at each row's LAST VALID column: the boundary column becomes the
    # sentinel. One entry per row moves, so the comparison above is shown able to fail.
    doctored = want.clone()
    for row, position in enumerate(POSITIONS):
        doctored[row, position] = SENTINEL
    differing = int((doctored != got).sum())
    assert not torch.equal(doctored, got), (
        "the doctored oracle must DISAGREE, or the equality assertion above proves nothing"
    )
    assert differing == ROWS, (
        f"the doctoring moves exactly one entry per row; {differing} of {ROWS} moved"
    )
    _emit("C1_DOCTORED_CONTROL", differing=differing, population=got.numel(), rows=ROWS)

    identity = causal_fill_kernel_identity()
    assert identity is not None, "the identity must be derived by TAKING the dispatch branch (D13.1)"
    assert identity[1] == "_causal_fill_nki", identity
    assert nki_n == 1 and fallback_n == 0, (nki_n, fallback_n)
    _emit("C1_ROUTE", can_run=admitted, nki_dispatch=nki_n, torch_fallback=fallback_n,
          kernel=identity[1])


# =========================================================================== #
# CONJUNCT 2 -- SENTINEL COUNT
# =========================================================================== #


def test_the_sentinel_count_per_row_is_the_width_less_the_position_less_one() -> None:
    """Conjunct 2. Per row, the number of ``-1`` columns is exactly ``width - position - 1``.

    WHY THIS IS A SEPARATE READING FROM CONJUNCT 1 rather than a restatement of it. Conjunct 1
    compares the kernel against a reference; if BOTH placed the boundary one column early they would
    still agree. This conjunct compares the kernel against ARITHMETIC on the declared positions, so
    it fails on a shared off-by-one that conjunct 1 cannot see.

    Certifying component (D1.4): the causal boundary in ``causal_fill._causal_fill_nki`` -- the
    ``clamp(positions[i] - c + 1, 0, 1)`` term.
    """
    positions = _positions(POSITIONS)

    reset_causal_fill_dispatch_counters()
    got = dsa_causal_fill(positions, WIDTH)
    nki_n, fallback_n = causal_fill_dispatch_counters()

    per_row = (got == SENTINEL).sum(dim=1).to(torch.int64)
    want = torch.tensor([WIDTH - p - 1 for p in POSITIONS], dtype=torch.int64)
    assert torch.equal(per_row, want), (per_row.tolist(), want.tolist())
    assert int(per_row.numel()) == ROWS, per_row.numel()
    _emit("C2_SENTINEL_COUNT", rows=ROWS, of=ROWS, counts=per_row.tolist(), want=want.tolist())

    # THE SATURATED ROWS carry ZERO sentinels. §79.1: the subset is asserted NON-EMPTY as well as
    # correct, so a selector that silently matched nothing cannot pass as a vacuous truth.
    saturated = [r for r, p in enumerate(POSITIONS) if p >= WIDTH - 1]
    assert len(saturated) > 0, "the declared shape must contain at least one saturated row"
    assert len(saturated) == 2, saturated
    for row in saturated:
        assert int(per_row[row]) == 0, (row, int(per_row[row]))
        # and every column of a saturated row is its own index, with nothing masked
        assert torch.equal(
            got[row], torch.arange(WIDTH, dtype=torch.int32)
        ), got[row].tolist()
    _emit("C2_SATURATED", rows=len(saturated), which=saturated,
          counts=[int(per_row[r]) for r in saturated])

    # The complement is non-empty too, or the saturated reading would be the only one taken.
    masked_rows = [r for r, p in enumerate(POSITIONS) if p < WIDTH - 1]
    assert len(masked_rows) == 3, masked_rows
    assert all(int(per_row[r]) > 0 for r in masked_rows), per_row.tolist()
    _emit("C2_MASKED", rows=len(masked_rows), which=masked_rows,
          counts=[int(per_row[r]) for r in masked_rows])

    assert nki_n == 1 and fallback_n == 0, (nki_n, fallback_n)
    _emit("C2_ROUTE", nki_dispatch=nki_n, torch_fallback=fallback_n)


# =========================================================================== #
# CONJUNCT 3 -- REGIME EQUIVALENCE, computed not typed
# =========================================================================== #


def test_the_fork_bypass_bound_differs_from_upstreams_and_agrees_where_it_matters() -> None:
    """Conjunct 3. The fork's predicate and upstream's, read at four lengths from the fixture's dials.

    THE TWO PREDICATES ARE NOT THE SAME PREDICATE, and this item is where that is stated as numbers
    rather than prose. The fork bounds on the CANDIDATE WIDTH (``seq_len // index_kpool <=
    select_k``) because that is what the landed selector refuses; upstream bounds on the TOKEN BUDGET
    (``max_seq_len <= index_topk``). They disagree on 2,049 to 2,051 -- and on exactly that span both
    routes attend every token anyway, because the complete-pool count equals ``select_k`` there, so
    upstream selecting 512 of 512 and the fork filling causal rows are the same attention set. That
    is why the fork may use its own bound without changing an answer.

    NOTHING HERE IS TYPED. The dials come from the pinned fixture and the minimum selecting length is
    found by scanning upward, so a checkpoint change moves this reading rather than leaving it
    accidentally true. This item calls no seam, which is itself asserted.

    Certifying component (D1.4): the bypass PREDICATE this block's dispatch site will use --
    ``seq_len // index_kpool <= select_k``.
    """
    index_topk, index_kpool = _fixture_dials()
    select_k = index_topk // index_kpool
    _emit("C3_DIALS", index_topk=index_topk, index_kpool=index_kpool, select_k=select_k)

    reset_causal_fill_dispatch_counters()

    fork = [seq // index_kpool <= select_k for seq in SEQ_LENS_ITEM3]
    upstream = [seq <= index_topk for seq in SEQ_LENS_ITEM3]
    assert fork == [True, True, True, False], (SEQ_LENS_ITEM3, fork)
    assert upstream == [True, False, False, False], (SEQ_LENS_ITEM3, upstream)
    assert fork != upstream, "the two bounds must differ, or this block's predicate is upstream's"
    _emit("C3_PREDICATES", lengths=SEQ_LENS_ITEM3, fork=fork, upstream=upstream)

    # WHERE THEY DISAGREE, THE ATTENTION SET IS THE SAME: the complete-pool count is exactly
    # `select_k`, so a selection would take every candidate. §79.1 -- the subset is non-empty.
    disagreeing = [
        seq for seq, f, u in zip(SEQ_LENS_ITEM3, fork, upstream) if f != u
    ]
    assert len(disagreeing) > 0, "the disagreement span must be non-empty for this reading to exist"
    assert disagreeing == [2049, 2051], disagreeing
    for seq in disagreeing:
        assert seq // index_kpool == select_k, (seq, seq // index_kpool, select_k)
    _emit("C3_DISAGREE", lengths=disagreeing, pools=[s // index_kpool for s in disagreeing],
          select_k=select_k)

    # THE MINIMUM SELECTING LENGTH, FOUND rather than typed: scan upward from 1 for the first length
    # the fork predicate lets through. The scan bound is derived from the dials, not a magic number.
    minimum = next(
        seq for seq in range(1, index_topk * 2 + index_kpool * 2)
        if not (seq // index_kpool <= select_k)
    )
    assert minimum == 2052, minimum
    assert minimum == (select_k + 1) * index_kpool, (minimum, select_k, index_kpool)
    assert minimum != index_topk, (
        f"the fork's bound is NOT the token budget: minimum selecting length {minimum} against "
        f"index_topk {index_topk}"
    )
    _emit("C3_MINIMUM", minimum=minimum, derived=(select_k + 1) * index_kpool,
          index_topk=index_topk, equal_to_index_topk=(minimum == index_topk))

    nki_n, fallback_n = causal_fill_dispatch_counters()
    assert (nki_n, fallback_n) == (0, 0), (
        f"this conjunct is arithmetic over the dials and must call no seam; got {(nki_n, fallback_n)}"
    )
    _emit("C3_ROUTE", nki_dispatch=nki_n, torch_fallback=fallback_n)


# =========================================================================== #
# CONJUNCT 4 -- REFUSAL, and the fallback firing control
# =========================================================================== #


def test_a_width_below_one_and_a_float_positions_tensor_are_refused_by_name() -> None:
    """Conjunct 4. Both malformed calls raise ``DsaCausalFillError`` naming what is wrong.

    THESE ARE RAISES, NOT GATE DECLINES, and that is the block's ruling rather than a style choice.
    Serving a zero width or a float position tensor through the torch oracle would hand a caller a
    correct-looking answer for a call that cannot be right, which is the ``k == width`` silent
    fallback this whole increment exists to make unreachable.

    THE ITEM ALSO CARRIES THE FIRING CONTROL FOR THE FALLBACK ZERO (D1.5). Because every malformed
    call now RAISES, the only way the gate declines is NKI being unavailable -- so that is the
    control: force ``can_run_kernel`` False and watch the same seam route to the oracle and the
    counter move off 0. Without it, the ``torch_fallback == 0`` of conjuncts 1 and 2 would be
    decoration.

    Certifying component (D1.4): ``causal_fill._validate`` and ``causal_fill.can_run_dsa_causal_fill``.
    """
    positions = _positions(POSITIONS)

    reset_causal_fill_dispatch_counters()
    with pytest.raises(DsaCausalFillError) as caught_width:
        dsa_causal_fill(positions, 0)
    width_message = str(caught_width.value)
    assert "width" in width_message, width_message
    assert "width=0" in width_message, width_message
    assert "at least 1 column" in width_message, width_message
    _emit("C4_REFUSAL_WIDTH", width=0, message=width_message.split(".")[0])

    with pytest.raises(DsaCausalFillError) as caught_dtype:
        dsa_causal_fill(positions.to(torch.float32), WIDTH)
    dtype_message = str(caught_dtype.value)
    assert "int32" in dtype_message, dtype_message
    assert "torch.float32" in dtype_message, dtype_message
    assert "positions" in dtype_message, dtype_message
    _emit("C4_REFUSAL_DTYPE", dtype="torch.float32", message=dtype_message.split(";")[0])

    refused_nki, refused_fallback = causal_fill_dispatch_counters()
    assert (refused_nki, refused_fallback) == (0, 0), (
        f"a refusal that dispatched first has already produced the wrong answer; got "
        f"{(refused_nki, refused_fallback)}"
    )
    _emit("C4_NOTHING_DISPATCHED", nki_dispatch=refused_nki, torch_fallback=refused_fallback)

    # THE FIRING CONTROL. `can_run_kernel` is read as a module global by the gate, so replacing it on
    # the module under test is what a call in a no-NKI process sees.
    reset_causal_fill_dispatch_counters()
    saved = mod.can_run_kernel
    try:
        mod.can_run_kernel = lambda: False
        assert can_run_dsa_causal_fill(positions, WIDTH) is False
        served = dsa_causal_fill(positions, WIDTH)
    finally:
        mod.can_run_kernel = saved
    control_nki, control_fallback = causal_fill_dispatch_counters()
    assert (control_nki, control_fallback) == (0, 1), (
        f"the fallback zero must be able to MOVE; got {(control_nki, control_fallback)}"
    )
    assert torch.equal(served, dsa_causal_fill_torch_oracle(positions, WIDTH))
    _emit("C4_FALLBACK_CONTROL", nki_dispatch=control_nki, torch_fallback=control_fallback)

    # and the gate is live again afterwards, so the control did not leak into the process
    assert mod.can_run_kernel is can_run_kernel
    assert can_run_dsa_causal_fill(positions, WIDTH) is True
    _emit("C4_GATE_RESTORED", can_run=can_run_dsa_causal_fill(positions, WIDTH))
