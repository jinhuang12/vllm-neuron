"""The DSA layer's selecting regime: the case that runs the selection chain.

Above the bypass length a query row cannot attend its whole span, so the indexer
scores, selects and expands pools, and every selected column must lie inside the
row's own causal length. The second test runs the same site at a ``select_k`` where
the vendored selector folds, pads and strikes, which is the arm that has to survive
the selector's own padding.
"""

from __future__ import annotations

import pytest
import torch

from test.vllm_neuron.model.glm5_next.test_dsa_layer import (
    PAGE_SIZE,
    PAGES,
    POOL_SIZE,
    SeamSpy,
    TOPK_POOLS,
    _bypass_width,
    _causal_fill_api,
    _check_two_instruments_agree,
    _discover_counter_api,
    _fill_constants,
    _seam_module,
    build_layer_stack,
    gate_live,
    prefill_slot_mapping,
    read_all_counters,
    reset_all_counters,
)

MAX8_ELEMENTS = 8

#: The shortest length that selects at this file's dials and that the selector can actually run.
SELECTING_SEQ_LEN = POOL_SIZE * max(MAX8_ELEMENTS, TOPK_POOLS + 1)

#: The families the selecting decision governs, each owing exactly one dispatch on one prefill leg.
#: `causal_fill` is deliberately not here: it is the bypass seam and owes a zero on
#: this case, which is a different claim and is read as such below.
SELECTING_ONE_FAMILIES: tuple[str, ...] = (
    "paged_gather", "score_gemm", "topk_select", "index_expand",
)


def _causal_bound_apis():
    """``{"bound": (reset, read), "sentinel": (reset, read)}`` for the causal bound's two
    seams.
    """
    module = _seam_module("causal_bound")
    names = [n for n in dir(module) if n.endswith("_dispatch_counters")]
    resets = sorted(n for n in names if n.startswith("reset_"))
    reads = sorted(n for n in names if not n.startswith("reset_"))
    assert len(resets) == 2 and len(reads) == 2, (module.__name__, resets, reads)
    with pytest.raises(AssertionError):
        _discover_counter_api(module)
    out = {}
    for key, token in (("bound", "causal_bound"), ("sentinel", "causal_sentinel")):
        reset = [n for n in resets if token in n]
        read = [n for n in reads if token in n]
        assert len(reset) == 1 and len(read) == 1, (key, reset, read)
        out[key] = (getattr(module, reset[0]), getattr(module, read[0]))
    return out


def _expected_sentinel_counts(seq_lens: torch.Tensor, select_k: int, pool: int,
                              width: int, *, offset: int = 0) -> list[int]:
    """Sentinels per row, computed from the predicate, the dials and this case's own
    lengths.
    """
    return [
        max(0, int(select_k) - min((int(s) + int(offset)) // int(pool), int(width)))
        for s in seq_lens
    ]


def _selecting_operands(cfg, *, seed: int, tokens: int = SELECTING_SEQ_LEN) -> dict:
    """Operands for ``forward``'s prefill leg one pool above the bypass boundary. """
    gen = torch.Generator().manual_seed(int(seed))
    rows = int(tokens)
    return {
        "hidden": torch.randn(
            rows, int(cfg.hidden_size), generator=gen, dtype=torch.float32
        ),
        "q_latent": torch.randn(
            rows, int(cfg.q_lora_rank), generator=gen, dtype=torch.float32
        ),
        "pool_cache": torch.zeros(
            PAGES * PAGE_SIZE, int(cfg.index_head_dim), dtype=torch.bfloat16
        ),
        "seq_lens": torch.arange(1, rows + 1, dtype=torch.int32),
        "slot_mapping": prefill_slot_mapping(rows, int(cfg.index_kpool)),
    }


def test_forward_bounds_the_selecting_regime_to_each_rows_own_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``forward`` above the bypass bound: every row selects only pools it completes.
    """
    if not gate_live():
        pytest.skip("the NKI gate is not live; the counter readings would be meaningless")

    stack, cfg, _gen = build_layer_stack(layers=1)
    indexer = stack[0].attention.indexer
    pool = int(cfg.index_kpool)
    select_k = int(indexer.select_k())

    # 1. The regime selects, read from the closed form before anything else.
    candidates = SELECTING_SEQ_LEN // pool
    assert candidates > select_k, (
        f"{SELECTING_SEQ_LEN} token(s) yields {candidates} complete pool(s) against select_k="
        f"{select_k}; this test must sit ABOVE the strict bound or the bypass serves it and every "
        f"counter below reads zero for a reason that has nothing to do with the causal bound"
    )
    assert candidates >= MAX8_ELEMENTS, (
        f"{SELECTING_SEQ_LEN} token(s) yields {candidates} candidate pool(s), under the "
        f"columns nisa.max8 emits; the selector refuses with 'max8 requires at least 8 "
        f"elements per partition' (rotational_topk_utils.py -> nki/isa/_validation.py) and "
        f"this test never reaches the bound it exists to read"
    )
    assert SELECTING_SEQ_LEN == 32, (
        f"the block declares a 32-token case and this file's dials now derive "
        f"{SELECTING_SEQ_LEN}; the figures below are computed, but the case identity is declared"
    )
    assert select_k == TOPK_POOLS == 2 and pool == POOL_SIZE == 4, (select_k, pool)

    bound_api, sentinel_api = _causal_bound_apis()["bound"], _causal_bound_apis()["sentinel"]
    fill_reset, fill_read = _causal_fill_api()
    ops = _selecting_operands(cfg, seed=9_103_001)

    # The pool ids are the block's subject and `forward` returns the expanded token indices, so the
    # sentinel seam's own output is recorded as it passes. The recorder delegates to the real seam,
    # so it moves no counter of its own -- which the (1, 0) reading below then confirms.
    recorded: list[torch.Tensor] = []
    recorded_widths: list[int] = []
    causal_bound_mod = _seam_module("causal_bound")
    real_sentinel = causal_bound_mod.dsa_causal_sentinel

    def recording_sentinel(values, indices, width):
        # `width` is forwarded and never defaulted. A spy that swallowed the third operand
        # would keep passing while the dispatch site stopped screening pads.
        out = real_sentinel(values, indices, width)
        recorded.append(out.clone())
        recorded_widths.append(int(width))
        return out

    reset_all_counters()
    bound_api[0]()
    sentinel_api[0]()
    fill_reset()
    spy = SeamSpy()
    spy.install(monkeypatch)
    monkeypatch.setattr(causal_bound_mod, "dsa_causal_sentinel", recording_sentinel)
    try:
        got = indexer.forward(
            ops["hidden"], ops["q_latent"], ops["pool_cache"], ops["seq_lens"],
            max_seq_len=SELECTING_SEQ_LEN, page_size=PAGE_SIZE,
            slot_mapping=ops["slot_mapping"],
        )
    finally:
        monkeypatch.undo()
    readings = read_all_counters()
    bound_count = tuple(int(v) for v in bound_api[1]())
    sentinel_count = tuple(int(v) for v in sentinel_api[1]())
    fill_count = tuple(int(v) for v in fill_read())

    assert len(recorded) == 1, (
        f"the sentinel seam ran {len(recorded)} time(s) on one prefill leg; the dispatch site "
        f"composes it exactly once"
    )
    # And the width it was given is the one the selector saw. The marker screens an index at
    # or past `width`, so a `width` wider than the real pool columns disables that arm silently. The
    # dispatch site reads it off the bounded tensor; this reads that the number arriving is the
    # candidate count and not, say, the pool-cache row count or `select_k`.
    assert recorded_widths == [candidates], (
        f"the dispatch site handed the marker width {recorded_widths} where the bounded tensor has "
        f"{candidates} real pool column(s); a wider width turns the pad arm off without failing"
    )
    pool_ids = recorded[0]
    assert pool_ids.dtype == torch.int32, pool_ids.dtype
    assert tuple(pool_ids.shape) == (SELECTING_SEQ_LEN, select_k), tuple(pool_ids.shape)

    # 2. The pool IDS carry the right sentinels, per row, computed and not typed.
    want = _expected_sentinel_counts(ops["seq_lens"], select_k, pool, candidates)
    per_row = (pool_ids == -1).sum(dim=1).to(torch.int64).tolist()
    assert per_row == want, (
        f"the bounded chain read {per_row} sentinel(s) per row where the predicate computes {want}"
    )
    ones = [r for r, n in enumerate(want) if n == 1]
    twos = [r for r, n in enumerate(want) if n == 2]
    zeros = [r for r, n in enumerate(want) if n == 0]
    assert twos[0] == 0 and per_row[0] == 2, (twos, per_row[0])
    assert ones == [3, 4, 5, 6], (
        f"the rows reading one sentinel computed to {ones}; the expected reading is "
        f"figure to 'rows 3-6 one each' on exactly this arithmetic"
    )
    assert per_row[-1] == 0 and (SELECTING_SEQ_LEN - 1) in zeros, (per_row[-1], zeros)

    complete = [min(int(s) // pool, candidates) for s in ops["seq_lens"]]
    illegal = [
        (r, int(p)) for r in range(SELECTING_SEQ_LEN) for p in pool_ids[r]
        if int(p) != -1 and not (0 <= int(p) < complete[r])
    ]
    assert illegal == [], f"a row selected a pool it does not complete: {illegal}"

    # 3. The off-by-one predicate fails, from the same function at offset -1.
    struck = _expected_sentinel_counts(ops["seq_lens"], select_k, pool, candidates, offset=-1)
    differing = [r for r in range(SELECTING_SEQ_LEN) if struck[r] != want[r]]
    assert struck != want, (
        "the struck reading (causal_len = position) agrees with the ruled one on this case, so this "
        "test cannot tell them apart and the control is not a control"
    )
    assert differing == [3, 7], (
        f"the two predicates differ at rows {differing}; on a 32-token case at these dials they "
        f"differ at exactly rows 3 and 7 -- the two pool boundaries the off-by-one moves -- which is "
        f"what makes this control able to fire. The wider case did not weaken it: the width cap does "
        f"not reach these rows, so the differing set is the one the 12-token case read"
    )
    assert per_row != struck, (
        f"the chain reproduced the STRUCK predicate {struck} rather than the ruled one {want}: the "
        f"dispatch site is passing position where it owes position + 1"
    )

    # 4. The route ran in NKI, form R-1, per entry point.
    assert bound_count == (1, 0), (
        f"the bound seam read {bound_count} and owes exactly one NKI dispatch with no fallback; a "
        f"torch masked_fill would read (0, 1) here, which this path forbids"
    )
    assert sentinel_count == (1, 0), (
        f"the sentinel seam read {sentinel_count} and owes exactly one NKI dispatch with no fallback"
    )
    assert fill_count == (0, 0), (
        f"the bypass seam read {fill_count} on a SELECTING case; the two regimes are "
        f"exclusive and a non-zero here means both fired"
    )
    for family in SELECTING_ONE_FAMILIES:
        assert readings[family] == (1, 0), (
            f"{family} read {readings[family]} where the selecting chain owes exactly one dispatch "
            f"and no fallback on one prefill leg"
        )
    assert all(v[1] == 0 for v in readings.values()), (
        f"a torch fallback ran on the selecting path: {readings}"
    )
    control = readings["kpool_hadamard"]
    assert control[0] > 0, (
        f"kpool_hadamard read {control} on a call that must rotate the indexer query, so the counter "
        f"instrument is not reading this call and the zeros above measure nothing"
    )
    _check_two_instruments_agree("selecting", spy, readings)
    identity = causal_bound_mod.causal_bound_kernel_identity()
    sentinel_identity = causal_bound_mod.causal_sentinel_kernel_identity()
    assert identity is not None and identity[1].endswith("_causal_bound_nki"), identity
    assert sentinel_identity is not None and sentinel_identity[1].endswith(
        "_causal_sentinel_nki"
    ), sentinel_identity

    # And the expansion still emits the shape every consumer reads, unchanged by the bound.
    width = _bypass_width(indexer, pool)
    assert got.dtype == torch.int32, got.dtype
    assert tuple(got.shape) == (SELECTING_SEQ_LEN, width), tuple(got.shape)
    live = got >= 0
    limit = ops["seq_lens"].to(torch.int64).reshape(SELECTING_SEQ_LEN, 1).expand_as(got)
    out_of_row = int((live & (got.to(torch.int64) >= limit)).sum())
    assert out_of_row == 0, (
        f"{out_of_row} live token index(es) point past their own row's causal length, which is the "
        f"defect the bound removes seen at the token level"
    )


# =========================================================================== #
# the same bound, at a `select_k` where the selector strikes and pads.
#
# The test above runs the dispatch site at `select_k = 2`, where the vendored selector takes its
# non-striking, non-folding branch. This one runs the same site at a `select_k` where the selector
# folds its input, pads the short fold with a finite value, strikes what it takes, and moves
# selected values across partitions with a matmul -- the four behaviours this arm has to survive.
#
# It is a separate test rather than a parametrisation because the two read different claims: that
# one reads the predicate (`causal_len = position + 1`) against an off-by-one control, this one
# reads the marker against the selector's own returned pads. Parametrising would make one failure
# message answer for two claims.
#
# The case needs three properties that no comment can promise, because the kernel's factories
# decide them from `(rows, width, k, dtype)`. So the test asks `create_rotational_topk_config`
# (through the seam's own `_nki_config`) for `n_stages`, `local_top_k_per_stage` and
# `padded_vocab_size`, and a failed premise is a finding about the dials, not about the module.
#: The striking case's ``select_k``, and why it is 16 rather than 8 or 2.
STRIKING_SELECT_K = 16

#: The candidate width: strictly above ``select_k`` -- `can_run_dsa_topk_select` refuses `k == width`
#: (`topk_select.py`) -- and odd, so the fold cannot divide it and a pad column must exist. The
#: pad's own existence is read from `padded_vocab_size` below, not from this comment.
STRIKING_CANDIDATES = STRIKING_SELECT_K + 1

#: The prefill length that yields those candidates, and the ``index_topk`` that yields that
#: ``select_k``. Both derived: ``select_k`` is ``index_topk // index_kpool``
#: (``model_fp8.py``), and ``candidates`` is ``max_seq_len // index_kpool``
#: (``model_fp8.py``).
STRIKING_SEQ_LEN = STRIKING_CANDIDATES * POOL_SIZE
STRIKING_INDEX_TOPK = STRIKING_SELECT_K * POOL_SIZE


def test_forward_at_a_striking_select_k_sentinelises_every_pad_the_selector_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``forward`` at a striking ``select_k``: no pad column survives into a row's pool
    ids.
    """
    if not gate_live():
        pytest.skip("the NKI gate is not live; the counter readings would be meaningless")

    fill, mark = _fill_constants()
    stack, cfg, _gen = build_layer_stack(
        layers=1, seed=9_103_016, index_topk=STRIKING_INDEX_TOPK
    )
    indexer = stack[0].attention.indexer
    pool = int(cfg.index_kpool)
    select_k = int(indexer.select_k())
    candidates = STRIKING_SEQ_LEN // pool

    # 1. The case is the one it claims. Read before the run: a wrong dial here spends nothing.
    assert select_k == STRIKING_SELECT_K, (
        f"the config built select_k={select_k} where this case declares {STRIKING_SELECT_K}; "
        f"index_topk={int(cfg.index_topk)} over index_kpool={pool} is the only route to it"
    )
    assert candidates == STRIKING_CANDIDATES and candidates > select_k, (
        f"{STRIKING_SEQ_LEN} token(s) yields {candidates} candidate pool(s) against select_k="
        f"{select_k}; the selector refuses k == width and the bypass serves anything below it"
    )
    assert candidates >= MAX8_ELEMENTS, (candidates, MAX8_ELEMENTS)

    bound_api, sentinel_api = _causal_bound_apis()["bound"], _causal_bound_apis()["sentinel"]
    fill_reset, fill_read = _causal_fill_api()
    ops = _selecting_operands(cfg, seed=9_103_016, tokens=STRIKING_SEQ_LEN)

    # Both seams are recorded, and each recorder delegates to the real seam, so it moves no counter of
    # its own -- which the (1, 0) readings below then confirm. The bound's output is recorded because
    # it is the exact tensor the selector was handed: the width, the row count and the dtype the
    # kernel's config is built from all come from it rather than from a number typed here.
    bounded_seen: list[torch.Tensor] = []
    values_seen: list[torch.Tensor] = []
    indices_seen: list[torch.Tensor] = []
    widths_seen: list[int] = []
    marked_seen: list[torch.Tensor] = []
    causal_bound_mod = _seam_module("causal_bound")
    real_bound = causal_bound_mod.dsa_causal_bound
    real_sentinel = causal_bound_mod.dsa_causal_sentinel

    def recording_bound(scores, causal_len, pool_size):
        out = real_bound(scores, causal_len, pool_size)
        bounded_seen.append(out.clone())
        return out

    def recording_sentinel(values, indices, width):
        values_seen.append(values.clone())
        indices_seen.append(indices.clone())
        widths_seen.append(int(width))
        out = real_sentinel(values, indices, width)
        marked_seen.append(out.clone())
        return out

    reset_all_counters()
    bound_api[0]()
    sentinel_api[0]()
    fill_reset()
    spy = SeamSpy()
    spy.install(monkeypatch)
    monkeypatch.setattr(causal_bound_mod, "dsa_causal_bound", recording_bound)
    monkeypatch.setattr(causal_bound_mod, "dsa_causal_sentinel", recording_sentinel)
    try:
        got = indexer.forward(
            ops["hidden"], ops["q_latent"], ops["pool_cache"], ops["seq_lens"],
            max_seq_len=STRIKING_SEQ_LEN, page_size=PAGE_SIZE,
            slot_mapping=ops["slot_mapping"],
        )
    finally:
        monkeypatch.undo()
    readings = read_all_counters()
    bound_count = tuple(int(v) for v in bound_api[1]())
    sentinel_count = tuple(int(v) for v in sentinel_api[1]())
    fill_count = tuple(int(v) for v in fill_read())

    assert len(bounded_seen) == len(marked_seen) == 1, (len(bounded_seen), len(marked_seen))
    bounded, values, raw_ids, pool_ids = (
        bounded_seen[0], values_seen[0], indices_seen[0], marked_seen[0]
    )
    assert tuple(bounded.shape) == (STRIKING_SEQ_LEN, candidates), tuple(bounded.shape)
    assert widths_seen == [candidates], (
        f"the marker was handed width {widths_seen} where the tensor the selector saw has "
        f"{candidates} real column(s); a wider width switches the index arm off silently"
    )
    assert tuple(pool_ids.shape) == (STRIKING_SEQ_LEN, select_k), tuple(pool_ids.shape)
    assert pool_ids.dtype == torch.int32, pool_ids.dtype

    # 2. The geometry is the striking, folding, padding one, asked of the kernel's own factories with
    # the run's own operands. A failure of any of these three is a dial finding.
    from vllm_neuron.functional.dsa.topk_select import _nki_config, _nki_dtype_of

    kernel_cfg = _nki_config(
        int(bounded.shape[0]), int(bounded.shape[1]), select_k, _nki_dtype_of(bounded)
    )
    n_stages = int(kernel_cfg.n_stages)
    local_k = int(kernel_cfg.local_top_k_per_stage)
    pad_columns = int(kernel_cfg.padded_vocab_size) - int(kernel_cfg.vocab_size)
    assert int(kernel_cfg.vocab_size) == candidates, (int(kernel_cfg.vocab_size), candidates)
    assert n_stages >= 2, (
        f"the factory chose n_stages={n_stages}, so the kernel takes naive_scanning_topk with the "
        f"ORIGINAL k and neither folds nor rotates; this case then reads no pad and no matmul. A "
        f"DIAL finding: raise select_k or the width until the factory chooses two stages"
    )
    assert local_k % MAX8_ELEMENTS == 0, (
        f"the per-stage k is {local_k}, not a whole number of {MAX8_ELEMENTS}, so topk_core takes the "
        f"non-striking max8 + nc_find_index8 branch and this case is the test above with a longer leg"
    )
    assert pad_columns >= 1, (
        f"padded_vocab_size equals vocab_size, so the fold divides the width evenly and the selector "
        f"pads NOTHING; the index arm cannot be read on this case. A DIAL finding: the width must not "
        f"be a multiple of n_stages={n_stages}"
    )

    # 3. The selector really returned a pad, and its value is above the mark. The first fact makes the
    # index arm reachable; the second is why the value arm alone cannot stand in for it.
    out_of_range = raw_ids.to(torch.int64) >= candidates
    pad_slots = int(out_of_range.sum())
    pad_values = values[out_of_range]
    distinct = sorted({float(v) for v in pad_values.to(torch.float32).flatten()})
    assert pad_slots >= 1, (
        f"the selector returned no index at or past the width on a geometry the factory says has "
        f"{pad_columns} pad column(s), so this case does not read the index arm at all. A DIAL "
        f"finding about the case, not a pass"
    )
    assert bool((pad_values > mark).all()), (
        f"every out-of-range slot came back at or below the mark {mark}, so the VALUE arm alone would "
        f"have caught them and this case still does not read the index arm. The pad values seen were "
        f"{distinct[:8]}"
    )

    # 4. No selected value is NaN, which is what the finite fill was chosen for.
    nan_slots = int(torch.isnan(values).sum())
    assert nan_slots == 0, (
        f"{nan_slots} selected value(s) came back NaN. The bound writes the FINITE {fill} so that the "
        f"rotation matmul has no infinity to turn into one; a NaN here is a reading about the "
        f"selector for the lead -- the marker still fires on it, so the ids stay legal, but the "
        f"per-row count below is then measuring something this case did not declare"
    )

    # 5. The output is legal, which is the reading the parent commit fails.
    complete = [min(int(s) // pool, candidates) for s in ops["seq_lens"]]
    illegal = [
        (r, int(p)) for r in range(STRIKING_SEQ_LEN) for p in pool_ids[r]
        if int(p) != -1 and not (0 <= int(p) < complete[r])
    ]
    assert int((pool_ids.to(torch.int64) >= candidates).sum()) == 0, (
        f"a pad column survived into the pool ids: {illegal[:12]}. A finite pad outranks a bounded "
        f"column at the dispatch site, so it wins a slot, and only the "
        f"index arm can tell it from a real selection"
    )
    assert illegal == [], f"a row selected a pool it does not complete: {illegal[:12]}"

    want = _expected_sentinel_counts(ops["seq_lens"], select_k, pool, candidates)
    per_row = (pool_ids == -1).sum(dim=1).to(torch.int64).tolist()
    assert per_row == want, (
        f"the bounded chain read {per_row} sentinel(s) per row where the predicate computes {want}. "
        f"Fewer means a fill or a pad survived; more means a slot was marked that carried a real "
        f"selection -- a NaN value is the one known way that happens and reading 4 covers it"
    )
    assert want[0] == select_k and want[-1] == 0, (
        f"the computed vector is {want[:3]}..{want[-3:]}; row 0 completes no pool so it owes "
        f"{select_k} sentinels and the longest row completes {complete[-1]} so it owes none. Without "
        f"both ends this case reads one regime only"
    )

    # 6. The one-arm marker disagrees, computed from this run's own recorded operands. Asserted red:
    # if the value arm alone reached the same answer, the index arm would be decoration on this case.
    value_only = torch.where(
        values > mark, raw_ids, torch.full_like(raw_ids, -1)
    )
    value_only_out_of_range = int((value_only.to(torch.int64) >= candidates).sum())
    value_only_counts = (value_only == -1).sum(dim=1).to(torch.int64).tolist()
    assert value_only_out_of_range >= 1, (
        f"the value arm ALONE left no out-of-range id, so this case cannot tell the two-arm marker "
        f"from the one-arm form the repair replaced"
    )
    assert value_only_counts != per_row, (
        f"the one-arm form read the same per-row counts as the two-arm marker, so the index arm "
        f"changed nothing measurable here"
    )

    # 7. The route ran in NKI, form R-1, per entry point.
    assert bound_count == (1, 0), (
        f"the bound seam read {bound_count} and owes exactly one NKI dispatch with no fallback"
    )
    assert sentinel_count == (1, 0), (
        f"the sentinel seam read {sentinel_count} and owes exactly one NKI dispatch with no fallback"
    )
    assert fill_count == (0, 0), (
        f"the bypass seam read {fill_count} on a SELECTING case"
    )
    for family in SELECTING_ONE_FAMILIES:
        assert readings[family] == (1, 0), (
            f"{family} read {readings[family]} where the selecting chain owes exactly one dispatch "
            f"and no fallback on one prefill leg"
        )
    assert all(v[1] == 0 for v in readings.values()), (
        f"a torch fallback ran on the selecting path: {readings}"
    )
    control = readings["kpool_hadamard"]
    assert control[0] > 0, (
        f"kpool_hadamard read {control} on a call that must rotate the indexer query, so the counter "
        f"instrument is not reading this call and the zeros above measure nothing"
    )
    _check_two_instruments_agree("striking", spy, readings)

    # And the expansion carries nothing past a row's own length, the same defect at the token level.
    width = _bypass_width(indexer, pool)
    live = got >= 0
    limit = ops["seq_lens"].to(torch.int64).reshape(STRIKING_SEQ_LEN, 1).expand_as(got)
    out_of_row = int((live & (got.to(torch.int64) >= limit)).sum())
    assert got.dtype == torch.int32, got.dtype
    assert tuple(got.shape) == (STRIKING_SEQ_LEN, width), tuple(got.shape)
    assert out_of_row == 0, (
        f"{out_of_row} live token index(es) point past their own row's causal length. A surviving pad "
        f"id expands to tokens at or past {candidates * pool}, which no row of this case can see"
    )
