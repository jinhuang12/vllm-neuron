"""The DSA layer's bypass regime: the short case that selects nothing.

At a prefill no longer than ``select_k`` complete pools every query row can attend
its whole causal span, so the selection chain is skipped and the causal fill serves
the rows directly. These tests read the rows against a plain torch answer and read
the skipped families' counters, on both entry points.
"""

from __future__ import annotations

import pytest
import torch

from test.vllm_neuron.model.glm5_next.test_dsa_layer import (
    DECLARED_PER_LAYER,
    DECLARED_PER_LAYER_RAGGED_ARM,
    ENTRY_POINTS,
    FAMILIES,
    PAGE_SIZE,
    PAGES,
    POOL_SIZE,
    SeamSpy,
    TOPK_POOLS,
    _bypass_width,
    _causal_fill_api,
    _check_two_instruments_agree,
    _seam_module,
    build_layer_stack,
    gate_live,
    prefill_slot_mapping,
    read_all_counters,
    reset_all_counters,
)

#: The largest length the bypass serves at this file's dials: exactly ``select_k`` complete pools.
#: derived from the two dials rather than typed, so a dial change moves the case instead of quietly
#: turning it into a selecting case that would pass for the wrong reason.
BYPASS_SEQ_LEN = TOPK_POOLS * POOL_SIZE

#: The five families the bypass decision governs. They must read ``(0, 0)`` on both entry points.
#: Four are the selection chain the bypass skips. ``decode_tail_update`` is here because neither
#: case below is a decode step, so its zero is a phase reading rather than a bypass reading -- said
#: plainly so a reader does not count it as evidence about selection.
BYPASS_ZERO_FAMILIES: tuple[str, ...] = (
    "paged_gather", "score_gemm", "topk_select", "index_expand", "decode_tail_update",
)


def _ref_causal_rows(seq_lens: torch.Tensor, width: int) -> torch.Tensor:
    """The bypass's answer computed by plain torch in this file: row ``i`` is
    ``0..seq_lens[i]-1``.
    """
    rows = int(seq_lens.shape[0])
    columns = torch.arange(int(width), dtype=torch.int32).expand(rows, int(width))
    positions = seq_lens.to(torch.int32).reshape(rows, 1) - 1
    return torch.where(columns <= positions, columns, torch.full_like(columns, -1))


def _declared_column(table: dict[str, tuple[int, int]], column: int) -> dict[str, int]:
    """One phase column of a declared table, folded onto families. """
    out = {family: 0 for family in FAMILIES}
    for entry, counts in table.items():
        out[ENTRY_POINTS[entry]] += int(counts[column])
    return out


def _bypass_forward_operands(cfg, *, seed: int) -> dict:
    """Operands for ``forward``'s prefill leg at the bypass boundary. One builder, two callers."""
    gen = torch.Generator().manual_seed(int(seed))
    return {
        "hidden": torch.randn(
            BYPASS_SEQ_LEN, int(cfg.hidden_size), generator=gen, dtype=torch.float32
        ),
        "q_latent": torch.randn(
            BYPASS_SEQ_LEN, int(cfg.q_lora_rank), generator=gen, dtype=torch.float32
        ),
        "pool_cache": torch.zeros(
            PAGES * PAGE_SIZE, int(cfg.index_head_dim), dtype=torch.bfloat16
        ),
        "seq_lens": torch.arange(1, BYPASS_SEQ_LEN + 1, dtype=torch.int32),
        "slot_mapping": prefill_slot_mapping(BYPASS_SEQ_LEN, int(cfg.index_kpool)),
    }


def _bypass_ragged_operands(cfg, *, seed: int) -> dict:
    """Operands for the non-uniform ragged arm at the bypass boundary. """
    lengths = [BYPASS_SEQ_LEN // 2, BYPASS_SEQ_LEN]
    max_len = max(lengths)
    gen = torch.Generator().manual_seed(int(seed))
    return {
        "hidden": torch.randn(
            len(lengths), max_len, int(cfg.hidden_size), generator=gen, dtype=torch.float32
        ),
        "q_latent": torch.randn(
            len(lengths), max_len, int(cfg.q_lora_rank), generator=gen, dtype=torch.float32
        ),
        "pool_cache": torch.zeros(
            PAGES * PAGE_SIZE, int(cfg.index_head_dim), dtype=torch.bfloat16
        ),
        "seq_lens": torch.cat([torch.arange(1, n + 1, dtype=torch.int32) for n in lengths]),
        "lengths": lengths,
    }


def _run_bypass(indexer, method: str, ops: dict, monkeypatch: pytest.MonkeyPatch):
    """Run one entry point on the short regime with both instruments installed and reset
    first.
    """
    reset_all_counters()
    _causal_fill_api()[0]()
    spy = SeamSpy()
    spy.install(monkeypatch)
    try:
        if method == "forward":
            result = indexer.forward(
                ops["hidden"], ops["q_latent"], ops["pool_cache"], ops["seq_lens"],
                max_seq_len=BYPASS_SEQ_LEN, page_size=PAGE_SIZE,
                slot_mapping=ops["slot_mapping"],
            )
        else:
            result = indexer.forward_ragged(
                ops["hidden"], ops["q_latent"], ops["pool_cache"], ops["seq_lens"],
                ops["lengths"], max_seq_len=BYPASS_SEQ_LEN, page_size=PAGE_SIZE,
            )
    finally:
        monkeypatch.undo()
    readings = read_all_counters()
    fill = tuple(int(v) for v in _causal_fill_api()[1]())
    return result, readings, fill, spy


def _assert_regime_is_the_boundary(label: str, indexer, pool: int) -> int:
    """The case sits on the strict bound, asserted from the closed form. Returns ``select_k``."""
    select_k = int(indexer.select_k())
    candidates = BYPASS_SEQ_LEN // int(pool)
    assert candidates == select_k, (
        f"{BYPASS_SEQ_LEN} token(s) yields {candidates} complete pool(s) against select_k="
        f"{select_k}; these tests exist to sit ON the boundary, where a non-strict bound and a "
        f"clamped k would both pass and a correct bypass is the only thing that ALSO reads zero on "
        f"topk_select"
    )
    return select_k


def _check_bypass_zeros(label: str, readings: dict[str, tuple[int, int]]) -> None:
    """The five zeros, plus the reading that makes them measurements rather than
    decoration.
    """
    control = readings["kpool_hadamard"]
    assert control[0] > 0, (
        f"kpool_hadamard read {control} on a call that must rotate the indexer query, so the "
        f"counter instrument is not reading this call and the five zeros below measure nothing"
    )
    for family in BYPASS_ZERO_FAMILIES:
        assert readings[family] == (0, 0), (
            f"{family} read {readings[family]} on the bypass; the bypass exists to skip the "
            f"selection chain, so anything it dispatched is work done for an answer it discards"
        )
    assert all(v[1] == 0 for v in readings.values()), (
        f"a torch fallback ran on the bypass path: {readings}"
    )


def _check_the_kernel_ran(label: str, fill: tuple[int, int]) -> None:
    """The route predicate, plus the kernel identity read through the call site."""
    identity = _seam_module("causal_fill").causal_fill_kernel_identity()
    assert fill == (1, 0), (
        f"the bypass read {fill} on its own seam and owes exactly one NKI dispatch with no "
        f"fallback; a torch-level fill would read (0, 1) here, which this path forbids"
    )
    assert identity is not None, "no kernel identity was recorded, so nothing certifies what ran"
    assert identity[1].endswith("_causal_fill_nki"), (
        f"the call site dispatched {identity}, not the NKI kernel"
    )


def _check_rows_are_exact(label: str, got: torch.Tensor, reference: torch.Tensor,
                          want_shape: tuple[int, int]) -> None:
    """Exact int32 equality, its shape and dtype, and a doctored control that must fail."""
    assert got.dtype == torch.int32, f"the bypass returned {got.dtype}, not int32"
    assert tuple(got.shape) == want_shape, (
        f"the bypass emitted {tuple(got.shape)} where it owes {want_shape}: it must emit the SAME "
        f"shape selection emits, or every consumer would have to branch on the regime"
    )
    diff = int((got.to(torch.int64) - reference.to(torch.int64)).abs().max())
    assert diff == 0, (
        f"the bypass rows differ from this file's torch reference by up to {diff}. No tolerance is "
        f"used and none is admissible: these are indices, and a tolerance would hide an off-by-one"
    )
    # The comparison must be able to fail, or `max_abs_diff == 0` is decoration. One entry of the
    # reference is moved by one and the same reader runs again over the same population.
    doctored = reference.clone()
    doctored[0, 0] = int(doctored[0, 0]) + 1
    differing = int((got != doctored).sum())
    assert differing == 1, (
        f"the element-wise reader found {differing} differing entries against a reference with "
        f"exactly one entry moved, so it is not reading the tensors it claims to compare"
    )


def test_forward_serves_the_short_regime_with_exact_causal_rows_and_no_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``forward``: at the bypass boundary the pool write still lands and selection never
    runs.
    """
    if not gate_live():
        pytest.skip("the NKI gate is not live; the counter readings would be meaningless")

    stack, cfg, _gen = build_layer_stack(layers=1)
    indexer = stack[0].attention.indexer
    pool = int(cfg.index_kpool)
    _assert_regime_is_the_boundary("bypass-forward", indexer, pool)
    width = _bypass_width(indexer, pool)

    ops = _bypass_forward_operands(cfg, seed=9_099_001)
    pool_cache = ops["pool_cache"]

    def written_rows() -> int:
        return int((pool_cache.to(torch.float32).abs().sum(dim=1) != 0).sum())

    before = written_rows()
    reference = _ref_causal_rows(ops["seq_lens"], width)
    got, readings, fill, spy = _run_bypass(indexer, "forward", ops, monkeypatch)

    # 2. The write stage ran, as a value and not a boolean.
    after = written_rows()
    assert before == 0, f"the fixture handed a pre-populated pool_cache ({before} row(s))"
    assert after > before, (
        "the pooled-key store is untouched after a prefill call, so the bypass returned BEFORE the "
        "write instead of after it -- the exact defect the placement exists to prevent"
    )

    # 3. Selection never dispatched.
    _check_bypass_zeros("bypass-forward", readings)
    _check_two_instruments_agree("bypass-forward", spy, readings)
    _check_the_kernel_ran("bypass-forward", fill)

    # 1. The answer is exact.
    _check_rows_are_exact("bypass-forward", got, reference, (BYPASS_SEQ_LEN, width))


def test_forward_ragged_serves_the_short_regime_with_exact_causal_rows_and_no_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``forward_ragged``: the same bypass on the non-uniform arm, and the pack still runs.
    """
    if not gate_live():
        pytest.skip("the NKI gate is not live; the counter readings would be meaningless")

    stack, cfg, _gen = build_layer_stack(layers=1)
    indexer = stack[0].attention.indexer
    pool = int(cfg.index_kpool)
    _assert_regime_is_the_boundary("bypass-ragged", indexer, pool)
    width = _bypass_width(indexer, pool)

    ops = _bypass_ragged_operands(cfg, seed=9_099_002)
    tokens = int(ops["seq_lens"].shape[0])
    assert tokens == sum(ops["lengths"]), "one seq_len per PACKED row"
    assert len(set(ops["lengths"])) > 1, "a uniform batch is refused by the arm, and rightly"

    reference = _ref_causal_rows(ops["seq_lens"], width)
    got, readings, fill, spy = _run_bypass(indexer, "forward_ragged", ops, monkeypatch)

    _check_bypass_zeros("bypass-ragged", readings)
    pack = readings["ragged_pack"]
    assert pack[0] > 0, (
        f"ragged_pack read {pack} on the bypass, so the bypass returned BEFORE the pack and zeroed "
        f"the seventh counter family on this regime. The placement is after the pack on purpose"
    )
    _check_two_instruments_agree("bypass-ragged", spy, readings)
    _check_the_kernel_ran("bypass-ragged", fill)
    _check_rows_are_exact("bypass-ragged", got, reference, (tokens, width))


def test_the_two_entry_points_read_the_same_bypass_governed_families(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bypass is one decision, so the families it governs must read the same on both
    calls.
    """
    if not gate_live():
        pytest.skip("the NKI gate is not live; the counter readings would be meaningless")

    stack, cfg, _gen = build_layer_stack(layers=1)
    indexer = stack[0].attention.indexer
    pool = int(cfg.index_kpool)
    _assert_regime_is_the_boundary("bypass-both", indexer, pool)

    # Each call gets its own operands, so neither call can see the other's pool write.
    _got_f, forward_readings, forward_fill, _spy_f = _run_bypass(
        indexer, "forward", _bypass_forward_operands(cfg, seed=9_099_003), monkeypatch
    )
    _got_r, ragged_readings, ragged_fill, _spy_r = _run_bypass(
        indexer, "forward_ragged", _bypass_ragged_operands(cfg, seed=9_099_004), monkeypatch
    )


    # The five that must match, and both claims are made: equal to each other, and equal to zero.
    # Equality alone would pass if both entry points selected identically.
    for family in BYPASS_ZERO_FAMILIES:
        assert forward_readings[family] == ragged_readings[family], (
            f"{family} read {forward_readings[family]} on forward and {ragged_readings[family]} on "
            f"forward_ragged; the bypass is one decision and the families it governs cannot differ"
        )
        assert forward_readings[family] == (0, 0), (
            f"{family} read {forward_readings[family]} on BOTH entry points, equally and non-zero, "
            f"so the two agree while both are still running selection"
        )

    # The fill ran on both, which is the positive half of the same claim.
    assert forward_fill == ragged_fill == (1, 0), (
        f"the bypass seam read {forward_fill} on forward and {ragged_fill} on forward_ragged; both "
        f"owe exactly one NKI dispatch and no fallback"
    )

    # The two that differ, each against its own arm's declared column, derived not typed.
    declared_forward = _declared_column(DECLARED_PER_LAYER, 0)
    declared_ragged = _declared_column(DECLARED_PER_LAYER_RAGGED_ARM, 1)
    for family in ("kpool_hadamard", "ragged_pack"):
        assert forward_readings[family][0] == declared_forward[family], (
            f"{family} read {forward_readings[family][0]} on forward's prefill leg where "
            f"DECLARED_PER_LAYER's prefill column declares {declared_forward[family]}"
        )
        assert ragged_readings[family][0] == declared_ragged[family], (
            f"{family} read {ragged_readings[family][0]} on the ragged arm where "
            f"DECLARED_PER_LAYER_RAGGED_ARM's decode column declares {declared_ragged[family]}"
        )
    differing = sorted(
        f for f in FAMILIES if forward_readings[f] != ragged_readings[f]
    )
    assert differing == ["kpool_hadamard", "ragged_pack"], (
        f"the two entry points differ on {differing}; exactly two families may differ on the short "
        f"regime and both are named by the arms' tables, so a third difference is either a "
        f"new dispatch on one path or a lost one on the other"
    )
