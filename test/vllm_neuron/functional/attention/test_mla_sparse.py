# SPDX-License-Identifier: Apache-2.0
"""Tests for the sparse MLA latent attention kernel and its tiling paths.

Every case is at RoPE width 0, this checkpoint's own ``qk_rope_head_dim``. An
exact-fit latent (512) at up to 512 selected rows takes the untiled body; a
ragged latent (2051) takes the latent-tiled body; a wider selection (2048 rows)
takes the row-tiled body. A ``-1`` selected-row column is a sentinel the kernel
masks rather than refuses.
"""

from __future__ import annotations

import pytest
import torch

import nki.language as nl
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.attention import mla_sparse as MS

# bf16 module-comparison tolerance.
RTOL = 1e-2
ATOL = 1e-5

# This checkpoint's own kv_lora_rank (512) and head count (64); rope is 0, not a
# placeholder. s_kv exceeds topk in every case so the selection is a real subset
# of the cache rather than the whole cache in some order.
DECLARED_CASE = dict(seq=1, heads=64, latent=512, topk=128, s_kv=256, rope=0)

# 2051 = 16 x 128 + 3: not a multiple of the 128-row partition tile nor of the
# 512-wide moving tile, so both tilings end in a ragged tail.
WIDTH_CASE = dict(seq=1, heads=64, latent=2051, topk=128, s_kv=256, rope=0)

# 17 x 128, the next partition-tile multiple above 2051. Both widths take the
# tiled body; the untiled body serves neither.
REFERENCE_WIDTH = 2176

# seq=2 walks the per-query loop and topk=512 walks MM2's four-chunk loop, at a
# latent the untiled body serves.
EXACT_FIT_CASE = dict(seq=2, heads=64, latent=512, topk=512, s_kv=1024, rope=0)

# The smallest latent that takes the tiled body: one full tile plus one column.
MINIMAL_TILED_CASE = dict(seq=1, heads=8, latent=129, topk=128, s_kv=256, rope=0)

# The production selected-row count (index_topk = 2048) on the checkpoint's
# latent rank and head count.
ROWS_CASE = dict(seq=2, heads=64, latent=512, topk=2048, s_kv=4096, rope=0)

# One score tile: topk == MOVING_MAX is the widest selection the untiled body serves.
SPLIT_CASE = dict(seq=2, heads=64, latent=512, topk=512, s_kv=1024, rope=0)

# Two score tiles at a small head count and latent.
MINIMAL_ROW_TILED_CASE = dict(seq=1, heads=8, latent=128, topk=1024, s_kv=2048, rope=0)

# A latent that needs the latent tiling, requested together with a selection
# that needs the row tiling. The module refuses the combination.
COMBINATION_LATENT = 2051

# Score tiles are 512 wide, so 600 sits inside tile 1 of 4.
PEAK_POSITION = 600

# Small head count and latent: the claim is about the mask, not the width.
SENTINEL_UNTILED = dict(seq=8, heads=4, latent=128, topk=128, s_kv=256, rope=0)

# topk=640 > MOVING_MAX takes the row-tiled body and splits as 512 + 128, so the
# carried-softmax merge runs rather than being elided. latent=128 is an exact fit.
SENTINEL_ROW_TILED = dict(seq=5, heads=4, latent=128, topk=640, s_kv=768, rope=0)

# latent=131 takes the latent-tiled body: one full tile plus a ragged 3.
SENTINEL_TILED = dict(seq=5, heads=4, latent=131, topk=128, s_kv=256, rope=0)

# Expected (seam_nki, seam_fb, tiled_nki, tiled_fb, row_nki, row_fb) after one
# dispatch through each body. The seam counter counts every dispatch; each
# tiling's counter additionally counts its own.
UNTILED = (1, 0, 0, 0, 0, 0)
LATENT_TILED = (1, 0, 1, 0, 0, 0)
ROW_TILED = (1, 0, 0, 0, 1, 0)


def make_case(seq: int, heads: int, latent: int, topk: int, s_kv: int, rope: int,
              seed: int = 40):
    """Inputs for one case, seeded so a failure reproduces from the shape.

    Each query selects a distinct random subset of cache rows, so a gather that
    ignored its index tensor could not agree with the reference.
    """
    gen = torch.Generator().manual_seed(seed)
    q_lift = torch.randn(seq, heads, latent, generator=gen, dtype=torch.float32) * 0.05
    c_kv = torch.randn(s_kv, latent, generator=gen, dtype=torch.float32) * 0.05
    idx = torch.stack(
        [torch.randperm(s_kv, generator=gen)[:topk] for _ in range(seq)]
    ).to(torch.int32)
    q_pe = k_pe = None
    if rope > 0:
        q_pe = torch.randn(seq, heads, rope, generator=gen, dtype=torch.float32) * 0.05
        k_pe = torch.randn(s_kv, rope, generator=gen, dtype=torch.float32) * 0.05
    return q_lift, c_kv, idx, q_pe, k_pe


def sparse_mla_torch_reference(q_lift, c_kv, idx, softmax_scale, q_pe=None, k_pe=None):
    """float64 reference written independently of the module's own oracle.

    The scores are an explicit ``einsum`` contraction, the softmax is a shifted
    exponential over an explicit sum, and the value pass is a second ``einsum``,
    so a shared mistake with the module's oracle would have to be made twice in
    two notations.
    """
    q = q_lift.to(torch.float64)
    cache = c_kv.to(torch.float64)
    rows = idx.to(torch.int64)
    seq = q.shape[0]
    out = []
    for s in range(seq):
        gathered = cache[rows[s]]                                  # [K, L]
        scores = torch.einsum("hl,kl->hk", q[s], gathered)
        if q_pe is not None:
            rope_rows = k_pe.to(torch.float64)[rows[s]]             # [K, R]
            scores = scores + torch.einsum(
                "hr,kr->hk", q_pe.to(torch.float64)[s], rope_rows
            )
        scaled = scores * softmax_scale
        shifted = scaled - scaled.max(dim=-1, keepdim=True).values
        expd = torch.exp(shifted)
        weights = expd / expd.sum(dim=-1, keepdim=True)
        out.append(torch.einsum("hk,kl->hl", weights, gathered))
    return torch.stack(out).to(torch.float32)


def case_scale(case: dict) -> float:
    """The softmax scale for a case: the inverse square root of its latent rank."""
    return float(case["latent"]) ** -0.5


def all_counters() -> tuple[int, int, int, int, int, int]:
    """``(seam_nki, seam_fb, tiled_nki, tiled_fb, row_nki, row_fb)``."""
    seam_nki, seam_fb = MS.mla_sparse_dispatch_counters()
    tiled_nki, tiled_fb = MS.mla_sparse_tiled_dispatch_counters()
    row_nki, row_fb = MS.mla_sparse_row_tiled_dispatch_counters()
    return seam_nki, seam_fb, tiled_nki, tiled_fb, row_nki, row_fb


def reset_all_counters() -> None:
    MS.reset_mla_sparse_dispatch_counters()
    MS.reset_mla_sparse_tiled_dispatch_counters()
    MS.reset_mla_sparse_row_tiled_dispatch_counters()


def dispatch_once(case: dict, q_lift, c_kv, idx):
    """Require the gate open, dispatch once through the seam, return ``(got, counters)``."""
    scale = case_scale(case)
    gate = MS.can_run_mla_sparse_attention(
        q_lift, case["seq"], case["heads"], case["latent"], case["rope"],
        case["topk"], case["s_kv"], scale,
    )
    assert gate, "the kernel route is closed for this geometry"
    reset_all_counters()
    got = MS.mla_sparse_attention(q_lift, c_kv, idx, scale)
    assert tuple(got.shape) == (case["seq"], case["heads"], case["latent"]), (
        f"kernel returned {tuple(got.shape)}"
    )
    assert bool(torch.isfinite(got).all()), (
        "the result holds a non-finite value, which is what a tile the kernel never "
        "wrote looks like"
    )
    return got, all_counters()


def admissibility_args(case: dict, **overrides) -> dict:
    args = dict(seq=case["seq"], heads=case["heads"], latent=case["latent"],
                rope=case["rope"], topk=case["topk"], s_kv=case["s_kv"],
                softmax_scale=case_scale(case))
    args.update(overrides)
    return args


# --------------------------------------------------------------------------- #
# The untiled body and the module's bounds.
# --------------------------------------------------------------------------- #
def test_no_rope_case_matches_the_torch_reference() -> None:
    """Latent 512 with no RoPE half takes the untiled body and agrees with float64 torch."""
    case = dict(DECLARED_CASE)
    q_lift, c_kv, idx, q_pe, k_pe = make_case(**case)
    assert q_pe is None and k_pe is None

    got, counters = dispatch_once(case, q_lift, c_kv, idx)
    assert counters == UNTILED

    expected = sparse_mla_torch_reference(q_lift, c_kv, idx, case_scale(case))
    torch.testing.assert_close(got, expected, rtol=RTOL, atol=ATOL)


def test_tile_bounds_match_nl_tile_size() -> None:
    """The module's tile constants are this image's axis extents."""
    assert MS.LATENT_TILE == nl.tile_size.pmax
    assert MS.KEY_CHUNK == nl.tile_size.pmax
    assert MS.HEAD_MAX == nl.tile_size.gemm_stationary_fmax
    assert MS.MOVING_MAX == nl.tile_size.gemm_moving_fmax


def test_admissibility_check_refuses_each_inadmissible_axis_by_name() -> None:
    """The declared case passes; each bad axis raises with a message naming that axis."""
    case = dict(DECLARED_CASE)
    MS._require_admissible(**admissibility_args(case))

    for override, fragment in (
        (dict(heads=MS.HEAD_MAX + 1), "stationary free axis"),
        (dict(latent=0), "latent rank must be positive"),
        (dict(topk=MS.KEY_CHUNK + 2), "multiple of it"),
        (dict(softmax_scale=0.0), "must be positive"),
        (dict(rope=-1), "Zero IS admissible"),
    ):
        with pytest.raises(MS.MlaSparseAttentionError) as caught:
            MS._require_admissible(**admissibility_args(case, **override))
        message = " ".join(str(caught.value).split())
        assert fragment in message, (
            f"{override} raised, but not with its own message: expected a mention of "
            f"{fragment!r}, got {message!r}"
        )


def test_admissibility_check_serves_tiled_widths_and_the_production_row_count() -> None:
    """Ragged and wide latents pass at a narrow selection; 2048 rows pass at an exact-fit latent."""
    case = dict(WIDTH_CASE)
    for latent in (
        MS.LATENT_TILE + 1,
        MS.MOVING_MAX + MS.LATENT_TILE,
        case["latent"],
        REFERENCE_WIDTH,
    ):
        MS._require_admissible(**admissibility_args(case, latent=latent))

    rows = dict(ROWS_CASE)
    MS._require_admissible(**admissibility_args(rows))
    MS._require_admissible(**admissibility_args(rows, topk=MS.KEY_CHUNK))
    MS._require_admissible(
        **admissibility_args(rows, latent=COMBINATION_LATENT, topk=MS.KEY_CHUNK)
    )


def test_admissibility_check_refuses_both_tilings_in_one_call() -> None:
    """A ragged latent together with a wide selection raises, and the message names both axes."""
    rows = dict(ROWS_CASE)
    with pytest.raises(MS.MlaSparseAttentionError) as caught:
        MS._require_admissible(**admissibility_args(rows, latent=COMBINATION_LATENT))
    message = " ".join(str(caught.value).split())
    for fragment in ("not served", "SAME call", f"topk={rows['topk']}",
                     f"latent={COMBINATION_LATENT}"):
        assert fragment in message, (
            f"the combination refusal does not name {fragment!r}: {message!r}"
        )


def test_kernel_identity_names_this_module_for_every_entry_point() -> None:
    """Every kernel the module reports is defined in the module itself."""
    identities = MS.mla_sparse_kernel_identity()
    assert identities
    for module_name, qualname in identities:
        assert module_name == MS.__name__, (
            f"{qualname} reports module {module_name}, not {MS.__name__}"
        )


# --------------------------------------------------------------------------- #
# The latent-tiled body.
# --------------------------------------------------------------------------- #
def test_ragged_latent_width_matches_the_torch_reference() -> None:
    """Latent 2051 takes the latent-tiled body and agrees with float64 torch."""
    case = dict(WIDTH_CASE)
    q_lift, c_kv, idx, _, _ = make_case(**case, seed=41)

    got, counters = dispatch_once(case, q_lift, c_kv, idx)
    assert counters == LATENT_TILED

    expected = sparse_mla_torch_reference(q_lift, c_kv, idx, case_scale(case))
    torch.testing.assert_close(got, expected, rtol=RTOL, atol=ATOL)


def test_ragged_width_is_bit_identical_to_the_same_data_zero_padded() -> None:
    """2051 columns equal the same inputs zero-extended to 2176, exactly, and the
    padded output columns are exactly zero."""
    case = dict(WIDTH_CASE)
    latent = case["latent"]
    scale = case_scale(case)
    q_lift, c_kv, idx, _, _ = make_case(**case, seed=41)

    q_ext = torch.zeros(case["seq"], case["heads"], REFERENCE_WIDTH, dtype=torch.float32)
    q_ext[:, :, :latent] = q_lift
    c_ext = torch.zeros(case["s_kv"], REFERENCE_WIDTH, dtype=torch.float32)
    c_ext[:, :latent] = c_kv

    reset_all_counters()
    got_real = MS.mla_sparse_attention(q_lift, c_kv, idx, scale)
    got_pad = MS.mla_sparse_attention(q_ext, c_ext, idx, scale)
    assert all_counters() == (2, 0, 2, 0, 0, 0)

    # The extension adds only exact zeros to MM1's contraction and x + 0.0 == x in
    # fp32, so the comparison is exact rather than a tolerance.
    diff = float((got_real - got_pad[:, :, :latent]).abs().max())
    assert diff == 0.0, (
        f"the padded run disagrees with the ragged run by {diff} over the real "
        f"{latent} columns; a tail tile is reading or writing outside its extent"
    )
    # Weights times zero rows.
    assert float(got_pad[:, :, latent:].abs().max()) == 0.0


def test_latent_and_output_tilings_partition_the_width() -> None:
    """Both tilings of 2051 are contiguous, sum to the width, stay within their axis
    bound, and end in a ragged tile."""
    latent = WIDTH_CASE["latent"]
    for tiles, bound in (
        (MS._latent_tiles(latent), nl.tile_size.pmax),
        (MS._output_tiles(latent), nl.tile_size.gemm_moving_fmax),
    ):
        expected_count = -(-latent // bound)
        assert len(tiles) == expected_count
        assert tiles[-1][1] == latent - (expected_count - 1) * bound
        offset = 0
        for tile_offset, extent in tiles:
            assert tile_offset == offset, "the tiling is not contiguous"
            assert 1 <= extent <= bound, "a tile exceeds its axis bound"
            offset += extent
        assert offset == latent
        assert tiles[-1][1] % bound != 0


def test_exact_fit_width_takes_the_untiled_body_and_matches_the_reference() -> None:
    """Latent 512 at seq=2 and topk=512 stays on the untiled body and agrees with float64 torch."""
    case = dict(EXACT_FIT_CASE)
    q_lift, c_kv, idx, _, _ = make_case(**case, seed=42)

    got, counters = dispatch_once(case, q_lift, c_kv, idx)
    assert counters == UNTILED

    expected = sparse_mla_torch_reference(q_lift, c_kv, idx, case_scale(case))
    torch.testing.assert_close(got, expected, rtol=RTOL, atol=ATOL)


def test_shipped_torch_oracle_matches_the_independent_reference() -> None:
    """The module's own torch oracle agrees with this file's reference and dispatches nothing."""
    case = dict(WIDTH_CASE)
    scale = case_scale(case)
    q_lift, c_kv, idx, _, _ = make_case(**case, seed=41)

    reset_all_counters()
    shipped = MS.mla_sparse_attention_torch_oracle(q_lift, c_kv, idx, scale)
    independent = sparse_mla_torch_reference(q_lift, c_kv, idx, scale)
    torch.testing.assert_close(shipped, independent, rtol=RTOL, atol=ATOL)
    assert all_counters() == (0, 0, 0, 0, 0, 0)


def test_ragged_tail_tile_is_load_bearing_in_the_output() -> None:
    """At latent 2051 the signal that orders the keys sits in the 3-wide tail tile,
    so a kernel that skipped the tail cannot agree with the reference."""
    case = dict(WIDTH_CASE)
    scale = case_scale(case)
    latent = case["latent"]
    tail = latent - (latent % MS.LATENT_TILE)   # 2048: where the tail tile starts
    assert latent - tail == 3

    q_lift, c_kv, idx, _, _ = make_case(**case, seed=41)
    # With random inputs alone the tail moves the output by about 9e-6, under ATOL.
    # A per-row ramp on the cache's tail columns and a fixed 1.0 on the query's
    # make every head see the same tail term. step=2.5 spreads the tail logits by
    # scale * 3 * 2.5 * 255 = 42 nats across the cache, so the tail orders the
    # keys, while adjacent selected rows differ by about 0.33 nats, so the weights
    # stay a graded ramp (denominator about 3.5) rather than a one-hot pick. The
    # ramp varies per row because softmax is invariant to a constant shift of a
    # row's logits.
    step = 2.5
    c_kv = c_kv.clone()
    c_kv[:, tail:] = torch.arange(case["s_kv"], dtype=torch.float32).unsqueeze(1) * step
    q_lift = q_lift.clone()
    q_lift[:, :, tail:] = 1.0

    got, counters = dispatch_once(case, q_lift, c_kv, idx)
    assert counters == LATENT_TILED

    expected = sparse_mla_torch_reference(q_lift, c_kv, idx, scale)
    torch.testing.assert_close(got, expected, rtol=RTOL, atol=ATOL)


# --------------------------------------------------------------------------- #
# The row-tiled body.
# --------------------------------------------------------------------------- #
def order_rows_so_the_peak_lands_in_tile_one(q_lift, c_kv, idx):
    """Rotate each query's selection so its highest-scoring row sits at ``PEAK_POSITION``.

    The row-tiled body exponentiates each score tile against that tile's max and
    rescales the running denominator and accumulator when a later tile raises it.
    With the global max in the last tile the tile rescale is exactly 1.0; in the
    first tile the accumulator rescale is. A peak in a middle tile makes both do
    work. A softmax-weighted sum over a set of rows does not depend on the order
    the rows are listed in, so this changes which arm runs and not the answer.
    """
    ordered = idx.clone()
    for s in range(int(idx.shape[0])):
        gathered = c_kv[idx[s].to(torch.int64)].to(torch.float64)      # [K, L]
        row_score = (q_lift[s].to(torch.float64) @ gathered.t()).max(dim=0).values
        ascending = torch.argsort(row_score)
        shift = int(ascending.numel()) - 1 - PEAK_POSITION
        rotated = torch.cat([ascending[shift:], ascending[:shift]])
        ordered[s] = idx[s][rotated]
    return ordered


def test_score_tiles_partition_the_selected_row_axis() -> None:
    """2048 rows split into whole-chunk tiles no wider than the moving axis that
    cover the axis exactly; 512 rows are one tile."""
    topk = ROWS_CASE["topk"]
    tiles = MS._score_tiles(topk)
    assert len(tiles) == -(-topk // MS.MOVING_MAX)
    covered = 0
    for lo, extent in tiles:
        assert lo == covered, f"the score tiles are not contiguous at offset {lo}"
        assert extent <= MS.MOVING_MAX, f"tile extent {extent} exceeds the moving axis"
        assert extent % MS.KEY_CHUNK == 0, (
            f"tile extent {extent} is not a whole number of {MS.KEY_CHUNK}-key chunks"
        )
        covered += extent
    assert covered == topk

    single = MS._score_tiles(SPLIT_CASE["topk"])
    assert len(single) == 1 and tuple(single[0]) == (0, SPLIT_CASE["topk"])


def test_production_selected_row_count_matches_the_torch_reference() -> None:
    """2048 selected rows take the row-tiled body, with the peak score in a middle
    tile, and agree with float64 torch."""
    case = dict(ROWS_CASE)
    q_lift, c_kv, idx, _, _ = make_case(**case, seed=93)
    idx = order_rows_so_the_peak_lands_in_tile_one(q_lift, c_kv, idx)

    got, counters = dispatch_once(case, q_lift, c_kv, idx)
    assert counters == ROW_TILED

    expected = sparse_mla_torch_reference(q_lift, c_kv, idx, case_scale(case))
    torch.testing.assert_close(got, expected, rtol=RTOL, atol=ATOL)


def test_row_tiled_body_is_bit_identical_to_the_untiled_body_at_one_score_tile() -> None:
    """At topk=512 the row-tiled kernel emits no merge, so it must equal the untiled kernel exactly."""
    case = dict(SPLIT_CASE)
    scale = case_scale(case)
    q_lift, c_kv, idx, _, _ = make_case(**case, seed=94)
    q_f32 = q_lift.contiguous().to(torch.float32)
    c_f32 = c_kv.contiguous().to(torch.float32)
    i_i32 = idx.contiguous().to(torch.int32)

    # Both kernels are called directly: the seam routes topk=512 to the untiled
    # body, so the row-tiled body has to be forced to compare them.
    untiled = wrap_nki(MS.mla_sparse_attention_nope_kernel)(q_f32, c_f32, i_i32, scale)
    forced = wrap_nki(MS.mla_sparse_attention_nope_row_tiled_kernel)(
        q_f32, c_f32, i_i32, scale
    )

    assert float(untiled.abs().max()) > 0.0
    diff = float((untiled - forced).abs().max())
    assert diff == 0.0, (
        f"the forced row-tiled body disagrees with the untiled body by {diff} at one "
        f"score tile, where it emits no merge at all"
    )


def test_each_body_counts_on_its_own_counter_and_resets_alone() -> None:
    """The seam counter counts every dispatch, each tiling's counter counts its own
    body, and resetting one counter leaves the others standing."""
    assert len({id(MS._MLA_SPARSE_COUNTERS), id(MS._MLA_SPARSE_TILED_COUNTERS),
                id(MS._MLA_SPARSE_ROW_TILED_COUNTERS)}) == 3

    reset_all_counters()
    assert all_counters() == (0, 0, 0, 0, 0, 0)

    q, c, i, _, _ = make_case(**SPLIT_CASE, seed=94)
    MS.mla_sparse_attention(q, c, i, case_scale(SPLIT_CASE))
    assert all_counters() == (1, 0, 0, 0, 0, 0)

    q, c, i, _, _ = make_case(**MINIMAL_TILED_CASE, seed=43)
    MS.mla_sparse_attention(q, c, i, case_scale(MINIMAL_TILED_CASE))
    assert all_counters() == (2, 0, 1, 0, 0, 0)

    q, c, i, _, _ = make_case(**MINIMAL_ROW_TILED_CASE, seed=95)
    MS.mla_sparse_attention(q, c, i, case_scale(MINIMAL_ROW_TILED_CASE))
    assert all_counters() == (3, 0, 1, 0, 1, 0)

    MS.reset_mla_sparse_tiled_dispatch_counters()
    assert all_counters() == (3, 0, 0, 0, 1, 0)
    MS.reset_mla_sparse_row_tiled_dispatch_counters()
    assert all_counters() == (3, 0, 0, 0, 0, 0)


# --------------------------------------------------------------------------- #
# The sentinel mask. All three bodies read the index tensor, so all three carry it.
# --------------------------------------------------------------------------- #

# Which row carries which sentinel pattern. Rows past the fifth are left clean.
ROW_LEADING_RUN = 0
ROW_TRAILING_RUN = 1
ROW_INTERIOR_COLUMN = 2
ROW_WHOLLY_SENTINEL = 3
ROW_ONE_VALID_COLUMN = 4


def sow_sentinels(idx):
    """Doctor a clean index tensor into the row layout above; return the kept column's cache row."""
    topk = int(idx.shape[1])
    run = max(1, topk // 8)
    idx[ROW_LEADING_RUN, 0:run] = MS.SENTINEL_INDEX
    idx[ROW_TRAILING_RUN, topk - run:topk] = MS.SENTINEL_INDEX
    idx[ROW_INTERIOR_COLUMN, topk // 2] = MS.SENTINEL_INDEX
    idx[ROW_WHOLLY_SENTINEL, :] = MS.SENTINEL_INDEX
    live = topk // 3
    kept = int(idx[ROW_ONE_VALID_COLUMN, live])
    idx[ROW_ONE_VALID_COLUMN, :] = MS.SENTINEL_INDEX
    idx[ROW_ONE_VALID_COLUMN, live] = kept
    return kept


def sentinel_reference(q_lift, c_kv, idx, softmax_scale):
    """float64 reference over each row's live columns only, sentinel columns dropped.

    The sentinel columns are removed from the tensors rather than weighted to
    zero, so a kernel leaking a small weight onto a masked column cannot hide
    behind a zero multiply. A row with no live column contributes exact zeros.
    """
    q = q_lift.to(torch.float64)
    cache = c_kv.to(torch.float64)
    rows = idx.to(torch.int64)
    out = []
    for s in range(q.shape[0]):
        live = rows[s][rows[s] >= 0]
        if int(live.numel()) == 0:
            out.append(torch.zeros(q.shape[1], q.shape[2], dtype=torch.float64))
            continue
        gathered = cache[live]                                     # [live, L]
        scores = torch.einsum("hl,kl->hk", q[s], gathered) * softmax_scale
        shifted = scores - scores.max(dim=-1, keepdim=True).values
        expd = torch.exp(shifted)
        weights = expd / expd.sum(dim=-1, keepdim=True)
        out.append(torch.einsum("hk,kl->hl", weights, gathered))
    return torch.stack(out).to(torch.float32)


def attending_reference(q_lift, c_kv, idx, softmax_scale):
    """The wrong answer the mask exists to avoid: the sentinel attended as cache row 0."""
    return sparse_mla_torch_reference(q_lift, c_kv, idx.clamp(min=0), softmax_scale)


def run_sentinel_case(case: dict, seed: int, expected_counters: tuple):
    """Build a case, sow the sentinel rows, dispatch once through the seam.

    Returns ``(got, q_lift, c_kv, idx, scale, kept)``.
    """
    q_lift, c_kv, idx, q_pe, k_pe = make_case(**case, seed=seed)
    assert q_pe is None and k_pe is None
    kept = sow_sentinels(idx)
    got, counters = dispatch_once(case, q_lift, c_kv, idx)
    assert counters == expected_counters, f"got {counters}, expected {expected_counters}"
    return got, q_lift, c_kv, idx, case_scale(case), kept


def assert_agrees_over_live_columns(got, q_lift, c_kv, idx, scale) -> None:
    """Kernel and shipped oracle both match the reference that drops the sentinel
    columns, and neither matches the one that attends them as row 0."""
    expected = sentinel_reference(q_lift, c_kv, idx, scale)
    attending = attending_reference(q_lift, c_kv, idx, scale)

    torch.testing.assert_close(got, expected, rtol=RTOL, atol=ATOL)
    assert float((got - attending).abs().max()) > ATOL, (
        "the kernel attended the sentinel as cache row 0 instead of masking it"
    )

    oracle = MS.mla_sparse_attention_torch_oracle(q_lift, c_kv, idx, scale)
    torch.testing.assert_close(oracle, expected, rtol=RTOL, atol=ATOL)
    assert float((oracle - attending).abs().max()) > ATOL, (
        "the oracle attended the sentinel as cache row 0 instead of masking it"
    )


def assert_empty_row_is_exactly_zero(got) -> None:
    """A row whose every column is the sentinel reads exact zeros; a row with one
    live column does not."""
    empty = float(got[ROW_WHOLLY_SENTINEL].abs().max())
    assert empty == 0.0, (
        f"a row whose every selected column is the sentinel must read exact zeros; got "
        f"{empty!r}. A NaN here is the divide-by-zero the mask avoids by leaving the "
        f"denominator at its unmasked value"
    )
    assert float(got[ROW_ONE_VALID_COLUMN].abs().max()) != 0.0


def test_sentinel_columns_are_masked_in_the_untiled_body() -> None:
    got, q_lift, c_kv, idx, scale, _ = run_sentinel_case(dict(SENTINEL_UNTILED), 98, UNTILED)
    assert_agrees_over_live_columns(got, q_lift, c_kv, idx, scale)


def test_wholly_sentinel_row_reads_zeros_and_one_live_column_reads_its_cache_row() -> None:
    case = dict(SENTINEL_UNTILED)
    got, _, c_kv, _, _, kept = run_sentinel_case(case, 198, UNTILED)
    assert_empty_row_is_exactly_zero(got)
    # With one live column the softmax weight is 1.0, so the row reads that cache row back.
    expected = c_kv[kept].to(torch.float32).expand(case["heads"], case["latent"])
    torch.testing.assert_close(got[ROW_ONE_VALID_COLUMN], expected, rtol=RTOL, atol=ATOL)


def test_index_range_admits_the_sentinel_and_refuses_beyond_it() -> None:
    """``-1`` is admitted; ``-2`` and ``s_kv`` are refused with messages naming the range."""
    case = dict(SENTINEL_UNTILED)
    got, q_lift, c_kv, idx, scale, _ = run_sentinel_case(case, 398, UNTILED)
    assert int(idx.min()) == MS.SENTINEL_INDEX

    below = idx.clone()
    below[0, 0] = MS.SENTINEL_INDEX - 1
    with pytest.raises(MS.MlaSparseAttentionError) as low:
        MS.mla_sparse_attention(q_lift, c_kv, below, scale)
    assert str(MS.SENTINEL_INDEX) in str(low.value), (
        "the refusal must name the sentinel it admits"
    )
    assert f"[{MS.SENTINEL_INDEX - 1}," in str(low.value), (
        "the refusal must name the offending range it read"
    )

    above = idx.clone()
    above[0, 0] = case["s_kv"]
    with pytest.raises(MS.MlaSparseAttentionError) as high:
        MS.mla_sparse_attention(q_lift, c_kv, above, scale)
    assert f"s_kv={case['s_kv']}" in str(high.value)


def test_sentinel_columns_are_masked_in_the_row_tiled_body() -> None:
    """Two score tiles with sentinels in both, one row wholly sentinel across both."""
    got, q_lift, c_kv, idx, scale, _ = run_sentinel_case(
        dict(SENTINEL_ROW_TILED), 598, ROW_TILED
    )
    assert_agrees_over_live_columns(got, q_lift, c_kv, idx, scale)
    assert_empty_row_is_exactly_zero(got)


def test_sentinel_columns_are_masked_in_the_tiled_latent_body() -> None:
    """The mask on the selected-row axis does not interfere with the latent tiling's ragged tail."""
    got, q_lift, c_kv, idx, scale, _ = run_sentinel_case(
        dict(SENTINEL_TILED), 698, LATENT_TILED
    )
    assert_agrees_over_live_columns(got, q_lift, c_kv, idx, scale)
    assert_empty_row_is_exactly_zero(got)
