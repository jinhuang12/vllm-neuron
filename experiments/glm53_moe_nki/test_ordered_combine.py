"""CPU contract checks for the private ordered expert-output combine.

These checks do not execute or simulate NKI. Native proof is a separate
campaign check; these source-derived CPU fixtures are packaged with the tests.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

from experiments.glm53_moe_nki import _ordered_combine_reference as baseline
from experiments.glm53_moe_nki import _ordered_combine_fixtures as comparison_helpers

ROOT = Path(__file__).resolve().parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


combine = _load("ordered_combine_test", "vllm_neuron/functional/moe/ordered_combine.py")


@pytest.fixture
def comparison():
    return comparison_helpers


def test_comparison_rejects_signed_zero_difference(comparison):
    result = comparison.compare(torch.tensor([-0.0]), torch.tensor([0.0]))
    assert result["finite"] and result["within_cpu_bounds"]
    assert result["different_bits_elements"] == 1 and not result["pass"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_comparison_rejects_nonfinite_real_output(comparison, value):
    result = comparison.compare(torch.tensor([value]), torch.tensor([value]))
    assert not result["finite"] and not result["pass"]


def test_comparison_computes_norm_in_float64(comparison):
    values = torch.randn(16384, generator=torch.Generator().manual_seed(13))
    result = comparison.compare(values, values.clone())
    assert result["pass"] and result["max_abs"] == 0 and result["difference_l2"] == 0
    assert result["reference_l2"] == float(torch.linalg.vector_norm(values.double()))
    assert abs(result["cosine_similarity"] - 1.0) < 1e-14


def test_concentrated_prefill_routes_cover_second_half(comparison):
    shape = dict(tokens=1024, hidden=16, global_experts=288, local_experts=18,
                 top_k=8, block_size=256, expert_parallel_rank=0)
    template, _ = baseline.mapping_fixture(shape, 530919, ROOT)
    fixture, proof = comparison.concentrated_fixture(template, ROOT, 530921)
    assert fixture["mapping_blocks"].shape == (49, 256)
    assert fixture["contribution"].shape == template["contribution"].shape
    assert proof["active_blocks"] == 32 and proof["valid_rows"] == 8192
    assert proof["valid_rows_in_second_flat_half"] == 1920
    assert proof["selected_pairs_emitted_once"] and proof["valid_id_uniqueness_per_block"]


@pytest.mark.parametrize("tokens", [128, 256, 257])
def test_rotating_routes_cover_all_local_experts_and_second_half(comparison, tokens):
    shape = dict(tokens=tokens, hidden=16, global_experts=288, local_experts=18,
                 top_k=8, block_size=256, expert_parallel_rank=0)
    template, _ = baseline.mapping_fixture(shape, 530919, ROOT)
    fixture, proof = comparison.concentrated_fixture(template, ROOT, 530922, "rotating-local")
    assert fixture["selected_global_experts"].unique().numel() == 18
    assert proof["valid_rows"] == tokens * 8
    assert proof["valid_rows_in_second_flat_half"] > 0
    assert proof["selected_pairs_emitted_once"] and proof["valid_id_uniqueness_per_block"]


def test_admitted_full_shape_cancellation_and_padding(comparison, original):
    variants = {name: fixture for name, fixture, _ in comparison.fixtures("t128-h1024", ROOT, 530919)}
    fixture = variants["full-shape-cancellation-and-scales"]
    ids, values = fixture["mapping_blocks"], fixture["contribution"]
    assert ids.shape == (21, 128) and values.shape == (2688, 1024)
    assert torch.equal(original(values, ids.reshape(-1), 128), torch.zeros(128, 1024))
    for token in range(128):
        positions = (ids.reshape(-1) == token).nonzero().flatten()
        assert positions.numel() == 8
        assert positions[0] // 128 < positions[-1] // 128
    fixture = variants["full-shape-padding-nonfinite"]
    padding = fixture["mapping_blocks"].reshape(-1) < 0
    assert (~torch.isfinite(fixture["contribution"][padding])).all()
    assert torch.isfinite(fixture["contribution"][~padding]).all()
    assert torch.isfinite(original(fixture["contribution"], fixture["mapping_blocks"].reshape(-1), 128)).all()


def test_full_prefill_sparse_controls_use_source_mapping(comparison, original, monkeypatch):
    monkeypatch.setitem(comparison.CASES, "prefill", (1024, 16))
    variants = {name: fixture for name, fixture, _ in comparison.fixtures("prefill", ROOT, 530919)}
    for name, count in (("full-shape-padding-only", 0), ("full-shape-one-active-row", 1)):
        fixture = variants[name]
        ids, data = fixture["mapping_blocks"], fixture["contribution"]
        assert ids.shape == (49, 256) and int((ids >= 0).sum()) == count
        actual = original(data, ids.reshape(-1), 1024)
        expected = torch.zeros(1024, 16)
        if count:
            expected[-1].fill_(1.25)
        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


@pytest.mark.parametrize("tokens,rows,hidden,expected", [
    (1024, 256, 4096, True), (128, 128, 1024, True),
    (256, 256, 1024, True), (257, 256, 2048, True),
    (3, 3, 130, False), (127, 127, 128, False),
    (128, 128, 384, False), (129, 129, 640, False),
    (257, 256, 1154, False),
])
def test_native_admission_requires_complete_input_tiles(tokens, rows, hidden, expected):
    values = torch.empty(2 * rows, hidden)
    ids = torch.empty(2, rows, dtype=torch.int32)
    assert combine._supported_geometry(values, ids, tokens) == expected


@pytest.mark.parametrize("blocks,rows,expected", [(20, 128, True), (21, 128, False),
                                                (25, 256, True), (29, 256, True)])
def test_native_admission_requires_complete_flattened_halves(blocks, rows, expected):
    assert combine._supported_geometry(torch.empty(blocks * rows, 1024),
                                       torch.empty(blocks, rows, dtype=torch.int32), 128) == expected


def test_structural_boundary_fixtures_keep_old_variants(comparison):
    cases = [("t128-h1024", 8, 21, False), ("t128-top6", 6, 20, True),
             ("t384", 8, 29, True)]
    for name, top_k, blocks, native in cases:
        variants = list(comparison.fixtures(name, ROOT, 530919))
        assert len(variants) == 5 and (name in comparison.NATIVE_CASES) == native
        for _, fixture, proof in variants:
            assert fixture["selected_global_experts"].shape[1] == top_k
            assert fixture["mapping_blocks"].shape[0] == blocks
            assert proof["valid_id_uniqueness_per_block"] and proof["selected_pairs_emitted_once"]
        assert variants[2][2]["valid_rows_in_second_flat_half"] > 0


@pytest.mark.parametrize("native,preserved,cpu_ok,expected", [
    (True, True, True, True), (True, True, False, False),
    (False, True, False, True), (False, False, True, False),
])
def test_preservation_gate_keeps_native_cpu_bounds(comparison, native, preserved, cpu_ok, expected):
    checks = {"baseline_vs_cpu": {"pass": cpu_ok}, "candidate_vs_cpu": {"pass": cpu_ok},
              "candidate_vs_baseline": {"pass": preserved}}
    assert comparison.preservation_pass(checks, native) == expected


@pytest.fixture
def original(tmp_path):
    function, _ = baseline.scatter_expression(ROOT, tmp_path)
    return function


@pytest.mark.parametrize("tokens", [1, 3, 127, 128, 129, 256, 257, 1024])
@pytest.mark.parametrize("hidden", [16, 130])
def test_actual_mapping_cpu_result_matches_original_bits(original, tokens, hidden):
    shape = dict(tokens=tokens, hidden=hidden, global_experts=288, local_experts=18,
                 top_k=8, block_size=256, expert_parallel_rank=0)
    fixture, proof = baseline.mapping_fixture(shape, 530919, ROOT)
    assert proof["valid_id_uniqueness_per_block"] and proof["selected_pairs_emitted_once"]
    values, rows = fixture["contribution"], fixture["mapping_blocks"]
    expected = original(values, rows.reshape(-1), tokens)
    actual = combine.ordered_combine(values, rows, tokens)
    assert actual.dtype == torch.float32 and actual.shape == (tokens, hidden)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


@pytest.mark.parametrize("rows", [2, 3, 127, 128, 129, 255, 256, 257, 385, 1025])
def test_waves_cover_rows_without_single_partition_tail(rows):
    waves = combine._row_waves(rows, 128)
    covered = [row for start, size in waves for row in range(start, start + size)]
    assert covered == list(range(rows))
    assert all(2 <= size <= 128 for _, size in waves)
    assert combine._row_waves(129, 128) == ((0, 127), (127, 2))


@pytest.mark.parametrize("padding", [float("nan"), float("inf"), -float("inf")])
def test_padding_cannot_contaminate_real_rows(original, padding):
    rows = torch.tensor([[0, -1, -1], [0, 2, -1], [-1, -1, -1]], dtype=torch.int32)
    values = torch.arange(rows.numel() * 6, dtype=torch.float32).reshape(-1, 6)
    values[rows.reshape(-1) < 0] = padding
    before = values.view(torch.int32).clone()
    actual = combine.ordered_combine(values, rows, 3)
    expected = original(values, rows.reshape(-1), 3)
    assert torch.isfinite(actual).all()
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    assert torch.equal(values.view(torch.int32), before)


def test_cross_block_cancellation_and_changed_routes(original):
    rows = torch.tensor([[0, 1], [0, 1], [0, 1], [-1, -1]], dtype=torch.int32)
    values = torch.tensor([2**24, -2**24, 1, -1, -(2**24), 2**24, 7, 9], dtype=torch.float32)
    values = values[:, None].expand(-1, 8).contiguous()
    for ids, data in ((rows, values), (rows.flip(0).contiguous(), values), (rows, -values)):
        actual = combine.ordered_combine(data, ids, 3)
        expected = original(data, ids.reshape(-1), 3)
        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


@pytest.mark.parametrize("tokens,rows,hidden", [(1, 1, 128), (3, 1, 128), (3, 3, 129)])
def test_unsupported_native_geometry_keeps_original(original, tokens, rows, hidden):
    ids = torch.arange(rows, dtype=torch.int32).remainder(tokens)[None, :]
    values = torch.randn(rows, hidden)
    assert not combine._supported_geometry(values, ids, tokens)
    actual = combine.ordered_combine(values, ids, tokens)
    expected = original(values, ids.reshape(-1), tokens)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


def test_noncontiguous_operands_keep_original(original):
    ids = torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.int32).t()
    values = torch.randn(8, ids.numel()).t()
    assert not combine._supported_geometry(values, ids, 3)
    actual = combine.ordered_combine(values, ids, 3)
    assert torch.equal(actual.view(torch.int32), original(values, ids.reshape(-1), 3).view(torch.int32))


@pytest.mark.parametrize("change", ["dtype", "indices", "rows", "tokens", "tile"])
def test_invalid_operands_refuse(change):
    values = torch.ones(6, 8)
    ids = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.int32)
    tokens, tile = 3, 512
    if change == "dtype":
        values = values.bfloat16()
    elif change == "indices":
        ids = ids.long()
    elif change == "rows":
        values = values[:-1]
    elif change == "tokens":
        tokens = True
    else:
        tile = 0
    with pytest.raises(ValueError):
        combine.ordered_combine(values, ids, tokens, BLOCK_H=tile)
