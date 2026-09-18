"""CPU-only contract tests. These tests do not import the production kernels."""

import pytest
import json
import torch

from .reference import (ATOL, RTOL, block_matmul, compact_reference,
                        make_fixture, metrics, routed_reference, swiglu)


@pytest.fixture(autouse=True, scope="module")
def one_cpu_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_scale_grid_has_independent_contraction_and_output_blocks():
    x = torch.cat((torch.ones(1, 128), torch.full((1, 128), -2.0)), dim=1)
    weight = torch.ones(256, 256).to(torch.float8_e4m3fn)
    scales = torch.tensor([[0.5, 1.5], [2.0, 0.25]])
    actual = block_matmul(x, weight, scales)
    assert torch.equal(actual[:, :128], torch.full((1, 128), -448.0))
    assert torch.equal(actual[:, 128:], torch.full((1, 128), 128.0))


def test_scale_applies_after_dot_without_bf16_dequant_rounding():
    x = torch.ones(1, 128)
    w = torch.full((128, 128), 1.125).to(torch.float8_e4m3fn)
    scale = torch.tensor([[0.0317]])
    actual = block_matmul(x, w, scale)
    expected = torch.full((1, 128), 144.0) * scale
    rounded_weight = (w.float() * scale).to(torch.bfloat16).float()
    assert torch.equal(actual, expected)
    assert not torch.equal(actual, x @ rounded_weight)


def test_swiglu_clamp_is_asymmetric_and_unbounded_mode_stays_unbounded():
    gates = torch.tensor([[-20.0, 20.0, -2.0, 2.0]])
    ups = torch.tensor([[-20.0, 20.0, -2.0, 2.0]])
    fused = torch.cat((gates, ups), dim=1)
    clamped = swiglu(fused, 3.0, 4.0)
    expected_gate = torch.tensor([[-20.0, 3.0, -2.0, 2.0]])
    expected_up = torch.tensor([[-4.0, 4.0, -2.0, 2.0]])
    assert torch.equal(clamped, expected_gate * torch.sigmoid(expected_gate) * expected_up)
    assert torch.equal(swiglu(fused), gates * torch.sigmoid(gates) * ups)
    wrong_symmetric = gates.clamp(-3, 3) * torch.sigmoid(gates.clamp(-3, 3)) * expected_up
    assert not torch.allclose(clamped, wrong_symmetric, atol=ATOL, rtol=RTOL)


def test_activation_rounds_to_bf16_before_down_but_affinity_stays_fp32():
    case = make_fixture(q=2, hidden=128, intermediate=128)
    gate, activated, output = compact_reference(
        case.hidden[:2], case.gate_weight[0], case.gate_scales[0],
        case.down_weight[0], case.down_scales[0], case.affinity[:2, 0])
    weight = case.down_weight[0].to(torch.bfloat16).float()
    expected = (activated.T.to(torch.bfloat16).float() @ weight) * case.down_scales[0, 0, 0]
    assert torch.equal(output, expected * case.affinity[:2, 0, None])
    no_round = (activated.T @ weight) * case.down_scales[0, 0, 0]
    assert not torch.equal(expected, no_round)
    assert not torch.equal(output, expected * case.affinity[:2, 0, None].to(torch.bfloat16).float())
    assert gate.dtype == activated.dtype == output.dtype == torch.float32


@pytest.mark.parametrize("q", [1, 2, 7, 127, 128])
@pytest.mark.parametrize("block", [128, 256])
def test_routing_holes_tails_and_multiple_expert_contributions(q, block):
    case = make_fixture(q=q, hidden=128, intermediate=128, experts=2, block=block, kind="routing")
    _, _, output = routed_reference(case)
    assert output.shape == (2 * block, 128)
    assert torch.count_nonzero(output[case.row_index < 0]) == 0
    for position, token in enumerate(case.row_index.tolist()):
        if token < 0:
            continue
        expert = position // block
        expected = compact_reference(
            case.hidden[token:token+1], case.gate_weight[expert], case.gate_scales[expert],
            case.down_weight[expert], case.down_scales[expert], case.affinity[token, expert])[2]
        torch.testing.assert_close(output[position], expected[0], atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize("kind", ["random", "cancellation", "zeros", "clamp"])
def test_fixture_is_finite_and_zero_padding_is_exact(kind):
    case = make_fixture(q=7, hidden=256, intermediate=256, kind=kind)
    stages = routed_reference(case)
    assert all(torch.isfinite(stage).all() for stage in stages)
    assert torch.count_nonzero(stages[-1][case.row_index < 0]) == 0
    if kind == "zeros":
        assert all(torch.count_nonzero(stage) == 0 for stage in stages)
    elif kind == "clamp":
        gate, up = stages[0].chunk(2, dim=1)
        assert (gate > case.gate_upper).any()
        assert (gate < -case.gate_upper).any()
        assert (up.abs() > case.up_upper).any()
    elif kind == "cancellation":
        assert torch.count_nonzero(stages[0][:7, 1:256]) == 0
        assert torch.count_nonzero(stages[0][:7, 257:]) == 0
        assert torch.count_nonzero(stages[0][:7, 0]) > 0


def test_metrics_fixed_gate_detects_sparse_outliers_and_zeros():
    expected = torch.zeros(2, 128)
    assert metrics(expected, expected)["cosine"] == 1.0
    actual = expected.clone()
    actual[0, 0] = 2 * ATOL
    result = metrics(actual, expected)
    assert not result["allclose"]
    assert result["outside_tolerance"] == 1
    assert result["l2"] == result["max_abs"]
    assert result["atol"] == 1e-5 and result["rtol"] == 3e-2


def test_device_timing_groups_cores_by_launch_instead_of_averaging_events():
    from .baseline import summarize_trace
    events = []
    for launch, durations in enumerate(((1000, 9000), (5000, 7000))):
        for core, duration in enumerate(durations):
            key = launch * 2 + core
            # Deliberately different clock origins must not alter the maximum
            # core duration used to compare a pair of kernel implementations.
            start = core * 1000000
            events += [
                {"event_type": "nc_exec_running", "phase": "start", "tracking_id": key,
                 "data": {"exec_id": launch, "device_core_idx": core, "nc_timestamp_ns": start}},
                {"event_type": "nc_exec_running", "phase": "stop", "tracking_id": key,
                 "data": {"nc_timestamp_ns": start + duration}},
            ]
    result = summarize_trace(json.dumps({"events": events}), 2)
    assert result["mean_us"] == 8.0
    assert result["iterations"] == 2
    assert result["launch_max_core_us"] == [9.0, 7.0]
    with pytest.raises(ValueError, match="Incomplete timing"):
        summarize_trace(json.dumps({"events": events}), 3)


@pytest.mark.parametrize("q", [256, 511, 512])
def test_large_bucket_fixture_preserves_real_rows_and_padding(q):
    case = make_fixture(q=q, hidden=128, intermediate=128, block=512)
    output = routed_reference(case)[-1]
    assert output.shape == (512, 128)
    assert int((case.row_index >= 0).sum()) == q
    assert torch.count_nonzero(output[q:]) == 0
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("block", [0, 64, 129, -128])
def test_fixture_refuses_nonblocked_extents(block):
    with pytest.raises(ValueError, match="positive multiple of 128"):
        make_fixture(q=1, hidden=128, intermediate=128, block=block)
