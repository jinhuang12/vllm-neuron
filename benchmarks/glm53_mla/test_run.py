import torch

from .run import compare_outputs


def test_default_grouping_rejects_a_difference_inside_cpu_tolerance():
    baseline = torch.ones(1, 1, 8)
    candidate = baseline + 1e-6
    for width in (None, 512):
        result = compare_outputs(candidate, baseline, width)
        assert result["accuracy"]["allclose"]
        assert not result["pass"]
    assert compare_outputs(baseline, baseline, None)["pass"]


def test_alternate_grouping_still_requires_cpu_tolerance():
    baseline = torch.ones(1, 1, 8)
    for width in (128, 256, 384):
        assert compare_outputs(baseline + 1e-6, baseline, width)["pass"]
        assert not compare_outputs(baseline + .1, baseline, width)["pass"]
