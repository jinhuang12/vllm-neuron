import math

import pytest
import torch

from .reference import Inputs, make_fixture, metrics, sparse_reference


def test_duplicates_are_softmax_terms():
    case = Inputs(torch.zeros(1, 1, 2), torch.tensor([[2., 0.], [0., 3.]]),
                  torch.tensor([[0, 0, 1]], dtype=torch.int32), 1.)
    torch.testing.assert_close(sparse_reference(case), torch.tensor([[[4 / 3, 1.]]]))


def test_sentinel_zero_and_one_selected_row():
    case = make_fixture(seq=3, heads=2, latent=7, cache_rows=13, topk=9, kind="sentinel")
    result = sparse_reference(case)
    assert torch.count_nonzero(result[0]) == 0
    torch.testing.assert_close(result[1], case.cache[-1].float().expand(2, -1))


def test_rope_limb_changes_scores_but_not_values():
    case = Inputs(torch.zeros(1, 1, 2), torch.eye(2),
                  torch.tensor([[0, 1]], dtype=torch.int32), 1.,
                  torch.tensor([[[1.]]]), torch.tensor([[math.log(3.)], [0.]]))
    torch.testing.assert_close(sparse_reference(case), torch.tensor([[[.75, .25]]]))


@pytest.mark.parametrize("bad", [-2, 3])
def test_invalid_indices_rejected(bad):
    case = make_fixture(latent=4, cache_rows=3, topk=2)
    case.indices[0, 0] = bad
    with pytest.raises(ValueError, match="Indices"):
        sparse_reference(case)


def test_scale_and_selected_rows_are_observable():
    case = Inputs(torch.tensor([[[2.]]]), torch.tensor([[1.], [3.]]),
                  torch.tensor([[0, 1]], dtype=torch.int32), .5)
    expected = (1 + 3 * math.exp(2)) / (1 + math.exp(2))
    torch.testing.assert_close(sparse_reference(case), torch.tensor([[[expected]]]))
    before = sparse_reference(case)
    case.scale = 1.
    assert not torch.allclose(before, sparse_reference(case))


def test_metrics_use_existing_tolerance():
    result = metrics(torch.ones(2), torch.tensor([1., 1.1]))
    assert not result["allclose"]
    assert result["outside_tolerance"] == 1
    assert result["atol"] == 1e-5 and result["rtol"] == 1e-2
