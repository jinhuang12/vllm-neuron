# SPDX-License-Identifier: Apache-2.0
"""Only sparse MLA may use the runner's static query prefix."""

from importlib import import_module
from types import SimpleNamespace

import pytest
import torch

from vllm_neuron.model.glm5_next.model_fp8 import (
    Glm5NextMLAAttention,
    Glm5NextMLADecodeError,
)

pytestmark = [pytest.mark.fast, pytest.mark.forked]


@pytest.fixture
def attention_path(monkeypatch):
    hidden = torch.arange(1, 25, dtype=torch.bfloat16).reshape(6, 4)
    cache = torch.full((10, 1, 4), -3, dtype=torch.bfloat16)
    indices = torch.arange(18, dtype=torch.int32).reshape(6, 3)
    calls = []

    def project(value):
        calls.append(("project", value.clone()))
        return value[:, None, :], value

    def absorb(value, weight):
        calls.append((weight, value.clone()))
        return value

    def sparse(query, latent, topk, scale):
        calls.append(("sparse", query.clone(), latent.clone(), topk.clone(), scale))
        return query.float() + 0.25

    def output(value, collector):
        calls.append(("output", value.clone()))
        return value[:, 0, :]

    module = SimpleNamespace(
        hidden_size=4,
        kv_lora_rank=4,
        NUM_LATENT_KV_HEADS=1,
        head_size=4,
        project_query_and_latent=project,
        _absorb_weight=lambda name: name,
        project_output=output,
    )
    monkeypatch.setattr(
        import_module("vllm_neuron.functional.attention.mla_absorb"),
        "mla_absorb", absorb,
    )
    monkeypatch.setattr(
        import_module("vllm_neuron.functional.attention.mla_sparse"),
        "mla_sparse_attention", sparse,
    )
    return module, hidden, cache, indices, calls


@pytest.mark.parametrize("prefix", [None, 6, 3, 1])
def test_only_sparse_attention_uses_the_query_prefix(attention_path, prefix):
    module, hidden, cache, indices, calls = attention_path
    collector = []
    # The existing positional API, including collector, remains valid.
    result = Glm5NextMLAAttention.attend(
        module, hidden, cache, 2, indices, 0.125, 1, None, collector,
        active_mla_query_rows=prefix,
    )
    rows = len(hidden) if prefix is None else prefix
    assert [call[0] for call in calls] == ["project", "W_UK", "sparse", "W_UV", "output"]
    for call in (calls[0], calls[1], calls[3], calls[4]):
        assert call[1].shape[0] == len(hidden)
    sparse = calls[2]
    torch.testing.assert_close(sparse[1], hidden[:rows, None, :], rtol=0, atol=0)
    torch.testing.assert_close(sparse[3], indices[:rows], rtol=0, atol=0)
    assert sparse[4] == 0.125
    assert sparse[2].shape == (10, 4)
    torch.testing.assert_close(sparse[2], cache[:, 0, :], rtol=0, atol=0)
    torch.testing.assert_close(cache[2:8, 0], hidden, rtol=0, atol=0)
    assert torch.all(cache[:2] == -3) and torch.all(cache[8:] == -3)

    attended = collector[-2]
    assert attended.dtype == torch.float32
    assert attended.shape == (6, 1, 4)
    torch.testing.assert_close(attended[:rows, 0], hidden[:rows].float() + 0.25)
    assert torch.count_nonzero(attended[rows:]) == 0
    assert calls[3][1].dtype == hidden.dtype
    torch.testing.assert_close(result, attended[:, 0].to(hidden.dtype), rtol=0, atol=0)


@pytest.mark.parametrize("prefix", [0, -1, 7, True, 1.5, torch.tensor(3)])
def test_invalid_prefix_refuses_before_projection_or_cache_write(attention_path, prefix):
    module, hidden, cache, indices, calls = attention_path
    before = cache.clone()
    with pytest.raises(Glm5NextMLADecodeError, match="active_mla_query_rows"):
        Glm5NextMLAAttention.attend(
            module, hidden, cache, 2, indices, 0.125,
            active_mla_query_rows=prefix,
        )
    assert calls == []
    assert torch.equal(cache, before)


def test_prefix_does_not_hide_mismatched_index_rows(attention_path):
    module, hidden, cache, indices, calls = attention_path
    before = cache.clone()
    with pytest.raises(Glm5NextMLADecodeError, match="topk_indices must have 6 query rows"):
        Glm5NextMLAAttention.attend(
            module, hidden, cache, 2, indices[:3], 0.125,
            active_mla_query_rows=3,
        )
    assert calls == []
    assert torch.equal(cache, before)
