# SPDX-License-Identifier: Apache-2.0
"""Only sparse MLA may use the runner's static query prefix."""

from importlib import import_module
from types import SimpleNamespace

import pytest
import torch

from test.vllm_neuron.model.glm5_next.test_mla_decode import paged_operands
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
    # The paged operands the seam was handed, kept beside the positional ones: the bank
    # travels whole now and the window comes from the block table, so a call that lost
    # the table would still look right in `calls` alone.
    paged_calls = []

    def project(value):
        calls.append(("project", value.clone()))
        return value[:, None, :], value

    def absorb(value, weight):
        calls.append((weight, value.clone()))
        return value

    def sparse(query, latent, topk, scale, **paged):
        calls.append(("sparse", query.clone(), latent.clone(), topk.clone(), scale))
        paged_calls.append(paged)
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
    return module, hidden, cache, indices, calls, paged_calls


#: The cache above is ten slots of five, so the identity table names both blocks.
PAGE = 5


def paged(cache, start, tokens):
    """The three paged operands at this file's page size, from the one derivation."""
    return paged_operands(cache, start, tokens, page=PAGE)


@pytest.mark.parametrize("prefix", [0, -1, 7, True, 1.5, torch.tensor(3)])
def test_invalid_prefix_refuses_before_projection_or_cache_write(attention_path, prefix):
    module, hidden, cache, indices, calls, _ = attention_path
    before = cache.clone()
    with pytest.raises(Glm5NextMLADecodeError, match="active_mla_query_rows"):
        Glm5NextMLAAttention.attend(
            module, hidden, cache, 2, indices, 0.125,
            active_mla_query_rows=prefix,
            **paged(cache, 2, len(hidden)),
        )
    assert calls == []
    assert torch.equal(cache, before)


def test_prefix_does_not_hide_mismatched_index_rows(attention_path):
    module, hidden, cache, indices, calls, _ = attention_path
    before = cache.clone()
    with pytest.raises(Glm5NextMLADecodeError, match="topk_indices must have 6 query rows"):
        Glm5NextMLAAttention.attend(
            module, hidden, cache, 2, indices[:3], 0.125,
            active_mla_query_rows=3,
            **paged(cache, 2, len(hidden)),
        )
    assert calls == []
    assert torch.equal(cache, before)
