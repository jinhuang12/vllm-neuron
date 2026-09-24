# SPDX-License-Identifier: Apache-2.0
"""Concurrent decode keeps singleton preparation and a batched post combine.

Run with NKI_SIMULATOR=1 NKI_PRECISE_FP=1. Native exactness remains a separate
gate; these tests check the public outputs, dispatch counts, and request routing.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch


def _impl():
    from vllm_neuron.model.glm5_next import model_fp8

    return model_fp8


def _case(tokens, dtype=torch.bfloat16):
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    site = _impl().Glm5NextHyperConnection(
        Glm5NextTextConfig(hidden_size=64, hc_mult=4, hc_sinkhorn_iters=20)
    )
    generator = torch.Generator().manual_seed(921)
    with torch.no_grad():
        site.fn.data = (torch.randn(site.fn.shape, generator=generator) * 0.04).to(
            torch.bfloat16
        )
        site.hc_scale.copy_(torch.tensor([0.7, 0.5, 0.3]))
        site.hc_base.copy_(torch.randn(site.hc_base.shape, generator=generator) * 0.1)
    residual = torch.randn(tokens, 4, 64, generator=generator).to(dtype)
    return site, residual


class _ProjectionRows:
    """Record Python matmul operands without intercepting the NKI higher-order op."""

    def __init__(self):
        self.rows = []

    def __enter__(self):
        original = torch.Tensor.__matmul__

        def record(left, right):
            self.rows.append(left.shape[0])
            return original(left, right)

        self._patch = patch.object(torch.Tensor, "__matmul__", record)
        self._patch.__enter__()
        return self

    def __exit__(self, *exc):
        return self._patch.__exit__(*exc)


@pytest.mark.parametrize("tokens", [2, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_concurrent_pre_matches_singletons_with_per_request_sinkhorn(tokens, dtype):
    from vllm_neuron.functional.mhc import sinkhorn

    site, residual = _case(tokens, dtype)
    # Reverse the rows so the comparison also fixes the request/output ordering.
    residual = residual.flip(0)
    expected = [site.mhc_pre(row.unsqueeze(0)) for row in residual]
    sinkhorn.reset_dispatch_counters()
    with _ProjectionRows() as projections:
        actual = site.mhc_pre(residual, num_requests=tokens)
    assert projections.rows == [1] * tokens
    assert sinkhorn.dispatch_counters() == (tokens, 0)
    for field, tensor in enumerate(actual):
        reference = torch.cat([row[field] for row in expected])
        assert tensor.dtype == reference.dtype
        assert torch.equal(tensor, reference)
    assert not torch.equal(actual[2][0], actual[2][1])


@pytest.mark.parametrize("tokens", [1, 2, 8])
def test_default_pre_keeps_original_batched_projection(tokens):
    site, residual = _case(tokens)
    with _ProjectionRows() as projections:
        implicit = site.mhc_pre(residual)
    assert projections.rows == [tokens]
    explicit = site.mhc_pre(residual, num_requests=1)
    assert all(torch.equal(left, right) for left, right in zip(implicit, explicit))


def test_concurrent_forward_calls_sublayer_and_post_once():
    from vllm_neuron.functional.mhc import hyper_connection, sinkhorn

    site, residual = _case(2)
    expected = torch.cat(
        [site.forward(row.unsqueeze(0), torch.tanh) for row in residual]
    )
    calls = []

    def sublayer(hidden):
        calls.append(hidden.clone())
        return torch.tanh(hidden)

    sinkhorn.reset_dispatch_counters()
    hyper_connection.reset_dispatch_counters()
    actual = site.forward(residual, sublayer, num_requests=2)
    assert [tuple(value.shape) for value in calls] == [(2, 64)]
    assert sinkhorn.dispatch_counters() == (2, 0)
    assert hyper_connection.dispatch_counters() == (1, 0)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("num_requests", [0, -1, True, 1.0, 1.5, 3])
@pytest.mark.parametrize("method", ["mhc_pre", "forward"])
def test_invalid_request_counts_refuse_before_projection_or_sublayer(
    num_requests, method
):
    from vllm_neuron.functional.mhc import hyper_connection, sinkhorn

    site, residual = _case(2)
    sinkhorn.reset_dispatch_counters()
    hyper_connection.reset_dispatch_counters()

    def forbidden(_hidden):
        pytest.fail("invalid request count reached the sublayer")

    args = (residual, forbidden) if method == "forward" else (residual,)
    with _ProjectionRows() as projections:
        with pytest.raises(_impl().Glm5NextHyperConnectionError):
            getattr(site, method)(*args, num_requests=num_requests)
    assert projections.rows == []
    assert sinkhorn.dispatch_counters() == (0, 0)
    assert hyper_connection.dispatch_counters() == (0, 0)


class _Site:
    def __init__(self):
        self.calls = []

    def forward(self, streams, sublayer, **kwargs):
        self.calls.append(kwargs)
        return streams + sublayer(streams.mean(dim=1)).unsqueeze(1)


def _carrier(family, count, is_prefill=False):
    state = tuple(torch.zeros(1) for _ in range(count)) if count else torch.zeros(1)
    if family == "kda":
        return dict(conv_state=state, recurrent_state=state, is_prefill=is_prefill)
    return dict(
        latent_cache=torch.zeros(1),
        pool_cache=state,
        seq_lens=torch.ones(1, dtype=torch.int32),
        start_position=0,
        softmax_scale=1.0,
        max_seq_len=16,
        page_size=4,
        block_table_row=torch.zeros(1, dtype=torch.int32),
        latent_slots=torch.zeros(1, dtype=torch.int32),
    )


@pytest.mark.parametrize("family", ["kda", "dsa"])
@pytest.mark.parametrize("count", [0, 1, 2, 8])
def test_attention_site_uses_request_tuple_count(monkeypatch, family, count):
    impl = _impl()
    site = _Site()
    monkeypatch.setattr(impl, "_mhc_attention_site", lambda *_args: site)
    calls = []

    def attention(hidden, **kwargs):
        calls.append((hidden.clone(), kwargs))
        return hidden * 2

    layer = SimpleNamespace(attention=attention, _input_norm=lambda hidden: hidden)
    tokens = max(count, 2)  # Multiple token rows alone must not select the repair.
    streams = torch.arange(tokens * 4 * 3).reshape(tokens, 4, 3).float()
    carrier = _carrier(family, count)
    cls = impl.Glm5NextKDALayer if family == "kda" else impl.Glm5NextDSALayer
    result = cls.forward(layer, streams, streams=streams, **carrier)
    assert site.calls == ([{"num_requests": count}] if count > 1 else [{}])
    assert len(calls) == 1
    assert tuple(calls[0][0].shape) == (tokens, 3)
    field = "conv_state" if family == "kda" else "pool_cache"
    assert calls[0][1][field] is carrier[field]
    assert torch.equal(result, streams + streams.mean(dim=1).unsqueeze(1) * 2)


def test_kda_prefill_tuple_keeps_default_site_call(monkeypatch):
    impl = _impl()
    site = _Site()
    monkeypatch.setattr(impl, "_mhc_attention_site", lambda *_args: site)
    layer = SimpleNamespace(
        attention=lambda hidden, **_kwargs: hidden,
        _input_norm=lambda hidden: hidden,
    )
    streams = torch.ones(8, 4, 3)
    impl.Glm5NextKDALayer.forward(
        layer, streams, streams=streams, **_carrier("kda", 2, is_prefill=True)
    )
    assert site.calls == [{}]


@pytest.mark.parametrize(
    "family,count,is_prefill",
    [
        ("kda", 0, False),
        ("kda", 1, False),
        ("kda", 2, False),
        ("kda", 8, False),
        ("kda", 2, True),
        ("dsa", 0, False),
        ("dsa", 1, False),
        ("dsa", 2, False),
        ("dsa", 8, False),
    ],
)
def test_ffn_site_uses_same_request_metadata(monkeypatch, family, count, is_prefill):
    impl = _impl()
    site = _Site()
    monkeypatch.setattr(impl, "_mhc_ffn_site", lambda *_args: site)
    attention_calls, ffn_calls = [], []

    def attention(hidden_states, **kwargs):
        attention_calls.append(kwargs)
        return hidden_states

    def ffn(layer, hidden, **kwargs):
        assert layer is attention
        ffn_calls.append(hidden.clone())
        return hidden * 2

    tokens = max(count, 2)
    table = torch.arange(tokens * 3).reshape(tokens, 3).float()
    model = SimpleNamespace(
        layers=[attention],
        embed_tokens_weight=table,
        text_config=SimpleNamespace(hc_mult=4),
        norm_weight=torch.ones(3),
        _ffn_half=ffn,
        _rms_norm=lambda hidden, _gain: hidden,
    )
    carrier = _carrier(family, count, is_prefill)
    result = impl.Glm5NextModel.forward(
        model, torch.arange(tokens), layer_carriers=[carrier], quant_config=None
    )
    expected_keywords = {"num_requests": count} if count > 1 and not is_prefill else {}
    assert site.calls == [expected_keywords]
    assert len(attention_calls) == 1
    assert len(ffn_calls) == 1
    assert tuple(ffn_calls[0].shape) == (tokens, 3)
    assert torch.equal(result, table * 3)
