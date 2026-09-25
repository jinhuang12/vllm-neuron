# SPDX-License-Identifier: Apache-2.0
"""DSA norm call-site contracts on CPU; these do not test compiler precision."""

import pytest
import torch
from torch import nn

from .test_mhc_composition import _dsa_layer, _impl, _norm_ready, _text_config


def _rows(value):
    return value if isinstance(value, tuple) else (value,)


class _RecordingAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.collector_before = None
        self.marker = torch.tensor([-17.0])
        self.output = None

    def forward(self, hidden, **kwargs):
        self.calls.append((hidden, kwargs))
        collector = kwargs.get("collector")
        if collector is not None:
            self.collector_before = list(collector)
            collector.append(self.marker)
        for row, (pool, tail, slot) in enumerate(
            zip(
                _rows(kwargs["pool_cache"]),
                _rows(kwargs["tail"]),
                _rows(kwargs["latent_slots"]),
                strict=True,
            )
        ):
            value = hidden[row, 0].float()
            pool.copy_(value.expand_as(pool))
            tail.copy_(value.expand_as(tail))
            kwargs["latent_cache"][int(slot.item())] = value
        kwargs["prefill_tail"].add_(1)
        self.output = hidden * 0.5
        return self.output


class _RecordingSite:
    def __init__(self, single_stream):
        self.single_stream = single_stream
        self.calls = []

    def forward(self, streams, sublayer, **kwargs):
        self.calls.append(kwargs)
        return streams + sublayer(self.single_stream).unsqueeze(1)


def _hidden(rows, dtype):
    generator = torch.Generator().manual_seed(530925)
    return torch.randn(rows, 64, generator=generator).to(dtype)


def _case(monkeypatch, hidden, request_order, tuple_carriers, streams_route):
    config = _text_config(hidden_size=64)
    layer = _norm_ready(_dsa_layer(config), config)
    with torch.no_grad():
        layer.input_layernorm_weight.copy_(torch.linspace(0.5, 1.5, 64))
    attention = _RecordingAttention()
    setattr(layer, layer.ATTENTION_ATTR, attention)
    original_norm = layer._input_norm
    norm_calls = []

    def recording_norm(value):
        norm_calls.append(value)
        return original_norm(value)

    monkeypatch.setattr(layer, "_input_norm", recording_norm)
    site = _RecordingSite(hidden) if streams_route else None
    monkeypatch.setattr(_impl(), "_mhc_attention_site", lambda *_args: site)

    def per_request(make):
        values = tuple(make(request) for request in request_order)
        return values if tuple_carriers else values[0]

    kwargs = {
        "latent_cache": torch.full((2 * len(request_order) + 3,), -99.0),
        "pool_cache": per_request(lambda request: torch.tensor([request + 10.0])),
        "seq_lens": per_request(
            lambda request: torch.tensor([request + 17], dtype=torch.int32)
        ),
        "start_position": per_request(
            lambda request: torch.tensor([request + 3], dtype=torch.int32)
        ),
        "softmax_scale": 0.25,
        "max_seq_len": 64,
        "page_size": 8,
        "block_table_row": per_request(
            lambda request: torch.tensor(
                [[request + 11, request + 31]], dtype=torch.int32
            )
        ),
        "latent_slots": per_request(
            lambda request: torch.tensor([2 * request + 1], dtype=torch.int64)
        ),
        "slot_mapping": torch.tensor(
            [request + 7 for request in request_order], dtype=torch.int64
        ),
        "tail": per_request(lambda request: torch.tensor([request + 20.0])),
        "position": per_request(
            lambda request: torch.tensor([request + 5], dtype=torch.int32)
        ),
        "prefill_tail": torch.tensor([61.0]),
        "prefill_end_position": torch.tensor([91], dtype=torch.int32),
        "active_mla_query_rows": 1,
    }
    if streams_route:
        kwargs["streams"] = (
            hidden.unsqueeze(1).expand(-1, 4, -1).clone()
            if hidden.dim() == 2
            else torch.zeros(2, 4, 64, dtype=hidden.dtype)
        )
    return layer, attention, original_norm, norm_calls, site, kwargs


def _bank_copies(kwargs):
    return {
        name: [tensor.clone() for tensor in _rows(kwargs[name])]
        for name in ("latent_cache", "pool_cache", "tail", "prefill_tail")
    }


def _assert_attention_and_result(actual, hidden, expected_norm, attention, kwargs):
    assert len(attention.calls) == 1
    received, passed = attention.calls[0]
    assert received.dtype == hidden.dtype
    assert torch.equal(received, expected_norm)
    for name in (
        "latent_cache", "pool_cache", "seq_lens", "start_position",
        "block_table_row", "latent_slots", "slot_mapping", "tail", "position",
        "prefill_tail", "prefill_end_position",
    ):
        assert passed[name] is kwargs[name], name
    for name in ("softmax_scale", "max_seq_len", "page_size", "active_mla_query_rows"):
        assert passed[name] == kwargs[name]
    expected = (
        kwargs["streams"] + (expected_norm * 0.5).unsqueeze(1)
        if "streams" in kwargs
        else hidden + expected_norm * 0.5
    )
    assert actual.dtype == hidden.dtype
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("requests", [2, 4])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("streams_route", [False, True])
def test_tuple_bf16_norm_preserves_request_carriers_and_collector_order(
    monkeypatch, requests, reverse, streams_route
):
    order = tuple(reversed(range(requests))) if reverse else tuple(range(requests))
    hidden = _hidden(requests, torch.bfloat16)[list(order)]
    layer, attention, original_norm, norm_calls, site, kwargs = _case(
        monkeypatch, hidden, order, True, streams_route
    )
    expected_norm = torch.cat(
        [original_norm(hidden[i : i + 1]) for i in range(requests)]
    )
    before = _bank_copies(kwargs)
    sentinel = torch.tensor([-7.0])
    collector = [sentinel]
    actual = layer.forward(hidden, collector=collector, **kwargs)

    assert [tuple(value.shape) for value in norm_calls] == [(1, 64)] * requests
    for row, value in enumerate(norm_calls):
        assert torch.equal(value, hidden[row : row + 1])
    assert not torch.equal(expected_norm[0], expected_norm[1])
    _assert_attention_and_result(actual, hidden, expected_norm, attention, kwargs)
    received, passed = attention.calls[0]
    assert passed["collector"] is collector
    assert len(attention.collector_before) == 3
    assert attention.collector_before[0] is sentinel
    assert attention.collector_before[1] is hidden
    assert attention.collector_before[2] is received
    assert len(collector) == 5
    assert all(
        left is right
        for left, right in zip(collector[:3], attention.collector_before, strict=True)
    )
    assert collector[3] is attention.marker
    assert collector[4] is attention.output

    expected_cache = before["latent_cache"][0]
    for row, request in enumerate(order):
        value = expected_norm[row, 0].float()
        expected_cache[2 * request + 1] = value
        for name in ("pool_cache", "tail"):
            assert torch.equal(kwargs[name][row], value.expand_as(kwargs[name][row]))
    assert torch.equal(kwargs["latent_cache"], expected_cache)
    assert torch.equal(kwargs["prefill_tail"], before["prefill_tail"][0] + 1)
    if site is not None:
        assert site.calls == [{"num_requests": requests}]


@pytest.mark.parametrize(
    "rows,requests,tuple_carriers,dtype",
    [
        (1, 1, False, torch.bfloat16),
        (1, 1, True, torch.bfloat16),
        (3, 1, False, torch.bfloat16),
        (2, 2, True, torch.float32),
        (2, 2, True, torch.float16),
    ],
)
@pytest.mark.parametrize("streams_route", [False, True])
def test_singleton_and_other_dtypes_keep_one_original_norm_call(
    monkeypatch, rows, requests, tuple_carriers, dtype, streams_route
):
    hidden = _hidden(rows, dtype)
    layer, attention, original_norm, norm_calls, site, kwargs = _case(
        monkeypatch, hidden, tuple(range(requests)), tuple_carriers, streams_route
    )
    expected_norm = original_norm(hidden)
    actual = layer.forward(hidden, **kwargs)

    assert len(norm_calls) == 1
    assert norm_calls[0] is hidden
    _assert_attention_and_result(actual, hidden, expected_norm, attention, kwargs)
    assert "collector" not in attention.calls[0][1]
    if site is not None:
        assert site.calls == ([{"num_requests": requests}] if requests > 1 else [{}])


@pytest.mark.parametrize("shape", [(0, 64), (1, 64), (3, 64), (64,), (2, 1, 64)])
@pytest.mark.parametrize("streams_route", [False, True])
def test_malformed_bf16_rows_fail_before_norm_attention_or_cache_writes(
    monkeypatch, shape, streams_route
):
    hidden = torch.ones(shape, dtype=torch.bfloat16)
    layer, attention, _, norm_calls, _, kwargs = _case(
        monkeypatch, hidden, (0, 1), True, streams_route
    )
    before = _bank_copies(kwargs)
    sentinel = torch.tensor([-7.0])
    collector = [sentinel]
    with pytest.raises(ValueError, match="DSA decode rows must match request carriers"):
        layer.forward(hidden, collector=collector, **kwargs)
    assert norm_calls == []
    assert attention.calls == []
    assert len(collector) == 1 and collector[0] is sentinel
    for name, saved in before.items():
        assert all(
            torch.equal(current, old)
            for current, old in zip(_rows(kwargs[name]), saved, strict=True)
        ), name
