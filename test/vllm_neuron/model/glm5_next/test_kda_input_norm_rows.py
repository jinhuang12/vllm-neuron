# SPDX-License-Identifier: Apache-2.0
"""KDA layer forward preserves singleton BF16 norm paths for tuple decode."""

import pytest
import torch
from torch import nn

from .test_mhc_composition import _impl, _kda_layer, _norm_ready, _text_config


class _RecordingAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, hidden, **kwargs):
        self.calls.append((hidden.clone(), kwargs))
        for name in ("conv_state", "recurrent_state"):
            state = kwargs[name]
            for tensor in state if isinstance(state, tuple) else (state,):
                tensor.add_(1)
        return hidden * 0.5


class _RecordingSite:
    def __init__(self, single_stream):
        self.single_stream = single_stream
        self.calls = []

    def forward(self, streams, sublayer, **kwargs):
        self.calls.append(kwargs)
        return streams + sublayer(self.single_stream).unsqueeze(1)


def _case(monkeypatch, hidden, count, is_prefill, streams_route):
    config = _text_config(hidden_size=64)
    layer = _norm_ready(_kda_layer(config), config)
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

    def state():
        return tuple(torch.zeros(1) for _ in range(count)) if count else torch.zeros(1)

    kwargs = dict(conv_state=state(), recurrent_state=state(), is_prefill=is_prefill)
    if streams_route:
        kwargs["streams"] = torch.zeros(2, 4, 64, dtype=hidden.dtype)
        if hidden.dim() == 2:
            kwargs["streams"] = hidden.unsqueeze(1).expand(-1, 4, -1).clone()
    return layer, attention, original_norm, norm_calls, site, kwargs


def _hidden(rows, dtype):
    generator = torch.Generator().manual_seed(530917)
    return torch.randn(rows, 64, generator=generator).to(dtype)


def _assert_forward_result(actual, hidden, expected_norm, kwargs, streams_route):
    attended = expected_norm * 0.5
    expected = (
        kwargs["streams"] + attended.unsqueeze(1)
        if streams_route
        else hidden + attended
    )
    assert actual.dtype == hidden.dtype
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("requests", [2, 4, 8])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("streams_route", [False, True])
def test_tuple_decode_norms_original_singletons_in_request_order(
    monkeypatch, requests, reverse, streams_route
):
    hidden = _hidden(requests, torch.bfloat16)
    if reverse:
        hidden = hidden.flip(0)
    layer, attention, original_norm, norm_calls, site, kwargs = _case(
        monkeypatch, hidden, requests, False, streams_route
    )
    expected_norm = torch.cat(
        [original_norm(row.unsqueeze(0)) for row in hidden], dim=0
    )
    collector = []
    actual = layer.forward(hidden, collector=collector, **kwargs)

    assert [tuple(value.shape) for value in norm_calls] == [(1, 64)] * requests
    for row, value in enumerate(norm_calls):
        assert torch.equal(value, hidden[row : row + 1])
    assert len(attention.calls) == 1
    received, passed = attention.calls[0]
    assert received.dtype == torch.bfloat16
    assert torch.equal(received, expected_norm)
    assert not torch.equal(received[0], received[1])
    for name in ("conv_state", "recurrent_state"):
        assert passed[name] is kwargs[name]
        assert all(torch.equal(tensor, torch.ones(1)) for tensor in kwargs[name])
    assert len(collector) == 1
    assert torch.equal(collector[0], expected_norm * 0.5)
    if site is not None:
        assert site.calls == [{"num_requests": requests}]
    _assert_forward_result(actual, hidden, expected_norm, kwargs, streams_route)


@pytest.mark.parametrize(
    "rows,count,is_prefill,dtype",
    [
        (1, 0, False, torch.bfloat16),
        (1, 1, False, torch.bfloat16),
        (4, 0, False, torch.bfloat16),
        (4, 2, True, torch.bfloat16),
        (4, 4, False, torch.float32),
        (4, 4, False, torch.float16),
    ],
)
@pytest.mark.parametrize("streams_route", [False, True])
def test_other_routes_keep_one_original_norm_call(
    monkeypatch, rows, count, is_prefill, dtype, streams_route
):
    hidden = _hidden(rows, dtype)
    layer, attention, original_norm, norm_calls, site, kwargs = _case(
        monkeypatch, hidden, count, is_prefill, streams_route
    )
    expected_norm = original_norm(hidden)
    actual = layer.forward(hidden, **kwargs)

    assert len(norm_calls) == 1
    assert norm_calls[0] is hidden
    assert len(attention.calls) == 1
    assert attention.calls[0][0].dtype == dtype
    assert torch.equal(attention.calls[0][0], expected_norm)
    if site is not None:
        expected_keywords = {"num_requests": count} if count > 1 and not is_prefill else {}
        assert site.calls == [expected_keywords]
    _assert_forward_result(actual, hidden, expected_norm, kwargs, streams_route)


@pytest.mark.parametrize("shape", [(1, 64), (3, 64), (64,), (2, 1, 64)])
@pytest.mark.parametrize("streams_route", [False, True])
def test_invalid_bf16_rows_refuse_before_norm_attention_or_state_update(
    monkeypatch, shape, streams_route
):
    hidden = torch.ones(shape, dtype=torch.bfloat16)
    layer, attention, _, norm_calls, _, kwargs = _case(
        monkeypatch, hidden, 2, False, streams_route
    )
    collector = []
    with pytest.raises(ValueError, match="concurrent BF16 KDA input norm"):
        layer.forward(hidden, collector=collector, **kwargs)
    assert norm_calls == []
    assert attention.calls == []
    assert collector == []
    for name in ("conv_state", "recurrent_state"):
        assert all(torch.equal(tensor, torch.zeros(1)) for tensor in kwargs[name])
