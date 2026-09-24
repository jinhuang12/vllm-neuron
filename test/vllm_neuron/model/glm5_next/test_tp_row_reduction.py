# SPDX-License-Identifier: Apache-2.0
"""Packing and call-site contracts; native TP64 exactness is a separate gate."""

from types import SimpleNamespace

import pytest
import torch


def _impl():
    from vllm_neuron.model.glm5_next import model_fp8

    return model_fp8


class _Group:
    def __init__(self):
        self.calls = []

    def all_reduce(self, value):
        self.calls.append((value.clone(), value.stride(), value.dtype))
        value.add_(0.03125)
        # The production seam must ignore this return value.
        return torch.full_like(value, float("nan"))


@pytest.mark.parametrize("requests", [2, 4, 8])
@pytest.mark.parametrize("contiguous", [False, True])
def test_concurrent_columns_share_contiguous_collective_rows(monkeypatch, requests, contiguous):
    impl = _impl()
    group = _Group()
    monkeypatch.setattr(impl, "_resolve_tp_group", lambda: group)
    # Both source layouts must produce the same contiguous collective layout.
    partials = torch.arange(requests * 7, dtype=torch.float32).reshape(7, requests).t()
    if contiguous:
        partials = partials.contiguous()
    before = partials.clone()
    actual = impl._reduce_tp_rows(partials, num_requests=requests)
    assert len(group.calls) == 1
    wire, stride, dtype = group.calls[0]
    assert wire.shape == (7, requests)
    assert stride == (requests, 1)
    assert dtype == torch.float32
    for column in range(7):
        assert torch.equal(wire[column], before[:, column])
    assert torch.equal(actual, before + 0.03125)
    assert actual.is_contiguous()


@pytest.mark.parametrize("tokens", [1, 9])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_default_single_request_and_prefill_keep_inplace_tensor(monkeypatch, tokens, dtype):
    impl = _impl()
    group = _Group()
    monkeypatch.setattr(impl, "_resolve_tp_group", lambda: group)
    partials = torch.ones(tokens, 7, dtype=dtype)
    actual = impl._reduce_tp_rows(partials)
    assert actual is partials
    assert len(group.calls) == 1
    assert group.calls[0][0].shape == (tokens, 7)
    assert group.calls[0][2] == dtype
    assert torch.equal(actual, torch.full_like(actual, 1.03125))


@pytest.mark.parametrize("requests", [1, 2, 8])
def test_no_group_returns_original_tensor(monkeypatch, requests):
    impl = _impl()
    monkeypatch.setattr(impl, "_resolve_tp_group", lambda: None)
    partials = torch.randn(requests, 7)
    assert impl._reduce_tp_rows(partials, num_requests=requests) is partials


@pytest.mark.parametrize(
    "requests,shape,dtype",
    [(0, (2, 7), torch.float32), (-1, (2, 7), torch.float32),
     (True, (2, 7), torch.float32), (2.0, (2, 7), torch.float32),
     (3, (2, 7), torch.float32), (2, (2, 1, 7), torch.float32),
     (2, (2, 0), torch.float32), (2, (2, 7), torch.bfloat16)],
)
def test_invalid_request_geometry_refuses_before_group_lookup(monkeypatch, requests, shape, dtype):
    impl = _impl()

    def forbidden():
        raise AssertionError("invalid inputs reached group lookup")

    monkeypatch.setattr(impl, "_resolve_tp_group", forbidden)
    with pytest.raises(ValueError):
        impl._reduce_tp_rows(torch.empty(shape, dtype=dtype), num_requests=requests)


def test_kda_finish_reduces_fp32_before_cast(monkeypatch):
    impl = _impl()
    group = _Group()
    monkeypatch.setattr(impl, "_resolve_tp_group", lambda: group)
    partials = torch.full((2, 7), 1.003, dtype=torch.float32)
    actual = impl.Glm5NextKDAAttention._finish_output(
        None, partials, torch.bfloat16, num_requests=2
    )
    assert group.calls[0][0].shape == (7, 2)
    assert group.calls[0][2] == torch.float32
    assert torch.equal(actual, (partials + 0.03125).bfloat16())


@pytest.mark.parametrize("family", ["dense", "moe"])
def test_ffn_reduces_interleaved_fp32_before_cast(monkeypatch, family):
    impl = _impl()
    group = _Group()
    monkeypatch.setattr(impl, "_resolve_tp_group", lambda: group)
    partials = torch.arange(14, dtype=torch.float32).reshape(2, 7) / 127

    class MLP:
        def __call__(self, *_args, **_kwargs):
            return partials.clone()

    monkeypatch.setattr(impl, "Glm5NextDenseMLP" if family == "dense" else "Glm5NextMoEBlock", MLP)
    model = SimpleNamespace(_rms_norm=lambda value, _gain: value, text_config=None)
    layer = SimpleNamespace(mlp=MLP(), post_attention_layernorm_weight=torch.ones(7))
    result = impl.Glm5NextModel._ffn_half(
        model, layer, torch.ones(2, 7, dtype=torch.bfloat16), quant_config=None,
        block_size=None, moe_group=None, tp_degree=64, expert_parallel_rank=0,
        num_requests=2,
    )
    assert len(group.calls) == 1
    assert group.calls[0][0].shape == (7, 2)
    assert group.calls[0][2] == torch.float32
    assert torch.equal(result, (partials + 0.03125).bfloat16())


@pytest.mark.parametrize("family", ["kda", "dsa"])
@pytest.mark.parametrize("count,is_prefill", [(1, False), (2, False), (8, False), (2, True)])
def test_model_threads_only_concurrent_decode_count_to_ffn(monkeypatch, family, count, is_prefill):
    impl = _impl()
    seen = []

    class Site:
        def forward(self, streams, sublayer, **_kwargs):
            return sublayer(streams.mean(1)).unsqueeze(1).expand_as(streams)

    monkeypatch.setattr(impl, "_mhc_ffn_site", lambda *_args: Site())

    def layer(value, **_kwargs):
        return value

    def ffn(_layer, hidden, **kwargs):
        seen.append(kwargs.get("num_requests"))
        return hidden

    tokens = max(count, 2)
    model = SimpleNamespace(layers=[layer], embed_tokens_weight=torch.ones(tokens, 7),
                            text_config=SimpleNamespace(hc_mult=4), norm_weight=torch.ones(7),
                            _ffn_half=ffn, _rms_norm=lambda hidden, _gain: hidden)
    carrier = {"conv_state" if family == "kda" else "pool_cache": tuple(range(count)),
               "is_prefill": is_prefill}
    impl.Glm5NextModel.forward(model, torch.arange(tokens), layer_carriers=[carrier], quant_config=None)
    assert seen == ([count] if count > 1 and not is_prefill else [None])
