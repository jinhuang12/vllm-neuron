# SPDX-License-Identifier: Apache-2.0
"""KDA channel sharding preserves the convolution and its carried state."""

import os
from unittest.mock import patch

import pytest
import torch

from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from vllm_neuron.functional.kda import depthwise_conv1d as conv
from test.vllm_neuron.model.glm5_next import test_kda_padded_prefill as state_fixture

pytestmark = [pytest.mark.fast, pytest.mark.forked]


@pytest.fixture(scope="module", autouse=True)
def require_simulator():
    assert os.environ.get("VLLM_NEURON_CPU_MODE") == "1"
    assert os.environ.get("NKI_SIMULATOR") == "1"


def _assert_bits(got, expected):
    assert got.dtype == expected.dtype
    assert got.shape == expected.shape
    assert torch.equal(
        got.contiguous().view(torch.uint8),
        expected.contiguous().view(torch.uint8),
    )


def _operands(tokens, channels=384, taps=4, dtype=torch.float32, batches=1):
    generator = torch.Generator().manual_seed(6103 + tokens + channels + taps)
    img = torch.randn(
        batches, channels, 1, tokens + taps - 1, generator=generator
    ).to(dtype)
    filt = torch.randn(channels, 1, 1, taps, generator=generator).to(dtype)
    return img, filt


def _grid_one(img, filt, stride=conv.UNIT_STRIDE):
    """The unchanged substrate launch, independent of the wrapper's new gate."""
    return wrap_nki(conv.depthwise_conv1d_implicit_gemm)(
        img_ref=img, filter_ref=filt, feature_group_count=int(img.shape[1]),
        padding=conv.NO_PADDING, stride=stride, rhs_dilation=conv.UNIT_DILATION,
        lhs_dilation=conv.UNIT_DILATION, batch_group_count=1,
    )


@pytest.mark.parametrize("tokens", [1, 2, 3, 7, 8, 9, 127, 128, 129, 511, 512, 513, 1024])
def test_channel_and_sequence_tile_boundaries(tokens, monkeypatch):
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    img, filt = _operands(tokens)
    old = _grid_one(img, filt)
    new = conv.depthwise_conv1d(img, filt)
    _assert_bits(new, old)
    reference = conv.depthwise_conv1d_torch_reference(img, filt)
    torch.testing.assert_close(new, reference, rtol=1e-2, atol=1e-5)
    # Distinct channel data makes shifted, omitted, or duplicated shards visible.
    for channel in (0, 127, 128, 191, 192, 255, 256, 383):
        _assert_bits(new[:, channel], old[:, channel])


@pytest.mark.parametrize("lnc", [None, "1", "2", "invalid"])
@pytest.mark.parametrize(
    "channels,taps,dtype,batches,stride",
    [
        (384, 4, torch.float32, 1, (1, 1)),
        (256, 4, torch.float32, 1, (1, 1)),
        (384, 5, torch.float32, 1, (1, 1)),
        (384, 4, torch.bfloat16, 1, (1, 1)),
        (384, 4, torch.float32, 2, (1, 1)),
        (384, 4, torch.float32, 1, (1, 2)),
    ],
)
def test_other_runtime_and_public_geometries_retain_grid_one(
    lnc, channels, taps, dtype, batches, stride, monkeypatch,
):
    if lnc is None:
        monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    else:
        monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", lnc)
    selected = []
    real_wrap = conv.wrap_nki

    class RecordCall:
        def __init__(self, call):
            self.call = call

        def __getitem__(self, grid):
            return RecordCall(self.call[grid])

        def __call__(self, **kwargs):
            selected.append(tuple(self.call.grid))
            return self.call(**kwargs)

    monkeypatch.setattr(conv, "wrap_nki", lambda kernel: RecordCall(real_wrap(kernel)))
    img, filt = _operands(1, channels, taps, dtype, batches)
    expected = _grid_one(img, filt, stride=stride)
    actual = conv.depthwise_conv1d(img, filt, stride=stride)
    _assert_bits(actual, expected)
    eligible = (
        lnc == "2" and channels == 384 and taps == 4
        and dtype == torch.float32 and batches == 1 and stride == (1, 1)
    )
    assert selected == [(2,) if eligible else ()]


def test_padded_prefill_continuation_decode_and_state_aliases(monkeypatch):
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    stack = state_fixture._stack()
    new_call = conv.depthwise_conv1d
    old_bank = state_fixture._fresh_bank(stack.layers)
    new_bank = state_fixture._fresh_bank(stack.layers)
    for old, new in zip(old_bank, new_bank, strict=True):
        # Opening ignores a prior owner's state; continuations retain ours.
        old[0].fill_(0.5)
        old[1].fill_(0.125)
        new[0].copy_(old[0])
        new[1].copy_(old[1])
    old_ptrs = [tuple(part.data_ptr() for part in pair) for pair in old_bank]
    new_ptrs = [tuple(part.data_ptr() for part in pair) for pair in new_bank]
    real = 19  # ragged within chunk 8 and below the 32-row padded bucket
    generator = torch.Generator().manual_seed(6313)
    steps = [
        (state_fixture._padded_rows(stack, real), real, True, 0),
        (torch.randn(8, stack.hidden, generator=generator), 8, True, real),
        (torch.randn(1, stack.hidden, generator=generator), 1, False, real + 8),
        (torch.randn(1, stack.hidden, generator=generator), 1, False, real + 9),
    ]
    for rows, count, prefill, position in steps:
        with patch.object(conv, "depthwise_conv1d", _grid_one):
            old = state_fixture._drive(
                stack.layers, old_bank, rows, real=count,
                is_prefill=prefill, start_position=position,
            )
        with patch.object(conv, "depthwise_conv1d", new_call):
            new = state_fixture._drive(
                stack.layers, new_bank, rows, real=count,
                is_prefill=prefill, start_position=position,
            )
        _assert_bits(new, old)
        for old_pair, new_pair in zip(old_bank, new_bank, strict=True):
            for old_state, new_state in zip(old_pair, new_pair, strict=True):
                _assert_bits(new_state, old_state)
        assert [tuple(part.data_ptr() for part in pair) for pair in old_bank] == old_ptrs
        assert [tuple(part.data_ptr() for part in pair) for pair in new_bank] == new_ptrs
