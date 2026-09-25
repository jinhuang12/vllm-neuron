# SPDX-License-Identifier: Apache-2.0
"""Exact concurrent-decode checks against the unchanged one-request route.

Run with the repository's NKI CPU simulator environment. The kernels are real
simulator calls; only the TP coordinator is replaced to count and apply a sum.
Native graph capture still needs its own state-alias and numerical checks.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from test.vllm_neuron.model.glm5_next import test_kda_layer as layer_tests


class _SumGroup:
    def __init__(self):
        self.calls = []

    def all_reduce(self, value):
        self.calls.append((tuple(value.shape), value.dtype))
        # A non-identity sum makes omitting the collective observable. The
        # constant peer partial does not depend on the request's batch index.
        value.add_(0.0625)
        return value


@pytest.fixture(scope="module")
def attention():
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    config = Glm5NextTextConfig(hidden_size=256)
    heads = 2
    world_size = int(config.linear_attn_config["num_heads"]) // heads
    module = layer_tests._impl().Glm5NextKDAAttention(config, world_size)
    torch.manual_seed(20260921)
    weights = layer_tests._make_weights(
        config.hidden_size, heads, module.head_dim, module.short_conv_kernel_size
    )
    for name, value in weights.items():
        if name != "input_layernorm_weight":
            setattr(module, name, nn.Parameter(value, requires_grad=False))
    return module


def _banks(module, slots):
    generator = torch.Generator().manual_seed(1729 + slots)
    conv = torch.randn(
        slots, *module.kda_conv_state_shape, generator=generator
    ).mul_(0.1).to(module.kda_conv_state_dtype)
    recurrent = torch.randn(
        slots, *module.kda_recurrent_state_shape, generator=generator
    ).mul_(0.1).to(module.kda_recurrent_state_dtype)
    return conv, recurrent


@pytest.mark.parametrize(
    ("requests", "input_dtype"),
    [(2, torch.float32), (4, torch.bfloat16), (8, torch.float32)],
)
def test_batched_decode_matches_separate_requests_across_slot_reorders(
    attention, requests, input_dtype, monkeypatch
):
    """Mixed starts and masks keep each original bank slot exact over three steps."""
    group = _SumGroup()
    monkeypatch.setattr(layer_tests._impl(), "_resolve_tp_group", lambda: group)
    actual_conv, actual_recurrent = _banks(attention, requests + 2)
    expected_conv, expected_recurrent = actual_conv.clone(), actual_recurrent.clone()
    unused_conv = actual_conv[[0, requests + 1]].clone()
    unused_recurrent = actual_recurrent[[0, requests + 1]].clone()
    # Even slots continue. Odd slots open on non-zero bytes from a prior owner.
    positions = torch.tensor(
        [0 if slot % 2 else 7 + slot for slot in range(requests + 2)],
        dtype=torch.int32,
    )
    generator = torch.Generator().manual_seed(20260922 + requests)
    saw_masked_continuation = False
    saw_active_update = False

    for step in range(3):
        slots = list(range(1, requests + 1))
        slots = slots[step % requests :] + slots[: step % requests]
        if step % 2 == 0:
            slots.reverse()
        real = torch.tensor(
            [int((slot + step) % 3 != 1) for slot in slots], dtype=torch.int32
        ).reshape(requests, 1)
        mask = real.to(torch.float32).reshape(requests, 1, 1)
        starts = positions[slots].clone()
        hidden = torch.randn(
            requests, attention.q_proj_weight.shape[1], generator=generator
        ).to(input_dtype)
        before_conv = actual_conv.clone()
        before_recurrent = actual_recurrent.clone()

        group.calls.clear()
        actual = attention(
            hidden,
            conv_state=tuple(actual_conv[slot] for slot in slots),
            recurrent_state=tuple(actual_recurrent[slot] for slot in slots),
            is_prefill=False,
            start_position=starts,
            real_tokens=real,
            row_mask=mask,
        )
        assert group.calls == [((hidden.shape[1], requests), torch.float32)]
        assert actual.dtype == input_dtype

        group.calls.clear()
        expected = torch.cat(
            [
                attention(
                    hidden[index : index + 1],
                    conv_state=expected_conv[slot],
                    recurrent_state=expected_recurrent[slot],
                    is_prefill=False,
                    start_position=starts[index],
                    real_tokens=real[index],
                    row_mask=mask[index],
                )
                for index, slot in enumerate(slots)
            ],
            dim=0,
        )
        assert group.calls == [((1, hidden.shape[1]), torch.float32)] * requests
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_conv, expected_conv, rtol=0, atol=0)
        torch.testing.assert_close(actual_recurrent, expected_recurrent, rtol=0, atol=0)
        assert torch.equal(actual_conv[[0, requests + 1]], unused_conv)
        assert torch.equal(actual_recurrent[[0, requests + 1]], unused_recurrent)

        for index, slot in enumerate(slots):
            if int(real[index]) == 0 and int(starts[index]) > 0:
                saw_masked_continuation = True
                assert torch.equal(actual_conv[slot], before_conv[slot])
                assert torch.equal(actual_recurrent[slot], before_recurrent[slot])
            elif int(real[index]) == 1:
                saw_active_update = True
                assert not torch.equal(actual_conv[slot], before_conv[slot])
                assert not torch.equal(actual_recurrent[slot], before_recurrent[slot])
            positions[slot] += real[index, 0]

    assert saw_masked_continuation
    assert saw_active_update


def test_batched_decode_keeps_the_optional_row_operands(attention, monkeypatch):
    """Omitting both operands still means one real token for each request."""
    monkeypatch.setattr(layer_tests._impl(), "_resolve_tp_group", lambda: None)
    conv, recurrent = _banks(attention, 2)
    reference_conv, reference_recurrent = conv.clone(), recurrent.clone()
    hidden = torch.randn(2, attention.q_proj_weight.shape[1])
    starts = torch.tensor([0, 9], dtype=torch.int32)
    actual = attention(
        hidden,
        conv_state=tuple(conv.unbind()),
        recurrent_state=tuple(recurrent.unbind()),
        is_prefill=False,
        start_position=starts,
    )
    expected = torch.cat(
        [
            attention(
                hidden[index : index + 1],
                conv_state=reference_conv[index],
                recurrent_state=reference_recurrent[index],
                is_prefill=False,
                start_position=starts[index],
            )
            for index in range(2)
        ]
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(conv, reference_conv)
    assert torch.equal(recurrent, reference_recurrent)
