# SPDX-License-Identifier: Apache-2.0
"""The model's KDA decode step takes the fused kernel, and agrees with the stages.

``Glm5NextKDAAttention.forward`` is driven as the runner drives a decode step:
one token per request, the carriers a tuple of views into one bank at the
requests' own slots, the position one int32 row per request. It is run twice on
copies of the same banks, once with the fused kernel (the default) and once with
``VLLM_NEURON_KDA_FUSED_DECODE=0``, which runs the conv, gate-clamp and
decode-state kernels this change replaced. Both runs must write the same bank
rows and return the same output, and each must have dispatched only its own
kernels.

Wave 1 keeps the per-request loop in the model, so ``B`` requests are ``B``
fused dispatches per layer here, not one.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from vllm_neuron.functional.kda import decode_state, depthwise_conv1d, gate_clamp
from vllm_neuron.functional.kda import fused_decode as fused

TP_WORLD_SIZE = 64
SLOTS = (5, 2, 7, 0)
BANK_SLOTS = 8

#: ``core`` passes through the gated RMSNorm and the output projection, which
#: neither amplifies nor hides the kernel's float32 reassociation error.
OUT_RTOL = 1e-4
OUT_ATOL = 1e-5
STATE_RTOL = 1e-4
STATE_ATOL = 1e-5


def _attention():
    from vllm_neuron.model.glm5_next import model_fp8
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    config = Glm5NextTextConfig()
    module = model_fp8.Glm5NextKDAAttention(config, TP_WORLD_SIZE)
    hidden = int(config.hidden_size)
    heads, kdim = int(module.num_kv_heads_per_rank), int(module.head_dim)
    width, taps = heads * kdim, int(module.short_conv_kernel_size)
    gen = torch.Generator().manual_seed(8100)

    def rnd(*shape, scale):
        return torch.randn(*shape, generator=gen) * scale

    weights = {
        "q_proj_weight": rnd(width, hidden, scale=hidden**-0.5),
        "k_proj_weight": rnd(width, hidden, scale=hidden**-0.5),
        "v_proj_weight": rnd(width, hidden, scale=hidden**-0.5),
        "b_proj_weight": rnd(heads, hidden, scale=hidden**-0.5),
        "f_a_proj_weight": rnd(kdim, hidden, scale=hidden**-0.5),
        "f_b_proj_weight": rnd(width, kdim, scale=kdim**-0.5),
        "g_a_proj_weight": rnd(kdim, hidden, scale=hidden**-0.5),
        "g_b_proj_weight": rnd(width, kdim, scale=kdim**-0.5),
        "q_conv1d_weight": rnd(width, 1, taps, scale=0.5).to(torch.bfloat16),
        "k_conv1d_weight": rnd(width, 1, taps, scale=0.5).to(torch.bfloat16),
        "v_conv1d_weight": rnd(width, 1, taps, scale=0.5).to(torch.bfloat16),
        "o_norm_weight": 1.0 + rnd(kdim, scale=0.05),
        "o_proj_weight": rnd(hidden, width, scale=width**-0.5),
        "A_log": rnd(heads, scale=0.3),
        "dt_bias": rnd(width, scale=0.3),
    }
    for name, tensor in weights.items():
        setattr(module, name, nn.Parameter(tensor, requires_grad=False))
    return module, hidden


def _reset():
    fused.reset_fused_decode_dispatch_counters()
    depthwise_conv1d.reset_dispatch_counters()
    gate_clamp.reset_gate_clamp_dispatch_counters()
    decode_state.reset_decode_dispatch_counters()


def _read():
    return {
        "fused": fused.fused_decode_dispatch_counters(),
        "conv": depthwise_conv1d.dispatch_counters(),
        "gate": gate_clamp.gate_clamp_dispatch_counters(),
        "decode": decode_state.decode_dispatch_counters(),
    }


def _case(module, hidden, batch):
    gen = torch.Generator().manual_seed(8200 + batch)
    conv_bank = torch.randn(
        (BANK_SLOTS, *module.kda_conv_state_shape), generator=gen
    ).to(module.kda_conv_state_dtype)
    rec_bank = torch.randn(
        (BANK_SLOTS, *module.kda_recurrent_state_shape), generator=gen
    ).to(module.kda_recurrent_state_dtype) * 0.1
    tokens = torch.randn(batch, hidden, generator=gen)
    if batch == 1:
        return dict(
            tokens=tokens, conv_bank=conv_bank, rec_bank=rec_bank,
            start_position=torch.tensor([11], dtype=torch.int32),
            real_tokens=None, row_mask=None,
        )
    # Request 0 opens on a slot whose previous owner diverged; request 2 is a
    # padding row of the bucket.
    conv_bank[SLOTS[0]] = float("nan")
    rec_bank[SLOTS[0]] = float("nan")
    return dict(
        tokens=tokens, conv_bank=conv_bank, rec_bank=rec_bank,
        start_position=torch.tensor([0, 9, 3, 12], dtype=torch.int32),
        real_tokens=torch.tensor([[1], [1], [0], [1]], dtype=torch.int32),
        row_mask=torch.tensor([[[1.0]], [[1.0]], [[0.0]], [[1.0]]]),
    )


def _decode(module, case, batch, monkeypatch, enabled):
    monkeypatch.setenv(fused.FUSED_DECODE_ENV, "1" if enabled else "0")
    conv_bank = case["conv_bank"].clone()
    rec_bank = case["rec_bank"].clone()
    slots = SLOTS[:batch]
    _reset()
    out = module(
        case["tokens"],
        conv_state=tuple(conv_bank[s] for s in slots),
        recurrent_state=tuple(rec_bank[s] for s in slots),
        is_prefill=False,
        start_position=case["start_position"],
        real_tokens=case["real_tokens"],
        row_mask=case["row_mask"],
    )
    return out, conv_bank, rec_bank, _read()


@pytest.mark.parametrize("batch", [1, 4])
def test_model_decode_takes_the_fused_kernel_and_matches_the_stages(
    batch, monkeypatch
):
    module, hidden = _attention()
    case = _case(module, hidden, batch)
    new_out, new_conv, new_rec, new_counts = _decode(
        module, case, batch, monkeypatch, enabled=True
    )
    old_out, old_conv, old_rec, old_counts = _decode(
        module, case, batch, monkeypatch, enabled=False
    )

    assert new_counts == {
        "fused": (batch, 0), "conv": (0, 0), "gate": (0, 0), "decode": (0, 0),
    }, f"the fused decode must serve every request on the NKI route: {new_counts}"
    assert old_counts == {
        "fused": (0, 0), "conv": (batch, 0), "gate": (batch, 0),
        "decode": (batch, 0),
    }, f"the switched-off control must run the old stages: {old_counts}"

    assert torch.isfinite(new_out).all()
    torch.testing.assert_close(new_out, old_out, rtol=OUT_RTOL, atol=OUT_ATOL)
    assert torch.equal(
        new_conv.contiguous().view(torch.uint8), old_conv.contiguous().view(torch.uint8)
    ), "the conv bank must agree bit for bit"
    torch.testing.assert_close(
        new_rec, old_rec, rtol=STATE_RTOL, atol=STATE_ATOL, equal_nan=True
    )
    untouched = [s for s in range(BANK_SLOTS) if s not in SLOTS[:batch]]
    for slot in untouched:
        assert torch.equal(
            new_rec[slot].nan_to_num(7.0), case["rec_bank"][slot].nan_to_num(7.0)
        ), f"slot {slot} belongs to no request in this step and was written"
    assert torch.isfinite(new_rec[list(SLOTS[:batch])]).all()
