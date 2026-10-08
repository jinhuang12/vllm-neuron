# SPDX-License-Identifier: Apache-2.0
"""The model's KDA layer serves a speculative verify step on checkpoint carriers.

A verify step hands ``Glm5NextKDAAttention.forward`` ``T = 1 + k`` tokens per
request (request-major rows) on carriers that hold ``T`` state rows per slot
(``state_checkpoints=T``): a view carrier is ``[T, ...]``, a bank
``[slots, T, ...]``. The step starts each request from the row of its
``checkpoint_rows`` and writes every row of its slot as that token's checkpoint,
in one fused launch (:func:`kda_fused_decode_tstep`).

The reference is the model itself, driven as it ran a multi-token decode before
this step existed: ``T`` one-token forwards on one-row carriers, each fed the
state the previous one wrote. Both forms project the same tokens through the
same weights; the stacked form projects ``[B*T, hidden]`` rows where the chain
projects ``[B, hidden]``, and torch's CPU GEMM rounds the two apart in the last
bit, so the comparison is to one rounding (the tolerances
``test_fused_decode_call_site.py`` uses), while the view form and the bank form
of the step -- the same operands through the same kernel -- must agree bit for
bit.
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
#: ``T = 1 + k`` at the gate's ``k = 3``.
TOKENS = 4

OUT_RTOL = 1e-4
OUT_ATOL = 1e-5
STATE_RTOL = 1e-4
STATE_ATOL = 1e-5
#: bf16 history rows agree to one rounding of the fp32 projection they were cast from.
CONV_TOL = 2.0**-7


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


def _case(module, hidden, batch, tokens=TOKENS):
    """The entering states (one row per slot), the step's tokens and its padding.

    Request 0 opens at position 0 on a slot whose previous owner diverged;
    request 1 carries two real rows of four, request 2 is a padding request of
    the bucket, request 3 carries one. The accepted counts name the checkpoint
    row each request starts from.
    """
    gen = torch.Generator().manual_seed(8300 + 10 * tokens + batch)
    conv_bank = torch.randn(
        (BANK_SLOTS, *module.kda_conv_state_shape), generator=gen
    ).to(module.kda_conv_state_dtype)
    rec_bank = torch.randn(
        (BANK_SLOTS, *module.kda_recurrent_state_shape), generator=gen
    ).to(module.kda_recurrent_state_dtype) * 0.1
    hidden_states = torch.randn(batch * tokens, hidden, generator=gen)
    accepted = torch.tensor([0, tokens - 1, 1, 2][:batch], dtype=torch.int32)
    if batch == 1:
        return dict(
            hidden_states=hidden_states, conv_bank=conv_bank, rec_bank=rec_bank,
            start_position=torch.tensor([11], dtype=torch.int32),
            real=None, accepted=accepted,
        )
    conv_bank[SLOTS[0]] = float("nan")
    rec_bank[SLOTS[0]] = float("nan")
    return dict(
        hidden_states=hidden_states, conv_bank=conv_bank, rec_bank=rec_bank,
        start_position=torch.tensor([0, 9, 3, 12], dtype=torch.int32),
        real=torch.tensor([tokens, 2, 0, 1], dtype=torch.int32), accepted=accepted,
    )


def _chained_reference(module, case, batch, tokens=TOKENS):
    """``T`` one-token forwards on one-row carriers; the checkpoints they leave."""
    conv_bank = case["conv_bank"].clone()
    rec_bank = case["rec_bank"].clone()
    slots = list(SLOTS[:batch])
    real = case["real"]
    outputs = torch.empty(batch * tokens, case["hidden_states"].shape[1])
    conv_ckpts = torch.empty((batch, tokens, *conv_bank.shape[1:]), dtype=conv_bank.dtype)
    rec_ckpts = torch.empty((batch, tokens, *rec_bank.shape[1:]), dtype=rec_bank.dtype)
    for t in range(tokens):
        rows = torch.arange(batch) * tokens + t
        kwargs = {}
        position = case["start_position"].clone()
        if real is not None:
            position = position + torch.minimum(torch.full_like(real, t), real)
            kwargs["real_tokens"] = (real > t).to(torch.int32).reshape(batch, 1)
            kwargs["row_mask"] = (real > t).to(torch.float32).reshape(batch, 1, 1)
        else:
            position = position + t
        outputs[rows] = module(
            case["hidden_states"][rows],
            conv_state=tuple(conv_bank[s] for s in slots),
            recurrent_state=tuple(rec_bank[s] for s in slots),
            is_prefill=False,
            start_position=position,
            **kwargs,
        )
        conv_ckpts[:, t] = conv_bank[slots]
        rec_ckpts[:, t] = rec_bank[slots]
    return outputs, conv_ckpts, rec_ckpts


def _checkpoint_banks(case, batch, tokens=TOKENS):
    """``[slots, T, ...]`` banks: the entering state at row ``accepted``, NaN elsewhere.

    The step reads only the accepted row and overwrites every row of the slot, so
    a NaN left anywhere in a request's slot after the step is a row it did not
    write; the slots no request holds keep their bytes.
    """
    gen = torch.Generator().manual_seed(8400 + batch)
    conv = torch.randn((BANK_SLOTS, tokens, *case["conv_bank"].shape[1:]), generator=gen)
    rec = torch.randn((BANK_SLOTS, tokens, *case["rec_bank"].shape[1:]), generator=gen)
    conv = conv.to(case["conv_bank"].dtype)
    rec = rec.to(case["rec_bank"].dtype)
    for b, slot in enumerate(SLOTS[:batch]):
        conv[slot] = float("nan")
        rec[slot] = float("nan")
        a = int(case["accepted"][b])
        conv[slot, a] = case["conv_bank"][slot]
        rec[slot, a] = case["rec_bank"][slot]
    return conv, rec


def _padding(case, batch, tokens=TOKENS):
    if case["real"] is None:
        return {}
    real = case["real"]
    mask = (torch.arange(tokens).reshape(1, tokens) < real.reshape(batch, 1)).to(torch.float32)
    return {"real_tokens": real.reshape(batch, 1), "row_mask": mask.reshape(batch, tokens, 1)}


def _step(module, case, batch, *, bank_form, accepted, tokens=TOKENS):
    conv, rec = _checkpoint_banks(case, batch, tokens)
    slots = torch.tensor(SLOTS[:batch], dtype=torch.int64)
    if bank_form:
        carriers = dict(conv_state=conv, recurrent_state=rec, state_slots=slots)
    else:
        carriers = dict(
            conv_state=tuple(conv[s] for s in SLOTS[:batch]),
            recurrent_state=tuple(rec[s] for s in SLOTS[:batch]),
        )
    _reset()
    out = module(
        case["hidden_states"],
        is_prefill=False,
        start_position=case["start_position"],
        checkpoint_rows=accepted,
        state_checkpoints=tokens,
        **carriers,
        **_padding(case, batch, tokens),
    )
    return out, conv, rec, _read()


def _assert_matches_chain(out, conv, rec, reference, case, batch, label):
    ref_out, conv_ckpts, rec_ckpts = reference
    slots = list(SLOTS[:batch])
    assert torch.isfinite(out).all(), label
    torch.testing.assert_close(out, ref_out, rtol=OUT_RTOL, atol=OUT_ATOL, msg=label)
    assert torch.isfinite(rec[slots]).all(), f"{label}: a checkpoint row was not written"
    assert torch.isfinite(conv[slots].float()).all(), f"{label}: a conv row was not written"
    torch.testing.assert_close(
        conv[slots].float(), conv_ckpts.float(), rtol=CONV_TOL, atol=CONV_TOL, msg=label
    )
    torch.testing.assert_close(
        rec[slots], rec_ckpts, rtol=STATE_RTOL, atol=STATE_ATOL, msg=label
    )
    untouched_conv, untouched_rec = _checkpoint_banks(case, batch)
    for slot in range(BANK_SLOTS):
        if slot in slots:
            continue
        assert torch.equal(conv[slot], untouched_conv[slot]), (slot, label)
        assert torch.equal(rec[slot], untouched_rec[slot]), (slot, label)


@pytest.mark.parametrize("batch", [1, 4])
def test_verify_step_on_checkpoint_views_matches_chained_one_token_steps(batch, monkeypatch):
    monkeypatch.setenv(fused.FUSED_DECODE_ENV, "1")
    module, hidden = _attention()
    case = _case(module, hidden, batch)
    reference = _chained_reference(module, case, batch)
    out, conv, rec, counts = _step(module, case, batch, bank_form=False,
                                   accepted=case["accepted"])
    assert counts == {"fused": (1, 0), "conv": (0, 0), "gate": (0, 0), "decode": (0, 0)}, (
        f"the verify step is one fused T-step launch: {counts}")
    _assert_matches_chain(out, conv, rec, reference, case, batch, f"views B={batch}")


@pytest.mark.parametrize("row", range(TOKENS))
def test_verify_step_reads_the_live_row_before_overwriting_it(row, monkeypatch):
    """The alias case of the pointer commit, bs=1: the slot's live row ``a`` is
    read, then every row ``0 .. T-1`` -- row ``a`` included -- is overwritten by
    the new checkpoints; the result must be the chain from the old row ``a``."""
    monkeypatch.setenv(fused.FUSED_DECODE_ENV, "1")
    module, hidden = _attention()
    case = _case(module, hidden, 1)
    case["accepted"] = torch.tensor([row], dtype=torch.int32)
    reference = _chained_reference(module, case, 1)
    out, conv, rec, counts = _step(module, case, 1, bank_form=False, accepted=case["accepted"])
    assert counts["fused"] == (1, 0)
    _assert_matches_chain(out, conv, rec, reference, case, 1, f"alias row {row}")


def test_verify_step_takes_host_checkpoint_rows_on_one_request(monkeypatch):
    """bs=1 with the row as a number: the row is a slice, not a gather."""
    monkeypatch.setenv(fused.FUSED_DECODE_ENV, "1")
    module, hidden = _attention()
    case = _case(module, hidden, 1)
    case["accepted"] = torch.tensor([2], dtype=torch.int32)
    reference = _chained_reference(module, case, 1)
    out, conv, rec, counts = _step(module, case, 1, bank_form=False, accepted=2)
    assert counts["fused"] == (1, 0)
    _assert_matches_chain(out, conv, rec, reference, case, 1, "host int")
    out_list, conv_list, rec_list, _ = _step(module, case, 1, bank_form=False, accepted=[2])
    assert torch.equal(out_list, out) and torch.equal(rec_list, rec)
    assert torch.equal(conv_list, conv)


def test_verify_step_on_whole_banks_is_bit_equal_to_the_view_form(monkeypatch):
    monkeypatch.setenv(fused.FUSED_DECODE_ENV, "1")
    batch = 4
    module, hidden = _attention()
    case = _case(module, hidden, batch)
    reference = _chained_reference(module, case, batch)
    view_out, view_conv, view_rec, _ = _step(module, case, batch, bank_form=False,
                                             accepted=case["accepted"])
    bank_out, bank_conv, bank_rec, counts = _step(module, case, batch, bank_form=True,
                                                  accepted=case["accepted"])
    assert counts["fused"] == (1, 0), counts
    _assert_matches_chain(bank_out, bank_conv, bank_rec, reference, case, batch, "bank")
    assert torch.equal(bank_out, view_out)
    assert torch.equal(bank_conv.view(torch.uint8), view_conv.view(torch.uint8))
    assert torch.equal(
        bank_rec.nan_to_num(7.0).view(torch.uint8), view_rec.nan_to_num(7.0).view(torch.uint8)
    )


def test_verify_step_refusals(monkeypatch):
    monkeypatch.setenv(fused.FUSED_DECODE_ENV, "1")
    module, hidden = _attention()
    batch = 4
    case = _case(module, hidden, batch)
    conv, rec = _checkpoint_banks(case, batch)
    views = dict(
        conv_state=tuple(conv[s] for s in SLOTS[:batch]),
        recurrent_state=tuple(rec[s] for s in SLOTS[:batch]),
    )
    common = dict(is_prefill=False, start_position=case["start_position"],
                  **_padding(case, batch))
    one_row = dict(
        conv_state=tuple(case["conv_bank"][s] for s in SLOTS[:batch]),
        recurrent_state=tuple(case["rec_bank"][s] for s in SLOTS[:batch]),
    )
    # T rows per request without checkpoint carriers.
    with pytest.raises(ValueError, match="state_checkpoints"):
        module(case["hidden_states"], **one_row, **common)
    # Fewer tokens than checkpoint rows.
    short = case["hidden_states"].reshape(batch, TOKENS, hidden)[:, : TOKENS - 1].reshape(-1, hidden)
    with pytest.raises(ValueError, match="exactly"):
        module(short, **views, is_prefill=False, start_position=case["start_position"],
               checkpoint_rows=case["accepted"], state_checkpoints=TOKENS)
    # One row per request.
    with pytest.raises(ValueError, match="checkpoint_rows"):
        module(case["hidden_states"], **views, **common,
               checkpoint_rows=torch.zeros(batch + 1, dtype=torch.int32),
               state_checkpoints=TOKENS)
    # A prefill never takes a checkpoint carrier.
    with pytest.raises(ValueError, match="prefill"):
        module(case["hidden_states"][:TOKENS], conv_state=(conv[SLOTS[0]],),
               recurrent_state=(rec[SLOTS[0]],), is_prefill=True,
               start_position=torch.tensor([0], dtype=torch.int32),
               checkpoint_rows=0, state_checkpoints=TOKENS)
    # The switched-off stage path serves one token per request.
    monkeypatch.setenv(fused.FUSED_DECODE_ENV, "0")
    with pytest.raises(ValueError, match="switched off"):
        module(case["hidden_states"][:TOKENS], conv_state=(conv[SLOTS[0]],),
               recurrent_state=(rec[SLOTS[0]],), is_prefill=False,
               start_position=torch.tensor([11], dtype=torch.int32),
               checkpoint_rows=0, state_checkpoints=TOKENS)
    with pytest.raises(ValueError, match="switched-off"):
        module(case["hidden_states"], **views, **common,
               checkpoint_rows=case["accepted"], state_checkpoints=TOKENS)
