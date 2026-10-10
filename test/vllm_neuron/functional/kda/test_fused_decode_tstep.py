# SPDX-License-Identifier: Apache-2.0
"""The T-token fused KDA decode step against T chained single-token fused steps.

A speculative verify step hands each request ``T = 1 + k`` tokens. The T-step
kernel steps them in one launch and writes one checkpoint of both carriers per
token; the next step starts from the checkpoint of the accepted count. The
reference here is the single-token kernel :func:`kda_fused_decode` chained
``T`` times on the host, each step fed the previous step's carriers, the way
the model ran a multi-token decode before this kernel existed. Both run the
same instructions on the same operands, so the comparison is bit equality:
``core`` rows, every conv checkpoint and every recurrent checkpoint.

Shapes are the TP=64 shapes: one head per rank, ``K = V = 128``, a 4-tap conv
over ``3 * 128 = 384`` channels, a bfloat16 conv carrier in the default ``SD``
layout and a float32 recurrent carrier.
"""

from __future__ import annotations

import pytest
import torch

from vllm_neuron.functional.kda import fused_decode as fused

HEADS = 1
KDIM = 128
TAPS = 4
LOWER = -5.0

#: The served speculative widths: ``T = 1 + k`` for the served k (3) and
#: its neighbours, plus the one-token form the non-speculative path uses.
TOKEN_COUNTS = (1, 2, 4, 6)
BATCHES = (1, 4)


def _weights(gen, heads=HEADS, kdim=KDIM, taps=TAPS):
    width = heads * kdim
    return {
        "q_conv1d_weight": (torch.randn(width, 1, taps, generator=gen) * 0.5).to(
            torch.bfloat16
        ),
        "k_conv1d_weight": (torch.randn(width, 1, taps, generator=gen) * 0.5).to(
            torch.bfloat16
        ),
        "v_conv1d_weight": (torch.randn(width, 1, taps, generator=gen) * 0.5).to(
            torch.bfloat16
        ),
        "A_log": torch.log(torch.rand(heads, generator=gen) * 15 + 1),
        "dt_bias": torch.randn(width, generator=gen) * 0.5,
    }


def _tokens(gen, batch, tokens, heads=HEADS, kdim=KDIM):
    """``T`` tokens per request, request-major rows (row ``b * T + t``)."""
    width = heads * kdim
    rows = batch * tokens
    return {
        "q_in": torch.randn(rows, width, generator=gen),
        "k_in": torch.randn(rows, width, generator=gen),
        "v_in": torch.randn(rows, width, generator=gen),
        "raw_gate": torch.randn(rows, width, generator=gen) * 2,
        "raw_beta": torch.randn(rows, heads, generator=gen),
    }


def _carriers(gen, batch, heads=HEADS, kdim=KDIM, taps=TAPS, dim_first=False):
    channels = 3 * heads * kdim
    rows = taps - 1
    shape = (batch, channels, rows) if dim_first else (batch, rows, channels)
    conv = torch.randn(shape, generator=gen).to(torch.bfloat16)
    rec = torch.randn(batch, heads, kdim, kdim, generator=gen) * 0.1
    return conv, rec


def _chain(step, weights, conv, rec, *, tokens, start_position, real_tokens=None,
           row_mask=None, dim_first=False):
    """``T`` single-token fused steps per request, carrying the carriers.

    Returns ``core`` ``[B*T, W]`` and the two checkpoint stacks ``[B, T, ...]``.
    Request ``b``'s step ``t`` stands at position ``start + min(t, real)``: a
    real token advances the position, a padding row does not.
    """
    batch = int(conv.shape[0])
    width = int(step["q_in"].shape[1])
    real = (torch.full((batch,), tokens, dtype=torch.int32)
            if real_tokens is None else real_tokens.reshape(batch).to(torch.int32))
    cores = torch.empty(batch * tokens, width)
    conv_ckpts = torch.empty((batch, tokens, *conv.shape[1:]), dtype=conv.dtype)
    rec_ckpts = torch.empty((batch, tokens, *rec.shape[1:]), dtype=rec.dtype)
    for t in range(tokens):
        rows = torch.arange(batch) * tokens + t
        position = start_position.to(torch.int32) + torch.minimum(
            torch.full_like(real, t), real
        )
        kwargs = {}
        if real_tokens is not None:
            kwargs["real_tokens"] = (real > t).to(torch.int32)
            kwargs["row_mask"] = row_mask.reshape(batch, tokens)[:, t].reshape(batch, 1)
        out = fused.kda_fused_decode(
            step["q_in"][rows], step["k_in"][rows], step["v_in"][rows],
            step["raw_gate"][rows], step["raw_beta"][rows],
            conv_state=conv, recurrent_state=rec,
            gate_lower_bound=LOWER, conv_state_dim_first=dim_first,
            start_position=position, **weights, **kwargs,
        )
        cores[rows] = out.core
        conv, rec = out.conv_state, out.recurrent_state
        conv_ckpts[:, t] = conv
        rec_ckpts[:, t] = rec
    return cores, conv_ckpts, rec_ckpts


def _tstep(step, weights, conv, rec, *, start_position, real_tokens=None,
           row_mask=None, dim_first=False):
    return fused.kda_fused_decode_tstep(
        **step, conv_state=conv, recurrent_state=rec,
        gate_lower_bound=LOWER, conv_state_dim_first=dim_first,
        start_position=start_position, real_tokens=real_tokens, row_mask=row_mask,
        **weights,
    )


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


def _assert_bit_equal(out, core, conv_ckpts, rec_ckpts, label):
    assert out.core.shape == core.shape, label
    assert out.conv_checkpoints.shape == conv_ckpts.shape, label
    assert out.recurrent_checkpoints.shape == rec_ckpts.shape, label
    assert out.conv_checkpoints.dtype == conv_ckpts.dtype
    assert out.recurrent_checkpoints.dtype == torch.float32
    tokens = int(conv_ckpts.shape[1])
    for t in range(tokens):
        assert torch.equal(
            _bytes(out.conv_checkpoints[:, t]), _bytes(conv_ckpts[:, t])
        ), f"{label}: conv checkpoint {t} is not bit-equal to the chained step"
        assert torch.equal(
            _bytes(out.recurrent_checkpoints[:, t]), _bytes(rec_ckpts[:, t])
        ), f"{label}: recurrent checkpoint {t} is not bit-equal to the chained step"
    assert torch.equal(_bytes(out.core), _bytes(core)), (
        f"{label}: core is not bit-equal to the chained steps"
    )


@pytest.mark.parametrize("batch", BATCHES)
@pytest.mark.parametrize("tokens", TOKEN_COUNTS)
def test_tstep_is_bit_equal_to_chained_single_token_steps(tokens, batch, monkeypatch):
    """Every request real on every row; request 0 opens on a NaN carrier."""
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    gen = torch.Generator().manual_seed(9000 + 10 * tokens + batch)
    weights = _weights(gen)
    step = _tokens(gen, batch, tokens)
    conv, rec = _carriers(gen, batch)
    conv[0] = float("nan")
    rec[0] = float("nan")
    start = torch.randint(1, 50, (batch,), generator=gen, dtype=torch.int32)
    start[0] = 0
    core, conv_ckpts, rec_ckpts = _chain(
        step, weights, conv, rec, tokens=tokens, start_position=start
    )
    fused.reset_fused_decode_dispatch_counters()
    out = _tstep(step, weights, conv, rec, start_position=start)
    assert fused.fused_decode_dispatch_counters() == (1, 0), (
        "the T-step must be one NKI dispatch and the torch fallback must not run"
    )
    _assert_bit_equal(out, core, conv_ckpts, rec_ckpts, f"T={tokens} B={batch}")
    assert torch.isfinite(out.recurrent_checkpoints).all()


@pytest.mark.parametrize("lnc", [None, "2"])
def test_tstep_partial_and_padding_requests(lnc, monkeypatch):
    """Rows past a request's real count leave its carriers where they stand.

    Request 1 carries two of four rows, request 2 none (a padding request of
    the bucket, position 1), request 3 opens at position 0 with one row.
    """
    if lnc is None:
        monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    else:
        monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", lnc)
    batch, tokens = 4, 4
    gen = torch.Generator().manual_seed(9100)
    weights = _weights(gen)
    step = _tokens(gen, batch, tokens)
    conv, rec = _carriers(gen, batch)
    conv[3] = float("nan")
    rec[3, 0, 5] = float("nan")
    start = torch.tensor([7, 12, 1, 0], dtype=torch.int32)
    real = torch.tensor([4, 2, 0, 1], dtype=torch.int32)
    row_mask = (torch.arange(tokens).reshape(1, tokens) < real.reshape(batch, 1)).to(
        torch.float32
    )
    core, conv_ckpts, rec_ckpts = _chain(
        step, weights, conv, rec, tokens=tokens, start_position=start,
        real_tokens=real, row_mask=row_mask,
    )
    out = _tstep(step, weights, conv, rec, start_position=start, real_tokens=real,
                 row_mask=row_mask)
    _assert_bit_equal(out, core, conv_ckpts, rec_ckpts, f"partial lnc={lnc}")
    # The padding request's checkpoints are all its entering carriers.
    for t in range(tokens):
        assert torch.equal(out.conv_checkpoints[2, t], conv[2])
        assert torch.equal(out.recurrent_checkpoints[2, t], rec[2])
    # Request 1's rows 2 and 3 are padding: checkpoints 1, 2, 3 are one state.
    for t in (2, 3):
        assert torch.equal(out.conv_checkpoints[1, t], out.conv_checkpoints[1, 1])
        assert torch.equal(
            out.recurrent_checkpoints[1, t], out.recurrent_checkpoints[1, 1]
        )
    assert torch.isfinite(out.recurrent_checkpoints[3]).all()


def test_tstep_dim_first_conv_layout(monkeypatch):
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    batch, tokens = 4, 4
    gen = torch.Generator().manual_seed(9200)
    weights = _weights(gen)
    step = _tokens(gen, batch, tokens)
    conv, rec = _carriers(gen, batch, dim_first=True)
    start = torch.tensor([3, 0, 8, 2], dtype=torch.int32)
    core, conv_ckpts, rec_ckpts = _chain(
        step, weights, conv, rec, tokens=tokens, start_position=start, dim_first=True
    )
    out = _tstep(step, weights, conv, rec, start_position=start, dim_first=True)
    _assert_bit_equal(out, core, conv_ckpts, rec_ckpts, "DS")


def test_tstep_two_heads_two_chunks(monkeypatch):
    """Two heads of 32 per rank, 36 requests (two request chunks), T=3."""
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    heads, kdim, batch, tokens = 2, 32, 36, 3
    gen = torch.Generator().manual_seed(9300)
    weights = _weights(gen, heads=heads, kdim=kdim)
    step = _tokens(gen, batch, tokens, heads=heads, kdim=kdim)
    conv, rec = _carriers(gen, batch, heads=heads, kdim=kdim)
    start = torch.randint(0, 9, (batch,), generator=gen, dtype=torch.int32)
    core, conv_ckpts, rec_ckpts = _chain(
        step, weights, conv, rec, tokens=tokens, start_position=start
    )
    out = _tstep(step, weights, conv, rec, start_position=start)
    _assert_bit_equal(out, core, conv_ckpts, rec_ckpts, "H=2 K=32 B=36 T=3")


def test_commit_selects_the_checkpoint_of_the_accepted_count(monkeypatch):
    """A bank of ``[slots, T, ...]`` rows: ``commit(accepted_counts=n)`` (``n``
    tokens kept, ``1 .. T``) returns row ``n - 1``, and that row holds the chained
    state after ``n`` tokens, for every ``n``; a device tensor passes unread."""
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    batch, tokens, bank_slots = 3, 4, 6
    slots = torch.tensor([4, 0, 2], dtype=torch.int64)
    gen = torch.Generator().manual_seed(9400)
    weights = _weights(gen)
    step = _tokens(gen, batch, tokens)
    conv, rec = _carriers(gen, batch)
    start = torch.tensor([5, 0, 11], dtype=torch.int32)
    _, conv_ckpts, rec_ckpts = _chain(
        step, weights, conv, rec, tokens=tokens, start_position=start
    )
    out = _tstep(step, weights, conv, rec, start_position=start)
    conv_bank = torch.zeros((bank_slots, tokens, *conv.shape[1:]), dtype=conv.dtype)
    rec_bank = torch.zeros((bank_slots, tokens, *rec.shape[1:]), dtype=rec.dtype)
    # The step writes whole slots: every checkpoint row of each request's slot.
    conv_bank.index_copy_(0, slots, out.conv_checkpoints)
    rec_bank.index_copy_(0, slots, out.recurrent_checkpoints)
    for kept in range(1, tokens + 1):
        a = kept - 1
        row = fused.commit_kda_checkpoints(
            (conv_bank, rec_bank), slots, torch.full((batch,), kept, dtype=torch.int32),
            state_checkpoints=tokens,
        )
        assert row.dtype == torch.int32 and tuple(row.shape) == (batch,)
        assert row.tolist() == [a] * batch
        rows = fused.kda_checkpoint_rows(slots, row, tokens)
        assert rows.tolist() == [s * tokens + a for s in slots.tolist()]
        live_conv = conv_bank.flatten(0, 1)[rows]
        live_rec = rec_bank.flatten(0, 1)[rows]
        assert torch.equal(_bytes(live_conv), _bytes(conv_ckpts[:, a])), a
        assert torch.equal(_bytes(live_rec), _bytes(rec_ckpts[:, a])), a
    # Mixed counts per request, host ints, and the refusals (0 kept and T + 1 kept).
    mixed = fused.commit_kda_checkpoints(
        (conv_bank, rec_bank), slots, [4, 1, 3], state_checkpoints=tokens
    )
    assert mixed.tolist() == [3, 0, 2]
    for bad in ([1, 0, 1], [1, tokens + 1, 1]):
        with pytest.raises(fused.KdaFusedDecodeError, match="accepted"):
            fused.commit_kda_checkpoints(
                (conv_bank, rec_bank), slots, bad, state_checkpoints=tokens
            )
    with pytest.raises(fused.KdaFusedDecodeError, match="checkpoint"):
        fused.commit_kda_checkpoints(
            (conv_bank, rec_bank), slots, [1, 1, 1], state_checkpoints=tokens + 1
        )
    with pytest.raises(fused.KdaFusedDecodeError, match="slot"):
        fused.commit_kda_checkpoints(
            (conv_bank, rec_bank), torch.tensor([bank_slots, 0, 1]), [1, 1, 1],
            state_checkpoints=tokens,
        )


def test_torch_fallback_matches_the_tstep_kernel(monkeypatch):
    """With NKI off, the counted fallback computes the same T-step."""
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    batch, tokens = 4, 3
    gen = torch.Generator().manual_seed(9500)
    weights = _weights(gen)
    step = _tokens(gen, batch, tokens)
    conv, rec = _carriers(gen, batch)
    start = torch.tensor([0, 4, 1, 9], dtype=torch.int32)
    real = torch.tensor([3, 3, 0, 2], dtype=torch.int32)
    row_mask = (torch.arange(tokens).reshape(1, tokens) < real.reshape(batch, 1)).to(
        torch.float32
    )
    kernel = _tstep(step, weights, conv, rec, start_position=start, real_tokens=real,
                    row_mask=row_mask)
    monkeypatch.setenv("NKI_SIMULATOR", "0")
    fused.reset_fused_decode_dispatch_counters()
    fallback = _tstep(step, weights, conv, rec, start_position=start, real_tokens=real,
                      row_mask=row_mask)
    assert fused.fused_decode_dispatch_counters() == (0, 1)
    assert torch.equal(_bytes(kernel.conv_checkpoints), _bytes(fallback.conv_checkpoints))
    torch.testing.assert_close(kernel.core, fallback.core, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(
        kernel.recurrent_checkpoints, fallback.recurrent_checkpoints, rtol=1e-4, atol=1e-5
    )


def test_tstep_refuses_mismatched_shapes():
    gen = torch.Generator().manual_seed(9600)
    weights = _weights(gen)
    batch, tokens = 2, 3
    step = _tokens(gen, batch, tokens)
    conv, rec = _carriers(gen, batch)
    start = torch.tensor([1, 1], dtype=torch.int32)
    # Rows that are not a whole number of tokens per request.
    bad = {k: v[: batch * tokens - 1] for k, v in step.items()}
    with pytest.raises(fused.KdaFusedDecodeError, match="request"):
        _tstep(bad, weights, conv, rec, start_position=start)
    with pytest.raises(fused.KdaFusedDecodeError, match="recurrent_state"):
        _tstep(step, weights, conv, rec[:, :, :, :64], start_position=start)
    with pytest.raises(fused.KdaFusedDecodeError, match="row_mask"):
        _tstep(step, weights, conv, rec, start_position=start,
               real_tokens=torch.tensor([3, 3]), row_mask=torch.ones(batch, tokens + 1))
    with pytest.raises(fused.KdaFusedDecodeError, match="real_tokens"):
        _tstep(step, weights, conv, rec, start_position=start,
               real_tokens=torch.tensor([3, tokens + 1]), row_mask=torch.ones(batch, tokens))
