# SPDX-License-Identifier: Apache-2.0
"""The fused KDA decode kernel against the 5938748 kernels it replaces.

The reference is the 5938748 decode region, run on the 5938748 kernel snapshots
in ``test/hardware/baselines/kda_5938748`` (conv, gate clamp, decode step, and
the model glue between them), one request at a time as the model ran it. The
fused kernel serves all ``B`` requests in one launch.

Shapes are the TP=64 shapes: one head per rank, ``K = V = 128``, a 4-tap conv
over ``3 * 128 = 384`` channels, a bfloat16 conv carrier in the default ``SD``
layout and a float32 recurrent carrier.

Tolerance, NEW vs OLD: the conv carrier is data movement plus one cast, so it
must agree bit for bit. ``core`` and the recurrent state agree to
``RTOL``/``ATOL`` below; the two paths sum the same products in a different
order (a 4-tap reduce instead of a 4x1 matmul, a ones-matmul norm instead of a
free-axis sum) and associate ``beta * kn * delta`` differently, which is
float32 rounding, not a formula change.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch

from vllm_neuron.functional.kda import fused_decode as fused

REPO = Path(__file__).resolve().parents[4]
BASELINE_DIR = REPO / "test" / "hardware" / "baselines" / "kda_5938748"

#: Relative and absolute tolerance on ``core`` and the recurrent state.
RTOL = 1e-4
ATOL = 1e-5

HEADS = 1
KDIM = 128
TAPS = 4
LOWER = -5.0

#: Shift on the raw gate for the long-memory case: sigmoid near zero, so the
#: per-step log decay ``lower * sigmoid`` is near zero and the state accumulates.
LONG_MEMORY_GATE_SHIFT = -6.0

#: Projection scale for the long-memory case, so ``v`` and the state it
#: accumulates reach several units, as a long decode's do.
LONG_INPUT_SCALE = 3.0

#: The 32-step state must exceed this max |s|, several times the 0.1-scale
#: random draw's (about 0.45 over 16k samples).
GROWN_STATE_MIN = 2.0


def _load_baseline():
    spec = importlib.util.spec_from_file_location(
        "_kda_5938748_loader", BASELINE_DIR / "loader.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module.load_baseline(BASELINE_DIR)


LOADER, OLD = _load_baseline()


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


def _step_inputs(gen, batch, heads=HEADS, kdim=KDIM, gate_shift=0.0, scale=1.0):
    width = heads * kdim
    return {
        "q_in": torch.randn(batch, width, generator=gen) * scale,
        "k_in": torch.randn(batch, width, generator=gen) * scale,
        "v_in": torch.randn(batch, width, generator=gen) * scale,
        "raw_gate": torch.randn(batch, width, generator=gen) * 2 + gate_shift,
        "raw_beta": torch.randn(batch, heads, generator=gen),
    }


def _carriers(gen, batch, heads=HEADS, kdim=KDIM, taps=TAPS, dim_first=False,
              conv_dtype=torch.bfloat16, state_scale=0.1):
    channels = 3 * heads * kdim
    rows = taps - 1
    shape = (batch, channels, rows) if dim_first else (batch, rows, channels)
    conv = torch.randn(shape, generator=gen).to(conv_dtype)
    rec = torch.randn(batch, heads, kdim, kdim, generator=gen) * state_scale
    return conv, rec


def _old(step, weights, conv, rec, *, dim_first=False, start_position=None,
         real_tokens=None, row_mask=None):
    """The 5938748 region, request by request, on clones of the carriers."""
    conv = conv.clone()
    rec = rec.clone()
    batch = int(step["q_in"].shape[0])
    cores = []
    for b in range(batch):
        cores.append(
            LOADER.old_decode_core(
                OLD,
                step["q_in"][b : b + 1],
                step["k_in"][b : b + 1],
                step["v_in"][b : b + 1],
                step["raw_gate"][b : b + 1],
                step["raw_beta"][b : b + 1],
                conv_state=conv[b],
                recurrent_state=rec[b],
                gate_lower_bound=LOWER,
                conv_state_dim_first=dim_first,
                start_position=1 if start_position is None else start_position[b],
                real_tokens=None if real_tokens is None else real_tokens[b : b + 1],
                row_mask=None if row_mask is None else row_mask[b : b + 1],
                **weights,
            )
        )
    return torch.cat(cores, dim=0), conv, rec


def _new(step, weights, conv, rec, *, dim_first=False, start_position=None,
         real_tokens=None, row_mask=None):
    batch = int(step["q_in"].shape[0])
    if start_position is None:
        start_position = torch.ones(batch, dtype=torch.int32)
    return fused.kda_fused_decode(
        **step,
        conv_state=conv,
        recurrent_state=rec,
        gate_lower_bound=LOWER,
        conv_state_dim_first=dim_first,
        start_position=start_position,
        real_tokens=real_tokens,
        row_mask=row_mask,
        **weights,
    )


def _assert_agrees(new, old_core, old_conv, old_rec, label):
    assert new.conv_state.dtype == old_conv.dtype
    assert torch.equal(
        new.conv_state.contiguous().view(torch.uint8),
        old_conv.contiguous().view(torch.uint8),
    ), f"{label}: conv carrier differs from 5938748 (must be bit-exact)"
    torch.testing.assert_close(
        new.core, old_core, rtol=RTOL, atol=ATOL, msg=lambda m: f"{label} core: {m}"
    )
    torch.testing.assert_close(
        new.recurrent_state, old_rec, rtol=RTOL, atol=ATOL,
        msg=lambda m: f"{label} state: {m}",
    )


def _run_and_compare(batch, seed, *, lnc=None, monkeypatch=None, **kwargs):
    gen = torch.Generator().manual_seed(seed)
    weights = _weights(gen)
    step = _step_inputs(gen, batch)
    conv, rec = _carriers(gen, batch)
    old_core, old_conv, old_rec = _old(step, weights, conv, rec, **kwargs)
    fused.reset_fused_decode_dispatch_counters()
    new = _new(step, weights, conv, rec, **kwargs)
    assert fused.fused_decode_dispatch_counters() == (1, 0), (
        "the fused NKI kernel must serve the whole batch in one dispatch and the "
        "torch fallback must not run"
    )
    _assert_agrees(new, old_core, old_conv, old_rec, f"B={batch}")
    return new


@pytest.mark.parametrize("batch", [1, 4])
def test_fused_matches_5938748_at_real_shapes(batch, monkeypatch):
    monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    assert fused.fused_decode_grid(KDIM) == ()
    _run_and_compare(batch, 7100 + batch)


@pytest.mark.parametrize("batch", [1, 4])
def test_fused_lnc2_grid_matches_5938748(batch, monkeypatch):
    """Both LNC2 programs: each serves half the value rows of every request."""
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    assert fused.fused_decode_grid(KDIM) == (2,)
    _run_and_compare(batch, 7200 + batch)


@pytest.mark.parametrize("batch", [1, 4])
def test_fused_matches_after_32_old_steps(batch, monkeypatch):
    """A state grown by 32 old decode steps, then 4 steps NEW vs OLD.

    The gate is shifted toward long memory (decay near one) and the projections
    scaled up, so the 32 old steps accumulate rather than forget and the state
    reaches several units, well above the 0.1-scale random draw. Both paths start from
    the same 32-step state and are chained independently for 4 more steps.
    """
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    gen = torch.Generator().manual_seed(7300 + batch)
    weights = _weights(gen)
    conv, rec = _carriers(gen, batch, state_scale=0.0)
    position = torch.zeros(batch, dtype=torch.int32)
    for _ in range(32):
        step = _step_inputs(
            gen, batch, gate_shift=LONG_MEMORY_GATE_SHIFT, scale=LONG_INPUT_SCALE
        )
        _, conv, rec = _old(step, weights, conv, rec, start_position=position)
        position = position + 1
    grown = float(rec.abs().max())
    assert math.isfinite(grown) and grown > GROWN_STATE_MIN, (
        f"32 steps grew the state only to max |s| = {grown}"
    )

    old_conv, old_rec = conv.clone(), rec.clone()
    new_conv, new_rec = conv.clone(), rec.clone()
    for k in range(4):
        step = _step_inputs(
            gen, batch, gate_shift=LONG_MEMORY_GATE_SHIFT, scale=LONG_INPUT_SCALE
        )
        old_core, old_conv, old_rec = _old(
            step, weights, old_conv, old_rec, start_position=position
        )
        new = _new(step, weights, new_conv, new_rec, start_position=position)
        _assert_agrees(new, old_core, old_conv, old_rec, f"B={batch} step {k}")
        new_conv, new_rec = new.conv_state, new.recurrent_state
        position = position + 1


def test_opening_and_padding_requests_match_5938748(monkeypatch):
    """Opening requests ignore NaN carriers; a padding request keeps its state.

    Request 0 and 2 open (position 0) on carriers holding NaN, as a slot whose
    previous owner diverged would; request 2 is also a padding row (no real
    token, mask 0), which must leave both carriers exactly as it found them.
    """
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    batch = 4
    gen = torch.Generator().manual_seed(7400)
    weights = _weights(gen)
    step = _step_inputs(gen, batch)
    conv, rec = _carriers(gen, batch)
    conv[0] = float("nan")
    rec[0] = float("nan")
    conv[2, 1] = float("nan")
    rec[2, 0, 3] = float("nan")
    start_position = torch.tensor([0, 5, 0, 9], dtype=torch.int32)
    real_tokens = torch.tensor([1, 1, 0, 1], dtype=torch.int32)
    row_mask = torch.tensor([[1.0], [1.0], [0.0], [1.0]])
    kwargs = dict(start_position=start_position, real_tokens=real_tokens,
                  row_mask=row_mask)
    old_core, old_conv, old_rec = _old(step, weights, conv, rec, **kwargs)
    new = _new(step, weights, conv, rec, **kwargs)
    _assert_agrees(new, old_core, old_conv, old_rec, "opening/padding")
    assert torch.isfinite(new.recurrent_state).all()
    assert torch.isfinite(new.conv_state.float()).all()
    # The padding request opened, so its carriers are zero and stay zero.
    assert torch.equal(new.recurrent_state[2], torch.zeros_like(rec[2]))
    assert torch.equal(new.conv_state[2], torch.zeros_like(conv[2]))
    # A continuing padding request is the identity on both carriers.
    start_position = torch.tensor([3, 5, 4, 9], dtype=torch.int32)
    clean_conv, clean_rec = _carriers(gen, batch)
    new = _new(step, weights, clean_conv, clean_rec,
               start_position=start_position, real_tokens=real_tokens,
               row_mask=row_mask)
    assert torch.equal(new.recurrent_state[2], clean_rec[2])
    assert torch.equal(new.conv_state[2], clean_conv[2])


@pytest.mark.parametrize("batch", [1, 4])
def test_fused_dim_first_conv_layout(batch, monkeypatch):
    """The ``DS`` conv carrier (``[C, R]`` per request) takes the same kernel."""
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    gen = torch.Generator().manual_seed(7500 + batch)
    weights = _weights(gen)
    step = _step_inputs(gen, batch)
    conv, rec = _carriers(gen, batch, dim_first=True)
    old_core, old_conv, old_rec = _old(step, weights, conv, rec, dim_first=True)
    new = _new(step, weights, conv, rec, dim_first=True)
    _assert_agrees(new, old_core, old_conv, old_rec, f"DS B={batch}")


@pytest.mark.parametrize("lnc", [None, "2"])
def test_fused_serves_several_heads_per_rank(lnc, monkeypatch):
    """Two heads of 32 per rank, a float32 conv carrier, 36 requests (2 chunks)."""
    if lnc is None:
        monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    else:
        monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", lnc)
    heads, kdim, batch = 2, 32, 36
    gen = torch.Generator().manual_seed(7600)
    weights = _weights(gen, heads=heads, kdim=kdim)
    step = _step_inputs(gen, batch, heads=heads, kdim=kdim)
    conv, rec = _carriers(gen, batch, heads=heads, kdim=kdim,
                          conv_dtype=torch.float32)
    old_core, old_conv, old_rec = _old(step, weights, conv, rec)
    fused.reset_fused_decode_dispatch_counters()
    new = _new(step, weights, conv, rec)
    assert fused.fused_decode_dispatch_counters() == (1, 0)
    _assert_agrees(new, old_core, old_conv, old_rec, "H=2 K=32 B=36")


def test_fused_kernel_is_authored_here_and_dispatched():
    module, name = fused.fused_decode_kernel_identity()
    assert module == "vllm_neuron.functional.kda.fused_decode"
    assert name == "kda_fused_decode_kernel"
    gen = torch.Generator().manual_seed(7700)
    assert fused.can_run_fused_decode(torch.zeros(1)) is True, (
        "the NKI path must be available here; False means the torch fallback "
        "would have served every comparison above"
    )


def test_torch_fallback_matches_the_kernel(monkeypatch):
    """With NKI off, the counted fallback computes the same step."""
    gen = torch.Generator().manual_seed(7800)
    weights = _weights(gen)
    step = _step_inputs(gen, 4)
    conv, rec = _carriers(gen, 4)
    kernel = _new(step, weights, conv, rec)
    monkeypatch.setenv("NKI_SIMULATOR", "0")
    fused.reset_fused_decode_dispatch_counters()
    fallback = _new(step, weights, conv, rec)
    assert fused.fused_decode_dispatch_counters() == (0, 1)
    _assert_agrees(kernel, fallback.core, fallback.conv_state,
                   fallback.recurrent_state, "fallback")


def test_fused_refuses_mismatched_shapes():
    gen = torch.Generator().manual_seed(7900)
    weights = _weights(gen)
    step = _step_inputs(gen, 2)
    conv, rec = _carriers(gen, 2)
    with pytest.raises(fused.KdaFusedDecodeError, match="recurrent_state"):
        _new(step, weights, conv, rec[:, :, :, :64])
    with pytest.raises(fused.KdaFusedDecodeError, match="conv_state"):
        _new(step, weights, conv[:1], rec)
    bad = dict(step, raw_beta=step["raw_beta"][:1])
    with pytest.raises(fused.KdaFusedDecodeError, match="raw_beta"):
        _new(bad, weights, conv, rec)


def test_baseline_snapshots_are_the_5938748_files():
    """The loader refuses a snapshot whose bytes moved; it loaded, so they did not."""
    assert set(LOADER.SNAPSHOT_SHA256) == {
        "chunked_recurrence.py", "decode_state.py", "depthwise_conv1d.py",
        "gate_clamp.py",
    }
    assert OLD.directory == BASELINE_DIR.resolve()
    assert math.isfinite(LOWER)
