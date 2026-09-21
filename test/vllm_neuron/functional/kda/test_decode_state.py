# SPDX-License-Identifier: Apache-2.0
"""Tests for the KDA decode state carry kernel.

One decode step applied ``k`` times from a zero state must reproduce the final
state of a ``k``-token prefill, for ``k`` of 1, 4 and 16.

The prefill reference is the sequential torch scan rather than the chunked
kernel, for two reasons: the chunked path needs ``chunk >= 2`` so it cannot
produce a one-token prefill at all, and it shares no formula with the decode
kernel, so agreement is a real claim rather than a kernel checked against a
kernel.

The three step counts are not three copies of one check. Decode starts from a
zero state and the decay multiplies that zero, so at ``k = 1`` the per-key-channel
decay is unobservable; it only becomes visible once state crosses a step boundary.
"""

from __future__ import annotations

import torch

from vllm_neuron.accuracy.testing import assert_close
from vllm_neuron.functional.kda.chunked_recurrence import kda_sequential_torch_oracle
from vllm_neuron.functional.kda.decode_state import (
    can_run_decode_step,
    decode_dispatch_counters,
    kda_decode_step,
    reset_decode_dispatch_counters,
)

RTOL = 1e-2
ATOL = 1e-5

#: Key and value widths. 64 is a real head width and matches what the chunked
#: tests use, so the two files' numbers are comparable.
KDIM = 64
VDIM = 64

#: Gate magnitude. KDA gates are log-space decays, so a negative draw is the
#: realistic sign.
GATE_SCALE = 0.05

#: One seed for the whole file, so a ``k``-token case draws the first ``k`` tokens
#: of the same stream and the three cases are nested rather than unrelated.
SEED = 20260904


def _flat_inputs(tokens: int):
    """Deterministic flat inputs for a ``tokens``-token sequence."""
    gen = torch.Generator().manual_seed(SEED)
    q = torch.randn((tokens, KDIM), generator=gen, dtype=torch.float32)
    k = torch.randn((tokens, KDIM), generator=gen, dtype=torch.float32)
    v = torch.randn((tokens, VDIM), generator=gen, dtype=torch.float32)
    beta = torch.rand(tokens, generator=gen, dtype=torch.float32) * 0.9 + 0.05
    gk = -torch.rand((tokens, KDIM), generator=gen, dtype=torch.float32) * GATE_SCALE
    return q, k, v, beta, gk


def _decode_scan(steps: int, inputs):
    """Apply the counted entry point ``steps`` times from a zero state.

    Returns ``(o_all, state, counters)``. The reset happens immediately before the
    first call and the read immediately after the last, so the reading covers
    exactly this case's calls.
    """
    q, k, v, beta, gk = inputs
    state = torch.zeros(VDIM, KDIM, dtype=torch.float32)
    reset_decode_dispatch_counters()
    outs = []
    for t in range(steps):
        got = kda_decode_step(
            state,
            q[t : t + 1],
            k[t : t + 1],
            v[t : t + 1],
            beta[t : t + 1].reshape(1, 1),
            gk[t : t + 1],
        )
        state = got.state
        outs.append(got.o)
    return torch.cat(outs, dim=0), state, decode_dispatch_counters()


def _run_case(steps: int) -> None:
    """Run one step count against its own ``steps``-token prefill reference.

    The same ``steps`` sets the reference's token count, the number of calls and
    the dispatch count asserted against, so the three cannot drift apart. The
    kernel advances exactly one token per dispatch and holds no loop, so the
    dispatch reading is an equality: a kernel that batched tokens internally would
    read fewer.
    """
    inputs = _flat_inputs(steps)
    reference = kda_sequential_torch_oracle(*inputs)

    gate_abs_max = float(inputs[4].abs().max().item())
    assert can_run_decode_step(inputs[0], KDIM, VDIM, gate_abs_max) is True, (
        "the NKI path must be available; a False here would mean the torch "
        "fallback served this case"
    )

    o_all, state, (nki_dispatch, torch_fallback) = _decode_scan(steps, inputs)

    assert tuple(state.shape) == (VDIM, KDIM), (
        f"the advanced state must come back in the [V, K] orientation "
        f"{(VDIM, KDIM)} that the chunked path stores final_state in, got "
        f"{tuple(state.shape)}"
    )

    assert_close(
        state, reference.final_state, rtol=RTOL, atol=ATOL,
        name=f"decode_state_after_{steps}_steps",
    )
    assert_close(
        o_all, reference.o, rtol=RTOL, atol=ATOL,
        name=f"decode_output_over_{steps}_steps",
    )

    assert nki_dispatch == steps, (
        f"expected exactly {steps} dispatches for a {steps}-step decode; "
        f"{nki_dispatch} means the kernel is not one dispatch per token"
    )
    assert torch_fallback == 0


def test_one_decode_step_reproduces_the_one_token_prefill_state():
    """One step from a zero state matches a one-token prefill.

    The weakest of the three, because a zero initial state hides the decay, but it
    pins the shape contract and is the only case the chunked path cannot produce a
    reference for.
    """
    _run_case(steps=1)


def test_four_decode_steps_reproduce_the_four_token_prefill_state():
    """Four steps match a four-token prefill.

    The first case that carries state across a step boundary, so the first that can
    observe the per-key-channel decay at all.
    """
    _run_case(steps=4)


def test_sixteen_decode_steps_reproduce_the_sixteen_token_prefill_state():
    """Sixteen steps match a sixteen-token prefill.

    Sixteen boundaries rather than three, so an error that compounds per step has
    more room to show.
    """
    _run_case(steps=16)
