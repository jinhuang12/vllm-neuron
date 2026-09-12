# SPDX-License-Identifier: Apache-2.0
"""Acceptance for `inc-glm53f-036` -- the KDA decode state carry NKI kernel.

**Three items, one per declared case, and no ``parametrize`` decorator in this
file** (D1.2). Each item names the component whose behaviour it certifies (D1.4).
The whole file is the selection, so the collected count is 3.

Run on the Tier N harness -- the NKI simulator on the host CPU, no device and no
lease::

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
    python -m pytest test/vllm_neuron/functional/kda/test_decode_state.py \
        -q -s --timeout 60 -p no:cacheprovider

WHAT EACH ITEM CLAIMS. One decode step applied ``k`` times, starting from a zero
state, reproduces the final state of a ``k``-token prefill -- for ``k`` of 1, 4 and
16, one item each.

THE PREFILL REFERENCE IS THE SEQUENTIAL TORCH SCAN, NOT THE LANDED CHUNKED
KERNEL, and the choice is load-bearing for three reasons. First, it serves
``k = 1``: the chunked path requires ``chunk >= 2``, so it cannot produce a
one-token prefill at all. Second, it shares no formula with the decode kernel --
it is torch walking tokens, the kernel is NKI advancing one -- so agreement is a
real claim rather than a kernel checked against a kernel. Third, `-035b`'s
conjunct 1 already tied the landed chunked prefill to THIS SAME oracle at
``5.960e-07``, so the statement "decode agrees with the landed prefill" is
already measured and citable and does not need re-deriving here.

WHY THE ROUTE READING IS AN EQUALITY AND NOT AN UPPER BOUND. The kernel advances
exactly one token per dispatch and contains no loop, so ``k`` steps make ``k``
dispatches. Each item below asserts the counter against ITS OWN declared step
count -- the same value that sets the reference's token count -- so the two cannot
drift apart, and a seam that batched tokens internally would read fewer than the
step count and fail.

ONE THING THE ``k = 1`` CASE CANNOT SEE, MEASURED RATHER THAN REASONED. Decode
starts from a zero state, and the decay multiplies that zero, so at ``k = 1`` the
per-key-channel decay is unobservable: removing it entirely changes the answer by
exactly ``0.000e+00``. At ``k = 4`` and ``k = 16`` it changes by ``6.241e-02`` and
``2.588e-01``. So the three cases are not three copies of one check, and the
acceptance driver's red arm A demonstrates that split instead of asserting it.
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

#: The frozen comparator pair, quoted from the plan block's Acceptance bullet:
#: "reproduces a ``k``-token prefill's final state at ``assert_close(rtol=1e-2,
#: atol=1e-5)``". Registered before measurement and not touched after (P9).
#:
#: WHAT THAT PAIR ACTUALLY BOUNDS, because the reported numbers look odd until you
#: know. ``assert_close`` forms ``rel_error = (abs_diff - atol) / max|expected|``
#: and requires ``rel_error <= rtol``, so the effective bound is
#: ``atol + rtol * max|expected|`` -- rtol normalised by the tensor's own scale
#: rather than per element. A NEGATIVE ``max_rel_error`` therefore means the worst
#: element sits below ``atol`` on its own, before the rtol term contributes
#: anything, which is the case in all three cases here.
RTOL = 1e-2
ATOL = 1e-5

#: Key and value widths. The block declares the STEP counts and leaves these
#: free; 64 is a real head width and matches what the chunked file uses, so the
#: two increments' numbers are comparable.
KDIM = 64
VDIM = 64

#: Gate magnitude. KDA gates are log-space decays, so a negative draw is the
#: realistic sign, and this range is the one the chunked file uses.
GATE_SCALE = 0.05

#: One seed for the whole file. A ``k``-token case draws the FIRST ``k`` tokens of
#: the same stream, so the three cases are nested rather than unrelated.
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
    """Apply the counted seam ``steps`` times from a zero state.

    Returns ``(o_all, state, counters)``. The reset happens immediately before the
    first call and the read immediately after the last, so the reading covers
    exactly this case's calls and no other's.
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


def _report(item: str, certifies: str) -> None:
    print(f"\nDECODE|{item}|certifies={certifies}", flush=True)


def _run_case(steps: int) -> None:
    """One declared case. ``steps`` is the case's own declared step count.

    THE SAME ``steps`` SETS THREE THINGS: the reference prefill's token count, the
    number of seam calls, and the dispatch count asserted against. They therefore
    cannot drift apart, which is what makes the route reading an equality against
    this case's own step count rather than against a constant.
    """
    inputs = _flat_inputs(steps)
    reference = kda_sequential_torch_oracle(*inputs)

    gate_abs_max = float(inputs[4].abs().max().item())
    assert can_run_decode_step(inputs[0], KDIM, VDIM, gate_abs_max) is True, (
        "the NKI route must be available on the Tier N harness; a False here "
        "would mean the seam took the torch fallback"
    )

    o_all, state, (nki_dispatch, torch_fallback) = _decode_scan(steps, inputs)

    assert tuple(state.shape) == (VDIM, KDIM), (
        f"the advanced state must come back in the [V, K] orientation "
        f"{(VDIM, KDIM)} that `-035b`'s final_state is stored in, got "
        f"{tuple(state.shape)}"
    )

    st_result = assert_close(
        state,
        reference.final_state,
        rtol=RTOL,
        atol=ATOL,
        name=f"decode_state_after_{steps}_steps",
    )
    o_result = assert_close(
        o_all,
        reference.o,
        rtol=RTOL,
        atol=ATOL,
        name=f"decode_output_over_{steps}_steps",
    )

    # The route predicate, as an equality against THIS case's step count.
    assert nki_dispatch == steps, (
        f"expected exactly {steps} dispatches for a {steps}-step decode; "
        f"{nki_dispatch} means the seam is not one dispatch per token"
    )
    assert torch_fallback == 0

    # The bound the frozen pair actually imposes, computed from the reference's own
    # scale, so the transcript carries the margin and not just the error.
    ref_absmax = reference.final_state.abs().max().item()
    bound = ATOL + RTOL * ref_absmax
    print(
        f"DECODE_CASE|steps={steps}|dispatch={nki_dispatch}|declared={steps}"
        f"|fallback={torch_fallback}"
        f"|state_max_abs={st_result.max_abs_error:.3e}"
        f"|state_linf_rel={st_result.linf_rel:.3e}"
        f"|state_max_rel_signed={st_result.max_rel_error:.3e}"
        f"|state_mismatches={st_result.num_mismatches}"
        f"|o_max_abs={o_result.max_abs_error:.3e}"
        f"|o_linf_rel={o_result.linf_rel:.3e}"
        f"|o_mismatches={o_result.num_mismatches}"
        f"|gate_abs_max={gate_abs_max:.4f}"
        f"|ref_state_absmax={ref_absmax:.4f}"
        f"|effective_bound={bound:.3e}"
        f"|passes_on_atol_alone={st_result.max_abs_error < ATOL}",
        flush=True,
    )


def test_one_decode_step_reproduces_the_one_token_prefill_state():
    """Case k = 1. Certifying component: the ``wrap_nki`` seam in ``decode_state``.

    The weakest of the three by construction and the docstring above says why: a
    zero initial state hides the decay. It is still the case that pins the shape
    contract and the single-dispatch reading, and it is the only one the chunked
    prefill path could not produce a reference for.
    """
    _report("k=1", "kda_decode_step (the wrap_nki seam this block authors)")
    _run_case(steps=1)


def test_four_decode_steps_reproduce_the_four_token_prefill_state():
    """Case k = 4. Certifying component: the ``wrap_nki`` seam in ``decode_state``.

    The first case that carries state across a step boundary, so the first one
    that can observe the per-key-channel decay at all.
    """
    _report("k=4", "kda_decode_step (the wrap_nki seam this block authors)")
    _run_case(steps=4)


def test_sixteen_decode_steps_reproduce_the_sixteen_token_prefill_state():
    """Case k = 16. Certifying component: the ``wrap_nki`` seam in ``decode_state``.

    Sixteen boundaries rather than three, so an error that compounds per step has
    four times the room to show, and the dispatch equality is read against the
    largest declared step count.
    """
    _report("k=16", "kda_decode_step (the wrap_nki seam this block authors)")
    _run_case(steps=16)
