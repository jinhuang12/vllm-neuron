# SPDX-License-Identifier: Apache-2.0
"""Acceptance for the inter-chunk seam's ENTERING RECURRENT STATE operand.

Three items, one per counted conjunct, no ``parametrize``. The seam already
dispatched a kernel that declares an entering-state input
(``chunked_recurrence.py:984``, loaded into the loop-carried tile at
``:1028-1029``); these items grade the operand's route from a caller to that
input, and the arithmetic through it.

Run on the Tier N harness -- the NKI simulator on the host CPU, no device and no
lease::

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
    python -m pytest \
        test/vllm_neuron/functional/kda/test_inter_chunk_entering_state_116.py \
        -q -s -rA --timeout 300 -p no:randomly -p no:cacheprovider

1. a sequence run in two halves, the second entered with the first's final
   state, equals the sequential scan over the same flat tokens -- and the same
   run with the state dropped does NOT, which is the arm that makes item 1
   discriminating rather than decorative;
2. the route predicate -- one NKI dispatch per seam call, no torch fallback,
   read around the calls and against a zero reading taken before them;
3. the operand is validated rather than trusted: a wrong rank, a wrong shape
   and a wrong dtype each refuse by name.
"""

from __future__ import annotations

import pytest
import torch

from test.vllm_neuron.functional.kda import test_chunked_recurrence as chunk_half

from vllm_neuron.accuracy.testing import assert_close
from vllm_neuron.functional.kda.chunked_recurrence import (
    ChunkedRecurrenceError,
    inter_dispatch_counters,
    kda_intra_chunk,
    kda_inter_chunk,
    kda_sequential_torch_oracle,
    reset_inter_dispatch_counters,
)

#: The comparator pair and the two head widths are CARRIED from the landed
#: chunked-recurrence acceptance, imported rather than retyped, so this file
#: mints no comparator of its own.
RTOL = chunk_half.RTOL
ATOL = chunk_half.ATOL
KDIM = chunk_half.KDIM
VDIM = chunk_half.VDIM

#: The chunk width these items run at, and the split. Two chunks per half keeps
#: the seam's own per-call dispatch reading discriminating -- a seam that
#: dispatched per chunk would read 2 where the design reads 1 -- and gives the
#: entering state a second chunk to be decayed through inside the kernel.
CHUNK = 32
CHUNKS_PER_HALF = 2
HALF_TOKENS = CHUNK * CHUNKS_PER_HALF
TOKENS = HALF_TOKENS * 2

#: One NKI dispatch per seam call and no torch fallback.
DECLARED_DISPATCH_PER_CALL = 1
DECLARED_FALLBACKS = 0


def _flat(start: int, stop: int):
    """The landed flat inputs, sliced to one token range."""
    return tuple(x[start:stop].contiguous() for x in chunk_half._flat_inputs())


def _pipeline(tokens, state):
    """One half through the real intra-chunk and inter-chunk kernels.

    Returns ``(o_flat, final_state, counters)``. The counter reading is taken
    around the inter-chunk call alone, so it belongs to that call and no other.
    """
    q, k, v, beta, gk = tokens
    n_chunks = q.shape[0] // CHUNK
    q_c = q.reshape(n_chunks, CHUNK, KDIM).contiguous()
    k_c = k.reshape(n_chunks, CHUNK, KDIM).contiguous()
    v_c = v.reshape(n_chunks, CHUNK, VDIM).contiguous()
    beta_c = beta.reshape(n_chunks, CHUNK).contiguous()
    gk_c = gk.reshape(n_chunks, CHUNK, KDIM).contiguous()
    intra = kda_intra_chunk(q_c, k_c, v_c, beta_c, gk_c)
    reset_inter_dispatch_counters()
    out = kda_inter_chunk(
        intra.kg, intra.w, intra.u, gk_c, q_c, intra.aqk, state=state
    )
    counters = inter_dispatch_counters()
    o_flat = chunk_half._t(out.o).reshape(q.shape[0], VDIM)
    return o_flat, chunk_half._t(out.final_state), counters


def _two_halves(carry: bool):
    """Both halves in order; the second entered with the first's state or not."""
    first = _pipeline(_flat(0, HALF_TOKENS), None)
    entering = first[1] if carry else None
    second = _pipeline(_flat(HALF_TOKENS, TOKENS), entering)
    return first, second


def _reference():
    """The sequential scan over the same flat tokens, walked one at a time."""
    return kda_sequential_torch_oracle(*_flat(0, TOKENS))


def _report(item: str, certifies: str) -> None:
    print(f"\nENTERSTATE|{item}|certifies={certifies}", flush=True)


def test_inter_chunk_entering_state_carries_a_split_sequence():
    """Item 1. Certifying component: the entering-state operand's arithmetic.

    Two halves, the second entered with the first's final state, against the
    sequential scan over the whole token range. The must-fail arm re-runs the
    identical halves with the state dropped and requires the SAME comparison to
    raise, so a seam that ignored the operand cannot pass this item.
    """
    _report("item1_split_equals_whole", "the entering-state operand")
    ref = _reference()
    (first_o, _, _), (second_o, second_state, _) = _two_halves(carry=True)
    joined = torch.cat((first_o, second_o), dim=0)
    worst_o = chunk_half._worst(joined, ref.o)
    worst_state = chunk_half._worst(second_state, ref.final_state)
    print(
        f"ENTERSTATE|item1|tokens={TOKENS}|split={HALF_TOKENS}+{HALF_TOKENS}|"
        f"chunk={CHUNK}|worst_abs_error_o={worst_o:.3e}|"
        f"worst_abs_error_state={worst_state:.3e}|"
        f"declared_rtol={RTOL}|declared_atol={ATOL}",
        flush=True,
    )
    assert_close(joined, ref.o, rtol=RTOL, atol=ATOL, name="split.o")
    assert_close(
        second_state, ref.final_state, rtol=RTOL, atol=ATOL, name="split.final_state"
    )

    (_, _, _), (bad_o, bad_state, _) = _two_halves(carry=False)
    bad_worst_o = chunk_half._worst(bad_o, ref.o[HALF_TOKENS:])
    bad_worst_state = chunk_half._worst(bad_state, ref.final_state)
    print(
        f"ENTERSTATE|item1_must_fail|worst_abs_error_o={bad_worst_o:.3e}|"
        f"worst_abs_error_state={bad_worst_state:.3e}",
        flush=True,
    )
    with pytest.raises(AssertionError):
        assert_close(
            bad_o, ref.o[HALF_TOKENS:], rtol=RTOL, atol=ATOL, name="no_carry.o"
        )
    with pytest.raises(AssertionError):
        assert_close(
            bad_state,
            ref.final_state,
            rtol=RTOL,
            atol=ATOL,
            name="no_carry.final_state",
        )


def test_inter_chunk_entering_state_reads_one_dispatch_per_call():
    """Item 2. Certifying component: the seam this file drives, form R-1.

    One NKI dispatch per seam call and no torch fallback, on the carried route
    and on the whole-sequence control. The pair read before either call is
    asserted at zero, so a stale counter cannot satisfy the equality.
    """
    _report("item2_route_predicate", "the inter-chunk seam's dispatch counters")
    reset_inter_dispatch_counters()
    before = inter_dispatch_counters()
    print(f"ENTERSTATE|item2|before={before}", flush=True)
    assert before == (0, 0), f"the counters read {before} before any call"

    (_, _, first_counters), (_, _, second_counters) = _two_halves(carry=True)
    _, _, whole_counters = _pipeline(_flat(0, TOKENS), None)
    print(
        f"ENTERSTATE|item2|first_half={first_counters}|second_half={second_counters}|"
        f"whole={whole_counters}|declared_dispatch_per_call="
        f"{DECLARED_DISPATCH_PER_CALL}|declared_fallbacks={DECLARED_FALLBACKS}",
        flush=True,
    )
    declared = (DECLARED_DISPATCH_PER_CALL, DECLARED_FALLBACKS)
    for label, counters in (
        ("first half", first_counters),
        ("second half, entered with a state", second_counters),
        ("the whole sequence", whole_counters),
    ):
        assert counters == declared, (
            f"{label} read {counters}; the declared reading is {declared}, one NKI "
            f"dispatch per seam call with no torch fallback"
        )


def test_inter_chunk_refuses_an_entering_state_it_cannot_serve():
    """Item 3. Certifying component: the seam's validation of the new operand.

    Three inputs the seam must refuse -- a wrong rank, a wrong shape and a
    wrong dtype -- each naming what it saw, and the declared state accepted.
    """
    _report("item3_operand_validated", "the seam's refusals on the new operand")
    tokens = _flat(0, HALF_TOKENS)
    good = torch.zeros(VDIM, KDIM, dtype=torch.float32)
    refused = {
        "wrong_rank": torch.zeros(VDIM, dtype=torch.float32),
        "wrong_shape": torch.zeros(VDIM, KDIM + 1, dtype=torch.float32),
        "wrong_dtype": torch.zeros(VDIM, KDIM, dtype=torch.bfloat16),
    }
    for label, state in refused.items():
        with pytest.raises(ChunkedRecurrenceError) as raised:
            _pipeline(tokens, state)
        message = str(raised.value)
        print(f"ENTERSTATE|item3|{label}|refusal={message}", flush=True)
        assert "state" in message, f"the {label} refusal does not name the operand"

    _, final_state, _ = _pipeline(tokens, good)
    print(
        f"ENTERSTATE|item3|accepted_shape={tuple(good.shape)}|"
        f"returned_shape={tuple(final_state.shape)}",
        flush=True,
    )
    assert tuple(final_state.shape) == (VDIM, KDIM), (
        f"the seam returned {tuple(final_state.shape)} for a declared "
        f"{(VDIM, KDIM)} entering state; the operand and the return are one "
        f"orientation"
    )
