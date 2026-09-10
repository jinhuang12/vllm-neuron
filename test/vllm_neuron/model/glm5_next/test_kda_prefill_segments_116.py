# SPDX-License-Identifier: Apache-2.0
"""Acceptance for a KDA prefill that CONTINUES a segmented prompt.

Six items, one per counted conjunct, no ``parametrize``. At the pinned serving
shape a prompt longer than one batch of tokens is prefilled in segments, so the
second segment must enter the recurrence with the state the first left. Every
declared value and both comparator numbers are CARRIED from the landed KDA layer
acceptance, imported rather than retyped.

Run on the Tier N harness -- the NKI simulator on the host CPU, no device and no
lease::

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
    python -m pytest \
        test/vllm_neuron/model/glm5_next/test_kda_prefill_segments_116.py \
        -q -s -rA --timeout 900 -p no:randomly -p no:cacheprovider

1. a two-segment prefill equals the one-shot prefill of the same tokens, on the
   stack output and on every layer's final recurrent state;
2. the must-fail arm of item 1 -- the same drive with the second segment entered
   at position 0 must NOT equal the one-shot run;
3. a continuing segment shorter than one chunk carries too, and its inter-chunk
   dispatch count is zero, so the state is shown to reach the single-token seam;
4. a fresh prefill is byte-unchanged: it ignores whatever the bank holds, and
   the same bank at a continuing position does NOT;
5. the runner hands the position, the layer takes it, and an unknown carrier key
   still raises;
6. the route predicate around item 1's drive.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from test.vllm_neuron.model.glm5_next import test_kda_layer as layer_half
from test.vllm_neuron.model.glm5_next import test_kda_runner_state as runner_half
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_e2e as e2e_half
from test.vllm_neuron.model.glm5_next.tiny import (
    test_tiny_glm5next_forward as seam_census,
)

from vllm_neuron.accuracy.testing import assert_close

#: Carried from the landed layer acceptance: the comparator pair, the stack
#: depth, the chunk width, the registered parallel degree and the seed.
DECLARED_RTOL = layer_half.DECLARED_RTOL
DECLARED_ATOL = layer_half.DECLARED_ATOL
DECLARED_STACK_LAYERS = layer_half.DECLARED_STACK_LAYERS
DECLARED_CHUNK = layer_half.DECLARED_CHUNK
DECLARED_PER_RANK_HEADS = layer_half.DECLARED_PER_RANK_HEADS
DECLARED_KDA_HEAD_SIZE = layer_half.DECLARED_KDA_HEAD_SIZE
DECLARED_KDA_CONV_KERNEL_SIZE = layer_half.DECLARED_KDA_CONV_KERNEL_SIZE
REGISTERED_TP_WORLD_SIZE = layer_half.REGISTERED_TP_WORLD_SIZE
SEED = layer_half.SEED

#: The aligned split: two segments of one chunk each. Aligned because that is
#: what the serving shape produces -- every segment but the last is exactly the
#: batched-token bound, which is a whole number of chunks.
ALIGNED_SEGMENT = DECLARED_CHUNK
ALIGNED_TOKENS = ALIGNED_SEGMENT * 2

#: The sub-chunk split: a continuing segment BELOW one chunk, so its recurrence
#: runs entirely on the single-token seam and takes no chunked dispatch at all.
SUB_FIRST = DECLARED_CHUNK * 2
SUB_SECOND = 3
SUB_TOKENS = SUB_FIRST + SUB_SECOND

#: Dispatches the sub-chunk continuing segment owes: none through either chunked
#: seam, and one single-token dispatch per token per head per layer.
DECLARED_SUB_CHUNKED_DISPATCHES = 0
DECLARED_SUB_DECODE_DISPATCHES = (
    SUB_SECOND * DECLARED_PER_RANK_HEADS * DECLARED_STACK_LAYERS
)


def _stack() -> SimpleNamespace:
    """Three KDA layers with deterministic weights, and the token block."""
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    text_config = Glm5NextTextConfig()
    hidden = int(text_config.hidden_size)
    heads = DECLARED_PER_RANK_HEADS
    torch.manual_seed(SEED)
    weights = [
        layer_half._make_weights(
            hidden, heads, DECLARED_KDA_HEAD_SIZE, DECLARED_KDA_CONV_KERNEL_SIZE
        )
        for _ in range(DECLARED_STACK_LAYERS)
    ]
    torch.manual_seed(SEED + 1)
    tokens = torch.randn(max(ALIGNED_TOKENS, SUB_TOKENS), hidden, dtype=torch.float32)

    layers = []
    for index, layer_weights in enumerate(weights):
        layer = layer_half._impl().Glm5NextKDALayer(
            text_config, index, REGISTERED_TP_WORLD_SIZE
        )
        for name, tensor in layer_weights.items():
            target = layer if name == "input_layernorm_weight" else layer.attention
            setattr(target, name, nn.Parameter(tensor.clone(), requires_grad=False))
        layers.append(layer)
    return SimpleNamespace(
        text_config=text_config, layers=layers, tokens=tokens, heads=heads
    )


@pytest.fixture(scope="module")
def stack() -> SimpleNamespace:
    """One built stack for every item; the state lives in a per-drive bank."""
    return _stack()


def _fresh_bank(layers) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """A zeroed carrier pair per layer, allocated from the reported state shapes."""
    bank = []
    for layer in layers:
        attention = layer.attention
        bank.append(
            (
                torch.zeros(
                    attention.kda_conv_state_shape,
                    dtype=attention.kda_conv_state_dtype,
                ),
                torch.zeros(
                    attention.kda_recurrent_state_shape,
                    dtype=attention.kda_recurrent_state_dtype,
                ),
            )
        )
    return bank


def _drive(layers, bank, tokens, *, start_position: int, reset: bool = True):
    """One prefill call through the whole stack, with its own counter reading.

    ``reset`` is off for the route item alone, which reads one window across
    both segments; every other caller wants the reading to be its own call's.
    """
    if reset:
        layer_half._reset_counters()
    out = tokens
    for layer, (conv_state, recurrent_state) in zip(layers, bank):
        out = layer(
            out,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            is_prefill=True,
            start_position=start_position,
            chunk_size=DECLARED_CHUNK,
        )
    return out, layer_half._read_counters()


def _one_shot(stack, total: int):
    """The whole prompt in one prefill, from a zeroed bank."""
    bank = _fresh_bank(stack.layers)
    out, counts = _drive(stack.layers, bank, stack.tokens[:total], start_position=0)
    return out, [rec.clone() for _, rec in bank], counts


def _segmented(stack, first: int, total: int, *, carry: bool, reset: bool = True):
    """The same prompt in two prefill calls over one bank."""
    bank = _fresh_bank(stack.layers)
    head, _ = _drive(
        stack.layers, bank, stack.tokens[:first], start_position=0, reset=reset
    )
    tail, tail_counts = _drive(
        stack.layers,
        bank,
        stack.tokens[first:total],
        start_position=first if carry else 0,
        reset=reset,
    )
    joined = torch.cat((head, tail), dim=0)
    return joined, [rec.clone() for _, rec in bank], tail_counts


def _worst(actual, expected) -> float:
    return (actual.float() - expected.float()).abs().max().item()


def _report(item: str, certifies: str) -> None:
    print(f"\nSEGPREFILL|{item}|certifies={certifies}", flush=True)


def test_kda_prefill_segments_c01_two_segments_equal_the_one_shot_prefill(
    stack: SimpleNamespace,
) -> None:
    """Item 1. Certifying component: the prefill arm's entering state.

    Two segments of one chunk each against one prefill of the same tokens, on
    the stack output and on every layer's final recurrent state.
    """
    _report("item1_segments_equal_one_shot", "the prefill arm's entering state")
    want, want_state, _ = _one_shot(stack, ALIGNED_TOKENS)
    got, got_state, _ = _segmented(stack, ALIGNED_SEGMENT, ALIGNED_TOKENS, carry=True)
    worst_out = _worst(got, want)
    worst_state = max(
        _worst(a, b) for a, b in zip(got_state, want_state)
    )
    print(
        f"SEGPREFILL|item1|tokens={ALIGNED_TOKENS}|"
        f"split={ALIGNED_SEGMENT}+{ALIGNED_TOKENS - ALIGNED_SEGMENT}|"
        f"chunk={DECLARED_CHUNK}|layers={DECLARED_STACK_LAYERS}|"
        f"worst_abs_error_out={worst_out:.3e}|worst_abs_error_state={worst_state:.3e}|"
        f"declared_rtol={DECLARED_RTOL}|declared_atol={DECLARED_ATOL}",
        flush=True,
    )
    assert_close(
        got.float(), want.float(), rtol=DECLARED_RTOL, atol=DECLARED_ATOL,
        name="segmented.out",
    )
    for index, (actual, expected) in enumerate(zip(got_state, want_state)):
        assert_close(
            actual.float(), expected.float(),
            rtol=DECLARED_RTOL, atol=DECLARED_ATOL,
            name=f"segmented.recurrent_state[{index}]",
        )


def test_kda_prefill_segments_c02_dropping_the_carry_misses_the_one_shot_run(
    stack: SimpleNamespace,
) -> None:
    """Item 2. The must-fail arm: item 1 passing must mean something.

    The identical two-segment drive with the second segment entered at position
    0 -- the behaviour before this change -- must NOT reach the comparator item
    1 passes at.
    """
    _report("item2_must_fail_arm", "item 1's discriminating power")
    want, want_state, _ = _one_shot(stack, ALIGNED_TOKENS)
    bad, bad_state, _ = _segmented(
        stack, ALIGNED_SEGMENT, ALIGNED_TOKENS, carry=False
    )
    worst_out = _worst(bad, want)
    worst_state = max(_worst(a, b) for a, b in zip(bad_state, want_state))
    outside = int(
        (
            (bad.float() - want.float()).abs()
            > DECLARED_ATOL + DECLARED_RTOL * want.float().abs()
        )
        .sum()
        .item()
    )
    print(
        f"SEGPREFILL|item2|worst_abs_error_out={worst_out:.3e}|"
        f"worst_abs_error_state={worst_state:.3e}|elements_outside_the_pair={outside}",
        flush=True,
    )
    assert outside > 0, (
        "a segment entered at position 0 reproduced the one-shot prefill within the "
        "declared pair, so item 1 grades nothing"
    )
    with pytest.raises(AssertionError):
        assert_close(
            bad.float(), want.float(),
            rtol=DECLARED_RTOL, atol=DECLARED_ATOL,
            name="no_carry.out",
        )


def test_kda_prefill_segments_c03_a_segment_below_one_chunk_carries_too(
    stack: SimpleNamespace,
) -> None:
    """Item 3. Certifying component: the single-token seam's entering state.

    A continuing segment shorter than one chunk takes no chunked dispatch, so
    this item is where the carried state is shown to reach the walked path.
    """
    _report("item3_sub_chunk_segment", "the single-token seam's entering state")
    want, want_state, _ = _one_shot(stack, SUB_TOKENS)
    got, got_state, counts = _segmented(stack, SUB_FIRST, SUB_TOKENS, carry=True)
    worst_out = _worst(got, want)
    worst_state = max(_worst(a, b) for a, b in zip(got_state, want_state))
    print(
        f"SEGPREFILL|item3|split={SUB_FIRST}+{SUB_SECOND}|"
        f"worst_abs_error_out={worst_out:.3e}|worst_abs_error_state={worst_state:.3e}|"
        f"tail_counts={counts}|declared_chunked={DECLARED_SUB_CHUNKED_DISPATCHES}|"
        f"declared_decode={DECLARED_SUB_DECODE_DISPATCHES}",
        flush=True,
    )
    assert counts["inter"][0] == DECLARED_SUB_CHUNKED_DISPATCHES, (
        f"the continuing segment took {counts['inter'][0]} inter-chunk dispatch(es); "
        f"a segment below one chunk must take {DECLARED_SUB_CHUNKED_DISPATCHES}, or "
        f"this item is not measuring the walked path"
    )
    assert counts["intra"][0] == DECLARED_SUB_CHUNKED_DISPATCHES
    assert counts["decode"][0] == DECLARED_SUB_DECODE_DISPATCHES, (
        f"the continuing segment took {counts['decode'][0]} single-token "
        f"dispatch(es); the declared reading is {DECLARED_SUB_DECODE_DISPATCHES}"
    )
    assert_close(
        got.float(), want.float(), rtol=DECLARED_RTOL, atol=DECLARED_ATOL,
        name="sub_chunk_segmented.out",
    )

    bad, _, _ = _segmented(stack, SUB_FIRST, SUB_TOKENS, carry=False)
    print(
        f"SEGPREFILL|item3_must_fail|worst_abs_error_out={_worst(bad, want):.3e}",
        flush=True,
    )
    with pytest.raises(AssertionError):
        assert_close(
            bad.float(), want.float(),
            rtol=DECLARED_RTOL, atol=DECLARED_ATOL,
            name="sub_chunk_no_carry.out",
        )


def test_kda_prefill_segments_c04_a_fresh_prefill_ignores_the_bank(
    stack: SimpleNamespace,
) -> None:
    """Item 4. Certifying component: the gate on the entering state.

    A prefill at position 0 must be bit-identical over a dirty bank and a clean
    one, and the same dirty bank at a continuing position must not be -- so the
    gate is shown live in both directions.
    """
    _report("item4_fresh_leg_unchanged", "the gate on the entering state")
    clean_out, dirty_source, _ = _one_shot(stack, ALIGNED_SEGMENT)

    dirty = _fresh_bank(stack.layers)
    for (_, recurrent), source in zip(dirty, dirty_source):
        recurrent.copy_(source)
    assert float(max(r.abs().max() for _, r in dirty)) > 0.0, (
        "the dirty bank is all zeros, so this item cannot tell the two routes apart"
    )
    fresh_out, _ = _drive(
        stack.layers, dirty, stack.tokens[:ALIGNED_SEGMENT], start_position=0
    )
    identical = bool(torch.equal(fresh_out, clean_out))

    dirty_again = _fresh_bank(stack.layers)
    for (_, recurrent), source in zip(dirty_again, dirty_source):
        recurrent.copy_(source)
    continued_out, _ = _drive(
        stack.layers,
        dirty_again,
        stack.tokens[:ALIGNED_SEGMENT],
        start_position=ALIGNED_SEGMENT,
    )
    continued_identical = bool(torch.equal(continued_out, clean_out))
    print(
        f"SEGPREFILL|item4|fresh_bit_identical={identical}|"
        f"continuing_bit_identical={continued_identical}|"
        f"worst_abs_error_continuing={_worst(continued_out, clean_out):.3e}",
        flush=True,
    )
    assert identical, (
        "a prefill at position 0 read the bank; a fresh sequence starts the "
        "recurrence at zero whatever the slot holds"
    )
    assert not continued_identical, (
        "the same bank at a continuing position produced the fresh result, so the "
        "position is not gating the entering state at all"
    )


def test_kda_prefill_segments_c05_the_runner_hands_the_position_to_the_layer(
    stack: SimpleNamespace,
) -> None:
    """Item 5. Certifying component: the runner's linear carrier.

    The converter's own carriers, splatted into the landed layer. The bogus-key
    arm is what makes the key-set reading load-bearing: an unknown key reaches
    the layer as a keyword and raises, which is how a carrier the layer does not
    accept fails at serve time.
    """
    _report("item5_carrier_position", "the runner's linear carrier")
    banks = runner_half._banks(stack.layers)
    runner = runner_half._runner(stack.text_config, banks)
    runner.max_model_len = SUB_TOKENS + ALIGNED_TOKENS + 1

    # THE ORDER IS LOAD-BEARING. The converter refuses a step that does not
    # continue the sequence its cursor names, and a prefill at position 0 is what
    # opens one (``neuron_model_runner.py:5195``, ``:5246-5259``), so the fresh
    # call has to come first for the continuing call to be servable at all.
    fresh = runner_half._carriers(runner, banks, tokens=ALIGNED_SEGMENT, cached=0)
    continuing = runner_half._carriers(
        runner, banks, tokens=ALIGNED_SEGMENT, cached=ALIGNED_SEGMENT
    )
    print(
        f"SEGPREFILL|item5|keys={sorted(fresh[0])}|"
        f"fresh_positions={[int(c['start_position']) for c in fresh]}|"
        f"continuing_positions={[int(c['start_position']) for c in continuing]}",
        flush=True,
    )
    assert all(int(c["start_position"]) == 0 for c in fresh), (
        f"a prefill at cached length 0 built {[int(c['start_position']) for c in fresh]}"
    )
    assert all(
        int(c["start_position"]) == ALIGNED_SEGMENT for c in continuing
    ), (
        f"a prefill at cached length {ALIGNED_SEGMENT} built "
        f"{[int(c['start_position']) for c in continuing]}"
    )

    hidden = stack.tokens[:ALIGNED_SEGMENT]
    out = hidden
    for layer, carrier in zip(stack.layers, continuing):
        out = layer(out, **carrier, chunk_size=DECLARED_CHUNK)
    print(f"SEGPREFILL|item5|splat_output_shape={tuple(out.shape)}", flush=True)
    assert tuple(out.shape) == tuple(hidden.shape)

    with pytest.raises(TypeError) as raised:
        stack.layers[0](
            hidden, **{**continuing[0], "not_a_layer_keyword": 0}
        )
    print(f"SEGPREFILL|item5|unknown_key_refusal={raised.value}", flush=True)
    assert "not_a_layer_keyword" in str(raised.value)


def test_kda_prefill_segments_c06_the_segmented_drive_took_the_kernel_route(
    stack: SimpleNamespace,
) -> None:
    """Item 6. The registered route predicate, form R-3, imported not re-written.

    Read around the two-segment drive: the kernel route is available, no seam
    this campaign owns fell back to torch, and the set that fired is not empty.
    """
    _report("item6_route_predicate", "the route the segmented drive took")
    seam_census._reset_seam_counters()
    before = seam_census._read_seam_counters()
    _segmented(stack, ALIGNED_SEGMENT, ALIGNED_TOKENS, carry=True, reset=False)
    after = seam_census._read_seam_counters()
    e2e_half._assert_route_predicate_r3("the two-segment prefill", before, after)
