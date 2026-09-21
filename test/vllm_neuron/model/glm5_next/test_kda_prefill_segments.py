# SPDX-License-Identifier: Apache-2.0
"""A KDA prefill that continues a segmented prompt.

A prompt longer than one batch of tokens is prefilled in segments, so the second
segment has to enter the recurrence with the state the first left.
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

#: Carried from the layer acceptance: the comparator pair, the stack
#: depth, the chunk width, the registered parallel degree and the seed.
DECLARED_RTOL = layer_half.DECLARED_RTOL
DECLARED_ATOL = layer_half.DECLARED_ATOL
DECLARED_STACK_LAYERS = layer_half.DECLARED_STACK_LAYERS
DECLARED_CHUNK = layer_half.DECLARED_CHUNK
DECLARED_PER_RANK_HEADS = layer_half.DECLARED_PER_RANK_HEADS
KDA_HEAD_SIZE = layer_half.KDA_HEAD_SIZE
KDA_CONV_KERNEL_SIZE = layer_half.KDA_CONV_KERNEL_SIZE
TP_WORLD_SIZE = layer_half.TP_WORLD_SIZE
SEED = layer_half.SEED

#: The aligned split: two segments of one chunk each. Aligned because that is
#: what the serving shape produces -- every segment but the last is exactly the
#: batched-token bound, which is a whole number of chunks.
ALIGNED_SEGMENT = DECLARED_CHUNK
ALIGNED_TOKENS = ALIGNED_SEGMENT * 2

#: The sub-chunk split: a continuing segment below one chunk, so its recurrence
#: runs entirely on the single-token seam and takes no chunked dispatch at all.
SUB_FIRST = DECLARED_CHUNK * 2
SUB_SECOND = 3
SUB_TOKENS = SUB_FIRST + SUB_SECOND

#: The value planted in a slot's convolution history to stand for whatever the
#: previous owner of that slot left there. It is exact in the state's dtype, which
#: the test checks before it plants it: a value the dtype rounds away is stored as
#: something else, and a comparison against it would then say nothing.
DIRTY_CONV_ROW = 0.5

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
            hidden, heads, KDA_HEAD_SIZE, KDA_CONV_KERNEL_SIZE
        )
        for _ in range(DECLARED_STACK_LAYERS)
    ]
    torch.manual_seed(SEED + 1)
    tokens = torch.randn(max(ALIGNED_TOKENS, SUB_TOKENS), hidden, dtype=torch.float32)

    layers = []
    for index, layer_weights in enumerate(weights):
        layer = layer_half._impl().Glm5NextKDALayer(
            text_config, index, TP_WORLD_SIZE
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
    """One built stack for every test; the state lives in a per-drive bank."""
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


def _drive(
    layers,
    bank,
    tokens,
    *,
    start_position: int,
    reset: bool = True,
    is_prefill: bool = True,
):
    """One call through the whole stack, with its own counter reading. """
    if reset:
        layer_half._reset_counters()
    out = tokens
    for layer, (conv_state, recurrent_state) in zip(layers, bank):
        out = layer(
            out,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            is_prefill=is_prefill,
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


def test_two_segments_equal_the_one_shot_prefill(
    stack: SimpleNamespace,
) -> None:
    """Check 1. Certifying component: the prefill arm's entering state."""
    want, want_state, _ = _one_shot(stack, ALIGNED_TOKENS)
    got, got_state, _ = _segmented(stack, ALIGNED_SEGMENT, ALIGNED_TOKENS, carry=True)
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


def test_dropping_the_carry_misses_the_one_shot_run(
    stack: SimpleNamespace,
) -> None:
    """Check 2. The must-fail arm: check-1 passing must mean something."""
    want, _want_state, _ = _one_shot(stack, ALIGNED_TOKENS)
    bad, _bad_state, _ = _segmented(
        stack, ALIGNED_SEGMENT, ALIGNED_TOKENS, carry=False
    )
    outside = int(
        (
            (bad.float() - want.float()).abs()
            > DECLARED_ATOL + DECLARED_RTOL * want.float().abs()
        )
        .sum()
        .item()
    )
    assert outside > 0, (
        "a segment entered at position 0 reproduced the one-shot prefill within the "
        "declared pair, so check-1 grades nothing"
    )
    with pytest.raises(AssertionError):
        assert_close(
            bad.float(), want.float(),
            rtol=DECLARED_RTOL, atol=DECLARED_ATOL,
            name="no_carry.out",
        )


def test_a_segment_below_one_chunk_carries_too(
    stack: SimpleNamespace,
) -> None:
    """Check 3. Certifying component: the single-token seam's entering state."""
    want, _want_state, _ = _one_shot(stack, SUB_TOKENS)
    got, _got_state, counts = _segmented(stack, SUB_FIRST, SUB_TOKENS, carry=True)
    assert counts["inter"][0] == DECLARED_SUB_CHUNKED_DISPATCHES, (
        f"the continuing segment took {counts['inter'][0]} inter-chunk dispatch(es); "
        f"a segment below one chunk must take {DECLARED_SUB_CHUNKED_DISPATCHES}, or "
        f"this test is not measuring the walked path"
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
    with pytest.raises(AssertionError):
        assert_close(
            bad.float(), want.float(),
            rtol=DECLARED_RTOL, atol=DECLARED_ATOL,
            name="sub_chunk_no_carry.out",
        )


def test_a_fresh_prefill_ignores_the_bank(
    stack: SimpleNamespace,
) -> None:
    """Check 4. Certifying component: the gate on the entering state."""
    clean_out, dirty_source, _ = _one_shot(stack, ALIGNED_SEGMENT)

    dirty = _fresh_bank(stack.layers)
    for (conv, recurrent), source in zip(dirty, dirty_source):
        recurrent.copy_(source)
        planted = float(torch.tensor(DIRTY_CONV_ROW, dtype=conv.dtype))
        assert planted == DIRTY_CONV_ROW, (
            f"the planted history {DIRTY_CONV_ROW} is stored as {planted} in "
            f"{conv.dtype}, so this test would plant a value it cannot name"
        )
        conv.fill_(DIRTY_CONV_ROW)
    assert float(max(r.abs().max() for _, r in dirty)) > 0.0, (
        "the dirty bank is all zeros, so this test cannot tell the two routes apart"
    )
    assert float(min(c.abs().min() for c, _ in dirty)) > 0.0, (
        "the planted convolution history is zero somewhere, so a gate that read it "
        "would still produce the fresh answer there"
    )
    fresh_out, _ = _drive(
        stack.layers, dirty, stack.tokens[:ALIGNED_SEGMENT], start_position=0
    )
    identical = bool(torch.equal(fresh_out, clean_out))

    dirty_again = _fresh_bank(stack.layers)
    for (conv, recurrent), source in zip(dirty_again, dirty_source):
        recurrent.copy_(source)
        conv.fill_(DIRTY_CONV_ROW)
    continued_out, _ = _drive(
        stack.layers,
        dirty_again,
        stack.tokens[:ALIGNED_SEGMENT],
        start_position=ALIGNED_SEGMENT,
    )
    continued_identical = bool(torch.equal(continued_out, clean_out))
    assert identical, (
        "a prefill at position 0 read the bank; a fresh sequence starts the "
        "recurrence at zero whatever the slot holds"
    )
    assert not continued_identical, (
        "the same bank at a continuing position produced the fresh result, so the "
        "position is not gating the entering state at all"
    )


def test_a_fresh_prefill_leaves_its_own_state_in_the_bank(
    stack: SimpleNamespace,
) -> None:
    """Check 7. Certifying component: the state write-back on the fresh leg."""
    clean = _fresh_bank(stack.layers)
    _drive(stack.layers, clean, stack.tokens[:ALIGNED_SEGMENT], start_position=0)
    clean_after = [(conv.clone(), recurrent.clone()) for conv, recurrent in clean]

    dirty = _fresh_bank(stack.layers)
    for conv, recurrent in dirty:
        conv.fill_(DIRTY_CONV_ROW)
        recurrent.fill_(DIRTY_CONV_ROW)
    _drive(stack.layers, dirty, stack.tokens[:ALIGNED_SEGMENT], start_position=0)

    same = [
        bool(torch.equal(conv, want_conv)) and bool(torch.equal(rec, want_rec))
        for (conv, rec), (want_conv, want_rec) in zip(dirty, clean_after)
    ]
    moved = [
        bool((conv != DIRTY_CONV_ROW).any()) and bool(conv.abs().max() > 0.0)
        for conv, _ in clean_after
    ]
    if not all(moved):
        raise AssertionError(
            "a clean run left a convolution state that is the planted value or all "
            "zeros, so this test could pass on a bank nothing wrote"
        )
    assert all(same), (
        "after a fresh prefill over a dirty slot the bank does not hold what the same "
        "prefill over a clean slot left; the write-back went somewhere else"
    )


def test_a_one_token_step_at_position_zero_ignores_the_bank(
    stack: SimpleNamespace,
) -> None:
    """Check 8. Certifying component: the entering state on the decode leg."""
    clean = _fresh_bank(stack.layers)
    clean_out, _ = _drive(
        stack.layers, clean, stack.tokens[:1], start_position=0, is_prefill=False
    )

    dirty = _fresh_bank(stack.layers)
    for conv, recurrent in dirty:
        planted = float(torch.tensor(DIRTY_CONV_ROW, dtype=conv.dtype))
        assert planted == DIRTY_CONV_ROW, (
            f"the planted state {DIRTY_CONV_ROW} is stored as {planted} in "
            f"{conv.dtype}, so this test would plant a value it cannot name"
        )
        conv.fill_(DIRTY_CONV_ROW)
        recurrent.fill_(DIRTY_CONV_ROW)
    assert float(min(c.abs().min() for c, _ in dirty)) > 0.0, (
        "the planted convolution history is zero somewhere, so a gate that read it "
        "would still produce the fresh answer there"
    )
    assert float(min(r.abs().min() for _, r in dirty)) > 0.0, (
        "the planted recurrent state is zero somewhere, so a gate that read it "
        "would still produce the fresh answer there"
    )
    fresh_out, _ = _drive(
        stack.layers, dirty, stack.tokens[:1], start_position=0, is_prefill=False
    )
    identical = bool(torch.equal(fresh_out, clean_out))

    dirty_again = _fresh_bank(stack.layers)
    for conv, recurrent in dirty_again:
        conv.fill_(DIRTY_CONV_ROW)
        recurrent.fill_(DIRTY_CONV_ROW)
    continued_out, _ = _drive(
        stack.layers,
        dirty_again,
        stack.tokens[:1],
        start_position=ALIGNED_SEGMENT,
        is_prefill=False,
    )
    continued_identical = bool(torch.equal(continued_out, clean_out))
    assert identical, (
        "a one-token step at position 0 read the bank; a sequence that has computed "
        "nothing enters at zero whatever leg it arrives on"
    )
    assert not continued_identical, (
        "the same bank at a continuing position produced the fresh result, so the "
        "position is not gating the entering state on this leg at all"
    )


def test_the_runner_hands_the_position_to_the_layer(
    stack: SimpleNamespace,
) -> None:
    """Check 5. Certifying component: the runner's linear carrier."""
    banks = runner_half._banks(stack.layers)
    runner = runner_half._runner(stack.text_config, banks)
    runner.max_model_len = SUB_TOKENS + ALIGNED_TOKENS + 1

    # The order is load-bearing. The converter refuses a step that does not
    # continue the sequence its cursor names, and a prefill at position 0 is what
    # opens one (``neuron_model_runner.py``), so the fresh
    # call has to come first for the continuing call to be servable at all.
    fresh = runner_half._carriers(runner, banks, tokens=ALIGNED_SEGMENT, cached=0)
    continuing = runner_half._carriers(
        runner, banks, tokens=ALIGNED_SEGMENT, cached=ALIGNED_SEGMENT
    )
    assert all(int(c["start_position"][0]) == 0 for c in fresh), (
        f"a prefill at cached length 0 built "
        f"{[int(c['start_position'][0]) for c in fresh]}"
    )
    assert all(
        int(c["start_position"][0]) == ALIGNED_SEGMENT for c in continuing
    ), (
        f"a prefill at cached length {ALIGNED_SEGMENT} built "
        f"{[int(c['start_position'][0]) for c in continuing]}"
    )

    hidden = stack.tokens[:ALIGNED_SEGMENT]
    out = hidden
    for layer, carrier in zip(stack.layers, continuing):
        out = layer(out, **carrier, chunk_size=DECLARED_CHUNK)
    assert tuple(out.shape) == tuple(hidden.shape)

    with pytest.raises(TypeError) as raised:
        stack.layers[0](
            hidden, **{**continuing[0], "not_a_layer_keyword": 0}
        )
    assert "not_a_layer_keyword" in str(raised.value)


def test_the_segmented_drive_took_the_kernel_route(
    stack: SimpleNamespace,
) -> None:
    """Check 6. The registered route predicate, form R-3, imported not re-written."""
    seam_census._reset_seam_counters()
    before = seam_census._read_seam_counters()
    _segmented(stack, ALIGNED_SEGMENT, ALIGNED_TOKENS, carry=True, reset=False)
    after = seam_census._read_seam_counters()
    e2e_half._assert_route_predicate_r3("the two-segment prefill", before, after)
