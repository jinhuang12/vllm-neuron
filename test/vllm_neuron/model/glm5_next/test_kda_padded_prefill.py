# SPDX-License-Identifier: Apache-2.0
"""A KDA prefill that arrives padded to its bucket.

The state it leaves behind has to be the state the same prompt leaves unpadded: the
recurrence must neither decay nor update on a padding row, and the convolution's
history must come from the last real rows rather than from the bucket's tail.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from test.vllm_neuron.model.glm5_next import test_kda_layer as layer_half

from vllm_neuron.accuracy.testing import assert_close

#: Carried from the layer acceptance: the comparator pair, the stack depth,
#: the chunk width, this rank's head geometry, the registered degree and the seed.
DECLARED_RTOL = layer_half.DECLARED_RTOL
DECLARED_ATOL = layer_half.DECLARED_ATOL
DECLARED_STACK_LAYERS = layer_half.DECLARED_STACK_LAYERS
DECLARED_CHUNK = layer_half.DECLARED_CHUNK
DECLARED_PER_RANK_HEADS = layer_half.DECLARED_PER_RANK_HEADS
KDA_HEAD_SIZE = layer_half.KDA_HEAD_SIZE
KDA_CONV_KERNEL_SIZE = layer_half.KDA_CONV_KERNEL_SIZE
TP_WORLD_SIZE = layer_half.TP_WORLD_SIZE
SEED = layer_half.SEED

#: The width both arms' padded operands take: four whole chunks, so the aligned and
#: the ragged real length both sit inside one bucket.
DECLARED_BUCKET = 4 * DECLARED_CHUNK

#: Check 1's real length. A whole number of chunks, so the real rows take the same
#: route on both arms and the two carriers can be asked for bit equality.
DECLARED_ALIGNED_REAL = 2 * DECLARED_CHUNK

#: Check 2's real length. Not a whole number of chunks: the bucket is chunk-aligned,
#: so the padded arm carries the tail rows through the chunked seams while the
#: unpadded arm walks them one at a time. The same value by two accumulation orders
#: agrees to a tolerance and not to the bit, which is what check-2 measures.
DECLARED_RAGGED_REAL = 2 * DECLARED_CHUNK + 3

#: The value every padding row carries. Exactly representable in bfloat16, and far
#: outside the unit-ish range the real rows are drawn from, so a carrier that moved
#: because of a padding row moves visibly rather than in the last bits.
DECLARED_PADDING_MARKER = 64.0


def _stack() -> SimpleNamespace:
    """Three KDA layers with deterministic weights, and one bucket of token rows."""
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    text_config = Glm5NextTextConfig()
    hidden = int(text_config.hidden_size)
    torch.manual_seed(SEED)
    weights = [
        layer_half._make_weights(
            hidden,
            DECLARED_PER_RANK_HEADS,
            KDA_HEAD_SIZE,
            KDA_CONV_KERNEL_SIZE,
        )
        for _ in range(DECLARED_STACK_LAYERS)
    ]
    # One row past the bucket: the token the decode step of check-3 is handed.
    torch.manual_seed(SEED + 1)
    tokens = torch.randn(DECLARED_BUCKET + 1, hidden, dtype=torch.float32)

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
        text_config=text_config, layers=layers, tokens=tokens, hidden=hidden
    )


@pytest.fixture(scope="module")
def stack() -> SimpleNamespace:
    """One built stack for every test; each drive takes its own bank."""
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


def _extras(layer, tokens: int, real: int) -> dict:
    """The two row operands, built by the runner, or none when the tree has neither. """
    runner = layer_half._runner_module().NeuronModelRunner
    builder = getattr(runner, "_glm5next_real_row_extent", None)
    accepted = inspect.signature(layer.attention.forward).parameters
    if builder is None or "row_mask" not in accepted or "real_tokens" not in accepted:
        return {}
    real_length, row_mask = builder(tokens, real, torch.device("cpu"))
    return {"real_tokens": real_length, "row_mask": row_mask}


def _drive(
    layers, bank, rows, *, real: int, is_prefill: bool, start_position: int,
    operands: bool = True,
):
    """One call per layer through the stack, each layer handed its own carrier. """
    out = rows
    for layer, (conv_state, recurrent_state) in zip(layers, bank):
        out = layer(
            out,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            is_prefill=is_prefill,
            start_position=start_position,
            chunk_size=DECLARED_CHUNK,
            **(_extras(layer, int(rows.shape[0]), real) if operands else {}),
        )
    return out


def _padded_rows(stack: SimpleNamespace, real: int) -> torch.Tensor:
    """One bucket wide: the request's own rows, then marker rows to the bucket."""
    rows = torch.full(
        (DECLARED_BUCKET, stack.hidden), DECLARED_PADDING_MARKER, dtype=torch.float32
    )
    rows[:real] = stack.tokens[:real]
    return rows


def _arms(stack: SimpleNamespace, real: int) -> SimpleNamespace:
    """The same request prefilled unpadded and padded, each over its own bank."""
    want_bank = _fresh_bank(stack.layers)
    want_out = _drive(
        stack.layers,
        want_bank,
        stack.tokens[:real],
        real=real,
        is_prefill=True,
        start_position=0,
    )
    got_bank = _fresh_bank(stack.layers)
    got_out = _drive(
        stack.layers,
        got_bank,
        _padded_rows(stack, real),
        real=real,
        is_prefill=True,
        start_position=0,
    )
    return SimpleNamespace(
        want_out=want_out, want_bank=want_bank, got_out=got_out, got_bank=got_bank
    )


def _worst(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return (actual.float() - expected.float()).abs().max().item()


def _bound(expected: torch.Tensor) -> float:
    """Eight steps of this carrier's own dtype, at the scale its values sit on. """
    scale = max(1.0, expected.float().abs().max().item())
    return 8.0 * torch.finfo(expected.dtype).eps * scale


def _worst_pair(got_bank, want_bank) -> tuple[float, float]:
    """The worst conv error and the worst recurrent error over every layer."""
    return (
        max(_worst(got[0], want[0]) for got, want in zip(got_bank, want_bank)),
        max(_worst(got[1], want[1]) for got, want in zip(got_bank, want_bank)),
    )


def _masked(stack: SimpleNamespace) -> bool:
    """Whether this tree takes the row operands at all; read by every test."""
    return bool(_extras(stack.layers[0], DECLARED_BUCKET, DECLARED_ALIGNED_REAL))


def test_a_chunk_multiple_length_leaves_the_same_state(
    stack: SimpleNamespace,
) -> None:
    """Check 1. Certifying component: which rows the recurrence and the history read."""
    real = DECLARED_ALIGNED_REAL
    marker_hits = int((stack.tokens[:real] == DECLARED_PADDING_MARKER).sum())
    assert marker_hits == 0, (
        f"{marker_hits} value(s) of the request's own rows hold the marker "
        f"{DECLARED_PADDING_MARKER}, so a state moved by a padding row could not be "
        f"told from one moved by a real row"
    )

    arms = _arms(stack, real)
    multiples = {"conv": 0.0, "state": 0.0}
    for index, (got, want) in enumerate(zip(arms.got_bank, arms.want_bank)):
        for half, half_name in ((0, "conv"), (1, "state")):
            bound = _bound(want[half])
            error = _worst(got[half], want[half])
            multiples[half_name] = max(multiples[half_name], error / bound)
    for index, (got, want) in enumerate(zip(arms.got_bank, arms.want_bank)):
        for half, half_name, what in (
            (0, "conv", "the history the next step convolves with is the bucket's tail"),
            (1, "state", "the padding rows entered the sequence's own state"),
        ):
            bound = _bound(want[half])
            error = _worst(got[half], want[half])
            assert error <= bound, (
                f"layer {index}'s {half_name} carrier differs between the padded and "
                f"the unpadded prefill of the same {real} tokens by {error:.3e}, which "
                f"is {error / bound:.3f} times the {bound:.3e} eight steps of "
                f"{want[half].dtype} allow at this carrier's scale; {what}"
            )


def test_a_ragged_length_carries_to_the_declared_comparator(
    stack: SimpleNamespace,
) -> None:
    """Check 2. The common case: a real length that is not a whole number of chunks."""
    real = DECLARED_RAGGED_REAL
    arms = _arms(stack, real)
    for index, (got, want) in enumerate(zip(arms.got_bank, arms.want_bank)):
        assert_close(
            got[0].float(), want[0].float(),
            rtol=DECLARED_RTOL, atol=DECLARED_ATOL,
            name=f"padded.conv_state[{index}]",
        )
        assert_close(
            got[1].float(), want[1].float(),
            rtol=DECLARED_RTOL, atol=DECLARED_ATOL,
            name=f"padded.recurrent_state[{index}]",
        )


def test_the_first_decode_after_it_equals_the_unpadded_arms(
    stack: SimpleNamespace,
) -> None:
    """Check 3. What the request actually feels: its next token."""
    real = DECLARED_ALIGNED_REAL
    arms = _arms(stack, real)
    step = stack.tokens[DECLARED_BUCKET : DECLARED_BUCKET + 1]
    want = _drive(
        stack.layers, arms.want_bank, step, real=1, is_prefill=False, start_position=real
    )
    got = _drive(
        stack.layers, arms.got_bank, step, real=1, is_prefill=False, start_position=real
    )
    assert_close(
        got.float(), want.float(),
        rtol=DECLARED_RTOL, atol=DECLARED_ATOL,
        name="padded.decode_out",
    )
    for index, (after, expected) in enumerate(zip(arms.got_bank, arms.want_bank)):
        assert_close(
            after[0].float(), expected[0].float(),
            rtol=DECLARED_RTOL, atol=DECLARED_ATOL,
            name=f"padded.decode_conv_state[{index}]",
        )
        assert_close(
            after[1].float(), expected[1].float(),
            rtol=DECLARED_RTOL, atol=DECLARED_ATOL,
            name=f"padded.decode_recurrent_state[{index}]",
        )


def test_the_row_operands_change_nothing_when_no_row_is_padding(
    stack: SimpleNamespace,
) -> None:
    """Check 4. Certifying component: that the masked route is exactly the old route."""
    real = DECLARED_BUCKET
    rows = _padded_rows(stack, real)
    assert bool(torch.equal(rows, stack.tokens[:real])), (
        "at the bucket's own length no row is padding, so these rows must be the "
        "request's own; a marker row here would make this test a padded comparison"
    )
    assert _masked(stack), (
        "this tree takes neither row operand, so both arms below are the same call and "
        "the test could not tell the masked route from the route it replaces"
    )

    want_bank = _fresh_bank(stack.layers)
    _drive(
        stack.layers, want_bank, rows, real=real, is_prefill=True, start_position=0,
        operands=False,
    )
    got_bank = _fresh_bank(stack.layers)
    _drive(
        stack.layers, got_bank, rows, real=real, is_prefill=True, start_position=0,
        operands=True,
    )
    worst_conv, worst_state = _worst_pair(got_bank, want_bank)
    for index, (got, want) in enumerate(zip(got_bank, want_bank)):
        assert torch.equal(got[0], want[0]), (
            f"layer {index}'s conv history moved by at most {worst_conv:.3e} when the "
            f"row operands were handed to a step with no padding row, so masking by "
            f"one is not the identity the design rests on"
        )
        assert torch.equal(got[1], want[1]), (
            f"layer {index}'s recurrent state moved by at most {worst_state:.3e} when "
            f"the row operands were handed to a step with no padding row, so masking "
            f"by one is not the identity the design rests on"
        )
