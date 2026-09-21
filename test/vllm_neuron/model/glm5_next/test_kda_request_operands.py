# SPDX-License-Identifier: Apache-2.0
"""The per-request row operands meeting a call that carries several requests.

Each request's own mask has to govern that request's own state, and an operand set
naming another request count has to refuse.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from test.vllm_neuron.model.glm5_next import test_kda_layer as layer_half

#: Carried from the layer acceptance: this rank's head geometry, the conv
#: kernel width, the registered degree and the seed.
DECLARED_PER_RANK_HEADS = layer_half.DECLARED_PER_RANK_HEADS
KDA_HEAD_SIZE = layer_half.KDA_HEAD_SIZE
KDA_CONV_KERNEL_SIZE = layer_half.KDA_CONV_KERNEL_SIZE
TP_WORLD_SIZE = layer_half.TP_WORLD_SIZE
SEED = layer_half.SEED

#: Two requests, one token each: the shape a concurrent decode arrives in.
DECLARED_REQUESTS = 2

#: A position above zero, which is what makes a state a state the layer reads: at zero the
#: layer enters with zero whatever the slot holds, so a step asked to leave a state alone
#: has to be a step of a sequence already under way. Any positive value serves; this one is
#: the conv history width plus one, so the history the step convolves with is full.
DECLARED_CONTINUING_POSITION = KDA_CONV_KERNEL_SIZE


def _layer() -> SimpleNamespace:
    """One KDA layer with deterministic weights, and one token row per request."""
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    text_config = Glm5NextTextConfig()
    hidden = int(text_config.hidden_size)
    torch.manual_seed(SEED)
    weights = layer_half._make_weights(
        hidden,
        DECLARED_PER_RANK_HEADS,
        KDA_HEAD_SIZE,
        KDA_CONV_KERNEL_SIZE,
    )
    layer = layer_half._impl().Glm5NextKDALayer(
        text_config, 0, TP_WORLD_SIZE
    )
    for name, tensor in weights.items():
        target = layer if name == "input_layernorm_weight" else layer.attention
        setattr(target, name, nn.Parameter(tensor.clone(), requires_grad=False))
    torch.manual_seed(SEED + 1)
    rows = torch.randn(DECLARED_REQUESTS, hidden, dtype=torch.float32)
    return SimpleNamespace(layer=layer, rows=rows, hidden=hidden)


@pytest.fixture(scope="module")
def one_layer() -> SimpleNamespace:
    """One built layer for every test; each drive takes its own carriers."""
    return _layer()


def _slots(layer, requests: int, *, position: int = 0):
    """One carrier pair per request, as views of one bank the way the runner hands them:
    two requests' states are two rows of one bank and cannot be one tensor without
    copying.
    """
    attention = layer.attention
    conv_bank = torch.zeros(
        (requests, *attention.kda_conv_state_shape),
        dtype=attention.kda_conv_state_dtype,
    )
    recurrent_bank = torch.zeros(
        (requests, *attention.kda_recurrent_state_shape),
        dtype=attention.kda_recurrent_state_dtype,
    )
    if int(position) > 0:
        for index in range(requests):
            conv_bank[index].fill_(float(index + 1))
            recurrent_bank[index].fill_(float(index + 1) / 8.0)
    # One row per request and one dimension. The splitter unbinds a position tensor
    # only when it is 1-D with more than one row; a `[requests, 1]` tensor stays whole
    # and the call is refused for describing unequal numbers of requests instead.
    return (
        tuple(conv_bank[index] for index in range(requests)),
        tuple(recurrent_bank[index] for index in range(requests)),
        torch.full((requests,), int(position), dtype=torch.int32),
        conv_bank,
        recurrent_bank,
    )


def _operands(layer, tokens: int, real: int) -> dict:
    """The two row operands, built by the runner, or none when the tree has neither. """
    runner = layer_half._runner_module().NeuronModelRunner
    builder = getattr(runner, "_glm5next_real_row_extent", None)
    accepted = inspect.signature(layer.attention.forward).parameters
    if builder is None or "row_mask" not in accepted or "real_tokens" not in accepted:
        return {}
    real_length, row_mask = builder(tokens, real, torch.device("cpu"))
    return {"real_tokens": real_length, "row_mask": row_mask}


def _per_request_operands(layer, reals) -> dict:
    """The two operands as the runner stacks them: one entry per request, one axis more.
    """
    pairs = [_operands(layer, 1, int(real)) for real in reals]
    if not all(pairs):
        return {}
    return {
        name: torch.stack([pair[name] for pair in pairs])
        for name in ("real_tokens", "row_mask")
    }


#: The two requests' real lengths for the runner-built carrier. They are equal, and one
#: row each, because that is the only reading a concurrent decode admits: several requests
#: bring one row apiece (``neuron_model_runner.py``) and a request's real length is
#: bounded by that width, so a step where they differed would be refused rather than built.
DECLARED_TWO_REALS = (1, 1)

#: The two requests' positions. These differ, which is the per-request value a concurrent
#: decode can differ in, so an axis assembled in another order shows up here.
DECLARED_TWO_POSITIONS = (
    DECLARED_CONTINUING_POSITION,
    DECLARED_CONTINUING_POSITION + 1,
)

#: The recurrent bank's slot count, wider than the request count so a slot number cannot
#: pass for a request index, and the slots the two requests hold -- not in slot order.
DECLARED_STATE_SLOTS = 3
DECLARED_TWO_SLOTS = (2, 0)


def _linear_bank(layer) -> dict:
    """One recurrent bank, built to the layer's own declared state shapes and filled."""
    attention = layer.attention
    bank = {
        "name": "model.layers.0.attention",
        "layer_index": 0,
        "family": "linear_attn",
        "state_slots": DECLARED_STATE_SLOTS,
        "conv_state": torch.zeros(
            (DECLARED_STATE_SLOTS, *attention.kda_conv_state_shape),
            dtype=attention.kda_conv_state_dtype,
        ),
        "recurrent_state": torch.zeros(
            (DECLARED_STATE_SLOTS, *attention.kda_recurrent_state_shape),
            dtype=attention.kda_recurrent_state_dtype,
        ),
    }
    for index, one_slot in enumerate(DECLARED_TWO_SLOTS):
        bank["conv_state"][one_slot].fill_(float(index + 1))
        bank["recurrent_state"][one_slot].fill_(float(index + 1) / 8.0)
    return bank


def _runner_carrier(runner, bank) -> dict:
    """The carrier the runner's own builder hands a two-request decode. """
    geometry = {
        "state_slot": DECLARED_TWO_SLOTS[0],
        "state_slots": list(DECLARED_TWO_SLOTS),
    }
    extra = {}
    accepted = inspect.signature(runner._glm5next_layer_carriers).parameters
    if "request_real_tokens" in accepted:
        extra["request_real_tokens"] = list(DECLARED_TWO_REALS)
    return runner._glm5next_layer_carriers(
        [bank],
        [None],
        geometries=[geometry],
        is_prefill=False,
        tokens=DECLARED_REQUESTS,
        start_position=DECLARED_TWO_POSITIONS[0],
        softmax_scale=float(KDA_HEAD_SIZE) ** -0.5,
        max_seq_len=int(DECLARED_TWO_POSITIONS[-1]) + DECLARED_REQUESTS,
        # The last two belong to the sparse family and are unread for a recurrent bank;
        # they are required keywords, so the call names them.
        index_kpool=1,
        requests=DECLARED_REQUESTS,
        request_starts=list(DECLARED_TWO_POSITIONS),
        **extra,
    )[0]


def _decode(layer, rows, requests: int, extras: dict, *, position: int = 0):
    """One concurrent-decode call: one token per request, each its own carrier. """
    conv_state, recurrent_state, start_position, conv_bank, recurrent_bank = _slots(
        layer, requests, position=position
    )
    before = SimpleNamespace(conv=conv_bank.clone(), recurrent=recurrent_bank.clone())
    out = layer(
        rows,
        conv_state=conv_state if requests > 1 else conv_state[0],
        recurrent_state=recurrent_state if requests > 1 else recurrent_state[0],
        is_prefill=False,
        start_position=start_position,
        **extras,
    )
    return SimpleNamespace(
        out=out, conv=conv_bank, recurrent=recurrent_bank, before=before
    )


def test_a_many_request_decode_without_the_row_operands_is_still_served(one_layer):
    """The operands are optional, and their absence says every row carries a token."""
    out = _decode(one_layer.layer, one_layer.rows, DECLARED_REQUESTS, {}).out
    assert tuple(out.shape) == (DECLARED_REQUESTS, one_layer.hidden)
    assert torch.isfinite(out).all()


def test_a_single_request_decode_with_the_row_operands_is_still_served(one_layer):
    """One sequence's pair, in the form every caller passes it."""
    operands = _operands(one_layer.layer, 1, 1)
    assert operands, "the tree carries the row operands and the runner's builder"
    out = _decode(one_layer.layer, one_layer.rows[:1], 1, operands).out
    assert tuple(out.shape) == (1, one_layer.hidden)
    assert torch.isfinite(out).all()


def test_per_request_operands_equal_the_single_request_decodes_they_are_made_of(
    one_layer,
):
    """One request's answer is the answer that request gets on its own, to the bit."""
    layer = one_layer.layer
    operands = _per_request_operands(layer, [1] * DECLARED_REQUESTS)
    assert operands, "the tree carries the row operands and the runner's builder"
    together = _decode(layer, one_layer.rows, DECLARED_REQUESTS, operands)
    assert tuple(together.out.shape) == (DECLARED_REQUESTS, one_layer.hidden)
    for index in range(DECLARED_REQUESTS):
        alone = _decode(
            layer,
            one_layer.rows[index : index + 1],
            1,
            _operands(layer, 1, 1),
        )
        gaps = (
            float((together.out[index : index + 1] - alone.out).abs().max()),
            float((together.conv[index] - alone.conv[0]).abs().max()),
            float((together.recurrent[index] - alone.recurrent[0]).abs().max()),
        )
        assert gaps == (0.0, 0.0, 0.0), (
            f"request {index} was served differently in the batch than on its own: "
            f"output, conv and recurrent gaps {gaps}. Each request's recursion carries "
            f"its own operands, so nothing about the batch may reach its arithmetic"
        )


def test_each_requests_own_mask_governs_that_requests_own_state(one_layer):
    """A padding row holds its request's states still, to the bit, and only its own."""
    layer = one_layer.layer
    still = {}
    for masked in range(DECLARED_REQUESTS):
        reals = [1] * DECLARED_REQUESTS
        reals[masked] = 0
        operands = _per_request_operands(layer, reals)
        assert operands, "the tree carries the row operands and the runner's builder"
        drive = _decode(
            layer,
            one_layer.rows,
            DECLARED_REQUESTS,
            operands,
            position=DECLARED_CONTINUING_POSITION,
        )
        held = [
            (
                torch.equal(drive.conv[index], drive.before.conv[index]),
                torch.equal(drive.recurrent[index], drive.before.recurrent[index]),
            )
            for index in range(DECLARED_REQUESTS)
        ]
        still[masked] = held
        assert held[masked] == (True, True), (
            f"request {masked}'s only row is padding, so both of its states must be the "
            f"bytes they were; conv and recurrent unchanged reads {held[masked]}"
        )
        for other in (index for index in range(DECLARED_REQUESTS) if index != masked):
            assert held[other] == (False, False), (
                f"request {other} carries a real token and its states {held[other]} did "
                f"not both move, so this test could not tell one request's mask from "
                f"another's"
            )


def test_an_operand_set_naming_another_request_count_is_refused(one_layer):
    """One pair for two requests names no request, and the count is what says so."""
    layer = one_layer.layer
    operands = _operands(layer, DECLARED_REQUESTS, DECLARED_REQUESTS)
    assert operands, "the tree carries the row operands and the runner's builder"
    with pytest.raises(ValueError) as refusal:
        _decode(layer, one_layer.rows, DECLARED_REQUESTS, operands)
    said = str(refusal.value)
    # The message names the count, which is what this test reads rather than the mere
    # raise: the refusal this one replaced named the concurrency, and both refuse the
    # same call.
    assert "one entry per request" in said, (
        f"the refusal must name the count it read; it said {said!r}"
    )
    assert "real_tokens" in said or "row_mask" in said
    assert str(DECLARED_REQUESTS) in said


def test_the_runner_builds_one_operand_pair_per_request_for_the_layer(one_layer):
    """The carrier the runner builds at two requests carries a stacked pair per request."""
    layer = one_layer.layer
    runner = layer_half._runner_module().NeuronModelRunner
    bank = _linear_bank(layer)
    before = SimpleNamespace(
        conv=bank["conv_state"].clone(), recurrent=bank["recurrent_state"].clone()
    )
    carrier = _runner_carrier(runner, bank)
    # One entry per request on a leading axis. The width is one row, because a decode
    # advances each sequence by one token, so a pair built from the batch's own reading
    # carries the request axis and nothing else.
    assert tuple(carrier["real_tokens"].shape) == (DECLARED_REQUESTS, 1)
    assert tuple(carrier["row_mask"].shape) == (DECLARED_REQUESTS, 1, 1)
    for index, one_real in enumerate(DECLARED_TWO_REALS):
        length, mask = runner._glm5next_real_row_extent(
            1, one_real, bank["recurrent_state"].device
        )
        assert torch.equal(carrier["real_tokens"][index], length)
        assert torch.equal(carrier["row_mask"][index], mask)
    # And the entries stand in the states' order, which is the batch's. Each request's
    # position is its own, and the positions differ, so an axis assembled in another
    # order is visible here rather than in the numbers a later step produces.
    assert len(carrier["conv_state"]) == DECLARED_REQUESTS
    assert len(carrier["recurrent_state"]) == DECLARED_REQUESTS
    assert [int(value) for value in carrier["start_position"]] == list(
        DECLARED_TWO_POSITIONS
    )
    for index, one_slot in enumerate(DECLARED_TWO_SLOTS):
        assert carrier["conv_state"][index].data_ptr() == (
            bank["conv_state"][one_slot].data_ptr()
        )
    # The layer runs on the carrier, because what the next block reads is whether each
    # request's own state moved -- which only a real step can move.
    layer(one_layer.rows, **carrier)
    advanced = tuple(
        (
            not torch.equal(bank["conv_state"][one_slot], before.conv[one_slot]),
            not torch.equal(
                bank["recurrent_state"][one_slot], before.recurrent[one_slot]
            ),
        )
        for one_slot in DECLARED_TWO_SLOTS
    )
    assert advanced == ((True, True),) * DECLARED_REQUESTS
