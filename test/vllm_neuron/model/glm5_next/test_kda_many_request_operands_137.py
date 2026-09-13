# SPDX-License-Identifier: Apache-2.0
"""Acceptance for the row operands meeting a call that carries several requests.

Six items, no ``parametrize``. The layer serves a concurrent decode by recursing once per
request, and each request's recursion receives THAT request's own row operands. They
arrive stacked on a leading request axis, which is one axis more than one sequence's pair
carries and so is a form no one-sequence operand can be mistaken for; the runner's carrier
builder stacks them the same way at one request, so both shapes run one derivation.

The item that graded a refusal here is gone with the refusal it graded: a masked
recurrence over several requests is served now, so that expectation no longer exists. What
survives of it is the COUNT -- an operand set naming a different number of requests than
the states do is still refused -- and the last item below reads that refusal.

Two items are the served path's scope guards and are unchanged: a concurrent decode
WITHOUT the operands must still be served, and a single-request call WITH them must still
be served.

THE DECODE LEG IS THE ONLY LEG THESE ITEMS CAN READ. A prefill of several requests is
refused by the layer -- its packed rows say nothing about where one request's tokens end --
so several requests' masks never coexist on a prefill, and one request's own padded prefill
is the landed acceptance of the other branch of this function.

Run on the Tier N harness -- the NKI simulator on the host CPU, no device and no
lease::

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
    python -m pytest \
        test/vllm_neuron/model/glm5_next/test_kda_many_request_operands_137.py \
        -q -s -rA --timeout 900 -p no:randomly -p no:cacheprovider

2. a two-request decode WITHOUT the operands is served, and its output carries one row
    per request;
3. a single-request decode WITH the operands is served;
4. a two-request decode with per-request operands equals the two single-request decodes
    it is made of -- every output row, and both banks each request wrote;
5. each request's own mask governs its own state: with one request's single decode row
    marked padding, that request's conv and recurrent states are bit-identical before and
    after the step while the other's advance, and exchanging the two masks exchanges which
    request stood still;
6. an operand set naming a different number of requests than the states do is refused, and
    the message names that count rather than the concurrency;
7. the RUNNER's own carrier builder, at two requests, hands the layer one stacked pair
    per request -- entry by entry, in the states' order -- and the layer serves it.

Every declared value is CARRIED from the landed KDA layer acceptance, imported rather
than retyped.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from test.vllm_neuron.model.glm5_next import test_kda_layer as layer_half

#: Carried from the landed layer acceptance: this rank's head geometry, the conv
#: kernel width, the registered degree and the seed.
DECLARED_PER_RANK_HEADS = layer_half.DECLARED_PER_RANK_HEADS
DECLARED_KDA_HEAD_SIZE = layer_half.DECLARED_KDA_HEAD_SIZE
DECLARED_KDA_CONV_KERNEL_SIZE = layer_half.DECLARED_KDA_CONV_KERNEL_SIZE
REGISTERED_TP_WORLD_SIZE = layer_half.REGISTERED_TP_WORLD_SIZE
SEED = layer_half.SEED

#: Two requests, one token each: the shape a concurrent decode arrives in.
DECLARED_REQUESTS = 2

#: A position above zero, which is what makes a state a state the layer READS: at zero the
#: layer enters with zero whatever the slot holds, so a step asked to leave a state alone
#: has to be a step of a sequence already under way. Any positive value serves; this one is
#: the conv history width plus one, so the history the step convolves with is full.
DECLARED_CONTINUING_POSITION = DECLARED_KDA_CONV_KERNEL_SIZE


def _layer() -> SimpleNamespace:
    """One KDA layer with deterministic weights, and one token row per request."""
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    text_config = Glm5NextTextConfig()
    hidden = int(text_config.hidden_size)
    torch.manual_seed(SEED)
    weights = layer_half._make_weights(
        hidden,
        DECLARED_PER_RANK_HEADS,
        DECLARED_KDA_HEAD_SIZE,
        DECLARED_KDA_CONV_KERNEL_SIZE,
    )
    layer = layer_half._impl().Glm5NextKDALayer(
        text_config, 0, REGISTERED_TP_WORLD_SIZE
    )
    for name, tensor in weights.items():
        target = layer if name == "input_layernorm_weight" else layer.attention
        setattr(target, name, nn.Parameter(tensor.clone(), requires_grad=False))
    torch.manual_seed(SEED + 1)
    rows = torch.randn(DECLARED_REQUESTS, hidden, dtype=torch.float32)
    return SimpleNamespace(layer=layer, rows=rows, hidden=hidden)


@pytest.fixture(scope="module")
def one_layer() -> SimpleNamespace:
    """One built layer for every item; each drive takes its own carriers."""
    return _layer()


def _slots(layer, requests: int, *, position: int = 0):
    """One carrier pair per request, as views of one bank the way the runner hands them:
    two requests' states are two rows of one bank and cannot be one tensor without
    copying.

    A POSITION ABOVE ZERO FILLS THE SLOTS, and each request's fill is its own value, so a
    request served with another's state is visible. At zero the slots stay zeroed, which
    is what the layer enters an opening sequence with whatever the bank holds.
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
    # ONE ROW PER REQUEST AND ONE DIMENSION. The splitter unbinds a position tensor
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
    """The two row operands, BUILT BY THE RUNNER, or none when the tree has neither.

    Reading the runner's own builder rather than assembling a mask here keeps one
    authority for what the layers are handed.
    """
    runner = layer_half._runner_module().NeuronModelRunner
    builder = getattr(runner, "_glm5next_real_row_extent", None)
    accepted = inspect.signature(layer.attention.forward).parameters
    if builder is None or "row_mask" not in accepted or "real_tokens" not in accepted:
        return {}
    real_length, row_mask = builder(tokens, real, torch.device("cpu"))
    return {"real_tokens": real_length, "row_mask": row_mask}


def _per_request_operands(layer, reals) -> dict:
    """The two operands as the runner stacks them: one entry per request, one axis more.

    Each entry is the runner's own builder at this leg's per-request width -- one row,
    because a decode advances each sequence by one token -- and the stack is what the
    carrier builder hands the layer.
    """
    pairs = [_operands(layer, 1, int(real)) for real in reals]
    if not all(pairs):
        return {}
    return {
        name: torch.stack([pair[name] for pair in pairs])
        for name in ("real_tokens", "row_mask")
    }


#: The two requests' real lengths for the runner-built carrier. They are EQUAL, and one
#: row each, because that is the only reading a concurrent decode admits: several requests
#: bring one row apiece (``neuron_model_runner.py:5707``) and a request's real length is
#: bounded by that width, so a step where they differed would be refused rather than built.
DECLARED_TWO_REALS = (1, 1)

#: The two requests' positions. These DIFFER, which is the per-request value a concurrent
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
    """The carrier the runner's own builder hands a two-request decode.

    The per-request lengths are passed only where the signature takes them, so a tree
    that has no such keyword builds its own way and is then read on what it produced.
    """
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
        softmax_scale=float(DECLARED_KDA_HEAD_SIZE) ** -0.5,
        max_seq_len=int(DECLARED_TWO_POSITIONS[-1]) + DECLARED_REQUESTS,
        # The last two belong to the sparse family and are unread for a recurrent bank;
        # they are required keywords, so the call names them.
        index_kpool=1,
        requests=DECLARED_REQUESTS,
        request_starts=list(DECLARED_TWO_POSITIONS),
        **extra,
    )[0]


def _decode(layer, rows, requests: int, extras: dict, *, position: int = 0):
    """One concurrent-decode call: one token per request, each its own carrier.

    The two banks come back beside the output, and beside the copy they held BEFORE the
    call, because whose state advanced is not a question the output rows answer.
    """
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


def test_c02_a_many_request_decode_without_the_row_operands_is_still_served(one_layer):
    """The operands are optional, and their absence says every row carries a token."""
    out = _decode(one_layer.layer, one_layer.rows, DECLARED_REQUESTS, {}).out
    print(
        f"MANYREQ|c02|route=served|rows={int(out.shape[0])}|hidden={int(out.shape[1])}|"
        f"operands=none|requests={DECLARED_REQUESTS}"
    )
    assert tuple(out.shape) == (DECLARED_REQUESTS, one_layer.hidden)
    assert torch.isfinite(out).all()


def test_c03_a_single_request_decode_with_the_row_operands_is_still_served(one_layer):
    """One sequence's pair, in the form every landed caller passes it."""
    operands = _operands(one_layer.layer, 1, 1)
    assert operands, "the tree carries the row operands and the runner's builder"
    out = _decode(one_layer.layer, one_layer.rows[:1], 1, operands).out
    print(
        f"MANYREQ|c03|route=served|rows={int(out.shape[0])}|hidden={int(out.shape[1])}|"
        f"operands=both|requests=1|real_tokens={int(operands['real_tokens'])}"
    )
    assert tuple(out.shape) == (1, one_layer.hidden)
    assert torch.isfinite(out).all()


def test_c04_per_request_operands_equal_the_single_request_decodes_they_are_made_of(
    one_layer,
):
    """One request's answer is the answer that request gets on its own, to the bit."""
    layer = one_layer.layer
    operands = _per_request_operands(layer, [1] * DECLARED_REQUESTS)
    assert operands, "the tree carries the row operands and the runner's builder"
    print(
        f"MANYREQ|c04|real_tokens={tuple(operands['real_tokens'].shape)}|"
        f"row_mask={tuple(operands['row_mask'].shape)}|requests={DECLARED_REQUESTS}"
    )
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
        print(
            f"MANYREQ|c04|request={index}|out_gap={gaps[0]}|conv_gap={gaps[1]}|"
            f"recurrent_gap={gaps[2]}"
        )
        assert gaps == (0.0, 0.0, 0.0), (
            f"request {index} was served differently in the batch than on its own: "
            f"output, conv and recurrent gaps {gaps}. Each request's recursion carries "
            f"its own operands, so nothing about the batch may reach its arithmetic"
        )
    print(
        f"MANYREQ|c04|route=served|rows={int(together.out.shape[0])}|"
        f"hidden={int(together.out.shape[1])}|operands=per_request|"
        f"requests={DECLARED_REQUESTS}"
    )


def test_c05_each_requests_own_mask_governs_that_requests_own_state(one_layer):
    """A padding row holds ITS request's states still, to the bit, and only its own."""
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
        print(
            f"MANYREQ|c05|masked_request={masked}|"
            f"real_tokens={operands['real_tokens'].reshape(-1).tolist()}|"
            f"position={DECLARED_CONTINUING_POSITION}|unchanged={held}"
        )
        still[masked] = held
        assert held[masked] == (True, True), (
            f"request {masked}'s only row is padding, so both of its states must be the "
            f"bytes they were; conv and recurrent unchanged reads {held[masked]}"
        )
        for other in (index for index in range(DECLARED_REQUESTS) if index != masked):
            assert held[other] == (False, False), (
                f"request {other} carries a real token and its states {held[other]} did "
                f"not both move, so this item could not tell one request's mask from "
                f"another's"
            )
    print(f"MANYREQ|c05|exchanged={[still[key] for key in sorted(still)]}")
    print(
        f"MANYREQ|c05|route=served|rows={int(drive.out.shape[0])}|"
        f"hidden={int(drive.out.shape[1])}|operands=per_request|"
        f"requests={DECLARED_REQUESTS}"
    )


def test_c06_an_operand_set_naming_another_request_count_is_refused(one_layer):
    """One pair for two requests names no request, and the count is what says so."""
    layer = one_layer.layer
    operands = _operands(layer, DECLARED_REQUESTS, DECLARED_REQUESTS)
    assert operands, "the tree carries the row operands and the runner's builder"
    print(
        f"MANYREQ|c06|real_tokens={tuple(operands['real_tokens'].shape)}|"
        f"row_mask={tuple(operands['row_mask'].shape)}|entries=1|"
        f"requests={DECLARED_REQUESTS}"
    )
    with pytest.raises(ValueError) as refusal:
        _decode(layer, one_layer.rows, DECLARED_REQUESTS, operands)
    said = str(refusal.value)
    print(f"MANYREQ|c06|refusal={said}")
    # THE MESSAGE NAMES THE COUNT, which is what this item reads rather than the mere
    # raise: the refusal this one replaced named the concurrency, and both refuse the
    # same call.
    assert "one entry per request" in said, (
        f"the refusal must name the count it read; it said {said!r}"
    )
    assert "real_tokens" in said or "row_mask" in said
    assert str(DECLARED_REQUESTS) in said
    print(
        f"MANYREQ|c06|route=refused|rows=0|hidden={one_layer.hidden}|"
        f"operands=one_pair|requests={DECLARED_REQUESTS}"
    )


def test_c07_the_runner_builds_one_operand_pair_per_request_for_the_layer(one_layer):
    """The carrier the runner builds at two requests carries a stacked pair per request."""
    layer = one_layer.layer
    runner = layer_half._runner_module().NeuronModelRunner
    bank = _linear_bank(layer)
    before = SimpleNamespace(
        conv=bank["conv_state"].clone(), recurrent=bank["recurrent_state"].clone()
    )
    carrier = _runner_carrier(runner, bank)
    print(
        f"MANYREQ|c07|real_tokens={tuple(carrier['real_tokens'].shape)}|"
        f"row_mask={tuple(carrier['row_mask'].shape)}|"
        f"positions={[int(value) for value in carrier['start_position']]}|"
        f"requests={DECLARED_REQUESTS}"
    )
    # ONE ENTRY PER REQUEST ON A LEADING AXIS. The width is one row, because a decode
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
    # AND THE ENTRIES STAND IN THE STATES' ORDER, which is the batch's. Each request's
    # position is its own, and the positions DIFFER, so an axis assembled in another
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
    out = layer(one_layer.rows, **carrier)
    advanced = tuple(
        (
            not torch.equal(bank["conv_state"][one_slot], before.conv[one_slot]),
            not torch.equal(
                bank["recurrent_state"][one_slot], before.recurrent[one_slot]
            ),
        )
        for one_slot in DECLARED_TWO_SLOTS
    )
    print(f"MANYREQ|c07|advanced={list(advanced)}")
    assert advanced == ((True, True),) * DECLARED_REQUESTS
    print(
        f"MANYREQ|c07|route=served|rows={int(out.shape[0])}|"
        f"hidden={int(out.shape[1])}|operands=runner_built|"
        f"requests={DECLARED_REQUESTS}"
    )
