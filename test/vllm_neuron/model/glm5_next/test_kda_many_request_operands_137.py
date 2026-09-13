# SPDX-License-Identifier: Apache-2.0
"""Acceptance for the row operands meeting a call that carries several requests.

Four items, no ``parametrize``. The layer serves a concurrent decode by recursing once per
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
5. each request's own mask is the one applied to IT: masking one request's only row
    leaves that request's state where it found it and advances the other's, and
    exchanging the two masks exchanges which request stood still. One pair for two
    requests names no request, and is refused.

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


def _slots(layer, requests: int):
    """One zeroed carrier pair per request, as views of one bank the way the runner
    hands them: two requests' states are two rows of one bank and cannot be one
    tensor without copying."""
    attention = layer.attention
    conv_bank = torch.zeros(
        (requests, *attention.kda_conv_state_shape),
        dtype=attention.kda_conv_state_dtype,
    )
    recurrent_bank = torch.zeros(
        (requests, *attention.kda_recurrent_state_shape),
        dtype=attention.kda_recurrent_state_dtype,
    )
    # ONE ROW PER REQUEST AND ONE DIMENSION. The splitter unbinds a position tensor
    # only when it is 1-D with more than one row; a `[requests, 1]` tensor stays whole
    # and the call is refused for describing unequal numbers of requests instead.
    return (
        tuple(conv_bank[index] for index in range(requests)),
        tuple(recurrent_bank[index] for index in range(requests)),
        torch.zeros((requests,), dtype=torch.int32),
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


def _decode(layer, rows, requests: int, extras: dict):
    """One concurrent-decode call: one token per request, each its own carrier.

    The two banks come back beside the output, because whose state advanced is not a
    question the output rows answer.
    """
    conv_state, recurrent_state, start_position, conv_bank, recurrent_bank = _slots(
        layer, requests
    )
    out = layer(
        rows,
        conv_state=conv_state if requests > 1 else conv_state[0],
        recurrent_state=recurrent_state if requests > 1 else recurrent_state[0],
        is_prefill=False,
        start_position=start_position,
        **extras,
    )
    return SimpleNamespace(out=out, conv=conv_bank, recurrent=recurrent_bank)


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


def test_c05_each_requests_own_mask_is_the_one_applied_to_that_request(one_layer):
    """A padding row holds ITS request's state still, and only that request's."""
    layer = one_layer.layer
    stood_still = {}
    for masked in range(DECLARED_REQUESTS):
        reals = [1] * DECLARED_REQUESTS
        reals[masked] = 0
        operands = _per_request_operands(layer, reals)
        assert operands, "the tree carries the row operands and the runner's builder"
        drive = _decode(layer, one_layer.rows, DECLARED_REQUESTS, operands)
        moved = [
            (
                float(drive.conv[index].abs().max()),
                float(drive.recurrent[index].abs().max()),
            )
            for index in range(DECLARED_REQUESTS)
        ]
        print(
            f"MANYREQ|c05|masked_request={masked}|"
            f"real_tokens={operands['real_tokens'].reshape(-1).tolist()}|moved={moved}"
        )
        stood_still[masked] = moved
        assert moved[masked] == (0.0, 0.0), (
            f"request {masked}'s only row is padding, so both of its states must stand "
            f"where they were; they moved to {moved[masked]}"
        )
        for other in (
            index for index in range(DECLARED_REQUESTS) if index != masked
        ):
            assert moved[other][1] > 0.0, (
                f"request {other} carries a real token and its recurrent state did not "
                f"move, so this item could not tell one request's mask from another's"
            )
    print(f"MANYREQ|c05|exchanged={[stood_still[key] for key in sorted(stood_still)]}")
    # ONE PAIR FOR TWO REQUESTS NAMES NO REQUEST, and the count is what says so.
    with pytest.raises(ValueError) as refusal:
        _decode(
            layer,
            one_layer.rows,
            DECLARED_REQUESTS,
            _operands(layer, DECLARED_REQUESTS, DECLARED_REQUESTS),
        )
    said = str(refusal.value)
    print(f"MANYREQ|c05|refusal={said}")
    assert "real_tokens" in said or "row_mask" in said
    assert str(DECLARED_REQUESTS) in said
    print(
        f"MANYREQ|c05|route=served|rows={int(drive.out.shape[0])}|"
        f"hidden={int(drive.out.shape[1])}|operands=per_request|"
        f"requests={DECLARED_REQUESTS}"
    )
