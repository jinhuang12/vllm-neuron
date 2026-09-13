# SPDX-License-Identifier: Apache-2.0
"""Acceptance for the row operands meeting a call that carries several requests.

Three items, one per conjunct of one refusal, no ``parametrize``. The layer serves a
concurrent decode by recursing once per request, and that recursion passes the two row
operands to nobody: a caller that asks for a masked recurrence over several requests is
served an UNMASKED one and told nothing. The operands also describe one sequence -- the
runner builds them from the first request's length -- so threading them unchanged would
be as wrong as dropping them. Until the operands are per-request, such a call is refused
by name.

The refusal's condition has two sides and both are items here, because a refusal that
fires wider than its condition is a new defect: a concurrent decode WITHOUT the operands
must still be served, and a single-request call WITH them must still be served.

Run on the Tier N harness -- the NKI simulator on the host CPU, no device and no
lease::

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
    python -m pytest \
        test/vllm_neuron/model/glm5_next/test_kda_many_request_operands_137.py \
        -q -s -rA --timeout 900 -p no:randomly -p no:cacheprovider

1. a two-request decode carrying the row operands is REFUSED, and the message names
    both operands and the request count; the half-passed spelling is refused too,
    because the pairing check that would have caught it stands below this branch and
    never runs on this path;
2. the same two-request decode WITHOUT the operands is served, and its output carries
    one row per request -- so item 1 refuses the operands and not the concurrency;
3. a single-request decode WITH the operands is served -- so item 1 refuses the
    concurrency and not the operands.

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


def _decode(layer, rows, requests: int, extras: dict):
    """One concurrent-decode call: one token per request, each its own carrier."""
    conv_state, recurrent_state, start_position = _slots(layer, requests)
    return layer(
        rows,
        conv_state=conv_state if requests > 1 else conv_state[0],
        recurrent_state=recurrent_state if requests > 1 else recurrent_state[0],
        is_prefill=False,
        start_position=start_position,
        **extras,
    )


def test_c01_a_many_request_decode_carrying_the_row_operands_is_refused(one_layer):
    """The refusal fires, and it names both operands and the request count."""
    operands = _operands(one_layer.layer, DECLARED_REQUESTS, DECLARED_REQUESTS)
    assert operands, "the tree carries the row operands and the runner's builder"
    print(
        f"MANYREQ|operands|real_tokens={int(operands['real_tokens'])}|"
        f"row_mask={tuple(operands['row_mask'].shape)}|requests={DECLARED_REQUESTS}"
    )
    with pytest.raises(ValueError) as refusal:
        _decode(one_layer.layer, one_layer.rows, DECLARED_REQUESTS, operands)
    said = str(refusal.value)
    print(f"MANYREQ|refusal|{said}")
    assert "real_tokens" in said and "row_mask" in said
    assert str(DECLARED_REQUESTS) in said
    # THE HALF-PASSED SPELLING IS REFUSED TOO. The check that reads the two operands
    # as one fact stands below this branch, so on this path nothing else would catch
    # a call that passed one of them.
    for name in ("real_tokens", "row_mask"):
        with pytest.raises(ValueError) as half:
            _decode(
                one_layer.layer,
                one_layer.rows,
                DECLARED_REQUESTS,
                {name: operands[name]},
            )
        print(f"MANYREQ|refusal_half|{name}|{str(half.value)}")
        assert "real_tokens" in str(half.value) and "row_mask" in str(half.value)


def test_c02_a_many_request_decode_without_the_row_operands_is_still_served(one_layer):
    """The refusal is the operands', not the concurrency's."""
    out = _decode(one_layer.layer, one_layer.rows, DECLARED_REQUESTS, {})
    print(f"MANYREQ|served_without_operands|out={tuple(out.shape)}")
    assert tuple(out.shape) == (DECLARED_REQUESTS, one_layer.hidden)
    assert torch.isfinite(out).all()


def test_c03_a_single_request_decode_with_the_row_operands_is_still_served(one_layer):
    """The refusal is the concurrency's, not the operands'."""
    operands = _operands(one_layer.layer, 1, 1)
    assert operands, "the tree carries the row operands and the runner's builder"
    out = _decode(one_layer.layer, one_layer.rows[:1], 1, operands)
    print(
        f"MANYREQ|served_with_operands|out={tuple(out.shape)}|"
        f"real_tokens={int(operands['real_tokens'])}"
    )
    assert tuple(out.shape) == (1, one_layer.hidden)
    assert torch.isfinite(out).all()
