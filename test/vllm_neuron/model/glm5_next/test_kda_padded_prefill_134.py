# SPDX-License-Identifier: Apache-2.0
"""Acceptance for a KDA prefill that arrives PADDED to its bucket.

Three items, one per counted conjunct, no ``parametrize``. A served prefill is padded
up to the width the captured graph was compiled for, so the layer is handed rows that
carry no token of the request. The state it leaves behind must be the state the same
prompt leaves unpadded: the recurrence must neither decay nor update on a padding row,
and the convolution's history must come from the last REAL rows rather than from the
bucket's tail. Every declared value and both comparator numbers are CARRIED from the
landed KDA layer acceptance, imported rather than retyped.

Run on the Tier N harness -- the NKI simulator on the host CPU, no device and no
lease::

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
    python -m pytest \
        test/vllm_neuron/model/glm5_next/test_kda_padded_prefill_134.py \
        -q -s -rA --timeout 900 -p no:randomly -p no:cacheprovider

1. at a real length that is a whole number of chunks, the padded prefill's conv and
   recurrent carriers are EXACTLY the unpadded prefill's, and the padding rows carry
   a marker no real row holds;
2. at a real length that is not a whole number of chunks -- where the padded arm
   carries the tail through the chunked seams and the unpadded arm walks the same
   rows on the single-token seam -- the two carriers meet the carried comparator, and
   the measured worst error is printed;
3. the first decode after the padded prefill equals the unpadded arm's, on the output
   and on both carriers, which is where a history taken from the bucket's tail shows.

THE OPERANDS ARE THE RUNNER'S OWN, and they are passed only where the tree accepts
them. A tree whose layer names no row mask is driven without one, which is the
bucket-width scan itself, so these items READ that tree rather than failing to call
it.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from test.vllm_neuron.model.glm5_next import test_kda_layer as layer_half

from vllm_neuron.accuracy.testing import assert_close

#: Carried from the landed layer acceptance: the comparator pair, the stack depth,
#: the chunk width, this rank's head geometry, the registered degree and the seed.
DECLARED_RTOL = layer_half.DECLARED_RTOL
DECLARED_ATOL = layer_half.DECLARED_ATOL
DECLARED_STACK_LAYERS = layer_half.DECLARED_STACK_LAYERS
DECLARED_CHUNK = layer_half.DECLARED_CHUNK
DECLARED_PER_RANK_HEADS = layer_half.DECLARED_PER_RANK_HEADS
DECLARED_KDA_HEAD_SIZE = layer_half.DECLARED_KDA_HEAD_SIZE
DECLARED_KDA_CONV_KERNEL_SIZE = layer_half.DECLARED_KDA_CONV_KERNEL_SIZE
REGISTERED_TP_WORLD_SIZE = layer_half.REGISTERED_TP_WORLD_SIZE
SEED = layer_half.SEED

#: The width both arms' padded operands take: four whole chunks, so the aligned and
#: the ragged real length both sit inside one bucket.
DECLARED_BUCKET = 4 * DECLARED_CHUNK

#: Item 1's real length. A whole number of chunks, so the real rows take the SAME
#: route on both arms and the two carriers can be asked for bit equality.
DECLARED_ALIGNED_REAL = 2 * DECLARED_CHUNK

#: Item 2's real length. Not a whole number of chunks: the bucket is chunk-aligned,
#: so the padded arm carries the tail rows through the chunked seams while the
#: unpadded arm walks them one at a time. The same value by two accumulation orders
#: agrees to a tolerance and not to the bit, which is what item 2 measures.
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
            DECLARED_KDA_HEAD_SIZE,
            DECLARED_KDA_CONV_KERNEL_SIZE,
        )
        for _ in range(DECLARED_STACK_LAYERS)
    ]
    # One row past the bucket: the token the decode step of item 3 is handed.
    torch.manual_seed(SEED + 1)
    tokens = torch.randn(DECLARED_BUCKET + 1, hidden, dtype=torch.float32)

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
        text_config=text_config, layers=layers, tokens=tokens, hidden=hidden
    )


@pytest.fixture(scope="module")
def stack() -> SimpleNamespace:
    """One built stack for every item; each drive takes its own bank."""
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
    """The two row operands, BUILT BY THE RUNNER, or none when the tree has neither.

    Reading the runner's own builder rather than assembling a mask here keeps one
    authority for what the layers are handed. A tree that carries neither the builder
    nor the keywords is driven without them, and that drive is the bucket-width scan.
    """
    runner = layer_half._runner_module().NeuronModelRunner
    builder = getattr(runner, "_glm5next_real_row_extent", None)
    accepted = inspect.signature(layer.attention.forward).parameters
    if builder is None or "row_mask" not in accepted or "real_tokens" not in accepted:
        return {}
    real_length, row_mask = builder(tokens, real, torch.device("cpu"))
    return {"real_tokens": real_length, "row_mask": row_mask}


def _drive(layers, bank, rows, *, real: int, is_prefill: bool, start_position: int):
    """One call per layer through the stack, each layer handed its own carrier."""
    out = rows
    for layer, (conv_state, recurrent_state) in zip(layers, bank):
        out = layer(
            out,
            conv_state=conv_state,
            recurrent_state=recurrent_state,
            is_prefill=is_prefill,
            start_position=start_position,
            chunk_size=DECLARED_CHUNK,
            **_extras(layer, int(rows.shape[0]), real),
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


def _worst_pair(got_bank, want_bank) -> tuple[float, float]:
    """The worst conv error and the worst recurrent error over every layer."""
    return (
        max(_worst(got[0], want[0]) for got, want in zip(got_bank, want_bank)),
        max(_worst(got[1], want[1]) for got, want in zip(got_bank, want_bank)),
    )


def _masked(stack: SimpleNamespace) -> bool:
    """Whether this tree takes the row operands at all; printed by every item."""
    return bool(_extras(stack.layers[0], DECLARED_BUCKET, DECLARED_ALIGNED_REAL))


def test_kda_padded_prefill_c01_a_chunk_multiple_length_leaves_the_same_state(
    stack: SimpleNamespace,
) -> None:
    """Item 1. Certifying component: which rows the recurrence and the history read.

    A whole number of chunks of real rows inside a chunk-aligned bucket, so the real
    rows are chunked identically on both arms and the only difference between them is
    the padding. Both carriers are asked for bit equality, which is what the masked
    gates make available: a zero decay rate leaves the state multiplied by one and a
    zero update rate adds nothing to it, and neither is an approximation.
    """
    real = DECLARED_ALIGNED_REAL
    rows = _padded_rows(stack, real)
    marker_hits = int((stack.tokens[:real] == DECLARED_PADDING_MARKER).sum())
    print(
        f"PADPREFILL|item1|bucket={DECLARED_BUCKET}|real={real}|chunk={DECLARED_CHUNK}|"
        f"padding_rows={DECLARED_BUCKET - real}|marker={DECLARED_PADDING_MARKER}|"
        f"real_values_holding_the_marker={marker_hits}|"
        f"every_padding_row_holds_it={bool((rows[real:] == DECLARED_PADDING_MARKER).all())}|"
        f"row_operands_accepted={_masked(stack)}",
        flush=True,
    )
    assert marker_hits == 0, (
        f"{marker_hits} value(s) of the request's own rows hold the marker "
        f"{DECLARED_PADDING_MARKER}, so a state moved by a padding row could not be "
        f"told from one moved by a real row"
    )

    arms = _arms(stack, real)
    worst_conv, worst_state = _worst_pair(arms.got_bank, arms.want_bank)
    print(
        f"PADPREFILL|item1|layers={DECLARED_STACK_LAYERS}|"
        f"worst_abs_error_conv={worst_conv:.3e}|worst_abs_error_state={worst_state:.3e}|"
        f"expected=bit_identical",
        flush=True,
    )
    for index, (got, want) in enumerate(zip(arms.got_bank, arms.want_bank)):
        assert torch.equal(got[0], want[0]), (
            f"layer {index}'s conv history differs between the padded and the "
            f"unpadded prefill of the same {real} tokens, by at most {worst_conv:.3e}; "
            f"the history the next step convolves with is the bucket's tail"
        )
        assert torch.equal(got[1], want[1]), (
            f"layer {index}'s recurrent state differs between the padded and the "
            f"unpadded prefill of the same {real} tokens, by at most {worst_state:.3e}; "
            f"the padding rows entered the sequence's own state"
        )


def test_kda_padded_prefill_c02_a_ragged_length_carries_to_the_declared_comparator(
    stack: SimpleNamespace,
) -> None:
    """Item 2. The common case: a real length that is not a whole number of chunks.

    The bucket is chunk-aligned, so the padded arm hands the tail rows to the chunked
    seams while the unpadded arm walks the same rows on the single-token seam. The two
    routes accumulate one value in two orders, so this item grades the carried
    comparator and prints what it measured.
    """
    real = DECLARED_RAGGED_REAL
    arms = _arms(stack, real)
    worst_conv, worst_state = _worst_pair(arms.got_bank, arms.want_bank)
    print(
        f"PADPREFILL|item2|bucket={DECLARED_BUCKET}|real={real}|chunk={DECLARED_CHUNK}|"
        f"remainder={real % DECLARED_CHUNK}|"
        f"worst_abs_error_conv={worst_conv:.3e}|worst_abs_error_state={worst_state:.3e}|"
        f"declared_rtol={DECLARED_RTOL}|declared_atol={DECLARED_ATOL}|"
        f"row_operands_accepted={_masked(stack)}",
        flush=True,
    )
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


def test_kda_padded_prefill_c03_the_first_decode_after_it_equals_the_unpadded_arms(
    stack: SimpleNamespace,
) -> None:
    """Item 3. What the request actually feels: its next token.

    Both arms prefill the same request, one padded and one not, and then take one
    decode step on the same token row. The decode reads the carriers and nothing else,
    so this is the reading that the state and the history a served request continues
    from are its own. A history taken from the bucket's tail shows here even when the
    recurrent state happens to survive.
    """
    real = DECLARED_ALIGNED_REAL
    arms = _arms(stack, real)
    step = stack.tokens[DECLARED_BUCKET : DECLARED_BUCKET + 1]
    want = _drive(
        stack.layers, arms.want_bank, step, real=1, is_prefill=False, start_position=real
    )
    got = _drive(
        stack.layers, arms.got_bank, step, real=1, is_prefill=False, start_position=real
    )
    worst_conv, worst_state = _worst_pair(arms.got_bank, arms.want_bank)
    worst_out = _worst(got, want)
    print(
        f"PADPREFILL|item3|real={real}|decode_position={real}|"
        f"worst_abs_error_out={worst_out:.3e}|worst_abs_error_conv={worst_conv:.3e}|"
        f"worst_abs_error_state={worst_state:.3e}|bit_identical_out={torch.equal(got, want)}|"
        f"declared_rtol={DECLARED_RTOL}|declared_atol={DECLARED_ATOL}|"
        f"row_operands_accepted={_masked(stack)}",
        flush=True,
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
