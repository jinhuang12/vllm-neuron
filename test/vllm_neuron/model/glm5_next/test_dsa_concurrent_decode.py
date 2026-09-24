# SPDX-License-Identifier: Apache-2.0
"""Exact DSA decode bridge checks against independent singleton execution."""

import pytest
import torch

from test.vllm_neuron.model.glm5_next.test_dsa_layer import build_layer_stack
from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

PAGE = 4
MAX_SEQUENCE = 64
STATE_SLOTS = (3, 1)
STARTS = (34, 39)
# Disjoint, noncontiguous pages in a shuffled order; neither row is a bank slice.
_ORDER = (8, 1, 13, 4, 10, 0, 15, 6, 3, 12, 2, 11, 7, 14, 5, 9)
PAGE_ROWS = tuple(tuple(2 * index + base for index in _ORDER) for base in (1, 2))


def _state(attention, cfg, generator):
    return {
        "latent": torch.randn(
            40 * PAGE, 1, attention.head_size, generator=generator
        ).to(torch.bfloat16),
        "pool": torch.randn(
            4,
            MAX_SEQUENCE // cfg.index_kpool + 1,
            cfg.index_head_dim,
            generator=generator,
        ).to(torch.bfloat16),
        "tail": torch.randn(
            4, 2, cfg.index_kpool, cfg.index_head_dim, generator=generator
        ).to(torch.bfloat16),
    }


def _carrier(state, cfg, starts, order):
    bank = {
        "name": "model.layers.0.self_attn",
        "family": "self_attn",
        "latent_cache": state["latent"],
        "block_size": PAGE,
    }
    slots = [STATE_SLOTS[index] for index in order]
    rows = [PAGE_ROWS[index] for index in order]
    positions = [starts[index] for index in order]
    return NeuronModelRunner._glm5next_layer_carriers(
        [bank],
        [{"pool_cache": state["pool"], "tail": state["tail"]}],
        geometries=[
            {
                "state_slot": slots[0],
                "state_slots": slots,
                "block_ids": rows[0],
                "block_id_rows": rows,
                "window_blocks": MAX_SEQUENCE // PAGE,
                "page_size": PAGE,
            }
        ],
        is_prefill=False,
        tokens=len(order),
        requests=len(order),
        start_position=positions[0],
        request_starts=positions,
        request_real_tokens=[1] * len(order),
        real_tokens=1,
        softmax_scale=(cfg.qk_nope_head_dim + cfg.qk_rope_head_dim) ** -0.5,
        max_seq_len=MAX_SEQUENCE,
        index_kpool=cfg.index_kpool,
    )[0]


def test_decode_output_and_all_cache_writes_match_independent_requests():
    """Unequal lengths cross pool boundaries while batch order changes."""
    layers, cfg, generator = build_layer_stack(layers=1)
    layer = layers[0]
    baseline = _state(layer.attention, cfg, generator)
    candidate = {name: value.clone() for name, value in baseline.items()}
    initial = {name: value.clone() for name, value in baseline.items()}
    hidden = torch.randn(3, 2, cfg.hidden_size, generator=generator).to(torch.bfloat16)
    touched_latent = set()
    touched_pool = set()
    touched_tail = set()

    for step, order in enumerate(((0, 1), (1, 0), (0, 1))):
        starts = tuple(position + step for position in STARTS)
        reference = torch.cat(
            [
                layer(
                    hidden[step, index : index + 1],
                    **_carrier(baseline, cfg, starts, [index]),
                )
                for index in order
            ]
        )
        got = layer(
            hidden[step, list(order)], **_carrier(candidate, cfg, starts, order)
        )
        assert torch.equal(got, reference)
        for name in baseline:
            assert torch.equal(candidate[name], baseline[name]), name
        for index, position in enumerate(starts):
            slot = STATE_SLOTS[index]
            touched_latent.add(
                PAGE_ROWS[index][position // PAGE] * PAGE + position % PAGE
            )
            touched_pool.add(
                (
                    slot,
                    (
                        position // cfg.index_kpool
                        if (position + 1) % cfg.index_kpool == 0
                        else candidate["pool"].shape[1] - 1
                    ),
                )
            )
            touched_tail.add((slot, position % cfg.index_kpool))

    for name in candidate:
        assert not torch.equal(
            candidate[name], initial[name]
        ), f"{name} writes were not exercised"
    untouched = torch.ones(candidate["latent"].shape[0], dtype=torch.bool)
    untouched[list(touched_latent)] = False
    assert torch.equal(candidate["latent"][untouched], initial["latent"][untouched])
    for slot in range(4):
        for row in range(candidate["pool"].shape[1]):
            if (slot, row) not in touched_pool:
                assert torch.equal(
                    candidate["pool"][slot, row], initial["pool"][slot, row]
                )
        for row in range(cfg.index_kpool):
            if (slot, row) not in touched_tail:
                assert torch.equal(
                    candidate["tail"][slot, :, row], initial["tail"][slot, :, row]
                )


@pytest.mark.parametrize(
    "bad_case", ["count", "mixed", "rows", "prefill", "missing_tail", "prefix"]
)
def test_invalid_batch_refuses_before_any_cache_write(bad_case):
    layers, cfg, generator = build_layer_stack(layers=1)
    attention = layers[0].attention
    state = _state(attention, cfg, generator)
    before = {name: value.clone() for name, value in state.items()}
    carrier = _carrier(state, cfg, STARTS, (0, 1))
    hidden = torch.randn(2, cfg.hidden_size, generator=generator).to(torch.bfloat16)
    if bad_case == "count":
        carrier["position"] = carrier["position"][:1]
    elif bad_case == "mixed":
        carrier["seq_lens"] = carrier["seq_lens"][0]
    elif bad_case == "rows":
        hidden = torch.cat((hidden, hidden))
    elif bad_case == "prefill":
        carrier["prefill_tail"] = carrier["tail"]
    elif bad_case == "missing_tail":
        carrier["tail"] = (carrier["tail"][0], None)
    else:
        carrier["active_mla_query_rows"] = 1
    with pytest.raises(ValueError, match="concurrent sparse"):
        attention(hidden, **carrier)
    for name in state:
        assert torch.equal(state[name], before[name])
