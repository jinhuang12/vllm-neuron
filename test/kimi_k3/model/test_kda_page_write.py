# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 KDA prefill reads a slot's page row once and writes it back whole.

Three KDA layers share one 251-block int8 page buffer at the runner's offsets. A prefill
chunk takes its conv rows and state from the slot's page row and stores the new ones in
the same row. The reads must equal the conv and state views' rows, and each write must
leave the buffer equal, byte for byte, to an ``index_copy_`` of the same row into a copy.
The kernels around the page access are stand-ins that record what they are given and
return set values, so only the page access runs. Set ``KIMI_K3_CHECKPOINT`` to the
checkpoint directory (default ``~/models/moonshotai/Kimi-K3``); run with
``VLLM_NEURON_CPU_MODE=1``.
"""

import copy
import os
from pathlib import Path

import pytest
import torch

from vllm_neuron.model.kimi_k3 import model as m

CHECKPOINT = Path(
    os.environ.get("KIMI_K3_CHECKPOINT", Path.home() / "models" / "moonshotai" / "Kimi-K3")
)

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "config.json").is_file(),
    reason=f"Kimi K3 checkpoint not found at {CHECKPOINT}",
)

BLOCKS, PAGE, HEADS, DIM, TOKENS = 251, 294_912, 3, 128, 8
CONV_BYTES, STATE_END = 3 * HEADS * 384 * 2, 203_520
SLOTS = (1, 7, 250)


def _bits(x: torch.Tensor) -> torch.Tensor:
    """The tensor's bytes, so NaN payloads compare as bits."""
    return x.contiguous().view(torch.uint8)


@pytest.fixture
def layers():
    """Three KDA layers at one rank's widths, bound to one page buffer of random bytes."""
    from vllm.transformers_utils.config import get_config

    config = copy.deepcopy(get_config(str(CHECKPOINT), trust_remote_code=False).to_dict())
    text = config["text_config"]
    text.update(num_hidden_layers=3, num_attention_heads=HEADS, intermediate_size=1056,
                vocab_size=5120, first_k_dense_replace=3)
    text["linear_attn_config"].update(num_heads=HEADS, kda_layers=[1, 2, 3], full_attn_layers=[])
    with torch.device("meta"):
        model = m.KimiK3ForConditionalGeneration.from_configs(config)
    gen = torch.Generator().manual_seed(0)
    pages = torch.randint(-128, 128, (BLOCKS, PAGE), dtype=torch.int8, generator=gen)
    conv = pages[:, :CONV_BYTES].view(torch.bfloat16).view(BLOCKS, 3, HEADS * 384)
    state = pages[:, CONV_BYTES:STATE_END].view(torch.float32).view(BLOCKS, HEADS, DIM, DIM)
    model.bind_kv_cache({f"layers.{i}.self_attn": [conv, state, pages] for i in range(3)})
    kdas = [layer.self_attn for layer in model.model.layers]
    assert all(kda.pages is pages for kda in kdas)
    return kdas, pages


@pytest.fixture
def chunk(monkeypatch):
    """Runs one prefill chunk with stand-in kernels; returns the conv rows and state it read."""
    seen = {}

    def run(kda, slot, start, conv_rows, new_state):
        width = HEADS * DIM
        proj = torch.zeros(TOKENS, 4 * width + DIM + HEADS, dtype=torch.bfloat16)
        # The chunk's last 3 q, k, v rows become the slot's next conv rows.
        proj[-3:, : 3 * width] = conv_rows
        monkeypatch.setattr(m, "_project", lambda *a: proj)
        monkeypatch.setattr(m, "_prefill_projection", lambda x, w: torch.zeros(TOKENS, width))

        def conv1d(history, *a):
            seen["conv"] = history.clone()
            return torch.zeros(3 * width, TOKENS, dtype=torch.bfloat16)

        def recurrence(*a, state, **k):
            seen["state"] = state.clone()
            return torch.zeros(HEADS, TOKENS, DIM), new_state

        monkeypatch.setattr(m, "depthwise_conv1d_silu", conv1d)
        monkeypatch.setattr(m, "kda_prefill", recurrence)
        monkeypatch.setattr(m, "kda_output_prefill", lambda *a: torch.zeros(TOKENS, 1))
        meta = {
            "block_table_tensor": torch.tensor([[slot]], dtype=torch.int32),
            "cached_seq_len": torch.tensor([[start]], dtype=torch.int32),
            "slot_mapping": torch.arange(start, start + TOKENS) + 1,
        }
        kda.forward_prefill(torch.zeros(TOKENS, 1), None, meta)
        return seen["conv"], seen["state"]

    return run


def _view_reads(kda, slot):
    """The slot's conv rows and state read through the conv and state views."""
    index = torch.tensor([slot])
    return kda.conv_state.index_select(0, index)[0], kda.recurrent_state.index_select(0, index)[0]


def _index_copy_write(pages, kda, slot, conv_rows, state):
    """Writes the slot's parts into ``pages`` by ``index_copy_`` of its whole row."""
    index = torch.tensor([slot])
    parts = ((kda._conv_layout, conv_rows.unsqueeze(0)), (kda._state_layout, state.unsqueeze(0)))
    pages.index_copy_(0, index, m._with_parts(pages.index_select(0, index), parts))


def test_row_reads_and_writes_match_the_views_and_index_copy(layers, chunk):
    """Three chunks per slot: reads equal the views' rows; 0 bytes differ after each write."""
    kdas, pages = layers
    expected, gen = pages.clone(), torch.Generator().manual_seed(1)
    for n in range(3):
        for kda, slot in zip(kdas, SLOTS):
            conv_view, state_view = _view_reads(kda, slot)
            conv_rows = torch.randn(3, HEADS * 384, generator=gen).to(torch.bfloat16)
            state = torch.randn(HEADS, DIM, DIM, generator=gen)
            conv_read, state_read = chunk(kda, slot, (n + 1) * TOKENS, conv_rows, state)
            assert torch.equal(_bits(conv_read), _bits(conv_view)), (n, slot, "conv read")
            assert torch.equal(_bits(state_read), _bits(state_view)), (n, slot, "state read")
            _index_copy_write(expected, kda, slot, conv_rows, state)
            differ = (pages != expected).sum().item()
            rows = (pages != expected).any(dim=1).nonzero().flatten().tolist()
            assert differ == 0, (n, slot, f"{differ} bytes differ in rows {rows}")


def test_odd_bytes_round_trip_at_the_last_slot(layers, chunk):
    """NaN payloads, subnormals and negative zero are written and read back bit for bit."""
    kdas, pages = layers
    kda, slot, before = kdas[2], SLOTS[2], pages.clone()
    # bf16: quiet NaN with payload, signalling NaN, negative quiet NaN, subnormals, -0.
    odd16 = torch.tensor([0x7FC1, 0x7F81, -0x003D, 0x0001, -0x7F81, -0x8000], dtype=torch.int16)
    # fp32: the same classes.
    odd32 = torch.tensor([0x7FC00001, 0x7F800001, -0x003FFFFD, 0x00000001, -0x7F800001,
                          -0x80000000], dtype=torch.int32)
    conv_rows = odd16.repeat(3 * HEADS * 384 // 6).view(torch.bfloat16).view(3, HEADS * 384)
    state = odd32.repeat(HEADS * DIM * DIM // 6).view(torch.float32).view(HEADS, DIM, DIM)

    chunk(kda, slot, TOKENS, conv_rows, state)
    conv_read, state_read = chunk(kda, slot, 2 * TOKENS, conv_rows, state)

    page = pages[slot]
    written = torch.cat([page[:CONV_BYTES], page[CONV_BYTES:STATE_END]]).view(torch.uint8)
    wanted = torch.cat([_bits(conv_rows).flatten(), _bits(state).flatten()])
    assert (written != wanted).sum().item() == 0, "the slot's bytes differ from the values'"
    assert (_bits(conv_read) != _bits(conv_rows)).sum().item() == 0, "conv read back differs"
    assert (_bits(state_read) != _bits(state)).sum().item() == 0, "state read back differs"
    assert torch.equal(page[STATE_END:], before[slot, STATE_END:]), "the page's tail changed"
    changed = (pages != before).any(dim=1).nonzero().flatten().tolist()
    assert [row for row in changed if row != slot] == [], f"other slots changed: {changed}"
