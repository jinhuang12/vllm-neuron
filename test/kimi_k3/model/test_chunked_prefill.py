# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 chunked prefill: a prompt in 4 chunks against the same prompt in 1 chunk.

The model is the checkpoint's first layer, KDA with the dense MLP, at one rank's widths.
Its weights are random except the gate's stored A_log and dt_bias, which set how far the
state carries: random ones decay it within a few tokens, so a lost carry would not show.
Each chunk starts from the conv rows and recurrent state the last one left in the slot.
Set ``KIMI_K3_CHECKPOINT`` to the checkpoint directory (default
``~/models/moonshotai/Kimi-K3``); run with ``VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1``.
"""

import copy
import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn

from vllm_neuron.model.kimi_k3.model import KimiK3ForConditionalGeneration

CHECKPOINT = Path(
    os.environ.get("KIMI_K3_CHECKPOINT", Path.home() / "models" / "moonshotai" / "Kimi-K3")
)

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "model.safetensors.index.json").is_file(),
    reason=f"Kimi K3 checkpoint not found at {CHECKPOINT}",
)

PROMPT, CHUNK, VOCAB, HEADS = 1024, 256, 5120, 3


def _stored_gate() -> dict[str, torch.Tensor]:
    """The first layer's stored A_log and dt_bias for its first ``HEADS`` heads."""
    with open(CHECKPOINT / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]
    gate = {}
    for leaf, size in (("A_log", HEADS), ("dt_bias", HEADS * 128)):
        key = f"language_model.model.layers.0.self_attn.{leaf}"
        with safe_open(str(CHECKPOINT / weight_map[key]), "pt") as f:
            gate[leaf] = f.get_tensor(key).float().flatten()[:size]
    return gate


def _first_layer() -> KimiK3ForConditionalGeneration:
    """The checkpoint's first layer at one rank's widths: random weights, the stored gate."""
    from vllm.transformers_utils.config import get_config

    config = copy.deepcopy(get_config(str(CHECKPOINT), trust_remote_code=False).to_dict())
    text = config["text_config"]
    text.update(num_hidden_layers=1, intermediate_size=1056, vocab_size=VOCAB)
    text["linear_attn_config"].update(num_heads=HEADS, kda_layers=[1], full_attn_layers=[])
    with torch.device("meta"):
        model = KimiK3ForConditionalGeneration.from_configs(config)
    gen, gate = torch.Generator().manual_seed(0), _stored_gate()
    for name, p in list(model.named_parameters()):
        path, _, leaf = name.rpartition(".")
        if leaf in gate:
            value = gate[leaf]
        elif "norm" in name:
            value = 1 + 0.1 * torch.randn(p.shape, generator=gen)
        else:
            scale = 0.5 if "conv" in name else 0.02
            value = scale * torch.randn(p.shape, generator=gen)
        param = nn.Parameter(value.to(p.dtype), requires_grad=False)
        setattr(model.get_submodule(path), leaf, param)
    model.prepare_after_load()
    pages = torch.zeros(3, 294_912, dtype=torch.int8)  # the runner's page layout, 3 slots
    conv = pages[:, :6912].view(torch.bfloat16).view(3, 3, HEADS * 384)
    state = pages[:, 6912:203_520].view(torch.float32).view(3, HEADS, 128, 128)
    model.bind_kv_cache({"layers.0.self_attn": [conv, state, pages]})
    return model


def _prefill(model, ids, slot: int, start: int) -> torch.Tensor:
    """One prefill chunk of ``ids`` at ``start`` in state slot ``slot``; returns its logits."""
    meta = {
        "block_table_tensor": torch.tensor([[slot]], dtype=torch.int32),
        "slot_mapping": torch.arange(start, start + ids.shape[0]) + 1,
        "cached_seq_len": torch.tensor([[start]], dtype=torch.int32),
        "max_query_len": ids.shape[0],
        "decode_token_threshold": 1,
    }
    return model(ids, None, {"layers.0.self_attn": meta})


@pytest.fixture(scope="module")
def runs() -> dict:
    """The prompt's final state, conv rows and last logits row, in 1 chunk and in 4."""
    model = _first_layer()
    kda = model.model.layers[0].self_attn
    ids = torch.randint(0, VOCAB, (PROMPT,), generator=torch.Generator().manual_seed(1))
    whole = _prefill(model, ids, slot=1, start=0)[-1]
    for start in range(0, PROMPT, CHUNK):
        chunked = _prefill(model, ids[start : start + CHUNK], slot=2, start=start)[-1]
    return {
        "state": (kda.recurrent_state[2].double(), kda.recurrent_state[1].double()),
        "conv": (kda.conv_state[2], kda.conv_state[1]),
        "logits": (chunked.double(), whole.double()),
    }


def test_final_state_matches_one_chunk(runs):
    """The final KDA state's relative Frobenius error against 1 chunk is at most 1e-5."""
    chunked, whole = runs["state"]
    assert torch.linalg.norm(chunked - whole) / torch.linalg.norm(whole) <= 1e-5


def test_conv_carry_rows_equal_one_chunk(runs):
    """The slot's 3 conv carry steps equal 1 chunk's, bit for bit."""
    chunked, whole = runs["conv"]
    assert torch.equal(chunked.view(torch.int16), whole.view(torch.int16))


def test_next_token_logits_match_one_chunk(runs):
    """The last prompt token's logits have cosine at least 0.99999 with 1 chunk's."""
    chunked, whole = runs["logits"]
    assert F.cosine_similarity(chunked, whole, dim=0) >= 0.99999
