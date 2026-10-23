# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 model tree, load-time products, embedding, sampler and prefill call sites.

Set ``KIMI_K3_CHECKPOINT`` to the checkpoint directory; the default is
``~/models/moonshotai/Kimi-K3``. The prefill call sites run their NKI kernels in the
simulator, so the run sets ``VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1``.
"""

import copy
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from vllm_neuron.functional.attention.attn_res_prefill import attn_res_prefill
from vllm_neuron.functional.embedding import embedding
from vllm_neuron.functional.kda.output import kda_output_prefill
from vllm_neuron.functional.sampling import sample
from vllm_neuron.model.kimi_k3.model import KimiK3ForConditionalGeneration
from vllm_neuron.model.kimi_k3.weight_loaders_mxfp4 import (
    EXPERT_PARTS,
    checkpoint_mappings,
    repack_expert,
)
from vllm_neuron.model.neuron_config import NeuronConfig, OnDeviceSamplingConfig

CHECKPOINT = Path(
    os.environ.get(
        "KIMI_K3_CHECKPOINT", Path.home() / "models" / "moonshotai" / "Kimi-K3"
    )
)

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "model.safetensors.index.json").is_file(),
    reason=f"Kimi K3 checkpoint not found at {CHECKPOINT}",
)

PREFIX = "language_model."
HIDDEN = 7168
EPS = 1e-5
ATTN_RES_WEIGHTS = ("_res_norm.weight", "_res_proj.weight")


@pytest.fixture(scope="module")
def weight_map() -> dict[str, str]:
    """The checkpoint's tensor-to-shard map."""
    with open(CHECKPOINT / "model.safetensors.index.json") as f:
        return json.load(f)["weight_map"]


@pytest.fixture(scope="module")
def hf_config() -> dict:
    """The checkpoint's HF config as vLLM builds it, as a dict."""
    from vllm.transformers_utils.config import get_config

    return get_config(str(CHECKPOINT), trust_remote_code=False).to_dict()


def _layers(config: dict, layers: int, kda_heads: int | None = None) -> dict:
    """The config cut to its first ``layers`` layers, optionally with fewer KDA heads."""
    config = copy.deepcopy(config)
    text = config["text_config"]
    attn = text["linear_attn_config"]
    for key in ("kda_layers", "full_attn_layers"):
        attn[key] = [i for i in attn[key] if i <= layers]
    text["num_hidden_layers"] = layers
    if kda_heads is not None:
        attn["num_heads"] = kda_heads
    return config


def _meta_model(config: dict, neuron_config=None) -> KimiK3ForConditionalGeneration:
    """The model built on the meta device, as the runner builds it."""
    with torch.device("meta"):
        return KimiK3ForConditionalGeneration.from_configs(config, neuron_config)


def _shapes(weight_map: dict[str, str], keys) -> dict[str, list[int]]:
    """Stored shapes of checkpoint tensors, read from the shard headers."""
    by_shard = defaultdict(list)
    for key in keys:
        by_shard[weight_map[key]].append(key)
    shapes = {}
    for shard, names in by_shard.items():
        with safe_open(CHECKPOINT / shard, framework="pt") as f:
            for name in names:
                shapes[name] = f.get_slice(name).get_shape()
    return shapes


def _read(weight_map: dict[str, str], keys) -> dict[str, torch.Tensor]:
    """Checkpoint tensors by key, as stored."""
    by_shard = defaultdict(list)
    for key in keys:
        by_shard[weight_map[key]].append(key)
    out = {}
    for shard, names in by_shard.items():
        with safe_open(CHECKPOINT / shard, framework="pt") as f:
            for name in names:
                out[name] = f.get_tensor(name)
    return out


def _random_weights(names_and_shapes, seed: int) -> dict[str, torch.Tensor]:
    """Random bf16 AttnRes weights: norm gains near 1, score projections near 0."""
    gen = torch.Generator().manual_seed(seed)
    out = {}
    for name, shape in names_and_shapes:
        if name.endswith("_proj.weight"):
            out[name] = (0.035 * torch.randn(shape, generator=gen)).bfloat16()
        else:
            out[name] = (1.0 + 0.1 * torch.randn(shape, generator=gen)).bfloat16()
    return out


def _same_bits(out: torch.Tensor, expected: torch.Tensor) -> bool:
    """True when both tensors have the same dtype, shape and bytes."""
    return (
        out.dtype == expected.dtype
        and out.shape == expected.shape
        and torch.equal(out.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))
    )


@pytest.fixture(scope="module")
def loaded_one_layer(hf_config) -> KimiK3ForConditionalGeneration:
    """The 1-layer model loaded from the checkpoint by its own load_weights."""
    model = _meta_model(_layers(hf_config, 1))
    model.load_weights(str(CHECKPOINT), torch.device("cpu"))
    return model


@pytest.fixture(scope="module")
def full_tree(hf_config) -> KimiK3ForConditionalGeneration:
    """The 93-layer model on the meta device, with a deterministic on-device sampler."""
    sampling = OnDeviceSamplingConfig(deterministic=True)
    return _meta_model(hf_config, NeuronConfig(on_device_sampling_config=sampling))


def test_four_layer_tree_maps_onto_checkpoint_keys(hf_config, weight_map):
    model = _meta_model(_layers(hf_config, 4))
    text = model.text_config
    assert [len(text.kda_layer_indices), len(text.mla_layer_indices)] == [3, 1]
    params = dict(model.named_parameters())
    mappings, _ = checkpoint_mappings(model, text)
    mapped = {k for v in mappings.values() for k in ([v] if isinstance(v, str) else v)}
    unmapped = sorted(k for k in mapped if k not in weight_map)
    assert unmapped == [], unmapped[:5]
    stored = _shapes(weight_map, mapped)

    expert = f"{PREFIX}model.layers.1.block_sparse_moe.experts.0"
    sources = [f"{w}.{kind}" for w in ("w1", "w3", "w2")
               for kind in ("weight_packed", "weight_scale")]
    one = _read(weight_map, [f"{expert}.{s}" for s in sources])
    parts = repack_expert(*(one[f"{expert}.{s}"] for s in sources), num_programs=model.num_programs)
    source_shapes = {s: stored[f"{expert}.{s}"] for s in sources}

    mismatched = []
    for name, keys in mappings.items():
        param, leaf = params[name], name.rpartition(".")[2]
        if leaf in EXPERT_PARTS:
            same_sources = all(stored[k] == source_shapes[k.split(".", 7)[7]] for k in keys)
            if tuple(param.shape) != (text.num_experts, *parts[leaf].shape) or not same_sources:
                mismatched.append(name)
        elif name.endswith("A_log"):
            if tuple(param.shape) != (text.kda_num_heads,) or math.prod(stored[keys]) < 96:
                mismatched.append(name)
        elif list(param.shape) != stored[keys]:
            mismatched.append(name)
    assert mismatched == [], mismatched[:5]

    in_scope = {
        k for k in weight_map
        if k.startswith(PREFIX)
        and (not k.startswith(f"{PREFIX}model.layers.") or int(k.split(".")[3]) < 4)
    }
    left_over = sorted(in_scope - mapped)
    assert left_over == [], left_over[:5]
    assert len(in_scope) == len(mapped)


def test_score_weights_computed_after_load(loaded_one_layer, hf_config, weight_map):
    model = loaded_one_layer
    assert [n for n, p in model.named_parameters() if p.is_meta] == []
    full = _meta_model(hf_config)
    names = [n for n, _ in full.named_parameters() if n.endswith(ATTN_RES_WEIGHTS)]
    stored = _read(weight_map, [PREFIX + n for n in names])
    full.load_state_dict({n[len(PREFIX):]: t for n, t in stored.items()}, strict=False, assign=True)
    full.prepare_after_load()

    checked = 0
    for tree in (model, full):
        sites = {"model.output_attn_res": tree.model.output_score_w}
        for i, layer in enumerate(tree.model.layers):
            sites[f"model.layers.{i}.self_attention_res"] = layer.attention_score_w
            sites[f"model.layers.{i}.mlp_res"] = layer.mlp_score_w
        for site, score_w in sites.items():
            norm = stored[f"{PREFIX}{site}_norm.weight"]
            proj = stored[f"{PREFIX}{site}_proj.weight"]
            assert score_w.dtype == torch.float32 and tuple(score_w.shape) == (HIDDEN,), site
            assert torch.equal(score_w, norm.float() * proj.squeeze(0).float()), site
            checked += 1
    assert checked == 3 + 2 * 93 + 1


def test_embedding_fills_the_stream_and_bank_row_zero(loaded_one_layer, full_tree):
    model = loaded_one_layer
    vocab = model.text_config.vocab_size
    ids = torch.tensor([0, 1, 163584, 163586, 7, 4242, 99999, vocab - 1])
    expected = embedding(ids, model.model.embed_tokens.weight)
    stream, bank = model.model.embed(ids)
    mix, base = model.model.layers[0].attention_mix(stream, bank)
    assert _same_bits(stream, expected)
    assert tuple(bank.shape) == (8, 1, HIDDEN)
    assert _same_bits(bank[:, 0], stream)
    assert mix is stream and base is None
    assert full_tree.model.num_bank_rows == math.ceil(93 / 12) == 8


def test_sampler_equals_the_shipped_sampling_function(full_tree):
    gen = torch.Generator().manual_seed(3)
    logits = torch.randn(8, 5120, generator=gen)
    for top_k, top_p, temperature in ((-1, 1.0, 0.0), (50, 0.9, 0.7)):
        params = torch.tensor([[top_k, top_p, temperature]] * 8)
        expected = sample(
            logits,
            temperature=params[:, 2],
            top_k=params[:, 0].to(torch.int32),
            top_p=params[:, 1],
            deterministic=True,
        )
        assert _same_bits(full_tree.sample(logits, params), expected), (top_k, top_p, temperature)


def test_landed_prefill_call_sites_equal_their_wrappers(full_tree, hf_config):
    model = full_tree
    layer = model.model.layers[84]
    names = [
        f"model.layers.84.{site}_{part}.weight"
        for site in ("self_attention_res", "mlp_res")
        for part in ("norm", "proj")
    ] + ["model.output_attn_res_norm.weight", "model.output_attn_res_proj.weight"]
    params = dict(model.named_parameters())
    weights = _random_weights([(n, tuple(params[n].shape)) for n in names], seed=11)
    model.load_state_dict(weights, strict=False, assign=True)
    model.prepare_after_load()
    # The reference appends a bank row after the attention-site mix of every 12th layer.
    rows, expected_rows = 0, []
    for i in range(len(model.model.layers)):
        attention_rows = rows
        rows += i % 12 == 0
        expected_rows.append((attention_rows, rows))
    assert [(x.attention_rows, x.mlp_rows) for x in model.model.layers] == expected_rows
    assert layer.block_start and expected_rows[84] == (7, 8)

    gen = torch.Generator().manual_seed(12)
    tokens = 128
    bank = torch.randn(tokens, 8, HIDDEN, generator=gen).bfloat16()
    streams = [torch.randn(tokens, HIDDEN, generator=gen).bfloat16() for _ in range(3)]
    w = {n.rsplit(".", 2)[-2]: weights[n] for n in names}
    score = {
        site: w[f"{site}_norm"].float() * w[f"{site}_proj"].squeeze(0).float()
        for site in ("self_attention_res", "mlp_res", "output_attn_res")
    }

    expected = attn_res_prefill(bank[:, :7], streams[0], score["self_attention_res"], EPS)
    assert _same_bits(layer.attention_mix(streams[0], bank)[0], expected)
    expected = attn_res_prefill(bank[:, :8], streams[1], score["mlp_res"], EPS)
    assert _same_bits(layer.mlp_mix(streams[1], bank), expected)
    expected = attn_res_prefill(bank, streams[2], score["output_attn_res"], EPS)
    assert _same_bits(model.model.output_mix(streams[2], bank), expected)

    kda_model = _meta_model(_layers(hf_config, 4, kda_heads=3))
    attn = kda_model.model.layers[1].self_attn
    o_norm = 1.0 + 0.1 * torch.randn(128, generator=gen)
    o_proj = (0.02 * torch.randn(HIDDEN, 384, generator=gen)).bfloat16()
    kda_model.load_state_dict(
        {"model.layers.1.self_attn.o_norm.weight": o_norm,
         "model.layers.1.self_attn.o_proj.weight": o_proj},
        strict=False,
        assign=True,
    )
    core = torch.randn(256, 384, generator=gen)
    gate = torch.randn(256, 384, generator=gen)
    expected = kda_output_prefill(core, gate, o_norm, o_proj, EPS)
    assert _same_bits(attn.output_prefill(core, gate), expected)
