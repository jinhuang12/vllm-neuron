# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 target taps: the streams a draft reads, at the layers it names.

Set ``KIMI_K3_CHECKPOINT`` and ``KIMI_K3_DRAFT`` to the target and draft directories;
the defaults are ``~/models/moonshotai/Kimi-K3`` and
``~/models/Inferact/Kimi-K3-DSpark-Block5``. The tap test runs the model's own layer
loop, bank rows and stream adds, with torch stand-ins for its kernels and sublayers,
so it needs no device and no NKI simulator.
"""

import json
import os
from pathlib import Path

import pytest
import torch
from torch import nn
from vllm.model_executor.models.interfaces import supports_eagle3

from vllm_neuron.functional.attention.attn_res_decode import AttnResDecodeOutput
from vllm_neuron.model.kimi_k3 import model as k3
from vllm_neuron.model.kimi_k3.model import KimiK3ForConditionalGeneration, aux_layers_for_draft

MODELS = Path.home() / "models"
CHECKPOINT = Path(os.environ.get("KIMI_K3_CHECKPOINT", MODELS / "moonshotai" / "Kimi-K3"))
DRAFT = Path(os.environ.get("KIMI_K3_DRAFT", MODELS / "Inferact" / "Kimi-K3-DSpark-Block5"))

pytestmark = pytest.mark.skipif(
    not ((CHECKPOINT / "config.json").is_file() and (DRAFT / "config.json").is_file()),
    reason=f"Kimi K3 target or draft config not found at {CHECKPOINT} and {DRAFT}",
)

HIDDEN = 7168
VOCAB = 1024
TAPPED = (1, 2, 3)


def _config(layers: int | None = None, **text_overrides) -> dict:
    """The target's config, cut to its first ``layers`` layers when given."""
    with open(CHECKPOINT / "config.json") as f:
        config = json.load(f)
    text = config["text_config"]
    if layers is not None:
        attn = text["linear_attn_config"]
        for key in ("kda_layers", "full_attn_layers"):
            attn[key] = [i for i in attn[key] if i <= layers]
        text["num_hidden_layers"] = layers
    text.update(text_overrides)
    return config


def _meta_model(config: dict) -> KimiK3ForConditionalGeneration:
    """The model built on the meta device, as the runner builds it."""
    with torch.device("meta"):
        return KimiK3ForConditionalGeneration.from_configs(config)


def _prefill_mix(bank, stream, score_weight, eps):
    """Stand-in prefill mix: any value that is not the stream."""
    return 0.5 * stream


def _decode_mix(bank, stream, score_weight, num_sources, *, residual=None, bank_slot=None,
                norm_weight=None, tp_rank=None, tp_size=1, dp_rank=None, dp_size=1, eps):
    """Stand-in decode mix: fold the residual into the stream and store it, as the kernel."""
    new = stream if residual is None else stream + residual
    if bank_slot is not None:
        bank[:, bank_slot] = new
    mix = 0.5 * new
    k_slice = mix if norm_weight is not None else None
    return AttnResDecodeOutput(mix, new, k_slice, bank if bank_slot is not None else None)


def _upstream_taps(attn, mlp, block_size, tapped):
    """Upstream's default capture in fp64: the running block prefix plus the pending MLP output."""
    prefix, pending, taps = None, None, []
    for i, (a, m) in enumerate(zip(attn, mlp)):
        if pending is not None:
            prefix = prefix + pending
        prefix = a.double() if i % block_size == 0 else prefix + a.double()
        pending = m.double()
        if i + 1 in tapped:
            taps.append(prefix + pending)
    return taps


def test_taps_equal_upstream_default_capture(monkeypatch):
    model = _meta_model(_config(4, vocab_size=VOCAB, dtype="float32"))
    gen = torch.Generator().manual_seed(7)
    weight = torch.randn(VOCAB, HIDDEN, generator=gen)
    model.model.embed_tokens.weight = nn.Parameter(weight, requires_grad=False)
    model.set_aux_hidden_state_layers(TAPPED)
    monkeypatch.setattr(k3, "attn_res_prefill", _prefill_mix)
    monkeypatch.setattr(k3, "attn_res_decode", _decode_mix)
    monkeypatch.setattr(model, "compute_logits", lambda hidden: hidden)
    layers = model.model.layers
    assert [layer.is_kda for layer in layers] == [True, True, True, False]

    for tokens, is_prefill in ((8, True), (5, False)):
        attn = [torch.randn(tokens, HIDDEN, generator=gen) for _ in layers]
        mlp = [torch.randn(tokens, HIDDEN, generator=gen) for _ in layers]
        for layer, a, m in zip(layers, attn, mlp):
            for name in ("forward_prefill", "forward_decode"):
                monkeypatch.setattr(layer.self_attn, name, lambda *args, out=a: out)
                monkeypatch.setattr(layer.feed_forward, name, lambda *args, out=m: out)
            if hasattr(layer, "block_sparse_moe"):
                monkeypatch.setattr(layer.block_sparse_moe, "prefill_output", lambda pack: pack)
        meta = {"max_query_len": tokens, "decode_token_threshold": 5}
        metas = {f"layers.{i}.self_attn": meta for i in range(len(layers))}
        ids = torch.randint(0, VOCAB, (tokens,), generator=gen)

        _, joined = model(ids, torch.arange(tokens), attn_metadata=metas)

        expected = _upstream_taps(attn, mlp, model.text_config.attn_res_block_size, TAPPED)
        assert joined.dtype == torch.float32
        assert tuple(joined.shape) == (tokens, len(TAPPED) * HIDDEN)
        for j, tap, ref in zip(TAPPED, joined.split(HIDDEN, dim=-1), expected):
            error = ((tap.double() - ref).abs().max() / ref.abs().max()).item()
            assert error <= 1e-5, (is_prefill, j, error)


def test_draft_target_layers_give_the_mla_taps():
    model = _meta_model(_config())
    with open(DRAFT / "config.json") as f:
        draft = json.load(f)
    layers = aux_layers_for_draft(draft, model.text_config)
    assert layers == (72, 88, 92)
    assert [model.model.layers[j - 1].is_kda for j in layers] == [False, False, False]
    assert supports_eagle3(model)
    model.set_aux_hidden_state_layers(layers)
    assert model.get_eagle3_aux_hidden_state_layers() == layers


def test_attn_res_stream_capture_refuses(monkeypatch):
    model = _meta_model(_config(4))
    monkeypatch.setenv("VLLM_KIMI_K3_AUX_ATTN_RES_STREAM", "1")
    with pytest.raises(ValueError, match="VLLM_KIMI_K3_AUX_ATTN_RES_STREAM"):
        model.set_aux_hidden_state_layers(TAPPED)
