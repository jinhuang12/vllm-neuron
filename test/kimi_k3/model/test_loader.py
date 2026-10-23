# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 quantization gate and checkpoint loading against the checkpoint on disk.

Set ``KIMI_K3_CHECKPOINT`` to the checkpoint directory; the default is
``~/models/moonshotai/Kimi-K3``.
"""

import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from safetensors import safe_open

from vllm_neuron.model.kimi_k3 import KimiK3Config
from vllm_neuron.model.kimi_k3.weight_loaders_mxfp4 import (
    F32_SUFFIXES,
    load_weights,
    repack_expert,
)

CHECKPOINT = Path(
    os.environ.get(
        "KIMI_K3_CHECKPOINT", Path.home() / "models" / "moonshotai" / "Kimi-K3"
    )
)

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "model.safetensors.index.json").is_file(),
    reason=f"Kimi K3 checkpoint not found at {CHECKPOINT}",
)

LAYER = 1
PREFIX = "language_model."
MOE = f"model.layers.{LAYER}.block_sparse_moe"
UP = f"{MOE}.routed_expert_up_proj.weight"
NORM = f"{MOE}.routed_expert_norm.weight"
G = 2


def _read(names: list[str]) -> dict[str, torch.Tensor]:
    """Read checkpoint tensors by key, as stored."""
    with open(CHECKPOINT / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]
    out = {}
    for name in names:
        with safe_open(CHECKPOINT / weight_map[name], framework="pt") as f:
            out[name] = f.get_tensor(name)
    return out


def _stub(shapes: dict[str, tuple[int, ...]]) -> nn.Module:
    """A meta-device module whose bf16 parameters carry the given dotted names."""
    root = nn.Module()
    for name, shape in shapes.items():
        *path, leaf = name.split(".")
        module = root
        for part in path:
            if not hasattr(module, part):
                module.add_module(part, nn.Module())
            module = getattr(module, part)
        param = torch.empty(shape, dtype=torch.bfloat16, device="meta")
        module.register_parameter(leaf, nn.Parameter(param, requires_grad=False))
    return root


def _gather(packed: torch.Tensor, *index: torch.Tensor) -> torch.Tensor:
    """Index a repacked part; uint16 parts are read through an int16 view."""
    if packed.dtype == torch.uint16:
        return packed.view(torch.int16)[index].view(torch.uint8)
    return packed[index]


def _unpack(parts: dict[str, torch.Tensor], i: int, d: int) -> dict[str, torch.Tensor]:
    """Rebuild the six stored tensors from the documented repacked layout."""
    ni, nk, kd = i // G // 512, d // 512, i // G // 512
    out = {}
    for name, proj in (("w1", 0), ("w3", 1)):
        r = torch.arange(i)[:, None]
        g, n, c = r // (i // G), (r % (i // G)) // 512 + proj * ni, r % 512
        j = torch.arange(d // 4)[None, :]
        out[f"{name}.weight_packed"] = _gather(
            parts["gate_up_weight"], g, n, j % 128, (j // 128) * 512 + c
        ).reshape(i, d // 2)
        j = torch.arange(d // 32)[None, :]
        out[f"{name}.weight_scale"] = _gather(
            parts["gate_up_scale"], g, j % 16, (n * nk + j // 16) * 512 + c
        )
    r = torch.arange(d)[:, None]
    dd, c = r // 512, r % 512
    j = torch.arange(i // 4)[None, :]
    g, jj = j // (i // G // 4), j % (i // G // 4)
    out["w2.weight_packed"] = _gather(
        parts["down_weight"], g, dd, jj % 128, (jj // 128) * 512 + c
    ).reshape(d, i // 2)
    j = torch.arange(i // 32)[None, :]
    g, jj = j // (i // G // 32), j % (i // G // 32)
    out["w2.weight_scale"] = _gather(
        parts["down_scale"], g, jj % 16, (dd * kd + jj // 16) * 512 + c
    )
    return out


@pytest.fixture(scope="module")
def text_config():
    """The parsed text config, built from vLLM's HF config for the checkpoint."""
    from vllm.transformers_utils.config import get_config

    hf_config = get_config(str(CHECKPOINT), trust_remote_code=False)
    return KimiK3Config.from_configs(hf_config).text_config


@pytest.fixture(scope="module")
def loaded_layer(text_config) -> nn.Module:
    """Layer 1's F32 kinds and latent up weight, loaded through load_weights."""
    shapes = {
        f"model.layers.{LAYER}.self_attn.q_conv1d.weight": (12288, 1, 4),
        f"model.layers.{LAYER}.self_attn.k_conv1d.weight": (12288, 1, 4),
        f"model.layers.{LAYER}.self_attn.v_conv1d.weight": (12288, 1, 4),
        f"model.layers.{LAYER}.self_attn.A_log": (text_config.kda_num_heads,),
        f"model.layers.{LAYER}.self_attn.dt_bias": (12288,),
        f"model.layers.{LAYER}.self_attn.o_norm.weight": (128,),
        f"{MOE}.gate.e_score_correction_bias": (text_config.num_experts,),
        UP: (text_config.hidden_size, text_config.routed_expert_hidden_size),
    }
    model = _stub(shapes)
    load_weights(model, text_config, CHECKPOINT, torch.device("cpu"), pipelined=False)
    return model


def test_unpack_of_repack_equals_stored_bytes(text_config):
    i, d = text_config.moe_intermediate_size, text_config.routed_expert_hidden_size
    mismatched, tensors = 0, 0
    for expert in (0, 1):
        keys = {
            f"{w}.{kind}": f"{PREFIX}{MOE}.experts.{expert}.{w}.{kind}"
            for w in ("w1", "w2", "w3")
            for kind in ("weight_packed", "weight_scale")
        }
        stored = _read(list(keys.values()))
        stored = {short: stored[key] for short, key in keys.items()}
        parts = repack_expert(
            stored["w1.weight_packed"],
            stored["w1.weight_scale"],
            stored["w3.weight_packed"],
            stored["w3.weight_scale"],
            stored["w2.weight_packed"],
            stored["w2.weight_scale"],
            num_programs=G,
        )
        assert parts["gate_up_weight"].shape == (G, 6, 128, 7 * 512)
        assert parts["down_weight"].shape == (G, 7, 128, 3 * 512)
        for short, rebuilt in _unpack(parts, i, d).items():
            assert rebuilt.shape == stored[short].shape
            mismatched += int((rebuilt != stored[short]).sum())
            tensors += 1
    assert (tensors, mismatched) == (12, 0)


def test_config_build_passes_quantization_check_and_plugin_gate():
    from vllm.engine.arg_utils import EngineArgs
    from vllm.platforms import current_platform

    vllm_config = EngineArgs(
        model=str(CHECKPOINT), max_model_len=4096
    ).create_engine_config()
    assert vllm_config.model_config.quantization == "mxfp4"
    current_platform._validate_quantization_config(vllm_config)


def test_f32_kinds_load_as_f32(loaded_layer, text_config):
    params = {
        name: p for name, p in loaded_layer.named_parameters() if name.endswith(F32_SUFFIXES)
    }
    assert len(params) == 7
    stored = _read([PREFIX + name for name in params])
    for name, param in params.items():
        expected = stored[PREFIX + name]
        if name.endswith("A_log"):
            expected = expected.reshape(-1)[: text_config.kda_num_heads]
        assert param.dtype == torch.float32, name
        assert torch.equal(param, expected), name


def test_folded_latent_up_within_one_bf16_rounding(loaded_layer):
    folded = dict(loaded_layer.named_parameters())[UP]
    stored = _read([PREFIX + UP, PREFIX + NORM])
    product = stored[PREFIX + UP].float() * stored[PREFIX + NORM].float()
    assert folded.dtype == torch.bfloat16
    over = (folded.float() - product).abs() > 2.0**-8 * product.abs()
    assert int(over.sum()) == 0
