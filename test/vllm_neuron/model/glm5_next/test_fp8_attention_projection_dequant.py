"""A blockwise-FP8 attention projection dequantises to its checkpoint value under the trn2 squeeze.

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \\
        python -m pytest -s -rA test/vllm_neuron/model/glm5_next/test_fp8_attention_projection_dequant.py

The trn2 squeeze halves fp8 weight bytes and doubles their block scales, so the product the
projection prep dequantises (``dequantise_blockwise`` over the loaded weight and the loaded scale
grid) must equal the checkpoint's own product: the bytes as fp32 times the scale grid broadcast
over 128x128 blocks. Three readings through the weight map the model builds and the loaders it
picks: a synthetic weight and grid written under one sparse-attention leaf's real checkpoint
keys, the same pair with the squeeze off, and, when ``GLM53F_CHECKPOINT_DIR`` names the
checkpoint, the real leaf itself.
"""

from __future__ import annotations

import json
import os
import pathlib

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from vllm_neuron.model.glm5_next import weight_loaders_fp8 as loaders
from vllm_neuron.model.glm5_next.config import Glm5NextConfig

pytestmark = [pytest.mark.fast, pytest.mark.forked]

FIXTURE = pathlib.Path(__file__).resolve().parent / "fixtures"
LAYER = 3
LEAF = "q_b_proj"
BLOCK = 128


def _mapping(config_dir: pathlib.Path) -> dict:
    """The weight map the model builds from a config directory."""
    config = Glm5NextConfig.from_configs(str(config_dir / "config.json"))
    return loaders.build_weight_mappings(
        config.text_config,
        quantised=config.is_block_quantized,
        modules_to_not_convert=tuple(config.modules_to_not_convert or ()),
    )


def _entries(mapping: dict) -> tuple[str, list[str], str, list[str]]:
    """The weight entry and the scale entry of the chosen leaf: (name, keys, name, keys)."""
    weight = next(n for n in mapping if n.endswith(f"layers.{LAYER}.self_attn.{LEAF}_weight"))
    scale = next(n for n in mapping if n.endswith(f"layers.{LAYER}.self_attn.{LEAF}_{loaders.FP8_SCALE_SUFFIX}"))
    as_list = lambda keys: [keys] if isinstance(keys, str) else list(keys)
    return weight, as_list(mapping[weight]), scale, as_list(mapping[scale])


def _load(handle, name: str, keys: list[str]) -> tuple[torch.Tensor, str]:
    """One parameter through the loader the model picks for its keys, and that loader's name."""
    loader = loaders.loader_for_mapped_keys(keys, param_name=name, owner=None, geometry=None)
    slices = [handle.get_slice(key) for key in keys]
    if loader is None:
        return slices[0][:], "none"
    return loader.load(slices, 0), getattr(loader.transform, "__qualname__", "identity")


def _reference(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """The checkpoint's own value: bytes as fp32 times the grid broadcast over the blocks."""
    grid = scale.float().repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
    return weight.float() * grid[: weight.shape[0], : weight.shape[1]]


def _read(handle, mapping: dict, label: str) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Dequantise the leaf the way the projection prep does and compare with the reference; returns (product, reference, message)."""
    weight_name, weight_keys, scale_name, scale_keys = _entries(mapping)
    raw_weight, raw_scale = handle.get_slice(weight_keys[0])[:], handle.get_slice(scale_keys[0])[:]
    weight, weight_via = _load(handle, weight_name, weight_keys)
    scale, scale_via = _load(handle, scale_name, scale_keys)
    product = loaders.dequantise_blockwise(weight, scale)
    reference = _reference(raw_weight, raw_scale)
    mask = reference != 0
    ratio = (product[mask] / reference[mask]).median().item() if mask.any() else float("nan")
    message = (
        f"{label}|squeeze={loaders.needs_240_downscale()}|weight={weight_keys[0]}|shape={tuple(raw_weight.shape)}"
        f"|weight_loader={weight_via}|scale_loader={scale_via}"
        f"|bytes_unchanged={torch.equal(weight.view(torch.uint8), raw_weight.view(torch.uint8))}"
        f"|scale_ratio={(scale.float() / raw_scale.float()).median().item():.4f}"
        f"|product_over_reference={ratio:.4f}|max_abs_diff={(product - reference).abs().max().item():.6g}"
        f"|bitwise_equal={torch.equal(product, reference)}"
    )
    print(f"FP8DQ|{message}")
    return product, reference, message


@pytest.fixture
def synthetic(tmp_path):
    """A random fp8 weight and grid saved under the leaf's real checkpoint keys; yields (handle, mapping)."""
    mapping = _mapping(FIXTURE)
    _, weight_keys, _, scale_keys = _entries(mapping)
    generator = torch.Generator().manual_seed(7)
    weight = (torch.randn(2 * BLOCK, 3 * BLOCK, generator=generator) * 0.05).to(torch.float8_e4m3fn)
    scale = torch.rand(2, 3, generator=generator) * 0.01 + 1e-3
    path = tmp_path / "leaf.safetensors"
    save_file({weight_keys[0]: weight, scale_keys[0]: scale}, str(path))
    with safe_open(str(path), "pt") as handle:
        yield handle, mapping


def test_the_loaded_pair_dequantises_to_the_checkpoint_value_under_the_squeeze(synthetic, monkeypatch):
    """With the trn2 squeeze on, the product of the loaded weight and the loaded grid is the checkpoint's value, bitwise."""
    monkeypatch.setattr(loaders, "needs_240_downscale", lambda: True)
    handle, mapping = synthetic
    product, reference, message = _read(handle, mapping, "synthetic")
    assert torch.equal(product, reference), message


def test_without_the_squeeze_the_loaded_pair_is_the_checkpoint_value(synthetic, monkeypatch):
    """With the squeeze off, the same route reproduces the checkpoint's value bitwise."""
    monkeypatch.setattr(loaders, "needs_240_downscale", lambda: False)
    handle, mapping = synthetic
    product, reference, message = _read(handle, mapping, "synthetic")
    assert torch.equal(product, reference), message


def test_the_real_leaf_dequantises_to_its_checkpoint_value_under_the_squeeze(monkeypatch):
    """The checkpoint's own leaf, when GLM53F_CHECKPOINT_DIR names the checkpoint, through the same route."""
    checkpoint = os.environ.get("GLM53F_CHECKPOINT_DIR")
    if not checkpoint:
        pytest.skip("GLM53F_CHECKPOINT_DIR names no checkpoint; the synthetic readings stand alone")
    monkeypatch.setattr(loaders, "needs_240_downscale", lambda: True)
    mapping = _mapping(pathlib.Path(checkpoint))
    _, weight_keys, _, _ = _entries(mapping)
    shard = json.load(open(os.path.join(checkpoint, "model.safetensors.index.json")))["weight_map"][weight_keys[0]]
    with safe_open(os.path.join(checkpoint, shard), "pt") as handle:
        product, reference, message = _read(handle, mapping, "checkpoint")
    assert torch.equal(product, reference), message


def _downscales(name: str, keys: list[str]) -> bool:
    """Whether the loader the model picks for these keys wraps the blockwise fp8 downscale."""
    loader = loaders.loader_for_mapped_keys(keys, param_name=name, owner=None, geometry=None)
    transform = getattr(loader, "transform", None)
    return "wrap_with_blockwise_fp8_downscale" in getattr(transform, "__qualname__", "")


def test_every_scaled_attention_weight_downscales_and_the_other_families_hold():
    """All 44 scaled attention weights take a downscaling loader; dense 9, shared 126 and routed 126 keep their kinds."""
    mapping = _mapping(FIXTURE)
    as_list = lambda keys: [keys] if isinstance(keys, str) else list(keys)
    kinds = {name: loaders.classify_mapped_keys(as_list(keys)) for name, keys in mapping.items()}
    attention = sorted(
        name for name in mapping
        if ".self_attn." in name and name.endswith("_weight")
        and name.rsplit(".", 1)[-1][: -len("_weight")] in loaders.DSA_SCALED_PROJECTIONS
    )
    downscaled = [name for name in attention if kinds[name] == "quantised_weight" and _downscales(name, as_list(mapping[name]))]
    dense = [n for n in mapping if ".mlp." in n and "experts" not in n and kinds[n] == "quantised_weight"]
    shared = [n for n in mapping if "shared_experts" in n and kinds[n] == "quantised_weight"]
    routed = [n for n in mapping if kinds[n] == "stacked_bank"]
    other = [n for n in mapping if kinds[n] == "quantised_weight" and n not in set(attention) | set(dense) | set(shared)]
    print(
        f"FP8DQ|census|attention_downscaled={len(downscaled)}/{len(attention)}|dense={len(dense)}"
        f"|shared={len(shared)}|routed_banks={len(routed)}|other_quantised={len(other)}"
        f"|dense_downscaled={sum(_downscales(n, as_list(mapping[n])) for n in dense)}"
        f"|shared_downscaled={sum(_downscales(n, as_list(mapping[n])) for n in shared)}"
    )
    assert len(attention) == 44, f"the fixture maps {len(attention)} scaled attention weights, not 44"
    assert len(downscaled) == 44, f"{44 - len(downscaled)} scaled attention weights miss the downscaling loader: {sorted(set(attention) - set(downscaled))[:4]}"
    assert (len(dense), len(shared), len(routed), len(other)) == (9, 126, 126, 0)
    assert all(_downscales(n, as_list(mapping[n])) for n in dense + shared)
