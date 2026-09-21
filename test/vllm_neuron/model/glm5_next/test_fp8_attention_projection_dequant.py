"""A blockwise-FP8 attention projection dequantises to its checkpoint value.

Under the trn2 squeeze the load path halves fp8 weight bytes and doubles their block
scales, so the product the projection prep dequantises -- ``dequantise_blockwise`` over
the loaded weight and the loaded scale grid -- must still be the checkpoint's own
product: the bytes as fp32 times the scale grid broadcast over 128x128 blocks. Halving an
e4m3 value is exact at or above 2^-5 and rounds to the nearest 2^-9 quantum below it, so
the product is exact in the normal band and within one quantum times the block scale
elsewhere; with the squeeze off it is bitwise the checkpoint's.

Three readings, all through the weight map the model builds and the loaders it picks: a
synthetic weight and grid written under one sparse-attention leaf's real checkpoint keys,
the same pair with the squeeze off, and, when ``GLM53F_CHECKPOINT_DIR`` names a
checkpoint, that leaf out of the checkpoint itself.
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import NamedTuple

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
QUANTUM = 2.0 ** -9
NORMAL_BAND = 2.0 ** -5
#: fp32 rounding of the two products, for the case where the difference sits exactly on
#: the bound.
EVALUATION_SLACK = 2.0 ** -20


class Reading(NamedTuple):
    """One leaf through the model's loaders, with the reference value and its inputs."""

    product: torch.Tensor
    reference: torch.Tensor
    weight: torch.Tensor
    scale: torch.Tensor
    label: str
    message: str


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


def _grid(scale: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """The block scale grid broadcast over the weight's blocks, cut to the weight's shape."""
    return scale.float().repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)[: shape[0], : shape[1]]


def _reference(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """The checkpoint's own value: bytes as fp32 times the grid broadcast over the blocks."""
    return weight.float() * _grid(scale, weight.shape)


def _within_one_quantum(reading: Reading) -> tuple[bool, str]:
    """Whether the product is the reference up to the halving's rounding.

    Ratio 1, within one quantum times the block scale per element, and exact at or above
    2^-5. Returns the verdict and a line describing the comparison for a failure message.
    """
    diff = (reading.product - reading.reference).abs()
    bound = QUANTUM * _grid(reading.scale, diff.shape)
    normal = reading.weight.float().abs() >= NORMAL_BAND
    mask = reading.reference != 0
    ratio = (reading.product[mask] / reading.reference[mask]).median().item()
    within = bool(torch.all(diff <= bound * (1.0 + EVALUATION_SLACK)))
    exact = bool(torch.equal(reading.product[normal], reading.reference[normal]))
    row = (
        f"moved {int((diff > 0).sum())}/{diff.numel()}, "
        f"max_abs_diff {diff.max().item():.6g}, max_bound {bound.max().item():.6g}, "
        f"within_one_quantum {within}, slack {EVALUATION_SLACK:.3g}, "
        f"normal_band_exact {exact} over {int(normal.sum())} elements, "
        f"ratio {ratio:.6f}"
    )
    return abs(ratio - 1.0) <= 1e-4 and within and exact, row


def _read(handle, mapping: dict, label: str) -> Reading:
    """Dequantise the leaf the way the projection prep does and compare with the reference."""
    weight_name, weight_keys, scale_name, scale_keys = _entries(mapping)
    raw_weight, raw_scale = handle.get_slice(weight_keys[0])[:], handle.get_slice(scale_keys[0])[:]
    weight, weight_via = _load(handle, weight_name, weight_keys)
    scale, scale_via = _load(handle, scale_name, scale_keys)
    product = loaders.dequantise_blockwise(weight, scale)
    reference = _reference(raw_weight, raw_scale)
    mask = reference != 0
    ratio = (product[mask] / reference[mask]).median().item() if mask.any() else float("nan")
    message = (
        f"{label}: squeeze {loaders.needs_240_downscale()}, weight {weight_keys[0]}, "
        f"shape {tuple(raw_weight.shape)}, weight loader {weight_via}, "
        f"scale loader {scale_via}, bytes unchanged "
        f"{torch.equal(weight.view(torch.uint8), raw_weight.view(torch.uint8))}, "
        f"scale ratio {(scale.float() / raw_scale.float()).median().item():.4f}, "
        f"product over reference {ratio:.4f}, "
        f"max_abs_diff {(product - reference).abs().max().item():.6g}, "
        f"bitwise equal {torch.equal(product, reference)}"
    )
    return Reading(product, reference, raw_weight, raw_scale, label, message)


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


def test_the_loaded_pair_dequantises_to_the_checkpoint_value_under_the_squeeze(
    synthetic, monkeypatch
):
    """With the squeeze on, the product is the checkpoint's value to one fp8 quantum.

    Exact at or above 2^-5, and within one quantum times the block scale below it.
    """
    monkeypatch.setattr(loaders, "needs_240_downscale", lambda: True)
    handle, mapping = synthetic
    reading = _read(handle, mapping, "synthetic")
    ok, row = _within_one_quantum(reading)
    assert ok, f"{reading.message}; {row}"


def test_without_the_squeeze_the_loaded_pair_is_the_checkpoint_value(synthetic, monkeypatch):
    """With the squeeze off, the same route reproduces the checkpoint's value bitwise."""
    monkeypatch.setattr(loaders, "needs_240_downscale", lambda: False)
    handle, mapping = synthetic
    reading = _read(handle, mapping, "synthetic")
    assert torch.equal(reading.product, reading.reference), reading.message


def test_the_real_leaf_dequantises_to_its_checkpoint_value_under_the_squeeze(monkeypatch):
    """The checkpoint's own leaf through the same route, under the same bound.

    Skipped unless ``GLM53F_CHECKPOINT_DIR`` names a checkpoint on disk.
    """
    checkpoint = os.environ.get("GLM53F_CHECKPOINT_DIR")
    if not checkpoint:
        pytest.skip(
            "GLM53F_CHECKPOINT_DIR names no checkpoint; the synthetic readings stand alone"
        )
    monkeypatch.setattr(loaders, "needs_240_downscale", lambda: True)
    mapping = _mapping(pathlib.Path(checkpoint))
    _, weight_keys, _, _ = _entries(mapping)
    shard = json.load(open(os.path.join(checkpoint, "model.safetensors.index.json")))["weight_map"][weight_keys[0]]
    with safe_open(os.path.join(checkpoint, shard), "pt") as handle:
        reading = _read(handle, mapping, "checkpoint")
    ok, row = _within_one_quantum(reading)
    assert ok, f"{reading.message}; {row}"


def _downscales(name: str, keys: list[str]) -> bool:
    """Whether the loader the model picks for these keys wraps the blockwise fp8 downscale."""
    loader = loaders.loader_for_mapped_keys(keys, param_name=name, owner=None, geometry=None)
    transform = getattr(loader, "transform", None)
    return "wrap_with_blockwise_fp8_downscale" in getattr(transform, "__qualname__", "")


def test_every_scaled_attention_weight_downscales_and_the_other_families_hold():
    """Every scaled attention weight takes a downscaling loader.

    The other families keep their own kinds: 44 attention weights, 9 dense, 126 shared
    and 126 routed banks.
    """
    mapping = _mapping(FIXTURE)
    as_list = lambda keys: [keys] if isinstance(keys, str) else list(keys)
    kinds = {name: loaders.classify_mapped_keys(as_list(keys)) for name, keys in mapping.items()}
    attention = sorted(
        name for name in mapping
        if ".self_attn." in name and name.endswith("_weight")
        and name.rsplit(".", 1)[-1][: -len("_weight")] in loaders.DSA_SCALED_PROJECTIONS
        and f"{name[: -len('_weight')]}_{loaders.FP8_SCALE_SUFFIX}" in mapping
    )
    downscaled = [name for name in attention if kinds[name] == "quantised_weight" and _downscales(name, as_list(mapping[name]))]
    dense = [n for n in mapping if ".mlp." in n and "experts" not in n and kinds[n] == "quantised_weight"]
    shared = [n for n in mapping if "shared_experts" in n and kinds[n] == "quantised_weight"]
    routed = [n for n in mapping if kinds[n] == "stacked_bank"]
    other = [n for n in mapping if kinds[n] == "quantised_weight" and n not in set(attention) | set(dense) | set(shared)]
    assert len(attention) == 44, f"the fixture maps {len(attention)} scaled attention weights, not 44"
    assert len(downscaled) == 44, f"{44 - len(downscaled)} scaled attention weights miss the downscaling loader: {sorted(set(attention) - set(downscaled))[:4]}"
    assert (len(dense), len(shared), len(routed), len(other)) == (9, 126, 126, 0)
    assert all(_downscales(n, as_list(mapping[n])) for n in dense + shared)
