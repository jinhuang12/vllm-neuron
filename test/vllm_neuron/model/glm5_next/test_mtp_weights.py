# SPDX-License-Identifier: Apache-2.0
"""The draft layer's weights against the real checkpoint's safetensors headers.

The multi-token-prediction (MTP) draft layer of GLM-5.3-Flash sits one past the main
stack (``layers.45``): a full decoder layer (sparse attention + indexer, the routed
expert bank, the shared expert, two norms) plus ``enorm``, ``hnorm``, ``eh_proj`` and
``shared_head.norm``, and no multi-hyper-connection tensors. With the shadow-draft
knob on, the root model builds ``self.mtp`` and the weight map claims every one of the
layer's keys; with the knob off neither happens and the map is the map it was at
82bee3b.

Ground truth here is the published checkpoint itself, read-only and headers only: the
8-byte length prefix and the JSON header of each shard that holds a draft-layer or
sibling-layer tensor. No tensor byte is read. Every expectation is derived from the
published config and headers -- the draft layer's index is the stack's length, its
sibling is the stack's last sparse-attention layer, and the parameter population is
whatever the head declares -- except the knob-off digest, which pins the map as it was
before the draft layer existed. The tests that need the checkpoint skip when it is not
on this machine; the knob tests run anywhere."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest
import torch

from vllm_neuron.model.glm5_next import factory, model_fp8
from vllm_neuron.model.glm5_next import mtp
from vllm_neuron.model.glm5_next import weight_loaders_fp8 as _fp8_module
from vllm_neuron.model.glm5_next.config import DSA_LAYER_TYPE, Glm5NextConfig
from vllm_neuron.model.glm5_next.model_fp8 import (
    Glm5NextForConditionalGeneration,
    _shard_geometry_for,
)
from vllm_neuron.model.glm5_next.weight_loaders_fp8 import (
    MAPPED_KEY_SCALE_GRID,
    build_weight_mappings,
    classify_mapped_keys,
    scale_keys,
)

from test.vllm_neuron import artifacts

FIXTURES_DIR = Path(__file__).parent / "fixtures"
REAL_CONFIG_PATH = FIXTURES_DIR / "hf-config.json"

MTP_KNOB = "VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT"
CKPT_LAYER_PREFIX = "model.language_model.layers."

#: The serving line's degrees, the inputs the geometry test builds the tree at:
#: TP=64 ranks, EP=16 expert-parallel groups.
WORLD = 64
EP_DEGREE = 16

#: The knob-off map over the published config at 82bee3b: entry count and the sha256
#: of its sorted compact JSON (``json.dumps(mappings, sort_keys=True,
#: separators=(",", ":"))``). Pinned before the draft layer existed, so a non-draft
#: entry that moves fails here by digest; regenerate with the same expression.
BASELINE_MAP_ENTRIES = 1_416
BASELINE_MAP_SHA256 = "2ced1a480ec43beb464ee5e7b0c22ec627e1e7d7af8d4aafdbd5ca93ace8710b"

HEADER_DTYPES = {
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F8_E4M3": torch.float8_e4m3fn,
}

def checkpoint_root() -> Path:
    """The served checkpoint ``VLLM_NEURON_GLM5NEXT_CHECKPOINT_DIR`` names
    (``test/vllm_neuron/artifacts.py``); skips the calling test by name, with the resolved
    path, where it holds no index."""
    return artifacts.require_checkpoint()


def _read_header(path: Path) -> dict:
    """The safetensors header of one shard: the JSON after the 8-byte length prefix."""
    with path.open("rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        return json.loads(handle.read(length))


@pytest.fixture(scope="module")
def real_config() -> Glm5NextConfig:
    return Glm5NextConfig.from_configs(json.loads(REAL_CONFIG_PATH.read_text()))


@pytest.fixture(scope="module")
def draft_layer(real_config) -> int:
    """The draft layer's index: the first one the stack does not use."""
    return int(real_config.text_config.num_hidden_layers)


@pytest.fixture(scope="module")
def sibling_layer(real_config) -> int:
    """The stack's last sparse-attention layer: the draft layer's closest sibling."""
    return max(
        index
        for index, kind in enumerate(real_config.text_config.layer_types)
        if kind == DSA_LAYER_TYPE
    )


@pytest.fixture(scope="module")
def real_headers(
    draft_layer, sibling_layer
) -> dict[str, tuple[torch.dtype, tuple[int, ...]]]:
    """``{key: (dtype, shape)}`` for every draft-layer and sibling tensor, headers only;
    skips by name where the served checkpoint is absent."""
    root = checkpoint_root()
    weight_map = json.loads((root / artifacts.CHECKPOINT_INDEX).read_text())["weight_map"]
    wanted = {
        key: shard
        for key, shard in weight_map.items()
        if key.startswith(f"{CKPT_LAYER_PREFIX}{draft_layer}.")
        or key.startswith(f"{CKPT_LAYER_PREFIX}{sibling_layer}.")
    }
    headers: dict[str, tuple[torch.dtype, tuple[int, ...]]] = {}
    for shard in sorted(set(wanted.values())):
        for key, entry in _read_header(root / shard).items():
            if key in wanted:
                headers[key] = (HEADER_DTYPES[entry["dtype"]], tuple(entry["shape"]))
    assert set(headers) == set(wanted), "a wanted key is missing from its shard header"
    return headers


def _mappings_as_load_weights_builds_them(config: Glm5NextConfig):
    """The same call ``load_weights`` makes, so the map under test is the production map."""
    return build_weight_mappings(
        config.text_config,
        quantised=config.is_block_quantized,
        modules_to_not_convert=tuple(config.modules_to_not_convert or ()),
    )


def _root(monkeypatch, config: Glm5NextConfig, *, knob: str | None, world: int = 1):
    if knob is None:
        monkeypatch.delenv(MTP_KNOB, raising=False)
    else:
        monkeypatch.setenv(MTP_KNOB, knob)
    monkeypatch.setattr(model_fp8, "_resolve_world_size", lambda: world)
    # ``Glm5NextRoutedExperts`` imports the getter from the factory at call time.
    monkeypatch.setattr(
        factory,
        "_resolve_ep_degree",
        lambda given=None: EP_DEGREE if given is None else int(given),
    )
    return Glm5NextForConditionalGeneration(config)


def _layer_keys(mappings, layer: int) -> dict[str, int]:
    """Reference count per checkpoint key on one layer."""
    prefix = f"{CKPT_LAYER_PREFIX}{layer}."
    counts: dict[str, int] = {}
    for value in mappings.values():
        for key in value if isinstance(value, list) else [value]:
            if key.startswith(prefix):
                counts[key] = counts.get(key, 0) + 1
    return counts


def _header_keys(real_headers, layer: int) -> set[str]:
    return {k for k in real_headers if k.startswith(f"{CKPT_LAYER_PREFIX}{layer}.")}


def _leaf(key: str, layer: int) -> str:
    return key[len(f"{CKPT_LAYER_PREFIX}{layer}.") :]


def _mtp_params(root) -> list[str]:
    prefix = _fp8_module.MTP_ROOT_ATTR + "."
    return [name for name in root.declared_parameter_names() if name.startswith(prefix)]


# --------------------------------------------------------------------------- #
# The knob.
# --------------------------------------------------------------------------- #


def test_knob_off_builds_no_head_and_the_map_is_the_82bee3b_map(
    monkeypatch, real_config, draft_layer
) -> None:
    """Unset knob: ``self.mtp is None``, no ``mtp.*`` parameter, no draft-layer key, pinned digest."""
    root = _root(monkeypatch, real_config, knob=None)
    assert root.mtp is None
    assert _mtp_params(root) == []

    mappings = _mappings_as_load_weights_builds_them(real_config)
    assert _layer_keys(mappings, draft_layer) == {}
    assert len(mappings) == BASELINE_MAP_ENTRIES
    blob = json.dumps(mappings, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(blob).hexdigest() == BASELINE_MAP_SHA256


def test_knob_zero_is_the_same_as_unset(monkeypatch, real_config, draft_layer) -> None:
    root = _root(monkeypatch, real_config, knob="0")
    assert root.mtp is None
    assert _layer_keys(_mappings_as_load_weights_builds_them(real_config), draft_layer) == {}


def test_knob_on_builds_the_head_and_maps_every_declared_mtp_parameter(
    monkeypatch, real_config
) -> None:
    """Knob 5: ``self.mtp`` is the draft head and every parameter it declares has a map entry.

    The second half is what keeps ``load_weights`` from leaving an unfilled placeholder:
    a declared ``mtp.*`` leaf with no entry would be looked up under its own name.
    """
    from vllm_neuron.model.glm5_next.mtp import Glm5NextMultiTokenPredictor

    root = _root(monkeypatch, real_config, knob="5")
    assert isinstance(root.mtp, Glm5NextMultiTokenPredictor)

    declared = _mtp_params(root)
    assert declared, "the head declares no parameter"
    mappings = _mappings_as_load_weights_builds_them(real_config)
    unmapped = [name for name in declared if name not in mappings]
    assert unmapped == [], unmapped[:8]
    assert [name for name in declared if name.rsplit(".", 1)[-1].startswith("hc_")] == []
    # And every mtp entry names a declared parameter: no entry hangs in the air.
    mapped = [name for name in mappings if name.startswith(_fp8_module.MTP_ROOT_ATTR + ".")]
    assert sorted(mapped) == sorted(declared)
    # Contract C2 spelled out: the map's paths are the head's published ones.
    assert sorted(mapped) == sorted(
        f"{_fp8_module.MTP_ROOT_ATTR}.{name}" for name in mtp.MTP_PARAMETER_NAMES
    )
    # The map with the knob off is this map less the mtp entries.
    monkeypatch.delenv(MTP_KNOB, raising=False)
    off = _mappings_as_load_weights_builds_them(real_config)
    assert {k: v for k, v in mappings.items() if k not in mapped} == off


# --------------------------------------------------------------------------- #
# The real headers.
# --------------------------------------------------------------------------- #


def test_every_draft_layer_header_key_is_mapped_as_often_as_the_sibling_maps_its_leaf(
    monkeypatch, real_config, real_headers, draft_layer, sibling_layer
) -> None:
    """Every header key of the draft layer is claimed; per leaf, exactly as many times as the sibling's leaf."""
    _root(monkeypatch, real_config, knob="5")
    mappings = _mappings_as_load_weights_builds_them(real_config)
    refs_draft = _layer_keys(mappings, draft_layer)
    refs_sibling = _layer_keys(mappings, sibling_layer)
    header_draft = _header_keys(real_headers, draft_layer)
    header_sibling = _header_keys(real_headers, sibling_layer)

    # The headers agree with the derivation: the sibling's leaves, less the six mHC
    # tensors, plus the four draft leaves (1,762 - 6 + 4 = 1,760 on this checkpoint).
    assert {_leaf(k, draft_layer) for k in header_draft} == (
        {_leaf(k, sibling_layer) for k in header_sibling} - set(model_fp8.MHC_LEAVES)
    ) | {ckpt for _, ckpt in _fp8_module.MTP_LEAVES}
    assert set(refs_draft) == header_draft, (
        f"unclaimed {sorted(header_draft - set(refs_draft))[:5]}; "
        f"claimed-but-absent {sorted(set(refs_draft) - header_draft)[:5]}"
    )

    by_leaf_draft = {_leaf(k, draft_layer): n for k, n in refs_draft.items()}
    by_leaf_sibling = {_leaf(k, sibling_layer): n for k, n in refs_sibling.items()}
    for name, count in by_leaf_draft.items():
        expected = by_leaf_sibling.get(name, 1)  # the four draft-only leaves: once
        assert count == expected, f"{name}: referenced {count}x, sibling {expected}x"


def test_every_mtp_parameter_matches_the_header_dtype_and_its_sibling_shard_geometry(
    monkeypatch, real_config, real_headers, draft_layer, sibling_layer
) -> None:
    """Per ``mtp.*`` parameter: placeholder dtype == header dtype; shard geometry == the sibling's for the same leaf; header shapes agree.

    Built at the serving degrees (TP=64, EP=16), so every sharded family has a
    geometry to compare and every shard count divides the header's extent.
    """
    root = _root(monkeypatch, real_config, knob="5", world=WORLD)
    mappings = _mappings_as_load_weights_builds_them(real_config)
    layer_prefix = f"{_fp8_module.MTP_ROOT_ATTR}."
    block_prefix = f"{layer_prefix}{mtp.BLOCK_ATTR}."
    hidden = int(real_config.text_config.hidden_size)
    block = tuple(int(b) for b in real_config.weight_block_size)
    mtp_only_shapes = {
        "enorm_weight": (hidden,),
        "hnorm_weight": (hidden,),
        "eh_proj_weight": (hidden, 2 * hidden),
        "shared_head_norm_weight": (hidden,),
    }
    assert set(mtp_only_shapes) == {param for param, _ in _fp8_module.MTP_LEAVES}

    declared = _mtp_params(root)
    assert declared
    for name in declared:
        keys = mappings[name]
        key_list = [keys] if isinstance(keys, str) else list(keys)
        module_path, _, leaf = name.rpartition(".")
        module = root.get_submodule(module_path)

        # dtype: the placeholder the reader casts to must be the header's dtype of the
        # tensor the loader hands back -- the weight for a weight entry, the grid for a
        # grid entry.
        kind = classify_mapped_keys(key_list)
        carried = (
            key_list
            if kind == MAPPED_KEY_SCALE_GRID
            else [k for k in key_list if k not in scale_keys(key_list)]
        )
        header_dtypes = {real_headers[k][0] for k in carried}
        assert len(header_dtypes) == 1, (name, header_dtypes)
        placeholder = root._placeholder_dtype(keys, param_name=name, mappings=mappings)
        assert placeholder == header_dtypes.pop(), name

        geometry = _shard_geometry_for(module, leaf, WORLD)
        if name.startswith(block_prefix):
            sibling_name = f"model.layers.{sibling_layer}.{name[len(block_prefix):]}"
            sibling_path, _, sibling_leaf = sibling_name.rpartition(".")
            sibling = root.get_submodule(sibling_path)
            assert type(sibling).__name__ == type(module).__name__, (name, sibling_name)
            sibling_geometry = _shard_geometry_for(sibling, sibling_leaf, WORLD)
            assert geometry == sibling_geometry, (name, geometry, sibling_geometry)
            # Header shapes of the same leaf agree between the two layers, key by key.
            sibling_keys = mappings[sibling_name]
            sibling_list = (
                [sibling_keys] if isinstance(sibling_keys, str) else list(sibling_keys)
            )
            assert len(sibling_list) == len(key_list), name
            for draft_key, sibling_key in zip(key_list, sibling_list):
                assert real_headers[draft_key] == real_headers[sibling_key], (
                    draft_key,
                    sibling_key,
                )
            # The shard count divides the sharded extent of every carried tensor.
            if geometry is not None:
                for k in carried:
                    shape = real_headers[k][1]
                    assert shape[geometry.shard_dim] % geometry.num_shards == 0, (
                        k,
                        shape,
                        geometry,
                    )
                    if hasattr(geometry, "shard_size"):
                        # ``shard_size`` counts weight rows; a scale grid holds one
                        # value per quantisation block, so its extent is the weight's
                        # over the block.
                        extent = shape[geometry.shard_dim]
                        if kind == MAPPED_KEY_SCALE_GRID:
                            extent *= block[geometry.shard_dim]
                        assert geometry.shard_size * geometry.num_shards == extent, (
                            k,
                            shape,
                            geometry,
                        )
        else:
            assert name.startswith(layer_prefix), name
            assert leaf in mtp_only_shapes, name
            if leaf == "eh_proj_weight":
                # Row-parallel (functional/mtp/tail_in.py): each rank holds H / world
                # rows and the layer_input slices are all-gathered in rank order.
                assert (geometry.shard_dim, geometry.shard_size, geometry.num_shards) == (
                    0, hidden // WORLD, WORLD,
                ), (name, geometry)
            else:
                assert geometry is None, (name, geometry)
            assert len(key_list) == 1, name
            assert real_headers[key_list[0]][1] == mtp_only_shapes[leaf], (
                name,
                real_headers[key_list[0]],
            )
