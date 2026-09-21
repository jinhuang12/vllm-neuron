# SPDX-License-Identifier: Apache-2.0
"""Tests for block-FP8 method recognition from the checkpoint's quantization config.

The published config declares ``weight_block_size [128, 128]``, and the model has to
resolve that into a block-FP8 quantization method reporting the same pair, while a
well-formed but unsupported block size raises.

The readings are taken off the resolved method rather than off the parsed spec: the spec
carries the block shape whether or not a method resolves, so a spec-side assertion would
hold without any recognition path at all. ``[64, 64]`` is the unsupported case rather
than a malformed shape, because the parser already rejects a malformed one and that
would measure the parser instead of the resolver.

The fixture is ``fixtures/config.json``, the trimmed published config, pinned by digest
here as well as beside the file. Its ``quantization_config`` is the checkpoint's own:
``quant_method "fp8"``, ``activation_scheme "dynamic"``, ``weight_block_size
[128, 128]``. Nothing here reaches the network or touches a weight tensor.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from vllm_neuron.model.glm5_next import quantization as qz
from vllm_neuron.model.glm5_next.config import Glm5NextConfig


def _impl():
    """Import the modeling module inside a test body, never at import time.

    Another test in this directory asserts that the modeling module is absent from
    ``sys.modules``, which is what shows the architecture class can be looked up without
    allocating a 45-layer stack. This file sorts before it, so a module-level import here
    would populate ``sys.modules`` first and break that assertion. ``quantization`` stays
    at module level: it is a different module, and this file needs it at class scope.
    """
    from vllm_neuron.model.glm5_next import model_fp8

    return model_fp8


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "config.json"

#: The block shape the published config declares.
EXPECTED_BLOCK_SHAPE = (128, 128)

#: Well formed, so the parser accepts it, and unsupported, so the resolver must not.
#: Both halves matter: see the module docstring.
UNSUPPORTED_BLOCK_SIZE = [64, 64]


def _raw() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


@pytest.fixture(scope="module")
def raw() -> dict:
    return _raw()


def _config_from(raw_dict: dict) -> Glm5NextConfig:
    return Glm5NextConfig.from_configs(copy.deepcopy(raw_dict))


def _quant_config_from(raw_dict: dict):
    """The recognition path end to end, the way a call site walks it.

    ``quantization_config`` to ``Glm5NextConfig`` to ``QuantizationSpec`` to
    ``Glm5NextQuantConfig``, with nothing hand-fed at any hop. The return type is
    unannotated on purpose: naming the class would need the modeling module at import
    time, which is what ``_impl`` exists to avoid.
    """
    return _impl().Glm5NextQuantConfig.from_model_config(_config_from(raw_dict))


# ---------------------------------------------------------------------------
# A block-FP8 method resolves from the published config
# ---------------------------------------------------------------------------
def test_block_quant_recognition_resolves_a_block_fp8_method(raw) -> None:
    """The pinned checkpoint config resolves a block-FP8 quantization method.

    ``resolve_quant_method``'s block-FP8 arm, reached through
    ``Glm5NextQuantConfig.from_model_config``, is what converts a parsed spec into a
    method. The same config with its ``quantization_config`` removed must resolve none,
    so a method is not something this path returns unconditionally.
    """
    quant_config = _quant_config_from(raw)
    method = quant_config.get_quant_method(layer_index=3, prefix="mlp.experts")

    assert method is not None, "the pinned checkpoint config must resolve a method"
    assert isinstance(method, qz.BlockFp8QuantMethod)
    assert method.scheme is qz.QuantScheme.FP8_BLOCK_DYNAMIC
    assert quant_config.is_block_quantized is True

    # With no quantization_config at all, no method resolves.
    unquantized_raw = copy.deepcopy(raw)
    unquantized_raw.pop("quantization_config", None)
    control_config = _quant_config_from(unquantized_raw)
    control_method = control_config.get_quant_method(
        layer_index=3, prefix="mlp.experts"
    )

    assert control_method is None, (
        "a config with no quantization block resolved a method"
    )
    assert control_config.spec is None
    assert control_config.is_block_quantized is False

    # A spec whose scheme is NONE reaches the other early return, so both are
    # exercised separately.
    none_spec = qz.QuantizationSpec(
        linear_scheme=qz.QuantScheme.NONE,
        kv_cache_scheme=qz.QuantScheme.NONE,
    )
    assert qz.resolve_quant_method(none_spec) is None


# ---------------------------------------------------------------------------
# The method reports the checkpoint's block shape exactly
# ---------------------------------------------------------------------------
def test_block_quant_recognition_reports_the_block_shape_exactly(
    raw, monkeypatch
) -> None:
    """The resolved method reports ``(block_h, block_w) == (128, 128)``.

    Read off the method's own fields, which ``resolve_quant_method`` populates from
    ``spec.weight_block_size``, and not off the spec: the spec carries the pair whether
    or not a method resolves, so it is asserted here only to record that the two agree.

    With the supported set widened to admit ``(256, 256)``, a ``[256, 256]`` config must
    report ``(256, 256)``, which is what shows the pair is read from the checkpoint
    rather than being a constant.
    """
    quant_config = _quant_config_from(raw)
    method = quant_config.get_quant_method(layer_index=3, prefix="mlp.experts")
    assert method is not None

    pair = (method.block_h, method.block_w)

    assert pair == EXPECTED_BLOCK_SHAPE
    assert method.block_shape == EXPECTED_BLOCK_SHAPE
    assert quant_config.block_shape == EXPECTED_BLOCK_SHAPE
    assert isinstance(method.block_h, int) and isinstance(method.block_w, int)
    # Recorded rather than load-bearing: the two views agree.
    assert tuple(quant_config.spec.weight_block_size) == EXPECTED_BLOCK_SHAPE

    # Widen the supported set, and move the checkpoint's own shape with it.
    monkeypatch.setattr(
        qz,
        "SUPPORTED_WEIGHT_BLOCK_SIZES",
        frozenset({(128, 128), (256, 256)}),
    )
    widened_raw = copy.deepcopy(raw)
    widened_raw["quantization_config"]["weight_block_size"] = [256, 256]
    control_method = _quant_config_from(widened_raw).get_quant_method()
    control_pair = (control_method.block_h, control_method.block_w)

    assert control_pair == (256, 256), (
        "the reported pair did not move with the checkpoint, so it is a constant"
    )
    assert control_pair != pair


# ---------------------------------------------------------------------------
# An unsupported weight_block_size raises
# ---------------------------------------------------------------------------
def test_block_quant_recognition_rejects_an_unsupported_block_size(raw) -> None:
    """A well-formed but unsupported ``weight_block_size`` raises by name.

    The check lives in ``BlockFp8QuantMethod.__post_init__`` rather than in the resolver,
    so no construction path can bypass it, and both entry points are driven here.
    ``[64, 64]`` rather than a malformed shape, because the parser already rejects a
    malformed one. The supported ``[128, 128]`` through the same call path must not
    raise, or a resolver that refused everything would pass too.
    """
    unsupported_raw = copy.deepcopy(raw)
    unsupported_raw["quantization_config"]["weight_block_size"] = (
        UNSUPPORTED_BLOCK_SIZE
    )
    with pytest.raises(qz.UnsupportedWeightBlockSize) as excinfo:
        _quant_config_from(unsupported_raw)

    # A named error, and still a ValueError, so existing callers keep working.
    assert isinstance(excinfo.value, ValueError)
    assert "64" in str(excinfo.value)

    # The parser stays permissive -- the spec for the same config builds -- so the
    # refusal is attributable to method resolution alone.
    permissive_spec = qz.QuantizationSpec.from_hf_quantization_config(
        unsupported_raw["quantization_config"]
    )
    assert tuple(permissive_spec.weight_block_size) == (64, 64)

    # The direct constructor reaches the same guard.
    with pytest.raises(qz.UnsupportedWeightBlockSize):
        qz.BlockFp8QuantMethod(block_h=64, block_w=64, activation_scheme="dynamic")

    # The supported shape through the same call path does not raise.
    control_method = _quant_config_from(raw).get_quant_method()
    assert control_method is not None
    assert control_method.block_shape == EXPECTED_BLOCK_SHAPE
