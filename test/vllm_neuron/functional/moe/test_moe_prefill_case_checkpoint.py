# SPDX-License-Identifier: Apache-2.0
"""Where ``moe_prefill_case`` reads its weights, and what it does when they are absent.

The checkpoint directory is ``VLLM_NEURON_GLM5NEXT_CHECKPOINT_DIR`` (``vllm_neuron/envs.py``),
read through ``test/vllm_neuron/artifacts.py``, as ``glue_case`` reads it. A ``_Source`` that
asks for the checkpoint either reads it or skips the calling test with the path it resolved;
it never draws random weights in its place. Random weights come only from
``_Source(use_checkpoint=False)``, which asks for them.

The tiny checkpoint built here (one leaf, one shard) pins the reading path on any host.
"""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from test.vllm_neuron import artifacts
from test.vllm_neuron.functional.moe import moe_prefill_case as case

KNOB = artifacts.CHECKPOINT_KNOB
#: A leaf of the tiny checkpoint, and the shape it is stored at (fp32).
LEAF = f"model.language_model.layers.{case.MOE_LAYER}.hc_attn_base"
LEAF_SHAPE = (24,)


def _tiny_checkpoint(directory):
    """A checkpoint directory holding :data:`LEAF` in one shard, and the stored value."""
    value = torch.arange(LEAF_SHAPE[0], dtype=torch.float32) / 8.0
    shard = "model-00001-of-00001.safetensors"
    save_file({LEAF: value}, str(directory / shard))
    (directory / artifacts.CHECKPOINT_INDEX).write_text(
        json.dumps({"metadata": {}, "weight_map": {LEAF: shard}})
    )
    return value


def test_absent_directory_skips_naming_the_path(monkeypatch, tmp_path):
    """A directory that does not exist: the test skips, and says which one and the knob."""
    missing = tmp_path / "nonexistent" / "ckpt"
    monkeypatch.setenv(KNOB, str(missing))
    with pytest.raises(pytest.skip.Exception) as skipped:
        case._Source(seed=3)
    assert str(missing / artifacts.CHECKPOINT_INDEX) in str(skipped.value)
    assert KNOB in str(skipped.value)


def test_directory_without_index_skips_naming_the_index(monkeypatch, tmp_path):
    """A directory without ``model.safetensors.index.json`` is not a checkpoint."""
    monkeypatch.setenv(KNOB, str(tmp_path))
    with pytest.raises(pytest.skip.Exception) as skipped:
        case._Source(seed=3)
    assert str(tmp_path / artifacts.CHECKPOINT_INDEX) in str(skipped.value)


def test_the_layer_builder_skips_before_it_builds_anything(monkeypatch, tmp_path):
    missing = tmp_path / "absent"
    monkeypatch.setenv(KNOB, str(missing))
    with pytest.raises(pytest.skip.Exception, match=str(missing)):
        case.moe_layer(model=None)


def test_present_checkpoint_is_read(monkeypatch, tmp_path):
    """A present checkpoint: ``real`` is True and a leaf comes back as stored, not drawn."""
    stored = _tiny_checkpoint(tmp_path)
    monkeypatch.setenv(KNOB, str(tmp_path))
    src = case._Source(seed=3)
    assert src.real is True
    torch.testing.assert_close(src.get(LEAF, LEAF_SHAPE, torch.float32, scale=0.5), stored, atol=0, rtol=0)


def test_leaf_absent_from_a_present_checkpoint_raises(monkeypatch, tmp_path):
    """A leaf the checkpoint does not hold is an error naming it, not a random draw."""
    _tiny_checkpoint(tmp_path)
    monkeypatch.setenv(KNOB, str(tmp_path))
    src = case._Source(seed=3)
    leaf = f"model.language_model.layers.{case.MOE_LAYER}.no_such_leaf"
    with pytest.raises(KeyError, match=leaf):
        src.get(leaf, LEAF_SHAPE, torch.float32)


def test_random_source_is_asked_for_and_ignores_the_knob(monkeypatch, tmp_path):
    """``use_checkpoint=False`` is the one random path: it reads no checkpoint."""
    monkeypatch.setenv(KNOB, str(tmp_path / "absent"))
    src = case._Source(seed=5, use_checkpoint=False)
    assert src.real is False
    again = case._Source(seed=5, use_checkpoint=False)
    torch.testing.assert_close(src.get(LEAF, LEAF_SHAPE, torch.float32),
                               again.get(LEAF, LEAF_SHAPE, torch.float32), atol=0, rtol=0)
