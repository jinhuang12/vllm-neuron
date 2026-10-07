# SPDX-License-Identifier: Apache-2.0
"""The MTP draft layer's KV state, model side: ``get_kv_spec`` and ``bind_kv_cache``.

GLM-5.3-Flash's checkpoint ships ``num_nextn_predict_layers`` draft layers past the
``num_hidden_layers`` stack. When the shadow-draft knob is above 0 the root builds the
draft head (``self.mtp``), whose one decoder block is a sparse-attention (DSA) layer, so
it needs the state a trunk DSA layer keeps: a latent page here, and the indexer side
caches and a carrier on the runner's side (``test/vllm_neuron/worker/
test_mtp_state_runner.py``). Read here, every expectation derived from the fixture's
config:

1. knob 0 or unset: the ``LayerSpec`` list is the recorded knob-off list
   (``glm5next_state_lists_knob_off.json`` beside the runner test; what it holds and how
   it is regenerated is in that file's ``about`` entry and the runner test's ``main``);
2. knob 1..5: exactly one more ``LayerSpec``, last, named for index
   ``num_hidden_layers + num_nextn_predict_layers - 1``, with the geometry of the stack's
   DSA layers (the same class built from the same config);
3. ``bind_kv_cache`` keeps that layer's bank last in ``glm5next_layer_banks``, as a
   sparse-family record over the runner's own tensor, and refuses a dict without it or a
   spec that dropped it;
4. a config placing the draft layer where the head's block does not sit is refused by name.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/model/glm5_next/test_mtp_state.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "config.json"
KNOB_OFF_RECORD = (
    Path(__file__).resolve().parents[2] / "worker" / "glm5next_state_lists_knob_off.json"
)
#: A small pool for the cache tensors handed to ``bind_kv_cache``, at the serve lines'
#: hybrid KV block size.
BLOCKS, BLOCK, SLOTS = 6, 128, 3


def _set_k(monkeypatch, value) -> None:
    """Set the shadow-draft knob through its environment variable (read via ``envs``)."""
    from vllm_neuron.model.glm5_next import mtp

    if value is None:
        monkeypatch.delenv(mtp.SHADOW_DRAFT_ENV, raising=False)
    else:
        monkeypatch.setenv(mtp.SHADOW_DRAFT_ENV, str(value))


def _model():
    """The skeleton the knob-off record was taken on: the fixture at one KDA head per rank."""
    from vllm_neuron.model.glm5_next.model_fp8 import Glm5NextForConditionalGeneration

    raw = json.loads(FIXTURE_PATH.read_text())
    raw["text_config"]["linear_attn_config"]["num_heads"] = 1  # per-rank at TP=64
    return Glm5NextForConditionalGeneration.from_configs(raw)


def _layout(model) -> dict:
    from test.vllm_neuron.worker.test_mtp_state_runner import layout

    return layout(model)


def _record(layer) -> dict:
    from test.vllm_neuron.worker.test_mtp_state_runner import spec_record

    return spec_record(layer)


def _kv_caches(spec_layers) -> dict:
    """The runner's ``layer name -> tensors`` dict for these specs, at a small pool."""
    caches = {}
    for index, layer in enumerate(spec_layers):
        if layer.kda_conv_state_shape is not None:
            caches[layer.name] = [
                torch.full((SLOTS, *layer.kda_conv_state_shape), float(index),
                           dtype=layer.kda_conv_state_dtype),
                torch.full((SLOTS, *layer.kda_recurrent_state_shape), float(index),
                           dtype=layer.kda_recurrent_state_dtype),
            ]
        else:
            caches[layer.name] = [
                torch.full((BLOCKS, layer.num_kv_heads, BLOCK, layer.head_size),
                           float(index), dtype=layer.dtype)
            ]
    return caches


# ── 1. knob 0 ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("k", [None, 0])
def test_knob_off_the_spec_list_is_the_recorded_list(k, monkeypatch) -> None:
    record = json.loads(KNOB_OFF_RECORD.read_text())["lines"]
    want = record["bs1"]["kv_spec"]
    assert want == record["bs64"]["kv_spec"]
    _set_k(monkeypatch, k)
    model = _model()
    assert model.mtp is None
    assert [_record(layer) for layer in model.get_kv_spec().layers] == want


def test_knob_off_bind_keeps_the_stack_and_nothing_else(monkeypatch) -> None:
    _set_k(monkeypatch, 0)
    model = _model()
    layers = model.get_kv_spec().layers
    model.bind_kv_cache(_kv_caches(layers))
    stack = _layout(model)["stack"]
    assert [bank["layer_index"] for bank in model.glm5next_layer_banks] == list(range(stack))


# ── 2. knob 1..5: one more DSA spec ──────────────────────────────────────────


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
def test_knob_on_adds_one_dsa_spec(k, monkeypatch) -> None:
    _set_k(monkeypatch, 0)
    off = _model().get_kv_spec().layers
    _set_k(monkeypatch, k)
    model = _model()
    on = model.get_kv_spec().layers
    assert len(on) == len(off) + 1
    assert sum(spec.latent_kv for spec in on) == sum(spec.latent_kv for spec in off) + 1
    assert on[-1].latent_kv


def test_knob_on_the_draft_spec_follows_the_stack_with_the_trunk_dsa_geometry(
    monkeypatch,
) -> None:
    from vllm_neuron.model.glm5_next.mtp import BLOCK_ATTR

    _set_k(monkeypatch, 0)
    off = [_record(layer) for layer in _model().get_kv_spec().layers]
    _set_k(monkeypatch, 5)
    model = _model()
    layout = _layout(model)
    on = [_record(layer) for layer in model.get_kv_spec().layers]
    assert on[: layout["stack"]] == off
    assert len(off) == layout["stack"]
    draft = on[layout["stack"]]
    block = getattr(model.mtp, BLOCK_ATTR)
    assert draft["name"] == f"layers.{layout['draft']}.{block.attention.CACHE_NAME_SUFFIX}"
    assert int(block.layer_idx) == layout["draft"]
    # Every trunk DSA layer has one geometry, and the draft layer's is that one.
    geometry = [{k: v for k, v in one.items() if k != "name"} for one in on]
    assert all(geometry[index] == geometry[layout["draft"]] for index in layout["trunk_dsa"])
    assert [i for i, one in enumerate(on) if one["latent_kv"]] == (
        layout["trunk_dsa"] + [layout["stack"]]
    )


def test_a_config_placing_the_draft_layer_elsewhere_is_refused_by_name(monkeypatch) -> None:
    _set_k(monkeypatch, 5)
    model = _model()
    model.text_config.num_nextn_predict_layers = 2
    with pytest.raises(ValueError, match="num_nextn_predict_layers 2"):
        model.get_kv_spec()


# ── 3. bind_kv_cache keeps the draft layer's bank last ───────────────────────


def test_knob_on_bind_keeps_the_draft_layers_bank_after_the_stack(monkeypatch) -> None:
    _set_k(monkeypatch, 5)
    model = _model()
    layout = _layout(model)
    stack, draft = layout["stack"], layout["draft"]
    trunk = layout["trunk_dsa"][-1]
    layers = model.get_kv_spec().layers
    caches = _kv_caches(layers)
    model.bind_kv_cache(caches)
    banks = model.glm5next_layer_banks
    assert [bank["layer_index"] for bank in banks] == list(range(stack + 1))
    mine, sibling = banks[stack], banks[trunk]
    assert mine["name"] == layers[stack].name
    assert mine["family"] == "self_attn"
    assert set(mine) == set(sibling)
    given = caches[mine["name"]][0]
    assert mine["latent_bank"] is given
    assert mine["latent_cache"].untyped_storage().data_ptr() == given.untyped_storage().data_ptr()
    assert tuple(mine["latent_cache"].shape) == (
        BLOCKS * BLOCK, layers[stack].num_kv_heads, layers[stack].head_size
    )
    for key in ("blocks", "block_size", "slots", "head_size"):
        assert mine[key] == sibling[key]
    assert f"layers.{draft}." in mine["name"]
    # The stack's records are the knob-0 records over the same tensors.
    _set_k(monkeypatch, 0)
    off_model = _model()
    off_model.bind_kv_cache({name: caches[name] for name in caches if name != mine["name"]})
    for one, two in zip(banks[:stack], off_model.glm5next_layer_banks, strict=True):
        assert set(one) == set(two)
        for key in one:
            if isinstance(one[key], torch.Tensor):
                assert one[key].data_ptr() == two[key].data_ptr() and one[key].shape == two[key].shape
            else:
                assert one[key] == two[key]


def test_knob_on_bind_refuses_a_cache_dict_without_the_draft_layer(monkeypatch) -> None:
    _set_k(monkeypatch, 5)
    model = _model()
    layout = _layout(model)
    layers = model.get_kv_spec().layers
    draft = layers[layout["stack"]].name
    assert draft.startswith(f"layers.{layout['draft']}.")
    caches = _kv_caches(layers)
    del caches[draft]
    with pytest.raises(ValueError, match=draft.replace(".", r"\.")):
        model.bind_kv_cache(caches)


def test_knob_on_bind_refuses_a_spec_that_dropped_the_draft_layer(monkeypatch) -> None:
    """The count check still guards the pairing: a head and a stack-only spec disagree."""
    from vllm_neuron.model.kv_cache import KVSpec

    _set_k(monkeypatch, 5)
    model = _model()
    full = model.get_kv_spec().layers
    caches = _kv_caches(full)
    monkeypatch.setattr(model, "get_kv_spec", lambda: KVSpec(layers=list(full[:-1])))
    with pytest.raises(ValueError, match="plus 1 draft layer"):
        model.bind_kv_cache(caches)
