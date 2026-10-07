# SPDX-License-Identifier: Apache-2.0
"""Layer 45's KV state, model side (MTP stage A, contract C4): ``get_kv_spec`` / ``bind_kv_cache``.

GLM-5.3-Flash's MTP layer is checkpoint layer 45, one past the 45-layer stack, and it is a
``deepseek_sparse_attention`` (DSA) layer. When contract C1's ``mtp.shadow_draft_k()`` is
above 0 the draft runs, so it needs the state a trunk DSA layer keeps: a latent page here,
and the indexer side caches and a carrier on the runner's side
(``test/vllm_neuron/worker/test_mtp_state_runner.py``). Read here:

1. knob 0 (and a tree whose ``mtp`` defines no ``shadow_draft_k`` yet, with the knob's
   variable unset, empty or 0): the ``LayerSpec`` list is the one 82bee3b produces (the
   record ``mtp_state_base_82bee3b.json`` beside the runner test, taken from 82bee3b by
   that file's ``main``);
2. knob 1..5: exactly one more ``LayerSpec``, ``layers.45.self_attn``, last, with layer
   43's geometry (layer 45 is built by the same DSA class from the same config);
3. ``bind_kv_cache`` keeps layer 45's bank at index 45 of ``glm5next_layer_banks``, as a
   sparse-family record over the runner's own tensor, and refuses a dict without it;
4. the knob is read through ``shadow_draft_k()``; on a tree without it, from the knob's
   variable, the fallback the sibling stubs (root construction, root forward) use.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/model/glm5_next/test_mtp_state.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "config.json"
BASE_RECORD = (
    Path(__file__).resolve().parents[2] / "worker" / "mtp_state_base_82bee3b.json"
)
#: Contract C1's variable, read directly only while ``mtp.shadow_draft_k`` is absent.
KNOB_ENV = "VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT"
STACK = 45
DRAFT_LAYER = 45
LAST_TRUNK_DSA = 43
BLOCKS, BLOCK, SLOTS = 6, 128, 3


def _set_k(monkeypatch, value) -> None:
    from vllm_neuron.model.glm5_next import mtp

    monkeypatch.setattr(mtp, "shadow_draft_k", lambda: int(value), raising=False)


def _model():
    """The skeleton the runner record was taken on: the fixture at one KDA head per rank."""
    from vllm_neuron.model.glm5_next.model_fp8 import Glm5NextForConditionalGeneration

    raw = json.loads(FIXTURE_PATH.read_text())
    raw["text_config"]["linear_attn_config"]["num_heads"] = 1  # per-rank at TP=64
    return Glm5NextForConditionalGeneration.from_configs(raw)


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
                           float(index), dtype=torch.bfloat16)
            ]
    return caches


# ── 1. knob 0 ────────────────────────────────────────────────────────────────


def test_knob_off_the_spec_list_is_the_82bee3b_record(monkeypatch) -> None:
    base = json.loads(BASE_RECORD.read_text())
    want = base["lines"]["bs1"]["kv_spec"]
    assert want == base["lines"]["bs64"]["kv_spec"] and len(want) == STACK
    _set_k(monkeypatch, 0)
    assert [_record(layer) for layer in _model().get_kv_spec().layers] == want


@pytest.mark.parametrize("raw", [None, "", "0"])
def test_a_tree_without_the_knob_function_and_the_knob_unset_or_0_is_82bee3b(
    raw, monkeypatch
) -> None:
    """Until worker-50's ``shadow_draft_k`` lands, unset, empty or 0 is off."""
    from vllm_neuron.model.glm5_next import mtp

    monkeypatch.delattr(mtp, "shadow_draft_k", raising=False)
    if raw is None:
        monkeypatch.delenv(KNOB_ENV, raising=False)
    else:
        monkeypatch.setenv(KNOB_ENV, raw)
    base = json.loads(BASE_RECORD.read_text())
    model = _model()
    layers = model.get_kv_spec().layers
    assert [_record(layer) for layer in layers] == base["lines"]["bs1"]["kv_spec"]
    model.bind_kv_cache(_kv_caches(layers))
    assert len(model.glm5next_layer_banks) == STACK


def test_a_tree_without_the_knob_function_reads_the_knob_variable(monkeypatch) -> None:
    """Without ``mtp.shadow_draft_k`` the knob's variable is read directly, as the root
    construction (worker-51, ``resolve_shadow_draft_k``) and forward (worker-53,
    ``_shadow_draft_k``) stubs read it, so a branch without worker-50's reader still
    builds the head, the 12th spec and the 46th carrier together."""
    from vllm_neuron.model.glm5_next import mtp

    monkeypatch.delattr(mtp, "shadow_draft_k", raising=False)
    monkeypatch.setenv(KNOB_ENV, "5")
    model = _model()
    layers = model.get_kv_spec().layers
    assert len(layers) == STACK + 1
    assert layers[DRAFT_LAYER].name == "layers.45.self_attn"
    model.bind_kv_cache(_kv_caches(layers))
    assert len(model.glm5next_layer_banks) == STACK + 1


def test_the_knob_function_wins_over_the_knob_variable(monkeypatch) -> None:
    monkeypatch.setenv(KNOB_ENV, "5")
    _set_k(monkeypatch, 0)
    model = _model()
    layers = model.get_kv_spec().layers
    assert len(layers) == STACK
    model.bind_kv_cache(_kv_caches(layers))
    assert len(model.glm5next_layer_banks) == STACK


def test_knob_off_bind_keeps_the_stack_and_nothing_else(monkeypatch) -> None:
    _set_k(monkeypatch, 0)
    model = _model()
    layers = model.get_kv_spec().layers
    model.bind_kv_cache(_kv_caches(layers))
    banks = model.glm5next_layer_banks
    assert len(banks) == STACK
    assert [bank["layer_index"] for bank in banks] == list(range(STACK))


# ── 2. knob 1..5: the 12th DSA spec ──────────────────────────────────────────


@pytest.mark.parametrize("k", [0, 1, 2, 3, 4, 5])
def test_the_spec_count_follows_shadow_draft_k(k, monkeypatch) -> None:
    _set_k(monkeypatch, k)
    layers = _model().get_kv_spec().layers
    assert len(layers) == STACK + (1 if k > 0 else 0)
    assert sum(1 for layer in layers if layer.latent_kv) == 11 + (1 if k > 0 else 0)


def test_knob_on_adds_layer_45_last_with_layer_43s_geometry(monkeypatch) -> None:
    _set_k(monkeypatch, 0)
    off = [_record(layer) for layer in _model().get_kv_spec().layers]
    _set_k(monkeypatch, 5)
    model = _model()
    layers = model.get_kv_spec().layers
    on = [_record(layer) for layer in layers]
    assert on[:STACK] == off
    draft, trunk = on[DRAFT_LAYER], on[LAST_TRUNK_DSA]
    assert draft["name"] == "layers.45.self_attn"
    assert trunk["name"] == "layers.43.self_attn"
    assert {k: v for k, v in draft.items() if k != "name"} == {
        k: v for k, v in trunk.items() if k != "name"
    }
    # Layer 43 is the trunk's last DSA layer, and the stack has no layer 45 to read.
    assert [i for i, one in enumerate(off) if one["latent_kv"]][-1] == LAST_TRUNK_DSA
    assert len(model.model.layers) == STACK
    assert int(model.text_config.num_hidden_layers) == DRAFT_LAYER


# ── 3. bind_kv_cache keeps layer 45's bank at index 45 ───────────────────────


def test_knob_on_bind_keeps_layer_45s_bank_at_index_45(monkeypatch) -> None:
    _set_k(monkeypatch, 5)
    model = _model()
    layers = model.get_kv_spec().layers
    caches = _kv_caches(layers)
    model.bind_kv_cache(caches)
    banks = model.glm5next_layer_banks
    assert len(banks) == STACK + 1
    assert [bank["layer_index"] for bank in banks] == list(range(STACK + 1))
    mine, trunk = banks[DRAFT_LAYER], banks[LAST_TRUNK_DSA]
    assert mine["name"] == "layers.45.self_attn"
    assert mine["family"] == "self_attn"
    assert set(mine) == set(trunk)
    given = caches["layers.45.self_attn"][0]
    assert mine["latent_bank"] is given
    assert mine["latent_cache"].untyped_storage().data_ptr() == given.untyped_storage().data_ptr()
    assert tuple(mine["latent_cache"].shape) == (BLOCKS * BLOCK, 1, 512)
    for key in ("blocks", "block_size", "slots", "head_size"):
        assert mine[key] == trunk[key]
    # The stack's records are the knob-0 records over the same tensors.
    _set_k(monkeypatch, 0)
    off_model = _model()
    off_model.bind_kv_cache({name: caches[name] for name in caches if name != mine["name"]})
    for one, two in zip(banks[:STACK], off_model.glm5next_layer_banks, strict=True):
        assert set(one) == set(two)
        for key in one:
            if isinstance(one[key], torch.Tensor):
                assert one[key].data_ptr() == two[key].data_ptr() and one[key].shape == two[key].shape
            else:
                assert one[key] == two[key]


def test_knob_on_bind_refuses_a_cache_dict_without_layer_45(monkeypatch) -> None:
    _set_k(monkeypatch, 5)
    model = _model()
    caches = _kv_caches(model.get_kv_spec().layers)
    del caches["layers.45.self_attn"]
    with pytest.raises(ValueError, match=r"layers\.45\.self_attn"):
        model.bind_kv_cache(caches)


def test_knob_on_bind_refuses_a_spec_that_dropped_layer_45(monkeypatch) -> None:
    """The count check still guards the pairing: knob on and 45 specs is a disagreement."""
    from vllm_neuron.model.kv_cache import KVSpec

    _set_k(monkeypatch, 5)
    model = _model()
    full = model.get_kv_spec().layers
    caches = _kv_caches(full)
    monkeypatch.setattr(model, "get_kv_spec", lambda: KVSpec(layers=list(full[:STACK])))
    with pytest.raises(ValueError, match="45 layer"):
        model.bind_kv_cache(caches)
