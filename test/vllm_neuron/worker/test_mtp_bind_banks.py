# SPDX-License-Identifier: Apache-2.0
"""A speculative server binds recurrent banks of ``1 + k`` state rows per slot.

Through the real runner (``NeuronModelRunner`` from a ``method: mtp`` engine config,
``initialize_kv_cache`` -> ``model.bind_kv_cache``): with ``num_speculative_tokens = k``
every ``linear_attn`` bank is ``[slots, 1 + k, *state]`` (the per-slot state the layer
spec declares, one row per verify-step token) and the bank record names the axis
(``state_checkpoints == 1 + k``), which the translator reads for the prefill leg's
one-row carrier ``bank[slot, 0]``; with speculation off the banks are ``[slots, *state]``
and the record carries no ``state_checkpoints`` key (absent = 1, the
plain record). Expectations come
from the layer spec the runner reports and from ``k``; no shape is pinned here.

The model is a double whose ``bind_kv_cache`` IS ``Glm5NextForConditionalGeneration``'s
(the class method applied to the double, so its exact shape check runs) over one real
``Glm5NextKDAAttention`` layer's spec: the tiny root fixture materialises DSA stacks only.

Seam: the bank allocation (``state_bank_regions(checkpoints=1 + k)``) and the bind's
acceptance of the axis are the KDA bank allocator's own. On a tree whose allocator has
no ``checkpoints`` parameter the two speculative
cases are a strict ``xfail`` naming that seam (plain banks, no record); on the merged tree
the condition drops and the cases must pass -- a bind that still refuses the axis, or a
record without ``state_checkpoints``, fails them.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

from test.vllm_neuron.model.glm5_next import test_mtp_e2e_spec as spec_test
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_batch_kda as kda
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_e2e as e2e
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_first_request as fr

BLOCKS = 4


def _tree_carves_checkpoint_banks() -> bool:
    """True once the allocator takes ``checkpoints`` (the allocator's ``state_bank_regions``)."""
    import inspect

    from vllm_neuron.vllm.worker.glm5next_state_banks import state_bank_regions

    return "checkpoints" in inspect.signature(state_bank_regions).parameters


SEAM = pytest.mark.xfail(
    condition=not _tree_carves_checkpoint_banks(),
    reason="this tree allocates one state row per slot: state_bank_regions(checkpoints=1 + k) "
    "and the bind_kv_cache checkpoint-bank hunk are not in it",
    strict=True,
)


def _double_root():
    """One real KDA attention layer behind the real class's ``bind_kv_cache``."""
    from vllm_neuron.model.glm5_next.model_fp8 import Glm5NextForConditionalGeneration

    text_config, _, layers = kda._layers()
    attention = layers[0]
    spec = KVSpec(layers=[LayerSpec(
        name=f"layers.0.{attention.CACHE_NAME_SUFFIX}",
        num_kv_heads=attention.num_kv_heads_per_rank,
        head_size=attention.head_size,
        dtype=attention.kda_conv_state_dtype,
        sliding_window_size=None,
        chunk_size=attention.cache_chunk_size,
        kda_conv_state_shape=attention.kda_conv_state_shape,
        kda_recurrent_state_shape=attention.kda_recurrent_state_shape,
        kda_conv_state_dtype=attention.kda_conv_state_dtype,
        kda_recurrent_state_dtype=attention.kda_recurrent_state_dtype,
        latent_kv=getattr(attention, "LATENT_KV_CACHE", False),
    )])
    root = SimpleNamespace(
        model=SimpleNamespace(layers=[attention]), mtp=None, text_config=text_config,
        get_kv_spec=lambda: spec,
    )
    root.bind_kv_cache = lambda caches: Glm5NextForConditionalGeneration.bind_kv_cache(root, caches)
    return root


def _bound_banks(tmp_path, monkeypatch, k: int | None):
    """The recurrent bank records the runner binds for speculative ``k`` (None: off), with the
    per-slot state shapes the runner's layer spec declares."""
    e2e._require_cpu_mode()
    fr._declaring_a_sampler(monkeypatch)
    monkeypatch.delenv(spec_test.KNOB, raising=False)
    config = spec_test._config(k)
    world = tmp_path / ("plain" if k is None else f"mtp-{k}")
    world.mkdir()
    with fr._parallel_state(world, config):
        runner = NeuronModelRunner(config, device=torch.device("cpu"))
        assert (runner.drafter is not None) == (k is not None)
        root = _double_root()
        runner.model = root
        specs = runner.get_kv_cache_spec()
        runner.initialize_kv_cache(fr._kv_cache_config(runner, num_blocks=BLOCKS))
        banks = [bank for bank in root.glm5next_layer_banks if bank["family"] == "linear_attn"]
        assert banks, "the double declares one recurrent layer"
        states = {}
        for bank in banks:
            spec = specs[bank["name"]]
            assert isinstance(spec, MambaSpec), (bank["name"], type(spec))
            states[bank["name"]] = tuple(tuple(int(v) for v in shape) for shape in spec.shapes)
        return banks, states


@SEAM
@pytest.mark.parametrize("k", [1, 3])
def test_a_speculative_server_binds_one_state_row_per_verify_token(tmp_path, monkeypatch, k):
    banks, states = _bound_banks(tmp_path, monkeypatch, k)
    for bank in banks:
        conv, recurrent = states[bank["name"]]
        assert int(bank.get("state_checkpoints", 1)) == 1 + k, (bank["name"], bank.get("state_checkpoints"))
        assert tuple(bank["conv_state"].shape[1:]) == (1 + k, *conv), bank["name"]
        assert tuple(bank["recurrent_state"].shape[1:]) == (1 + k, *recurrent), bank["name"]
        assert int(bank["conv_state"].shape[0]) == int(bank["recurrent_state"].shape[0]) == int(bank["state_slots"])


def test_a_plain_server_binds_one_state_row_per_slot(tmp_path, monkeypatch):
    banks, states = _bound_banks(tmp_path, monkeypatch, None)
    for bank in banks:
        conv, recurrent = states[bank["name"]]
        assert int(bank.get("state_checkpoints", 1)) == 1, bank["name"]
        assert tuple(bank["conv_state"].shape[1:]) == conv, bank["name"]
        assert tuple(bank["recurrent_state"].shape[1:]) == recurrent, bank["name"]
        assert int(bank["conv_state"].shape[0]) == int(bank["state_slots"])
