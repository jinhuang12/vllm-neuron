# SPDX-License-Identifier: Apache-2.0
"""Host-only attention metadata for GLM-5.3-Flash: the per-step device copies it never reads.

Each step the runner uploads every KV group's block table (with a sentinel remap on the
device), every group's slot mapping, a cached-length scalar and a block-table clone. The
GLM-5.3-Flash translator (``_glm5next_model_kwargs``) builds its carriers from the host
block table and host cached lengths and hands the root no attention metadata, so none of
those device tensors reaches its graph. ``VLLM_NEURON_GLM5NEXT_HOST_ONLY_METADATA`` skips
them; the tokens must not move.

In CPU mode an upload is a no-op, so what is read here is that the upload paths are not
taken, and that the root receives the same carriers and returns the same tokens.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from vllm.v1.kv_cache_interface import SlidingWindowSpec

import vllm_neuron.vllm.worker.neuron_model_runner as runner_module
from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_first_request as fr
from test.vllm_neuron.vllm import test_glm5next_on_device_sampling as ods

pytestmark = [pytest.mark.fast]

KNOB = "VLLM_NEURON_GLM5NEXT_HOST_ONLY_METADATA"


def _counting(monkeypatch) -> dict:
    """Count the device-upload paths: the two save methods and the sentinel remap."""
    counts = {"block_table": 0, "slot_mapping": 0, "remap": 0}
    for name, key in (("_save_block_table_to_device", "block_table"),
                      ("_save_slot_mapping_to_device", "slot_mapping")):
        original = getattr(NeuronModelRunner, name)

        def counted(self, _original=original, _key=key):
            counts[_key] += 1
            return _original(self)

        monkeypatch.setattr(NeuronModelRunner, name, counted)
    remap = runner_module._remap_null_block_to_sentinel

    def counted_remap(table):
        counts["remap"] += 1
        return remap(table)

    monkeypatch.setattr(runner_module, "_remap_null_block_to_sentinel", counted_remap)
    return counts


def _carrier_tables(calls) -> list:
    """Every sparse carrier's block-table row and latent slots, per call."""
    rows = []
    for call in calls:
        rows.append([
            (carrier["block_table_row"].tolist(), carrier["latent_slots"].tolist())
            for carrier in call["kwargs"]["layer_carriers"]
            if "block_table_row" in carrier
        ])
    return rows


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    """Knob off, as built: tokens, carriers and how often each upload path ran."""
    saved = {name: os.environ.pop(name, None) for name in (KNOB, ods.KNOB)}
    patch = pytest.MonkeyPatch()
    try:
        counts = _counting(patch)
        ids, _, calls = ods._generate(tmp_path_factory.mktemp("ref"), fr._engine_config(),
                                      record=True)
    finally:
        patch.undo()
        for name, value in saved.items():
            if value is not None:
                os.environ[name] = value
    steps = 1 + ods.DECODES
    # The control: without the knob every step takes every upload path.
    assert counts["block_table"] == steps and counts["slot_mapping"] == steps, counts
    assert counts["remap"] >= steps, counts
    return ids, _carrier_tables(calls)


def test_knob_on_skips_every_upload_and_keeps_the_tokens(tmp_path, monkeypatch, reference):
    ref_ids, ref_tables = reference
    monkeypatch.setenv(KNOB, "1")
    monkeypatch.delenv(ods.KNOB, raising=False)
    counts = _counting(monkeypatch)
    ids, _, calls = ods._generate(tmp_path, fr._engine_config(), record=True)
    assert counts == {"block_table": 0, "slot_mapping": 0, "remap": 0}, counts
    assert ids == ref_ids, (ids, ref_ids)
    assert _carrier_tables(calls) == ref_tables


def test_knob_on_with_on_device_sampling_and_async_keeps_the_tokens(tmp_path, monkeypatch,
                                                                     reference):
    ref_ids, _ = reference
    monkeypatch.setenv(KNOB, "1")
    monkeypatch.setenv(ods.KNOB, "1")
    counts = _counting(monkeypatch)
    config = fr._engine_config(async_scheduling=True, on_device_sampling=True)
    ids, runner, _ = ods._generate(tmp_path, config)
    assert runner.use_async_scheduling and runner.on_device_sampling
    assert counts == {"block_table": 0, "slot_mapping": 0, "remap": 0}, counts
    assert ids == ref_ids, (ids, ref_ids)


def _shell(**overrides) -> NeuronModelRunner:
    """A runner carrying only what the predicate reads."""
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.model = SimpleNamespace(glm5next_layer_banks=({"name": "layers.0"},))
    runner.speculative_config = None
    runner.vllm_config = SimpleNamespace(kv_transfer_config=None)
    runner._kv_snapshot_enabled = False
    runner._tensor_replacer = None
    runner._target_tensor_capture = None
    runner.cp_world_size = 1
    for name, value in overrides.items():
        setattr(runner, name, value)
    return runner


def test_predicate_needs_the_knob_and_a_glm_root(monkeypatch):
    monkeypatch.delenv(KNOB, raising=False)
    assert _shell()._glm5next_host_only_metadata() is False
    monkeypatch.setenv(KNOB, "1")
    assert _shell()._glm5next_host_only_metadata() is True
    assert _shell(model=SimpleNamespace())._glm5next_host_only_metadata() is False


@pytest.mark.parametrize("override", [
    {"speculative_config": SimpleNamespace(num_speculative_tokens=1)},
    {"vllm_config": SimpleNamespace(kv_transfer_config=object())},
    {"_kv_snapshot_enabled": True},
    {"_tensor_replacer": object()},
    {"_target_tensor_capture": object()},
    {"cp_world_size": 2},
    {"kv_cache_config": SimpleNamespace(kv_cache_groups=[
        SimpleNamespace(kv_cache_spec=object.__new__(SlidingWindowSpec))])},
])
def test_predicate_keeps_the_device_metadata_for_its_other_readers(monkeypatch, override):
    """Speculative decoding, KV transfer, snapshots, tensor replacement/capture, context
    parallelism and sliding-window trims read the device metadata, so the knob does not
    apply there."""
    monkeypatch.setenv(KNOB, "1")
    assert _shell(**override)._glm5next_host_only_metadata() is False


def test_predicate_refuses_runtime_input_snapshots(monkeypatch):
    monkeypatch.setenv(KNOB, "1")
    monkeypatch.setenv("VLLM_NEURON_RUNTIME_INPUT_SNAPSHOT_ENABLE", "1")
    assert _shell()._glm5next_host_only_metadata() is False
