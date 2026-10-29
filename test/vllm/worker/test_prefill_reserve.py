# SPDX-License-Identifier: Apache-2.0
"""Neuron worker KV budget: the prefill reserve caps the default budget, and the KV floor.

The worker is bare, at LNC 2 with no headroom setting. Its runner holds a neuron config and a
model of known bytes. The device-free compile's caller reads the platform's HBM size, and the
device run's caller the runtime's memory query; both are stubbed. The served forms' reserves
come from the Kimi K3 cells file. Run with ``VLLM_NEURON_CPU_MODE=1``.
"""

import json
import re
import types
from pathlib import Path

import libtorch_neuronx_lite.compile.platform as neuron_platform
import pytest
import torch

from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.vllm.worker.neuron_worker import NeuronWorker

GIB = 1024**3
HBM = 36 * GIB
# One rank's device bytes at 2 replica slots: weights, load-time copies and the slots.
FORM_DEVICE_BYTES = 33_007_600_700 - 1_614_348_288
CELLS = Path(__file__).resolve().parents[2] / "kimi_k3" / "serving" / "cells.json"
PATHS = ("device-free compile", "device run")


@pytest.fixture
def lnc2(monkeypatch):
    """LNC 2, no headroom, floor or cap setting, and a platform HBM size of 36 GiB."""
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    for name in ("VLLM_NEURON_RESERVED_HEADROOM_GIB", "VLLM_NEURON_MIN_KV_BUDGET_GIB",
                 "VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(neuron_platform, "get_total_available_memory", lambda: HBM / GIB)
    return monkeypatch


def _worker(monkeypatch, model_bytes: int, **neuron) -> NeuronWorker:
    """A bare worker whose runner holds ``neuron`` and a model of ``model_bytes``; the
    runtime reports those bytes used of 36 GiB."""
    model = torch.nn.Module()
    model.register_buffer("weights", torch.empty(model_bytes, dtype=torch.uint8, device="meta"))
    worker = NeuronWorker.__new__(NeuronWorker)
    worker.cache_config = types.SimpleNamespace(gpu_memory_utilization=0.90)
    worker.model_runner = types.SimpleNamespace(
        model=model, drafter=None, neuron_config=NeuronConfig(**neuron))
    monkeypatch.setattr(worker, "_query_runtime_memory_stats",
                        lambda: (model_bytes, HBM - model_bytes))
    return worker


def _callers(worker) -> dict:
    """The two Neuron callers of the KV budget."""
    return {"device-free compile": worker._estimate_available_memory_neuron,
            "device run": worker._determine_available_memory_neuron}


def _form_reserves() -> dict:
    """Each Kimi K3 form's prefill reserve, from its serve flags."""
    reserves = {}
    for name, form in json.loads(CELLS.read_text(encoding="utf-8"))["forms"].items():
        flags = form["server_flags"]
        neuron = json.loads(flags[flags.index("--additional-config") + 1])["neuron_config"]
        if "prefill_reserve_gib" in neuron:
            reserves[name] = neuron["prefill_reserve_gib"]
    return reserves


def test_reserve_caps_the_default_budget_on_both_paths(lnc2):
    """Item 1: at 36 GiB and 28 GiB of weights, a 5 GiB reserve gives 3.00 GiB where the rule
    without it gives 4.40, and a 2 GiB reserve 4.40. At each form's device bytes and reserve
    R, 2 slots: min(3.16, 6.76 - R) GiB. Both callers at every point: 0 differ."""
    points = [(28 * GIB, 0.0, 4.40), (28 * GIB, 5.0, 3.00), (28 * GIB, 2.0, 4.40)]
    reserves = _form_reserves()
    assert sorted(reserves) == ["dcp16", "dcp4"]
    points += [(FORM_DEVICE_BYTES, r, min(3.16, 6.76 - r)) for r in reserves.values()]
    differ = []
    for weights, reserve, want in points:
        worker = _worker(lnc2, weights, prefill_reserve_gib=reserve)
        got = {name: run() / GIB for name, run in _callers(worker).items()}
        if any(round(gib, 2) != round(want, 2) for gib in got.values()):
            differ.append((weights / GIB, reserve, want, got))
    assert differ == []


def test_reserve_that_leaves_too_little_refuses_to_start(lnc2):
    """Item 2: at 36 GiB and 30 GiB of weights a 5.5 GiB reserve leaves 0.5 GiB, under the
    1 GiB minimum: each caller raises the below-minimum error, whose formula names the
    reserve."""
    worker = _worker(lnc2, 30 * GIB, prefill_reserve_gib=5.5)
    raised = {}
    for name, run in _callers(worker).items():
        with pytest.raises(RuntimeError, match="below minimum threshold") as error:
            run()
        raised[name] = str(error.value)
    assert sorted(raised) == sorted(PATHS)
    assert all("effective=0.50 GiB" in m and "prefill_reserve_bytes" in m
               for m in raised.values()), raised


def test_kv_floor_field_replaces_the_environment_value(lnc2):
    """Item 3: with min_kv_budget_gib 0.54, a budget of 0.55 GiB starts and one of 0.53 GiB
    raises the below-minimum error naming the field. With the field unset, the environment
    value, 1 GiB, applies: 1.01 GiB starts and 0.55 GiB raises naming the variable. Each
    budget is 36 GiB less 28 GiB of weights and a reserve, on both callers."""
    cases = {(0.54, 0.55): "starts", (0.54, 0.53): "min_kv_budget_gib",
             (None, 1.01): "starts", (None, 0.55): "VLLM_NEURON_MIN_KV_BUDGET_GIB"}
    got = {}
    for floor, budget in cases:
        worker = _worker(lnc2, 28 * GIB, prefill_reserve_gib=8 - budget,
                         min_kv_budget_gib=floor)
        for name, run in _callers(worker).items():
            try:
                run()
                got[floor, budget, name] = "starts"
            except RuntimeError as error:
                named = re.search(r"below minimum threshold.*minimum=[\d.]+ GiB \((\w+)\)",
                                  str(error))
                got[floor, budget, name] = named.group(1) if named else str(error)
    assert got == {(f, b, name): want for (f, b), want in cases.items() for name in PATHS}
