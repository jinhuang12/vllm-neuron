# SPDX-License-Identifier: Apache-2.0
"""The on-device sampling parameters are built without reading the device back.

``build_sampling_params_tensor`` runs every on-device-sampling step. A debug log that
formats ``result.tolist()`` eagerly reads the device tensor back to the host on every
step, whatever the log level; under async scheduling that read can wait on the step in
flight. A ``meta`` device makes any such read raise, so the build is run on one.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch

from vllm_neuron.vllm.worker.neuron_model_runner import build_sampling_params_tensor

pytestmark = [pytest.mark.fast]

META = torch.device("meta")
LOGGER = "vllm_neuron.vllm.worker.neuron_model_runner"


@pytest.mark.parametrize("batch", [1, 4, 64])
def test_greedy_parameters_build_on_the_device_with_no_read_back(batch):
    metadata = SimpleNamespace(top_k=None, top_p=None, temperature=None)
    logging.getLogger(LOGGER).setLevel(logging.INFO)
    params = build_sampling_params_tensor(metadata, batch, META)
    assert params.device.type == "meta" and tuple(params.shape) == (batch, 3)


def test_per_request_parameters_build_on_the_device_with_no_read_back():
    metadata = SimpleNamespace(
        top_k=torch.tensor([5, -1, 40, 1], dtype=torch.int32),
        top_p=torch.ones(4, device=META),
        temperature=torch.ones(4, device=META),
    )
    logging.getLogger(LOGGER).setLevel(logging.INFO)
    params = build_sampling_params_tensor(metadata, 4, META)
    assert params.device.type == "meta" and tuple(params.shape) == (4, 3)


def test_the_values_are_unchanged_on_the_host():
    metadata = SimpleNamespace(
        top_k=torch.tensor([5, -1], dtype=torch.int32),
        top_p=torch.tensor([0.9, 1.0]),
        temperature=torch.tensor([0.0, 0.7]),
    )
    params = build_sampling_params_tensor(metadata, 2, torch.device("cpu"))
    assert params.tolist() == [[5.0, pytest.approx(0.9), 0.0], [-1.0, 1.0, pytest.approx(0.7)]]


def test_debug_logging_still_reports_the_values(caplog):
    metadata = SimpleNamespace(top_k=None, top_p=None, temperature=None)
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        build_sampling_params_tensor(metadata, 2, torch.device("cpu"))
    assert any("[[-1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]]" in record.getMessage()
               for record in caplog.records), [r.getMessage() for r in caplog.records]
