# SPDX-License-Identifier: Apache-2.0
"""DSpark proposer warm-up under async scheduling: the readback skips meta outputs.

The device-free compile runs each warm-up on the meta device, where a host copy has no data to
read. A stand-in proposer holds a ``propose`` that returns the draft's two outputs. Run with
``VLLM_NEURON_CPU_MODE=1``.
"""

import types

import torch

from vllm_neuron.vllm.spec_decode.dspark import DSparkProposer


class Probe(torch.Tensor):
    """A CPU tensor that counts its ``cpu()`` calls."""

    reads = 0

    def cpu(self, *args, **kwargs):
        self.reads += 1
        return self


def _warmup(outputs) -> None:
    """Run the warm-up, async scheduling on, with a ``propose`` that returns ``outputs``."""
    proposer = DSparkProposer.__new__(DSparkProposer)
    proposer.vllm_config = types.SimpleNamespace(
        scheduler_config=types.SimpleNamespace(async_scheduling=True))
    proposer.model = object()
    proposer._build_synthetic_inputs = lambda num_tokens, num_reqs: {}
    proposer.propose = lambda **kwargs: outputs
    proposer.warmup(8, 1, None)


def test_warmup_on_meta_outputs_returns_without_error():
    _warmup(tuple(torch.empty(1, n, dtype=torch.int32, device="meta") for n in (5, 4)))


def test_warmup_reads_each_cpu_output_back_once():
    probes = tuple(torch.zeros(1, n, dtype=torch.int32).as_subclass(Probe) for n in (5, 4))
    _warmup(probes)
    assert [probe.reads for probe in probes] == [1, 1]
