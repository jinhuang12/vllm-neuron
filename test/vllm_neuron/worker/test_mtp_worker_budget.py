# SPDX-License-Identifier: Apache-2.0
"""The worker's model-bytes count with a drafter that shares the target's parameters.

``NeuronWorker._get_byte_used_from_model`` adds the drafter's parameters to the
target's; the mtp head is a submodule of the GLM-5.3-Flash root, so a drafter that
declares ``shares_target_parameters`` is skipped, or its bytes would be counted twice
and the KV budget shrink by the head's size (team-lead ruling 10:10Z, hunk 3). An
eagle drafter, a model of its own, is still counted.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/worker/test_mtp_worker_budget.py
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from vllm_neuron.vllm.spec_decode.mtp import MtpProposer
from vllm_neuron.vllm.worker.neuron_worker import NeuronWorker


def _bytes(module: torch.nn.Module) -> int:
    return sum(p.nbytes for p in module.parameters()) + sum(b.nbytes for b in module.buffers())


class _Target(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = torch.nn.Linear(8, 4)
        self.mtp = torch.nn.Linear(4, 4)
        self.register_buffer("scale", torch.ones(3))


def _worker(target, drafter):
    return SimpleNamespace(model_runner=SimpleNamespace(model=target, drafter=drafter))


def test_the_mtp_proposer_declares_that_it_shares_the_targets_parameters():
    assert MtpProposer.shares_target_parameters is True


def test_a_drafter_sharing_the_targets_parameters_is_not_counted_twice():
    target = _Target()
    shared = SimpleNamespace(shares_target_parameters=True, model=target.mtp)
    assert NeuronWorker._get_byte_used_from_model(_worker(target, shared)) == _bytes(target)


def test_a_drafter_with_a_model_of_its_own_is_counted():
    target = _Target()
    own = SimpleNamespace(model=torch.nn.Linear(2, 2))
    expected = _bytes(target) + _bytes(own.model)
    assert NeuronWorker._get_byte_used_from_model(_worker(target, own)) == expected


def test_no_drafter_counts_the_target_alone():
    target = _Target()
    assert NeuronWorker._get_byte_used_from_model(_worker(target, None)) == _bytes(target)
