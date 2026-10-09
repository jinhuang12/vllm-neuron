# SPDX-License-Identifier: Apache-2.0
"""The prefill key pooling ``dsa_kpool_hadamard``: how it reads and launches its grid.

The pooling's numerics against the reference live in ``test_kpool_hadamard.py`` and
``test_kpool_hadamard_error_bound.py``. This file pins the launch. The grid comes from
``launch_grid.lnc_pair``, which a trace folds to a constant, as the decode grids do; a raw
environment read in ``kpool_hadamard.py`` would become a guard that reads the environment
again before every step of the prefill graph. Under an LNC2 launch the pools are split over
two programs, a single pool takes one program, and a setting the kernels do not serve is
refused by name.
"""

from __future__ import annotations

import ast
import pathlib

import pytest
import torch

from vllm_neuron.functional.dsa import kpool_hadamard as KH
from vllm_neuron.utils.neuron_utils import can_run_kernel


@pytest.fixture(autouse=True)
def _kernels_on():
    assert can_run_kernel(), "run with VLLM_NEURON_CPU_MODE=1 and NKI_SIMULATOR=1"
    KH.reset_kpool_hadamard_dispatch_counters()


def _inputs(n_pools: int, pool_size: int, score_dtype, seed: int):
    gen = torch.Generator().manual_seed(seed)
    shape = (n_pools, pool_size, KH.INDEX_HEAD_DIM)
    slot_k = (torch.randn(shape, generator=gen) * 2.0).to(torch.bfloat16)
    slot_score = (torch.randn(shape, generator=gen) * 3.0).to(score_dtype)
    ape = torch.randn((pool_size, KH.INDEX_HEAD_DIM), generator=gen) * 0.5
    return slot_k, slot_score, ape


def test_the_pooling_grid_reads_the_setting_through_lnc_pair_and_refuses_an_unserved_one(
        monkeypatch):
    from vllm_neuron.functional.dsa.launch_grid import LaunchGridError

    tree = ast.parse(pathlib.Path(KH.__file__).read_text())
    raw = [node.lineno for node in ast.walk(tree)
           if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
           and node.value.id in ("os", "envs")]
    assert not raw, f"kpool_hadamard.py reads the environment itself at lines {raw}"
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "4")
    with pytest.raises(LaunchGridError, match="NEURON_LOGICAL_NC_CONFIG=4"):
        KH._programs(2)


def _launch_grids(monkeypatch, n_pools: int) -> list:
    """The grid of every launch of the pooling kernel during one call."""
    seen = []
    real = KH.wrap_nki

    def spy(kernel):
        call = real(kernel)

        class _Spy:
            def __getitem__(self, grid):
                seen.append(grid)
                return call[grid]

            def __call__(self, *a):
                seen.append(1)
                return call(*a)
        return _Spy()

    monkeypatch.setattr(KH, "wrap_nki", spy)
    KH.dsa_kpool_hadamard(*_inputs(n_pools, KH.DEFAULT_POOL_SIZE, torch.bfloat16, seed=1))
    return seen


def test_an_lnc2_launch_is_a_two_program_grid(monkeypatch):
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    assert _launch_grids(monkeypatch, 64) == [2]


def test_one_pool_is_one_program(monkeypatch):
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    assert _launch_grids(monkeypatch, 1) == [1]
