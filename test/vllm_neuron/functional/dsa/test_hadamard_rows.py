# SPDX-License-Identifier: Apache-2.0
"""The query rotation ``dsa_hadamard128``: its launch grid, and the two-program split.

The rotation's numerics against the reference live in ``test_kpool_hadamard.py`` and
``test_kpool_hadamard_error_bound.py``. This file pins the launch: the rotation splits its
rows over both programs only under an LNC2 launch (``NEURON_LOGICAL_NC_CONFIG=2``, read
through ``launch_grid.lnc_pair``), a single row takes one program, and the split changes
no output bit, because each row is rotated by the same instructions whichever program
owns it. ``test_decode_ctx.py`` pins the decode_batch and decode_select grids and leaves
the rotation's grid to this file.
"""

from __future__ import annotations

import pytest
import torch

from vllm_neuron import envs
from vllm_neuron.functional.dsa import kpool_hadamard as KH
from vllm_neuron.utils.neuron_utils import can_run_kernel


@pytest.fixture(autouse=True)
def _kernels_on():
    assert can_run_kernel(), "run with VLLM_NEURON_CPU_MODE=1 and NKI_SIMULATOR=1"
    KH.reset_kpool_hadamard_dispatch_counters()


def _rows(n: int, dtype, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return (torch.randn(n, KH.INDEX_HEAD_DIM, generator=gen) * 3.0).to(dtype)


#: Two rows, a ragged tile, the decode query at B=64 (32 heads per request), and a count
#: that leaves a short final tile on each of two programs.
SPLIT_ROWS = (2, 37, 64 * 32, 2 * 1024 + 2 * 128 + 6)


@pytest.mark.parametrize("n", SPLIT_ROWS)
def test_two_programs_split_the_rows_and_equal_one(monkeypatch, n):
    x = _rows(n, torch.bfloat16, seed=7 + n)
    monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    assert KH._programs(n) == 1
    one = KH.dsa_hadamard128(x)
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
    assert KH._programs(n) == 2
    two = KH.dsa_hadamard128(x)
    assert KH.kpool_hadamard_dispatch_counters() == (2, 0)
    assert torch.equal(one.view(torch.int16), two.view(torch.int16))


@pytest.mark.parametrize("lnc", [None, 1, 2])
def test_the_rotation_grid_follows_the_logical_core_config(monkeypatch, lnc):
    """Both programs only under an LNC2 launch; one row always takes one program."""
    # The entry, not a module attribute: ``envs`` resolves names lazily, so an attribute
    # set here would outlive the test and shadow the variable for every later test.
    monkeypatch.setitem(envs.environment_variables, "NEURON_LOGICAL_NC_CONFIG", lambda: lnc)
    assert KH._programs(1) == 1
    assert KH._programs(2) == (2 if lnc == 2 else 1)
