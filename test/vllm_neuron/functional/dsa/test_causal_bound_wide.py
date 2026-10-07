# SPDX-License-Identifier: Apache-2.0
"""The prefill causal bound at candidate widths past 16,384 (contexts past 64k).

``_causal_bound_nki`` held a whole ``[rows, width]`` score row and five same-width
helper tiles in SBUF per row tile, so neuronx-cc refused it from width 32,768 (context
131,072) on (``test_dsa_wide_cpu_compile.py``). The kernel now walks the width in
column tiles of :data:`causal_bound.COLUMN_TILE`. Two claims are checked here, on the
simulator:

* the result equals b17526a's kernel bit for bit at the widths it served (512 to
  16,384), the reference being b17526a's own file loaded from git;
* past them it equals the torch oracle exactly (a kept column carries the loaded bits,
  a bounded one holds ``BOUND_FILL``), with lengths that end inside every column tile.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

from vllm_neuron.functional.dsa import causal_bound as CB
from vllm_neuron.utils.neuron_utils import can_run_kernel

BASE_COMMIT = "b17526a"
POOL = 4


def _load_base_module():
    root = Path(CB.__file__).resolve().parents[3]
    source = subprocess.run(
        ["git", "-C", str(root), "show",
         f"{BASE_COMMIT}:vllm_neuron/functional/dsa/causal_bound.py"],
        check=True, capture_output=True).stdout
    path = Path(tempfile.mkdtemp(prefix="causal_bound_base_")) / "causal_bound_b17526a.py"
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location("causal_bound_b17526a", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def base():
    return _load_base_module()


@pytest.fixture(autouse=True)
def _kernels_on():
    assert can_run_kernel(), "run with VLLM_NEURON_CPU_MODE=1 and NKI_SIMULATOR=1"
    CB.reset_causal_bound_dispatch_counters()


def _case(rows: int, width: int, seed: int):
    """Scores with a ``-0.0`` column, and lengths spread over the whole width."""
    gen = torch.Generator().manual_seed(seed)
    scores = torch.randn(rows, width, generator=gen)
    scores[:, 1] = -0.0
    top = width * POOL
    lengths = torch.randint(0, top + 1, (rows,), generator=gen)
    if rows == 1:
        lengths[0] = (width * 3 // 4) * POOL + 1   # the bound lands in a late column tile
    else:
        lengths[0] = top      # every pool complete
        lengths[-1] = 3       # no pool complete
    return scores, lengths.to(torch.int32).reshape(rows, 1)


@pytest.mark.parametrize("rows", [1, 130])
@pytest.mark.parametrize("width", [512, 2048, 8192, 16384])
def test_column_tiles_equal_b17526a_bit_for_bit(base, rows, width):
    scores, lengths = _case(rows, width, seed=width + rows)
    want = base.dsa_causal_bound(scores, lengths, POOL)
    got = CB.dsa_causal_bound(scores, lengths, POOL)
    assert CB.causal_bound_dispatch_counters() == (1, 0)
    assert torch.equal(got.view(torch.int32), want.view(torch.int32))


@pytest.mark.parametrize("rows", [1, 130])
@pytest.mark.parametrize("width", [16385, 32768, 65536])
def test_column_tiles_match_the_oracle_past_the_old_width(rows, width):
    scores, lengths = _case(rows, width, seed=3 * width + rows)
    got = CB.dsa_causal_bound(scores, lengths, POOL)
    assert CB.causal_bound_dispatch_counters() == (1, 0)
    want = CB.dsa_causal_bound_torch_oracle(scores, lengths, POOL)
    assert torch.equal(got.view(torch.int32), want.view(torch.int32))
