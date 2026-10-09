# SPDX-License-Identifier: Apache-2.0
"""The prefill causal bound at candidate widths past 16,384 (contexts past 64k).

``_causal_bound_nki`` held a whole ``[rows, width]`` score row and five same-width
helper tiles in SBUF per row tile, so neuronx-cc refused it from width 32,768 (context
131,072) on (``test_dsa_wide_cpu_compile.py``). The kernel now walks the width in
column tiles of :data:`causal_bound.COLUMN_TILE`. Two claims are checked here, on the
simulator:

* the result equals the untiled kernel bit for bit at the widths it served (512 to
  16,384), the reference being the last untiled commit's own file loaded from git;
* past them it equals the torch oracle exactly (a kept column carries the loaded bits,
  a bounded one holds ``BOUND_FILL``), with lengths that end inside every column tile.
"""

from __future__ import annotations

import inspect
import re

import pytest
import torch

from test.vllm_neuron.functional.reference_at_commit import (
    load_reference,
    needs_reference,
)
from vllm_neuron.functional.dsa import causal_bound as CB
from vllm_neuron.functional.dsa import decode_batch as DB
from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig
from vllm_neuron.utils.neuron_utils import SBUF_BYTES_PER_PARTITION, can_run_kernel

#: The last commit whose bound holds a whole candidate row: the bit-equality reference.
UNTILED_COMMIT = "b17526a"
#: Tokens per candidate pool, GLM-5.3-Flash's.
POOL = Glm5NextTextConfig().index_kpool
#: The widest candidate row of the indexer chain at :data:`UNTILED_COMMIT`: the old decode
#: kernel's ``MAX_CANDIDATES``, one ``PARTITIONS x PARTITIONS`` block of candidates (65,536
#: tokens at pool 4). The untiled bound held a whole row in SBUF, which neuronx-cc refuses
#: from twice this width on (``causal_bound.COLUMN_TILE``); ``test_dsa_wide_cpu_compile.py``
#: builds the tiled bound at those widths.
UNTILED_WIDTH_MAX = DB.PARTITIONS**2


#: The reference file at :data:`UNTILED_COMMIT`.
UNTILED_PATH = "vllm_neuron/functional/dsa/causal_bound.py"
#: The bit-equality tests need the untiled bound from git; without it they skip, and say why.
needs_untiled = needs_reference(UNTILED_COMMIT, UNTILED_PATH)


@pytest.fixture(scope="module")
def base():
    return load_reference(UNTILED_COMMIT, UNTILED_PATH, "causal_bound_untiled")


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


def test_the_column_tile_is_the_largest_power_of_two_that_fits_twice():
    tile = CB.COLUMN_TILE
    assert tile & (tile - 1) == 0
    per_tile = CB._BOUND_BYTES_PER_COLUMN * CB._BOUND_TILES_IN_FLIGHT
    assert tile * per_tile <= SBUF_BYTES_PER_PARTITION < 2 * tile * per_tile


def test_the_bytes_per_column_are_the_kernels_column_tiles():
    """The kernel source allocates the tiles `_BOUND_BYTES_PER_COLUMN` counts."""
    source = inspect.getsource(CB._causal_bound_nki.func)   # the traced Python function
    itemsize = {"float32": 4, "uint8": 1}
    tiles = re.findall(r"nl\.ndarray\(\(height, cw\), dtype=nl\.(\w+), buffer=nl\.sbuf\)", source)
    assert sum(itemsize[t] for t in tiles) == CB._BOUND_BYTES_PER_COLUMN


@needs_untiled
@pytest.mark.parametrize("rows", [1, 130])
@pytest.mark.parametrize("width", [512, 2048, 8192, UNTILED_WIDTH_MAX])
def test_column_tiles_equal_the_untiled_kernel_bit_for_bit(base, rows, width):
    scores, lengths = _case(rows, width, seed=width + rows)
    want = base.dsa_causal_bound(scores, lengths, POOL)
    got = CB.dsa_causal_bound(scores, lengths, POOL)
    assert CB.causal_bound_dispatch_counters() == (1, 0)
    assert torch.equal(got.view(torch.int32), want.view(torch.int32))


@pytest.mark.parametrize("rows", [1, 130])
@pytest.mark.parametrize("width", [UNTILED_WIDTH_MAX + 1, 2 * UNTILED_WIDTH_MAX, 4 * UNTILED_WIDTH_MAX])
def test_column_tiles_match_the_oracle_past_the_old_width(rows, width):
    scores, lengths = _case(rows, width, seed=3 * width + rows)
    got = CB.dsa_causal_bound(scores, lengths, POOL)
    assert CB.causal_bound_dispatch_counters() == (1, 0)
    want = CB.dsa_causal_bound_torch_oracle(scores, lengths, POOL)
    assert torch.equal(got.view(torch.int32), want.view(torch.int32))
