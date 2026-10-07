# SPDX-License-Identifier: Apache-2.0
"""The fused ``noaux_tc`` router at prefill chunks above 2048 tokens.

The kernel holds the RMSNorm output of every token of a launch in SBUF
(``[128, T, H / 128]`` bf16). At H 4096 a launch of 8192 tokens needs 524,288 B per
partition of the 229,376 B there are, and neuronx-cc refused the chunk-8192 prefill
graph with ``NCC_IGCA037`` in this kernel (runs/chunk8k, b17526a). The entry now runs
a long chunk as launches of at most :data:`router.NOAUX_TC_TOKEN_TILE` tokens. No stage
reduces across tokens, so on the simulator:

* the result equals b17526a's single launch bit for bit (b17526a's own file, loaded
  from git, is the reference), at chunks that need two and three launches;
* the launch count is the tile count, and a chunk that fits takes one launch.

``test_dsa_wide_cpu_compile.py`` holds the compile at H 4096, E 288 and 8192 tokens.
"""

from __future__ import annotations

import importlib.util
import inspect
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

from vllm_neuron.functional.moe import router as seam

from test.vllm_neuron.functional.moe.test_router_token_axis import (
    NORM_TOPK_PROB,
    ROUTED_SCALING_FACTOR,
    TOP_K,
    _SimulatorCounter,
    build_hidden,
    set_equal_rows,
)

BASE_COMMIT = "b17526a"


def _load_base_module():
    root = Path(seam.__file__).resolve().parents[3]
    source = subprocess.run(
        ["git", "-C", str(root), "show",
         f"{BASE_COMMIT}:vllm_neuron/functional/moe/router.py"],
        check=True, capture_output=True).stdout
    path = Path(tempfile.mkdtemp(prefix="router_base_")) / "router_b17526a.py"
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location("router_b17526a", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def base():
    return _load_base_module()


def _route(entry, tokens: int):
    hidden_states, gamma, router_weights, bias = build_hidden(tokens)
    eps = inspect.signature(entry).parameters["eps"].default
    with _SimulatorCounter() as sim:
        out = entry(
            hidden_states=hidden_states, gamma=gamma, router_weights=router_weights,
            correction_bias=bias, top_k=TOP_K, eps=eps, norm_topk_prob=NORM_TOPK_PROB,
            routed_scaling_factor=ROUTED_SCALING_FACTOR)
    return out, sim.calls


def test_the_token_tile_fits_the_partition_and_the_two_core_split():
    tile = seam.NOAUX_TC_TOKEN_TILE
    assert tile % seam._NOAUX_TC_T_MULTIPLE == 0
    # RMSNorm output of one launch at the served H 4096, bf16, per partition.
    assert tile * (4096 // 128) * 2 <= 229_376


@pytest.mark.parametrize("tokens", [
    seam.NOAUX_TC_TOKEN_TILE + 300,        # two launches, the second partial
    2 * seam.NOAUX_TC_TOKEN_TILE + 1,      # three launches, the last one 256 rows
])
def test_a_long_chunk_equals_b17526a_one_launch_bit_for_bit(base, tokens):
    tile = seam.NOAUX_TC_TOKEN_TILE
    padded = -(-tokens // seam._NOAUX_TC_T_MULTIPLE) * seam._NOAUX_TC_T_MULTIPLE
    seam.reset_noaux_tc_counters()
    got, launches = _route(seam.noaux_tc_rmsnorm_router_topk, tokens)
    assert launches == -(-padded // tile)
    assert seam.noaux_tc_dispatch_counters() == (1, 0)
    want, base_launches = _route(base.noaux_tc_rmsnorm_router_topk, tokens)
    assert base_launches == 1
    for name, g, w in zip(("logits", "index", "affinities", "substrate_index"), got, want):
        assert g.shape == w.shape == (tokens, w.shape[1]), name
        assert torch.equal(g, w), name
    # And it is the router: the corrected selection matches the torch oracle.
    hidden_states, gamma, router_weights, bias = build_hidden(tokens)
    eps = inspect.signature(seam.noaux_tc_rmsnorm_router_topk).parameters["eps"].default
    _, oracle_index, _, _ = seam.noaux_tc_rmsnorm_router_topk_torch_oracle(
        hidden_states, gamma, router_weights, bias, eps, NORM_TOPK_PROB,
        ROUTED_SCALING_FACTOR)
    assert set_equal_rows(got[1], oracle_index) == tokens


def test_a_chunk_that_fits_takes_one_launch():
    seam.reset_noaux_tc_counters()
    _, launches = _route(seam.noaux_tc_rmsnorm_router_topk, seam.NOAUX_TC_TOKEN_TILE)
    assert launches == 1
