# SPDX-License-Identifier: Apache-2.0
"""The fused ``noaux_tc`` router at prefill chunks above one launch.

The kernel holds the RMSNorm output of every token of a launch in SBUF
(``[128, T, H / 128]`` in the router matmul dtype). At GLM-5.3-Flash's H 4096 in bf16 a
launch of 8192 tokens needs 524,288 B per partition, more than SBUF has, and
neuronx-cc refused the chunk-8192 prefill graph with ``NCC_IGCA037`` in this kernel.
The entry now runs a long chunk as launches of at most
:func:`router.noaux_tc_token_tile` tokens. No stage reduces across tokens, so on the
simulator:

* the result equals the one-launch router bit for bit (the last commit before the
  token tiles, loaded from git, is the reference), at chunks that need two and three
  launches;
* the launch count is the tile count, and a chunk that fits takes one launch.

``test_dsa_wide_cpu_compile.py`` holds the compile at the served H, E and 8192 tokens.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from vllm_neuron.functional.moe import router as seam

from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig
from vllm_neuron.utils.neuron_utils import SBUF_BYTES_PER_PARTITION

from test.vllm_neuron.functional.moe.test_router_token_axis import (
    NORM_TOPK_PROB,
    ROUTED_SCALING_FACTOR,
    TINY_H,
    TOP_K,
    _SimulatorCounter,
    build_hidden,
    set_equal_rows,
)
from test.vllm_neuron.functional.reference_at_commit import load_reference, needs_reference

#: The last commit whose router runs every chunk as one launch: the bit-equality reference.
ONE_LAUNCH_COMMIT = "b17526a"

#: The fixture's tile: `build_hidden` makes bf16 hidden states of width `TINY_H`.
TILE = seam.noaux_tc_token_tile(TINY_H, torch.bfloat16)


#: The reference file at :data:`ONE_LAUNCH_COMMIT`.
ONE_LAUNCH_PATH = "vllm_neuron/functional/moe/router.py"
#: The bit-equality tests need the one-launch router from git; without it they skip, and say why.
needs_one_launch = needs_reference(ONE_LAUNCH_COMMIT, ONE_LAUNCH_PATH)


@pytest.fixture(scope="module")
def base():
    return load_reference(ONE_LAUNCH_COMMIT, ONE_LAUNCH_PATH, "router_one_launch")


def _route(entry, tokens: int):
    hidden_states, gamma, router_weights, bias = build_hidden(tokens)
    eps = inspect.signature(entry).parameters["eps"].default
    with _SimulatorCounter() as sim:
        out = entry(
            hidden_states=hidden_states, gamma=gamma, router_weights=router_weights,
            correction_bias=bias, top_k=TOP_K, eps=eps, norm_topk_prob=NORM_TOPK_PROB,
            routed_scaling_factor=ROUTED_SCALING_FACTOR)
    return out, sim.calls


def _norm_bytes(tokens: int, hidden: int, dtype: torch.dtype) -> int:
    """SBUF bytes per partition of one launch's RMSNorm output, `[128, T, H / 128]`."""
    return tokens * (hidden // 128) * dtype.itemsize


# 7168 in bf16 is a width where SBUF, not the nkilib bound, sets the tile.
@pytest.mark.parametrize("hidden", [TINY_H, Glm5NextTextConfig().hidden_size, 7168])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_the_token_tile_is_the_most_tokens_that_fit_one_launch(hidden, dtype):
    tile = seam.noaux_tc_token_tile(hidden, dtype)
    multiple = seam._NOAUX_TC_T_MULTIPLE
    assert tile % multiple == 0
    assert tile <= seam._NKILIB_ROUTER_MAX_TOKENS
    assert _norm_bytes(tile, hidden, dtype) <= SBUF_BYTES_PER_PARTITION
    # Most: one more multiple is past the nkilib bound or past SBUF.
    more = tile + multiple
    assert (more > seam._NKILIB_ROUTER_MAX_TOKENS
            or _norm_bytes(more, hidden, dtype) > SBUF_BYTES_PER_PARTITION)


def test_the_served_chunk_8192_does_not_fit_one_launch():
    hidden = Glm5NextTextConfig().hidden_size
    assert _norm_bytes(8192, hidden, torch.bfloat16) > SBUF_BYTES_PER_PARTITION
    assert seam.noaux_tc_token_tile(hidden, torch.bfloat16) < 8192


def test_a_hidden_size_too_wide_for_one_multiple_raises():
    multiple = seam._NOAUX_TC_T_MULTIPLE
    hidden = 128 * (SBUF_BYTES_PER_PARTITION // (multiple * 2) + 1)
    assert _norm_bytes(multiple, hidden, torch.bfloat16) > SBUF_BYTES_PER_PARTITION
    with pytest.raises(seam.NoauxTcRouterError, match="SBUF"):
        seam.noaux_tc_token_tile(hidden, torch.bfloat16)


@needs_one_launch
@pytest.mark.parametrize("tokens", [
    TILE + 300,        # two launches, the second partial
    2 * TILE + 1,      # three launches, the last one 256 rows
])
def test_a_long_chunk_equals_the_one_launch_router_bit_for_bit(base, tokens):
    tile = TILE
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
    _, launches = _route(seam.noaux_tc_rmsnorm_router_topk, TILE)
    assert launches == 1
