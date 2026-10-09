# SPDX-License-Identifier: Apache-2.0
"""The LNC-aware dispatches outside ``functional/dsa`` follow ``NEURON_LOGICAL_NC_CONFIG``.

Every kernel seam here that may split its work over both cores of an LNC2 pair asks
:func:`vllm_neuron.functional.dsa.launch_grid.lnc_pair`, the reader the DSA kernels use.
Each case calls one seam at a geometry that takes the LNC2 route under ``2``, so the three
served settings show both answers: ``2`` takes the two-core route, ``1`` and unset take the
one-program route. Any other setting is refused by name with ``LaunchGridError``; before
the shared reader, these seams compared the raw string with ``"2"`` and silently launched
one program for it.
"""

import importlib

import pytest
import torch

from vllm_neuron.functional.attention import mla_decode as MD
from vllm_neuron.functional.attention import mla_dense_window as DW
from vllm_neuron.functional.attention import mla_sparse as MS
from vllm_neuron.functional.dsa.launch_grid import LaunchGridError
from vllm_neuron.functional.glue import kda_output, kda_projections, mhc_pre
from vllm_neuron.functional.kda import fused_decode as FD
from vllm_neuron.functional.mhc import hyper_connection as HC
from vllm_neuron.functional.moe import expert_decode as ED
from vllm_neuron.functional.moe import fused_fp8 as FF
from vllm_neuron.functional.moe import token_gather_combine as TGC
from vllm_neuron.functional.moe.fused_fp8_pack import PackedExperts

# ``vllm_neuron.functional`` re-exports a function of the same name over the module.
BW = importlib.import_module("vllm_neuron.functional.blockwise_fp8_mm")

NC_CONFIG = "NEURON_LOGICAL_NC_CONFIG"
#: The two-core answer of a seam that launches a ``[2]`` grid on an LNC2 pair.
PAIR = 2
#: The one-program answer.
ONE = 1


def _grid_programs(grid: tuple[int, ...]) -> int:
    """Programs of an NKI launch grid: ``()`` is one program, ``(n,)`` is ``n``."""
    return grid[0] if grid else ONE


def _recorded_programs(monkeypatch, module, output, seam) -> int:
    """Run ``seam()`` with ``module.wrap_nki`` recording; the programs of its one launch.

    The stand-in launches nothing: indexing selects the grid, calling records it and
    returns ``output(*operands)``.
    """
    launches = []

    class Launch:
        def __init__(self, kernel):
            self.grid = ONE

        def __getitem__(self, grid):
            self.grid = grid
            return self

        def __call__(self, *operands):
            launches.append(self.grid)
            return output(*operands)

    monkeypatch.setattr(module, "wrap_nki", Launch)
    seam()
    assert len(launches) == 1
    return launches[0]


def _mla_sparse(monkeypatch) -> int:
    """Row-tiled nope entry, one head, the served latent, two query blocks."""
    seq, latent, topk = 32, MS.TARGET_LATENT_RANK, 2176
    return _recorded_programs(
        monkeypatch, MS, lambda q, *rest: torch.zeros(q.shape, dtype=torch.float32),
        lambda: MS.mla_sparse_attention(
            torch.zeros(seq, 1, latent), torch.zeros(2048, latent),
            torch.zeros(seq, topk, dtype=torch.int32), latent**-0.5))


def _token_gather_combine(monkeypatch) -> int:
    """Two token tiles of one hidden tile, two slots per token."""
    tokens, slots = 2 * TGC.TOKEN_TILE, 2
    return _recorded_programs(
        monkeypatch, TGC, lambda contribution, index, valid: torch.zeros(
            index.shape[0], contribution.shape[1]),
        lambda: TGC.token_gather_combine(
            torch.zeros(tokens * slots, TGC.HIDDEN_TILE),
            torch.zeros(tokens, slots, dtype=torch.int64),
            torch.ones(tokens, slots)))


def _fused_fp8_route(monkeypatch) -> int:
    """The measured decode geometry: two programs on the compact LNC2 kernel, else one.

    Both entries are ``[2]`` launches; only the compact decode kernel is the LNC2 pair
    route, so it answers :data:`PAIR` and the generic entry :data:`ONE`.
    """
    hidden, intermediate, experts, blocks = 4096, 512, 18, 8
    routes = []

    def entry(answer):
        def run(hidden_states, weights, scales, rows, *rest, **tiles):
            routes.append(answer)
            return torch.empty((*rows.shape, hidden_states.shape[1]), device="meta")
        return run

    monkeypatch.setattr(FF, "_FUSED_DECODE_EXPERTS", entry(PAIR))
    monkeypatch.setattr(FF, "_FUSED_EXPERTS", entry(ONE))
    panels, blocks_h = 3 * (intermediate // 128), hidden // 128
    FF.fused_fp8_experts(
        torch.empty((2, hidden), dtype=torch.bfloat16, device="meta"),
        PackedExperts(
            torch.empty((experts, panels, 128, blocks_h, 128), dtype=torch.float8_e4m3fn,
                        device="meta"),
            torch.empty((experts, panels, blocks_h), dtype=torch.float32, device="meta")),
        torch.empty((blocks, 1), dtype=torch.int32, device="meta"),
        torch.empty((blocks, 1), dtype=torch.int32, device="meta"),
        torch.empty((2, experts), dtype=torch.float32, device="meta"),
        torch.empty((128, 3), dtype=torch.float32, device="meta"))
    assert len(routes) == 1
    return routes[0]


#: (seam, programs of one launch at a geometry that splits under LNC2).
SEAMS = {
    "mla_decode": lambda mp: MD._programs(2),
    "mla_dense_window": lambda mp: DW._programs(DW.ROW_TILE + 1),
    "mla_sparse": _mla_sparse,
    "blockwise_fp8_mlp": lambda mp: _grid_programs(BW.mlp_launch_grid(2 * BW.TILE_SIZE)),
    "kda_output": lambda mp: kda_output.launch_programs(),
    "kda_projections": lambda mp: kda_projections.launch_programs(),
    "mhc_pre": lambda mp: mhc_pre.launch_programs(),
    "kda_fused_decode": lambda mp: _grid_programs(FD.fused_decode_grid(2)),
    "hyper_connection": lambda mp: HC.launch_programs(2 * HC.PARTITION_MAX),
    "expert_decode": lambda mp: ED.default_programs(2, 2),
    "fused_fp8": _fused_fp8_route,
    "token_gather_combine": _token_gather_combine,
}


@pytest.fixture(params=sorted(SEAMS))
def seam(request, monkeypatch):
    return lambda: SEAMS[request.param](monkeypatch)


def test_lnc2_takes_the_two_core_route(seam, monkeypatch):
    monkeypatch.setenv(NC_CONFIG, "2")
    assert seam() == PAIR


@pytest.mark.parametrize("setting", [None, "1"], ids=["unset", "lnc1"])
def test_one_core_settings_launch_one_program(seam, setting, monkeypatch):
    if setting is None:
        monkeypatch.delenv(NC_CONFIG, raising=False)
    else:
        monkeypatch.setenv(NC_CONFIG, setting)
    assert seam() == ONE


@pytest.mark.parametrize("setting", ["3", "abc"])
def test_an_unserved_setting_is_refused_by_name(seam, setting, monkeypatch):
    monkeypatch.setenv(NC_CONFIG, setting)
    with pytest.raises(LaunchGridError, match=NC_CONFIG):
        seam()
