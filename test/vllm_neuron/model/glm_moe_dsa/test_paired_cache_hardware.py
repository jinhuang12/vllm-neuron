# SPDX-License-Identifier: Apache-2.0
"""Bounded fullgraph smoke for GLM paired MLA and indexer caches."""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import pytest
import torch
from torch import nn

from vllm_neuron.envs import get_compile_backend_name
from vllm_neuron.model.glm_moe_dsa.cache import (
    INDEXER_CACHE_BYTES,
    INDEXER_CACHE_PART_BYTES,
    MLA_CACHE_HEAD_SIZE,
    MLA_CACHE_PART_SIZE,
    gather_paged_cache_pair,
    write_paged_cache_pair,
)


_BLOCK_SIZE = 16
_PHYSICAL_BLOCKS = 8
_LOGICAL_BLOCKS = 4
_TOKEN_COUNT = 9


class _PairedCacheRoundTripProbe(nn.Module):
    """Write and gather both production split-cache payload contracts."""

    def forward(
        self,
        mla_k_cache: torch.Tensor,
        mla_v_cache: torch.Tensor,
        indexer_k_cache: torch.Tensor,
        indexer_v_cache: torch.Tensor,
        mla_values: torch.Tensor,
        indexer_values: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_table: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        write_paged_cache_pair(
            mla_k_cache,
            mla_v_cache,
            mla_values,
            slot_mapping,
            _BLOCK_SIZE,
        )
        write_paged_cache_pair(
            indexer_k_cache,
            indexer_v_cache,
            indexer_values,
            slot_mapping,
            _BLOCK_SIZE,
        )
        return (
            gather_paged_cache_pair(mla_k_cache, mla_v_cache, block_table),
            gather_paged_cache_pair(
                indexer_k_cache,
                indexer_v_cache,
                block_table,
            ),
        )


def _round_trip_inputs() -> tuple[torch.Tensor, ...]:
    mla_k_cache = torch.zeros(
        _PHYSICAL_BLOCKS,
        1,
        _BLOCK_SIZE,
        MLA_CACHE_PART_SIZE,
        dtype=torch.bfloat16,
    ).to(torch.float8_e4m3fn)
    mla_v_cache = torch.zeros(
        _PHYSICAL_BLOCKS,
        1,
        _BLOCK_SIZE,
        MLA_CACHE_PART_SIZE,
        dtype=torch.bfloat16,
    ).to(torch.float8_e4m3fn)
    indexer_k_cache = torch.zeros(
        _PHYSICAL_BLOCKS,
        1,
        _BLOCK_SIZE,
        INDEXER_CACHE_PART_BYTES,
        dtype=torch.uint8,
    )
    indexer_v_cache = torch.zeros_like(indexer_k_cache)

    mla_values = (
        torch.remainder(
            torch.arange(_TOKEN_COUNT * MLA_CACHE_HEAD_SIZE, dtype=torch.int32),
            127,
        )
        .sub(63)
        .view(1, _TOKEN_COUNT, MLA_CACHE_HEAD_SIZE)
        .to(torch.bfloat16)
    )
    indexer_values = (
        torch.remainder(
            torch.arange(_TOKEN_COUNT * INDEXER_CACHE_BYTES, dtype=torch.int32),
            251,
        )
        .view(1, _TOKEN_COUNT, INDEXER_CACHE_BYTES)
        .to(torch.uint8)
    )

    # The final valid row is written before -1 and upper-OOB sentinels. This
    # catches sentinel aliasing to the final physical slot.
    slot_mapping = torch.tensor(
        [83, 87, 18, 100, 33, 15, 127, -1, 128],
        dtype=torch.int64,
    )
    block_table = torch.tensor(
        [[5, 1, -1, 8], [6, 2, 0, 7]],
        dtype=torch.int32,
    )
    return (
        mla_k_cache,
        mla_v_cache,
        indexer_k_cache,
        indexer_v_cache,
        mla_values,
        indexer_values,
        slot_mapping,
        block_table,
    )


def _expected_round_trip(
    values: torch.Tensor,
) -> torch.Tensor:
    expected = torch.zeros(
        2,
        _LOGICAL_BLOCKS * _BLOCK_SIZE,
        values.shape[-1],
        dtype=torch.bfloat16 if values.dtype is torch.float8_e4m3fn else values.dtype,
    )
    flat_values = values.reshape(_TOKEN_COUNT, values.shape[-1]).to(expected.dtype)
    for request, logical_row, value_row in (
        (0, 3, 0),
        (0, 7, 1),
        (0, 18, 2),
        (1, 4, 3),
        (1, 17, 4),
        (1, 47, 5),
        (1, 63, 6),
    ):
        expected[request, logical_row] = flat_values[value_row]
    return expected.to(values.dtype)


def _assert_invalid_pages_are_zero(
    actual_mla: torch.Tensor,
    actual_indexer: torch.Tensor,
) -> None:
    for actual in (actual_mla, actual_indexer):
        # Request 0 maps logical block 2 to -1 and block 3 to one-past-end.
        assert torch.count_nonzero(actual[0, 32:48].float()) == 0
        assert torch.count_nonzero(actual[0, 48:64].float()) == 0


def test_paired_cache_round_trip_cpu_contract() -> None:
    inputs = _round_trip_inputs()
    actual_mla, actual_indexer = _PairedCacheRoundTripProbe()(*inputs)

    assert actual_mla.dtype is torch.float8_e4m3fn
    assert torch.equal(
        actual_mla,
        _expected_round_trip(inputs[4].to(torch.float8_e4m3fn)),
    )
    assert torch.equal(actual_indexer, _expected_round_trip(inputs[5]))
    _assert_invalid_pages_are_zero(actual_mla, actual_indexer)


def test_paired_cache_probe_is_one_production_fullgraph() -> None:
    forward_source = inspect.getsource(_PairedCacheRoundTripProbe.forward)
    hardware_source = inspect.getsource(test_paired_cache_round_trip_fullgraph_neuron)

    assert forward_source.count("write_paged_cache_pair(") == 2
    assert forward_source.count("gather_paged_cache_pair(") == 2
    assert ").to(torch.float8_e4m3fn)" in inspect.getsource(_round_trip_inputs)
    assert "backend=get_compile_backend_name()" in hardware_source
    assert "fullgraph=True" in hardware_source
    assert "dynamic=False" in hardware_source
    assert hardware_source.count("torch.compile(") == 1
    assert 'os.environ["GLM_STAGE5_PAIRED_CACHE_COMPILE_DIR"]' in hardware_source


@pytest.mark.skipif(
    os.environ.get("GLM_STAGE5_PAIRED_CACHE_HARDWARE") != "1"
    or not Path("/dev/neuron0").exists(),
    reason="requires explicit GLM_STAGE5_PAIRED_CACHE_HARDWARE=1 on Neuron",
)
def test_paired_cache_round_trip_fullgraph_neuron() -> None:
    device = torch.device("neuron:0")
    cpu_inputs = _round_trip_inputs()
    expected_mla = _expected_round_trip(cpu_inputs[4].to(torch.float8_e4m3fn))
    expected_indexer = _expected_round_trip(cpu_inputs[5])
    module = _PairedCacheRoundTripProbe().eval().to(device)
    compile_root = Path(os.environ["GLM_STAGE5_PAIRED_CACHE_COMPILE_DIR"])
    compiled = torch.compile(
        module,
        backend=get_compile_backend_name(),
        fullgraph=True,
        dynamic=False,
        options={"compiler_workdir": str(compile_root / "paired-cache-round-trip")},
    )

    actual_mla, actual_indexer = tuple(
        value.cpu() for value in compiled(*(value.to(device) for value in cpu_inputs))
    )

    assert torch.equal(actual_mla, expected_mla)
    assert torch.equal(actual_indexer, expected_indexer)
    _assert_invalid_pages_are_zero(actual_mla, actual_indexer)
