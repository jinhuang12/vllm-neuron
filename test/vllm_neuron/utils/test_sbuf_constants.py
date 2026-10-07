# SPDX-License-Identifier: Apache-2.0
"""The host-side SBUF constants of ``neuron_utils`` equal the values the vendor stack states.

Host code sizes kernel tiles before any kernel is traced, where ``nl.tile_size`` has no
backend for its byte sizes. So ``neuron_utils`` writes the numbers down, and these tests
hold them to NKI's partition bound and to nkilib's usable SBUF size.
"""

from __future__ import annotations

import nki.language as nl
import pytest

from vllm_neuron.functional.dsa import causal_bound, sentinel_order
from vllm_neuron.utils import neuron_utils as NU


def test_the_partition_count_is_the_nki_partition_bound():
    assert NU.SBUF_PARTITIONS == nl.tile_size.pmax


@pytest.mark.parametrize("module", [causal_bound, sentinel_order],
                         ids=lambda m: m.__name__.rsplit(".", 1)[1])
def test_the_dsa_row_tiles_are_one_partition_count_high(module):
    assert module.PARTITION_MAX == NU.SBUF_PARTITIONS


def test_the_usable_bytes_are_nkilib_max_available_sbuf_size():
    from nkilib.experimental.moe.bwd.bwmm_bwd_dropless import MAX_AVAILABLE_SBUF_SIZE

    assert NU.SBUF_BYTES_PER_PARTITION == MAX_AVAILABLE_SBUF_SIZE


def test_the_usable_bytes_are_the_total_less_the_named_reservations():
    reserved = (NU.SBUF_DYNAMIC_DMA_SCRATCH_BYTES + NU.SBUF_EVAL_ACCEL_RESERVED_BYTES
                + NU.SBUF_TAIL_RESERVED_BYTES)
    assert NU.SBUF_BYTES_PER_PARTITION == NU.SBUF_TOTAL_BYTES_PER_PARTITION - reserved
