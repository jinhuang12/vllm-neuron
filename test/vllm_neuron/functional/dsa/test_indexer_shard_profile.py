# SPDX-License-Identifier: Apache-2.0
"""How ``indexer_shard_profile.py`` finds a compiler copy's transfer among the DMA packets."""

from __future__ import annotations

from test.vllm_neuron.functional.dsa import indexer_shard_profile as profile

#: Bytes the copy under test reads from HBM.
SIZE = 2048
#: When the copy is issued (ns).
ISSUED = 1000
#: Bytes each DMA queue adds to a copy's payload on the device (a 2 MiB copy over 16
#: queues moved 131076 bytes a queue in the C = 2048 profile).
COMPLETION_WRITE = 4


def _packets(*rows):
    """``(start, queue, end, bytes)`` rows as the profile's packets of physical core 0."""
    return sorted((start, queue, end, size, 0) for start, queue, end, size in rows)


def test_the_transfer_is_the_first_burst_after_the_issue_that_moves_the_copys_bytes():
    half = SIZE // 2 + COMPLETION_WRITE
    gap = profile.BURST_GAP_NS
    packets = _packets(
        # Before the issue: never this copy, though within a burst gap of it.
        (ISSUED - gap // 2, "Q1-E64", ISSUED - gap // 4, SIZE // 2),
        # After the issue, but another family's burst of three times the bytes.
        (ISSUED + 100, "Q2-E64", ISSUED + 200, 3 * SIZE),
        # The copy: two queues of one family, each with its completion write.
        (ISSUED + 500, "Q1-E64", ISSUED + 1100, half),
        (ISSUED + 510, "Q1-E65", ISSUED + 1400, half),
        # More than a burst gap later: a second burst that fits as well.
        (ISSUED + 1400 + gap + 1, "Q1-E66", ISSUED + 1400 + gap + 100, SIZE),
    )
    took, matches = profile.copy_transfer(packets, ISSUED, SIZE)
    assert took == 1400
    assert matches == 2


def test_a_copy_whose_bytes_no_burst_moves_has_no_transfer():
    packets = _packets((ISSUED + 10, "Q1-E64", ISSUED + 20, SIZE // 2),
                       (ISSUED + 10, "Q2-E64", ISSUED + 20, 2 * SIZE))
    assert profile.copy_transfer(packets, ISSUED, SIZE) == (None, 0)


def test_a_queue_name_without_an_engine_suffix_is_its_own_family():
    packets = _packets((ISSUED, "qSyncSpillReload0-Q3", ISSUED + 50, SIZE))
    assert profile.copy_transfer(packets, ISSUED, SIZE) == (50, 1)
