# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Collective latency model with the source engine's lookup interface.

Ported for the GLM-5.3-Flash kernel ledger from
``inference-engine-overview/container-source/model/test/kernels/kernel_to_model_benchmarking/collectives.py``.
The source ``CollectiveLookup`` interpolates a measured CSV
(``collective_latencies/collective_ops_performance_trn2.csv``) that is not in the
snapshot, so it is not ported. ``ConstantCollectiveModel`` keeps its interface
(``lookup_us(op_type, replica_group, data_size_bytes)`` and ``lookup_ms``) and
returns ``latency + bytes / bandwidth``.

GLM defaults: 17.5 us per collective over 64 ranks, the fastest of the 630 traced
TP=64 all-reduces (``waits.md``: "min 17.5 us"; planner ``trn2_constants_glm.json``:
"about 17.5 us each = 2*log2(64)=12 hops x 1.5 us/hop"), and 100 GB/s achieved
volume bandwidth (planner doc constant, ``PLANNER_REPORT.md`` caveat 2).
"""

from __future__ import annotations

from dataclasses import dataclass

#: Measured minimum TP=64 all-reduce latency, us (waits.md, waits_1_collective.py).
TP64_COLLECTIVE_LATENCY_US = 17.5

#: Achieved collective volume bandwidth, bytes/s (planner constant).
COLLECTIVE_BANDWIDTH_BYTES_PER_S = 100e9


@dataclass(frozen=True)
class ConstantCollectiveModel:
    """``latency_us + data_size_bytes / bandwidth`` for every op and replica group."""

    latency_us: float = TP64_COLLECTIVE_LATENCY_US
    bandwidth_bytes_per_s: float = COLLECTIVE_BANDWIDTH_BYTES_PER_S

    def lookup_us(self, op_type: str, replica_group: str, data_size_bytes: int) -> float:
        """Latency in microseconds. ``op_type`` and ``replica_group`` are accepted for
        interface parity with the source ``CollectiveLookup`` and do not change the result."""
        return self.latency_us + max(data_size_bytes, 0) / self.bandwidth_bytes_per_s * 1e6

    def lookup_ms(self, op_type: str, replica_group: str, data_size_bytes: int) -> float:
        return self.lookup_us(op_type, replica_group, data_size_bytes) / 1000.0

    def __call__(self, op_type: str, replica_group: str, data_size_bytes: int) -> float:
        return self.lookup_us(op_type, replica_group, data_size_bytes)
