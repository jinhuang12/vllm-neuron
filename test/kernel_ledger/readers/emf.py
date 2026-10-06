# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Measured-kernel record, EMF style.

Ported for the GLM-5.3-Flash kernel ledger from
``inference-engine-overview/container-source/model/test/kernels/kernel_to_model_benchmarking/emf_reader.py``.
The source parses one EMF JSON per kernel test run into a ``KernelResult`` keyed by
the model config name. The team's hardware microbenchmarks write one JSON per kernel
family instead (``reports/*_micro.json``), so the per-file parser is replaced by the
family readers in ``micro.py``; ``KernelResult`` keeps the source fields and adds:

  - ``variant``: "before" (the 5938748 kernel) or "after" (the worker's kernel);
  - ``p90_us``, ``iterations``: the benchmark's spread and sample count;
  - ``location``: "device" or "host" (the 5938748 sampler runs on the host);
  - ``record``: the shapes the benchmark recorded, in its own vocabulary;
  - ``points``: ``((distinct local experts, us), ...)`` for the routed-expert curve;
  - ``sources``: ``file#case`` of every case averaged into this result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class KernelResult:
    """A single kernel measurement (median over the benchmark's iterations)."""

    config_name: str
    kernel_api: str
    latency_us: float
    mbu_percent: float = -1.0
    mfu_percent: float = -1.0
    accuracy_hw: float = -1.0
    test_name: str = ""
    # -- added for the microbenchmark JSONs ---------------------------------------------
    variant: str = "before"
    p90_us: Optional[float] = None
    iterations: Optional[int] = None
    location: str = "device"
    record: Dict = field(default_factory=dict, compare=False, hash=False)
    points: Optional[Tuple[Tuple[int, float], ...]] = None
    sources: Tuple[str, ...] = ()
    #: other cases at the same config that repeat a measurement: (source, median us); not used
    repeats: Tuple[Tuple[str, float], ...] = ()
