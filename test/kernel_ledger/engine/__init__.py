# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Kernel-to-model engine (ported subset) for the GLM-5.3-Flash kernel ledger.

Node-graph framework: define the decode forward pass, propagate per-rank shapes,
compute a roofline per node, and roll measured kernel times up into a step.
Ported from ``inference-engine-overview/.../test/model_config_gen/engine``; see each
module's docstring for what changed.
"""

from .collectives import ConstantCollectiveModel
from .graph import Edge, ForwardPassGraph, RollupResult, compute_rollup
from .node import (
    CollectiveDim,
    CollectiveNode,
    CollectiveType,
    InputSharding,
    KernelName,
    KernelNode,
    MLPNode,
    Node,
    OpType,
    Port,
    RooflineMode,
    ShardingSpec,
    WeightSharding,
)

__all__ = [
    "CollectiveDim",
    "CollectiveNode",
    "CollectiveType",
    "ConstantCollectiveModel",
    "Edge",
    "ForwardPassGraph",
    "InputSharding",
    "KernelName",
    "KernelNode",
    "MLPNode",
    "Node",
    "OpType",
    "Port",
    "RooflineMode",
    "RollupResult",
    "ShardingSpec",
    "WeightSharding",
    "compute_rollup",
]
