# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Forward pass graph, topological ordering, validation, and rollup.

Ported for the GLM-5.3-Flash kernel ledger from
``inference-engine-overview/container-source/model/test/model_config_gen/engine/graph.py``.
Changes against the source (everything else is the source text):
  - New ``ForwardPassGraph.merge()``: a zero-cost join of same-shaped branches (GLM
    layers pick KDA or DSA attention, dense or MoE MLP, and both feed one all-reduce).
  - The test registry (register_model / get_configs / KERNEL_TEST_MAP) is not ported:
    the ledger emits shapes with ``--emit-shapes`` instead of parametrizing tests.

This module provides:
  - ForwardPassGraph: ordered node collection with edge wiring and validation
  - compute_rollup(): walks graph, sums latencies from EMF + collectives
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .node import KernelName, KernelNode, Node, Port, RooflineMode, CollectiveNode

logger = logging.getLogger(__name__)


# ============================================================================
# Edge
# ============================================================================


@dataclass(frozen=True)
class Edge:
    """A directed connection from one node's output port to another node's input port."""

    src_node: Node
    src_port: str
    dst_node: Node
    dst_port: str


# ============================================================================
# ForwardPassGraph
# ============================================================================


class ForwardPassGraph:
    """Ordered collection of nodes with validated edge wiring.

    Nodes are added via add_node(), then wired via connect().
    The engine computes execution order via topological sort.
    """

    def __init__(self, model_name: str, phase: str):
        """
        Args:
            model_name: identifier for this model (e.g., "vega")
            phase: "prefill" or "decode"
        """
        self.model_name = model_name
        self.phase = phase
        self._nodes: List[Node] = []
        self._edges: List[Edge] = []
        self._node_set: set = set()

    @property
    def nodes(self) -> List[Node]:
        return list(self._nodes)

    @property
    def edges(self) -> List[Edge]:
        return list(self._edges)

    def add_node(self, node: Node) -> Node:
        """Add a node to the graph. Returns the node for chaining."""
        if id(node) in self._node_set:
            raise ValueError(f"Node '{node.name}' already in graph")
        self._nodes.append(node)
        self._node_set.add(id(node))
        return node

    def connect(self, src: Node, src_port: str, dst: Node, dst_port: str):
        """Wire src's output port to dst's input port.

        - Sets dst's input port shape from src's output port shape
        - Validates that src_port exists on src
        - Validates that dst_port is a declared input on dst
        - For CollectiveNode dst: input is set, output auto-derived
        """
        if id(src) not in self._node_set:
            raise ValueError(f"Source node '{src.name}' not in graph")
        if id(dst) not in self._node_set:
            raise ValueError(f"Destination node '{dst.name}' not in graph")

        # Validate src has the output port
        src_output = src.get_output_port(src_port)
        if src_output is None:
            available = src.output_port_names
            raise ValueError(
                f"Node '{src.name}' has no output port '{src_port}'. Available: {available}"
            )

        # Wire: set dst's input port from src's output port
        # Any port name is accepted — the wiring IS the declaration
        dst.set_input_port(dst_port, src_output)

        self._edges.append(Edge(src_node=src, src_port=src_port, dst_node=dst, dst_port=dst_port))

    def reshape(self, src: Node, src_port: str, new_shape: tuple) -> Node:
        """Insert a zero-cost reshape node between src's output and downstream consumers.

        Validates that total elements match. Returns the reshape node (use its
        "output" port to connect downstream).
        """
        src_output = src.get_output_port(src_port)
        if src_output is None:
            raise ValueError(f"Node '{src.name}' has no output port '{src_port}'")

        old_elements = math.prod(src_output.shape)
        new_elements = math.prod(new_shape)
        if old_elements != new_elements:
            raise ValueError(
                f"Reshape: element count mismatch — {src_output.shape} ({old_elements}) "
                f"vs {new_shape} ({new_elements})"
            )

        reshape_node = Node(
            f"{src.name}_{src_port}_reshape",
            layer_type=src.layer_type if hasattr(src, 'layer_type') else "",
            layer_count=src.layer_count if hasattr(src, 'layer_count') else 1,
        )
        reshape_node._output_ports_explicit = {
            "output": Port(name="output", shape=new_shape, dtype=src_output.dtype)
        }
        self.add_node(reshape_node)
        self.connect(src, src_port, reshape_node, "input")
        return reshape_node

    def slice(self, src: Node, src_port: str, dim: int, size: int) -> Node:
        """Insert a zero-cost slice node that takes a chunk along a dimension.

        Models operations like: each rank selects its local heads from a
        replicated/gathered tensor. Zero cost (no communication, no compute).

        Args:
            src: source node
            src_port: output port name on source
            dim: dimension to slice along (supports negative indexing)
            size: size of the slice along that dim

        Returns the slice node (use its "output" port to connect downstream).
        """
        src_output = src.get_output_port(src_port)
        if src_output is None:
            raise ValueError(f"Node '{src.name}' has no output port '{src_port}'")

        shape = src_output.shape
        ndims = len(shape)
        actual_dim = dim if dim >= 0 else ndims + dim

        if size > shape[actual_dim]:
            raise ValueError(
                f"slice: requested size ({size}) > source dim {dim} size ({shape[actual_dim]})"
            )

        out_shape = list(shape)
        out_shape[actual_dim] = size

        slice_node = Node(
            f"{src.name}_{src_port}_slice",
            layer_type=src.layer_type if hasattr(src, 'layer_type') else "",
            layer_count=src.layer_count if hasattr(src, 'layer_count') else 1,
        )
        slice_node._output_ports_explicit = {
            "output": Port(name="output", shape=tuple(out_shape), dtype=src_output.dtype)
        }
        self.add_node(slice_node)
        self.connect(src, src_port, slice_node, "input")
        return slice_node

    def split(self, src: Node, src_port: str, dim: int, sizes: Dict[str, int]) -> Node:
        """Insert a zero-cost split node that splits a source port along a dimension.

        Inverse of concat. Creates named output ports for each split chunk.

        Args:
            src: source node
            src_port: output port name on source
            dim: dimension to split along (supports negative indexing)
            sizes: dict of {output_port_name: size_along_dim}

        Validates that sum of sizes equals the source dim size.
        Returns the split node (connect downstream from its named ports).
        """
        src_output = src.get_output_port(src_port)
        if src_output is None:
            raise ValueError(f"Node '{src.name}' has no output port '{src_port}'")

        shape = src_output.shape
        ndims = len(shape)
        actual_dim = dim if dim >= 0 else ndims + dim

        total_size = sum(sizes.values())
        if total_size != shape[actual_dim]:
            raise ValueError(
                f"split: sizes sum ({total_size}) != source dim {dim} size ({shape[actual_dim]})"
            )

        # Build output ports with split shapes
        output_ports = {}
        for port_name, size in sizes.items():
            out_shape = list(shape)
            out_shape[actual_dim] = size
            output_ports[port_name] = Port(name=port_name, shape=tuple(out_shape), dtype=src_output.dtype)

        split_node = Node(
            f"{src.name}_{src_port}_split",
            layer_type=src.layer_type if hasattr(src, 'layer_type') else "",
            layer_count=src.layer_count if hasattr(src, 'layer_count') else 1,
        )
        split_node._output_ports_explicit = output_ports
        self.add_node(split_node)
        self.connect(src, src_port, split_node, "input")
        return split_node

    def concat(self, sources: List[tuple], dim: int) -> Node:
        """Insert a zero-cost concat node that joins multiple source ports along a dimension.

        Args:
            sources: list of (node, port_name) tuples to concatenate
            dim: dimension to concatenate along (supports negative indexing)

        Validates that all shapes match except on the concat dim.
        Returns the concat node (use its "output" port to connect downstream).
        """
        if len(sources) < 2:
            raise ValueError("concat requires at least 2 sources")

        ports = []
        for src, port_name in sources:
            port = src.get_output_port(port_name)
            if port is None:
                raise ValueError(f"Node '{src.name}' has no output port '{port_name}'")
            ports.append(port)

        # Validate shapes match except on concat dim
        ndims = len(ports[0].shape)
        actual_dim = dim if dim >= 0 else ndims + dim
        for i, port in enumerate(ports[1:], 1):
            if len(port.shape) != ndims:
                raise ValueError(
                    f"concat: all ports must have same number of dims. "
                    f"Port 0 has {ndims} dims, port {i} has {len(port.shape)} dims"
                )
            for d in range(ndims):
                if d == actual_dim:
                    continue
                if port.shape[d] != ports[0].shape[d]:
                    raise ValueError(
                        f"concat: shape mismatch on dim {d} — "
                        f"port 0 has {ports[0].shape[d]}, port {i} has {port.shape[d]}"
                    )

        # Compute output shape
        out_shape = list(ports[0].shape)
        out_shape[actual_dim] = sum(p.shape[actual_dim] for p in ports)

        # Get layer info from first source
        first_src = sources[0][0]
        name_parts = [f"{s.name}_{p}" for s, p in sources]
        concat_name = f"concat_{'_'.join(p for _, p in sources)}"

        concat_node = Node(
            concat_name,
            layer_type=first_src.layer_type if hasattr(first_src, 'layer_type') else "",
            layer_count=first_src.layer_count if hasattr(first_src, 'layer_count') else 1,
        )
        concat_node._output_ports_explicit = {
            "output": Port(name="output", shape=tuple(out_shape), dtype=ports[0].dtype)
        }
        self.add_node(concat_node)

        # Connect all sources
        for i, (src, port_name) in enumerate(sources):
            self.connect(src, port_name, concat_node, f"input_{i}")

        return concat_node

    def merge(self, sources: List[tuple], name: str) -> Node:
        """Insert a zero-cost join of branches that produce the same tensor.

        Written new for the ledger. Each source is ``(node, port_name)``; all ports
        must have the same shape and dtype, and the output is that port. Used where a
        layer runs one of two sub-blocks (KDA or DSA attention, dense or MoE MLP) and
        both feed the same downstream node, whose ``layer_count`` is the sum.
        """
        if len(sources) < 2:
            raise ValueError("merge requires at least 2 sources")
        ports = []
        for src, port_name in sources:
            port = src.get_output_port(port_name)
            if port is None:
                raise ValueError(f"Node '{src.name}' has no output port '{port_name}'")
            ports.append(port)
        for i, port in enumerate(ports[1:], 1):
            if port.shape != ports[0].shape or port.dtype != ports[0].dtype:
                raise ValueError(
                    f"merge '{name}': port {i} is {port.shape}/{port.dtype}, "
                    f"port 0 is {ports[0].shape}/{ports[0].dtype}"
                )
        merge_node = Node(name, layer_type="merge", layer_count=0)
        merge_node._output_ports_explicit = {
            "output": Port(name="output", shape=ports[0].shape, dtype=ports[0].dtype)
        }
        self.add_node(merge_node)
        for i, (src, port_name) in enumerate(sources):
            self.connect(src, port_name, merge_node, f"input_{i}")
        return merge_node

    def resolve_order(self) -> List[Node]:
        """Topological sort using Kahn's algorithm. Tiebreaks by insertion order."""
        # Build adjacency and in-degree
        in_degree: Dict[int, int] = {id(n): 0 for n in self._nodes}
        adjacency: Dict[int, List[Node]] = {id(n): [] for n in self._nodes}

        for edge in self._edges:
            src_id = id(edge.src_node)
            dst_id = id(edge.dst_node)
            if dst_id not in adjacency.get(src_id, []):
                adjacency[src_id].append(edge.dst_node)
                in_degree[dst_id] += 1

        # Deduplicate adjacency edges (multiple ports between same pair)
        seen_edges: set = set()
        clean_adjacency: Dict[int, List[Node]] = {id(n): [] for n in self._nodes}
        clean_in_degree: Dict[int, int] = {id(n): 0 for n in self._nodes}

        for edge in self._edges:
            pair = (id(edge.src_node), id(edge.dst_node))
            if pair not in seen_edges:
                seen_edges.add(pair)
                clean_adjacency[id(edge.src_node)].append(edge.dst_node)
                clean_in_degree[id(edge.dst_node)] += 1

        # Kahn's with insertion-order tiebreak
        insertion_order = {id(n): i for i, n in enumerate(self._nodes)}
        queue = deque(
            sorted(
                [n for n in self._nodes if clean_in_degree[id(n)] == 0],
                key=lambda n: insertion_order[id(n)],
            )
        )

        result: List[Node] = []
        while queue:
            node = queue.popleft()
            result.append(node)
            for neighbor in clean_adjacency[id(node)]:
                clean_in_degree[id(neighbor)] -= 1
                if clean_in_degree[id(neighbor)] == 0:
                    queue.append(neighbor)
            # Re-sort queue by insertion order for stable tiebreaking
            queue = deque(sorted(queue, key=lambda n: insertion_order[id(n)]))

        if len(result) != len(self._nodes):
            raise ValueError("Graph has a cycle — topological sort failed")

        return result

    def validate(self) -> List[str]:
        """Validate graph integrity. Returns list of warnings/errors."""
        errors = []

        # Check for unconnected input ports (non-root nodes)
        root_nodes = set()
        for edge in self._edges:
            # Nodes that have incoming edges are not roots
            pass

        dst_nodes = {id(e.dst_node) for e in self._edges}
        for node in self._nodes:
            if id(node) in dst_nodes:
                # Non-root: check all declared inputs are wired
                for port_name in node.input_port_names:
                    port = node.get_input_port(port_name)
                    if port is None:
                        errors.append(f"Node '{node.name}': input port '{port_name}' is not connected")

        # Check topological sort succeeds (no cycles)
        try:
            self.resolve_order()
        except ValueError as e:
            errors.append(str(e))

        return errors

    def get_kernel_nodes(self) -> List[KernelNode]:
        """All KernelNodes with a kernel_name (for test generation)."""
        return [
            n for n in self.resolve_order()
            if isinstance(n, KernelNode) and n.kernel_name is not None
        ]

    def get_all_configs(self) -> Dict[KernelName, List[Dict]]:
        """Extract kernel configs grouped by KernelName for test registration."""
        configs: Dict[KernelName, List[Dict]] = defaultdict(list)
        for node in self.resolve_order():
            if isinstance(node, KernelNode) and node.kernel_name is not None:
                cfg = node.get_kernel_config()
                if cfg is not None:
                    configs[node.kernel_name].append(cfg)
        return configs

    def __repr__(self) -> str:
        return f"ForwardPassGraph('{self.model_name}', phase='{self.phase}', nodes={len(self._nodes)}, edges={len(self._edges)})"


# ============================================================================
# Rollup
# ============================================================================


@dataclass
class RollupResult:
    """Aggregated latency result for one forward pass configuration."""

    model_name: str
    phase: str
    total_latency_us: float = 0.0
    kernel_latency_us: float = 0.0
    collective_latency_us: float = 0.0
    roofline_latency_us: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    breakdown: Dict[str, Dict] = field(default_factory=dict)
    missing_kernels: List[str] = field(default_factory=list)


def compute_rollup(
    graph: ForwardPassGraph,
    emf_results: Optional[Dict[str, float]] = None,
    collective_lookup=None,
    tokens: int = 0,
    world_size: int = 64,
) -> RollupResult:
    """Walk graph in topological order, sum latencies.

    Args:
        graph: the forward pass graph to evaluate
        emf_results: dict mapping config_name → measured latency_us
        collective_lookup: CollectiveLookup instance (has .lookup_us method) or
                          callable(op_type, replica_group, data_size_bytes) → latency_us
        tokens: total tokens for throughput calculation
        world_size: total number of ranks (for collective replica_group formation)

    For each node:
      - CollectiveNode: uses collective_lookup if available, else hardcoded/roofline
      - KernelNode with name: looks up EMF measured latency, falls back to roofline
      - KernelNode without name: uses roofline as the estimate
    """
    emf_results = emf_results or {}
    result = RollupResult(model_name=graph.model_name, phase=graph.phase)
    missing = []

    for node in graph.resolve_order():
        lat_us = 0.0

        # Common shape info for breakdown
        input_shapes = {k: v.shape for k, v in node.input_ports.items()}
        output_shapes = {k: v.shape for k, v in node.output_ports.items()}
        weight_shapes = dict(node.weights) if isinstance(node, KernelNode) else {}

        if isinstance(node, CollectiveNode):
            if node.roofline_mode == RooflineMode.HARDCODED and node.hardcoded_latency_us is not None:
                lat_us = node.hardcoded_latency_us
            elif collective_lookup is not None:
                # Form replica_group string: g{group_size}_s{stride}
                # Use node.stride if explicitly set, else default to world_size/group_size
                stride = node.stride if node.stride is not None else (world_size // node.group_size if node.group_size > 0 else 1)
                replica_group = f"g{node.group_size}_s{stride}"
                if hasattr(collective_lookup, 'lookup_us'):
                    looked_up = collective_lookup.lookup_us(
                        node.collective_type.value,
                        replica_group,
                        node.message_size_bytes,
                    )
                else:
                    looked_up = collective_lookup(
                        node.collective_type.value,
                        replica_group,
                        node.message_size_bytes,
                    )
                lat_us = looked_up if looked_up else 0.0
            total_us = lat_us * node.layer_count
            result.collective_latency_us += total_us
            result.breakdown[f"{node.layer_type}/{node.name}"] = {
                "latency_us": total_us,
                "per_layer_us": lat_us,
                "count": node.layer_count,
                "type": "collective",
                "input_shapes": input_shapes,
                "output_shapes": output_shapes,
            }

        elif isinstance(node, KernelNode) and node.kernel_name is not None:
            # Try measured EMF data
            measured = emf_results.get(node.name)
            if measured is not None and measured > 0:
                lat_us = measured
            else:
                lat_us = node.compute_roofline()
                missing.append(node.name)
            total_us = lat_us * node.layer_count
            result.kernel_latency_us += total_us
            result.breakdown[f"{node.layer_type}/{node.name}"] = {
                "latency_us": total_us,
                "per_layer_us": lat_us,
                "count": node.layer_count,
                "type": "kernel",
                "kernel_name": node.kernel_name.value,
                "source": "measured" if node.name in emf_results else "roofline",
                "input_shapes": input_shapes,
                "output_shapes": output_shapes,
                "weight_shapes": weight_shapes,
            }

        elif isinstance(node, KernelNode) and node.kernel_name is None:
            # KernelNode without kernel_name — use roofline if it has hardcoded latency
            lat_us = node.compute_roofline()
            if lat_us > 0:
                total_us = lat_us * node.layer_count
                result.kernel_latency_us += total_us
                result.breakdown[f"{node.layer_type}/{node.name}"] = {
                    "latency_us": total_us,
                    "per_layer_us": lat_us,
                    "count": node.layer_count,
                    "type": "kernel",
                    "kernel_name": "none",
                    "source": "roofline",
                    "input_shapes": input_shapes,
                    "output_shapes": output_shapes,
                    "weight_shapes": weight_shapes,
                }

        else:
            # Plain Node (reshape, concat, split, slice) — zero cost, skip
            pass

    result.total_latency_us = result.kernel_latency_us + result.collective_latency_us + result.roofline_latency_us
    result.missing_kernels = missing

    if tokens > 0 and result.total_latency_us > 0:
        result.throughput_tokens_per_sec = tokens / (result.total_latency_us / 1e6)

    return result
