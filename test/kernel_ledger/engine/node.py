# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Node primitives: Node, KernelNode, CollectiveNode, MLPNode, Port, enums, and roofline computation.

Ported for the GLM-5.3-Flash kernel ledger from
``inference-engine-overview/container-source/model/test/model_config_gen/engine/node.py``.
Changes against the source (everything else is the source text):
  - Hardware constants: one trn2 logical core at LNC=2 (PEAK_FLOPS 79e12 bf16,
    HBM_BW 716e9 B/s, the parallelism planner's trn2 constants). The source used
    150e12 / 740e9 per NeuronCore.
  - MIN_KERNEL_LATENCY_US: 2.0 us, one dependent DMA (``moe_host.md`` "dependent-DMA
    latency 2 us ASSUMED"). The source used 5.0.
  - KernelName lists the GLM decode kernels the ledger measures, not Vega's.
  - MoENode (prefill block skew) and SDPANode (dense/banded SDPA) are not ported:
    GLM decode uses the node types in ``decode_nodes.py`` instead.

This module defines:
  - Enums: KernelName, CollectiveType, CollectiveDim, RooflineMode, OpType,
            InputSharding, WeightSharding
  - Port: typed tensor descriptor with shape and dtype
  - ShardingSpec: combined sharding declaration for a node
  - Node: base computation node with auto FLOPs + roofline
  - CollectiveNode: collective communication (auto-derives output, model lookup)
  - MLPNode: dense gated MLP (gate_up + down)

Assumptions:
  1. Input tensors are always shaped (B, S, H) — batch, sequence, hidden.
  2. Weight tensors have heads as a SEPARATE dimension when applicable,
     e.g., (Nq, Dh, H) not (Nq*Dh, H). This keeps sharding unambiguous.
  3. Matmul convention: input (..., K) x weight (..., K, N) → output (..., N).
     Last dim of input is contraction dim. Last dim of weight is output dim.
  4. Input sharding divides: BATCH_SHARDED → dim 0 (B),
     SEQUENCE_PARALLEL/CONTEXT_PARALLEL → dim -2 (S).
  5. Weight sharding default dims: TP_HEAD/TP_MONARCH_BLOCKS/EP → dim 0,
     TP_HIDDEN/TP_MOE → dim -1. User can override with explicit dim.
  6. Memory bandwidth roofline defaults to weight-load only (assumes activations
     are in SBUF from fused megakernels). Set include_activation_mem=True to
     include input reads + output writes.
  7. User provides GLOBAL (unsharded) weight shapes. The engine applies
     ShardingSpec to derive per-rank shapes.
  8. Input shapes are INFERRED from upstream node's output port via graph.connect()
     and are always PER-RANK. Only root nodes (no upstream connection) need
     explicit input_ports in the constructor.
  9. OpType.NONE nodes are pure shape transforms (slice/select for CP/DP) —
     zero FLOPs, output = input after input sharding applied.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

# ============================================================================
# Hardware Constants (TRN2, one logical core = 2 physical cores at LNC=2)
# ============================================================================

# Source: /home/ubuntu/glm53f-planner-20261005/scripts/configs/trn2_constants_glm.json
# (te_peak_flops 79e12, hbm_bw_bytes_per_s 716e9, "PER NeuronCore (logical)").
PEAK_FLOPS = 79e12  # 79 TFLOPS bf16 per logical core
HBM_BW = 716e9  # 716 GB/s per logical core (profile Metadata hbm_ddr_bandwidth)

# Minimum kernel latency (us): one dependent DMA hop. No kernel can run faster
# than this regardless of compute/memory roofline.
MIN_KERNEL_LATENCY_US = 2.0

DTYPE_BYTES = {
    "bf16": 2,
    "fp32": 4,
    "fp16": 2,
    "fp8": 1,
    "mxfp4": 1,
    "int32": 4,
    "int8": 1,
}


# ============================================================================
# Enums
# ============================================================================


class KernelName(Enum):
    """GLM-5.3-Flash decode kernels the ledger measures (one entry per measured unit).

    Naming convention (source engine): _TKG suffix = decode (token-group) kernels,
    no suffix = shared kernels (lm_head, sampling).
    """

    MHC_SINKHORN_TKG = "mhc_sinkhorn_tkg"
    MHC_COMBINE_TKG = "mhc_combine_tkg"
    KDA_DECODE_TKG = "kda_decode_tkg"
    DSA_MLA_LAYER_TKG = "dsa_mla_layer_tkg"
    RMSNORM_TKG = "rmsnorm_tkg"
    DENSE_MLP_TKG = "dense_mlp_tkg"
    SHARED_EXPERT_TKG = "shared_expert_tkg"
    RMSNORM_ROUTER_TOPK_TKG = "rmsnorm_router_topk_tkg"
    MOE_EXPERTS_TKG = "moe_experts_tkg"
    LM_HEAD = "lm_head"
    SAMPLING = "sampling"


class CollectiveType(Enum):
    """Type of collective communication operation."""

    ALL_GATHER = "AllGather"
    REDUCE_SCATTER = "ReduceScatter"
    ALL_REDUCE = "AllReduce"
    ALL_TO_ALL = "AllToAll"


class CollectiveDim(Enum):
    """Which logical dimension the collective operates on."""

    SEQUENCE = "sequence"
    HEADS = "heads"
    HIDDEN = "hidden"
    EXPERTS = "experts"
    BATCH = "batch"
    MONARCH_BLOCK = "monarch_block"


class RooflineMode(Enum):
    """How roofline latency is computed for a node."""

    MAX = "max"  # max(compute_time, memory_time) — DEFAULT
    ADDITIVE = "additive"  # compute_time + memory_time
    COMPUTE_ONLY = "compute"  # only flops / (peak * mfu)
    MEMORY_ONLY = "memory"  # only bytes / (bw * mbu)
    HARDCODED = "hardcoded"  # user provides latency_us directly


class OpType(Enum):
    """Determines how FLOPs are auto-computed from shapes."""

    MATMUL = "matmul"  # 2 * M * K * N
    ELEMENTWISE = "elementwise"  # 2 * num_elements
    CUSTOM = "custom"  # user passes flops explicitly
    NONE = "none"  # no compute, pure shape transform (slice/select for CP/DP)


class InputSharding(Enum):
    """How the input activation is distributed across ranks."""

    REPLICATED = "replicated"
    SEQUENCE_PARALLEL = "sequence_parallel"
    CONTEXT_PARALLEL = "context_parallel"
    BATCH_SHARDED = "batch_sharded"
    HEAD_PARALLEL = "head_parallel"


class WeightSharding(Enum):
    """How the weight tensor is distributed across ranks."""

    REPLICATED = "replicated"
    TP_HEAD = "tp_head"
    TP_HIDDEN = "tp_hidden"
    TP_MOE = "tp_moe"
    TP_MONARCH_BLOCKS = "tp_monarch_blocks"
    EP = "ep"


# ============================================================================
# Port
# ============================================================================


@dataclass(frozen=True)
class Port:
    """Typed tensor descriptor — a named tensor with shape and dtype."""

    name: str
    shape: Tuple[int, ...]
    dtype: str = "bf16"

    @property
    def size_bytes(self) -> int:
        return math.prod(self.shape) * DTYPE_BYTES[self.dtype]

    @property
    def num_elements(self) -> int:
        return math.prod(self.shape)

    def with_shape(self, new_shape: Tuple[int, ...]) -> Port:
        """Return a copy with a different shape."""
        return Port(name=self.name, shape=new_shape, dtype=self.dtype)


# ============================================================================
# ShardingSpec
# ============================================================================


# Default dim for each weight sharding type (used when user omits dim)
_WEIGHT_SHARDING_DEFAULT_DIM = {
    WeightSharding.REPLICATED: 0,
    WeightSharding.TP_HEAD: 0,           # heads always at dim 0
    WeightSharding.TP_HIDDEN: -1,        # hidden/output always last dim
    WeightSharding.TP_MOE: -1,           # intermediate always last dim
    WeightSharding.TP_MONARCH_BLOCKS: 0, # blocks at dim 0
    WeightSharding.EP: 0,                # experts at dim 0
}


@dataclass(frozen=True)
class ShardingSpec:
    """Sharding specification for a node. Supports combined sharding.

    Input sharding: List of (type, degree) — dim is fixed by convention:
      - BATCH_SHARDED: dim 0
      - SEQUENCE_PARALLEL / CONTEXT_PARALLEL: dim -2

    Weight sharding: List of (type, degree) or (type, degree, dim).
      If dim is omitted, uses default:
        - TP_HEAD: dim 0 (heads always separate dim at 0)
        - TP_HIDDEN / TP_MOE: dim -1 (output/intermediate always last)
        - TP_MONARCH_BLOCKS / EP: dim 0

    Examples:
        # TP on heads (default dim 0): weight (Nq, Dh, H) → (Nq//64, Dh, H)
        ShardingSpec(weight=[(WeightSharding.TP_HEAD, 64)])

        # TP on hidden (default dim -1): weight (H_in, H_out) → (H_in, H_out//64)
        ShardingSpec(weight=[(WeightSharding.TP_HIDDEN, 64)])

        # Override dim: TP_HEAD on dim 1 instead of 0
        ShardingSpec(weight=[(WeightSharding.TP_HEAD, 64, 1)])

        # MoE: EP on dim 0 + TP_moe on dim -1
        ShardingSpec(weight=[(WeightSharding.EP, 16), (WeightSharding.TP_MOE, 4)])

        # Sequence parallel input + TP on weight hidden
        ShardingSpec(input=[(InputSharding.SEQUENCE_PARALLEL, 64)],
                     weight=[(WeightSharding.TP_HIDDEN, 64)])
    """

    input: List[Tuple[InputSharding, int]] = field(default_factory=lambda: [(InputSharding.REPLICATED, 1)])
    weight: list = field(default_factory=lambda: [(WeightSharding.REPLICATED, 1)])


# ============================================================================
# Node (base class)
# ============================================================================


class Node:
    """Base node: minimal identity + port interface. No compute, no roofline.

    Used by ForwardPassGraph for topological sort and edge wiring.
    Subclass KernelNode for compute nodes with weights/FLOPs/roofline.
    """

    def __init__(self, name: str, *, layer_type: str = "", layer_count: int = 1, fused_in: Optional["Node"] = None):
        self.name = name
        self.layer_type = layer_type
        self.layer_count = layer_count
        self.fused_in = fused_in
        self._input_ports: Dict[str, Port] = {}
        self._output_ports_explicit: Dict[str, Port] = {}

    @property
    def input_ports(self) -> Dict[str, Port]:
        return self._input_ports

    @property
    def output_ports(self) -> Dict[str, Port]:
        return self._output_ports_explicit

    def set_input_port(self, port_name: str, port: Port):
        self._input_ports[port_name] = port

    def get_input_port(self, port_name: str) -> Optional[Port]:
        return self._input_ports.get(port_name)

    def get_output_port(self, port_name: str) -> Optional[Port]:
        return self.output_ports.get(port_name)

    @property
    def input_port_names(self) -> List[str]:
        return list(self._input_ports.keys()) if self._input_ports else ["input"]

    @property
    def output_port_names(self) -> List[str]:
        return list(self.output_ports.keys())

    def __repr__(self) -> str:
        return f"Node('{self.name}', layer_type='{self.layer_type}', count={self.layer_count})"


# ============================================================================
# KernelNode
# ============================================================================


class KernelNode(Node):
    """Computation node with weights, sharding, FLOPs, and roofline.

    Supports 1-to-1, 1-to-many, many-to-1, and many-to-many port configs.

    Input ports: INFERRED from connected upstream nodes via graph.connect().
                 Only root nodes (no upstream) need explicit input_ports.

    Output ports: Can be explicit (user provides dict of named Ports) or
                  AUTO-DERIVED for single-output MATMUL/ELEMENTWISE cases.

    Weight shapes: USER-PROVIDED as GLOBAL (unsharded) shapes. Engine applies
                   ShardingSpec to derive per-rank shapes.

    For exotic FLOPs (SDPA, MoE routing, etc.): use OpType.CUSTOM + flops=.
    """

    def __init__(
        self,
        name: str,
        *,
        kernel_name: Optional[KernelName] = None,
        layer_type: str = "",
        layer_count: int = 1,
        fused_in: Optional["Node"] = None,
        input_ports: Optional[Dict[str, Port]] = None,
        output_ports: Optional[Dict[str, Port]] = None,
        weights: Optional[Dict[str, Tuple[int, ...]]] = None,
        weight_dtype: str = "bf16",
        sharding: Optional[ShardingSpec] = None,
        op_type: OpType = OpType.MATMUL,
        roofline_mode: RooflineMode = RooflineMode.COMPUTE_ONLY,
        roofline_mfu: float = 1.0,
        roofline_mbu: float = 1.0,
        target_mfu: Optional[float] = None,
        target_mbu: Optional[float] = None,
        include_activation_mem: bool = False,
        hardcoded_latency_us: Optional[float] = None,
        hardcoded_target_us: Optional[float] = None,
        min_latency_us: float = 5.0,
        latency_offset_us: float = 0.0,
        flops: Optional[int] = None,
    ):
        super().__init__(name, layer_type=layer_type, layer_count=layer_count, fused_in=fused_in)
        if fused_in is not None and kernel_name is not None:
            raise ValueError(
                f"Node '{name}': fused_in and kernel_name are mutually exclusive. "
                f"A node fused into '{fused_in.name}' cannot have its own kernel_name."
            )
        self._kernel_name = kernel_name
        self._output_ports_explicit = output_ports or {}
        self.weights = weights or {}
        self.weight_dtype = weight_dtype
        self.sharding = sharding or ShardingSpec()
        self._validate_sharding()
        self.op_type = op_type
        self.roofline_mode = roofline_mode
        self.roofline_mfu = roofline_mfu
        self.roofline_mbu = roofline_mbu
        self.target_mfu = target_mfu if target_mfu is not None else roofline_mfu
        self.target_mbu = target_mbu if target_mbu is not None else roofline_mbu
        self.include_activation_mem = include_activation_mem
        self.hardcoded_latency_us = hardcoded_latency_us
        self.hardcoded_target_us = hardcoded_target_us
        self.min_latency_us = min_latency_us
        self.latency_offset_us = latency_offset_us
        self._custom_flops = flops
        if input_ports:
            self._input_ports = input_ports

    def _validate_sharding(self):
        """A node must have EITHER input sharding OR weight sharding, not both."""
        has_input = any(t != InputSharding.REPLICATED for t, *_ in self.sharding.input)
        has_weight = any(t != WeightSharding.REPLICATED for t, *_ in self.sharding.weight)
        if has_input and has_weight:
            raise ValueError(
                f"Node '{self.name}': cannot have both non-trivial input sharding "
                f"({self.sharding.input}) and weight sharding ({self.sharding.weight}). "
                f"Use one or the other."
            )

    @property
    def kernel_name(self) -> Optional[KernelName]:
        return self._kernel_name

    # ------------------------------------------------------------------
    # Port interface (extends base Node with auto-derivation)
    # ------------------------------------------------------------------

    @property
    def output_ports(self) -> Dict[str, Port]:
        """Named output ports (explicit or auto-derived from input + weight shapes).

        Auto-derivation handles:
          - 1 input + 1 weight → 1 output named "output"
          - 1 input + N weights → N outputs named by weight key
          - N inputs + 1 weight → N outputs named by input key
          - ELEMENTWISE: outputs mirror inputs
          - CUSTOM/NONE: must provide output_ports explicitly (or use input sharding for NONE)
        """
        if self._output_ports_explicit:
            return self._output_ports_explicit
        return self._derive_output_ports()

    @property
    def output_port_names(self) -> List[str]:
        return list(self.output_ports.keys())

    # ------------------------------------------------------------------
    # Sharding resolution
    # ------------------------------------------------------------------

    def _apply_input_sharding(self, port: Port) -> Port:
        """Apply input sharding to get per-rank input shape.

        Assumes input layout is (B, S, H) or (S, H).

        Convention:
          - REPLICATED: no change
          - BATCH_SHARDED: shape[0] // degree (divides B, dim 0)
          - SEQUENCE_PARALLEL: shape[-2] // degree (divides S, second-to-last dim)
          - CONTEXT_PARALLEL: shape[-2] // degree (divides S, second-to-last dim)
        """
        shape = list(port.shape)
        for entry in self.sharding.input:
            sharding_type, degree = entry[0], entry[1]
            apply = entry[2] if len(entry) > 2 else True
            if not apply or sharding_type == InputSharding.REPLICATED or degree <= 1:
                continue

            if sharding_type == InputSharding.BATCH_SHARDED:
                if len(shape) < 1:
                    raise ValueError(
                        f"Node '{self.name}': BATCH_SHARDED requires at least 1 dim, "
                        f"got shape {port.shape}"
                    )
                if shape[0] % degree != 0:
                    raise ValueError(
                        f"Node '{self.name}': input dim 0 ({shape[0]}) not divisible "
                        f"by batch_sharded degree ({degree})"
                    )
                shape[0] = shape[0] // degree

            elif sharding_type in (InputSharding.SEQUENCE_PARALLEL, InputSharding.CONTEXT_PARALLEL):
                if len(shape) < 2:
                    raise ValueError(
                        f"Node '{self.name}': {sharding_type.value} requires at least 2 dims, "
                        f"got shape {port.shape}"
                    )
                dim_idx = -2  # S is second-to-last in (B, S, H) or (S, H)
                actual_idx = len(shape) + dim_idx  # convert to positive index
                if shape[actual_idx] % degree != 0:
                    raise ValueError(
                        f"Node '{self.name}': input dim {actual_idx} ({shape[actual_idx]}) not divisible "
                        f"by {sharding_type.value} degree ({degree})"
                    )
                shape[actual_idx] = shape[actual_idx] // degree

        return Port(name=port.name, shape=tuple(shape), dtype=port.dtype)

    def _apply_weight_sharding(self, w_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Apply weight sharding to get per-rank weight shape.

        Each sharding entry is (type, degree) or (type, degree, dim).
        If dim is omitted, uses the default from _WEIGHT_SHARDING_DEFAULT_DIM.
        """
        shape = list(w_shape)
        for entry in self.sharding.weight:
            if len(entry) == 3:
                sharding_type, degree, dim = entry
            else:
                sharding_type, degree = entry
                dim = _WEIGHT_SHARDING_DEFAULT_DIM[sharding_type]

            if sharding_type == WeightSharding.REPLICATED or degree <= 1:
                continue

            # Resolve negative dim index
            actual_idx = dim if dim >= 0 else len(shape) + dim

            if actual_idx < 0 or actual_idx >= len(shape):
                raise ValueError(
                    f"Node '{self.name}': {sharding_type.value} sharding specifies dim={dim}, "
                    f"but weight only has {len(shape)} dims (shape={w_shape})"
                )
            if shape[actual_idx] % degree != 0:
                raise ValueError(
                    f"Node '{self.name}': weight dim {dim} ({shape[actual_idx]}) not divisible "
                    f"by {sharding_type.value} degree ({degree})"
                )
            shape[actual_idx] = shape[actual_idx] // degree

        return tuple(shape)

    # ------------------------------------------------------------------
    # Output and FLOPs computation
    # ------------------------------------------------------------------

    def _resolve_matmul_pairs(self) -> List[Tuple[Port, Tuple[int, ...], str]]:
        """Match inputs to weights, validate K-dims.

        Input shapes: used AS-IS (already per-rank from upstream graph.connect).
        Weight shapes: sharded via _apply_weight_sharding (user provides global shapes).

        For root nodes with explicit input_ports: _apply_input_sharding is applied
        to transform global input shapes to per-rank.

        Returns list of (per_rank_input_port, per_rank_weight_shape, output_name) tuples.

        Handles:
          - 1 input + 1 weight → [("output", inp, w)]
          - 1 input + N weights → [(w_name, inp, w) for each weight]
          - N inputs + 1 weight → [(inp_name, inp, w) for each input]

        Raises ValueError on K-dim mismatch or ambiguous multi-multi case.
        """
        if not self._input_ports or not self.weights:
            return []

        # Apply input sharding if specified (e.g., CP sharding on dense KV proj)
        sharded_inputs = {
            name: self._apply_input_sharding(port)
            for name, port in self._input_ports.items()
        }

        # Weight shapes need sharding applied (user provides global)
        sharded_weights = {
            name: self._apply_weight_sharding(shape)
            for name, shape in self.weights.items()
        }

        n_in = len(sharded_inputs)
        n_wt = len(sharded_weights)

        pairs = []

        if n_in == 1 and n_wt == 1:
            inp = list(sharded_inputs.values())[0]
            w_name = list(sharded_weights.keys())[0]
            w_shape = list(sharded_weights.values())[0]
            if len(w_shape) >= 2:
                self._validate_k_dim(inp, w_shape, w_name)
                pairs.append((inp, w_shape, w_name))

        elif n_in == 1 and n_wt > 1:
            inp = list(sharded_inputs.values())[0]
            for w_name, w_shape in sharded_weights.items():
                if len(w_shape) < 2:
                    continue
                self._validate_k_dim(inp, w_shape, w_name)
                pairs.append((inp, w_shape, w_name))

        elif n_in > 1 and n_wt == 1:
            w_name = list(sharded_weights.keys())[0]
            w_shape = list(sharded_weights.values())[0]
            if len(w_shape) >= 2:
                for p_name, inp in sharded_inputs.items():
                    self._validate_k_dim(inp, w_shape, w_name)
                    pairs.append((inp, w_shape, p_name))

        elif n_in == n_wt and n_in > 1:
            # 1:1 positional match: zip inputs and weights by insertion order
            inputs_list = list(sharded_inputs.items())
            weights_list = list(sharded_weights.items())
            for (p_name, inp), (w_name, w_shape) in zip(inputs_list, weights_list):
                if len(w_shape) < 2:
                    continue
                self._validate_k_dim(inp, w_shape, w_name)
                pairs.append((inp, w_shape, w_name))

        else:
            raise ValueError(
                f"Node '{self.name}': can't auto-resolve "
                f"{n_in} inputs + {n_wt} weights. Use OpType.CUSTOM."
            )

        return pairs

    def _derive_output_ports(self) -> Dict[str, Port]:
        """Auto-derive output ports from input + weight shapes."""
        if not self._input_ports:
            return {}

        if self.op_type == OpType.NONE:
            # Pure shape transform: output = input after input sharding applied
            if len(self._input_ports) == 1:
                inp = list(self._input_ports.values())[0]
                sharded = self._apply_input_sharding(inp)
                return {"output": Port(name="output", shape=sharded.shape, dtype=sharded.dtype)}
            return {
                name: self._apply_input_sharding(p)
                for name, p in self._input_ports.items()
            }

        if self.op_type == OpType.ELEMENTWISE:
            if len(self._input_ports) == 1:
                inp = list(self._input_ports.values())[0]
                return {"output": Port(name="output", shape=inp.shape, dtype=inp.dtype)}
            return {
                name: Port(name=name, shape=p.shape, dtype=p.dtype)
                for name, p in self._input_ports.items()
            }

        if self.op_type != OpType.MATMUL or not self.weights:
            return {}

        pairs = self._resolve_matmul_pairs()
        dtype = list(self._input_ports.values())[0].dtype
        # Output shape depends on input vs weight dimensionality:
        # - input has MORE dims than weight → batched matmul, no new dim: input[:-1] + (N,)
        # - input has SAME or FEWER dims → new dim from weight batch: input[:-1] + weight[:-2] + (N,)
        ports = {}
        for inp, w_shape, out_name in pairs:
            if len(inp.shape) > len(w_shape):
                out_shape = inp.shape[:-1] + (w_shape[-1],)
            else:
                out_shape = inp.shape[:-1] + w_shape[:-2] + (w_shape[-1],)
            ports[out_name] = Port(name=out_name, shape=out_shape, dtype=dtype)
        return ports

    def _validate_k_dim(self, inp: Port, w_shape: Tuple[int, ...], w_name: str):
        """Validate contraction dim matches between input and weight."""
        K_input = inp.shape[-1]
        K_weight = w_shape[-2]
        if K_input != K_weight:
            raise ValueError(
                f"Node '{self.name}': shape mismatch — "
                f"input last dim ({K_input}) != weight '{w_name}' "
                f"second-to-last dim ({K_weight}). "
                f"Input shape: {inp.shape}, Weight shape: {w_shape}"
            )

    def compute_flops(self) -> int:
        """Auto-compute FLOPs based on OpType and port/weight shapes."""
        if self.op_type == OpType.CUSTOM:
            if self._custom_flops is None:
                raise ValueError(f"Node '{self.name}': OpType.CUSTOM requires flops= argument")
            return self._custom_flops
        elif self.op_type == OpType.MATMUL:
            return self._matmul_flops()
        elif self.op_type == OpType.ELEMENTWISE:
            return 2 * sum(p.num_elements for p in self.input_ports.values())
        elif self.op_type == OpType.NONE:
            return 0
        return 0

    def _matmul_flops(self) -> int:
        """Auto-compute matmul FLOPs based on input vs weight dimensionality.

        Same logic as output shape derivation:
        - input MORE dims than weight → batched matmul: 2 * prod(input) * weight[-1]
        - input SAME or FEWER dims → new dim: 2 * prod(input) * prod(weight) / K
        """
        pairs = self._resolve_matmul_pairs()
        total = 0
        for inp, w_shape, _ in pairs:
            if len(inp.shape) > len(w_shape):
                total += 2 * math.prod(inp.shape) * w_shape[-1]
            else:
                total += 2 * math.prod(inp.shape) * math.prod(w_shape) // w_shape[-2]
        return total

    # ------------------------------------------------------------------
    # Roofline computation
    # ------------------------------------------------------------------

    def _compute_latency(self, mfu: float, mbu: float) -> float:
        """Internal: compute latency for given utilization values."""
        if self.roofline_mode == RooflineMode.HARDCODED:
            if self.hardcoded_latency_us is None:
                raise ValueError(f"Node '{self.name}': HARDCODED mode requires hardcoded_latency_us")
            return self.hardcoded_latency_us

        flops = self.compute_flops()
        compute_us = (flops / (PEAK_FLOPS * mfu) * 1e6) if (flops > 0 and mfu > 0) else 0.0

        mem_bytes = self._total_memory_bytes()
        memory_us = (mem_bytes / (HBM_BW * mbu) * 1e6) if (mem_bytes > 0 and mbu > 0) else 0.0

        if self.roofline_mode == RooflineMode.MAX:
            return max(compute_us, memory_us)
        elif self.roofline_mode == RooflineMode.ADDITIVE:
            return compute_us + memory_us
        elif self.roofline_mode == RooflineMode.COMPUTE_ONLY:
            return compute_us
        elif self.roofline_mode == RooflineMode.MEMORY_ONLY:
            return memory_us
        return 0.0

    def compute_roofline(self) -> float:
        """Compute roofline latency (best achievable) in microseconds.
        Uses min_latency_us if set on node, otherwise falls back to MIN_KERNEL_LATENCY_US."""
        floor = self.min_latency_us if self.min_latency_us > 0 else MIN_KERNEL_LATENCY_US
        return max(self._compute_latency(self.roofline_mfu, self.roofline_mbu), floor) + self.latency_offset_us

    def compute_target(self) -> float:
        """Compute target latency (expected from current kernel) in microseconds.
        Uses hardcoded_target_us if set, otherwise computes from target_mfu/mbu."""
        if self.hardcoded_target_us is not None:
            return self.hardcoded_target_us
        floor = self.min_latency_us if self.min_latency_us > 0 else MIN_KERNEL_LATENCY_US
        return max(self._compute_latency(self.target_mfu, self.target_mbu), floor) + self.latency_offset_us

    def _total_memory_bytes(self) -> int:
        """Memory bytes for bandwidth roofline.

        By default only counts weight loads (assumes activations are in SBUF).
        Set include_activation_mem=True to also count input reads + output writes.
        Weight shapes are sharded (per-rank) for accurate memory accounting.
        """
        weight_bytes = sum(
            math.prod(self._apply_weight_sharding(s)) * DTYPE_BYTES[self.weight_dtype]
            for s in self.weights.values()
        )
        if not self.include_activation_mem:
            return weight_bytes
        input_bytes = sum(p.size_bytes for p in self.input_ports.values())
        output_bytes = sum(p.size_bytes for p in self.output_ports.values())
        return input_bytes + weight_bytes + output_bytes

    # ------------------------------------------------------------------
    # Kernel config extraction (for test parametrization)
    # TODO: get_kernel_config should return actual kernel config dataclasses
    # imported from test/model_config_gen/engine/kernel_configs.py
    # (e.g., QRKVConfig, MoEConfig, AttentionBlockConfig) instead of raw dicts.
    # ------------------------------------------------------------------

    def get_kernel_config(self) -> Optional[Dict]:
        """Extract a config dict for test registration. Override in subclasses."""
        if self.kernel_name is None:
            return None
        return {
            "name": self.name,
            "kernel_name": self.kernel_name.value,
            "layer_type": self.layer_type,
            "layer_count": self.layer_count,
            "input_shapes": {k: v.shape for k, v in self.input_ports.items()},
            "output_shapes": {k: v.shape for k, v in self.output_ports.items()},
            "weight_shapes": dict(self.weights),
            "weight_dtype": self.weight_dtype,
            "sharding": self.sharding,
            "flops": self.compute_flops(),
            "roofline_mfu": self.roofline_mfu,
            "roofline_mbu": self.roofline_mbu,
            "target_mfu": self.target_mfu,
            "target_mbu": self.target_mbu,
            "roofline_us": self.compute_roofline(),
            "target_us": self.compute_target(),
        }

    def __repr__(self) -> str:
        kernel = self.kernel_name.value if self.kernel_name else "None"
        return f"KernelNode('{self.name}', kernel={kernel}, layer_type='{self.layer_type}', count={self.layer_count})"


# ============================================================================
# CollectiveNode
# ============================================================================


class CollectiveNode(Node):
    """Collective communication node. Output shape auto-derived from input + collective semantics.

    User specifies:
      - collective_type: AG / RS / A2A
      - group_size: number of ranks in the collective group
      - dim: which logical dimension is affected (CollectiveDim enum)
      - dim_idx: which positional index in the shape tuple corresponds to `dim`

    Output shape is computed automatically:
      - AG:  output_shape[dim_idx] = input_shape[dim_idx] * group_size
      - RS:  output_shape[dim_idx] = input_shape[dim_idx] // group_size
      - A2A: user provides output_ports explicitly (redistribution is model-specific)

    Roofline: message_size from input port → CSV lookup, or HARDCODED.

    TODO: dim_idx should be auto-derived from CollectiveDim enum since input is
    always (B, S, H): SEQUENCE→1, BATCH→0, HIDDEN→2, etc. Remove dim_idx param
    and use a default mapping like _WEIGHT_SHARDING_DEFAULT_DIM.
    """

    def __init__(
        self,
        name: str,
        *,
        collective_type: CollectiveType,
        group_size: int,
        dim: CollectiveDim,
        dim_idx: int = 0,
        stride: Optional[int] = None,
        layer_type: str = "",
        layer_count: int = 1,
        roofline_mode: RooflineMode = RooflineMode.MAX,
        hardcoded_latency_us: Optional[float] = None,
        output_ports: Optional[Dict[str, Port]] = None,
        fused_in: Optional["Node"] = None,
    ):
        super().__init__(name, layer_type=layer_type, layer_count=layer_count, fused_in=fused_in)
        self.collective_type = collective_type
        self.group_size = group_size
        self.dim = dim
        self.dim_idx = dim_idx
        self.stride = stride
        self.roofline_mode = roofline_mode
        self.hardcoded_latency_us = hardcoded_latency_us
        self._collective_output_override = output_ports

    @property
    def output_ports(self) -> Dict[str, Port]:
        if self._collective_output_override:
            return self._collective_output_override
        derived = self._derive_collective_output()
        if derived is not None:
            return {"output": derived}
        return {}

    def _derive_collective_output(self) -> Optional[Port]:
        """Compute output shape from input + collective semantics."""
        inp = self._input_ports.get("input")
        if inp is None:
            return None

        in_shape = list(inp.shape)
        idx = self.dim_idx

        if self.collective_type == CollectiveType.ALL_GATHER:
            in_shape[idx] = in_shape[idx] * self.group_size
        elif self.collective_type == CollectiveType.REDUCE_SCATTER:
            in_shape[idx] = in_shape[idx] // self.group_size
        elif self.collective_type == CollectiveType.ALL_REDUCE:
            pass  # shape unchanged
        elif self.collective_type == CollectiveType.ALL_TO_ALL:
            return None  # A2A: require explicit output_ports

        return Port(name="output", shape=tuple(in_shape), dtype=inp.dtype)

    def compute_roofline(self) -> float:
        if self.roofline_mode == RooflineMode.HARDCODED:
            if self.hardcoded_latency_us is None:
                raise ValueError(f"CollectiveNode '{self.name}': HARDCODED requires hardcoded_latency_us")
            return self.hardcoded_latency_us
        return 0.0

    def compute_target(self) -> float:
        return self.compute_roofline()

    @property
    def message_size_bytes(self) -> int:
        """Total message volume for this collective (for CSV lookup).

        For AllGather: output size (= input_size * group_size)
        For ReduceScatter: input size (= output_size * group_size)
        For AllReduce: input size (same as output)
        """
        inp = self._input_ports.get("input")
        if inp is None:
            return 0
        if self.collective_type == CollectiveType.ALL_GATHER:
            return inp.size_bytes * self.group_size
        else:
            return inp.size_bytes

    def get_kernel_config(self) -> Optional[Dict]:
        return None

    def __repr__(self) -> str:
        return (
            f"CollectiveNode('{self.name}', type={self.collective_type.value}, "
            f"group={self.group_size}, dim={self.dim.value})"
        )


# ============================================================================
# MLPNode
# ============================================================================


class MLPNode(KernelNode):
    """Dense MLP node (gate_up + down) with auto FLOPs computation.

    A fused two-stage matmul: input × gate_up → intermediate, then intermediate × down → output.
    Since the two stages have different K-dims (H for gate_up, I for down), standard matmul
    auto-derive can't handle this. MLPNode computes FLOPs and memory explicitly.

    Weights:
        gate_up: (H, 2*I) — gate and up projections fused (SwiGLU style)
        down: (I, H) — down projection

    FLOPs = 2 * 3 * S * H * I per sample (gate + up + down, each 2*M*K*N)
    where S = prod(input.shape[:-1]), H = input.shape[-1], I = gate_up.shape[-1] // 2

    Sharding: TP on last dim of gate_up and first dim of down.
    """

    def __init__(
        self,
        name: str,
        *,
        kernel_name: Optional[KernelName] = KernelName.DENSE_MLP_TKG,
        layer_type: str = "",
        layer_count: int = 1,
        output_ports: Optional[Dict[str, Port]] = None,
        weights: Optional[Dict[str, Tuple[int, ...]]] = None,
        weight_dtype: str = "bf16",
        sharding: Optional[ShardingSpec] = None,
        roofline_mode: RooflineMode = RooflineMode.ADDITIVE,
        roofline_mfu: float = 0.60,
        roofline_mbu: float = 0.60,
        target_mfu: Optional[float] = None,
        target_mbu: Optional[float] = None,
        hardcoded_latency_us: Optional[float] = None,
        min_latency_us: float = 0.0,
    ):
        super().__init__(
            name,
            kernel_name=kernel_name,
            layer_type=layer_type,
            layer_count=layer_count,
            output_ports=output_ports,
            weights=weights,
            weight_dtype=weight_dtype,
            sharding=sharding,
            op_type=OpType.CUSTOM,
            roofline_mode=roofline_mode,
            roofline_mfu=roofline_mfu,
            roofline_mbu=roofline_mbu,
            target_mfu=target_mfu,
            target_mbu=target_mbu,
            hardcoded_latency_us=hardcoded_latency_us,
            min_latency_us=min_latency_us,
            flops=0,  # computed internally
        )

    def compute_flops(self) -> int:
        """FLOPs = 2 * 3 * S * H * I_local (gate + up + down projections).

        Infers from input shape and sharded weight shapes:
          - S = prod(input.shape[:-1])
          - H = input.shape[-1]
          - I_local = gate_up sharded last dim // 2
        """
        inp = self._input_ports.get("input")
        if inp is None:
            return 0

        S = math.prod(inp.shape[:-1])
        H = inp.shape[-1]

        gate_up_global = self.weights.get("gate_up")
        if gate_up_global is None:
            return 0

        gate_up_sharded = self._apply_weight_sharding(gate_up_global)
        I_local = gate_up_sharded[-1] // 2

        return 2 * 3 * S * H * I_local

    def __repr__(self) -> str:
        kernel = self.kernel_name.value if self.kernel_name else "None"
        return f"MLPNode('{self.name}', kernel={kernel}, layer_type='{self.layer_type}', count={self.layer_count})"
