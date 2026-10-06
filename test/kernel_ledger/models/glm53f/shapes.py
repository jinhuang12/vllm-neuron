# SPDX-License-Identifier: Apache-2.0
"""Per-kernel decode shapes of the GLM-5.3-Flash graph (``--emit-shapes``).

One entry per measured node (a node with a ``kernel_name``), in the vocabulary of the
hardware microbenchmark that times it (``benchmark``). The DSA layer is emitted once per
kernel set: its decode window differs (5938748 = ``max_model_len``, current = the
decode context bucket). The DSA members fused into the layer are also emitted in the
vocabulary of ``dsa_kernels.json`` (per-projection and attention tables), marked
``fused_in``: they are not separate measured units of the ledger.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from ...engine.decode_nodes import DSAIndexerNode, SparseMLANode
from ...engine.node import KernelNode
from ...readers.micro import MATCH_KEYS, config_name
from .arch import GLM53F, GLM53F_SHARDING, DEFAULT_CONFIG, Glm53fArch, Glm53fSharding
from .configs import KERNEL_SETS, DecodePoint
from .decode import build_glm53f_decode_graph

#: The benchmark JSON (``reports/``) that records each kernel's shapes.
BENCHMARK = {
    "mhc_sinkhorn_tkg": "mhc_micro.json",
    "mhc_combine_tkg": "mhc_micro.json",
    "kda_decode_tkg": "kda_micro.json",
    "dsa_mla_layer_tkg": "dsa_micro.json",
    "rmsnorm_router_topk_tkg": "moe-t_micro.json",
    "moe_experts_tkg": "moe-t_micro.json",
    "shared_expert_tkg": "dense_micro.json",
    "dense_mlp_tkg": "dense_micro.json",
    "lm_head": "dense_micro.json",
    "rmsnorm_tkg": "dense_micro.json",
    "sampling": "host_sampler_micro.json",
    "dsa_projection": "dsa_kernels.json",
    "dsa_attention": "dsa_kernels.json",
}

#: Graph weight key -> ``dsa_kernels.json`` projection case.
DSA_PROJECTION_CASE = {
    "q_a": "q_a_proj",
    "kv_a": "kv_a_proj_with_mqa",
    "q_b": "q_b_proj",
    "o_proj": "o_proj",
    "wq_b": "wq_b",
    "wk": "wk",
    "weights_proj": "weights_proj",
}

_DSA_WEIGHT = {"fp8": "fp8-e4m3 + 128x128 fp32 scale", "bf16": "bf16"}


def _entry(node, kernel: str, shape: Dict, kernel_set: Optional[str] = None, fused_in: Optional[str] = None):
    e = {"node": node.name, "kernel": kernel, "kernel_set": kernel_set, "layer_count": node.layer_count,
         "benchmark": BENCHMARK[kernel], "shape": shape}
    if kernel in MATCH_KEYS:
        e["config_name"] = config_name(kernel, shape)
    if fused_in:
        e["fused_in"] = fused_in
    return e


def _projection_entries(node: KernelNode, rows: int) -> List[Dict]:
    out = []
    for key, w_shape in node.weights.items():
        case = DSA_PROJECTION_CASE.get(key)
        if case is None:
            continue  # not timed by the projection table (kpool_gate)
        local = node._apply_weight_sharding(w_shape)
        k_in, n_out = (local[0], local[1]) if len(local) == 2 else (local[1], local[0] * local[2])
        shape = {"case": case, "in": k_in, "out": n_out, "rows": rows, "weight": _DSA_WEIGHT[node.weight_dtype]}
        out.append(_entry(node, "dsa_projection", shape, fused_in=node.fused_in.name))
    return out


def point_entries(point: DecodePoint, shard: Glm53fSharding = GLM53F_SHARDING,
                  arch: Glm53fArch = GLM53F) -> List[Dict]:
    g = build_glm53f_decode_graph(point, shard, arch)
    entries: List[Dict] = []
    indexer = next(n for n in g.nodes if isinstance(n, DSAIndexerNode))
    for node in g.resolve_order():
        if not isinstance(node, KernelNode):
            continue
        if node.kernel_name is not None:
            kernel = node.kernel_name.value
            if isinstance(node, SparseMLANode):
                for ks in KERNEL_SETS:
                    entries.append(_entry(node, kernel, node.shape_record(ks), kernel_set=ks))
                window = node.shape_record("current")["window_rows"]
                dense = indexer.regime == "bypass"
                shape = {"case": "dense" if dense else "selected", "ctx": point.ctx, "window_rows": window,
                         "indices": window if dense else indexer.topk, "batch": point.bs}
                entries.append(_entry(node, "dsa_attention", shape, kernel_set="current", fused_in=node.name))
            else:
                entries.append(_entry(node, kernel, node.shape_record()))
        elif node.fused_in is not None and node.weights:
            entries.extend(_projection_entries(node, point.bs))
    return entries


def emit_shapes(points: Iterable[DecodePoint], shard: Glm53fSharding = GLM53F_SHARDING,
                arch: Glm53fArch = GLM53F) -> Dict:
    """The ``ledger_shapes.json`` document for ``points``."""
    return {
        "generator": "python -m test.kernel_ledger --emit-shapes",
        "config": str(DEFAULT_CONFIG),
        "sharding": {"tp": shard.tp, "ep": shard.ep, "tp_moe": shard.tp_moe},
        "points": [
            {"point": {"bs": p.bs, "ctx": p.ctx, "max_model_len": p.max_model_len,
                       "decode_window_rows": p.decode_window_rows},
             "entries": point_entries(p, shard, arch)}
            for p in points
        ],
    }
