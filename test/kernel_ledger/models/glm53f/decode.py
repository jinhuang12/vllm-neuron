# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash decode step as a node graph, TP=64 / EP=16, per-rank shapes.

One decode step for ``point.bs`` requests (one token each). The graph is a layer
template: each node carries ``layer_count`` = how many times a step runs it
(the engine's convention), so 45 layers do not need 45 copies.

```
embed -> mhc_expand -> [ mHC pre (fn GEMV, sinkhorn, mix) -> attn_norm
          -> KDA x34 (proj, kda_step, gate, o_proj) | DSA x11 (projections, indexer, sparse MLA, o_proj)
          -> all-reduce x45 -> mHC post (combine)
          -> mHC pre -> ffn_norm -> dense MLP x3 | MoE x42 (router, experts, shared expert, sum)
          -> all-reduce x45 -> mHC post ] -> mhc_collapse -> final_norm -> lm_head -> all-gather -> sampler
```

Node kinds:
  - nodes with a ``kernel_name``: one measured unit each (a hardware microbenchmark exists);
  - DSA members are ``fused_in`` the measured ``mla_sparse`` node: the DSA layer
    microbenchmark times the whole ``Glm5NextMLAAttention.forward``;
  - KernelNodes without a ``kernel_name``: compiler-lowered ops (fn GEMV, KDA
    projections, gates, casts). No microbenchmark: roofline only, their time is in the
    residual ("compiler glue");
  - plain Nodes (reshape, merge, cast): zero cost.

``layer_type`` is the ledger bucket of each node.
"""

from __future__ import annotations

from typing import Optional

from ...engine.decode_nodes import (
    BlockQuantMLPNode,
    DSAIndexerNode,
    ExpertDecodeNode,
    FusedRouterNode,
    GreedySamplerNode,
    KDADecodeStepNode,
    MHCCombineNode,
    MHCSinkhornNode,
    RMSNormNode,
    SparseMLANode,
    VocabParallelLMHeadNode,
)
from ...engine.graph import ForwardPassGraph
from ...engine.node import (
    CollectiveDim,
    CollectiveNode,
    CollectiveType,
    KernelName,
    KernelNode,
    Node,
    OpType,
    Port,
    RooflineMode,
    ShardingSpec,
    WeightSharding,
)
from .arch import GLM53F, GLM53F_SHARDING, Glm53fArch, Glm53fSharding
from .configs import DecodePoint

#: Ledger buckets (``layer_type`` of every node), split as DECODE_BREAKDOWN.md's master
#: table splits the step: "dense" holds the blockwise MLPs (shared expert x42, dense MLP
#: x3) and the 1-row norms (its ``dense.md`` rows #8, #9); "MoE" holds routing and the
#: routed experts (its ``moe_host.md`` rows).
MHC, KDA, DSA, DENSE, MOE, LM_HEAD, SAMPLER, COLL, TAIL = (
    "mHC", "KDA", "DSA/MLA", "dense", "MoE", "lm_head", "sampler", "collectives", "embed/tail",
)
NORM = DENSE
BUCKETS = (MHC, KDA, DSA, MOE, DENSE, LM_HEAD, SAMPLER, COLL, TAIL)


def _gemv(name, weights, *, dtype, bucket, count, tp_head: Optional[int] = None) -> KernelNode:
    """A matmul node with GLOBAL weights; ``tp_head`` shards dim 0 (heads)."""
    sharding = ShardingSpec(weight=[(WeightSharding.TP_HEAD, tp_head)]) if tp_head else ShardingSpec()
    return KernelNode(
        name, weights=weights, weight_dtype=dtype, sharding=sharding, layer_type=bucket,
        layer_count=count, op_type=OpType.MATMUL, roofline_mode=RooflineMode.MAX, min_latency_us=2.0,
    )


def _compiler_op(name, outputs, *, flops, bucket, count) -> KernelNode:
    """A compiler-lowered elementwise op: explicit outputs, activations in and out of HBM."""
    return KernelNode(
        name, output_ports=outputs, op_type=OpType.CUSTOM, flops=flops, layer_type=bucket,
        layer_count=count, roofline_mode=RooflineMode.MAX, include_activation_mem=True, min_latency_us=2.0,
    )


def _cast(g, src, port, dtype, name) -> Node:
    """Zero-cost dtype change (the fp32 accumulate ahead of each all-reduce)."""
    p = src.get_output_port(port)
    node = Node(name, layer_type="cast")
    node._output_ports_explicit = {"output": Port("output", p.shape, dtype)}
    g.add_node(node)
    g.connect(src, port, node, "input")
    return node


def _all_reduce(name, tp, count) -> CollectiveNode:
    return CollectiveNode(name, collective_type=CollectiveType.ALL_REDUCE, group_size=tp,
                          dim=CollectiveDim.HIDDEN, dim_idx=2, layer_type=COLL, layer_count=count)


def _mhc_pre(g, streams_src, streams_port, side, arch, B, L):
    """fn GEMV (compiler) -> sinkhorn (measured) -> pre-mix (compiler). Returns (sinkhorn, mix)."""
    n, H = arch.hc_mult, arch.hidden_size
    flat = g.reshape(streams_src, streams_port, (B, 1, n * H))
    fn = g.add_node(_gemv(f"mhc_pre_{side}_fn", {"mix": (n * H, arch.mhc_fn_out)}, dtype="bf16",
                          bucket=MHC, count=L))
    g.connect(flat, "output", fn, "input")
    sk = g.add_node(MHCSinkhornNode(f"mhc_pre_{side}_sinkhorn", streams=n, iters=arch.hc_sinkhorn_iters,
                                    hidden=H, layer_type=MHC, layer_count=L))
    g.connect(fn, "mix", sk, "mix")
    mix = g.add_node(_compiler_op(f"mhc_pre_{side}_mix", {"x": Port("x", (B, 1, H), "bf16")},
                                  flops=2 * B * n * H, bucket=MHC, count=L))
    g.connect(streams_src, streams_port, mix, "streams")
    g.connect(sk, "pre", mix, "pre")
    return sk, mix


def build_glm53f_decode_graph(point: DecodePoint, shard: Glm53fSharding = GLM53F_SHARDING,
                              arch: Glm53fArch = GLM53F) -> ForwardPassGraph:
    B, H, n, tp = point.bs, arch.hidden_size, arch.hc_mult, shard.tp
    L = arch.num_layers
    nK, nD = len(arch.kda_layers), len(arch.dsa_layers)
    nDense, nMoE = len(arch.dense_layers), len(arch.moe_layers)
    g = ForwardPassGraph("glm53f", "decode")

    # -- embedding and stream expansion (compiler) --------------------------------------
    embed = g.add_node(KernelNode(
        "embed", input_ports={"ids": Port("ids", (B, 1, 1), "int32")},
        output_ports={"x": Port("x", (B, 1, H), "bf16")}, op_type=OpType.CUSTOM, flops=0,
        layer_type=TAIL, roofline_mode=RooflineMode.MAX, include_activation_mem=True, min_latency_us=2.0,
    ))
    expand = g.add_node(_compiler_op("mhc_expand", {"streams": Port("streams", (B, 1, n, H), "bf16")},
                                     flops=0, bucket=MHC, count=1))
    g.connect(embed, "x", expand, "input")

    # -- attention half ----------------------------------------------------------------
    sk_a, mix_a = _mhc_pre(g, expand, "streams", "attn", arch, B, L)
    attn_norm = g.add_node(RMSNormNode("attn_norm", layer_type=NORM, layer_count=L))
    g.connect(mix_a, "x", attn_norm, "input")

    # KDA x34: one head of 64 per rank
    Hk, d = arch.kda_heads, arch.kda_head_dim
    kda_in = g.add_node(_gemv("kda_in_proj", {"q": (Hk, H, d), "k": (Hk, H, d), "v": (Hk, H, d),
                                              "b": (Hk, H, 1)}, dtype="bf16", bucket=KDA, count=nK, tp_head=tp))
    g.connect(attn_norm, "output", kda_in, "input")
    kda_ga = g.add_node(_gemv("kda_gate_a_proj", {"f_a": (H, d), "g_a": (H, d)}, dtype="bf16",
                              bucket=KDA, count=nK))
    g.connect(attn_norm, "output", kda_ga, "input")
    kda_gb = g.add_node(_gemv("kda_gate_b_proj", {"f_b": (Hk, d, d), "g_b": (Hk, d, d)}, dtype="bf16",
                              bucket=KDA, count=nK, tp_head=tp))
    g.connect(kda_ga, "f_a", kda_gb, "f_a")
    g.connect(kda_ga, "g_a", kda_gb, "g_a")
    hpr = Hk // tp
    kda = g.add_node(KDADecodeStepNode("kda_step", heads_per_rank=hpr, head_dim=d,
                                       conv_taps=arch.kda_conv_taps, layer_type=KDA, layer_count=nK))
    for p in ("q", "k", "v"):
        g.connect(kda_in, p, kda, p)
    g.connect(kda_gb, "f_b", kda, "g")
    g.connect(kda_in, "b", kda, "beta")
    kda_gate = g.add_node(_compiler_op("kda_out_gate", {"o": Port("o", (B, 1, hpr, d), "bf16")},
                                       flops=6 * B * hpr * d, bucket=KDA, count=nK))
    g.connect(kda, "core", kda_gate, "core")
    g.connect(kda_gb, "g_b", kda_gate, "g_b")
    kda_flat = g.reshape(kda_gate, "o", (B, 1, hpr * d))
    kda_o = g.add_node(_gemv("kda_o_proj", {"o": (Hk * d, H)}, dtype="bf16", bucket=KDA, count=nK, tp_head=tp))
    g.connect(kda_flat, "output", kda_o, "input")

    # DSA x11: MLA one head of 64 per rank; q_a / kv_a and the indexer replicated
    Hm = arch.num_attention_heads
    mla_x = g.add_node(_gemv("mla_x_proj", {"q_a": (H, arch.q_lora_rank), "kv_a": (H, arch.kv_lora_rank)},
                             dtype="fp8", bucket=DSA, count=nD))
    g.connect(attn_norm, "output", mla_x, "input")
    mla_qb = g.add_node(_gemv("mla_q_b", {"q_b": (Hm, arch.q_lora_rank, arch.qk_head_dim)}, dtype="fp8",
                              bucket=DSA, count=nD, tp_head=tp))
    g.connect(mla_x, "q_a", mla_qb, "input")
    idx_q = g.add_node(_gemv("idx_q_proj", {"wq_b": (arch.index_n_heads, arch.q_lora_rank, arch.index_head_dim)},
                             dtype="bf16", bucket=DSA, count=nD))
    g.connect(mla_x, "q_a", idx_q, "input")
    idx_x = g.add_node(_gemv("idx_x_proj", {"wk": (H, arch.index_head_dim), "weights_proj": (H, arch.index_n_heads),
                                            "kpool_gate": (H, arch.index_head_dim)},
                             dtype="bf16", bucket=DSA, count=nD))
    g.connect(attn_norm, "output", idx_x, "input")
    indexer = g.add_node(DSAIndexerNode("dsa_indexer", n_heads=arch.index_n_heads, head_dim=arch.index_head_dim,
                                        kpool=arch.index_kpool, topk=arch.index_topk, ctx=point.ctx,
                                        layer_type=DSA, layer_count=nD))
    g.connect(idx_q, "wq_b", indexer, "q_idx")
    g.connect(idx_x, "wk", indexer, "k_idx")
    g.connect(idx_x, "weights_proj", indexer, "w_idx")
    mla = g.add_node(SparseMLANode("mla_sparse", heads_per_rank=Hm // tp, qk_head_dim=arch.qk_head_dim,
                                   v_head_dim=arch.v_head_dim, kv_lora_rank=arch.kv_lora_rank, tp=tp,
                                   layer_type=DSA, layer_count=nD))
    g.connect(mla_qb, "q_b", mla, "q")
    g.connect(mla_x, "kv_a", mla, "latent")
    g.connect(indexer, "indices", mla, "indices")
    # the DSA layer benchmark's shape fields (dsa_micro.json layer cases)
    mla.layer_record = {
        "common": {"batch": B, "ctx": point.ctx, "max_seq_len": point.max_model_len},
        # 5938748 serves no decode context bucket: the window is max_model_len
        "5938748": {"window_rows": point.max_model_len},
        "current": {"window_rows": point.decode_window_rows},
    }
    mla_flat = g.reshape(mla, "out", ((B, 1, (Hm // tp) * arch.v_head_dim)))
    mla_o = g.add_node(_gemv("mla_o_proj", {"o_proj": (Hm * arch.v_head_dim, H)}, dtype="fp8",
                             bucket=DSA, count=nD, tp_head=tp))
    g.connect(mla_flat, "output", mla_o, "input")
    for member in (mla_x, mla_qb, idx_q, idx_x, indexer, mla_o):
        member.fused_in = mla

    attn_out = g.merge([(kda_o, "o"), (mla_o, "o_proj")], name="attn_out")
    ar_attn = g.add_node(_all_reduce("ar_attn", tp, L))
    g.connect(_cast(g, attn_out, "output", "fp32", "attn_out_fp32"), "output", ar_attn, "input")
    post_a = g.add_node(MHCCombineNode("mhc_post_attn", streams=n, layer_type=MHC, layer_count=L))
    g.connect(expand, "streams", post_a, "streams")
    g.connect(ar_attn, "output", post_a, "y")
    g.connect(sk_a, "res", post_a, "res")
    g.connect(sk_a, "post", post_a, "post")

    # -- MLP half ----------------------------------------------------------------------
    sk_m, mix_m = _mhc_pre(g, post_a, "streams", "mlp", arch, B, L)
    ffn_norm = g.add_node(RMSNormNode("ffn_norm", layer_type=NORM, layer_count=L))
    g.connect(mix_m, "x", ffn_norm, "input")

    dense = g.add_node(BlockQuantMLPNode("dense_mlp", hidden=H, intermediate_size=arch.intermediate_size, tp=tp,
                                         kernel_name=KernelName.DENSE_MLP_TKG, layer_type=DENSE,
                                         layer_count=nDense))
    g.connect(ffn_norm, "output", dense, "input")

    router = g.add_node(FusedRouterNode("moe_router", num_experts=arch.n_routed_experts,
                                        top_k=arch.num_experts_per_tok, layer_type=MOE, layer_count=nMoE))
    g.connect(mix_m, "x", router, "x")  # the router kernel normalises the pre-norm row itself
    experts = g.add_node(ExpertDecodeNode("moe_experts", num_experts=arch.n_routed_experts,
                                          top_k=arch.num_experts_per_tok,
                                          intermediate_size=arch.moe_intermediate_size, ep=shard.ep,
                                          tp_moe=shard.tp_moe, layer_type=MOE, layer_count=nMoE))
    g.connect(ffn_norm, "output", experts, "x")
    g.connect(router, "topk_ids", experts, "topk_ids")
    shared = g.add_node(BlockQuantMLPNode("shared_expert", hidden=H,
                                          intermediate_size=arch.moe_intermediate_size * arch.n_shared_experts,
                                          tp=tp, kernel_name=KernelName.SHARED_EXPERT_TKG, layer_type=DENSE,
                                          layer_count=nMoE))
    g.connect(ffn_norm, "output", shared, "input")
    moe_sum = g.add_node(_compiler_op("moe_sum", {"output": Port("output", (B, 1, H), "bf16")},
                                      flops=B * H, bucket=MOE, count=nMoE))
    g.connect(experts, "y", moe_sum, "routed")
    g.connect(shared, "output", moe_sum, "shared")

    mlp_out = g.merge([(dense, "output"), (moe_sum, "output")], name="mlp_out")
    ar_mlp = g.add_node(_all_reduce("ar_mlp", tp, L))
    g.connect(_cast(g, mlp_out, "output", "fp32", "mlp_out_fp32"), "output", ar_mlp, "input")
    post_m = g.add_node(MHCCombineNode("mhc_post_mlp", streams=n, layer_type=MHC, layer_count=L))
    g.connect(post_a, "streams", post_m, "streams")
    g.connect(ar_mlp, "output", post_m, "y")
    g.connect(sk_m, "res", post_m, "res")
    g.connect(sk_m, "post", post_m, "post")

    # -- tail --------------------------------------------------------------------------
    collapse = g.add_node(_compiler_op("mhc_collapse", {"x": Port("x", (B, 1, H), "bf16")},
                                       flops=2 * B * n * H, bucket=MHC, count=1))
    g.connect(post_m, "streams", collapse, "streams")
    final_norm = g.add_node(RMSNormNode("final_norm", layer_type=NORM, layer_count=1))
    g.connect(collapse, "x", final_norm, "input")
    lm = g.add_node(VocabParallelLMHeadNode("lm_head", hidden=H, vocab=arch.vocab_size, tp=tp,
                                            layer_type=LM_HEAD))
    g.connect(final_norm, "output", lm, "input")
    gather = g.add_node(CollectiveNode("lm_head_gather", collective_type=CollectiveType.ALL_GATHER, group_size=tp,
                                       dim=CollectiveDim.HIDDEN, dim_idx=2, layer_type=COLL))
    # 5938748 runs a replicated lm_head and has no logit gather: wt/dense adds both
    gather.requires_after = "dense"
    g.connect(lm, "logits", gather, "input")
    sampler = g.add_node(GreedySamplerNode("sampler", layer_type=SAMPLER))
    g.connect(gather, "output", sampler, "logits")
    return g
