# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash architecture parse and the TP=64/EP=16 decode graph."""

from __future__ import annotations

import pytest

from test.kernel_ledger.engine.decode_nodes import (
    DSAIndexerNode,
    ExpertDecodeNode,
    KDADecodeStepNode,
    SparseMLANode,
)
from test.kernel_ledger.engine.node import CollectiveNode, CollectiveType, KernelName, KernelNode
from test.kernel_ledger.models.glm53f.arch import GLM53F, GLM53F_SHARDING, load_arch
from test.kernel_ledger.models.glm53f.configs import BS1_CTX1K, BS64_CTX8K, DecodePoint
from test.kernel_ledger.models.glm53f.decode import build_glm53f_decode_graph


def _sites(graph, kernel_name):
    return sum(n.layer_count for n in graph.nodes if isinstance(n, KernelNode) and n.kernel_name == kernel_name)


def test_arch_parses_the_checkpoint_config():
    arch = load_arch()
    assert arch == GLM53F
    assert arch.num_layers == 45
    assert arch.kda_layers == tuple(i for i in range(45) if i not in arch.dsa_layers)
    assert arch.dsa_layers == tuple(range(3, 45, 4))
    assert len(arch.kda_layers) == 34 and len(arch.dsa_layers) == 11
    assert arch.dense_layers == (0, 1, 2) and len(arch.moe_layers) == 42
    assert (arch.hc_mult, arch.hc_sinkhorn_iters) == (4, 20)
    assert (arch.kda_heads, arch.kda_head_dim, arch.kda_conv_taps) == (64, 128, 4)
    assert (arch.kv_lora_rank, arch.index_n_heads, arch.index_kpool, arch.index_topk) == (512, 32, 4, 2048)
    assert (arch.n_routed_experts, arch.num_experts_per_tok, arch.moe_intermediate_size) == (288, 8, 2048)
    assert (arch.intermediate_size, arch.vocab_size) == (12288, 154880)


def test_sharding_is_tp64_ep16_in_groups_of_4():
    s = GLM53F_SHARDING
    assert (s.tp, s.ep, s.tp_moe) == (64, 16, 4)
    assert s.ep * s.tp_moe == s.tp


@pytest.fixture(scope="module")
def g1():
    return build_glm53f_decode_graph(BS1_CTX1K)


@pytest.fixture(scope="module")
def g64():
    return build_glm53f_decode_graph(BS64_CTX8K)


def test_graph_validates(g1, g64):
    assert g1.validate() == []
    assert g64.validate() == []


def test_site_counts_per_step(g1):
    assert _sites(g1, KernelName.MHC_SINKHORN_TKG) == 90
    assert _sites(g1, KernelName.MHC_COMBINE_TKG) == 90
    assert _sites(g1, KernelName.KDA_DECODE_TKG) == 34
    assert _sites(g1, KernelName.DSA_MLA_LAYER_TKG) == 11
    assert _sites(g1, KernelName.RMSNORM_TKG) == 91
    assert _sites(g1, KernelName.DENSE_MLP_TKG) == 3
    assert _sites(g1, KernelName.SHARED_EXPERT_TKG) == 42
    assert _sites(g1, KernelName.RMSNORM_ROUTER_TOPK_TKG) == 42
    assert _sites(g1, KernelName.MOE_EXPERTS_TKG) == 42
    assert _sites(g1, KernelName.LM_HEAD) == 1
    assert _sites(g1, KernelName.SAMPLING) == 1


def test_ninety_all_reduces_plus_the_logit_gather(g1):
    colls = [n for n in g1.nodes if isinstance(n, CollectiveNode)]
    ar = sum(n.layer_count for n in colls if n.collective_type == CollectiveType.ALL_REDUCE)
    ag = [n for n in colls if n.collective_type == CollectiveType.ALL_GATHER]
    assert ar == 90
    assert len(ag) == 1 and ag[0].output_ports["output"].shape == (1, 1, 154880)
    for n in colls:
        if n.collective_type == CollectiveType.ALL_REDUCE:
            assert n.message_size_bytes == 16384  # fp32 [1, 4096]


def test_per_rank_shapes_propagate(g1):
    by = {n.name: n for n in g1.nodes}
    assert by["kda_in_proj"].output_ports["q"].shape == (1, 1, 1, 128)
    assert by["kda_gate_a_proj"].output_ports["f_a"].shape == (1, 1, 128)
    assert by["kda_step"].input_ports["g"].shape == (1, 1, 1, 128)
    assert by["mla_q_b"].output_ports["q_b"].shape == (1, 1, 1, 256)
    assert by["idx_q_proj"].output_ports["wq_b"].shape == (1, 1, 32, 128)
    assert by["mla_x_proj"].output_ports["kv_a"].shape == (1, 1, 512)
    assert by["lm_head"].output_ports["logits"].shape == (1, 1, 2420)
    assert by["sampler"].output_ports["tokens"].shape == (1, 1)
    assert by["mhc_post_mlp"].output_ports["streams"].shape == (1, 1, 4, 4096)


def test_dsa_regime_and_experts_follow_the_operating_point(g1, g64):
    idx1 = next(n for n in g1.nodes if isinstance(n, DSAIndexerNode))
    idx64 = next(n for n in g64.nodes if isinstance(n, DSAIndexerNode))
    assert (idx1.regime, idx1.selected_tokens) == ("bypass", 1024)
    assert (idx64.regime, idx64.selected_tokens) == ("selected", 2048)
    mla64 = next(n for n in g64.nodes if isinstance(n, SparseMLANode))
    assert mla64.input_ports["indices"].shape == (64, 1, 2048)
    e1 = next(n for n in g1.nodes if isinstance(n, ExpertDecodeNode))
    e64 = next(n for n in g64.nodes if isinstance(n, ExpertDecodeNode))
    assert e1.expected_distinct_local_experts == pytest.approx(0.5)
    assert e64.expected_distinct_local_experts == pytest.approx(15.03, abs=0.01)


def test_dsa_members_are_fused_into_the_measured_layer(g1):
    rep = next(n for n in g1.nodes if isinstance(n, SparseMLANode))
    members = {n.name for n in g1.nodes if getattr(n, "fused_in", None) is rep}
    assert {"mla_x_proj", "mla_q_b", "idx_q_proj", "idx_x_proj", "dsa_indexer", "mla_o_proj"} <= members


def test_kda_state_carriers_scale_with_batch(g64):
    kda = next(n for n in g64.nodes if isinstance(n, KDADecodeStepNode))
    rec = kda.shape_record()
    assert rec["conv_carrier"]["shape"] == [64, 3, 384]
    assert rec["recurrent_carrier"]["shape"] == [64, 1, 128, 128]


def test_decode_window_rows():
    assert DecodePoint(bs=1, ctx=1024, max_model_len=4096).decode_window_rows == 2048
    assert DecodePoint(bs=1, ctx=4096, max_model_len=4096).decode_window_rows == 4096
    assert DecodePoint(bs=64, ctx=8192, max_model_len=8192).decode_window_rows == 8192
