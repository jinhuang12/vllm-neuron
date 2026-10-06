# SPDX-License-Identifier: Apache-2.0
"""New GLM decode node types: output shapes, FLOPs, bytes, regimes, shape records."""

from __future__ import annotations

import math

import pytest

from test.kernel_ledger.engine.decode_nodes import (
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
    expected_distinct_experts,
)
from test.kernel_ledger.engine.node import HBM_BW, Port

H = 4096


def _wire(node, **ports):
    for name, port in ports.items():
        node.set_input_port(name, port)
    return node


def test_expected_distinct_experts_formula():
    assert expected_distinct_experts(288, 8, 1) == pytest.approx(8.0)
    assert expected_distinct_experts(288, 8, 64) == pytest.approx(288 * (1 - (1 - 8 / 288) ** 64))
    assert expected_distinct_experts(288, 8, 0) == 0.0


def test_mhc_sinkhorn_outputs_and_record():
    n = MHCSinkhornNode("s", streams=4, iters=20, hidden=H)
    _wire(n, mix=Port("mix", (1, 1, 24), "fp32"))
    assert n.output_ports["res"].shape == (1, 1, 4, 4)
    assert n.output_ports["pre"].shape == (1, 1, 4)
    assert n.output_ports["post"].shape == (1, 1, 4)
    # exp once, then 20 x (row + column normalisation, 2 n^2 each)
    assert n.compute_flops() == 1 * (16 + 20 * 4 * 16)
    assert n.shape_record() == {"B": 1, "hidden": 4096, "streams": 4, "iters": 20}


def test_mhc_sinkhorn_refuses_a_mix_of_the_wrong_width():
    n = MHCSinkhornNode("s", streams=4, iters=20, hidden=H)
    _wire(n, mix=Port("mix", (1, 1, 16), "fp32"))
    with pytest.raises(ValueError, match="mix"):
        n.output_ports


def test_mhc_combine_flops_bytes_and_record():
    n = MHCCombineNode("c", streams=4)
    _wire(
        n,
        streams=Port("streams", (4, 1, 4, H), "bf16"),
        y=Port("y", (4, 1, H), "bf16"),
        res=Port("res", (4, 1, 4, 4), "fp32"),
        post=Port("post", (4, 1, 4), "fp32"),
    )
    assert n.output_ports["streams"].shape == (4, 1, 4, H)
    assert n.compute_flops() == 4 * (2 * 16 * H + 2 * 4 * H)
    # read streams + y + mix weights, write streams
    expected = 4 * 4 * H * 2 + 4 * H * 2 + 4 * 16 * 4 + 4 * 4 * 4 + 4 * 4 * H * 2
    assert n._total_memory_bytes() == expected
    assert n.shape_record() == {"B": 4, "hidden": H, "streams": 4}


def _kda(b):
    n = KDADecodeStepNode("kda", heads_per_rank=1, head_dim=128, conv_taps=4)
    head = Port("q", (b, 1, 1, 128), "bf16")
    return _wire(n, q=head, k=head, v=head, g=head, beta=Port("beta", (b, 1, 1, 1), "bf16"))


def test_kda_step_record_matches_the_carrier_layout():
    n = _kda(4)
    assert n.output_ports["core"].shape == (4, 1, 1, 128)
    assert n.shape_record() == {
        "B": 4,
        "heads_per_rank": 1,
        "head_dim": 128,
        "conv_taps": 4,
        "conv_channels": 384,
        "conv_carrier": {"dtype": "bfloat16", "shape": [4, 3, 384]},
        "recurrent_carrier": {"dtype": "float32", "shape": [4, 1, 128, 128]},
    }


def test_kda_step_bytes_are_carrier_round_trips_plus_io():
    n = _kda(1)
    carriers = 2 * (3 * 384 * 2 + 128 * 128 * 4)
    io = 4 * 128 * 2 + 1 * 2 + 128 * 2
    assert n._total_memory_bytes() == carriers + io
    assert n.compute_flops() == 2 * 4 * 384 + 7 * 128 * 128 + 8 * 128


def _indexer(ctx, b=1):
    n = DSAIndexerNode("idx", n_heads=32, head_dim=128, kpool=4, topk=2048, ctx=ctx)
    return _wire(
        n,
        q_idx=Port("q", (b, 1, 32, 128), "bf16"),
        k_idx=Port("k", (b, 1, 128), "bf16"),
        w_idx=Port("w", (b, 1, 32), "bf16"),
    )


def test_indexer_bypass_regime_up_to_ctx_2051():
    for ctx in (1024, 2051):
        n = _indexer(ctx)
        assert n.regime == "bypass"
        assert n.selected_tokens == ctx
        assert n.compute_flops() == 0
        assert n._total_memory_bytes() == 0
    assert _indexer(2052).regime == "selected"


def test_indexer_selected_regime_scores_every_pool():
    n = _indexer(8192, b=64)
    pools = 8192 // 4
    assert n.selected_tokens == 2048
    assert n.output_ports["indices"].shape == (64, 1, 2048)
    assert n.compute_flops() == 64 * (2 * 32 * 128 * pools + 2 * 32 * pools)
    assert n._total_memory_bytes() == 64 * (pools * 128 * 2 + 32 * 128 * 2 + 32 * 2)


def test_sparse_mla_reads_the_selected_latent_rows():
    n = SparseMLANode("mla", heads_per_rank=1, qk_head_dim=256, v_head_dim=256, kv_lora_rank=512, tp=64)
    _wire(
        n,
        q=Port("q", (2, 1, 1, 256), "bf16"),
        latent=Port("latent", (2, 1, 512), "bf16"),
        indices=Port("indices", (2, 1, 1024), "int32"),
    )
    assert n.output_ports["out"].shape == (2, 1, 1, 256)
    t = 1024
    assert n.compute_flops() == 2 * (2 * 256 * 512 + 2 * 512 * t + 2 * 512 * t + 2 * 512 * 256)
    weights = (256 * 512 + 512 * 256) * 2
    assert n._total_memory_bytes() == weights + 2 * t * 512 * 2 + 2 * 256 * 2 + 2 * 512 * 2 + 2 * 256 * 2


def test_fused_router_record_and_outputs():
    n = FusedRouterNode("r", num_experts=288, top_k=8)
    _wire(n, x=Port("x", (1, 1, H), "bf16"))
    assert n.output_ports["topk_ids"].shape == (1, 1, 8)
    assert n.output_ports["topk_weights"].shape == (1, 1, 8)
    assert n.shape_record() == {"T": 1, "H": H, "E_global": 288, "top_k": 8}
    assert n.compute_flops() == 4 * H + 2 * H * 288 + 10 * 288


def test_expert_decode_bytes_follow_expected_distinct_local_experts():
    n = ExpertDecodeNode("e", num_experts=288, top_k=8, intermediate_size=2048, ep=16, tp_moe=4)
    _wire(n, x=Port("x", (64, 1, H), "bf16"), topk_ids=Port("ids", (64, 1, 8), "int32"))
    d_local = 18 * (1 - (1 - 8 / 288) ** 64)
    assert n.expected_distinct_local_experts == pytest.approx(d_local)
    assert d_local == pytest.approx(15.03, abs=0.01)
    per_expert = 3 * H * 512
    assert n._weight_bytes() == pytest.approx(d_local * per_expert)
    assert n.compute_flops() == pytest.approx(2 * 3 * H * 512 * (64 * 8 * 18 / 288))
    assert n.shape_record() == {
        "T": 64, "H": H, "E_global": 288, "top_k": 8, "E_local": 18, "I_local": 512,
        "expected_distinct_local_experts": pytest.approx(d_local),
    }


def test_expert_decode_at_one_token_hits_half_an_expert_per_rank():
    n = ExpertDecodeNode("e", num_experts=288, top_k=8, intermediate_size=2048, ep=16, tp_moe=4)
    _wire(n, x=Port("x", (1, 1, H), "bf16"), topk_ids=Port("ids", (1, 1, 8), "int32"))
    assert n.expected_distinct_local_experts == pytest.approx(0.5)


def test_block_quant_mlp_pads_the_local_intermediate_to_128():
    shared = BlockQuantMLPNode("shared", hidden=H, intermediate_size=2048, tp=64, block=128)
    dense = BlockQuantMLPNode("dense", hidden=H, intermediate_size=12288, tp=64, block=128)
    for n in (shared, dense):
        n.set_input_port("input", Port("x", (1, 1, H), "bf16"))
    assert shared.local_intermediate == 128  # 2048 / 64 = 32, padded
    assert dense.local_intermediate == 256  # 12288 / 64 = 192, padded
    assert dense.local_weight_shapes() == {"gate_up": (H, 512), "down": (256, H)}
    assert dense._total_memory_bytes() == 3 * H * 256
    assert dense.compute_flops() == 2 * 3 * 1 * H * 256
    assert dense.output_ports["output"].shape == (1, 1, H)
    assert shared.shape_record() == {"M": 1, "H": H, "I": 128}


def test_vocab_parallel_lm_head_shards_rows():
    n = VocabParallelLMHeadNode("lm", hidden=H, vocab=154880, tp=64)
    n.set_input_port("input", Port("x", (1, 1, H), "bf16"))
    assert n.output_ports["logits"].shape == (1, 1, 2420)
    assert n._total_memory_bytes() == 2420 * H * 2
    assert n.compute_roofline() == pytest.approx(2420 * H * 2 / HBM_BW * 1e6)
    assert n.shape_record() == {"M": 1, "H": H, "vocab": 154880, "shard_rows": 2420}


def test_vocab_parallel_lm_head_refuses_a_remainder_vocabulary():
    with pytest.raises(ValueError, match="divisible"):
        VocabParallelLMHeadNode("lm", hidden=H, vocab=154881, tp=64)


def test_greedy_sampler_reads_the_logits_row():
    n = GreedySamplerNode("s")
    n.set_input_port("logits", Port("logits", (1, 1, 154880), "bf16"))
    assert n.output_ports["tokens"].shape == (1, 1)
    assert n._total_memory_bytes() == 154880 * 2 + 4
    assert n.shape_record() == {"B": 1, "vocab": 154880, "logits_dtype": "bfloat16", "mode": "all_greedy"}


def test_rmsnorm_record_and_bytes():
    n = RMSNormNode("norm")
    n.set_input_port("input", Port("x", (1, 1, H), "bf16"))
    assert n.output_ports["output"].shape == (1, 1, H)
    assert n._total_memory_bytes() == 3 * H * 2
    assert n.shape_record() == {"M": 1, "H": H}
    assert math.isclose(n.compute_flops(), 4 * H)
