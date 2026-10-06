# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash architecture constants (from the checkpoint config.json) and sharding.

``load_arch()`` parses ``text_config`` of the served checkpoint's ``config.json``.
``GLM53F`` is the parsed default; ``test_glm_graph.py`` re-parses the file and checks
the two agree, so a checkpoint change shows up as a test failure, not a silent drift.

Sharding as deployed (``PLANNER_REPORT.md`` section 1, ``dense.md`` E2): TP=64 over all
64 logical cores; experts EP=16 groups of 4 ranks (each expert's intermediate split
4 ways); KDA one head per rank; MLA one head per rank with q_a / kv_a and the DSA
indexer replicated; router and mHC fn replicated; lm_head vocab-parallel in the
ledger's graph (replicated at 5938748, sharded by ``wt/dense``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

#: The served checkpoint's config (read-only).
DEFAULT_CONFIG = Path("/home/ubuntu/glm53f-campaign/lane-serve/models/GLM-5.3-Flash-04c4e9e9/config.json")


@dataclass(frozen=True)
class Glm53fArch:
    hidden_size: int
    num_layers: int
    dsa_layers: Tuple[int, ...]
    kda_layers: Tuple[int, ...]
    dense_layers: Tuple[int, ...]
    moe_layers: Tuple[int, ...]
    # mHC
    hc_mult: int
    hc_sinkhorn_iters: int
    # KDA (linear attention)
    kda_heads: int
    kda_head_dim: int
    kda_conv_taps: int
    # MLA + DSA indexer
    num_attention_heads: int
    q_lora_rank: int
    qk_head_dim: int
    v_head_dim: int
    kv_lora_rank: int
    index_n_heads: int
    index_head_dim: int
    index_kpool: int
    index_topk: int
    # MLP / MoE
    intermediate_size: int
    moe_intermediate_size: int
    n_routed_experts: int
    num_experts_per_tok: int
    n_shared_experts: int
    # tail
    vocab_size: int

    @property
    def mhc_fn_out(self) -> int:
        """fn GEMV width: n*n residual mix + n pre gates + n post gates (24 at n=4)."""
        n = self.hc_mult
        return n * n + 2 * n


def load_arch(path: Path = DEFAULT_CONFIG) -> Glm53fArch:
    """Parse ``text_config`` of a GLM-5.3-Flash ``config.json``."""
    cfg = json.loads(Path(path).read_text())["text_config"]
    lin = cfg["linear_attn_config"]
    n = cfg["num_hidden_layers"]
    dsa = tuple(i for i, t in enumerate(cfg["layer_types"]) if t == "deepseek_sparse_attention")
    kda = tuple(i for i, t in enumerate(cfg["layer_types"]) if t == "linear_attention")
    if tuple(lin["full_attn_layers"]) != dsa or tuple(lin["kda_layers"]) != kda:
        raise ValueError("config.json: layer_types disagree with linear_attn_config layer lists")
    dense = tuple(i for i, t in enumerate(cfg["mlp_layer_types"]) if t == "dense")
    moe = tuple(i for i, t in enumerate(cfg["mlp_layer_types"]) if t == "sparse")
    if dense != tuple(range(cfg["first_k_dense_replace"])):
        raise ValueError("config.json: mlp_layer_types disagree with first_k_dense_replace")
    return Glm53fArch(
        hidden_size=cfg["hidden_size"],
        num_layers=n,
        dsa_layers=dsa,
        kda_layers=kda,
        dense_layers=dense,
        moe_layers=moe,
        hc_mult=cfg["hc_mult"],
        hc_sinkhorn_iters=cfg["hc_sinkhorn_iters"],
        kda_heads=lin["num_heads"],
        kda_head_dim=lin["head_dim"],
        kda_conv_taps=lin["short_conv_kernel_size"],
        num_attention_heads=cfg["num_attention_heads"],
        q_lora_rank=cfg["q_lora_rank"],
        qk_head_dim=cfg["qk_head_dim"],
        v_head_dim=cfg["v_head_dim"],
        kv_lora_rank=cfg["kv_lora_rank"],
        index_n_heads=cfg["index_n_heads"],
        index_head_dim=cfg["index_head_dim"],
        index_kpool=cfg["index_kpool"],
        index_topk=cfg["index_topk"],
        intermediate_size=cfg["intermediate_size"],
        moe_intermediate_size=cfg["moe_intermediate_size"],
        n_routed_experts=cfg["n_routed_experts"],
        num_experts_per_tok=cfg["num_experts_per_tok"],
        n_shared_experts=cfg["n_shared_experts"],
        vocab_size=cfg["vocab_size"],
    )


@dataclass(frozen=True)
class Glm53fSharding:
    """TP over all ranks; experts in ``ep`` groups of ``tp_moe`` ranks."""

    tp: int = 64
    ep: int = 16
    tp_moe: int = 4

    def __post_init__(self):
        if self.ep * self.tp_moe != self.tp:
            raise ValueError(f"ep x tp_moe ({self.ep} x {self.tp_moe}) != tp ({self.tp})")


GLM53F = load_arch()
GLM53F_SHARDING = Glm53fSharding()
