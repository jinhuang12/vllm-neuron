# SPDX-License-Identifier: Apache-2.0
"""Which tensor-parallel degree GLM-5.3-Flash (+MTP) should run at on trn2: a byte-level entitlement.

Inputs: the checkpoint's bytes by weight class (``checkpoint_bytes_by_class.json``, summed from the
safetensors headers: ``kda:routed_experts``, ``dsa:attention``, ...), the HF config, and trn2
constants per LOGICAL core (LNC2): 24 GiB HBM, 716 GB/s. For TP in {8, 16, 32, 64}:

* weights per rank  = sharded classes / TP + replicated classes (norms, mHC);
* active bytes per decode step at bs=1 = attention + routers + shared/dense MLP + lm_head shard +
  8/288 of the routed experts, all / TP  (expectation; the slowest rank at EP=e hosts the busiest
  of e expert groups and reads more, see the report);
* KV bytes per token per rank: the MLA latent (kv_lora_rank + rope) x 11 DSA layers x bf16 is ONE
  latent per token shared by all heads, so it is replicated at every TP (planner rule: the KV
  divisor is min(TP, n_kv) with n_kv = 1); the indexer side caches likewise; the KDA recurrent
  state is per sequence and shards by head (64 heads -> 1 head/rank at TP=64);
* KV budget per rank = min(user budget = GMU x HBM - weights, cap = GMU x HBM x 0.30, physical-core bound =
  min(HBM/2 - 5 GiB reserve, 2 x (HBM/2 - 5 GiB) - weights)), the worker's own formula (``neuron_worker.py:1116-1135``
  ``_physical_core_kv_bound`` and ``:1273-1290`` ``_compute_kv_budget``; inputs as the served configuration logs them:
  ``total_hbm=24.00 GiB``, ``gpu_memory_utilization`` 0.92, ``cap=6.62 GiB``, ``physical_core_bound=7.00 GiB``,
  the recorded bs=64 @ 8k serve run's server log). ``free after the 5 GiB reserve`` (HBM - 5 GiB - weights) is reported too: it is
  the memory left, not the code's bound (section 10 A1 names the per-core term as a software heuristic);
* ITL byte term = active bytes / 716 GB/s; the fixed part of the step (collectives, launch, glue)
  is taken from today's measured 16.96 ms at TP=64 and held constant, which is what makes the
  comparison an entitlement, not a prediction.

The MTP draft layer (``mtp45:*``) is listed separately: its per-iteration bytes at any TP are a few
MiB, so the measured 654 us per draft iteration is latency, not bandwidth, and is TP-independent.

Usage::

    python -m test.vllm_neuron.model.glm5_next.mtp_assess.tp_choice \
        --bytes <reports>/mtp-logs/checkpoint_bytes_by_class.json \
        --output <reports>/mtp-logs/tp_choice.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

GIB = 2**30
HBM_PER_LOGICAL_CORE = 24 * GIB          # total_hbm=24.00 GiB as the worker logs it (the recorded bs=64 @ 8k serve run)
GPU_MEM_UTIL = 0.92                      # the served configuration's gpu_memory_utilization (total_budget=22.08 GiB = 0.92 x 24.00)
HBM_BW = 716e9                           # bytes/s per logical core
GRAPH_RESERVE = 5 * GIB                  # envs.py VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB default
KV_CAP_FRACTION = 0.30                   # envs.py VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION default
ITL_TODAY_MS = 16.96                     # TP=64 EP=16 bs=1 ctx~1k
N_EXPERTS, TOPK = 288, 8
N_DSA, N_KDA = 11, 34
KV_LORA, ROPE = 512, 0                   # MLA latent per token (bf16)
KDA_HEADS, KDA_HEAD_DIM = 64, 128        # recurrent state per head: 128 x 128 fp32; conv state 4 x 3 x 128 x ... small
INDEXER_SIDE_BYTES_PER_TOKEN = 0.3466 * GIB / (64 * 8192)   # 0.3466 GiB for 64 x 8k at TP=64 (per rank)
MLA_HEADS = 64                           # TP ceiling (query heads); KDA also 64 heads
EP_TODAY = 16

REPLICATED = ("norms_other", "mhc")       # tiny; everything else shards by head / row / expert


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bytes", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--tps", type=int, nargs="+", default=[8, 16, 32, 64])
    args = ap.parse_args()
    by = json.load(open(args.bytes))
    trunk = {k: v for k, v in by.items() if not k.startswith("mtp45") and k != "vision"}
    mtp = {k: v for k, v in by.items() if k.startswith("mtp45")}

    def split(classes: dict) -> tuple[int, int, int]:
        """(sharded bytes, replicated bytes, active-per-step bytes at bs=1) before dividing by TP."""
        sharded = sum(v for k, v in classes.items() if not k.endswith(REPLICATED))
        replicated = sum(v for k, v in classes.items() if k.endswith(REPLICATED))
        active = sum(v * (TOPK / N_EXPERTS if k.endswith("routed_experts") else 1.0)
                     for k, v in classes.items() if not k.endswith(REPLICATED) and k != "embed")
        return sharded, replicated, active

    t_sh, t_rep, t_act = split(trunk)
    m_sh, m_rep, m_act = split(mtp)
    kv_token = (KV_LORA + ROPE) * 2 * N_DSA + INDEXER_SIDE_BYTES_PER_TOKEN    # replicated at every TP
    rows = []
    for tp in args.tps:
        if tp > MLA_HEADS:
            continue
        w_trunk = t_sh / tp + t_rep
        w_mtp = m_sh / tp + m_rep
        free = HBM_PER_LOGICAL_CORE - GRAPH_RESERVE - w_trunk - w_mtp          # memory left after the reserve (not the code's bound)
        per_core = HBM_PER_LOGICAL_CORE // 2 - GRAPH_RESERVE                    # neuron_worker.py:1129-1131
        core_bound = max(min(per_core, 2 * per_core - (w_trunk + w_mtp)), 0)   # :1132-1135 (`bytes_used` = the weights here)
        user_budget = GPU_MEM_UTIL * HBM_PER_LOGICAL_CORE - w_trunk - w_mtp
        cap = GPU_MEM_UTIL * HBM_PER_LOGICAL_CORE * KV_CAP_FRACTION
        kv_budget = max(min(user_budget, cap, core_bound), 0)
        verdict = "no" if free <= 0 or kv_budget <= 0 else ("marginal (bs=1 only)" if kv_budget < 2 * GIB else "yes")
        kda_state_per_seq = (KDA_HEADS / tp) * KDA_HEAD_DIM * KDA_HEAD_DIM * 4 * N_KDA
        kv_tokens_1seq = max(kv_budget - kda_state_per_seq, 0) / kv_token
        active = t_act / tp
        byte_ms = active / HBM_BW * 1e3
        draft_bytes = m_act / tp
        rows.append({
            "TP": tp, "EP_options_dividing_288_and_TP": [e for e in (1, 2, 4, 8, 16, 32) if e <= tp and tp % e == 0 and N_EXPERTS % e == 0],
            "weights_per_rank_GiB": round(w_trunk / GIB, 2), "mtp_layer_per_rank_GiB": round(w_mtp / GIB, 3),
            "free_after_graph_reserve_GiB": round(free / GIB, 2), "physical_core_bound_GiB": round(core_bound / GIB, 2),
            "kv_budget_GiB (<= 0.30 cap)": round(kv_budget / GIB, 2),
            "kv_cap_GiB": round(cap / GIB, 2), "fits": verdict,
            "kda_state_per_seq_MiB": round(kda_state_per_seq / 2**20, 1),
            "kv_tokens_one_sequence": int(kv_tokens_1seq),
            "active_bytes_per_step_MiB": round(active / 2**20, 1), "itl_byte_term_ms": round(byte_ms, 3),
            "itl_estimate_ms (today's fixed part held)": round(ITL_TODAY_MS - (t_act / 64 / HBM_BW * 1e3) + byte_ms, 2),
            "draft_iteration_bytes_MiB": round(draft_bytes / 2**20, 2),
            "draft_iteration_byte_term_us": round(draft_bytes / HBM_BW * 1e6, 1),
        })
    out = {
        "inputs": {"total_checkpoint_GiB": round(sum(by.values()) / GIB, 2), "trunk_sharded_GiB": round(t_sh / GIB, 2),
                   "trunk_replicated_GiB": round(t_rep / GIB, 4), "trunk_active_per_step_GiB": round(t_act / GIB, 2),
                   "mtp_layer_GiB": round((m_sh + m_rep) / GIB, 3), "mtp_active_per_iteration_GiB": round(m_act / GIB, 3),
                   "kv_bytes_per_token_per_rank_replicated": round(kv_token), "hbm_per_logical_core_GiB": 24.0, "gpu_mem_util": GPU_MEM_UTIL,
                   "graph_reserve_GiB": 5, "kv_cap_fraction": KV_CAP_FRACTION, "hbm_bw_GBps": 716, "itl_today_ms": ITL_TODAY_MS,
                   "tp_ceiling_query_heads": MLA_HEADS},
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["inputs"], indent=1))
    keys = ["TP", "weights_per_rank_GiB", "free_after_graph_reserve_GiB", "physical_core_bound_GiB", "kv_budget_GiB (<= 0.30 cap)", "kv_tokens_one_sequence",
            "active_bytes_per_step_MiB", "itl_byte_term_ms", "itl_estimate_ms (today's fixed part held)", "draft_iteration_byte_term_us", "fits"]
    print(" | ".join(keys))
    for r in rows:
        print(" | ".join(str(r[k]) for k in keys))


if __name__ == "__main__":
    main()
