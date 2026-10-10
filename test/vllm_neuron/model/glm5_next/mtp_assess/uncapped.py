# SPDX-License-Identifier: Apache-2.0
"""MTP on GLM-5.3-Flash with the KV budget caps removed.

Two questions, both for TP=64 EP=16 on this fork:

(a) Does the GPU recipe line, unchanged, fit the KV budget once the 0.30 cap and the fixed
    5 GiB graph reserve are gone?  Per-rank KV need from the kvseg hand formula
    (the KV-segment budget record, checked here against its three measured points) for
    the recipe line as written (prefix caching on, no ``--mamba-block-size``) and for the
    two-flag variant the bs=64 gate line uses (``--no-enable-prefix-caching
    --mamba-block-size <max_model_len>``, kv.md:153).

(b) How the verify step (T = k+1 rows) and the draft iterations grow with context C:
    DSA indexer (query rotation + ring + scores + top-k select; bypassed at C <= 2051),
    sparse MLA (reads min(C, 2048) latent rows per query row), KDA (independent of C).
    Anchors are the dsa8k and mla slice microbenchmarks (B requests of one row stand in for
    one request of T rows); 32k is an extrapolation (no device shape above 8192 until
    wt2/indexer-ctx lands).  The 1k rows reproduce ``projection.py``'s pessimistic bound
    ((ii) + KDA sequential) exactly, as a self-check.

Writes JSON to ``--output`` and prints the tables.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from . import projection as P

GIB = 1024 ** 3
MIB = 1024 ** 2

# --- (a) KV accounting on this fork, per rank at TP=64 (kvseg_budget.json hand_formula) ----
PAGE_BYTES = 131072          # 128 tokens x 512 latent x bf16, one DSA layer
LAYERS_PER_POOL = 11         # vLLM shares one tensor per layer index across the groups
ATTN_BLOCK = 128             # hybrid_kv_block_size
KDA_GROUPS = 4               # 34 KDA layers grouped 11-wide -> 4 recurrent groups
KDA_LAYERS = 34
KDA_SLOT_BYTES = 67840       # conv + recurrent state per layer per request slot (fp32, 1 head/rank)
INDEX_KPOOL = 4
INDEX_HEAD_DIM = 128
SIDE_ELEM_BYTES = 2          # latent bank dtype (bf16)
HBM_PER_RANK_GIB = 24.00     # neuron_worker log: total_hbm=24.00 GiB (gate runs)
FREE_AT_BUDGET_GIB = 16.94   # the bs=64 tip: 24.00 total, 7.06 used at budget time
CAP_TODAY_GIB = 6.62         # min(user, 0.30 x 22.08, 12 - 5 reserve) today
ASSUMED_UNCAPPED_GIB = (12.0, 15.0)   # KV budget study: free - graphs' real need - margin (16.94 - 5 / - 2, rounded)
# Today's per-physical-core staging bound (neuron_worker.py:1116-1135 `_physical_core_kv_bound`):
# min(HBM/2 - reserve, 2 x (HBM/2 - reserve) - used) = min(12 - 5, 14 - 1.63) = 7.00 GiB (tip-b64-C server.log:6032).
# A1 assumes wave 4 drops that term too; the alternative keeps it with the measured prefill graph need
# (4.567 GiB on one physical core, kv.md) in place of the 5 GiB placeholder: 12 - 4.567 = 7.43 GiB.
GRAPH_NEED_MEASURED_GIB = 4.567
PER_CORE_TERM_KEPT_GIB = HBM_PER_RANK_GIB / 2 - GRAPH_NEED_MEASURED_GIB
MTP_LAYER_PER_RANK_GIB = 6.979 / 64   # the draft layer's weights, sharded (tp_choice.py)


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def kv_need_bytes(L: int, S: int, *, mamba_block_is_L: bool, spec_tokens: int = 0,
                  kda_snapshot: bool = False) -> dict:
    """Per-rank bytes the runner allocates for ``S`` slots of ``L`` tokens.

    ``mamba_block_is_L``: ``--mamba-block-size <max_model_len>`` (one pool block per KDA group
    per request) vs. the default (``cdiv(L, 128)`` blocks per group: kv_spec_patch.py:331,
    "vLLM's mamba 'none' mode reserves cdiv(tokens, block_size) blocks at admission").
    ``spec_tokens``: k draft slots the scheduler reserves per request on top of L.
    ``kda_snapshot``: a second KDA bank per slot for the verify step's rollback (plan B3).
    """
    tokens = L + spec_tokens
    attn_blocks = cdiv(tokens, ATTN_BLOCK)
    kda_blocks = 1 if mamba_block_is_L else cdiv(tokens, ATTN_BLOCK)
    pool_blocks = S * (attn_blocks + KDA_GROUPS * kda_blocks) + 1
    pool = pool_blocks * PAGE_BYTES * LAYERS_PER_POOL
    banks = KDA_LAYERS * S * KDA_SLOT_BYTES * (2 if kda_snapshot else 1)
    side = LAYERS_PER_POOL * S * (L // INDEX_KPOOL + 1 + 2 * 2 * INDEX_KPOOL) * INDEX_HEAD_DIM * SIDE_ELEM_BYTES
    return {"pool_blocks": pool_blocks, "pool_bytes": pool, "kda_bank_bytes": banks,
            "side_cache_bytes": side, "total_bytes": pool + banks + side}


def largest_bs(L: int, budget_bytes: int, **kw) -> int:
    """Largest S whose allocation fits ``budget_bytes`` (0 if one sequence does not)."""
    lo, hi = 0, 1 << 16
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if kv_need_bytes(L, mid, **kw)["total_bytes"] <= budget_bytes:
            lo = mid
        else:
            hi = mid - 1
    return lo


def check_against_kvseg_points() -> list[dict]:
    """The formula must reproduce the KV-segment budget record's three measured points."""
    pts = [
        # (S, L, mamba_block_is_L, need_bytes, footprint_kvseg_bytes)
        (1, 4096, False, 232128512, 237366528),
        (64, 8192, True, 6276120576, 6795902976),
        (119, 4096, True, 6178078720, 6801402624),
    ]
    out = []
    for S, L, flag, need, foot in pts:
        got = kv_need_bytes(L, S, mamba_block_is_L=flag)
        out.append({"S": S, "L": L, "mamba_block_is_L": flag, "pool_bytes": got["pool_bytes"],
                    "kvseg_need_bytes": need, "pool_match": got["pool_bytes"] == need,
                    "total_bytes": got["total_bytes"], "kvseg_footprint_bytes": foot,
                    "total_match": got["total_bytes"] == foot})
    return out


# --- (b) context-dependent kernels, us per DSA layer -------------------------------------
# dsa8k_micro.json (slice, 11-layer graphs, medians): chain = hadamard + ring + scores + select.
# 'tip' = 0a08ff4 code (= b17526a for these kernels), 'after' = wt2/dsa8k 342e93e (DONE, gate queue).
INDEXER = {
    "tip":   {"bypass_b1": 99.4, "bypass_b64": 100.7,
              "chain_8k": {1: 411.7, 4: 438.7, 64: 861.5},
              "scores_8k": {1: 36.2, 4: 60.4, 64: 305.2},
              "select_8k": {1: 326.7, 4: 336.8, 64: 551.4}},
    "after": {"bypass_b1": 82.5, "bypass_b64": 84.5,
              "chain_8k": {1: 161.8, 4: 173.0, 64: 419.4},
              "scores_8k": {1: 36.0, 4: 64.2, 64: 333.5},
              "select_8k": {1: 108.1, 4: 112.4, 64: 168.4}},
}
INDEXER_BYPASS_MAX_CTX = 2051   # selection is the identity at or below this (dsa8k.md:127)
EMPTY_PER_LAYER_US = 13.0       # fixed per-call cost each separately timed sub-graph carries (dsa8k.md:82: 134-189 us per
                                # call / 11 layers); subtracted from scores and select before they are scaled with C
INDEX_TOPK = 2048               # rows the sparse MLA reads per query row above the bypass

# mla_micro.json / mla.md section 5: profiled device time per 11-layer step.
#   fixed per layer at B=1 and per-row cost per 1024 latent rows read.
MLA = {
    "tip":   {"fixed_us": 421.6 / 11 - 11280.3 / 64 / 11, "per_row_per_1k_us": 11280.3 / 64 / 11},
    "after": {"fixed_us": 278.3 / 11 - 5467.2 / 64 / 11, "per_row_per_1k_us": 5467.2 / 64 / 11},
}


def interp_rows(anchors: dict, T: int) -> float:
    """Piecewise-linear in the row count between the measured B anchors (1, 4, 64)."""
    xs = sorted(anchors)
    if T <= xs[0]:
        return anchors[xs[0]]
    for a, b in zip(xs, xs[1:]):
        if T <= b:
            return anchors[a] + (anchors[b] - anchors[a]) * (T - a) / (b - a)
    a, b = xs[-2], xs[-1]
    return anchors[b] + (anchors[b] - anchors[a]) / (b - a) * (T - b)


def indexer_us(C: int, T: int, variant: str, select_scales_with_ctx: bool = True) -> float:
    """One DSA layer's indexer at context C for T query rows of one request.

    C <= 2051: the bypass chain (ring step only; flat in rows, dsa8k.md:127-129).
    C = 8192: the measured chain, rows interpolated between B = 1, 4, 64.
    C > 8192: the C-dependent kernels extrapolated linearly in C/8192: scores (T rows x C/4
    pooled keys) always; select (whole-row counts over C/4 candidates) when
    ``select_scales_with_ctx`` (pessimistic) else held flat.  Device shapes above 8192 exist
    only once wt2/indexer-ctx lands; these rows are to be replaced by its measurements.
    """
    v = INDEXER[variant]
    if C <= INDEXER_BYPASS_MAX_CTX:
        return v["bypass_b1"] + (v["bypass_b64"] - v["bypass_b1"]) * (T - 1) / 63
    base = interp_rows(v["chain_8k"], T)
    if C <= 8192:
        return base
    f = C / 8192 - 1.0
    extra = f * max(interp_rows(v["scores_8k"], T) - EMPTY_PER_LAYER_US, 0.0)
    if select_scales_with_ctx:
        extra += f * max(interp_rows(v["select_8k"], T) - EMPTY_PER_LAYER_US, 0.0)
    return base + extra


def mla_us(C: int, T: int, variant: str) -> float:
    """One DSA layer's sparse MLA decode at context C for T query rows (device time)."""
    v = MLA[variant]
    rows_read = min(C, INDEX_TOPK)
    return v["fixed_us"] + T * v["per_row_per_1k_us"] * rows_read / 1024.0


# Whole-layer cross-check (the indexer microbenchmark, indexer tree = the tip's DSA layer code; the
# same "one layer's decode step" unit as dsa_micro.json): one DSA layer in the SELECTED regime at B=1 takes
# 671.4 us at 4k and 711.2 us at 8k (B=4 939.8, B=16 2141.2 at 8k); the BYPASS layer at 1k B=1 takes 286.64 us
# (dsa_micro.json, the number projection.py's attention half rests on). Composing the tip chain and MLA above
# gives 615 us at 8k B=1, so the selected regime carries a residual neither kernel benchmark accounts for
# (selected-row gather set-up, index glue between the chain and the MLA). It is applied per DSA layer call at
# C > 2051, for both variants (dsa8k and mla replaced the chain and the MLA kernel, not this glue).
LAYER_SELECTED_8K_B1_TIP_US = 711.2
LAYER_BYPASS_1K_B1_TIP_US = 286.64
LAYER_SELECTED_8K_B4_TIP_US = 939.8


def selected_residual_us() -> float:
    composed = (LAYER_BYPASS_1K_B1_TIP_US - indexer_us(1024, 1, "tip") - mla_us(1024, 1, "tip")
                + indexer_us(8192, 1, "tip") + mla_us(8192, 1, "tip"))
    return LAYER_SELECTED_8K_B1_TIP_US - composed


def residual_us(C: int, T: int, mode: str) -> float:
    """``none`` | ``flat`` (one residual per layer call, the default) | ``per_row`` (one per query row: the
    sensitivity case; indexer_micro's selected layer grows 76 us per extra row at 8k against the 41 us the
    chain slope + MLA rows compose, so part of the residual may be per row)."""
    if C <= INDEXER_BYPASS_MAX_CTX or mode == "none":
        return 0.0
    return selected_residual_us() * (T if mode == "per_row" else 1)


def delta_verify_ms(C: int, T: int, variant: str, residual: str = "flat", **kw) -> dict:
    """Extra device time of the T-row attention half at context C over the same T rows at 1k."""
    di = indexer_us(C, T, variant, **kw) - indexer_us(1024, T, variant, **kw)
    dm = mla_us(C, T, variant) - mla_us(1024, T, variant)
    dr = residual_us(C, T, residual)
    return {"indexer_ms": 11 * di / 1000.0, "mla_ms": 11 * dm / 1000.0, "residual_ms": 11 * dr / 1000.0,
            "total_ms": 11 * (di + dm + dr) / 1000.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft-us", type=float, default=654.0, help="fused draft iteration, us (mtp.md 3.3)")
    ap.add_argument("--draft-fixed-us", type=float, default=162.0, help="one draft graph launch per verify step, us")
    ap.add_argument("--verify-overhead-us", type=float, default=200.0)
    ap.add_argument("--ks", type=int, nargs="+", default=[3, 5])
    ap.add_argument("--ctxs", type=int, nargs="+", default=[1024, 8192, 32768])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.6, 0.7, 0.8, 0.9])
    ap.add_argument("--no-index-share", action="store_true",
                    help="draft iterations 1..k-1 re-run the indexer (upstream shares iteration 0's top-k)")
    ap.add_argument("--select-flat-in-ctx", action="store_true", help="32k lower bound: top-k select held flat in C")
    ap.add_argument("--residual", choices=("none", "flat", "per_row"), default="flat",
                    help="selected-regime residual per DSA layer call at C > 2051 (indexer_micro whole layer minus the composed "
                         "kernels): none | flat (default) | per_row (sensitivity)")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    sel_kw = {"select_scales_with_ctx": not args.select_flat_in_ctx}

    out: dict = {"inputs": vars(args) | {"output": str(args.output), "selected_residual_us_per_layer": round(selected_residual_us(), 1),
                                         "indexer_micro_anchors_us": {"selected_8k_b1": LAYER_SELECTED_8K_B1_TIP_US, "selected_8k_b4": LAYER_SELECTED_8K_B4_TIP_US,
                                                                      "bypass_1k_b1": LAYER_BYPASS_1K_B1_TIP_US}},
                 "kvseg_check": check_against_kvseg_points()}
    assert all(p["pool_match"] and p["total_match"] for p in out["kvseg_check"]), out["kvseg_check"]

    # ---- (a) feasibility ------------------------------------------------------------------
    budgets = {"today 6.62 (cap)": CAP_TODAY_GIB, "A1-alt 7.43 (per-core term kept)": round(PER_CORE_TERM_KEPT_GIB, 2),
               "assumed 12 (A1 low)": ASSUMED_UNCAPPED_GIB[0], "assumed 15 (A1 high)": ASSUMED_UNCAPPED_GIB[1]}
    variants = {"recipe line as written (prefix caching on, no --mamba-block-size)": False,
                "+ --no-enable-prefix-caching --mamba-block-size <max_model_len>": True}
    feas = []
    for vname, flag in variants.items():
        for L in (1 << 20, 32768, 8192):
            for S in (1,):
                need = kv_need_bytes(L, S, mamba_block_is_L=flag, spec_tokens=5, kda_snapshot=True)
                row = {"variant": vname, "max_model_len": L, "bs": S, "need_gib": round(need["total_bytes"] / GIB, 3),
                       "pool_blocks": need["pool_blocks"], "side_cache_gib": round(need["side_cache_bytes"] / GIB, 3)}
                for bname, b in budgets.items():
                    row[f"fits @ {bname}"] = need["total_bytes"] <= b * GIB
                feas.append(row)
            if L != 1 << 20:
                for bname, b in budgets.items():
                    S = largest_bs(L, int(b * GIB), mamba_block_is_L=flag, spec_tokens=5, kda_snapshot=True)
                    need = kv_need_bytes(L, max(S, 1), mamba_block_is_L=flag, spec_tokens=5, kda_snapshot=True)
                    one = kv_need_bytes(L, 1, mamba_block_is_L=flag, spec_tokens=5, kda_snapshot=True)["total_bytes"]
                    feas.append({"variant": vname, "max_model_len": L, "bs": S, "budget": bname,
                                 "largest_bs": True, "need_gib": round(need["total_bytes"] / GIB, 3),
                                 "per_seq_gib_incl_k5": round(one / GIB, 3)})
    # the recipe's absent --max-num-seqs: the side caches and banks are allocated at max_num_seqs
    # slots (runner:5254 _glm5next_side_caches), not at the live batch.
    default_s = 256
    alloc_default = kv_need_bytes(1 << 20, default_s, mamba_block_is_L=True)
    out["feasibility"] = {"budgets_gib": budgets, "rows": feas,
                          "default_max_num_seqs_at_1M": {"max_num_seqs": default_s,
                                                         "side_cache_gib": round(alloc_default["side_cache_bytes"] / GIB, 1),
                                                         "kda_banks_gib": round(alloc_default["kda_bank_bytes"] / GIB, 3),
                                                         "pool_gib": round(alloc_default["pool_bytes"] / GIB, 1)},
                          "per_token_bytes": {
                              "attention latent (11 layers)": LAYERS_PER_POOL * PAGE_BYTES // ATTN_BLOCK,
                              "kda groups when mamba_block != L (4 x 11 layers worth of pool)": KDA_GROUPS * LAYERS_PER_POOL * PAGE_BYTES // ATTN_BLOCK,
                              "indexer pool_cache (11 layers)": LAYERS_PER_POOL * INDEX_HEAD_DIM * SIDE_ELEM_BYTES // INDEX_KPOOL},
                          "mtp_layer_weights_per_rank_gib": round(MTP_LAYER_PER_RANK_GIB, 3)}

    # ---- (b) context scaling --------------------------------------------------------------
    anchors = []
    for variant in ("tip", "after"):
        for C in args.ctxs:
            for T in (1, 4, 6):
                anchors.append({"variant": variant, "ctx": C, "T": T,
                                "indexer_us_per_layer": round(indexer_us(C, T, variant, **sel_kw), 1),
                                "mla_us_per_layer": round(mla_us(C, T, variant), 1),
                                "residual_us_per_layer": round(residual_us(C, T, args.residual), 1),
                                "delta_vs_1k_ms_per_step": round(delta_verify_ms(C, T, variant, args.residual, **sel_kw)["total_ms"], 3)})
    out["context_anchors"] = anchors

    ii_1 = P.verify_bound_ii(1)["step_ms"]
    kda_count, kda_b1, _b4, _src = P.KERNELS["kda fused step"]
    rows = []
    break_even = {}
    for variant in ("after", "tip"):
        for C in args.ctxs:
            base_itl = P.ITL_TODAY_MS + delta_verify_ms(C, 1, variant, args.residual, **sel_kw)["total_ms"]
            for k in args.ks:
                T = k + 1
                bii = P.verify_bound_ii(T)
                verify_1k = P.ITL_TODAY_MS * (bii["step_ms"] + kda_count * kda_b1 * k / 1000.0) / ii_1
                dv = delta_verify_ms(C, T, variant, args.residual, **sel_kw)
                verify = verify_1k + dv["total_ms"]
                d_mla = (mla_us(C, 1, variant) - mla_us(1024, 1, variant)) * k
                d_idx = (indexer_us(C, 1, variant, **sel_kw) - indexer_us(1024, 1, variant, **sel_kw)) * (k if args.no_index_share else 1)
                d_res = residual_us(C, 1, args.residual) * k          # the MTP layer is a DSA layer: one residual per iteration
                draft = (k * args.draft_us + args.draft_fixed_us + d_mla + d_idx + d_res) / 1000.0
                num = draft + verify + args.verify_overhead_us / 1000.0
                for alpha in args.alphas:
                    n = P.expected_accepted(k, alpha)
                    itl = num / n
                    rows.append({"variant": variant, "ctx": C, "k": k, "alpha": alpha, "N": round(n, 4),
                                 "baseline_itl_ms": round(base_itl, 3), "draft_ms": round(draft, 3),
                                 "verify_ms": round(verify, 3), "verify_ctx_extra_ms": round(dv["total_ms"], 3),
                                 "itl_eff_ms": round(itl, 3), "speedup_vs_baseline_at_ctx": round(base_itl / itl, 4)})
                target_n = num / base_itl
                lo, hi = 0.0, 0.999999
                if P.expected_accepted(k, hi) < target_n:
                    break_even[f"{variant}_ctx{C}_k{k}"] = None
                else:
                    for _ in range(60):
                        mid = (lo + hi) / 2
                        if P.expected_accepted(k, mid) < target_n:
                            lo = mid
                        else:
                            hi = mid
                    break_even[f"{variant}_ctx{C}_k{k}"] = round(hi, 4)
    out["projection"] = {"bound": "(ii) kernel-by-kernel + KDA sequential (projection.py) + context extra for the attention half",
                         "rows": rows, "break_even_alpha": break_even}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=1))

    # ---- print ------------------------------------------------------------------------------
    print("kvseg points reproduced:", all(p["pool_match"] and p["total_match"] for p in out["kvseg_check"]))
    print(f"selected-regime residual: {selected_residual_us():.1f} us per DSA layer call at C > 2051 (mode {args.residual})")
    print(f"\n(a) KV need per rank, TP=64, incl. k=5 slots + KDA rollback snapshot; budgets GiB: {budgets}")
    for r in feas:
        if r.get("largest_bs"):
            print(f"  {r['variant'][:60]:60s} L={r['max_model_len']:>8d} budget {r['budget']:18s} largest bs {r['bs']:>4d} ({r['need_gib']} GiB)")
        else:
            fits = ", ".join(f"{k[7:]}: {'yes' if v else 'NO'}" for k, v in r.items() if k.startswith("fits"))
            print(f"  {r['variant'][:60]:60s} L={r['max_model_len']:>8d} bs={r['bs']} need {r['need_gib']:>7.3f} GiB  [{fits}]")
    d = out["feasibility"]["default_max_num_seqs_at_1M"]
    print(f"  absent --max-num-seqs -> {d['max_num_seqs']} slots at 1M: side caches {d['side_cache_gib']} GiB, pool {d['pool_gib']} GiB")
    print("\n(b) attention half per DSA layer, us (indexer / sparse MLA) and the per-step extra over 1k, ms")
    for a in anchors:
        print(f"  {a['variant']:5s} ctx {a['ctx']:>6d} T={a['T']}  idx {a['indexer_us_per_layer']:>7.1f}  mla {a['mla_us_per_layer']:>6.1f}  res {a['residual_us_per_layer']:>6.1f}  +{a['delta_vs_1k_ms_per_step']:>6.3f} ms/step")
    print("\n(b) ITL projection, pessimistic bound, vs the baseline ITL at the same context")
    print("variant ctx     k  alpha  N      base   draft  verify  ITL_eff  x")
    for r in rows:
        print(f"{r['variant']:5s} {r['ctx']:>7d} {r['k']}  {r['alpha']:.1f}  {r['N']:.3f}  {r['baseline_itl_ms']:6.2f}  {r['draft_ms']:5.2f}  {r['verify_ms']:6.2f}  {r['itl_eff_ms']:6.2f}  {r['speedup_vs_baseline_at_ctx']:.3f}")
    print("break-even alpha:", break_even)


if __name__ == "__main__":
    main()
