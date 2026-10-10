# SPDX-License-Identifier: Apache-2.0
"""bs=1 ITL projection for MTP speculative decoding on GLM-5.3-Flash (acceptance item 4).

    ITL_eff(k, alpha) = (k * draft + verify(k+1) + overhead) / N(k, alpha)
    N(k, alpha)       = (1 - alpha**(k+1)) / (1 - alpha)        (expected accepted + bonus)

``verify(k+1)`` is bounded two ways: (i) today's step, as if T=k+1 cost what T=1 costs;
(ii) a kernel-by-kernel estimate from the per-kernel microbenchmarks at B=1 and B=4
(B requests of one token each stand in for one request of T tokens: identical weight
traffic and activation rows; attention differs only in reading one context instead of
four, which makes (ii) an upper bound for attention) and the DECODE_BREAKDOWN_v2 point-A
in-model buckets for the parts no microbenchmark covers (glue, waits, collectives).

Every input number is a constant below with its source. Writes JSON to ``--output``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# --- inputs ----------------------------------------------------------------------------
ITL_TODAY_MS = 16.96            # bs=1 ctx~1k standard line, tip 8aa22fa
DEVICE_STEP_MS = 16.63          # unprofiled device period, point A
HOST_SLACK_MS = ITL_TODAY_MS - DEVICE_STEP_MS

# point-A in-model buckets (DECODE_BREAKDOWN_v2 master table, profiled step 18.34 ms)
POINT_A = {
    "compiler ops (glue)": 5.01, "wait: collective": 2.55, "idle": 1.98, "wait: DMA": 1.56,
    "mhc/sinkhorn": 1.36, "moe/expert_decode": 0.90, "kda/fused_decode": 0.69,
    "wait: core barrier": 0.69, "wait: semaphore": 0.64, "mla_projections": 0.53,
    "mla_decode": 0.49, "moe/router_decode": 0.48, "dma issue": 0.41, "blockwise_fp8_mm": 0.29,
    "mhc/hyper_connection": 0.24, "dsa/kpool_hadamard": 0.20, "dsa/decode_batch": 0.17,
    "mla_absorb": 0.14,
}
PROFILED_STEP_MS = 18.34

# per-kernel device microbenchmarks, us per call at B=1 and B=4 ("after" medians), with counts per step
# (ledger --tip b17526a --bs 1 --ctx 1024). mla_sparse: dsa_micro layer/bypass B1 and B4 (B4 = four
# one-request forwards in one graph: the per-request serial form the layer has today).
KERNELS = {
    #                      count, us@B1,   us@B4,  source
    "mhc sinkhorn x2/layer": (90, 23.30, 24.92, "mhc_micro.json cases B1/B4 after.sinkhorn.device_per_call"),
    "mhc combine x2/layer": (90, 6.16, 10.38, "mhc_micro.json after.combine.device_per_call"),
    "kda fused step": (34, 32.72, 33.39, "kda_micro.json summary B=1/B=4 after (34-layer graph)"),
    "dsa/mla layer (attention half)": (11, 286.64, 1033.22, "dsa_micro.json layer/bypass B1/B4 after"),
    "moe router": (42, 22.86, 23.65, "moe-t_micro.json router T1/T4 after"),
    "moe experts": (42, 47.15, 88.38, "moe-t_micro.json experts T1 (0/1/2 hits weighted, ledger) / T4 2_distinct after"),
    "shared expert mlp": (42, 22.80, 23.95, "dense_micro.json shared_mlp_b1/b4 after per_site"),
    "dense mlp": (3, 30.26, 30.26, "dense_micro.json dense_mlp_b1 (b4 not measured; GEMV, taken flat)"),
    "norms (attn+ffn)": (90, 5.32, 5.32, "dense_micro.json norm_b1 before (XLA lowering stays wired); per-row op, taken flat"),
    "lm_head shard": (1, 37.14, 37.14, "dense_micro.json lm_head_b1 after (GEMV; taken flat)"),
    "lm_head all-gather": (1, 20.60, 21.00, "ledger model 17.5 us + bytes/100 GB/s"),
    "all-reduce x2/layer": (90, 17.66, 18.15, "ledger model 17.5 us + bytes/100 GB/s (8 KiB -> 32 KiB)"),
    "sampler (all_greedy argmax)": (1, 146.32, 1322.94, "host_sampler_micro.json B1/B4 all_greedy after_device_sampler"),
}

# the draft step (one MTP layer = DSA attention half + MoE half + head extras), us per step at B=1,
# filled in from benchmark_mtp_draft.py (device) + the MoE/dense microbenchmarks.
DRAFT_DEFAULT_US = {"head_block_measured_or_estimated": None}


def expected_accepted(k: int, alpha: float, decay: float = 1.0) -> float:
    """Expected tokens per verify step with k drafts.

    Constant acceptance ``alpha``: N = (1 - alpha^(k+1)) / (1 - alpha). With ``decay`` < 1 the
    acceptance at draft position i is ``alpha * decay**(i-1)`` (the per-position decay GPU
    users see), and N = 1 + sum_j prod_{i<=j} alpha_i.
    """
    if decay == 1.0:
        return (1.0 - alpha ** (k + 1)) / (1.0 - alpha) if alpha < 1.0 else float(k + 1)
    n, run = 1.0, 1.0
    for i in range(1, k + 1):
        run *= alpha * decay ** (i - 1)
        n += run
    return n


def verify_bound_ii(T: int) -> dict:
    """Kernel-by-kernel step at T tokens per request, ms, from the B=1 -> B=4 micro slopes.

    T <= 4 interpolates each kernel between its B=1 and B=4 medians; T > 4 (k=5 -> T=6) extrapolates
    the same per-row slope, i.e. assumes the per-request-serial forms keep their B=1..4 slope.
    """
    rows = {}
    kernel_total = 0.0
    for name, (count, b1, b4, _src) in KERNELS.items():
        per_call = b1 + (b4 - b1) * (T - 1) / 3.0 if T <= 4 else b4 + (b4 - b1) / 3.0 * (T - 4)
        ms = count * per_call / 1000.0
        rows[name] = round(ms, 3)
        kernel_total += ms
    # in-model parts no microbenchmark covers: glue + waits + idle, taken flat in T
    # (optimistic: glue is per-row elementwise and norms/casts grow with rows; DMA waits grow with
    # weight-streaming kernels' extra tiles), scaled from the profiled to the unprofiled step.
    flat = sum(POINT_A[k] for k in ("compiler ops (glue)", "idle", "wait: DMA", "wait: core barrier",
                                    "wait: semaphore", "dma issue"))
    flat_unprofiled = flat * DEVICE_STEP_MS / PROFILED_STEP_MS
    # collective exposed wait beyond the modelled transfer is in POINT_A; keep its excess flat too.
    coll_excess = POINT_A["wait: collective"] - 1.61
    return {"T": T, "kernels_ms": rows, "kernels_total_ms": round(kernel_total, 3),
            "glue_waits_flat_ms": round(flat_unprofiled, 3), "collective_excess_flat_ms": round(coll_excess, 3),
            "step_ms": round(kernel_total + flat_unprofiled + coll_excess, 3)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-us", type=float, required=True,
                        help="one draft step at B=1, us (device, incl. its own launch)")
    parser.add_argument("--draft-launch-us", type=float, default=0.0,
                        help="extra per-draft-step overhead (host dispatch, hidden-state hand-off), us")
    parser.add_argument("--draft-fixed-us", type=float, default=0.0,
                        help="per-VERIFY-step draft overhead paid once whatever k (the launch of one fused draft graph "
                             "that unrolls the k iterations), us")
    parser.add_argument("--alpha-decay", type=float, default=1.0,
                        help="per-position acceptance decay: alpha_i = alpha * decay**(i-1) (1.0 = constant alpha)")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 5])
    parser.add_argument("--kda-sequential", action="store_true",
                        help="bound (ii): the KDA recurrence takes the T verify tokens one after another (34 layers x "
                             "the fused one-token step x (T-1)) instead of the batched B=1->B=4 slope the micro numbers give")
    parser.add_argument("--verify-overhead-us", type=float, default=0.0,
                        help="extra per-verify-step overhead (rejection sampling, k+1 rows, extra graph inputs), us")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ks = tuple(args.ks)
    alphas = (0.6, 0.7, 0.8, 0.9)
    # calibrate bound (ii) so that its T=1 reproduces today's device period: the micro numbers are
    # standalone medians, the in-model numbers are smaller (the ledger's k); use the RATIO at T>1 vs T=1.
    ii_1 = verify_bound_ii(1)["step_ms"]
    table = {"inputs": {"itl_today_ms": ITL_TODAY_MS, "device_step_ms": DEVICE_STEP_MS, "host_slack_ms": round(HOST_SLACK_MS, 3),
                        "draft_us": args.draft_us, "draft_launch_us": args.draft_launch_us,
                        "draft_fixed_us": args.draft_fixed_us, "alpha_decay": args.alpha_decay,
                        "kda_sequential": bool(args.kda_sequential),
                        "verify_overhead_us": args.verify_overhead_us,
                        "bound_ii_raw_T1_ms": round(ii_1, 3)},
             "bound_ii": {}, "rows": [], "break_even_alpha": {}, "best_case_alpha_0_8": {}}
    for k in ks:
        bii = verify_bound_ii(k + 1)
        if args.kda_sequential:
            count, kda_b1, _b4, _src = KERNELS["kda fused step"]
            extra = count * kda_b1 * k / 1000.0          # (T - 1) = k extra sequential tokens
            bii = bii | {"step_ms": bii["step_ms"] + extra, "kda_sequential_extra_ms": round(extra, 3)}
        ratio = bii["step_ms"] / ii_1
        verify_i = ITL_TODAY_MS                       # (i) today's graph, T=k+1 free
        verify_ii = ITL_TODAY_MS * ratio              # (ii) kernel-by-kernel scaling applied to today's ITL
        table["bound_ii"][str(k + 1)] = bii | {"ratio_vs_T1": round(ratio, 4), "verify_ms_i": verify_i,
                                               "verify_ms_ii": round(verify_ii, 3)}
        draft_total_ms = (k * (args.draft_us + args.draft_launch_us) + args.draft_fixed_us) / 1000.0
        for alpha in alphas:
            n = expected_accepted(k, alpha, args.alpha_decay)
            for bound, verify in (("i", verify_i), ("ii", verify_ii)):
                itl = (draft_total_ms + verify + args.verify_overhead_us / 1000.0) / n
                table["rows"].append({"k": k, "alpha": alpha, "bound": bound, "N": round(n, 4),
                                      "draft_ms": round(draft_total_ms, 3), "verify_ms": round(verify, 3),
                                      "itl_eff_ms": round(itl, 3), "speedup": round(ITL_TODAY_MS / itl, 4)})
        # break-even alpha: ITL_eff == ITL_today  <=>  N = (k*draft + verify + ovh) / ITL_today
        for bound, verify in (("i", verify_i), ("ii", verify_ii)):
            target_n = (draft_total_ms + verify + args.verify_overhead_us / 1000.0) / ITL_TODAY_MS
            lo, hi = 0.0, 0.999999
            if expected_accepted(k, hi, args.alpha_decay) < target_n:
                table["break_even_alpha"][f"k{k}_{bound}"] = None   # never breaks even
                continue
            for _ in range(60):
                mid = (lo + hi) / 2
                if expected_accepted(k, mid, args.alpha_decay) < target_n:
                    lo = mid
                else:
                    hi = mid
            table["break_even_alpha"][f"k{k}_{bound}"] = round(hi, 4)
    for bound in ("i", "ii"):
        best = min((r for r in table["rows"] if r["alpha"] == 0.8 and r["bound"] == bound), key=lambda r: r["itl_eff_ms"])
        table["best_case_alpha_0_8"][bound] = best
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(table, indent=1))
    # human table
    print(f"draft {args.draft_us:.0f} us + launch {args.draft_launch_us:.0f} us per draft iteration, + {args.draft_fixed_us:.0f} us per verify step; "
          f"verify overhead {args.verify_overhead_us:.0f} us; alpha decay {args.alpha_decay}; ITL today {ITL_TODAY_MS} ms")
    print("k  alpha  N      verify(i)  ITL(i)   x(i)    verify(ii)  ITL(ii)  x(ii)")
    for k in ks:
        for alpha in alphas:
            ri = next(r for r in table["rows"] if r["k"] == k and r["alpha"] == alpha and r["bound"] == "i")
            rii = next(r for r in table["rows"] if r["k"] == k and r["alpha"] == alpha and r["bound"] == "ii")
            print(f"{k}  {alpha:.1f}    {ri['N']:.3f}  {ri['verify_ms']:7.2f}   {ri['itl_eff_ms']:6.2f}  {ri['speedup']:.3f}   "
                  f"{rii['verify_ms']:7.2f}     {rii['itl_eff_ms']:6.2f}  {rii['speedup']:.3f}")
    print("break-even alpha:", table["break_even_alpha"])
    print("bound (ii) ratios:", {T: v["ratio_vs_T1"] for T, v in table["bound_ii"].items()})


if __name__ == "__main__":
    main()
