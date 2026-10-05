# SPDX-License-Identifier: Apache-2.0
"""MoE decode microbenchmark: 5938748 router and routed experts vs the decode kernels.

Run under ``devlease.py slice moe-t`` (one trn2 chip, LNC2). This script does not
select cores. All references are computed on CPU or on the other variant.

What is timed. Two families, each per decoder layer:

* ``router``: ``route_tokens`` of ``[T, H]`` pre-norm activations. Before: the
  5938748 ``noaux_tc_rmsnorm_router_topk`` (pads T to 256 rows). After:
  ``router_decode.noaux_tc_router_decode``.
* ``experts``: one rank's 18-expert shard (I = 512 per rank, H = 4096) on the
  router's global ``[T, 288]`` output and a device rank tensor, the packed branch
  of ``block_quant_expert_mm``. Before: 5938748 local gather,
  ``build_blockwise_mapping``, padding, ``fused_fp8_experts`` (compact kernel at
  T=1, general kernel above), fp32 token-gather combine, bf16 cast. After:
  ``expert_decode.fused_fp8_decode_experts`` (one launch, bf16 out).

Method. Host launch and sync cost far more than one layer, so each variant is
compiled twice into one graph: 1 layer and ``L`` layers (distinct inputs and
distinct weight copies per layer; each layer's output is summed into one small
tensor so no layer is dead code). The per-layer cost is the slope:
``(t_L - median(t_1)) / (L - 1)`` per timed ``t_L`` sample, whose median and p90
are reported. The sum adds one ``[T, E]`` or ``[T, H]`` add per layer to both
variants alike. Expert scenarios fix the number of distinct local experts per
layer, so the hit count is a controlled input rather than a routing accident.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time

import torch

import vllm_neuron  # noqa: F401 -- registers the Neuron compilation backend
import libtorch_neuronx_lite.envs as libtorch_envs

from vllm_neuron.functional.moe.expert_decode import expert_decode_torch_oracle
from vllm_neuron.functional.moe.fused_fp8 import fused_fp8_decode_experts
from vllm_neuron.functional.moe.fused_fp8_pack import PackedExperts, pack_experts
from vllm_neuron.functional.moe.moe_blockwise_fp8 import _swiglu_bound_operand
from vllm_neuron.functional.moe.router_decode import (
    noaux_tc_router_decode,
    router_decode_torch_oracle,
)

HIDDEN, EXPERTS, TOP_K, SCALING, EPS = 4096, 288, 8, 2.5, 1e-5
LOCAL_EXPERTS, LOCAL_INTERMEDIATE, SWIGLU_LIMIT, RANK = 18, 512, 10.0, 5
CHECKPOINT = Path("/home/ubuntu/glm53f-campaign/lane-serve/models/GLM-5.3-Flash-04c4e9e9")
DEVICE = "neuron:0"

#: Distinct local experts per layer (token -> local expert ids) per scenario.
SCENARIOS = {
    1: {
        "0_hits": [[]],
        "1_hit": [[7]],
        "2_hits": [[3, 11]],
    },
    4: {
        # Expected distinct experts at T=4: 288 * (1 - (1 - 8/288)**4) = 30.7,
        # 1.9 per 18-expert group; a loaded group sees 4.
        "2_distinct": [[3], [11], [3, 11], []],
        "4_distinct": [[3, 11], [11], [], [0, 3, 17]],
    },
    64: {
        # 288 * (1 - (1 - 8/288)**64) = 240.6 distinct, 15.0 per group.
        "15_distinct": None,
    },
}


def load_baseline(path: Path):
    name = "moe_5938748"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, path / "__init__.py", submodule_search_locations=[str(path)])
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load baseline package {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return importlib.import_module(f"{name}.pipeline")


# ---- Inputs ----------------------------------------------------------------- #


def router_inputs(layers: int, seed: int):
    """Per-layer router weight, bias and gain: the checkpoint's when present."""
    gen = torch.Generator().manual_seed(seed)
    out = []
    index = None
    if (CHECKPOINT / "model.safetensors.index.json").exists():
        index = json.loads((CHECKPOINT / "model.safetensors.index.json").read_text())
    for layer in range(layers):
        if index is not None:
            from safetensors import safe_open

            prefix = f"model.language_model.layers.{3 + layer}"
            tensors = []
            for name in (f"{prefix}.mlp.gate.weight", f"{prefix}.mlp.gate.e_score_correction_bias",
                         f"{prefix}.post_attention_layernorm.weight"):
                with safe_open(str(CHECKPOINT / index["weight_map"][name]), framework="pt") as f:
                    tensors.append(f.get_tensor(name))
            weight, bias, gamma = tensors
            out.append((weight.to(torch.bfloat16).t().contiguous(), bias.to(torch.float32),
                        gamma.to(torch.bfloat16)))
        else:
            weight = (torch.randn(HIDDEN, EXPERTS, generator=gen) / HIDDEN ** 0.5)
            bias = (torch.rand(EXPERTS, generator=gen) - 0.5) * 0.1
            gamma = 1.0 + 0.1 * torch.randn(HIDDEN, generator=gen)
            out.append((weight.to(torch.bfloat16), bias, gamma.to(torch.bfloat16)))
    return out, index is not None


def activations(tokens: int, layers: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    scale = torch.ones(HIDDEN)
    scale[torch.randperm(HIDDEN, generator=gen)[:24]] = 20.0
    return [(torch.randn(tokens, HIDDEN, generator=gen) * scale).to(torch.bfloat16)
            for _ in range(layers)]


def expert_bank(seed: int = 23) -> PackedExperts:
    gen = torch.Generator().manual_seed(seed)
    nh, ni = HIDDEN // 128, LOCAL_INTERMEDIATE // 128

    def fp8(*shape):
        return (torch.randn(*shape, generator=gen) * 48).clamp(-240, 240).to(torch.float8_e4m3fn)

    gate_up = fp8(LOCAL_EXPERTS, HIDDEN, 2 * LOCAL_INTERMEDIATE)
    down = fp8(LOCAL_EXPERTS, LOCAL_INTERMEDIATE, HIDDEN)
    gu_scales = (0.5 + torch.rand(LOCAL_EXPERTS, nh, 2, ni, generator=gen)) * (3.0 / (48 * 64))
    d_scales = (0.5 + torch.rand(LOCAL_EXPERTS, ni, nh, generator=gen)) * (
        1.0 / (48 * LOCAL_INTERMEDIATE ** 0.5))
    return pack_experts(gate_up, down, gu_scales, d_scales)


def routed(local_hits, seed: int):
    """Scattered [T, 288] router output with these local hits on group RANK."""
    gen = torch.Generator().manual_seed(seed)
    first = RANK * LOCAL_EXPERTS
    others = [e for e in range(EXPERTS) if not first <= e < first + LOCAL_EXPERTS]
    rows = []
    for hits in local_hits:
        picks = [first + h for h in hits]
        rest = torch.randperm(len(others), generator=gen)[: TOP_K - len(picks)]
        picks += [others[i] for i in rest.tolist()]
        weight = torch.rand(TOP_K, generator=gen) + 0.1
        row = torch.zeros(EXPERTS)
        row[torch.tensor(picks)] = weight / weight.sum() * SCALING
        rows.append(row)
    return torch.stack(rows)


def scenario_hits(tokens: int, pattern, layer: int):
    if pattern is not None:
        return pattern
    # T=64: 15 of 18 local experts, each token choosing 0-2 of them.
    gen = torch.Generator().manual_seed(1000 + layer)
    chosen = torch.randperm(LOCAL_EXPERTS, generator=gen)[:15].tolist()
    hits = [[] for _ in range(tokens)]
    for slot, expert in enumerate(chosen):
        hits[slot % tokens].append(expert)
    for t in range(tokens):
        extra = chosen[int(torch.randint(0, 15, (1,), generator=gen))]
        if extra not in hits[t] and len(hits[t]) < TOP_K and t % 3 == 0:
            hits[t].append(extra)
    return [sorted(h) for h in hits]


# ---- Graphs ----------------------------------------------------------------- #


def compile_fn(fn):
    return torch.compile(fn, backend="neuron_libtorch", fullgraph=True, dynamic=False,
                         options={"compiler_args": os.environ.get("NEURON_CC_FLAGS", "")})


def router_graph(variant: str, layers: int, baseline):
    def chained(xs, weights, biases, gammas):
        total = None
        for layer in range(layers):
            if variant == "before":
                _, _, aff = baseline.route_tokens(
                    xs[layer].unsqueeze(0), gammas[layer], weights[layer], biases[layer],
                    top_k=TOP_K, eps=EPS, norm_topk_prob=True,
                    routed_scaling_factor=SCALING)
            else:
                _, _, aff = noaux_tc_router_decode(
                    xs[layer], gammas[layer], weights[layer], biases[layer], top_k=TOP_K,
                    eps=EPS, norm_topk_prob=True, routed_scaling_factor=SCALING)
            total = aff if total is None else total + aff
        return total

    return compile_fn(chained)


def expert_graph(variant: str, layers: int, baseline, bounds, weight_fp8=False):
    def chained(xs, affs, weights, scales, rank):
        total = None
        for layer in range(layers):
            if variant == "before":
                y = baseline.routed_experts(
                    xs[layer], affs[layer], weights[layer], scales[layer], rank,
                    top_k=TOP_K, swiglu_limit=SWIGLU_LIMIT)
            else:
                y = fused_fp8_decode_experts(
                    xs[layer], affs[layer], PackedExperts(weights[layer], scales[layer]),
                    bounds, rank, weight_fp8=weight_fp8)
            total = y if total is None else total + y
        return total

    return compile_fn(chained)


# ---- Timing ----------------------------------------------------------------- #


def samples_ms(fn, inputs, warmup, iterations):
    for _ in range(warmup):
        fn(*inputs).to("cpu")
    out = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        fn(*inputs).to("cpu")
        out.append((time.perf_counter_ns() - started) / 1e6)
    return out


def percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def per_layer(one, many, layers):
    base = statistics.median(one)
    slopes = [(t - base) / (layers - 1) * 1000.0 for t in many]  # us
    return {
        "median_us": statistics.median(slopes),
        "p90_us": percentile(slopes, 0.9),
        "min_us": min(slopes),
        "one_layer_call_median_ms": base,
        f"{layers}_layer_call_median_ms": statistics.median(many),
    }


def timed(graph_one, graph_many, inputs_one, inputs_many, layers, args):
    one = samples_ms(graph_one, inputs_one, args.warmup, args.iterations)
    many = samples_ms(graph_many, inputs_many, args.warmup, args.iterations)
    result = per_layer(one, many, layers)
    result["iterations"] = args.iterations
    result["warmup"] = args.warmup
    return result


def profile(label, graphs_and_inputs, args):
    if args.profile_dir is None:
        return None
    path = args.profile_dir / label
    path.mkdir(parents=True, exist_ok=True)
    runtime = torch.classes.neuron.Runtime()
    runtime.start_profiling(str(path), ["device_profile", "system_profile"], None, None,
                            libtorch_envs.get_neuron_compile_cache_dir())
    try:
        for _ in range(args.profile_iterations):
            for graph, inputs in graphs_and_inputs:
                graph(*inputs).to("cpu")
    finally:
        runtime.stop_profiling()
    return str(path)


# ---- Cases ------------------------------------------------------------------ #


def router_case(tokens, variants, baseline, args):
    torch._dynamo.reset()
    layers = args.layers
    params, from_checkpoint = router_inputs(layers, seed=97)
    xs = activations(tokens, layers, seed=101 + tokens)
    dev = lambda ts: [t.to(DEVICE) for t in ts]  # noqa: E731
    inputs = {
        n: (dev(xs[:n]), dev([p[0] for p in params[:n]]), dev([p[1] for p in params[:n]]),
            dev([p[2] for p in params[:n]]))
        for n in (1, layers)
    }
    case = {"family": "router", "T": tokens, "layers": layers,
            "router_weights": "checkpoint layers 3.." if from_checkpoint else "random"}
    outputs = {}
    for variant in variants:
        one, many = router_graph(variant, 1, baseline), router_graph(variant, layers, baseline)
        outputs[variant] = one(*inputs[1]).to("cpu")
        case[variant] = timed(one, many, inputs[1], inputs[layers], layers, args)
        case[variant]["profile_dir"] = profile(f"router_T{tokens}_{variant}",
                                               [(one, inputs[1])], args)
    # Correctness on device: the layer-0 affinities against the torch oracle.
    w, b, g = params[0]
    _, ref_index, ref_aff = router_decode_torch_oracle(
        xs[0], g.reshape(1, -1), w, b.reshape(1, -1), EPS, True, SCALING)
    checks = {}
    for variant, aff in outputs.items():
        support = (aff != 0)
        same = torch.equal(support, ref_aff != 0)
        checks[variant] = {
            "index_sets_equal_oracle": bool(same),
            "max_abs_affinity_vs_oracle": float((aff - ref_aff).abs().max()),
        }
        if not same or not torch.allclose(aff, ref_aff, rtol=1e-3, atol=1e-5):
            raise AssertionError(f"router {variant} T={tokens} disagrees with the oracle: "
                                 f"{checks[variant]}")
    case["checks"] = checks
    if "before" in case and "after" in case:
        case["speedup_median"] = case["before"]["median_us"] / case["after"]["median_us"]
    return case


def expert_cases(tokens, variants, baseline, args):
    torch._dynamo.reset()
    layers = args.layers
    bank = expert_bank()
    bounds_cpu = _swiglu_bound_operand(SWIGLU_LIMIT, SWIGLU_LIMIT, torch.device("cpu"))
    bounds = bounds_cpu.to(DEVICE)
    # Distinct device copies of the bank per layer: no layer re-reads another's bytes.
    weights = [bank.weights.to(DEVICE) for _ in range(layers)]
    scales = [bank.scales.to(DEVICE) for _ in range(layers)]
    rank = torch.tensor([RANK], dtype=torch.int64).to(DEVICE)
    xs = [(torch.randn(tokens, HIDDEN, generator=torch.Generator().manual_seed(200 + l))
           ).to(torch.bfloat16) for l in range(layers)]
    graphs = {}
    for variant in variants:
        fp8 = variant == "after_fp8_stationary"
        kind = "before" if variant == "before" else "after"
        graphs[variant] = (expert_graph(kind, 1, baseline, bounds, fp8),
                           expert_graph(kind, layers, baseline, bounds, fp8))
    cases = []
    for name, pattern in SCENARIOS[tokens].items():
        hits = [scenario_hits(tokens, pattern, l) for l in range(layers)]
        affs = [routed(hits[l], seed=300 + l) for l in range(layers)]
        distinct = [len({e for row in h for e in row}) for h in hits]
        pairs = [sum(len(row) for row in h) for h in hits]
        inputs = {
            n: ([x.to(DEVICE) for x in xs[:n]], [a.to(DEVICE) for a in affs[:n]],
                weights[:n], scales[:n], rank)
            for n in (1, layers)
        }
        case = {"family": "experts", "T": tokens, "scenario": name, "layers": layers,
                "distinct_local_experts_per_layer": distinct,
                "token_expert_pairs_per_layer": pairs}
        ref = expert_decode_torch_oracle(xs[0], affs[0], bank, bounds_cpu, RANK,
                                         out_dtype=torch.float32)
        checks = {}
        for variant, (one, many) in graphs.items():
            out = one(*inputs[1]).to("cpu").to(torch.float32)
            diff = (out - ref).abs()
            checks[variant] = {
                "max_abs_vs_oracle": float(diff.max()),
                "max_abs_ref": float(ref.abs().max()),
            }
            if not torch.allclose(out, ref, rtol=2.0 ** -7, atol=1e-3 * float(ref.abs().max()) + 1e-6):
                raise AssertionError(f"experts {variant} T={tokens} {name} disagrees with the "
                                     f"oracle: {checks[variant]}")
            case[variant] = timed(one, many, inputs[1], inputs[layers], layers, args)
            case[variant]["profile_dir"] = profile(
                f"experts_T{tokens}_{name}_{variant}", [(one, inputs[1])], args)
        case["checks"] = checks
        if "before" in case and "after" in case:
            case["speedup_median"] = case["before"]["median_us"] / case["after"]["median_us"]
        cases.append(case)
        print(json.dumps({k: v for k, v in case.items() if k != "checks"}), flush=True)
    return cases


def per_hit(report):
    """Fixed and marginal per-expert cost per variant at T=1 (0, 1, 2 hit layers)."""
    t1 = {c["scenario"]: c for c in report["cases"]
          if c["family"] == "experts" and c["T"] == 1}
    if not {"0_hits", "1_hit", "2_hits"} <= set(t1):
        return None
    out = {}
    for variant in ("before", "after", "after_fp8_stationary"):
        if variant not in t1["1_hit"]:
            continue
        zero, one, two = (t1[k][variant]["median_us"] for k in ("0_hits", "1_hit", "2_hits"))
        out[variant] = {"zero_hit_layer_us": zero, "first_hit_us": one - zero,
                        "second_hit_us": two - one}
    return out


def summarize(report):
    """Router + experts per layer at each T and scenario, and the 0.6 criterion."""
    rows = []
    routers = {c["T"]: c for c in report["cases"] if c["family"] == "router"}
    for case in report["cases"]:
        if case["family"] != "experts" or case["T"] not in routers:
            continue
        router = routers[case["T"]]
        row = {"T": case["T"], "scenario": case["scenario"]}
        for variant in ("before", "after"):
            if variant in case and variant in router:
                row[f"{variant}_median_us"] = (router[variant]["median_us"]
                                               + case[variant]["median_us"])
                row[f"{variant}_p90_us"] = router[variant]["p90_us"] + case[variant]["p90_us"]
        if "before_median_us" in row and "after_median_us" in row:
            row["after_over_before"] = row["after_median_us"] / row["before_median_us"]
            row["meets_0_6"] = row["after_over_before"] <= 0.6
            row["gain_us_per_layer"] = row["before_median_us"] - row["after_median_us"]
            row["bounded_e2e_ms_per_step_42_layers"] = row["gain_us_per_layer"] * 42 / 1000
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline-module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 4, 64])
    parser.add_argument("--before-tokens", type=int, nargs="+", default=[1, 4],
                        help="token counts that also time the 5938748 path")
    parser.add_argument("--families", nargs="+", default=["router", "experts"])
    parser.add_argument("--fp8-stationary", action="store_true",
                        help="also time the expert kernel with fp8 stationaries")
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--profile-iterations", type=int, default=3)
    args = parser.parse_args()
    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1":
        raise ValueError("Hardware benchmark cannot run in VLLM_NEURON_CPU_MODE=1")
    if not os.environ.get("NEURON_RT_VISIBLE_CORES"):
        raise ValueError("Run under devlease.py, which pins NEURON_RT_VISIBLE_CORES")
    if any(t not in SCENARIOS for t in args.tokens) or args.layers < 2:
        raise ValueError(f"tokens must be in {sorted(SCENARIOS)} and layers >= 2")
    baseline = load_baseline(args.baseline_module.resolve())
    args.output = args.output.resolve()
    if args.profile_dir is not None:
        args.profile_dir = args.profile_dir.resolve()
    # The compiler leaves per-kernel work directories in the current directory;
    # keep them out of the source tree.
    workdir = Path(os.environ.get("NEURON_LIBTORCH_CACHE_ROOT") or "/tmp") / "benchmark_workdir"
    workdir.mkdir(parents=True, exist_ok=True)
    os.chdir(workdir)
    lnc2 = os.environ.get("NEURON_LOGICAL_NC_CONFIG") == "2"
    report = {
        "environment": {k: os.environ.get(k) for k in (
            "NEURON_RT_VISIBLE_CORES", "NEURON_LOGICAL_NC_CONFIG", "NEURON_CC_FLAGS",
            "NEURON_PLATFORM_TARGET_OVERRIDE", "NEURON_LIBTORCH_CACHE_ROOT")},
        "baseline_module": str(args.baseline_module.resolve()),
        "method": __doc__.split("Method.")[1].strip(),
        "programs": {
            "router_after": 1,
            "experts_after": 2 if lnc2 else 1,
            "experts_before": "compact_decode_kernel / moe_fused_fp8_kernel, grid [2]",
            "router_before": "noaux_tc_rmsnorm_router_topk, both programs (128 rows each)",
        },
        "shapes": {"H": HIDDEN, "E_global": EXPERTS, "top_k": TOP_K,
                   "E_local": LOCAL_EXPERTS, "I_local": LOCAL_INTERMEDIATE, "rank": RANK},
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for tokens in args.tokens:
        before = ["before"] if tokens in args.before_tokens else []
        if "router" in args.families:
            case = router_case(tokens, before + ["after"], baseline, args)
            report["cases"].append(case)
            print(json.dumps({k: v for k, v in case.items() if k != "checks"}), flush=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        if "experts" in args.families:
            variants = before + ["after"] + (["after_fp8_stationary"] if args.fp8_stationary else [])
            report["cases"].extend(expert_cases(tokens, variants, baseline, args))
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    report["router_plus_experts_per_layer"] = summarize(report)
    report["expert_cost_per_hit_T1"] = per_hit(report)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["router_plus_experts_per_layer"], indent=2), flush=True)


if __name__ == "__main__":
    main()
