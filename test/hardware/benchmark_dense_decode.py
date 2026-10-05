# SPDX-License-Identifier: Apache-2.0
"""Dense decode sites, 5938748 vs this tree, on one Neuron chip slice.

Run through ``devlease.py slice`` (it pins the cores). All references are on
CPU. Timing: each case compiles one graph that chains ``repeat`` instances of
the site (distinct weights; for the MLP and the norm the output of one instance
feeds the next, as in the decode step), so the runtime's fixed per-graph launch
and sync cost is amortised. The reported per-site time is
``(graph time - null graph time) / repeat``; the null graph has the same inputs
and outputs and no site in it. Median and p90 are over ``--iterations`` timed
executions after ``--warmup``. Correctness: separate one-instance graphs of the
5938748 route and of this tree's route, each compared with the CPU reference.

Cases (per-rank decode shapes at TP=64):
  shared_mlp_b1 / _b4: shared expert, H=4096, I=128 (2048/64 = 32, padded).
  dense_mlp_b1:        dense MLP (layers 0-2), H=4096, I=256 (12288/64 = 192),
                       before timed with load-time scale operands (a lower bound;
                       5938748 builds them in the forward: the non-default case
                       dense_mlp_b1_layout_in_forward times that).
  lm_head_b1:          the 154880/64 = 2420-row local shard GEMV vs the
                       replicated 154880-row GEMV. The all-gather cannot run on
                       one chip slice; its cost is stated from bytes / bandwidth.
                       The null graph for this case takes only the rows.
  norm_b1:             one RMSNorm of [1, 4096] bf16, the 5938748 torch formula
                       vs functional/norm.py's kernel (measured; the kernel is
                       not wired into the model). Non-default: norm_b4, norm_b64.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import statistics
import sys
import time

import torch
import torch.nn.functional as functional
from torch.nn.functional import silu

#: This worktree. The venv has an editable vllm_neuron that points at another
#: checkout, so the tree under test goes first on sys.path and is checked below.
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import vllm_neuron  # noqa: E402 -- registers the Neuron compilation backend

if not Path(vllm_neuron.__file__).resolve().is_relative_to(REPO):
    raise ImportError(f"vllm_neuron came from {vllm_neuron.__file__}, not {REPO}")

HIDDEN = 4096
VOCAB = 154880
TP = 64
SWIGLU_LIMIT = 10.0
EPS = 1e-5
DEVICE = "neuron:0"
#: Gather entitlement inputs for the vocab-parallel head (see lm_head case).
GATHER_BYTES_PER_TOKEN = VOCAB * 2  # bf16 logits, the dtype the sampler sees
ALL_REDUCE_16KB_US = 44.2  # measured 64-rank all-reduce, dense.md E1 / [15]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load baseline snapshot {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def compile_fn(fn):
    return torch.compile(
        fn,
        backend="neuron_libtorch",
        fullgraph=True,
        dynamic=False,
        options={"compiler_args": os.environ.get("NEURON_CC_FLAGS", "")},
    )


def first(value):
    return value[0] if isinstance(value, (tuple, list)) else value


def timed(fn, inputs, warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        first(fn(*inputs)).to("cpu")
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        first(fn(*inputs)).to("cpu")
        samples.append((time.perf_counter_ns() - started) / 1000.0)
    return samples


def pct(ordered, q):
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def summarise(samples, null_median, repeat) -> dict:
    ordered = sorted(samples)
    per_site = [(s - null_median) / repeat for s in ordered]
    return {
        "iterations": len(samples),
        "repeat_per_graph": repeat,
        "graph_median_us": statistics.median(ordered),
        "graph_p90_us": pct(ordered, 0.9),
        "per_site_median_us": statistics.median(per_site),
        "per_site_p90_us": pct(per_site, 0.9),
    }


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    actual = actual.to(torch.float32)
    expected = expected.to(torch.float32)
    residual = actual - expected
    return {
        "max_abs_diff": residual.abs().max().item(),
        "max_abs_ref": expected.abs().max().item(),
        "relative_l2": (residual.norm() / expected.norm().clamp_min(1e-30)).item(),
        "cosine": functional.cosine_similarity(
            actual.reshape(1, -1), expected.reshape(1, -1)
        ).item(),
    }


def run_case(name, *, check, timing, args, meta) -> dict:
    """``check`` = (before_fn, before_in, after_fn, after_in, reference, tol).

    ``timing`` = dict with keys before/after/null, each (fn, inputs, repeat).
    """
    torch._dynamo.reset()
    before_fn, before_in, after_fn, after_in, reference, rel_tol = check
    old = first(compile_fn(before_fn)(*before_in)).to("cpu")
    new = first(compile_fn(after_fn)(*after_in)).to("cpu")
    for label, value in (("before", old), ("after", new)):
        if not torch.isfinite(value.float()).all():
            raise AssertionError(f"{name}: {label} returned nonfinite values")
    checks = {
        "before_vs_cpu_reference": metrics(old, reference),
        "after_vs_cpu_reference": metrics(new, reference),
        "after_vs_before": metrics(new, old),
    }
    if checks["after_vs_cpu_reference"]["relative_l2"] > rel_tol:
        raise AssertionError(f"{name}: device result differs from CPU: {checks}")
    compiled = {k: (compile_fn(fn), inputs, rep) for k, (fn, inputs, rep)
                in timing.items()}
    for fn, inputs, _ in compiled.values():
        first(fn(*inputs)).to("cpu")
    null_fn, null_in, _ = compiled["null"]
    null_median = statistics.median(timed(null_fn, null_in, args.warmup,
                                          args.iterations))
    result = {"case": name, **meta, "null_graph_median_us": null_median}
    for key in ("before", "after"):
        fn, inputs, rep = compiled[key]
        result[key] = summarise(timed(fn, inputs, args.warmup, args.iterations),
                                null_median, rep)
    b, a = result["before"], result["after"]
    result["speedup_median"] = b["per_site_median_us"] / a["per_site_median_us"]
    result["after_over_before_median"] = (a["per_site_median_us"]
                                          / b["per_site_median_us"])
    result["after_over_before_p90"] = a["per_site_p90_us"] / b["per_site_p90_us"]
    result["correctness_rel_l2_bound"] = rel_tol
    result.update(checks)
    print(json.dumps(result), flush=True)
    return result


# --------------------------------------------------------------------------- #
# MLP: shared expert and dense MLP.
# --------------------------------------------------------------------------- #

def mlp_layers(inter: int, count: int, g):
    def weight(rows, cols):
        return (torch.randn((rows, cols), generator=g) * 64).clamp(-240, 240).to(
            torch.float8_e4m3fn)

    def scale(rows, cols, mag):
        return ((torch.rand((rows // 128, cols // 128), generator=g) + 0.37)
                * mag).float()

    return [(weight(HIDDEN, inter), weight(HIDDEN, inter), weight(inter, HIDDEN),
             scale(HIDDEN, inter, 3e-3), scale(HIDDEN, inter, 3e-3),
             scale(inter, HIDDEN, 1e-4)) for _ in range(count)]


def mlp_case(name, tokens, inter, prebuilt, baseline, current, args) -> dict:
    """``prebuilt``: the before route gets load-time scale operands (as the
    5938748 shared expert does); otherwise it builds the scale layout inside the
    forward (as the 5938748 dense MLP does)."""
    repeat = args.repeat
    g = torch.Generator().manual_seed(1000 + tokens + inter)
    x = torch.randn((tokens, HIDDEN), generator=g).to(torch.bfloat16)
    layers = mlp_layers(inter, repeat, g)
    oracle = current.blockwise_fp8_mm_torch_oracle
    gw, uw, dw, gs, us, ds = layers[0]
    gate = oracle(x, gw, gs).clamp(max=SWIGLU_LIMIT)
    up = oracle(x, uw, us).clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
    reference = oracle((silu(gate) * up).to(torch.bfloat16), dw, ds)

    def scale_ts(layer):
        lgw, luw, ldw, lgs, lus, lds = layer
        return [baseline.to_kernel_scale_layout(s, w.shape[0], w.shape[1])
                for w, s in ((lgw, lgs), (luw, lus), (ldw, lds))]

    dev = [[t.to(DEVICE) for t in layer] for layer in layers]
    dev_t = [[t.to(DEVICE) for t in scale_ts(layer)] for layer in layers]
    xd = x.to(DEVICE)

    def old_site(h, lgw, luw, ldw, lgs, lus, lds, gt=None, ut=None, dt=None):
        kw = (lambda t: {"prebuilt_scale_t": t}) if prebuilt else (lambda t: {})
        g_ = baseline.blockwise_fp8_mm(h, lgw, lgs, **kw(gt))
        u_ = baseline.blockwise_fp8_mm(h, luw, lus, **kw(ut))
        g_ = g_.clamp(min=None, max=SWIGLU_LIMIT)
        u_ = u_.clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
        a_ = (silu(g_) * u_).to(h.dtype)
        return baseline.blockwise_fp8_mm(a_, ldw, lds, **kw(dt))

    def new_site(h, lgw, luw, ldw, lgs, lus, lds):
        return current.blockwise_fp8_mlp(h, lgw, luw, ldw, lgs, lus, lds,
                                         swiglu_limit=SWIGLU_LIMIT)

    def before_chain(h, *t):
        w, s = t[:6 * repeat], t[6 * repeat:]
        for r in range(repeat):
            out = old_site(h, *w[6 * r:6 * r + 6], *s[3 * r:3 * r + 3])
            h = out.to(torch.bfloat16)
        return out

    def after_chain(h, *w):
        for r in range(repeat):
            out = new_site(h, *w[6 * r:6 * r + 6])
            h = out.to(torch.bfloat16)
        return out

    def null_chain(h, *t):
        for _ in range(repeat):
            out = h.to(torch.float32) * 1.0009765625
            h = out.to(torch.bfloat16)
        return out

    flat_w = [t for layer in dev for t in layer]
    flat_s = [t for ts in dev_t for t in ts]
    return run_case(
        name,
        check=(lambda h, *t: old_site(h, *t), (xd, *dev[0], *dev_t[0]),
               lambda h, *t: new_site(h, *t), (xd, *dev[0]), reference, 1e-2),
        timing={"before": (before_chain, (xd, *flat_w, *flat_s), repeat),
                "after": (after_chain, (xd, *flat_w), repeat),
                "null": (null_chain, (xd, *flat_w), repeat)},
        args=args,
        meta={"M": tokens, "H": HIDDEN, "I": inter,
              "before_route": "5938748: 3 x blockwise_fp8_mm small-M kernel + torch "
              "clamp/silu/mul/bf16 cast" + (" (load-time scale operands)" if prebuilt
                                            else " (scale layout built in forward)"),
              "after_route": "blockwise_fp8_mlp -> blockwise_fp8_mlp_small_m_kernel"},
    )


# --------------------------------------------------------------------------- #
# lm_head: local vocab-shard GEMV vs replicated GEMV.
# --------------------------------------------------------------------------- #

def lm_head_case(args) -> dict:
    g = torch.Generator().manual_seed(7)
    rows = torch.randn((1, HIDDEN), generator=g).to(torch.bfloat16)
    head = (torch.randn((VOCAB, HIDDEN), generator=g) * 0.02).to(torch.bfloat16)
    width = VOCAB // TP
    repeat = args.lm_head_shards
    reference = functional.linear(rows.float(), head[:width].float())
    # Distinct shards per instance: rank-local rows of the real head.
    shards = [head[r * width:(r + 1) * width].contiguous() for r in range(repeat)]
    rows_d = rows.to(DEVICE)
    head_d = head.to(DEVICE)
    shards_d = [s.to(DEVICE) for s in shards]

    def full(x, w):
        return functional.linear(x, w)

    def shard(x, w):
        return functional.linear(x, w)

    def shard_chain(x, *ws):
        return torch.cat([functional.linear(x, w) for w in ws], dim=-1)

    def null(x, *ws):
        return x * 1.0009765625

    case = run_case(
        "lm_head_b1",
        check=(lambda x, w: full(x, w)[:, :width], (rows_d, head_d),
               shard, (rows_d, shards_d[0]), reference, 1e-2),
        timing={"before": (full, (rows_d, head_d), 1),
                "after": (shard_chain, (rows_d, *shards_d), repeat),
                "null": (null, (rows_d,), 1)},
        args=args,
        meta={"M": 1, "H": HIDDEN, "vocab": VOCAB, "shard_rows": width,
              "before_route": "5938748: replicated [154880, 4096] bf16 GEMV",
              "after_route": "local [2420, 4096] bf16 shard GEMV (all-gather excluded)"},
    )
    gather_bytes = GATHER_BYTES_PER_TOKEN * (TP - 1) // TP
    case["all_gather_entitlement"] = {
        "bytes_received_per_rank_per_token": gather_bytes,
        "basis": "bf16 [1, 154880] logits; each rank receives 63/64 of them",
        "bandwidth_term_us_at_100GBps": gather_bytes / 100e9 * 1e6,
        "latency_term_us": ALL_REDUCE_16KB_US,
        "latency_basis": "measured 64-rank 16 KB all-reduce (dense.md), taken as "
                         "the fixed cost of one 64-rank collective",
        "estimated_us": gather_bytes / 100e9 * 1e6 + ALL_REDUCE_16KB_US,
    }
    after = case["after"]["per_site_median_us"] + case["all_gather_entitlement"][
        "estimated_us"]
    case["after_plus_gather_median_us"] = after
    case["saving_per_step_us"] = case["before"]["per_site_median_us"] - after
    print(json.dumps({"lm_head_gather": case["all_gather_entitlement"],
                      "saving_per_step_us": case["saving_per_step_us"]}), flush=True)
    return case


# --------------------------------------------------------------------------- #
# One fused RMSNorm.
# --------------------------------------------------------------------------- #

def norm_case(tokens, baseline_norm, current_norm, args) -> dict:
    repeat = args.repeat
    g = torch.Generator().manual_seed(5)
    x = (torch.randn((tokens, HIDDEN), generator=g) * 3).to(torch.bfloat16)
    gains = [(torch.rand((HIDDEN,), generator=g) + 0.5).to(torch.bfloat16)
             for _ in range(repeat)]
    reference = baseline_norm(x, gains[0], EPS)
    xd = x.to(DEVICE)
    gd = [t.to(DEVICE) for t in gains]

    def before_chain(h, *gs):
        for gain in gs:
            h = baseline_norm(h, gain, EPS)
        return h

    def after_chain(h, *gs):
        for gain in gs:
            h = current_norm.rms_norm(h, gain, EPS)
        return h

    def null(h, *gs):
        return h * 1.0009765625

    return run_case(
        f"norm_b{tokens}",
        check=(lambda h, gain: baseline_norm(h, gain, EPS), (xd, gd[0]),
               lambda h, gain: current_norm.rms_norm(h, gain, EPS), (xd, gd[0]),
               reference, 1e-2),
        timing={"before": (before_chain, (xd, *gd), repeat),
                "after": (after_chain, (xd, *gd), repeat),
                "null": (null, (xd, *gd), repeat)},
        args=args,
        meta={"M": tokens, "H": HIDDEN,
              "before_route": "5938748: torch fp32 RMSNorm (upcast, pow, mean, "
              "rsqrt, 2 mul, downcast) lowered by the compiler",
              "after_route": "vllm_neuron.functional.norm.rms_norm_kernel"},
    )


def baseline_rms_norm(hidden_states, gain, eps):
    """``Glm5NextModel._rms_norm`` / ``_input_norm`` body at 5938748."""
    x = hidden_states.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    normed = x * torch.rsqrt(variance + eps)
    normed = normed * gain.to(torch.float32)
    return normed.to(hidden_states.dtype)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-module", type=Path, required=True,
                        help="directory holding the 5938748 kernel snapshots")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+",
                        default=["shared_mlp_b1", "shared_mlp_b4", "dense_mlp_b1",
                                 "lm_head_b1", "norm_b1"])
    parser.add_argument("--repeat", type=int, default=16)
    parser.add_argument("--lm-head-shards", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--time-limit", type=int, default=570,
                        help="seconds before the run aborts (device jobs stay < 10 min)")
    args = parser.parse_args()
    signal.alarm(args.time_limit)
    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1":
        raise ValueError("Hardware benchmark cannot run in VLLM_NEURON_CPU_MODE=1")
    if not os.environ.get("NEURON_RT_VISIBLE_CORES"):
        raise ValueError("Run through devlease.py, which pins NEURON_RT_VISIBLE_CORES")
    baseline_dir = args.baseline_module.resolve()
    if baseline_dir.is_file():
        baseline_dir = baseline_dir.parent
    args.output = args.output.resolve()
    # The NKI and neuronx-cc drivers write artifacts into the working directory;
    # keep them out of the worktree.
    scratch = Path(os.environ.get("NEURON_LIBTORCH_CACHE_ROOT") or "/tmp") / "bench-cwd"
    scratch.mkdir(parents=True, exist_ok=True)
    os.chdir(scratch)
    baseline = load_module(baseline_dir / "blockwise_fp8_mm.py",
                           "_dense_5938748_blockwise_fp8_mm")
    current = importlib.import_module("vllm_neuron.functional.blockwise_fp8_mm")
    current_norm = importlib.import_module("vllm_neuron.functional.norm")

    report = {
        "environment": {key: os.environ.get(key) for key in (
            "NEURON_RT_VISIBLE_CORES", "NEURON_LOGICAL_NC_CONFIG",
            "NEURON_PLATFORM_TARGET_OVERRIDE", "NEURON_CC_FLAGS")},
        "cores": "graphs run on neuron:0 = 1 logical core (LNC2: 2 physical "
                 "cores) of the 4-core slice",
        "baseline_module": str(baseline_dir),
        "vllm_neuron": str(Path(vllm_neuron.__file__).resolve().parent),
        "timing": "per-site = (graph - null graph) / repeat; median/p90 over "
                  "iterations of host-timed graph executions incl. output sync",
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    runners = {
        "shared_mlp_b1": lambda: mlp_case("shared_mlp_b1", 1, 128, True,
                                          baseline, current, args),
        "shared_mlp_b4": lambda: mlp_case("shared_mlp_b4", 4, 128, True,
                                          baseline, current, args),
        # Conservative: the 5938748 dense MLP builds its scale layout inside the
        # forward; timing it with load-time operands gives a lower before-bound.
        "dense_mlp_b1": lambda: mlp_case("dense_mlp_b1", 1, 256, True,
                                         baseline, current, args),
        "dense_mlp_b1_layout_in_forward": lambda: mlp_case(
            "dense_mlp_b1_layout_in_forward", 1, 256, False, baseline, current, args),
        "lm_head_b1": lambda: lm_head_case(args),
        "norm_b1": lambda: norm_case(1, baseline_rms_norm, current_norm, args),
        "norm_b4": lambda: norm_case(4, baseline_rms_norm, current_norm, args),
        "norm_b64": lambda: norm_case(64, baseline_rms_norm, current_norm, args),
    }
    for name in args.cases:
        report["cases"].append(runners[name]())
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
