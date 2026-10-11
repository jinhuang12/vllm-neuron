# SPDX-License-Identifier: Apache-2.0
"""Time the MoE prefill router kernels against a baseline commit, and compare their outputs.

Run it through the device lease, which pins the cores and the LNC; this script does not
select cores::

    devlease.py slice NAME -- python3 test/hardware/benchmark_router_topk.py --out FILE

Kernels (``--kernels``), each launched as its seam launches it (``[2]`` programs):

* ``prefill`` -- ``router_prefill.noaux_tc_router_prefill_kernel`` on ``[T, H]`` rows already
  scaled by ``router_rms_scale``: the served prefill route.
* ``fused`` -- ``router._noaux_tc_rmsnorm_router_topk_nki`` on ``[1, T, H]`` pre-norm rows: the
  route ``noaux_tc_rmsnorm_router_topk`` takes for a call the prefill route declines.

``T`` (``--tokens``) is the padded token extent of one launch, a multiple of 256: 1024 and 2048
are the served prefill chunks (chunk 1024 on the standard lines, chunk 2048 at 64k context),
512 the 256k line's chunk, and 256 the pad of a 65..256-token call. ``H`` and ``E`` are
GLM-5.3-Flash's router extents.

Variants: ``before`` is ``--baseline-commit``'s ``router.py`` / ``router_prefill.py`` /
``router_decode.py``, read with ``git show`` into a package beside ``--records`` and imported
under its own name (the kernel cache keys a kernel by its qualified name); ``after`` is this
tree's modules.

Timing. Per-call device time is the slope between an ``R``-call graph and a one-call graph
(``R = --chain``), ``(device(R) - device(1)) / (R - 1)``, paired per iteration, with device
time from the runtime system trace (the LNC2 physical-core intervals of an execution merged).
Two ``R``-call graphs give two measures. Independent (``per_call``): every call reads its own
rows and nothing orders the calls, so one call's loads may overlap the previous call's work.
Chained (``per_call_chained``): each call's correction bias is the previous call's first logits
row, so the calls run in series and a call's whole latency counts (plus the one-row slice
that feeds the next call). Both graphs return one element of each call's logits. ``--reps``
rounds time every graph ``--iterations`` times, the graphs interleaved in a rotating order.
Reported per variant and measure: the median of each round, the median and spread over rounds,
and the noise floor (the larger relative spread ``(max - min) / median`` of the two variants'
round medians).

Numerics. For each seed (``--seeds``) and each input case, the one-call graphs of both
variants run on the same inputs; all outputs are saved under ``--records``/``numerics`` as
``.pt`` and compared with ``torch.equal``. Cases: ``random`` (synthetic weights, gain and
correction bias with the checkpoint's spread), ``checkpoint`` (``--checkpoint``'s layer
``--layer`` gate weight, correction bias and FFN gain, when given), and three tie cases that
make corrected scores equal on purpose: ``tie_pair`` (two experts with equal weight columns and
bias, both selected), ``tie_block`` (sixteen such experts, so the top-8 cut falls inside a tie)
and ``tie_flat`` (a zero bias, and every seventh row zero, so those rows tie on all experts).
Device identity: the ``after`` one-call graph runs ``--identity-runs`` more times on each case's
inputs, and every run's outputs must equal its first run's bit for bit.

Compilation and warmup are excluded. The compile cache is off unless ``--use-compile-cache``:
the graph cache key ignores kernel bodies, so an edited kernel could be timed from a stale NEFF.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

#: This file's repository. Run as a script, Python puts ``test/hardware`` on ``sys.path`` and
#: the venv's own ``vllm_neuron`` is another checkout; the "after" side must be this tree.
#: Bytecode is not written, so a run leaves no ``__pycache__`` in the tree.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import torch  # noqa: E402

import vllm_neuron  # noqa: E402,F401 -- registers the Neuron compilation backend
import nki.language as nl  # noqa: E402
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki  # noqa: E402
from vllm_neuron.functional.moe import router as current_router  # noqa: E402
from vllm_neuron.functional.moe import router_prefill as current_prefill  # noqa: E402

DEVICE = "neuron:0"
#: The modules the baseline package is built from (``router_prefill`` imports the other two).
MODULES = ("router", "router_prefill", "router_decode")
MODULE_DIR = "vllm_neuron/functional/moe"
#: GLM-5.3-Flash router extents and routing settings (the checkpoint's config.json):
#: hidden_size, n_routed_experts, rms_norm_eps, routed_scaling_factor, norm_topk_prob.
HIDDEN, EXPERTS, EPS, SCALING, NORM_TOPK = 4096, 288, 1e-5, 2.5, True
#: Spread of the synthetic operands, from the checkpoint's layer-3 router: the gate weight's
#: standard deviation and the correction bias's mean and standard deviation.
WEIGHT_STD, BIAS_MEAN, BIAS_STD = 0.031, 7.6, 0.05
#: Tie cases: experts made equal on purpose, and the bias lift that puts them in the top 8.
TIE_PAIR, TIE_BLOCK, TIE_LIFT, TIE_FLAT_ROW_STRIDE = (3, 17), range(40, 56), 1.0, 7
KERNELS = ("prefill", "fused")
CASES = ("random", "checkpoint", "tie_pair", "tie_block", "tie_flat")
VARIANTS = ("before", "after")


def load_baseline(commit: str, records: Path):
    """The baseline commit's router modules as a package beside the records."""
    name = f"_router_baseline_{commit}"
    package = records / "baseline" / name
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("")
    digests = {}
    for module in MODULES:
        source = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{commit}:{MODULE_DIR}/{module}.py"],
            check=True, capture_output=True, text=True).stdout
        (package / f"{module}.py").write_text(source)
        digests[module] = hashlib.sha256(source.encode()).hexdigest()
    sys.path.insert(0, str(package.parent))
    return (importlib.import_module(f"{name}.router"),
            importlib.import_module(f"{name}.router_prefill"), str(package), digests)


def compiled(fn):
    return torch.compile(fn, backend="neuron_libtorch", fullgraph=True, dynamic=False,
                         options={"compiler_args": os.environ.get("NEURON_CC_FLAGS", "")})


def launcher(kernel: str, router_module, prefill_module):
    """The kernel's ``[2]`` launch, as its seam makes it; returns ``(logits, index, aff, ...)``."""
    if kernel == "prefill":
        jit = wrap_nki(prefill_module.noaux_tc_router_prefill_kernel)

        def launch(rows, gamma, weights, bias):
            return jit[2](scaled=rows, gamma=gamma, router_weights=weights,
                          correction_bias=bias, norm_topk_prob=NORM_TOPK,
                          routed_scaling_factor=SCALING)
    else:
        jit = wrap_nki(router_module._noaux_tc_rmsnorm_router_topk_nki)

        def launch(rows, gamma, weights, bias):
            return jit[2](hidden_states=rows.unsqueeze(0), gamma=gamma,
                          router_weights=weights, correction_bias=bias, eps=EPS,
                          norm_topk_prob=NORM_TOPK, routed_scaling_factor=SCALING,
                          router_mm_dtype=nl.bfloat16)
    return launch


def checkpoint_router(directory: Path, layer: int):
    """``(weights [H, E] bf16, bias [1, E] fp32, gain [1, H] bf16)`` of one MoE layer."""
    from safetensors import safe_open

    index = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"model.language_model.layers.{layer}."
    names = {"weights": prefix + "mlp.gate.weight",
             "bias": prefix + "mlp.gate.e_score_correction_bias",
             "gain": prefix + "post_attention_layernorm.weight"}
    out = {}
    for key, name in names.items():
        with safe_open(directory / index[name], "pt") as handle:
            out[key] = handle.get_tensor(name)
    return (out["weights"].t().contiguous().to(torch.bfloat16),
            out["bias"].reshape(1, -1).to(torch.float32),
            out["gain"].reshape(1, -1).to(torch.bfloat16))


def operands(case: str, tokens: int, seed: int, checkpoint):
    """Pre-norm rows ``[T, H]`` bf16, gain ``[1, H]`` bf16, weights ``[H, E]`` bf16, bias ``[1, E]``."""
    gen = torch.Generator().manual_seed(seed)
    rows = torch.randn((tokens, HIDDEN), generator=gen).to(torch.bfloat16)
    if case == "checkpoint":
        weights, bias, gain = checkpoint
        return rows, gain, weights, bias
    gain = (1.0 + 0.1 * torch.randn((1, HIDDEN), generator=gen)).to(torch.bfloat16)
    weights = (WEIGHT_STD * torch.randn((HIDDEN, EXPERTS), generator=gen)).to(torch.bfloat16)
    bias = BIAS_MEAN + BIAS_STD * torch.randn((1, EXPERTS), generator=gen)
    if case == "tie_pair":
        first, second = TIE_PAIR
        weights[:, second] = weights[:, first]
        bias[0, first] += TIE_LIFT
        bias[0, second] = bias[0, first]
    elif case == "tie_block":
        lead = TIE_BLOCK[0]
        bias[0, lead] += TIE_LIFT
        for column in TIE_BLOCK:
            weights[:, column] = weights[:, lead]
            bias[0, column] = bias[0, lead]
    elif case == "tie_flat":
        bias.zero_()
        rows[::TIE_FLAT_ROW_STRIDE] = 0
    return rows, gain, weights, bias


def kernel_rows(kernel: str, rows):
    """The rows the kernel reads: scaled by this tree's ``router_rms_scale`` (``prefill``, the
    same rows for both variants) or pre-norm (``fused``)."""
    if kernel == "prefill":
        return current_prefill.router_rms_scale(rows, EPS)
    return rows


def one_call(launch):
    def graph(rows, gamma, weights, bias):
        return launch(rows, gamma, weights, bias)
    return graph


def independent_calls(launch, links: int):
    """``links`` calls on their own rows: nothing orders one call after another."""
    def graph(gamma, weights, bias, *rows):
        return torch.cat([launch(r, gamma, weights, bias)[0][0, 0:1] for r in rows[:links]])
    return graph


def dependent_calls(launch, links: int):
    """``links`` calls in series: each call's correction bias is the previous call's first
    logits row, so a call starts only after the one before it has finished."""
    def graph(gamma, weights, bias, *rows):
        firsts = []
        for r in rows[:links]:
            logits = launch(r, gamma, weights, bias)[0]
            firsts.append(logits[0, 0:1])
            bias = logits[0:1, :]
        return torch.cat(firsts)
    return graph


def device_intervals(events_json: str) -> list[float]:
    """Per-execution device time in us, in execution order; LNC2 core intervals merged."""
    starts, intervals = {}, {}
    for event in json.loads(events_json)["events"]:
        if event["event_type"] != "nc_exec_running":
            continue
        key = (event["nc_idx"], event["tracking_id"])
        if event["phase"] == "start":
            starts[key] = event
        elif event["phase"] == "stop":
            start = starts.pop(key)
            intervals.setdefault(start["data"]["exec_id"], []).append(
                (start["data"]["nc_timestamp_ns"], event["data"]["nc_timestamp_ns"]))
    return [(max(end for _, end in intervals[e]) - min(b for b, _ in intervals[e])) / 1000.0
            for e in sorted(intervals)]


def time_round(graphs: dict, inputs: dict, iterations: int) -> dict:
    """One round: every graph ``iterations`` times, in a rotating order; device us per call."""
    from nrtpy._nrtpy import SystemTraceSession

    names = list(graphs)
    order = []
    with SystemTraceSession() as trace:
        for iteration in range(iterations):
            shift = iteration % len(names)
            for name in names[shift:] + names[:shift]:
                graphs[name](*inputs[name])[0].to("cpu")
                order.append(name)
        events = trace.fetch_events_json()
    device = device_intervals(events)
    if len(device) != len(order):
        raise AssertionError(f"system trace has {len(device)} executions for {len(order)} calls")
    samples = {name: [] for name in names}
    for name, value in zip(order, device):
        samples[name].append(value)
    return samples


def spread(values: list[float]) -> dict:
    middle = statistics.median(values)
    return {"median_us": middle, "min_us": min(values), "max_us": max(values),
            "relative_spread": (max(values) - min(values)) / middle}


def numerics(graphs: dict, kernel: str, tokens: int, case: str, seed: int, args) -> dict:
    rows, gamma, weights, bias = operands(case, tokens, seed, args.checkpoint_router)
    device_inputs = [t.to(DEVICE) for t in (kernel_rows(kernel, rows), gamma, weights, bias)]
    outputs, files = {}, {}
    for variant in VARIANTS:
        outputs[variant] = [o.to("cpu") for o in graphs[f"{variant}_k1"](*device_inputs)]
        files[variant] = args.records / "numerics" / f"{kernel}_t{tokens}_{case}_seed{seed}_{variant}.pt"
        torch.save({"outputs": outputs[variant], "operands": (rows, gamma, weights, bias)},
                   files[variant])
    equal = [bool(torch.equal(a, b)) for a, b in zip(outputs["before"], outputs["after"])]
    differing = [int(torch.ne(a, b).sum()) for a, b in zip(outputs["before"], outputs["after"])]
    # Device identity: the "after" graph run again on the same inputs gives the same bits.
    repeats = [[o.to("cpu") for o in graphs["after_k1"](*device_inputs)]
               for _ in range(args.identity_runs)]
    identical = all(torch.equal(a, b) for run in repeats for a, b in zip(run, outputs["after"]))
    index = outputs["after"][1].to(torch.int64)
    distinct = bool((index.sort(dim=1).values.diff(dim=1) != 0).all())
    return {"case": case, "seed": seed, "outputs_equal": equal, "elements_differing": differing,
            "all_equal": all(equal), "after_indices_distinct_per_row": distinct,
            "identity_runs": 1 + args.identity_runs, "identity_all_equal": identical,
            "files": {variant: str(path) for variant, path in files.items()}}


def shape_case(kernel: str, tokens: int, launches: dict, args) -> dict:
    torch._dynamo.reset()
    links = args.chain
    graphs, inputs, compile_s = {}, {}, {}
    _, gamma, weights, bias = operands("random", tokens, args.timing_seed, None)
    gamma_d, weights_d, bias_d = gamma.to(DEVICE), weights.to(DEVICE), bias.to(DEVICE)
    rows = [kernel_rows(kernel, operands("random", tokens, args.timing_seed + link, None)[0])
            .to(DEVICE) for link in range(links)]
    for variant in VARIANTS:
        for key, graph, graph_inputs in (
                (f"{variant}_k1", one_call(launches[variant]), (rows[0], gamma_d, weights_d, bias_d)),
                (f"{variant}_k{links}", independent_calls(launches[variant], links),
                 (gamma_d, weights_d, bias_d, *rows)),
                (f"{variant}_d{links}", dependent_calls(launches[variant], links),
                 (gamma_d, weights_d, bias_d, *rows))):
            graphs[key] = compiled(graph)
            inputs[key] = graph_inputs
            started = time.perf_counter()
            graphs[key](*graph_inputs)[0].to("cpu")
            compile_s[key] = time.perf_counter() - started
    for _ in range(args.warmup):
        for key in graphs:
            graphs[key](*inputs[key])[0].to("cpu")
    rounds = {variant: [] for variant in VARIANTS}
    for _ in range(args.reps):
        samples = time_round(graphs, inputs, args.iterations)
        for variant in VARIANTS:
            one = samples[f"{variant}_k1"]
            pairs = [(long - first) / (links - 1)
                     for long, first in zip(samples[f"{variant}_k{links}"], one)]
            serial = [(long - first) / (links - 1)
                      for long, first in zip(samples[f"{variant}_d{links}"], one)]
            rounds[variant].append({
                "per_call_median_us": statistics.median(pairs),
                "per_call_chained_median_us": statistics.median(serial),
                "one_call_graph_median_us": statistics.median(one),
                f"k{links}_graph_median_us": statistics.median(samples[f"{variant}_k{links}"]),
                f"d{links}_graph_median_us": statistics.median(samples[f"{variant}_d{links}"]),
            })
    checks = [numerics(graphs, kernel, tokens, case, seed, args)
              for case in args.cases for seed in args.seeds]
    case = {"kernel": kernel, "tokens": tokens, "hidden": HIDDEN, "experts": EXPERTS,
            "chain_links": links, "compile_and_first_call_s": compile_s, "numerics": checks}
    for variant in VARIANTS:
        case[variant] = {
            "rounds": rounds[variant],
            "per_call": spread([r["per_call_median_us"] for r in rounds[variant]]),
            "per_call_chained": spread([r["per_call_chained_median_us"]
                                        for r in rounds[variant]]),
        }
    for measure in ("per_call", "per_call_chained"):
        suffix = "" if measure == "per_call" else "_chained"
        case["noise_floor_relative" + suffix] = max(
            case[v][measure]["relative_spread"] for v in VARIANTS)
        case["speedup" + suffix] = (case["before"][measure]["median_us"]
                                    / case["after"][measure]["median_us"])
    return case


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="the JSON report")
    parser.add_argument("--records", type=Path,
                        help="directory for the baseline package and the .pt outputs "
                             "(default: the report's directory)")
    parser.add_argument("--baseline-commit", default="ab4f37f")
    parser.add_argument("--kernels", nargs="+", choices=KERNELS, default=list(KERNELS))
    parser.add_argument("--tokens", type=int, nargs="+", default=[1024, 2048, 512, 256])
    parser.add_argument("--cases", nargs="+", choices=CASES,
                        default=["random", "tie_pair", "tie_block", "tie_flat"])
    parser.add_argument("--checkpoint", type=Path, help="checkpoint directory for the "
                        "'checkpoint' case")
    parser.add_argument("--layer", type=int, default=3, help="MoE layer of --checkpoint")
    parser.add_argument("--chain", type=int, default=4, help="calls in the long graph")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10, help="timed calls per graph per round")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    parser.add_argument("--timing-seed", type=int, default=7)
    parser.add_argument("--identity-runs", type=int, default=8,
                        help="extra runs of the 'after' one-call graph per numerics case")
    parser.add_argument("--use-compile-cache", action="store_true")
    parser.add_argument("--time-limit", type=int, default=7200, help="seconds before the run aborts")
    args = parser.parse_args()
    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1" or os.environ.get("NKI_SIMULATOR"):
        raise ValueError("Hardware benchmark cannot run in CPU mode or the NKI simulator")
    if not os.environ.get("NEURON_RT_VISIBLE_CORES"):
        raise ValueError("Run under the device lease, which pins NEURON_RT_VISIBLE_CORES")
    for module in (current_router, current_prefill):
        if not Path(module.__file__).resolve().is_relative_to(ROOT):
            raise ValueError(f"imported {module.__name__} from {module.__file__}, not {ROOT}")
    if (args.chain < 2 or args.reps < 1 or args.iterations < 1 or args.warmup < 0
            or args.identity_runs < 0):
        raise ValueError("Use --chain >= 2, positive rounds and iterations, and "
                         "non-negative warmup and identity runs")
    if any(t % current_router._NOAUX_TC_T_MULTIPLE for t in args.tokens):
        raise ValueError(f"--tokens must be multiples of {current_router._NOAUX_TC_T_MULTIPLE}")
    if "checkpoint" in args.cases and args.checkpoint is None:
        raise ValueError("the 'checkpoint' case needs --checkpoint")
    signal.alarm(args.time_limit)
    if not args.use_compile_cache:
        os.environ["NEURON_LIBTORCH_DISABLE_COMPILE_CACHE"] = "1"
    args.out = args.out.resolve()
    args.records = (args.records or args.out.parent).resolve()
    (args.records / "numerics").mkdir(parents=True, exist_ok=True)
    args.checkpoint_router = (checkpoint_router(args.checkpoint, args.layer)
                              if args.checkpoint is not None else None)
    base_router, base_prefill, base_path, base_digests = load_baseline(args.baseline_commit,
                                                                       args.records)
    routers = {"before": base_router, "after": current_router}
    prefill_modules = {"before": base_prefill, "after": current_prefill}
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", MODULE_DIR],
                           check=True, capture_output=True, text=True).stdout.strip()
    # The NKI and neuronx-cc drivers write artifacts into the working directory.
    scratch = Path(os.environ.get("NEURON_LIBTORCH_CACHE_ROOT") or args.records) / "benchmark_cwd"
    scratch.mkdir(parents=True, exist_ok=True)
    os.chdir(scratch)
    report = {
        "environment": {key: os.environ.get(key) for key in (
            "NEURON_RT_VISIBLE_CORES", "NEURON_LOGICAL_NC_CONFIG", "NEURON_CC_FLAGS",
            "NEURON_PLATFORM_TARGET_OVERRIDE", "NEURON_LIBTORCH_CACHE_ROOT",
            "NEURON_LIBTORCH_DISABLE_COMPILE_CACHE")},
        "tree": {"root": str(ROOT), "head": head, "modules_modified_vs_head": bool(dirty)},
        "baseline": {"commit": args.baseline_commit, "package": base_path,
                     "sha256": base_digests},
        "args": {key: (str(value) if isinstance(value, Path) else value)
                 for key, value in vars(args).items() if key != "checkpoint_router"},
        "method": "per-call = (R-call graph - 1-call graph) / (R - 1), paired per iteration; "
                  "R-call graph independent (per_call) or chained through the correction bias "
                  "(per_call_chained); device time from the runtime system trace, LNC2 core "
                  "intervals merged; graphs interleaved in a rotating order",
        "cases": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for kernel in args.kernels:
        launches = {v: launcher(kernel, routers[v], prefill_modules[v]) for v in VARIANTS}
        for tokens in args.tokens:
            case = shape_case(kernel, tokens, launches, args)
            report["cases"].append(case)
            args.out.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"kernel": kernel, "tokens": tokens,
                              "before_us": case["before"]["per_call"]["median_us"],
                              "after_us": case["after"]["per_call"]["median_us"],
                              "before_chained_us": case["before"]["per_call_chained"]["median_us"],
                              "after_chained_us": case["after"]["per_call_chained"]["median_us"],
                              "speedup": case["speedup"],
                              "noise_floor": case["noise_floor_relative"],
                              "bit_equal": [c["all_equal"] for c in case["numerics"]]}),
                  flush=True)


if __name__ == "__main__":
    main()
