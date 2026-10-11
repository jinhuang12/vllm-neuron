# SPDX-License-Identifier: Apache-2.0
"""Time the vendored rotational top-k at the DSA selection shapes, against a baseline commit.

Run it through the device lease, which pins the cores and the LNC; this script does not
select cores::

    devlease.py slice NAME -- python3 test/hardware/benchmark_rotational_topk.py --out FILE

Shapes are ``[rows, cands]`` fp32 scores with ``k = index_topk // index_kpool`` (the
checkpoint's ``select_k``, 512), the calls ``dsa_topk_select`` makes:

* prefill (``--shapes``, timed and checked): ``1k`` = ``[1024, 2048]``, a 1024-token chunk at
  ``max_model_len`` 8192, and ``64k`` = ``[2048, 16384]``, a 2048-token chunk at 65536.
  ``cands`` is the pooled key count, ``max_model_len // index_kpool``.
* decode (``--decode-shapes``, checked, timed with ``--time-decode``): the decode leg's
  selection over the same two candidate widths, one request and the request counts the
  kernel serves there.

Variants:

* ``before`` -- the ``vendored_kernels/rotational_topk`` package of ``--baseline-commit``,
  read with ``git show`` into a directory beside ``--out`` and imported from there under its
  own package name (the kernel cache keys a kernel by its qualified name).
* ``after`` -- this tree's package.

Both are launched as ``dsa_topk_select`` launches them: the package's own config factories
at two programs, the kernel on the config's grid, the indices cast to int64.

Per-call device time is the slope between an ``R``-call graph and a one-call graph
(``R = --chain``; every call of the long graph reads its own scores, so no two calls are the
same computation), ``(device(R) - device(1)) / (R - 1)``, paired per iteration, with device
time from the runtime system trace (the LNC2 physical-core intervals of an execution merged).
``--reps`` rounds each time every graph ``--iterations`` times, the four graphs interleaved in
a rotating order. Reported per variant: the median of each round, the median and spread over
rounds, and the noise floor, the larger relative spread ``(max - min) / median`` of the two
variants' round medians.

Numerics, at every shape: for each of ``--seeds`` seeds and each score pattern the one-call
graphs' outputs are saved under ``--records``/``numerics`` as ``.pt`` files and compared bit
for bit (values as their int32 bit patterns, so ``-0.0`` and ``+0.0`` differ). The patterns:

* ``normal`` -- standard normal scores, ties only by chance;
* ``causal`` -- normal scores with each row's columns past its visible pool count set to the
  causal bound's fill, as ``dsa_causal_bound`` writes them: every masked column ties;
* ``coarse`` -- integers in ``[-8, 8]`` with signed zeros, so every value ties.

The values are also compared with ``torch.topk``'s (the set and order of values are exact;
the indices of tied values are the kernel's own and are not compared with torch's).
Identity: the ``after`` one-call graph runs ``--identity-runs`` more times on the ``causal``
scores of the first seed and every output must be bit-identical to the first.

Compilation and warmup are excluded. The compile cache is off unless ``--use-compile-cache``:
the graph cache key ignores kernel bodies, so an edited kernel could be timed from a stale NEFF.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

#: This file's repository. Run as a script, Python puts ``test/hardware`` on ``sys.path`` and
#: the venv's own ``vllm_neuron`` is another checkout; the "after" side must be this tree.
#: Bytecode is not written, so a run leaves no ``__pycache__`` in the tree.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import torch  # noqa: E402

import nki.language as nl  # noqa: E402
from libtorch_neuronx_lite.nki.nki_dtype import torch_to_nki_dtype  # noqa: E402
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki  # noqa: E402

import vllm_neuron  # noqa: E402,F401 -- registers the Neuron compilation backend
from vllm_neuron.functional.dsa import topk_select  # noqa: E402
from vllm_neuron.functional.dsa.causal_bound import BOUND_FILL  # noqa: E402
from vllm_neuron.functional.vendored_kernels import rotational_topk as current  # noqa: E402
from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig  # noqa: E402

DEVICE = "neuron:0"
PACKAGE_PATH = "vllm_neuron/functional/vendored_kernels/rotational_topk"
PACKAGE_FILES = ("__init__.py", "rotational_topk.py", "rotational_topk_utils.py",
                 "cascaded_max_utils.py")
_TEXT_CONFIG = Glm5NextTextConfig()
#: The checkpoint's ``select_k``: ``index_topk`` tokens in pools of ``index_kpool``.
SELECT_K = _TEXT_CONFIG.index_topk // _TEXT_CONFIG.index_kpool
#: ``[rows, cands]`` of the served prefill calls (reports/op_inventory.json, family
#: "DSA attention | rotational_topk #0": a 1024-token chunk at max_model_len 8192, and a
#: 2048-token chunk of a 65534-token prompt at max_model_len 65536).
SHAPES = {"1k": (1024, 2048), "64k": (2048, 16384)}
#: ``[requests, cands]`` of the decode leg's selection at the same two widths.
DECODE_SHAPES = {"d1x2048": (1, 2048), "d8x2048": (8, 2048), "d1x16384": (1, 16384),
                 "d8x16384": (8, 16384), "d32x16384": (32, 16384), "d64x8192": (64, 8192),
                 "d64x16384": (64, 16384)}
PATTERNS = ("normal", "causal", "coarse")
VARIANTS = ("before", "after")


def load_baseline(commit: str, records: Path):
    """The baseline commit's package, written beside the records and imported from there."""
    directory = records / f"baseline_rotational_topk_{commit}"
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    for name in PACKAGE_FILES:
        source = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{PACKAGE_PATH}/{name}"],
                                check=True, capture_output=True, text=True).stdout
        (directory / name).write_text(source)
        digest.update(source.encode())
    package = f"_rotational_topk_baseline_{commit}"
    spec = importlib.util.spec_from_file_location(
        package, directory / "__init__.py", submodule_search_locations=[str(directory)])
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load the baseline package {directory}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[package] = module
    spec.loader.exec_module(module)
    return module, directory, digest.hexdigest()


def seam(package):
    """``dsa_topk_select``'s launch of ``package``'s kernel: ``(values, int64 indices)``."""

    @torch._dynamo.assume_constant_result
    def config_of(n_rows: int, width: int, k: int, nki_dtype):
        topk_config = package.create_topk_config(inp_shape=(n_rows, width), inp_dtype=nki_dtype,
                                                 k=k, num_programs=topk_select._NUM_PROGRAMS)
        return package.create_rotational_topk_config(inp_shape=(n_rows, width),
                                                     topk_config=topk_config)

    def select(scores):
        n_rows, width = (int(d) for d in scores.shape)
        config = config_of(n_rows, width, SELECT_K, getattr(nl, torch_to_nki_dtype(scores.dtype)))
        values, indices = wrap_nki(package.rotational_topk)[config.n_prgs](scores, config)
        return values, indices.to(torch.int64)

    return select


def compiled(fn):
    return torch.compile(fn, backend="neuron_libtorch", fullgraph=True, dynamic=False,
                         options={"compiler_args": os.environ.get("NEURON_CC_FLAGS", "")})


def scores_of(rows: int, cands: int, seed: int, pattern: str) -> torch.Tensor:
    """``[rows, cands]`` fp32 scores of one pattern (see the module docstring)."""
    gen = torch.Generator().manual_seed(seed)
    if pattern == "coarse":
        values = torch.randint(-8, 9, (rows, cands), generator=gen).to(torch.float32)
        negative_zero = (values == 0) & (torch.rand((rows, cands), generator=gen) < 0.5)
        return torch.where(negative_zero, torch.tensor(-0.0), values)
    scores = torch.randn((rows, cands), generator=gen, dtype=torch.float32)
    if pattern == "causal":
        # Row r of a chunk that starts at a random position sees (position + 1) // pool
        # complete pools, at most every candidate; the rest carry the causal bound's fill.
        # The start is drawn below 2 * k pools, so that rows on both sides of k visible
        # pools occur and the fill ties inside the selection.
        pool = _TEXT_CONFIG.index_kpool
        latest = max(1, min(cands * pool - rows, 2 * SELECT_K * pool))
        start = int(torch.randint(0, latest, (1,), generator=gen))
        visible = ((torch.arange(rows) + start + 1) // pool).clamp(max=cands)
        masked = torch.arange(cands).unsqueeze(0) >= visible.unsqueeze(1)
        scores = torch.where(masked, torch.tensor(BOUND_FILL, dtype=torch.float32), scores)
    return scores


def one_call(select):
    def graph(scores):
        return select(scores)
    return graph


def chained_calls(select, links: int):
    def graph(*scores):
        return torch.cat([select(s)[0][0, 0:1] for s in scores[:links]])
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
                out = graphs[name](*inputs[name])
                (out[0] if isinstance(out, tuple) else out).to("cpu")
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


def bits(values: torch.Tensor) -> torch.Tensor:
    return values.contiguous().view(torch.int32)


def outputs_of(graph, scores_device) -> tuple[torch.Tensor, torch.Tensor]:
    values, indices = graph(scores_device)
    return values.to("cpu"), indices.to("cpu")


def numerics(graphs: dict, rows: int, cands: int, seed: int, pattern: str, name: str,
             records: Path) -> dict:
    scores = scores_of(rows, cands, seed, pattern)
    scores_device = scores.to(DEVICE)
    outputs, files = {}, {}
    for variant in VARIANTS:
        outputs[variant] = outputs_of(graphs[f"{variant}_k1"], scores_device)
        files[variant] = records / f"{name}_{pattern}_seed{seed}_{variant}.pt"
        torch.save({"values": outputs[variant][0], "indices": outputs[variant][1]}, files[variant])
    (bv, bi), (av, ai) = outputs["before"], outputs["after"]
    reference = torch.topk(scores, SELECT_K, dim=-1)
    return {
        "seed": seed,
        "pattern": pattern,
        "values_bit_equal": bool(torch.equal(bits(av), bits(bv))),
        "indices_equal": bool(torch.equal(ai, bi)),
        "values_differing": int(torch.ne(bits(av), bits(bv)).sum()),
        "indices_differing": int(torch.ne(ai, bi).sum()),
        "after_values_equal_torch": bool(torch.equal(av, reference.values)),
        "after_indices_select_their_values": bool(torch.equal(scores.gather(1, ai), av)),
        "files": {variant: str(path) for variant, path in files.items()},
    }


def identity(graph, rows: int, cands: int, seed: int, runs: int) -> dict:
    """``runs`` emissions of the one-call graph on one input, each against the first."""
    scores_device = scores_of(rows, cands, seed, "causal").to(DEVICE)
    first_values, first_indices = outputs_of(graph, scores_device)
    same = []
    for _ in range(runs):
        values, indices = outputs_of(graph, scores_device)
        same.append(torch.equal(bits(values), bits(first_values))
                    and torch.equal(indices, first_indices))
    return {"pattern": "causal", "seed": seed, "emissions": runs, "bit_identical": all(same),
            "identical_count": sum(same)}


def shape_case(name: str, rows: int, cands: int, seams: dict, args, timed: bool) -> dict:
    torch._dynamo.reset()
    links = args.chain
    graphs, inputs, compile_s = {}, {}, {}
    timing_scores = [scores_of(rows, cands, args.timing_seed + link, "normal").to(DEVICE)
                     for link in range(links)]
    for variant in VARIANTS:
        plan = [(f"{variant}_k1", one_call(seams[variant]), (timing_scores[0],))]
        if timed:
            plan.append((f"{variant}_k{links}", chained_calls(seams[variant], links),
                         tuple(timing_scores)))
        for key, graph, graph_inputs in plan:
            graphs[key] = compiled(graph)
            inputs[key] = graph_inputs
            started = time.perf_counter()
            out = graphs[key](*graph_inputs)
            (out[0] if isinstance(out, tuple) else out).to("cpu")
            compile_s[key] = time.perf_counter() - started
    case = {"shape": name, "rows": rows, "cands": cands, "k": SELECT_K, "chain_links": links,
            "compile_and_first_call_s": compile_s,
            "bytes_per_call": rows * cands * 4 + rows * SELECT_K * (4 + 4)}
    if timed:
        for _ in range(args.warmup):
            for key in graphs:
                out = graphs[key](*inputs[key])
                (out[0] if isinstance(out, tuple) else out).to("cpu")
        rounds = {variant: [] for variant in VARIANTS}
        for _ in range(args.reps):
            samples = time_round(graphs, inputs, args.iterations)
            for variant in VARIANTS:
                pairs = [(long - one) / (links - 1) for long, one in
                         zip(samples[f"{variant}_k{links}"], samples[f"{variant}_k1"])]
                rounds[variant].append({
                    "per_call_median_us": statistics.median(pairs),
                    "one_call_graph_median_us": statistics.median(samples[f"{variant}_k1"]),
                    f"k{links}_graph_median_us": statistics.median(samples[f"{variant}_k{links}"]),
                })
        for variant in VARIANTS:
            per_round = [r["per_call_median_us"] for r in rounds[variant]]
            case[variant] = {"rounds": rounds[variant], "per_call": spread(per_round)}
        case["noise_floor_relative"] = max(case[v]["per_call"]["relative_spread"] for v in VARIANTS)
        case["speedup"] = (case["before"]["per_call"]["median_us"]
                           / case["after"]["per_call"]["median_us"])
    records = args.records / "numerics"
    records.mkdir(parents=True, exist_ok=True)
    case["numerics"] = [numerics(graphs, rows, cands, seed, pattern, name, records)
                        for seed in args.seeds for pattern in PATTERNS]
    case["identity"] = identity(graphs["after_k1"], rows, cands, args.seeds[0], args.identity_runs)
    if args.profile_dir is not None and timed:
        case["profile_dir"] = capture_profile(graphs, inputs, [f"{v}_k1" for v in VARIANTS],
                                              args.profile_dir / name)
    return case


def capture_profile(graphs: dict, inputs: dict, names, directory: Path) -> dict:
    """One device profile per graph, each of one execution, in ``directory / name``.

    ``neuron-explorer view`` reads a profile against the NEFF of its first execution, so
    every graph gets a session of its own.
    """
    import libtorch_neuronx_lite.envs as libtorch_envs

    runtime = torch.classes.neuron.Runtime()
    out = {}
    for name in names:
        target = directory / name
        target.mkdir(parents=True, exist_ok=True)
        runtime.start_profiling(str(target), ["device_profile", "system_profile"], None, None,
                                libtorch_envs.get_neuron_compile_cache_dir())
        try:
            graphs[name](*inputs[name])[0].to("cpu")
        finally:
            runtime.stop_profiling()
        out[name] = str(target)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="the JSON report")
    parser.add_argument("--records", type=Path,
                        help="directory for the baseline source and the .pt outputs "
                             "(default: the report's directory)")
    parser.add_argument("--baseline-commit", default="ab4f37f")
    parser.add_argument("--shapes", nargs="*", choices=sorted(SHAPES), default=list(SHAPES))
    parser.add_argument("--decode-shapes", nargs="*", choices=sorted(DECODE_SHAPES),
                        default=list(DECODE_SHAPES))
    parser.add_argument("--time-decode", action="store_true", help="also time the decode shapes")
    parser.add_argument("--chain", type=int, default=3, help="calls in the long graph")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10, help="timed calls per graph per round")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    parser.add_argument("--timing-seed", type=int, default=7)
    parser.add_argument("--identity-runs", type=int, default=8)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--use-compile-cache", action="store_true")
    parser.add_argument("--time-limit", type=int, default=7200, help="seconds before the run aborts")
    args = parser.parse_args()
    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1" or os.environ.get("NKI_SIMULATOR"):
        raise ValueError("Hardware benchmark cannot run in CPU mode or the NKI simulator")
    if not os.environ.get("NEURON_RT_VISIBLE_CORES"):
        raise ValueError("Run under the device lease, which pins NEURON_RT_VISIBLE_CORES")
    if not Path(current.__file__).resolve().is_relative_to(ROOT):
        raise ValueError(f"imported {current.__name__} from {current.__file__}, not {ROOT}")
    if args.chain < 2 or args.reps < 1 or args.iterations < 1 or args.warmup < 0:
        raise ValueError("Use --chain >= 2 and positive rounds and iterations")
    signal.alarm(args.time_limit)
    if not args.use_compile_cache:
        os.environ["NEURON_LIBTORCH_DISABLE_COMPILE_CACHE"] = "1"
    args.out = args.out.resolve()
    args.records = (args.records or args.out.parent).resolve()
    args.records.mkdir(parents=True, exist_ok=True)
    if args.profile_dir is not None:
        args.profile_dir = args.profile_dir.resolve()
    baseline, baseline_dir, baseline_sha = load_baseline(args.baseline_commit, args.records)
    seams = {"before": seam(baseline), "after": seam(current)}
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", PACKAGE_PATH],
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
        "tree": {"root": str(ROOT), "head": head, "package_modified_vs_head": bool(dirty),
                 "package": current.__file__},
        "baseline": {"commit": args.baseline_commit, "directory": str(baseline_dir),
                     "sha256": baseline_sha},
        "args": {key: (str(value) if isinstance(value, Path) else value)
                 for key, value in vars(args).items()},
        "method": "per-call = (R-call graph - 1-call graph) / (R - 1), paired per iteration; "
                  "device time from the runtime system trace, LNC2 core intervals merged; "
                  "graphs interleaved in a rotating order",
        "select_k": SELECT_K,
        "cases": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plan = ([(name, *SHAPES[name], True) for name in args.shapes]
            + [(name, *DECODE_SHAPES[name], args.time_decode) for name in args.decode_shapes])
    for name, rows, cands, timed in plan:
        case = shape_case(name, rows, cands, seams, args, timed)
        report["cases"].append(case)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        summary = {"shape": name,
                   "bit_equal": [c["values_bit_equal"] and c["indices_equal"]
                                 for c in case["numerics"]],
                   "identity": case["identity"]["bit_identical"]}
        if timed:
            summary.update({"before_us": case["before"]["per_call"]["median_us"],
                            "after_us": case["after"]["per_call"]["median_us"],
                            "speedup": case["speedup"], "noise_floor": case["noise_floor_relative"]})
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
