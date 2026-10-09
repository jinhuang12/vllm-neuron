# SPDX-License-Identifier: Apache-2.0
"""Time the MLA low-precision projections and their folded glue at the served shapes,
against a baseline commit.

Run it through the device lease, which pins the cores and the LNC; this script does not
select cores::

    devlease.py slice NAME -- python3 test/hardware/benchmark_mla_projection.py --out FILE

Projection cases (``--sites``, ``--rows``): one TP=64 rank's projections of a DSA layer,
``SITE@ROWS``. ``SITES`` holds each site's ``(x dtype, in_features, out_features, weight
dtype)`` as the model calls it (``model_fp8.py``: ``project_query_and_latent``,
``project_output``, the indexer's ``project_stage``); the indexer's
``index_kpool_compress_gate`` has ``wk``'s shape and dtypes, so ``wk`` stands for both.
``--rows`` are prefill chunk rows and are timed; ``--decode-rows`` are decode batch rows
and are checked for numerics only.

Glue cases (``--glue``), the traced torch each fold replaces, at ``--rows``:

* ``kv_latent`` -- the KV latent of ``Glm5NextMLAAttention.attend`` on a padded prefill
  chunk (its last ``rows // 8`` rows clamped): before, the baseline projection,
  ``_latent_norm``'s fp32 steps and the cast to bf16, traced; after,
  ``mla_latent_projection``. Both end in ``attend``'s clamp-gather, which stays traced.
* ``input_norm`` -- the DSA layer's input RMSNorm at its mHC site: before, the fused
  ``mhc_pre`` without the norm and ``Glm5NextDSALayer._input_norm``'s steps, traced;
  after, ``mhc_pre`` with the norm folded (``functional/glue/mhc_pre.py``). Only at the
  row counts ``VLLM_NEURON_GLUE_FUSED`` selects ``mhc_pre`` for (1024 by default); other
  rows are skipped.

Variants: ``before`` is ``--baseline-commit``'s ``mla_projections.py``, read with
``git show`` into a file beside ``--out`` and imported from there under its own module
name (the kernel cache keys a kernel by its qualified name); ``after`` is this tree.

Per-call device time is the slope between an ``R``-call graph and a one-call graph
(``R = --chain``; every call of the long graph reads its own input and returns its whole
output), ``(device(R) - device(1)) / (R - 1)``, paired per iteration, with device time from the runtime system
trace (the LNC2 physical-core intervals of an execution merged). ``--reps`` rounds each time
every graph ``--iterations`` times, the four graphs of a case interleaved in a rotating
order. Reported per variant: the median of each round, the median and spread over rounds,
and the noise floor, the larger relative spread ``(max - min) / median`` of the two
variants' round medians.

Numerics: for each of ``--seeds`` seeds the one-call outputs of both variants are saved
under ``--records``/``numerics`` as ``.pt`` files and compared with ``torch.equal``. A
projection's ``after`` is also compared with ``mla_projection_lowp_torch_oracle``. A glue
case's two outputs are also compared, by bf16 ulps, with the correctly rounded value: the
same norm in float64 of the same fp32 (``kv_latent``: the baseline projection's) or bf16
(``input_norm``: the collapse ``mhc_pre`` returns) input, rounded once. Device identity:
``after`` runs ``--emissions`` times on each seed's input and every output is bit-compared
with the first. An fp8 weight holds
every byte the weight loader can store -- exponent field below 15, because the loader
halves the checkpoint's e4m3fn bytes into legacy e4m3's finite range
(``weight_loaders_fp8.py`` ``_FP8_WEIGHT_DOWNSCALE``) -- and its scales are arbitrary fp32.

Compilation and warmup are excluded. The compile cache is off unless ``--use-compile-cache``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
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

import torch

import vllm_neuron  # noqa: F401 -- registers the Neuron compilation backend
from vllm_neuron.functional.attention import mla_projections as current
from vllm_neuron.functional.glue import glue_selected
from vllm_neuron.functional.glue import mhc_pre as glue_mhc_pre
from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

DEVICE = "neuron:0"
MODULE_PATH = "vllm_neuron/functional/attention/mla_projections.py"
FP8 = torch.float8_e4m3fn
#: ``(x dtype, in_features, out_features, weight dtype)`` of one TP=64 rank's DSA-layer
#: projections (reports/op_inventory.json families "DSA attention |
#: mla_projection_lowp_kernel"; x is the hidden state, the fp32 query latent, or the bf16
#: attention output).
SITES = {
    "q_a_proj": (torch.bfloat16, 4096, 1536, FP8),
    "kv_a_proj_with_mqa": (torch.bfloat16, 4096, 512, FP8),
    "q_b_proj": (torch.float32, 1536, 256, FP8),
    "o_proj": (torch.bfloat16, 256, 4096, FP8),
    "wq_b": (torch.float32, 1536, 4096, torch.bfloat16),
    "wk": (torch.bfloat16, 4096, 128, torch.bfloat16),
    "weights_proj": (torch.bfloat16, 4096, 32, torch.bfloat16),
}
#: The folded glue, by case name (see the module docstring).
GLUE = ("kv_latent", "input_norm")
#: The mHC site's streams (``hc_mult``) and its post-gate multiplier
#: (``Glm5NextHyperConnection``'s default).
MHC_STREAMS = 4
MHC_POST_MULT = 2.0
#: Prefill chunk rows: a 1024-token chunk, and the 2048-token chunk of a long prompt.
PREFILL_ROWS = (1024, 2048)
#: Decode batch rows of the served buckets, and the 128-row prefill bucket.
DECODE_ROWS = (1, 2, 4, 8, 16, 32, 64, 128)
VARIANTS = ("before", "after")
#: bf16 has 8 significant bits: one ulp of a value in ``[2^e, 2^(e+1))`` is ``2^(e-7)``.
BF16_ULP_SHIFT = 7


def load_baseline(commit: str, records: Path):
    """The baseline commit's module, written beside the records and imported from there."""
    source = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{MODULE_PATH}"],
                            check=True, capture_output=True, text=True).stdout
    path = records / "baseline_mla_projections.py"
    path.write_text(source)
    name = f"_mla_projections_baseline_{commit}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load the baseline module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, path, hashlib.sha256(source.encode()).hexdigest()


def compiled(fn):
    return torch.compile(fn, backend="neuron_libtorch", fullgraph=True, dynamic=False,
                         options={"compiler_args": os.environ.get("NEURON_CC_FLAGS", "")})


def storable_fp8(shape, gen) -> torch.Tensor:
    """Uniform over every e4m3fn byte the weight loader can store (exponent field < 15)."""
    codes = torch.tensor([b for b in range(256) if (b >> 3) & 0xF != 0xF], dtype=torch.uint8)
    picks = torch.randint(0, codes.numel(), (shape[0] * shape[1],), generator=gen)
    return codes[picks].reshape(shape).view(FP8)


def prepared_weight(site: str, gen):
    """One prepared ``(weight, scale)`` of ``site``; host tensors."""
    _, idim, odim, wdt = SITES[site]
    if wdt == FP8:
        weight = storable_fp8((odim, idim), gen)
        # Arbitrary fp32 multipliers of the checkpoint's magnitude, 2^-14 .. 2^-7.
        scale = torch.exp2(torch.rand((odim // 128, idim // 128), generator=gen) * 7 - 14)
    else:
        weight = (torch.randn((odim, idim), generator=gen) * idim ** -0.5).to(torch.bfloat16)
        scale = None
    return current.prepare_lowp_projection(weight, scale)


def rms_norm_reference(x: torch.Tensor, gain: torch.Tensor, eps: float,
                       dtype: torch.dtype) -> torch.Tensor:
    """``x * rsqrt(mean(x**2) + eps) * gain`` in float64, rounded to ``dtype`` once."""
    x64 = x.double()
    rstd = torch.rsqrt(x64.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x64 * rstd * gain.double()).to(dtype)


class Case:
    """One timed shape: per variant a function ``f(x, *consts) -> (output, *aux)``.

    ``output`` is what both variants must agree on; ``aux`` is what the numerics check
    needs beside it. ``inputs(seed, xs)`` gives ``xs`` host inputs and the host constants.
    """

    def __init__(self, name: str, rows: int, fns: dict, inputs, check, meta: dict):
        self.name, self.rows, self.fns, self.inputs, self.check = name, rows, fns, inputs, check
        self.meta = meta


def projection_case(site: str, rows: int, seams: dict) -> Case:
    xdt, idim, odim, wdt = SITES[site]

    def inputs(seed: int, xs: int):
        gen = torch.Generator().manual_seed(seed)
        x = [torch.randn((rows, idim), generator=gen).to(xdt) for _ in range(xs)]
        return x, prepared_weight(site, gen)

    def fn(seam):
        return lambda x, weight, scale: (seam(x, weight, scale),)

    def check(outputs, x, consts):
        reference = current.mla_projection_lowp_torch_oracle(x, *consts)
        residual = outputs["after"][0].double() - reference.double()
        return {"after_finite": bool(torch.isfinite(outputs["after"][0]).all()),
                "after_vs_oracle_relative_l2": float(residual.norm()
                                                     / reference.double().norm()),
                "after_vs_oracle_max_abs": float(residual.abs().max())}

    programs = current.lowp_programs(odim, wdt == FP8)
    row_bytes = idim * torch.empty((), dtype=xdt).element_size()
    meta = {"site": site, "in_features": idim, "out_features": odim, "x_dtype": str(xdt),
            "weight_dtype": str(wdt), "after_programs": programs,
            "after_splits_rows": current.lowp_splits_rows(rows, programs, row_bytes)}
    return Case(f"{site}_rows{rows}", rows, {v: fn(seams[v]) for v in VARIANTS}, inputs,
                check, meta)


def ulp_report(output: torch.Tensor, reference: torch.Tensor) -> dict:
    """Elements of a bf16 ``output`` that differ from ``reference``, and by how many ulps."""
    diff = (output.double() - reference.double()).abs()
    exponent = torch.floor(torch.log2(reference.double().abs().clamp_min(2.0 ** -126)))
    ulps = diff / torch.exp2(exponent - BF16_ULP_SHIFT)
    return {"elements_differing": int(torch.ne(output, reference).sum()),
            "max_ulps": float(ulps.max())}


def kv_latent_case(rows: int, baseline, eps: float) -> Case:
    """``_kv_latent`` of a chunk whose last ``rows // 8`` rows are padding."""
    site = "kv_a_proj_with_mqa"
    xdt, idim, odim, _ = SITES[site]
    last = rows - 1 - rows // 8

    def inputs(seed: int, xs: int):
        gen = torch.Generator().manual_seed(seed)
        x = [torch.randn((rows, idim), generator=gen).to(xdt) for _ in range(xs)]
        weight, scale = prepared_weight(site, gen)
        gain = (torch.rand((odim,), generator=gen) + 0.5).to(torch.bfloat16)
        return x, (weight, scale, gain, torch.tensor(last, dtype=torch.int64))

    def clamp(latent, last_row):
        # attend()'s clamp of a padded chunk.
        offsets = torch.minimum(torch.arange(rows, device=latent.device), last_row)
        return latent.index_select(0, offsets)

    def before(x, weight, scale, gain, last_row):
        # Glm5NextMLAAttention._latent_norm and the cast, at ab4f37fc.
        projected = baseline.mla_projection_lowp(x, weight, scale)
        variance = projected.pow(2).mean(dim=-1, keepdim=True)
        normed = projected * torch.rsqrt(variance + eps)
        normed = (normed * gain.to(torch.float32)).to(x.dtype)
        return clamp(normed, last_row), projected

    def after(x, weight, scale, gain, last_row):
        return (clamp(current.mla_latent_projection(x, weight, scale, gain, eps), last_row),)

    def check(outputs, x, consts):
        projected = outputs["before"][1]
        reference = rms_norm_reference(projected, consts[2], eps, xdt)
        reference = reference.index_select(0, torch.clamp(torch.arange(rows), max=last))
        return {variant: ulp_report(outputs[variant][0], reference) | {"vs": "rounded"}
                for variant in VARIANTS} | {"after_vs_before": ulp_report(
                    outputs["after"][0], outputs["before"][0])}

    meta = {"glue": "kv_latent", "in_features": idim, "out_features": odim,
            "last_row": last, "eps": eps}
    return Case(f"kv_latent_rows{rows}", rows, {"before": before, "after": after}, inputs,
                check, meta)


def input_norm_case(rows: int, eps: float, hc_eps: float) -> Case:
    """The DSA layer's input norm at its mHC site, beside the fused ``mhc_pre``."""
    hidden = SITES["q_a_proj"][1]
    streams = MHC_STREAMS
    mix = 2 * streams + streams * streams

    def inputs(seed: int, xs: int):
        gen = torch.Generator().manual_seed(seed)
        residual = [torch.randn((rows, streams, hidden), generator=gen).to(torch.bfloat16)
                    for _ in range(xs)]
        fn = torch.randn((mix, streams * hidden), generator=gen) * (streams * hidden) ** -0.5
        hc_scale = torch.rand((3,), generator=gen) + 0.5
        hc_base = torch.randn((mix,), generator=gen) * 0.1
        gain = (torch.rand((hidden,), generator=gen) + 0.5).to(torch.bfloat16)
        return residual, (fn, hc_scale, hc_base, gain)

    def site(residual, fn, hc_scale, hc_base, gain=None):
        return glue_mhc_pre.mhc_pre_fused(residual, fn, hc_scale, hc_base, rms_eps=eps,
                                          hc_eps=hc_eps, post_mult=MHC_POST_MULT,
                                          norm_gain=gain, norm_eps=eps)

    def before(residual, fn, hc_scale, hc_base, gain):
        # Glm5NextDSALayer._input_norm at ab4f37fc, on the collapse mhc_pre returns.
        _, _, layer_input, _ = site(residual, fn, hc_scale, hc_base)
        x = layer_input.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        normed = x * torch.rsqrt(variance + eps)
        normed = normed * gain.to(torch.float32)
        return normed.to(layer_input.dtype), layer_input

    def after(residual, fn, hc_scale, hc_base, gain):
        _, _, layer_input, normed = site(residual, fn, hc_scale, hc_base, gain)
        return normed, layer_input

    def check(outputs, x, consts):
        reference = rms_norm_reference(outputs["after"][1], consts[3], eps, torch.bfloat16)
        return {variant: ulp_report(outputs[variant][0], reference) | {"vs": "rounded"}
                for variant in VARIANTS} | {
                    "after_vs_before": ulp_report(outputs["after"][0], outputs["before"][0]),
                    "layer_input_equal": bool(torch.equal(outputs["after"][1],
                                                          outputs["before"][1]))}

    meta = {"glue": "input_norm", "streams": streams, "hidden": hidden, "eps": eps}
    return Case(f"input_norm_rows{rows}", rows, {"before": before, "after": after}, inputs,
                check, meta)


def one_call(fn):
    def graph(x, *consts):
        return fn(x, *consts)
    return graph


def chained_calls(fn, links: int, n_consts: int):
    """``links`` calls, every output returned whole: a sliced output would let the
    compiler compute only the slice of a traced-torch variant."""
    def graph(*args):
        consts, xs = args[:n_consts], args[n_consts:]
        return tuple(fn(x, *consts)[0] for x in xs[:links])
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
                to_host(graphs[name](*inputs[name]))
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


def to_device(value):
    return None if value is None else value.to(DEVICE)


def to_host(value):
    if isinstance(value, (tuple, list)):
        return tuple(to_host(v) for v in value)
    return value.to("cpu")


def numerics(graphs: dict, case: Case, seed: int, records: Path, emissions: int) -> dict:
    """One seed: both variants' one-call outputs, saved and compared, and ``after``
    emitted ``emissions`` times on the same input, every emission bit-compared."""
    (x,), consts = case.inputs(seed, 1)
    device_inputs = (x.to(DEVICE), *(to_device(c) for c in consts))
    outputs, files = {}, {}
    for variant in VARIANTS:
        outputs[variant] = to_host(graphs[variant](*device_inputs))
        files[variant] = records / f"{case.name}_seed{seed}_{variant}.pt"
        torch.save(outputs[variant][0], files[variant])
    repeats = [to_host(graphs["after"](*device_inputs))[0] for _ in range(emissions - 1)]
    return {
        "seed": seed,
        "after_equals_before": bool(torch.equal(outputs["after"][0], outputs["before"][0])),
        "elements_differing": int(torch.ne(outputs["after"][0], outputs["before"][0]).sum()),
        **case.check(outputs, x, consts),
        "after_emissions": emissions,
        "after_emissions_identical": all(torch.equal(r, outputs["after"][0]) for r in repeats),
        "files": {variant: str(path) for variant, path in files.items()},
    }


def timed_case(case: Case, args) -> tuple[dict, dict]:
    torch._dynamo.reset()
    links = args.chain
    xs, consts = case.inputs(args.timing_seed, links)
    xd = [x.to(DEVICE) for x in xs]
    cd = tuple(to_device(c) for c in consts)
    graphs, inputs, compile_s = {}, {}, {}
    for variant in VARIANTS:
        fn = case.fns[variant]
        for key, graph, graph_inputs in (
                (f"{variant}_k1", one_call(fn), (xd[0], *cd)),
                (f"{variant}_k{links}", chained_calls(fn, links, len(cd)), (*cd, *xd))):
            graphs[key] = compiled(graph)
            inputs[key] = graph_inputs
            started = time.perf_counter()
            to_host(graphs[key](*graph_inputs))
            compile_s[key] = time.perf_counter() - started
    for _ in range(args.warmup):
        for key, graph in graphs.items():
            to_host(graph(*inputs[key]))
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
    result = {"case": case.name, "rows": case.rows, **case.meta, "chain_links": links,
              "compile_and_first_call_s": compile_s}
    for variant in VARIANTS:
        per_round = [r["per_call_median_us"] for r in rounds[variant]]
        result[variant] = {"rounds": rounds[variant], "per_call": spread(per_round)}
    result["noise_floor_relative"] = max(result[v]["per_call"]["relative_spread"]
                                         for v in VARIANTS)
    result["speedup"] = (result["before"]["per_call"]["median_us"]
                         / result["after"]["per_call"]["median_us"])
    if args.profile_dir is not None:
        result["profile_dir"] = capture_profile(graphs, inputs, [f"{v}_k1" for v in VARIANTS],
                                                args.profile_dir / case.name)
    return result, {variant: graphs[f"{variant}_k1"] for variant in VARIANTS}


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
            to_host(graphs[name](*inputs[name]))
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
    parser.add_argument("--baseline-commit", default="ab4f37fc")
    parser.add_argument("--sites", nargs="*", choices=sorted(SITES), default=list(SITES))
    parser.add_argument("--glue", nargs="*", choices=GLUE, default=list(GLUE))
    parser.add_argument("--rows", type=int, nargs="*", default=list(PREFILL_ROWS),
                        help="prefill rows: timed and checked")
    parser.add_argument("--decode-rows", type=int, nargs="*", default=list(DECODE_ROWS),
                        help="decode rows: projections checked only")
    parser.add_argument("--chain", type=int, default=3, help="calls in the long graph")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10,
                        help="timed calls per graph per round")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    parser.add_argument("--timing-seed", type=int, default=7)
    parser.add_argument("--emissions", type=int, default=8,
                        help="one-call runs of 'after' per seed, bit-compared")
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--use-compile-cache", action="store_true")
    parser.add_argument("--time-limit", type=int, default=7200,
                        help="seconds before the run aborts")
    args = parser.parse_args()
    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1" or os.environ.get("NKI_SIMULATOR"):
        raise ValueError("Hardware benchmark cannot run in CPU mode or the NKI simulator")
    if not os.environ.get("NEURON_RT_VISIBLE_CORES"):
        raise ValueError("Run under the device lease, which pins NEURON_RT_VISIBLE_CORES")
    if not Path(current.__file__).resolve().is_relative_to(ROOT):
        raise ValueError(f"imported {current.__name__} from {current.__file__}, not {ROOT}")
    if (args.chain < 2 or args.reps < 1 or args.iterations < 1 or args.warmup < 0
            or args.emissions < 1):
        raise ValueError("Use --chain >= 2 and positive rounds, iterations and emissions")
    signal.alarm(args.time_limit)
    if not args.use_compile_cache:
        os.environ["NEURON_LIBTORCH_DISABLE_COMPILE_CACHE"] = "1"
    args.out = args.out.resolve()
    args.records = (args.records or args.out.parent).resolve()
    args.records.mkdir(parents=True, exist_ok=True)
    if args.profile_dir is not None:
        args.profile_dir = args.profile_dir.resolve()
    baseline, baseline_path, baseline_sha = load_baseline(args.baseline_commit, args.records)
    seams = {"before": baseline.mla_projection_lowp, "after": current.mla_projection_lowp}
    text_config = Glm5NextTextConfig()
    eps, hc_eps = float(text_config.rms_norm_eps), float(text_config.hc_eps)
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                           check=True, capture_output=True, text=True).stdout.strip()
    # The NKI and neuronx-cc drivers write artifacts into the working directory.
    scratch = Path(os.environ.get("NEURON_LIBTORCH_CACHE_ROOT") or args.records) / "benchmark_cwd"
    scratch.mkdir(parents=True, exist_ok=True)
    os.chdir(scratch)
    report = {
        "environment": {key: os.environ.get(key) for key in (
            "NEURON_RT_VISIBLE_CORES", "NEURON_LOGICAL_NC_CONFIG", "NEURON_CC_FLAGS",
            "NEURON_PLATFORM_TARGET_OVERRIDE", "NEURON_LIBTORCH_CACHE_ROOT",
            "NEURON_LIBTORCH_DISABLE_COMPILE_CACHE", "VLLM_NEURON_GLUE_FUSED")},
        "tree": {"root": str(ROOT), "head": head, "modified_vs_head": dirty.splitlines(),
                 "module": current.__file__},
        "baseline": {"commit": args.baseline_commit, "file": str(baseline_path),
                     "sha256": baseline_sha},
        "method": {"chain": args.chain, "reps": args.reps, "iterations": args.iterations,
                   "warmup": args.warmup, "seeds": args.seeds, "timing_seed": args.timing_seed,
                   "emissions": args.emissions},
        "timed": [], "decode_numerics": [], "skipped": [],
    }
    numerics_dir = args.records / "numerics"
    numerics_dir.mkdir(parents=True, exist_ok=True)

    def save():
        args.out.write_text(json.dumps(report, indent=1))

    def run(case: Case):
        result, graphs = timed_case(case, args)
        result["numerics"] = [numerics(graphs, case, seed, numerics_dir, args.emissions)
                              for seed in args.seeds]
        report["timed"].append(result)
        save()
        print(json.dumps({"case": case.name,
                          "before_us": result["before"]["per_call"]["median_us"],
                          "after_us": result["after"]["per_call"]["median_us"],
                          "noise": result["noise_floor_relative"],
                          "bit_equal": [n["after_equals_before"] for n in result["numerics"]],
                          "identical": [n["after_emissions_identical"]
                                        for n in result["numerics"]]}),
              flush=True)

    for site in args.sites:
        for rows in args.rows:
            run(projection_case(site, rows, seams))
        for rows in args.decode_rows:
            torch._dynamo.reset()
            case = projection_case(site, rows, seams)
            graphs = {v: compiled(one_call(case.fns[v])) for v in VARIANTS}
            checks = [numerics(graphs, case, seed, numerics_dir, args.emissions)
                      for seed in args.seeds]
            report["decode_numerics"].append({"case": case.name, "rows": rows, **case.meta,
                                              "numerics": checks})
            save()
            print(json.dumps({"case": case.name,
                              "bit_equal": [n["after_equals_before"] for n in checks],
                              "identical": [n["after_emissions_identical"] for n in checks]}),
                  flush=True)
    for glue in args.glue:
        for rows in args.rows:
            if glue == "kv_latent":
                run(kv_latent_case(rows, baseline, eps))
            elif glue_selected("mhc_pre", rows, "prefill"):
                run(input_norm_case(rows, eps, hc_eps))
            else:
                report["skipped"].append({"case": f"input_norm_rows{rows}", "reason":
                                          "VLLM_NEURON_GLUE_FUSED does not select mhc_pre"})
    save()


if __name__ == "__main__":
    main()
