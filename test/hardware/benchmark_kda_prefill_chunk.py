# SPDX-License-Identifier: Apache-2.0
"""The KDA prefill chunk kernels on Neuron: a baseline revision against this tree.

Times :func:`kda_intra_chunk` (stages 1 to 3) and :func:`kda_inter_chunk` (stages 4
and 5) at the served prefill shapes, ``[NC, C, K] = [128, 8, 128]`` (a 1024-token
chunk) and ``[256, 8, 128]`` (a 2048-token chunk), for the baseline revision's
``chunked_recurrence.py`` (read with ``git show``) and for this checkout's.

One compiled graph holds ``--layers`` independent calls of one entry point, each on
its own operands, because a prefill chunk calls each kernel once per KDA layer. A
graph's time is the host wall time of one launch plus the copy of its first output
to CPU; an empty graph over the same operands that returns a tensor of the same
shape (``floor``) is timed alongside, and the per-call figure is
``(graph - floor) / layers``.

The variants are timed interleaved: every repetition times every variant once, in
the same order, ``--iterations`` launches each after ``--warmup``. A graph's figure
is the median over repetitions of the per-repetition medians. A kernel's
per-call figure in one repetition is that repetition's ``(graph - floor) /
layers``, and its spread is ``(max - min) / median`` of those over the
repetitions. The noise floor is the largest per-call spread over the four kernel
variants, i.e. how far one tree's own per-call figure moves between repetitions
in this run; the graphs' own spreads, the floor graph's included, are reported
beside it.

Numerics. For each ``--seeds`` entry and each shape, one-call graphs of both
revisions run on the same operands; the outputs are saved as ``.pt`` files under
``--numerics-dir`` and compared with ``torch.equal`` (baseline against this tree)
and against the CPU references (:func:`kda_intra_chunk_torch_oracle`,
:func:`rebuild_i_plus_a`, :func:`kda_inter_chunk_torch_oracle`). The inter-chunk
operands are the CPU reference's intra-chunk outputs, so both revisions' inter-chunk
kernels read identical bits.

Run it through the device lease, which pins the cores and the LNC; this script
refuses to run without ``NEURON_RT_VISIBLE_CORES``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

#: The worktree root, ahead of any installed copy: the lease command sets no
#: PYTHONPATH, and the venv's own ``vllm_neuron`` is another tree.
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True

import torch

import vllm_neuron  # noqa: F401 -- registers the Neuron compilation backend
from vllm_neuron.functional.kda import chunked_recurrence as live

DEVICE = "neuron:0"

#: The file both revisions are read from.
KERNEL_PATH = "vllm_neuron/functional/kda/chunked_recurrence.py"

#: The revision this series starts from; ``--baseline-rev`` names another.
BASELINE_REV = "f3a833f"

#: Chunks per call at the two served prefill chunk sizes, 1024 and 2048 tokens.
SERVED_N_CHUNKS = (128, 256)

#: The chunk width the KDA layer resolves, and the per-rank key and value widths
#: at TP=64 (one head per rank). ``test_chunked_recurrence`` reads the same three
#: values off the layer.
SERVED_CHUNK = 8
SERVED_KDIM = 128
SERVED_VDIM = 128

#: KDA layers per prefill chunk, i.e. calls of each kernel per chunk.
KDA_LAYERS = 34

#: The checkpoint's gate lower bound: each per-token log gate lies in
#: ``(GATE_LOWER_BOUND, 0)``, as the gate clamp produces it.
GATE_LOWER_BOUND = -5.0

INTRA_FIELDS = ("w", "u", "kg", "a_inv", "aqk")
INTER_FIELDS = ("o", "final_state", "v_new")


def load_baseline(rev: str, workdir: Path):
    """The baseline revision's ``chunked_recurrence`` as a fresh module.

    The source is written into ``workdir`` because the NKI front end reads a
    kernel's source from its file. The module is registered under its own name,
    because a compiled graph resolves the kernel's globals by module name.
    """
    source = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{rev}:{KERNEL_PATH}"],
        check=True, capture_output=True, text=True,
    ).stdout
    path = workdir / "baseline_chunked_recurrence.py"
    path.write_text(source)
    name = "_kda_prefill_baseline_chunked_recurrence"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load the baseline source {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, hashlib.sha256(source.encode()).hexdigest()


def make_inputs(n_chunks: int, seed: int) -> dict:
    """One call's operands for both kernels, as CPU float32 tensors.

    The intra-chunk operands are drawn; the inter-chunk operands are the CPU
    reference's intra-chunk outputs on them, plus an entering state.
    """
    gen = torch.Generator().manual_seed(seed)
    shape_k = (n_chunks, SERVED_CHUNK, SERVED_KDIM)
    q = torch.randn(shape_k, generator=gen)
    k = torch.randn(shape_k, generator=gen)
    v = torch.randn((n_chunks, SERVED_CHUNK, SERVED_VDIM), generator=gen)
    beta = torch.sigmoid(torch.randn((n_chunks, SERVED_CHUNK), generator=gen))
    gk = GATE_LOWER_BOUND * torch.sigmoid(torch.randn(shape_k, generator=gen) * 2)
    state = torch.randn((SERVED_VDIM, SERVED_KDIM), generator=gen) * 0.5
    ref = live.kda_intra_chunk_torch_oracle(q, k, v, beta, gk)
    return {
        "intra": (q, k, v, beta, gk),
        "inter": (ref.kg, ref.w, ref.u, gk, q, ref.aqk, state),
        "intra_reference": ref,
    }


def intra_graph(module, layers: int):
    def run(*args):
        outs = []
        for layer in range(layers):
            got = module.kda_intra_chunk(*args[5 * layer : 5 * (layer + 1)])
            outs.extend(got)
        return tuple(outs)

    return run


def inter_graph(module, layers: int):
    def run(*args):
        outs = []
        for layer in range(layers):
            kg, w, u, gk, q, aqk, state = args[7 * layer : 7 * (layer + 1)]
            outs.extend(module.kda_inter_chunk(kg, w, u, gk, q, aqk, state=state))
        return tuple(outs)

    return run


def floor_graph(*args):
    # The same operands and no kernel: the launch, the first-output copy and the
    # synchronisation every timed graph contains.
    return (args[0] + 0.0,)


def compiled(fn):
    return torch.compile(
        fn, backend="neuron_libtorch", fullgraph=True, dynamic=False,
        options={"compiler_args": os.environ.get("NEURON_CC_FLAGS", "")},
    )


def to_device(tensors) -> tuple:
    return tuple(t.contiguous().to(DEVICE) for t in tensors)


def time_once(model, inputs, warmup: int, iterations: int) -> dict:
    for _ in range(warmup):
        model(*inputs)[0].to("cpu")
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        model(*inputs)[0].to("cpu")
        samples.append((time.perf_counter_ns() - started) / 1_000)
    return {"median_us": statistics.median(samples), "samples_us": samples}


def summarize_reps(per_rep: list[float]) -> dict:
    median = statistics.median(per_rep)
    return {
        "median_us": median,
        "min_us": min(per_rep),
        "max_us": max(per_rep),
        "spread": (max(per_rep) - min(per_rep)) / median,
        "per_rep_median_us": per_rep,
    }


def bench_shape(n_chunks: int, baseline, args) -> dict:
    """Timed graphs for one shape: both kernels, both revisions, and the floor."""
    torch._dynamo.reset()
    layers = args.layers
    cases = [make_inputs(n_chunks, args.timing_seed + layer) for layer in range(layers)]
    intra_in = to_device([t for c in cases for t in c["intra"]])
    inter_in = to_device([t for c in cases for t in c["inter"]])
    variants = {
        "intra_before": (compiled(intra_graph(baseline, layers)), intra_in),
        "intra_after": (compiled(intra_graph(live, layers)), intra_in),
        "inter_before": (compiled(inter_graph(baseline, layers)), inter_in),
        "inter_after": (compiled(inter_graph(live, layers)), inter_in),
        "floor": (compiled(floor_graph), intra_in),
    }
    result = {"n_chunks": n_chunks, "layers": layers, "first_call_s": {}}
    for name, (model, inputs) in variants.items():
        started = time.perf_counter()
        first = model(*inputs)[0].to("cpu")
        result["first_call_s"][name] = time.perf_counter() - started
        if not torch.isfinite(first).all():
            raise AssertionError(f"{name} at n_chunks={n_chunks} returned non-finite values")

    per_rep = {name: [] for name in variants}
    raw = {name: [] for name in variants}
    for _ in range(args.reps):
        for name, (model, inputs) in variants.items():
            got = time_once(model, inputs, args.warmup, args.iterations)
            per_rep[name].append(got["median_us"])
            raw[name].append(got["samples_us"])
    graphs = {name: summarize_reps(values) for name, values in per_rep.items()}
    floor_us = graphs["floor"]["median_us"]
    per_call = {}
    for name, summary in graphs.items():
        if name == "floor":
            continue
        per_rep_us = [
            (value - floor_rep) / layers
            for value, floor_rep in zip(summary["per_rep_median_us"],
                                        graphs["floor"]["per_rep_median_us"])
        ]
        per_call[name] = {
            "median_us": (summary["median_us"] - floor_us) / layers,
            "per_rep_us": per_rep_us,
            "spread": (max(per_rep_us) - min(per_rep_us)) / statistics.median(per_rep_us),
        }
    kernel_variants = tuple(per_call)
    for kernel in ("intra", "inter"):
        before = per_call[f"{kernel}_before"]["median_us"]
        after = per_call[f"{kernel}_after"]["median_us"]
        per_call[f"{kernel}_speedup"] = before / after
    result.update({
        "graphs_us": graphs,
        "per_call_us": per_call,
        "noise_floor_spread": max(per_call[name]["spread"] for name in kernel_variants),
        "graph_spread": {name: summary["spread"] for name, summary in graphs.items()},
        "samples_us": raw,
    })
    return result


def tensor_diff(got: torch.Tensor, want: torch.Tensor) -> dict:
    got, want = got.double(), want.double()
    residual = got - want
    return {
        "max_abs": float(residual.abs().max()),
        "rel_l2": float(residual.norm() / want.norm().clamp_min(1e-300)),
        "mismatched_elements": int((got != want).sum()),
    }


def numerics_shape(n_chunks: int, baseline, args) -> list[dict]:
    """One-call graphs of both revisions on ``--seeds`` operands; outputs saved."""
    torch._dynamo.reset()
    models = {
        "intra_before": compiled(intra_graph(baseline, 1)),
        "intra_after": compiled(intra_graph(live, 1)),
        "inter_before": compiled(inter_graph(baseline, 1)),
        "inter_after": compiled(inter_graph(live, 1)),
    }
    rows = []
    for seed in args.seeds:
        case = make_inputs(n_chunks, seed)
        directory = args.numerics_dir / f"n{n_chunks}" / f"seed{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        torch.save({"intra": case["intra"], "inter": case["inter"]},
                   directory / "inputs.pt")
        outputs = {}
        for name, model in models.items():
            kernel = name.split("_")[0]
            fields = INTRA_FIELDS if kernel == "intra" else INTER_FIELDS
            got = model(*to_device(case[kernel]))
            outputs[name] = {f: t.to("cpu") for f, t in zip(fields, got)}
            torch.save(outputs[name], directory / f"{name}.pt")

        _, k, _, beta, gk = case["intra"]
        intra_ref = case["intra_reference"]
        inter_ref = live.kda_inter_chunk_torch_oracle(
            *case["inter"][:6], state=case["inter"][6]
        )
        row = {"n_chunks": n_chunks, "seed": seed, "dir": str(directory)}
        for kernel, fields, ref in (("intra", INTRA_FIELDS, intra_ref),
                                    ("inter", INTER_FIELDS, inter_ref)):
            before, after = outputs[f"{kernel}_before"], outputs[f"{kernel}_after"]
            row[kernel] = {
                f: {
                    "bit_equal": bool(torch.equal(before[f], after[f])),
                    "after_vs_before": tensor_diff(after[f], before[f]),
                    "before_vs_reference": tensor_diff(before[f], getattr(ref, f)),
                    "after_vs_reference": tensor_diff(after[f], getattr(ref, f)),
                }
                for f in fields
            }
        i_plus_a = live.rebuild_i_plus_a(k, beta, gk)
        identity = torch.eye(SERVED_CHUNK).expand_as(i_plus_a)
        row["intra_inverse_residual_max_abs"] = {
            rev: float((i_plus_a @ outputs[f"intra_{rev}"]["a_inv"] - identity).abs().max())
            for rev in ("before", "after")
        }
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline-rev", default=BASELINE_REV)
    parser.add_argument("--n-chunks", type=int, nargs="+", default=list(SERVED_N_CHUNKS))
    parser.add_argument("--layers", type=int, default=KDA_LAYERS)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--timing-seed", type=int, default=20261008)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--numerics-dir", type=Path)
    parser.add_argument("--workdir", type=Path)
    args = parser.parse_args()
    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1":
        raise ValueError("Hardware benchmark cannot run in VLLM_NEURON_CPU_MODE=1")
    if not os.environ.get("NEURON_RT_VISIBLE_CORES"):
        raise ValueError("Run through the device lease, which pins NEURON_RT_VISIBLE_CORES")
    if args.reps < 1 or args.iterations < 1 or args.warmup < 0 or args.layers < 1:
        raise ValueError("Use positive reps, iterations and layers, nonnegative warmup")
    args.out = args.out.resolve()
    args.workdir = (args.workdir or args.out.parent / f"{args.out.stem}_workdir").resolve()
    args.numerics_dir = (
        args.numerics_dir or args.out.parent / f"{args.out.stem}_numerics"
    ).resolve()
    args.workdir.mkdir(parents=True, exist_ok=True)
    baseline, baseline_sha256 = load_baseline(args.baseline_rev, args.workdir)
    # The kernel compiler writes its artifacts into the working directory; keep
    # them out of the checkout.
    os.chdir(args.workdir)
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", KERNEL_PATH],
                           check=True, capture_output=True, text=True).stdout.strip()
    report = {
        "environment": {
            key: os.environ.get(key)
            for key in ("NEURON_RT_VISIBLE_CORES", "NEURON_LOGICAL_NC_CONFIG",
                        "NEURON_CC_FLAGS", "NEURON_PLATFORM_TARGET_OVERRIDE")
        },
        "tree": {"head": head, "kernel_file_dirty": bool(dirty),
                 "kernel_sha256": hashlib.sha256(
                     (REPO / KERNEL_PATH).read_bytes()).hexdigest()},
        "baseline": {"rev": args.baseline_rev, "kernel_sha256": baseline_sha256},
        "shape": {"chunk": SERVED_CHUNK, "kdim": SERVED_KDIM, "vdim": SERVED_VDIM},
        "timing_unit": (
            "microseconds per kernel call = (graph median - floor median) / layers; "
            "graph time is host wall time of one launch plus its first-output copy"
        ),
        "args": {k: str(v) for k, v in vars(args).items()},
        "timing": [],
        "numerics": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for n_chunks in args.n_chunks:
        result = bench_shape(n_chunks, baseline, args)
        report["timing"].append(result)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"n_chunks": n_chunks, "per_call_us": result["per_call_us"],
                          "noise_floor_spread": result["noise_floor_spread"]}), flush=True)
    for n_chunks in args.n_chunks:
        rows = numerics_shape(n_chunks, baseline, args)
        report["numerics"].extend(rows)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        for row in rows:
            print(json.dumps({
                "n_chunks": n_chunks, "seed": row["seed"],
                "intra_bit_equal": {f: row["intra"][f]["bit_equal"] for f in INTRA_FIELDS},
                "inter_bit_equal": {f: row["inter"][f]["bit_equal"] for f in INTER_FIELDS},
            }), flush=True)


if __name__ == "__main__":
    main()
