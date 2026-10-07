# SPDX-License-Identifier: Apache-2.0
"""Time the MLA decode attention step on Neuron against commit 0a08ff4.

Run it through the device lease, which pins the cores and sets the LNC (2); this script
does not select cores. Each case is one (B, ctx) point at the per-rank TP=64 shape of
GLM-5.3-Flash: one head, latent 512, page 128, bf16, a 557184-row latent bank (the B=64
serving graph's), each request on its own pages:

* ctx 1024 -- the bypass regime: dense causal prefix in the 2048-row decode bucket
  (16 pages), no ``topk_indices``.
* ctx 8192 -- the selecting regime: 2048 distinct selected rows of an 8192-row window
  (64 pages), this step's own row among them.

One compiled graph runs the model's call sequence ``--layers`` times (default 11, the DSA
layer count of one decode step): ``mla_decode_attention(q, bank, table.t(), position,
written, ...)`` then ``.to(bfloat16)``, each layer with its own query and own row, the
table passed transposed as the model passes it. So a sample is one decode step's MLA
decode attention plus the glue right around it, and one dispatch (the output is copied to
CPU on every timed call); per-layer figures are the sample over ``--layers``. Before is
the 0a08ff4 module loaded from ``--baseline-module``; after is this tree's. Before and
after samples interleave, so a change in host load reaches both alike.

``--profile-dir`` captures a device profile of ``--profile-iterations`` warmed iterations
per variant after timing. ``cores`` (a separate CPU-only run, after ``neuron-explorer view
--ingest-only``) counts each variant's kernel instructions per physical core.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time

#: The worktree root, ahead of any installed copy (the lease sets no PYTHONPATH).
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

HEADS, LATENT, PAGE, SCALE = 1, 512, 128, 0.0625
BANK_ROWS = 557184
#: ctx -> (window pages, selected width or None for the dense bypass).
REGIMES = {1024: (16, None), 8192: (64, 2048)}
CASES = ((1, 1024), (4, 1024), (64, 1024), (64, 8192))
#: Before vs after, both cast to bf16 as the model casts them: a few one-ulp flips.
AGREE_REL_L2 = 4e-3
#: One call, fp32 output, against the fp32 CPU oracle: the simulator tests' bound
#: (2e-5) with room for the device's own fp32 summation order.
DEVICE_REL_L2 = 1e-4


def make_case(batch: int, ctx: int, layers: int, seed: int):
    import torch

    pages, width = REGIMES[ctx]
    gen = torch.Generator().manual_seed(seed)
    bank = torch.randn(BANK_ROWS, LATENT, generator=gen).to(torch.bfloat16)
    q = torch.randn(layers, batch, HEADS, LATENT, generator=gen).to(torch.bfloat16)
    written = torch.randn(layers, batch, LATENT, generator=gen).to(torch.bfloat16)
    table = torch.randperm(BANK_ROWS // PAGE, generator=gen)[:batch * pages]
    table = table.reshape(batch, pages).to(torch.int32)
    pos = (ctx - 1 - torch.randint(0, 97, (batch,), generator=gen)).to(torch.int32)
    for b in range(batch):
        table[b, int(pos[b]) // PAGE + 1:] = -1
    topk = None
    if width is not None:
        topk = torch.empty(batch, width, dtype=torch.int32)
        for b in range(batch):
            pick = torch.randperm(int(pos[b]), generator=gen)[:width - 1]
            row = torch.cat([pick, pos[b:b + 1].to(torch.int64)])
            topk[b] = row[torch.randperm(width, generator=gen)].to(torch.int32)
    return q, bank, table.t().contiguous(), pos, written, topk


def chain(module, layers: int, dense: bool):
    import torch

    attend = module.mla_decode_attention

    def step_dense(q, bank, table_t, pos, written):
        return torch.stack([
            attend(q[i], bank, table_t.t(), pos, written[i], SCALE, PAGE).to(torch.bfloat16)
            for i in range(layers)])

    def step_selected(q, bank, table_t, pos, written, topk):
        return torch.stack([
            attend(q[i], bank, table_t.t(), pos, written[i], SCALE, PAGE,
                   topk_indices=topk).to(torch.bfloat16)
            for i in range(layers)])

    return step_dense if dense else step_selected


def single(module, dense: bool):
    """One layer's call, fp32 output, for the accuracy check."""
    attend = module.mla_decode_attention

    def call_dense(q, bank, table_t, pos, written):
        return attend(q, bank, table_t.t(), pos, written, SCALE, PAGE)

    def call_selected(q, bank, table_t, pos, written, topk):
        return attend(q, bank, table_t.t(), pos, written, SCALE, PAGE, topk_indices=topk)

    return call_dense if dense else call_selected


def compiled(fn):
    import torch

    return torch.compile(fn, backend="neuron_libtorch", fullgraph=True, dynamic=False,
                         options={"compiler_args": os.environ.get("NEURON_CC_FLAGS", "")})


def rel_l2(got, want) -> float:
    got, want = got.double(), want.double()
    return float((got - want).norm() / want.norm().clamp_min(1e-30))


def measure(fns, inputs, layers: int, warmup: int, iterations: int) -> list[dict]:
    """Median and p90 of one sample (one step, ``layers`` calls) per function, in us.

    The functions' samples interleave (ABAB..., the order swapped every iteration), so a
    change in host load during the run reaches every function alike.
    """
    for fn in fns:
        for _ in range(warmup):
            fn(*inputs).to("cpu")
    samples = [[] for _ in fns]
    for i in range(iterations):
        order = range(len(fns)) if i % 2 == 0 else reversed(range(len(fns)))
        for k in order:
            started = time.perf_counter_ns()
            fns[k](*inputs).to("cpu")
            samples[k].append((time.perf_counter_ns() - started) / 1000.0)
    return [_stats(one, layers, warmup) for one in samples]


def _stats(samples: list, layers: int, warmup: int) -> dict:
    samples = sorted(samples)
    iterations = len(samples)
    median = statistics.median(samples)
    return {
        "iterations": iterations,
        "warmup": warmup,
        "layers_per_sample": layers,
        "median_us": median,
        "p90_us": samples[min(len(samples) - 1, int(0.9 * len(samples)))],
        "min_us": samples[0],
        "max_us": samples[-1],
        "median_us_per_layer": median / layers,
    }


def load_baseline(path: Path, name: str = "mla_decode_0a08ff4_benchmark"):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load baseline {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_case(batch: int, ctx: int, base, live, args) -> dict:
    import torch
    import libtorch_neuronx_lite.envs as libtorch_envs

    torch._dynamo.reset()
    dense = REGIMES[ctx][1] is None
    q, bank, table_t, pos, written, topk = make_case(batch, ctx, args.layers,
                                                    seed=1000 * batch + ctx)
    host = (q, bank, table_t, pos, written) + (() if dense else (topk,))
    inputs = tuple(t.to("neuron:0") for t in host)
    before_fn = compiled(chain(base, args.layers, dense))
    after_fn = compiled(chain(live, args.layers, dense))
    started = time.perf_counter()
    old = before_fn(*inputs).to("cpu")
    before_first_s = time.perf_counter() - started
    started = time.perf_counter()
    new = after_fn(*inputs).to("cpu")
    after_first_s = time.perf_counter() - started
    for value in (old, new):
        if not torch.isfinite(value.float()).all():
            raise AssertionError("device decode attention returned nonfinite values")
    agreement = rel_l2(new, old)
    # fp32, one layer: the kernels' own accuracy on device, before any bf16 cast.
    one = (inputs[0][0], inputs[1], inputs[2], inputs[3], inputs[4][0]) + inputs[5:]
    want0 = live.mla_decode_attention_torch_oracle(q[0], bank, table_t.t(), pos, written[0],
                                                   SCALE, PAGE, topk)
    fp32 = {"after_vs_oracle": rel_l2(compiled(single(live, dense))(*one).to("cpu"), want0),
            "before_vs_oracle": rel_l2(compiled(single(base, dense))(*one).to("cpu"), want0)}
    if fp32["after_vs_oracle"] > DEVICE_REL_L2:
        raise AssertionError(f"B={batch} ctx={ctx}: fp32 after vs oracle {fp32}")
    oracle = {}
    for layer in sorted({0, args.layers - 1}):
        want = live.mla_decode_attention_torch_oracle(
            q[layer], bank, table_t.t(), pos, written[layer], SCALE, PAGE, topk)
        oracle[f"layer{layer}"] = {"after": rel_l2(new[layer], want),
                                   "before": rel_l2(old[layer], want)}
    if agreement > AGREE_REL_L2 or any(v["after"] > AGREE_REL_L2 for v in oracle.values()):
        raise AssertionError(f"B={batch} ctx={ctx}: after vs before {agreement:.3e}, "
                             f"vs oracle {oracle}")
    before, after = measure((before_fn, after_fn), inputs, args.layers, args.warmup,
                            args.iterations)
    profile_dir = None
    if args.profile_dir is not None:
        profile_dir = args.profile_dir.resolve() / f"b{batch}_ctx{ctx}"
        profile_dir.mkdir(parents=True, exist_ok=True)
        runtime = torch.classes.neuron.Runtime()
        runtime.start_profiling(str(profile_dir), ["device_profile", "system_profile"],
                                None, None, libtorch_envs.get_neuron_compile_cache_dir())
        try:
            for _ in range(args.profile_iterations):
                before_fn(*inputs).to("cpu")
                after_fn(*inputs).to("cpu")
        finally:
            runtime.stop_profiling()
    return {
        "batch": batch,
        "ctx": ctx,
        "mode": "dense" if dense else "selected",
        "window_pages": REGIMES[ctx][0],
        "selected_width": REGIMES[ctx][1],
        "first_call_s": {"before": before_first_s, "after": after_first_s},
        "after_vs_before_rel_l2_bf16": agreement,
        "vs_oracle_rel_l2_bf16": oracle,
        "fp32_layer0_rel_l2": fp32,
        "before": before,
        "after": after,
        "speedup_median": before["median_us"] / after["median_us"],
        "after_over_before_median": after["median_us"] / before["median_us"],
        "profile_dir": str(profile_dir) if profile_dir is not None else None,
        "profile_iterations_per_variant": args.profile_iterations,
    }


# ---- cores: CPU only, on an ingested profile ----------------------------------------- #

def cores(explorer_global: Path, name: str, source: str) -> list[dict]:
    """Per execution: the kernel's instructions on each physical core, and its span.

    ``source`` is the kernel file's basename; ``nki_source_location`` names the file
    each instruction came from, which tells the live kernel (``mla_decode.py`` in this
    tree) from the 0a08ff4 copy (the same basename under ``baselines/``).
    """
    import duckdb

    found = sorted(p for p in explorer_global.glob(f"{name}_*@latest")
                   if (p / "Instruction.parquet").exists())
    if not found:
        raise ValueError(f"no ingested profile {name}_*@latest under {explorer_global}")
    rows = []
    for session in found:
        instr = session / "Instruction.parquet"
        execs = session / "ExecutionInfo.parquet"
        query = f"""
            with e as (select execution_index, execution_start_ts s0, execution_end_ts e0
                       from read_parquet('{execs}')),
                 i as (select pcore_idx, start_ts, end_ts, nki_source_location nsl
                       from read_parquet('{instr}') where nki_source_location like '%{source}%')
            select e.execution_index,
                   case when i.nsl like '%baselines/mla_0a08ff4/%' then 'before' else 'after' end,
                   i.pcore_idx, count(*), min(i.start_ts), max(i.end_ts), sum(i.end_ts - i.start_ts)
            from e join i on i.start_ts >= e.s0 and i.end_ts <= e.e0
            group by all order by 1, 2, 3"""
        for index, variant, pcore, count, first, last, busy in duckdb.sql(query).fetchall():
            rows.append({"session": session.name, "execution": int(index),
                         "variant": variant, "pcore": int(pcore),
                         "instructions": int(count), "first_ts": int(first),
                         "last_ts": int(last), "span_us": (last - first) / 1000.0,
                         "instruction_time_us": busy / 1000.0})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")
    bench = sub.add_parser("bench")
    bench.add_argument("--baseline-module", type=Path, required=True)
    bench.add_argument("--after-module", type=Path,
                       help="an experimental after-module file (default: this tree's)")
    bench.add_argument("--output", type=Path, required=True)
    bench.add_argument("--profile-dir", type=Path)
    bench.add_argument("--case", action="append", help="B,ctx (repeatable)")
    bench.add_argument("--layers", type=int, default=11)
    bench.add_argument("--warmup", type=int, default=10)
    bench.add_argument("--iterations", type=int, default=50)
    bench.add_argument("--profile-iterations", type=int, default=1)
    count = sub.add_parser("cores")
    count.add_argument("--explorer-global", type=Path, required=True)
    count.add_argument("--name", required=True)
    count.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "cores":
        rows = cores(args.explorer_global, args.name, "mla_decode.py")
        args.output.write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps(rows, indent=2))
        return
    if args.command != "bench":
        parser.error("choose bench or cores")

    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1" or os.environ.get("NKI_SIMULATOR"):
        raise ValueError("Hardware benchmark cannot run in CPU mode or the NKI simulator")
    if not os.environ.get("NEURON_RT_VISIBLE_CORES"):
        raise ValueError("Run under the device lease, which pins NEURON_RT_VISIBLE_CORES")
    if args.layers < 1 or args.iterations < 1 or args.warmup < 0:
        raise ValueError("Use positive layer and iteration counts")
    cases = CASES
    if args.case:
        cases = tuple(tuple(int(v) for v in one.split(",")) for one in args.case)
    for batch, ctx in cases:
        if ctx not in REGIMES or batch < 1:
            raise ValueError(f"unknown case B={batch} ctx={ctx}")

    import vllm_neuron  # noqa: F401 -- registers the Neuron compilation backend
    from vllm_neuron.functional.attention import mla_decode as live

    if not vllm_neuron.__file__.startswith(str(ROOT) + "/"):
        raise ValueError(f"imported vllm_neuron from {vllm_neuron.__file__}, not {ROOT}")
    baseline_path = args.baseline_module.resolve()
    base = load_baseline(baseline_path)
    if args.after_module is not None:
        live = load_baseline(args.after_module.resolve(), "mla_decode_after_benchmark")
    output = args.output.resolve()
    # The compiler leaves per-graph debug files in its working directory.
    scratch = Path(os.environ.get("NEURON_LIBTORCH_CACHE_ROOT") or tempfile.gettempdir())
    scratch = scratch / "benchmark_cwd"
    scratch.mkdir(parents=True, exist_ok=True)
    os.chdir(scratch)
    report = {
        "environment": {
            key: os.environ.get(key)
            for key in ("NEURON_RT_VISIBLE_CORES", "NEURON_LOGICAL_NC_CONFIG",
                        "NEURON_CC_FLAGS", "NEURON_PLATFORM_TARGET_OVERRIDE",
                        "NEURON_LIBTORCH_CACHE_ROOT")
        },
        "tree": str(ROOT),
        "after_module": live.__file__,
        "after_source_digest": live.SOURCE_DIGEST,
        "baseline_module": str(baseline_path),
        "baseline_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        "args": {"layers": args.layers, "warmup": args.warmup,
                 "iterations": args.iterations, "cases": [list(c) for c in cases]},
        "shape": {"heads": HEADS, "latent": LATENT, "page": PAGE, "softmax_scale": SCALE,
                  "bank_rows": BANK_ROWS},
        "synchronization": "graph output copied to CPU on every timed call",
        "sampling": "before and after samples interleaved, order swapped every iteration",
        "cases": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    for batch, ctx in cases:
        row = run_case(batch, ctx, base, live, args)
        report["cases"].append(row)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: row[k] for k in ("batch", "ctx", "speedup_median",
                                              "after_vs_before_rel_l2_bf16")}
                         | {"before_us": row["before"]["median_us"],
                            "after_us": row["after"]["median_us"]}), flush=True)


if __name__ == "__main__":
    main()
