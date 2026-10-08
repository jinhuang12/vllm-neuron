# SPDX-License-Identifier: Apache-2.0
"""Time the sparse MLA prefill attention kernel on Neuron against commit 8aa22fa.

Run it through the device lease, which pins the cores and sets the LNC (2); this script
does not select cores. Each case is one (tokens, ctx) point at the per-rank TP=64 shape of
GLM-5.3-Flash: one head, latent 512, page 128, bf16 query and cache, the selector's
2176-column index rows (512 pools of 4 rows, a 3-row tail, -1 padding to whole 128-column
chunks). ``tokens`` queries at positions ``ctx - tokens .. ctx - 1`` attend a paged window
of ``ctx`` rows whose last ``tokens`` rows are the chunk's own (``written``), as the model
calls it.

One compiled graph runs the model's call ``mla_sparse_attention(q, bank, indices, scale,
block_table_row=..., written=..., write_offset=..., page_size=128)`` once; a sample is one
dispatch with the output copied to CPU. Variants, each its own graph:

* ``before``: the 8aa22fa module from ``--baseline-module``.
* ``after_fp32``: this tree with ``VLLM_NEURON_MLA_SPARSE_FP32=1`` (structural changes only:
  the sentinel mask on the head partitions and two rotating per-query buffer sets; the
  arithmetic is 8aa22fa's).
* ``after_fp32_1buf``: as ``after_fp32`` with one buffer set (isolates the rotation).
* ``after``: this tree as served (bf16 PE operands, fp32 accumulation, hi/lo split of p).
* ``after_1core``: ``after`` on a one-program grid (isolates the two-core split).

``--profile-dir`` captures a device profile of ``--profile-iterations`` warmed iterations of
``before`` and ``after`` after timing. ``cores`` (a separate CPU-only run, after
``neuron-explorer view --ingest-only``) counts each variant's kernel instructions per
physical core.
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
KPOOL, SELECT_K = 4, 512
#: The selector's emitted width: 512 * 4 + 3 = 2051 meaningful columns, padded to 17 x 128.
WIDTH = ((SELECT_K * KPOOL + KPOOL - 1 + 127) // 128) * 128
BANK_ROWS = 65536
CASES = ((1024, 1024), (1024, 8192), (64, 8192))
#: after vs before, fp32 outputs: the hi/lo split carries 16 significand bits of p.
AGREE_REL_L2 = 1e-4
#: The fp32 kill-switch path against before: the same arithmetic, so bitwise.
FP32_REL_L2 = 0.0


def prefill_selection(ctx: int, tokens: int, gen, kpool: int = KPOOL,
                      select_k: int = SELECT_K, width: int = WIDTH):
    """Index rows as ``dsa_index_expand`` lays them out, for the chunk's ``tokens`` queries.

    Query ``t`` sits at position ``pos = ctx - tokens + t`` and sees ``pos + 1`` rows: the
    complete pools ``0 .. (pos + 1) // kpool - 1`` (all of them when there are at most
    ``select_k``, else ``select_k`` of them, drawn at random), each expanded to its
    ``kpool`` consecutive rows in the selector's own (score) order, then the tail rows
    after the last complete pool, then -1 to ``width``.
    """
    import torch

    out = torch.full((tokens, width), -1, dtype=torch.int32)
    for t in range(tokens):
        seen = ctx - tokens + t + 1
        n_pools = seen // kpool
        chosen = torch.randperm(n_pools, generator=gen)[:select_k]
        rows = (chosen.reshape(-1, 1) * kpool + torch.arange(kpool).reshape(1, -1)).reshape(-1)
        tail = torch.arange(n_pools * kpool, seen)
        row = torch.cat([rows, tail])
        out[t, :row.numel()] = row.to(torch.int32)
    return out


def make_case(tokens: int, ctx: int, seed: int):
    """``(q, bank, indices, table, written, offset, window)``: the model's operands and the window they name."""
    import torch

    gen = torch.Generator().manual_seed(seed)
    pages = (ctx + PAGE - 1) // PAGE
    q = (torch.randn(tokens, HEADS, LATENT, generator=gen) * 0.5).to(torch.bfloat16)
    bank = torch.randn(BANK_ROWS, LATENT, generator=gen).to(torch.bfloat16)
    table = torch.randperm(BANK_ROWS // PAGE, generator=gen)[:pages].reshape(pages, 1).to(torch.int32)
    written = torch.randn(tokens, LATENT, generator=gen).to(torch.bfloat16)
    offset = torch.tensor([[ctx - tokens]], dtype=torch.int32)
    indices = prefill_selection(ctx, tokens, gen)
    window = bank.reshape(-1, PAGE, LATENT)[table[:, 0].long()].reshape(pages * PAGE, LATENT).clone()
    window[ctx - tokens:ctx] = written
    return q, bank, indices, table, written, offset, window


def call_of(module, grid: int | None = None):
    """The model's call through ``module``'s seam; ``grid`` forces the row-tiled entry's program count."""
    attend = module.mla_sparse_attention

    def call(q, bank, indices, table, written, offset):
        return attend(q, bank, indices, SCALE, block_table_row=table, written=written,
                      write_offset=offset, page_size=PAGE)

    return call


def compiled(fn):
    import torch

    return torch.compile(fn, backend="neuron_libtorch", fullgraph=True, dynamic=False,
                         options={"compiler_args": os.environ.get("NEURON_CC_FLAGS", "")})


def rel_l2(got, want) -> float:
    got, want = got.double(), want.double()
    return float((got - want).norm() / want.norm().clamp_min(1e-30))


def measure(fn, inputs, warmup: int, iterations: int) -> dict:
    """Median and p90 of one sample (one dispatch), in microseconds."""
    for _ in range(warmup):
        fn(*inputs).to("cpu")
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        fn(*inputs).to("cpu")
        samples.append((time.perf_counter_ns() - started) / 1000.0)
    samples.sort()
    return {
        "iterations": iterations,
        "warmup": warmup,
        "median_us": statistics.median(samples),
        "p90_us": samples[min(len(samples) - 1, int(0.9 * len(samples)))],
        "min_us": samples[0],
        "max_us": samples[-1],
    }


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _Env:
    """Set environment variables for one ``with`` block (the seam reads them at trace time)."""

    def __init__(self, **values):
        self.values = values
        self.saved = {}

    def __enter__(self):
        for key, value in self.values.items():
            self.saved[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def __exit__(self, *exc):
        for key, value in self.saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def variants(base, live):
    """name -> (module, env overrides) for every graph the benchmark times."""
    return {
        "before": (base, {}),
        "after_fp32": (live, {"VLLM_NEURON_MLA_SPARSE_FP32": "1"}),
        "after_fp32_1buf": (live, {"VLLM_NEURON_MLA_SPARSE_FP32": "1",
                                   "VLLM_NEURON_MLA_SPARSE_QUERY_BUFFERS": "1"}),
        "after": (live, {}),
        "after_1core": (live, {"VLLM_NEURON_MLA_SPARSE_PROGRAMS": "1"}),
    }


def run_case(tokens: int, ctx: int, base, live, args) -> dict:
    import torch
    import libtorch_neuronx_lite.envs as libtorch_envs

    q, bank, indices, table, written, offset, window = make_case(tokens, ctx, seed=1000 * tokens + ctx)
    host = (q, bank, indices, table, written, offset)
    inputs = tuple(t.to("neuron:0") for t in host)
    want = live.mla_sparse_attention_torch_oracle(q, window, indices, SCALE)
    outputs, timings, firsts, fns = {}, {}, {}, {}
    for name, (module, env) in variants(base, live).items():
        if args.variant and name not in args.variant:
            continue
        torch._dynamo.reset()
        with _Env(**env):
            fn = compiled(call_of(module))
            started = time.perf_counter()
            got = fn(*inputs).to("cpu")
            firsts[name] = time.perf_counter() - started
            if not torch.isfinite(got).all():
                raise AssertionError(f"{name}: nonfinite output")
            outputs[name] = got
            timings[name] = measure(fn, inputs, args.warmup, args.iterations)
            fns[name] = (fn, env)
    before = outputs.get("before")
    accuracy = {name: {"vs_oracle_rel_l2": rel_l2(got, want),
                       "vs_before_rel_l2": None if before is None else rel_l2(got, before),
                       "vs_before_max_abs": None if before is None else float((got - before).abs().max())}
                for name, got in outputs.items()}
    if before is not None:
        for name in ("after_fp32", "after_fp32_1buf"):
            if name in outputs and accuracy[name]["vs_before_rel_l2"] > FP32_REL_L2:
                raise AssertionError(f"{name} is not bitwise 8aa22fa: {accuracy[name]}")
        for name in ("after", "after_1core"):
            if name in outputs and accuracy[name]["vs_before_rel_l2"] > AGREE_REL_L2:
                raise AssertionError(f"{name} vs before {accuracy[name]}")
    profile_dir = None
    if args.profile_dir is not None and "before" in fns and "after" in fns:
        profile_dir = args.profile_dir.resolve() / f"t{tokens}_ctx{ctx}"
        profile_dir.mkdir(parents=True, exist_ok=True)
        runtime = torch.classes.neuron.Runtime()
        runtime.start_profiling(str(profile_dir), ["device_profile", "system_profile"],
                                None, None, libtorch_envs.get_neuron_compile_cache_dir())
        try:
            for _ in range(args.profile_iterations):
                fns["before"][0](*inputs).to("cpu")
                fns["after"][0](*inputs).to("cpu")
        finally:
            runtime.stop_profiling()
    if "after" in accuracy and accuracy["after"].get("vs_before_rel_l2") is not None:
        # The positive control: the default path is a different arithmetic, so a row that
        # equals 8aa22fa bitwise is the fp32 fallback wearing the wrong name.
        assert accuracy["after"]["vs_before_rel_l2"] > 0.0, \
            "the default path reproduced 8aa22fa bitwise: the low-precision body did not run"
    ref = timings.get("before", {}).get("median_us")
    rows = {}
    for name, timing in timings.items():
        rows[name] = dict(timing)
        rows[name]["first_call_s"] = firsts[name]
        rows[name]["speedup_vs_before"] = None if ref is None else ref / timing["median_us"]
        rows[name]["after_over_before"] = None if ref is None else timing["median_us"] / ref
        rows[name]["cores"] = 1 if name == "after_1core" else 2
    return {
        "tokens": tokens,
        "ctx": ctx,
        "window_pages": (ctx + PAGE - 1) // PAGE,
        "index_width": WIDTH,
        "accuracy": accuracy,
        "variants": rows,
        "profile_dir": str(profile_dir) if profile_dir is not None else None,
        "profile_iterations_per_variant": args.profile_iterations,
    }


def merged_case(earlier: dict, latest: dict) -> dict:
    """One case's rows across runs: the latest run's variants over the earlier ones, the
    speedups re-read against whichever ``before`` row the merge holds."""
    row = dict(latest)
    row["variants"] = {**earlier.get("variants", {}), **latest.get("variants", {})}
    row["accuracy"] = {**earlier.get("accuracy", {}), **latest.get("accuracy", {})}
    if latest.get("profile_dir") is None:
        row["profile_dir"] = earlier.get("profile_dir")
    ref = row["variants"].get("before", {}).get("median_us")
    for one in row["variants"].values():
        one["speedup_vs_before"] = None if ref is None else ref / one["median_us"]
        one["after_over_before"] = None if ref is None else one["median_us"] / ref
    return row


# ---- cores: CPU only, on an ingested profile ----------------------------------------- #

def cores(explorer_global: Path, name: str, source: str = "mla_sparse.py") -> list[dict]:
    """Per execution: the kernel's instructions on each physical core, and its span."""
    import duckdb

    found = sorted(p for p in explorer_global.glob(f"{name}_*@latest")
                   if (p / "Instruction.parquet").exists())
    if not found:
        raise ValueError(f"no ingested profile {name}_*@latest under {explorer_global}")
    rows = []
    for session in found:
        instr = session / "Instruction.parquet"
        execs = session / "ExecutionInfo.parquet"
        # Instructions on one engine overlap (the Tensor engine's weight loads run under its
        # matmuls), so the busy time is the union of the intervals, not their sum: an
        # island starts where an instruction begins after every earlier one has ended.
        query = f"""
            with e as (select execution_index, execution_start_ts s0, execution_end_ts e0
                       from read_parquet('{execs}')),
                 i as (select pcore_idx, engine, start_ts, end_ts, nki_source_location nsl
                       from read_parquet('{instr}') where nki_source_location like '%{source}%'),
                 j as (select e.execution_index ex,
                              case when i.nsl like '%baselines/mla_sparse_8aa22fa/%'
                                   then 'before' else 'after' end variant,
                              i.pcore_idx pcore, i.engine, i.start_ts, i.end_ts
                       from e join i on i.start_ts >= e.s0 and i.end_ts <= e.e0),
                 k as (select *, max(end_ts) over (partition by ex, variant, pcore, engine
                                                   order by start_ts, end_ts
                                                   rows between unbounded preceding and 1 preceding) reach
                       from j),
                 m as (select *, sum(case when reach is null or start_ts > reach then 1 else 0 end)
                                 over (partition by ex, variant, pcore, engine order by start_ts, end_ts
                                       rows unbounded preceding) island
                       from k),
                 u as (select ex, variant, pcore, engine, island, min(start_ts) a, max(end_ts) b
                       from m group by all)
            select j.ex, j.variant, j.pcore, j.engine, count(*), min(j.start_ts), max(j.end_ts),
                   sum(j.end_ts - j.start_ts),
                   (select sum(b - a) from u where u.ex = j.ex and u.variant = j.variant
                                              and u.pcore = j.pcore and u.engine = j.engine)
            from j group by all order by 1, 2, 3, 4"""
        for index, variant, pcore, engine, count, first, last, raw, busy in duckdb.sql(query).fetchall():
            rows.append({"session": session.name, "execution": int(index),
                         "variant": variant, "pcore": int(pcore), "engine": engine,
                         "instructions": int(count), "first_ts": int(first),
                         "last_ts": int(last), "span_us": (last - first) / 1000.0,
                         "busy_us": busy / 1000.0,
                         "instruction_time_us": raw / 1000.0})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")
    bench = sub.add_parser("bench")
    bench.add_argument("--baseline-module", type=Path, required=True)
    bench.add_argument("--output", type=Path, required=True)
    bench.add_argument("--profile-dir", type=Path)
    bench.add_argument("--case", action="append", help="tokens,ctx (repeatable)")
    bench.add_argument("--variant", action="append", help="variant name (repeatable; default all)")
    bench.add_argument("--warmup", type=int, default=3)
    bench.add_argument("--iterations", type=int, default=20)
    bench.add_argument("--profile-iterations", type=int, default=1)
    count = sub.add_parser("cores")
    count.add_argument("--explorer-global", type=Path, required=True)
    count.add_argument("--name", required=True)
    count.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "cores":
        rows = cores(args.explorer_global, args.name)
        args.output.write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps(rows, indent=2))
        return
    if args.command != "bench":
        parser.error("choose bench or cores")

    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1" or os.environ.get("NKI_SIMULATOR"):
        raise ValueError("Hardware benchmark cannot run in CPU mode or the NKI simulator")
    if not os.environ.get("NEURON_RT_VISIBLE_CORES"):
        raise ValueError("Run under the device lease, which pins NEURON_RT_VISIBLE_CORES")
    if args.iterations < 1 or args.warmup < 0:
        raise ValueError("Use positive iteration counts")
    cases = CASES
    if args.case:
        cases = tuple(tuple(int(v) for v in one.split(",")) for one in args.case)

    import vllm_neuron  # noqa: F401 -- registers the Neuron compilation backend
    from vllm_neuron.functional.attention import mla_sparse as live

    if not vllm_neuron.__file__.startswith(str(ROOT) + "/"):
        raise ValueError(f"imported vllm_neuron from {vllm_neuron.__file__}, not {ROOT}")
    baseline_path = args.baseline_module.resolve()
    base = load_module(baseline_path, "mla_sparse_8aa22fa_benchmark")
    output = args.output.resolve()
    scratch = Path(os.environ.get("NEURON_LIBTORCH_CACHE_ROOT") or tempfile.gettempdir())
    scratch = scratch / "benchmark_cwd"
    scratch.mkdir(parents=True, exist_ok=True)
    os.chdir(scratch)
    report = {"environment": {key: os.environ.get(key) for key in (
                  "NEURON_RT_VISIBLE_CORES", "NEURON_LOGICAL_NC_CONFIG", "NEURON_CC_FLAGS",
                  "NEURON_PLATFORM_TARGET_OVERRIDE", "NEURON_LIBTORCH_CACHE_ROOT")},
              "tree": str(ROOT),
              "after_module": live.__file__,
              "after_source_digest": getattr(live, "SOURCE_DIGEST", None),
              "baseline_module": str(baseline_path),
              "baseline_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
              "args": {"warmup": args.warmup, "iterations": args.iterations,
                       "cases": [list(c) for c in cases], "variants": args.variant},
              "shape": {"heads": HEADS, "latent": LATENT, "page": PAGE, "softmax_scale": SCALE,
                        "bank_rows": BANK_ROWS, "index_width": WIDTH, "kpool": KPOOL,
                        "select_k": SELECT_K, "dtype": "bfloat16"},
              "synchronization": "graph output copied to CPU on every timed call",
              "cases": []}
    if output.exists():
        try:
            report["cases"] = json.loads(output.read_text()).get("cases", [])
        except ValueError:
            pass
    output.parent.mkdir(parents=True, exist_ok=True)
    for tokens, ctx in cases:
        row = run_case(tokens, ctx, base, live, args)
        earlier = [c for c in report["cases"] if (c["tokens"], c["ctx"]) == (tokens, ctx)]
        report["cases"] = [c for c in report["cases"] if (c["tokens"], c["ctx"]) != (tokens, ctx)]
        if earlier:
            row = merged_case(earlier[0], row)
        report["cases"].append(row)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"tokens": tokens, "ctx": ctx,
                          **{f"{n}_us": round(v["median_us"], 1) for n, v in row["variants"].items()},
                          **{f"{n}_x": round(v["speedup_vs_before"], 3) for n, v in row["variants"].items()
                             if v["speedup_vs_before"] is not None}}), flush=True)


if __name__ == "__main__":
    main()
