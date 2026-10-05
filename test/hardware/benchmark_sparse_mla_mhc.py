# SPDX-License-Identifier: Apache-2.0
"""Compare sparse MLA and mHC kernels with unchanged source snapshots on Neuron.

Select a core and LNC configuration in the environment before launching. CPU
references include duplicate indices, sentinel chunks and fully masked queries.
Compilation and warmup are excluded. Each timed call copies the same-sized output
to CPU to synchronize, so the timings include dispatch and synchronization.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time

import torch
import torch.nn.functional as functional

import vllm_neuron  # noqa: F401 -- registers the Neuron compilation backend
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

from vllm_neuron.functional.attention import mla_sparse as mla
from vllm_neuron.functional.mhc import hyper_connection as mhc


def load_snapshot(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load baseline snapshot {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def save_inputs(inputs, label: str, args) -> dict | None:
    """Save raw, contiguous inputs for a matching neuron-explorer replay."""
    if args.input_dir is None:
        return None
    destination = args.input_dir / label
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for index, value in enumerate(inputs):
        path = destination / f"input{index}.bin"
        data = value.contiguous().view(torch.uint8).numpy().tobytes()
        path.write_bytes(data)
        manifest[f"input{index}"] = {
            "path": str(path),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "bytes": len(data),
        }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    # Large vectors need FP64 reductions for a useful norm/cosine diagnostic.
    actual, expected = actual.double(), expected.double()
    residual = actual - expected
    return {
        "max_absolute_difference": residual.abs().max().item(),
        "difference_norm": residual.norm().item(),
        "relative_l2": (
            residual.norm() / expected.norm().clamp_min(1e-30)
        ).item(),
        "cosine_similarity": functional.cosine_similarity(
            actual.reshape(1, -1), expected.reshape(1, -1)
        ).item(),
    }


def timing_stats(samples: list[float]) -> dict:
    ordered = sorted(samples)
    return {
        "iterations": len(samples),
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "p90_ms": ordered[min(len(samples) - 1, int(0.9 * len(samples)))],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "samples_ms": samples,
    }


def device_timing(events_json: str, call_order: list[str]) -> dict:
    """Merge physical-core execution intervals for each LNC2 invocation."""
    starts = {}
    intervals = {}
    for event in json.loads(events_json)["events"]:
        if event["event_type"] != "nc_exec_running":
            continue
        key = (event["nc_idx"], event["tracking_id"])
        if event["phase"] == "start":
            starts[key] = event
        elif event["phase"] == "stop":
            start = starts.pop(key)
            execution = start["data"]["exec_id"]
            intervals.setdefault(execution, []).append((
                start["data"]["nc_timestamp_ns"],
                event["data"]["nc_timestamp_ns"],
            ))
    if starts or len(intervals) != len(call_order):
        raise AssertionError("System trace does not cover every timed invocation")
    samples = {"baseline": [], "optimized": []}
    for name, execution in zip(call_order, sorted(intervals)):
        windows = intervals[execution]
        elapsed = max(end for _, end in windows) - min(begin for begin, _ in windows)
        samples[name].append(elapsed / 1_000_000)
    baseline = timing_stats(samples["baseline"])
    optimized = timing_stats(samples["optimized"])
    return {
        "baseline": baseline,
        "optimized": optimized,
        "median_speedup": baseline["median_ms"] / optimized["median_ms"],
        "clock": "device nc_timestamp_ns, physical-core intervals merged per exec_id",
    }


def compare(old_model, new_model, inputs, expected, args) -> dict:
    old = old_model(*inputs).to("cpu")
    new = new_model(*inputs).to("cpu")
    for value in (old, new):
        if not torch.isfinite(value).all():
            raise AssertionError("Device kernel returned nonfinite values")
        torch.testing.assert_close(value, expected, rtol=1e-2, atol=1e-5)
    torch.testing.assert_close(new, old, rtol=0.0, atol=0.0)
    for _ in range(args.warmup):
        old_model(*inputs).to("cpu")
        new_model(*inputs).to("cpu")
    samples = {"baseline": [], "optimized": []}
    call_order = []
    if args.system_trace:
        from nrtpy._nrtpy import SystemTraceSession

        trace_context = SystemTraceSession()
    else:
        trace_context = nullcontext()
    trace_path = None
    device = None
    with trace_context as trace:
        for iteration in range(args.iterations):
            order = (
                (("baseline", old_model), ("optimized", new_model))
                if iteration % 2 == 0
                else (("optimized", new_model), ("baseline", old_model))
            )
            for name, model in order:
                started = time.perf_counter_ns()
                model(*inputs).to("cpu")
                samples[name].append((time.perf_counter_ns() - started) / 1_000_000)
                call_order.append(name)
        if args.system_trace:
            trace_path = args.output.with_name(
                args.output.stem + f"_trace_{args.trace_index}.json"
            )
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            events_json = trace.fetch_events_json()
            trace_path.write_text(events_json + "\n")
            device = device_timing(events_json, call_order)
            args.trace_index += 1
    baseline = timing_stats(samples["baseline"])
    optimized = timing_stats(samples["optimized"])
    return {
        "baseline_vs_cpu": metrics(old, expected),
        "optimized_vs_cpu": metrics(new, expected),
        "optimized_vs_baseline": metrics(new, old),
        "baseline": baseline,
        "optimized": optimized,
        "median_speedup": baseline["median_ms"] / optimized["median_ms"],
        "timing_order": "alternating baseline/optimized, reversed every iteration",
        "system_trace": str(trace_path) if trace_path else None,
        "system_trace_call_order": call_order if args.system_trace else None,
        "device": device,
    }


def run_mla(seq: int, baseline, args) -> dict:
    dtype = getattr(torch, args.dtype)
    generator = torch.Generator().manual_seed(741 + seq)
    cache_rows = args.cache_rows or args.topk + 129
    q = (torch.randn(seq, args.heads, args.latent, generator=generator) * 0.1).to(
        dtype
    )
    cache = (torch.randn(cache_rows, args.latent, generator=generator) * 0.1).to(
        dtype
    )
    indices = torch.randint(
        cache_rows, (seq, args.topk), generator=generator, dtype=torch.int32
    )
    indices[0, :128] = cache_rows - 1
    indices[0, 128:256] = -1
    if seq > 1:
        indices[-1, :] = -1
    scale = args.latent**-0.5
    if args.rope:
        if args.page_size:
            raise ValueError("The RoPE entry does not accept paged operands")
        q_pe = (
            torch.randn(seq, args.heads, args.rope, generator=generator) * 0.1
        ).to(dtype)
        k_pe = (torch.randn(cache_rows, args.rope, generator=generator) * 0.1).to(
            dtype
        )
        old_call = wrap_nki(baseline.mla_sparse_attention_rope_row_tiled_kernel)
        new_call = wrap_nki(mla.mla_sparse_attention_rope_row_tiled_kernel)

        def old_model(query, query_pe, keys, keys_pe, selected):
            return old_call(
                query, query_pe, keys, keys_pe, selected, scale, args.block_n, True
            )

        def new_model(query, query_pe, keys, keys_pe, selected):
            return new_call(
                query, query_pe, keys, keys_pe, selected, scale, args.block_n, True
            )

        host_inputs = (q, q_pe, cache, k_pe, indices)
    else:
        q_pe = k_pe = None
        old_call = wrap_nki(baseline.mla_sparse_attention_nope_row_tiled_kernel)
        new_call = wrap_nki(mla.mla_sparse_attention_nope_row_tiled_kernel)

        if args.page_size:
            if cache_rows % args.page_size or cache_rows < seq:
                raise ValueError("Paged cache rows must contain the written query rows")
            bank_rows = args.bank_rows or 2 * cache_rows
            if bank_rows % args.page_size or bank_rows < cache_rows:
                raise ValueError("Bank must contain at least as many full pages as window")
            bank = (torch.randn(bank_rows, args.latent, generator=generator) * 0.1).to(
                dtype
            )
            pages = cache_rows // args.page_size
            table = torch.randperm(
                bank_rows // args.page_size, generator=generator
            )[:pages].to(torch.int32).reshape(pages, 1)
            cache = bank.reshape(-1, args.page_size, args.latent)[
                table[:, 0].long()
            ].reshape(cache_rows, args.latent).clone()
            written = (
                torch.randn(seq, args.latent, generator=generator) * 0.1
            ).to(dtype)
            offset = torch.tensor([[cache_rows - seq]], dtype=torch.int32)
            cache[cache_rows - seq:] = written
            # Explicitly select freshly written rows as well as untouched pages.
            indices[0, 0] = cache_rows - seq

            def old_model(query, keys, selected, page_table, fresh, write_at):
                return old_call(
                    query, keys, selected, scale, page_table, fresh, write_at,
                    args.page_size, args.block_n, True,
                )

            def new_model(query, keys, selected, page_table, fresh, write_at):
                return new_call(
                    query, keys, selected, scale, page_table, fresh, write_at,
                    args.page_size, args.block_n, True,
                )

            host_inputs = (q, bank, indices, table, written, offset)
        else:
            def old_model(query, keys, selected):
                return old_call(
                    query, keys, selected, scale,
                    None, None, None, 0, args.block_n, True,
                )

            def new_model(query, keys, selected):
                return new_call(
                    query, keys, selected, scale,
                    None, None, None, 0, args.block_n, True,
                )

            host_inputs = (q, cache, indices)
    expected = mla.mla_sparse_attention_torch_oracle(
        q, cache, indices, scale, q_pe=q_pe, k_pe=k_pe
    )
    compiled_old = torch.compile(
        old_model, backend="neuron_libtorch", fullgraph=True, dynamic=False,
        options={"compiler_args": args.compiler_args},
    )
    compiled_new = torch.compile(
        new_model, backend="neuron_libtorch", fullgraph=True, dynamic=False,
        options={"compiler_args": args.compiler_args},
    )
    device_inputs = tuple(value.to("neuron:0") for value in host_inputs)
    result = compare(compiled_old, compiled_new, device_inputs, expected, args)
    profile_inputs = save_inputs(
        host_inputs, f"mla_s{seq}_k{args.topk}_page{args.page_size}", args
    )
    result.update({
        "kind": "mla",
        "seq": seq,
        "heads": args.heads,
        "latent": args.latent,
        "topk": args.topk,
        "rope": args.rope,
        "block_n": args.block_n,
        "dtype": args.dtype,
        "cache_rows": cache_rows,
        "page_size": args.page_size,
        "bank_rows": bank_rows if args.page_size else None,
        "profile_inputs": profile_inputs,
    })
    return result


def run_mhc(tokens: int, hidden: int, baseline, args) -> dict:
    generator = torch.Generator().manual_seed(748 + tokens + hidden)
    streams = mhc.MHC_STREAMS
    x = torch.randn(tokens, hidden, generator=generator)
    residual = torch.randn(tokens, streams, hidden, generator=generator)
    post = torch.rand(tokens, streams, 1, generator=generator)
    mix = torch.softmax(
        torch.randn(tokens, streams, streams, generator=generator), dim=-1
    )
    expected = mhc.hyper_connection_torch_oracle(x, residual, post, mix)
    old_call = wrap_nki(baseline.hyper_connection_kernel)
    new_call = wrap_nki(mhc.hyper_connection_kernel)

    def old_model(layer, residuals, post_mix, residual_mix):
        return old_call(layer, residuals, post_mix, residual_mix)

    def new_model(layer, residuals, post_mix, residual_mix):
        return new_call(layer, residuals, post_mix, residual_mix)

    compiled_old = torch.compile(
        old_model, backend="neuron_libtorch", fullgraph=True, dynamic=False,
        options={"compiler_args": args.compiler_args},
    )
    compiled_new = torch.compile(
        new_model, backend="neuron_libtorch", fullgraph=True, dynamic=False,
        options={"compiler_args": args.compiler_args},
    )
    inputs = tuple(value.to("neuron:0") for value in (x, residual, post, mix))
    result = compare(compiled_old, compiled_new, inputs, expected, args)
    profile_inputs = save_inputs(
        (x, residual, post, mix), f"mhc_t{tokens}_h{hidden}", args
    )
    result.update({
        "kind": "mhc",
        "tokens": tokens,
        "hidden": hidden,
        "hidden_tile": mhc.HIDDEN_TILE,
        "profile_inputs": profile_inputs,
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-mla", type=Path)
    parser.add_argument("--baseline-mhc", type=Path)
    parser.add_argument("--kind", choices=("mla", "mhc", "both"), default="both")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--seq", type=int, nargs="+", default=[1, 128])
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--latent", type=int, default=512)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--cache-rows", type=int)
    parser.add_argument("--page-size", type=int, default=0)
    parser.add_argument("--bank-rows", type=int)
    parser.add_argument("--rope", type=int, default=0)
    parser.add_argument("--block-n", type=int, default=512)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--mhc-tokens", type=int, nargs="+", default=[1, 128])
    parser.add_argument("--hidden", type=int, nargs="+", default=[4096, 7168])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--system-trace", action="store_true")
    parser.add_argument("--compiler-args", default=os.environ.get("NEURON_CC_FLAGS", ""))
    args = parser.parse_args()
    args.trace_index = 0
    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1":
        raise ValueError("Hardware benchmark cannot run in VLLM_NEURON_CPU_MODE=1")
    if not os.environ.get("NEURON_RT_VISIBLE_CORES"):
        raise ValueError("Pin NEURON_RT_VISIBLE_CORES before running the benchmark")
    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("Use nonnegative warmup and positive iteration counts")
    report = {
        "environment": {
            key: os.environ.get(key)
            for key in (
                "NEURON_RT_VISIBLE_CORES",
                "NEURON_RT_NUM_CORES",
                "NEURON_LOGICAL_NC_CONFIG",
                "NEURON_CC_FLAGS",
                "NEURON_PLATFORM_TARGET_OVERRIDE",
            )
        },
        "synchronization": "same-sized output copied to CPU every timed call",
        "compiler_args": args.compiler_args,
        "cases": [],
    }

    def record(case):
        report["cases"].append(case)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(case), flush=True)

    if args.kind in ("mla", "both"):
        if args.baseline_mla is None:
            parser.error("--baseline-mla is required for MLA")
        baseline = load_snapshot(args.baseline_mla, "_sparse_mla_baseline_snapshot")
        for seq in args.seq:
            record(run_mla(seq, baseline, args))
    if args.kind in ("mhc", "both"):
        if args.baseline_mhc is None:
            parser.error("--baseline-mhc is required for mHC")
        baseline = load_snapshot(args.baseline_mhc, "_mhc_baseline_snapshot")
        for tokens in args.mhc_tokens:
            for hidden in args.hidden:
                record(run_mhc(tokens, hidden, baseline, args))


if __name__ == "__main__":
    main()
