# SPDX-License-Identifier: Apache-2.0
"""Compare dense and padding-aware routed experts on one isolated NeuronCore.

Run with the campaign venv and PYTHONPATH set to this checkout. Device mode
requires NEURON_RT_VISIBLE_CORES to be explicitly set. CPU mode is a simulator
correctness check at small dimensions, not a performance measurement.
"""

import argparse
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path

# examples/vllm_neuron is also a package. Direct script execution must resolve
# the implementation from this checkout before that examples package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_case(torch, scenario, small=False):
    from vllm_neuron.functional.moe.fused_fp8_pack import pack_experts

    torch.manual_seed(20261003)
    h, intermediate, experts = (256, 128, 3) if small else (4096, 512, 18)
    decode = scenario.startswith("decode")
    tokens = 1 if decode else (64 if small else 1024)
    width = 1 if decode else (32 if small else 256)
    blocks = min(8, experts) if decode else (5 if small else 49)
    hidden = (torch.randn(tokens + 1, h) / 4).to(torch.bfloat16)
    hidden[-1].zero_()
    gate_up = (torch.randn(experts, h, 2 * intermediate) / 16).to(
        torch.float8_e4m3fn
    )
    down = (torch.randn(experts, intermediate, h) / 16).to(torch.float8_e4m3fn)
    gu_scale = 0.5 + torch.rand(experts, h // 128, 2, intermediate // 128)
    down_scale = 0.5 + torch.rand(experts, intermediate // 128, h // 128)
    packed = pack_experts(gate_up, down, gu_scale, down_scale)
    row_ids = torch.full((blocks, width), -1, dtype=torch.int32)
    expert_ids = torch.zeros(blocks, 1, dtype=torch.int32)
    affinity = torch.zeros(tokens + 1, experts)

    if scenario == "balanced":
        # About T*8/16 valid local memberships for the recorded EP16 workload.
        memberships = tokens // 2
        for expert in range(experts):
            count = memberships // experts + int(expert < memberships % experts)
            rows = (torch.arange(count) * 13 + expert * 29) % tokens
            row_ids[expert, :count] = rows
            expert_ids[expert, 0] = expert
            affinity[rows, expert] = 0.25 + torch.rand(count) / 2
    elif scenario == "skew":
        for block in range((tokens + width - 1) // width):
            start = block * width
            rows = torch.arange(start, min(start + width, tokens))
            row_ids[block, :rows.numel()] = rows
        affinity[:tokens, 0] = 0.5
    elif scenario == "full":
        # Maximum top-8 memberships, distributed across full expert blocks.
        # Even this case includes unused capacity blocks from the row bucket.
        blocks_per_expert = (tokens + width - 1) // width
        routed_experts = min(8, experts, blocks // blocks_per_expert)
        block = 0
        for expert in range(routed_experts):
            for start in range(0, tokens, width):
                rows = torch.arange(start, min(start + width, tokens))
                row_ids[block, :rows.numel()] = rows
                expert_ids[block, 0] = expert
                block += 1
            affinity[:tokens, expert] = 1 / routed_experts
    elif scenario == "full_mixed":
        # Max top-8 routing with all 49 capacity blocks active: full blocks
        # alternate with small expert tails, stressing pair-union decisions.
        counts = [33, 33, 17] if small else [525] * 13 + [273, 273, 273, 274, 274]
        block, token_offset = 0, 0
        for expert, count in enumerate(counts):
            rows = (torch.arange(count) + token_offset) % tokens
            token_offset += count
            for start in range(0, count, width):
                chunk = rows[start:start + width]
                row_ids[block, :chunk.numel()] = chunk
                expert_ids[block, 0] = expert
                block += 1
            affinity[rows, expert] = 1 / min(8, experts)
        assert block == blocks
        if not small:
            assert token_offset == tokens * 8
            assert torch.all(torch.bincount(row_ids[row_ids >= 0].long()) == 8)
    elif scenario == "capacity_full":
        # Kernel-level control-flow diagnostic: every capacity row is valid.
        # This intentionally exceeds a model router's top-8 membership count.
        for block in range(blocks):
            expert = block % experts
            row_ids[block] = torch.arange(width) % tokens
            expert_ids[block, 0] = expert
        affinity[:tokens].fill_(0.25)
    elif scenario in ("holes", "tail"):
        routed = list(enumerate([2, experts - 1, 0]))
        if scenario == "tail":
            routed = [(blocks - 1, experts - 1)]
        for block, expert in routed:
            positions = sorted({
                0, width - 1, width // 2,
                *[i for i in [31, 32, 63, 64, 95, 96, 127, 128, 159, 160,
                              191, 192, 223, 224, 255] if i < width],
            })
            if block == 1:
                positions = positions[::2]
            rows = (torch.tensor(positions) * 13 + block * 17) % tokens
            row_ids[block, positions] = rows.to(torch.int32)
            expert_ids[block, 0] = expert
            affinity[rows, expert] = 0.375
    elif decode:
        active = (0 if scenario == "decode_zero"
                  else 1 if scenario == "decode_sparse" else blocks)
        row_ids[:active, 0] = 0
        expert_ids[:, 0] = torch.arange(blocks)
        if active:
            affinity[0, :active] = 1 / active
    elif scenario != "zero":
        raise ValueError(scenario)
    bounds = torch.tensor([10.0, -10.0, 10.0]).repeat(128, 1)
    args = (hidden, packed.weights, packed.scales, row_ids, expert_ids,
            affinity.reshape(-1, 1), bounds)
    return args, (gate_up.float(), down.float(), gu_scale, down_scale)


def reference(torch, args, source):
    hidden, _, _, ids, expert_ids, affinity_flat, bounds = args
    gate_up, down, gu_scale, down_scale = source
    affinity = affinity_flat.reshape(hidden.shape[0], down.shape[0])
    intermediate = down.shape[1]
    result = torch.zeros(*ids.shape, hidden.shape[1])
    for block in range(ids.shape[0]):
        valid = ids[block] >= 0
        rows = ids[block, valid].long()
        if not rows.numel():
            continue
        expert = int(expert_ids[block, 0])
        x = hidden[rows].float()
        gu = torch.zeros(rows.numel(), 2 * intermediate)
        for k in range(hidden.shape[1] // 128):
            product = x[:, k * 128:(k + 1) * 128] @ gate_up[
                expert, k * 128:(k + 1) * 128
            ]
            gu += product * gu_scale[expert, k].reshape(-1).repeat_interleave(128)
        gate, up = gu.split(intermediate, dim=1)
        activated = (torch.nn.functional.silu(gate.clamp(max=float(bounds[0, 0])))
                     * up.clamp(min=float(bounds[0, 1]),
                                max=float(bounds[0, 2]))).to(torch.bfloat16).float()
        contribution = torch.zeros(rows.numel(), hidden.shape[1])
        for k in range(intermediate // 128):
            contribution += (
                activated[:, k * 128:(k + 1) * 128]
                @ down[expert, k * 128:(k + 1) * 128]
            ) * down_scale[expert, k].repeat_interleave(128)
        result[block, valid] = contribution * affinity[rows, expert, None]
    return result


def device_latencies(events_json, call_order):
    """Merge physical core execution intervals for each logical invocation."""
    starts, executions = {}, {}
    for event in json.loads(events_json)["events"]:
        if event["event_type"] != "nc_exec_running":
            continue
        key = (event["nc_idx"], event["tracking_id"])
        if event["phase"] == "start":
            starts[key] = event
        elif event["phase"] == "stop":
            start = starts.pop(key)
            executions.setdefault(start["data"]["exec_id"], []).append((
                start["data"]["nc_timestamp_ns"],
                event["data"]["nc_timestamp_ns"],
            ))
    if starts or len(executions) != len(call_order):
        raise AssertionError("System trace does not cover exactly the timed calls")
    samples = {name: [] for name in set(call_order)}
    for name, execution in zip(call_order, sorted(executions)):
        total, previous_end = 0, 0
        for begin, end in sorted(executions[execution]):
            total += max(0, end - max(begin, previous_end))
            previous_end = max(previous_end, end)
        samples[name].append(total / 1_000_000)
    medians = {name: statistics.median(values) for name, values in samples.items()}
    return {
        "median_ms": medians,
        "samples_ms": samples,
        "speedup": medians["dense"] / medians["padding_aware"],
        "clock": "union of physical nc_exec_running intervals per exec_id",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["cpu", "device"], default="cpu")
    parser.add_argument("--scenario", choices=[
        "balanced", "skew", "full", "full_mixed", "capacity_full", "holes", "tail", "zero", "decode",
        "decode_sparse", "decode_zero", "all"
    ], default="all")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--baseline-kernel-file", type=Path)
    parser.add_argument("--candidate-kernel-file", type=Path)
    parser.add_argument("--include-ablation", action="store_true")
    parser.add_argument("--system-trace", action="store_true")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    if options.mode == "device" and not os.environ.get("NEURON_RT_VISIBLE_CORES"):
        parser.error("Device mode requires an explicitly reserved NEURON_RT_VISIBLE_CORES")
    if options.mode == "device" and options.baseline_kernel_file is None:
        parser.error("Device mode requires --baseline-kernel-file from the untouched checkout")
    if options.mode == "cpu":
        os.environ["VLLM_NEURON_CPU_MODE"] = "1"
        os.environ["NEURON_LIBTORCH_CPU_MODE"] = "1"
        os.environ["NKI_SIMULATOR"] = "1"
    else:
        os.environ.pop("VLLM_NEURON_CPU_MODE", None)
        os.environ.pop("NEURON_LIBTORCH_CPU_MODE", None)
        os.environ.pop("NKI_SIMULATOR", None)
        os.environ["NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS"] = "0"
        # The outer FX/NEFF cache can retain a function's cache identity across
        # NKI source edits. Isolate every source revision as well as its BIR cache.
        source_file = options.candidate_kernel_file or (
            Path(__file__).resolve().parents[1]
            / "vllm_neuron/functional/moe/moe_fused_fp8.py"
        )
        source_digest = hashlib.sha256(source_file.read_bytes()).hexdigest()
        cache_root = os.environ.get("NEURON_LIBTORCH_CACHE_ROOT")
        if cache_root:
            revision_cache = str(Path(cache_root) / ("source-" + source_digest[:16]))
            os.environ["NEURON_LIBTORCH_CACHE_ROOT"] = revision_cache
            os.environ["NKI_COMPILE_CACHE_URL"] = revision_cache + "-nki"
            os.environ["NKI_TRACE_CACHE_URL"] = revision_cache + "-trace"
            os.environ["NKI_DISABLE_TRACE_CACHE"] = "1"
            Path(revision_cache).mkdir(parents=True, exist_ok=True)
            (Path(revision_cache) / "moe_fused_fp8.py").write_bytes(
                source_file.read_bytes()
            )

    import torch
    import nki
    import libtorch_neuronx_lite
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
    from vllm_neuron.functional.moe.moe_fused_fp8 import moe_fused_fp8_kernel

    torch.set_num_threads(8)
    if options.candidate_kernel_file is not None:
        spec = importlib.util.spec_from_file_location(
            "_moe_fp8_profile_candidate", options.candidate_kernel_file
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        moe_fused_fp8_kernel = module.moe_fused_fp8_kernel
    kernel = wrap_nki(moe_fused_fp8_kernel)[2]
    baseline_kernel = None
    if options.baseline_kernel_file is not None:
        spec = importlib.util.spec_from_file_location(
            "_moe_fp8_profile_baseline", options.baseline_kernel_file
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        baseline_kernel = wrap_nki(module.moe_fused_fp8_kernel)[2]
    scenarios = ["balanced", "skew", "full", "full_mixed", "holes", "tail", "zero", "decode",
                 "decode_sparse", "decode_zero"]
    if options.scenario != "all":
        scenarios = [options.scenario]
    reports = []
    for scenario in scenarios:
        args, source = make_case(torch, scenario, options.mode == "cpu")
        if options.input_dir is not None:
            input_dir = options.input_dir / scenario
            input_dir.mkdir(parents=True, exist_ok=True)
            for index, value in enumerate(args):
                (input_dir / f"input{index}.bin").write_bytes(
                    value.contiguous().view(torch.uint8).numpy().tobytes()
                )
        expected = reference(torch, args, source)
        width = args[3].shape[1]

        def dense(*tensors):
            if baseline_kernel is not None:
                return baseline_kernel(*tensors, BLOCK_M=width, BLOCK_N=4096,
                                       BLOCK_K=4096)
            return kernel(*tensors, BLOCK_M=width, BLOCK_N=4096, BLOCK_K=4096,
                          SKIP_PADDING=False)

        def ablation(*tensors):
            return kernel(*tensors, BLOCK_M=width, BLOCK_N=4096, BLOCK_K=4096,
                          SKIP_PADDING=False)

        def sparse(*tensors):
            return kernel(*tensors, BLOCK_M=width, BLOCK_N=4096, BLOCK_K=4096,
                          SKIP_PADDING=True)

        functions = {"dense": dense, "padding_aware": sparse}
        if options.include_ablation:
            functions["refactored_dense_ablation"] = ablation
        if options.mode == "device":
            functions = {
                name: torch.compile(
                    fn, backend="neuron_libtorch", fullgraph=True,
                    options={"compiler_args": os.environ.get("NEURON_CC_FLAGS", "")},
                )
                for name, fn in functions.items()
            }
            call_args = tuple(t.to("neuron:0") for t in args)
        else:
            call_args = args
        outputs = {name: fn(*call_args).cpu() for name, fn in functions.items()}
        baseline_comparison = {}
        oracle_metrics = {}
        for name, actual in outputs.items():
            torch.testing.assert_close(
                actual, outputs["dense"],
                rtol=0.0 if options.mode == "device" else 1e-5,
                atol=0.0 if options.mode == "device" else 1e-7,
            )
            bitwise_equal = torch.equal(
                actual.view(torch.int32), outputs["dense"].view(torch.int32)
            )
            if options.mode == "device":
                assert bitwise_equal, (scenario, name, "FP32 output bits differ")
            baseline_difference = actual - outputs["dense"]
            baseline_comparison[name] = {
                "max_abs_difference": float(baseline_difference.abs().max()),
                "relative_l2": float(
                    baseline_difference.norm()
                    / outputs["dense"].norm().clamp_min(1e-30)
                ),
                "bitwise_equal": bitwise_equal,
            }
            residual = actual - expected
            expected_norm = expected.norm().clamp_min(1e-30)
            relative_l2 = float(residual.norm() / expected_norm)
            valid = args[3] >= 0
            selected_actual = actual[valid].double().reshape(-1)
            selected_expected = expected[valid].double().reshape(-1)
            if selected_expected.numel():
                cosine = float(
                    (selected_actual * selected_expected).sum()
                    / (selected_actual.norm() * selected_expected.norm()).clamp_min(1e-30)
                )
            else:
                cosine = 1.0
            oracle_metrics[name] = {
                "max_abs_difference": float(residual.abs().max()),
                "relative_l2": relative_l2,
                "cosine_similarity": cosine,
            }
            # Device SiLU followed by BF16 conversion can round near a BF16
            # boundary differently from Torch's CPU SiLU, including in the
            # untouched kernel. Hardware-to-hardware comparison above is exact.
            torch.testing.assert_close(
                actual, expected, rtol=3e-2,
                atol=2e-3 if options.mode == "device" else 1e-5,
            )
            assert relative_l2 < 1e-4, (scenario, name, oracle_metrics[name])
            assert torch.count_nonzero(actual[args[3] < 0]) == 0
        actual = outputs["padding_aware"]
        difference = actual - expected
        baseline_difference = actual - outputs["dense"]
        baseline_norm = outputs["dense"].norm().clamp_min(1e-30)
        report = {
            "scenario": scenario,
            "mode": options.mode,
            "hidden": args[0].shape[1],
            "intermediate": source[1].shape[1],
            "experts": source[1].shape[0],
            "capacity_rows": args[3].numel(),
            "valid_rows": int((args[3] >= 0).sum()),
            "max_memberships_per_token": int(
                torch.bincount(args[3][args[3] >= 0].long(), minlength=args[0].shape[0] - 1).max()
            ),
            "max_abs_error": float(difference.abs().max()),
            "relative_l2_error": float(difference.norm() / expected.norm().clamp_min(1e-30)),
            "baseline_max_abs_difference": float(baseline_difference.abs().max()),
            "baseline_relative_l2_difference": float(baseline_difference.norm() / baseline_norm),
            "baseline_bitwise_equal": baseline_comparison["padding_aware"]["bitwise_equal"],
            "baseline_comparison": baseline_comparison,
            "cpu_oracle_metrics": oracle_metrics,
            "nki_version": nki.__version__,
            "libtorch_neuronx_lite_version": libtorch_neuronx_lite.__version__,
            "compiler_flags": os.environ.get("NEURON_CC_FLAGS", ""),
            "visible_cores": os.environ.get("NEURON_RT_VISIBLE_CORES", ""),
            "baseline_kernel_file": str(options.baseline_kernel_file),
            "candidate_kernel_file": str(options.candidate_kernel_file),
            "compile_cache": os.environ.get("NEURON_LIBTORCH_CACHE_ROOT", ""),
        }
        if options.mode == "device" and options.runs > 0:
            # libtorch schedules asynchronously. Copy one FP32 element to wait
            # for each result; profile nc_exec_running is the device-only metric.
            def invoke_and_wait(fn):
                result = fn(*call_args)
                result.reshape(-1)[:1].cpu()

            timings = {name: [] for name in functions}
            for _ in range(options.warmup):
                for fn in functions.values():
                    invoke_and_wait(fn)
            if options.system_trace:
                from nrtpy._nrtpy import SystemTraceSession
                trace_context = SystemTraceSession()
            else:
                trace_context = nullcontext()
            call_order = []
            with trace_context as trace:
                for iteration in range(options.runs):
                    order = list(functions.items())
                    if iteration % 2:
                        order.reverse()
                    for name, fn in order:
                        start = time.perf_counter()
                        invoke_and_wait(fn)
                        timings[name].append((time.perf_counter() - start) * 1000)
                        call_order.append(name)
                if options.system_trace:
                    events_json = trace.fetch_events_json()
                    report["device_timing"] = device_latencies(events_json, call_order)
                    if options.output is not None:
                        trace_file = options.output.with_name(
                            options.output.stem + f"-{scenario}-trace.json"
                        )
                        trace_file.write_text(events_json + "\n")
                        report["system_trace_file"] = str(trace_file)
            report["median_ms"] = {
                name: statistics.median(samples) for name, samples in timings.items()
            }
            report["speedup"] = report["median_ms"]["dense"] / report["median_ms"]["padding_aware"]
            report["timing_includes"] = "host dispatch and one FP32 element copy for synchronization"
        reports.append(report)
        print(json.dumps(report), flush=True)
    if options.output:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
