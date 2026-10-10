# SPDX-License-Identifier: Apache-2.0
"""One KDA layer's speculative verify step on Neuron: ``T`` chained launches vs the T-step kernel.

A verify step hands each request ``T = 1 + k`` tokens. ``chain`` is what the model
would do without the T-step kernel: ``T`` launches of :func:`kda_fused_decode` per
layer, each fed the carriers the previous one wrote. ``tstep`` is one
:func:`kda_fused_decode_tstep` launch per layer, which writes the ``T`` per-token
checkpoints of both carriers. ``one_token`` is the plain decode step (one
:func:`kda_fused_decode` per layer on the first token of each request), the number
the decode microbenchmark (``benchmark_kda_decode.py``, ``after``) reported, so a
change to the shared kernel body shows up here as a change to that line.

Each timed graph holds ``L`` independent layers (own carriers, shared weights),
because a layer runs inside a decode graph that holds 34 of them; per-layer times
are graph time / ``L``. ``floor`` is the empty graph with the same inputs and
outputs: the launch and synchronisation cost every graph time contains.

The entitlement of a step is the kernel ledger's roofline (``docs/kernel_ledger.md``):
the larger of its HBM bytes (operands, the carriers it reads, the checkpoints it
writes) at 716 GB/s, its arithmetic at 79 TFLOP/s, and the 2 us floor of one
LNC2 core. The kernel's per-token recurrence is a dependency chain, so the step's
second yardstick is ``T`` times the one-token step's above-floor time. Both ratios
are in the JSON.

Shapes are the TP=64 decode shapes: one KDA head per rank, ``K = V = 128``, a
4-tap conv over 384 channels, a bfloat16 ``SD`` conv carrier, a float32 recurrent
carrier, ``gate_lower_bound = -5``. References run on CPU.

Set the Neuron core and LNC environment before launching (``devlease.py`` does);
this script refuses to run without ``NEURON_RT_VISIBLE_CORES``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
if sys.path[:1] != [str(REPO)]:
    sys.path.insert(0, str(REPO))

import vllm_neuron  # noqa: E402,F401 -- registers the Neuron compilation backend

from vllm_neuron.functional.kda.fused_decode import (  # noqa: E402
    fused_decode_grid,
    kda_fused_decode,
    kda_fused_decode_torch_reference,
    kda_fused_decode_tstep,
    kda_fused_decode_tstep_torch_reference,
)

HEADS = 1
KDIM = 128
TAPS = 4
LOWER = -5.0
STEP_KEYS = ("q_in", "k_in", "v_in", "raw_gate", "raw_beta")
WEIGHT_KEYS = ("q_conv1d_weight", "k_conv1d_weight", "v_conv1d_weight", "A_log", "dt_bias")
#: The kernel ledger's roofline basis for one LNC2 core (``docs/kernel_ledger.md``):
#: ``max(FLOPs / 79 TFLOP/s, HBM bytes / 716 GB/s, 2 us)``.
DEFAULT_HBM_GBPS = 716.0
PEAK_TFLOPS = 79.0
ROOFLINE_FLOOR_US = 2.0


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    a = actual.float().flatten()
    e = expected.float().flatten()
    diff = (a - e).abs()
    return {
        "bit_equal": bool(torch.equal(
            actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
        )),
        "max_abs": float(diff.max()),
        "relative_l2": float(diff.norm() / e.norm().clamp_min(1e-30)),
    }


def make_case(batch: int, tokens: int, layers: int, seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed)
    width = HEADS * KDIM
    channels = 3 * width
    rows = batch * tokens
    weights = {
        key: (torch.randn(width, 1, TAPS, generator=gen) * 0.5).to(torch.bfloat16)
        for key in WEIGHT_KEYS[:3]
    }
    weights["A_log"] = torch.log(torch.rand(HEADS, generator=gen) * 15 + 1)
    weights["dt_bias"] = torch.randn(width, generator=gen) * 0.5
    step = {
        "q_in": torch.randn(layers, rows, width, generator=gen),
        "k_in": torch.randn(layers, rows, width, generator=gen),
        "v_in": torch.randn(layers, rows, width, generator=gen),
        "raw_gate": torch.randn(layers, rows, width, generator=gen) * 2,
        "raw_beta": torch.randn(layers, rows, HEADS, generator=gen),
    }
    carriers = {
        "conv": torch.randn(layers, batch, TAPS - 1, channels, generator=gen).to(torch.bfloat16),
        "rec": torch.randn(layers, batch, HEADS, KDIM, KDIM, generator=gen) * 0.5,
    }
    position = torch.arange(1, batch + 1, dtype=torch.int32) * 17
    return {"weights": weights, "step": step, "carriers": carriers, "position": position,
            "batch": batch, "tokens": tokens, "layers": layers}


def step_bytes(batch: int, tokens: int) -> dict:
    """HBM bytes one layer's T-step must move: operands in, carriers in, checkpoints out."""
    width = HEADS * KDIM
    conv_row = (TAPS - 1) * 3 * width * 2
    rec_row = HEADS * KDIM * KDIM * 4
    reads = {
        "tokens": batch * tokens * (4 * width + HEADS) * 4,
        "carriers": batch * (conv_row + rec_row),
        "weights": 3 * width * TAPS * 2 + (HEADS + width) * 4,
        "position": batch * 4,
    }
    writes = {
        "core": batch * tokens * width * 4,
        "checkpoints": batch * tokens * (conv_row + rec_row),
    }
    return {"reads": reads, "writes": writes,
            "total": sum(reads.values()) + sum(writes.values())}


def step_flops(batch: int, tokens: int) -> int:
    """Arithmetic of one layer's T-step: the conv taps and the ``K x V`` state update."""
    width = HEADS * KDIM
    per_token = 3 * width * TAPS * 2 + 8 * HEADS * KDIM * KDIM
    return batch * tokens * per_token


def build_variants(layers: int, batch: int, tokens: int):
    """Graphs returning ``(core, conv, rec)``; the carriers leave as outputs."""

    def unpack(args):
        step = dict(zip(STEP_KEYS, args[0:5]))
        conv, rec, position = args[5], args[6], args[7]
        weights = dict(zip(WEIGHT_KEYS, args[8:13]))
        return step, conv, rec, position, weights

    def token_rows(step, layer, t):
        return tuple(
            step[key][layer].reshape(batch, tokens, -1)[:, t] for key in STEP_KEYS
        )

    def one_token(*args):
        step, conv, rec, position, weights = unpack(args)
        cores, convs, recs = [], [], []
        for layer in range(layers):
            out = kda_fused_decode(
                *token_rows(step, layer, 0), conv_state=conv[layer],
                recurrent_state=rec[layer], gate_lower_bound=LOWER,
                conv_state_dim_first=False, start_position=position, **weights,
            )
            cores.append(out.core)
            convs.append(out.conv_state)
            recs.append(out.recurrent_state)
        return torch.stack(cores), torch.stack(convs), torch.stack(recs)

    def chain(*args):
        step, conv, rec, position, weights = unpack(args)
        cores, convs, recs = [], [], []
        for layer in range(layers):
            conv_t, rec_t = conv[layer], rec[layer]
            per_token, conv_ck, rec_ck = [], [], []
            for t in range(tokens):
                out = kda_fused_decode(
                    *token_rows(step, layer, t), conv_state=conv_t,
                    recurrent_state=rec_t, gate_lower_bound=LOWER,
                    conv_state_dim_first=False, start_position=position + t, **weights,
                )
                conv_t, rec_t = out.conv_state, out.recurrent_state
                per_token.append(out.core)
                conv_ck.append(conv_t)
                rec_ck.append(rec_t)
            cores.append(torch.stack(per_token, dim=1).reshape(batch * tokens, -1))
            convs.append(torch.stack(conv_ck, dim=1))
            recs.append(torch.stack(rec_ck, dim=1))
        return torch.stack(cores), torch.stack(convs), torch.stack(recs)

    def tstep(*args):
        step, conv, rec, position, weights = unpack(args)
        cores, convs, recs = [], [], []
        for layer in range(layers):
            out = kda_fused_decode_tstep(
                *(step[key][layer] for key in STEP_KEYS), conv_state=conv[layer],
                recurrent_state=rec[layer], gate_lower_bound=LOWER,
                conv_state_dim_first=False, start_position=position, **weights,
            )
            cores.append(out.core)
            convs.append(out.conv_checkpoints)
            recs.append(out.recurrent_checkpoints)
        return torch.stack(cores), torch.stack(convs), torch.stack(recs)

    def floor(*args):
        return args[0] + 0.0, args[5].clone(), args[6].clone()

    return {"one_token": one_token, "chain": chain, "tstep": tstep, "floor": floor}


def cpu_reference(case: dict) -> dict:
    batch, tokens, layers = case["batch"], case["tokens"], case["layers"]
    position = case["position"].reshape(1, -1)
    common = dict(gate_lower_bound=LOWER, conv_state_dim_first=False,
                  start_position=position, **case["weights"])
    one = {"core": [], "conv": [], "rec": []}
    full = {"core": [], "conv": [], "rec": []}
    for layer in range(layers):
        carriers = dict(conv_state=case["carriers"]["conv"][layer],
                        recurrent_state=case["carriers"]["rec"][layer])
        first = kda_fused_decode_torch_reference(
            *(case["step"][key][layer].reshape(batch, tokens, -1)[:, 0] for key in STEP_KEYS),
            **carriers, **common,
        )
        one["core"].append(first.core)
        one["conv"].append(first.conv_state)
        one["rec"].append(first.recurrent_state)
        out = kda_fused_decode_tstep_torch_reference(
            *(case["step"][key][layer] for key in STEP_KEYS), **carriers, **common,
        )
        full["core"].append(out.core)
        full["conv"].append(out.conv_checkpoints)
        full["rec"].append(out.recurrent_checkpoints)
    return {
        "one_token": {k: torch.stack(v) for k, v in one.items()},
        "tstep": {k: torch.stack(v) for k, v in full.items()},
    }


def device_inputs(case: dict, device: str) -> tuple:
    step = tuple(case["step"][key].to(device) for key in STEP_KEYS)
    carriers = (
        case["carriers"]["conv"].clone().to(device),
        case["carriers"]["rec"].clone().to(device),
        case["position"].to(device),
    )
    weights = tuple(case["weights"][key].to(device) for key in WEIGHT_KEYS)
    return step + carriers + weights


def measure(model, inputs, warmup: int, iterations: int, layers: int) -> dict:
    for _ in range(warmup):
        model(*inputs)[0].to("cpu")
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        model(*inputs)[0].to("cpu")
        samples.append((time.perf_counter_ns() - started) / 1_000)
    samples.sort()
    p90 = samples[min(len(samples) - 1, int(0.9 * len(samples)))]
    median = statistics.median(samples)
    return {
        "iterations": iterations, "warmup": warmup,
        "graph_median_us": median, "graph_p90_us": p90,
        "graph_min_us": samples[0], "graph_max_us": samples[-1],
        "median_us": median / layers, "p90_us": p90 / layers, "samples_us": samples,
    }


def run_case(batch: int, tokens: int, layers: int, args) -> dict:
    torch._dynamo.reset()
    case = make_case(batch, tokens, layers, seed=9700 + 13 * batch + 7 * tokens + layers)
    reference = cpu_reference(case)
    compiled = {
        name: torch.compile(
            fn, backend="neuron_libtorch", fullgraph=True, dynamic=False,
            options={"compiler_args": os.environ.get("NEURON_CC_FLAGS", "")},
        )
        for name, fn in build_variants(layers, batch, tokens).items()
    }
    result = {"B": batch, "T": tokens, "layers": layers, "variants": {},
              "bytes_per_layer_step": step_bytes(batch, tokens)}
    outputs = {}
    for name, model in compiled.items():
        inputs = device_inputs(case, "neuron:0")
        started = time.perf_counter()
        returned = model(*inputs)
        core = returned[0].to("cpu")
        first_call_s = time.perf_counter() - started
        conv, rec = returned[1].to("cpu"), returned[2].to("cpu")
        if not torch.isfinite(core).all() or not torch.isfinite(rec).all():
            raise AssertionError(f"{name} returned non-finite values")
        outputs[name] = {"core": core, "conv": conv, "rec": rec}
        result["variants"][name] = {"first_call_s": first_call_s}
        expected = reference.get("tstep" if name == "chain" else name)
        if expected is None:
            continue
        against = {key: metrics(outputs[name][key], expected[key]) for key in expected}
        result["variants"][name]["vs_cpu_reference"] = against
        for key, m in against.items():
            if m["relative_l2"] > 1e-3:
                raise AssertionError(f"{name} {key} differs from the CPU reference: {m}")
    # The T-step kernel against the chained launches: the same instructions on the
    # same operands, so the carriers are expected bit-equal on the simulator; the
    # device result is recorded either way.
    result["tstep_vs_chain"] = {
        key: metrics(outputs["tstep"][key], outputs["chain"][key]) for key in ("core", "conv", "rec")
    }
    for key, m in result["tstep_vs_chain"].items():
        if m["relative_l2"] > 1e-3:
            raise AssertionError(f"tstep {key} differs from the chain: {m}")
    for name, model in compiled.items():
        inputs = device_inputs(case, "neuron:0")
        result["variants"][name]["timing"] = measure(
            model, inputs, args.warmup, args.iterations, layers
        )
    timing = {name: result["variants"][name]["timing"] for name in compiled}
    floor_us = timing["floor"]["graph_median_us"]
    above = {name: (timing[name]["graph_median_us"] - floor_us) / layers
             for name in compiled if name != "floor"}
    bytes_us = result["bytes_per_layer_step"]["total"] / (args.hbm_gbps * 1e3)
    flops_us = step_flops(batch, tokens) / (PEAK_TFLOPS * 1e6)
    entitlement_us = max(bytes_us, flops_us, ROOFLINE_FLOOR_US)
    result.update({
        "launch_floor_us": floor_us,
        "above_floor_per_layer_median_us": above,
        "one_token_median_us": timing["one_token"]["median_us"],
        "chain_median_us": timing["chain"]["median_us"],
        "tstep_median_us": timing["tstep"]["median_us"],
        "tstep_over_T_one_token": timing["tstep"]["median_us"] / (tokens * timing["one_token"]["median_us"]),
        "tstep_over_chain": timing["tstep"]["median_us"] / timing["chain"]["median_us"],
        "tstep_above_floor_over_T_one_token_above_floor": above["tstep"] / (tokens * above["one_token"]),
        "roofline_basis": {"hbm_gbps": args.hbm_gbps, "peak_tflops": PEAK_TFLOPS,
                           "floor_us": ROOFLINE_FLOOR_US, "flops": step_flops(batch, tokens),
                           "bytes_us": bytes_us, "flops_us": flops_us},
        "entitlement_us": entitlement_us,
        "tstep_above_floor_over_entitlement": above["tstep"] / entitlement_us,
        "one_token_above_floor_over_entitlement": above["one_token"] / max(
            step_bytes(batch, 1)["total"] / (args.hbm_gbps * 1e3),
            step_flops(batch, 1) / (PEAK_TFLOPS * 1e6), ROOFLINE_FLOOR_US),
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--tokens", type=int, default=4, help="T = 1 + k tokens per request")
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 34])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--hbm-gbps", type=float, default=DEFAULT_HBM_GBPS,
                        help="HBM bandwidth of the leased cores, for the bytes entitlement")
    args = parser.parse_args()
    if os.environ.get("VLLM_NEURON_CPU_MODE") == "1":
        raise ValueError("Hardware benchmark cannot run in VLLM_NEURON_CPU_MODE=1")
    if not os.environ.get("NEURON_RT_VISIBLE_CORES"):
        raise ValueError("Pin NEURON_RT_VISIBLE_CORES before running the benchmark")
    if args.warmup < 0 or args.iterations < 1 or args.tokens < 2:
        raise ValueError("Use nonnegative warmup, positive iterations and T >= 2")
    if any(b < 1 for b in args.batches) or any(n < 1 for n in args.layers):
        raise ValueError("Batches and layer counts must be positive")
    args.output = args.output.resolve()
    cache_root = os.environ.get("NEURON_LIBTORCH_CACHE_ROOT")
    workdir = (Path(cache_root) / "benchmark_kda_tstep_workdir" if cache_root
               else Path(tempfile.mkdtemp(prefix="benchmark_kda_tstep_")))
    workdir.mkdir(parents=True, exist_ok=True)
    os.chdir(workdir)
    report = {
        "environment": {
            key: os.environ.get(key)
            for key in ("NEURON_RT_VISIBLE_CORES", "NEURON_RT_NUM_CORES",
                        "NEURON_LOGICAL_NC_CONFIG", "NEURON_CC_FLAGS",
                        "NEURON_PLATFORM_TARGET_OVERRIDE", "NEURON_LIBTORCH_CACHE_ROOT")
        },
        "vllm_neuron": str(Path(vllm_neuron.__file__).resolve().parent),
        "fused_grid": list(fused_decode_grid(KDIM)),
        "shapes": {
            "heads_per_rank": HEADS, "head_dim": KDIM, "conv_taps": TAPS,
            "conv_channels": 3 * HEADS * KDIM,
            "conv_carrier": "bfloat16 [B, 3, 384] (SD)",
            "recurrent_carrier": "float32 [B, 1, 128, 128]",
        },
        "timing_unit": (
            "microseconds per layer step = host wall time of one graph launch plus its "
            "core copy to CPU, divided by the graph's layer count"
        ),
        "compiler_workdir": str(workdir),
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for layers in args.layers:
        for batch in args.batches:
            case = run_case(batch, args.tokens, layers, args)
            report["cases"].append(case)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            summary = {k: v for k, v in case.items() if k != "variants"}
            summary["timing_us"] = {
                name: {k: v for k, v in variant["timing"].items() if k != "samples_us"}
                for name, variant in case["variants"].items()
            }
            print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
