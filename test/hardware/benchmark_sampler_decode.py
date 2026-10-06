# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash decode sampling: the host path at 5938748 against the on-device sampler.

before (5938748, as served): the decode graph returns ``[B, 154880]`` bf16 logits; the
    runner copies them to the host and vLLM's own ``Sampler`` picks the tokens there,
    on one torch thread as vLLM's multiproc executor runs each worker.
after (this tree, ``VLLM_NEURON_GLM5NEXT_ON_DEVICE_SAMPLING=1``): the root hands the
    same logits to ``sample_full_vocab`` inside its graph and only ``[B]`` int32 tokens
    come back.

The device sampler is timed here as its own compiled graph, so each "after" sample
pays one extra graph launch that the served root, where the sampler is fused into the
decode graph, does not: "after" is an upper bound. Both variants end with the tokens
on the host, which is what the scheduler needs each step.

Two sampler configurations, both of which the knob accepts:

* ``all_greedy`` -- ``{"all_greedy": true}``: argmax only; every request is greedy.
* ``full`` -- ``{}``: the default; per-request top-k (max_top_k 256) / top-p /
  temperature, greedy rows by argmax. Rows alternate greedy and sampled here.

For the full configuration the script also times its two parts alone: argmax over the
full row, and the top-k over the full row that dominates it.

Correctness, checked on every case before timing: every greedy row equals CPU
``torch.argmax`` on the same logits, including two rows with a planted tie at the
maximum (the first index must win, as on the host); every sampled row is inside its
row's top-k.

The sampler is torch ops lowered by the Neuron compiler (``torch.topk``, ``argmax``,
softmax, cumsum), not an NKI kernel. Set the cores before launching (devlease does);
this script does not select cores.

    python3 devlease.py slice host -- env ... python test/hardware/benchmark_sampler_decode.py \\
        --output reports/host_sampler_micro.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time

import torch

# Run this tree's vllm_neuron even without PYTHONPATH (the venv's editable install
# points at another checkout).
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import vllm_neuron  # noqa: E402,F401 -- registers the Neuron compilation backend

if REPO not in Path(vllm_neuron.__file__).resolve().parents:
    raise RuntimeError(f"vllm_neuron resolved to {vllm_neuron.__file__}, not under {REPO}")

from vllm.v1.sample.logits_processor import LogitsProcessors  # noqa: E402
from vllm.v1.sample.metadata import SamplingMetadata  # noqa: E402
from vllm.v1.sample.ops import topk_topp_sampler  # noqa: E402
from vllm.v1.sample.sampler import Sampler  # noqa: E402

from vllm_neuron.functional.full_vocab_sampling import sample_full_vocab  # noqa: E402
from vllm_neuron.model.neuron_config import OnDeviceSamplingConfig  # noqa: E402

VOCAB = 154880
# [top_k, top_p, temperature] rows as vLLM's InputBatch encodes them once any request
# in the batch sets top_k: a request without one carries the vocabulary size.
GREEDY_ROW = (float(VOCAB), 1.0, 0.0)
SAMPLED_ROW = (50.0, 0.9, 0.8)
TIES = ((1000, 90000), (5, VOCAB - 1))  # planted at the maximum of the first two greedy rows
MODES = {"all_greedy": {"all_greedy": True}, "full": {}}


def _stats(samples_ns: list[int]) -> dict:
    us = sorted(value / 1000.0 for value in samples_ns)
    return {
        "iterations": len(us),
        "median_us": statistics.median(us),
        "p90_us": us[min(len(us) - 1, int(0.9 * len(us)))],
        "mean_us": statistics.mean(us),
        "min_us": us[0],
        "max_us": us[-1],
    }


def _measure(call, warmup: int, iterations: int) -> dict:
    for _ in range(warmup):
        call()
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        call()
        samples.append(time.perf_counter_ns() - started)
    return _stats(samples)


def _inputs(batch: int, mode: str):
    generator = torch.Generator().manual_seed(4242 + batch)
    logits = (torch.randn(batch, VOCAB, generator=generator) * 4.0).to(torch.bfloat16)
    rows = [GREEDY_ROW if mode == "all_greedy" or index % 2 == 0 else SAMPLED_ROW
            for index in range(batch)]
    params = torch.tensor(rows, dtype=torch.float32)
    for row, (first, second) in zip(_tie_rows(params), TIES):
        peak = logits[row].float().max() + 1.0
        logits[row, first] = peak
        logits[row, second] = peak
    return logits, params


def _tie_rows(params: torch.Tensor) -> list[int]:
    return (params[:, 2] == 0).nonzero().reshape(-1).tolist()[: len(TIES)]


def _host_metadata(params: torch.Tensor, mode: str) -> SamplingMetadata:
    """What vLLM's InputBatch hands its Sampler for these rows (no penalties/logprobs)."""
    batch = int(params.shape[0])
    greedy = bool((params[:, 2] == 0).all())
    empty = torch.zeros(batch)
    return SamplingMetadata(
        temperature=None if greedy else params[:, 2].clone(),
        all_greedy=greedy, all_random=bool((params[:, 2] > 0).all()),
        top_p=None if greedy else params[:, 1].clone(),
        top_k=None if greedy else params[:, 0].to(torch.int32),
        generators={}, max_num_logprobs=None, no_penalties=True, prompt_token_ids=None,
        frequency_penalties=empty, presence_penalties=empty, repetition_penalties=empty,
        output_token_ids=[[] for _ in range(batch)], allowed_token_ids_mask=None,
        bad_words_token_ids={}, logitsprocs=LogitsProcessors())


def _check(tokens: torch.Tensor, logits: torch.Tensor, params: torch.Tensor, label: str) -> dict:
    tokens = tokens.reshape(-1).to(torch.int64)
    expected = torch.argmax(logits, dim=-1)
    greedy_rows = (params[:, 2] == 0).nonzero().reshape(-1).tolist()
    sampled_rows = (params[:, 2] > 0).nonzero().reshape(-1).tolist()
    wrong = [row for row in greedy_rows if int(tokens[row]) != int(expected[row])]
    if wrong:
        raise AssertionError(
            f"{label}: greedy rows {wrong} differ from torch.argmax: "
            f"{[(int(tokens[r]), int(expected[r])) for r in wrong]}")
    tie_rows = _tie_rows(params)
    for row, (first, _) in zip(tie_rows, TIES):
        if int(tokens[row]) != first:
            raise AssertionError(f"{label}: planted tie in row {row} resolved to "
                                 f"{int(tokens[row])}, not the first index {first}")
    outside = []
    for row in sampled_rows:
        k = int(params[row, 0])
        threshold = torch.topk(logits[row].float(), k).values[-1]
        if float(logits[row, int(tokens[row])].float()) < float(threshold):
            outside.append(row)
    if outside:
        raise AssertionError(f"{label}: sampled rows {outside} are outside their top-k")
    return {"greedy_rows_equal_torch_argmax": len(greedy_rows),
            "planted_ties_resolved_to_first_index": len(tie_rows),
            "sampled_rows_inside_top_k": len(sampled_rows)}


def run_case(batch: int, mode: str, args) -> dict:
    torch._dynamo.reset()
    logits, params = _inputs(batch, mode)
    config = OnDeviceSamplingConfig(**MODES[mode])

    def device_sampler(device_logits, device_params):
        return sample_full_vocab(device_logits, device_params, config)

    on_device = args.device != "cpu"
    compiled = torch.compile(
        device_sampler, backend="neuron_libtorch" if on_device else "eager",
        fullgraph=True, dynamic=False,
        **({"options": {"compiler_args": os.environ.get("NEURON_CC_FLAGS", "")}}
           if on_device else {}))
    logits_dev = logits.to(args.device)
    params_dev = params.to(args.device)

    compile_started = time.perf_counter()
    after_tokens = compiled(logits_dev, params_dev).to("cpu")
    compile_s = time.perf_counter() - compile_started
    sampler = Sampler()
    metadata = _host_metadata(params, mode)
    before_note = None
    has_triton = topk_topp_sampler.HAS_TRITON
    try:
        try:
            before_tokens = sampler(logits_dev.to("cpu"), metadata).sampled_token_ids
        except NotImplementedError as error:
            # At B>=8 vLLM's top-k/top-p takes its Triton kernel, which asks the
            # platform for num_compute_units; the Neuron platform has none, so the
            # host path as shipped cannot sample this batch. Time vLLM's own PyTorch
            # top-k/top-p instead, and say so.
            before_note = (f"vLLM's host Sampler raised {error!r} (its Triton top-k/top-p "
                           f"at B>=8); timed with vLLM's PyTorch top-k/top-p instead")
            topk_topp_sampler.HAS_TRITON = False
            before_tokens = sampler(logits_dev.to("cpu"), metadata).sampled_token_ids
        checks = {"after": _check(after_tokens, logits, params, f"after B={batch} {mode}"),
                  "before": _check(before_tokens.to("cpu"), logits, params,
                                   f"before B={batch} {mode}")}
        after = _measure(lambda: compiled(logits_dev, params_dev).to("cpu"),
                         args.warmup, args.iterations)
        before = _measure(lambda: sampler(logits_dev.to("cpu"), metadata).sampled_token_ids,
                          args.warmup, args.iterations)
        copy_only = _measure(lambda: logits_dev.to("cpu"), args.warmup, args.iterations)
    finally:
        topk_topp_sampler.HAS_TRITON = has_triton
    components = {}
    if mode == "full":
        # Where the full sampler's time goes: its top-k over every vocabulary entry,
        # against the argmax that is all a greedy-only config runs.
        max_top_k = config.max_top_k
        for name, part in (
            ("argmax_full_row", lambda x: torch.argmax(x, dim=-1).to(torch.int32)),
            (f"topk{max_top_k}_full_row",
             lambda x: torch.topk(x, max_top_k, dim=-1)[1].to(torch.int32)),
        ):
            torch._dynamo.reset()
            graph = torch.compile(
                part, backend="neuron_libtorch" if on_device else "eager",
                fullgraph=True, dynamic=False,
                **({"options": {"compiler_args": os.environ.get("NEURON_CC_FLAGS", "")}}
                   if on_device else {}))
            components[name] = _measure(lambda: graph(logits_dev).to("cpu"),
                                        args.warmup, args.iterations)
    return {
        "B": batch, "vocab": VOCAB, "logits_dtype": "bfloat16", "mode": mode,
        "on_device_sampling_config": MODES[mode],
        "rows": [list(map(float, row)) for row in params[: min(batch, 2)]],
        "checks": checks,
        "first_call_s_including_compile": compile_s,
        "before_host_path": before,
        "before_note": before_note,
        "before_logits_copy_only": copy_only,
        "after_device_sampler": after,
        "components": components,
        "median_speedup": before["median_us"] / after["median_us"],
        "median_saved_us": before["median_us"] - after["median_us"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 64])
    parser.add_argument("--modes", nargs="+", choices=sorted(MODES), default=sorted(MODES))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--device", default="neuron:0",
                        help="'cpu' is a dry run of the checks only; its times mean nothing")
    args = parser.parse_args()
    if args.device != "cpu":
        if os.environ.get("VLLM_NEURON_CPU_MODE") == "1":
            raise ValueError("Hardware benchmark cannot run in VLLM_NEURON_CPU_MODE=1")
        if not os.environ.get("NEURON_RT_VISIBLE_CORES"):
            raise ValueError("Pin NEURON_RT_VISIBLE_CORES before running the benchmark")
    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("Use nonnegative warmup and positive iteration counts")
    # neuronx-cc writes metrics and, on a failed compile, its artifacts into the
    # working directory; keep them out of the source tree.
    args.output = args.output.resolve()
    os.chdir(tempfile.mkdtemp(prefix="benchmark_sampler_decode_"))
    # vLLM's multiproc executor runs each worker on one torch thread; the host
    # sampler is timed on the same.
    torch.set_num_threads(1)
    report = {
        "what": __doc__.splitlines()[0],
        "device": args.device,
        "dry_run": args.device == "cpu",
        "vllm_neuron": str(Path(vllm_neuron.__file__).resolve().parent),
        "torch_threads": torch.get_num_threads(),
        "environment": {key: os.environ.get(key) for key in (
            "NEURON_RT_VISIBLE_CORES", "NEURON_RT_NUM_CORES", "NEURON_LOGICAL_NC_CONFIG",
            "NEURON_CC_FLAGS", "NEURON_PLATFORM_TARGET_OVERRIDE",
            "NEURON_LIBTORCH_CACHE_ROOT")},
        "kernel": "torch ops lowered by the Neuron compiler (no NKI kernel)",
        "after_is_upper_bound": "standalone graph: one launch the fused root does not pay",
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for batch in args.batches:
        for mode in args.modes:
            case = run_case(batch, mode, args)
            report["cases"].append(case)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({key: case[key] for key in (
                "B", "mode", "median_speedup", "first_call_s_including_compile")}
                | {"before_us": case["before_host_path"]["median_us"],
                   "after_us": case["after_device_sampler"]["median_us"]}), flush=True)


if __name__ == "__main__":
    main()
