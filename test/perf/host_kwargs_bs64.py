# SPDX-License-Identifier: Apache-2.0
"""CPU-mode timing of ``_glm5next_model_kwargs`` and the decode graph's operand inputs.

What is measured, for a tree and a batch size ``B``:

* ``kwargs_ms``: the wall time of one ``_glm5next_model_kwargs`` call on a real
  ``NeuronModelRunner`` driving a real decode step of ``B`` requests, median over the
  timed steps (``--steps``, default 60, after ``--warmup`` steps). The runner holds the
  real GLM-5.3-Flash root geometry (45 layers: 34 KDA, 11 sparse attention, the served
  hybrid KV grouping at block 128) and the root's ``forward`` is a stub, so the number is
  the converter alone; the step is driven through ``execute_model`` so the metadata it
  reads is the runner's own.
* ``operand_inputs``: how many distinct tensors the converter hands the root in the
  translated kwargs. Every tensor leaf there becomes one placeholder of the captured
  decode graph (dynamo lifts each tensor it meets; the backend then drops duplicates by
  storage identity, offset, shape, stride and dtype, which this count also does). The
  served graph's input count is these operands plus the weights the forward reads, and
  the weights are the same before and after, so::

      graph_inputs(tree, B) = SERVED_INPUTS[B] - operand_inputs(base, B) + operand_inputs(tree, B)

  where ``SERVED_INPUTS`` are the counts read off the served graphs of 0a08ff4
  (``example_inputs.txt`` of the bs=64 and bs=1 compile-cache entries: 7298 and 1630).
  The derivation is per batch size (each served graph against the base operands of
  the same ``B``). The derived weight count ``SERVED_INPUTS[B] - operand_inputs(base,
  B)`` is reported per ``B`` beside its spread across batches as a check on the
  accounting: it is 1497 at ``B = 1`` and 1496 at ``B = 64`` (one operand of the bs=1
  graph -- the one-request leg's own -- is not in the bs=64 operand set), so the spread
  is 1 and a spread above that would say the operand count misses something.

The base tree is a read-only ``git worktree`` of 0a08ff4 (``--base-tree``); each case runs
in its own process with ``PYTHONPATH`` and the working directory set to its tree, so the
two trees' modules never mix. Every case runs with one torch thread, as vLLM's
multiproc executor runs each worker.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 OMP_NUM_THREADS=4 PYTHONPATH=$PWD \\
        python test/perf/host_kwargs_bs64.py --json reports/hostpath_kwargs.json \\
        --base-tree <0a08ff4 worktree>

``--profile`` additionally prints the top cumulative frames of the timed converter calls
(cProfile) for each case, to stderr.
"""

from __future__ import annotations

import argparse
import cProfile
import gc
import io
import json
import os
import pathlib
import pstats
import statistics
import subprocess
import sys
import tempfile
import time

#: Input counts of the decode graphs served from 0a08ff4 (compile cache
#: 816ef60b6a5632fa5a4a9d3746ca2c87 for bs=64; DECODE_BREAKDOWN_v2.md section 3 for bs=1).
SERVED_INPUTS = {64: 7298, 1: 1630}
BATCHES = (1, 64)
VOCAB = 154880
PROMPT = 100
PREFILL_BUCKET = 1024
#: The bs=64 serve line: 8k model length, decode context bucket 2048, hybrid block 128.
MAX_MODEL_LEN = 8192
DECODE_CTX_BUCKET = 2048
BLOCK = 128
FIXTURE = pathlib.Path("test/vllm_neuron/model/glm5_next/fixtures")
KV_BYTES = 3 * 2**30


def _stats(samples_ms: list[float]) -> dict:
    ordered = sorted(samples_ms)
    return {
        "median_ms": statistics.median(ordered),
        "p90_ms": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
        "mean_ms": statistics.mean(ordered),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "n": len(ordered),
    }


def _tensor_leaves(value):
    import torch

    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensor_leaves(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensor_leaves(item)


def _count_operand_inputs(kwargs: dict) -> dict:
    """Distinct tensor leaves of the translated kwargs, the backend's dedup key."""
    seen = set()
    total = 0
    strided = 0  # leaves the Neuron executor would refuse (non-contiguous inputs)
    per_key: dict[str, int] = {}
    for key, value in kwargs.items():
        before = len(seen)
        for leaf in _tensor_leaves(value):
            total += 1
            strided += 0 if leaf.is_contiguous() else 1
            seen.add(
                (
                    id(leaf.untyped_storage()),
                    leaf.storage_offset(),
                    tuple(leaf.shape),
                    tuple(leaf.stride()),
                    leaf.dtype,
                )
            )
        per_key[key] = len(seen) - before
    carriers = kwargs.get("layer_carriers", ())
    families: dict[str, int] = {}
    for carrier in carriers:
        family = "kda" if "conv_state" in carrier else "dsa"
        families[family] = families.get(family, 0) + sum(1 for _ in _tensor_leaves(carrier))
    return {
        "operand_inputs": len(seen),
        "operand_inputs_noncontiguous": strided,
        "tensor_leaves": total,
        "distinct_by_key": per_key,
        "carrier_leaves_by_family": families,
    }


def _child(case: dict) -> dict:
    import torch

    torch.set_num_threads(1)
    from vllm.config import set_current_vllm_config
    from vllm.distributed import parallel_state as dist_state
    from vllm.engine.arg_utils import EngineArgs
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
    from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput

    import vllm_neuron
    import vllm_neuron.vllm.worker.neuron_model_runner as runner_module
    from vllm_neuron.model.glm5_next.model_fp8 import Glm5NextForConditionalGeneration as Root

    tree = pathlib.Path(case["tree_path"]).resolve()
    package = pathlib.Path(vllm_neuron.__file__).resolve()
    if tree not in package.parents:
        raise RuntimeError(f"vllm_neuron resolved to {package}, not under {tree}")

    batch = int(case["batch"])
    steps = int(case["steps"])
    warmup = int(case["warmup"])
    neuron_config = {
        "num_batched_tokens_buckets": [PREFILL_BUCKET],
        "kv_segment_size_buckets": [MAX_MODEL_LEN],
        "decode_context_length_buckets": [DECODE_CTX_BUCKET],
        "hybrid_kv_block_size": BLOCK,
        "on_device_sampling_config": {"all_greedy": True},
    }
    config = EngineArgs(
        model=str(tree / FIXTURE), skip_tokenizer_init=True, max_model_len=MAX_MODEL_LEN,
        max_num_seqs=batch, max_num_batched_tokens=PREFILL_BUCKET, block_size=BLOCK,
        enforce_eager=True, enable_prefix_caching=False, async_scheduling=False,
        additional_config={"neuron_config": neuron_config},
    ).create_engine_config()

    hf = json.load(open(tree / FIXTURE / "config.json"))
    text = hf.get("text_config", hf)
    text["linear_attn_config"]["num_heads"] = 1  # per-rank KDA heads at TP=64

    generator = torch.Generator().manual_seed(1234 + batch)
    device_logits = (torch.randn(batch, VOCAB, generator=generator) * 4.0).to(torch.bfloat16)

    def stub_forward(input_ids, *, layer_carriers, sampling_positions,
                     device_sampling_params=None, device_logit_mask=None, **_):
        rows = int(sampling_positions.shape[0])
        if device_sampling_params is not None:
            return torch.argmax(device_logits[:rows], dim=-1).to(torch.int32)
        return device_logits[:rows].clone()

    rendezvous = tempfile.mkdtemp(prefix="host_kwargs_rdv_")
    result: dict = {"batch": batch, "tree": str(tree)}
    with set_current_vllm_config(config, check_compile=False):
        dist_state.init_distributed_environment(
            world_size=1, rank=0, distributed_init_method=f"file://{rendezvous}/rdv",
            local_rank=0, backend="gloo")
        dist_state.ensure_model_parallel_initialized(1, 1)
        try:
            runner = runner_module.NeuronModelRunner(config, device=torch.device("cpu"))
            root = Root.from_configs(hf, text_neuron_config=runner.neuron_config)
            root.forward = stub_forward
            runner.model = root
            runner.vocab_size = VOCAB
            spec = runner.get_kv_cache_spec()
            kv_config = get_kv_cache_configs(config, [spec], [KV_BYTES])[0]
            runner.initialize_kv_cache(kv_config)
            groups = kv_config.kv_cache_groups
            runner.warmup_prefill(PREFILL_BUCKET, PREFILL_BUCKET)
            runner.warmup_decode(batch, ctx_bucket=DECODE_CTX_BUCKET)

            requests = [f"req-{index}" for index in range(batch)]
            needed = PROMPT + warmup + steps + 2
            next_block = [1]

            def blocks_for(spec_) -> list[int]:
                count = -(-needed // BLOCK) if "Attention" in type(spec_).__name__ else 1
                first = next_block[0]
                next_block[0] += count
                return list(range(first, first + count))

            prompt_generator = torch.Generator().manual_seed(99)

            def run(step):
                out = runner.execute_model(step)
                return runner.sample_tokens(None) if out is None else out

            all_token_ids: dict[str, list[int]] = {}
            for request in requests:
                prompt = torch.randint(0, VOCAB, (PROMPT,), generator=prompt_generator).tolist()
                new = NewRequestData(
                    req_id=request, prompt_token_ids=prompt, mm_features=[],
                    sampling_params=SamplingParams(temperature=0.0, max_tokens=4096),
                    pooling_params=None,
                    block_ids=tuple(blocks_for(group.kv_cache_spec) for group in groups),
                    num_computed_tokens=0, lora_request=None)
                step = SchedulerOutput(
                    scheduled_new_reqs=[new], scheduled_cached_reqs=CachedRequestData.make_empty(),
                    num_scheduled_tokens={request: PROMPT}, total_num_scheduled_tokens=PROMPT,
                    scheduled_spec_decode_tokens={}, scheduled_encoder_inputs={},
                    num_common_prefix_blocks=[0] * len(groups), finished_req_ids=set(),
                    free_encoder_mm_hashes=[])
                step.num_scheduled_tokens_padded = {request: PREFILL_BUCKET}
                output = run(step)
                if hasattr(output, "get_output"):
                    output = output.get_output()
                all_token_ids[request] = prompt + [int(output.sampled_token_ids[0][0])]

            position = [PROMPT]

            def decode_step():
                first = position[0] == PROMPT
                cached = CachedRequestData(
                    req_ids=list(requests), resumed_req_ids=set(),
                    new_token_ids=[[] for _ in requests],
                    all_token_ids={r: all_token_ids[r] for r in requests[:-1]} if first else {},
                    new_block_ids=[None] * batch,
                    num_computed_tokens=[position[0]] * batch,
                    num_output_tokens=[position[0] - PROMPT + 1] * batch)
                step = SchedulerOutput(
                    scheduled_new_reqs=[], scheduled_cached_reqs=cached,
                    num_scheduled_tokens={request: 1 for request in requests},
                    total_num_scheduled_tokens=batch, scheduled_spec_decode_tokens={},
                    scheduled_encoder_inputs={}, num_common_prefix_blocks=[0] * len(groups),
                    finished_req_ids=set(), free_encoder_mm_hashes=[])
                step.num_scheduled_tokens_padded = {request: 1 for request in requests}
                position[0] += 1
                return step

            samples_ms: list[float] = []
            last: dict = {}
            original = runner._glm5next_model_kwargs
            profiler = cProfile.Profile() if case.get("profile") else None
            timing = {"on": False}

            def timed(kwargs):
                if not timing["on"]:
                    return original(kwargs)
                gc.collect()
                gc.disable()
                try:
                    if profiler is not None:
                        profiler.enable()
                    started = time.perf_counter_ns()
                    out = original(kwargs)
                    samples_ms.append((time.perf_counter_ns() - started) / 1e6)
                    if profiler is not None:
                        profiler.disable()
                finally:
                    gc.enable()
                last["kwargs"] = out
                return out

            runner._glm5next_model_kwargs = timed
            for index in range(warmup + steps):
                timing["on"] = index >= warmup
                out = run(decode_step())
                if hasattr(out, "get_output"):
                    out.get_output()
            result["kwargs"] = _stats(samples_ms)
            result["kwargs_samples_ms"] = samples_ms
            result.update(_count_operand_inputs(last["kwargs"]))
            carriers = last["kwargs"]["layer_carriers"]
            result["carrier_forms"] = {
                "kda_conv_state": type(carriers[0].get("conv_state")).__name__,
                "kda_has_state_slots": "state_slots" in carriers[0],
            }
            result["load_average_1m"] = os.getloadavg()[0]
            if profiler is not None:
                buffer = io.StringIO()
                pstats.Stats(profiler, stream=buffer).sort_stats("cumulative").print_stats(28)
                result["profile"] = buffer.getvalue()
        finally:
            dist_state.destroy_model_parallel()
            dist_state.destroy_distributed_environment()
    return result


def _run_case(script: pathlib.Path, tree: pathlib.Path, case: dict) -> dict:
    env = dict(os.environ)
    env.update({"PYTHONPATH": str(tree), "VLLM_NEURON_CPU_MODE": "1", "NKI_SIMULATOR": "1",
                "NEURON_PLATFORM_TARGET_OVERRIDE": "trn2", "VLLM_LOGGING_LEVEL": "WARNING",
                "OMP_NUM_THREADS": "1",
                # The served recipe: on-device greedy sampling, host-only metadata.
                "VLLM_NEURON_GLM5NEXT_ON_DEVICE_SAMPLING": "1",
                "VLLM_NEURON_GLM5NEXT_HOST_ONLY_METADATA": "1"})
    payload = json.dumps(dict(case, tree_path=str(tree)))
    proc = subprocess.run([sys.executable, str(script), "--child", payload], cwd=str(tree),
                          env=env, capture_output=True, text=True)
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
    if proc.returncode != 0 or not lines:
        raise RuntimeError(
            f"case tree={tree} B={case['batch']} failed (exit {proc.returncode}):\n"
            f"{proc.stderr[-6000:]}")
    return json.loads(lines[-1][len("RESULT "):])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=pathlib.Path, required=False)
    parser.add_argument("--base-tree", type=pathlib.Path,
                        help="read-only worktree of the base revision; needed for --trees base")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--batches", type=int, nargs="+", default=list(BATCHES))
    parser.add_argument("--trees", nargs="+", default=["base", "head"],
                        help="which trees to run: base, head (this one), or both")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--child")
    args = parser.parse_args()
    if args.child:
        print("RESULT " + json.dumps(_child(json.loads(args.child))), flush=True)
        return 0

    if "base" in args.trees and args.base_tree is None:
        parser.error("--trees base needs --base-tree")
    script = pathlib.Path(__file__).resolve()
    head = script.parents[2]
    trees = {"head": head}
    if args.base_tree is not None:
        trees["base"] = args.base_tree.resolve()
    revs = {
        name: subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                             check=True, capture_output=True, text=True).stdout.strip()
        for name, path in trees.items() if name in args.trees
    }
    report: dict = {
        "what": __doc__.splitlines()[0],
        "trees": {name: {"path": str(trees[name]), "rev": revs[name]} for name in revs},
        "served_inputs": SERVED_INPUTS,
        "steps": args.steps, "warmup": args.warmup, "cpu_count": os.cpu_count(),
        "load_average_1m_at_start": os.getloadavg()[0],
        "cases": {},
    }
    for batch in args.batches:
        for name in args.trees:
            print(f"[host_kwargs_bs64] {name} B={batch} ...", file=sys.stderr, flush=True)
            case = {"batch": batch, "steps": args.steps, "warmup": args.warmup,
                    "profile": args.profile}
            result = _run_case(script, trees[name], case)
            report["cases"][f"{name}_B{batch}"] = result
            if args.profile and "profile" in result:
                print(f"--- profile {name} B={batch} ---\n{result['profile']}",
                      file=sys.stderr, flush=True)

    # Derived graph input counts, with the weight-count consistency check.
    summary: dict = {}
    weights = {}
    if "base" in args.trees:
        for batch in args.batches:
            if batch in SERVED_INPUTS:
                weights[batch] = SERVED_INPUTS[batch] - report["cases"][f"base_B{batch}"]["operand_inputs"]
    spread = (max(weights.values()) - min(weights.values())) if weights else 0
    report["derived_weight_inputs"] = weights
    report["derived_weight_inputs_spread"] = spread
    for batch in args.batches:
        row: dict = {}
        for name in args.trees:
            case = report["cases"][f"{name}_B{batch}"]
            row[f"{name}_operand_inputs"] = case["operand_inputs"]
            row[f"{name}_kwargs_median_ms"] = case["kwargs"]["median_ms"]
            row[f"{name}_kwargs_p90_ms"] = case["kwargs"]["p90_ms"]
            if weights and batch in SERVED_INPUTS:
                row[f"{name}_graph_inputs"] = (
                    SERVED_INPUTS[batch]
                    - report["cases"][f"base_B{batch}"]["operand_inputs"]
                    + case["operand_inputs"]
                )
        summary[f"B{batch}"] = row
    report["summary"] = summary
    report["load_average_1m_at_end"] = os.getloadavg()[0]
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"summary": summary, "derived_weight_inputs": weights,
                      "derived_weight_inputs_spread": spread}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
