# SPDX-License-Identifier: Apache-2.0
"""CPU-mode timing of GLM-5.3-Flash's per-step host path: 5938748 against this tree.

What is timed is one decode step as the worker drives it, ``execute_model`` then
``sample_tokens`` (and, under async scheduling, ``get_output``), on a real
``NeuronModelRunner`` holding the real GLM-5.3-Flash root geometry: 45 layers (34 KDA,
11 sparse attention), the real 154,880-entry vocabulary, and vLLM's own KV grouping
(5 groups at hybrid block size 128, as served). The root's ``forward`` is replaced by a
stub that returns what the device returns -- ``[B, vocab]`` bf16 logits, or ``[B]`` int32
tokens when the root samples on device -- and the stub's own time is subtracted, so the
number is host time only. KDA heads are the per-rank count at TP=64 (one), which is what
makes the page unifier accept the stack on one CPU rank, as it does on each served rank.

Cases, each in its own process so a tree's modules never mix:

* ``before``: the 5938748 tree, as served today (host vLLM Sampler, sync scheduling).
* ``head_default``: this tree with no knob set; must match ``before``.
* ``after_sync``: this tree, on-device sampling + host-only metadata, sync scheduling.
* ``after_async``: the same with async scheduling; ``get_output`` runs on vLLM's async
  output thread in a server, so it is reported apart from the critical path.

Every case runs with one torch thread, as vLLM's multiproc executor runs each worker.
Each case runs once per round (``--rounds``, default 3) and the case order reverses every
round, so drift in machine load hits before and after alike; the step samples of all
rounds are pooled, and the JSON keeps each round's median and the 1-minute load average.

B=1 runs the real layer mix. B=4 runs a KDA-only stack of the same 45 layers, because
the sparse-attention family refuses more than one request per forward
(``_glm5next_layer_carriers``); B=1 is also run on that stack so the two batch
sizes compare on one geometry.

In CPU mode device work is a no-op, so what the device path pays per step for it
(the decode breakdown's phase E1, 1.63 ms) costs almost nothing here. The script therefore
also counts it, in a separate untimed pass with the runner's device set to ``meta``
(a read back from the device would raise there): host->device copies, device factories
and device ops, by call name, and the sentinel remaps by name.

    PYTHONPATH=$PWD python test/perf/host_step_timing.py --output reports/host_step_timing.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pathlib
import statistics
import subprocess
import sys
import tempfile
import time

BASE_REV = "5938748"
VOCAB = 154880
PROMPT = 100
PREFILL_BUCKET = 1024
MAX_MODEL_LEN = 4096
BLOCK = 128
FIXTURE = pathlib.Path("test/vllm_neuron/model/glm5_next/fixtures")
KNOBS = ("VLLM_NEURON_GLM5NEXT_ON_DEVICE_SAMPLING", "VLLM_NEURON_GLM5NEXT_HOST_ONLY_METADATA")
CASES = {
    "before": {"tree": "base", "knobs": (), "async": False, "ods": False},
    "head_default": {"tree": "head", "knobs": (), "async": False, "ods": False},
    "after_sync": {"tree": "head", "knobs": KNOBS, "async": False, "ods": True},
    "after_async": {"tree": "head", "knobs": KNOBS, "async": True, "ods": True},
}
PLAN = (
    # (batch, stack, cases)
    (1, "real", ("before", "head_default", "after_sync", "after_async")),
    (1, "kda_only", ("before", "after_sync", "after_async")),
    (4, "kda_only", ("before", "head_default", "after_sync", "after_async")),
)


# ── child: one case in one process ───────────────────────────────────────────


def _stats(samples_ns: list[int]) -> dict:
    us = sorted(value / 1000.0 for value in samples_ns)
    return {
        "median_us": statistics.median(us),
        "p90_us": us[min(len(us) - 1, int(0.9 * len(us)))],
        "mean_us": statistics.mean(us),
        "min_us": us[0],
        "n": len(us),
    }


def _stats_us(samples_us: list[float]) -> dict:
    return _stats([value * 1000.0 for value in samples_us])


def _merge(results: list[dict]) -> dict:
    """One case's rounds as one record: pooled step samples, the rest from round 0."""
    merged = {key: value for key, value in results[0].items()
              if key not in ("critical_samples_us", "output_samples_us")}
    merged["host_critical_path"] = _stats_us(
        [value for result in results for value in result["critical_samples_us"]])
    merged["async_output_thread"] = _stats_us(
        [value for result in results for value in result["output_samples_us"]])
    merged["host_total"] = _stats_us([
        a + b for result in results
        for a, b in zip(result["critical_samples_us"], result["output_samples_us"])])
    merged["round_medians_us"] = [result["host_critical_path"]["median_us"] for result in results]
    merged["phases_median_us"] = {
        name: statistics.median(result["phases_median_us"][name] for result in results)
        for name in results[0]["phases_median_us"]}
    merged["load_average_1m"] = [result["load_average_1m"] for result in results]
    merged["tokens_same_every_round"] = len({json.dumps(r["all_tokens"]) for r in results}) == 1
    return merged


def _child(case: dict) -> dict:
    for name in KNOBS:
        os.environ.pop(name, None)
    for name in case["knobs"]:
        os.environ[name] = "1"

    import torch
    from torch.overrides import TorchFunctionMode

    # vLLM's multiproc executor runs each worker with one torch thread
    # (multiproc_executor.py, "Reducing Torch parallelism ... to 1"); so does the serve.
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
    neuron_config = {
        "num_batched_tokens_buckets": [PREFILL_BUCKET],
        "num_seqs_buckets": [batch],
        "kv_segment_size_buckets": [PREFILL_BUCKET],
        "hybrid_kv_block_size": BLOCK,
        "on_device_sampling_config": {} if case["ods"] else None,
    }
    config = EngineArgs(
        model=str(tree / FIXTURE), skip_tokenizer_init=True, max_model_len=MAX_MODEL_LEN,
        max_num_seqs=batch, max_num_batched_tokens=PREFILL_BUCKET, block_size=BLOCK,
        enforce_eager=True, enable_prefix_caching=False, async_scheduling=case["async"],
        additional_config={"neuron_config": neuron_config},
    ).create_engine_config()

    hf = json.load(open(tree / FIXTURE / "config.json"))
    text = hf.get("text_config", hf)
    text["linear_attn_config"]["num_heads"] = 1  # per-rank KDA heads at TP=64
    if case["stack"] == "kda_only":
        layers = int(text["num_hidden_layers"])
        text["layer_types"] = ["linear_attention"] * layers
        text["linear_attn_config"]["kda_layers"] = list(range(layers))
        text["linear_attn_config"]["full_attn_layers"] = []

    generator = torch.Generator().manual_seed(1234 + batch)
    device_logits = (torch.randn(batch, VOCAB, generator=generator) * 4.0).to(torch.bfloat16)
    stub_ns = [0]

    def stub_forward(input_ids, *, layer_carriers, sampling_positions,
                     device_sampling_params=None, device_logit_mask=None, **_):
        started = time.perf_counter_ns()
        rows = int(sampling_positions.shape[0])
        if device_sampling_params is not None:
            out = torch.argmax(device_logits[:rows], dim=-1).to(torch.int32)
        else:
            out = device_logits[:rows].clone()
        stub_ns[0] += time.perf_counter_ns() - started
        return out

    rendezvous = tempfile.mkdtemp(prefix="host_step_rdv_")
    result: dict = {}
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
            kv_config = get_kv_cache_configs(config, [spec], [512 * 2**20])[0]
            runner.initialize_kv_cache(kv_config)
            groups = kv_config.kv_cache_groups
            runner.warmup_prefill(PREFILL_BUCKET, PREFILL_BUCKET)
            # Warmup primes compiles only. At B>1 both trees' synthetic decode inputs
            # hand this family B tokens for one request, which the carrier builder
            # refuses; the timed steps below carry B real requests and are not refused.
            result["warmup_decode"] = batch == 1
            if batch == 1:
                runner.warmup_decode(batch, ctx_bucket=runner.max_model_len)

            requests = [f"req-{index}" for index in range(batch)]
            mla_blocks = -(-(PROMPT + 2 * case["steps"] + 16) // BLOCK)
            next_block = [1]

            def blocks_for(spec_) -> list[int]:
                count = mla_blocks if "Attention" in type(spec_).__name__ else 1
                first = next_block[0]
                next_block[0] += count
                return list(range(first, first + count))

            prompt_generator = torch.Generator().manual_seed(99)

            def run(step):
                out = runner.execute_model(step)
                return runner.sample_tokens(None) if out is None else out

            # One prefill per request, as the scheduler serves this family.
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
                # The scheduler sends a request's tokens when the previous step did not
                # schedule it (the runner dropped it from the batch); at the first decode
                # that is every request but the last one prefilled.
                first = position[0] == PROMPT
                cached = CachedRequestData(
                    req_ids=list(requests), resumed_req_ids=set(), new_token_ids=[[] for _ in requests],
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

            phases: dict[str, list[int]] = {}
            current: dict[str, int] = {}
            for name in ("_update_states", "_prepare_model_input", "_glm5next_model_kwargs",
                         "_sample"):
                original = getattr(runner, name)

                def timed(*args, _original=original, _name=name, **kwargs):
                    started = time.perf_counter_ns()
                    try:
                        return _original(*args, **kwargs)
                    finally:
                        current[_name] = current.get(_name, 0) + time.perf_counter_ns() - started

                setattr(runner, name, timed)

            critical: list[int] = []
            output_thread: list[int] = []
            tokens: list = []
            for index in range(case["warmup"] + case["steps"]):
                step = decode_step()
                current.clear()
                stub_ns[0] = 0
                gc.collect()
                gc.disable()
                started = time.perf_counter_ns()
                out = runner.execute_model(step)
                if out is None:
                    out = runner.sample_tokens(None)
                middle = time.perf_counter_ns()
                if hasattr(out, "get_output"):
                    out = out.get_output()
                ended = time.perf_counter_ns()
                gc.enable()
                model_ns = stub_ns[0]
                if index < case["warmup"]:
                    continue
                critical.append(middle - started - model_ns)
                output_thread.append(ended - middle)
                tokens.append([row[0] for row in out.sampled_token_ids[:batch]])
                for name, value in current.items():
                    phases.setdefault(name, []).append(value)
            # _glm5next_model_kwargs runs inside the model call site, not around the model.
            result["host_critical_path"] = _stats(critical)
            result["critical_samples_us"] = [value / 1000.0 for value in critical]
            result["async_output_thread"] = _stats(output_thread)
            result["output_samples_us"] = [value / 1000.0 for value in output_thread]
            result["load_average_1m"] = os.getloadavg()[0]
            result["host_total"] = _stats([a + b for a, b in zip(critical, output_thread)])
            result["phases_median_us"] = {
                name: statistics.median(values) / 1000.0 for name, values in phases.items()
            }

            # Untimed pass: the runner's device is swapped for ``meta`` so every tensor the
            # step puts on the device is told apart from host tensors, and any read back
            # from the device raises. A call that returns a meta tensor from host inputs
            # only is a host->device copy (``to`` / ``copy_``) or a device factory
            # (``full``, ``zeros``, ...); one that returns a meta tensor from a meta input
            # is a device op. On Neuron each copy is one host->device write and each
            # factory or op one eager dispatch. A ``to`` that returns its own input does
            # nothing and is not counted. The root stub only reads shapes.
            counts = {"h2d_copies": 0, "device_factories": 0, "device_ops": 0, "remaps": 0}
            by_name: dict[str, int] = {}

            def _tensors(value):
                if isinstance(value, torch.Tensor):
                    yield value
                elif isinstance(value, (list, tuple)):
                    for item in value:
                        yield from _tensors(item)
                elif isinstance(value, dict):
                    for item in value.values():
                        yield from _tensors(item)

            class DeviceWork(TorchFunctionMode):
                def __torch_function__(self, func, types, args=(), kwargs=None):
                    kwargs = kwargs or {}
                    out = func(*args, **kwargs)
                    made = [t for t in _tensors(out) if t.device.type == "meta"]
                    inputs = list(_tensors((args, kwargs)))
                    if made and not any(t is i for t in made for i in inputs):
                        name = getattr(func, "__name__", str(func))
                        if any(t.device.type == "meta" for t in inputs):
                            kind = "device_ops"
                        elif name in ("to", "copy_", "cuda"):
                            kind = "h2d_copies"
                        else:
                            kind = "device_factories"
                        counts[kind] += 1
                        by_name[f"{kind}:{name}"] = by_name.get(f"{kind}:{name}", 0) + 1
                    return out

            remap = runner_module._remap_null_block_to_sentinel

            def counted_remap(table):
                counts["remaps"] += 1
                return remap(table)

            runner_module._remap_null_block_to_sentinel = counted_remap
            host_device, runner.device = runner.device, torch.device("meta")
            counted_steps = 5
            try:
                for _ in range(counted_steps):
                    with DeviceWork():
                        out = run(decode_step())
                    if hasattr(out, "get_output"):
                        out.get_output()
            finally:
                runner.device = host_device
                runner_module._remap_null_block_to_sentinel = remap
            result["per_step_device_work"] = {
                key: value / counted_steps for key, value in counts.items()
            }
            result["per_step_device_work_by_call"] = {
                key: value / counted_steps for key, value in sorted(by_name.items())
            }
            result["first_tokens"] = tokens[0]
            result["all_tokens"] = tokens
            result["async_steps"] = getattr(runner, "_async_steps", None)
            result["kv_groups"] = [
                [type(group.kv_cache_spec).__name__, len(group.layer_names)] for group in groups
            ]
            result["vllm_neuron"] = str(package.parent)
        finally:
            dist_state.destroy_model_parallel()
            dist_state.destroy_distributed_environment()
    return result


# ── parent: build the base tree, run every case, compare ─────────────────────


def _base_tree(repo: pathlib.Path, rev: str, into: pathlib.Path) -> pathlib.Path:
    tree = into / f"tree-{rev}"
    tree.mkdir()
    archive = subprocess.run(
        ["git", "-C", str(repo), "archive", "--format=tar", rev, "vllm_neuron", str(FIXTURE)],
        check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(tree)], input=archive, check=True)
    return tree


def _run_case(script: pathlib.Path, tree: pathlib.Path, case: dict) -> dict:
    env = dict(os.environ)
    env.update({"PYTHONPATH": str(tree), "VLLM_NEURON_CPU_MODE": "1", "NKI_SIMULATOR": "1",
                "NEURON_PLATFORM_TARGET_OVERRIDE": "trn2", "VLLM_LOGGING_LEVEL": "WARNING",
                "OMP_NUM_THREADS": "1"})
    for name in KNOBS:
        env.pop(name, None)
    payload = json.dumps(dict(case, tree_path=str(tree)))
    proc = subprocess.run([sys.executable, str(script), "--child", payload], cwd=str(tree),
                          env=env, capture_output=True, text=True)
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
    if proc.returncode != 0 or not lines:
        raise RuntimeError(
            f"case {case['name']} B={case['batch']} {case['stack']} failed "
            f"(exit {proc.returncode}):\n{proc.stderr[-4000:]}")
    return json.loads(lines[-1][len("RESULT "):])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--base-rev", default=BASE_REV)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3,
                        help="each case runs once per round; the case order reverses "
                             "every round so machine-load drift hits before and after alike")
    parser.add_argument("--child")
    args = parser.parse_args()
    if args.child:
        print("RESULT " + json.dumps(_child(json.loads(args.child))), flush=True)
        return 0
    if args.output is None:
        parser.error("--output is required")

    script = pathlib.Path(__file__).resolve()
    repo = script.parents[2]
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(repo), "status", "--short", "--untracked-files=no"],
                                check=True, capture_output=True, text=True).stdout.strip())
    report: dict = {
        "what": "median per-decode-step host time, CPU mode, root forward stubbed and "
                "subtracted; see the module docstring",
        "base_rev": args.base_rev, "head_rev": head, "head_dirty": dirty,
        "steps": args.steps, "warmup_steps": args.warmup, "rounds": args.rounds,
        "prompt_tokens": PROMPT, "cpu_count": os.cpu_count(),
        "load_average_1m_at_start": os.getloadavg()[0],
        "cases": {}, "comparison": {},
    }
    ok = True
    with tempfile.TemporaryDirectory(prefix="host_step_timing_") as scratch:
        base = _base_tree(repo, args.base_rev, pathlib.Path(scratch))
        for batch, stack, names in PLAN:
            key = f"B{batch}_{stack}"
            runs: dict[str, list[dict]] = {name: [] for name in names}
            for round_index in range(args.rounds):
                order = names if round_index % 2 == 0 else tuple(reversed(names))
                for name in order:
                    spec = CASES[name]
                    case = dict(spec, name=name, batch=batch, stack=stack, steps=args.steps,
                                warmup=args.warmup, knobs=list(spec["knobs"]))
                    tree = base if spec["tree"] == "base" else repo
                    print(f"[host_step_timing] {key} round {round_index} {name} ...",
                          file=sys.stderr, flush=True)
                    runs[name].append(_run_case(script, tree, case))
            report["cases"][key] = cases = {name: _merge(runs[name]) for name in names}
            before = cases["before"]["host_critical_path"]["median_us"]
            comparison = {"before_median_us": before}
            for name in names[1:]:
                value = cases[name]["host_critical_path"]["median_us"]
                comparison[f"{name}_median_us"] = value
                comparison[f"{name}_saved_us"] = before - value
                comparison[f"{name}_saved_pct"] = 100.0 * (before - value) / before
            for name in ("after_sync", "after_async"):
                lower = cases[name]["host_critical_path"]["median_us"] < before
                comparison[f"{name}_lower_than_before"] = lower
                comparison[f"{name}_lower_in_every_round"] = all(
                    after < first for after, first in zip(
                        cases[name]["round_medians_us"], cases["before"]["round_medians_us"]))
                ok = ok and lower
            comparison["tokens_agree"] = len({
                json.dumps(cases[name]["all_tokens"]) for name in names}) == 1 and all(
                cases[name]["tokens_same_every_round"] for name in names)
            ok = ok and comparison["tokens_agree"]
            report["comparison"][key] = comparison
    report["after_lower_than_before_everywhere"] = ok
    report["load_average_1m_at_end"] = os.getloadavg()[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["comparison"], indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
