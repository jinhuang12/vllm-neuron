# SPDX-License-Identifier: Apache-2.0
"""CPU instrument: what the worker's post-warmup GC policy costs a cached decode step.

The question it answers: does the GC policy a serving worker applies after warmup
(``vllm_neuron/vllm/worker/gc_policy.py``) change the host time of a bs=1 cached decode
step, and by which route? Each regime (a tree plus a ``VLLM_NEURON_GC_POLICY`` value)
runs in its own child process on a real ``NeuronModelRunner`` holding the real
GLM-5.3-Flash root geometry (45 layers, 154,880-entry vocabulary, the served bs=1
recipe: on-device sampling, host-only metadata, async scheduling). The root's
``forward`` is a stub that returns what the device returns; its own time is subtracted,
so what is timed is the worker's host path (``NeuronWorker.execute_model`` then
``sample_tokens``; ``get_output`` apart, as the async output thread runs it).

The worker's heap is part of what the policy acts on, so before warmup the child
builds ``--heap`` tracked objects (default 1.65 M: the std-line TP=64 worker froze
1,647,121-1,647,123 objects). After warmup the child applies the policy the way the
worker does: through the tree's own ``apply_post_warmup_gc_policy()`` reading
``VLLM_NEURON_GC_POLICY`` (a tree without ``gc_policy`` -- the drop tree -- applies
nothing, as its worker does). Then ``--requests`` requests are served one after
another, each one prefill and ``--tokens`` decode steps, with automatic GC on.

Per decode step it records, outside the timed window:

* host time (critical path, and ``get_output``);
* every collection (``gc.callbacks``): generation and pause, and the tracked-object
  count ``gc.get_count()[0]`` the step started from (H-A, H-E);
* minor and major page faults and context switches (``getrusage``), and RSS every
  100 steps (H-B);
* ``len(gc.callbacks)`` before and after the policy, the instrument's own excluded
  (H-D: a callback the policy attaches costs every collection).

Then ``--profile-steps`` more steps run under ``sys.setprofile`` and count every
Python and C call per step by name: a regime that sends the step down a different host
path (a cache that the end-of-warmup collection emptied, say) shows up there.

The parent pools each regime's rounds (the regime order reverses every round, so host
load drift hits all alike) and models the TP=64 lock-step wait (H-C): every rank draws
its step time independently from the regime's pooled distribution, the collective waits
for the slowest of 64, and ``E[max of 64] - median`` is the barrier a regime's per-step
spread costs. ``--gc-only`` restricts that draw to the collection pauses. An
independent draw is the worst case for 64 unsynchronised ranks; ranks that collect on
the same step pay the pause once.

    PYTHONPATH=$PWD python test/perf/gc_step_cost.py --output gc_step_cost.json \\
        --regime tip_default=01ca4ca:freeze_rare_gen2 --regime tip_off=01ca4ca:off \\
        --regime drop=b7d0439:none --regime fix=worktree:default

A regime is ``name=TREE:POLICY``. ``TREE`` is a git revision (exported with
``git archive``) or ``worktree`` (this checkout, as is); ``POLICY`` is a
``VLLM_NEURON_GC_POLICY`` value, ``default`` (variable unset) or ``none`` (the policy is
not applied: a tree without ``gc_policy``).
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import pathlib
import random
import resource
import statistics
import subprocess
import sys
import tempfile
import time

VOCAB = 154880
PROMPT = 100
PREFILL_BUCKET = 1024
MAX_MODEL_LEN = 4096
#: The served bs=1 std line's buckets (gate serve config).
KV_SEGMENTS = [1024, 2048, 4096]
DECODE_CTX = 2048
BLOCK = 128
FIXTURE = pathlib.Path("test/vllm_neuron/model/glm5_next/fixtures")
SERVED_KNOBS = ("VLLM_NEURON_GLM5NEXT_ON_DEVICE_SAMPLING", "VLLM_NEURON_GLM5NEXT_HOST_ONLY_METADATA")
POLICY_ENV = "VLLM_NEURON_GC_POLICY"
#: ``POLICY`` values that are not ``VLLM_NEURON_GC_POLICY`` values.
POLICY_DEFAULT = "default"
POLICY_NONE = "none"
WORKTREE = "worktree"
RANKS = 64
RESULT_MARK = "GC_STEP_COST_RESULT "
HERE = pathlib.Path(__file__).resolve().parent


def _load_sibling(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))]


def _dist(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "median": round(statistics.median(values), 4),
        "mean": round(statistics.mean(values), 4),
        "p90": round(_percentile(values, 0.90), 4),
        "p99": round(_percentile(values, 0.99), 4),
        "max": round(max(values), 4),
    }


def _rss_mb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return float("nan")


def _old_garbage(heap: list, objects: int, buffers: int, buffer_kib: int) -> None:
    """Leave cyclic garbage in gen 2, interleaved with ``heap``, as warmup can.

    ``objects`` tracked lists in two-list cycles, each pair allocated between two live
    heap entries (so freeing them leaves holes among live objects), and ``buffers``
    ``buffer_kib`` KiB bytearrays each held by a cycle (large blocks the C allocator
    serves). They are unreachable at once, and ``gc.freeze(); gc.unfreeze()`` moves
    everything to the oldest generation without collecting: only a full collection --
    the one ``freeze_gc_heap()`` runs at the end of warmup -- frees them. Call with
    automatic GC disabled.
    """
    a: list | None = None
    b: list | None = None
    stride = max(1, len(heap) // max(1, objects // 2))
    for index in range(objects // 2):
        a = []
        b = [a]
        a.append(b)
        slot = (index * stride) % max(1, len(heap))
        heap[slot] = [heap[slot]]  # a live object allocated next to the garbage pair
    for _ in range(buffers):
        a = [bytearray(buffer_kib * 1024)]
        a.append(a)
    del a, b
    gc.freeze()
    gc.unfreeze()


# ── child: one regime in one process ─────────────────────────────────────────


def _child(case: dict) -> dict:
    import torch

    torch.set_num_threads(1)  # as vLLM's multiproc executor runs each worker
    from vllm.config import set_current_vllm_config
    from vllm.distributed import parallel_state as dist_state
    from vllm.engine.arg_utils import EngineArgs
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
    from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput

    import vllm_neuron
    import vllm_neuron.vllm.worker.neuron_model_runner as runner_module
    from vllm_neuron.model.glm5_next.model_fp8 import Glm5NextForConditionalGeneration as Root
    from vllm_neuron.vllm.worker.neuron_worker import NeuronWorker

    tree = pathlib.Path(case["tree_path"]).resolve()
    package = pathlib.Path(vllm_neuron.__file__).resolve()
    if tree not in package.parents:
        raise RuntimeError(f"vllm_neuron resolved to {package}, not under {tree}")
    print(f"vllm_neuron.__file__ {package} torch_threads {torch.get_num_threads()}",
          file=sys.stderr, flush=True)

    neuron_config = {
        "num_batched_tokens_buckets": [PREFILL_BUCKET],
        "num_seqs_buckets": [1],
        "kv_segment_size_buckets": KV_SEGMENTS,
        "decode_context_length_buckets": [DECODE_CTX],
        "hybrid_kv_block_size": BLOCK,
        "on_device_sampling_config": {"all_greedy": True},
    }
    config = EngineArgs(
        model=str(tree / FIXTURE), skip_tokenizer_init=True, max_model_len=MAX_MODEL_LEN,
        max_num_seqs=1, max_num_batched_tokens=PREFILL_BUCKET, block_size=BLOCK,
        enforce_eager=True, enable_prefix_caching=False, async_scheduling=True,
        additional_config={"neuron_config": neuron_config},
    ).create_engine_config()
    hf = json.load(open(tree / FIXTURE / "config.json"))
    text = hf.get("text_config", hf)
    text["linear_attn_config"]["num_heads"] = 1  # per-rank KDA heads at TP=64

    device_tokens = torch.arange(1, 65, dtype=torch.int32)
    stub_ns = [0]

    def stub_forward(input_ids, *, layer_carriers, sampling_positions, **_):
        started = time.perf_counter_ns()
        out = device_tokens[: int(sampling_positions.shape[0])].clone()
        stub_ns[0] += time.perf_counter_ns() - started
        return out

    # The worker's static heap (see the module docstring); built before warmup so the
    # end-of-warmup policy finds it where the worker's heap is.
    gc.disable()
    heap = _load_sibling("gc_decode_pattern")._build_heap(case["heap"])
    if case["garbage"] or case["garbage_buffers"]:
        _old_garbage(heap, case["garbage"], case["garbage_buffers"], case["garbage_buffer_kib"])
    gc.enable()

    rendezvous = tempfile.mkdtemp(prefix="gc_step_cost_rdv_")
    result: dict = {"vllm_neuron": str(package.parent), "torch_threads": torch.get_num_threads()}
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
            kv_config = get_kv_cache_configs(config, [runner.get_kv_cache_spec()], [512 * 2**20])[0]
            runner.initialize_kv_cache(kv_config)
            groups = kv_config.kv_cache_groups
            for segment in KV_SEGMENTS:
                runner.warmup_prefill(PREFILL_BUCKET, segment)
            runner.warmup_decode(1, ctx_bucket=DECODE_CTX)

            # The worker's own per-step entry points, on a worker that holds this runner.
            worker = NeuronWorker.__new__(NeuronWorker)
            worker.model_runner = runner
            worker._profiler = None

            # End of warmup: the worker applies its GC policy here.
            callbacks_before = len(gc.callbacks)
            collected_before = sum(g["collected"] for g in gc.get_stats())
            rss_before_apply = _rss_mb()
            started = time.perf_counter()
            applied = None
            if case["policy"] != POLICY_NONE:
                from vllm_neuron.vllm.worker import gc_policy

                applied = gc_policy.apply_post_warmup_gc_policy()
            result["policy_apply_ms"] = round((time.perf_counter() - started) * 1e3, 1)
            result["collected_by_policy"] = (
                sum(g["collected"] for g in gc.get_stats()) - collected_before)
            result["rss_change_at_apply_mb"] = round(_rss_mb() - rss_before_apply, 2)
            result["applied"] = repr(applied)
            result["callbacks_added_by_policy"] = len(gc.callbacks) - callbacks_before
            result["threshold_after"] = list(gc.get_threshold())
            result["gc_enabled_after"] = gc.isenabled()
            result["frozen_objects"] = gc.get_freeze_count()

            events: list[tuple[int, float, int]] = []  # (generation, ms, step)
            state = {"t0": 0, "step": -1}

            def on_gc(phase: str, info: dict) -> None:
                if phase == "start":
                    state["t0"] = time.perf_counter_ns()
                else:
                    events.append((info["generation"],
                                   (time.perf_counter_ns() - state["t0"]) / 1e6, state["step"]))

            gc.callbacks.append(on_gc)

            # One request at a time, so every request reuses the same pages: attention groups
            # get the pages its tokens need, the KDA groups their one state page.
            pages = -(-(PROMPT + case["tokens"] + case["profile_steps"] + 16) // BLOCK)
            blocks, next_page = [], 1
            for group in groups:
                count = pages if "Attention" in type(group.kv_cache_spec).__name__ else 1
                blocks.append(list(range(next_page, next_page + count)))
                next_page += count

            def schedule(new=None, cached=None, tokens=None, finished=()):
                step = SchedulerOutput(
                    scheduled_new_reqs=[new] if new else [],
                    scheduled_cached_reqs=cached or CachedRequestData.make_empty(),
                    num_scheduled_tokens=tokens, total_num_scheduled_tokens=sum(tokens.values()),
                    scheduled_spec_decode_tokens={}, scheduled_encoder_inputs={},
                    num_common_prefix_blocks=[0] * len(groups), finished_req_ids=set(finished),
                    free_encoder_mm_hashes=[])
                return step

            def run(step):
                out = NeuronWorker.execute_model(worker, step)
                if out is None:
                    out = NeuronWorker.sample_tokens(worker, None)
                return out

            prompt_generator = torch.Generator().manual_seed(99)
            steps: list[dict] = []
            rss: list[tuple[int, float]] = [(0, _rss_mb())]
            profile_counts: list[dict] = []
            finished: tuple = ()
            total_tokens = case["tokens"]
            step_index = 0
            for request_index in range(case["requests"] + 1):
                profiling = request_index == case["requests"]
                request = f"req-{request_index}"
                prompt = torch.randint(0, VOCAB, (PROMPT,), generator=prompt_generator).tolist()
                new = NewRequestData(
                    req_id=request, prompt_token_ids=prompt, mm_features=[],
                    sampling_params=SamplingParams(temperature=0.0, max_tokens=total_tokens + 1),
                    pooling_params=None, block_ids=tuple(blocks), num_computed_tokens=0,
                    lora_request=None)
                step = schedule(new=new, tokens={request: PROMPT}, finished=finished)
                step.num_scheduled_tokens_padded = {request: PREFILL_BUCKET}
                out = run(step)
                if hasattr(out, "get_output"):
                    out = out.get_output()
                position = PROMPT
                count = case["profile_steps"] if profiling else total_tokens
                for token in range(count):
                    cached = CachedRequestData(
                        req_ids=[request], resumed_req_ids=set(), new_token_ids=[[]],
                        all_token_ids={}, new_block_ids=[None],
                        num_computed_tokens=[position], num_output_tokens=[position - PROMPT + 1])
                    step = schedule(cached=cached, tokens={request: 1})
                    step.num_scheduled_tokens_padded = {request: 1}
                    position += 1
                    if profiling:
                        calls: dict[str, int] = {}

                        def tracer(frame, event, arg, _calls=calls):
                            if event == "call":
                                code = frame.f_code
                                key = f"{pathlib.Path(code.co_filename).name}:{code.co_name}"
                            elif event == "c_call":
                                key = f"C:{getattr(arg, '__qualname__', repr(arg))}"
                            else:
                                return
                            _calls[key] = _calls.get(key, 0) + 1

                        sys.setprofile(tracer)
                        try:
                            out = run(step)
                            if hasattr(out, "get_output"):
                                out = out.get_output()
                        finally:
                            sys.setprofile(None)
                        profile_counts.append(calls)
                        continue
                    state["step"] = step_index
                    first_event = len(events)
                    count0 = gc.get_count()[0]
                    usage0 = resource.getrusage(resource.RUSAGE_SELF)
                    stub_ns[0] = 0
                    t0 = time.perf_counter_ns()
                    out = run(step)
                    t1 = time.perf_counter_ns()
                    if hasattr(out, "get_output"):
                        out = out.get_output()
                    t2 = time.perf_counter_ns()
                    usage1 = resource.getrusage(resource.RUSAGE_SELF)
                    step_events = events[first_event:]
                    steps.append({
                        "host_ms": (t1 - t0 - stub_ns[0]) / 1e6,
                        "output_ms": (t2 - t1) / 1e6,
                        "gc_ms": sum(e[1] for e in step_events),
                        "gc_passes": [e[0] for e in step_events],
                        "gen0_count_at_start": count0,
                        "minflt": usage1.ru_minflt - usage0.ru_minflt,
                        "majflt": usage1.ru_majflt - usage0.ru_majflt,
                        "nivcsw": usage1.ru_nivcsw - usage0.ru_nivcsw,
                        "token": token,
                    })
                    step_index += 1
                    if step_index % 100 == 0:
                        rss.append((step_index, _rss_mb()))
                finished = (request,)
            state["step"] = -1
            gc.callbacks.remove(on_gc)
            rss.append((step_index, _rss_mb()))
            result["steps"] = steps
            result["events"] = events
            result["rss_mb"] = rss
            result["gc_stats_end"] = gc.get_stats()
            result["profile_counts"] = profile_counts
        finally:
            dist_state.destroy_model_parallel()
            dist_state.destroy_distributed_environment()
    del heap
    return result


# ── parent ───────────────────────────────────────────────────────────────────


def _tree(repo: pathlib.Path, rev: str, into: pathlib.Path) -> pathlib.Path:
    if rev == WORKTREE:
        return repo
    tree = into / f"tree-{rev}"
    if not tree.exists():
        tree.mkdir()
        archive = subprocess.run(
            ["git", "-C", str(repo), "archive", "--format=tar", rev, "vllm_neuron", str(FIXTURE)],
            check=True, capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", str(tree)], input=archive, check=True)
    return tree


def _parse_regime(text: str) -> tuple[str, str, str]:
    name, sep, spec = text.partition("=")
    rev, sep2, policy = spec.partition(":")
    if not (sep and sep2 and name and rev and policy):
        raise argparse.ArgumentTypeError(f"regime {text!r} is not NAME=TREE:POLICY")
    return name, rev, policy


def _run_case(tree: pathlib.Path, case: dict) -> dict:
    env = dict(os.environ)
    env.update({"PYTHONPATH": str(tree), "VLLM_NEURON_CPU_MODE": "1", "NKI_SIMULATOR": "1",
                "NEURON_PLATFORM_TARGET_OVERRIDE": "trn2", "VLLM_LOGGING_LEVEL": "WARNING",
                "OMP_NUM_THREADS": "1"})
    for name in SERVED_KNOBS:
        env[name] = "1"
    env.pop("VLLM_GC_DEBUG", None)
    env.pop(POLICY_ENV, None)
    if case["policy"] not in (POLICY_DEFAULT, POLICY_NONE):
        env[POLICY_ENV] = case["policy"]
    payload = json.dumps(dict(case, tree_path=str(tree)))
    proc = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve()), "--child", payload],
                          cwd=str(tree), env=env, capture_output=True, text=True)
    lines = [line for line in proc.stdout.splitlines() if line.startswith(RESULT_MARK)]
    stamp = [line for line in proc.stderr.splitlines() if line.startswith("vllm_neuron.__file__")]
    if proc.returncode != 0 or not lines:
        raise RuntimeError(f"regime {case['name']} failed (exit {proc.returncode}):\n"
                           f"{proc.stderr[-4000:]}")
    result = json.loads(lines[-1][len(RESULT_MARK):])
    result["stamp"] = stamp
    return result


def _lockstep(samples: list[float], draws: int, seed: int) -> dict:
    """E[max of RANKS independent draws] - median, and how often the slowest rank is
    more than 1 ms above the median."""
    if not samples:
        return {}
    rng = random.Random(seed)
    median = statistics.median(samples)
    excess = []
    for _ in range(draws):
        excess.append(max(rng.choice(samples) for _ in range(RANKS)) - median)
    return {
        "e_max_minus_median_ms": round(statistics.mean(excess), 4),
        "p90_max_minus_median_ms": round(_percentile(excess, 0.9), 4),
        "steps_slowest_over_1ms_frac": round(sum(1 for e in excess if e > 1.0) / draws, 4),
    }


def _summarise(name: str, runs: list[dict], draws: int) -> dict:
    steps = [s for run in runs for s in run["steps"]]
    host = [s["host_ms"] for s in steps]
    gc_ms = [s["gc_ms"] for s in steps]
    passes = [g for s in steps for g in s["gc_passes"]]
    pauses = {g: [e[1] for run in runs for e in run["events"] if e[0] == g and e[2] >= 0]
              for g in range(3)}
    n = len(steps)
    # Gen-1 pause against step index (H-E): first vs last fifth of each run.
    early, late = [], []
    for run in runs:
        total = len(run["steps"])
        for g, ms, step in run["events"]:
            if g == 1 and step >= 0:
                (early if step < total / 5 else late if step >= 4 * total / 5 else []).append(ms)
    rss = [run["rss_mb"][-1][1] - run["rss_mb"][0][1] for run in runs]
    calls = [sum(c.values()) for run in runs for c in run["profile_counts"]]
    return {
        "regime": name,
        "rounds": len(runs),
        "decode_steps": n,
        "stamp": runs[0]["stamp"],
        "applied": runs[0]["applied"],
        "threshold_after": runs[0]["threshold_after"],
        "gc_enabled_after": runs[0]["gc_enabled_after"],
        "frozen_objects": runs[0]["frozen_objects"],
        "callbacks_added_by_policy": sorted({run["callbacks_added_by_policy"] for run in runs}),
        "policy_apply_ms": [run["policy_apply_ms"] for run in runs],
        "collected_by_policy": [run["collected_by_policy"] for run in runs],
        "rss_change_at_apply_mb": [run["rss_change_at_apply_mb"] for run in runs],
        "host_ms": _dist(host),
        "host_ms_round_medians": [round(statistics.median([s["host_ms"] for s in run["steps"]]), 4)
                                  for run in runs],
        "output_ms": _dist([s["output_ms"] for s in steps]),
        "gc_passes_per_1000_steps": {f"gen{g}": round(passes.count(g) * 1000.0 / n, 3)
                                     for g in range(3)},
        "gc_pause_ms": {f"gen{g}": _dist(pauses[g]) for g in range(3)},
        "gc_ms_per_step": _dist(gc_ms),
        "steps_with_a_collection_frac": round(sum(1 for s in steps if s["gc_passes"]) / n, 4),
        "gen0_net_tracked_per_step": _dist([
            b["gen0_count_at_start"] - a["gen0_count_at_start"]
            for run in runs for a, b in zip(run["steps"], run["steps"][1:])
            if not a["gc_passes"]]),
        "gen1_pause_ms_first_fifth": _dist(early),
        "gen1_pause_ms_last_fifth": _dist(late),
        "minflt_per_step": _dist([float(s["minflt"]) for s in steps]),
        "majflt_total": sum(s["majflt"] for s in steps),
        "nivcsw_per_step": _dist([float(s["nivcsw"]) for s in steps]),
        "rss_growth_mb": [round(v, 2) for v in rss],
        "calls_per_profiled_step": sorted(set(calls)),
        "lockstep_64_host": _lockstep(host, draws, 1),
        "lockstep_64_gc_only": _lockstep(gc_ms, draws, 2),
    }


def _call_diff(results: dict[str, list[dict]], reference: str) -> dict:
    """Per-step call counts of every regime against ``reference`` (last profiled step)."""
    ref = results[reference][0]["profile_counts"][-1]
    diff = {}
    for name, runs in results.items():
        counts = runs[0]["profile_counts"][-1]
        delta = {key: counts.get(key, 0) - ref.get(key, 0)
                 for key in set(ref) | set(counts) if counts.get(key, 0) != ref.get(key, 0)}
        diff[name] = dict(sorted(delta.items()))
    return diff


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--regime", action="append", type=_parse_regime, default=[])
    parser.add_argument("--heap", type=int, default=1_650_000)
    parser.add_argument("--garbage", type=int, default=0,
                        help="cyclic garbage objects left in gen 2 at the end of warmup")
    parser.add_argument("--garbage-buffers", type=int, default=0)
    parser.add_argument("--garbage-buffer-kib", type=int, default=1024)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=150, help="decode steps per request")
    parser.add_argument("--profile-steps", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--child")
    args = parser.parse_args()
    if args.child:
        print(RESULT_MARK + json.dumps(_child(json.loads(args.child))), flush=True)
        return 0
    if args.output is None or not args.regime:
        parser.error("--output and at least one --regime are required")
    names = [name for name, _, _ in args.regime]
    if len(set(names)) != len(names):
        parser.error(f"regime names repeat: {names}")

    repo = pathlib.Path(__file__).resolve().parents[2]
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--short", "--untracked-files=no"],
                           check=True, capture_output=True, text=True).stdout.strip()
    print(f"[gc_step_cost] worktree {repo} head {head} dirty={bool(dirty)} "
          f"python {sys.version.split()[0]} cpus {os.cpu_count()} "
          f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}", flush=True)
    raw: dict[str, list[dict]] = {name: [] for name in names}
    with tempfile.TemporaryDirectory(prefix="gc_step_cost_") as scratch:
        trees = {name: _tree(repo, rev, pathlib.Path(scratch)) for name, rev, _ in args.regime}
        specs = {name: (rev, policy) for name, rev, policy in args.regime}
        for round_index in range(args.rounds):
            order = names if round_index % 2 == 0 else list(reversed(names))
            for name in order:
                rev, policy = specs[name]
                case = {"name": name, "rev": rev, "policy": policy, "heap": args.heap,
                        "garbage": args.garbage, "garbage_buffers": args.garbage_buffers,
                        "garbage_buffer_kib": args.garbage_buffer_kib,
                        "requests": args.requests, "tokens": args.tokens,
                        "profile_steps": args.profile_steps}
                started = time.perf_counter()
                raw[name].append(_run_case(trees[name], case))
                print(f"[gc_step_cost] round {round_index} {name} ({rev}:{policy}) "
                      f"{time.perf_counter() - started:.0f} s load1 {os.getloadavg()[0]:.1f} "
                      f"{raw[name][-1]['stamp']}", flush=True)
    report = {
        "what": __doc__.splitlines()[0],
        "head": head, "head_dirty": bool(dirty),
        "args": {k: v for k, v in vars(args).items() if k not in ("output", "child")},
        "regimes": {name: _summarise(name, raw[name], args.draws) for name in names},
        "call_diff_vs_first_regime": _call_diff(raw, names[0]),
        "loadavg_end": os.getloadavg(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    for name in names:
        r = report["regimes"][name]
        print(f"{name:>14} host ms {r['host_ms']}  gc/1k {r['gc_passes_per_1000_steps']} "
              f"minflt {r['minflt_per_step']['median']} cb+{r['callbacks_added_by_policy']} "
              f"lockstep {r['lockstep_64_host']} gc-only {r['lockstep_64_gc_only']}", flush=True)
    print(json.dumps(report["call_diff_vs_first_regime"], indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
