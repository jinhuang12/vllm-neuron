# SPDX-License-Identifier: Apache-2.0
"""CPU-mode timing of the DSA side-cache slot hand-out for one new request at the bs=64 line.

What is measured, for a tree, on a real ``NeuronModelRunner`` at the served bs=64 geometry
(the GLM-5.3-Flash fixture config: 45 layers, 11 sparse-attention (DSA) layers with
``index_kpool`` 4 and ``index_head_dim`` 128, ``max_num_seqs`` 64, ``max_model_len`` 8192,
hybrid block 128; the side caches are the runner's own, ``[65, 2049, 128]`` and
``[65, 2, 4, 128]`` bf16 per DSA layer):

1. 63 requests are prefilled one at a time through ``execute_model`` (stub forward), so
   63 of the 64 request slots are held;
2. then each sample step finishes one held request (scattered over the slots) and admits
   a new one in the same step, the scheduler's own order, so the new request is handed
   the slot just freed. Per sample:

   * ``handout_ms``: wall time of ``_glm5next_request_slots`` (the hand-out, which
     empties both side caches of the slot in every DSA layer) plus
     ``_glm5next_position_arm`` (the opening at position 0, which empties the ring);
   * ``request_slots_ms`` and ``position_arm_ms``: the two parts;
   * ``converter_ms``: the whole ``_glm5next_model_kwargs`` call of that prefill step.

   Before each sample every side cache is planted with a non-zero value, so the reset
   always clears a dirty slot (a slot freed by a sequence that ran).
3. ``bytes``: for one extra, untimed sample, every aten op the hand-out and the opening
   dispatch is recorded (``TorchDispatchMode``): the bytes of their outputs are the bytes
   written (an in-place op's output is the tensor it wrote) and the bytes of their inputs
   that share a side cache's storage are the bytes read from the side caches.

The base tree is a read-only ``git worktree`` (``--base-tree``, default 3098da3); each
case runs in its own process with ``PYTHONPATH`` and the working directory set to its
tree, so the two trees' modules never mix. One torch thread per case, as vLLM's multiproc
executor runs each worker.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 OMP_NUM_THREADS=4 PYTHONPATH=$PWD \\
        python test/perf/host_slot_reset_bs64.py \\
        --json /home/ubuntu/glm53f-wt2/reports/slotreset_harness.json \\
        --base-tree /home/ubuntu/glm53f-wt2/slotreset-base
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

VOCAB = 154880
BATCH = 64
PROMPT = 100
PREFILL_BUCKET = 1024
MAX_MODEL_LEN = 8192
DECODE_CTX_BUCKET = 2048
BLOCK = 128
FIXTURE = pathlib.Path("test/vllm_neuron/model/glm5_next/fixtures")
KV_BYTES = 3 * 2**30
#: A bf16 value whose two bytes are both non-zero, so a cleared element always differs.
PLANTED = 1.5


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
    samples = int(case["samples"])

    neuron_config = {
        "num_batched_tokens_buckets": [PREFILL_BUCKET],
        "kv_segment_size_buckets": [MAX_MODEL_LEN],
        "decode_context_length_buckets": [DECODE_CTX_BUCKET],
        "hybrid_kv_block_size": BLOCK,
        "on_device_sampling_config": {"all_greedy": True},
    }
    config = EngineArgs(
        model=str(tree / FIXTURE), skip_tokenizer_init=True, max_model_len=MAX_MODEL_LEN,
        max_num_seqs=BATCH, max_num_batched_tokens=PREFILL_BUCKET, block_size=BLOCK,
        enforce_eager=True, enable_prefix_caching=False, async_scheduling=False,
        additional_config={"neuron_config": neuron_config},
    ).create_engine_config()
    hf = json.load(open(tree / FIXTURE / "config.json"))
    text = hf.get("text_config", hf)
    text["linear_attn_config"]["num_heads"] = 1  # per-rank KDA heads at TP=64

    device_logits = (torch.randn(1, VOCAB, generator=torch.Generator().manual_seed(7)) * 4.0
                     ).to(torch.bfloat16)

    def stub_forward(input_ids, *, layer_carriers, sampling_positions,
                     device_sampling_params=None, device_logit_mask=None, **_):
        rows = int(sampling_positions.shape[0])
        logits = device_logits.expand(rows, -1)
        if device_sampling_params is not None:
            return torch.argmax(logits, dim=-1).to(torch.int32)
        return logits.clone()

    rendezvous = tempfile.mkdtemp(prefix="host_slot_reset_rdv_")
    result: dict = {"tree": str(tree), "samples_requested": samples}
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

            next_block = [1]
            needed = PROMPT + 2

            def blocks_for(spec_) -> list[int]:
                count = -(-needed // BLOCK) if "Attention" in type(spec_).__name__ else 1
                first = next_block[0]
                next_block[0] += count
                return list(range(first, first + count))

            prompt_gen = torch.Generator().manual_seed(99)
            held_blocks: dict[str, tuple] = {}

            def prefill_step(request: str, finished: set, reuse=None):
                prompt = torch.randint(0, VOCAB, (PROMPT,), generator=prompt_gen).tolist()
                block_ids = reuse or tuple(blocks_for(group.kv_cache_spec) for group in groups)
                held_blocks[request] = block_ids
                new = NewRequestData(
                    req_id=request, prompt_token_ids=prompt, mm_features=[],
                    sampling_params=SamplingParams(temperature=0.0, max_tokens=16),
                    pooling_params=None, block_ids=block_ids,
                    num_computed_tokens=0, lora_request=None)
                step = SchedulerOutput(
                    scheduled_new_reqs=[new], scheduled_cached_reqs=CachedRequestData.make_empty(),
                    num_scheduled_tokens={request: PROMPT}, total_num_scheduled_tokens=PROMPT,
                    scheduled_spec_decode_tokens={}, scheduled_encoder_inputs={},
                    num_common_prefix_blocks=[0] * len(groups), finished_req_ids=set(finished),
                    free_encoder_mm_hashes=[])
                step.num_scheduled_tokens_padded = {request: PREFILL_BUCKET}
                out = runner.execute_model(step)
                out = runner.sample_tokens(None) if out is None else out
                if hasattr(out, "get_output"):
                    out.get_output()

            # 1. 63 of the 64 slots held.
            for index in range(BATCH - 1):
                prefill_step(f"held-{index}", set())
            table = runner._glm5next_request_slot_table
            sides = runner._glm5next_side_cache_set
            dsa = [side for side in sides if side]
            result["geometry"] = {
                "layers": len(sides), "dsa_layers": len(dsa),
                "pool_cache": list(dsa[0]["pool_cache"].shape),
                "tail": list(dsa[0]["tail"].shape),
                "dtype": str(dsa[0]["pool_cache"].dtype),
                "held_slots": len(table),
                "side_cache_bytes": sum(v.nbytes for side in dsa for v in side.values()),
            }

            def plant():
                for side in dsa:
                    for key in ("pool_cache", "tail"):
                        side[key].fill_(PLANTED)

            timers = {"request_slots": [], "position_arm": [], "converter": []}
            current: dict[str, float] = {}
            originals = {
                "request_slots": runner._glm5next_request_slots,
                "position_arm": runner._glm5next_position_arm,
                "converter": runner._glm5next_model_kwargs,
            }

            def timed(name):
                original = originals[name]

                def wrapper(*args, **kwargs):
                    started = time.perf_counter_ns()
                    try:
                        return original(*args, **kwargs)
                    finally:
                        current[name] = current.get(name, 0.0) + (
                            time.perf_counter_ns() - started) / 1e6
                return wrapper

            runner._glm5next_request_slots = timed("request_slots")
            runner._glm5next_position_arm = timed("position_arm")
            runner._glm5next_model_kwargs = timed("converter")

            # 2. Each sample: one held request finishes, a new one is admitted.
            order = torch.randperm(BATCH - 1, generator=torch.Generator().manual_seed(5)).tolist()
            victims = [f"held-{index}" for index in order]
            slot_of_new = []
            for sample in range(samples + 1):
                victim = victims[sample % len(victims)]
                if victim not in table:
                    victim = next(rid for rid in table if rid.startswith(("held-", "new-")))
                freed = table[victim]
                plant()
                gc.collect()
                gc.disable()
                current.clear()
                try:
                    rid = f"new-{sample}"
                    prefill_step(rid, {victim}, reuse=held_blocks.pop(victim))
                finally:
                    gc.enable()
                if table[rid] != freed:
                    raise RuntimeError(f"{rid} took slot {table[rid]}, not the freed {freed}")
                slot_of_new.append(freed)
                if sample == 0:
                    continue  # the first admission after the table settles is not timed
                for name in timers:
                    timers[name].append(current.get(name, 0.0))
            handout = [a + b for a, b in zip(timers["request_slots"], timers["position_arm"])]
            result["handout"] = _stats(handout)
            result["request_slots"] = _stats(timers["request_slots"])
            result["position_arm"] = _stats(timers["position_arm"])
            result["converter"] = _stats(timers["converter"])
            result["handout_samples_ms"] = handout
            result["slots_handed"] = slot_of_new

            # 3. Bytes moved by one hand-out (an extra, untimed sample), op by op: every
            # aten op the two calls dispatch is recorded; its outputs are bytes written
            # (an in-place op's output is the tensor it wrote), its inputs that share a
            # side cache's storage are bytes read from the side caches.
            from torch.utils._python_dispatch import TorchDispatchMode

            runner._glm5next_request_slots = originals["request_slots"]
            runner._glm5next_position_arm = originals["position_arm"]
            storages = {v.untyped_storage().data_ptr() for side in dsa for v in side.values()}

            class Bytes(TorchDispatchMode):
                def __init__(self):
                    super().__init__()
                    self.written = self.read_side = self.ops = 0
                    self.kinds: dict[str, int] = {}

                def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                    kwargs = kwargs or {}
                    out = func(*args, **kwargs)
                    if func.overloadpacket.__name__ in {"select", "slice", "view", "alias",
                                                        "as_strided", "detach", "_reshape_alias"}:
                        return out  # a view moves nothing
                    self.ops += 1
                    self.kinds[str(func)] = self.kinds.get(str(func), 0) + 1
                    outs = out if isinstance(out, (tuple, list)) else (out,)
                    self.written += sum(t.nbytes for t in outs if isinstance(t, torch.Tensor))
                    self.read_side += sum(
                        t.nbytes for t in list(args) + list(kwargs.values())
                        if isinstance(t, torch.Tensor)
                        and t.untyped_storage().data_ptr() in storages
                        and not (func._schema.is_mutable and t is args[0])
                        # a *_like factory reads its argument's metadata, not its data
                        and not func.overloadpacket.__name__.endswith("_like"))
                    return out

            victim = next(rid for rid in table if rid.startswith("held-"))
            freed = table[victim]
            plant()
            identities = [{k: id(side[k]) for k in ("pool_cache", "tail")} for side in dsa]
            runner._glm5next_note_finished_requests({victim})
            probe = Bytes()
            with probe:
                slots = runner._glm5next_request_slots(
                    root.glm5next_layer_banks, ["bytes-probe"], synthetic=False,
                    side_caches=runner._glm5next_side_cache_set)
                runner._glm5next_position_arm(int(slots[0]), 0,
                                              side_caches=runner._glm5next_side_cache_set,
                                              is_prefill=True)
            replaced = sum(1 for side, ids in zip(dsa, identities)
                           for k in ("pool_cache", "tail") if id(side[k]) != ids[k])
            fresh = all(bool((side[k][int(slots[0])] == 0).all())
                        for side in dsa for k in ("pool_cache", "tail"))
            slot_bytes = sum(side[k][0].nbytes for side in dsa for k in ("pool_cache", "tail"))
            result["bytes"] = {
                "written_per_handout": probe.written,
                "read_from_side_caches_per_handout": probe.read_side,
                "aten_ops_per_handout": probe.ops,
                "aten_op_kinds": probe.kinds,
                "side_cache_tensors_replaced": replaced,
                "one_slot_bytes_all_dsa_layers": slot_bytes,
                "slot_handed": int(slots[0]),
                "slot_freed": int(freed),
                "slot_fresh_after": fresh,
            }
            result["load_average_1m"] = os.getloadavg()[0]
        finally:
            dist_state.destroy_model_parallel()
            dist_state.destroy_distributed_environment()
    return result


def _run_case(script: pathlib.Path, tree: pathlib.Path, case: dict) -> dict:
    env = dict(os.environ)
    env.update({"PYTHONPATH": str(tree), "VLLM_NEURON_CPU_MODE": "1", "NKI_SIMULATOR": "1",
                "NEURON_PLATFORM_TARGET_OVERRIDE": "trn2", "VLLM_LOGGING_LEVEL": "WARNING",
                "OMP_NUM_THREADS": "1",
                "VLLM_NEURON_GLM5NEXT_ON_DEVICE_SAMPLING": "1",
                "VLLM_NEURON_GLM5NEXT_HOST_ONLY_METADATA": "1"})
    payload = json.dumps(dict(case, tree_path=str(tree)))
    proc = subprocess.run([sys.executable, str(script), "--child", payload], cwd=str(tree),
                          env=env, capture_output=True, text=True)
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
    if proc.returncode != 0 or not lines:
        raise RuntimeError(f"case tree={tree} failed (exit {proc.returncode}):\n"
                           f"{proc.stderr[-6000:]}")
    return json.loads(lines[-1][len("RESULT "):])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=pathlib.Path, required=False)
    parser.add_argument("--base-tree", type=pathlib.Path,
                        default=pathlib.Path("/home/ubuntu/glm53f-wt2/slotreset-base"))
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--trees", nargs="+", default=["base", "head"])
    parser.add_argument("--child")
    args = parser.parse_args()
    if args.child:
        print("RESULT " + json.dumps(_child(json.loads(args.child))), flush=True)
        return 0
    if args.samples < 20:
        parser.error("--samples must be at least 20")

    script = pathlib.Path(__file__).resolve()
    trees = {"base": args.base_tree.resolve(), "head": script.parents[2]}
    revs = {name: subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                                 check=True, capture_output=True, text=True).stdout.strip()
            for name, path in trees.items() if name in args.trees}
    dirty = {name: bool(subprocess.run(["git", "-C", str(trees[name]), "status", "--porcelain",
                                        "--untracked-files=no"],
                                       check=True, capture_output=True, text=True).stdout.strip())
             for name in revs}
    report: dict = {
        "what": __doc__.splitlines()[0],
        "trees": {name: {"path": str(trees[name]), "rev": revs[name], "dirty": dirty[name]}
                  for name in revs},
        "samples": args.samples, "cpu_count": os.cpu_count(),
        "load_average_1m_at_start": os.getloadavg()[0],
        "cases": {},
    }
    for name in args.trees:
        print(f"[host_slot_reset_bs64] {name} ...", file=sys.stderr, flush=True)
        report["cases"][name] = _run_case(script, trees[name], {"samples": args.samples})
    summary: dict = {}
    for name, case in report["cases"].items():
        summary[name] = {
            "handout_median_ms": case["handout"]["median_ms"],
            "handout_p90_ms": case["handout"]["p90_ms"],
            "request_slots_median_ms": case["request_slots"]["median_ms"],
            "position_arm_median_ms": case["position_arm"]["median_ms"],
            "converter_median_ms": case["converter"]["median_ms"],
            "n": case["handout"]["n"],
            "bytes_written_per_handout": case["bytes"]["written_per_handout"],
            "bytes_read_from_side_caches_per_handout":
                case["bytes"]["read_from_side_caches_per_handout"],
        }
    if {"base", "head"} <= set(summary):
        summary["speedup_handout_median"] = (
            summary["base"]["handout_median_ms"] / summary["head"]["handout_median_ms"])
        summary["bytes_ratio"] = (
            summary["base"]["bytes_written_per_handout"]
            / max(1, summary["head"]["bytes_written_per_handout"]))
        summary["acceptance"] = {
            "head_handout_below_20ms": summary["head"]["handout_median_ms"] < 20.0,
            "speedup_at_least_50x": summary["speedup_handout_median"] >= 50.0,
        }
    report["summary"] = summary
    report["load_average_1m_at_end"] = os.getloadavg()[0]
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
