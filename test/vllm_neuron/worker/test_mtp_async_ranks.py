# SPDX-License-Identifier: Apache-2.0
"""The async drafter on two gloo ranks: a prefill and two verify steps, every output comes back.

What the async drafter must never do is read a step's device future on the worker's main
thread: the hook, the draft proposal and the next step's translator all run while the
step's output is still in flight, and the engine's output thread is the one reader of
each future (``AsyncNeuronModelRunnerOutput.get_output``, one materialisation under one
lock; ``test_shadow_draft_async_ranks`` pins the hang a second reader caused). Here two
ranks (a real tensor-parallel group over gloo, so the runner's rank bookkeeping is the
production path) each run the real ``NeuronModelRunner`` on the tiny GLM-5.3-Flash root
with speculative method "mtp", ``--async-scheduling``, on-device sampling and
``VLLM_NEURON_GLM5NEXT_MTP_ASYNC=1``, through a prefill and two drafted decode steps the
way the async engine drives the worker (the scheduler's optimistic positions and ``k``
placeholder drafts), with one output thread per step and the simulated device of the
shadow-draft test: outputs are futures, a host read waits for the execution, which
completes a step behind the host, and the completion wakes one waiter.

The on-device take and correction run their torch route in the rank processes: the NKI
simulator reads a kernel's inputs on the host to run it, which on the real device a
launch does not do, and the reads this test watches are the runner's own. The kernels are
held to the torch route bit for bit in ``functional/test_mtp_async_step.py``.

What fails on a runner that reads a future back on the main thread: a reader that never
returns trips the step bound (the rank's progress stops at ``dispatched 1`` or
``dispatched 2``, its stacks name the read); a reader that returns once the simulated
device completes the step shows as a second waiter on that future (``waiters``) -- the
serialisation the async drafter removes. A runner that forces the synchronous drafter
under async scheduling (its refusal bypassed) fails earlier still: the
synchronous proposal reads the sampler's output as host lists and the device future is a
tensor (``len() of a 0-d tensor`` at the prefill's proposal), the first of the host reads
(``_glm5next_propose_drafts``, then the hook's ``list(sampled_token_ids)``) that path makes.
On the async drafter every step is materialised once (``waiters`` one per step), both
ranks return the same ids, each decode keeps ``1 .. 1 + k`` of them, and the one step
counted as a synchronous fallback is the prefill, whose input ids are the prompt itself;
every decode step takes the async input path.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/worker/test_mtp_async_ranks.py
"""

from __future__ import annotations

import faulthandler
import json
import os
import pathlib
import threading
import time

import pytest
import torch
import torch.multiprocessing as mp

from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_e2e as e2e

pytestmark = [pytest.mark.forked]

K = 3
T = 1 + K
WORLD = 2
HEAD_KNOB = "VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT"
ASYNC_KNOB = "VLLM_NEURON_GLM5NEXT_MTP_ASYNC"
# The simulated device: see ``test_shadow_draft_async_ranks`` for the evidence this models.
DEVICE_LAG_SECONDS = 0.3
DEVICE_STEP_SECONDS = 10.0
# A prefill and two verify steps: the first verifies the prefill's placeholder drafts,
# the second the root's drafts from the first, each built while the previous is in flight.
STEPS = 3
SETUP_TIMEOUT_SECONDS = 240.0
STEP_TIMEOUT_SECONDS = 60.0
OUTPUT_THREAD_TIMEOUT_SECONDS = 30.0
STACK_DUMP_MARGIN_SECONDS = 10.0


# ── the simulated device (the shadow-draft test's) ──────────────────────────


class _Execution:
    def __init__(self) -> None:
        self._completion = threading.Semaphore(0)
        self._done = threading.Event()
        self.waiters = 0
        self.dispatched_at = time.monotonic()

    def complete(self) -> None:
        self._completion.release()

    def wait(self) -> None:
        if self._done.is_set():
            return
        self.waiters += 1
        self._completion.acquire()
        self._done.set()


class _Future(torch.Tensor):
    """An output still in the device queue: a host read waits; tensor operations return
    plain tensors and wait for nothing, as a NEFF input would."""

    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def of(tensor: torch.Tensor, execution: _Execution) -> "_Future":
        future = tensor.as_subclass(_Future)
        future._execution = execution
        return future

    def _resolved(self) -> torch.Tensor:
        self._execution.wait()
        return self.as_subclass(torch.Tensor)

    def cpu(self):
        return self._resolved()

    def tolist(self):
        return self._resolved().tolist()

    def item(self):
        return self._resolved().item()

    def numpy(self, *args, **kwargs):
        return self._resolved().numpy(*args, **kwargs)

    def to(self, *args, **kwargs):
        return torch.Tensor.to(self.as_subclass(torch.Tensor), *args, **kwargs)


class _Device:
    def __init__(self, group) -> None:
        self._group = group
        self._condition = threading.Condition()
        self._flushing = False
        self.executions: list[_Execution] = []
        self._thread = threading.Thread(target=self._run, name="simulated-device", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        index = 0
        while True:
            with self._condition:
                while len(self.executions) <= index:
                    self._condition.wait()
                execution = self.executions[index]
                deadline = execution.dispatched_at + DEVICE_STEP_SECONDS
                while not (
                    len(self.executions) > index + 1 or self._flushing
                ) and time.monotonic() < deadline:
                    self._condition.wait(timeout=deadline - time.monotonic())
            time.sleep(DEVICE_LAG_SECONDS)
            torch.distributed.barrier(group=self._group)
            execution.complete()
            index += 1

    def dispatch(self, outputs):
        execution = _Execution()
        with self._condition:
            self.executions.append(execution)
            self._condition.notify_all()
        return tuple(
            _Future.of(value, execution) if torch.is_tensor(value) else value
            for value in outputs
        )

    def flush(self) -> None:
        with self._condition:
            self._flushing = True
            self._condition.notify_all()

    def attach(self, root) -> None:
        forward = root.forward

        def dispatched(*args, **kwargs):
            output = forward(*args, **kwargs)
            if isinstance(output, tuple):
                return self.dispatch(output)
            return self.dispatch((output,))[0]

        root.forward = dispatched


# ── one rank ────────────────────────────────────────────────────────────────


def _rank_main(rank: int, world: int, run_dir: str) -> None:
    run = pathlib.Path(run_dir)
    progress = run / f"progress_rank{rank}.txt"

    def mark(stage: str) -> None:
        with progress.open("a", encoding="utf-8") as handle:
            handle.write(stage + "\n")

    stacks = (run / f"stacks_rank{rank}.txt").open("w", encoding="utf-8")
    faulthandler.dump_traceback_later(SETUP_TIMEOUT_SECONDS - STACK_DUMP_MARGIN_SECONDS, file=stacks)
    os.environ[HEAD_KNOB] = str(K)
    os.environ[ASYNC_KNOB] = "1"
    from vllm.config import set_current_vllm_config
    from vllm.distributed import parallel_state as dist_state

    from vllm_neuron.functional.mtp import async_step
    from vllm_neuron.model.glm5_next import Glm5NextForConditionalGeneration, model_fp8
    from test.vllm_neuron.model.glm5_next import test_mtp_e2e_spec as spec
    from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_first_request as first
    from test.vllm_neuron.worker import test_mtp_async_identity as identity

    Glm5NextForConditionalGeneration.supports_on_device_sampling = True
    model_fp8._resolve_world_size = lambda: 1
    torch._C._get_accelerator = lambda check=False: torch.device("cpu")
    # The torch route for the take and the correction (module docstring).
    async_step.can_run_kernel = lambda tensor: False
    config = identity._async_config(K)
    prompt = spec._prompts()[0]
    with set_current_vllm_config(config, check_compile=False):
        dist_state.init_distributed_environment(
            world_size=world,
            rank=rank,
            distributed_init_method=f"file://{run / 'rendezvous'}",
            local_rank=rank,
            backend="gloo",
        )
        dist_state.ensure_model_parallel_initialized(world, 1)
        try:
            device_group = torch.distributed.new_group(backend="gloo")
            root, _head = spec._root()
            runner = spec._runner(config, root)
            assert runner.use_async_scheduling and runner.on_device_sampling
            assert runner.is_mtp_spec and runner.drafter.async_steps is True
            device = _Device(device_group)
            device.attach(root)
            mark("warm")
            faulthandler.dump_traceback_later(
                STEP_TIMEOUT_SECONDS - STACK_DUMP_MARGIN_SECONDS, file=stacks
            )
            groups = first._groups(runner)
            blocks = list(range(spec.FIRST_BLOCK, spec.FIRST_BLOCK + -(-len(prompt) // spec.PAGE)))
            materialized: dict[int, list] = {}
            drafts_seen: list[list[int]] = []

            def output_loop(index: int, output) -> None:
                materialized[index] = output.get_output().sampled_token_ids
                mark(f"materialized {index}")

            threads = []
            handed = len(prompt)
            for index in range(STEPS):
                if index == 0:
                    step = spec._prefill("async-ranks", prompt, groups, blocks, set())
                else:
                    drafts = identity._scheduler_drafts(runner)
                    drafts_seen.append(drafts)
                    rows = 1 + len(drafts)
                    needed = -(-(handed + rows) // spec.PAGE)
                    new = list(range(spec.FIRST_BLOCK + len(blocks), spec.FIRST_BLOCK + needed))
                    blocks += new
                    step = spec._decode("async-ranks", handed, index - 1, groups, drafts, new)
                    handed += rows
                mark(f"dispatched {index}")
                _logits, output = first._step(runner, step)
                mark(f"returned {index}")
                thread = threading.Thread(target=output_loop, args=(index, output), daemon=True)
                thread.start()
                threads.append(thread)
            device.flush()
            for thread in threads:
                thread.join(timeout=OUTPUT_THREAD_TIMEOUT_SECONDS)
            runner.ensure_kv_transfer_shutdown()
            mark("shutdown")
            result = {
                "rank": rank,
                "tp_rank": int(dist_state.get_tp_group().rank_in_group),
                "ids": [materialized.get(index) for index in range(STEPS)],
                "drafts_seen": drafts_seen,
                "output_threads_alive": [thread.is_alive() for thread in threads],
                "waiters": [execution.waiters for execution in device.executions],
                "async_steps": int(runner._async_steps),
                "sync_fallbacks": int(runner._sync_fallback_steps),
            }
            (run / f"result_rank{rank}.json").write_text(json.dumps(result))
        finally:
            dist_state.destroy_model_parallel()
            dist_state.destroy_distributed_environment()


# ── the test ────────────────────────────────────────────────────────────────


def _progress(run: pathlib.Path) -> dict[int, list[str]]:
    report = {}
    for rank in range(WORLD):
        path = run / f"progress_rank{rank}.txt"
        report[rank] = path.read_text().split("\n")[:-1] if path.exists() else []
    return report


def test_two_ranks_finish_a_prefill_and_two_verify_steps_reading_nothing_back_on_the_main_thread(tmp_path):
    e2e._require_cpu_mode()
    context = mp.start_processes(
        _rank_main, args=(WORLD, str(tmp_path)), nprocs=WORLD, join=False, start_method="spawn"
    )
    deadline = time.monotonic() + SETUP_TIMEOUT_SECONDS
    bound = "setup"
    finished = False
    while not finished and time.monotonic() < deadline:
        finished = context.join(timeout=1.0)
        if bound == "setup" and all("warm" in stages for stages in _progress(tmp_path).values()):
            bound = "step"
            deadline = time.monotonic() + STEP_TIMEOUT_SECONDS
    if not finished:
        progress = _progress(tmp_path)
        for process in context.processes:
            if process.is_alive():
                process.kill()
        stacks = {
            rank: (tmp_path / f"stacks_rank{rank}.txt").read_text()
            for rank in range(WORLD) if (tmp_path / f"stacks_rank{rank}.txt").exists()
        }
        limit = STEP_TIMEOUT_SECONDS if bound == "step" else SETUP_TIMEOUT_SECONDS
        pytest.fail(
            f"the ranks did not finish within the {bound} bound of {limit:.0f} s; progress per "
            f"rank: {progress}; thread stacks per rank: {stacks}"
        )
    results = [json.loads((tmp_path / f"result_rank{rank}.json").read_text()) for rank in range(WORLD)]
    assert [result["tp_rank"] for result in results] == list(range(WORLD))
    for result in results:
        assert result["output_threads_alive"] == [False] * STEPS, result
        assert all(ids is not None for ids in result["ids"]), result
        prefill, *decodes = result["ids"]
        assert len(prefill) == 1 and len(prefill[0]) == 1, prefill
        for ids in decodes:
            assert len(ids) == 1 and 1 <= len(ids[0]) <= T, ids
            assert all(type(value) is int and value >= 0 for value in ids[0]), ids
        # The scheduler reserved k placeholders before each verify step.
        assert result["drafts_seen"] == [[-1] * K] * (STEPS - 1), result
        # One reader per future: the output thread of each step.
        assert result["waiters"] == [1] * STEPS, result
        # Both verify steps took their input ids from the previous step's future; the
        # prefill builds its own from the prompt and is counted as the generic swap's one
        # host-built step.
        assert (result["async_steps"], result["sync_fallbacks"]) == (STEPS - 1, 1), result
    assert results[0]["ids"] == results[1]["ids"], results
