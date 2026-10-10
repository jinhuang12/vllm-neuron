"""The shadow draft under async scheduling, two ranks on the CPU: every step's output comes back.

The incident this pins (a served run of 2026-10-08, ``VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT=5``,
TP=64): the server hung at the first decode step. Every device execution had completed; no
rank raised; the engine core timed out waiting for a free broadcast block because rank 0
never returned from ``sample_tokens``. Rank 0 is the one rank that scores drafts, and the
glue it ran read the PREVIOUS step's sampled ids back from the device on the worker's main
thread (``_glm5next_shadow_resolve``: ``sampled.cpu()``) while the async output thread was
reading the same future back inside ``AsyncNeuronModelRunnerOutput.get_output()``. The
runner's own contract for that future is one materialization under one lock ("prevents both
threads from materializing the same future concurrently", the class docstring); the glue
broke it, and one of the two waiters never woke.

What runs here: two ranks over gloo (a real tensor-parallel group of two, so the runner's
rank bookkeeping is the production code path and rank 1 is a rank that must stay silent),
each with the real ``NeuronModelRunner`` on the tiny GLM-5.3-Flash root with the knob on,
async scheduling and on-device sampling, driven through a prefill and two decodes the way
the worker drives it, with one thread per step calling ``get_output()`` the way the
worker's output loop does. Each rank runs the whole tiny root: the root's head sharding is
pinned to one rank (``_resolve_world_size``) because the hang is in the runner's host
threads, not in the model, and a model that shards over gloo would only add time.

The device is simulated, and the simulation is the point: on the CPU a tensor's ``.cpu()``
never waits, so the hang cannot show without a model of the NEFF future. The model, with
its evidence:

* an execution's outputs are futures; a host read (``.cpu()``, ``.tolist()``, ``.item()``,
  ``.numpy()``) waits for the execution (``neuron_model_runner.py``: "sampled_token_ids
  is still a device tensor future", "reading the output tensor(s) back to CPU forces the
  future to resolve");
* the device runs a step behind the host (the premise of async scheduling): an execution
  completes ``DEVICE_LAG_SECONDS`` after the host dispatched the following step, and on its
  own within ``DEVICE_STEP_SECONDS`` whatever the host does;
* an execution completes on every rank together (the NEFF's collectives): the simulated
  device threads meet in a barrier before completing a step;
* the completion is delivered to exactly ONE waiter (``libtorchneuron.so`` completes a
  future through ``nrta_event_register_seq_id_completion`` on an ``eventfd`` read under
  ``epoll_wait``; an eventfd read consumes the event). A later read of a completed output
  returns at once (the knob-5 warmup read ``(sampled, drafts)`` back in sequence on the
  device and finished), so sibling outputs and sequential reads are fine; two threads
  waiting on one incomplete future is the case the runner's lock exists for, and here it
  leaves one of them waiting forever, which is what the server did.

On the glue that read the previous step back on the main thread (``b654dc1``) this test
fails by the 60 s step bound with rank 0's progress stopping at ``dispatched 1`` (its main
thread) or its output thread never reporting step 0; on the fix every step is materialized, the
sampled ids agree across the two ranks, rank 0's log holds the two draft records (the
first scored against the token that followed, the second retired unscored at shutdown)
and rank 1 wrote no log.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/worker/test_shadow_draft_async_ranks.py
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
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_forward as tiny

pytestmark = [pytest.mark.forked]

K = 5
WORLD = 2
KNOB = "VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT"
LOG_KNOB = "VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT_LOG"
# The simulated device runs a step behind the host, which is the regime async scheduling
# exists for and the one the server hung in: an execution completes DEVICE_LAG_SECONDS after
# the host has dispatched the following step (or has nothing more to dispatch), so the
# previous step's futures are still incomplete while the host runs its glue for the next.
# Nothing the host does can hold the device beyond DEVICE_STEP_SECONDS per execution: a real
# step finishes on its own, and a host read that waits for it stalls, it does not deadlock.
DEVICE_LAG_SECONDS = 0.3
DEVICE_STEP_SECONDS = 10.0
# A prefill and two decodes: the first decode's drafts are scored against the second decode's
# token, the second decode's drafts retire unscored at shutdown (the server's one record).
STEPS = 3
# Both ranks build the tiny root and warm the runner (about 30 s on this machine) within the
# setup bound; from the moment both are warm, the three steps, their outputs and the shutdown
# must finish within the step bound (60 s). A rank that has not reported by then
# is hung: its progress file says where, its stack file (dumped shortly before the bound) says
# on which line.
SETUP_TIMEOUT_SECONDS = 240.0
STEP_TIMEOUT_SECONDS = 60.0
OUTPUT_THREAD_TIMEOUT_SECONDS = 30.0
STACK_DUMP_MARGIN_SECONDS = 10.0


# ── the simulated device ────────────────────────────────────────────────────


class _Execution:
    """One execution in the device queue: its outputs complete together, once.

    ``wait`` returns at once for a completed execution; otherwise it takes the one
    completion the device delivers. Two waiters on an incomplete execution leave one of
    them waiting forever (see the module docstring for the evidence this models).
    """

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
    """An output of an execution still in the device queue: a host read waits for it.

    Tensor operations on it (the next step consumes the sampled ids as its input on the
    device) return plain tensors and wait for nothing, as a NEFF input would.
    """

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
        # The runner's device is the CPU here, so ``.to(self.device)`` is the device-side
        # no-op it is in production and cannot be told from a host copy; the host reads
        # this model watches are ``cpu``, ``tolist``, ``item`` and ``numpy``.
        return torch.Tensor.to(self.as_subclass(torch.Tensor), *args, **kwargs)


class _Device:
    """A rank's device: completes the dispatched steps in order, a step behind the host, in
    lockstep with the other ranks' devices, each one once."""

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
        """The host has nothing more to dispatch: the device finishes what it holds."""
        with self._condition:
            self._flushing = True
            self._condition.notify_all()

    def attach(self, root) -> None:
        """Every forward of ``root`` from now on is an execution whose outputs are futures."""
        forward = root.forward

        def dispatched(*args, **kwargs):
            output = forward(*args, **kwargs)
            if isinstance(output, tuple):
                return self.dispatch(output)
            return self.dispatch((output,))[0]

        root.forward = dispatched


# ── one rank ────────────────────────────────────────────────────────────────


def _rank_main(rank: int, world: int, run_dir: str) -> None:
    """The worker of one rank: build, warm, three steps with an output thread each, shutdown."""
    run = pathlib.Path(run_dir)
    progress = run / f"progress_rank{rank}.txt"

    def mark(stage: str) -> None:
        with progress.open("a", encoding="utf-8") as handle:
            handle.write(stage + "\n")

    # A rank that hangs reports every thread's stack shortly before the parent gives up.
    stacks = (run / f"stacks_rank{rank}.txt").open("w", encoding="utf-8")
    faulthandler.dump_traceback_later(SETUP_TIMEOUT_SECONDS - STACK_DUMP_MARGIN_SECONDS, file=stacks)
    log = run / f"shadow_rank{rank}.jsonl"
    os.environ[KNOB] = str(K)
    os.environ[LOG_KNOB] = str(log)
    from vllm.config import set_current_vllm_config
    from vllm.distributed import parallel_state as dist_state

    from vllm_neuron.model.glm5_next import Glm5NextForConditionalGeneration, model_fp8
    from test.vllm_neuron.model.glm5_next import test_shadow_draft_e2e as shadow
    from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_first_request as first

    # The class declares the on-device sampler the engine config asks for (the tiny
    # first-request file's ``_declaring_a_sampler``), and the root shards over one rank.
    Glm5NextForConditionalGeneration.supports_on_device_sampling = True
    model_fp8._resolve_world_size = lambda: 1
    # The Neuron backend registers itself as the accelerator but, with no device in this
    # process, registers no hooks; ``torch.distributed.barrier`` asks for the accelerator and
    # would raise. A machine without one answers CPU, which is this machine's truth.
    torch._C._get_accelerator = lambda check=False: torch.device("cpu")
    config = first._engine_config(async_scheduling=True, on_device_sampling=True)
    prompt = first._prompt()
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
            root = shadow._world(max_model_len=first.e2e.E2E_MAX_SEQ_LEN).root
            runner = first._runner(config, root)
            assert runner.use_async_scheduling and runner.on_device_sampling
            assert runner._glm5next_shadow_k() == K
            first._warm(runner)
            device = _Device(device_group)
            device.attach(root)
            mark("warm")
            faulthandler.dump_traceback_later(
                STEP_TIMEOUT_SECONDS - STACK_DUMP_MARGIN_SECONDS, file=stacks
            )
            groups = first._groups(runner)
            steps = [first._prefill_step(prompt, groups)] + [
                first._decode_step(position=len(prompt) + n, generated=n + 1, groups=groups)
                for n in range(STEPS - 1)
            ]
            materialized: dict[int, list] = {}

            def output_loop(index: int, output) -> None:
                materialized[index] = output.get_output().sampled_token_ids
                mark(f"materialized {index}")

            threads = []
            for index, step in enumerate(steps):
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
                "ids": [materialized.get(index) for index in range(len(steps))],
                "output_threads_alive": [thread.is_alive() for thread in threads],
                "waiters": [execution.waiters for execution in device.executions],
                "log_exists": log.exists(),
                "records": [
                    json.loads(line) for line in log.read_text().splitlines() if line.strip()
                ] if log.exists() else [],
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


def test_two_ranks_finish_a_prefill_and_two_decodes_and_only_rank_zero_scores(tmp_path):
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
        for ids in result["ids"]:
            assert len(ids) == 1 and len(ids[0]) == 1 and type(ids[0][0]) is int, ids
        # One reader per future: the output thread of each step.
        assert result["waiters"] == [1] * STEPS, result
    assert results[0]["ids"] == results[1]["ids"], results
    # Rank 0 scores: the first decode's drafts against the second decode's token, the
    # second decode's drafts retired unscored at shutdown.
    records = results[0]["records"]
    tokens = [ids[0][0] for ids in results[0]["ids"]]
    assert [record["step"] for record in records] == [1, 2], records
    assert [record["position"] for record in records] == [tiny.STACK_TOKENS, tiny.STACK_TOKENS + 1]
    assert records[0]["actual"] == [tokens[2]] and records[0]["scored"] == 1, records
    assert records[1]["actual"] == [] and records[1]["scored"] == 0, records
    for record in records:
        assert len(record["drafts"]) == K and all(type(d) is int and d >= 0 for d in record["drafts"])
    # Rank 1 wrote nothing: its rank came from the tensor-parallel group, on the host.
    assert results[1]["log_exists"] is False and results[1]["records"] == []
