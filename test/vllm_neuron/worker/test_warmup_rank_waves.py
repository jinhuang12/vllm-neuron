# SPDX-License-Identifier: Apache-2.0
"""Acceptance: warmup runs in rank waves, and every rank barriers once per wave.

THE DECLARED ACCEPTANCE COMMAND, verbatim:

    VLLM_NEURON_CPU_MODE=1 NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \\
      python -m pytest test/vllm_neuron/worker/test_warmup_rank_waves.py \\
      -q -rA -p no:randomly -p no:cacheprovider --timeout 600

Warmup traces and compiles in each rank's own process, so 64 ranks warming up at
once ask the host for 64 compiles at the same time. The waves bound that number.
Rank 0 goes alone first, because if its compiled graph is reusable the ranks
behind it read it instead of compiling; the wave timings in the log say whether
it was. Six items, ONE test each, no ``parametrize``:

* T01 -- rank 0 is alone in the first wave.
* T02 -- no wave after the first holds more than the wave size.
* T03 -- every rank appears exactly once, in rank order.
* T04 -- a wave size at or above the world size leaves one wave after rank 0's.
* T05 -- rank 0 barriers once per wave, and works in the first wave only.
* T06 -- a rank in the last wave barriers once per wave too, and works once.

T02 and T05 print the numbers they measured, so the transcript carries the wave
sizes and the barrier count rather than only a verdict. Run pytest with ``-s``.
"""

from types import SimpleNamespace

from vllm_neuron.vllm.worker import neuron_worker
from vllm_neuron.vllm.worker.neuron_worker import warmup_rank_waves


class _Group:
    """A TP group that reports only the world size and this rank's index."""

    def __init__(self, world_size: int, rank: int) -> None:
        self.world_size = world_size
        self.rank_in_group = rank


def _run_on(monkeypatch, world_size: int, rank: int, wave: int):
    """Run the wave runner as ``rank`` and return (barrier calls, work calls)."""
    barriers = []
    works = []
    monkeypatch.setattr(
        neuron_worker, "get_tp_group", lambda: _Group(world_size, rank)
    )
    monkeypatch.setattr(neuron_worker, "tp_barrier", lambda: barriers.append(1))
    monkeypatch.setattr(
        neuron_worker.envs, "VLLM_NEURON_WARMUP_WAVE_SIZE", wave, raising=False
    )
    neuron_worker.NeuronWorker._run_warmup_in_waves(
        SimpleNamespace(), "prefill", "2048/kv2048", lambda: works.append(rank)
    )
    return barriers, works


def test_first_wave_is_rank_zero_alone():
    assert warmup_rank_waves(64, 16)[0] == [0]


def test_no_wave_after_the_first_exceeds_the_wave_size():
    sizes = [len(ranks) for ranks in warmup_rank_waves(64, 16)]
    print(f"warmup_waves|world=64|wave=16|waves={len(sizes)}|sizes={sizes}")
    assert sizes == [1, 16, 16, 16, 15]


def test_every_rank_appears_exactly_once_in_order():
    flattened = [rank for ranks in warmup_rank_waves(64, 16) for rank in ranks]
    assert flattened == list(range(64))


def test_wave_size_at_or_above_the_world_leaves_one_wave_after_rank_zero():
    assert warmup_rank_waves(64, 64) == [[0], list(range(1, 64))]
    assert warmup_rank_waves(64, 1000) == [[0], list(range(1, 64))]


def test_rank_zero_barriers_once_per_wave_and_works_in_the_first(monkeypatch):
    barriers, works = _run_on(monkeypatch, world_size=64, rank=0, wave=16)
    print(f"warmup_barriers|rank=0|barriers={len(barriers)}|works={works}")
    assert len(barriers) == len(warmup_rank_waves(64, 16))
    assert works == [0]


def test_a_rank_in_the_last_wave_barriers_once_per_wave_and_works_once(monkeypatch):
    barriers, works = _run_on(monkeypatch, world_size=64, rank=63, wave=16)
    assert len(barriers) == len(warmup_rank_waves(64, 16))
    assert works == [63]
