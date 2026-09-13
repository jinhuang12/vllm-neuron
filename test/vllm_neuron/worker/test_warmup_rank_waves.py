# SPDX-License-Identifier: Apache-2.0
"""Acceptance: warmup runs in rank waves, and one status exchange per wave holds the group.

THE DECLARED ACCEPTANCE COMMAND, verbatim:

    VLLM_NEURON_CPU_MODE=1 NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \\
      python -m pytest test/vllm_neuron/worker/test_warmup_rank_waves.py \\
      -q -rA -s -p no:randomly -p no:cacheprovider --timeout 600

Warmup traces and compiles in each rank's own process, so 64 ranks warming up at
once ask the host for 64 compiles at the same time. The waves bound that number
to the wave size, which ships at eight. Rank 0 goes alone first, because if its
compiled graph is reusable the ranks behind it read it instead of compiling; the
wave timings in the log say whether it was. After every wave every rank sums one
integer across the group: the sum is the wave's barrier AND it carries how many
ranks failed, so a rank that raises is reported to the others in that wave rather
than left for them to wait out the barrier timeout.

Sixteen items, ONE test each, no ``parametrize``:

* T01 -- rank 0 is alone in the first wave.
* T02 -- no wave after the first holds more than the wave size.
* T03 -- every rank appears exactly once, in rank order.
* T04 -- a wave size at or above the world size leaves one wave after rank 0's.
* T05 -- rank 0 exchanges once per wave, and works in the first wave only.
* T06 -- a rank in the last wave exchanges once per wave too, and works once.
* T07 -- a rank in a middle wave does the same.
* T08 -- the loop partitions by the size the variable holds, at a size that is NOT the default.
* T09 -- with the variable unset the wave size reads eight.
* T10 -- a set value is read through, not defaulted.
* T11 -- a set "0" is refused instead of read as the default.
* T12 -- a negative value is refused.
* T13 -- the rank that raises re-raises its own error.
* T14 -- the rank that raises contributes its 1 to the exchange BEFORE it re-raises.
* T15 -- rank 0 learns of that failure on the same wave and raises, naming the count.
* T16 -- a rank whose own wave is later learns on that wave too, and never works.

T02, T05 and T08 print the numbers they measured, so the transcript carries the wave
sizes, the exchange count and the off-default partition rather than only a verdict.
Run pytest with ``-s``.
"""

import pytest

from vllm_neuron import envs
from vllm_neuron.vllm.worker import neuron_worker
from vllm_neuron.vllm.worker.neuron_worker import warmup_rank_waves


class _Group:
    """A TP group that reports only the world size and this rank's index."""

    def __init__(self, world_size: int, rank: int) -> None:
        self.world_size = world_size
        self.rank_in_group = rank


def _run_on(monkeypatch, world_size, rank, wave, work=None, fail_on_wave=0, exchanges=None):
    """Run the wave runner as ``rank``; return (values exchanged, work calls made).

    ``fail_on_wave`` makes the exchange report one OTHER rank as failed on that wave,
    which is what a rank hears when a rank it cannot see has raised. Pass ``exchanges``
    to read what this rank contributed even when the run ends in a raise.
    """
    exchanges = [] if exchanges is None else exchanges
    already = len(exchanges)
    works = []
    monkeypatch.setattr(
        neuron_worker, "get_tp_group", lambda: _Group(world_size, rank)
    )
    # The variable, never the module attribute. ``vllm_neuron.envs`` serves these names from a
    # module ``__getattr__``, which Python consults only while the name is absent from the module
    # dict; patching the attribute makes monkeypatch write the value it read back as a real
    # attribute on undo, and that shadows the reader for every item after this one.
    monkeypatch.setenv("VLLM_NEURON_WARMUP_WAVE_SIZE", str(wave))

    def _sum(value):
        exchanges.append(value)
        # Counted from what THIS run exchanged, so a caller that passes a list already holding
        # values does not shift the wave the planted failure lands on.
        return value + (1 if len(exchanges) - already == fail_on_wave else 0)

    monkeypatch.setattr(neuron_worker, "tp_sum_int", _sum)
    neuron_worker.run_warmup_in_waves(
        "prefill", "2048/kv2048", work or (lambda: works.append(rank))
    )
    return exchanges, works


def test_first_wave_is_rank_zero_alone():
    assert warmup_rank_waves(64, 8)[0] == [0]


def test_no_wave_after_the_first_exceeds_the_wave_size():
    sizes = [len(ranks) for ranks in warmup_rank_waves(64, 8)]
    print(f"warmup_waves|world=64|wave=8|waves={len(sizes)}|sizes={sizes}")
    assert sizes == [1, 8, 8, 8, 8, 8, 8, 8, 7]


def test_every_rank_appears_exactly_once_in_order():
    flattened = [rank for ranks in warmup_rank_waves(64, 8) for rank in ranks]
    assert flattened == list(range(64))


def test_wave_size_at_or_above_the_world_leaves_one_wave_after_rank_zero():
    assert warmup_rank_waves(64, 64) == [[0], list(range(1, 64))]
    assert warmup_rank_waves(64, 1000) == [[0], list(range(1, 64))]


def test_rank_zero_exchanges_once_per_wave_and_works_in_the_first(monkeypatch):
    wave = 8
    exchanges, works = _run_on(monkeypatch, world_size=64, rank=0, wave=wave)
    print(f"warmup_exchanges|rank=0|exchanges={len(exchanges)}|works={works}")
    assert len(exchanges) == len(warmup_rank_waves(64, wave))
    assert works == [0]


def test_a_rank_in_the_last_wave_exchanges_once_per_wave_and_works_once(monkeypatch):
    wave = 8
    exchanges, works = _run_on(monkeypatch, world_size=64, rank=63, wave=wave)
    assert len(exchanges) == len(warmup_rank_waves(64, wave))
    assert works == [63]


def test_a_rank_in_a_middle_wave_exchanges_once_per_wave_and_works_once(monkeypatch):
    wave = 8
    exchanges, works = _run_on(monkeypatch, world_size=64, rank=30, wave=wave)
    assert len(exchanges) == len(warmup_rank_waves(64, wave))
    assert works == [30]


def test_the_loop_partitions_by_the_size_the_knob_holds(monkeypatch):
    # A size the shipped default is not. Every other item runs at eight, so a loop that ignored the
    # knob and partitioned by a literal eight would satisfy them all; this one it cannot satisfy.
    wave = 4
    exchanges, works = _run_on(monkeypatch, world_size=64, rank=30, wave=wave)
    print(f"warmup_knob|wave={wave}|exchanges={len(exchanges)}|works={works}")
    assert len(exchanges) == len(warmup_rank_waves(64, wave))
    assert works == [30]


def test_an_unset_wave_size_reads_eight(monkeypatch):
    monkeypatch.delenv("VLLM_NEURON_WARMUP_WAVE_SIZE", raising=False)
    assert envs.VLLM_NEURON_WARMUP_WAVE_SIZE == 8


def test_a_set_wave_size_is_read_through(monkeypatch):
    monkeypatch.setenv("VLLM_NEURON_WARMUP_WAVE_SIZE", "5")
    assert envs.VLLM_NEURON_WARMUP_WAVE_SIZE == 5


def test_a_set_zero_is_refused_rather_than_read_as_the_default(monkeypatch):
    monkeypatch.setenv("VLLM_NEURON_WARMUP_WAVE_SIZE", "0")
    with pytest.raises(ValueError, match="must be at least 1"):
        envs.VLLM_NEURON_WARMUP_WAVE_SIZE  # noqa: B018


def test_a_negative_wave_size_is_refused(monkeypatch):
    monkeypatch.setenv("VLLM_NEURON_WARMUP_WAVE_SIZE", "-3")
    with pytest.raises(ValueError, match="must be at least 1"):
        envs.VLLM_NEURON_WARMUP_WAVE_SIZE  # noqa: B018


def test_the_rank_that_raises_re_raises_its_own_error(monkeypatch):
    def _boom():
        raise RuntimeError("this rank could not compile")

    with pytest.raises(RuntimeError, match="this rank could not compile"):
        _run_on(monkeypatch, world_size=64, rank=9, wave=8, work=_boom)


def test_the_raising_rank_exchanges_before_it_re_raises(monkeypatch):
    exchanges = []

    def _boom():
        raise RuntimeError("this rank could not compile")

    with pytest.raises(RuntimeError, match="this rank could not compile"):
        _run_on(
            monkeypatch,
            world_size=64,
            rank=5,
            wave=8,
            work=_boom,
            exchanges=exchanges,
        )
    # Rank 5 warms up in the second wave: it contributed 0 to the first wave's exchange and 1 to
    # its own, so the group learned of the failure before this rank re-raised it.
    assert exchanges == [0, 1]


def test_rank_zero_learns_of_a_failure_on_the_wave_it_happened(monkeypatch):
    with pytest.raises(RuntimeError, match=r"wave=2/9: 1 rank\(s\) failed"):
        _run_on(monkeypatch, world_size=64, rank=0, wave=8, fail_on_wave=2)


def test_a_rank_whose_own_wave_is_later_learns_on_that_wave_too(monkeypatch):
    done = []

    with pytest.raises(RuntimeError, match=r"wave=2/9: 1 rank\(s\) failed"):
        _run_on(
            monkeypatch,
            world_size=64,
            rank=40,
            wave=8,
            work=lambda: done.append("worked"),
            fail_on_wave=2,
        )
    assert done == []
