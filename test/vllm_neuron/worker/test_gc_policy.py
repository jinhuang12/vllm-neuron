# SPDX-License-Identifier: Apache-2.0
"""The GC policy a serving worker applies once after warmup (``gc_policy.py``).

CPython 3.12 runs a full (gen-2) collection when more than ``threshold2`` gen-1
collections ran since the last one AND ``long_lived_pending`` is at least
``long_lived_total / 4``. With CPython's ``threshold2`` = 10, bs=64 decode on the
TP=64 server reached a full pass over the whole heap every few hundred steps (3.5-6 s
each, a decode stall). ``rare_gen2`` raises only ``threshold2``.

It does not run vLLM's ``freeze_gc_heap()`` (a full collection, then ``gc.freeze()``)
at the end of warmup: that one call slowed every later bs=1 decode step of the TP=64
server, although a bs=1 step runs almost no collection. The default is ``off``
(``test_gc_policy_default.py``) until ``rare_gen2`` passes its bs=1 measurement. The
pin test below fails if ``rare_gen2`` or the default runs a collection or freezes
anything again.

Unit cases replace the freeze, ``gc.set_threshold``, ``gc.disable`` and the vLLM
GC-debug hook with recorders: the real calls would change the pytest process. The pin
runs the real policies with automatic GC disabled and restores the thresholds;
one case runs real policies in child processes (``test/perf/gc_decode_pattern.py``).

Run with ``VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1``.
"""

import gc
import importlib.util
import os
import subprocess
import sys
import weakref
from pathlib import Path

import pytest
import vllm.utils.gc_utils as gc_utils

from vllm_neuron import envs
from vllm_neuron.vllm.worker import gc_policy

ENV = "VLLM_NEURON_GC_POLICY"
REPO = Path(__file__).resolve().parents[3]
HARNESS = REPO / "test" / "perf" / "gc_decode_pattern.py"


@pytest.fixture
def calls(monkeypatch):
    """Ordered record of every GC call the policy makes."""
    record: list[tuple] = []
    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.setattr(gc_utils, "freeze_gc_heap", lambda: record.append(("freeze_gc_heap",)))
    monkeypatch.setattr(
        gc_utils, "maybe_attach_gc_debug_callback", lambda: record.append(("gc_debug",))
    )
    monkeypatch.setattr(gc, "set_threshold", lambda *a: record.append(("set_threshold", *a)))
    monkeypatch.setattr(gc, "disable", lambda: record.append(("disable",)))
    monkeypatch.setattr(gc, "freeze", lambda: record.append(("freeze",)))
    return record


def _young_thresholds() -> tuple[int, int]:
    t0, t1, _ = gc.get_threshold()
    return t0, t1


def test_rare_gen2_policy_raises_only_the_gen2_threshold(calls, monkeypatch):
    """Only threshold2 changes; gen-0/1 keep their thresholds; no freeze."""
    monkeypatch.setenv(ENV, "rare_gen2")
    t0, t1 = _young_thresholds()

    applied = gc_policy.apply_post_warmup_gc_policy()

    assert calls == [
        ("set_threshold", t0, t1, gc_policy.GEN2_THRESHOLD),
        ("gc_debug",),
    ]
    assert applied.policy == gc_policy.RARE_GEN2
    assert applied.frozen is False


def test_envs_default_is_the_policy_default(monkeypatch):
    """envs.py and gc_policy.py name the same default."""
    monkeypatch.delenv(ENV, raising=False)

    assert envs.VLLM_NEURON_GC_POLICY == gc_policy.DEFAULT_POLICY


class _Cycle:
    """A reference cycle with a weak handle: alive until a collection of its generation."""

    def __init__(self) -> None:
        self.me = self


@pytest.mark.parametrize("value", ["rare_gen2", None], ids=["rare_gen2", "unset-default"])
def test_rare_gen2_and_the_default_collect_nothing_and_freeze_nothing(monkeypatch, value):
    """PIN: neither ``rare_gen2`` nor the served default collects or freezes at warmup end.

    The full collection + freeze (``freeze_gc_heap()``) at the end of warmup is what
    slowed every bs=1 decode step of the TP=64 server. A cycle that sits in the oldest
    generation survives anything but a full collection, so it is the witness: it must
    still be alive after the policy, the freeze count and the collection counts must
    not move, and at most ``threshold2`` changes.
    """
    if value is None:
        monkeypatch.delenv(ENV, raising=False)  # the served default: knob unset
    else:
        monkeypatch.setenv(ENV, value)
    name = value or f"default ({gc_policy.DEFAULT_POLICY})"
    saved = gc.get_threshold()
    was_enabled = gc.isenabled()
    gc.disable()  # no automatic pass between the two snapshots
    try:
        witness = _Cycle()
        handle = weakref.ref(witness)
        del witness
        # Move every tracked object, the cycle included, to the oldest generation
        # without collecting it.
        gc.freeze()
        gc.unfreeze()
        frozen_before = gc.get_freeze_count()
        collections_before = [g["collections"] for g in gc.get_stats()]

        applied = gc_policy.apply_post_warmup_gc_policy()

        raised = applied.policy in gc_policy.RAISING_POLICIES
        assert handle() is not None, f"{name} ran a full collection"
        assert gc.get_freeze_count() == frozen_before, f"{name} froze objects"
        assert [g["collections"] for g in gc.get_stats()] == collections_before
        assert gc.get_threshold() == (
            saved[0], saved[1], gc_policy.GEN2_THRESHOLD if raised else saved[2]
        )
        assert applied.frozen is False
    finally:
        gc.unfreeze()
        gc.set_threshold(*saved)
        if was_enabled:
            gc.enable()
        gc.collect()


def test_freeze_rare_gen2_policy_freezes_then_raises_the_gen2_threshold(calls, monkeypatch):
    """``freeze_rare_gen2`` (the default before ``off``), for A/B runs."""
    monkeypatch.setenv(ENV, "freeze_rare_gen2")
    t0, t1 = _young_thresholds()

    applied = gc_policy.apply_post_warmup_gc_policy()

    assert calls == [
        ("freeze_gc_heap",),
        ("set_threshold", t0, t1, gc_policy.GEN2_THRESHOLD),
        ("gc_debug",),
    ]
    assert applied.policy == gc_policy.FREEZE_RARE_GEN2
    assert applied.frozen is True


def test_gen2_threshold_allows_at_most_one_full_pass_per_1000_steps():
    """A full pass needs threshold2 gen-1 passes; bs=64 runs <= 1 gen-1 per step."""
    assert gc_policy.GEN2_THRESHOLD >= 1000


def test_freeze_policy_is_the_freeze_alone(calls, monkeypatch):
    """``freeze`` is the e9aa679 behaviour: freeze, CPython thresholds kept."""
    monkeypatch.setenv(ENV, "freeze")

    applied = gc_policy.apply_post_warmup_gc_policy()

    assert calls == [("freeze_gc_heap",), ("gc_debug",)]
    assert applied.policy == gc_policy.FREEZE_ONLY
    assert applied.frozen is True


@pytest.mark.parametrize("value", ["off", " OFF "])
def test_kill_switch_leaves_cpython_default_gc(calls, monkeypatch, value):
    """``off``: no freeze, no threshold change, automatic collection stays on."""
    monkeypatch.setenv(ENV, value)
    before = gc.get_threshold()

    applied = gc_policy.apply_post_warmup_gc_policy()

    assert calls == [("gc_debug",)]
    assert gc.get_threshold() == before
    assert gc.isenabled()
    assert applied.policy == gc_policy.OFF
    assert applied.frozen is False


def test_unknown_policy_is_refused_before_any_gc_change(calls, monkeypatch):
    """A typo must not silently pick a policy."""
    monkeypatch.setenv(ENV, "freez")

    with pytest.raises(ValueError, match="'freez' is not one of rare_gen2, freeze_rare_gen2"):
        gc_policy.apply_post_warmup_gc_policy()

    assert calls == []


def _load_harness():
    spec = importlib.util.spec_from_file_location("gc_decode_pattern", HARNESS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_harness_measures_the_policy_the_worker_applies(monkeypatch):
    """The harness's chosen policy, ``rare_gen2``, runs through the worker's code."""
    harness = _load_harness()
    seen: list[str] = []
    monkeypatch.setattr(
        gc_policy,
        "apply_post_warmup_gc_policy",
        lambda policy=None: seen.append(policy),
    )

    harness.apply_policy(harness.CHOSEN_POLICY)

    assert harness.CHOSEN_POLICY == gc_policy.RARE_GEN2
    assert seen == [gc_policy.RARE_GEN2]


def _run_child(policy: str) -> dict:
    """One real policy run in a fresh interpreter (real freeze, real thresholds)."""
    env = dict(os.environ)
    env.pop(ENV, None)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    )
    out = subprocess.run(
        [sys.executable, str(HARNESS), "--child", policy, "--heap", "300000",
         "--steps", "1500", "--ranks", "4"],
        env=env, cwd=str(REPO), capture_output=True, text=True, timeout=600, check=True,
    )
    return _load_harness().parse_child_output(out.stdout)


def test_real_freeze_alone_runs_frequent_full_passes_and_rare_gen2_does_not():
    """Real CPython: freeze alone -> a full pass every few steps; rare_gen2 -> none."""
    freeze_only = _run_child(gc_policy.FREEZE_ONLY)
    rare = _run_child(gc_policy.RARE_GEN2)

    assert freeze_only["per_1000_steps"]["gen2"] >= 20, freeze_only["per_1000_steps"]
    assert rare["per_1000_steps"]["gen2"] <= 1, rare["per_1000_steps"]
    assert rare["per_1000_steps"]["gen0"] > 0
    assert rare["per_1000_steps"]["gen1"] > 0
    assert freeze_only["frozen_objects"] >= 300000
    assert rare["frozen_objects"] < 300000, "rare_gen2 froze the heap"
    assert rare["threshold_after"][2] == gc_policy.GEN2_THRESHOLD
