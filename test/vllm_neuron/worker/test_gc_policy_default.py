# SPDX-License-Identifier: Apache-2.0
"""The worker's default GC policy is ``off``: CPython's own GC after warmup.

``freeze_rare_gen2`` (freeze the heap after warmup, then raise ``threshold2``) stops
the bs=64 full-pass stalls, but on the TP=64 server it made every bs=1 decode step
slower (``gc_policy.py``). So the default changes nothing until a policy without
that cost is measured. These cases pin the default, keep every policy selectable by
its name, and refuse any other name before the GC changes.

The freeze, ``gc.set_threshold``, ``gc.disable``, ``gc.freeze`` and the vLLM
GC-debug hook are replaced with recorders: the real calls would change the pytest
process.

Run with ``VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1``.
"""

import gc

import pytest
import vllm.utils.gc_utils as gc_utils

from vllm_neuron import envs
from vllm_neuron.vllm.worker import gc_policy

ENV = "VLLM_NEURON_GC_POLICY"
#: The knob's accepted values, spelled out: they are the operator interface.
ACCEPTED = ("freeze_rare_gen2", "freeze", "off")


@pytest.fixture
def calls(monkeypatch):
    """Ordered record of every GC call the policy makes; the knob starts unset."""
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


def test_unset_knob_applies_off_and_leaves_cpython_gc_unchanged(calls):
    """No knob set: no freeze, no threshold change, automatic collection stays on."""
    before = gc.get_threshold()

    applied = gc_policy.apply_post_warmup_gc_policy()

    assert envs.VLLM_NEURON_GC_POLICY == "off"
    assert gc_policy.DEFAULT_POLICY == "off"
    assert applied == gc_policy.AppliedGCPolicy(
        policy="off", frozen=False, threshold=before
    )
    assert calls == [("gc_debug",)]
    assert gc.get_threshold() == before
    assert gc.isenabled()


@pytest.mark.parametrize("value", ACCEPTED)
def test_every_policy_is_still_selectable_by_name(calls, monkeypatch, value):
    """The opt-in policies stay available (``freeze_rare_gen2``: the bs=64 stall fix)."""
    monkeypatch.setenv(ENV, f" {value.upper()} ")

    applied = gc_policy.apply_post_warmup_gc_policy()

    assert gc_policy.POLICIES == ACCEPTED
    assert applied.policy == value
    assert applied.frozen is (value != "off")
    assert (("freeze_gc_heap",) in calls) is (value != "off")


def test_unknown_policy_is_refused_by_name_before_any_gc_change(calls, monkeypatch):
    """A typo must not silently fall back to a policy."""
    monkeypatch.setenv(ENV, "freez")

    with pytest.raises(
        ValueError,
        match=(
            r"^VLLM_NEURON_GC_POLICY='freez' is not one of freeze_rare_gen2, freeze, off$"
        ),
    ):
        gc_policy.apply_post_warmup_gc_policy()

    assert calls == []
