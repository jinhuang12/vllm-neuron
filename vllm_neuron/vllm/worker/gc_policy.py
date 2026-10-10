# SPDX-License-Identifier: Apache-2.0
"""The garbage-collector policy a serving worker applies once, after warmup.

Why a policy and not only the freeze
------------------------------------
After warmup a TP=64 GLM-5.3-Flash worker tracks millions of objects. Without a
freeze, each gen-2 (full) pass scans all of them, for seconds, and every one is a
decode stall, because the other 63 ranks wait for the rank in GC. ``freeze_gc_heap()``
(vLLM's ``gc.collect(0/1/2)`` then ``gc.freeze()``) takes that heap out of the scans.

CPython 3.12 runs a full pass when more than ``threshold2`` gen-1 passes ran since
the last full one AND ``long_lived_pending >= long_lived_total / 4``. Each full pass
sets ``long_lived_total`` to the objects it kept, and frozen objects are not in it.
So after the freeze, the first full pass leaves a small total, the 25 % rule stops
holding gen-2 back, and every 11th gen-1 trigger (``threshold2`` = 10) becomes a full
pass: in batch-64 decode, several per rank per hundred steps, each a few
milliseconds, on a different rank each step. On 64 lock-stepped ranks that is a late
rank on most steps. On a TP=64 server the worker full passes after the freeze
collected no object.

Why the default raises one threshold and freezes nothing
--------------------------------------------------------
On a TP=64 server, ``freeze_rare_gen2`` made every batch-1 decode step slower,
although no collection ran during decode. The time went to the all-rank start
barrier of a captured step; the device compute per step did not change. ``off``,
the same build without the policy's code, and ``rare_gen2`` did not have that cost.

The default, ``rare_gen2``, sets only ``threshold2`` to :data:`GEN2_THRESHOLD`. A
full pass then needs that many gen-1 passes: hours of continuous batch-64 decode,
and none in practice at batch 1. Gen-0/1 thresholds do not change, so those passes
still collect the per-step cyclic garbage. It does not call ``freeze_gc_heap()``.
The batch-1 cost followed that one call at the end of warmup: the launches without
it served the first decode steps faster than the launches with it. The decode steps
do the same GC work either way: at batch 1 a worker runs tens of gen-0 passes and one
gen-1 pass in a whole launch (``VLLM_GC_DEBUG``), and on CPU
(``test/perf/gc_step_cost.py``) a decode step makes the same collections, callbacks
and host calls with and without the call. A freeze only shortens full passes, and
with ``threshold2`` raised there are none to shorten.

The cost of the freeze also depends on the host: on a second server of the same
type it was smaller at the times measured there; its first decode steps after startup
were not measured. The barrier time also varies between launches without the
freeze; that spread comes from the launch or the host, not from this code: ``off``
imports this module, reads the knob and logs one line, and changes no GC state. The
freeze alone (``freeze``) was rejected earlier for a batch-1 cost of the same kind.

The trade: the first full pass, after :data:`GEN2_THRESHOLD` gen-1 passes, scans the
whole heap (seconds), where a frozen heap leaves it the unfrozen objects only. With
``off``, batch-64 decode has its full-pass stalls again (seconds each, several per
hundred steps).

Policies (``VLLM_NEURON_GC_POLICY``)
------------------------------------
* ``rare_gen2`` (default): ``threshold2`` = :data:`GEN2_THRESHOLD`, no collection,
  no freeze.
* ``off``: CPython's default GC, nothing changed.
* ``freeze_rare_gen2``: ``freeze_gc_heap()``, then ``threshold2`` raised, for A/B
  runs.
* ``freeze``: ``freeze_gc_heap()`` alone (CPython thresholds kept), for A/B runs.

Every policy then calls vLLM's ``maybe_attach_gc_debug_callback()`` (as vLLM's GPU
worker does), so ``VLLM_GC_DEBUG=1`` logs every collection with its time on each
worker; without ``VLLM_GC_DEBUG`` it attaches nothing (``gc_step_cost.py`` counts the
callbacks).

On CPU (``test/perf/gc_decode_pattern.py``: a 5 M-object heap, 5000 batch-64-like
steps), the freeze alone runs full passes, ``freeze_rare_gen2`` and ``rare_gen2``
none, with the same gen-0/1 rates; RSS grows a little more with either raised
threshold than with CPython's GC.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass

import vllm.utils.gc_utils as gc_utils

from vllm_neuron import envs

logger = logging.getLogger(__name__)

RARE_GEN2 = "rare_gen2"
FREEZE_RARE_GEN2 = "freeze_rare_gen2"
FREEZE_ONLY = "freeze"
OFF = "off"
POLICIES = (RARE_GEN2, FREEZE_RARE_GEN2, FREEZE_ONLY, OFF)
DEFAULT_POLICY = RARE_GEN2
#: Policies that run ``freeze_gc_heap()`` (a full collection and a freeze).
FREEZING_POLICIES = (FREEZE_RARE_GEN2, FREEZE_ONLY)
#: Policies that raise ``threshold2`` to :data:`GEN2_THRESHOLD`.
RAISING_POLICIES = (RARE_GEN2, FREEZE_RARE_GEN2)

#: ``threshold2`` after warmup: gen-1 passes between two full passes. A batch-64
#: decode step runs about one gen-1 pass per rank, so this is about one full pass per
#: 100 k steps (hours of continuous decode), and none in practice at batch 1. Cyclic
#: garbage that reaches gen 2 waits that long to be freed; on a TP=64 server the full
#: passes of a batch-64 run freed no object.
GEN2_THRESHOLD = 100_000


@dataclass(frozen=True)
class AppliedGCPolicy:
    """What :func:`apply_post_warmup_gc_policy` did."""

    policy: str
    frozen: bool
    threshold: tuple[int, int, int]


def apply_post_warmup_gc_policy(policy: str | None = None) -> AppliedGCPolicy:
    """Apply the post-warmup GC policy to this process, once.

    ``policy`` defaults to ``VLLM_NEURON_GC_POLICY``. An unknown name raises
    ``ValueError`` before any GC state changes.
    """
    if policy is None:
        policy = envs.VLLM_NEURON_GC_POLICY
    policy = policy.strip().lower()
    if policy not in POLICIES:
        raise ValueError(
            f"VLLM_NEURON_GC_POLICY={policy!r} is not one of {', '.join(POLICIES)}"
        )

    frozen = policy in FREEZING_POLICIES
    if frozen:
        gc_utils.freeze_gc_heap()
    if policy in RAISING_POLICIES:
        t0, t1, _ = gc.get_threshold()
        gc.set_threshold(t0, t1, GEN2_THRESHOLD)
    gc_utils.maybe_attach_gc_debug_callback()

    threshold = gc.get_threshold()
    if frozen:
        # gc.get_freeze_count() walks the frozen list (slow on a served heap); once.
        logger.info(
            "GC heap frozen after warmup: %d objects; GC policy %s, threshold %s",
            gc.get_freeze_count(),
            policy,
            threshold,
        )
    else:
        logger.info(
            "GC policy %s after warmup: heap not frozen, threshold %s",
            policy,
            threshold,
        )
    return AppliedGCPolicy(policy=policy, frozen=frozen, threshold=threshold)
