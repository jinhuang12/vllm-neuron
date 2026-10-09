# SPDX-License-Identifier: Apache-2.0
"""The garbage-collector policy a serving worker applies once, after warmup.

Why a policy
------------
After warmup a TP=64 GLM-5.3-Flash worker tracks 1.6-5.4 M objects. CPython 3.12 runs
a full (gen-2) pass when more than ``threshold2`` gen-1 passes ran since the last one
AND ``long_lived_pending >= long_lived_total / 4``. With CPython's ``threshold2`` = 10,
bs=64 decode (about one gen-1 pass per step on each rank) keeps reaching full passes:
2-6 stalls of 3.8-6 s per 119 steps on the TP=64 bs=64 line. Each pass scans the whole
heap, and each one is a decode stall, because the other 63 ranks wait for the rank in GC.

Why the default raises one threshold and freezes nothing
--------------------------------------------------------
The default, ``rare_gen2``, sets only ``threshold2`` to :data:`GEN2_THRESHOLD`. A full
pass then needs that many gen-1 passes: about one per 100 k bs=64 steps (4-6 h of
continuous bs=64 decode), none in practice at bs=1. Gen-0/1 thresholds do not change,
so those passes still collect the per-step cyclic garbage.

The default does not call vLLM's ``freeze_gc_heap()`` (``gc.collect(0/1/2)``, then
``gc.freeze()``), which vLLM's GPU worker runs at this point. On the TP=64 server that
one call at the end of warmup made every later bs=1 decode step slower: the same tree
with and without it served bs=1 at 22.4-24.2 ms against 19.6 ms ITL, with device compute
16.6 ms in both; the time is host-side lateness of single ranks at the all-rank start
barrier. It is not a collection: at bs=1 a worker ran ~20 gen-0 passes and one gen-1
pass in a whole launch (``VLLM_GC_DEBUG``), and on CPU (``test/perf/gc_step_cost.py``)
a decode step makes the same collections, callbacks and host calls with and without the
call. A freeze only shortens full passes, and with ``threshold2`` raised there are none
to shorten. The trade: the first full pass, after :data:`GEN2_THRESHOLD` gen-1 passes,
scans the whole heap (seconds), where a frozen heap leaves it the unfrozen objects only.

Policies (``VLLM_NEURON_GC_POLICY``)
------------------------------------
* ``rare_gen2`` (default): ``threshold2`` = :data:`GEN2_THRESHOLD`, no collection, no
  freeze.
* ``freeze_rare_gen2``: ``freeze_gc_heap()``, then ``threshold2`` raised (the default
  before ``rare_gen2``), for A/B runs.
* ``freeze``: ``freeze_gc_heap()`` alone (CPython thresholds kept), for A/B runs.
* ``off``: CPython's default GC, nothing changed.

Every policy then calls vLLM's ``maybe_attach_gc_debug_callback()`` (as vLLM's GPU
worker does), so ``VLLM_GC_DEBUG=1`` logs every collection with its time on each worker;
without ``VLLM_GC_DEBUG`` it attaches nothing (``gc_step_cost.py`` counts the callbacks).
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

#: ``threshold2`` after warmup: gen-1 passes between two full passes. A bs=64 decode
#: step runs ~1 gen-1 pass per rank, so this is about one full pass per 100 k steps
#: (~4-6 h at 160 ms per step), and none in practice at bs=1. Cyclic garbage that
#: reaches gen 2 waits that long to be freed; on the TP=64 server the full passes of a
#: bs=64 run freed 0 objects in total.
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
        # gc.get_freeze_count() walks the frozen list (~0.7 s at 4.7 M objects); once.
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
