# SPDX-License-Identifier: Apache-2.0
"""The garbage-collector policy a serving worker applies once, after warmup.

Why a policy and not only the freeze
------------------------------------
After warmup a TP=64 GLM-5.3-Flash worker tracks ~5 M objects. Without a freeze,
each gen-2 (full) pass scans all of them: 3.5-6 s, and every one is a decode stall,
because the other 63 ranks wait for the rank in GC. ``freeze_gc_heap()`` (vLLM's
``gc.collect(0/1/2)`` then ``gc.freeze()``) takes that heap out of the scans.

CPython 3.12 runs a full pass when more than ``threshold2`` gen-1 passes ran since
the last full one AND ``long_lived_pending >= long_lived_total / 4``. Each full pass
sets ``long_lived_total`` to the objects it kept, and frozen objects are not in it.
So after the freeze, the first full pass leaves a small total, the 25 % rule stops
holding gen-2 back, and every 11th gen-1 trigger (``threshold2`` = 10) becomes a full
pass: ~11 per rank per 120 bs=64 steps, 3-12 ms each, on a different rank each step.
On 64 lock-stepped ranks that is a late rank on most steps. In the TP=64 A/B the
5,248 worker full passes after the freeze collected 0 objects in total.

Why the default raises one threshold and freezes nothing
--------------------------------------------------------
On the TP=64 server, ``freeze_rare_gen2`` made every bs=1 decode step slower,
although no collection ran during decode. bs=1 ITL in ms, at the first point after
startup (then two quiet points ~2.5 and ~17 min later), and the all-rank start
barrier of a captured step, which is where the time went (device compute was
16.4-16.6 ms per step in all of them):

* policy on (3 launches of trees that contain it): 22.4-24.8 (21.1-21.9 /
  20.3-21.5), barrier 5.6-10.3;
* ``off`` on such a tree (1 launch): 20.1 (20.0 / 20.2), barrier 5.0;
* that tree with the policy's code removed (1 launch): 19.6 (19.6 / 18.7), barrier
  1.95;
* ``rare_gen2`` on such a tree (1 launch): 19.3 (18.5 / 18.7), barrier 1.3.

The default, ``rare_gen2``, sets only ``threshold2`` to :data:`GEN2_THRESHOLD`. A
full pass then needs that many gen-1 passes: about one per 100 k bs=64 steps (4-6 h
of continuous bs=64 decode), none in practice at bs=1. Gen-0/1 thresholds do not
change, so those passes still collect the per-step cyclic garbage. It does not call
``freeze_gc_heap()``. On this host the early bs=1 cost above followed that one call
at the end of warmup: the three launches without it served the first point at
19.3-20.1 ms, the three with it at 22.4-24.8 ms. The later points and the barrier
vary between launches (below). The decode steps do the same GC work either way: at
bs=1 a worker ran ~20 gen-0 passes and one gen-1 pass in a whole launch
(``VLLM_GC_DEBUG``), and on CPU (``test/perf/gc_step_cost.py``) a decode step makes
the same collections, callbacks and host calls with and without the call. A freeze
only shortens full passes, and with ``threshold2`` raised there are none to shorten.

The cost of the freeze also depends on the host. On a second server of the same
type, the tree with ``freeze_rare_gen2`` on served bs=1 at 19.3 ms 10 min after
startup (18.9 ms warm). Between ~2.5 and ~17 min after startup this server served
20.3-21.9 ms with the policy on (above), 1-2.6 ms slower at comparable times. The
first point after startup was not measured there. The spread between launches
without the freeze (barrier 1.3-5.0 ms) comes from the launch or the host, not from
this code: ``off`` imports this module, reads the knob and logs one line, and
changes no GC state. The freeze alone (``freeze``) was rejected earlier for a bs=1
cost of the same kind (+1.4 ms per step, barrier 0.04 -> 3.05 ms).

The trade: the first full pass, after :data:`GEN2_THRESHOLD` gen-1 passes, scans the
whole heap (seconds), where a frozen heap leaves it the unfrozen objects only. With
``off``, bs=64 decode has its full-pass stalls again (3.8-6 s each, several per 120
steps).

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

On CPU (``test/perf/gc_decode_pattern.py``: 5 M-object heap, 5000 bs=64-like steps),
the freeze alone ran 61.8 full passes per 1000 steps, ``freeze_rare_gen2`` and
``rare_gen2`` 0, with the same gen-0/1 rates (~8.2 k / 0.75 k per 1000 steps); RSS
grew +7.4-7.5 MB with either raised threshold against +6.8-6.9 MB with CPython's GC.
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
