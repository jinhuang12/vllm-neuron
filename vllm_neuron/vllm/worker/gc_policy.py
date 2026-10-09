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

Why the default is ``off``
--------------------------
On the TP=64 server, ``freeze_rare_gen2`` made every bs=1 decode step slower,
although no collection ran during decode. bs=1 ITL in ms, at the first point after
startup (then two quiet points ~2.5 and ~17 min later), and the all-rank start
barrier of a captured step, which is where the time went (device compute was
16.4-16.6 ms per step in all of them):

* policy on (3 launches of trees that contain it): 22.4-24.8 (21.1-21.9 /
  20.3-21.5), barrier 5.6-10.3;
* ``off`` on such a tree (1 launch): 20.1 (20.0 / 20.2), barrier 5.0;
* that tree with the policy's code removed (1 launch): 19.6 (19.6 / 18.7), barrier
  1.95.

With ``off`` the first two points came back; the late point and the barrier stayed
at the level of the launches with the policy. At runtime ``off`` is the same as the
code removed: the code that stays imports this module (about 90 GC-tracked objects,
once), reads the knob and logs one line. It changes no threshold, freezes nothing,
attaches no callback, and adds no barrier or warmup step. So the difference between
those two launches is launch-to-launch variance, not this code.

The freeze alone (``freeze``) was rejected earlier for a bs=1 cost of the same kind
(+1.4 ms per step, barrier 0.04 -> 3.05 ms). So the default keeps CPython's GC until
a policy that removes the bs=64 stalls passes its own bs=1 measurement. The trade:
with ``off``, bs=64 decode has its full-pass stalls again (3.8-6 s each, several per
120 steps); ``VLLM_NEURON_GC_POLICY=freeze_rare_gen2`` removes them and has the bs=1
cost above.

Policies (``VLLM_NEURON_GC_POLICY``)
------------------------------------
* ``off`` (default): no freeze and no threshold change: CPython's default GC.
* ``freeze_rare_gen2``: freeze, then raise only ``threshold2`` to
  :data:`GEN2_THRESHOLD`. Gen-0/1 passes keep their thresholds and still collect
  the per-step cyclic garbage; a full pass needs :data:`GEN2_THRESHOLD` gen-1 passes.
* ``freeze``: the freeze alone (CPython thresholds kept), for A/B runs.

All three then call vLLM's ``maybe_attach_gc_debug_callback()`` (as vLLM's GPU
worker does after its freeze), so ``VLLM_GC_DEBUG=1`` logs every collection with its
time on each worker. Without ``VLLM_GC_DEBUG`` it does nothing.

On CPU (``test/perf/gc_decode_pattern.py``: 5 M frozen objects, 5000 bs=64-like
steps), the freeze alone ran 61.8 full passes per 1000 steps and ``freeze_rare_gen2``
0, with the same gen-0/1 rates (~8.2 k / 0.75 k per 1000 steps) and RSS growth
within 1 MB of the run without a freeze (+7.4-7.5 MB against +6.8-6.9 MB, two runs).
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass

import vllm.utils.gc_utils as gc_utils

from vllm_neuron import envs

logger = logging.getLogger(__name__)

FREEZE_RARE_GEN2 = "freeze_rare_gen2"
FREEZE_ONLY = "freeze"
OFF = "off"
POLICIES = (FREEZE_RARE_GEN2, FREEZE_ONLY, OFF)
DEFAULT_POLICY = OFF

#: ``threshold2`` after the freeze: gen-1 passes between two full passes. A bs=64
#: decode step runs ~1 gen-1 pass per rank (gen-2 every ~11 steps with
#: ``threshold2`` = 10), so this is about one full pass per 100k steps (~4 h at
#: 160 ms per step), and none in practice at bs=1. Cyclic garbage that reaches
#: gen-2 waits that long to be freed; the A/B above saw none.
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

    frozen = policy != OFF
    if frozen:
        gc_utils.freeze_gc_heap()
    if policy == FREEZE_RARE_GEN2:
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
