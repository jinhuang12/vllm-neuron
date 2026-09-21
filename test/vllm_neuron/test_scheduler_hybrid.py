# SPDX-License-Identifier: Apache-2.0
"""Scheduler support for a hybrid KDA/DSA KV cache.

``NeuronScheduler.schedule`` resolves a per-request block demand from the KV
cache groups it was configured with, uses it to size a concurrency window, and
admits requests against that window instead of against the uniform one. The
tests below drive the real ``schedule()`` on an instance built with ``__new__``
(no ``VllmConfig``, no device) and read admission at the delegate boundary, so
they check the scheduling decision itself.

Both ``NeuronScheduler`` and ``NeuronAsyncScheduler`` override
``_call_base_schedule``, so the unchanged-decision test covers both bodies.
Every window here is counted in blocks, never bytes; no test asserts a page
size.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from types import SimpleNamespace

from test.vllm_neuron.worker.test_get_kv_cache_spec_hybrid import (
    _call,
    _fake_layers,
    _raw_fixture,
)

HYBRID_BLOCK_SIZE = 128

KDA_LAYERS = 34
DSA_LAYERS = 11
TOTAL_LAYERS = 45

# The fixture's request budget and batch, stated here so the block demand below
# is hand-checkable from this file alone.
FIXTURE_MAX_MODEL_LEN = 1024
FIXTURE_BATCH = 4

# A recurrent group holds one constant page; an attention group grows as
# cdiv(tokens, block_size).
BLOCKS_PER_REQUEST = KDA_LAYERS * 1 + DSA_LAYERS * (
    -(-FIXTURE_MAX_MODEL_LEN // HYBRID_BLOCK_SIZE)
)  # 34 + 11*8 = 122

# The uniform (non-hybrid) window, used as the input the hybrid resolution has
# to replace.
UNIFORM_KV_WINDOW = 1

# Decision digests for the non-hybrid fixtures, one per live scheduler class.
EXPECTED_DECISION_DIGESTS = {
    "NeuronScheduler/L1_idle_new_prefill": (
        "06b09bde45a91697f8c4107ce071e77afa7917551881c980aa1b651ef73181aa"
    ),
    "NeuronScheduler/L2_decode_only": (
        "07338de5b9d6be81c76e069bad9629504f0054ead65c80c613ccfa97997f8716"
    ),
    "NeuronScheduler/L3_prefill_in_running_hides_decode": (
        "720a6d10897beb7508eaabaf7f2ba160aa67c9b9614ca0b9dd84c16eda101185"
    ),
    "NeuronAsyncScheduler/L1_idle_new_prefill": (
        "534bac1af3531db479f807f4da45a2fb09ce0194553668cce0ce566b9c287363"
    ),
    "NeuronAsyncScheduler/L2_decode_only": (
        "ac30b68e65566b1e8a99935ddfdb319a6cde071a487a3cffc261ea07466fbf18"
    ),
    "NeuronAsyncScheduler/L3_prefill_in_running_hides_decode": (
        "ef2596df7883d4be2aeac68c124e0e4b6b967894f37e7a32c43beb8b21e7cb0b"
    ),
}

LEGACY_NUM_BLOCKS = 4096
LEGACY_KV_WINDOW = 4


def _specs() -> dict:
    """The 45 real KV cache specs, from the spec test's own helpers."""
    return _call(_fake_layers(_raw_fixture()))


def _groups(specs: dict, *, only_attention: bool = False) -> list:
    """One ``KVCacheGroupSpec`` per layer: 34 recurrent + 11 attention."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

    return [
        KVCacheGroupSpec([name], spec)
        for name, spec in specs.items()
        if not only_attention or isinstance(spec, FullAttentionSpec)
    ]


def _fake_request(rid: str, *, prompt: int, computed: int):
    """Only the attributes the admission path actually reads."""
    return SimpleNamespace(
        request_id=rid,
        num_prompt_tokens=prompt,
        num_computed_tokens=computed,
        status=None,
        structured_output_request=None,
        kv_transfer_params=None,
    )


def _stub_output():
    return SimpleNamespace(
        num_scheduled_tokens={},
        scheduled_new_reqs=[],
        scheduled_cached_reqs=None,
        kv_connector_metadata=None,
        _grammar_bitmask=None,
        _structured_output_request_ids=None,
    )


class _BasePatch:
    """Patch the two base ``schedule`` methods and record which one fired.

    Patching the base classes rather than ``self._call_base_schedule`` keeps the
    real per-class delegate bodies executing: an instance attribute would shadow
    both of them and leave the override unmeasured.
    """

    def __init__(self, seen: dict) -> None:
        self.seen = seen

    def __enter__(self):
        from vllm.v1.core.sched.async_scheduler import AsyncScheduler
        from vllm.v1.core.sched.scheduler import Scheduler

        self._cls = (Scheduler, AsyncScheduler)
        self._orig = (Scheduler.schedule, AsyncScheduler.schedule)
        seen = self.seen

        def mk(tag):
            def f(self_, throttle_prefills=False):
                seen["delegate_fired"] = tag
                seen["at_delegate_running"] = [r.request_id for r in self_.running]
                seen["at_delegate_waiting"] = [r.request_id for r in self_.waiting]
                seen["at_delegate_max_num_running_reqs"] = self_.max_num_running_reqs
                seen["at_delegate_max_kv_concurrent"] = self_._max_kv_concurrent
                seen["throttle"] = throttle_prefills
                return _stub_output()

            return f

        Scheduler.schedule = mk("Scheduler.schedule")
        AsyncScheduler.schedule = mk("AsyncScheduler.schedule")
        return self

    def __exit__(self, *exc):
        self._cls[0].schedule = self._orig[0]
        self._cls[1].schedule = self._orig[1]
        return False


def _build(cls, *, kv_window: int, num_blocks: int, groups, max_num_seqs: int):
    """A real instance with ``__init__`` bypassed.

    Real ``schedule``, real ``can_schedule``, real properties, real delegates --
    and no ``VllmConfig``, so the scheduling logic is what is under test and
    engine construction is not.
    """
    from vllm.v1.kv_cache_interface import KVCacheConfig

    from vllm_neuron.vllm.core.scheduler import SchedulerState

    s = cls.__new__(cls)
    s.waiting = []
    s.running = []
    s.skipped_waiting = []
    s.holdback_queue = deque()
    s.requests = {}
    s._state = SchedulerState.IDLE
    s._kv_exhaustion_warned = False
    s._max_kv_concurrent = kv_window
    s.max_num_seqs = max_num_seqs
    s.max_num_running_reqs = max_num_seqs
    s.max_prefills_per_batch = 1
    s.max_model_len = FIXTURE_MAX_MODEL_LEN
    s.num_batched_tokens_buckets = []
    s.num_seqs_buckets = [max_num_seqs]
    s.total_padding_tokens = 0
    s.total_scheduled_tokens = 0
    s.model_name = "test-scheduler"
    s.kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups
    )
    s.cache_config = SimpleNamespace(mamba_cache_mode="none")
    return s


def _drive(
    cls,
    *,
    kv_window,
    num_blocks,
    groups,
    max_num_seqs=FIXTURE_BATCH,
    running=(),
    holdback=(),
    waiting=(),
):
    """Run the real ``schedule()`` once; return ``(scheduler, seen, post)``."""
    s = _build(
        cls,
        kv_window=kv_window,
        num_blocks=num_blocks,
        groups=groups,
        max_num_seqs=max_num_seqs,
    )
    for rid, p, c in running:
        s.running.append(_fake_request(rid, prompt=p, computed=c))
    for rid, p, c in holdback:
        s.holdback_queue.append(_fake_request(rid, prompt=p, computed=c))
    for rid, p, c in waiting:
        s.waiting.append(_fake_request(rid, prompt=p, computed=c))
    s._apply_padding_and_log_stats = lambda o: o
    s.get_grammar_bitmask = lambda o: None
    seen: dict = {}
    with _BasePatch(seen):
        s.schedule()
    post = {
        "post_running": [r.request_id for r in s.running],
        "post_waiting": [r.request_id for r in s.waiting],
        "post_holdback": [r.request_id for r in s.holdback_queue],
        "post_state": str(s._state),
        "post_max_num_running_reqs": s.max_num_running_reqs,
        "post_max_kv_concurrent": s._max_kv_concurrent,
    }
    return s, seen, post


def _ramp(groups, num_blocks, kv_window):
    """Admitted-of-batch over a 4-request ramp, read at the delegate boundary.

    Admission cannot be read off ``self.waiting`` after ``schedule()`` returns:
    the last step drains the whole holdback queue back into ``waiting``, so a
    post-schedule reading reports every request as admitted.
    """
    from vllm_neuron.vllm.core.scheduler import NeuronScheduler

    admitted = 0
    trace = []
    for k in range(FIXTURE_BATCH):
        _, seen, post = _drive(
            NeuronScheduler,
            kv_window=kv_window,
            num_blocks=num_blocks,
            groups=groups,
            running=[(f"dec{i}", 64, 64) for i in range(k)],
            holdback=[(f"new{k}", 64, 0)],
        )
        at_delegate = seen.get("at_delegate_waiting", [])
        ok = f"new{k}" in at_delegate
        admitted += int(ok)
        trace.append(
            {
                "running_before": k,
                "admitted": ok,
                "at_delegate_waiting": at_delegate,
                "waiting_after_schedule": post["post_waiting"],
            }
        )
    return admitted, trace


def _legacy_fixtures():
    """Three non-hybrid fixtures: idle-prefill, decode-only, mixed."""
    return [
        ("L1_idle_new_prefill", {"running": [], "holdback": [("n0", 64, 0)], "waiting": []}),
        (
            "L2_decode_only",
            {"running": [("d0", 64, 64), ("d1", 64, 64)], "holdback": [], "waiting": []},
        ),
        (
            "L3_prefill_in_running_hides_decode",
            {
                "running": [("p0", 64, 10), ("d0", 64, 64)],
                "holdback": [("n1", 64, 0)],
                "waiting": [],
            },
        ),
    ]


def _decision_digest(cls, fixture, *, groups, num_blocks, kv_window):
    """A canonical, byte-comparable record of one schedule decision."""
    _, seen, post = _drive(
        cls,
        kv_window=kv_window,
        num_blocks=num_blocks,
        groups=groups,
        max_num_seqs=4,
        running=fixture["running"],
        holdback=fixture["holdback"],
        waiting=fixture["waiting"],
    )
    decision = {"delegate": seen, **post}
    blob = json.dumps(decision, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest(), decision


def test_four_request_batch_schedules_without_starvation() -> None:
    """All four requests of a ramp are admitted on the hybrid stack."""
    specs = _specs()
    assert len(specs) == TOTAL_LAYERS
    groups = _groups(specs)
    num_blocks = FIXTURE_BATCH * BLOCKS_PER_REQUEST

    admitted, trace = _ramp(groups, num_blocks, UNIFORM_KV_WINDOW)
    assert admitted == FIXTURE_BATCH, trace


def test_block_demand_equals_the_closed_form() -> None:
    """Block demand is ``34*1 + 11*cdiv(1024, 128) == 122``, exactly."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    specs = _specs()
    groups = _groups(specs)
    kda = sum(1 for s in specs.values() if isinstance(s, MambaSpec))
    dsa = sum(1 for s in specs.values() if isinstance(s, FullAttentionSpec))
    assert (kda, dsa) == (KDA_LAYERS, DSA_LAYERS)

    from vllm_neuron.vllm.core.scheduler import NeuronScheduler

    s, _, _ = _drive(
        NeuronScheduler,
        kv_window=UNIFORM_KV_WINDOW,
        num_blocks=FIXTURE_BATCH * BLOCKS_PER_REQUEST,
        groups=groups,
        holdback=[("n0", 64, 0)],
    )
    assert s._hybrid_kv_blocks_per_request == BLOCKS_PER_REQUEST
    assert s._hybrid_kv_window == FIXTURE_BATCH

    # Keep the group count and change only the recurrent/paged split: one
    # recurrent group is enough to engage the hybrid path, and the demand has to
    # follow the split rather than the number of groups.
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec

    attention_spec = _groups(specs, only_attention=True)[0].kv_cache_spec
    one_kda = _groups(specs, only_attention=True) + [
        KVCacheGroupSpec([f"retyped.{i}"], attention_spec)
        for i in range(KDA_LAYERS - 1)
    ]
    one_kda.append(
        KVCacheGroupSpec(
            ["kept.kda"],
            next(s for s in specs.values() if isinstance(s, MambaSpec)),
        )
    )
    s2, _, _ = _drive(
        NeuronScheduler,
        kv_window=UNIFORM_KV_WINDOW,
        num_blocks=FIXTURE_BATCH * BLOCKS_PER_REQUEST,
        groups=one_kda,
        holdback=[("n0", 64, 0)],
    )
    per_page = -(-FIXTURE_MAX_MODEL_LEN // HYBRID_BLOCK_SIZE)
    assert len(one_kda) == TOTAL_LAYERS
    assert s2._hybrid_kv_blocks_per_request == 1 * 1 + (TOTAL_LAYERS - 1) * per_page

    # With every recurrent group retyped away the hybrid path does not engage at
    # all: no window, no demand.
    all_attention = _groups(specs, only_attention=True) + [
        KVCacheGroupSpec([f"retyped.{i}"], attention_spec)
        for i in range(KDA_LAYERS)
    ]
    s3, _, _ = _drive(
        NeuronScheduler,
        kv_window=UNIFORM_KV_WINDOW,
        num_blocks=FIXTURE_BATCH * BLOCKS_PER_REQUEST,
        groups=all_attention,
        holdback=[("n0", 64, 0)],
    )
    assert s3._hybrid_kv_window == 0
    assert s3._hybrid_kv_blocks_per_request == 0


def test_non_hybrid_decisions_are_unchanged_for_both_classes() -> None:
    """A non-hybrid stack keeps its byte-identical schedule decision."""
    from vllm.v1.kv_cache_interface import MambaSpec

    from vllm_neuron.vllm.core.scheduler import (
        NeuronAsyncScheduler,
        NeuronScheduler,
    )

    specs = _specs()
    legacy_groups = _groups(specs, only_attention=True)
    assert legacy_groups, "the non-hybrid fixture must have attention groups"
    assert not any(
        isinstance(g.kv_cache_spec, MambaSpec) for g in legacy_groups
    ), "a fixture with a recurrent group is not a non-hybrid fixture"

    observed: dict = {}
    delegates: dict = {}
    for cls in (NeuronScheduler, NeuronAsyncScheduler):
        for name, fx in _legacy_fixtures():
            digest, decision = _decision_digest(
                cls,
                fx,
                groups=legacy_groups,
                num_blocks=LEGACY_NUM_BLOCKS,
                kv_window=LEGACY_KV_WINDOW,
            )
            observed[f"{cls.__name__}/{name}"] = digest
            delegates[f"{cls.__name__}/{name}"] = decision["delegate"]["delegate_fired"]

    assert observed == EXPECTED_DECISION_DIGESTS

    # Each class routed through its own delegate body, so neither reading is
    # measuring the other class's.
    assert delegates["NeuronScheduler/L2_decode_only"] == "Scheduler.schedule"
    assert (
        delegates["NeuronAsyncScheduler/L2_decode_only"] == "AsyncScheduler.schedule"
    )
