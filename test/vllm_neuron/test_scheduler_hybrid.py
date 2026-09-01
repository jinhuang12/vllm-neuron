# SPDX-License-Identifier: Apache-2.0
"""``inc-glm53f-021`` acceptance -- WP2: scheduler support for the hybrid window.

THE DECLARED ACCEPTANCE, composed in full (Tier T, D1/D2):

    VLLM_NEURON_CPU_MODE=1 NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \\
      python -m pytest test/vllm_neuron/test_scheduler_hybrid.py -q \\
      --timeout 60 -p no:cacheprovider

Expected numeric exit **0** (D1.1). Both env terms live in the process
invocation, never a fixture (D2). ``VLLM_SSM_CONV_STATE_LAYOUT`` is deliberately
ABSENT: every count here is a product of extents or a queue membership, so no
fixture asserts a conv extent and the term would pin nothing.
``VLLM_CACHE_ROOT`` appears nowhere. No collected-item count is declared
(D1.2, ``-q``); the count is RECORDED off ``--collect-only -q``.

THREE COUNTED CONJUNCTS, one item each, no ``parametrize`` (section 6 rule 6).
The block predates the D1.4/D1.5 declaration form, so each conjunct SUPPLIES its
certifying component and a non-vacuity control that MOVES -- the accepted
``-017``/``-019``/``-020`` precedent.

  1. **0 of 4 requests starved** over a 4-request ramp. PARENT: **3** starved
     (1 of 4 admitted). Certifying component: the Step 1.5 hybrid window plus
     the Step 2 scoped override and the Step 3 prefill gate in
     ``NeuronScheduler.schedule``. Control: neutralise the window on the
     instance and the count moves **0 -> 3**.
  2. **Block demand equals the closed form exactly**: ``34*1 + 11*cdiv(1024,128)
     == 122``. PARENT: the method computes no demand at all, and vLLM's own
     byte-priced arithmetic implies **375** blocks per request -- so the
     equality cannot be satisfied by the pre-existing number. Control: retyping
     the recurrent groups as attention moves the demand **122 -> 360**.
  3. **Byte-identical schedule decision on 3/3 legacy fixtures, for BOTH live
     scheduler classes** (6 digests). Certifying component: the ``MambaSpec``
     opt-in guard -- the reason a non-hybrid stack keeps the pin's decision.
     Control: injecting one recurrent group into a legacy fixture moves the
     digest, so byte-identity is a discriminating reading and not a tautology.

WHY BOTH CLASSES. ``_call_base_schedule`` is defined twice -- ``NeuronScheduler``
and ``NeuronAsyncScheduler`` -- in two DIFFERENT classes, so it is a subclass
override and BOTH bodies are live (measured: ``ast`` reports owners
``NeuronScheduler`` and ``NeuronAsyncScheduler``; at runtime the two bind
distinct ``co_firstlineno``). Covering one class would leave one live delegate
unmeasured. This widens no declared value: 3/3 stays 3/3, measured per class.

UNITS. Every window here is in BLOCKS, never bytes. A byte-priced window would
read ``page_size_padded`` / the mamba page size, whose mechanism and prohibition
belong to the KV-spec increment that owns that field. This file asserts nothing
about either, and asserts no page size.

NOT SELF-REFERENTIAL. The 45 real specs come from the LANDED ``inc-glm53f-016``
acceptance module's own helpers, so this file's construction and that
increment's construction are the same one (the ``-017`` precedent). No spec
geometry is hand-written here.

FALSE-PASS DOOR, CLOSED. Admission is read AT THE DELEGATE BOUNDARY, never from
``self.waiting`` after ``schedule()`` returns: Step 5 drains the whole holdback
queue back into ``waiting`` unconditionally, so a post-schedule reading contains
REFUSED requests too and reports "admitted" for every case. Measured: the parent
ramp reads 4/4 admitted on the post-schedule queue and 1/4 at the delegate.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections import deque
from types import SimpleNamespace

from test.vllm_neuron.worker.test_get_kv_cache_spec_hybrid import (
    _call,
    _fake_layers,
    _raw_fixture,
)

#: ``approvals/DECISIONS.md`` section 6: the frozen user-decided block size.
#: CITED, never re-derived (P9).
REGISTERED_HYBRID_BLOCK_SIZE = 128

DECLARED_KDA_ENTRIES = 34
DECLARED_DSA_ENTRIES = 11
DECLARED_TOTAL_ENTRIES = 45

#: The fixture's request budget and batch. Declared here so the closed form is
#: hand-checkable from this file alone.
FIXTURE_MAX_MODEL_LEN = 1024
FIXTURE_BATCH = 4

#: THE CLOSED FORM, stated as an algebraic expectation rather than a bare int:
#: a recurrent group holds a CONSTANT page, an attention group grows as
#: ``cdiv(tokens, block_size)``.
CLOSED_FORM_BLOCKS_PER_REQUEST = DECLARED_KDA_ENTRIES * 1 + DECLARED_DSA_ENTRIES * (
    -(-FIXTURE_MAX_MODEL_LEN // REGISTERED_HYBRID_BLOCK_SIZE)
)  # 34 + 11*8 = 122

#: Measured PARENT readings, named here so an arm that stops discriminating is
#: visible in this file and not only in the evidence record.
#: ``increments/probe-021-parent-readings.py``, ``PROBE_EXIT_CODE=0``.
PARENT_STARVED_COUNT = 3
PARENT_ADMITTED_OF_BATCH = 1
PARENT_VLLM_IMPLIED_BLOCKS_PER_REQUEST = 375
PARENT_UNIFORM_WINDOW = 1

#: Parent decision digests, 3 legacy fixtures x 2 live classes.
PARENT_LEGACY_DIGESTS = {
    ("NeuronScheduler", "L1_idle_new_prefill"): (
        "06b09bde45a91697f8c4107ce071e77afa7917551881c980aa1b651ef73181aa"
    ),
    ("NeuronScheduler", "L2_decode_only"): (
        "07338de5b9d6be81c76e069bad9629504f0054ead65c80c613ccfa97997f8716"
    ),
    ("NeuronScheduler", "L3_prefill_in_running_hides_decode"): (
        "720a6d10897beb7508eaabaf7f2ba160aa67c9b9614ca0b9dd84c16eda101185"
    ),
    ("NeuronAsyncScheduler", "L1_idle_new_prefill"): (
        "534bac1af3531db479f807f4da45a2fb09ce0194553668cce0ce566b9c287363"
    ),
    ("NeuronAsyncScheduler", "L2_decode_only"): (
        "ac30b68e65566b1e8a99935ddfdb319a6cde071a487a3cffc261ea07466fbf18"
    ),
    ("NeuronAsyncScheduler", "L3_prefill_in_running_hides_decode"): (
        "ef2596df7883d4be2aeac68c124e0e4b6b967894f37e7a32c43beb8b21e7cb0b"
    ),
}

LEGACY_NUM_BLOCKS = 4096
LEGACY_KV_WINDOW = 4


def _record(**readings: object) -> None:
    """Put a reading in the ``-q`` transcript (``-075``'s convention)."""
    for key, value in readings.items():
        warnings.warn(f"RECORDED {key}={value!r}", UserWarning, stacklevel=2)


def _specs() -> dict:
    """The 45 real specs, from ``-016``'s landed helpers."""
    return _call(_fake_layers(_raw_fixture()))


def _groups(specs: dict, *, only_attention: bool = False) -> list:
    """One ``KVCacheGroupSpec`` per layer: 34 KDA + 11 DSA layers DECLARED."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec

    return [
        KVCacheGroupSpec([name], spec)
        for name, spec in specs.items()
        if not only_attention or isinstance(spec, FullAttentionSpec)
    ]


def _fake_request(rid: str, *, prompt: int, computed: int):
    """Only the four attributes the admission path actually reads."""
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
    """Patch the two BASE ``schedule`` methods, so the REAL
    ``_call_base_schedule`` bodies execute and record WHICH one fired.

    Overriding ``self._call_base_schedule`` on the instance would shadow both
    bodies and leave the per-class delegate unmeasured. Patching a vLLM function
    is not an env-setting fixture, so section 6 rule 3 is untouched.
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
    """A real instance with ``__init__`` BYPASSED.

    Real ``schedule``, real ``can_schedule``, real properties, real delegates --
    and no ``VllmConfig``, so the vehicle certifies the scheduling logic and
    nothing about engine construction.
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
    s.model_name = "acceptance"
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
    neutralise_window=False,
):
    """Run the REAL ``schedule()`` once; return ``(scheduler, seen, post)``."""
    s = _build(
        cls,
        kv_window=kv_window,
        num_blocks=num_blocks,
        groups=groups,
        max_num_seqs=max_num_seqs,
    )
    if neutralise_window:
        # CONTROL: pre-seed the cache so the Step 1.5 resolution is skipped and
        # the window stays 0 -- the parent's behaviour reproduced with the
        # child's code, which is what makes the control a control.
        s._hybrid_kv_window = 0
        s._hybrid_kv_blocks_per_request = 0
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


def _ramp(groups, num_blocks, kv_window, *, neutralise_window=False):
    """Admitted-of-batch over a 4-request ramp, read at the delegate boundary."""
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
            neutralise_window=neutralise_window,
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
    """Three non-hybrid fixtures, declared: idle-prefill, decode-only, mixed."""
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


def _digest_of(cls, groups, fixture, num_blocks, *, neutralise_window: bool) -> str:
    """Decision digest for one fixture, with the window active or neutralised.

    Same canonical record shape as ``_decision_digest``; the only difference is
    that the window can be switched off, which is what makes it a control.
    """
    _, seen, post = _drive(
        cls,
        kv_window=PARENT_UNIFORM_WINDOW,
        num_blocks=num_blocks,
        groups=groups,
        running=fixture["running"],
        holdback=fixture["holdback"],
        waiting=fixture["waiting"],
        neutralise_window=neutralise_window,
    )
    decision = {"delegate": seen, **post}
    return hashlib.sha256(json.dumps(decision, sort_keys=True).encode()).hexdigest()


def _decision_digest(cls, fixture, *, groups, num_blocks, kv_window):
    """Canonical, byte-comparable record of the schedule DECISION.

    Structure is byte-identical to the parent probe's, so the digests compare.
    """
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


# --- Conjunct 1 -------------------------------------------------------------
def test_c01_four_request_batch_schedules_with_zero_starved() -> None:
    """0 of 4 starved on the hybrid stack; the count MOVES to 3 without it."""
    specs = _specs()
    assert len(specs) == DECLARED_TOTAL_ENTRIES
    groups = _groups(specs)
    num_blocks = FIXTURE_BATCH * CLOSED_FORM_BLOCKS_PER_REQUEST

    admitted, trace = _ramp(groups, num_blocks, PARENT_UNIFORM_WINDOW)
    starved = FIXTURE_BATCH - admitted
    _record(
        c01_admitted_of_batch=admitted,
        c01_starved_count=starved,
        c01_num_blocks=num_blocks,
        c01_trace=trace,
    )
    assert starved == 0, trace
    assert admitted == FIXTURE_BATCH

    # The parent read 3 starved under this same predicate, so the pre-election
    # behaviour cannot satisfy this arm.
    assert starved != PARENT_STARVED_COUNT

    # NON-VACUITY CONTROL, and the counted zero MOVES: with the hybrid window
    # neutralised the same ramp starves 3 of 4 -- the parent's reading, so the
    # zero tracks this increment's own work and not the fixture's generosity.
    c_admitted, c_trace = _ramp(
        groups, num_blocks, PARENT_UNIFORM_WINDOW, neutralise_window=True
    )
    c_starved = FIXTURE_BATCH - c_admitted
    _record(c01_control_starved_count=c_starved, c01_control_trace=c_trace)
    assert c_starved == PARENT_STARVED_COUNT
    assert c_admitted == PARENT_ADMITTED_OF_BATCH


# --- Conjunct 2 -------------------------------------------------------------
def test_c02_computed_block_demand_equals_closed_form_exactly() -> None:
    """Block demand == 34*1 + 11*cdiv(1024,128) == 122, exact equality."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    specs = _specs()
    groups = _groups(specs)
    kda = sum(1 for s in specs.values() if isinstance(s, MambaSpec))
    dsa = sum(1 for s in specs.values() if isinstance(s, FullAttentionSpec))
    assert (kda, dsa) == (DECLARED_KDA_ENTRIES, DECLARED_DSA_ENTRIES)

    from vllm_neuron.vllm.core.scheduler import NeuronScheduler

    s, _, _ = _drive(
        NeuronScheduler,
        kv_window=PARENT_UNIFORM_WINDOW,
        num_blocks=FIXTURE_BATCH * CLOSED_FORM_BLOCKS_PER_REQUEST,
        groups=groups,
        holdback=[("n0", 64, 0)],
    )
    demand = s._hybrid_kv_blocks_per_request
    _record(
        c02_blocks_per_request=demand,
        c02_closed_form=CLOSED_FORM_BLOCKS_PER_REQUEST,
        c02_window=s._hybrid_kv_window,
        c02_algebra=f"{kda}*1 + {dsa}*cdiv({FIXTURE_MAX_MODEL_LEN},"
        f"{REGISTERED_HYBRID_BLOCK_SIZE})",
    )
    assert demand == CLOSED_FORM_BLOCKS_PER_REQUEST
    assert s._hybrid_kv_window == FIXTURE_BATCH

    # The parent computed NO demand, and vLLM's own byte-priced arithmetic
    # implies 375 blocks per request -- so this equality is not satisfiable by
    # the pre-existing number.
    assert demand != PARENT_VLLM_IMPLIED_BLOCKS_PER_REQUEST

    # NON-VACUITY CONTROL (a) -- THE ARITHMETIC MOVES WHILE THE GUARD STILL
    # FIRES. Retype 33 of the 34 recurrent groups as attention, keeping ONE, so
    # the opt-in still engages and only the recurrent/paged SPLIT changes. The
    # demand must move 122 -> 1*1 + 44*8 == 353, which shows the count tracks
    # the split rather than the group count (both stacks carry 45 groups).
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec

    attention_spec = _groups(specs, only_attention=True)[0].kv_cache_spec
    one_kda = _groups(specs, only_attention=True) + [
        KVCacheGroupSpec([f"retyped.{i}"], attention_spec)
        for i in range(DECLARED_KDA_ENTRIES - 1)
    ]
    one_kda.append(
        KVCacheGroupSpec(
            ["kept.kda"],
            next(s for s in specs.values() if isinstance(s, MambaSpec)),
        )
    )
    s2, _, _ = _drive(
        NeuronScheduler,
        kv_window=PARENT_UNIFORM_WINDOW,
        num_blocks=FIXTURE_BATCH * CLOSED_FORM_BLOCKS_PER_REQUEST,
        groups=one_kda,
        holdback=[("n0", 64, 0)],
    )
    per_page = -(-FIXTURE_MAX_MODEL_LEN // REGISTERED_HYBRID_BLOCK_SIZE)
    expected_split_moved = 1 * 1 + (DECLARED_TOTAL_ENTRIES - 1) * per_page
    _record(
        c02_control_a_group_count=len(one_kda),
        c02_control_a_blocks_per_request=s2._hybrid_kv_blocks_per_request,
        c02_control_a_expected=expected_split_moved,
    )
    assert len(one_kda) == DECLARED_TOTAL_ENTRIES
    assert s2._hybrid_kv_blocks_per_request == expected_split_moved
    assert s2._hybrid_kv_blocks_per_request != CLOSED_FORM_BLOCKS_PER_REQUEST

    # NON-VACUITY CONTROL (b) -- THE GUARD ITSELF MOVES. With every recurrent
    # group retyped away, the opt-in does not fire at all: no window, no demand.
    all_attention = _groups(specs, only_attention=True) + [
        KVCacheGroupSpec([f"retyped.{i}"], attention_spec)
        for i in range(DECLARED_KDA_ENTRIES)
    ]
    s3, _, _ = _drive(
        NeuronScheduler,
        kv_window=PARENT_UNIFORM_WINDOW,
        num_blocks=FIXTURE_BATCH * CLOSED_FORM_BLOCKS_PER_REQUEST,
        groups=all_attention,
        holdback=[("n0", 64, 0)],
    )
    _record(
        c02_control_b_window=s3._hybrid_kv_window,
        c02_control_b_blocks_per_request=s3._hybrid_kv_blocks_per_request,
    )
    assert s3._hybrid_kv_window == 0
    assert s3._hybrid_kv_blocks_per_request == 0


# --- Conjunct 3 -------------------------------------------------------------
def test_c03_legacy_non_hybrid_decisions_are_byte_identical_both_classes() -> None:
    """3/3 legacy fixtures byte-identical, per live scheduler class."""
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec, MambaSpec

    from vllm_neuron.vllm.core.scheduler import (
        NeuronAsyncScheduler,
        NeuronScheduler,
    )

    specs = _specs()
    legacy_groups = _groups(specs, only_attention=True)
    assert legacy_groups, "legacy fixture must have attention groups"
    assert not any(
        isinstance(g.kv_cache_spec, MambaSpec) for g in legacy_groups
    ), "a legacy fixture with a recurrent group is not legacy"

    matched = 0
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
            key = (cls.__name__, name)
            observed[f"{cls.__name__}/{name}"] = digest
            delegates[f"{cls.__name__}/{name}"] = decision["delegate"]["delegate_fired"]
            matched += int(digest == PARENT_LEGACY_DIGESTS[key])
    _record(
        c03_matched=matched,
        c03_of=len(PARENT_LEGACY_DIGESTS),
        c03_observed=observed,
        c03_delegates=delegates,
    )
    # 3/3 per class, measured per class -- 6 digests, all byte-identical.
    assert matched == len(PARENT_LEGACY_DIGESTS), observed

    # Each class routed through ITS OWN live delegate body, so neither arm is
    # measuring the other's.
    assert delegates["NeuronScheduler/L2_decode_only"] == "Scheduler.schedule"
    assert (
        delegates["NeuronAsyncScheduler/L2_decode_only"] == "AsyncScheduler.schedule"
    )

    # NON-VACUITY CONTROL, and the byte-identity MOVES -- measured WHERE THE
    # WINDOW CAN ACT. The control is deliberately NOT "inject a recurrent group
    # into all three legacy fixtures and demand 6 digests change": on
    # `L2_decode_only` nothing is waiting, so no admission is at stake and the
    # decision is legitimately identical either way. Demanding movement there
    # would be demanding the increment change a decision it must not change.
    #
    # Instead: hold the fixture at the shape where admission IS at stake (one
    # decode running, one candidate held back, uniform window 1) on a HYBRID
    # config, and digest it twice -- window active vs window neutralised. The
    # two must differ, which proves the digest function discriminates the
    # hybrid path, and therefore that the 6 matches above are readings.
    recurrent = next(s for s in specs.values() if isinstance(s, MambaSpec))
    hybrid_groups = _groups(specs)
    contested = {"running": [("dec0", 64, 64)], "holdback": [("new1", 64, 0)], "waiting": []}
    num_blocks = FIXTURE_BATCH * CLOSED_FORM_BLOCKS_PER_REQUEST

    moved = 0
    control_observed: dict = {}
    for cls in (NeuronScheduler, NeuronAsyncScheduler):
        d_active = _digest_of(
            cls, hybrid_groups, contested, num_blocks, neutralise_window=False
        )
        d_off = _digest_of(
            cls, hybrid_groups, contested, num_blocks, neutralise_window=True
        )
        control_observed[f"{cls.__name__}/active"] = d_active
        control_observed[f"{cls.__name__}/neutralised"] = d_off
        moved += int(d_active != d_off)
    _record(c03_control_moved=moved, c03_control_observed=control_observed)
    assert moved == 2, control_observed

    # And the injected-recurrent-group reading, RECORDED rather than asserted:
    # it is the same discrimination seen from the config side.
    hybridised = legacy_groups + [KVCacheGroupSpec(["injected.kda"], recurrent)]
    injected: dict = {}
    for name, fx in _legacy_fixtures():
        digest, _ = _decision_digest(
            NeuronScheduler,
            fx,
            groups=hybridised,
            num_blocks=LEGACY_NUM_BLOCKS,
            kv_window=LEGACY_KV_WINDOW,
        )
        injected[name] = (
            digest,
            digest != PARENT_LEGACY_DIGESTS[("NeuronScheduler", name)],
        )
    _record(c03_injected_recurrent_group=injected)
