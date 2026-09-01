# SPDX-License-Identifier: Apache-2.0
"""``inc-glm53f-018`` acceptance -- WP2: the campaign's ONE new vLLM patch.

THE DECLARED ACCEPTANCE (increment plan revision 32, L753), verbatim:

    "Tier T -- ``pytest test/vllm_neuron/patches/test_kv_spec_patch.py``.
    **Expected:** applying twice leaves **exactly one** wrapper layer
    (idempotence asserted by identity comparison of the patched attribute across
    two applications); the original callable is still reachable and returns its
    pin behaviour on **1/1** unwidened input (wrap-not-replace); and ``coverage``
    config resolves ``vllm_neuron/vllm/patches/*`` to **exactly 4** existing
    files -- stated additionally as a **delta 0 -> 4** so the predicate is
    invariant to the pin's own file count."

Three declared conjuncts, measured below as C01 / C02 / C03. **They are the
gate.** The plan declares no exit code for this block, so exit 0 is the
expectation D1.1 supplies centrally, and it declares no collected-item count, so
the count here is RECORDED from a dedicated ``--collect-only -q`` run and is not
claimed in this docstring (D1.2).

Items prefixed ``E`` are RECORDED EVIDENCE, not criteria. They author no
threshold, move no declared value and add no conjunct; they exist because C01
through C03 are all satisfiable by a patch that widens nothing -- a wrapper that
simply forwards would leave one layer, keep the original reachable and change no
file count. E01 and E02 close that false-pass door by measuring the widening
itself and the refusal it deliberately preserves; E03 keeps this increment's
reachability reading live and repeatable instead of leaving it in a transcript;
and **E04 exists because attempt 2 of this acceptance passed 7/7 while the patch
made vLLM unable to load the plugin at all** -- every item except E04 imports
``vllm_neuron`` first, an order production never uses. E04 is the only item that
runs the production order, and it is the reason this file is worth more than its
three declared conjuncts.

WHY THE FIXTURE PAGES ARE FIXTURE PAGES
---------------------------------------
The spec objects below are constructed locally with arithmetic-chosen sizes.
They are shaped like the campaign's hybrid set -- a recurrent-state
``MambaSpec`` beside an attention spec, the recurrent page smaller than and not
a divisor of the attention page -- but they assert **nothing** about the real
model's geometry, which is ``-013``/``-016``'s subject and not this file's. What
they exercise is upstream's branch structure, which is what this patch touches.

ONE UPSTREAM FINDING, RECORDED HERE AND ROUTED, NOT FIXED HERE
--------------------------------------------------------------
Upstream's re-block remedy at ``:1084`` multiplies a spec's ``block_size`` and
then asserts at ``:1098`` that the page now equals the maximum.
``MambaSpec.page_size_bytes`` has no ``block_size`` term, so a recurrent page
that DIVIDES the maximum makes upstream fail its own post-condition with a bare
``AssertionError``. This patch's declared scope is pages the maximum does NOT
divide, so that case is out of scope here and is routed to the lead with
``increments/probe-018-divisible-mamba.py`` as its transcript. The same probe
records that it is unreachable at this campaign's registered geometry: the KDA
state page carries odd prime factors, and an integer with an odd factor cannot
divide a power-of-two attention page.

WHY THE VENDOR MODULE IS MUTATED IN E03
---------------------------------------
E03 measures whether upstream's ``get_kv_cache_groups`` actually calls the
target at ``kv_cache_utils.py:1751``. The only honest instrument is a counter in
the call's own position, so the item installs one over the module attribute and
restores the previous binding in a ``finally``. It asserts the restoration.
"""

from __future__ import annotations

import subprocess  # noqa: F401  -- imported in the item that needs it
import sys
import textwrap
import tomllib
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]

TARGET_MODULE = "vllm.v1.core.kv_cache_utils"
TARGET_ATTR = "unify_kv_cache_spec_page_size"

# Coverage's omit entry for the patch package, and the stale value this
# increment replaces. Both are literal here on purpose: C03's delta compares the
# two, so the old one has to survive somewhere the test can read it.
OMIT_PATTERN_NEW = "vllm_neuron/vllm/patches/*"
OMIT_PATTERN_STALE = "vllm_neuron/patches/*"
EXPECTED_PATCH_FILES = 4


class KvSpecProbeError(RuntimeError):
    """Two instruments in this file disagreed about the same world reading.

    Raised instead of asserting so the failure names the instrument that
    disagreed rather than surfacing as a bare comparison.
    """


class _StubSchedulerConfig:
    # Read once by upstream, at kv_cache_utils.py:1710. False is the path this
    # campaign runs on; True would route through unify_hybrid_kv_cache_specs,
    # which is a different increment's subject.
    disable_hybrid_kv_cache_manager = False


class _StubVllmConfig:
    scheduler_config = _StubSchedulerConfig()


def _kv_cache_utils():
    """The live vendor module. Imported inside bodies, never at module scope."""
    from vllm.v1.core import kv_cache_utils

    return kv_cache_utils


def _specs():
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        MambaSpec,
        SlidingWindowSpec,
    )

    return FullAttentionSpec, MambaSpec, SlidingWindowSpec


def _attention_spec(block_size: int = 128):
    """Page = 2 * block_size * num_kv_heads * head_size * itemsize = 262144 B."""
    FullAttentionSpec, _, _ = _specs()
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.bfloat16,
        sliding_window=None,
    )


def _recurrent_spec(shapes, block_size: int = 128):
    _, MambaSpec, _ = _specs()
    return MambaSpec(
        block_size=block_size,
        shapes=shapes,
        dtypes=(torch.float32, torch.float32),
    )


def _attention_spec_small(block_size: int = 128):
    """Page = 2 * 128 * 1 * 128 * 2 = 65536 B, a divisor of the 262144 B page.

    C02's "unwidened input" is built from two ATTENTION specs, deliberately.
    A ``MambaSpec`` cannot serve there, and the reason is a recorded upstream
    finding rather than a preference: upstream's re-block remedy at ``:1084``
    multiplies ``block_size``, and ``MambaSpec.page_size_bytes`` has no
    ``block_size`` term, so a recurrent page that DIVIDES the maximum makes
    upstream fail its own post-condition at ``:1098`` with a bare
    ``AssertionError``. That case is out of this patch's declared scope (the
    design scopes it to pages the maximum does not divide) and is routed to the
    lead with ``increments/probe-018-divisible-mamba.py`` as its transcript.
    Using it here would have measured that upstream defect instead of this
    increment's wrap-not-replace property.
    """
    FullAttentionSpec, _, _ = _specs()
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        sliding_window=None,
    )


# Chosen so 262144 % 196608 == 65536: upstream can neither re-block it nor pad
# it, which is exactly the refusal this patch narrows.
NON_DIVIDING_SHAPES = ((16384,), (32768,))  # 65536 + 131072 = 196608 B


def _wrapper_chain(fn):
    """(depth, innermost) over ``__wrapped__``, cycle-safe."""
    depth = 0
    seen = {id(fn)}
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
        if id(fn) in seen:
            raise KvSpecProbeError(
                "__wrapped__ chain is cyclic; the depth instrument cannot read it"
            )
        seen.add(id(fn))
        depth += 1
    return depth, fn


def test_kv_spec_patch_is_live_after_importing_the_plugin():
    """Precondition, measured rather than assumed: the import-time wiring ran.

    Every other item in this file reads a patched binding, so a silent failure
    of the wiring would make them measure the pin instead of this increment.
    """
    import vllm_neuron  # noqa: F401  -- the import IS the mechanism under test

    from vllm_neuron.vllm.patches import kv_spec_patch

    assert kv_spec_patch._applied is True, (
        "importing vllm_neuron did not wire the KV-spec patch; every reading "
        "below would be about the pin, not about this increment"
    )
    assert kv_spec_patch._bound is True, (
        "the patch is wired but not bound: in THIS import order the eager path "
        "should have rebound the attribute without needing the audit hook"
    )
    installed = getattr(_kv_cache_utils(), TARGET_ATTR)
    assert installed is kv_spec_patch._unify_kv_cache_spec_page_size_widened, (
        f"{TARGET_MODULE}.{TARGET_ATTR} is not this patch's wrapper; "
        f"got {installed!r}"
    )
    assert TARGET_MODULE in sys.modules, (
        "the patch claims to have rebound an attribute on a module that is not "
        "loaded, which is impossible -- the instrument is wrong"
    )


def test_c01_applying_twice_leaves_exactly_one_wrapper_layer():
    """C01 -- exactly one wrapper layer, by identity across two applications.

    The vacuity door this closes: ``before is after`` is trivially true when the
    patch never installed anything at all. So the item also measures that the
    installed attribute is NOT the original, and that the chain depth is 1
    rather than 0 or 2 -- with a control proving the depth counter can read 2.
    """
    import vllm_neuron  # noqa: F401

    from vllm_neuron.vllm.patches.kv_spec_patch import apply_kv_spec_patch

    kv_cache_utils = _kv_cache_utils()

    before = getattr(kv_cache_utils, TARGET_ATTR)
    apply_kv_spec_patch()
    after_first = getattr(kv_cache_utils, TARGET_ATTR)
    apply_kv_spec_patch()
    after_second = getattr(kv_cache_utils, TARGET_ATTR)

    assert before is after_first is after_second, (
        "the patched attribute changed identity across applications, so a "
        f"wrapper was stacked: {before!r} / {after_first!r} / {after_second!r}"
    )

    depth, innermost = _wrapper_chain(after_second)
    assert depth == 1, f"expected exactly 1 wrapper layer, measured {depth}"
    assert innermost is not after_second, (
        "the installed attribute IS the innermost callable, so nothing was "
        "wrapped and the identity comparison above is vacuous"
    )
    assert innermost.__module__ == TARGET_MODULE, (
        f"the innermost callable is not upstream's: {innermost.__module__}"
    )
    assert innermost.__name__ == TARGET_ATTR, (
        f"the innermost callable is not upstream's: {innermost.__name__}"
    )

    # Non-vacuity control on the depth instrument itself: a deliberately
    # double-wrapped callable must read 2. Built locally; the vendor module is
    # not touched.
    def _second_layer(spec):  # pragma: no cover -- never called
        raise KvSpecProbeError("control callable must never be invoked")

    _second_layer.__wrapped__ = after_second
    control_depth, control_innermost = _wrapper_chain(_second_layer)
    assert control_depth == 2, (
        "the depth instrument cannot distinguish one wrapper from two, so its "
        f"reading of 1 above is not a measurement (control read {control_depth})"
    )
    assert control_innermost is innermost, (
        "the control walked to a different innermost callable than the real "
        "chain, so the two readings are not comparable"
    )


def test_c02_original_is_reachable_and_pin_behaviour_survives_unwidened_input():
    """C02 -- original reachable; 1/1 unwidened input keeps pin behaviour.

    "Unwidened input" is a hybrid set upstream already unifies unaided: the
    smaller attention page (65536 B) divides the larger one (262144 B), so
    upstream takes its re-blocking remedy at ``:1084`` and never reaches the
    refusal. The wrapper must therefore return what the original returns, and
    must pad nothing -- which is what makes this a wrap and not a replacement.

    See ``_attention_spec_small`` for why both specs here are attention specs.
    """
    import vllm_neuron  # noqa: F401

    from vllm_neuron.vllm.patches import kv_spec_patch

    kv_cache_utils = _kv_cache_utils()
    wrapper = getattr(kv_cache_utils, TARGET_ATTR)
    original = kv_spec_patch._original_unify

    assert callable(original), (
        f"the original callable is not reachable through the patch: {original!r}"
    )
    _, innermost = _wrapper_chain(wrapper)
    assert innermost is original, (
        "the saved original and the __wrapped__ chain's innermost callable "
        "disagree, so 'reachable' has two different answers"
    )

    big = _attention_spec()
    small = _attention_spec_small()
    big_page = big.page_size_bytes
    small_page = small.page_size_bytes

    # The arm is only about "unwidened input" if upstream really can unify it.
    assert small_page < big_page, "fixture is not a below-maximum case"
    assert big_page % small_page == 0, (
        f"fixture is not upstream-unifiable: {big_page} % {small_page} "
        f"== {big_page % small_page}; C02 would be measuring the widened path "
        "instead of the pin path"
    )

    def _fresh():
        return {"attn.big": _attention_spec(), "attn.small": _attention_spec_small()}

    from_original = original(_fresh())
    from_wrapper = wrapper(_fresh())

    assert from_original == from_wrapper, (
        "the wrapper changed upstream's result on an input upstream can already "
        f"unify: original={from_original!r} wrapper={from_wrapper!r}"
    )
    unified_pages = {spec.page_size_bytes for spec in from_wrapper.values()}
    assert unified_pages == {big_page}, (
        f"pages did not unify to the maximum: {unified_pages}"
    )
    padded = {
        name: spec.page_size_padded
        for name, spec in from_wrapper.items()
        if getattr(spec, "page_size_padded", None) is not None
    }
    assert padded == {}, (
        "the wrapper padded a page upstream re-blocks unaided, so it is "
        f"displacing upstream's preferred remedy: {padded}"
    )
    # Upstream's re-blocking multiplied the logical block size; recorded so the
    # arm shows WHICH pin remedy ran, not merely that one did.
    assert from_wrapper["attn.small"].block_size == small.block_size * (
        big_page // small_page
    ), "upstream's re-blocking remedy did not run on the unwidened input"


def test_c03_coverage_omit_resolves_to_exactly_four_patch_files():
    """C03 -- the omit entry resolves to exactly 4 files; delta 0 -> 4.

    The delta is over the SOURCE path. This test file's own package
    (``test/vllm_neuron/patches/``) is not in it, per the plan's D15 note.
    """
    pyproject = REPO_ROOT / "pyproject.toml"
    assert pyproject.is_file(), f"pyproject.toml not found at {pyproject}"
    config = tomllib.loads(pyproject.read_text())
    omit = config["tool"]["coverage"]["run"]["omit"]

    assert OMIT_PATTERN_NEW in omit, (
        f"coverage omit does not carry {OMIT_PATTERN_NEW!r}; got {omit!r}"
    )
    assert OMIT_PATTERN_STALE not in omit, (
        f"the stale omit path {OMIT_PATTERN_STALE!r} survives in {omit!r}"
    )

    def resolve(pattern: str) -> list[str]:
        return sorted(
            str(p.relative_to(REPO_ROOT))
            for p in REPO_ROOT.glob(pattern)
            if p.is_file()
        )

    resolved_new = resolve(OMIT_PATTERN_NEW)
    resolved_stale = resolve(OMIT_PATTERN_STALE)

    assert len(resolved_new) == EXPECTED_PATCH_FILES, (
        f"{OMIT_PATTERN_NEW} resolved to {len(resolved_new)} files, expected "
        f"{EXPECTED_PATCH_FILES}: {resolved_new}"
    )
    assert "vllm_neuron/vllm/patches/kv_spec_patch.py" in resolved_new, (
        f"this increment's module is not among the resolved files: {resolved_new}"
    )
    # The declared delta: the stale pattern matched nothing, which is why the
    # repair is a repair. This is the 0 half of "0 -> 4".
    assert resolved_stale == [], (
        f"the stale omit pattern matches real files, so the repair's premise is "
        f"wrong: {resolved_stale}"
    )

    # Non-vacuity control: the glob instrument must be able to return non-zero
    # for a pattern that does match, or the 0 above is a blind reading.
    control = resolve("vllm_neuron/vllm/*.py")
    assert len(control) > 0, (
        "the glob instrument returns 0 for a pattern with known matches, so its "
        "0 for the stale path is not a reading"
    )


def test_e01_evidence_widening_admits_the_recurrent_state_refusal():
    """E01 (evidence, not a criterion) -- the widening actually widens.

    Upstream raises ``NotImplementedError`` on this input; the wrapper returns a
    unified set by setting the field ``MambaSpec`` already carries. Both halves
    are measured, so neither the raise nor the repair is assumed.
    """
    import vllm_neuron  # noqa: F401

    kv_cache_utils = _kv_cache_utils()
    wrapper = getattr(kv_cache_utils, TARGET_ATTR)
    _, innermost = _wrapper_chain(wrapper)

    attention = _attention_spec()
    recurrent = _recurrent_spec(NON_DIVIDING_SHAPES)
    attention_page = attention.page_size_bytes
    recurrent_page = recurrent.page_size_bytes
    assert recurrent_page < attention_page
    assert attention_page % recurrent_page != 0, (
        "fixture is divisible, so upstream would re-block it and this item "
        "would not exercise the refusal at all"
    )

    from vllm.v1.kv_cache_interface import AttentionSpec

    assert not isinstance(recurrent, AttentionSpec), (
        "the recurrent-state spec is an AttentionSpec at this pin, so upstream's "
        "own gate would cover it and this patch would be unnecessary"
    )

    def _fresh():
        return {
            "attn.0": _attention_spec(),
            "kda.0": _recurrent_spec(NON_DIVIDING_SHAPES),
        }

    # The pin's behaviour, measured through the reachable original.
    with pytest.raises(NotImplementedError):
        innermost(_fresh())

    widened = wrapper(_fresh())
    assert {spec.page_size_bytes for spec in widened.values()} == {attention_page}, (
        "the widened set does not report one page size"
    )
    assert widened["kda.0"].page_size_padded == attention_page, (
        "the recurrent-state page was not padded to the maximum: "
        f"{widened['kda.0'].page_size_padded}"
    )
    assert widened["kda.0"].block_size == recurrent.block_size, (
        "the widening moved a LOGICAL block size; it must only pad the physical "
        "page"
    )
    assert widened["attn.0"] == attention, (
        "the attention spec was modified; it was already at the maximum page"
    )


def test_e02_evidence_the_refusal_is_narrowed_not_removed():
    """E02 (evidence, not a criterion) -- a non-recurrent refusal still raises.

    A sliding-window attention spec that has not opted into stride indexing is
    refused by upstream's own gate. This patch must leave that refusal standing:
    it admits the recurrent-state class only, and an ``except NotImplementedError``
    that swallowed everything would be a much larger change than the one
    declared.
    """
    import vllm_neuron  # noqa: F401

    _, _, SlidingWindowSpec = _specs()
    from vllm.v1.kv_cache_interface import AttentionSpec

    kv_cache_utils = _kv_cache_utils()
    wrapper = getattr(kv_cache_utils, TARGET_ATTR)

    narrow = SlidingWindowSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=384,
        dtype=torch.bfloat16,
        sliding_window=256,
    )
    assert isinstance(narrow, AttentionSpec), (
        "the control spec is not an AttentionSpec, so it does not exercise "
        "upstream's own gate"
    )
    assert narrow.indexes_kv_by_block_stride is False, (
        "the control spec opted into stride indexing, so upstream would pad it "
        "and no refusal would be exercised"
    )
    attention_page = _attention_spec().page_size_bytes
    assert narrow.page_size_bytes < attention_page
    assert attention_page % narrow.page_size_bytes != 0, (
        "the control spec's page divides the maximum, so upstream would "
        "re-block it and no refusal would be exercised"
    )

    with pytest.raises(NotImplementedError):
        wrapper({"attn.0": _attention_spec(), "swa.0": narrow})


def test_e03_evidence_upstream_reaches_the_patched_call_site():
    """E03 (evidence, not a criterion) -- the reachability reading, kept live.

    This increment's pre-declared first action asked whether a Neuron run
    reaches the target's sole call site, ``kv_cache_utils.py:1751`` inside
    ``get_kv_cache_groups``. The static half is in
    ``increments/probe-018-reachability.out``. This is the dynamic half: a
    counter in the call's own position, driven by upstream's real
    ``get_kv_cache_groups``.

    It moves: the hybrid set reaches the call, and a uniform set -- which
    upstream answers at ``:1718``, before the call -- does not. A counter that
    read 1 for both would be counting something else.
    """
    import vllm_neuron  # noqa: F401

    kv_cache_utils = _kv_cache_utils()
    installed = getattr(kv_cache_utils, TARGET_ATTR)
    calls: list[int] = []

    def _counting(kv_cache_spec):
        calls.append(len(kv_cache_spec))
        return installed(kv_cache_spec)

    _counting.__wrapped__ = installed

    hybrid = {
        "attn.0": _attention_spec(),
        "kda.0": _recurrent_spec(NON_DIVIDING_SHAPES),
    }
    uniform = {"attn.0": _attention_spec(), "attn.1": _attention_spec()}

    try:
        setattr(kv_cache_utils, TARGET_ATTR, _counting)

        calls.clear()
        try:
            kv_cache_utils.get_kv_cache_groups(_StubVllmConfig(), dict(hybrid))
        except Exception:
            # Grouping needs more of VllmConfig than this stub carries. The
            # reading is whether the CALL happened, which the counter has
            # already recorded by then.
            pass
        hybrid_calls = len(calls)

        calls.clear()
        try:
            kv_cache_utils.get_kv_cache_groups(_StubVllmConfig(), dict(uniform))
        except Exception:
            pass
        uniform_calls = len(calls)
    finally:
        setattr(kv_cache_utils, TARGET_ATTR, installed)

    assert getattr(kv_cache_utils, TARGET_ATTR) is installed, (
        "the counter was not removed from the vendor module; later items would "
        "read an instrumented binding"
    )

    if hybrid_calls < 1:
        raise KvSpecProbeError(
            "upstream's get_kv_cache_groups did not reach "
            f"{TARGET_MODULE}.{TARGET_ATTR} for a hybrid recurrent-state spec "
            "set. That is this increment's reachability reading coming back "
            "negative and it contradicts the design; it is not a test bug to "
            "be silenced."
        )
    assert hybrid_calls == 1, (
        f"expected exactly 1 call for the hybrid set, counted {hybrid_calls}"
    )
    assert uniform_calls == 0, (
        "a uniform spec set reached the call site, which upstream answers at "
        f":1718 before it; the counter is measuring the wrong thing "
        f"({uniform_calls} calls)"
    )


def test_e04_evidence_production_import_order_keeps_the_plugin_loadable():
    """E04 (evidence, not a criterion) -- ``import vllm`` FIRST must still work.

    THE DEFECT THIS ITEM EXISTS TO CATCH, and it caught it. Every other item in
    this file reaches the patch by importing ``vllm_neuron`` directly, so they
    all run in an import order production never uses. In production vLLM is
    imported first and loads this plugin from inside its own initialisation. An
    eager ``from vllm.v1.core import kv_cache_utils`` in ``apply_kv_spec_patch``
    then walks back into a partially initialised ``vllm.utils.torch_utils`` and
    raises, and vLLM reports ``Failed to load plugin neuron`` -- the Neuron
    platform does not register AT ALL. Attempt 2 of this increment's acceptance
    passed 7/7 while that was true, because no item used the production order.

    Import order is process-global, so this is measured in a CHILD PROCESS, on
    ``-014``'s two-subprocess precedent. Three readings, all from the child's own
    stdout: the plugin loaded, the attribute is the wrapper, and the wrapper's
    ``__wrapped__`` is upstream's function.
    """
    import subprocess

    child = textwrap.dedent(
        """
        import sys
        # PRODUCTION ORDER: vllm first. This triggers vLLM's plugin discovery,
        # which imports vllm_neuron from inside vllm's own initialisation.
        import vllm  # noqa: F401
        from vllm.v1.core import kv_cache_utils

        attr = getattr(kv_cache_utils, "unify_kv_cache_spec_page_size")
        print("PLUGIN_LOADED=%s" % ("vllm_neuron" in sys.modules))
        print("ATTR_MODULE=%s" % attr.__module__)
        print("ATTR_IS_WRAPPER=%s" % attr.__module__.startswith("vllm_neuron"))
        inner = getattr(attr, "__wrapped__", None)
        print("HAS_WRAPPED=%s" % (inner is not None))
        print(
            "INNER_IS_UPSTREAM=%s"
            % (inner is not None and inner.__module__ == "vllm.v1.core.kv_cache_utils")
        )
        import vllm_neuron.vllm.patches.kv_spec_patch as p
        print("APPLIED=%s" % p._applied)
        print("BOUND=%s" % p._bound)
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", child],
        capture_output=True,
        text=True,
        timeout=300,
    )
    readings = dict(
        line.split("=", 1)
        for line in completed.stdout.splitlines()
        if "=" in line and not line.startswith(("INFO", "ERROR", "WARNING"))
    )

    if completed.returncode != 0:
        raise KvSpecProbeError(
            "importing vllm before vllm_neuron failed outright, so the plugin "
            f"is unloadable in the production order.\nrc={completed.returncode}\n"
            f"stdout:\n{completed.stdout[-3000:]}\nstderr:\n{completed.stderr[-3000:]}"
        )
    if "Failed to load plugin" in completed.stderr + completed.stdout:
        raise KvSpecProbeError(
            "vLLM reported 'Failed to load plugin' in the production import "
            "order: the Neuron platform did not register, so nothing about this "
            f"campaign works on that path.\nstderr:\n{completed.stderr[-3000:]}"
        )

    assert readings.get("PLUGIN_LOADED") == "True", (
        f"vllm_neuron is not in the child's sys.modules: {readings}"
    )
    assert readings.get("ATTR_IS_WRAPPER") == "True", (
        "in the production import order the patched attribute is NOT this "
        f"plugin's wrapper: {readings}"
    )
    assert readings.get("INNER_IS_UPSTREAM") == "True", (
        f"the wrapper's __wrapped__ is not upstream's callable: {readings}"
    )
    assert readings.get("APPLIED") == "True", (
        f"the patch was not wired in the production order: {readings}"
    )
    assert readings.get("BOUND") == "True", (
        "the patch was wired but never bound in the production order, so the "
        f"deferred audit hook did not fire: {readings}"
    )
