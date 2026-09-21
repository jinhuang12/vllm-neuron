# SPDX-License-Identifier: Apache-2.0
"""The KV-spec patch: widening upstream's page-size unification.

``vllm_neuron/vllm/patches/kv_spec_patch.py`` wraps
``vllm.v1.core.kv_cache_utils.unify_kv_cache_spec_page_size`` so a hybrid cache
set whose recurrent-state page does not divide the attention page is padded up to
the maximum instead of refused. Covered here: importing the plugin wires the
patch, applying it twice leaves exactly one wrapper layer, the wrapped original
stays reachable and returns upstream's own result on input upstream already
unifies, the widening pads a recurrent-state page, a non-recurrent refusal still
raises, upstream really reaches the wrapped call site, and the plugin still loads
when ``vllm`` is imported first.

The spec objects are built here with arithmetic-chosen sizes. They are shaped
like a hybrid set -- a recurrent-state ``MambaSpec`` beside an attention spec,
the recurrent page smaller than and not a divisor of the attention page -- but
they assert nothing about any real model's geometry. All they do is select the
upstream branches this patch touches.

One upstream defect is out of scope here. Upstream's re-block remedy at
``kv_cache_utils.py:1084`` multiplies a spec's ``block_size`` and then asserts at
``:1098`` that the page equals the maximum; ``MambaSpec.page_size_bytes`` has no
``block_size`` term, so a recurrent page that DIVIDES the maximum makes upstream
fail its own post-condition with a bare ``AssertionError``. This patch is scoped
to pages the maximum does not divide, so nothing below builds that case.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest
import torch

TARGET_MODULE = "vllm.v1.core.kv_cache_utils"
TARGET_ATTR = "unify_kv_cache_spec_page_size"


class _StubSchedulerConfig:
    # Read once by upstream, at kv_cache_utils.py:1710. True would route through
    # unify_hybrid_kv_cache_specs, which this patch does not touch.
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
    """Page = 2 * 128 * 1 * 128 * 2 = 65536 B, a divisor of the 262144 B page."""
    # An attention spec, not a MambaSpec: a recurrent page that divides the
    # maximum trips the upstream post-condition described in the module
    # docstring, so it would measure that defect instead of this patch.
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
    """Return ``(depth, innermost)`` over ``__wrapped__``, cycle-safe."""
    depth = 0
    seen = {id(fn)}
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
        assert id(fn) not in seen, "__wrapped__ chain is cyclic, so depth is unreadable"
        seen.add(id(fn))
        depth += 1
    return depth, fn


def test_kv_spec_patch_is_live_after_importing_the_plugin():
    """Importing ``vllm_neuron`` binds the wrapper onto the vendor module."""
    # Every other test here reads the patched binding, so a silent failure of the
    # import-time wiring would make them measure upstream instead of the patch.
    import vllm_neuron  # noqa: F401  -- the import IS the mechanism under test

    from vllm_neuron.vllm.patches import kv_spec_patch

    assert kv_spec_patch._applied is True, (
        "importing vllm_neuron did not wire the KV-spec patch"
    )
    assert kv_spec_patch._bound is True, (
        "the patch is wired but not bound: in THIS import order the eager path "
        "should have rebound the attribute without needing the deferral"
    )
    installed = getattr(_kv_cache_utils(), TARGET_ATTR)
    assert installed is kv_spec_patch._unify_kv_cache_spec_page_size_widened, (
        f"{TARGET_MODULE}.{TARGET_ATTR} is not this patch's wrapper; "
        f"got {installed!r}"
    )
    assert TARGET_MODULE in sys.modules, (
        "the patch claims to have rebound an attribute on a module that is not "
        "loaded, which is impossible"
    )


def test_applying_twice_leaves_exactly_one_wrapper_layer():
    """Re-applying the patch does not stack a second wrapper."""
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

    # Identity alone would also hold if the patch had installed nothing at all,
    # so the chain is measured too.
    depth, innermost = _wrapper_chain(after_second)
    assert depth == 1, f"expected exactly 1 wrapper layer, measured {depth}"
    assert innermost is not after_second, (
        "the installed attribute IS the innermost callable, so nothing was wrapped"
    )
    assert innermost.__module__ == TARGET_MODULE, (
        f"the innermost callable is not upstream's: {innermost.__module__}"
    )
    assert innermost.__name__ == TARGET_ATTR, (
        f"the innermost callable is not upstream's: {innermost.__name__}"
    )


def test_original_is_reachable_and_unwidened_input_is_unchanged():
    """On input upstream can unify unaided, the wrapper returns upstream's result."""
    # The two attention pages divide, so upstream takes its re-blocking remedy and
    # never reaches the refusal: the wrapper must pad nothing, which is what makes
    # it a wrap rather than a replacement.
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

    assert small_page < big_page, "fixture is not a below-maximum case"
    assert big_page % small_page == 0, (
        f"fixture is not upstream-unifiable: {big_page} % {small_page} "
        f"== {big_page % small_page}; this would measure the widened path instead"
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
    # Which upstream remedy ran, not merely that one did: re-blocking multiplied
    # the logical block size.
    assert from_wrapper["attn.small"].block_size == small.block_size * (
        big_page // small_page
    ), "upstream's re-blocking remedy did not run on the unwidened input"


def test_widening_pads_a_recurrent_state_page_upstream_refuses():
    """Where upstream raises, the wrapper pads the recurrent page to the maximum."""
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
        "fixture is divisible, so upstream would re-block it and the refusal "
        "would never be reached"
    )

    from vllm.v1.kv_cache_interface import AttentionSpec

    assert not isinstance(recurrent, AttentionSpec), (
        "the recurrent-state spec is an AttentionSpec here, so upstream's own "
        "gate would cover it and this patch would be unnecessary"
    )

    def _fresh():
        return {
            "attn.0": _attention_spec(),
            "kda.0": _recurrent_spec(NON_DIVIDING_SHAPES),
        }

    # Upstream's behaviour, through the reachable original.
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


def test_non_recurrent_refusal_still_raises():
    """A sliding-window spec upstream refuses is still refused after the patch."""
    # The widening admits the recurrent-state class only: an
    # ``except NotImplementedError`` that swallowed everything would be a far
    # larger change than this patch claims.
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
        "the spec is not an AttentionSpec, so it does not exercise upstream's gate"
    )
    assert narrow.indexes_kv_by_block_stride is False, (
        "the spec opted into stride indexing, so upstream would pad it and no "
        "refusal would be exercised"
    )
    attention_page = _attention_spec().page_size_bytes
    assert narrow.page_size_bytes < attention_page
    assert attention_page % narrow.page_size_bytes != 0, (
        "the spec's page divides the maximum, so upstream would re-block it and "
        "no refusal would be exercised"
    )

    with pytest.raises(NotImplementedError):
        wrapper({"attn.0": _attention_spec(), "swa.0": narrow})


def test_upstream_reaches_the_patched_call_site():
    """``get_kv_cache_groups`` calls the wrapped function for a hybrid set only."""
    # The only way to see the call is a counter in its own position, so this
    # installs one over the vendor module attribute and restores the previous
    # binding in a finally. A uniform set is the comparison: upstream answers it
    # at kv_cache_utils.py:1718, before the call.
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
        "the counter was not removed from the vendor module; later tests would "
        "read the counter instead of the wrapper"
    )
    assert hybrid_calls == 1, (
        f"upstream's get_kv_cache_groups reached {TARGET_MODULE}.{TARGET_ATTR} "
        f"{hybrid_calls} times for a hybrid recurrent-state spec set, expected 1"
    )
    assert uniform_calls == 0, (
        "a uniform spec set reached the call site, which upstream answers at "
        f":1718 before it; the counter is measuring the wrong thing "
        f"({uniform_calls} calls)"
    )


def test_plugin_loads_when_vllm_is_imported_first():
    """In production order -- ``import vllm`` first -- the wrapper is wired and bound."""
    # In production vLLM is imported first and loads this plugin from inside its
    # own initialisation, an order no other test here uses. An eager
    # ``from vllm.v1.core import kv_cache_utils`` inside apply_kv_spec_patch()
    # then walks back into a partially initialised vllm.utils.torch_utils and
    # raises, and vLLM reports "Failed to load plugin neuron" -- the Neuron
    # platform does not register at all. With the eager import replaced by an
    # audit-hook fallback, the plugin loads but can end up wired and never bound,
    # which leaves the widening inert and engine start still raising. Import order
    # is process-global, so both readings need a child process.
    child = textwrap.dedent(
        """
        import sys
        import vllm  # noqa: F401  -- production order: vllm first
        from vllm.v1.core import kv_cache_utils

        attr = getattr(kv_cache_utils, "unify_kv_cache_spec_page_size")
        print("PLUGIN_LOADED=%s" % ("vllm_neuron" in sys.modules))
        print("ATTR_MODULE=%s" % attr.__module__)
        print("ATTR_IS_WRAPPER=%s" % attr.__module__.startswith("vllm_neuron"))
        inner = getattr(attr, "__wrapped__", None)
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

    assert completed.returncode == 0, (
        "importing vllm before vllm_neuron failed outright, so the plugin is "
        f"unloadable in the production order.\nrc={completed.returncode}\n"
        f"stdout:\n{completed.stdout[-3000:]}\nstderr:\n{completed.stderr[-3000:]}"
    )
    assert "Failed to load plugin" not in completed.stderr + completed.stdout, (
        "vLLM reported 'Failed to load plugin' in the production import order, so "
        f"the Neuron platform did not register.\nstderr:\n{completed.stderr[-3000:]}"
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
        f"widening is inert there and engine start would still raise: {readings}"
    )
