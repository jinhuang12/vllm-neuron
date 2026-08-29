"""``inc-glm53f-005`` items 0 and 2-7, and extraction-fidelity guard B.

**Why this file parses a fixture instead of asserting literals.** The registered
values live in the campaign's frozen acceptance registration, which is outside
this repository. Hand-copying them into a test and asserting the code against
the same hand-copied numbers proves only that one transcription matches itself:
a single slip passes every equality. So the registration's machine-readable
block is vendored **byte-exact** to ``fixtures/m4_registered_values.json``, its
sha256 is pinned in this file, and the tests compare the code registry against
the **parsed** fixture. A transcription slip now changes the digest and item 0
fails first.

The fixture is a *copy* under the campaign's comparator freeze, never a second
source of truth. If it ever disagrees with the registration, that is a
contradiction to route to the lead -- never something to reconcile here.

**The two tuple orders are inverted on purpose.** The accuracy ``tol_map`` is
``(atol, rtol)``; ``testing._DEFAULT_DTYPE_TOLERANCE`` is ``(rtol, atol)``.
Item 7 exists to fail if a later hand "normalises" one to the other, which
would set the fp8 rtol to 1e-5 and its atol to 3e-2 -- a
three-orders-of-magnitude loosening that still reads plausibly in review.

**Item 6 reads N from the instrument.** How many fp8 dtypes exist is a property
of the installed ``torch``, not a constant this file may declare. The test
enumerates them by a classified screen and asserts **key-set equality** against
the registry, so the predicate keeps adjudicating when ``torch`` is
re-resolved and N moves.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

from vllm_neuron.accuracy.constants import (
    ARCH_DIVERGENCE_CONFIG,
    ARCH_TOLERANCE_MAP,
    DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE,
    DEFAULT_TOLERANCE_MAP,
    GLM5NEXT_ARCH,
)
from vllm_neuron.accuracy.logit_validation import (
    ARCH_AGGREGATE_CONFIG,
    DEFAULT_AGGREGATE_CONFIG,
)
from vllm_neuron.accuracy.testing import (
    _DEFAULT_DTYPE_TOLERANCE,
    _FALLTHROUGH_DTYPE_TOLERANCE,
    FP8_DTYPE_TOLERANCE,
    resolve_dtype_tolerance,
)

#: sha256 of ``fixtures/m4_registered_values.json``: the registration's JSON
#: **object only**, fences excluded. The fenced range digests to
#: ``6f09a1e1908ce05acfacca0769a107ff5f57d1f3a2d506dfa3f3b79a4d26a526`` and is
#: the recorded rejected alternative -- it is not JSON, so it cannot be parsed,
#: and the whole point of the fixture is that it is parsed rather than
#: transcribed.
FIXTURE_SHA256 = "133e6057672626a34309b925e68f383186a3796ba958e2a857ec959a7f60cb39"
FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "m4_registered_values.json"

#: The pin's three pre-existing dtype entries, in ``_DEFAULT_DTYPE_TOLERANCE``'s
#: own ``(rtol, atol)`` order. Item 7 guards this order, so the order is stated
#: here explicitly rather than read back out of the map under test.
PIN_DTYPE_TOLERANCE = {
    torch.float16: (1e-3, 1e-5),
    torch.bfloat16: (1.6e-2, 1e-5),
    torch.float32: (1.3e-6, 1e-5),
}

#: The pin's shared defaults, which item 5 asserts have NOT moved. These are the
#: other architectures' regression guard: the registered entries are arch-scoped
#: additions, so every one of these must read exactly as it did at the pin.
PIN_DEFAULT_TOLERANCE_MAP = {
    "5": (1e-5, 0.011),
    "50": (1e-5, 0.02),
    "1000": (1e-5, 0.03),
    "all": (1e-5, 0.05),
}
PIN_DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE = 0.001
PIN_DEFAULT_PP_STATIC_THRESHOLDS = [0.03, 0.05]

#: Dtypes guard B uses for its "absent from the map" leg. Both are real dtypes
#: that no entry registers. ``int8`` is deliberately 1 byte wide: it proves the
#: fp8 screen classifies on dtype identity, not on width alone.
UNREGISTERED_DTYPES = (torch.int8, torch.float64)

_SINK = Path(
    os.environ.get("VLLM_NEURON_INC005_REGISTRY_JSON")
    or Path(tempfile.gettempdir()) / "vllm_neuron_inc005_registry_readings.json"
)
_RECORD: dict[str, Any] = {}
_SINK.write_text("{}\n")  # truncate a stale run's values


def _rec(**values: Any) -> None:
    """Persist as we go, so a failing conjunct still leaves its readings behind."""
    _RECORD.update(values)
    _SINK.write_text(json.dumps(_RECORD, indent=2, sort_keys=True, default=str) + "\n")


_FIXTURE_BYTES = FIXTURE_PATH.read_bytes()
#: The registered values, PARSED from the vendored copy -- never transcribed.
REGISTERED = json.loads(_FIXTURE_BYTES)


def _exposed_fp8_dtypes_by_declared_screen() -> list[torch.dtype]:
    """Enumerate this interpreter's fp8 dtypes by the classified screen.

    Deliberately a second, independent implementation of the screen the plugin
    uses to build its registration, and deliberately over ``dir(torch)`` +
    ``getattr`` rather than the module ``__dict__``. If the two ever disagree,
    the key-set equality below fails and that disagreement is the finding.

    Classified, never name-matched: the attribute must **be** a ``torch.dtype``
    instance, its ``itemsize`` must be 1, and its ``str()`` must name
    ``float8``. De-duplicated by object identity so two attribute names bound
    to one dtype are counted once.
    """
    found: list[torch.dtype] = []
    for name in dir(torch):
        candidate = getattr(torch, name, None)
        if not isinstance(candidate, torch.dtype):
            continue
        if candidate.itemsize != 1 or "float8" not in str(candidate):
            continue
        if any(candidate is seen for seen in found):
            continue
        found.append(candidate)
    return found


def _is_fp8(dtype: torch.dtype) -> bool:
    return dtype.itemsize == 1 and "float8" in str(dtype)


def test_item0_fixture_is_the_byte_exact_registered_block() -> None:
    """Item 0 -- 1/1 fixture-hash match, and the fixture parses."""
    digest = hashlib.sha256(_FIXTURE_BYTES).hexdigest()
    _rec(
        interpreter=sys.executable,
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        torch_file=getattr(torch, "__file__", None),
        item0_fixture_path=str(FIXTURE_PATH),
        item0_fixture_bytes=len(_FIXTURE_BYTES),
        item0_measured_sha256=digest,
        item0_declared_sha256=FIXTURE_SHA256,
        item0_top_level_key_count=len(REGISTERED),
        item0_campaign=REGISTERED.get("campaign"),
        item0_design_entry_id=REGISTERED.get("design_entry_id"),
    )

    assert digest == FIXTURE_SHA256, (
        f"item 0: {FIXTURE_PATH} digests to {digest}, not the declared "
        f"{FIXTURE_SHA256}. The vendored copy is not byte-exact -- route this to "
        f"the lead as a contradiction; do not re-vendor to make it match."
    )
    assert isinstance(REGISTERED, dict) and REGISTERED, (
        "item 0: the fixture did not parse to a non-empty JSON object"
    )


def test_item2_tol_map_eight_equalities_and_zero_extra_keys() -> None:
    """Item 2 -- 8/8 exact equalities, plus counted zero (a): 0 extra keys."""
    registered_map = REGISTERED["tol_map"]
    code_map = ARCH_TOLERANCE_MAP[GLM5NEXT_ARCH]

    assert REGISTERED["tol_map_order"] == "(atol, rtol)", (
        "item 2: the registration no longer declares (atol, rtol) order; the "
        "element-by-element comparison below assumes it"
    )

    extra_keys = sorted(set(code_map) - set(registered_map))
    missing_keys = sorted(set(registered_map) - set(code_map))

    equalities = 0
    failures: list[str] = []
    for key in sorted(registered_map):
        want = registered_map[key]
        got = code_map[key]
        if len(got) != len(want):
            failures.append(f"{key}: arity {len(got)} != {len(want)}")
            continue
        for index, (got_value, want_value) in enumerate(zip(got, want)):
            # By value, element by element -- never by truthiness, and never by
            # comparing the containers, so a (atol, rtol) flip cannot hide.
            if got_value == want_value:
                equalities += 1
            else:
                failures.append(
                    f"{key}[{index}]: {got_value!r} != registered {want_value!r}"
                )

    _rec(
        item2_equalities=equalities,
        item2_key_count=len(registered_map),
        item2_counted_zero_a_extra_keys=len(extra_keys),
        item2_extra_keys=extra_keys,
        item2_missing_keys=missing_keys,
        item2_code_map={key: list(value) for key, value in code_map.items()},
        item2_failures=failures,
    )

    assert not failures, "item 2: " + "; ".join(failures)
    assert not missing_keys, f"item 2: registered keys absent from code: {missing_keys}"
    assert extra_keys == [], (
        f"item 2 counted zero (a): {len(extra_keys)} extra keys {extra_keys}, want 0"
    )
    assert equalities == 8, f"item 2: {equalities} equalities checked, want 8"


def test_item3_divergence_two_equalities() -> None:
    """Item 3 -- 2/2. The registered ``None`` is asserted PRESENT, not absent."""
    code_config = ARCH_DIVERGENCE_CONFIG[GLM5NEXT_ARCH]

    _rec(
        item3_code_config={
            key: value for key, value in sorted(code_config.items())
        },
        item3_registered_difference_tol=REGISTERED["divergence_difference_tol"],
        item3_registered_n_ulps=REGISTERED["divergence_n_ulps"],
        item3_n_ulps_key_present="divergence_n_ulps" in code_config,
    )

    assert code_config["divergence_difference_tol"] == REGISTERED[
        "divergence_difference_tol"
    ], (
        f"item 3: divergence_difference_tol "
        f"{code_config['divergence_difference_tol']!r} != registered "
        f"{REGISTERED['divergence_difference_tol']!r}"
    )

    # The key must EXIST holding None. A missing key also reads as None through
    # `.get`, and that difference matters: the fixed difference tolerance is
    # consulted only when the ULP count is None, so an entry that merely omitted
    # the key would leave the registered 0.003 as dead code.
    assert "divergence_n_ulps" in code_config, (
        "item 3: divergence_n_ulps is not registered at all; the registered None "
        "is load-bearing and an absent key is not the same registration"
    )
    assert code_config["divergence_n_ulps"] is None, (
        f"item 3: divergence_n_ulps is {code_config['divergence_n_ulps']!r}, "
        f"want None"
    )
    assert REGISTERED["divergence_n_ulps"] is None, (
        "item 3: the registration no longer declares divergence_n_ulps as null"
    )


def test_item4_aggregate_config_nine_values_zero_missing_zero_extra() -> None:
    """Item 4 -- 9/9 values exact, plus counted zeros (b) and (c).

    Completeness *is* the measurement. Consumers read these keys through
    ``.get`` with their own fallbacks, so a partial dict does not fail -- it
    gates on a different number and still reports green.
    """
    registered_config = REGISTERED["aggregate_config"]
    code_config = ARCH_AGGREGATE_CONFIG[GLM5NEXT_ARCH]

    missing = sorted(set(registered_config) - set(code_config))  # counted zero (b)
    extra = sorted(set(code_config) - set(registered_config))  # counted zero (c)

    values_checked = 0
    failures: list[str] = []
    for key in sorted(registered_config):
        if key not in code_config:
            continue
        want = registered_config[key]
        got = code_config[key]
        if isinstance(want, list):
            if list(got) != list(want):
                failures.append(f"{key}: {got!r} != registered {want!r}")
            else:
                for index, (got_value, want_value) in enumerate(zip(got, want)):
                    if got_value != want_value:
                        failures.append(
                            f"{key}[{index}]: {got_value!r} != {want_value!r}"
                        )
        elif got != want:
            failures.append(f"{key}: {got!r} != registered {want!r}")
        values_checked += 1

    _rec(
        item4_values_checked=values_checked,
        item4_registered_key_count=len(registered_config),
        item4_counted_zero_b_missing=len(missing),
        item4_counted_zero_c_extra=len(extra),
        item4_missing_keys=missing,
        item4_extra_keys=extra,
        item4_code_config=code_config,
        item4_failures=failures,
    )

    assert not failures, "item 4: " + "; ".join(failures)
    assert missing == [], (
        f"item 4 counted zero (b): {len(missing)} registered keys missing "
        f"{missing}, want 0 -- a missing agg_bc_threshold silently downgrades "
        f"0.99 to the consumer's 0.95 fallback"
    )
    assert extra == [], (
        f"item 4 counted zero (c): {len(extra)} keys not in the registration "
        f"{extra}, want 0"
    )
    assert values_checked == 9, f"item 4: {values_checked} values checked, want 9"


def test_item5_shared_defaults_unmoved() -> None:
    """Item 5 -- 3/3. The other architectures' regression guard.

    A negative test: it fails if a later hand retunes a shared default in place
    instead of adding an arch-scoped entry beside it.
    """
    _rec(
        item5_default_tolerance_map={
            key: list(value) for key, value in DEFAULT_TOLERANCE_MAP.items()
        },
        item5_default_divergence_difference_tolerance=(
            DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE
        ),
        item5_default_pp_static_thresholds=list(
            DEFAULT_AGGREGATE_CONFIG["pp_static_thresholds"]
        ),
    )

    assert set(DEFAULT_TOLERANCE_MAP) == set(PIN_DEFAULT_TOLERANCE_MAP), (
        f"item 5: DEFAULT_TOLERANCE_MAP key set {sorted(DEFAULT_TOLERANCE_MAP)} "
        f"!= the pin's {sorted(PIN_DEFAULT_TOLERANCE_MAP)}"
    )
    for key, want in PIN_DEFAULT_TOLERANCE_MAP.items():
        got = DEFAULT_TOLERANCE_MAP[key]
        assert len(got) == len(want), f"item 5: {key}: arity {len(got)} != {len(want)}"
        for index, (got_value, want_value) in enumerate(zip(got, want)):
            assert got_value == want_value, (
                f"item 5: DEFAULT_TOLERANCE_MAP[{key!r}][{index}] is "
                f"{got_value!r}, was {want_value!r} at the pin -- the shared "
                f"default moved, which changes every other architecture"
            )

    assert (
        DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE
        == PIN_DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE
    ), (
        f"item 5: DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE is "
        f"{DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE!r}, was "
        f"{PIN_DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE!r} at the pin"
    )

    got_thresholds = list(DEFAULT_AGGREGATE_CONFIG["pp_static_thresholds"])
    assert got_thresholds == PIN_DEFAULT_PP_STATIC_THRESHOLDS, (
        f"item 5: DEFAULT_AGGREGATE_CONFIG['pp_static_thresholds'] is "
        f"{got_thresholds!r}, was {PIN_DEFAULT_PP_STATIC_THRESHOLDS!r} at the "
        f"pin -- the 0.09 rung belongs in the arch-scoped entry, not here"
    )


def test_item6_every_exposed_fp8_dtype_has_an_explicit_entry() -> None:
    """Item 6 -- N/N fp8 keys resolved, plus counted zero (d).

    N is READ FROM THE INSTRUMENT. The adjudicating predicate is key-set
    equality between the registry's fp8 keys and this interpreter's own fp8
    dtype enumeration, so counted zero (d) -- the number of fp8 dtypes falling
    through to the bf16 default -- is a consequence of that equality rather
    than a second count that can drift.
    """
    exposed = _exposed_fp8_dtypes_by_declared_screen()
    registered_fp8 = [dtype for dtype in _DEFAULT_DTYPE_TOLERANCE if _is_fp8(dtype)]

    missing = [dtype for dtype in exposed if dtype not in _DEFAULT_DTYPE_TOLERANCE]
    extra = [dtype for dtype in registered_fp8 if dtype not in set(exposed)]
    # Counted zero (d): a dtype absent from the map falls through the resolver
    # to the bf16 pair. With the key sets equal this is 0 by construction.
    fall_through = [
        dtype
        for dtype in exposed
        if tuple(resolve_dtype_tolerance(dtype)) == tuple(_FALLTHROUGH_DTYPE_TOLERANCE)
    ]

    pair_failures: list[str] = []
    for dtype in exposed:
        if dtype not in _DEFAULT_DTYPE_TOLERANCE:
            continue
        got = _DEFAULT_DTYPE_TOLERANCE[dtype]
        if len(got) != len(FP8_DTYPE_TOLERANCE):
            pair_failures.append(f"{dtype}: arity {len(got)}")
            continue
        for index, (got_value, want_value) in enumerate(zip(got, FP8_DTYPE_TOLERANCE)):
            if got_value != want_value:
                pair_failures.append(
                    f"{dtype}[{index}]: {got_value!r} != {want_value!r}"
                )

    _rec(
        item6_N=len(exposed),
        item6_exposed_names=[str(dtype) for dtype in exposed],
        item6_exposed_itemsizes=[dtype.itemsize for dtype in exposed],
        item6_exposed_is_floating_point=[
            bool(dtype.is_floating_point) for dtype in exposed
        ],
        item6_total_torch_dtype_attributes=sum(
            1
            for name in dir(torch)
            if isinstance(getattr(torch, name, None), torch.dtype)
        ),
        item6_registered_fp8_names=sorted(str(dtype) for dtype in registered_fp8),
        item6_registered_pair=list(FP8_DTYPE_TOLERANCE),
        item6_fallthrough_pair=list(_FALLTHROUGH_DTYPE_TOLERANCE),
        item6_missing=[str(dtype) for dtype in missing],
        item6_extra=[str(dtype) for dtype in extra],
        item6_counted_zero_d_fall_through=len(fall_through),
        item6_pair_failures=pair_failures,
        item6_full_dtype_tolerance_map={
            str(dtype): list(pair) for dtype, pair in _DEFAULT_DTYPE_TOLERANCE.items()
        },
    )

    assert exposed, (
        "item 6: this interpreter's torch exposes NO fp8 dtype, so the "
        "registration has no population -- route this reading to the lead"
    )
    assert not missing, (
        f"item 6: {len(missing)} exposed fp8 dtypes have no explicit entry: "
        f"{[str(d) for d in missing]}"
    )
    assert not extra, (
        f"item 6: {len(extra)} registered fp8 keys this torch does not expose: "
        f"{[str(d) for d in extra]}"
    )
    assert not pair_failures, "item 6: " + "; ".join(pair_failures)
    assert len(fall_through) == 0, (
        f"item 6 counted zero (d): {len(fall_through)} fp8 dtypes fall through to "
        f"{_FALLTHROUGH_DTYPE_TOLERANCE!r}: {[str(d) for d in fall_through]}"
    )

    # The order is this map's own (rtol, atol) -- the REVERSE of tol_map's.
    # Stated as a value assertion so a normalising edit fails here too.
    assert FP8_DTYPE_TOLERANCE[0] == 3e-2, (
        f"item 6: fp8 rtol is {FP8_DTYPE_TOLERANCE[0]!r}, want 3e-2 at index 0"
    )
    assert FP8_DTYPE_TOLERANCE[1] == 1e-5, (
        f"item 6: fp8 atol is {FP8_DTYPE_TOLERANCE[1]!r}, want 1e-5 at index 1"
    )


def test_item7_tuple_order_guards() -> None:
    """Item 7 -- 3/3. The resolver is CALLED, not read.

    A negative test. If a later hand normalises ``_DEFAULT_DTYPE_TOLERANCE`` to
    ``tol_map``'s ``(atol, rtol)`` order, every one of these three fails --
    which is the point, because that edit would silently set the fp8 rtol to
    1e-5 and its atol to 3e-2.
    """
    guards = 0
    failures: list[str] = []
    readings: dict[str, list[Any]] = {}

    for dtype, (want_rtol, want_atol) in PIN_DTYPE_TOLERANCE.items():
        pair = resolve_dtype_tolerance(dtype)  # called, never read out of the map
        readings[str(dtype)] = list(pair)
        if pair[0] != want_rtol:
            failures.append(f"{dtype}: rtol at [0] is {pair[0]!r}, want {want_rtol!r}")
        elif pair[1] != want_atol:
            failures.append(f"{dtype}: atol at [1] is {pair[1]!r}, want {want_atol!r}")
        else:
            guards += 1

    _rec(item7_guards_passed=guards, item7_readings=readings, item7_failures=failures)

    assert not failures, "item 7: " + "; ".join(failures)
    assert guards == 3, f"item 7: {guards} order guards passed, want 3"


def test_guard_b_resolver_matches_the_pin_inline_expression() -> None:
    """Guard B -- 1/1 extraction fidelity for ``resolve_dtype_tolerance``.

    The pin's own inline expression is evaluated **here**, and the extracted
    resolver's return is compared to it element by element, by value. Every key
    in the map is covered, plus dtypes absent from it so the fall-through leg is
    proved too.
    """
    compared = 0
    failures: list[str] = []
    readings: dict[str, dict[str, list[Any]]] = {}

    for dtype in UNREGISTERED_DTYPES:
        assert dtype not in _DEFAULT_DTYPE_TOLERANCE, (
            f"guard B: {dtype} was expected to be absent from the map but is "
            f"registered; the fall-through leg would not be exercised"
        )

    for dtype in list(_DEFAULT_DTYPE_TOLERANCE) + list(UNREGISTERED_DTYPES):
        # The pin's expression, verbatim: the same `.get` with the same literal
        # fall-through pair the two consumer sites used before the extraction.
        inline = _DEFAULT_DTYPE_TOLERANCE.get(dtype, (1.6e-2, 1e-5))
        got = resolve_dtype_tolerance(dtype)
        readings[str(dtype)] = {"inline": list(inline), "resolver": list(got)}

        if len(got) != len(inline):
            failures.append(f"{dtype}: arity {len(got)} != {len(inline)}")
            continue
        for index, (got_value, want_value) in enumerate(zip(got, inline)):
            if got_value != want_value:
                failures.append(
                    f"{dtype}[{index}]: resolver {got_value!r} != inline {want_value!r}"
                )
        compared += 1

    _rec(
        guard_b_dtypes_compared=compared,
        guard_b_registered_key_count=len(_DEFAULT_DTYPE_TOLERANCE),
        guard_b_unregistered_probed=[str(d) for d in UNREGISTERED_DTYPES],
        guard_b_readings=readings,
        guard_b_failures=failures,
    )

    assert not failures, "guard B: " + "; ".join(failures)
    assert compared == len(_DEFAULT_DTYPE_TOLERANCE) + len(UNREGISTERED_DTYPES), (
        f"guard B: compared {compared} dtypes, want "
        f"{len(_DEFAULT_DTYPE_TOLERANCE) + len(UNREGISTERED_DTYPES)}"
    )
