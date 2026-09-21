"""The glm5next accuracy tolerances, checked against a fixture of expected values.

The expected values live in ``fixtures/glm5next_tolerances.json`` and are parsed,
not transcribed into this file, so one hand-copied slip cannot match itself.

One detail is load-bearing throughout: the accuracy ``tol_map`` is ordered
``(atol, rtol)`` while ``testing._DEFAULT_DTYPE_TOLERANCE`` is ordered
``(rtol, atol)``. Normalising either one to the other would set the fp8 rtol to
1e-5 and its atol to 3e-2 -- a three-orders-of-magnitude loosening that still
reads plausibly -- so the tests below assert the orders separately.
"""

from __future__ import annotations

import json
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

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "glm5next_tolerances.json"
EXPECTED = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

#: The dtype entries that predate the glm5next additions, in
#: ``_DEFAULT_DTYPE_TOLERANCE``'s own ``(rtol, atol)`` order. Stated here rather
#: than read back out of the map under test, so the order itself is checked.
BASE_DTYPE_TOLERANCE = {
    torch.float16: (1e-3, 1e-5),
    torch.bfloat16: (1.6e-2, 1e-5),
    torch.float32: (1.3e-6, 1e-5),
}

#: Shared defaults every other architecture reads. The glm5next entries are
#: architecture-scoped additions, so these must not move.
BASE_DEFAULT_TOLERANCE_MAP = {
    "5": (1e-5, 0.011),
    "50": (1e-5, 0.02),
    "1000": (1e-5, 0.03),
    "all": (1e-5, 0.05),
}
BASE_DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE = 0.001
BASE_DEFAULT_PP_STATIC_THRESHOLDS = [0.03, 0.05]

#: Dtypes no entry covers, used for the resolver's fall-through leg. ``int8`` is
#: deliberately 1 byte wide: it proves the fp8 screen classifies on dtype
#: identity, not on width alone.
UNREGISTERED_DTYPES = (torch.int8, torch.float64)


def _exposed_fp8_dtypes() -> list[torch.dtype]:
    """Every fp8 dtype this interpreter's ``torch`` exposes.

    Deliberately a second, independent implementation of the screen the plugin
    uses, and over ``dir(torch)`` rather than the module ``__dict__``: if the two
    disagree, the key-set equality below fails and that disagreement is the
    finding. Classified rather than name-matched -- the attribute must be a
    ``torch.dtype`` one byte wide whose ``str()`` names ``float8`` -- and
    de-duplicated by identity so two names bound to one dtype count once.
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


def test_tolerance_map_matches_the_expected_values() -> None:
    """The glm5next ``tol_map`` equals the fixture, element by element."""
    expected_map = EXPECTED["tol_map"]
    code_map = ARCH_TOLERANCE_MAP[GLM5NEXT_ARCH]

    assert EXPECTED["tol_map_order"] == "(atol, rtol)", (
        "the fixture no longer declares (atol, rtol) order; the comparison below "
        "assumes it"
    )

    assert sorted(code_map) == sorted(expected_map), (
        f"tol_map keys {sorted(code_map)} != expected {sorted(expected_map)}"
    )

    failures: list[str] = []
    for key in sorted(expected_map):
        expected = expected_map[key]
        got = code_map[key]
        if len(got) != len(expected):
            failures.append(f"{key}: arity {len(got)} != {len(expected)}")
            continue
        # Element by element, never by comparing the containers, so an
        # (atol, rtol) flip cannot hide behind an equal pair of pairs.
        for index, (got_value, expected_value) in enumerate(zip(got, expected)):
            if got_value != expected_value:
                failures.append(f"{key}[{index}]: {got_value!r} != expected {expected_value!r}")
    assert not failures, "; ".join(failures)


def test_divergence_config_matches_the_expected_values() -> None:
    """The glm5next divergence entry equals the fixture, ``None`` included."""
    code_config = ARCH_DIVERGENCE_CONFIG[GLM5NEXT_ARCH]

    assert (
        code_config["divergence_difference_tol"]
        == EXPECTED["divergence_difference_tol"]
    ), (
        f"divergence_difference_tol {code_config['divergence_difference_tol']!r} != "
        f"expected {EXPECTED['divergence_difference_tol']!r}"
    )

    # The key must EXIST holding None. A missing key also reads as None through
    # `.get`, and the difference matters: the fixed difference tolerance is
    # consulted only when the ULP count is None, so an entry that merely omitted
    # the key would leave the 0.003 above as dead code.
    assert "divergence_n_ulps" in code_config, (
        "divergence_n_ulps is absent; the None is load-bearing and a missing key "
        "is not the same entry"
    )
    assert code_config["divergence_n_ulps"] is None, (
        f"divergence_n_ulps is {code_config['divergence_n_ulps']!r}, expected None"
    )
    assert EXPECTED["divergence_n_ulps"] is None, (
        "the fixture no longer expects divergence_n_ulps to be null"
    )


def test_aggregate_config_matches_the_expected_values() -> None:
    """The glm5next aggregate entry equals the fixture, with no key missing.

    Completeness is part of the check: consumers read these keys through ``.get``
    with their own fallbacks, so a partial dict does not fail -- it gates on a
    different number and still reports green.
    """
    expected_config = EXPECTED["aggregate_config"]
    code_config = ARCH_AGGREGATE_CONFIG[GLM5NEXT_ARCH]

    missing = sorted(set(expected_config) - set(code_config))
    extra = sorted(set(code_config) - set(expected_config))

    failures: list[str] = []
    for key in sorted(expected_config):
        if key not in code_config:
            continue
        expected = expected_config[key]
        got = code_config[key]
        if isinstance(expected, list):
            if list(got) != list(expected):
                failures.append(f"{key}: {got!r} != expected {expected!r}")
        elif got != expected:
            failures.append(f"{key}: {got!r} != expected {expected!r}")

    assert not failures, "; ".join(failures)
    assert missing == [], (
        f"expected keys missing from the aggregate config {missing} -- a missing "
        f"agg_bc_threshold silently downgrades 0.99 to the consumer's 0.95 fallback"
    )
    assert extra == [], f"aggregate config holds keys the fixture does not expect: {extra}"


def test_shared_defaults_are_unchanged() -> None:
    """The architecture-independent defaults did not move.

    Fails if a later change retunes a shared default in place instead of adding
    an architecture-scoped entry beside it, which would move every other model.
    """
    assert set(DEFAULT_TOLERANCE_MAP) == set(BASE_DEFAULT_TOLERANCE_MAP), (
        f"DEFAULT_TOLERANCE_MAP key set {sorted(DEFAULT_TOLERANCE_MAP)} != "
        f"{sorted(BASE_DEFAULT_TOLERANCE_MAP)}"
    )
    for key, expected in BASE_DEFAULT_TOLERANCE_MAP.items():
        got = DEFAULT_TOLERANCE_MAP[key]
        assert len(got) == len(expected), f"{key}: arity {len(got)} != {len(expected)}"
        for index, (got_value, expected_value) in enumerate(zip(got, expected)):
            assert got_value == expected_value, (
                f"DEFAULT_TOLERANCE_MAP[{key!r}][{index}] is {got_value!r}, expected "
                f"{expected_value!r} -- the shared default moved"
            )

    assert (
        DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE
        == BASE_DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE
    ), (
        f"DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE is "
        f"{DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE!r}, expected "
        f"{BASE_DEFAULT_DIVERGENCE_DIFFERENCE_TOLERANCE!r}"
    )

    got_thresholds = list(DEFAULT_AGGREGATE_CONFIG["pp_static_thresholds"])
    assert got_thresholds == BASE_DEFAULT_PP_STATIC_THRESHOLDS, (
        f"DEFAULT_AGGREGATE_CONFIG['pp_static_thresholds'] is {got_thresholds!r}, "
        f"expected {BASE_DEFAULT_PP_STATIC_THRESHOLDS!r} -- the 0.09 rung belongs in "
        f"the architecture-scoped entry, not here"
    )


def test_every_exposed_fp8_dtype_has_an_explicit_entry() -> None:
    """Each fp8 dtype torch exposes resolves to the fp8 pair, not the fallback.

    How many fp8 dtypes exist is a property of the installed torch, so the check
    is key-set equality between the map's fp8 keys and this interpreter's own
    enumeration rather than a count this file declares.
    """
    exposed = _exposed_fp8_dtypes()
    registered_fp8 = [dtype for dtype in _DEFAULT_DTYPE_TOLERANCE if _is_fp8(dtype)]

    missing = [dtype for dtype in exposed if dtype not in _DEFAULT_DTYPE_TOLERANCE]
    extra = [dtype for dtype in registered_fp8 if dtype not in set(exposed)]
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
        for index, (got_value, expected_value) in enumerate(zip(got, FP8_DTYPE_TOLERANCE)):
            if got_value != expected_value:
                pair_failures.append(f"{dtype}[{index}]: {got_value!r} != {expected_value!r}")

    assert exposed, "this interpreter's torch exposes no fp8 dtype at all"
    assert not missing, (
        f"exposed fp8 dtypes with no explicit entry: {[str(d) for d in missing]}"
    )
    assert not extra, (
        f"fp8 entries this torch does not expose: {[str(d) for d in extra]}"
    )
    assert not pair_failures, "; ".join(pair_failures)
    assert not fall_through, (
        f"fp8 dtypes falling through to {_FALLTHROUGH_DTYPE_TOLERANCE!r}: "
        f"{[str(d) for d in fall_through]}"
    )

    # This map's order is (rtol, atol) -- the reverse of tol_map's. Asserted by
    # value so a normalising edit fails here too.
    assert FP8_DTYPE_TOLERANCE[0] == 3e-2, (
        f"fp8 rtol is {FP8_DTYPE_TOLERANCE[0]!r}, expected 3e-2 at index 0"
    )
    assert FP8_DTYPE_TOLERANCE[1] == 1e-5, (
        f"fp8 atol is {FP8_DTYPE_TOLERANCE[1]!r}, expected 1e-5 at index 1"
    )


def test_resolve_dtype_tolerance_returns_rtol_then_atol() -> None:
    """The resolver is called, and returns ``(rtol, atol)`` for each base dtype."""
    failures: list[str] = []
    for dtype, (expected_rtol, expected_atol) in BASE_DTYPE_TOLERANCE.items():
        pair = resolve_dtype_tolerance(dtype)  # called, never read out of the map
        if pair[0] != expected_rtol:
            failures.append(f"{dtype}: rtol at [0] is {pair[0]!r}, expected {expected_rtol!r}")
        elif pair[1] != expected_atol:
            failures.append(f"{dtype}: atol at [1] is {pair[1]!r}, expected {expected_atol!r}")
    assert not failures, "; ".join(failures)


def test_resolve_dtype_tolerance_matches_the_inline_lookup() -> None:
    """The resolver returns what the call sites' inline ``.get`` expression did.

    Covers every key in the map plus dtypes absent from it, so the fall-through
    leg is exercised too.
    """
    failures: list[str] = []
    readings: dict[str, dict[str, list[Any]]] = {}

    for dtype in UNREGISTERED_DTYPES:
        assert dtype not in _DEFAULT_DTYPE_TOLERANCE, (
            f"{dtype} was expected to be absent from the map but has an entry; the "
            f"fall-through leg would not be exercised"
        )

    for dtype in list(_DEFAULT_DTYPE_TOLERANCE) + list(UNREGISTERED_DTYPES):
        inline = _DEFAULT_DTYPE_TOLERANCE.get(dtype, (1.6e-2, 1e-5))
        got = resolve_dtype_tolerance(dtype)
        readings[str(dtype)] = {"inline": list(inline), "resolver": list(got)}

        if len(got) != len(inline):
            failures.append(f"{dtype}: arity {len(got)} != {len(inline)}")
            continue
        for index, (got_value, expected_value) in enumerate(zip(got, inline)):
            if got_value != expected_value:
                failures.append(
                    f"{dtype}[{index}]: resolver {got_value!r} != inline {expected_value!r}"
                )

    assert not failures, f"{'; '.join(failures)} (readings: {readings})"
