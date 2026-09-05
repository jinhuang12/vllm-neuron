# SPDX-License-Identifier: Apache-2.0
"""The M4 gate harness for GLM-5.3-Flash: it assembles the instrument, it does not set the criterion.

WHAT THIS FILE IS. The M4 gate compares Neuron logits against a GPU reference under registered
tolerances. This file builds and proves the *instrument* that gate will use. It authors no
tolerance, no threshold and no pass criterion. Every number it compares against is loaded at run
time from the arch-scoped registry entries in ``vllm_neuron.accuracy``; the single source of truth
for those numbers is the campaign's acceptance pre-registration, section 2.7.

WHAT IT PROVES, IN FOUR PARTS.

1. All fifteen registered load sites resolve, each by import from the module that holds it: four
   tolerance rows and a two-value divergence pair from ``accuracy.constants``, and a nine-key
   aggregate config from ``accuracy.logit_validation``. The *count* is asserted as well as the
   values, so a registry entry that silently loses a key fails this harness rather than surviving
   to distort the gate. ``AssertCloseResult`` is imported too but is deliberately NOT one of the
   fifteen: it is a result type, not a registered value, and counting it would give sixteen.

2. The harness holds no tolerance value of its own. A source-level census over this very file
   requires zero hits across three populations: every float literal, every int literal outside a
   small printed structural allow-list, and every string literal equal to one of the registry's
   row keys. A harness carrying its own copy of a tolerance would keep passing after the registry
   moved, which is the one failure a loading harness must not have.

   THE THIRD POPULATION IS THERE BECAUSE OF A GAP IN THIS FILE'S FIRST VERSION. The registry's
   row keys are strings, not numbers, so a numeric-only census could not see ``tol_rows["1000"]``
   at all -- a spelled row key would hard-code a registry value and pass the census untouched.
   The keys searched for are read from the loaded map at census time, and the control's source
   text is built from one of them, so this file spells no row key even while testing for spelled
   row keys. Each of the three controls must read exactly one hit; a census that cannot report a
   hit cannot report a zero either.

3. One synthetic logit pair goes end to end through ``assert_close_logit_pair`` and yields exactly
   one ``AssertCloseResult``. Both the tensors and the perturbation applied to them are derived
   from loaded values, so even the test data authors no number.

4. With the arch-scoped entry absent, resolution RAISES instead of falling back to the plugin
   default. This is the part with teeth. ``DEFAULT_TOLERANCE_MAP`` carries the same four row keys
   as the arch-scoped map and every one of its rows is *tighter*, so a silent fallback would not
   crash and would not look sloppy -- it would look stricter while measuring the wrong comparator.
   That is why resolution here refuses rather than defaults, and why the negative case asserts the
   exception type rather than merely that something went wrong.

WHY A RESOLVER LIVES HERE. No arch-scoped resolver exists in plugin source; today's only consumers
subscript the maps directly. The strict resolver below is therefore part of the instrument, and its
contract is the thing part 4 measures: an absent entry is an error, never a default.
"""

import ast
import pathlib

import pytest
import torch

from vllm_neuron.accuracy.constants import (
    ARCH_DIVERGENCE_CONFIG,
    ARCH_TOLERANCE_MAP,
    DEFAULT_TOLERANCE_MAP,
    GLM5NEXT_ARCH,
)
from vllm_neuron.accuracy.logit_validation import ARCH_AGGREGATE_CONFIG
from vllm_neuron.accuracy.testing import AssertCloseResult, assert_close_logit_pair

# --------------------------------------------------------------------------------------------
# The declared shape of the registration. These are COUNTS and KEY NAMES, never values.
#
# The tolerance map's row keys are NOT spelled here on purpose. They are numeric ("5", "50",
# "1000"), so writing them would put registry content in the harness -- the very thing part 2
# forbids. They are iterated from the loaded map instead, and their completeness is proved by
# comparing the arch-scoped row set against the default map's row set: two loaded sources, no
# literal on either side. That comparison is also part 4's ground, since identical row keys are
# exactly what makes a silent fallback possible rather than a loud KeyError.
# --------------------------------------------------------------------------------------------
_TOLERANCE_ROW_COUNT = 4
_DIVERGENCE_KEYS = ("divergence_difference_tol", "divergence_n_ulps")
_AGGREGATE_KEYS = (
    "pp_static_thresholds",
    "pp_linf_multipliers",
    "pp_l2_multipliers",
    "pp_tok_linf_multipliers",
    "pp_tok_l2_multipliers",
    "agg_bc_threshold",
    "agg_linf_multipliers",
    "agg_l2_multipliers",
    "agg_sigma_ratio_threshold",
)
_REGISTERED_LOAD_SITE_TOTAL = 15

#: Integers this file may spell: the counted zero, the controls' one, and the declared counts.
#: Everything else -- a tolerance, a ULP count, a registry row key -- is a census hit.
_STRUCTURAL_INT_ALLOW_LIST = frozenset({0, 1, 2, 4, 9, 15})

_UNREGISTERED_ARCH = "unregistered architecture"


class UnregisteredArchError(LookupError):
    """An arch-scoped registry entry is absent.

    Raised instead of returning a default. The plugin default exists and would resolve
    cleanly, which is the hazard: see part 4 in this module's docstring.
    """


def load_arch_entry(mapping, arch, what):
    """Return ``mapping[arch]``, or raise. Never falls back to a default.

    Args:
        mapping: An arch-scoped registry map.
        arch: The architecture key.
        what: Human-readable label for the message.

    Raises:
        UnregisteredArchError: If ``arch`` has no entry. The message carries the verbatim
            prefix ``unregistered architecture`` and names the refusal explicitly, so a
            reader of the failure sees that defaulting was declined rather than overlooked.
    """
    try:
        return mapping[arch]
    except KeyError as exc:
        raise UnregisteredArchError(
            f"{_UNREGISTERED_ARCH}: {what} has no entry for {arch!r}; refusing to fall back "
            f"to a plugin default, which would compare against a different comparator"
        ) from exc


def census_authored_literals(source, allow_list, row_keys):
    """Count authored tolerance literals in ``source``: numerics, plus spelled row keys.

    The population has three parts:

    1. every float literal;
    2. every int literal whose value is not allow-listed;
    3. every string literal equal to one of ``row_keys``.

    Part 3 exists because the registry's row keys are STRINGS, not numbers. A numeric-only
    census cannot see ``tol_rows["1000"]`` at all, so a spelled row key would evade it while
    still hard-coding a value that belongs to the registry. That was a real gap in this
    harness's first version; it is closed here rather than left latent.

    ``row_keys`` is supplied by the caller, which reads it from the loaded map at census
    time. That is the point: this file must never spell a row key, not even in order to
    search for one, so the thing being looked for is loaded rather than typed.

    Two notes on what this still cannot see:

    * Booleans are excluded explicitly. ``bool`` is a subclass of ``int`` in Python, so a
      plain ``isinstance(value, int)`` test would count every ``True`` and ``False``.
    * A negative literal is invisible as such. ``ast`` parses ``-1`` as a unary minus applied
      to the constant ``1``, so the census sees the magnitude only. This file writes no
      negative numeric literal, so nothing hides there, but the limit is real and stated
      rather than left for a reader to discover.

    Returns:
        A list of ``(lineno, value)`` pairs, one per hit.
    """
    keys = frozenset(row_keys)
    hits = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Constant):
            continue
        value = node.value
        if isinstance(value, bool):
            continue
        if isinstance(value, float):
            hits.append((node.lineno, value))
        elif isinstance(value, int) and value not in allow_list:
            hits.append((node.lineno, value))
        elif isinstance(value, str) and value in keys:
            hits.append((node.lineno, value))
    return hits


def _load_all():
    """The three arch-scoped entries, each through the strict resolver."""
    return (
        load_arch_entry(ARCH_TOLERANCE_MAP, GLM5NEXT_ARCH, "arch tolerance map"),
        load_arch_entry(ARCH_DIVERGENCE_CONFIG, GLM5NEXT_ARCH, "arch divergence config"),
        load_arch_entry(ARCH_AGGREGATE_CONFIG, GLM5NEXT_ARCH, "arch aggregate config"),
    )


# --------------------------------------------------------------------------------------------
# Part 1 -- fifteen named load sites resolve, and the count is asserted both ways.
# --------------------------------------------------------------------------------------------
def _load_site_ids():
    """One id per registered load site, built from loaded keys plus declared names."""
    tol_rows, _, _ = _load_all()
    sites = [("arch_tolerance_map", key) for key in sorted(tol_rows)]
    sites += [("arch_divergence_config", key) for key in _DIVERGENCE_KEYS]
    sites += [("arch_aggregate_config", key) for key in _AGGREGATE_KEYS]
    return sites


@pytest.mark.parametrize("entry_name,key", _load_site_ids())
def test_harness_resolves_each_registered_load_site(entry_name, key):
    """Every one of the fifteen registered load sites resolves to a present value."""
    tol_rows, divergence, aggregate = _load_all()
    holder = {
        "arch_tolerance_map": tol_rows,
        "arch_divergence_config": divergence,
        "arch_aggregate_config": aggregate,
    }[entry_name]
    assert key in holder, f"{entry_name} lost its registered key {key!r}"


def test_harness_asserts_the_registered_load_site_count(capsys):
    """The counts are 4 + 2 + 9 = 15, asserted so a lost or added key fails here."""
    tol_rows, divergence, aggregate = _load_all()
    default_rows = DEFAULT_TOLERANCE_MAP

    with capsys.disabled():
        print(f"\n  arch tolerance rows      : {sorted(tol_rows)}")
        print(f"  arch divergence keys     : {sorted(divergence)}")
        print(f"  arch aggregate keys      : {len(aggregate)} keys")
        print(f"  total registered sites   : {len(tol_rows) + len(divergence) + len(aggregate)}")

    assert len(tol_rows) == _TOLERANCE_ROW_COUNT
    # Row completeness without spelling a row key: both sides are loaded.
    assert sorted(tol_rows) == sorted(default_rows), (
        "the arch-scoped tolerance map no longer covers exactly the default map's rows"
    )
    assert sorted(divergence) == sorted(_DIVERGENCE_KEYS)
    assert sorted(aggregate) == sorted(_AGGREGATE_KEYS)

    total = len(tol_rows) + len(divergence) + len(aggregate)
    assert total == _REGISTERED_LOAD_SITE_TOTAL

    # The result type is imported but is not a registered value; counting it would give 16.
    assert isinstance(AssertCloseResult, type)
    assert total + 1 != _REGISTERED_LOAD_SITE_TOTAL


# --------------------------------------------------------------------------------------------
# Part 2 -- the counted zero: this harness holds no number of its own.
# --------------------------------------------------------------------------------------------
def test_harness_authors_no_authored_tolerance_literal(capsys):
    """Zero authored tolerance literals in this file: no numerics, no spelled row keys.

    Three planted controls run first and must each read exactly one hit: a float, an int
    outside the allow-list, and a spelled registry row key. A census that cannot report a hit
    cannot report a zero either.

    The row keys are read from the loaded tolerance map, and the third control's source text is
    built from one of them. So this file spells no row key even while testing for spelled row
    keys -- which is the only way that test can be honest about its own subject.
    """
    tol_rows, _, _ = _load_all()
    row_keys = tuple(tol_rows)

    planted_float = "tolerance = 0.033\n"
    planted_int = "ulps = 3\n"
    planted_key = f"row = tol_rows[{sorted(row_keys)[0]!r}]\n"

    float_hits = census_authored_literals(planted_float, _STRUCTURAL_INT_ALLOW_LIST, row_keys)
    int_hits = census_authored_literals(planted_int, _STRUCTURAL_INT_ALLOW_LIST, row_keys)
    key_hits = census_authored_literals(planted_key, _STRUCTURAL_INT_ALLOW_LIST, row_keys)

    own_source = pathlib.Path(__file__).read_text()
    own_hits = census_authored_literals(own_source, _STRUCTURAL_INT_ALLOW_LIST, row_keys)

    with capsys.disabled():
        print(f"\n  allow-list (structural ints) : {sorted(_STRUCTURAL_INT_ALLOW_LIST)}")
        print(f"  row keys, loaded not spelled : {sorted(row_keys)}")
        print(f"  planted float control        : {len(float_hits)} hit(s) {float_hits}")
        print(f"  planted int control          : {len(int_hits)} hit(s) {int_hits}")
        print(f"  planted row-key control      : {len(key_hits)} hit(s) {key_hits}")
        print(f"  census over this file        : {len(own_hits)} hit(s) {own_hits}")
        print("  population: floats + non-allow-listed ints + strings equal to a loaded row key")
        print("  booleans excluded explicitly; a negative literal is seen as its magnitude")

    assert len(float_hits) == 1, "the census cannot see a planted float"
    assert len(int_hits) == 1, "the census cannot see a planted non-structural int"
    assert len(key_hits) == 1, "the census cannot see a planted spelled row key"
    assert own_hits == [], f"this harness authors tolerance values of its own: {own_hits}"


# --------------------------------------------------------------------------------------------
# Part 3 -- one synthetic pair, one AssertCloseResult, end to end.
# --------------------------------------------------------------------------------------------
def test_harness_produces_one_comparison_result(capsys):
    """One synthetic logit pair yields exactly one ``AssertCloseResult``.

    The tensors and the perturbation are both derived from loaded values, so the test data
    authors no number either. A perturbation of exactly the registered absolute tolerance sits
    on the passing boundary; twice the full tolerance band is outside it, which gives the
    comparator a firing control rather than a single one-sided reading.
    """
    tol_rows, _, _ = _load_all()
    atol, rtol = tol_rows[sorted(tol_rows)[0]]

    expected = (torch.arange(4, dtype=torch.float32) + 1).reshape(2, 2)
    near = expected + atol
    far = expected + (atol + rtol * expected.abs()) * 2

    result = assert_close_logit_pair(near, expected, rtol=rtol, atol=atol, name="m4_harness")
    control = assert_close_logit_pair(far, expected, rtol=rtol, atol=atol, name="m4_control")

    with capsys.disabled():
        print(f"\n  registered (atol, rtol) used : ({atol}, {rtol})")
        print(f"  within-tolerance pair        : allclose={result.allclose}")
        print(f"  beyond-tolerance control     : allclose={control.allclose}")

    assert isinstance(result, AssertCloseResult)
    assert result.allclose is True
    assert control.allclose is False, "the comparator cannot fail, so its pass proves nothing"
    assert result.num_mismatches == 0


# --------------------------------------------------------------------------------------------
# Part 4 -- an absent arch entry raises; it does not fall back to the tighter default.
# --------------------------------------------------------------------------------------------
def test_harness_refuses_absent_arch_entry_rather_than_falling_back(monkeypatch, capsys):
    """Removing the arch-scoped entry raises ``UnregisteredArchError``, never a default.

    The ground for this test is measured, not assumed: ``DEFAULT_TOLERANCE_MAP`` carries the
    same row keys as the arch-scoped map and every row is tighter, so falling back would
    return a usable, stricter-looking map for the wrong comparator.
    """
    arch_rows = load_arch_entry(ARCH_TOLERANCE_MAP, GLM5NEXT_ARCH, "arch tolerance map")
    tighter = {
        key: (DEFAULT_TOLERANCE_MAP[key][1], arch_rows[key][1]) for key in sorted(arch_rows)
    }

    with capsys.disabled():
        print("\n  per-row rtol, default vs arch (item 4's ground):")
        for key, (default_rtol, arch_rtol) in tighter.items():
            print(f"    row {key!r:>8}: default={default_rtol} arch={arch_rtol} "
                  f"default_tighter={default_rtol < arch_rtol}")

    assert all(pair[0] < pair[1] for pair in tighter.values()), (
        "the default rows are no longer tighter, so this test's stated ground has moved"
    )

    # Control: with the entry present, resolution returns the ARCH map and not the default.
    assert load_arch_entry(ARCH_TOLERANCE_MAP, GLM5NEXT_ARCH, "control") is arch_rows

    monkeypatch.delitem(ARCH_TOLERANCE_MAP, GLM5NEXT_ARCH)
    monkeypatch.delitem(ARCH_DIVERGENCE_CONFIG, GLM5NEXT_ARCH)
    monkeypatch.delitem(ARCH_AGGREGATE_CONFIG, GLM5NEXT_ARCH)

    with pytest.raises(UnregisteredArchError) as excinfo:
        load_arch_entry(ARCH_TOLERANCE_MAP, GLM5NEXT_ARCH, "arch tolerance map")

    message = str(excinfo.value)
    assert _UNREGISTERED_ARCH in message
    assert "refusing to fall back" in message
    assert excinfo.type is UnregisteredArchError

    # The other two entries refuse on the same contract, not just the one that was asked first.
    for mapping, label in (
        (ARCH_DIVERGENCE_CONFIG, "arch divergence config"),
        (ARCH_AGGREGATE_CONFIG, "arch aggregate config"),
    ):
        with pytest.raises(UnregisteredArchError):
            load_arch_entry(mapping, GLM5NEXT_ARCH, label)

    # The default map is still intact, which is what made the fallback tempting in the first
    # place: nothing about the absent arch entry disturbs it.
    assert sorted(DEFAULT_TOLERANCE_MAP) == sorted(tighter)
