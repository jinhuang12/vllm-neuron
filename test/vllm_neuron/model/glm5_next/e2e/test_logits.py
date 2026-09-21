# SPDX-License-Identifier: Apache-2.0
"""The logit comparison instrument for GLM-5.3-Flash, and the numbers it loads.

Comparing Neuron logits against a GPU reference needs three arch-scoped entries from
``vllm_neuron.accuracy``: a tolerance map, a divergence pair and an aggregate config. This
file holds no tolerance of its own. It loads all three through a strict resolver, checks
that every key they are expected to carry is present, runs one synthetic logit pair through
``assert_close_logit_pair``, and requires an absent arch entry to raise rather than fall
back to the plugin default.

The fallback is the hazard worth a test: ``DEFAULT_TOLERANCE_MAP`` carries the same row keys
as the arch-scoped map and every one of its rows is tighter, so a silent fallback would not
crash and would not look sloppy -- it would look stricter while comparing against the wrong
tolerances.
"""

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

# The keys each entry carries. These are names and counts, never values. The tolerance map's
# row keys are numeric strings, so they are iterated from the loaded map rather than spelled
# here, and their completeness is read by comparing the arch-scoped row set against the
# default map's: two loaded sources, no literal on either side.
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

_UNREGISTERED_ARCH = "unregistered architecture"


class UnregisteredArchError(LookupError):
    """An arch-scoped entry is absent.

    Raised instead of returning a default. The plugin default exists and would resolve
    cleanly, which is the hazard this error exists to prevent.
    """


def load_arch_entry(mapping, arch, what):
    """Return ``mapping[arch]``, or raise. Never falls back to a default.

    Args:
        mapping: An arch-scoped map.
        arch: The architecture key.
        what: Human-readable label for the message.

    Raises:
        UnregisteredArchError: If ``arch`` has no entry. The message carries the prefix
            ``unregistered architecture`` and names the refusal, so a reader of the failure
            sees that defaulting was declined rather than overlooked.
    """
    try:
        return mapping[arch]
    except KeyError as exc:
        raise UnregisteredArchError(
            f"{_UNREGISTERED_ARCH}: {what} has no entry for {arch!r}; refusing to fall back "
            f"to a plugin default, which would compare against different tolerances"
        ) from exc


def _load_all():
    """The three arch-scoped entries, each through the strict resolver."""
    return (
        load_arch_entry(ARCH_TOLERANCE_MAP, GLM5NEXT_ARCH, "arch tolerance map"),
        load_arch_entry(ARCH_DIVERGENCE_CONFIG, GLM5NEXT_ARCH, "arch divergence config"),
        load_arch_entry(ARCH_AGGREGATE_CONFIG, GLM5NEXT_ARCH, "arch aggregate config"),
    )


def _load_site_ids():
    """One id per value the instrument loads, built from loaded keys plus declared names."""
    tol_rows, _, _ = _load_all()
    sites = [("arch_tolerance_map", key) for key in sorted(tol_rows)]
    sites += [("arch_divergence_config", key) for key in _DIVERGENCE_KEYS]
    sites += [("arch_aggregate_config", key) for key in _AGGREGATE_KEYS]
    return sites


@pytest.mark.parametrize("entry_name,key", _load_site_ids())
def test_every_loaded_value_resolves(entry_name, key):
    """Every value the instrument loads resolves to a present key."""
    tol_rows, divergence, aggregate = _load_all()
    holder = {
        "arch_tolerance_map": tol_rows,
        "arch_divergence_config": divergence,
        "arch_aggregate_config": aggregate,
    }[entry_name]
    assert key in holder, f"{entry_name} lost its key {key!r}"


def test_the_three_entries_carry_exactly_the_keys_they_are_read_for():
    """4 tolerance rows, 2 divergence keys and 9 aggregate keys, so a lost key fails here."""
    tol_rows, divergence, aggregate = _load_all()

    assert len(tol_rows) == _TOLERANCE_ROW_COUNT
    # Row completeness without spelling a row key: both sides are loaded.
    assert sorted(tol_rows) == sorted(DEFAULT_TOLERANCE_MAP), (
        "the arch-scoped tolerance map no longer covers exactly the default map's rows"
    )
    assert sorted(divergence) == sorted(_DIVERGENCE_KEYS)
    assert sorted(aggregate) == sorted(_AGGREGATE_KEYS)


def test_one_logit_pair_yields_one_comparison_result():
    """One synthetic logit pair yields exactly one ``AssertCloseResult``.

    The tensors and the perturbation are both derived from the loaded tolerances, so the
    test data holds no number of its own. A perturbation of exactly the absolute tolerance
    sits on the passing boundary; twice the full tolerance band is outside it, which shows
    the comparator can fail as well as pass.
    """
    tol_rows, _, _ = _load_all()
    atol, rtol = tol_rows[sorted(tol_rows)[0]]

    expected = (torch.arange(4, dtype=torch.float32) + 1).reshape(2, 2)
    near = expected + atol
    far = expected + (atol + rtol * expected.abs()) * 2

    result = assert_close_logit_pair(near, expected, rtol=rtol, atol=atol, name="near")
    beyond = assert_close_logit_pair(far, expected, rtol=rtol, atol=atol, name="far")

    assert isinstance(result, AssertCloseResult)
    assert result.allclose is True
    assert result.num_mismatches == 0
    assert beyond.allclose is False, "the comparator cannot fail, so its pass proves nothing"


def test_an_absent_arch_entry_raises_rather_than_falling_back(monkeypatch):
    """Removing an arch-scoped entry raises ``UnregisteredArchError``, never a default.

    The ground for this is read and not assumed: ``DEFAULT_TOLERANCE_MAP`` carries the same
    row keys as the arch-scoped map and every row is tighter, so falling back would return a
    usable, stricter-looking map for the wrong comparison.
    """
    arch_rows = load_arch_entry(ARCH_TOLERANCE_MAP, GLM5NEXT_ARCH, "arch tolerance map")
    tighter = {
        key: (DEFAULT_TOLERANCE_MAP[key][1], arch_rows[key][1]) for key in sorted(arch_rows)
    }
    assert all(pair[0] < pair[1] for pair in tighter.values()), (
        "the default rows are no longer tighter, so this test's stated ground has moved"
    )
    assert load_arch_entry(ARCH_TOLERANCE_MAP, GLM5NEXT_ARCH, "present") is arch_rows

    monkeypatch.delitem(ARCH_TOLERANCE_MAP, GLM5NEXT_ARCH)
    monkeypatch.delitem(ARCH_DIVERGENCE_CONFIG, GLM5NEXT_ARCH)
    monkeypatch.delitem(ARCH_AGGREGATE_CONFIG, GLM5NEXT_ARCH)

    with pytest.raises(UnregisteredArchError) as excinfo:
        load_arch_entry(ARCH_TOLERANCE_MAP, GLM5NEXT_ARCH, "arch tolerance map")

    message = str(excinfo.value)
    assert _UNREGISTERED_ARCH in message
    assert "refusing to fall back" in message

    # The other two entries refuse on the same contract, not just the one asked first.
    for mapping, label in (
        (ARCH_DIVERGENCE_CONFIG, "arch divergence config"),
        (ARCH_AGGREGATE_CONFIG, "arch aggregate config"),
    ):
        with pytest.raises(UnregisteredArchError):
            load_arch_entry(mapping, GLM5NEXT_ARCH, label)

    # Nothing about the absent arch entry disturbs the default map, which is what would have
    # made a fallback tempting.
    assert sorted(DEFAULT_TOLERANCE_MAP) == sorted(tighter)
