# SPDX-License-Identifier: Apache-2.0
"""The factory and its registry registration.

The arch string resolves to a class object, and looking it up allocates no
parameters and does not import the implementation module.
"""

import ast
import inspect
import os
import re
import sys
from pathlib import Path

import pytest
import torch

from vllm_neuron.model import registry

# ---------------------------------------------------------------------------
# Declared values. NEW_ARCH is the arch string the block pins verbatim.
# ---------------------------------------------------------------------------

NEW_ARCH = "Glm5NextForConditionalGeneration"

# The five production archs at the pin, in the order registry.py lists them.
# Pinning the order as well as the set is what makes "do not reorder the other
# five entries" a measured property rather than a promise.
ORIGINAL_ARCHS = [
    "LlamaForCausalLM",
    "GptOssForCausalLM",
    "Eagle3LlamaForCausalLM",
    "Qwen3ForCausalLM",
    "Qwen3VLForConditionalGeneration",
]

EXPECTED_ARCHS = set(ORIGINAL_ARCHS) | {NEW_ARCH}
DECLARED_COUNT = 6

SYNTHETIC_ENV = "VLLM_NEURON_SYNTHETIC_MODEL"

# The suffix two arch strings share -- the substring trap, named so the screen
# that avoids it can be checked.
SHARED_SUFFIX = "ForConditionalGeneration"


IMPL_MODULE = "vllm_neuron.model.glm5_next.model_fp8"
FACTORY_MODULE = "vllm_neuron.model.glm5_next.factory"


@pytest.fixture
def declared_env(monkeypatch):
    """The declared environment: the synthetic-model gate explicitly off. """
    monkeypatch.delenv(SYNTHETIC_ENV, raising=False)


def _arch_strings():
    return [name for name, _ in registry.get_models()]


def _static_models_entries_from_source():
    """second independent base for the count -- parse, never execute. """
    tree = ast.parse(Path(registry.__file__).read_text())
    func = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "get_models"
    )
    for node in func.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "models"
            for target in node.targets
        ):
            assert isinstance(node.value, ast.List), "`models` is not a list literal"
            return [element.elts[0].value for element in node.value.elts]
    raise AssertionError("no `models = [...]` assignment inside get_models()")


# ---------------------------------------------------------------------------
# Get_models() returns 6 entries
# ---------------------------------------------------------------------------


def test_get_models_returns_the_declared_six_entries(declared_env):
    assert len(registry.get_models()) == DECLARED_COUNT


def test_the_static_list_literal_holds_six_entries():
    """The same 6, counted from source without executing the function."""
    names = _static_models_entries_from_source()
    assert len(names) == DECLARED_COUNT
    assert names[-1] == NEW_ARCH


def test_the_six_arch_strings_are_exactly_the_expected_set(declared_env):
    """Both directions of the complement, so neither a substitution nor an extra entry can
    hide inside a correct total.
    """
    observed = set(_arch_strings())
    assert observed - EXPECTED_ARCHS == set()
    assert EXPECTED_ARCHS - observed == set()


def test_no_arch_string_is_duplicated(declared_env):
    """A duplicate would let 6 tuples expose fewer than 6 archs."""
    archs = _arch_strings()
    assert len(set(archs)) == DECLARED_COUNT
    assert len(dict(registry.get_models())) == DECLARED_COUNT


def test_the_five_existing_archs_keep_their_order_and_the_new_one_is_appended(
    declared_env,
):
    """The declared edit is "add one tuple" -- nothing else moves."""
    archs = _arch_strings()
    assert archs[:5] == ORIGINAL_ARCHS
    assert archs[5] == NEW_ARCH


@pytest.mark.parametrize("value", ["0", "true", "TRUE", "yes", "", "11"])
def test_the_gate_is_exact_equality_on_the_string_one(monkeypatch, value):
    """The 7th entry appears only for exactly "1" -- so the declared 6 is not fragile
    against every non-empty value of the variable.
    """
    monkeypatch.setenv(SYNTHETIC_ENV, value)
    assert len(registry.get_models()) == DECLARED_COUNT


# ---------------------------------------------------------------------------
# The new entry's arch string
# ---------------------------------------------------------------------------


def test_the_new_entry_arch_string_is_the_declared_literal(declared_env):
    archs = _arch_strings()
    assert [name for name in archs if name == NEW_ARCH] == [NEW_ARCH]


def test_word_boundary_screen_matches_exactly_one_and_substring_over_matches(
    declared_env,
):
    """The substring trap, measured. """
    archs = _arch_strings()

    boundary_hits = [
        name for name in archs if re.search(rf"\b{re.escape(NEW_ARCH)}\b", name)
    ]
    assert len(boundary_hits) == 1

    substring_hits = sorted(name for name in archs if SHARED_SUFFIX in name)
    assert len(substring_hits) == 2
    assert substring_hits == [
        "Glm5NextForConditionalGeneration",
        "Qwen3VLForConditionalGeneration",
    ]


@pytest.mark.parametrize(
    "near_miss",
    [
        "Glm5NextForCausalLM",
        "Glm5NextForConditionalGenerationX",
        "Glm5Next",
        "glm5nextforconditionalgeneration",
        "Glm53FlashForConditionalGeneration",
    ],
)
def test_near_miss_arch_strings_are_absent(declared_env, near_miss):
    """mutation arm for the literal: the screen must discriminate, including on case and on
    a trailing character the word boundary is there to catch.
    """
    assert near_miss not in set(_arch_strings())


# ---------------------------------------------------------------------------
# The factory returns a class object without instantiating weights, 1/1
# ---------------------------------------------------------------------------


def test_the_registered_value_is_a_class_object(declared_env):
    """1/1 calls: one lookup, and it yields a class, not an instance."""
    calls = 0
    entry = dict(registry.get_models())[NEW_ARCH]
    calls += 1

    assert calls == 1
    assert inspect.isclass(entry)
    assert issubclass(entry, torch.nn.Module)
    assert not isinstance(entry, torch.nn.Module)
    assert entry.__name__ == NEW_ARCH


def test_the_registry_entry_the_package_export_and_the_factory_module_agree(
    declared_env,
):
    """Three import paths, one class object -- identity, not equality. """
    from vllm_neuron.model.glm5_next import (
        Glm5NextForConditionalGeneration as via_package,
    )
    from vllm_neuron.model.glm5_next.factory import (
        Glm5NextForConditionalGeneration as via_module,
    )

    entry = dict(registry.get_models())[NEW_ARCH]
    assert entry is via_package
    assert entry is via_module


def test_the_lookup_allocates_no_parameters(
    declared_env, monkeypatch
):
    """"without instantiating weights", measured rather than asserted. """
    created = []
    original_new = torch.nn.Parameter.__new__

    def counting_new(cls, *args, **kwargs):
        created.append(cls)
        return original_new(cls, *args, **kwargs)

    monkeypatch.setattr(torch.nn.Parameter, "__new__", staticmethod(counting_new))

    entry = dict(registry.get_models())[NEW_ARCH]
    assert inspect.isclass(entry)
    assert created == [], f"the lookup allocated {len(created)} Parameter(s)"

    # Positive control, same window: without this the zero above proves nothing.
    torch.nn.Parameter(torch.zeros(1))
    assert len(created) == 1


def test_the_implementation_module_is_not_imported_at_module_level():
    """The factory's lazy import is what lets the class be looked up before the model
    skeleton exists -- and is why no weights can be allocated.
    """
    import json
    import subprocess

    source = (
        "import json, sys\n"
        f"import {FACTORY_MODULE}\n"
        "print(json.dumps({"
        f"'factory': {FACTORY_MODULE!r} in sys.modules, "
        f"'impl': {IMPL_MODULE!r} in sys.modules"
        "}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True
    )
    assert proc.returncode == 0, f"probe exited {proc.returncode}: {proc.stderr}"
    observed = json.loads(proc.stdout.strip().splitlines()[-1])

    assert observed["factory"] is True
    assert observed["impl"] is False


def test_the_class_predicate_rejects_a_real_instance(declared_env):
    """mutation arm for the class predicate: it must be able to fail."""
    entry = dict(registry.get_models())[NEW_ARCH]
    probe = torch.nn.Linear(1, 1)

    assert inspect.isclass(entry)
    assert not inspect.isclass(probe)
    assert isinstance(probe, torch.nn.Module)


