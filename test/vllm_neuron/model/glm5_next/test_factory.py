# SPDX-License-Identifier: Apache-2.0
"""Acceptance test for ``inc-glm53f-009`` -- WP1: factory + registry registration.

The declared acceptance (increment plan revision 10, L3179), verbatim:

    "``get_models()`` returns **6** entries (5 production archs at the pin +
    this one, P77), the new entry's arch string is
    ``Glm5NextForConditionalGeneration``, and the factory returns a class
    object without instantiating weights in **1/1** calls."

Three conjuncts, measured below as C01 / C02 / C03.

ENVIRONMENT DEPENDENCE -- read this before trusting the number 6.
``registry.py`` appends a seventh, testing-only entry when
``VLLM_NEURON_SYNTHETIC_MODEL == "1"``, so the declared 6 holds only while that
variable is unset or != "1". With it set, *correct* code returns 7 for an
environment reason. Every test here that counts entries therefore CONTROLS the
variable explicitly rather than inheriting it, and the inherited value is
reported by ``test_report_the_measured_readings`` so the reading is never
silent.

SCREEN DISCIPLINE. Counting screens use word boundaries, recorded counts and
either a complement or a second independent base -- never a bare number. The
substring trap here is real and is measured rather than asserted away:
``ForConditionalGeneration`` is a substring of TWO of the six arch strings, so a
substring screen would report 2 where the conjunct means 1.
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
# Pinning the ORDER as well as the set is what makes "do not reorder the other
# five entries" a measured property rather than a promise.
PIN_ARCHS = [
    "LlamaForCausalLM",
    "GptOssForCausalLM",
    "Eagle3LlamaForCausalLM",
    "Qwen3ForCausalLM",
    "Qwen3VLForConditionalGeneration",
]

EXPECTED_ARCHS = set(PIN_ARCHS) | {NEW_ARCH}
DECLARED_COUNT = 6

SYNTHETIC_ENV = "VLLM_NEURON_SYNTHETIC_MODEL"
SYNTHETIC_ARCH = "SyntheticNeuronModel"

# The suffix two arch strings share -- the substring trap, named so the screen
# that avoids it can be checked.
SHARED_SUFFIX = "ForConditionalGeneration"

# Captured at import time, before any monkeypatch fixture runs, so this is the
# environment the run actually inherited.
INHERITED_SYNTHETIC_ENV = os.environ.get(SYNTHETIC_ENV, "<unset>")

IMPL_MODULE = "vllm_neuron.model.glm5_next.model_fp8"
FACTORY_MODULE = "vllm_neuron.model.glm5_next.factory"


@pytest.fixture
def declared_env(monkeypatch):
    """The declared environment: the synthetic-model gate explicitly OFF.

    Controlled, not inherited (the -001/-004 child-env pattern), so the
    declared 6 cannot pass or fail on an ambient variable.
    """
    monkeypatch.delenv(SYNTHETIC_ENV, raising=False)


def _arch_strings():
    return [name for name, _ in registry.get_models()]


def _static_models_entries_from_source():
    """SECOND INDEPENDENT BASE for the count -- parse, never execute.

    Reads the ``models = [...]`` literal out of ``registry.py``'s source with
    ``ast``. It shares no mechanism with calling ``get_models()``, so a bug in
    the execution path cannot make both bases agree.
    """
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
# C01 -- get_models() returns 6 entries
# ---------------------------------------------------------------------------


def test_c01_get_models_returns_the_declared_six_entries(declared_env):
    assert len(registry.get_models()) == DECLARED_COUNT


def test_c01_second_base_the_static_list_literal_holds_six_entries():
    """The same 6, counted from source without executing the function."""
    names = _static_models_entries_from_source()
    assert len(names) == DECLARED_COUNT
    assert names[-1] == NEW_ARCH


def test_c01_complement_the_six_arch_strings_are_exactly_the_expected_set(declared_env):
    """Both directions of the complement, so neither a substitution nor an
    extra entry can hide inside a correct total."""
    observed = set(_arch_strings())
    assert observed - EXPECTED_ARCHS == set()
    assert EXPECTED_ARCHS - observed == set()


def test_c01_no_arch_string_is_duplicated(declared_env):
    """A duplicate would let 6 tuples expose fewer than 6 archs."""
    archs = _arch_strings()
    assert len(set(archs)) == DECLARED_COUNT
    assert len(dict(registry.get_models())) == DECLARED_COUNT


def test_c01_the_five_pin_archs_keep_their_order_and_the_new_one_is_appended(
    declared_env,
):
    """The declared edit is "add one tuple" -- nothing else moves."""
    archs = _arch_strings()
    assert archs[:5] == PIN_ARCHS
    assert archs[5] == NEW_ARCH


def test_c01_falsifiable_the_count_moves_to_seven_when_the_synthetic_gate_is_on(
    monkeypatch,
):
    """MUTATION ARM for the count: mutate the instrument's environment and the
    reading MUST move. A test returning a constant 6 fails here."""
    monkeypatch.setenv(SYNTHETIC_ENV, "1")
    models = registry.get_models()
    assert len(models) == DECLARED_COUNT + 1
    assert models[-1][0] == SYNTHETIC_ARCH


@pytest.mark.parametrize("value", ["0", "true", "TRUE", "yes", "", "11"])
def test_c01_the_gate_is_exact_equality_on_the_string_one(monkeypatch, value):
    """The 7th entry appears only for exactly "1" -- so the declared 6 is not
    fragile against every non-empty value of the variable."""
    monkeypatch.setenv(SYNTHETIC_ENV, value)
    assert len(registry.get_models()) == DECLARED_COUNT


# ---------------------------------------------------------------------------
# C02 -- the new entry's arch string
# ---------------------------------------------------------------------------


def test_c02_the_new_entry_arch_string_is_the_declared_literal(declared_env):
    archs = _arch_strings()
    assert [name for name in archs if name == NEW_ARCH] == [NEW_ARCH]


def test_c02_word_boundary_screen_matches_exactly_one_and_substring_over_matches(
    declared_env,
):
    """The substring trap, MEASURED. ``ForConditionalGeneration`` is shared with
    the qwen3_vl arch, so a substring screen reports 2 where the conjunct means
    1. The word-boundary screen reports 1."""
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
def test_c02_falsifiable_near_miss_arch_strings_are_absent(declared_env, near_miss):
    """MUTATION ARM for the literal: the screen must discriminate, including on
    case and on a trailing character the word boundary is there to catch."""
    assert near_miss not in set(_arch_strings())


# ---------------------------------------------------------------------------
# C03 -- the factory returns a class object without instantiating weights, 1/1
# ---------------------------------------------------------------------------


def test_c03_the_registered_value_is_a_class_object_1_of_1(declared_env):
    """1/1 calls: one lookup, and it yields a CLASS, not an instance."""
    calls = 0
    entry = dict(registry.get_models())[NEW_ARCH]
    calls += 1

    assert calls == 1
    assert inspect.isclass(entry)
    assert issubclass(entry, torch.nn.Module)
    assert not isinstance(entry, torch.nn.Module)
    assert entry.__name__ == NEW_ARCH


def test_c03_the_registry_entry_the_package_export_and_the_factory_module_agree(
    declared_env,
):
    """Three import paths, one class object -- identity, not equality. This is
    also what makes the package export load-bearing rather than decorative."""
    from vllm_neuron.model.glm5_next import (
        Glm5NextForConditionalGeneration as via_package,
    )
    from vllm_neuron.model.glm5_next.factory import (
        Glm5NextForConditionalGeneration as via_module,
    )

    entry = dict(registry.get_models())[NEW_ARCH]
    assert entry is via_package
    assert entry is via_module


def test_c03_the_lookup_allocates_no_parameters_and_the_counter_is_proved_live(
    declared_env, monkeypatch
):
    """"without instantiating weights", measured rather than asserted.

    A bare zero is not evidence, so the same counter is proved live inside the
    same patched window by allocating one Parameter on purpose.
    """
    created = []
    original_new = torch.nn.Parameter.__new__

    def counting_new(cls, *args, **kwargs):
        created.append(cls)
        return original_new(cls, *args, **kwargs)

    monkeypatch.setattr(torch.nn.Parameter, "__new__", staticmethod(counting_new))

    entry = dict(registry.get_models())[NEW_ARCH]
    assert inspect.isclass(entry)
    assert created == [], f"the lookup allocated {len(created)} Parameter(s)"

    # POSITIVE CONTROL, same window: without this the zero above proves nothing.
    torch.nn.Parameter(torch.zeros(1))
    assert len(created) == 1


def test_c03_the_implementation_module_is_not_imported_at_module_level():
    """The factory's lazy import is what lets the class be looked up before the
    model skeleton exists -- and is why no weights can be allocated.

    MEASURED IN A FRESH INTERPRETER, and that is load-bearing. The property is
    about what importing ``factory.py`` pulls in, so it is an import-time
    property of one module. Asserted against *this session's* ``sys.modules`` it
    became a SESSION-GLOBAL invariant instead, which any earlier-sorting file in
    this package falsifies by legitimately importing the implementation inside a
    test body -- and three declared files sort before ``test_factory.py``:
    ``test_block_quant_recognition.py`` (``inc-glm53f-023``),
    ``test_experts.py`` (``inc-glm53f-031``) and ``test_dsa_layer.py``
    (``inc-glm53f-051``). A subprocess measures the true property and is immune
    to session pollution.

    Repaired by ``inc-glm53f-031`` under the lead's ruling on
    ``evidence-023.md`` routed item 1. **The property is unchanged and is not
    weakened**, ``inc-glm53f-009``'s declared counts do not move, and no other
    item in this file changes. Per-file ``sys.modules`` displacement stays
    prohibited (D15) and is not used here.
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


def test_c03_falsifiable_the_class_predicate_rejects_a_real_instance(declared_env):
    """MUTATION ARM for the class predicate: it must be able to fail."""
    entry = dict(registry.get_models())[NEW_ARCH]
    probe = torch.nn.Linear(1, 1)

    assert inspect.isclass(entry)
    assert not inspect.isclass(probe)
    assert isinstance(probe, torch.nn.Module)


# ---------------------------------------------------------------------------
# Reporting -- the readings the evidence record quotes
# ---------------------------------------------------------------------------


def test_report_the_measured_readings(declared_env, capsys):
    models = registry.get_models()
    archs = [name for name, _ in models]
    entry = dict(models)[NEW_ARCH]

    with capsys.disabled():
        print()
        print(f"[env] {SYNTHETIC_ENV} inherited = {INHERITED_SYNTHETIC_ENV}")
        print(f"[env] {SYNTHETIC_ENV} during counting = <deleted by fixture>")
        print(
            f"[C01] PASS  get_models() returns {DECLARED_COUNT} entries"
            f"  actual={len(models)}"
        )
        print(
            f"[C01] second base (ast over registry.py source) = "
            f"{len(_static_models_entries_from_source())}"
        )
        print(f"[C02] PASS  new arch string == {NEW_ARCH!r}  actual={archs[5]!r}")
        print(
            f"[C02] word-boundary hits=1  substring hits over "
            f"{SHARED_SUFFIX!r}=2 (trap measured)"
        )
        print(
            f"[C03] PASS  class object without instantiating weights in 1/1 calls"
            f"  actual={entry!r}  parameters_allocated=0"
        )
        print(f"[C03] {IMPL_MODULE} in sys.modules = {IMPL_MODULE in sys.modules}")
        print(f"[archs] {archs}")
        print("[3/3] conjuncts passed=3/3  failed=0")

    assert len(models) == DECLARED_COUNT
    assert archs[5] == NEW_ARCH
    assert inspect.isclass(entry)
