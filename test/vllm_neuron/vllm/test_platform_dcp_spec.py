# SPDX-License-Identifier: Apache-2.0
"""Speculative decoding and the shadow draft under decode context parallelism: the admission.

``NeuronPlatform._refuse_speculative_decoding_under_dcp`` runs at
``decode_context_parallel_size > 1`` and admits two forms only when the paths they need are in
the tree, each read from its module's source, without importing the module, as a module-level
constant set to ``True``:

* the GLM-5.3-Flash shadow draft (``VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT >= 1``, no speculative
  config) on ``mtp.DCP_DRAFT_SLOTS``: the draft iterations past the first take their latent
  slot from the DCP ownership map;
* a speculative config (method "mtp", GLM-5.3-Flash) on the conjunction of four: the T-row
  DCP partial-attention entry (``mla_decode.DCP_SPECULATIVE_ROWS``), the DSA indexer's T rows
  under DCP (``decode_trow.DCP_SPECULATIVE_ROWS``), the draft head's DCP ownership slots
  (``mtp.DCP_DRAFT_SLOTS``, which its draft iterations past the first take) and, with
  ``VLLM_NEURON_GLM5NEXT_MTP_ASYNC`` set, the async step's DCP latent slots
  (``async_step.DCP_LATENT_SLOTS``).

Where a form is not admitted it is refused with today's message, byte for byte, then a
sentence naming what is missing and the file read. Every admission test points the reader at
a source file of its own for every capability it reads (``_sources``), so the result does
not depend on which members the tree holds; the tests of the tree read the tree. The reader
itself (``NeuronPlatform._missing_capability``) is pinned on source texts and package
layouts, and with the probed modules blocked from import. No test builds an engine or
touches a device.
"""

from __future__ import annotations

import importlib
import py_compile
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import vllm_neuron
from vllm_neuron.vllm.platform import NeuronPlatform

pytestmark = [pytest.mark.fast]

KNOB = "VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT"
ASYNC_KNOB = "VLLM_NEURON_GLM5NEXT_MTP_ASYNC"
GLM5_NEXT_ARCH = "Glm5NextForConditionalGeneration"
DCP_SIZES = (2, 4, 8)

DRAFT_SLOTS = ("vllm_neuron.model.glm5_next.mtp", "DCP_DRAFT_SLOTS")
ATTENTION_ROWS = ("vllm_neuron.functional.attention.mla_decode", "DCP_SPECULATIVE_ROWS")
INDEXER_ROWS = ("vllm_neuron.functional.dsa.decode_trow", "DCP_SPECULATIVE_ROWS")
ASYNC_SLOTS = ("vllm_neuron.functional.mtp.async_step", "DCP_LATENT_SLOTS")
SPECULATIVE = (ATTENTION_ROWS, INDEXER_ROWS, DRAFT_SLOTS, ASYNC_SLOTS)


def shadow_message(k: int, dcp: int) -> str:
    """The shadow-draft refusal as b5b516b3 words it (platform.py :1302-:1310)."""
    return (
        f"VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT={k} is not supported with "
        f"decode_context_parallel_size={dcp}: from its second draft iteration on, the "
        f"shadow draft computes the MTP layer's KV slots from the sequence's page index, "
        f"and a decode context parallel rank holds only its own pages. Unset "
        f"VLLM_NEURON_GLM5NEXT_SHADOW_DRAFT or set decode_context_parallel_size=1."
    )


def speculative_message(k: int, dcp: int) -> str:
    """The speculative-config refusal as b5b516b3 words it (platform.py :1316-:1324)."""
    return (
        f"Speculative decoding is not supported with decode_context_parallel_size > 1 "
        f"(dcp={dcp}, num_speculative_tokens={k}): spec decode under DCP (item 5a T > 1 "
        f"follow-up) is not implemented. Disable speculative decoding or set "
        f"decode_context_parallel_size=1."
    )


def _config(dcp: int, k: int | None = None, *, method: str = "mtp",
            architecture: str = GLM5_NEXT_ARCH) -> SimpleNamespace:
    """The fields the refusal reads: the DCP size, the speculative config, the architecture."""
    speculative = None if k is None else SimpleNamespace(method=method, num_speculative_tokens=k)
    return SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=dcp),
        speculative_config=speculative,
        model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=[architecture])),
    )


def _sources(monkeypatch, tmp_path: Path, texts: dict[str, str | bytes | None]) -> dict[str, Path]:
    """Point the capability reader at a file under ``tmp_path`` for each module in
    ``texts``, holding that text (no file when it is None); every other module resolves as
    before. Returns the files by module."""
    root = tmp_path / "capability-sources"
    root.mkdir(parents=True, exist_ok=True)
    files = {module: root / f"{module}.py" for module in texts}
    for module, text in texts.items():
        if isinstance(text, bytes):
            files[module].write_bytes(text)
        elif text is not None:
            files[module].write_text(text)
    earlier = NeuronPlatform._capability_source

    def resolve(module: str) -> str | None:
        return str(files[module]) if module in files else earlier(module)

    monkeypatch.setattr(NeuronPlatform, "_capability_source", staticmethod(resolve))
    return files


def _set(monkeypatch, tmp_path: Path, capability: tuple[str, str], present: bool) -> None:
    _sources(monkeypatch, tmp_path, {capability[0]: f"{capability[1]} = {present}\n"})


def _speculative_tree(monkeypatch, tmp_path: Path, *, attention: bool, indexer: bool,
                      draft: bool, slots: bool) -> None:
    present = zip(SPECULATIVE, (attention, indexer, draft, slots), strict=True)
    _sources(monkeypatch, tmp_path,
             {module: f"{attribute} = {value}\n" for (module, attribute), value in present})


def _label(capability: tuple[str, str]) -> str:
    return f"{capability[0]}.{capability[1]}"


# --------------------------------------------------------------------------- #
# the shadow draft.


@pytest.mark.parametrize("draft_k", [1, 3])
@pytest.mark.parametrize("dcp", DCP_SIZES)
def test_the_shadow_draft_is_admitted_when_its_slots_follow_ownership(
    monkeypatch, tmp_path, dcp, draft_k
):
    monkeypatch.setenv(KNOB, str(draft_k))
    _set(monkeypatch, tmp_path, DRAFT_SLOTS, True)
    assert NeuronPlatform._refuse_speculative_decoding_under_dcp(_config(dcp)) is None


@pytest.mark.parametrize(
    "text",
    ["DCP_DRAFT_SLOTS = False\n", "DCP_DRAFT_SLOTS = 1\n", "DCP_DRAFT_SLOTS = 'True'\n",
     "DCP_DRAFT_SLOTS = bool(1)\n", ""],
    ids=["False", "1", "str-True", "bool-call", "absent"],
)
def test_only_a_literal_true_draft_slot_constant_admits_the_shadow_draft(
    monkeypatch, tmp_path, text
):
    monkeypatch.setenv(KNOB, "3")
    files = _sources(monkeypatch, tmp_path, {DRAFT_SLOTS[0]: text})
    with pytest.raises(ValueError) as refused:
        NeuronPlatform._refuse_speculative_decoding_under_dcp(_config(2))
    message = str(refused.value)
    assert message.startswith(shadow_message(3, 2))
    assert "mtp.DCP_DRAFT_SLOTS" in message and str(files[DRAFT_SLOTS[0]]) in message


def test_the_tree_publishes_the_draft_slot_attribute() -> None:
    module = importlib.import_module(DRAFT_SLOTS[0])
    assert getattr(module, DRAFT_SLOTS[1]) is True


# --------------------------------------------------------------------------- #
# a speculative config: the capability conjunction.


@pytest.mark.parametrize("async_knob", [None, "1"])
@pytest.mark.parametrize("dcp", DCP_SIZES)
def test_speculative_mtp_is_admitted_when_every_capability_is_present(
    monkeypatch, tmp_path, dcp, async_knob
):
    if async_knob is None:
        monkeypatch.delenv(ASYNC_KNOB, raising=False)
    else:
        monkeypatch.setenv(ASYNC_KNOB, async_knob)
    _speculative_tree(monkeypatch, tmp_path, attention=True, indexer=True, draft=True,
                      slots=True)
    assert NeuronPlatform._refuse_speculative_decoding_under_dcp(_config(dcp, 3)) is None


def test_the_async_slots_are_not_needed_without_the_async_knob(monkeypatch, tmp_path):
    monkeypatch.delenv(ASYNC_KNOB, raising=False)
    _speculative_tree(monkeypatch, tmp_path, attention=True, indexer=True, draft=True,
                      slots=False)
    assert NeuronPlatform._refuse_speculative_decoding_under_dcp(_config(2, 3)) is None


@pytest.mark.parametrize(
    "attention, indexer, draft, slots",
    [
        (False, False, False, False),
        (True, False, True, False),
        (False, True, True, True),
        (True, True, True, False),
        (True, False, True, True),
        (True, True, False, True),
        (False, False, False, True),
    ],
)
def test_a_missing_capability_is_refused_with_todays_message_and_its_name(
    monkeypatch, tmp_path, attention, indexer, draft, slots
):
    monkeypatch.setenv(ASYNC_KNOB, "1")
    _speculative_tree(monkeypatch, tmp_path, attention=attention, indexer=indexer,
                      draft=draft, slots=slots)
    with pytest.raises(ValueError) as refused:
        NeuronPlatform._refuse_speculative_decoding_under_dcp(_config(2, 3))
    message = str(refused.value)
    assert message.startswith(speculative_message(3, 2))
    for capability, present in zip(SPECULATIVE, (attention, indexer, draft, slots),
                                   strict=True):
        assert (_label(capability) in message) is (not present), (capability, message)


@pytest.mark.parametrize("async_knob", [None, "1"])
def test_the_draft_slots_are_needed_with_and_without_the_async_knob(
    monkeypatch, tmp_path, async_knob
):
    """The verify step's draft iterations past the first take their latent slots from the
    draft head's DCP ownership path, so the speculative form names it when it is absent."""
    if async_knob is None:
        monkeypatch.delenv(ASYNC_KNOB, raising=False)
    else:
        monkeypatch.setenv(ASYNC_KNOB, async_knob)
    _speculative_tree(monkeypatch, tmp_path, attention=True, indexer=True, draft=False,
                      slots=True)
    with pytest.raises(ValueError) as refused:
        NeuronPlatform._refuse_speculative_decoding_under_dcp(_config(2, 3))
    message = str(refused.value)
    assert message.startswith(speculative_message(3, 2))
    assert [capability for capability in SPECULATIVE if _label(capability) in message] == [
        DRAFT_SLOTS]


def test_the_tree_alone_names_what_it_lacks(monkeypatch):
    """No capability is set here: the refusal reads the tree as it is, and names exactly
    the members' attributes that are not ``True`` in it."""
    monkeypatch.setenv(ASYNC_KNOB, "1")
    absent = [
        capability for capability in SPECULATIVE
        if getattr(importlib.import_module(capability[0]), capability[1], None) is not True
    ]
    if not absent:
        assert NeuronPlatform._refuse_speculative_decoding_under_dcp(_config(2, 3)) is None
        return
    with pytest.raises(ValueError) as refused:
        NeuronPlatform._refuse_speculative_decoding_under_dcp(_config(2, 3))
    for capability in SPECULATIVE:
        assert (_label(capability) in str(refused.value)) is (capability in absent)


@pytest.mark.parametrize(
    "method, architecture",
    [("eagle3", GLM5_NEXT_ARCH), ("mtp", "LlamaForCausalLM"), ("mtp", "DeepseekV3ForCausalLM")],
)
def test_other_methods_and_architectures_stay_refused(
    monkeypatch, tmp_path, method, architecture
):
    monkeypatch.setenv(ASYNC_KNOB, "1")
    _speculative_tree(monkeypatch, tmp_path, attention=True, indexer=True, draft=True,
                      slots=True)
    config = _config(2, 3, method=method, architecture=architecture)
    with pytest.raises(ValueError) as refused:
        NeuronPlatform._refuse_speculative_decoding_under_dcp(config)
    message = str(refused.value)
    assert message.startswith(speculative_message(3, 2))
    assert f"'{method}'" in message and architecture in message


def test_no_speculative_tokens_is_not_a_speculative_step(monkeypatch, tmp_path):
    _speculative_tree(monkeypatch, tmp_path, attention=False, indexer=False, draft=False,
                      slots=False)
    assert NeuronPlatform._refuse_speculative_decoding_under_dcp(_config(2, 0)) is None


# --------------------------------------------------------------------------- #
# the capability reader: the source text, not an import.

PROBE = "capability_probe_case"
FLAG = "CAPABILITY"
#: (case, source text, whether it vouches): the last module-scope binding wins, and only a
#: literal ``True`` assigned at top level vouches.
READER_CASES = [
    ("true", "CAPABILITY = True\n", True),
    ("annotated", "CAPABILITY: bool = True\n", True),
    ("chained", "OTHER = CAPABILITY = True\n", True),
    ("parenthesized", "CAPABILITY = (True)\n", True),
    ("false-then-true", "CAPABILITY = False\nCAPABILITY = True\n", True),
    ("function-local", "CAPABILITY = True\ndef f():\n    CAPABILITY = False\n", True),
    ("class-local", "CAPABILITY = True\nclass C:\n    CAPABILITY = False\n", True),
    ("lambda-argument", "CAPABILITY = True\nf = lambda CAPABILITY=False: CAPABILITY\n", True),
    ("comprehension-variable", "CAPABILITY = True\n[CAPABILITY for CAPABILITY in ()]\n", True),
    ("item-store", "CAPABILITY = True\nCAPABILITY.x = 1\nCAPABILITY[0] = 1\n", True),
    ("annotation-only-after", "CAPABILITY = True\nCAPABILITY: bool\n", True),
    ("latin-1-cookie", "# -*- coding: latin-1 -*-\nCAPABILITY = True\n".encode("latin-1"), True),
    ("absent", "OTHER = True\n", False),
    ("annotation-only", "CAPABILITY: bool\n", False),
    ("false", "CAPABILITY = False\n", False),
    ("one", "CAPABILITY = 1\n", False),
    ("str-true", "CAPABILITY = 'True'\n", False),
    ("bool-call", "CAPABILITY = bool(1)\n", False),
    ("not-false", "CAPABILITY = not False\n", False),
    ("tuple", "CAPABILITY, OTHER = True, True\n", False),
    ("true-then-false", "CAPABILITY = True\nCAPABILITY = False\n", False),
    ("deleted", "CAPABILITY = True\ndel CAPABILITY\n", False),
    ("augmented", "CAPABILITY = True\nCAPABILITY += 1\n", False),
    ("conditional-last", "CAPABILITY = True\nif FLAG:\n    CAPABILITY = False\n", False),
    ("conditional-only", "if FLAG:\n    CAPABILITY = True\n", False),
    ("in-try", "CAPABILITY = True\ntry:\n    pass\nexcept OSError:\n    CAPABILITY = False\n",
     False),
    ("for-target", "CAPABILITY = True\nfor CAPABILITY in ():\n    pass\n", False),
    ("with-target", "CAPABILITY = True\nwith f() as CAPABILITY:\n    pass\n", False),
    ("walrus", "CAPABILITY = True\n(CAPABILITY := 0)\n", False),
    ("walrus-in-default", "CAPABILITY = True\ndef f(a=(CAPABILITY := 0)):\n    pass\n", False),
    ("import-as", "CAPABILITY = True\nimport os as CAPABILITY\n", False),
    ("from-import", "CAPABILITY = True\nfrom os import CAPABILITY\n", False),
    ("star-import", "CAPABILITY = True\nfrom os import *\n", False),
    ("def", "CAPABILITY = True\ndef CAPABILITY():\n    pass\n", False),
    ("class", "CAPABILITY = True\nclass CAPABILITY:\n    pass\n", False),
    ("except-as", "CAPABILITY = True\ntry:\n    pass\nexcept OSError as CAPABILITY:\n    pass\n",
     False),
    ("match-capture", "CAPABILITY = True\nmatch v:\n    case [CAPABILITY]:\n        pass\n", False),
    ("match-rest", "CAPABILITY = True\nmatch v:\n    case {**CAPABILITY}:\n        pass\n", False),
    ("syntax-error", "CAPABILITY = True\nCAPABILITY = (\n", False),
    ("null-byte", "CAPABILITY = True\n\x00\n", False),
]


@pytest.mark.parametrize("text, vouches", [case[1:] for case in READER_CASES],
                         ids=[case[0] for case in READER_CASES])
def test_only_a_last_module_level_literal_true_vouches(monkeypatch, tmp_path, text, vouches):
    _sources(monkeypatch, tmp_path, {PROBE: text})
    assert (NeuronPlatform._missing_capability(PROBE, FLAG) is None) is vouches


@pytest.mark.parametrize(
    "text, reason",
    [
        (None, "{file} cannot be read (No such file or directory)"),
        ("CAPABILITY = (\n", "{file} does not parse ("),
        ("OTHER = True\n", "{file} does not set CAPABILITY = True at module level"),
        ("CAPABILITY = True\nCAPABILITY = False\n",
         "{file} binds CAPABILITY other than to True at module level (line 2)"),
    ],
    ids=["missing-file", "syntax-error", "absent", "other-binding"],
)
def test_each_refusal_reason_names_the_file_read(monkeypatch, tmp_path, text, reason):
    files = _sources(monkeypatch, tmp_path, {PROBE: text})
    assert NeuronPlatform._missing_capability(PROBE, FLAG).startswith(
        reason.format(file=files[PROBE]))


def test_a_module_with_no_source_on_the_path_is_named(monkeypatch):
    monkeypatch.setattr(NeuronPlatform, "_capability_source", staticmethod(lambda module: None))
    assert NeuronPlatform._missing_capability(PROBE, FLAG) == (
        f"no Python source file for {PROBE} on the import path")


@pytest.mark.parametrize("capability", SPECULATIVE, ids=_label)
def test_each_capability_resolves_to_the_trees_own_source(capability):
    module = capability[0]
    want = Path(vllm_neuron.__file__).parent.joinpath(*module.split(".")[1:]).with_suffix(".py")
    assert Path(NeuronPlatform._capability_source(module)) == want


@pytest.mark.parametrize("capability", SPECULATIVE, ids=_label)
def test_the_reader_agrees_with_an_import_of_the_tree(capability):
    imported = getattr(importlib.import_module(capability[0]), capability[1], None) is True
    assert (NeuronPlatform._missing_capability(*capability) is None) is imported


def test_resolving_a_module_runs_no_package_init(monkeypatch, tmp_path):
    package = tmp_path / "probe_pkg_init_raises"
    (package / "inner").mkdir(parents=True)
    for init in (package / "__init__.py", package / "inner" / "__init__.py"):
        init.write_text("raise RuntimeError('a package init ran')\n")
    leaf = package / "inner" / "leaf.py"
    leaf.write_text("raise RuntimeError('the module ran')\nCAPABILITY = True\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    name = "probe_pkg_init_raises.inner.leaf"
    assert NeuronPlatform._capability_source(name) == str(leaf)
    assert NeuronPlatform._missing_capability(name, FLAG) is None
    assert not [module for module in sys.modules if module.startswith("probe_pkg_init_raises")]


def test_a_namespace_package_resolves(monkeypatch, tmp_path):
    (tmp_path / "probe_namespace").mkdir()
    leaf = tmp_path / "probe_namespace" / "leaf.py"
    leaf.write_text("CAPABILITY = True\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert NeuronPlatform._capability_source("probe_namespace.leaf") == str(leaf)
    assert "probe_namespace" not in sys.modules


def test_an_imported_package_resolves_through_its_own_path(monkeypatch, tmp_path):
    """A package already imported is searched where its ``__path__`` says, as an import
    would, not where the import path would find it now."""
    (tmp_path / "elsewhere").mkdir()
    leaf = tmp_path / "elsewhere" / "leaf.py"
    leaf.write_text("CAPABILITY = True\n")
    package = ModuleType("probe_imported_package")
    package.__path__ = [str(tmp_path / "elsewhere")]
    monkeypatch.setitem(sys.modules, "probe_imported_package", package)
    assert NeuronPlatform._capability_source("probe_imported_package.leaf") == str(leaf)


@pytest.mark.parametrize("layout", ["sourceless", "module-as-package", "no-package", "no-leaf"])
def test_a_module_without_a_source_file_of_its_own_does_not_resolve(
    monkeypatch, tmp_path, layout
):
    package = tmp_path / "probe_layout"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (tmp_path / "probe_plain_module.py").write_text("CAPABILITY = True\n")
    source = package / "compiled.py"
    source.write_text("CAPABILITY = True\n")
    py_compile.compile(str(source), cfile=str(package / "compiled.pyc"), doraise=True)
    source.unlink()
    monkeypatch.syspath_prepend(str(tmp_path))
    name = {
        "sourceless": "probe_layout.compiled",
        "module-as-package": "probe_plain_module.leaf",
        "no-package": "probe_no_such_package.leaf",
        "no-leaf": "probe_layout.no_such_leaf",
    }[layout]
    assert NeuronPlatform._capability_source(name) is None
    assert NeuronPlatform._missing_capability(name, FLAG) == (
        f"no Python source file for {name} on the import path")


def test_the_admission_does_not_depend_on_importing_the_probed_modules(monkeypatch):
    """Ruling (ib) row (d): with each probed module blocked from import (None in
    ``sys.modules``), both forms get the verdict they get unblocked; this tree vouches
    for all four, so both are admitted."""
    monkeypatch.setenv(ASYNC_KNOB, "1")
    monkeypatch.setenv(KNOB, "3")
    forms = (_config(2, 3), _config(2))

    def verdicts() -> list[str | None]:
        seen = []
        for config in forms:
            try:
                seen.append(NeuronPlatform._refuse_speculative_decoding_under_dcp(config))
            except ValueError as refused:
                seen.append(str(refused))
        return seen

    unblocked = verdicts()
    for module, _attribute in SPECULATIVE:
        monkeypatch.setitem(sys.modules, module, None)
    assert verdicts() == unblocked == [None, None]


# --------------------------------------------------------------------------- #
# the message phrase the startup log is searched for.


def test_each_refusal_holds_its_phrase_on_one_source_line() -> None:
    """``grep 'is not supported with decode_context_parallel_size' platform.py`` finds both
    messages: the phrase sits in one string literal in each."""
    import inspect

    source = inspect.getsource(NeuronPlatform._refuse_speculative_decoding_under_dcp)
    lines = [line for line in source.splitlines()
             if "is not supported with decode_context_parallel_size" in line]
    assert len(lines) == 2, lines
