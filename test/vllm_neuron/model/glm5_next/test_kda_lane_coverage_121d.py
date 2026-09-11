# SPDX-License-Identifier: Apache-2.0
"""Three readings the KDA lane was missing: its allocations, its stack dial, its reader.

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 python -m pytest -s -rA \\
      test/vllm_neuron/model/glm5_next/test_kda_lane_coverage_121d.py

A01 censuses every module of ``vllm_neuron/functional/kda/`` with a factory set the package
itself supplies: every ``torch.<name>(`` its modules call, kept when that name accepts a
``device=``. Which place answered that question for which name is a row of its own, and a set
that keeps nothing fails the item, because a census with an empty set reads every module as
empty and reports a hollow zero offenders on any tree at all. Nothing is exempt for the name
of the function around it, because no class here
is an ``nn.Module`` and so nothing carries a later ``to(device)``. The walk is armed twice: once
off the derived set, and once off a set this file spells out, because an arming that reads only
what the predicate kept says nothing when the predicate keeps the wrong names. A01 reports the driven
model file as well, through the LANDED census, whose narrower scope is named on its own row
and whose call total is a reading rather than a pin. A01B is the reading the product repair
answers: no allocation in that package decides a device for itself.

A02 reads the tiny stack's layer schedule as a dial. The schedule was a fixed argument beside
``**overrides``, so a caller asking for another shape got a duplicate keyword and a
``TypeError``. The default is unchanged, and that is the reading: every landed item still gets
the sparse-attention stack it got before.

A03 reads the meta-forward's device-mismatch reporter. It keyed on one wording, while the
fake-tensor propagation and the eager check say two others, so a mismatch in either arrived
unlabelled. All three read as one, each armed, with the old needle's own miss beside them.

With ``GLM53F_121D_EXPECT_BASE=1`` A01B asserts the PRE-REPAIR reading of the tree it is
copied into: the one allocation that names no device, keyed by module, function and factory.
A01, A02 and A03 read the same on both trees, because this lap changes one product line and
neither the dial nor the reporter.
"""

from __future__ import annotations

import ast
import inspect
import os
import pathlib

import pytest
import torch

from vllm_neuron.functional import kda as kda_package

from test.vllm_neuron.model.glm5_next import test_device_follows_activation_121 as census
from test.vllm_neuron.model.glm5_next import test_meta_forward_119 as meta
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_forward as tiny

pytestmark = [pytest.mark.fast, pytest.mark.forked]

EXPECT_BASE_VAR = "GLM53F_121D_EXPECT_BASE"

#: The package this file censuses, resolved off the imported module rather than typed.
KDA_PACKAGE = pathlib.Path(kda_package.__file__).parent

#: The allocation the pre-repair tree carries, keyed by module, enclosing function and factory.
#: NOT by line: a line moves whenever anything above it does, and this one already drifted nine
#: lines between the tree that recorded it and the tree that repaired it.
RECORDED_ALLOCATION = ("gate_clamp.py", "kda_gate_clamp", "zeros")

#: What the landed census reads, said where its numbers are printed rather than left implied.
MODEL_SCOPE = "landed_walk:six_typed_factories,init_exempt,this_file_only"

#: One device-less call per derived factory, so every name in the set is shown live.
PLANTED = "import torch\n\n\ndef f(rows):\n{calls}\n"

#: A SECOND ARMING, OFF A SET SPELLED HERE RATHER THAN DERIVED. The arming above shows the walk
#: working on whatever the predicate kept, so a predicate that keeps the wrong names arms a wrong
#: walk and says nothing. These five calls and their three expected offenders do not move with the
#: predicate, the installed torch or the package: three name no device, one copies the device of
#: what it reads, and one names a device and so is not an offender.
FIXED_SET = ("arange", "empty", "empty_like", "zeros")
FIXED_CALLS = (
    "    torch.zeros(rows)",
    "    torch.empty(rows)",
    "    torch.arange(rows)",
    "    torch.empty_like(rows)",
    "    torch.zeros(rows, device=rows.device)",
)
FIXED_OFFENDERS = ["zeros", "empty", "arange"]

#: The wordings a device mismatch arrives in on this path.
FAKE_TENSOR_WORDING = (
    "Unhandled FakeTensor Device Propagation for aten.mm.default, found two different "
    "devices cpu, meta"
)
EAGER_WORDING = (
    "Expected all tensors to be on the same device, but found at least two devices, "
    "meta and cpu!"
)
OP_WORDING = "expected device meta but got device cpu"

#: A failure that is not a device mismatch, so the predicate is read in both directions.
NOT_A_MISMATCH = (
    "error: failed to specialize NKI kernel: Collected 1 different diagnostics: "
    "- [x1] error: unsupported expression"
)

#: What the reporter keyed on before, and what keying on it alone missed.
OLD_NEEDLE = "expected device"


def _expect_base() -> bool:
    return os.environ.get(EXPECT_BASE_VAR) == "1"


def _require_cpu_lane() -> None:
    """Both flags come from the process environment; the seams read them at import."""
    missing = [
        name
        for name in ("VLLM_NEURON_CPU_MODE", "NKI_SIMULATOR")
        if os.environ.get(name) != "1"
    ]
    if missing:
        raise RuntimeError(f"{missing} must be 1 in the process environment")


def _emit(tag: str, **values: object) -> None:
    """One row per reading, pipe-separated, prefix first."""
    fields = "|".join(f"{key}={value}" for key, value in values.items())
    print(f"KDA121D|{tag}|{fields}", flush=True)


def _model_file() -> pathlib.Path:
    """The driven model file, the same way the landed census resolves it."""
    from vllm_neuron.model.glm5_next import model_fp8

    return pathlib.Path(os.environ.get(census.MODEL_FILE_VAR) or model_fp8.__file__)


def _package_modules() -> list[pathlib.Path]:
    """Every module of the package, any sub-directory included."""
    return sorted(KDA_PACKAGE.rglob("*.py"))


def _torch_calls(tree: ast.AST):
    """Each ``torch.<name>(...)`` call as ``(name, enclosing def, keyword names)``."""

    def walk(node: ast.AST, where: str):
        for child in ast.iter_child_nodes(node):
            func = getattr(child, "func", None)
            if isinstance(func, ast.Attribute) and getattr(func.value, "id", None) == "torch":
                yield func.attr, where, {keyword.arg for keyword in child.keywords}
            inner = isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            yield from walk(child, child.name if inner else where)

    yield from walk(tree, "<module>")


def _device_sources(name: str) -> list[str]:
    """Which places say ``torch.<name>`` accepts a ``device=``, so a call omitting it picks one.

    THREE PLACES, AND NO ONE OF THEM ANSWERS FOR EVERY NAME. A Python-level function has a
    signature the inspect module can read. A C-level one may answer that call with a bare
    ``(*args, **kwargs)`` instead of refusing it, and then carries its real signature in
    ``__text_signature__`` or on the first NON-BLANK line of its docstring -- torch writes
    that docstring with a leading newline, so the first line itself is empty.
    """
    function = getattr(torch, name, None)
    if function is None:
        return []
    reads = []
    try:
        if "device" in inspect.signature(function).parameters:
            reads.append("signature")
    except (TypeError, ValueError):
        pass
    if "device=" in (getattr(function, "__text_signature__", None) or ""):
        reads.append("text_signature")
    doc = getattr(function, "__doc__", None) or ""
    if "device=" in next((line for line in doc.splitlines() if line.strip()), ""):
        reads.append("docstring")
    return reads


def _factory_set(modules: list[pathlib.Path]) -> tuple[list[str], list[str]]:
    """``(the names the package calls, those that take a device)``, off its own parse trees."""
    called = sorted(
        {
            name
            for path in modules
            for name, _where, _keywords in _torch_calls(ast.parse(path.read_text()))
        }
    )
    return called, [name for name in called if _device_sources(name)]


def _census(
    path: pathlib.Path, factories: frozenset[str]
) -> tuple[int, list[tuple[str, str, str]]]:
    """``(calls read, offenders)``; an offender is ``(module, enclosing def, factory)``.

    A ``*_like`` factory takes the device of the tensor it copies, so a call that omits
    ``device=`` there has named one after all. Every other factory falls back to the default
    device, which under graph capture is not where the activations live.
    """
    calls, offenders = 0, []
    for factory, where, keywords in _torch_calls(ast.parse(path.read_text())):
        if factory not in factories:
            continue
        calls += 1
        if not factory.endswith("_like") and "device" not in keywords:
            offenders.append((path.name, where, factory))
    return calls, offenders


def test_a01_the_package_census_reads_every_factory_its_modules_call(tmp_path) -> None:
    """The derived set, one row per module, the walk shown finding one call per kept name."""
    _require_cpu_lane()
    modules = _package_modules()
    called, factories = _factory_set(modules)
    _emit(
        "factories",
        called=",".join(called),
        kept=",".join(factories),
        exempt=",".join(name for name in factories if name.endswith("_like")),
    )
    reads = {name: _device_sources(name) for name in called}
    _emit(
        "predicate",
        **{
            place: ",".join(name for name, places in reads.items() if place in places) or "none"
            for place in ("signature", "text_signature", "docstring")
        },
    )
    assert factories, (
        f"the device predicate kept NONE of the {len(called)} names this package calls: {called}. "
        f"Every census below would then read zero calls and every offender count would be a "
        f"hollow zero. The predicate row above says which place answered for which name"
    )
    kept = frozenset(factories)
    calls, found = 0, []
    for path in modules:
        module_calls, offenders = _census(path, kept)
        calls += module_calls
        found += offenders
        _emit("census", module=path.name, calls=module_calls, offenders=len(offenders))
    _emit(
        "census_totals",
        files=len(modules),
        calls=calls,
        offenders=len(found),
        keyed=";".join(":".join(entry) for entry in sorted(found)) or "none",
    )
    assert calls, (
        f"the census read no call at all over {len(modules)} modules with the kept set "
        f"{factories}, so its zero offenders say nothing about this package"
    )

    planted_file = tmp_path / "planted.py"
    planted_file.write_text(
        PLANTED.format(calls="\n".join(f"    torch.{name}(rows)" for name in factories))
    )
    planted_calls, planted = _census(planted_file, kept)
    _emit(
        "armed",
        planted_calls=planted_calls,
        planted_offenders=len(planted),
        planted=",".join(factory for _module, _where, factory in planted) or "none",
    )
    assert planted_calls == len(factories), (planted_calls, factories)
    assert [factory for _module, _where, factory in planted] == [
        name for name in factories if not name.endswith("_like")
    ], planted

    # AND THE SAME WALK OVER A SET THIS FILE SPELLS, so the arming above cannot be the only one.
    # It reads whatever the predicate kept, and a predicate that keeps the wrong names arms a
    # wrong walk. These five calls and their three offenders are fixed here.
    fixed_file = tmp_path / "fixed.py"
    fixed_file.write_text(PLANTED.format(calls="\n".join(FIXED_CALLS)))
    fixed_calls, fixed_found = _census(fixed_file, frozenset(FIXED_SET))
    _emit(
        "armed_fixed",
        factories=",".join(FIXED_SET),
        planted_calls=fixed_calls,
        planted_offenders=len(fixed_found),
        planted=",".join(factory for _module, _where, factory in fixed_found) or "none",
    )
    assert fixed_calls == len(FIXED_CALLS), (
        f"the walk read {fixed_calls} of the {len(FIXED_CALLS)} calls planted from a set spelled "
        f"in this file, so it is not reading every call form it is given"
    )
    assert [factory for _module, _where, factory in fixed_found] == FIXED_OFFENDERS, (
        f"over the fixed arming the walk read {fixed_found} and this file declares "
        f"{FIXED_OFFENDERS}: the copying factory and the call that names a device are not "
        f"offenders, and the other three are"
    )

    model_file = _model_file()
    model_calls, model_offenders = census._census(model_file)
    _emit(
        "model_census",
        calls=model_calls,
        offenders=len(model_offenders),
        keyed=";".join(f"{where}:{factory}" for _line, factory, where in sorted(model_offenders))
        or "none",
        scope=MODEL_SCOPE,
    )
    assert not model_offenders, (
        f"{len(model_offenders)} allocation(s) outside __init__ in {model_file.name} name no "
        f"device=: {model_offenders}. The call count beside them is a reading, not a pin, so a "
        f"new allocation that names its device is not a failure here"
    )


def test_a01b_no_allocation_in_the_kda_package_decides_a_device_for_itself() -> None:
    """Zero offenders on the repaired tree; exactly one, by shape, on the tree without it."""
    _require_cpu_lane()
    modules = _package_modules()
    called, factories = _factory_set(modules)
    kept = frozenset(factories)
    read = [_census(path, kept) for path in modules]
    calls = sum(count for count, _offenders in read)
    found = sorted(entry for _count, offenders in read for entry in offenders)
    _emit(
        "offenders",
        count=len(found),
        keyed=";".join(":".join(entry) for entry in found) or "none",
        calls_read=calls,
    )
    # THE FLOOR UNDER THE READING BELOW. A predicate that keeps no name, or a walk that finds no
    # call, gives zero offenders on any tree at all, and a zero of that kind is not the reading
    # this item is for.
    assert factories, (
        f"the device predicate kept NONE of the {len(called)} names this package calls: {called}"
    )
    assert calls, f"no call was read over the {len(modules)} modules with the kept set {factories}"
    if _expect_base():
        assert found == [RECORDED_ALLOCATION], (
            f"this arm reads the tree that still builds its zero bias column without a device, so "
            f"the walk must find exactly that one allocation, keyed by module, function and "
            f"factory: declared [{RECORDED_ALLOCATION}], read {found} over {calls} calls in "
            f"{len(modules)} modules"
        )
        return
    assert not found, (
        f"{len(found)} allocation(s) in {KDA_PACKAGE.name}/ name no device= and take the "
        f"default device, which under capture is not where the activations are: {found}"
    )


def test_a02_the_tiny_stacks_layer_schedule_is_a_dial_with_its_landed_default() -> None:
    """The default schedule is what it was, and another schedule can now be asked for."""
    _require_cpu_lane()
    from vllm_neuron.model.glm5_next.config import DSA_LAYER_TYPE, KDA_LAYER_TYPE

    default = tiny._stack_text_config()
    landed_schedule = [DSA_LAYER_TYPE] * tiny.STACK_LAYERS
    _emit(
        "schedule_default",
        layers=default.num_hidden_layers,
        types=",".join(default.layer_types),
        landed=",".join(landed_schedule),
    )
    assert list(default.layer_types) == landed_schedule
    assert default.num_hidden_layers == tiny.STACK_LAYERS

    asked = [KDA_LAYER_TYPE] * (tiny.STACK_LAYERS - 1) + [DSA_LAYER_TYPE]
    mixed = tiny._stack_text_config(layer_types=asked)
    moved = [
        name
        for name in vars(default)
        if name != "layer_types" and getattr(mixed, name, None) != getattr(default, name)
    ]
    _emit(
        "schedule_asked",
        types=",".join(mixed.layer_types),
        recurrent=sum(kind == KDA_LAYER_TYPE for kind in mixed.layer_types),
        other_fields_moved=len(moved),
    )
    assert list(mixed.layer_types) == asked
    assert not moved, moved


def test_a03_the_meta_forwards_device_mismatch_reader_reads_every_wording() -> None:
    """One predicate for three wordings, each armed, and the old needle's own miss."""
    _require_cpu_lane()
    wordings = {
        "fake_tensor": FAKE_TENSOR_WORDING,
        "eager": EAGER_WORDING,
        "op": OP_WORDING,
    }
    _emit(
        "old_needle",
        needle=OLD_NEEDLE,
        reads=";".join(
            f"{name}={int(OLD_NEEDLE in text)}" for name, text in wordings.items()
        ),
    )
    assert OLD_NEEDLE not in FAKE_TENSOR_WORDING
    assert OLD_NEEDLE not in EAGER_WORDING

    read = {
        name: int(meta._names_a_device_mismatch(RuntimeError(text)))
        for name, text in wordings.items()
    }
    other = int(meta._names_a_device_mismatch(RuntimeError(NOT_A_MISMATCH)))
    _emit(
        "reader",
        reads=";".join(f"{name}={value}" for name, value in read.items()),
        reads_a_kernel_refusal=other,
    )
    assert all(read.values()), read
    assert not other

    site = meta._site_of(_a_failure_raised_inside_the_plugin())
    _emit("site", site=site)
    assert site.startswith("functional/kda/gate_clamp.py:"), site
    assert site.rsplit(":", 1)[-1].isdigit(), site


def _a_failure_raised_inside_the_plugin() -> BaseException:
    """A real plugin failure, so the site reader has a real frame and a real line."""
    from vllm_neuron.functional.kda.gate_clamp import GateClampError, kda_gate_clamp

    try:
        kda_gate_clamp(torch.zeros(3), torch.zeros(1), lower=-5.0)
    except GateClampError as caught:
        return caught
    raise AssertionError("the seam accepted a one-dimensional gate")
