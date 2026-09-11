# SPDX-License-Identifier: Apache-2.0
"""Three readings the KDA lane was missing: its allocations, its stack dial, its reader.

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 python -m pytest -s -rA \\
      test/vllm_neuron/model/glm5_next/test_kda_lane_coverage_121d.py

A01 censuses every allocation in ``vllm_neuron/functional/kda/`` and names the ONE that
decides a device for itself, keyed by module, function and factory rather than by a line
that moves. A01B reads why that one is inert: the single shipped call site always passes
``bias``, so the branch holding it is never taken. Neither item repairs it -- the
allocation is on record and its repair is a product change -- and A01 fails the moment a
SECOND such allocation appears in that package.

A02 reads the tiny stack's layer schedule as a dial. The schedule was a fixed argument
beside ``**overrides``, so a caller asking for another shape got a duplicate keyword and a
``TypeError``, and a stack with a recurrent layer in it could not be asked for at all. The
default is unchanged, and that is the reading: every landed item still gets the
sparse-attention stack it got before.

A03 reads the meta-forward's device-mismatch reporter. It keyed on one wording,
``expected device``, while the fake-tensor propagation says ``two different devices`` and
the eager check says ``at least two devices`` -- so a mismatch in either of those arrived
unlabelled. All three read as one now, each is armed, and the old needle's own miss is a
reading beside them.

With ``GLM53F_121D_EXPECT_BASE=1`` A02 and A03 assert the PRE-REPAIR reading of the tree
they are copied into: the schedule refuses an override and the reporter carries no shared
predicate. A01 and A01B read the same on both trees, because what they name is recorded
rather than repaired.
"""

from __future__ import annotations

import ast
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

#: The allocation on record, keyed by module, enclosing function and factory. NOT by line:
#: a line moves whenever anything above it does, and this one already drifted nine lines
#: between the tree that recorded it and the tree that carries it.
RECORDED_ALLOCATION = ("gate_clamp.py", "kda_gate_clamp", "zeros")

#: The seam that holds it, and the argument whose absence would reach it.
RECORDED_SEAM = "kda_gate_clamp"
RECORDED_BRANCH_ARGUMENT = "bias"

#: The repaired model file's own census, which must not slip back.
MODEL_CALLS, MODEL_OFFENDERS = 19, 0

#: A device-less factory, planted so the census walk is shown finding one.
PLANTED = "import torch\n\n\ndef f(rows):\n    return torch.zeros((rows, 1))\n"

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


def test_a01_every_kda_allocation_names_its_device_but_the_one_on_record(tmp_path) -> None:
    """The package census, the recorded allocation by shape, and the walk shown working."""
    _require_cpu_lane()
    modules = sorted(KDA_PACKAGE.glob("*.py"))
    calls, found = 0, []
    for path in modules:
        module_calls, offenders = census._census(path)
        calls += module_calls
        found += [(path.name, where, factory) for _line, factory, where in offenders]
        _emit("census", module=path.name, calls=module_calls, offenders=len(offenders))

    model_file = _model_file()
    model_calls, model_offenders = census._census(model_file)
    _emit(
        "census_totals",
        files=len(modules),
        calls=calls,
        offenders=len(found),
        keyed=";".join(":".join(entry) for entry in sorted(found)) or "none",
        model_calls=model_calls,
        model_offenders=len(model_offenders),
    )

    planted_file = tmp_path / "planted.py"
    planted_file.write_text(PLANTED)
    planted_calls, planted = census._census(planted_file)
    _emit("armed", planted_calls=planted_calls, planted_offenders=len(planted))
    assert planted_calls == 1
    assert [(factory, where) for _line, factory, where in planted] == [("zeros", "f")]

    assert sorted(found) == [RECORDED_ALLOCATION], sorted(found)
    assert (model_calls, len(model_offenders)) == (MODEL_CALLS, MODEL_OFFENDERS)


def test_a01b_the_recorded_allocation_sits_on_a_branch_the_product_never_takes() -> None:
    """Every shipped call of that seam passes ``bias``, so the branch holding it is dead."""
    _require_cpu_lane()
    calls = [
        node
        for node in ast.walk(ast.parse(_model_file().read_text()))
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == RECORDED_SEAM
    ]
    passes = [
        any(keyword.arg == RECORDED_BRANCH_ARGUMENT for keyword in node.keywords)
        for node in calls
    ]
    _emit("call_sites", seam=RECORDED_SEAM, sites=len(calls), pass_the_argument=sum(passes))
    assert len(calls) == 1, f"{len(calls)} call sites of {RECORDED_SEAM}"
    assert all(passes), f"a call site of {RECORDED_SEAM} passes no {RECORDED_BRANCH_ARGUMENT}"


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
    if _expect_base():
        with pytest.raises(TypeError) as refused:
            tiny._stack_text_config(layer_types=asked)
        _emit("schedule_refused", error=" ".join(str(refused.value).split()))
        assert "layer_types" in str(refused.value)
        return

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

    if _expect_base():
        _emit("reader_absent", has_predicate=int(hasattr(meta, "_names_a_device_mismatch")))
        assert not hasattr(meta, "_names_a_device_mismatch")
        assert f'"{OLD_NEEDLE}" in str(error)' in pathlib.Path(meta.__file__).read_text()
        return

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
