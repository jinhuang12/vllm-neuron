"""Arm 2 of ``inc-glm53f-001``: D15's test-import mechanism, measured at the pin.

Read-and-record under D1.3. Every reading is taken inside this collected body, in
the pytest process itself -- no subprocess and no ``sys.modules`` manipulation --
because ``sys.path[0]`` is an *invocation* fact, not a repo fact. So the
instrument is pinned and the pin is asserted rather than assumed: cwd = the repo
root, the Tier T environment in the invocation, the campaign venv's interpreter.
Launched from anywhere else the stdlib ``test`` wins and (e) reads FALSE for a
reason that has nothing to do with D15. The ``test.*`` enumeration is taken at
TWO points and both are recorded.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

#: D3's literal -- the same one ``inc-glm53f-005`` item 5 guards.
DECLARED_TOLERANCE_MAP = {
    "5": (1e-5, 0.011),
    "50": (1e-5, 0.02),
    "1000": (1e-5, 0.03),
    "all": (1e-5, 0.05),
}
#: D15's one declared exception. It carries no ``__init__.py`` on purpose, which
#: makes its dotted name a PEP-420 namespace portion with ``__file__ is None``.
D15_NAMESPACE_EXCEPTIONS = frozenset({"test.vllm_neuron.upstream"})
#: reading key -> (severity, route). The arm's route is the JOIN: the most severe
#: route any member selects. ``"2u"`` is reading 2 for a member the exception
#: list does NOT name.
ROUTES: dict[Any, tuple[int, str]] = {
    1: (0, "RECORD"),
    2: (1, "RECORD -- D15 declared exception, expected and benign"),
    "2u": (2, "ROUTE TO LEAD -- plan_unrealizable_as_designed signal (unnamed namespace portion)"),
    3: (2, "ROUTE TO LEAD -- no __file__ attribute"),
    4: (3, "evidence_contradicts_design on D15 -- member outside the repo overlay"),
    5: (3, "evidence_contradicts_design on D15 -- the test.* set is EMPTY"),
}

_SINK = Path(
    os.environ.get("VLLM_NEURON_INC001_ARM2_JSON")
    or Path(tempfile.gettempdir()) / "vllm_neuron_inc001_arm2_readings.json"
)
_RECORD: dict[str, Any] = {}
_SINK.write_text("{}\n")  # truncate a stale run's values


def _rec(**values: Any) -> None:
    """Persist as we go, so a failing conjunct still leaves its readings behind."""
    _RECORD.update(values)
    _SINK.write_text(json.dumps(_RECORD, indent=2, sort_keys=True, default=str) + "\n")


def _under(path: Path, root: Path) -> bool:
    return str(path) == str(root) or str(path).startswith(str(root) + os.sep)


def _enumerate() -> dict[str, dict[str, Any]]:
    """Every ``sys.modules`` key equal to ``test`` or beginning ``test.``."""
    found: dict[str, dict[str, Any]] = {}
    for name, module in sorted(sys.modules.items()):
        if name == "test" or name.startswith("test."):
            found[name] = {
                "has_file_attr": hasattr(module, "__file__"),
                "file": getattr(module, "__file__", None),
            }
    return found


def _classify(name: str, info: dict[str, Any], root: Path) -> dict[str, Any]:
    """Put one member in exactly one of readings 1-4 and attach its route."""
    if not info["has_file_attr"]:
        key: Any = 3
    elif info["file"] is None:
        key = 2 if name in D15_NAMESPACE_EXCEPTIONS else "2u"
    elif _under(Path(info["file"]).resolve(), root):
        key = 1
    else:
        key = 4
    severity, route = ROUTES[key]
    return {**info, "reading": key, "severity": severity, "route": route}


def _join(snapshot: dict[str, dict[str, Any]], root: Path) -> dict[str, Any]:
    """The arm's route for one snapshot = the most severe route any member selects."""
    members = {name: _classify(name, info, root) for name, info in snapshot.items()}
    if not members:
        severity, route = ROUTES[5]
    else:
        severity = max(member["severity"] for member in members.values())
        route = next(m["route"] for m in members.values() if m["severity"] == severity)
    return {"count": len(members), "members": members, "severity": severity, "route": route}


def test_d15_import_mechanism(pytestconfig: pytest.Config) -> None:
    root = pytestconfig.rootpath.resolve()
    _rec(repo_root=str(root), cwd=str(Path.cwd().resolve()), interpreter=sys.executable)

    # The instrument pin is a conjunct, not advice.
    assert (root / "pyproject.toml").is_file(), f"rootdir {root} does not hold pyproject.toml"
    assert Path.cwd().resolve() == root, (
        f"instrument pin violated: cwd {Path.cwd().resolve()} != repo root {root}"
    )

    import vllm_neuron

    after_plugin = _enumerate()  # point 1 -- immediately after import vllm_neuron

    plugin_file = Path(vllm_neuron.__file__).resolve()
    _rec(a_vllm_neuron_file=str(plugin_file))
    assert _under(plugin_file, root), f"(a) {plugin_file} does not resolve under {root}"
    assert not _under(plugin_file, root / "test"), (
        f"(a) vllm_neuron resolved to the test overlay {plugin_file}, not the real package"
    )

    _rec(b_has_register=hasattr(vllm_neuron, "register"))
    assert _RECORD["b_has_register"] is True, "(b) vllm_neuron has no attribute 'register'"

    from vllm_neuron.accuracy import constants

    measured = dict(constants.DEFAULT_TOLERANCE_MAP)
    _rec(
        c_constants_file=str(Path(constants.__file__).resolve()),
        c_tolerance_map={key: list(value) for key, value in measured.items()},
    )
    assert set(measured) == set(DECLARED_TOLERANCE_MAP), (
        f"(c) key set {sorted(measured)} != {sorted(DECLARED_TOLERANCE_MAP)}"
    )
    for key, want in DECLARED_TOLERANCE_MAP.items():
        got = tuple(measured[key])
        assert len(got) == len(want), f"(c) {key}: arity {len(got)} != {len(want)}"
        for index, (measured_value, declared_value) in enumerate(zip(got, want)):
            assert measured_value == declared_value, (
                f"(c) {key}[{index}]: {measured_value!r} != {declared_value!r}"
            )

    _rec(d_module_name=__name__)
    assert __name__.startswith("test.vllm_neuron."), f"(d) __name__ is {__name__!r}"

    import test as overlay

    after_overlay = _enumerate()  # point 2 -- after the arm's own import test

    overlay_file = Path(overlay.__file__).resolve()
    _rec(e_test_file=str(overlay_file))
    assert overlay_file == (root / "test" / "__init__.py").resolve(), (
        f"(e) the bare name `test` resolved to {overlay_file}, not the repo overlay"
    )

    snapshots = {
        "point_1_after_import_vllm_neuron": _join(after_plugin, root),
        "point_2_after_import_test": _join(after_overlay, root),
    }
    _rec(e_enumeration=snapshots)

    for label, reading in snapshots.items():
        assert reading["count"] > 0, f"(e) reading 5 at {label}: set EMPTY; record at {_SINK}"
        assert reading["severity"] <= 1, (
            f"(e) {label} selects {reading['route']}; record at {_SINK}; members "
            + repr({n: (m["reading"], m["file"]) for n, m in reading["members"].items()})
        )
