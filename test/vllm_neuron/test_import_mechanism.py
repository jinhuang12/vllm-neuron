"""Import behaviour of the repo's own packages under pytest.

The repo carries a ``test`` package that shadows the stdlib ``test`` module, so
these readings only hold when pytest runs from the repo root: the bare name
``test`` must resolve to ``test/__init__.py`` here, ``vllm_neuron`` must resolve
to the real package rather than to anything under ``test/``, and every already
imported ``test.*`` module must live inside the repo.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

# The tolerance table ``vllm_neuron.accuracy`` publishes.
EXPECTED_TOLERANCE_MAP = {
    "5": (1e-5, 0.011),
    "50": (1e-5, 0.02),
    "1000": (1e-5, 0.03),
    "all": (1e-5, 0.05),
}

# ``test.vllm_neuron.upstream`` carries no ``__init__.py`` on purpose, which
# makes its dotted name a PEP-420 namespace portion with ``__file__ is None``.
NAMESPACE_PORTIONS = frozenset({"test.vllm_neuron.upstream"})


def _under(path: Path, root: Path) -> bool:
    return str(path) == str(root) or str(path).startswith(str(root) + os.sep)


def _imported_test_modules() -> dict[str, Any]:
    """Every ``sys.modules`` entry named ``test`` or beginning ``test.``."""
    return {
        name: module
        for name, module in sorted(sys.modules.items())
        if name == "test" or name.startswith("test.")
    }


def _assert_all_inside_repo(modules: dict[str, Any], root: Path, label: str) -> None:
    for name, module in modules.items():
        assert hasattr(module, "__file__"), f"{label}: {name} has no __file__"
        file = module.__file__
        if file is None:
            assert name in NAMESPACE_PORTIONS, (
                f"{label}: {name} has __file__ None but is not a known "
                "namespace portion"
            )
            continue
        resolved = Path(file).resolve()
        assert _under(resolved, root), (
            f"{label}: {name} resolves to {resolved}, outside {root}"
        )


def test_repo_packages_resolve_inside_the_repo(pytestconfig: pytest.Config) -> None:
    root = pytestconfig.rootpath.resolve()
    assert (root / "pyproject.toml").is_file(), f"rootdir {root} holds no pyproject.toml"
    # ``sys.path[0]`` follows the invocation, not the repo, so the bare name
    # ``test`` only resolves to this repo's package when cwd is the repo root.
    assert Path.cwd().resolve() == root, (
        f"run pytest from the repo root; cwd is {Path.cwd().resolve()}"
    )

    import vllm_neuron

    after_plugin = _imported_test_modules()

    plugin_file = Path(vllm_neuron.__file__).resolve()
    assert _under(plugin_file, root), f"{plugin_file} does not resolve under {root}"
    assert not _under(plugin_file, root / "test"), (
        f"vllm_neuron resolved to the test tree at {plugin_file}, not the package"
    )
    assert hasattr(vllm_neuron, "register"), "vllm_neuron has no attribute 'register'"

    from vllm_neuron.accuracy import constants

    measured = dict(constants.DEFAULT_TOLERANCE_MAP)
    assert set(measured) == set(EXPECTED_TOLERANCE_MAP), (
        f"key set {sorted(measured)} != {sorted(EXPECTED_TOLERANCE_MAP)}"
    )
    for key, expected in EXPECTED_TOLERANCE_MAP.items():
        got = tuple(measured[key])
        assert len(got) == len(expected), f"{key}: arity {len(got)} != {len(expected)}"
        for index, (measured_value, expected_value) in enumerate(zip(got, expected)):
            assert measured_value == expected_value, (
                f"{key}[{index}]: {measured_value!r} != {expected_value!r}"
            )

    assert __name__.startswith("test.vllm_neuron."), f"__name__ is {__name__!r}"

    import test as overlay

    after_overlay = _imported_test_modules()

    overlay_file = Path(overlay.__file__).resolve()
    assert overlay_file == (root / "test" / "__init__.py").resolve(), (
        f"the bare name `test` resolved to {overlay_file}, not this repo's package"
    )

    for label, modules in (
        ("after import vllm_neuron", after_plugin),
        ("after import test", after_overlay),
    ):
        assert modules, f"{label}: no test.* module is imported at all"
        _assert_all_inside_repo(modules, root, label)
