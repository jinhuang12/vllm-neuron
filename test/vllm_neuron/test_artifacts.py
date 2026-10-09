# SPDX-License-Identifier: Apache-2.0
"""The knobs ``test/vllm_neuron/artifacts.py`` reads, and what it does with them.

Each knob is declared in ``vllm_neuron/envs.py`` and resolved by
``vllm_neuron/_artifact_paths.py``, which both ``envs`` and the helper call, so they
agree; an empty value is the documented default, never the working directory. The
helper imports neither torch, vLLM nor ``vllm_neuron/__init__.py``, and ``pytest``
only to skip, so the scripts under ``test/hardware``, ``test/perf`` and
``test/kernel_ledger`` can import it and start fast. What a missing checkpoint does
to a test is pinned beside its user, ``functional/glue/test_glue_case_checkpoint.py``.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from test.vllm_neuron import artifacts

REPO_ROOT = Path(__file__).resolve().parents[2]

#: (reader in the helper, the knob it reads).
KNOBS = [
    pytest.param(artifacts.campaign_path, artifacts.CAMPAIGN_KNOB, id="campaign"),
    pytest.param(artifacts.checkpoint_dir, artifacts.CHECKPOINT_KNOB, id="checkpoint"),
]


def test_campaign_path_is_below_the_knob(monkeypatch, tmp_path):
    """``campaign_path(*parts)`` joins ``parts`` below the knob's directory."""
    monkeypatch.setenv(artifacts.CAMPAIGN_KNOB, str(tmp_path))
    assert artifacts.campaign_path("glm53f-wt", "reports") == tmp_path / "glm53f-wt" / "reports"
    assert artifacts.campaign_path() == tmp_path


@pytest.mark.parametrize("read, knob", KNOBS)
def test_an_empty_knob_is_the_default(monkeypatch, read, knob):
    """An empty value resolves as an unset one, to an absolute default."""
    monkeypatch.delenv(knob, raising=False)
    default = read()
    monkeypatch.setenv(knob, "")
    assert read() == default
    assert default.is_absolute()


@pytest.mark.parametrize("read, knob", KNOBS)
def test_envs_and_the_helper_resolve_the_same_directory(monkeypatch, tmp_path, read, knob):
    """``vllm_neuron.envs`` gives the helper's directory, set and at the default."""
    from vllm_neuron import envs

    monkeypatch.delenv(knob, raising=False)
    assert Path(getattr(envs, knob)) == read()
    monkeypatch.setenv(knob, str(tmp_path))
    assert Path(getattr(envs, knob)) == read() == tmp_path


def test_the_helper_imports_no_torch_no_vllm_and_not_the_package():
    """A fresh interpreter that imports the helper and resolves both paths loads none."""
    probe = (
        "import json, sys\n"
        "from test.vllm_neuron import artifacts\n"
        "artifacts.checkpoint_dir(), artifacts.campaign_path('x')\n"
        "print(json.dumps(sorted(m for m in sys.modules\n"
        "                        if m.split('.')[0] in ('torch', 'vllm', 'pytest')\n"
        "                        or m == 'vllm_neuron')))\n"
    )
    done = subprocess.run([sys.executable, "-c", probe], cwd=REPO_ROOT, check=True,
                          capture_output=True, text=True)
    assert json.loads(done.stdout.strip().splitlines()[-1]) == []


def test_the_module_imports_without_pytest(monkeypatch):
    """A script that has no pytest can still resolve its default paths."""
    monkeypatch.setitem(sys.modules, "pytest", None)  # ``import pytest`` now raises
    try:
        module = importlib.reload(artifacts)
        assert module.campaign_path("x") == module.campaign_path() / "x"
    finally:
        monkeypatch.undo()
        importlib.reload(artifacts)
