# SPDX-License-Identifier: Apache-2.0
"""Where ``test_mtp_weights`` finds the served checkpoint, and what it does when it is absent.

The directory is ``VLLM_NEURON_GLM5NEXT_CHECKPOINT_DIR`` (``vllm_neuron/envs.py``), read
through ``test/vllm_neuron/artifacts.py``; the header-reading tests skip by name, with the
resolved path and the knob, where it holds no index. No other path and no other variable
names it.
"""

from __future__ import annotations

import json

import pytest

from test.vllm_neuron import artifacts
from test.vllm_neuron.model.glm5_next import test_mtp_weights as weights

KNOB = artifacts.CHECKPOINT_KNOB


def test_an_absent_checkpoint_skips_naming_the_path_and_the_knob(monkeypatch, tmp_path):
    missing = tmp_path / "absent"
    monkeypatch.setenv(KNOB, str(missing))
    with pytest.raises(pytest.skip.Exception) as skipped:
        weights.checkpoint_root()
    assert str(missing / artifacts.CHECKPOINT_INDEX) in str(skipped.value)
    assert KNOB in str(skipped.value)


def test_the_knob_names_the_checkpoint_the_headers_are_read_from(monkeypatch, tmp_path):
    """The resolver answers the directory the knob names once it holds an index."""
    (tmp_path / artifacts.CHECKPOINT_INDEX).write_text(json.dumps({"metadata": {}, "weight_map": {}}))
    monkeypatch.setenv(KNOB, str(tmp_path))
    assert weights.checkpoint_root() == tmp_path
