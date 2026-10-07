# SPDX-License-Identifier: Apache-2.0
"""Greedy ids at B in {1, 4, 64} over eight decode steps equal the ones 0a08ff4 produces.

The fixture ``fixtures/greedy_ids_0a08ff4.json`` was produced by
``gen_greedy_fixture.py`` run against a read-only worktree of 0a08ff4 (the command is
recorded inside it). This tree runs the identical scenario -- the bank-form carriers at
``B > 1`` and the untouched one-request path at ``B = 1`` -- and every token of every
request at every step must match.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/model/glm5_next/tiny/test_tiny_glm5next_greedy_fixture.py
"""

from __future__ import annotations

import json

import pytest

from test.vllm_neuron.model.glm5_next.tiny import gen_greedy_fixture as gen
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_e2e as e2e

pytestmark = [pytest.mark.fast, pytest.mark.forked]

BASE_REV = "0a08ff4"


def _fixture() -> dict:
    record = json.loads(gen.FIXTURE.read_text())
    assert record["rev"] == BASE_REV, record["rev"]
    assert "hostpath-base" in record["vllm_neuron_tree"], record["vllm_neuron_tree"]
    assert record["env"]["VLLM_NEURON_CPU_MODE"] == "1" and record["env"]["NKI_SIMULATOR"] == "1"
    return record


@pytest.mark.parametrize("batch", list(gen.BATCHES))
def test_greedy_ids_match_the_base_revision(batch):
    e2e._require_cpu_mode()
    want = _fixture()["runs"][str(batch)]
    assert want["steps"] >= 8
    got = gen.greedy_ids(batch, want["steps"])
    assert got["prompts"] == want["prompts"]
    for step, (mine, theirs) in enumerate(zip(got["ids"], want["ids"])):
        assert mine == theirs, f"B={batch} step {step}: {mine} != {theirs}"
    assert len(got["ids"]) == len(want["ids"]) == want["steps"] + 1
