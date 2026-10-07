# SPDX-License-Identifier: Apache-2.0
"""Every number the state carriers can change is bitwise what 0a08ff4 produces, at B in {1, 4, 64}.

The fixture ``fixtures/numerics_0a08ff4.json`` was produced by ``gen_numerics_fixture.py``
run against a read-only worktree of 0a08ff4 (the command is recorded inside it). It holds
sha256 digests, per batch size, of:

* the DSA root world: the final step's logits, every layer's pooled store and ring for
  the request slots, every KV latent cache;
* the KDA world (real KDA modules, one bank row per request): every decode step's layer
  output and both state banks of every layer after the last step.

This tree runs the identical scenarios -- the bank-form carriers at ``B > 1`` and the
untouched one-request path at ``B = 1`` -- and every digest must be equal. Two negative
controls show the digests have power: a broken bank gather and a broken ring write each
change them.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/model/glm5_next/tiny/test_tiny_glm5next_numerics_fixture.py
"""

from __future__ import annotations

import json

import pytest
import torch

from test.vllm_neuron.model.glm5_next.tiny import gen_numerics_fixture as gen
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_e2e as e2e

pytestmark = [pytest.mark.fast, pytest.mark.forked]

BASE_REV = "0a08ff4"


def _fixture() -> dict:
    record = json.loads(gen.FIXTURE.read_text())
    assert record["rev"] == BASE_REV, record["rev"]
    assert "hostpath-base" in record["vllm_neuron_tree"], record["vllm_neuron_tree"]
    assert record["env"]["VLLM_NEURON_CPU_MODE"] == "1" and record["env"]["NKI_SIMULATOR"] == "1"
    return record


def _assert_same_digests(got: dict, want: dict, label: str) -> None:
    assert set(got) == set(want), (label, sorted(set(got) ^ set(want)))
    mismatched = [key for key in want if got[key] != want[key]]
    assert not mismatched, f"{label}: digests differ from {BASE_REV} at {mismatched}"


@pytest.mark.parametrize("batch", list(gen.BATCHES))
def test_the_dsa_world_is_bitwise_the_base_revision(batch):
    e2e._require_cpu_mode()
    want = _fixture()["dsa"][str(batch)]
    assert want["steps"] >= 8
    got = gen.dsa_numerics(batch, want["steps"])
    assert got["prompts"] == want["prompts"]
    _assert_same_digests(got["digests"], want["digests"], f"DSA B={batch}")
    # Information, not evidence (the seeded head pins the argmax): still equal.
    assert got["ids"] == want["ids"]


@pytest.mark.parametrize("batch", list(gen.BATCHES))
def test_the_kda_world_is_bitwise_the_base_revision(batch):
    e2e._require_cpu_mode()
    want = _fixture()["kda"][str(batch)]
    assert want["steps"] >= 8
    got = gen.kda_numerics(batch, want["steps"])
    assert got["prompts"] == want["prompts"]
    _assert_same_digests(got["digests"], want["digests"], f"KDA B={batch}")


# ── negative controls: the digests must move when the bank form is broken ───────


def test_the_kda_digests_have_power_against_a_broken_bank_gather(monkeypatch):
    """Zeroed gathered rows at B=4 change every step output and every bank."""
    e2e._require_cpu_mode()
    import vllm_neuron.functional.state_banks as state_banks

    want = _fixture()["kda"]["4"]
    monkeypatch.setattr(
        state_banks, "gather_bank_rows",
        lambda bank, slots: torch.zeros_like(bank.index_select(0, slots)),
    )
    got = gen.kda_numerics(4, want["steps"])["digests"]
    unchanged = [key for key in want["digests"] if got[key] == want["digests"][key]]
    assert not unchanged, f"a zeroed gather left these digests equal: {unchanged}"


def test_the_dsa_digests_have_power_against_a_broken_ring_write(monkeypatch):
    """Zeroed rings at B=4 change every layer's ring digest."""
    e2e._require_cpu_mode()
    from vllm_neuron.functional.dsa import decode_batch

    original = decode_batch.dsa_decode_ring_step

    def zero_rings(*args, **kwargs):
        pooled, rings = original(*args, **kwargs)
        return pooled, torch.zeros_like(rings)

    monkeypatch.setattr(decode_batch, "dsa_decode_ring_step", zero_rings)
    want = _fixture()["dsa"]["4"]
    got = gen.dsa_numerics(4, want["steps"])["digests"]
    rings = [key for key in want["digests"] if key.endswith(".tail[:B]")]
    assert rings
    unchanged = [key for key in rings if got[key] == want["digests"][key]]
    assert not unchanged, f"zeroed rings left these ring digests equal: {unchanged}"
