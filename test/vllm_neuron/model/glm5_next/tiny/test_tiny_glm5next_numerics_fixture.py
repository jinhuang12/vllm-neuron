# SPDX-License-Identifier: Apache-2.0
"""Every number the state carriers can change is 0a08ff4's at B in {1, 4, 64}: bitwise, or bounded.

The fixture ``fixtures/numerics_0a08ff4.json`` was produced by ``gen_numerics_fixture.py``
run against a read-only worktree of 0a08ff4 (the command is recorded inside it). It holds
sha256 digests, per batch size, of:

* the DSA root world: the final step's logits, every layer's pooled store and ring for
  the request slots, every KV latent cache;
* the KDA world (real KDA modules, one bank row per request): every decode step's layer
  output and both state banks of every layer after the last step.

This tree runs the identical scenarios -- the bank-form carriers at ``B > 1`` and the
untouched one-request path at ``B = 1`` -- and every digest must be equal, with one
exception: the DSA pooled stores, in a world where this tree's pooling kernel rounds a row
other than 0a08ff4's kernel does on the same operands. Every pooling call is held to the
kernel's derived error bound (see ``POOLED_STORES``). Two negative controls show the
digests have power: a broken bank gather and a broken ring write each change them.

The digests are of CPU float arithmetic whose order follows the thread count, so the run
uses the producing command's ``OMP_NUM_THREADS``:

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 OMP_NUM_THREADS=4 python -m pytest \\
        test/vllm_neuron/model/glm5_next/tiny/test_tiny_glm5next_numerics_fixture.py
"""

from __future__ import annotations

import json

import pytest
import torch

from test.hardware import benchmark_dsa_hadamard as bench
from test.vllm_neuron.functional.dsa import test_kpool_hadamard_error_bound as bound
from test.vllm_neuron.model.glm5_next.tiny import gen_numerics_fixture as gen
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_e2e as e2e
from vllm_neuron.functional.dsa import kpool_hadamard

pytestmark = [pytest.mark.fast, pytest.mark.forked]

BASE_REV = "0a08ff4"

#: The DSA digests held to a bound rather than bit for bit: every DSA layer's pooled store.
#: Its prefill rows are ``dsa_kpool_hadamard``'s, and this tree's kernel sums in another
#: order than 0a08ff4's (the rotation on the Tensor Engine, the reciprocal on the Scalar
#: Engine), so a row may round to another bf16 value. Where one does, the stores' digests
#: may move and no other digest may; where none does, the stores are held bit for bit too.
POOLED_STORES = ("side.0.pool_cache[:B]", "side.1.pool_cache[:B]", "side.2.pool_cache[:B]")


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


def _base_kpool_hadamard(where):
    """0a08ff4's ``kpool_hadamard`` module, read by ``git show`` into ``where``.

    It is pinned to the blob the hardware benchmark measures as its baseline (f3a833f's,
    the same file), so a different base kernel cannot pass silently.
    """
    path, blob = bench.BASELINE_SOURCES["kpool_hadamard"]
    return bench.load_revision(
        BASE_REV, where, f"numerics_base_{BASE_REV}",
        {"kpool_hadamard": path}, {"kpool_hadamard": blob},
    ).kpool_hadamard


def _record_pooling(monkeypatch) -> list:
    """Keep the operands and the result of every ``dsa_kpool_hadamard`` call the world makes.

    The hook calls the served function and returns its result unchanged: it observes the
    world and substitutes nothing. The model imports the function from its module at each
    call, so the hook sees every call.
    """
    served = kpool_hadamard.dsa_kpool_hadamard
    calls = []

    def record(slot_k, slot_score, ape):
        pooled = served(slot_k, slot_score, ape)
        calls.append(tuple(t.detach().clone() for t in (slot_k, slot_score, ape, pooled)))
        return pooled

    monkeypatch.setattr(kpool_hadamard, "dsa_kpool_hadamard", record)
    return calls


def _pooling_against_base(calls: list, base) -> dict:
    """Each recorded call against the derived bound, beside 0a08ff4's kernel on its operands.

    Returns the number of calls, how many of them differ from 0a08ff4's bit for bit, and the
    largest ``|out - exact| / bound`` of each kernel over every call.
    """
    worst = {"this tree": 0.0, BASE_REV: 0.0}
    differing = 0
    base.reset_kpool_hadamard_dispatch_counters()
    for slot_k, slot_score, ape, pooled in calls:
        base_pooled = base.dsa_kpool_hadamard(slot_k, slot_score, ape)
        exact = bound.exact_pooling(slot_k, slot_score, ape)
        limit = bound.pooling_error_bound(slot_k, slot_score, ape)
        for name, out in (("this tree", pooled), (BASE_REV, base_pooled)):
            ratio = float(bound.error_over_bound(out, exact, limit).max())
            worst[name] = max(worst[name], ratio)
        differing += gen.digest(pooled) != gen.digest(base_pooled)
    # 0a08ff4's results are its NKI kernel's, not its torch fallback's.
    assert base.kpool_hadamard_dispatch_counters() == (len(calls), 0)
    return {"calls": len(calls), "differing": differing, "max_error_over_bound": worst}


@pytest.mark.parametrize("batch", list(gen.BATCHES))
def test_the_dsa_world_is_bitwise_the_base_revision_except_the_pooled_stores_which_hold_the_bound(
    batch, monkeypatch, tmp_path
):
    """Every DSA digest is 0a08ff4's bit for bit, except ``POOLED_STORES`` where a pooling
    call rounds other than 0a08ff4's kernel on the same operands; every call is within the
    derived bound, and so is 0a08ff4's kernel."""
    e2e._require_cpu_mode()
    want = _fixture()["dsa"][str(batch)]
    assert want["steps"] >= 8
    assert set(POOLED_STORES) <= set(want["digests"])
    base = _base_kpool_hadamard(tmp_path)
    calls = _record_pooling(monkeypatch)
    kpool_hadamard.reset_kpool_hadamard_dispatch_counters()
    got = gen.dsa_numerics(batch, want["steps"])
    assert got["prompts"] == want["prompts"]
    # The world pooled, and on the NKI kernel, not the torch fallback.
    assert calls and kpool_hadamard.kpool_hadamard_dispatch_counters()[1] == 0
    pooling = _pooling_against_base(calls, base)
    label = f"DSA B={batch} (pooling: {pooling})"
    assert max(pooling["max_error_over_bound"].values()) <= 1.0, (
        f"{label}: a pooled row is outside the derived bound"
    )
    exempt = set(POOLED_STORES) if pooling["differing"] else set()
    _assert_same_digests(
        {key: value for key, value in got["digests"].items() if key not in exempt},
        {key: value for key, value in want["digests"].items() if key not in exempt},
        label,
    )
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
