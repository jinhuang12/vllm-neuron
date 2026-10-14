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
untouched one-request path at ``B = 1`` -- and every digest must be equal, but for the four
values :data:`SELECTION_ORDER_DIGESTS` pins instead (in the DSA world at ``B = 64``, which
the decode selection's pool order moves) and the KDA values :data:`KDA_CONV1D_DIGESTS` pins
(which the in-tree prefill conv1d kernel moves). Two negative controls show the
digests have power: a broken bank gather and a broken ring write each change them.

The KDA pins are not the only guard on the conv1d. The KDA test also records every prefill
conv1d call, checks that each one ran the NKI kernel, and checks each output against the
module's torch reference within a bound derived from float32 and the tap count, never
fitted. Each output is a sum of ``S`` float32 products (``S = 4`` taps here). Any order of
``S`` products and ``S - 1`` sums is within ``gamma_S * M`` of the exact sum, where
``M = sum_s |w_s * x_s|``, ``gamma_S = S * u / (1 - S * u)`` and ``u = 2**-24`` (the
dot-product bound). So two float32 forms of one output differ by at most ``2 * gamma_S * M``.

Derivation: this test's own conv1d calls were recorded at OMP 4 and replayed through
a01e99c2's kernel and through the base's nkilib implicit GEMM, both in the NKI simulator.
Per batch size: outputs that differ / outputs, max ``|delta|``, and max ``|delta|`` over
the bound:

* ``B = 1``: 292 / 2,304, 5.96e-08, 0.44;
* ``B = 4``: 7,603 / 23,808, 5.96e-08, 0.40;
* ``B = 64``: 127,885 / 395,520, 8.94e-08, 0.43.

Every output is within the bound (0 over, at every ``B``). Each kernel is also within
``gamma_S * M`` of a float64 sum (max ratio 0.61 for a01e99c2's kernel, 0.59 for nkilib's).
Against the torch reference, a01e99c2's kernel is at most 0.46 of the bound (0.44 / 0.44 /
0.46 at ``B = 1 / 4 / 64``).

The fixture was produced at ``OMP_NUM_THREADS=4`` (the producing command in
``gen_numerics_fixture.py``). Torch splits a CPU op's work by the intra-op thread count,
and some of the worlds' sums round apart at another count: an unchanged tree moves the
DSA ``B = 64`` digests at 1 thread, and the DSA ``B = 64`` and KDA ``B = 1`` digests at
96. So every test here runs at :data:`FIXTURE_THREADS`, whatever count the process
started with.

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

#: The digests this tree's decode selection moves from the base, by batch size: in the DSA
#: world at B = 64, the final logits and the last DSA layer's pooled store, ring and KV
#: cache. The one-kernel selection (``decode_select.py``) lists a request's selected pools
#: in ascending pool order, where the base's four-kernel route listed them by descending
#: score. The selected sets are the same (``test_decode_select.py`` pins them against the
#: base's chain); the sparse attention sums the selected rows in column order, so its bf16
#: output rounds apart in the last bits from the second layer on. The first two layers'
#: state keeps the base's digests; the third layer's input is the second layer's output,
#: so what it writes moves, and the logits after it. The greedy ids stay the base's. Each
#: value is this tree's own, bitwise, so any further move fails. B = 1 and B = 4 move
#: nothing.
SELECTION_ORDER_DIGESTS = {
    64: {
        "final_logits": "c9c5a400f85df1c43c1c35aad8cf24b47e77923260111cb873f538b35f90aa21",
        "side.2.pool_cache[:B]":
            "f2270e6ad69a9c5e19e8e57f73e23dd0496387bb6f128d585dd8616e56a2d42a",
        "side.2.tail[:B]": "955d01bff2d68a2a96624205827f0264b83f16bb82eff2797d239c795c74cbf6",
        "kv.layers.2.self_attn.0":
            "8a3f808a04c7cc026db25badd23b8abf56070d39ca247cd7e914e01950d0a0a1",
    },
}

#: The digests this tree's KDA prefill conv1d moves from the base, by batch size. a01e99c2
#: moved the conv1d (four float32 taps) from nkilib's implicit GEMM to the in-tree kernel
#: (``depthwise_conv1d_kernel.py``). The in-tree kernel adds the four tap products in a
#: different order (two float32 chains), so its outputs round apart from the base's in the
#: last bits (the module docstring gives the measured differences and their bound). The
#: conv1d runs at prefill only; what moves is the decode-step outputs listed here and, at
#: B = 4 and B = 64, the second layer's recurrent state. Each value is this tree's own,
#: bitwise, so any further move fails.
KDA_CONV1D_DIGESTS = {
    # a01e99c2: the in-tree conv1d's tap summation order moves these 5 of 12.
    1: {
        "step.0.output": "ef0f07409807a3c21e7cb07695f165d49219f762fb4ab51493b3d3226b1d423e",
        "step.1.output": "61c77420dae010753c18893991f8e006c3d07c5356129c8ad7fd6c7eb8868c41",
        "step.2.output": "e60953bc338a6a5615b77994dd1abd220f3e2d26a03a8232e91f51b641a30853",
        "step.3.output": "0624def209016bfc64ae5c3fa4d107848e6011c6b97828177218f540f6d98586",
        "step.5.output": "925d9f30ef16a512598b6f3198866363134bfdf5eeedb7cc1d2e5c3ca101a118",
    },
    # a01e99c2: the in-tree conv1d's tap summation order moves these 6 of 12.
    4: {
        "step.0.output": "ad9b3a643caba8fe88b95465866a036a3696ffd1234234941ee8f49122fb0eb1",
        "step.1.output": "0c26ac75520a5ea8d096247357d3987dd630517858ad9cda7536bed42f39b6e4",
        "step.2.output": "9e45eb04178b2a24a5eaf53c641855b87c059abe8445352e2b84cd7837d7cf50",
        "step.3.output": "60454d1883887308df111cd82c370873cfeccf464e3283ba97753f0eb0d7c669",
        "step.6.output": "9622a6052a83691b1b1c10f46f7e0f4b67469896c429c66ee6d49033960ffe7c",
        "bank.1.recurrent_state":
            "41af9b6559ae2eefee17f0b0f2cf407d12a9ed410d0b2320e332eb9781a4b6f7",
    },
    # a01e99c2: the in-tree conv1d's tap summation order moves these 7 of 12.
    64: {
        "step.0.output": "802e86207b8993456869b79a75e1ec22239afe5436179792dfd5ab2bc0d850c1",
        "step.1.output": "9109caea026902ee96fc9adf91eccc39792cdc2e90cd6c0c597ecb790d9fa931",
        "step.2.output": "b1501327678fc790ff972f845b0b0872a59020f2bd6f15abcdd5316fbfda2504",
        "step.3.output": "0bd3a093cd57085499bc4b7a18f731872ce52f5962ae54b9d501556f7a5971ca",
        "step.4.output": "2a687b4f096e539ff1055b826f94f1ef2fe8462ce2ded657bc6c194932c798ec",
        "step.5.output": "71281fcaf101067fe9cee317e05ddc11ee33de228ae0bc3ff1003fef5a895572",
        "bank.1.recurrent_state":
            "80e9817acc902fe2d1eb70452f64f200bc4765277d09b28f1f826f6c2f437ebc",
    },
}

#: The unit roundoff of float32, for the conv1d bound (module docstring).
FLOAT32_UNIT_ROUNDOFF = 2.0**-24

#: The intra-op thread count the fixture was produced at; every test runs at it.
FIXTURE_THREADS = 4


@pytest.fixture(autouse=True)
def _at_the_fixture_thread_count():
    before = torch.get_num_threads()
    torch.set_num_threads(FIXTURE_THREADS)
    try:
        yield
    finally:
        torch.set_num_threads(before)


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


def _kda_digests(batch: int, want: dict) -> dict:
    """The fixture's KDA digests with this tree's conv1d pins merged over them."""
    moved = KDA_CONV1D_DIGESTS[batch]
    assert set(moved) <= set(want), sorted(set(moved) - set(want))
    return {**want, **moved}


def _record_conv1d_calls(monkeypatch) -> list:
    """Keeps ``(img, filt, out)`` of every prefill conv1d call; the call itself is unchanged."""
    from vllm_neuron.functional.kda import depthwise_conv1d as conv

    calls = []
    original = conv.depthwise_conv1d

    def recording(img, filt, *args, **kwargs):
        assert not args and not kwargs, (args, kwargs)
        out = original(img, filt)
        calls.append((img.detach().clone(), filt.detach().clone(), out.detach().clone()))
        return out

    monkeypatch.setattr(conv, "depthwise_conv1d", recording)
    return calls


def _assert_conv1d_within_the_float32_bound(calls: list, label: str) -> None:
    """Each output is within ``2 * gamma_S * M`` of the torch reference (module docstring)."""
    from vllm_neuron.functional.kda.depthwise_conv1d import depthwise_conv1d_torch_reference

    u = FLOAT32_UNIT_ROUNDOFF
    for index, (img, filt, out) in enumerate(calls):
        assert img.dtype == filt.dtype == out.dtype == torch.float32, (img.dtype, out.dtype)
        x, w = img.double()[:, :, 0, :], filt.double()[:, 0, 0, :]
        taps = w.shape[1]
        columns = x.shape[-1] - taps + 1
        assert tuple(out.shape) == (x.shape[0], x.shape[1], 1, columns), tuple(out.shape)
        magnitude = sum(
            (w[None, :, s:s + 1] * x[:, :, s:s + columns]).abs() for s in range(taps)
        )
        bound = 2 * taps * u / (1 - taps * u) * magnitude
        reference = depthwise_conv1d_torch_reference(img, filt).double()[:, :, 0, :]
        excess = (out.double()[:, :, 0, :] - reference).abs() - bound
        assert (excess <= 0).all(), (
            f"{label}: conv1d call {index}: {int((excess > 0).sum())} of {excess.numel()} "
            f"outputs outside 2 gamma_{taps} M of the torch reference, worst by "
            f"{excess.max().item():.3e}"
        )


@pytest.mark.parametrize("batch", list(gen.BATCHES))
def test_the_dsa_world_is_bitwise_the_base_revision(batch):
    e2e._require_cpu_mode()
    want = _fixture()["dsa"][str(batch)]
    assert want["steps"] >= 8
    got = gen.dsa_numerics(batch, want["steps"])
    assert got["prompts"] == want["prompts"]
    moved = SELECTION_ORDER_DIGESTS.get(batch, {})
    assert set(moved) <= set(want["digests"]), sorted(set(moved) - set(want["digests"]))
    _assert_same_digests(got["digests"], {**want["digests"], **moved}, f"DSA B={batch}")
    # Information, not evidence (the seeded head pins the argmax): still equal.
    assert got["ids"] == want["ids"]


@pytest.mark.parametrize("batch", list(gen.BATCHES))
def test_the_kda_world_is_bitwise_the_base_revision(batch, monkeypatch):
    from vllm_neuron.functional.kda.depthwise_conv1d import (
        dispatch_counters,
        reset_dispatch_counters,
    )

    e2e._require_cpu_mode()
    want = _fixture()["kda"][str(batch)]
    assert want["steps"] >= 8
    calls = _record_conv1d_calls(monkeypatch)
    reset_dispatch_counters()
    got = gen.kda_numerics(batch, want["steps"])
    assert got["prompts"] == want["prompts"]
    _assert_same_digests(got["digests"], _kda_digests(batch, want["digests"]), f"KDA B={batch}")
    # The second guard: every conv1d call ran the NKI kernel (not the torch fallback, which
    # would compare torch with itself), and each output is within the float32 bound.
    assert calls, "the KDA world made no conv1d call"
    assert dispatch_counters() == (len(calls), 0), (dispatch_counters(), len(calls))
    _assert_conv1d_within_the_float32_bound(calls, f"KDA B={batch}")


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
    expected = _kda_digests(4, want["digests"])
    got = gen.kda_numerics(4, want["steps"])["digests"]
    unchanged = [key for key in expected if got[key] == expected[key]]
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
