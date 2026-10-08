# SPDX-License-Identifier: Apache-2.0
"""A decode step of the tiny root runs the DSA indexer as one launch per stage per layer.

The tiny root (three sparse-attention layers) is driven through the runner's own
carrier builder, with worker-9's harness (``test_tiny_glm5next_batch_decode``): the
requests are prefilled alone into their own pages and slots, the caches are
snapshotted, and two decode steps of all requests run on each arm.

* batched: ``Glm5NextMLAAttention._forward_requests`` as it is, which calls
  ``Glm5NextDSAIndexer.forward_requests`` once per layer: one ring-step launch and
  (when selecting) one score launch for all requests. A one-request step reaches it
  through ``forward`` as the batch-of-one case.
* per-request: the same step with ``forward_requests`` replaced by the loop it
  replaced (e75995b's ``_forward_requests``): ``Glm5NextDSAIndexer.forward`` once per
  request on that request's own ring and pooled store, the one-request ring-step and
  score-GEMM kernels.

Every other stage is the same code on the same rows on both arms, so the arms differ
only in the indexer.

Two regimes, each at B=4 and B=1. The tiny selection keeps ``index_topk = 8`` tokens
(two pools of four), so it is a no-op only up to ``bypass_max_context(8, 4) = 11``
tokens -- the tiny model's 2051.

* selecting: model length 136, every request past 11 tokens at a mix of pool phases,
  so each request scores and drops pools on every layer.
* inside the bound: model length 11, the bypass -- the ring step and the pool write
  only, then dense attention.

Asserted: each batched launch served every row, exactly once per layer and step; no
one-request indexer kernel ran on the batched arm; the per-request arm ran them once
per request, layer and step; the attention route is the same on both arms; the greedy
tokens are identical; the logits and every request's state are bit-identical (the two
score paths agree to fp32 rounding and select the same pools here, after which the
two arms compute the same values), except each pooled store's never-read trash row.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/functional/dsa/test_decode_batch_tiny_step.py
"""

from __future__ import annotations

import pytest
import torch

import vllm_neuron.model.glm5_next.model_fp8 as model_fp8
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_batch_decode as BD
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_forward as tiny
from vllm_neuron.functional.dsa import decode_batch as DB
from vllm_neuron.functional.dsa.decode_bypass import bypass_max_context
from vllm_neuron.functional.dsa.decode_tail_update import decode_tail_dispatch_counters
from vllm_neuron.functional.dsa.score_gemm import score_gemm_dispatch_counters

pytestmark = [pytest.mark.fast, pytest.mark.forked]

BATCH = 4
STEPS = 2
LAYERS = tiny.STACK_LAYERS
MAX_MODEL_LEN = tiny.STACK_TOKENS + 8
#: Positions 13, 18, 27, 34 then one more each: a pool closes at 19, 27 and 35.
PROMPTS = [13, 18, 27, 34]


def _per_request_indexer(self, hidden_states, q_latent, pool_bank, tail_bank, slots,
                         seq_lens, position, *, max_seq_len, indices_wanted=True,
                         projected=None):
    """e75995b's indexer loop: one-request ``forward`` per request, on its own views."""
    assert slots is None and isinstance(pool_bank, tuple) and isinstance(tail_bank, tuple)
    selected = []
    for b in range(int(hidden_states.shape[0])):
        rows = slice(b, b + 1)
        selected.append(self(
            hidden_states[rows], q_latent[rows], pool_bank[b], seq_lens[rows],
            max_seq_len=max_seq_len, page_size=BD.PAGE, indices_wanted=indices_wanted,
            tail=tail_bank[b], position=position[b],
            projected=tuple(None if part is None else part[rows] for part in projected),
        ))
    return torch.cat(selected) if indices_wanted else None


def _spy(monkeypatch, name, rows_arg, seen):
    """Record the request count (rows of positional ``rows_arg``) of every call of
    ``decode_batch.<name>``."""
    real = getattr(DB, name)

    def wrapped(*args, **kwargs):
        seen.append((name, int(args[rows_arg].shape[0])))
        return real(*args, **kwargs)

    monkeypatch.setattr(DB, name, wrapped)


def _counters():
    return {"batch": DB.decode_batch_dispatch_counters(),
            "routes": DB.decode_batch_route_counts()[:2],
            "score_gemm": score_gemm_dispatch_counters(),
            "tail": decode_tail_dispatch_counters()}


def _delta(after, before):
    return {k: tuple(a - b for a, b in zip(after[k], before[k])) for k in after}


def _decode(world):
    """``STEPS`` decode steps of every request together; logits per step and what ran."""
    batch = len(world.req_ids)
    DB.reset_decode_batch_dispatch_counters()
    BD._reset_routes()
    before = _counters()
    logits, fed = [], list(world.first)
    for step in range(STEPS):
        out = BD._step(world, list(range(batch)), torch.tensor(fed),
                       cached=[n + step for n in world.lengths],
                       sampling=list(range(batch)))
        logits.append(out)
        fed = out.argmax(-1).tolist()
    ran = _delta(_counters(), before)
    ran["attention"] = BD._routes()
    return logits, ran


def _assert_state_identical(got, want):
    """Every request's latent pages, rings and stored pools, bit for bit, on every layer.

    The pooled store's last row is its write trash: a step that closes no pool writes a
    meaningless row there on both arms, computed from differently ordered rings (the
    one-request step rotates its ring first), and nothing reads it. That row alone is
    left out.
    """
    assert got.keys() == want.keys()
    for (index, key), value in got.items():
        other = want[(index, key)]
        if key == "pool_cache":
            value, other = value[:, :-1], other[:, :-1]
        assert torch.equal(value, other), f"layer {index} {key} differs"


def _both_arms(monkeypatch, world):
    """The batched indexer and the per-request loop from one snapshot; what each did."""
    monkeypatch.setenv("VLLM_NEURON_GLM5NEXT_STATE_BANKS", "0")  # this test pins the per-request VIEW form
    slots = [world.runner._glm5next_request_slot_table[r] for r in world.req_ids]
    snapshot = BD._snapshot(world)
    seen: list = []
    with monkeypatch.context() as patch:
        _spy(patch, "dsa_decode_ring_step", 1, seen)  # slots, [B]
        _spy(patch, "dsa_decode_scores", 0, seen)  # query, [B, heads, dim]
        batched, batched_ran = _decode(world)
    batched_state = BD._owned_state(world, slots)
    BD._restore(world, snapshot)
    with monkeypatch.context() as patch:
        patch.setattr(model_fp8.Glm5NextDSAIndexer, "forward_requests",
                      _per_request_indexer)
        single, single_ran = _decode(world)
    single_state = BD._owned_state(world, slots)
    for step, (got, want) in enumerate(zip(batched, single)):
        top = want.topk(2, dim=-1).values
        assert float((top[:, 0] - top[:, 1]).min()) > 0.0, f"step {step}: a top-2 tie"
        assert got.argmax(-1).tolist() == want.argmax(-1).tolist(), f"step {step}"
        assert torch.equal(got, want), f"step {step}: logits differ"
    _assert_state_identical(batched_state, single_state)
    return seen, batched_ran, single_ran


def _bound() -> int:
    return bypass_max_context(int(tiny._stack_text_config().index_topk),
                              int(tiny.MLA_INDEX_KPOOL))


def _attention(kind: str, count: int) -> dict:
    return {**{"dense": 0, "selected": 0, "sparse": 0}, kind: count}


@pytest.mark.parametrize("batch", [BATCH, 1])
def test_a_selecting_step_scores_all_requests_in_one_launch_per_layer(monkeypatch, batch):
    """B=4 (the batched step) and B=1 (its batch-of-one case) past the bound."""
    prompts = PROMPTS[-batch:]
    assert _bound() == 11 and min(prompts) > _bound(), "every request must select"
    world = BD._world(batch, max_model_len=MAX_MODEL_LEN, prompts=prompts,
                      window_blocks=-(-MAX_MODEL_LEN // BD.PAGE))
    seen, batched_ran, single_ran = _both_arms(monkeypatch, world)
    launches = LAYERS * STEPS
    # B > 1 attends through the batched decode kernel; one request keeps its own
    # one-request attention (``attend`` at batch_size 1), only its indexer moved.
    attention = _attention("selected" if batch > 1 else "sparse", launches)
    # One ring step, one score launch and one selection per layer and step, each on every
    # row (the selection counts in the batched decode family).
    assert seen == [("dsa_decode_ring_step", batch), ("dsa_decode_scores", batch)] * launches
    assert batched_ran == {"batch": (3 * launches, 0), "routes": (launches, launches),
                           "score_gemm": (0, 0), "tail": (0, 0),
                           "attention": attention}, batched_ran
    # The loop it replaced: the one-request kernels, once per request, layer and step.
    assert single_ran == {"batch": (0, 0), "routes": (0, 0),
                          "score_gemm": (batch * launches, 0),
                          "tail": (batch * launches, 0),
                          "attention": attention}, single_ran


@pytest.mark.parametrize("batch", [BATCH, 1])
def test_a_step_inside_the_bound_writes_its_pools_and_attends_densely(monkeypatch, batch):
    """At or below the bound selection keeps every token: the ring step runs, no scores."""
    prompts = [4, 5, 6, 7][-batch:]
    assert max(prompts) + STEPS <= _bound()
    world = BD._world(batch, max_model_len=_bound(), prompts=prompts,
                      window_blocks=BD.DENSE_WINDOW_BLOCKS)
    seen, batched_ran, single_ran = _both_arms(monkeypatch, world)
    launches = LAYERS * STEPS
    assert seen == [("dsa_decode_ring_step", batch)] * launches
    assert batched_ran == {"batch": (launches, 0), "routes": (launches, 0),
                           "score_gemm": (0, 0), "tail": (0, 0),
                           "attention": _attention("dense", launches)}, batched_ran
    assert single_ran == {"batch": (0, 0), "routes": (0, 0), "score_gemm": (0, 0),
                          "tail": (batch * launches, 0),
                          "attention": _attention("dense", launches)}, single_ran
