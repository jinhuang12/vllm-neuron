# SPDX-License-Identifier: Apache-2.0
"""Concurrent decode on the tiny root: B requests in one step give one-request answers.

The tiny root (three sparse-attention layers, dense MLP and MoE halves, the head) is
driven through the runner's own carrier builder, ``_glm5next_model_kwargs``, the way a
served step reaches it:

1. every request is prefilled alone (the scheduler schedules one prefill per step), at
   its own prompt and length, into its own pages and its own slot;
2. the caches are snapshotted;
3. the batched arm decodes all B requests together for two steps, each step feeding back
   its own greedy tokens;
4. the reference arm restores the snapshot and decodes the same requests one at a time,
   the one-request path.

Read at B in {4, 64} in the selecting regime (the context exceeds the selection's
bound, so the indexer scores and selects per request) and at B = 4 in the dense regime
(the selection would keep every token, so the layer attends the causal prefix). The
greedy tokens must be identical and the logits must agree to bf16 rounding: in the
selecting regime the batched decode kernel stands against the one-request sparse kernel,
which sums in another order, and in both regimes the row-batched kernels (projections,
dense MLP, MoE) round a [B, ...] operand apart from B [1, ...] ones in the last bit. The
first layer's input is the embedding row on both arms, so the caches it writes must be
bit-identical; every later layer's must agree to bf16 rounding. The batched arm must take
one attention launch per layer, not one per request.

A padded step (3 requests in the bucket of 4) is read too: the padding row is served
from the null block, a scratch ring and a slot no request holds, and every request's
state comes back as three one-request steps leave it.

The root's own head scores two ids equally for every row, which would make a token
comparison vacuous, so a seeded random head replaces it; the minimum top-2 gap is
asserted to exceed the logit tolerance, so the token check discriminates.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/model/glm5_next/tiny/test_tiny_glm5next_batch_decode.py
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_neuron.functional.attention import mla_decode as decode_seam
from vllm_neuron.functional.attention import mla_sparse as sparse_seam
from vllm_neuron.functional.dsa.decode_bypass import bypass_max_context
from vllm_neuron.vllm.worker.neuron_model_runner import NULL_BLOCK_ID, NeuronModelRunner

from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_e2e as e2e
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_forward as tiny

pytestmark = [pytest.mark.fast, pytest.mark.forked]

PAGE = tiny.MLA_PAGE_SIZE
DECODE_STEPS = 2
SEED = 20261006
#: bf16 logits: the selecting regime's two attention kernels differ in summation order,
#: which moves a logit by an ulp or two.
LOGIT_RTOL = 2.0**-6
#: The dense window must be whole 128-row chunks for the decode kernel.
DENSE_WINDOW_BLOCKS = 128 // PAGE


def _world(batch: int, *, max_model_len: int, prompts: list[int], window_blocks: int,
           slots: int | None = None):
    """The tiny root with a random head, its caches, a runner shell, and B prefilled requests."""
    e2e._require_cpu_mode()
    fixture = e2e._fixture()
    root = fixture["root"]
    head = torch.randn(
        tiny.STACK_VOCAB_SIZE, int(root.text_config.hidden_size),
        generator=torch.Generator().manual_seed(SEED),
    )
    root.lm_head_weight = torch.nn.Parameter(head.to(torch.bfloat16), requires_grad=False)
    per_request = -(-max_model_len // PAGE)
    blocks = 1 + batch * per_request
    caches = {}
    for spec in root.get_kv_spec().layers:
        shape = (blocks, int(spec.num_kv_heads), PAGE, int(spec.head_size))
        caches[spec.name] = [
            torch.zeros(shape, dtype=spec.dtype) for _ in range(1 if spec.latent_kv else 2)
        ]
    root.bind_kv_cache(caches)
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.input_batch = SimpleNamespace(req_ids=[])
    runner.model = root
    runner.max_model_len = int(max_model_len)
    runner.max_num_reqs = int(slots or batch)
    # Block 0 is the null block; request r owns a run of pages after it, so no two
    # requests share a page.
    tables = [
        [1 + r * per_request + k for k in range(per_request)]
        + [NULL_BLOCK_ID] * (window_blocks - per_request)
        for r in range(batch)
    ]
    gen = torch.Generator().manual_seed(SEED + batch)
    world = SimpleNamespace(
        root=root, runner=runner, caches=caches, tables=tables,
        window_blocks=window_blocks, req_ids=[f"req-{r}" for r in range(batch)],
        lengths=list(prompts),
    )
    first = []
    for r in range(batch):
        prompt = torch.randint(0, tiny.STACK_VOCAB_SIZE, (prompts[r],), generator=gen)
        logits = _step(world, [r], prompt, cached=[0], sampling=[prompts[r] - 1])
        first.append(int(logits[-1].float().argmax()))
    world.first = first
    return world


def _metadata(world, rows: list[int], *, cached: list[int], tokens: int) -> dict:
    table = torch.tensor([world.tables[r] for r in rows], dtype=torch.int32)
    entry = {
        "block_table_tensor": table,
        "full_block_table_tensor": table,
        "slot_mapping": torch.zeros(tokens, dtype=torch.int64),
        "max_query_len": tokens if len(rows) == 1 else 1,
        "block_size": PAGE,
        "max_blocks_per_seq": world.window_blocks,
        "decode_token_threshold": 1,
        "cached_seq_len": torch.tensor([[cached[0]]], dtype=torch.int32),
        "host_block_table": table,
        "host_num_computed_tokens": torch.tensor(cached, dtype=torch.int32),
        "kv_segment_size": 0,
    }
    return {bank["name"]: entry for bank in world.root.glm5next_layer_banks}


def _step(world, rows, input_ids, *, cached, sampling, real=None, out=None):
    """One step for requests ``rows`` through the converter and the root; returns logits."""
    runner = world.runner
    runner.input_batch.req_ids = [world.req_ids[r] for r in rows]
    runner._glm5next_request_tokens = None if real is None else np.array(real, np.int32)
    converted = runner._glm5next_model_kwargs(
        {
            "input_ids": input_ids,
            "positions": None,
            "attn_metadata": _metadata(world, rows, cached=cached,
                                       tokens=int(input_ids.shape[0])),
            "sampling_positions": torch.tensor(sampling, dtype=torch.long),
            "sampling_params": None,
            "spec_decode_metadata": None,
            "rank": None,
            "logit_mask": None,
        }
    )
    if out is not None:
        out.append(converted["layer_carriers"])
    return world.root.forward(**converted).float()


def _snapshot(world):
    runner = world.runner
    return (
        {name: [t.clone() for t in tensors] for name, tensors in world.caches.items()},
        [{k: v.clone() for k, v in side.items()} for side in runner._glm5next_side_cache_set],
        dict(runner._glm5next_side_cache_positions),
    )


def _restore(world, snapshot):
    caches, sides, positions = snapshot
    for name, tensors in caches.items():
        for live, saved in zip(world.caches[name], tensors):
            live.copy_(saved)
    for live, saved in zip(world.runner._glm5next_side_cache_set, sides):
        for key, value in saved.items():
            live[key] = value.clone()
    world.runner._glm5next_side_cache_positions = dict(positions)


def _reset_routes():
    decode_seam.reset_mla_decode_dispatch_counters()
    sparse_seam.reset_mla_sparse_dispatch_counters()


def _routes():
    dense, selected, _ = decode_seam.mla_decode_route_counts()
    return {"dense": dense, "selected": selected,
            "sparse": sparse_seam.mla_sparse_dispatch_counters()[0]}


def _owned_state(world, slots):
    """Every request's own state, per layer: its latent pages, its slot's side caches."""
    pages = sorted(
        {b for r in range(len(world.req_ids)) for b in world.tables[r]} - {NULL_BLOCK_ID}
    )
    sides = world.runner._glm5next_side_cache_set
    out = {}
    for index, bank in enumerate(world.root.glm5next_layer_banks):
        out[(index, "latent")] = world.caches[bank["name"]][0][pages].clone()
        for key in ("pool_cache", "tail"):
            if key in sides[index]:
                out[(index, key)] = sides[index][key][slots].clone()
    return out


def _assert_state_equal(left, right, label, *, exact_layers):
    """Bit for bit up to ``exact_layers``; past it, to bf16 rounding of the cached values.

    In the selecting regime the two arms attend through different kernels, so a layer's
    output differs by an ulp and every later layer's cached keys inherit it. The first
    layer's input is the embedding row on both arms, so its writes must be identical.
    """
    for (index, key), value in left.items():
        other = right[(index, key)]
        if index < exact_layers:
            assert torch.equal(value, other), f"{label}: layer {index} {key} differs"
        else:
            peak = float(other.float().abs().max())
            torch.testing.assert_close(
                value.float(), other.float(), rtol=LOGIT_RTOL, atol=LOGIT_RTOL * peak,
                msg=lambda m: f"{label}: layer {index} {key}: {m}",
            )


def _batched_against_single(world, *, dense: bool):
    batch = len(world.req_ids)
    layers = tiny.STACK_LAYERS
    slots = [world.runner._glm5next_request_slot_table[r] for r in world.req_ids]
    assert sorted(slots) == list(range(batch)), slots
    snapshot = _snapshot(world)

    _reset_routes()
    batched, fed = [], list(world.first)
    for step in range(DECODE_STEPS):
        logits = _step(world, list(range(batch)), torch.tensor(fed),
                       cached=[n + step for n in world.lengths], sampling=list(range(batch)))
        batched.append(logits)
        fed = logits.argmax(-1).tolist()
    batched_routes = _routes()
    batched_state = _owned_state(world, slots)

    _restore(world, snapshot)
    _reset_routes()
    single, fed = [], list(world.first)
    for step in range(DECODE_STEPS):
        rows = [
            _step(world, [r], torch.tensor([fed[r]]), cached=[world.lengths[r] + step],
                  sampling=[0])
            for r in range(batch)
        ]
        logits = torch.cat(rows)
        single.append(logits)
        fed = logits.argmax(-1).tolist()
    single_routes = _routes()
    single_state = _owned_state(world, slots)

    kind = "dense" if dense else "selected"
    assert batched_routes == {**{"dense": 0, "selected": 0, "sparse": 0},
                              kind: layers * DECODE_STEPS}, batched_routes
    expected_single = (
        {"dense": layers * DECODE_STEPS * batch, "selected": 0, "sparse": 0}
        if dense else {"dense": 0, "selected": 0, "sparse": layers * DECODE_STEPS * batch}
    )
    assert single_routes == expected_single, single_routes

    for step, (got, want) in enumerate(zip(batched, single)):
        top = want.topk(2, dim=-1).values
        gap = float((top[:, 0] - top[:, 1]).min())
        peak = float(want.abs().max())
        tolerance = LOGIT_RTOL * peak
        torch.testing.assert_close(got, want, rtol=0.0, atol=tolerance)
        assert gap > 2 * tolerance, (
            f"step {step}: the closest top-2 gap {gap} is inside twice the logit "
            f"tolerance {tolerance}, so the token check would not discriminate"
        )
        assert got.argmax(-1).tolist() == want.argmax(-1).tolist(), f"step {step}"
    _assert_state_equal(batched_state, single_state, f"B={batch} {kind}", exact_layers=1)


@pytest.mark.parametrize("batch", [4, 64])
def test_a_batched_selecting_decode_gives_the_one_request_tokens(batch):
    max_model_len = tiny.STACK_TOKENS + 8
    prompts = [5 + (r * 7) % 13 for r in range(batch)]
    bound = bypass_max_context(
        int(tiny._stack_text_config().index_topk), int(tiny.MLA_INDEX_KPOOL)
    )
    assert max_model_len > bound, "this regime must select"
    world = _world(batch, max_model_len=max_model_len, prompts=prompts,
                   window_blocks=-(-max_model_len // PAGE))
    _batched_against_single(world, dense=False)


def test_a_batched_dense_decode_gives_the_one_request_tokens():
    batch = 4
    max_model_len = bypass_max_context(
        int(tiny._stack_text_config().index_topk), int(tiny.MLA_INDEX_KPOOL)
    )
    # A prefill must complete at least one pool of four.
    prompts = [4, 5, 6, 7]
    assert max(prompts) + DECODE_STEPS <= max_model_len
    world = _world(batch, max_model_len=max_model_len, prompts=prompts,
                   window_blocks=DENSE_WINDOW_BLOCKS)
    _batched_against_single(world, dense=True)


def test_a_padded_decode_serves_the_padding_row_from_the_null_block_and_an_idle_slot(monkeypatch):
    """Three requests in the bucket of four: the requests' answers and state are unchanged."""
    monkeypatch.setenv("VLLM_NEURON_GLM5NEXT_STATE_BANKS", "0")  # this test pins the per-request VIEW form
    batch, bucket = 3, 4
    max_model_len = tiny.STACK_TOKENS + 8
    world = _world(batch, max_model_len=max_model_len, prompts=[6, 9, 11],
                   window_blocks=-(-max_model_len // PAGE), slots=8)
    slots = [world.runner._glm5next_request_slot_table[r] for r in world.req_ids]
    snapshot = _snapshot(world)
    idle_before = {
        index: side["pool_cache"].clone()
        for index, side in enumerate(world.runner._glm5next_side_cache_set) if side
    }

    carriers: list = []
    padded = _step(world, list(range(batch)),
                   torch.tensor(world.first + [0] * (bucket - batch)),
                   cached=list(world.lengths), sampling=list(range(batch)),
                   real=[1] * batch, out=carriers)
    padded_state = _owned_state(world, slots)
    carrier = carriers[0][0]
    assert int(carrier["block_table_row"].shape[1]) == bucket
    assert carrier["block_table_row"][:, batch].tolist() == (
        [NULL_BLOCK_ID] + [-1] * (world.window_blocks - 1)
    )
    assert int(carrier["latent_slots"][batch]) == NULL_BLOCK_ID * PAGE
    assert int(carrier["position"][batch]) == 0 and int(carrier["seq_lens"][batch]) == 1
    side = world.runner._glm5next_side_cache_set[0]
    assert carrier["tail"][batch].data_ptr() == side["pad_tail"][0].data_ptr()
    stride = side["pool_cache"][0].numel() * side["pool_cache"].element_size()
    pad_slot = (carrier["pool_cache"][batch].data_ptr() - side["pool_cache"].data_ptr()) // stride
    assert pad_slot not in slots, (pad_slot, slots)
    # The padding row's pooled store took only its trash row.
    for index, before in idle_before.items():
        after = world.runner._glm5next_side_cache_set[index]["pool_cache"]
        assert torch.equal(after[pad_slot, :-1], before[pad_slot, :-1])

    _restore(world, snapshot)
    single = torch.cat([
        _step(world, [r], torch.tensor([world.first[r]]), cached=[world.lengths[r]],
              sampling=[0])
        for r in range(batch)
    ])
    torch.testing.assert_close(padded, single, rtol=0.0,
                               atol=LOGIT_RTOL * float(single.abs().max()))
    assert padded.argmax(-1).tolist() == single.argmax(-1).tolist()
    _assert_state_equal(padded_state, _owned_state(world, slots), "padded B=3 in 4",
                        exact_layers=1)
