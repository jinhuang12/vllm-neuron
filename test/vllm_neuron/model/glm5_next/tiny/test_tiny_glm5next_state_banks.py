# SPDX-License-Identifier: Apache-2.0
"""Bank-form decode at B=64: whole state banks and slot tensors, written back in place.

At ``B > 1`` the runner's carrier builder hands each layer its whole banks and one ``[B]``
slot tensor instead of ``B`` per-request views (``test_glm5next_state_banks`` reads the
carriers themselves). Read here, on the real consumers:

1. the sparse (DSA) family on the tiny root and the recurrent (KDA) family on real
   ``Glm5NextKDAAttention`` modules at the TP=64 geometry, both driven through
   ``_glm5next_model_kwargs``: 64 requests whose slots are scattered over 128
   (``{0, 5, 17, 63, 127, ...}``), decoded together for two steps, leave every bank row
   as the one-request steps leave it, and every slot no request holds untouched;
2. a padded step: the padding rows name the scratch slot on the sparse family (the last
   store, past the engine's concurrency bound) and idle slots on the recurrent family;
   a request that holds a slot but is not scheduled this step keeps its ring and state;
3. the write-back proof: the consumers are traced with Dynamo and the backend's own
   aliasing and in-place rewrite passes are run on the graph. Every bank placeholder is
   an aliased output (``io_map``) and the rewritten, out-of-place graph's aliased outputs
   equal the banks the eager step leaves -- the write-back is the whole bank, not a view
   the backend would drop;
4. request removal and re-admission: a finished request frees its slot, a new request
   takes it with fresh state, and a preempted request re-prefilled from position 0
   continues correctly; every request's state matches the one-request reference.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/model/glm5_next/tiny/test_tiny_glm5next_state_banks.py
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch._dynamo as dynamo

from vllm_neuron.vllm.worker import glm5next_state_banks as runner_side
from vllm_neuron.vllm.worker.neuron_model_runner import NULL_BLOCK_ID, NeuronModelRunner

from test.vllm_neuron.model.glm5_next import test_kda_layer as layer_half
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_batch_decode as dsa
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_batch_kda as kda
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_e2e as e2e
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_forward as tiny

pytestmark = [pytest.mark.fast, pytest.mark.forked]

BATCH = 64
CAPACITY = 128
DECODE_STEPS = 2
SEED = 20261007


def _scattered_slots(count: int, capacity: int) -> list[int]:
    """``count`` distinct slots over ``capacity``, non-contiguous, with the corners in."""
    fixed = [0, 5, 17, 63, capacity - 1]
    gen = torch.Generator().manual_seed(SEED)
    rest = [s for s in torch.randperm(capacity, generator=gen).tolist() if s not in fixed]
    slots = fixed + rest[: count - len(fixed)]
    gen2 = torch.Generator().manual_seed(SEED + 1)
    order = torch.randperm(len(slots), generator=gen2).tolist()
    return [slots[i] for i in order]


@pytest.fixture(autouse=True)
def _bank_form_on(monkeypatch):
    monkeypatch.delenv(runner_side.STATE_BANKS_ENV, raising=False)
    monkeypatch.delenv(kda.layer_half._fused_seam().FUSED_DECODE_ENV, raising=False)


# ── the sparse family on the tiny root ───────────────────────────────────────


def _dsa_world(batch: int, slot_map: list[int], *, capacity: int, prompts: list[int]):
    """``dsa._world`` with the request slots seeded before the prefills."""
    e2e._require_cpu_mode()
    max_model_len = tiny.STACK_TOKENS + 8
    window_blocks = -(-max_model_len // dsa.PAGE)
    fixture = e2e._fixture()
    root = fixture["root"]
    head = torch.randn(
        tiny.STACK_VOCAB_SIZE, int(root.text_config.hidden_size),
        generator=torch.Generator().manual_seed(dsa.SEED),
    )
    root.lm_head_weight = torch.nn.Parameter(head.to(torch.bfloat16), requires_grad=False)
    per_request = -(-max_model_len // dsa.PAGE)
    blocks = 1 + batch * per_request
    caches = {}
    for spec in root.get_kv_spec().layers:
        shape = (blocks, int(spec.num_kv_heads), dsa.PAGE, int(spec.head_size))
        caches[spec.name] = [
            torch.zeros(shape, dtype=spec.dtype) for _ in range(1 if spec.latent_kv else 2)
        ]
    root.bind_kv_cache(caches)
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.input_batch = SimpleNamespace(req_ids=[])
    runner.model = root
    runner.max_model_len = int(max_model_len)
    runner.max_num_reqs = int(capacity)
    tables = [
        [1 + r * per_request + k for k in range(per_request)]
        + [NULL_BLOCK_ID] * (window_blocks - per_request)
        for r in range(batch)
    ]
    world = SimpleNamespace(
        root=root, runner=runner, caches=caches, tables=tables,
        window_blocks=window_blocks, req_ids=[f"req-{r}" for r in range(batch)],
        lengths=list(prompts),
    )
    # Allocate the side caches (which resets the slot table), then seed the table.
    runner._glm5next_live_side_caches(root.glm5next_layer_banks)
    runner._glm5next_request_slot_table = dict(zip(world.req_ids, slot_map))
    gen = torch.Generator().manual_seed(dsa.SEED + batch)
    first = []
    for r in range(batch):
        prompt = torch.randint(0, tiny.STACK_VOCAB_SIZE, (prompts[r],), generator=gen)
        logits = dsa._step(world, [r], prompt, cached=[0], sampling=[prompts[r] - 1])
        first.append(int(logits[-1].float().argmax()))
    world.first = first
    return world


def _unowned_side_state(world, owned: set[int]):
    sides = world.runner._glm5next_side_cache_set
    out = {}
    for index, side in enumerate(sides):
        if not side:
            continue
        free = [s for s in range(int(side["pool_cache"].shape[0])) if s not in owned]
        for key in ("pool_cache", "tail"):
            out[(index, key)] = side[key][free].clone()
    return out


def test_a_b64_bank_form_decode_over_scattered_slots_matches_the_one_request_steps():
    slot_map = _scattered_slots(BATCH, CAPACITY)
    assert {0, 5, 17, 63, CAPACITY - 1} <= set(slot_map) and slot_map != sorted(slot_map)
    prompts = [5 + (r * 7) % 13 for r in range(BATCH)]
    world = _dsa_world(BATCH, slot_map, capacity=CAPACITY, prompts=prompts)
    assert [world.runner._glm5next_request_slot_table[r] for r in world.req_ids] == slot_map
    snapshot = dsa._snapshot(world)
    untouched_before = _unowned_side_state(world, set(slot_map))

    carriers: list = []
    batched, fed = [], list(world.first)
    for step in range(DECODE_STEPS):
        logits = dsa._step(world, list(range(BATCH)), torch.tensor(fed),
                           cached=[n + step for n in world.lengths],
                           sampling=list(range(BATCH)), out=carriers)
        batched.append(logits)
        fed = logits.argmax(-1).tolist()
    batched_state = dsa._owned_state(world, slot_map)
    untouched_after = _unowned_side_state(world, set(slot_map))

    # The bank form reached the layers: the banks themselves and the scattered slots.
    sides = world.runner._glm5next_side_cache_set
    for index, carrier in enumerate(carriers[0]):
        # The pooled store goes flat: the bank's own storage, [slots * rows, dim].
        pool, flat = sides[index]["pool_cache"], carrier["pool_cache"]
        assert flat.untyped_storage().data_ptr() == pool.untyped_storage().data_ptr()
        assert flat.storage_offset() == 0
        assert tuple(flat.shape) == (pool.shape[0] * pool.shape[1], pool.shape[2])
        assert carrier["tail"] is sides[index]["tail"]
        assert carrier["state_slots"].tolist() == slot_map
        assert int(sides[index]["pool_cache"].shape[0]) == CAPACITY + 1
    for key, before in untouched_before.items():
        assert torch.equal(untouched_after[key], before), f"{key}: an unowned slot changed"

    dsa._restore(world, snapshot)
    single, fed = [], list(world.first)
    for step in range(DECODE_STEPS):
        rows = [
            dsa._step(world, [r], torch.tensor([fed[r]]), cached=[world.lengths[r] + step],
                      sampling=[0])
            for r in range(BATCH)
        ]
        logits = torch.cat(rows)
        single.append(logits)
        fed = logits.argmax(-1).tolist()
    for step, (got, want) in enumerate(zip(batched, single)):
        peak = float(want.abs().max())
        torch.testing.assert_close(got, want, rtol=0.0, atol=dsa.LOGIT_RTOL * peak)
        assert got.argmax(-1).tolist() == want.argmax(-1).tolist(), f"step {step}"
    dsa._assert_state_equal(batched_state, dsa._owned_state(world, slot_map),
                            "B=64 scattered slots", exact_layers=1)


def test_a_padded_bank_form_decode_serves_padding_from_the_scratch_slot_and_spares_an_unscheduled_ring():
    batch, bucket, capacity = 3, 4, 4
    # Four requests hold all four slots; the last one is not scheduled in the padded step.
    slot_map = [2, 0, 3, 1]
    prompts = [6, 9, 11, 7]
    world = _dsa_world(4, slot_map, capacity=capacity, prompts=prompts)
    scheduled = list(range(batch))
    snapshot = dsa._snapshot(world)
    sides = world.runner._glm5next_side_cache_set
    scratch = capacity
    unscheduled_slot = slot_map[3]
    before_unscheduled = {
        (i, k): sides[i][k][unscheduled_slot].clone()
        for i, side in enumerate(sides) if side for k in ("pool_cache", "tail")
    }

    carriers: list = []
    padded = dsa._step(world, scheduled,
                       torch.tensor([world.first[r] for r in scheduled] + [0] * (bucket - batch)),
                       cached=[world.lengths[r] for r in scheduled], sampling=scheduled,
                       real=[1] * batch, out=carriers)
    padded_state = dsa._owned_state(world, slot_map)
    carrier = carriers[0][0]
    assert carrier["state_slots"].tolist() == [slot_map[r] for r in scheduled] + [scratch]
    assert int(carrier["position"][batch]) == 0 and int(carrier["seq_lens"][batch]) == 1
    assert carrier["block_table_row"][:, batch].tolist() == (
        [NULL_BLOCK_ID] + [-1] * (world.window_blocks - 1)
    )
    for (i, k), before in before_unscheduled.items():
        assert torch.equal(sides[i][k][unscheduled_slot], before), (
            f"layer {i} {k}: the padding row wrote the unscheduled request's slot"
        )

    dsa._restore(world, snapshot)
    single = torch.cat([
        dsa._step(world, [r], torch.tensor([world.first[r]]), cached=[world.lengths[r]],
                  sampling=[0])
        for r in scheduled
    ])
    torch.testing.assert_close(padded, single, rtol=0.0,
                               atol=dsa.LOGIT_RTOL * float(single.abs().max()))
    assert padded.argmax(-1).tolist() == single.argmax(-1).tolist()
    dsa._assert_state_equal(padded_state, dsa._owned_state(world, slot_map),
                            "padded B=3 in 4", exact_layers=1)


# ── the recurrent family on real KDA modules ─────────────────────────────────


def _kda_banks(layers, slots: int) -> list[dict]:
    gen = torch.Generator().manual_seed(kda.SEED + 7)
    banks = []
    for index, attention in enumerate(layers):
        banks.append({
            "name": f"model.layers.{index}.attention",
            "family": "linear_attn",
            "state_slots": slots,
            "conv_state": torch.randn(
                (slots, *attention.kda_conv_state_shape), generator=gen
            ).to(attention.kda_conv_state_dtype),
            "recurrent_state": (
                torch.randn((slots, *attention.kda_recurrent_state_shape), generator=gen) * 0.1
            ).to(attention.kda_recurrent_state_dtype),
        })
    return banks


def _kda_world(batch: int, slot_map: list[int], *, capacity: int, prompts: list[int]):
    text_config, hidden, layers = kda._layers()
    banks = _kda_banks(layers, capacity)
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.input_batch = SimpleNamespace(req_ids=[])
    runner.model = SimpleNamespace(text_config=text_config, glm5next_layer_banks=banks)
    runner.max_model_len = max(prompts) + DECODE_STEPS + 4
    runner.max_num_reqs = capacity
    req_ids = [f"req-{r}" for r in range(batch)]
    runner._glm5next_live_side_caches(banks)
    runner._glm5next_request_slot_table = dict(zip(req_ids, slot_map))
    gen = torch.Generator().manual_seed(kda.SEED + batch)
    prompt_rows = [kda._grid(prompts[r], hidden, low=-1, high=1, exponent=-3, gen=gen)
                   for r in range(batch)]
    decodes = [kda._grid(batch, hidden, low=-1, high=1, exponent=-3, gen=gen)
               for _ in range(DECODE_STEPS)]
    for r in range(batch):
        kda._step(runner, banks, layers, req_ids=[req_ids[r]], rows=prompt_rows[r],
                  cached=[0], max_query_len=prompts[r])
    return SimpleNamespace(runner=runner, banks=banks, layers=layers, req_ids=req_ids,
                           decodes=decodes, hidden=hidden, lengths=list(prompts))


def _bank_rows(banks, slots):
    return [{k: bank[k][slots].clone() for k in ("conv_state", "recurrent_state")}
            for bank in banks]


def test_a_b64_kda_bank_form_decode_over_scattered_slots_advances_only_those_rows():
    slot_map = _scattered_slots(BATCH, CAPACITY)
    prompts = [3 + (r * 5) % 11 for r in range(BATCH)]
    world = _kda_world(BATCH, slot_map, capacity=CAPACITY, prompts=prompts)
    snapshot = kda._snapshot(world)
    unowned = [s for s in range(CAPACITY) if s not in slot_map]

    layer_half._reset_counters()
    batched_rows = []
    for step in range(DECODE_STEPS):
        out, carriers = kda._step(
            world.runner, world.banks, world.layers, req_ids=world.req_ids,
            rows=world.decodes[step], cached=[n + step for n in world.lengths],
            max_query_len=1,
        )
        batched_rows.append(out)
        assert carriers[0]["conv_state"] is world.banks[0]["conv_state"]
        assert carriers[0]["recurrent_state"] is world.banks[0]["recurrent_state"]
        assert carriers[0]["state_slots"].tolist() == slot_map
    counts = layer_half._read_counters()
    assert counts["fused"] == (kda.LAYERS * DECODE_STEPS, 0), counts
    batched_owned = _bank_rows(world.banks, slot_map)
    for index, (after, before) in enumerate(zip(world.banks, snapshot[0])):
        for key in ("conv_state", "recurrent_state"):
            assert torch.equal(kda._bytes(after[key][unowned]), kda._bytes(before[key][unowned])), (
                f"layer {index} {key}: a slot no request holds changed"
            )

    kda._restore(world, snapshot)
    single_rows = []
    for step in range(DECODE_STEPS):
        rows = []
        for r, rid in enumerate(world.req_ids):
            out, _ = kda._step(
                world.runner, world.banks, world.layers, req_ids=[rid],
                rows=world.decodes[step][r : r + 1], cached=[world.lengths[r] + step],
                max_query_len=1,
            )
            rows.append(out)
        single_rows.append(torch.cat(rows, dim=1))
    single_owned = _bank_rows(world.banks, slot_map)
    for index, (a, b) in enumerate(zip(batched_owned, single_owned)):
        for key in ("conv_state", "recurrent_state"):
            assert torch.equal(kda._bytes(a[key]), kda._bytes(b[key])), (
                f"layer {index} {key} differs between the bank-form and one-at-a-time decode"
            )
    for step in range(DECODE_STEPS):
        torch.testing.assert_close(batched_rows[step], single_rows[step],
                                   rtol=kda.OUT_RTOL, atol=kda.OUT_ATOL)


def test_a_padded_kda_bank_form_decode_leaves_the_unscheduled_requests_slot_untouched():
    batch, bucket, capacity = 3, 4, 4
    slot_map = [2, 0, 3, 1]
    prompts = [9, 3, 11, 6]
    world = _kda_world(4, slot_map, capacity=capacity, prompts=prompts)
    snapshot = kda._snapshot(world)
    scheduled = world.req_ids[:batch]
    rows = torch.cat([world.decodes[0][:batch], torch.zeros(bucket - batch, world.hidden)])
    out, carriers = kda._step(
        world.runner, world.banks, world.layers, req_ids=scheduled, rows=rows,
        cached=[world.lengths[r] for r in range(batch)], max_query_len=1, real=[1] * batch,
    )
    carrier = carriers[0]
    # The padding row names the one idle slot: the unscheduled request's, which it
    # must hand back unchanged (the kernel's masked step is the identity).
    assert carrier["state_slots"].tolist()[:batch] == slot_map[:batch]
    pad_slot = carrier["state_slots"].tolist()[batch]
    assert pad_slot == slot_map[3]
    assert carrier["real_tokens"].reshape(-1).tolist() == [1, 1, 1, 0]
    for index, (after, before) in enumerate(zip(world.banks, snapshot[0])):
        for key in ("conv_state", "recurrent_state"):
            assert torch.equal(kda._bytes(after[key][pad_slot]), kda._bytes(before[key][pad_slot])), (
                f"layer {index} {key}: the padding row changed the unscheduled request's slot"
            )
    padded = _bank_rows(world.banks, slot_map[:batch])
    kda._restore(world, snapshot)
    single = []
    for r, rid in enumerate(scheduled):
        row, _ = kda._step(world.runner, world.banks, world.layers, req_ids=[rid],
                           rows=world.decodes[0][r : r + 1], cached=[world.lengths[r]],
                           max_query_len=1)
        single.append(row)
    reference = _bank_rows(world.banks, slot_map[:batch])
    for index, (a, b) in enumerate(zip(padded, reference)):
        for key in ("conv_state", "recurrent_state"):
            assert torch.equal(kda._bytes(a[key]), kda._bytes(b[key])), (index, key)
    torch.testing.assert_close(out[:, :batch], torch.cat(single, dim=1),
                               rtol=kda.OUT_RTOL, atol=kda.OUT_ATOL)


# ── the write-back proof through the backend's own passes ────────────────────


def _trace_through_the_backend_passes(fn, kwargs: dict):
    """Dynamo-trace ``fn(**kwargs)``, run the aliasing and in-place passes, keep both."""
    from libtorch_neuronx_lite.fx_passes.aliasing_pass import AliasingOutputRewritePass
    from libtorch_neuronx_lite.fx_passes.inplace_rewrite_pass import InPlaceToOutOfPlacePass

    kept: dict = {}

    def backend(gm, example_inputs):
        placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
        rewritten, meta = AliasingOutputRewritePass().run(gm)
        rewritten, _ = InPlaceToOutOfPlacePass().run(rewritten)
        kept.update(gm=rewritten, io_map=dict(meta["io_map"]),
                    original_output_count=meta["original_output_count"],
                    example_inputs=list(example_inputs), placeholders=placeholders)
        return rewritten.forward

    dynamo.reset()
    compiled = torch.compile(fn, backend=backend, dynamic=False)
    out = compiled(**kwargs)
    dynamo.reset()
    return out, kept


def _aliased_outputs_for(kept: dict, banks: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Run the rewritten (out-of-place) graph on the example inputs; return each bank's output."""
    inputs = kept["example_inputs"]
    outputs = kept["gm"](*inputs)
    if not isinstance(outputs, (tuple, list)):
        outputs = (outputs,)
    by_input = {}
    for output_index, input_index in kept["io_map"].items():
        by_input[input_index] = outputs[output_index]
    found = {}
    for name, bank in banks.items():
        matches = [i for i, inp in enumerate(inputs)
                   if torch.is_tensor(inp) and inp.untyped_storage().data_ptr() == bank.untyped_storage().data_ptr()
                   and inp.storage_offset() == bank.storage_offset() and tuple(inp.shape) == tuple(bank.shape)]
        assert len(matches) == 1, (name, matches)
        assert matches[0] in by_input, (
            f"{name} is graph input {matches[0]} but no output aliases it: "
            f"io_map={kept['io_map']}"
        )
        found[name] = by_input[matches[0]]
    return found


def test_the_kda_bank_write_back_is_a_whole_bank_aliased_output():
    slot_map = _scattered_slots(BATCH, CAPACITY)
    prompts = [3 + (r * 5) % 11 for r in range(BATCH)]
    world = _kda_world(BATCH, slot_map, capacity=CAPACITY, prompts=prompts)
    _, carriers = kda._step(
        world.runner, world.banks, world.layers, req_ids=world.req_ids,
        rows=world.decodes[0], cached=list(world.lengths), max_query_len=1,
    )
    kda._restore(world, kda._snapshot(world))  # the step above advanced the banks once
    layer, carrier = world.layers[0], dict(carriers[0])
    rows = world.decodes[1]
    # The eager reference on copies of the banks.
    conv_ref = carrier["conv_state"].clone()
    rec_ref = carrier["recurrent_state"].clone()
    layer(rows, **dict(carrier, conv_state=conv_ref, recurrent_state=rec_ref), chunk_size=kda.CHUNK)
    # The traced step on the live banks, through the backend's passes.
    out, kept = _trace_through_the_backend_passes(
        lambda **kw: layer(rows, **kw, chunk_size=kda.CHUNK), carrier
    )
    aliased = _aliased_outputs_for(
        kept, {"conv_state": carrier["conv_state"], "recurrent_state": carrier["recurrent_state"]}
    )
    for name, reference in (("conv_state", conv_ref), ("recurrent_state", rec_ref)):
        assert tuple(aliased[name].shape) == tuple(reference.shape), name
        assert torch.equal(kda._bytes(aliased[name].to(reference.dtype)), kda._bytes(reference)), (
            f"{name}: the aliased output is not the bank the eager step leaves"
        )


def test_the_dsa_bank_write_back_is_a_whole_bank_aliased_output():
    slot_map = _scattered_slots(BATCH, CAPACITY)
    prompts = [5 + (r * 7) % 13 for r in range(BATCH)]
    world = _dsa_world(BATCH, slot_map, capacity=CAPACITY, prompts=prompts)
    carriers: list = []
    dsa._step(world, list(range(BATCH)), torch.tensor(world.first),
              cached=list(world.lengths), sampling=list(range(BATCH)), out=carriers)
    snapshot = dsa._snapshot(world)
    layer_index = 0
    carrier = dict(carriers[0][layer_index])
    carrier.pop("is_prefill", None)
    attention = world.root.model.layers[layer_index].self_attn
    gen = torch.Generator().manual_seed(SEED + 3)
    normed = torch.randn(BATCH, int(world.root.text_config.hidden_size), generator=gen).to(
        carrier["latent_cache"].dtype
    )
    # Eager reference on copies of the three banks this layer writes.
    copies = {k: carrier[k].clone() for k in ("latent_cache", "pool_cache", "tail")}
    attention(normed, **dict(carrier, **copies))
    dsa._restore(world, snapshot)
    _, kept = _trace_through_the_backend_passes(lambda **kw: attention(normed, **kw), carrier)
    aliased = _aliased_outputs_for(kept, {k: carrier[k] for k in copies})
    for name, reference in copies.items():
        assert tuple(aliased[name].shape) == tuple(reference.shape), name
        assert torch.equal(aliased[name].to(reference.dtype), reference), (
            f"{name}: the aliased output is not the bank the eager step leaves"
        )


# ── the warmup's synthetic step in the bank form ─────────────────────────────


def test_a_synthetic_bank_form_decode_names_slots_zero_to_b_and_takes_no_claim():
    """The warmup's B rows are served from slots 0..B-1 on every bank, none claimed."""
    from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_batch_capture as capture

    _, runner, _ = capture._capture_runner()
    kwargs = runner._build_decode_synthetic_inputs(4, compiled_graph_input=True)
    converted = runner._glm5next_model_kwargs(kwargs)
    banks = runner.model.glm5next_layer_banks
    sides = runner._glm5next_side_cache_set
    read = 0
    for index, carrier in enumerate(converted["layer_carriers"]):
        assert carrier["state_slots"].tolist() == [0, 1, 2, 3], (index, carrier["state_slots"])
        if "tail" in carrier:
            assert carrier["tail"] is sides[index]["tail"]
            # The pooled store goes flat: the bank's own storage, [slots * rows, dim].
            pool, flat = sides[index]["pool_cache"], carrier["pool_cache"]
            assert flat.untyped_storage().data_ptr() == pool.untyped_storage().data_ptr()
            assert flat.storage_offset() == 0
            assert tuple(flat.shape) == (pool.shape[0] * pool.shape[1], pool.shape[2])
            read += 1
        else:
            assert carrier["conv_state"] is banks[index]["conv_state"]
            assert carrier["recurrent_state"] is banks[index]["recurrent_state"]
    assert read == tiny.STACK_LAYERS
    assert runner._glm5next_request_slot_table == {}
    assert runner._glm5next_side_cache_positions == {}


# ── removal, re-admission and preemption keep every slot consistent ──────────


def test_removal_readmission_and_preemption_keep_every_requests_kda_state_consistent():
    capacity = 4
    world = _kda_world(4, [0, 1, 2, 3], capacity=capacity, prompts=[9, 3, 11, 6])
    runner, banks, layers = world.runner, world.banks, world.layers
    gen = torch.Generator().manual_seed(SEED + 9)
    grid = lambda n: kda._grid(n, world.hidden, low=-1, high=1, exponent=-3, gen=gen)  # noqa: E731
    lengths = dict(zip(world.req_ids, world.lengths))
    decode_inputs = [grid(4) for _ in range(4)]
    reference_banks = [{k: v.clone() for k, v in bank.items() if torch.is_tensor(v)} for bank in banks]
    # The runner's own request bookkeeping after the four prefills: the slot table and
    # the per-slot sequence cursor a continuing step must find.
    reference_table = dict(runner._glm5next_request_slot_table)
    reference_cursors = dict(runner._glm5next_side_cache_positions)

    def decode(ids, rows):
        kda._step(runner, banks, layers, req_ids=ids, rows=rows,
                  cached=[lengths[i] for i in ids], max_query_len=1)
        for i in ids:
            lengths[i] += 1

    # Step 0: all four together. Then req-1 finishes and frees slot 1.
    decode(world.req_ids, decode_inputs[0])
    runner._glm5next_note_finished_requests({"req-1"})
    # Step 1: req-4 is admitted; its prefill takes the freed slot and opens fresh state.
    new_prompt = grid(5)
    kda._step(runner, banks, layers, req_ids=["req-4"], rows=new_prompt, cached=[0],
              max_query_len=5)
    lengths["req-4"] = 5
    assert runner._glm5next_request_slot_table["req-4"] == 1
    assert "req-1" not in runner._glm5next_request_slot_table
    live = ["req-0", "req-2", "req-3", "req-4"]
    decode(live, decode_inputs[1])
    # Step 2: req-2 is preempted (its slot freed) and resumes from computed length 0: a
    # fresh prefill of its whole prompt into whatever slot is free.
    runner._glm5next_note_finished_requests({"req-2"})
    resumed = grid(lengths["req-2"])
    kda._step(runner, banks, layers, req_ids=["req-2"], rows=resumed, cached=[0],
              max_query_len=int(resumed.shape[0]))
    assert runner._glm5next_request_slot_table["req-2"] == 2
    decode(live, decode_inputs[2])
    final = {rid: {k: bank[k][runner._glm5next_request_slot_table[rid]].clone()
                   for k in ("conv_state", "recurrent_state")}
             for rid in live for bank in banks[:1]}

    # Reference: each live request alone, in its own slot, over the same inputs.
    for bank, saved in zip(banks, reference_banks):
        for key, value in saved.items():
            bank[key].copy_(value)
    runner._glm5next_request_slot_table = dict(reference_table)
    runner._glm5next_side_cache_positions = dict(reference_cursors)
    ref_lengths = dict(zip(world.req_ids, world.lengths))

    def one(rid, rows, cached, max_query_len):
        kda._step(runner, banks, layers, req_ids=[rid], rows=rows, cached=[cached],
                  max_query_len=max_query_len)

    index = {rid: i for i, rid in enumerate(world.req_ids)}
    for rid in world.req_ids:  # step 0, one at a time
        one(rid, decode_inputs[0][index[rid]: index[rid] + 1], ref_lengths[rid], 1)
        ref_lengths[rid] += 1
    runner._glm5next_note_finished_requests({"req-1"})
    one("req-4", new_prompt, 0, 5)
    ref_lengths["req-4"] = 5
    live_index = {rid: i for i, rid in enumerate(live)}
    for rid in live:
        one(rid, decode_inputs[1][live_index[rid]: live_index[rid] + 1], ref_lengths[rid], 1)
        ref_lengths[rid] += 1
    runner._glm5next_note_finished_requests({"req-2"})
    one("req-2", resumed, 0, int(resumed.shape[0]))
    for rid in live:
        one(rid, decode_inputs[2][live_index[rid]: live_index[rid] + 1], ref_lengths[rid], 1)
        ref_lengths[rid] += 1
    for rid in live:
        slot = runner._glm5next_request_slot_table[rid]
        for key in ("conv_state", "recurrent_state"):
            assert torch.equal(kda._bytes(final[rid][key]), kda._bytes(banks[0][key][slot])), (
                f"{rid} {key}: the bank-form run and the one-request run disagree"
            )
