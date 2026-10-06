# SPDX-License-Identifier: Apache-2.0
"""Concurrent KDA decode at B in {2, 4}: one fused launch per layer, banks advanced exactly.

Real ``Glm5NextKDAAttention`` modules at the TP=64 geometry (one head of 128 per rank)
are driven through the runner's own carrier builder, ``_glm5next_model_kwargs``, the way a
served step reaches them:

1. every request is prefilled alone, at its own prompt length, into its own bank slot;
2. the banks are snapshotted;
3. the batched arm decodes all B requests together for two steps (one row per request);
4. the reference arm restores the snapshot and decodes the same requests one at a time
   (B one-request steps per decode step), the path wave 1 served.

Both arms must leave every bank row bit-identical. The batched arm must take ONE
``kda_fused_decode`` dispatch per layer per step, where the reference takes one per
request.

Why the inputs sit on a coarse grid. torch's CPU matmul takes a GEMV for one row and a
GEMM for several, and the two round differently in the last bit, so a [B, 4096]
projection is not bit-equal to B [1, 4096] ones. The inputs and the projection weights are
therefore small multiples of powers of two whose every partial sum is exact in fp32, so
the kernel receives the same operands on both arms and the banks can be compared bit for
bit. The modules are driven one by one on the same input rows (no layer norm or residual
between them), which keeps every module's input on the grid. The attention OUTPUT passes
through ``o_proj`` on rows off the grid, so it is compared at fp32 reassociation
tolerance.

A padded batch (3 requests in the bucket of 4) is read too: the padding row is served
from a slot no request in the step holds, and that slot's rows come back unchanged.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/model/glm5_next/tiny/test_tiny_glm5next_batch_kda.py
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

from test.vllm_neuron.model.glm5_next import test_kda_layer as layer_half

pytestmark = [pytest.mark.fast, pytest.mark.forked]

TP_WORLD_SIZE = layer_half.TP_WORLD_SIZE
LAYERS = 2
CHUNK = layer_half.DECLARED_CHUNK
#: One prompt length per request, all different, so no two requests share a state.
PROMPTS = (9, 3, 11, 6)
DECODE_STEPS = 2
#: The engine's concurrency bound, and so the banks' slot count. More than the largest
#: batch, so the padded case has a slot no request owns.
MAX_NUM_SEQS = 8
PAGE = layer_half.HYBRID_BLOCK_SIZE
SEED = 20261006
#: ``o_proj`` on off-grid rows: a [B, 128] GEMM against B [1, 128] GEMVs.
OUT_RTOL, OUT_ATOL = 1e-5, 1e-6


def _grid(*shape, low: int, high: int, exponent: int, gen) -> torch.Tensor:
    """Integers in [low, high] times 2**exponent: exact in fp32 under any summation order."""
    return torch.randint(low, high + 1, shape, generator=gen).to(torch.float32) * (
        2.0**exponent
    )


def _layers():
    """``LAYERS`` attention modules with grid projection weights (see the docstring)."""
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    text_config = Glm5NextTextConfig()
    hidden = int(text_config.hidden_size)
    linear = text_config.linear_attn_config
    heads = int(linear["num_heads"]) // TP_WORLD_SIZE
    head_dim = int(linear["head_dim"])
    kernel = int(linear["short_conv_kernel_size"])
    width = heads * head_dim
    torch.manual_seed(SEED)
    gen = torch.Generator().manual_seed(SEED + 1)
    layers = []
    for index in range(LAYERS):
        weights = layer_half._make_weights(hidden, heads, head_dim, kernel)
        # x (multiples of 2**-3, |x| <= 2**-3) times these (multiples of 2**-8,
        # |w| <= 2**-5) over 4096 terms stays below 2**4 on a 2**-11 grid: exact.
        for name, rows_ in (("q_proj_weight", width), ("k_proj_weight", width),
                            ("v_proj_weight", width), ("b_proj_weight", heads),
                            ("f_a_proj_weight", head_dim), ("g_a_proj_weight", head_dim)):
            weights[name] = _grid(rows_, hidden, low=-8, high=8, exponent=-8, gen=gen)
        # The low-rank second stage: a 2**-11 grid below 2**4, times multiples of
        # 2**-4 at most 2**-4, over 128 terms: below 2**7 on a 2**-15 grid, exact.
        for name in ("f_b_proj_weight", "g_b_proj_weight"):
            weights[name] = _grid(width, head_dim, low=-1, high=1, exponent=-4, gen=gen)
        module = layer_half._impl().Glm5NextKDAAttention(text_config, TP_WORLD_SIZE)
        for name, tensor in weights.items():
            if name != "input_layernorm_weight":
                setattr(module, name, nn.Parameter(tensor.clone(), requires_grad=False))
        layers.append(module)
    return text_config, hidden, layers


def _banks(layers) -> list[dict]:
    """The bank mapping ``bind_kv_cache`` leaves, one slot per admitted sequence."""
    gen = torch.Generator().manual_seed(SEED + 7)
    banks = []
    for index, attention in enumerate(layers):
        # Random rows, not zeros: a slot that is read where it should not be, or written
        # where it should not be, shows up against them.
        banks.append(
            {
                "name": f"model.layers.{index}.attention",
                "family": "linear_attn",
                "state_slots": MAX_NUM_SEQS,
                "conv_state": torch.randn(
                    (MAX_NUM_SEQS, *attention.kda_conv_state_shape), generator=gen
                ).to(attention.kda_conv_state_dtype),
                "recurrent_state": (
                    torch.randn(
                        (MAX_NUM_SEQS, *attention.kda_recurrent_state_shape), generator=gen
                    )
                    * 0.1
                ).to(attention.kda_recurrent_state_dtype),
            }
        )
    return banks


def _runner(text_config, banks) -> NeuronModelRunner:
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.input_batch = SimpleNamespace(req_ids=[])
    runner.model = SimpleNamespace(text_config=text_config, glm5next_layer_banks=banks)
    runner.max_model_len = max(PROMPTS) + DECODE_STEPS + 1
    runner.max_num_reqs = MAX_NUM_SEQS
    return runner


def _metadata(banks, *, rows: int, max_query_len: int, cached: list[int]) -> dict:
    """One entry per layer; ``cached`` holds one computed length per request."""
    table = torch.tensor([[1 + index] for index in range(len(cached))], dtype=torch.int32)
    entry = {
        "max_query_len": int(max_query_len),
        "block_size": PAGE,
        "max_blocks_per_seq": 1,
        "decode_token_threshold": 1,
        "host_block_table": table,
        "host_num_computed_tokens": torch.tensor(cached, dtype=torch.int32),
        "kv_segment_size": 0,
    }
    return {str(bank["name"]): entry for bank in banks}


def _step(runner, banks, layers, *, req_ids, rows, cached, max_query_len, real=None):
    """One step through the converter and the layers; returns the stack's output rows."""
    runner.input_batch.req_ids = list(req_ids)
    runner._glm5next_request_tokens = (
        None if real is None else np.array(real, dtype=np.int32)
    )
    converted = runner._glm5next_model_kwargs(
        {
            "input_ids": torch.zeros(int(rows.shape[0]), dtype=torch.long),
            "attn_metadata": _metadata(
                banks, rows=int(rows.shape[0]), max_query_len=max_query_len, cached=cached
            ),
            "sampling_positions": torch.zeros(1, dtype=torch.long),
        }
    )
    # Every module reads the same grid rows; their outputs are stacked.
    out = torch.stack(
        [layer(rows, **carrier, chunk_size=CHUNK)
         for layer, carrier in zip(layers, converted["layer_carriers"])]
    )
    return out, converted["layer_carriers"]


def _world(batch: int):
    text_config, hidden, layers = _layers()
    banks = _banks(layers)
    runner = _runner(text_config, banks)
    gen = torch.Generator().manual_seed(SEED + batch)
    prompts = [_grid(PROMPTS[r], hidden, low=-1, high=1, exponent=-3, gen=gen)
               for r in range(batch)]
    decodes = [_grid(batch, hidden, low=-1, high=1, exponent=-3, gen=gen)
               for _ in range(DECODE_STEPS)]
    req_ids = [f"req-{r}" for r in range(batch)]
    for r in range(batch):
        _step(runner, banks, layers, req_ids=[req_ids[r]], rows=prompts[r], cached=[0],
              max_query_len=PROMPTS[r])
    return SimpleNamespace(runner=runner, banks=banks, layers=layers, req_ids=req_ids,
                           decodes=decodes, hidden=hidden)


def _snapshot(world):
    return (
        [{k: v.clone() for k, v in bank.items() if torch.is_tensor(v)} for bank in world.banks],
        dict(world.runner._glm5next_side_cache_positions),
    )


def _restore(world, snapshot):
    tensors, positions = snapshot
    for bank, saved in zip(world.banks, tensors):
        for key, value in saved.items():
            bank[key].copy_(value)
    world.runner._glm5next_side_cache_positions = dict(positions)


def _bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(torch.uint8)


def _assert_banks_equal(left, right, label):
    for index, (a, b) in enumerate(zip(left, right)):
        for key in ("conv_state", "recurrent_state"):
            assert torch.equal(_bytes(a[key]), _bytes(b[key])), (
                f"{label}: layer {index} {key} differs between the batched and the "
                f"one-at-a-time decode"
            )


@pytest.mark.parametrize("batch", [2, 4])
def test_a_batched_kda_decode_advances_every_bank_row_as_one_request_steps_do(batch):
    world = _world(batch)
    slots = [world.runner._glm5next_request_slot_table[r] for r in world.req_ids]
    assert sorted(slots) == list(range(batch)), slots
    snapshot = _snapshot(world)
    lengths = list(PROMPTS[:batch])

    # Batched arm: all requests in one step.
    layer_half._reset_counters()
    batched_rows = []
    for step in range(DECODE_STEPS):
        out, carriers = _step(
            world.runner, world.banks, world.layers, req_ids=world.req_ids,
            rows=world.decodes[step], cached=[n + step for n in lengths], max_query_len=1,
        )
        batched_rows.append(out)
        assert len(carriers[0]["conv_state"]) == batch
    batched_counts = layer_half._read_counters()
    batched_banks = [{k: v.clone() for k, v in bank.items() if torch.is_tensor(v)}
                     for bank in world.banks]

    # Reference arm: the same steps, one request at a time.
    _restore(world, snapshot)
    layer_half._reset_counters()
    single_rows = []
    for step in range(DECODE_STEPS):
        rows = []
        for r, rid in enumerate(world.req_ids):
            out, _ = _step(
                world.runner, world.banks, world.layers, req_ids=[rid],
                rows=world.decodes[step][r : r + 1], cached=[lengths[r] + step],
                max_query_len=1,
            )
            rows.append(out)
        single_rows.append(torch.cat(rows, dim=1))
    single_counts = layer_half._read_counters()

    assert batched_counts["fused"] == (LAYERS * DECODE_STEPS, 0), batched_counts
    assert single_counts["fused"] == (LAYERS * DECODE_STEPS * batch, 0), single_counts
    for name in ("conv", "gate", "decode", "intra", "inter"):
        assert batched_counts[name] == (0, 0), (name, batched_counts)

    _assert_banks_equal(batched_banks, world.banks, f"B={batch}")
    for step in range(DECODE_STEPS):
        single = single_rows[step]
        torch.testing.assert_close(batched_rows[step], single, rtol=OUT_RTOL, atol=OUT_ATOL)
    # The slots no request holds were never touched.
    for index, (after, before) in enumerate(zip(world.banks, snapshot[0])):
        for slot in range(batch, MAX_NUM_SEQS):
            for key in ("conv_state", "recurrent_state"):
                assert torch.equal(_bytes(after[key][slot]), _bytes(before[key][slot])), (
                    f"layer {index} slot {slot} {key} belongs to no request and changed"
                )


def test_a_padded_kda_decode_serves_the_padding_row_from_an_idle_slot_and_leaves_it():
    """Three requests in the bucket of four: one padding row, whose slot is returned unchanged."""
    batch, bucket = 3, 4
    world = _world(batch)
    snapshot = _snapshot(world)
    lengths = list(PROMPTS[:batch])
    layer_half._reset_counters()
    rows = torch.cat([world.decodes[0], torch.zeros(bucket - batch, world.hidden)])
    assert rows.shape[0] == bucket
    out, carriers = _step(
        world.runner, world.banks, world.layers, req_ids=world.req_ids, rows=rows,
        cached=list(lengths), max_query_len=1, real=[1] * batch,
    )
    counts = layer_half._read_counters()
    assert counts["fused"] == (LAYERS, 0), counts
    carrier = carriers[0]
    assert len(carrier["conv_state"]) == bucket
    assert carrier["real_tokens"].reshape(-1).tolist() == [1] * batch + [0]
    assert carrier["row_mask"].reshape(-1).tolist() == [1.0] * batch + [0.0]
    # The padding row's views are a slot outside the batch, and its start is not zero,
    # so the kernel keeps the slot's carriers instead of opening them.
    pad_slot = next(
        slot for slot in range(MAX_NUM_SEQS)
        if carrier["conv_state"][batch].data_ptr()
        == world.banks[0]["conv_state"][slot].data_ptr()
    )
    owned = {world.runner._glm5next_request_slot_table[r] for r in world.req_ids}
    assert pad_slot not in owned, (pad_slot, owned)
    assert int(carrier["start_position"].reshape(-1)[batch]) != 0
    for index, (after, before) in enumerate(zip(world.banks, snapshot[0])):
        for slot in range(batch, MAX_NUM_SEQS):
            for key in ("conv_state", "recurrent_state"):
                assert torch.equal(_bytes(after[key][slot]), _bytes(before[key][slot])), (
                    f"layer {index} slot {slot} {key} was changed by the padding row"
                )

    # The three real rows match three one-request steps.
    padded_banks = [{k: v.clone() for k, v in bank.items() if torch.is_tensor(v)}
                    for bank in world.banks]
    _restore(world, snapshot)
    single = []
    for r, rid in enumerate(world.req_ids):
        row, _ = _step(world.runner, world.banks, world.layers, req_ids=[rid],
                       rows=world.decodes[0][r : r + 1], cached=[lengths[r]],
                       max_query_len=1)
        single.append(row)
    _assert_banks_equal(padded_banks, world.banks, "padded B=3 in 4")
    torch.testing.assert_close(out[:, :batch], torch.cat(single, dim=1), rtol=OUT_RTOL,
                               atol=OUT_ATOL)
