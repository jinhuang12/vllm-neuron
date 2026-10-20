# SPDX-License-Identifier: Apache-2.0
"""The runner's async-correction handoff under decode context parallelism.

The served async form (``VLLM_NEURON_GLM5NEXT_MTP_ASYNC=1``, method "mtp", ``k = 3``) at
``--decode-context-parallel-size 2``: each rank's sparse carriers take their latent slots
from the device correction (``_glm5next_async_correction`` ->
``functional/mtp/async_step.mtp_async_correct``), so the correction must be handed this
rank's DCP geometry and its slots must be the runner's own DCP slot function's
(``_dcp_owned_latent_slots``) at the true positions: the rows this rank owns land on its
local rows, the others on the null slot. Covered on the translator tests' tiny world (its
runner, its tables), at both ranks, on the verify form (the batch carrier) and the one-row
form (the one-ring carrier), and the warmup: it compiles the correction at the geometry the
served step launches, so a served DCP step compiles nothing.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/vllm/worker/test_async_correction_dcp.py
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_neuron.functional.mtp import async_step
from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner, _dcp_owned_latent_slots

from test.vllm_neuron.worker import test_mtp_async_runner as async_runner
from test.vllm_neuron.worker import test_mtp_translator as translator
from test.vllm_neuron.worker.test_mtp_async_runner import proposer  # noqa: F401  (fixture)

pytestmark = [pytest.mark.forked]

K = translator.K
T = translator.T
PAGE = translator.PAGE
DCP = 2


def _dcp_world(rank: int):
    """The translator's one-request sparse world, its runner made rank ``rank`` of a DCP-2
    group: the attributes ``NeuronModelRunner.__init__`` and ``cp_rank`` set, and the config
    fields the DCP indexer context reads (``indexer_context_per_rank``)."""
    world = translator._world(1)
    runner = world.runner
    runner._dcp_size = DCP
    runner.cp_world_size = DCP
    runner._cp_rank = rank
    runner.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=translator.MAX_MODEL_LEN),
        parallel_config=SimpleNamespace(decode_context_parallel_size=DCP),
        cache_config=SimpleNamespace(block_size=PAGE),
    )
    return world


def _want(world, true: int, width: int, rank: int) -> list[int]:
    """The runner's own DCP slots for the request's rows at the true position."""
    positions = torch.arange(width, dtype=torch.int64) + true
    return _dcp_owned_latent_slots(world.tables[0], positions, block_size=PAGE, dcp_size=DCP, cp_rank=rank)


@pytest.mark.parametrize("rank", range(DCP))
def test_a_verify_step_hands_the_carriers_this_ranks_dcp_slots(proposer, rank):
    """Three verify steps of one request, each corrected one step late; the rows cross a
    block boundary, so each rank owns some rows of a step and writes the null slot for the
    others. The synchronous translator's DCP mapping gives the same slots."""
    world = _dcp_world(rank)
    runner = world.runner
    async_runner._async_server(runner, proposer)
    runner._glm5next_async_carry = async_runner._carry(world.req_ids, prev_width=1, checkpoint_rows=[0])
    optimistic, rejected = world.lengths[0], 0
    owners = set()
    for kept in (2, 1, T):
        true = optimistic - rejected
        converted = translator._convert(world, [0], cached=[optimistic], tokens=T, real=[T])
        want = _want(world, true, T, rank)
        sync = NeuronModelRunner._glm5next_latent_slot_mapping(
            rows=[world.tables[0]], starts=[true], tokens=T, block_size=PAGE, reals=[T],
            device=torch.device("cpu"), dcp_size=DCP, cp_rank=rank,
        ).tolist()
        assert want == sync
        for carrier in translator._sparse(world, converted["layer_carriers"]):
            assert carrier["latent_slots"].tolist() == want, (kept, true)
            assert carrier["latent_slots"].dtype == torch.int64
            assert carrier["position"].tolist() == [true]
        owners |= {(true + t) // PAGE % DCP for t in range(T)}
        rejected = T - kept
        runner._glm5next_async_carry = async_runner._carry(world.req_ids, prev_width=T, checkpoint_rows=[kept - 1])
        optimistic += T
    assert owners == set(range(DCP)), "the steps cross blocks of both ranks"


@pytest.mark.parametrize("rank", range(DCP))
def test_a_one_row_step_hands_the_one_ring_carrier_this_ranks_dcp_slot(proposer, rank):
    """The spec-to-non-spec transition at both owners: the one-row step's slot is this rank's
    local row when it owns the position and the null slot when the other rank does."""
    world = _dcp_world(rank)
    runner = world.runner
    async_runner._async_server(runner, proposer)
    start = world.lengths[0]
    runner._glm5next_async_carry = async_runner._carry(world.req_ids, prev_width=1, checkpoint_rows=[0])
    translator._convert(world, [0], cached=[start], tokens=T, real=[T])
    owners = set()
    # The transition step pulls back the verify step's two rejected rows; the one-row steps
    # after it have nothing to pull back. Two blocks of them reach both owners.
    for step in range(2 * PAGE):
        transition = step == 0
        runner._glm5next_async_carry = async_runner._carry(
            world.req_ids, prev_width=T if transition else 1, checkpoint_rows=[1 if transition else 0]
        )
        cached = start + T + step
        true = start + 2 if transition else cached
        converted = translator._convert(world, [0], cached=[cached], tokens=1, real=[1], width=1)
        for carrier in translator._sparse(world, converted["layer_carriers"]):
            assert carrier["latent_slots"].tolist() == _want(world, true, 1, rank), step
            assert int(carrier["position"]) == true
            assert torch.is_tensor(carrier["tail"]), "the one-ring form"
        owners.add(true // PAGE % DCP)
    assert owners == set(range(DCP))


@pytest.fixture
def fresh_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


@pytest.mark.parametrize("rank", range(DCP))
def test_the_warmup_compiles_the_correction_at_the_served_dcp_geometry(proposer, fresh_dynamo, monkeypatch, rank):
    """The warmup launches the correction at every width pair; the served steps after it
    (a verify step, then the one-row step) compile nothing, and hand this rank's slots."""
    world = _dcp_world(rank)
    runner = world.runner
    async_runner._async_server(runner, proposer)
    compiles: list = []

    def backend(graph_module, example_inputs):
        compiles.append(tuple(tuple(value.shape) for value in example_inputs))
        return graph_module.forward

    monkeypatch.setattr(async_step, "_launches_in_a_graph", lambda tensor: True)
    launch = async_step.DeviceLaunch(backend)
    monkeypatch.setattr(runner, "_glm5next_async_launch", lambda: launch, raising=False)
    # The warmup forward's sampled row keeps one id (the rest are placeholders), so the
    # correction can pull back at every width pair.
    sampled = torch.tensor([[5] + [async_step.PLACEHOLDER_TOKEN_ID] * K], dtype=torch.int32)
    drafts = torch.zeros((1, K), dtype=torch.int32)
    runner._glm5next_async_warmup((sampled, drafts), batch_size=1, is_prefill=False)
    warmed = len(compiles)
    assert launch.calls["correct"] == 4, launch.calls
    start = world.lengths[0]
    runner._glm5next_async_carry = async_runner._carry(world.req_ids, prev_width=1, checkpoint_rows=[0])
    converted = translator._convert(world, [0], cached=[start], tokens=T, real=[T])
    for carrier in translator._sparse(world, converted["layer_carriers"]):
        assert carrier["latent_slots"].tolist() == _want(world, start, T, rank)
    runner._glm5next_async_carry = async_runner._carry(world.req_ids, prev_width=T, checkpoint_rows=[1])
    converted = translator._convert(world, [0], cached=[start + T], tokens=1, real=[1], width=1)
    for carrier in translator._sparse(world, converted["layer_carriers"]):
        assert carrier["latent_slots"].tolist() == _want(world, start + 2, 1, rank)
    assert launch.calls["correct"] == 6, launch.calls
    assert len(compiles) == warmed, ("a served DCP step compiled", compiles[warmed:])
