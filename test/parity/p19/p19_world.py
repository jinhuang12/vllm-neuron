# SPDX-License-Identifier: Apache-2.0
"""A minimal GLM-5.3-Flash runner for the request lifecycle suite: one sparse layer, one KDA layer.

Each step goes through the runner's own code, in the order a real step does: the state
update (``_update_states``) reads vLLM's ``SchedulerOutput``, the slot table hands out
slots for the requests in the persistent batch, and the position check opens or
continues each request's indexer ring. Only the persistent batch is a stand-in, and it
keeps only what the slot table reads from it: the ordered request ids.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput

from vllm_neuron.vllm.worker import neuron_model_runner
from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

#: vLLM's KV block for this model with the hybrid KV cache (128 tokens).
BLOCK = 128

#: The engine's longest sequence; it sizes the indexer's pooled store.
MAX_MODEL_LEN = 4096

#: GLM-5.3-Flash's indexer pool and key width (``index_kpool``, ``index_head_dim``).
INDEX_KPOOL = 4
INDEX_HEAD_DIM = 128

#: A prompt length that is not a block multiple, so a cursor cannot equal a cache hit.
PROMPT = 300


class PersistentBatch:
    """The runner's persistent batch, reduced to its ordered request ids.

    ``_update_states`` adds and removes requests here, and the slot table reads the
    step's request ids from ``req_ids`` (``_glm5next_batch_request_ids``). The other
    members are the ones ``_update_states`` writes for a running request; they hold
    nothing this suite reads.
    """

    def __init__(self) -> None:
        self.req_ids: list[str] = []
        self.num_computed_tokens_cpu: dict[int, int] = {}
        self.block_table = SimpleNamespace(append_row=lambda block_ids, row: None)
        self.prev_req_id_to_index = None

    @property
    def req_id_to_index(self) -> dict[str, int]:
        return {req_id: row for row, req_id in enumerate(self.req_ids)}

    def remove_request(self, req_id: str) -> None:
        if req_id in self.req_ids:
            self.req_ids.remove(req_id)

    def add_request(self, request) -> None:
        self.req_ids.append(request.req_id)

    def update_req_spec_token_ids(self, request, scheduled_spec_tokens) -> None:
        pass

    def condense(self) -> None:
        pass

    def refresh_metadata(self) -> None:
        pass


def banks(state_slots: int) -> list[dict]:
    """The two bank records ``bind_kv_cache`` leaves on the model: sparse first, then KDA."""
    latent = torch.zeros((8, 1, BLOCK, 16), dtype=torch.bfloat16)
    sparse = {"family": "self_attn", "latent_cache": latent.view(-1, 1, 16)}
    linear = {
        "family": "linear_attn",
        "state_slots": int(state_slots),
        "conv_state": torch.zeros((state_slots, 3, 8), dtype=torch.bfloat16),
        "recurrent_state": torch.zeros((state_slots, 1, 4, 4), dtype=torch.float32),
    }
    return [sparse, linear]


def runner(max_num_seqs: int) -> NeuronModelRunner:
    """A runner holding only what the state update, the slot table and the position check read."""
    built = NeuronModelRunner.__new__(NeuronModelRunner)
    built.max_num_reqs = int(max_num_seqs)
    built.max_model_len = MAX_MODEL_LEN
    built.model = SimpleNamespace(
        text_config=SimpleNamespace(index_kpool=INDEX_KPOOL, index_head_dim=INDEX_HEAD_DIM)
    )
    # Tensor-parallel rank 0, read from the cache so no process group is needed.
    built._glm5next_shadow_rank_cache = 0
    # What ``_update_states`` reads for a text-only generation model.
    built.requests = {}
    built.num_prompt_logprobs = {}
    built.input_batch = PersistentBatch()
    built.device = torch.device("cpu")
    built.is_pooling_model = False
    built.uses_mrope = False
    built.uses_xdrope_dim = 0
    built.use_async_scheduling = True
    built.speculative_config = None
    return built


@dataclass
class World:
    runner: NeuronModelRunner
    banks: list[dict]
    side: list[dict]


def world(monkeypatch, max_num_seqs: int = 2) -> World:
    """A runner, its banks (one KDA slot per sequence) and its live side caches.

    ``_update_states`` asks the pipeline-parallel group whether this is the last
    stage; a single-stage serve is, so that answer is patched in.
    """
    monkeypatch.setattr(
        neuron_model_runner, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True)
    )
    built = runner(max_num_seqs)
    stack = banks(max_num_seqs)
    side = built._glm5next_live_side_caches(stack)
    return World(built, stack, side)


def output(
    *,
    new: dict[str, int | tuple[int, int]] | None = None,
    running: dict[str, int] | None = None,
    resumed: dict[str, int] | None = None,
    finished: set[str] | frozenset[str] = frozenset(),
    preempted: set[str] | frozenset[str] | None = frozenset(),
) -> SchedulerOutput:
    """One ``SchedulerOutput`` as vLLM's scheduler builds it.

    ``new``: first-scheduled requests at computed length 0, id -> prompt tokens, or
    id -> ``(scheduled tokens, prompt tokens)`` for a first prefill segment.
    ``running``: requests continuing a decode, id -> computed length, one token each.
    ``resumed``: preempted requests scheduled again, id -> prompt tokens, at computed
    length 0 (vLLM resets a preempted request's computed length).
    ``finished``: ids finished (completed or aborted) since the last step.
    ``preempted``: ids preempted in this step (``preempted_req_ids``).
    """
    new = {
        req_id: tokens if isinstance(tokens, tuple) else (tokens, tokens)
        for req_id, tokens in (new or {}).items()
    }
    running = dict(running or {})
    resumed = dict(resumed or {})
    scheduled_new = [
        NewRequestData(
            req_id=req_id,
            prompt_token_ids=[1] * prompt,
            mm_features=[],
            sampling_params=None,
            pooling_params=None,
            block_ids=([],),
            num_computed_tokens=0,
            lora_request=None,
        )
        for req_id, (_, prompt) in new.items()
    ]
    cached_ids = list(running) + list(resumed)
    cached = CachedRequestData(
        req_ids=cached_ids,
        resumed_req_ids=set(resumed),
        new_token_ids=[],
        all_token_ids={},
        new_block_ids=[None] * len(running) + [([],)] * len(resumed),
        num_computed_tokens=list(running.values()) + [0] * len(resumed),
        num_output_tokens=[0] * len(cached_ids),
    )
    tokens = {
        **{req_id: scheduled for req_id, (scheduled, _) in new.items()},
        **{req_id: 1 for req_id in running},
        **resumed,
    }
    return SchedulerOutput(
        scheduled_new_reqs=scheduled_new,
        scheduled_cached_reqs=cached,
        num_scheduled_tokens=tokens,
        total_num_scheduled_tokens=sum(tokens.values()),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[0],
        finished_req_ids=set(finished),
        free_encoder_mm_hashes=[],
        preempted_req_ids=None if preempted is None else set(preempted),
    )


def step(w: World, scheduled: SchedulerOutput) -> dict[str, int]:
    """Serve one engine step up to its position checks; return each request's slot.

    The order is the runner's: ``execute_model`` runs the state update first and
    returns before building a step that schedules no token
    (``total_num_scheduled_tokens == 0``). A real step then takes the request ids
    from the persistent batch, takes their slots, checks each request's start
    position and records where its step ends (the cursor the step builder writes).
    """
    built = w.runner
    built._update_states(scheduled)
    if scheduled.total_num_scheduled_tokens == 0:
        return {}
    request_ids = built._glm5next_request_identities(synthetic=False)
    slots = built._glm5next_request_slots(
        w.banks, request_ids, synthetic=False, side_caches=w.side
    )
    for request_id, slot in zip(request_ids, slots):
        start = built.requests[request_id].num_computed_tokens
        built._glm5next_position_arm(
            int(slot), int(start), side_caches=w.side, is_prefill=start == 0
        )
        built._glm5next_side_cache_positions[int(slot)] = int(start) + int(
            scheduled.num_scheduled_tokens[request_id]
        )
    return {request_id: int(slot) for request_id, slot in zip(request_ids, slots)}


def synthetic_step(w: World) -> int:
    """A warmup or idle dummy step: slot 0, with the slot table neither read nor written."""
    (slot,) = w.runner._glm5next_request_slots(
        w.banks, [None], synthetic=True, side_caches=w.side
    )
    return int(slot)


def table(w: World) -> dict[str, int]:
    """The slot table: request id -> slot."""
    return dict(getattr(w.runner, "_glm5next_request_slot_table", {}))


def cursor(w: World, slot: int):
    """Where the slot's indexer ring stands, or ``None`` if it holds no sequence."""
    return w.runner._glm5next_side_cache_positions.get(int(slot))


def dirty(w: World, slot: int, value: float) -> None:
    """Fill one slot of every pooled store and ring, so a later zero read is not vacuous."""
    for entry in sparse_entries(w):
        entry["pool_cache"][slot].fill_(value)
        entry["tail"][slot].fill_(value)


def sparse_entries(w: World) -> list[dict]:
    """The side-cache sets of the sparse layers (the KDA layer's entry is empty)."""
    entries = [entry for entry in w.side if entry]
    if not entries:
        raise AssertionError("the world has no sparse layer, so no side cache can be read")
    return entries
