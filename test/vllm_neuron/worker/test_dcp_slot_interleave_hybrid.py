# SPDX-License-Identifier: Apache-2.0
"""Under decode context parallelism the prefill slot interleave applies to token-paged groups only.

At DCP > 1 a rank holds every DCP-th ``block_size`` run of a sequence's tokens in a
token-paged attention group. So both prefill builders, the warmup one
(``_build_warmup_attention_metadata``) and the serving one (``_build_attention_metadata``),
keep only the entries of the step's per-token slot mapping at this rank's positions
(``view(S // (W * I), W, I)[:, R, :]`` with ``I = block_size``).

A recurrent group (``MambaSpec``) is addressed by request slot, not by token, and its one
block spans the whole sequence (``block_size`` = ``max_model_len``). It has no per-rank
token shard, and the view has no rows for it. These tests build both builders' prefill
inputs for a hybrid layout (a latent attention group of 128-token blocks and a recurrent
group of ``max_model_len``-token blocks) at DCP 1, 2 and 4, on every rank, and pin:

1. both builders run, chunk after chunk, and the recurrent group keeps the step's per-token
   slot mapping as the host computed it;
2. the attention group's slot mapping is the interleave's: this rank's positions, in order,
   at the slots its block table names. It is the same for the hybrid layout as for an
   attention-only layout;
3. each group's slot mapping has the same length in warmup as in serving, so a captured
   prefill graph and the inputs it is fed agree.

Which groups are token-paged is ``kv_group_blocks.is_token_paged``, the rule the block-table
rows are sized with; ``test_kv_group_blocks.py`` pins it per spec class.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, MambaSpec, MLAAttentionSpec
from vllm.v1.worker.block_table import MultiGroupBlockTable

from vllm_neuron.vllm.worker import neuron_model_runner
from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

pytestmark = [pytest.mark.fast]

MAX_MODEL_LEN = 131072
PAGE = 128
QUERY = 1024
ATTENTION_LAYERS = ["model.layers.3.self_attn"]
RECURRENT_LAYERS = ["model.layers.0.linear_attn"]
REQUEST = "dcp-hybrid-request"
#: (dcp, rank) for every rank at DCP 1, 2 and 4.
RANKS = [(dcp, rank) for dcp in (1, 2, 4) for rank in range(dcp)]


def _attention_spec() -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=PAGE,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
        sliding_window=None,
        attention_chunk_size=None,
    )


def _recurrent_spec() -> MambaSpec:
    return MambaSpec(
        block_size=MAX_MODEL_LEN,
        shapes=((3, 384), (1, 128, 128)),
        dtypes=(torch.bfloat16, torch.float32),
    )


def _groups(hybrid: bool) -> list[KVCacheGroupSpec]:
    groups = [KVCacheGroupSpec(list(ATTENTION_LAYERS), _attention_spec())]
    if hybrid:
        groups.append(KVCacheGroupSpec(list(RECURRENT_LAYERS), _recurrent_spec()))
    return groups


def _runner(dcp: int, rank: int, *, hybrid: bool) -> NeuronModelRunner:
    """A runner shell carrying what both metadata builders read, with real block tables."""
    groups = _groups(hybrid)
    block_sizes = [group.kv_cache_spec.block_size for group in groups]
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.max_model_len = MAX_MODEL_LEN
    runner.max_num_reqs = 1
    runner.device = torch.device("cpu")
    runner.cp_world_size = dcp
    # ``cp_rank`` caches the rank here; unset, it would read the TP group.
    runner._cp_rank = rank
    runner._dcp_size = dcp
    runner._is_synthetic_model = False
    runner.drafter = None
    runner.neuron_config = SimpleNamespace(
        kv_segment_size_buckets=[1024, 2048, 4096, MAX_MODEL_LEN],
        num_batched_tokens_buckets=[QUERY],
        decode_context_length_buckets=None,
        enable_structured_outputs=False,
    )
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=groups)
    runner.input_batch = SimpleNamespace(
        req_ids=[REQUEST],
        num_reqs=1,
        block_table=MultiGroupBlockTable(
            max_num_reqs=1,
            max_model_len=MAX_MODEL_LEN,
            max_num_batched_tokens=QUERY,
            pin_memory=False,
            device=torch.device("cpu"),
            block_sizes=block_sizes,
            kernel_block_sizes=block_sizes,
            # Each rank stores 1 / dcp of a sequence's tokens (vLLM's per-rank rule).
            max_num_blocks=[-(-MAX_MODEL_LEN // (size * dcp)) for size in block_sizes],
        ),
        num_computed_tokens_cpu_tensor=torch.zeros(1, dtype=torch.int32),
        num_tokens_no_spec=np.zeros(1, dtype=np.int32),
    )
    return runner


def _serve_chunk(runner, *, cached: int, total: int) -> dict:
    """Allocate the chunk's blocks and compute its slot mapping the way the runner does,
    then build the serving metadata for it.

    Each group's table holds the blocks this rank needs for ``cached + QUERY`` tokens. Block
    0 is vLLM's null block, so a request's blocks start at 1. The slot mapping is the
    runner's own host computation, with the interleave size equal to each group's block
    size.
    """
    dcp, rank = runner.cp_world_size, runner.cp_rank
    end = cached + QUERY
    for table in runner.input_batch.block_table.block_tables:
        held = int(table.num_blocks_per_row[0])
        needed = -(-end // (table.block_size * dcp))
        if needed > held:
            table.append_row(list(range(1 + held, 1 + needed)), 0)
    positions = np.arange(cached, end, dtype=np.int64)
    requests = np.zeros(QUERY, dtype=np.int64)
    for table in runner.input_batch.block_table.block_tables:
        neuron_model_runner._compute_slot_mapping_cpu(
            table.block_table.cpu,
            table.slot_mapping.np,
            positions,
            requests,
            table.block_size,
            cp_world_size=dcp,
            cp_rank=rank,
            cp_kv_cache_interleave_size=table.block_size,
        )
    runner.input_batch.num_computed_tokens_cpu_tensor[0] = cached
    runner.input_batch.num_tokens_no_spec[0] = total
    return runner._build_attention_metadata(1, QUERY, QUERY, 0, cached, host_only=True)


def _owned(start: int, dcp: int, rank: int) -> list[int]:
    """This rank's positions of the chunk ``start .. start + QUERY - 1``, in order."""
    return [p for p in range(start, start + QUERY) if (p // PAGE) % dcp == rank]


def _attention_slots(runner, start: int) -> list[int]:
    """The slots the attention group's table names for this rank's positions of a chunk.

    A rank's ``j``-th block holds the sequence's tokens ``(j * dcp + rank) * PAGE ..``, so
    position ``p`` it owns sits at offset ``p % PAGE`` of its block ``p // (PAGE * dcp)``.
    """
    dcp, rank = runner.cp_world_size, runner.cp_rank
    table = runner.input_batch.block_table.block_tables[0].block_table.cpu[0]
    return [int(table[p // (PAGE * dcp)]) * PAGE + p % PAGE for p in _owned(start, dcp, rank)]


def _slots(metadata: dict, layers: list[str]) -> list[int]:
    return metadata[layers[0]]["slot_mapping"].tolist()


@pytest.mark.parametrize("dcp, rank", RANKS)
def test_warmup_prefill_inputs_build_for_a_hybrid_layout(dcp, rank):
    runner = _runner(dcp, rank, hybrid=True)
    metadata = runner._build_warmup_attention_metadata(
        QUERY, 1, cached_seq_len=0, decode_token_threshold=1, kv_segment_size=1024
    )
    assert sorted(metadata) == sorted(ATTENTION_LAYERS + RECURRENT_LAYERS)
    # The attention group: this rank's positions of the dummy per-token slots.
    assert _slots(metadata, ATTENTION_LAYERS) == _owned(0, dcp, rank)
    # The recurrent group: the step's per-token slots, uninterleaved.
    assert _slots(metadata, RECURRENT_LAYERS) == list(range(QUERY))


@pytest.mark.parametrize("dcp, rank", RANKS)
def test_serving_prefill_inputs_build_for_a_hybrid_layout_chunk_by_chunk(dcp, rank):
    runner = _runner(dcp, rank, hybrid=True)
    recurrent_table = runner.input_batch.block_table.block_tables[1]
    for cached in (0, QUERY, 2 * QUERY):
        metadata = _serve_chunk(runner, cached=cached, total=4 * QUERY)
        assert _slots(metadata, ATTENTION_LAYERS) == _attention_slots(runner, cached)
        # The recurrent group keeps the host's per-token mapping for the whole step.
        host = recurrent_table.slot_mapping.cpu[:QUERY].tolist()
        assert _slots(metadata, RECURRENT_LAYERS) == host


@pytest.mark.parametrize("dcp, rank", RANKS)
def test_an_attention_only_layout_keeps_the_interleave(dcp, rank):
    runner = _runner(dcp, rank, hybrid=False)
    warm = runner._build_warmup_attention_metadata(
        QUERY, 1, cached_seq_len=0, decode_token_threshold=1, kv_segment_size=1024
    )
    assert _slots(warm, ATTENTION_LAYERS) == _owned(0, dcp, rank)
    for cached in (0, QUERY, 2 * QUERY):
        served = _serve_chunk(runner, cached=cached, total=4 * QUERY)
        assert _slots(served, ATTENTION_LAYERS) == _attention_slots(runner, cached)


@pytest.mark.parametrize("dcp, rank", RANKS)
def test_attention_slots_match_an_attention_only_layout(dcp, rank):
    hybrid, alone = _runner(dcp, rank, hybrid=True), _runner(dcp, rank, hybrid=False)
    warm = [
        runner._build_warmup_attention_metadata(
            QUERY, 1, cached_seq_len=0, decode_token_threshold=1, kv_segment_size=1024
        )
        for runner in (hybrid, alone)
    ]
    assert _slots(warm[0], ATTENTION_LAYERS) == _slots(warm[1], ATTENTION_LAYERS)
    for cached in (0, QUERY):
        served = [_serve_chunk(runner, cached=cached, total=2 * QUERY) for runner in (hybrid, alone)]
        assert _slots(served[0], ATTENTION_LAYERS) == _slots(served[1], ATTENTION_LAYERS)
        assert _slots(served[1], ATTENTION_LAYERS) == _attention_slots(alone, cached)


@pytest.mark.parametrize("dcp, rank", RANKS)
def test_each_group_has_the_same_slot_mapping_length_in_warmup_and_in_serving(dcp, rank):
    runner = _runner(dcp, rank, hybrid=True)
    warm = runner._build_warmup_attention_metadata(
        QUERY, 1, cached_seq_len=0, decode_token_threshold=1, kv_segment_size=1024
    )
    served = _serve_chunk(runner, cached=0, total=QUERY)
    for layers in (ATTENTION_LAYERS, RECURRENT_LAYERS):
        assert warm[layers[0]]["slot_mapping"].shape == served[layers[0]]["slot_mapping"].shape
    assert warm[ATTENTION_LAYERS[0]]["slot_mapping"].shape == (QUERY // dcp,)
    assert warm[RECURRENT_LAYERS[0]]["slot_mapping"].shape == (QUERY,)

