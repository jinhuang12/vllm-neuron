# SPDX-License-Identifier: Apache-2.0
"""The worker's KV budget check counts the DSA indexer side caches, pad_tail included.

Each sparse-attention (DSA) layer of GLM-5.3-Flash keeps three caches that no
``KVCacheSpec`` declares (``NeuronModelRunner._glm5next_side_caches``): the pooled-key
store ``pool_cache [slots, max_model_len // index_kpool + 1, index_head_dim]``, the
decode ring ``tail [slots, 2, index_kpool, index_head_dim]`` and the padding ring
``pad_tail`` of the same shape. At 0a08ff4 they sat outside
``NeuronWorker._kv_cache_footprint_bytes``: 0.3466 GiB per rank at 64 x 8192.

Hand computation at TP=64, block 128, 11 DSA + 34 KDA layers (bf16 KV, fp32 state),
the served TP=64 configuration (``--no-enable-prefix-caching --mamba-block-size 8192``):

* vLLM pool: (64 x (8192 / 128 + 4) + 1) = 4353 blocks x 131072 B x 11 = 6276120576 B
* KDA banks: 34 layers x 64 slots x 67840 B = 147619840 B
* side caches: 11 x (64 + 1) x (2049 x 128 + 2 x (2 x 4 x 128)) x 2 B = 377977600 B,
  one set per request slot and one for the scratch slot (``SCRATCH_SLOTS``) the
  runner allocates past them
* footprint: 6801718016 B = 6.3346 GiB (6.0 GiB = 6423740416 B without the side caches)

Run with ``VLLM_NEURON_CPU_MODE=1`` (``test/conftest.py`` pins it).
"""

from __future__ import annotations

import types
from types import SimpleNamespace

import pytest
import torch

from test.vllm_neuron.worker import test_kv_budget_glm53f as kv
from vllm_neuron.vllm.worker.glm5next_state_banks import SCRATCH_SLOTS

GIB = 1024**3

INDEX_KPOOL = 4
INDEX_HEAD_DIM = 128
BF16_BYTES = 2
KDA_SLOT_BYTES = 128 * 128 * 4 + 3 * 384 * 2
POOL_PAGE_BYTES = kv.BLOCK_SIZE_TOKENS * kv.LATENT_BYTES_PER_TOKEN

TIP_PARAM_BYTES = kv.TIP_PARAM_BYTES
TIP_RESIDENT_BYTES = kv.TIP_RESIDENT_BYTES


def hand_need_bytes(*, seqs: int, length: int, gate_line: bool) -> int:
    """vLLM's pool: one block per 128 tokens for the latent group, plus the KDA groups."""
    kda_groups = -(-kv.KDA_LAYERS // kv.MLA_LAYERS)
    latent_blocks = -(-length // kv.BLOCK_SIZE_TOKENS)
    kda_blocks = 1 if gate_line else latent_blocks
    blocks = (latent_blocks + kda_groups * kda_blocks) * seqs + 1
    return blocks * POOL_PAGE_BYTES * kv.MLA_LAYERS


def hand_side_cache_bytes(*, seqs: int, length: int) -> int:
    """``pool_cache`` + ``tail`` + ``pad_tail`` of every DSA layer, by hand.

    One set per request slot (``seqs``) and one per scratch slot (``SCRATCH_SLOTS``).
    """
    pool_rows = length // INDEX_KPOOL + 1
    ring_rows = 2 * INDEX_KPOOL
    per_slot = (pool_rows + 2 * ring_rows) * INDEX_HEAD_DIM * BF16_BYTES
    return kv.MLA_LAYERS * (seqs + SCRATCH_SLOTS) * per_slot


def hand_footprint_bytes(*, seqs: int, length: int, gate_line: bool) -> int:
    """Pool + KDA banks + side caches: what the worker check now compares."""
    return (
        hand_need_bytes(seqs=seqs, length=length, gate_line=gate_line)
        + kv.KDA_LAYERS * seqs * KDA_SLOT_BYTES
        + hand_side_cache_bytes(seqs=seqs, length=length)
    )


def _worker(*, seqs: int, length: int, gate_line: bool = True, **residency):
    layers, text_config = kv.glm53f_layer_specs()
    runner = kv.fake_runner(
        layers,
        max_num_seqs=seqs,
        max_model_len=length,
        gate_knobs=gate_line,
        text_config=text_config,
    )
    return kv.fake_worker(runner, **residency)


def _clear_knobs(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_NEURON_CPU_MODE", raising=False)
    monkeypatch.delenv("VLLM_NEURON_CPU_COMPILE", raising=False)
    monkeypatch.delenv("VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB", raising=False)
    monkeypatch.delenv("VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION", raising=False)


def test_the_hand_computation_at_64_by_8k() -> None:
    """The numbers in the module docstring."""
    assert hand_need_bytes(seqs=64, length=8192, gate_line=True) == 6276120576
    assert hand_side_cache_bytes(seqs=64, length=8192) == 377977600
    assert round(hand_side_cache_bytes(seqs=64, length=8192) / GIB, 4) == 0.352
    assert hand_footprint_bytes(seqs=64, length=8192, gate_line=True) == 6801718016


def test_the_footprint_at_64_by_8k_includes_the_side_caches_and_pad_tail() -> None:
    """The worker's footprint is the hand total: pool, KDA banks and all three side caches."""
    from vllm_neuron.vllm.worker.neuron_model_runner import kv_cache_allocations

    worker = _worker(seqs=64, length=8192)
    need = worker._kv_cache_need_bytes()
    footprint = worker._kv_cache_footprint_bytes(need)
    config = kv.served_kv_cache_config(worker.model_runner, worker)
    kv_tensors = sum(size for size, _ in kv_cache_allocations(config, state_slots=64))

    assert need == hand_need_bytes(seqs=64, length=8192, gate_line=True)
    assert kv_tensors == 6423740416
    assert footprint == hand_footprint_bytes(seqs=64, length=8192, gate_line=True)
    assert footprint - kv_tensors == 377977600


def test_the_side_cache_price_is_the_runners_own_allocation() -> None:
    """The priced bytes are the nbytes of the tensors the runner's builder allocates."""
    from vllm_neuron.vllm.worker.neuron_model_runner import (
        NeuronModelRunner,
        indexer_side_cache_bytes,
    )

    seqs, length = 3, 512
    layers, text_config = kv.glm53f_layer_specs()
    runner = kv.fake_runner(
        layers, max_num_seqs=seqs, max_model_len=length, text_config=text_config
    )
    specs = runner.get_kv_cache_spec()
    banks = [
        {"family": "linear_attn"}
        if type(spec).__name__ == "MambaSpec"
        else {
            "family": "self_attn",
            "latent_cache": torch.zeros((1, 1, 512), dtype=spec.dtype),
        }
        for spec in specs.values()
    ]
    allocated = NeuronModelRunner._glm5next_side_caches(
        banks,
        index_kpool=int(text_config.index_kpool),
        index_head_dim=int(text_config.index_head_dim),
        max_seq_len=length,
        request_slots=seqs + SCRATCH_SLOTS,
    )
    sparse = [entry for entry in allocated if entry]

    assert len(sparse) == kv.MLA_LAYERS
    assert {tuple(sorted(entry)) for entry in sparse} == {
        ("pad_tail", "pool_cache", "tail")
    }
    assert indexer_side_cache_bytes(
        specs, text_config, max_seq_len=length, request_slots=seqs + SCRATCH_SLOTS
    ) == sum(tensor.nbytes for entry in sparse for tensor in entry.values())
    assert indexer_side_cache_bytes(
        specs, text_config, max_seq_len=length, request_slots=seqs + SCRATCH_SLOTS
    ) == hand_side_cache_bytes(seqs=seqs, length=length)


def _live_side_cache_bytes(runner) -> int:
    """The nbytes of the set ``NeuronModelRunner._glm5next_live_side_caches`` allocates.

    The class's own allocation path, bound to the fake runner, over one bank per KV
    spec; the latent banks are on the meta device, so the set takes no memory.
    """
    from vllm.v1.kv_cache_interface import MambaSpec

    from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

    banks = [
        {"family": "linear_attn", "state_slots": runner.max_num_reqs}
        if isinstance(spec, MambaSpec)
        else {
            "family": "self_attn",
            "latent_cache": torch.empty((0,), dtype=spec.dtype, device="meta"),
        }
        for spec in runner.get_kv_cache_spec().values()
    ]
    for name in (
        "_glm5next_indexer_context",
        "_glm5next_request_slot_capacity",
        "_glm5next_state_slot_count",
        "_glm5next_speculative_tokens",
    ):
        setattr(runner, name, types.MethodType(getattr(NeuronModelRunner, name), runner))
    runner._glm5next_side_caches = NeuronModelRunner._glm5next_side_caches
    runner._glm5next_indexer_ring_rows = NeuronModelRunner._glm5next_indexer_ring_rows
    live = NeuronModelRunner._glm5next_live_side_caches(runner, banks)
    return sum(tensor.nbytes for entry in live for tensor in entry.values())


@pytest.mark.parametrize(
    "seqs,length,draft_tokens,slot_bytes",
    [
        # One sequence of 4096 tokens drafting 3 mtp tokens: 12 sparse layers (the
        # draft head's included), rings 8 rows deep;
        # per slot 12 x (1025 x 128 + 2 x (2 x 8 x 128)) x 2 B.
        (1, 4096, 3, 3247104),
        # 64 sequences of 8192 tokens, no draft: 11 sparse layers, rings 4 rows deep;
        # per slot 11 x (2049 x 128 + 2 x (2 x 4 x 128)) x 2 B.
        (64, 8192, 0, 5815040),
    ],
)
def test_the_price_is_every_slot_the_runners_live_set_allocates(
    seqs, length, draft_tokens, slot_bytes, monkeypatch
) -> None:
    """The fit check prices the runner's live side caches, the scratch slot included."""
    from vllm.config import SpeculativeConfig

    from vllm_neuron.model.glm5_next import mtp
    from vllm_neuron.vllm.worker.neuron_worker import _indexer_side_cache_bytes

    monkeypatch.setenv(mtp.SHADOW_DRAFT_ENV, str(draft_tokens))
    worker = _worker(seqs=seqs, length=length)
    runner = worker.model_runner
    if draft_tokens:
        spec = SimpleNamespace(method="mtp", num_speculative_tokens=draft_tokens)
        spec.use_eagle = lambda: SpeculativeConfig.use_eagle(spec)
        runner.speculative_config = spec
        runner.is_mtp_spec = True
        runner.drafter = SimpleNamespace(num_speculative_tokens=draft_tokens)

    priced = _indexer_side_cache_bytes(worker.vllm_config, runner)

    assert priced == _live_side_cache_bytes(runner)
    assert priced == (seqs + SCRATCH_SLOTS) * slot_bytes


def test_a_model_with_no_indexer_prices_no_side_caches() -> None:
    """A text config without ``index_kpool``/``index_head_dim`` has no side caches."""
    from vllm_neuron.vllm.worker.neuron_model_runner import indexer_side_cache_bytes

    layers, text_config = kv.glm53f_layer_specs()
    runner = kv.fake_runner(
        layers, max_num_seqs=4, max_model_len=4096, text_config=text_config
    )
    specs = runner.get_kv_cache_spec()

    assert indexer_side_cache_bytes(specs, None, max_seq_len=4096, request_slots=4) == 0
    assert (
        indexer_side_cache_bytes(
            specs, object(), max_seq_len=4096, request_slots=4
        )
        == 0
    )


def test_a_point_that_fits_only_without_side_caches_is_refused_naming_the_shortfall(
    monkeypatch,
) -> None:
    """64 x 8k against a budget between the KV tensors and the footprint: refused.

    The graph-need override puts the budget midway between the KV tensors
    (5.983 GiB) and the footprint with the side caches (6.335 GiB). Before the
    budget change the 5938748 residency put it there by itself (6.04 GiB, the
    per-physical-core bound, short by 0.289 GiB); that bound is gone.
    """
    from vllm_neuron.vllm.worker.neuron_model_runner import kv_cache_allocations
    from vllm_neuron.vllm.worker.neuron_worker import KV_BUDGET_MARGIN_BYTES

    _clear_knobs(monkeypatch)
    worker = _worker(seqs=64, length=8192)
    config = kv.served_kv_cache_config(worker.model_runner, worker)
    kv_tensors = sum(size for size, _ in kv_cache_allocations(config, state_slots=64))
    footprint = hand_footprint_bytes(seqs=64, length=8192, gate_line=True)
    free = kv.TOTAL_HBM_BYTES - kv.MEASURED_RESIDENT_BYTES
    graph_bytes = free - KV_BUDGET_MARGIN_BYTES - (kv_tensors + footprint) // 2
    monkeypatch.setenv("VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB", repr(graph_bytes / GIB))
    budget = worker._compute_kv_budget(
        kv.TOTAL_HBM_BYTES, kv.MEASURED_PARAM_BYTES, kv.GPU_MEMORY_UTILIZATION
    )
    # The premise: this point fits only while the side caches are left out.
    assert kv_tensors <= budget < footprint

    with pytest.raises(RuntimeError) as refusal:
        worker.determine_available_memory()

    message = str(refusal.value)
    assert f"{footprint / GIB:.3f} GiB" in message
    assert f"side caches {377977600 / GIB:.3f} GiB" in message
    assert f"short by {(footprint - budget) / GIB:.3f} GiB" in message
    assert "VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB override" in message


def test_the_0a08ff4_residency_admits_64_by_8k_with_the_side_caches(monkeypatch) -> None:
    """On the measured tip the 6.33 GiB footprint fits with 5.6 GiB to spare.

    A cold cache (the fake worker reads none) assumes the 5 GiB default reserve:
    24 - 6.79 resident - 5 - 0.25 margin = 11.96 GiB. Before the budget change the 0.30
    cap bound by default at 6.62 GiB, 0.295 GiB above the footprint.
    """
    from vllm_neuron.vllm.worker.neuron_worker import KV_BUDGET_MARGIN_BYTES

    _clear_knobs(monkeypatch)
    worker = _worker(
        seqs=64,
        length=8192,
        param_bytes=TIP_PARAM_BYTES,
        resident_bytes=TIP_RESIDENT_BYTES,
    )
    budget = worker._compute_kv_budget(
        kv.TOTAL_HBM_BYTES, TIP_PARAM_BYTES, kv.GPU_MEMORY_UTILIZATION
    )

    available = worker.determine_available_memory()

    footprint = hand_footprint_bytes(seqs=64, length=8192, gate_line=True)
    assert budget == (
        kv.TOTAL_HBM_BYTES - TIP_RESIDENT_BYTES - 5 * GIB - KV_BUDGET_MARGIN_BYTES
    )
    assert round(budget / GIB, 2) == 11.96
    assert available == hand_need_bytes(seqs=64, length=8192, gate_line=True)
    assert round((budget - footprint) / GIB, 2) == 5.63


@pytest.mark.parametrize(
    "seqs,length,gate_line,before,after",
    [
        # bs=1 recipe: max_model_len 4096, prefix caching on (no --mamba-block-size).
        (1, 4096, False, 234435072, 240297984),
        (64, 8192, True, 6423740416, 6801718016),
        (119, 4096, True, 6452559360, 6804334080),
    ],
)
def test_the_budget_table_points(seqs, length, gate_line, before, after) -> None:
    """Per (bs, ctx) point: the KV tensors alone and the footprint with the side caches."""
    from vllm_neuron.vllm.worker.neuron_model_runner import kv_cache_allocations

    worker = _worker(seqs=seqs, length=length, gate_line=gate_line)
    need = worker._kv_cache_need_bytes()
    config = kv.served_kv_cache_config(worker.model_runner, worker)

    assert (
        sum(size for size, _ in kv_cache_allocations(config, state_slots=seqs))
        == before
    )
    assert worker._kv_cache_footprint_bytes(need) == after
    assert after == hand_footprint_bytes(seqs=seqs, length=length, gate_line=gate_line)
