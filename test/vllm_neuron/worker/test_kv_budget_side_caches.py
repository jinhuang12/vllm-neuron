# SPDX-License-Identifier: Apache-2.0
"""The worker's KV budget check counts the DSA indexer side caches, pad_tail included.

Each sparse-attention (DSA) layer of GLM-5.3-Flash keeps three caches that no
``KVCacheSpec`` declares (``NeuronModelRunner._glm5next_side_caches``): the pooled-key
store ``pool_cache [slots, max_model_len // index_kpool + 1, index_head_dim]``, the
decode ring ``tail [slots, 2, index_kpool, index_head_dim]`` and the padding ring
``pad_tail`` of the same shape. At 0a08ff4 they sat outside
``NeuronWorker._kv_cache_footprint_bytes``: 0.3466 GiB per rank at 64 x 8192.

Hand computation at TP=64, block 128, 11 DSA + 34 KDA layers (bf16 KV, fp32 state),
the gate's serve line (``--no-enable-prefix-caching --mamba-block-size 8192``):

* vLLM pool: (64 x (8192 / 128 + 4) + 1) = 4353 blocks x 131072 B x 11 = 6276120576 B
* KDA banks: 34 layers x 64 slots x 67840 B = 147619840 B
* side caches: 11 x 64 x (2049 x 128 + 2 x (2 x 4 x 128)) x 2 B = 372162560 B
* footprint: 6795902976 B = 6.3292 GiB (6.0 GiB = 6423740416 B without the side caches)

The same numbers are in ``/home/ubuntu/glm53f-wt2/reports/kvseg_budget.json``.

Run with ``VLLM_NEURON_CPU_MODE=1`` (``test/conftest.py`` pins it).
"""

from __future__ import annotations

import pytest
import torch

from test.vllm_neuron.worker import test_kv_budget_glm53f as kv

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
    """``pool_cache`` + ``tail`` + ``pad_tail`` of every DSA layer, by hand."""
    pool_rows = length // INDEX_KPOOL + 1
    ring_rows = 2 * INDEX_KPOOL
    per_slot = (pool_rows + 2 * ring_rows) * INDEX_HEAD_DIM * BF16_BYTES
    return kv.MLA_LAYERS * seqs * per_slot


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
    """The numbers in the module docstring and in kvseg_budget.json."""
    assert hand_need_bytes(seqs=64, length=8192, gate_line=True) == 6276120576
    assert hand_side_cache_bytes(seqs=64, length=8192) == 372162560
    assert round(hand_side_cache_bytes(seqs=64, length=8192) / GIB, 4) == 0.3466
    assert hand_footprint_bytes(seqs=64, length=8192, gate_line=True) == 6795902976


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
    assert footprint - kv_tensors == 372162560


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
        request_slots=seqs,
    )
    sparse = [entry for entry in allocated if entry]

    assert len(sparse) == kv.MLA_LAYERS
    assert {tuple(sorted(entry)) for entry in sparse} == {
        ("pad_tail", "pool_cache", "tail")
    }
    assert indexer_side_cache_bytes(
        specs, text_config, max_seq_len=length, request_slots=seqs
    ) == sum(tensor.nbytes for entry in sparse for tensor in entry.values())
    assert indexer_side_cache_bytes(
        specs, text_config, max_seq_len=length, request_slots=seqs
    ) == hand_side_cache_bytes(seqs=seqs, length=length)


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
    """64 x 8k on the 5938748 residency: the KV tensors fit 6.04 GiB, the side caches do not."""
    from vllm_neuron.vllm.worker.neuron_model_runner import kv_cache_allocations

    _clear_knobs(monkeypatch)
    worker = _worker(seqs=64, length=8192)
    budget = worker._compute_kv_budget(
        kv.TOTAL_HBM_BYTES, kv.MEASURED_PARAM_BYTES, kv.GPU_MEMORY_UTILIZATION
    )
    config = kv.served_kv_cache_config(worker.model_runner, worker)
    kv_tensors = sum(size for size, _ in kv_cache_allocations(config, state_slots=64))
    footprint = hand_footprint_bytes(seqs=64, length=8192, gate_line=True)
    # The premise: this point fits only while the side caches are left out.
    assert kv_tensors <= budget < footprint

    with pytest.raises(RuntimeError) as refusal:
        worker.determine_available_memory()

    message = str(refusal.value)
    assert f"{footprint / GIB:.3f} GiB" in message
    assert f"side caches {372162560 / GIB:.3f} GiB" in message
    assert f"short by {(footprint - budget) / GIB:.3f} GiB" in message
    assert round((footprint - budget) / GIB, 3) == 0.289


def test_the_0a08ff4_residency_admits_64_by_8k_with_the_side_caches(monkeypatch) -> None:
    """On the measured tip the cap binds at 6.62 GiB and the 6.33 GiB footprint fits."""
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
    assert round(budget / GIB, 2) == 6.62
    assert available == hand_need_bytes(seqs=64, length=8192, gate_line=True)
    assert round((budget - footprint) / GIB, 3) == 0.295


@pytest.mark.parametrize(
    "seqs,length,gate_line,before,after",
    [
        # bs=1 recipe: max_model_len 4096, prefix caching on (no --mamba-block-size).
        (1, 4096, False, 234435072, 237366528),
        (64, 8192, True, 6423740416, 6795902976),
        (119, 4096, True, 6452559360, 6801402624),
    ],
)
def test_the_budget_table_points(seqs, length, gate_line, before, after) -> None:
    """The (bs, ctx) rows of kvseg_budget.json: 0a08ff4 footprint and this footprint."""
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
