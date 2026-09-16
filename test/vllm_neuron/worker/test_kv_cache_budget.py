# SPDX-License-Identifier: Apache-2.0
"""Acceptance for the worker's KV cache budget on a hybrid recurrent stack.

The declared acceptance command:

    VLLM_NEURON_CPU_MODE=1 python -m pytest \\
      test/vllm_neuron/worker/test_kv_cache_budget.py -s -rA --timeout 120 \\
      -p no:cacheprovider

Four items, one per declared conjunct. The KV cache layout is the registered one
for this stack -- 11 latent-attention layers and 34 recurrent-state layers at
block size 128, bfloat16, tensor-parallel degree 64 -- and every expected number
below is arithmetic over those registered values, never a copy of what the code
returns. The recurrent-state geometry is derived from the vendor's own state
calculators rather than written out here.

The methods under test are driven UNBOUND on a fake worker, so no runtime, no
device and no model are constructed. The fake carries the real methods by name,
so reverting any one of them fails every item in this file rather than silently
skipping it.
"""

from __future__ import annotations

import types
from types import SimpleNamespace

import pytest
import torch

# Registered layout for this stack. Read, never re-derived here.
BLOCK_SIZE_TOKENS = 128
TP_WORLD_SIZE = 64
LATENT_LAYERS = 11
RECURRENT_LAYERS = 34
LATENT_KV_HEADS = 1
LATENT_WIDTH = 512  # kv_lora_rank + qk_rope_head_dim
BF16_BYTES = 2

# The served configuration of the run this budget is for.
MAX_MODEL_LEN = 4096
MAX_NUM_SEQS = 1
MAX_NUM_BATCHED_TOKENS = 1024
GPU_MEMORY_UTILIZATION = 0.9

GIB = 1024**3
# One logical NeuronCore of a trn2 device at logical-core size 2, and the
# per-rank device residency measured before any graph staged.
TOTAL_HBM_BYTES = 24 * GIB
MEASURED_BYTES_USED = 13_733_871_288  # 12.79 GiB of weights and operands
DEFAULT_RESERVE_GIB = 5.0

# The page one block of the latent cache occupies: one buffer, not a key/value
# pair, so no factor two.
PAGE_SIZE_BYTES = BLOCK_SIZE_TOKENS * LATENT_KV_HEADS * LATENT_WIDTH * BF16_BYTES

# Grouping: the recurrent layers are split into groups no larger than the
# smallest layer family, so the pool holds one tensor per layer of the largest
# group and every group draws blocks from that one pool.
LAYERS_PER_POOL = LATENT_LAYERS
RECURRENT_GROUPS = -(-RECURRENT_LAYERS // LATENT_LAYERS)
TOTAL_GROUPS = 1 + RECURRENT_GROUPS
# A request fills whole blocks of latent cache and keeps one state block in each
# recurrent group; one further block is the pool's null block.
LATENT_BLOCKS_PER_REQUEST = -(-MAX_MODEL_LEN // BLOCK_SIZE_TOKENS)
BLOCKS_PER_REQUEST = LATENT_BLOCKS_PER_REQUEST + RECURRENT_GROUPS
EXPECTED_BLOCKS = BLOCKS_PER_REQUEST * MAX_NUM_SEQS + 1
EXPECTED_NEED_BYTES = EXPECTED_BLOCKS * PAGE_SIZE_BYTES * LAYERS_PER_POOL

# The pre-change budget: the cap fraction applied to the whole logical core.
PARENT_CAP_FRACTION = 0.30
PARENT_BUDGET_BYTES = int(
    int(TOTAL_HBM_BYTES * GPU_MEMORY_UTILIZATION) * PARENT_CAP_FRACTION
)

_WORKER_METHODS = (
    "determine_available_memory",
    "_compute_kv_budget",
    "_kv_cache_need_bytes",
    "_physical_core_kv_bound",
    "_get_graph_reserve_bytes",
    "_get_kv_cap_fraction",
    "_determine_available_memory_cpu",
    "_determine_available_memory_neuron",
)


def _recurrent_state() -> tuple[tuple, tuple, tuple]:
    """Conv and recurrent shapes and dtypes, from the vendor's calculators."""
    from vllm.model_executor.layers.mamba.mamba_utils import (
        MambaStateDtypeCalculator,
        MambaStateShapeCalculator,
    )

    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    linear_attn = Glm5NextTextConfig().linear_attn_config
    conv, recurrent = MambaStateShapeCalculator.kda_state_shape(
        tp_world_size=TP_WORLD_SIZE,
        num_heads=linear_attn["num_heads"],
        head_dim=linear_attn["head_dim"],
        conv_kernel_size=linear_attn["short_conv_kernel_size"],
    )
    dtypes = MambaStateDtypeCalculator.kda_state_dtype(torch.bfloat16, "auto")
    return tuple(conv), tuple(recurrent), dtypes


def _registered_specs() -> dict:
    """The KV cache specs the runner reports for this stack."""
    from vllm.v1.kv_cache_interface import MambaSpec, MLAAttentionSpec

    conv_shape, recurrent_shape, state_dtypes = _recurrent_state()
    specs = {}
    for index in range(LATENT_LAYERS):
        specs[f"latent.{index}"] = MLAAttentionSpec(
            block_size=BLOCK_SIZE_TOKENS,
            num_kv_heads=LATENT_KV_HEADS,
            head_size=LATENT_WIDTH,
            dtype=torch.bfloat16,
            sliding_window=None,
            attention_chunk_size=None,
        )
    for index in range(RECURRENT_LAYERS):
        specs[f"recurrent.{index}"] = MambaSpec(
            block_size=BLOCK_SIZE_TOKENS,
            shapes=(conv_shape, recurrent_shape),
            dtypes=state_dtypes,
            page_size_padded=PAGE_SIZE_BYTES,
        )
    return specs


def _worker(*, bytes_used: int = MEASURED_BYTES_USED, host_bytes: int = 1024 * GIB):
    """A fake worker carrying the real methods and stubbed memory readings."""
    from vllm_neuron.vllm.worker.neuron_worker import NeuronWorker

    cache_config = SimpleNamespace(
        block_size=BLOCK_SIZE_TOKENS,
        cache_dtype="auto",
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        mamba_cache_mode="none",
    )
    worker = SimpleNamespace(
        cache_config=cache_config,
        vllm_config=SimpleNamespace(
            cache_config=cache_config,
            model_config=SimpleNamespace(
                max_model_len=MAX_MODEL_LEN, dtype=torch.bfloat16
            ),
            scheduler_config=SimpleNamespace(
                max_num_seqs=MAX_NUM_SEQS,
                max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
                disable_hybrid_kv_cache_manager=False,
            ),
            # A latent page is per-rank, so context parallelism would divide the
            # tokens a rank keeps. Neither degree is engaged on this stack.
            parallel_config=SimpleNamespace(
                tensor_parallel_size=TP_WORLD_SIZE,
                decode_context_parallel_size=1,
                prefill_context_parallel_size=1,
            ),
        ),
        model_runner=SimpleNamespace(get_kv_cache_spec=_registered_specs),
    )
    for name in _WORKER_METHODS:
        setattr(worker, name, types.MethodType(getattr(NeuronWorker, name), worker))
    worker._query_host_runtime_memory = lambda: host_bytes
    worker._query_runtime_memory_stats = lambda: (
        bytes_used,
        TOTAL_HBM_BYTES - bytes_used,
    )
    worker._get_byte_used_from_model = lambda: bytes_used
    return worker


def test_available_memory_is_the_block_rounded_served_need(monkeypatch) -> None:
    """The budget equals the blocks the served sequences can fill, and no more."""
    monkeypatch.setenv("VLLM_NEURON_CPU_MODE", "1")
    monkeypatch.delenv("VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB", raising=False)
    worker = _worker()

    available = worker.determine_available_memory()

    print(
        f"served tokens={MAX_MODEL_LEN * MAX_NUM_SEQS} "
        f"blocks={EXPECTED_BLOCKS} page={PAGE_SIZE_BYTES} B "
        f"pools={LAYERS_PER_POOL} groups={TOTAL_GROUPS} "
        f"available={available} B ({available / GIB:.4f} GiB)"
    )
    assert available == EXPECTED_NEED_BYTES


def test_the_heuristic_is_bounded_by_one_physical_core(monkeypatch) -> None:
    """A logical core reporting 24 GiB free budgets against 12 GiB, less reserve."""
    monkeypatch.delenv("VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB", raising=False)
    monkeypatch.delenv("VLLM_NEURON_CPU_MODE", raising=False)
    monkeypatch.delenv("VLLM_NEURON_CPU_COMPILE", raising=False)
    worker = _worker()
    reserve_bytes = int(DEFAULT_RESERVE_GIB * GIB)
    room_on_one_core = TOTAL_HBM_BYTES // 2 - reserve_bytes

    bound = worker._physical_core_kv_bound(TOTAL_HBM_BYTES, MEASURED_BYTES_USED)
    heuristic = worker._compute_kv_budget(
        TOTAL_HBM_BYTES, MEASURED_BYTES_USED, GPU_MEMORY_UTILIZATION
    )
    available = worker.determine_available_memory()

    print(
        f"bound={bound / GIB:.4f} GiB room_on_one_core={room_on_one_core / GIB:.4f} GiB "
        f"heuristic={heuristic / GIB:.4f} GiB available={available / GIB:.4f} GiB "
        f"need={EXPECTED_NEED_BYTES / GIB:.4f} GiB"
    )
    assert bound <= room_on_one_core
    assert heuristic == bound
    assert available == EXPECTED_NEED_BYTES


def test_the_graph_reserve_is_overridable_and_validated(monkeypatch) -> None:
    """The override moves the bound; an unusable value is refused by name."""
    worker = _worker()

    monkeypatch.delenv("VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB", raising=False)
    default_bound = worker._physical_core_kv_bound(
        TOTAL_HBM_BYTES, MEASURED_BYTES_USED
    )
    monkeypatch.setenv("VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB", "1.0")
    overridden_bound = worker._physical_core_kv_bound(
        TOTAL_HBM_BYTES, MEASURED_BYTES_USED
    )
    print(
        f"reserve {DEFAULT_RESERVE_GIB} GiB -> bound={default_bound / GIB:.4f} GiB; "
        f"reserve 1.0 GiB -> bound={overridden_bound / GIB:.4f} GiB"
    )
    assert overridden_bound > default_bound

    for offending in ("0", "-1", "not-a-number"):
        monkeypatch.setenv("VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB", offending)
        with pytest.raises(RuntimeError) as refusal:
            worker._get_graph_reserve_bytes()
        print(f"refused {offending!r}: {refusal.value}")
        assert "VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB" in str(refusal.value)


def test_control_the_logical_core_budget_over_allocates(monkeypatch) -> None:
    """Restoring the unbounded logical-core budget over-allocates the cache."""
    monkeypatch.delenv("VLLM_NEURON_CPU_MODE", raising=False)
    monkeypatch.delenv("VLLM_NEURON_CPU_COMPILE", raising=False)
    monkeypatch.delenv("VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB", raising=False)
    worker = _worker()
    shipped = worker.determine_available_memory()

    # The pre-change path: no physical-core bound and no served-need cap.
    worker._physical_core_kv_bound = lambda *_: TOTAL_HBM_BYTES
    parent = worker._compute_kv_budget(
        TOTAL_HBM_BYTES, MEASURED_BYTES_USED, GPU_MEMORY_UTILIZATION
    )

    print(
        f"parent={parent} B ({parent / GIB:.4f} GiB) "
        f"shipped={shipped} B ({shipped / GIB:.4f} GiB) "
        f"ratio={parent / shipped:.1f}x"
    )
    assert parent == PARENT_BUDGET_BYTES
    assert shipped == EXPECTED_NEED_BYTES
    assert parent >= 100 * shipped
