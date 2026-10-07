# SPDX-License-Identifier: Apache-2.0
"""The worker's KV cache budget on a hybrid recurrent stack.

The layout is the one this stack serves -- 11 latent-attention layers and 34
recurrent-state layers at block size 128, bfloat16, tensor-parallel degree 64 --
and every expected number below is arithmetic over those values rather than a
copy of what the code returns. The recurrent-state geometry comes from vLLM's own
state calculators.

The methods under test are driven unbound on a fake worker, so no runtime, no
device and no model are constructed. The fake carries the real methods by name,
so replacing any one of them fails every test here instead of skipping it.

Run with ``VLLM_NEURON_CPU_MODE=1``.
"""

from __future__ import annotations

import types
from types import SimpleNamespace

import pytest
import torch

# The layout this budget is computed for. Read, never re-derived here.
BLOCK_SIZE_TOKENS = 128
TP_WORLD_SIZE = 64
LATENT_LAYERS = 11
RECURRENT_LAYERS = 34
LATENT_KV_HEADS = 1
LATENT_WIDTH = 512  # kv_lora_rank + qk_rope_head_dim
BF16_BYTES = 2

# The served configuration this budget is for.
MAX_MODEL_LEN = 4096
MAX_NUM_SEQS = 1
MAX_NUM_BATCHED_TOKENS = 1024
GPU_MEMORY_UTILIZATION = 0.9

GIB = 1024**3
# One logical NeuronCore of a trn2 device at logical-core size 2, and the
# per-rank device residency measured before any graph staged.
TOTAL_HBM_BYTES = 24 * GIB
MEASURED_BYTES_USED = 13_733_871_288  # 12.79 GiB of weights and operands
# The graph need assumed on a cold compile cache (the fake worker reads none).
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
# The allocator takes a block per block_size tokens in EVERY group, recurrent
# groups included, because this stack uses 128-token recurrent blocks. One
# further block is the pool's null block, which no request can be given.
BLOCKS_PER_GROUP_PER_SEQUENCE = -(-MAX_MODEL_LEN // BLOCK_SIZE_TOKENS)
BLOCKS_PER_REQUEST = TOTAL_GROUPS * BLOCKS_PER_GROUP_PER_SEQUENCE
EXPECTED_BLOCKS = BLOCKS_PER_REQUEST * MAX_NUM_SEQS + 1
EXPECTED_NEED_BYTES = EXPECTED_BLOCKS * PAGE_SIZE_BYTES * LAYERS_PER_POOL
# One request slot of one recurrent layer at this degree: the bf16 short-conv
# state (kernel - 1 = 3 rows of 3 x 128 channels for the one head per rank) and
# the fp32 128 x 128 recurrent state. 67840 bytes, already a multiple of the
# runner's 256-byte slot alignment.
RECURRENT_SLOT_BYTES = 3 * 3 * 128 * BF16_BYTES + 128 * 128 * 4
# What the runner allocates once vLLM has sized the blocks from that need: the
# latent pools as vLLM laid them out, plus, for every recurrent layer, its own
# bank of one slot per concurrent sequence, because a recurrent bank is
# addressed by request slot and cannot share a tensor with any other layer.
EXPECTED_FOOTPRINT_BYTES = (
    EXPECTED_BLOCKS * PAGE_SIZE_BYTES * LAYERS_PER_POOL
    + RECURRENT_LAYERS * MAX_NUM_SEQS * RECURRENT_SLOT_BYTES
)

# A pool priced at one block per recurrent group: the allocator refuses a
# full-length request in it, which one test below measures.
UNDERPRICED_BLOCKS = BLOCKS_PER_GROUP_PER_SEQUENCE + RECURRENT_GROUPS + 1
UNDERPRICED_BYTES = UNDERPRICED_BLOCKS * PAGE_SIZE_BYTES * LAYERS_PER_POOL

# The budget the cap fraction alone gives, applied to the whole logical core.
KV_CAP_FRACTION = 0.30
CAP_FRACTION_BUDGET_BYTES = int(
    int(TOTAL_HBM_BYTES * GPU_MEMORY_UTILIZATION) * KV_CAP_FRACTION
)

_WORKER_METHODS = (
    "determine_available_memory",
    "_kv_budget",
    "_compute_kv_budget",
    "_kv_cache_need_bytes",
    "_kv_cache_allocation_sizes",
    "_kv_cache_footprint_bytes",
    "_max_num_seqs_that_fit",
    "_kv_cache_footprint_at",
    "_log_recurrent_blocks",
    "_kv_cache_largest_tensor_bytes",
    "_graph_need",
    "_prepared_operand_bytes",
    "_get_graph_reserve_bytes",
    "_get_kv_cap_fraction",
    "_determine_available_memory_cpu",
    "_determine_available_memory_neuron",
    "_estimate_available_memory_neuron",
)


def _margin() -> int:
    from vllm_neuron.vllm.worker.neuron_worker import KV_BUDGET_MARGIN_BYTES

    return KV_BUDGET_MARGIN_BYTES


class _FakeModel(torch.nn.Module):
    """One parameter, plus whatever prepared attributes a test needs."""

    def __init__(self, prepared: dict | None = None) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))
        for name, value in (prepared or {}).items():
            setattr(self, name, value)


def _recurrent_state() -> tuple[tuple, tuple, tuple]:
    """Conv and recurrent shapes and dtypes, from vLLM's calculators."""
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


def _kv_cache_specs() -> dict:
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


def _worker(
    *,
    bytes_used: int = MEASURED_BYTES_USED,
    host_bytes: int = 1024 * GIB,
    total_hbm: int = TOTAL_HBM_BYTES,
    model: torch.nn.Module | None = None,
):
    """A fake worker carrying the real methods and stubbed memory readings."""
    from vllm_neuron.vllm.worker.neuron_worker import NeuronWorker

    # num_gpu_blocks_override and the fields below it are read by vLLM's own
    # allocator, which one test drives with the budget this worker returns.
    cache_config = SimpleNamespace(
        block_size=BLOCK_SIZE_TOKENS,
        cache_dtype="auto",
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        mamba_cache_mode="none",
        mamba_block_size=None,
        num_gpu_blocks_override=None,
        # Prefix caching is on in the served configuration, so the allocator this
        # fixture drives hashes blocks exactly as the served run does.
        enable_prefix_caching=True,
        prefix_caching_hash_algo="sha256",
    )
    worker = SimpleNamespace(
        cache_config=cache_config,
        vllm_config=SimpleNamespace(
            cache_config=cache_config,
            model_config=SimpleNamespace(
                max_model_len=MAX_MODEL_LEN,
                original_max_model_len=MAX_MODEL_LEN,
                dtype=torch.bfloat16,
            ),
            scheduler_config=SimpleNamespace(
                max_num_seqs=MAX_NUM_SEQS,
                max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
                disable_hybrid_kv_cache_manager=False,
                # The served default: admission reserves the whole input length
                # rather than only the first chunk.
                scheduler_reserve_full_isl=True,
            ),
            # A latent page is per-rank, so context parallelism would divide the
            # tokens a rank keeps. Neither degree is engaged on this stack.
            parallel_config=SimpleNamespace(
                tensor_parallel_size=TP_WORLD_SIZE,
                pipeline_parallel_size=1,
                decode_context_parallel_size=1,
                prefill_context_parallel_size=1,
            ),
            kv_transfer_config=None,
        ),
        model_runner=SimpleNamespace(
            get_kv_cache_spec=_kv_cache_specs,
            model=model if model is not None else _FakeModel(),
            drafter=None,
        ),
    )
    for name in _WORKER_METHODS:
        setattr(worker, name, types.MethodType(getattr(NeuronWorker, name), worker))
    worker._query_host_runtime_memory = lambda: host_bytes
    worker._query_runtime_memory_stats = lambda: (
        bytes_used,
        total_hbm - bytes_used,
    )
    worker._get_byte_used_from_model = lambda: bytes_used
    # No compile cache: the graph need is the cold-cache reserve on every host.
    worker._neff_cache_dir = lambda: None
    worker._graphs_to_load = lambda: (3, None)
    return worker


def test_available_memory_is_the_block_rounded_served_need(monkeypatch) -> None:
    """The budget equals the blocks the served sequences can fill, and no more."""
    monkeypatch.setenv("VLLM_NEURON_CPU_MODE", "1")
    monkeypatch.delenv("VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB", raising=False)
    worker = _worker()

    assert worker.determine_available_memory() == EXPECTED_NEED_BYTES


def test_the_physical_core_bound_is_retired_for_the_logical_core(monkeypatch) -> None:
    """Retired premise: a rank's KV cache must fit one physical core (24 / 2 - 5 GiB).

    The bound assumed the runtime splits a rank's tensors over the two physical
    cores and stages a graph on one of them, so the cache had to leave a 5 GiB
    reserve inside 12 GiB. The runtime's own accounting says otherwise. At
    gate run tip-b64-C (server log,
    ``TDRV:dml_log_dev_neff_mem``, after each rank's 15 NEFF loads; the runtime
    prints binary units as GB): ``:529036`` ND 0 NC 0 holds 14.457 GiB = 13.378
    GiB of tensors + 0.75 GiB of shared scratchpad + the graphs' code, constants
    and rings, while ``:528992`` ND 0 NC 1 holds 0.196 GiB; ``:532035`` ND 13 NC 6
    holds 14.474 GiB, 13.378 GiB of it tensors. One physical core holds more than
    12 GiB and serves; the only limit is the logical core's 24 GiB. So the budget
    is that core's free memory (24 - 12.79 GiB resident here) less the graph
    need (the 5 GiB reserve: this fake reads no compile cache) less the margin:
    5.96 GiB, where the bound gave min(12 - 5, 2 x 7 - 12.79) = 1.21 GiB. The
    September serve that ran out of device memory at this residency held 12.79
    GiB of weights, 6.48 GiB of KV cache and 4.57 GiB of graph, about the
    logical core's 24 GiB; this budget refuses that 6.48 GiB cache.
    """
    monkeypatch.delenv("VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB", raising=False)
    monkeypatch.delenv("VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION", raising=False)
    monkeypatch.delenv("VLLM_NEURON_CPU_MODE", raising=False)
    monkeypatch.delenv("VLLM_NEURON_CPU_COMPILE", raising=False)
    worker = _worker()
    reserve_bytes = int(DEFAULT_RESERVE_GIB * GIB)

    heuristic = worker._compute_kv_budget(
        TOTAL_HBM_BYTES, MEASURED_BYTES_USED, GPU_MEMORY_UTILIZATION
    )
    available = worker.determine_available_memory()

    room_per_core = TOTAL_HBM_BYTES // 2 - reserve_bytes
    retired_bound = min(room_per_core, 2 * room_per_core - MEASURED_BYTES_USED)

    assert heuristic == TOTAL_HBM_BYTES - MEASURED_BYTES_USED - reserve_bytes - _margin()
    assert round(heuristic / GIB, 2) == 5.96
    assert round(retired_bound / GIB, 2) == 1.21
    assert heuristic < int(6.48 * GIB)
    assert available == EXPECTED_NEED_BYTES


def test_the_graph_reserve_is_overridable_and_validated(monkeypatch) -> None:
    """The override moves the budget; an unusable value is refused by name."""
    worker = _worker()

    def budget() -> int:
        return worker._compute_kv_budget(
            TOTAL_HBM_BYTES, MEASURED_BYTES_USED, GPU_MEMORY_UTILIZATION
        )

    monkeypatch.delenv("VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB", raising=False)
    default_budget = budget()
    monkeypatch.setenv("VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB", "measured")
    assert budget() == default_budget
    # 3 GiB keeps the measured term under the GMU limit (0.9 x 24 - 12.79 GiB).
    monkeypatch.setenv("VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB", "3.0")
    assert budget() == default_budget + int(DEFAULT_RESERVE_GIB * GIB) - 3 * GIB

    for offending in ("0", "-1", "not-a-number"):
        monkeypatch.setenv("VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB", offending)
        with pytest.raises(RuntimeError) as refusal:
            worker._get_graph_reserve_bytes()
        assert "VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB" in str(refusal.value)


def _pool_blocks(worker, budget_bytes: int) -> int:
    """The block count vLLM's allocator builds from a budget."""
    from vllm.v1.core.kv_cache_utils import get_kv_cache_configs

    return get_kv_cache_configs(
        worker.vllm_config, [_kv_cache_specs()], [budget_bytes]
    )[0].num_blocks


def test_cpu_compilation_and_execution_agree_on_the_bytes(monkeypatch) -> None:
    """Both modes return the same bytes AND the same block count."""
    from libtorch_neuronx_lite.compile.platform import get_total_available_memory

    monkeypatch.delenv("VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB", raising=False)
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    table_gib = get_total_available_memory()

    # The runtime's own footprint makes the device total a few MiB short of the
    # static table the compile path reads. The need is what closes that gap.
    monkeypatch.delenv("VLLM_NEURON_CPU_MODE", raising=False)
    monkeypatch.delenv("VLLM_NEURON_CPU_COMPILE", raising=False)
    device_worker = _worker(total_hbm=TOTAL_HBM_BYTES - 3 * 1024**2)
    device_bytes = device_worker.determine_available_memory()
    device_blocks = _pool_blocks(device_worker, device_bytes)

    monkeypatch.setenv("VLLM_NEURON_CPU_COMPILE", "1")
    compile_worker = _worker()
    compile_bytes = compile_worker.determine_available_memory()
    compile_blocks = _pool_blocks(compile_worker, compile_bytes)

    assert table_gib * GIB == TOTAL_HBM_BYTES
    assert device_bytes == compile_bytes == EXPECTED_NEED_BYTES
    assert device_blocks == compile_blocks == EXPECTED_BLOCKS


def test_vllm_accepts_the_budget_and_one_request_fits(monkeypatch) -> None:
    """vLLM's own allocator accepts the budget and holds the served tokens."""
    from vllm.v1.core.kv_cache_utils import (
        check_enough_kv_cache_memory,
        get_kv_cache_capacity,
        get_kv_cache_configs,
    )

    monkeypatch.delenv("VLLM_NEURON_CPU_MODE", raising=False)
    monkeypatch.delenv("VLLM_NEURON_CPU_COMPILE", raising=False)
    monkeypatch.delenv("VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB", raising=False)
    worker = _worker()
    available = worker.determine_available_memory()

    # Raises ValueError when the budget cannot hold one max_model_len request.
    check_enough_kv_cache_memory(worker.vllm_config, _kv_cache_specs(), available)
    config = get_kv_cache_configs(
        worker.vllm_config, [_kv_cache_specs()], [available]
    )[0]
    num_tokens, max_concurrency = get_kv_cache_capacity(worker.vllm_config, config)

    assert config.num_blocks >= BLOCKS_PER_REQUEST + 1
    assert num_tokens >= MAX_MODEL_LEN * MAX_NUM_SEQS
    assert max_concurrency >= MAX_NUM_SEQS


def _admit_one_full_length_request(worker, budget_bytes: int):
    """Drive vLLM's own KV cache manager over one max_model_len request.

    Returns the pool's block count and what ``allocate_slots`` decided. Prefix
    caching, the hash algorithm, the block sizes, the batched-token chunk and the
    full-input-length admission gate are the served configuration's.
    """
    from vllm.sampling_params import SamplingParams
    from vllm.utils.hashing import get_hash_fn_by_name
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import (
        get_kv_cache_configs,
        get_request_block_hasher,
        init_none_hash,
    )
    from vllm.v1.request import Request

    config = get_kv_cache_configs(
        worker.vllm_config, [_kv_cache_specs()], [budget_bytes]
    )[0]
    manager = KVCacheManager(
        kv_cache_config=config,
        max_model_len=MAX_MODEL_LEN,
        scheduler_block_size=BLOCK_SIZE_TOKENS,
        hash_block_size=BLOCK_SIZE_TOKENS,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        enable_caching=True,
    )
    hash_fn = get_hash_fn_by_name("sha256")
    init_none_hash(hash_fn)
    request = Request(
        request_id="one-full-length-request",
        prompt_token_ids=list(range(MAX_MODEL_LEN)),
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE_TOKENS, hash_fn),
    )
    allocated = manager.allocate_slots(
        request,
        MAX_NUM_BATCHED_TOKENS,
        full_sequence_must_fit=True,
    )
    return config.num_blocks, allocated


def test_the_allocator_admits_one_full_length_request(monkeypatch) -> None:
    """The allocator admits a max_model_len request; an underpriced pool refuses.

    This is the only path that prices what is allocated rather than what is
    admitted, so it is what stops a pool the admission arithmetic accepts and the
    allocator cannot serve.
    """
    monkeypatch.delenv("VLLM_NEURON_CPU_MODE", raising=False)
    monkeypatch.delenv("VLLM_NEURON_CPU_COMPILE", raising=False)
    monkeypatch.delenv("VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB", raising=False)
    worker = _worker()
    available = worker.determine_available_memory()

    blocks, allocated = _admit_one_full_length_request(worker, available)
    underpriced_blocks, underpriced = _admit_one_full_length_request(
        worker, UNDERPRICED_BYTES
    )

    assert blocks == EXPECTED_BLOCKS
    assert allocated is not None
    assert underpriced_blocks == UNDERPRICED_BLOCKS
    assert underpriced is None


def test_prepared_operands_count_once_towards_residency(monkeypatch) -> None:
    """Operands kept on module attributes are counted, and counted once."""
    monkeypatch.delenv("VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB", raising=False)
    operand_a = torch.zeros(1024, 512, dtype=torch.bfloat16)
    operand_b = torch.zeros(256, dtype=torch.float32)
    model = _FakeModel(
        {
            # The two shapes of holder the model uses: a dict of operands and a
            # single tensor.
            "_prepared_kernel_operands": {"gate_up": operand_a, "down": operand_b},
            # Already a parameter, so residency must not count it twice.
            "_prepared_alias": None,
            # Not tensors, and not prepared: both ignored.
            "_prepared_retile_health": {"gate": (0, 0, 0)},
            "cached_host_copy": torch.zeros(4096, dtype=torch.float32),
        }
    )
    model._prepared_alias = model.weight.data
    worker = _worker(model=model)

    prepared_bytes = worker._prepared_operand_bytes()
    # The estimate taken without a device charges them against the free memory.
    estimate_with = worker._compute_kv_budget(
        TOTAL_HBM_BYTES, MEASURED_BYTES_USED, GPU_MEMORY_UTILIZATION
    )
    estimate_without = _worker()._compute_kv_budget(
        TOTAL_HBM_BYTES, MEASURED_BYTES_USED, GPU_MEMORY_UTILIZATION
    )

    assert prepared_bytes == operand_a.nbytes + operand_b.nbytes
    assert estimate_without - estimate_with == prepared_bytes


def test_prepared_operands_on_the_meta_device_are_counted(monkeypatch) -> None:
    """A model on meta reports its operands, and its parameter only once.

    The compile path can hold the model on the meta device, where a tensor has no
    memory and reports a zero pointer instead of raising. Reading that zero as an
    identity would make every meta tensor after the first look already counted,
    and the operand bytes would vanish in exactly the mode whose block count has
    to match the device's.
    """
    monkeypatch.delenv("VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB", raising=False)
    operand = torch.zeros(1024, 512, dtype=torch.bfloat16, device="meta")
    model = _FakeModel({"_prepared_kernel_operands": {"gate_up": operand}})
    model.to("meta")
    # The parameter itself, reachable twice: as a parameter and as a prepared
    # attribute. It must be counted neither twice nor at all.
    model._prepared_alias = model.weight

    prepared_bytes = _worker(model=model)._prepared_operand_bytes()

    assert model.weight.device.type == "meta"
    assert prepared_bytes == operand.nbytes


def test_a_need_over_the_budget_is_refused_by_name(monkeypatch) -> None:
    """A budget under the need refuses, naming both figures and the mode."""
    monkeypatch.setenv("VLLM_NEURON_CPU_MODE", "1")
    monkeypatch.delenv("VLLM_NEURON_CPU_COMPILE", raising=False)
    monkeypatch.delenv("VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB", raising=False)
    # A fair share far under the need: the cache cannot take the served shape.
    worker = _worker(host_bytes=32 * 1024**2)

    with pytest.raises(RuntimeError) as refusal:
        worker.determine_available_memory()

    message = str(refusal.value)
    assert f"{EXPECTED_NEED_BYTES / GIB:.3f} GiB" in message
    assert "cpu mode" in message
    assert "VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB" in message


def test_the_allocation_adds_a_bank_per_recurrent_layer_and_fits_the_budget(
    monkeypatch,
) -> None:
    """The allocation is the need plus one bank per recurrent layer, inside the budget."""
    monkeypatch.setenv("VLLM_NEURON_CPU_MODE", "1")
    monkeypatch.delenv("VLLM_NEURON_CPU_COMPILE", raising=False)
    monkeypatch.delenv("VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB", raising=False)
    monkeypatch.delenv("VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION", raising=False)
    worker = _worker()

    need = worker._kv_cache_need_bytes()
    allocated = worker._kv_cache_footprint_bytes(need)
    budget = worker._compute_kv_budget(
        TOTAL_HBM_BYTES, MEASURED_BYTES_USED, GPU_MEMORY_UTILIZATION
    )
    available = worker.determine_available_memory()

    assert need == EXPECTED_NEED_BYTES
    assert allocated == EXPECTED_FOOTPRINT_BYTES
    assert available == need
    assert allocated <= budget

    # A budget the need fits and the allocation does not refuses, naming the
    # allocation as well as the need. CPU mode: the host share x GMU, uncapped.
    share = (need + allocated) // 2
    between = _worker(host_bytes=int(share / GPU_MEMORY_UTILIZATION))
    with pytest.raises(RuntimeError) as refusal:
        between.determine_available_memory()
    message = str(refusal.value)
    assert f"{allocated / GIB:.3f} GiB" in message
    assert f"{need / GIB:.3f} GiB" in message


def test_the_cap_fraction_is_retired_as_a_default(monkeypatch) -> None:
    """Retired premise: a fixed 0.30 of GMU x HBM is a safe KV budget by default.

    The cap stood in for the graphs' device memory, which nothing measured. The
    graph need is now read from the compiled NEFFs (or the reserve stands in on a
    cold cache), so the cap applies only when
    ``VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION`` is set. Set to 0.30 it still gives
    30x the served need at bs=1 x 4k, and the measured budget gives more; either
    way the returned bytes are the need, so the block count is the compiled
    graph's.
    """
    monkeypatch.delenv("VLLM_NEURON_CPU_MODE", raising=False)
    monkeypatch.delenv("VLLM_NEURON_CPU_COMPILE", raising=False)
    # A 3 GiB graph need keeps the measured term under the GMU limit (21.6 - 1 GiB).
    monkeypatch.setenv("VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB", "3.0")
    monkeypatch.setenv("VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION", "0.30")
    worker = _worker(bytes_used=GIB)

    capped = worker._compute_kv_budget(TOTAL_HBM_BYTES, GIB, GPU_MEMORY_UTILIZATION)
    returned = worker.determine_available_memory()
    monkeypatch.delenv("VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION")
    measured = worker._compute_kv_budget(TOTAL_HBM_BYTES, GIB, GPU_MEMORY_UTILIZATION)

    assert capped == CAP_FRACTION_BUDGET_BYTES
    assert measured == TOTAL_HBM_BYTES - GIB - 3 * GIB - _margin()
    assert returned == EXPECTED_NEED_BYTES
    # About 30x on these figures: an order of magnitude over the served need.
    assert capped >= 25 * returned
    assert measured > capped


def test_an_over_allocation_is_refused_with_the_shortfall_named(monkeypatch) -> None:
    """A footprint above free - graph need - margin is refused, naming the shortfall.

    On the device path, with the graph need set so the budget is half the footprint.
    The 1 GiB minimum-budget floor is off so the footprint check is the one that
    refuses.
    """
    monkeypatch.delenv("VLLM_NEURON_CPU_MODE", raising=False)
    monkeypatch.delenv("VLLM_NEURON_CPU_COMPILE", raising=False)
    monkeypatch.delenv("VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION", raising=False)
    monkeypatch.setenv("VLLM_NEURON_MIN_KV_BUDGET_GIB", "0")
    free = TOTAL_HBM_BYTES - MEASURED_BYTES_USED
    graph_bytes = free - _margin() - EXPECTED_FOOTPRINT_BYTES // 2
    monkeypatch.setenv("VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB", repr(graph_bytes / GIB))
    worker = _worker()
    budget = worker._compute_kv_budget(
        TOTAL_HBM_BYTES, MEASURED_BYTES_USED, GPU_MEMORY_UTILIZATION
    )
    assert budget < EXPECTED_FOOTPRINT_BYTES

    with pytest.raises(RuntimeError) as refusal:
        worker.determine_available_memory()

    message = str(refusal.value)
    assert "neuron mode" in message
    assert f"free {free / GIB:.2f} GiB (runtime)" in message
    assert f"graph need {graph_bytes / GIB:.2f} GiB" in message
    assert f"budget {budget / GIB:.3f} GiB" in message
    assert f"{EXPECTED_FOOTPRINT_BYTES / GIB:.3f} GiB" in message
    assert f"short by {(EXPECTED_FOOTPRINT_BYTES - budget) / GIB:.3f} GiB" in message
