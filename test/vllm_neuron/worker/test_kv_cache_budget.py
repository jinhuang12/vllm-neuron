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
# What the runner allocates once vLLM has sized the blocks from that need: the
# latent pools as vLLM laid them out, plus one bank of the same size for every
# recurrent layer, because a recurrent bank is addressed by request slot and
# cannot share a tensor with any other layer.
EXPECTED_FOOTPRINT_BYTES = (
    EXPECTED_BLOCKS * PAGE_SIZE_BYTES * (LAYERS_PER_POOL + RECURRENT_LAYERS)
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
    "_compute_kv_budget",
    "_kv_cache_need_bytes",
    "_kv_cache_footprint_bytes",
    "_physical_core_kv_bound",
    "_prepared_operand_bytes",
    "_get_graph_reserve_bytes",
    "_get_kv_cap_fraction",
    "_determine_available_memory_cpu",
    "_determine_available_memory_neuron",
    "_estimate_available_memory_neuron",
)


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
    return worker


def test_available_memory_is_the_block_rounded_served_need(monkeypatch) -> None:
    """The budget equals the blocks the served sequences can fill, and no more."""
    monkeypatch.setenv("VLLM_NEURON_CPU_MODE", "1")
    monkeypatch.delenv("VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB", raising=False)
    worker = _worker()

    assert worker.determine_available_memory() == EXPECTED_NEED_BYTES


def test_the_heuristic_is_bounded_by_one_physical_core(monkeypatch) -> None:
    """A logical core reporting 24 GiB budgets against 12 GiB, less reserve."""
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
    assert overridden_bound > default_bound

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
    bound_with = worker._physical_core_kv_bound(
        TOTAL_HBM_BYTES, MEASURED_BYTES_USED + prepared_bytes
    )
    bound_without = worker._physical_core_kv_bound(
        TOTAL_HBM_BYTES, MEASURED_BYTES_USED
    )

    assert prepared_bytes == operand_a.nbytes + operand_b.nbytes
    assert bound_with < bound_without


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


def test_the_allocation_adds_a_bank_per_recurrent_layer_and_fits_the_bound(
    monkeypatch,
) -> None:
    """The allocation is the need plus one bank per recurrent layer, inside the bound."""
    monkeypatch.setenv("VLLM_NEURON_CPU_MODE", "1")
    monkeypatch.delenv("VLLM_NEURON_CPU_COMPILE", raising=False)
    monkeypatch.delenv("VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB", raising=False)
    worker = _worker()

    need = worker._kv_cache_need_bytes()
    allocated = worker._kv_cache_footprint_bytes(need)
    bound = worker._physical_core_kv_bound(
        TOTAL_HBM_BYTES, MEASURED_BYTES_USED + worker._prepared_operand_bytes()
    )
    available = worker.determine_available_memory()

    assert need == EXPECTED_NEED_BYTES
    assert allocated == EXPECTED_FOOTPRINT_BYTES
    assert available == need
    assert allocated <= bound

    # A budget the need fits and the allocation does not refuses, naming the
    # allocation as well as the need.
    share = (need + allocated) // 2
    between = _worker(
        host_bytes=int(share / (GPU_MEMORY_UTILIZATION * worker._get_kv_cap_fraction()))
    )
    with pytest.raises(RuntimeError) as refusal:
        between.determine_available_memory()
    message = str(refusal.value)
    assert f"{allocated / GIB:.3f} GiB" in message
    assert f"{need / GIB:.3f} GiB" in message


def test_the_unbounded_logical_core_budget_over_allocates(monkeypatch) -> None:
    """Without the core bound and the need cap, the cap fraction over-allocates."""
    monkeypatch.delenv("VLLM_NEURON_CPU_MODE", raising=False)
    monkeypatch.delenv("VLLM_NEURON_CPU_COMPILE", raising=False)
    monkeypatch.delenv("VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB", raising=False)
    worker = _worker()
    bounded = worker.determine_available_memory()

    worker._physical_core_kv_bound = lambda *_: TOTAL_HBM_BYTES
    unbounded = worker._compute_kv_budget(
        TOTAL_HBM_BYTES, MEASURED_BYTES_USED, GPU_MEMORY_UTILIZATION
    )

    assert unbounded == CAP_FRACTION_BUDGET_BYTES
    assert bounded == EXPECTED_NEED_BYTES
    # About 30x on these figures: an order of magnitude over the served need.
    assert unbounded >= 25 * bounded
