# SPDX-License-Identifier: Apache-2.0
"""The KV and state budget of GLM-5.3-Flash at TP=64: bs=64 at max_model_len 8192.

The layer specs are the real model's (``Glm5NextForConditionalGeneration.get_kv_spec``
on the 45-layer fixture, built at world size 64), translated by the real
``NeuronModelRunner.get_kv_cache_spec``, grouped by vLLM's own
``get_kv_cache_groups``, priced by the real ``NeuronWorker`` methods, and sized by
the real ``kv_cache_allocations``. Nothing on the "after" side is a hand formula.

The "before" side is the pricing at commit 5938748, kept here as
:func:`before_fix_footprint_bytes`. It is anchored to the serve log of that commit
(``/home/ubuntu/glm53f-moefix-20261005T115338Z/server_decode.log``: "KV cache need:
0.216 GiB for 1 sequence(s) of 4096 tokens (161 blocks of 131072 B, ...)" and
"allocated=0.884 GiB").

Per-rank numbers at TP=64, block size 128 (``hybrid_kv_block_size``):

* KDA state per request per layer: recurrent ``(1, 128, 128)`` fp32 = 65536 B plus
  conv ``(3, 384)`` bf16 = 2304 B, so 67840 B.
* MLA per token per layer: latent 512 x bf16 = 1024 B in the vLLM page, plus the
  indexer pool 128 x bf16 per 4 tokens = 64 B in the runner's side cache, so
  1088 B.

Run with ``VLLM_NEURON_CPU_MODE=1`` (``test/conftest.py`` pins it).
"""

from __future__ import annotations

import copy
import functools
import json
import math
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

GIB = 1024**3

TP_WORLD_SIZE = 64
BLOCK_SIZE_TOKENS = 128
MLA_LAYERS = 11
KDA_LAYERS = 34

#: The served point the gate needs.
SERVED_MAX_NUM_SEQS = 64
SERVED_MAX_MODEL_LEN = 8192
MAX_NUM_BATCHED_TOKENS = 1024
#: vLLM 0.24's default ``gpu_memory_utilization``; the 5938748 serve sets none, and
#: its log reads ``total_budget=22.08 GiB`` = 0.92 x 24 GiB.
GPU_MEMORY_UTILIZATION = 0.92

#: The recurrent state of the one KDA head a rank holds at TP=64: 128 x 128 fp32.
RECURRENT_STATE_BYTES = 128 * 128 * 4

#: The MLA layout this budget prices, per token per layer.
LATENT_BYTES_PER_TOKEN = 512 * 2
INDEXER_BYTES_PER_TOKEN = 128 * 2 // 4
MLA_BYTES_PER_TOKEN = 1088

#: Ceiling on KV + state per rank at the served point.
SERVED_BUDGET_CEILING_BYTES = 8 * GIB

#: The 5938748 pricing must be at least this at the served point.
BEFORE_FLOOR_BYTES = 50 * GIB

#: The measured device at 5938748 (server_decode.log, every rank): 24.00 GiB per
#: logical core, 2.79 GiB of parameters and buffers, 7.96 GiB resident once the
#: prepared kernel operands are counted.
TOTAL_HBM_BYTES = 24 * GIB
MEASURED_PARAM_BYTES = int(2.79 * GIB)
MEASURED_RESIDENT_BYTES = int(7.96 * GIB)

#: The measured device at 0a08ff4 on the bs=64 @ 8k serve line, every rank
#: (``gate/runs/indexer-A/server.log``): "bytes_used=1.63 GiB, resident=6.79 GiB ...
#: effective=6.62 GiB".
TIP_PARAM_BYTES = int(1.63 * GIB)
TIP_RESIDENT_BYTES = int(6.79 * GIB)

FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "model"
    / "glm5_next"
    / "fixtures"
    / "config.json"
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
    "_get_graph_reserve_bytes",
    "_get_kv_cap_fraction",
    "_determine_available_memory_cpu",
    "_determine_available_memory_neuron",
    "_estimate_available_memory_neuron",
)

#: Graphs the bs=64 @ 8k line warms: 1 prefill target, 7 batch x 2 ctx decode.
SERVED_GRAPHS = 15


# ---------------------------------------------------------------------------
# The 5938748 pricing, kept as the reference the change is measured against.
# ---------------------------------------------------------------------------


def before_fix_footprint_bytes(
    *,
    max_num_seqs: int,
    max_model_len: int,
    block_size: int = BLOCK_SIZE_TOKENS,
    mla_layers: int = MLA_LAYERS,
    kda_layers: int = KDA_LAYERS,
    latent_width: int = 512,
    dtype_bytes: int = 2,
) -> int:
    """Bytes per rank the allocator at 5938748 sets aside for this served point.

    At 5938748 every recurrent spec carried the attention block size and the
    attention page (``page_size_padded``), so the worker's need priced each of the
    four recurrent groups at ``cdiv(max_model_len, block_size)`` blocks per request
    like the latent group, and ``kv_cache_allocations`` then gave every recurrent
    layer its own buffer the size of a whole shared pool tensor. The KV-cache
    tensors therefore total ``num_blocks x page x (latent layers + recurrent
    layers)``. The indexer side caches are not included (they were not priced).
    """
    page = block_size * latent_width * dtype_bytes
    groups = 1 + -(-kda_layers // mla_layers)
    blocks_per_request = groups * -(-max_model_len // block_size)
    num_blocks = blocks_per_request * max_num_seqs + 1
    return num_blocks * page * (mla_layers + kda_layers)


# ---------------------------------------------------------------------------
# The real model's specs, and fakes that carry the real methods.
# ---------------------------------------------------------------------------


def glm53f_layer_specs():
    """The real model's ``LayerSpec`` list at TP=64, built on the CPU without weights."""
    from vllm_neuron.model.glm5_next import model_fp8

    raw = json.loads(FIXTURE_PATH.read_text())
    original = model_fp8._resolve_world_size
    model_fp8._resolve_world_size = lambda: TP_WORLD_SIZE
    try:
        model = model_fp8.Glm5NextForConditionalGeneration.from_configs(
            copy.deepcopy(raw)
        )
    finally:
        model_fp8._resolve_world_size = original
    return model.get_kv_spec().layers, model.text_config


class _SpecOnlyModel:
    """A model exposing ``get_kv_spec``, a recording ``bind_kv_cache`` and its text config.

    ``text_config`` carries the two indexer dials (``index_kpool``,
    ``index_head_dim``) the worker prices the DSA side caches from.
    """

    def __init__(self, layers, text_config=None) -> None:
        self._spec = SimpleNamespace(layers=list(layers))
        self.text_config = text_config
        self.bound: list[dict] = []

    def get_kv_spec(self):
        return self._spec

    def bind_kv_cache(self, kv_caches) -> None:
        self.bound.append(kv_caches)


def served_vllm_config(
    *, max_num_seqs: int, max_model_len: int, gate_knobs: bool = True
):
    """The engine config fields vLLM's KV-cache code and the worker read.

    ``gate_knobs`` is the gate's serve line: ``--no-enable-prefix-caching
    --mamba-block-size <max_model_len>``. Without it the config is the 5938748 serve
    line's (prefix caching on, no mamba block size).
    """
    cache_config = SimpleNamespace(
        block_size=BLOCK_SIZE_TOKENS,
        cache_dtype="auto",
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        mamba_cache_mode="none",
        mamba_block_size=max_model_len if gate_knobs else None,
        num_gpu_blocks_override=None,
        enable_prefix_caching=not gate_knobs,
        prefix_caching_hash_algo="sha256",
        hash_block_size=None,
    )
    return SimpleNamespace(
        cache_config=cache_config,
        model_config=SimpleNamespace(
            max_model_len=max_model_len,
            original_max_model_len=max_model_len,
            dtype=torch.bfloat16,
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
            disable_hybrid_kv_cache_manager=False,
            scheduler_reserve_full_isl=True,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=TP_WORLD_SIZE,
            pipeline_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        kv_transfer_config=None,
    )


@functools.lru_cache(maxsize=1)
def _real_text_config():
    """The real model's text config (indexer dials included), built once."""
    return glm53f_layer_specs()[1]


def fake_runner(
    layers,
    *,
    max_num_seqs: int,
    max_model_len: int,
    gate_knobs: bool = True,
    text_config=None,
):
    """A runner-shaped object carrying the real KV-cache methods of ``NeuronModelRunner``.

    The model carries the real text config unless ``text_config`` is given, so the
    worker prices the DSA indexer side caches as it does for the served model.
    """
    from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

    fake = SimpleNamespace(
        vllm_config=served_vllm_config(
            max_num_seqs=max_num_seqs,
            max_model_len=max_model_len,
            gate_knobs=gate_knobs,
        ),
        neuron_config=SimpleNamespace(fp8_packed_kv=False),
        speculative_config=None,
        drafter=None,
        device=torch.device("cpu"),
        max_num_reqs=max_num_seqs,
        max_model_len=max_model_len,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        vocab_size=128,
        is_pooling_model=False,
        model=_SpecOnlyModel(
            layers,
            text_config if text_config is not None else _real_text_config(),
        ),
        _kv_cache_full_tensors={},
    )
    fake.get_kv_cache_spec = types.MethodType(NeuronModelRunner.get_kv_cache_spec, fake)
    fake._kv_cache_is_fp8_packed = types.MethodType(
        NeuronModelRunner._kv_cache_is_fp8_packed, fake
    )
    fake._k_cache_alloc_shape = NeuronModelRunner._k_cache_alloc_shape
    return fake


def fake_worker(
    runner,
    *,
    param_bytes: int = MEASURED_PARAM_BYTES,
    resident_bytes: int = MEASURED_RESIDENT_BYTES,
    total_hbm: int = TOTAL_HBM_BYTES,
    runtime_used_bytes: int | None = None,
    graph_cache_dir: Path | None = None,
    expected_graphs: int = SERVED_GRAPHS,
):
    """A worker-shaped object carrying the real budget methods of ``NeuronWorker``.

    The runtime reports ``runtime_used_bytes`` used (``resident_bytes`` unless
    given) out of ``total_hbm``. The graph need is read from ``graph_cache_dir``,
    which warmup is taken to fill with ``expected_graphs`` graphs; without one the
    worker reads a cache that does not exist, so the need is the cold-cache
    reserve and no test depends on the compile caches of the host it runs on.
    """
    from vllm_neuron.vllm.worker.neuron_worker import NeuronWorker

    worker = SimpleNamespace(
        cache_config=runner.vllm_config.cache_config,
        vllm_config=runner.vllm_config,
        model_runner=runner,
    )
    for name in _WORKER_METHODS:
        setattr(worker, name, types.MethodType(getattr(NeuronWorker, name), worker))
    used = resident_bytes if runtime_used_bytes is None else runtime_used_bytes
    worker._query_runtime_memory_stats = lambda: (used, total_hbm - used)
    worker._get_byte_used_from_model = lambda: param_bytes
    worker._prepared_operand_bytes = lambda: resident_bytes - param_bytes
    cache_dir = graph_cache_dir if graph_cache_dir is not None else Path("/nonexistent/compile_cache")
    worker._neff_cache_dir = lambda: cache_dir
    worker._graphs_to_load = lambda: (expected_graphs, None)
    return worker


def served_kv_cache_config(runner, worker):
    """The KV cache config vLLM builds from the budget the worker returns."""
    from vllm.v1.core.kv_cache_utils import get_kv_cache_configs

    available = worker._kv_cache_need_bytes()
    return get_kv_cache_configs(
        runner.vllm_config, [runner.get_kv_cache_spec()], [available]
    )[0]


def side_cache_bytes(text_config, *, max_num_seqs: int, max_model_len: int) -> int:
    """Bytes of the indexer side caches the runner allocates, read off its own builder.

    The builder allocates on the latent bank's device, so a meta latent bank sizes
    them without touching memory.
    """
    from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

    latent = torch.empty((0,), dtype=torch.bfloat16, device="meta")
    banks = [{"family": "self_attn", "latent_cache": latent}] * MLA_LAYERS + [
        {"family": "linear_attn"}
    ] * KDA_LAYERS
    side = NeuronModelRunner._glm5next_side_caches(
        banks,
        index_kpool=int(text_config.index_kpool),
        index_head_dim=int(text_config.index_head_dim),
        max_seq_len=max_model_len,
        request_slots=max_num_seqs,
    )
    return sum(t.nbytes for entry in side for t in entry.values())


def after_fix_bytes(
    *,
    max_num_seqs: int,
    max_model_len: int,
    gate_knobs: bool = True,
    model_specs=None,
) -> dict:
    """What the worker prices and the runner allocates at one served point, by part.

    ``model_specs`` is a ``glm53f_layer_specs()`` result to reuse; built when None.
    ``footprint_bytes`` is the worker's check, which counts the indexer side caches,
    so it equals ``total_bytes``.
    """
    from vllm.v1.kv_cache_interface import MambaSpec

    from vllm_neuron.vllm.worker.neuron_model_runner import kv_cache_allocations

    layers, text_config = model_specs or glm53f_layer_specs()
    runner = fake_runner(
        layers,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        gate_knobs=gate_knobs,
        text_config=text_config,
    )
    worker = fake_worker(runner)
    need = worker._kv_cache_need_bytes()
    footprint = worker._kv_cache_footprint_bytes(need)
    config = served_kv_cache_config(runner, worker)
    specs = runner.get_kv_cache_spec()
    allocations = kv_cache_allocations(config, state_slots=max_num_seqs)
    recurrent = sum(
        size for size, owners in allocations if isinstance(specs[owners[0]], MambaSpec)
    )
    side = side_cache_bytes(
        text_config, max_num_seqs=max_num_seqs, max_model_len=max_model_len
    )
    return {
        "need_bytes": need,
        "footprint_bytes": footprint,
        "pool_bytes": footprint - recurrent - side,
        "recurrent_state_bytes": recurrent,
        "indexer_side_cache_bytes": side,
        "total_bytes": footprint,
        "num_blocks": config.num_blocks,
    }


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def served():
    layers, text_config = glm53f_layer_specs()
    runner = fake_runner(
        layers, max_num_seqs=SERVED_MAX_NUM_SEQS, max_model_len=SERVED_MAX_MODEL_LEN
    )
    worker = fake_worker(runner)
    return SimpleNamespace(
        layers=layers,
        text_config=text_config,
        runner=runner,
        worker=worker,
        specs=runner.get_kv_cache_spec(),
        config=served_kv_cache_config(runner, worker),
    )


def _kda_state_bytes(spec) -> tuple[int, int]:
    """``(conv_bytes, recurrent_bytes)`` of one recurrent spec, from its own carriers."""
    conv, recurrent = (
        math.prod(shape) * dtype.itemsize
        for shape, dtype in zip(spec.shapes, spec.dtypes, strict=True)
    )
    return conv, recurrent


def test_the_reference_reproduces_the_as_built_serve_log() -> None:
    """At bs=1 and 4096 tokens the 5938748 pricing is the 0.884 GiB the serve logged."""
    before = before_fix_footprint_bytes(max_num_seqs=1, max_model_len=4096)

    assert before == 161 * 131072 * 45
    assert round(before / GIB, 3) == 0.884


def test_the_budget_reproduces_the_as_built_serve_log(monkeypatch) -> None:
    """The serve log's cap still binds where its knob is set; by default it is gone.

    ``server_decode.log`` (5938748): "cap=6.62 GiB, physical_core_bound=6.04 GiB ...
    effective=6.04 GiB" with the default knobs of the time. Since wt2/kvbudget the
    0.30 cap applies only when ``VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION`` is set and
    the per-physical-core bound not at all (``test_kv_budget_measured.py``). The
    default budget is the free memory (24 - 7.96 resident) less the graph need (the
    5 GiB reserve: the fake worker reads no compile cache) less the margin.
    """
    from vllm_neuron.vllm.worker.neuron_worker import KV_BUDGET_MARGIN_BYTES

    monkeypatch.delenv("VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB", raising=False)
    monkeypatch.delenv("VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION", raising=False)
    layers, _ = glm53f_layer_specs()
    worker = fake_worker(fake_runner(layers, max_num_seqs=1, max_model_len=4096))

    def budget() -> int:
        return worker._compute_kv_budget(
            TOTAL_HBM_BYTES, MEASURED_PARAM_BYTES, GPU_MEMORY_UTILIZATION
        )

    measured = budget()
    monkeypatch.setenv("VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION", "0.30")
    capped = budget()

    cap = int(int(TOTAL_HBM_BYTES * GPU_MEMORY_UTILIZATION) * 0.30)
    assert round(cap / GIB, 2) == 6.62
    assert capped == cap
    assert measured == (
        TOTAL_HBM_BYTES - MEASURED_RESIDENT_BYTES - 5 * GIB - KV_BUDGET_MARGIN_BYTES
    )
    assert round(measured / GIB, 2) == 10.79


def test_the_default_line_prices_bs1_at_4k_exactly_as_5938748() -> None:
    """Without the two flags, bs=1 at 4096 tokens prices vLLM's pool as 5938748 did.

    The 5938748 serve log: "KV cache need: 0.216 GiB for 1 sequence(s) of 4096 tokens
    (161 blocks of 131072 B, 160 allocator block(s) per request, 5 group(s), 11
    layer(s) per pool)". The specs, the groups, the block count, the need and the
    pool tensors are those numbers, so the gate's bs=1 point needs no flag change.
    The one difference is the fix itself: each of the 34 KDA layers now holds one
    67840 B request slot instead of a copy of a 161-block pool tensor.
    """
    from vllm.v1.kv_cache_interface import MambaSpec

    layers, _ = glm53f_layer_specs()
    runner = fake_runner(layers, max_num_seqs=1, max_model_len=4096, gate_knobs=False)
    worker = fake_worker(runner)
    specs = runner.get_kv_cache_spec()
    config = served_kv_cache_config(runner, worker)
    need = worker._kv_cache_need_bytes()
    page = BLOCK_SIZE_TOKENS * LATENT_BYTES_PER_TOKEN

    assert {spec.block_size for spec in specs.values()} == {BLOCK_SIZE_TOKENS}
    assert {spec.page_size_bytes for spec in specs.values()} == {page} == {131072}
    assert sum(isinstance(spec, MambaSpec) for spec in specs.values()) == KDA_LAYERS
    assert len(config.kv_cache_groups) == 5
    assert config.num_blocks == 161
    assert need == 161 * page * MLA_LAYERS
    assert round(need / GIB, 3) == 0.216
    assert [tensor.size for tensor in config.kv_cache_tensors] == [161 * page] * MLA_LAYERS

    slot = RECURRENT_STATE_BYTES + 3 * 384 * 2
    footprint = worker._kv_cache_footprint_bytes(need)
    before = before_fix_footprint_bytes(max_num_seqs=1, max_model_len=4096)
    # Since wt2/kvseg the footprint also counts the DSA indexer side caches, which
    # 5938748 did not price.
    side = side_cache_bytes(_real_text_config(), max_num_seqs=1, max_model_len=4096)
    assert footprint == need + KDA_LAYERS * 1 * slot + side
    assert before - (footprint - side) == KDA_LAYERS * (161 * page - slot)


def test_the_opt_in_line_prices_one_block_per_kda_group_per_request() -> None:
    """With ``--mamba-block-size <max_model_len>`` a KDA group costs 1 block per request.

    The latent group still costs ``cdiv(max_model_len, 128)`` blocks per request; each
    of the four KDA groups costs one. Checked at the served point and at bs=1, 4k.
    vLLM's allocator holding exactly that is
    ``test_vllms_allocator_admits_64_full_length_requests_in_the_priced_pool``.
    """
    from vllm.v1.kv_cache_interface import MambaSpec

    layers, _ = glm53f_layer_specs()
    page = BLOCK_SIZE_TOKENS * LATENT_BYTES_PER_TOKEN
    kda_groups = -(-KDA_LAYERS // MLA_LAYERS)
    for seqs, length in ((SERVED_MAX_NUM_SEQS, SERVED_MAX_MODEL_LEN), (1, 4096)):
        runner = fake_runner(layers, max_num_seqs=seqs, max_model_len=length)
        worker = fake_worker(runner)
        config = served_kv_cache_config(runner, worker)
        recurrent_groups = [
            group
            for group in config.kv_cache_groups
            if isinstance(group.kv_cache_spec, MambaSpec)
        ]
        blocks_per_request = -(-length // BLOCK_SIZE_TOKENS) + kda_groups * 1

        assert len(recurrent_groups) == kda_groups
        assert {
            -(-length // group.kv_cache_spec.block_size) for group in recurrent_groups
        } == {1}
        assert config.num_blocks == blocks_per_request * seqs + 1
        assert worker._kv_cache_need_bytes() == (
            (blocks_per_request * seqs + 1) * page * MLA_LAYERS
        )
    # The served point: 64 x (64 + 4) + 1 blocks.
    assert (64 + 4) * SERVED_MAX_NUM_SEQS + 1 == 4353


def test_each_kda_layer_is_charged_its_real_per_request_state(served) -> None:
    """Every recurrent layer's buffer is ``max_num_seqs`` slots of about its own state."""
    from vllm.v1.kv_cache_interface import MambaSpec

    from vllm_neuron.vllm.worker.neuron_model_runner import kv_cache_allocations

    kda = {n: s for n, s in served.specs.items() if isinstance(s, MambaSpec)}
    assert len(kda) == KDA_LAYERS
    conv_bytes, recurrent_bytes = {_kda_state_bytes(s) for s in kda.values()}.pop()
    assert recurrent_bytes == RECURRENT_STATE_BYTES
    ceiling = 1.5 * (RECURRENT_STATE_BYTES + conv_bytes)

    allocations = kv_cache_allocations(served.config, state_slots=SERVED_MAX_NUM_SEQS)
    by_owner = {owners[0]: size for size, owners in allocations if len(owners) == 1}
    per_slot = {by_owner[name] / SERVED_MAX_NUM_SEQS for name in kda}

    assert len({by_owner[name] for name in kda}) == 1
    assert len(per_slot) == 1
    (slot_bytes,) = per_slot
    assert recurrent_bytes + conv_bytes <= slot_bytes <= ceiling


def test_mla_bytes_per_token_per_layer_are_the_latent_plus_indexer_layout(served) -> None:
    """1088 B per token per MLA layer: 1024 latent in the page plus 64 indexer."""
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    mla = [s for s in served.specs.values() if isinstance(s, MLAAttentionSpec)]
    assert len(mla) == MLA_LAYERS
    latent = {s.page_size_bytes // s.block_size for s in mla}
    assert {s.page_size_bytes % s.block_size for s in mla} == {0}

    # The side caches: per layer and slot, ``max_model_len // index_kpool + 1``
    # pooled rows of ``index_head_dim`` plus a fixed tail ring. The per-token part
    # is the pooled rows; the spare row and the ring are per slot.
    length = SERVED_MAX_MODEL_LEN
    side = side_cache_bytes(served.text_config, max_num_seqs=1, max_model_len=length)
    side_longer = side_cache_bytes(
        served.text_config, max_num_seqs=1, max_model_len=2 * length
    )
    indexer = (side_longer - side) // (MLA_LAYERS * length)

    assert latent == {LATENT_BYTES_PER_TOKEN}
    assert indexer == INDEXER_BYTES_PER_TOKEN
    assert LATENT_BYTES_PER_TOKEN + INDEXER_BYTES_PER_TOKEN == MLA_BYTES_PER_TOKEN


def test_kv_and_state_for_64_sequences_at_8k_fit_in_8_gib() -> None:
    """Pool + recurrent banks + indexer side caches, per rank, at bs=64 and 8192."""
    after = after_fix_bytes(
        max_num_seqs=SERVED_MAX_NUM_SEQS, max_model_len=SERVED_MAX_MODEL_LEN
    )
    latent_floor = (
        SERVED_MAX_NUM_SEQS
        * SERVED_MAX_MODEL_LEN
        * LATENT_BYTES_PER_TOKEN
        * MLA_LAYERS
    )

    assert after["pool_bytes"] >= latent_floor
    assert after["num_blocks"] >= SERVED_MAX_NUM_SEQS * SERVED_MAX_MODEL_LEN // 128 + 1
    assert after["total_bytes"] <= SERVED_BUDGET_CEILING_BYTES


def test_the_5938748_pricing_is_over_50_gib_at_the_same_point() -> None:
    """The reference gives at least 50 GiB where the change gives at most 8."""
    before = before_fix_footprint_bytes(
        max_num_seqs=SERVED_MAX_NUM_SEQS, max_model_len=SERVED_MAX_MODEL_LEN
    )
    after = after_fix_bytes(
        max_num_seqs=SERVED_MAX_NUM_SEQS, max_model_len=SERVED_MAX_MODEL_LEN
    )

    assert before >= BEFORE_FLOOR_BYTES
    assert after["footprint_bytes"] * 10 < before


def test_the_recurrent_banks_follow_max_num_seqs_and_the_pool_follows_max_model_len(
    monkeypatch,
) -> None:
    """The real ``initialize_kv_cache`` on a small point: slots, slot bytes, pages."""
    from vllm.v1.kv_cache_interface import MambaSpec, MLAAttentionSpec

    from vllm_neuron.vllm.worker import neuron_model_runner as module

    max_num_seqs, max_model_len = 3, 512
    layers, _ = glm53f_layer_specs()
    runner = fake_runner(layers, max_num_seqs=max_num_seqs, max_model_len=max_model_len)
    worker = fake_worker(runner)
    config = served_kv_cache_config(runner, worker)
    specs = runner.get_kv_cache_spec()

    monkeypatch.setattr(module, "InputBatch", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(module, "has_kv_transfer_group", lambda: False)
    caches = module.NeuronModelRunner.initialize_kv_cache(runner, config)

    blocks_per_seq = -(-max_model_len // BLOCK_SIZE_TOKENS)
    for name, spec in specs.items():
        if isinstance(spec, MambaSpec):
            conv_bytes, recurrent_bytes = _kda_state_bytes(spec)
            conv, recurrent = caches[name]
            assert conv.shape[0] == recurrent.shape[0] == max_num_seqs
            assert tuple(conv.shape[1:]) == spec.shapes[0]
            assert tuple(recurrent.shape[1:]) == spec.shapes[1]
            # Both banks are contiguous regions of the layer's one buffer (the bank form
            # hands them whole to the graph), and the buffer prices one slot per sequence.
            assert conv.is_contiguous() and recurrent.is_contiguous()
            assert conv.untyped_storage().data_ptr() == recurrent.untyped_storage().data_ptr()
            slot_bytes = conv.untyped_storage().nbytes() // max_num_seqs
            assert conv_bytes + recurrent_bytes <= slot_bytes <= 1.5 * (conv_bytes + recurrent_bytes)
        else:
            assert isinstance(spec, MLAAttentionSpec)
            (latent,) = caches[name]
            assert latent.shape[0] == config.num_blocks
            assert latent.shape[0] >= blocks_per_seq * max_num_seqs + 1

    # Two recurrent layers never share bytes, and a slot is not shared by two layers.
    kda_names = [n for n, s in specs.items() if isinstance(s, MambaSpec)]
    starts = {caches[n][1].untyped_storage().data_ptr() for n in kda_names}
    assert len(starts) == len(kda_names)


def test_vllms_allocator_admits_64_full_length_requests_in_the_priced_pool(served) -> None:
    """vLLM's own KV cache manager holds 64 requests of 8191 tokens, and no 65th.

    The gate's serve line is used: prefix caching off, the full-input admission
    gate, the 1024-token chunk. A recurrent group costs one pool block per request,
    so the need the worker returns is exactly enough for the served concurrency.
    """
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
    from vllm.v1.request import Request

    vllm_config = served.runner.vllm_config
    assert vllm_config.cache_config.enable_prefix_caching is False
    scheduler_block, hash_block = resolve_kv_cache_block_sizes(served.config, vllm_config)
    manager = KVCacheManager(
        kv_cache_config=served.config,
        max_model_len=SERVED_MAX_MODEL_LEN,
        scheduler_block_size=scheduler_block,
        hash_block_size=hash_block,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        enable_caching=False,
    )
    prompt_len = SERVED_MAX_MODEL_LEN - 1

    def _request(index: int) -> Request:
        return Request(
            request_id=f"request-{index}",
            prompt_token_ids=[index] + [7] * (prompt_len - 1),
            sampling_params=SamplingParams(max_tokens=1),
            pooling_params=None,
        )

    admitted = [
        manager.allocate_slots(_request(i), prompt_len, full_sequence_must_fit=True)
        for i in range(SERVED_MAX_NUM_SEQS)
    ]
    one_more = manager.allocate_slots(
        _request(SERVED_MAX_NUM_SEQS), prompt_len, full_sequence_must_fit=True
    )
    held_per_request = {
        len(group_blocks) for blocks in admitted for group_blocks in blocks.get_block_ids()
    }

    assert all(blocks is not None for blocks in admitted)
    assert one_more is None
    # Each request holds one block in each recurrent group and 64 latent blocks.
    assert held_per_request == {1, SERVED_MAX_MODEL_LEN // BLOCK_SIZE_TOKENS}


def test_without_the_gate_knobs_the_recurrent_block_stays_the_attention_block() -> None:
    """The 5938748 serve line keeps 128-token recurrent blocks; only the banks shrink."""
    from vllm.v1.kv_cache_interface import MambaSpec

    layers, _ = glm53f_layer_specs()
    runner = fake_runner(
        layers,
        max_num_seqs=SERVED_MAX_NUM_SEQS,
        max_model_len=SERVED_MAX_MODEL_LEN,
        gate_knobs=False,
    )
    gated = fake_runner(
        layers, max_num_seqs=SERVED_MAX_NUM_SEQS, max_model_len=SERVED_MAX_MODEL_LEN
    )

    def recurrent_blocks(fake) -> set:
        return {
            s.block_size
            for s in fake.get_kv_cache_spec().values()
            if isinstance(s, MambaSpec)
        }

    assert recurrent_blocks(runner) == {BLOCK_SIZE_TOKENS}
    assert recurrent_blocks(gated) == {SERVED_MAX_MODEL_LEN}


def test_a_mamba_block_size_with_prefix_caching_on_is_refused_by_name() -> None:
    """vLLM's hybrid coordinator would assert at engine start; the spec refuses first."""
    layers, _ = glm53f_layer_specs()
    runner = fake_runner(
        layers, max_num_seqs=SERVED_MAX_NUM_SEQS, max_model_len=SERVED_MAX_MODEL_LEN
    )
    runner.vllm_config.cache_config.enable_prefix_caching = True

    with pytest.raises(ValueError) as refusal:
        runner.get_kv_cache_spec()

    assert "--no-enable-prefix-caching" in str(refusal.value)


def test_the_worker_does_not_refuse_64_sequences_at_8k_with_the_default_knobs(
    served, monkeypatch
) -> None:
    """On the measured 0a08ff4 residency the default knobs admit the served point.

    Even on a cold cache, where the 5 GiB reserve stands in for the graph need. A
    budget between the KV tensors and the footprint is refused since wt2/kvseg:
    ``test_kv_budget_side_caches.py::test_a_point_that_fits_only_without_side_caches_is_refused_naming_the_shortfall``.
    """
    monkeypatch.delenv("VLLM_NEURON_CPU_MODE", raising=False)
    monkeypatch.delenv("VLLM_NEURON_CPU_COMPILE", raising=False)
    monkeypatch.delenv("VLLM_NEURON_DEVICE_GRAPH_RESERVE_GIB", raising=False)
    monkeypatch.delenv("VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION", raising=False)
    worker = fake_worker(
        served.runner,
        param_bytes=TIP_PARAM_BYTES,
        resident_bytes=TIP_RESIDENT_BYTES,
    )

    available = worker.determine_available_memory()

    assert available == worker._kv_cache_need_bytes()
    assert (
        before_fix_footprint_bytes(
            max_num_seqs=SERVED_MAX_NUM_SEQS, max_model_len=SERVED_MAX_MODEL_LEN
        )
        > worker._compute_kv_budget(
            TOTAL_HBM_BYTES, TIP_PARAM_BYTES, GPU_MEMORY_UTILIZATION
        )
    )
