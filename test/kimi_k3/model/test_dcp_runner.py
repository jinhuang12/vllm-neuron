# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 runner under decode context parallelism: latent groups shard, state groups do not.

The latent (MLA) cache groups split their blocks over the DCP group. The KDA state groups
keep a DCP size of 1, because every rank holds the whole state. The first three tests run
the runner on two gloo ranks at TP 2 and DCP 2, on a 4-layer config (KDA layers 0 to 2, MLA
layer 3) at the served block and page sizes, against the same runner at DCP 1. The last two
read the 93-layer config at the served forms. Set ``KIMI_K3_CHECKPOINT`` to the checkpoint
directory (default ``~/models/moonshotai/Kimi-K3``); run with ``VLLM_NEURON_CPU_MODE=1``.
"""

import copy
import json
import os
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.config import get_config
from vllm.v1.attention.backends.utils import resolve_kv_cache_layout
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_config_from_groups,
    get_kv_cache_configs,
    get_kv_cache_groups,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.worker.gpu_input_batch import CachedRequestState

from vllm_neuron.model.kimi_k3 import model as kimi_model
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner
from vllm_neuron.vllm.worker.neuron_worker import NeuronWorker

CHECKPOINT = Path(
    os.environ.get("KIMI_K3_CHECKPOINT", Path.home() / "models" / "moonshotai" / "Kimi-K3")
)

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "config.json").is_file(),
    reason=f"Kimi K3 checkpoint not found at {CHECKPOINT}",
)

TP, DCP, MAX_LEN = 2, 2, 16384
BLOCK, CHUNK, BUCKET = 256, 4 * 256, 8192
NULL_BLOCK = 0

SERVED_TP, SERVED_DP = 32, 2
# One LNC 2 trn3 core's HBM region, the model's own parameter bytes per rank of the TP 32 by
# EP 64 layout (the embedding and the LM head at their TP 32 shards, the load-time copies not
# counted), and the bytes of one expert replica slot.
HBM_PER_RANK = 36 * 1024**3
WEIGHT_BYTES = 26_789_456_700
SLOT_BYTES = 1_614_348_288
# Per served form: DCP size, max-model-len, replica slots, users per replica and context of the
# largest cell, then the worker's KV budget (GB) and the headroom (GB) at that cell, as the
# worker's budget function gives them at the weight bytes above.
FORMS = {
    4: (32768, 3, 16, 2048, 1.78, 1.21),
    16: (131072, 3, 16, 8192, 1.78, 1.21),
    32: (131072, 2, 16, 131072, 3.40, 1.24),
}


def _text_overrides() -> dict:
    """The 4-layer text config: three KDA layers and one MLA layer at 3 heads per rank."""
    return dict(num_hidden_layers=4, num_attention_heads=6, intermediate_size=2112,
                moe_intermediate_size=96, routed_expert_hidden_size=112, num_experts=16,
                vocab_size=10240, first_k_dense_replace=1)


def _small_model() -> kimi_model.KimiK3ForConditionalGeneration:
    """The 4-layer model on the meta device."""
    config = copy.deepcopy(get_config(str(CHECKPOINT), trust_remote_code=False).to_dict())
    text = config["text_config"]
    text.update(_text_overrides())
    text["linear_attn_config"].update(num_heads=6, kda_layers=[1, 2, 3], full_attn_layers=[4])
    with torch.device("meta"):
        return kimi_model.KimiK3ForConditionalGeneration.from_configs(config)


def _small_config_dir(root: Path) -> Path:
    """A model directory holding the 4-layer config, for the engine config build."""
    config = json.loads((CHECKPOINT / "config.json").read_text())
    text = config["text_config"]
    text.update({k: v for k, v in _text_overrides().items() if k in text})
    text["linear_attn_config"].update(num_heads=6, kda_layers=[1, 2, 3], full_attn_layers=[4])
    root.mkdir(exist_ok=True)
    (root / "config.json").write_text(json.dumps(config))
    return root


def _engine_config(model: Path, tp: int, dp: int, dcp: int, max_len: int, **kwargs):
    """An engine config after the platform's block-size step."""
    config = EngineArgs(
        model=str(model), tensor_parallel_size=tp, data_parallel_size=dp,
        decode_context_parallel_size=dcp, max_model_len=max_len, max_num_seqs=32,
        enable_prefix_caching=False, gpu_memory_utilization=0.90, **kwargs).create_engine_config()
    current_platform.update_block_size_for_backend(config)
    return config


def _runner(config, model, dcp: int) -> NeuronModelRunner:
    """A model runner holding what cache setup, slot mapping and attention metadata read."""
    runner = NeuronModelRunner.__new__(NeuronModelRunner)
    runner.vllm_config, runner.model, runner.speculative_config = config, model, None
    runner.drafter, runner.neuron_config, runner.is_dflash_spec = None, NeuronConfig(), False
    runner.max_num_reqs, runner.max_model_len = 2, config.model_config.max_model_len
    runner.max_num_batched_tokens = config.scheduler_config.max_num_batched_tokens
    runner.device, runner.vocab_size, runner.is_pooling_model = torch.device("cpu"), 10240, False
    runner._kv_cache_full_tensors, runner._block_table_groups, runner._swa_groups = {}, {}, {}
    runner._constant_tensors, runner._kv_snapshot_enabled = {}, False
    runner._is_synthetic_model = False
    runner._dcp_size = runner.cp_world_size = dcp
    runner.cp_kv_cache_interleave_size = config.parallel_config.cp_kv_cache_interleave_size
    runner._cp_rank = None
    return runner


def _kv_specs(config, runner) -> dict:
    """The runner's layer specs, with the KV layout resolved as the engine core does."""
    specs = runner.get_kv_cache_spec()
    layouts = NeuronWorker.__new__(NeuronWorker).get_supported_kv_cache_layouts()
    resolve_kv_cache_layout(config, [layouts], list(specs.values()))
    return specs


def _small_pool(config, groups, blocks: int):
    """The KV cache config of ``blocks`` blocks over ``groups``."""
    slots = max(len(g.layer_names) for g in groups)
    return get_kv_cache_config_from_groups(
        config, groups, blocks * slots * config.cache_config.mamba_page_size_padded)


def _group_run(model_dir: Path, dcp: int) -> dict:
    """One runner at ``dcp`` on this rank: slot mappings, block-table ids and widths."""
    config = _engine_config(model_dir, TP, 1, dcp, MAX_LEN, skip_tokenizer_init=True)
    runner = _runner(config, _small_model(), dcp)
    groups = get_kv_cache_groups(config, _kv_specs(config, runner))
    kv_config = _small_pool(config, groups, blocks=80)
    runner.initialize_kv_cache(kv_config)
    state = [i for i in range(len(groups)) if runner._is_state_group(i)]
    latent = [i for i in range(len(groups)) if i not in state]
    block = groups[latent[0]].kv_cache_spec.block_size
    assert (block, len(latent), len(state)) == (BLOCK, 1, 3)

    batch = runner.input_batch
    per_dcp = MAX_LEN // (block * dcp)
    batch.add_request(CachedRequestState(
        req_id="a", prompt_token_ids=[1, 2, 3], mm_features=[], sampling_params=SamplingParams(),
        generator=None, num_computed_tokens=0, output_token_ids=[],
        block_ids=tuple(list(range(3, 3 + per_dcp)) if i in latent else [1] for i in range(4))))
    layers = [g.layer_names[0] for g in groups]
    out = {"latent": latent, "state": state, "dcp_block": block * dcp, "layers": layers}

    for name, start in (("first", 0), ("second", BUCKET)):
        positions = np.arange(start, start + CHUNK)
        runner._compute_slot_mappings(positions, np.zeros(CHUNK, dtype=np.int64))
        runner._save_block_table_to_device()
        runner._save_slot_mapping_to_device(CHUNK)
        try:
            meta = runner._build_attention_metadata(
                padded_num_reqs=1, total_num_scheduled_tokens=CHUNK, max_query_len=CHUNK,
                max_num_draft_tokens=0, cached_seq_len=start)
        except ValueError as error:
            out[name] = str(error)
            continue
        out[name] = [(meta[n]["slot_mapping"].clone(), meta[n]["block_table_tensor"][:, 0].clone())
                     for n in layers]

    runner.neuron_config.decode_context_length_buckets = [BUCKET]
    meta = runner._build_attention_metadata(
        padded_num_reqs=1, total_num_scheduled_tokens=1, max_query_len=1, max_num_draft_tokens=0,
        cached_seq_len=BUCKET - 100, max_decode_ctx_len=BUCKET - 100)
    out["decode_width"] = [meta[n]["block_table_tensor"].shape[1] for n in layers]
    warm = runner._build_warmup_attention_metadata(num_tokens=1, num_reqs=1, ctx_bucket=BUCKET)
    out["warmup_width"] = [warm[n]["block_table_tensor"].shape[1] for n in layers]
    out["scheduler_block"] = resolve_kv_cache_block_sizes(kv_config, config)[0]
    return out


def _rank(rank: int, root: Path, model_dir: Path) -> None:
    """One gloo rank at TP 2 and DCP 2: the runner at DCP 2, then at DCP 1."""
    from vllm_neuron.parallel.neuron_parallel_state import initialize_neuron_parallel_state

    dist.init_process_group("gloo", init_method=f"file://{root}/rdzv", rank=rank, world_size=TP)
    initialize_neuron_parallel_state(tp_global_ranks=list(range(TP)), local_rank=rank,
                                     dcp_size=DCP)
    torch.save({dcp: _group_run(model_dir, dcp) for dcp in (DCP, 1)}, root / f"rank{rank}.pt")
    dist.destroy_process_group()


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> list[dict]:
    """Each rank's runner outputs at DCP 2 and at DCP 1."""
    root = tmp_path_factory.mktemp("dcp")
    mp.spawn(_rank, args=(root, _small_config_dir(root / "model")), nprocs=TP, join=True)
    return [torch.load(root / f"rank{r}.pt") for r in range(TP)]


def test_each_latent_row_has_a_slot_on_its_owner_only(runs):
    """Over 4 blocks of positions: a real slot on the owner rank, the null block on the other."""
    owner = np.arange(CHUNK) // BLOCK % DCP
    real = []
    for run in runs:
        (slots, _), = (run[DCP]["first"][g] for g in run[DCP]["latent"])
        real.append(slots.numpy() != NULL_BLOCK)
    on_both = int((real[0] & real[1]).sum())
    on_neither = int((~real[0] & ~real[1]).sum())
    assert (on_both, on_neither) == (0, 0)
    for rank in range(TP):
        assert real[rank].tolist() == (owner == rank).tolist(), f"rank {rank}"


def test_state_slot_ids_match_dcp_1_on_both_ranks(runs):
    """The KDA groups' slot mapping and state slot, across two chunks, equal the DCP 1 ids."""
    differ = checked = 0
    for run in runs:
        for chunk in ("first", "second"):
            assert isinstance(run[DCP][chunk], list), run[DCP][chunk]
            for g in run[DCP]["state"]:
                for at_dcp, at_one in zip(run[DCP][chunk][g], run[1][chunk][g]):
                    differ += int((at_dcp != at_one).sum())
                    checked += at_one.numel()
                    assert bool((at_one > 0).all())
    assert checked > 0 and differ == 0


def test_block_table_widths_at_the_8k_bucket(runs):
    """The latent width is the bucket over DCP blocks; the state width is its DCP 1 width."""
    for run in runs:
        for width in ("decode_width", "warmup_width"):
            for g in run[DCP]["latent"]:
                assert run[DCP][width][g] == BUCKET // (DCP * BLOCK)
            for g in run[DCP]["state"]:
                assert run[DCP][width][g] == run[1][width][g]
        assert run[DCP]["decode_width"] == run[DCP]["warmup_width"]


@pytest.fixture(scope="module")
def model():
    """The 93-layer model on the meta device, built on one rank of a 32-way TP group.

    The MoE layers hold that rank's experts of the 64-rank expert group, TP 32 by DP 2. The
    embedding and the LM head take their shard size from torch.distributed, which is not
    initialized here, so its size queries answer with the group size.
    """
    group = types.SimpleNamespace(world_size=SERVED_TP, rank_in_group=0, device_group=None)
    experts = types.SimpleNamespace(world_size=SERVED_TP * SERVED_DP, rank_in_group=0,
                                    device_group=None)
    with pytest.MonkeyPatch.context() as patch, torch.device("meta"):
        patch.setattr(kimi_model, "_tp_group", lambda: group)
        patch.setattr(kimi_model, "get_world_group", lambda: experts)
        patch.setattr(kimi_model, "get_wide_ep_group", lambda: experts)
        patch.setattr(dist, "is_initialized", lambda: True)
        patch.setattr(dist, "get_world_size", lambda group=None: SERVED_TP)
        patch.setattr(dist, "get_rank", lambda group=None: 0)
        built = kimi_model.KimiK3ForConditionalGeneration.from_configs(
            get_config(str(CHECKPOINT), trust_remote_code=False))
        built.prepare_after_load()
        return built


def _served(model, dcp: int, max_len: int):
    """The served engine config at ``dcp`` with the runner's layer specs."""
    config = _engine_config(CHECKPOINT, SERVED_TP, SERVED_DP, dcp, max_len,
                            enable_expert_parallel=True)
    return config, _kv_specs(config, _runner(config, model, dcp))


def test_scheduler_block_size_is_the_same_at_every_dcp_size(runs, model):
    """Upstream's scheduler block size for the two groups at DCP 2, 4, 16 and 32 is DCP 1's."""
    differ = [run[DCP]["scheduler_block"] != run[1]["scheduler_block"] for run in runs]
    for dcp, (max_len, *_) in FORMS.items():
        sizes = []
        for size in (dcp, 1):
            config, specs = _served(model, size, max_len)
            groups = get_kv_cache_groups(config, specs)
            kv_config = _small_pool(config, groups, blocks=4)
            sizes.append(resolve_kv_cache_block_sizes(kv_config, config)[0])
        differ.append(sizes[0] != sizes[1])
    assert len(differ) == 2 + len(FORMS) and not any(differ)


def _device_bytes(model, slots: int) -> int:
    """Bytes per rank: weights, the model's load-time copies (its buffers), and replica slots."""
    return WEIGHT_BYTES + sum(b.nbytes for b in model.buffers()) + slots * SLOT_BYTES


def _worker_budget(config, device_bytes: int) -> int:
    """The worker's KV budget for a rank holding ``device_bytes``, at LNC 2 with no headroom set."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("NEURON_LOGICAL_NC_CONFIG", "2")
        for name in ("VLLM_NEURON_RESERVED_HEADROOM_GIB", "VLLM_NEURON_MIN_KV_BUDGET_GIB",
                     "VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION"):
            patch.delenv(name, raising=False)
        worker = NeuronWorker.__new__(NeuronWorker)
        worker.cache_config = config.cache_config
        return worker._compute_kv_budget(HBM_PER_RANK, device_bytes)


def test_weight_bytes_are_the_models_own_parameter_bytes_at_tp_32(model):
    """One rank's parameters, the embedding and LM head at TP 32 shards, are WEIGHT_BYTES."""
    own = sum(p.nbytes for p in model.parameters())
    assert own == WEIGHT_BYTES, f"the model holds {own} bytes, WEIGHT_BYTES is {WEIGHT_BYTES}"


def test_each_served_form_holds_its_cells_inside_the_rank_memory(model):
    """The worker's KV pool holds each form's largest cell and one max-length sequence."""
    for dcp, (max_len, slots, users, context, budget_gb, headroom_gb) in FORMS.items():
        config, specs = _served(model, dcp, max_len)
        device = _device_bytes(model, slots)
        budget = _worker_budget(config, device)
        kv_config = get_kv_cache_configs(config, [specs], [budget])[0]
        pool = max(t.offset + t.size for t in kv_config.kv_cache_tensors)
        per_block = pool // kv_config.num_blocks

        def need(sequences: int, length: int) -> int:
            per_request = sum(g.kv_cache_spec.max_num_blocks_per_req(config, length)
                              for g in kv_config.kv_cache_groups)
            return sequences * per_request + 1  # block 0 is the null block

        cell, longest = need(users, context), need(1, max_len)
        usable = config.cache_config.gpu_memory_utilization * HBM_PER_RANK
        headroom = (usable - device - cell * per_block) / 1e9
        assert abs(budget / 1e9 - budget_gb) <= 0.05, f"DCP {dcp}: budget {budget / 1e9:.2f} GB"
        assert budget >= 1024**3 and max(cell, longest) <= kv_config.num_blocks, f"DCP {dcp}"
        assert headroom >= 0, f"DCP {dcp}: headroom {headroom:.2f} GB"
        assert longest < cell, f"DCP {dcp}: the largest cell does not bind"
        assert abs(headroom - headroom_gb) <= 0.10, f"DCP {dcp}: headroom {headroom:.2f} GB"
    config, _ = _served(model, 4, 32768)
    with pytest.raises(RuntimeError, match="below minimum threshold"):
        _worker_budget(config, _device_bytes(model, 4))
