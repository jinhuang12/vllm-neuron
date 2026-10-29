# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 prefill chunks under decode context parallelism that end inside a round of DCP blocks.

Kimi K3's DCP prefill runs every token of the chunk on every rank, so its prefill stride is the
block size, not DCP x block size. The first three tests run 16 gloo ranks at TP 16 and DCP 16
on a 4-layer config (KDA layers 0 to 2, MLA layer 3): the runner's metadata, slots and the MLA
layer's prefix gather for chunks of 2,048 tokens, half a round. The next run two gloo ranks at
TP 2 and DCP 2 against DCP 1, with the torch stand-ins of ``test_dcp_mla``. The last builds the
served forms' engine config, scheduler and runner in one process at each chunk, and a
grouped-query model that keeps the whole-round stride. Set ``KIMI_K3_CHECKPOINT`` to the
checkpoint directory (default ``~/models/moonshotai/Kimi-K3``); run with
``VLLM_NEURON_CPU_MODE=1``.
"""

import contextlib
import copy
import json
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import test_dcp_mla as dcp_mla
import test_dcp_runner as dcp_runner
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.entrypoints.launchers.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.platforms import current_platform
from vllm.transformers_utils.config import get_config
from vllm.usage.usage_lib import UsageContext
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_config_from_groups,
    get_kv_cache_groups,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.structured_output import StructuredOutputManager

from vllm_neuron.model.kimi_k3 import model as m
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.vllm.core.scheduler import NeuronScheduler
from vllm_neuron.vllm.worker import neuron_model_runner
from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

CHECKPOINT = dcp_mla.CHECKPOINT
pytestmark = dcp_mla.pytestmark
model = dcp_runner.model  # the 93-layer model on the meta device, one rank of TP 32

BLOCK, LATENT, NULL_SLOT = dcp_mla.BLOCK, dcp_mla.LATENT, dcp_mla.NULL_SLOT
WIDE, WIDE_MAX_LEN, WIDE_CHUNK, WIDE_PROMPT = 16, 16384, 2048, 8192
TURN = WIDE * BLOCK
# At DCP 2 a round is 512 tokens: the first set ends chunks at 256 and 1,792, inside a round.
CHUNK_SETS = {"inside-a-round": (256, 768, 768, 1280, 1023), 512: (512,) * 7 + (511,),
              1024: (1024,) * 3 + (1023,), 2048: (2048, 2047)}
PROMPT = sum(CHUNK_SETS[512])
SERVED_CHUNKS = (512, 1024, 2048)
CELLS = Path(__file__).resolve().parents[1] / "serving" / "cells.json"
STUB_TP_GROUP = types.SimpleNamespace(rank_in_group=0, world_size=32)


def _wide_overrides():
    """The 4-layer config at 48 heads, so each of 16 tensor-parallel ranks holds 3, as at TP 2."""
    heads = 3 * WIDE
    return ({**dcp_mla.OVERRIDES, "num_attention_heads": heads},
            {**dcp_mla.LINEAR, "num_heads": heads})


def _wide_config_dir(root: Path) -> Path:
    """A model directory holding the 16-head 4-layer config."""
    text, linear = _wide_overrides()
    config = json.loads((CHECKPOINT / "config.json").read_text())
    config["text_config"].update({k: v for k, v in text.items() if k in config["text_config"]})
    config["text_config"]["linear_attn_config"].update(linear)
    root.mkdir(exist_ok=True)
    (root / "config.json").write_text(json.dumps(config))
    return root


def _recording_prefill(reads: list):
    """The DCP prefill kernel's cache reads and writes: the prefix through the window table,
    kept in ``reads``, then each chunk row at its slot of the rank's cache, a slot outside it
    at the null block's first row; the row is the kernel's first input."""
    def run(qkv_a, gate, q_norm, kv_norm, w_qb_t, w_uk, w_uv_t, pages, offset, block_size,
            window, table, prefix, slots, eps):
        rows = window.view(torch.bfloat16).view(window.shape[0], -1, LATENT)
        ids = table.long().clamp(0, window.shape[0] - 1)
        reads.append(rows[ids, :block_size].reshape(-1, LATENT)[: int(prefix)])
        s = slots.long()
        s = torch.where((s < 0) | (s >= pages.shape[0] * block_size), 0, s)
        row = 2 * LATENT
        target = s // block_size * (pages.shape[1] // row) + offset // row + s % block_size
        pages.view(torch.bfloat16).view(-1, LATENT).index_copy_(0, target, qkv_a.bfloat16())
        return qkv_a.new_zeros(qkv_a.shape[0], 1), pages
    return run


def _wide_rank(rank: int, root: Path, model_dir: Path) -> None:
    """One of 16 gloo ranks at TP 16 and DCP 16: metadata, slots and prefix reads per chunk."""
    from vllm_neuron.parallel.neuron_parallel_state import (
        initialize_neuron_parallel_state, stamp_spmd_rank_tensors)

    dist.init_process_group("gloo", init_method=f"file://{root}/rdzv", rank=rank, world_size=WIDE)
    initialize_neuron_parallel_state(tp_global_ranks=list(range(WIDE)), local_rank=rank,
                                     dcp_size=WIDE)
    stamped = stamp_spmd_rank_tensors("cpu")
    config = EngineArgs(
        model=str(model_dir), tensor_parallel_size=WIDE, decode_context_parallel_size=WIDE,
        max_model_len=WIDE_MAX_LEN, max_num_batched_tokens=2 * WIDE_CHUNK, max_num_seqs=32,
        enable_prefix_caching=False, gpu_memory_utilization=0.90,
        skip_tokenizer_init=True).create_engine_config()
    current_platform.update_block_size_for_backend(config)
    text, linear = _wide_overrides()
    full = copy.deepcopy(get_config(str(CHECKPOINT), trust_remote_code=False).to_dict())
    full["text_config"].update(text)
    full["text_config"]["linear_attn_config"].update(linear)
    with torch.device("meta"):
        model = m.KimiK3ForConditionalGeneration.from_configs(full)
    runner = dcp_mla._runner(config, model, WIDE)
    name = "layers.3.self_attn"
    table = list(range(3, 3 + WIDE_MAX_LEN // TURN))
    dcp_mla._add(runner, "prompt", [1] * WIDE_PROMPT, table, 1)
    rows = torch.randn(WIDE_PROMPT, LATENT, generator=torch.Generator().manual_seed(11))
    groups = runner.kv_cache_config.kv_cache_groups
    block = [g.kv_cache_spec.block_size for g in groups if g.layer_names == [name]][0]
    out = {"table": table, "block": block, "errors": {}, "slots": {}, "reads": []}
    for size in (WIDE_CHUNK, TURN):
        out["slots"][size] = []
        for start in range(0, WIDE_PROMPT, size):
            positions = np.arange(start, start + size)
            try:
                meta = dcp_mla._metadata(runner, positions, np.zeros(size, dtype=np.int64), 1,
                                         size, start)
            except ValueError as error:
                out["errors"][(size, start)] = str(error)
                continue
            out["slots"][size].append((start, meta[name]["slot_mapping"].clone()))
            if size == WIDE_CHUNK:
                reads = []
                with pytest.MonkeyPatch.context() as patch:
                    patch.setattr(m, "mla_paged_prefill_dcp", _recording_prefill(reads))
                    model.model.layers[3].self_attn._attend(rows[start : start + size], None,
                                                            meta[name])
                want = rows[:start].bfloat16()
                out["reads"].append((start, len(reads[0]), int((reads[0] != want).any(-1).sum())))
    torch.save(out, root / f"rank{rank}.pt")
    del stamped
    dist.destroy_process_group()


@pytest.fixture(scope="module")
def wide(tmp_path_factory) -> list[dict]:
    """Each of the 16 ranks' metadata errors, latent slots and prefix reads."""
    root = tmp_path_factory.mktemp("dcp16")
    mp.spawn(_wide_rank, args=(root, _wide_config_dir(root / "model")), nprocs=WIDE, join=True)
    return [torch.load(root / f"rank{r}.pt") for r in range(WIDE)]


def test_dcp16_metadata_builds_at_each_half_round_prefix(wide):
    """The metadata builds at the prefixes 0, 2,048, 4,096 and 6,144 on every rank."""
    errors = {(r, key): e for r, out in enumerate(wide) for key, e in out["errors"].items()}
    assert errors == {}


def test_dcp16_each_row_has_one_real_slot_as_in_a_whole_round(wide):
    """Each token's latent row: a real slot on its owner rank only, the one a 4,096-token chunk
    gives it; the null slot on the other 15."""
    differ = 0
    position = np.arange(WIDE_PROMPT)
    assert [out["block"] for out in wide] == [BLOCK] * WIDE
    for rank, out in enumerate(wide):
        half, whole = (torch.cat([s for _, s in out["slots"][size]]).numpy()
                       for size in (WIDE_CHUNK, TURN))
        owner = position // BLOCK % WIDE == rank
        block = np.array(out["table"])[position // TURN]
        want = np.where(owner, block * BLOCK + position % BLOCK, NULL_SLOT)
        differ += int((half != want).sum()) + int((half != whole).sum())
    assert differ == 0


def test_dcp16_prefix_gather_reads_the_rows_written_before(wide):
    """Before each chunk, every rank's MLA layer reads the rows of every earlier token, in
    sequence order: 0 rows differ."""
    reads = [(start, n, bad) for out in wide for start, n, bad in out["reads"]]
    assert len(reads) == WIDE * WIDE_PROMPT // WIDE_CHUNK
    assert all(n == start and bad == 0 for start, n, bad in reads), reads


def _pair_prefill(config, root: Path, rank: int, dcp: int) -> dict:
    """Each chunk set of the prompt on a fresh cache: the MLA layer outputs and the logits."""
    model, ids = dcp_mla._small_model(), torch.load(root / "ids.pt")
    dcp_mla._load(model, torch.load(root / "weights.pt", mmap=True), rank)
    rank_t, out, seen = torch.tensor([rank], dtype=torch.int32), {}, []
    table = list(range(3, 3 + dcp_mla.MAX_LEN // (BLOCK * dcp)))
    with dcp_mla._stand_ins() as patch:
        patch.setattr(m.KimiK3MLA, "forward_prefill",
                      dcp_mla._tap(seen, m.KimiK3MLA.forward_prefill))
        for name, chunks in CHUNK_SETS.items():
            runner, start = dcp_mla._runner(config, model, dcp), 0
            dcp_mla._add(runner, "prompt", ids.tolist(), table, 1)
            seen.clear()
            try:
                for size in chunks:
                    positions = np.arange(start, start + size)
                    meta = dcp_mla._metadata(runner, positions, np.zeros(size, dtype=np.int64),
                                             1, size, start)
                    logits = model(ids[start : start + size], torch.from_numpy(positions),
                                   attn_metadata=meta, sampling_positions=torch.tensor([size - 1]),
                                   rank=rank_t)
                    start += size
            except ValueError as error:
                out[name] = {"error": str(error)}
                continue
            out[name] = {"chunks": list(seen), "next": logits}
    return out


def _pair_rank(rank: int, root: Path, model_dir: Path, dcp: int) -> None:
    """One gloo rank at TP 2 and ``dcp``: every chunk set."""
    from vllm_neuron.parallel.neuron_parallel_state import (
        initialize_neuron_parallel_state, stamp_spmd_rank_tensors)

    dist.init_process_group("gloo", init_method=f"file://{root}/rdzv-{dcp}", rank=rank,
                            world_size=dcp_mla.TP)
    initialize_neuron_parallel_state(tp_global_ranks=list(range(dcp_mla.TP)), local_rank=rank,
                                     dcp_size=dcp)
    stamped = stamp_spmd_rank_tensors("cpu")
    torch.save(_pair_prefill(dcp_mla._engine_config(model_dir, dcp), root, rank, dcp),
               root / f"dcp{dcp}-rank{rank}.pt")
    del stamped
    dist.destroy_process_group()


@pytest.fixture(scope="module")
def pair(tmp_path_factory) -> dict[int, list[dict]]:
    """Each world's outputs by rank, at DCP 2 and at DCP 1."""
    root = tmp_path_factory.mktemp("dcp2")
    torch.save(dcp_mla._weights(dcp_mla._small_model()), root / "weights.pt")
    gen = torch.Generator().manual_seed(5)
    torch.save(torch.randint(0, dcp_mla.VOCAB, (PROMPT,), generator=gen), root / "ids.pt")
    model_dir = dcp_mla._small_config_dir(root / "model")
    for dcp in (2, 1):
        mp.spawn(_pair_rank, args=(root, model_dir, dcp), nprocs=dcp_mla.TP, join=True)
    return {dcp: [torch.load(root / f"dcp{dcp}-rank{r}.pt") for r in range(dcp_mla.TP)]
            for dcp in (2, 1)}


@pytest.mark.parametrize("chunks", CHUNK_SETS)
def test_dcp2_prefill_chunks_match_dcp1(pair, chunks):
    """Each chunk's MLA layer output, then the next-token logits, within 1e-5 of DCP 1."""
    readings = []
    for got, ref in zip(pair[2], pair[1]):
        got, ref = got[chunks], ref[chunks]
        assert "error" not in got and "error" not in ref, (got.get("error"), ref.get("error"))
        assert len(got["chunks"]) == len(ref["chunks"]) == len(CHUNK_SETS[chunks])
        readings += [dcp_mla._error(a, b) for a, b in zip(got["chunks"], ref["chunks"])]
        readings.append(dcp_mla._error(got["next"], ref["next"]))
    assert all(r <= dcp_mla.BOUND for r in readings), readings


def _serve_config(model_dir: Path, flags: list, tmp_path: Path):
    """The engine config ``vllm serve`` builds from a form's flags, after the block-size step."""
    flags = [f.replace("{checkpoint}", str(model_dir)).replace("{measurements}", str(tmp_path))
             for f in flags]
    args = make_arg_parser(FlexibleArgumentParser()).parse_args([str(model_dir), *flags])
    args.model = args.model_tag
    validate_parsed_serve_args(args)
    config = AsyncEngineArgs.from_cli_args(args).create_engine_config(
        usage_context=UsageContext.OPENAI_API_SERVER)
    current_platform.update_block_size_for_backend(config)
    return config


def _at_chunk(flags: list, chunk: int, threshold: bool = True) -> list:
    """A form's flags with the prefill chunk and the long-prefill threshold at ``chunk``."""
    flags = list(flags)
    flags[flags.index("--max-num-batched-tokens") + 1] = str(chunk)
    at = flags.index("--long-prefill-token-threshold")
    if threshold:
        flags[at + 1] = str(chunk)
    else:
        del flags[at : at + 2]
    return flags


def _kv_config(config, runner):
    """The KV cache config of 64 blocks over the runner's layer groups."""
    groups = get_kv_cache_groups(config, dcp_runner._kv_specs(config, runner))
    page = max(len(g.layer_names) * g.kv_cache_spec.page_size_bytes for g in groups)
    return get_kv_cache_config_from_groups(config, groups, 64 * page)


def _scheduler(config, kv_config) -> NeuronScheduler:
    """The scheduler on ``kv_config`` at upstream's scheduler block size, as the engine core
    builds it."""
    config.cache_config.num_gpu_blocks = kv_config.num_blocks
    return NeuronScheduler(config, kv_config, StructuredOutputManager(config),
                           block_size=resolve_kv_cache_block_sizes(kv_config, config)[0])


@contextlib.contextmanager
def _stub_tp_group():
    """The runner init's tensor-parallel group, which one process with no workers lacks."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(neuron_model_runner, "get_tp_group", lambda: STUB_TP_GROUP)
        yield


class _GroupedQueryKV:
    """The KV needs of the two-layer grouped-query model: one KV head per rank, head size 128."""

    def get_kv_spec(self) -> KVSpec:
        return KVSpec(layers=[LayerSpec(f"layers.{i}.self_attn", 1, 128, torch.bfloat16)
                              for i in range(2)])

    def bind_kv_cache(self, kv_caches) -> None:
        pass


def _grouped_query_errors(model_dir: Path, flags: list, chunk: int, tmp_path: Path) -> dict:
    """The four DCP stride checks a grouped-query model meets at ``chunk``: what each raises."""
    errors = {}
    try:
        _serve_config(model_dir, _at_chunk(flags, chunk), tmp_path)
    except Exception as error:
        errors["platform"] = str(error)
    config = _serve_config(model_dir, _at_chunk(flags, chunk, threshold=False), tmp_path)
    runner = dcp_runner._runner(config, _GroupedQueryKV(), 16)
    kv_config = _kv_config(config, runner)
    for site, build in (("scheduler", lambda: _scheduler(config, kv_config)),
                        ("runner", lambda: NeuronModelRunner(config, torch.device("cpu")))):
        try:
            with _stub_tp_group():
                build()
        except ValueError as error:
            errors[site] = str(error)
    runner._cp_rank = 0
    runner.initialize_kv_cache(kv_config)
    dcp_mla._add(runner, "a", [1] * 2 * chunk, list(range(1, 33)), 1)
    try:
        dcp_mla._metadata(runner, np.arange(chunk, 2 * chunk), np.zeros(chunk, dtype=np.int64),
                          1, chunk, chunk)
    except ValueError as error:
        errors["metadata"] = str(error)
    return errors


@pytest.mark.parametrize("chunk", SERVED_CHUNKS)
def test_served_forms_select_their_chunk(model, chunk, tmp_path):
    """At ``chunk``: dcp16 and dcp32, and dcp4 at 512, pass the DCP check; the scheduler and
    the runner select [chunk]: 0 errors. A grouped-query model at DCP 16 on dcp16's flags
    raises at the platform check, the scheduler, the runner and the metadata: 4 errors."""
    forms = json.loads(CELLS.read_text())["forms"]
    names = ["dcp16", "dcp32"] + (["dcp4"] if chunk == 512 else [])
    errors, selected = {}, {}
    for name in names:
        try:
            config = _serve_config(CHECKPOINT, _at_chunk(forms[name]["server_flags"], chunk),
                                   tmp_path)
            current_platform._validate_dcp_config(config)
            dcp = config.parallel_config.decode_context_parallel_size
            kv_config = _kv_config(config, dcp_runner._runner(config, model, dcp))
            state = sum(isinstance(g.kv_cache_spec, MambaSpec) for g in kv_config.kv_cache_groups)
            with _stub_tp_group():
                runner = NeuronModelRunner(config, torch.device("cpu"))
            assert not dist.is_initialized()
            selected[name] = (len(kv_config.kv_cache_groups) - state, state,
                              _scheduler(config, kv_config).num_batched_tokens_buckets,
                              runner.neuron_config.num_batched_tokens_buckets)
        except Exception as error:
            errors[name] = f"{type(error).__name__}: {error}"
    assert errors == {} and selected == {name: (1, 3, [chunk], [chunk]) for name in names}

    model_dir = tmp_path / "grouped-query"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({
        "architectures": ["MixtralForCausalLM"], "model_type": "mixtral", "hidden_size": 4096,
        "intermediate_size": 14336, "num_hidden_layers": 2, "num_attention_heads": 32,
        "num_key_value_heads": 2, "num_local_experts": 8, "num_experts_per_tok": 2,
        "vocab_size": 1024, "max_position_embeddings": 131072, "torch_dtype": "bfloat16"}))
    # The checkpoint's tokenizer, for the reasoning parser; Kimi K3's block size, which the
    # platform sets for it and not for this model.
    flags = forms["dcp16"]["server_flags"] + ["--tokenizer", str(CHECKPOINT),
                                              "--block-size", str(BLOCK)]
    raised = _grouped_query_errors(model_dir, flags, chunk, tmp_path)
    assert sorted(raised) == ["metadata", "platform", "runner", "scheduler"], raised
    assert all(str(16 * BLOCK) in text for text in raised.values()), raised
