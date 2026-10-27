# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 under decode context parallelism with speculative decoding: owners over a window.

The platform admits DCP with speculation for Kimi K3 and refuses it for a grouped-query model.
Then the 4-layer config (KDA layers 0 to 2, MLA layer 3) runs on two gloo ranks at TP 2, once at
DCP 2 and once at DCP 1: a prompt of 4 prefill chunks, then 16 verify steps of 5 tokens for two
sequences. The prompt sequence's windows start in block 7, on rank 1, and cross into block 8, on
rank 0; the other sequence's context stays inside its first block, so rank 1 holds none of it.
Each step's window start moves on by 1 to 5 tokens, as accepted drafts move it. The page buffer
starts filled with NaN, as a block reused from an earlier owner can be, so a read of a row that
no step wrote shows as NaN. The torch stand-ins are test_dcp_mla's; at DCP 1 the paged MLA decode
is the window mode over a one-rank group. Set ``KIMI_K3_CHECKPOINT`` and ``KIMI_K3_DSPARK_DRAFT``
to the checkpoint and draft directories (defaults under ``~/models``); run with
``VLLM_NEURON_CPU_MODE=1``.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_dcp_mla import (  # noqa: F401  (pytestmark is used by pytest)
    BLOCK,
    CHECKPOINT,
    MAX_LEN,
    NULL_SLOT,
    ONE_RANK,
    TP,
    VOCAB,
    _add,
    _engine_config,
    _load,
    _metadata,
    _runner,
    _small_config_dir,
    _small_model,
    _stand_ins,
    _tap,
    _weights,
    m,
    pytestmark,
)
from vllm.engine.arg_utils import EngineArgs

from vllm_neuron.functional.attention.mla_paged import mla_paged_window_dcp

DRAFT = Path(os.environ.get("KIMI_K3_DSPARK_DRAFT",
                            Path.home() / "models" / "Inferact" / "Kimi-K3-DSpark-Block5"))
DCP, K, STEPS, BOUND = 2, 4, 16, 1e-5
Q = K + 1
CHUNKS = (512, 512, 512, 504)
PROMPT = sum(CHUNKS)
# Each sequence's window start per step: the prompt's from its end, the other's from 0.
ADVANCE = ((1, 2, 1, 2, 1, 1, 3, 5, 4, 2, 5, 1, 3, 2, 4),
           (2, 5, 1, 3, 4, 1, 2, 5, 3, 1, 4, 2, 5, 1, 3))
STARTS = np.stack([np.cumsum((s,) + a) for s, a in zip((PROMPT, 0), ADVANCE)], axis=1)
SERVED_LEN = {4: 32768, 16: 131072}


def _target_config(dcp: int):
    """The checkpoint's engine config with the DSpark drafter, TP 32 by DP 2, no workers."""
    return EngineArgs(
        model=str(CHECKPOINT), tensor_parallel_size=32, data_parallel_size=2,
        decode_context_parallel_size=dcp, enable_expert_parallel=True,
        max_model_len=SERVED_LEN[dcp], enable_prefix_caching=False, gpu_memory_utilization=0.90,
        speculative_config={"method": "dspark", "model": str(DRAFT), "num_speculative_tokens": K},
    ).create_engine_config()


def _grouped_query_config(root: Path):
    """A grouped-query model's engine config with n-gram speculation at DCP 4."""
    (root / "config.json").write_text(json.dumps({
        "architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": 4096,
        "intermediate_size": 11008, "num_hidden_layers": 2, "num_attention_heads": 64,
        "num_key_value_heads": 2, "vocab_size": 1024, "max_position_embeddings": 4096,
        "torch_dtype": "bfloat16"}))
    return EngineArgs(
        model=str(root), tensor_parallel_size=8, decode_context_parallel_size=4,
        max_model_len=4096, skip_tokenizer_init=True,
        speculative_config={"method": "ngram", "num_speculative_tokens": K,
                            "prompt_lookup_max": 4, "prompt_lookup_min": 2},
    ).create_engine_config()


def _verify_one_rank(*args):
    """The paged MLA decode at DCP 1: the window mode's torch form over a one-rank group."""
    *inputs, table, positions, slots, eps = args
    window = positions.view(table.shape[0], -1)
    owner = torch.ones(window.shape, dtype=torch.int32)
    return mla_paged_window_dcp(*inputs, table, window[:, 0], slots, owner, eps, ONE_RANK)


def _verify_metadata(runner, step: int) -> tuple[np.ndarray, dict]:
    """One verify step's positions and the runner's metadata: each sequence's window of Q."""
    starts = STARTS[step]
    runner.input_batch.num_computed_tokens_cpu[: len(starts)] = starts
    positions = (starts[:, None] + np.arange(Q)).reshape(-1)
    runner._compute_slot_mappings(positions, np.repeat(np.arange(len(starts)), Q))
    runner._save_block_table_to_device()
    runner._save_slot_mapping_to_device(len(positions))
    cached = int(starts.max())
    return positions, runner._build_attention_metadata(
        padded_num_reqs=len(starts), total_num_scheduled_tokens=len(positions), max_query_len=Q,
        max_num_draft_tokens=K, cached_seq_len=cached, max_decode_ctx_len=cached + Q)


def _serve(config, root: Path, rank: int, dcp: int) -> dict:
    """The prompt's chunks, then the verify steps, on this rank."""
    model, ids = _small_model(), torch.load(root / "ids.pt")
    _load(model, torch.load(root / "weights.pt", mmap=True), rank)
    runner = _runner(config, model, dcp)
    mla = next(layer for layer in model.modules() if isinstance(layer, m.KimiK3MLA))
    mla.pages.view(torch.bfloat16).fill_(float("nan"))
    per_rank = MAX_LEN // (BLOCK * dcp)
    tables = [list(range(3, 3 + per_rank)), [3 + per_rank]]
    _add(runner, "prompt", ids["prompt"].tolist(), tables[0], 1)
    _add(runner, "short", ids["verify"][0, 1, :1].tolist(), tables[1], 2)
    latent = next(g.layer_names[0] for i, g in enumerate(runner.kv_cache_config.kv_cache_groups)
                  if not runner._is_state_group(i))
    rank_t, start, out = torch.tensor([rank], dtype=torch.int32), 0, {"attention": []}
    slots, steps = [], []
    with _stand_ins() as patch:
        patch.setattr(m, "mla_paged_decode", _verify_one_rank)
        patch.setattr(m.KimiK3MLA, "forward_decode",
                      _tap(out["attention"], m.KimiK3MLA.forward_decode))
        for size in CHUNKS:
            positions = np.arange(start, start + size)
            meta = _metadata(runner, positions, np.zeros(size, dtype=np.int64), 1, size, start)
            model(ids["prompt"][start : start + size], torch.from_numpy(positions),
                  attn_metadata=meta, sampling_positions=torch.tensor([size - 1]), rank=rank_t)
            start += size
        for step in range(STEPS):
            positions, meta = _verify_metadata(runner, step)
            slots.append(meta[latent]["slot_mapping"].clone())
            steps.append(model(ids["verify"][step].reshape(-1), torch.from_numpy(positions),
                               attn_metadata=meta, rank=rank_t))
    return {"attention": torch.stack(out["attention"]), "logits": torch.stack(steps),
            "slots": torch.stack(slots), "tables": tables}


def _rank(rank: int, root: Path, model_dir: Path, dcp: int) -> None:
    """One gloo rank at TP 2 and ``dcp``: the prompt, then the verify steps."""
    from vllm_neuron.parallel.neuron_parallel_state import (
        initialize_neuron_parallel_state, stamp_spmd_rank_tensors)

    dist.init_process_group("gloo", init_method=f"file://{root}/rdzv-{dcp}", rank=rank,
                            world_size=TP)
    initialize_neuron_parallel_state(tp_global_ranks=list(range(TP)), local_rank=rank,
                                     dcp_size=dcp)
    stamped = stamp_spmd_rank_tensors("cpu")
    torch.save(_serve(_engine_config(model_dir, dcp), root, rank, dcp),
               root / f"dcp{dcp}-rank{rank}.pt")
    del stamped
    dist.destroy_process_group()


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> dict[int, list[dict]]:
    """Each world's outputs by rank, at DCP 2 and at DCP 1."""
    root = tmp_path_factory.mktemp("dcp-spec")
    torch.save(_weights(_small_model()), root / "weights.pt")
    gen = torch.Generator().manual_seed(11)
    torch.save({"prompt": torch.randint(0, VOCAB, (PROMPT,), generator=gen),
                "verify": torch.randint(0, VOCAB, (STEPS, 2, Q), generator=gen)}, root / "ids.pt")
    model_dir = _small_config_dir(root / "model")
    for dcp in (DCP, 1):
        mp.spawn(_rank, args=(root, model_dir, dcp), nprocs=TP, join=True)
    return {dcp: [torch.load(root / f"dcp{dcp}-rank{r}.pt") for r in range(TP)]
            for dcp in (DCP, 1)}


def _owners(step: int, seq: int) -> np.ndarray:
    """The owner rank of each row of a sequence's window at a step."""
    return (STARTS[step, seq] + np.arange(Q)) // BLOCK % DCP


def _misses(runs, seq: int, steps) -> list:
    """Each step of ``seq`` whose MLA output on a rank is over the bound, or whose picks differ."""
    rows, misses = slice(seq * Q, (seq + 1) * Q), []
    for s in steps:
        for rank, (got, ref) in enumerate(zip(runs[DCP], runs[1])):
            a, b = got["attention"][s, rows].double(), ref["attention"][s, rows].double()
            error = float((a - b).abs().max() / b.abs().max())
            if not error <= BOUND:
                misses.append((s, rank, error))
        picks = [torch.cat([out["logits"][s, rows] for out in run], -1).argmax(-1)
                 for run in (runs[DCP], runs[1])]
        if not torch.equal(*picks):
            misses.append((s, "picks", picks[0].tolist(), picks[1].tolist()))
    return misses


def test_config_admits_target_and_refuses_grouped_query(tmp_path):
    """The target with DSpark at DCP 4 and DCP 16 builds; a grouped-query model still raises."""
    raised = {}
    for dcp in SERVED_LEN:
        try:
            config = _target_config(dcp)
        except Exception as error:
            raised[dcp] = f"{type(error).__name__}: {error}"
            continue
        assert config.parallel_config.decode_context_parallel_size == dcp
        assert config.speculative_config.method == "dspark"
    refused = 0
    try:
        _grouped_query_config(tmp_path)
    except Exception as error:
        refused = int("DCP does not support speculative decoding" in str(error))
    assert raised == {} and refused == 1, (raised, refused)


def test_window_slots_on_their_owner_only(runs):
    """Each window row has a real slot on its owner rank and the null block on the other.

    Every row of every step is read, the windows that cross a block boundary among them.
    """
    both = neither = misplaced = crossing = 0
    for s in range(STEPS):
        for seq in range(2):
            owners = _owners(s, seq)
            crossing += len(set(owners)) > 1
            for i, owner in enumerate(owners):
                pos = STARTS[s, seq] + i
                slots = [int(out["slots"][s, seq * Q + i]) for out in runs[DCP]]
                real = [r for r, slot in enumerate(slots) if slot != NULL_SLOT]
                block = runs[DCP][owner]["tables"][seq][pos // (BLOCK * DCP)]
                both += len(real) == DCP
                neither += not real
                misplaced += len(real) == 1 and (real != [owner]
                                                 or slots[owner] != block * BLOCK + pos % BLOCK)
    assert crossing > 0 and (both, neither, misplaced) == (0, 0, 0), (both, neither, misplaced)


def test_verify_steps_match_dcp_1(runs):
    """The prompt sequence's 16 windows: the MLA output within the bound, the picks equal."""
    crossing = [s for s in range(STEPS) if len(set(_owners(s, 0))) > 1]
    misses = _misses(runs, 0, range(STEPS))
    assert crossing and not misses, (crossing, misses)


def test_other_rank_windows_and_short_context_without_nan(runs):
    """Windows whose rows all sit on one rank, and a context inside one block: no NaN.

    The prompt sequence's one-owner windows and every window of the short sequence meet the
    bound and the picks; no MLA output or logit of either sequence is NaN.
    """
    single = [s for s in range(STEPS) if len(set(_owners(s, 0))) == 1]
    misses = _misses(runs, 0, single) + _misses(runs, 1, range(STEPS))
    nan = sum(int(out[key].isnan().sum()) for run in runs.values() for out in run
              for key in ("attention", "logits"))
    assert single and nan == 0 and not misses, (single, nan, misses)
