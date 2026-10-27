# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 slots under async speculative decoding with decode context parallelism (DCP).

The 4-layer config (KDA layers 0 to 2, MLA layer 3) builds on two gloo ranks at TP 2, once at DCP 2
and once at DCP 1, with the runner's KV cache and metadata. Each verify step has three sequences of
5 rows and one padding row. The runner builds the step's metadata at the optimistic positions, as
the scheduler counts every draft accepted; the previous step's sampled rows then name the rejected
drafts. The target forward runs its async prologue, and the slots its body receives are read. The
DSpark drafter's forward runs after each step, and the slots of its context write and of its block
rows are read. The host rule is the runner's slot mapping at the corrected positions. A slot is
real when it is outside the null block 0. Set ``KIMI_K3_CHECKPOINT`` and ``KIMI_K3_DSPARK_DRAFT``
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
from test_dcp_mla import (
    BLOCK,
    MAX_LEN,
    TP,
    _add,
    _engine_config,
    _runner,
    _small_config_dir,
    _small_model,
    m,
)
from test_dcp_mla import pytestmark as checkpoint_mark

from vllm_neuron.model.kimi_k3 import dspark
from vllm_neuron.model.qwen3.dflash_model import _paged_slots
from vllm_neuron.vllm.worker.neuron_model_runner import _compute_slot_mapping_cpu

DRAFT = Path(os.environ.get("KIMI_K3_DSPARK_DRAFT",
                            Path.home() / "models" / "Inferact" / "Kimi-K3-DSpark-Block5"))
pytestmark = [checkpoint_mark, pytest.mark.skipif(
    not (DRAFT / "config.json").is_file(), reason=f"DSpark draft not found at {DRAFT}")]

DCP, K, SEQS, ROWS = 2, 4, 3, 4
Q = K + 1
LATENT = "layers.3.self_attn"
# Per step: each sequence's corrected window start, the drafts the previous step rejected, and the
# tokens this step accepts with its bonus token, which set the drafter's block rows.
STEPS = {
    "rejected": ((1790, 2045, 100), (3, 2, 4), (1, 2, 5)),
    "none": ((1795, 2050, 254), (0, 0, 0), (5, 3, 1)),
    "first_row": ((1791, 2047, 2559), (1, 3, 2), (3, 4, 2)),
    "last_row": ((1788, 2044, 3068), (4, 4, 1), (5, 1, 2)),
}


class _Reached(Exception):
    """Raised by a recording stand-in once the forward has handed it the step's slots."""


def _sampled(counts, k: int) -> torch.Tensor:
    """Rejection-sampler rows ``[B, k + 1]``: ``counts`` valid tokens, then -1."""
    rows = torch.full((len(counts), k + 1), -1, dtype=torch.int32)
    for i, n in enumerate(counts):
        rows[i, :n] = 7
    return rows


def _target_slots(model, rank: int, positions: np.ndarray, meta: dict, **prev) -> dict:
    """The positions and each layer's slots that the target forward's body receives."""
    seen = {}

    def record(self, input_ids, attn_metadata, is_prefill, positions=None, rank=None, **kwargs):
        seen["positions"] = positions.clone()
        seen["slots"] = {name: meta["slot_mapping"].clone() for name, meta in attn_metadata.items()}
        raise _Reached

    with pytest.MonkeyPatch.context() as patch, pytest.raises(_Reached):
        patch.setattr(m.KimiK3Model, "forward", record)
        model(input_ids=torch.zeros(len(positions), dtype=torch.int32),
              positions=torch.from_numpy(positions), attn_metadata=meta,
              rank=torch.tensor([rank], dtype=torch.int32), **prev)
    return seen


def _draft_slots(draft, table, starts, rejected, valid) -> dict:
    """The slots of the drafter's context write and block rows after a verify step."""
    seen, batch = {}, len(starts)

    def block_pass(embedded, positions, attn_metadata):
        seen["block"] = attn_metadata[draft.layers[0].cache_name]["slot_mapping"].clone()
        raise _Reached

    draft.precompute_and_store_context_kv = lambda s, p, slots: seen.update(context=slots.clone())
    draft.block_pass = block_pass
    optimistic = torch.from_numpy((starts + rejected)[:, None] + np.arange(Q)).reshape(-1)
    meta = {"block_table_tensor": table, "block_size": BLOCK, "max_blocks_per_seq": table.shape[1]}
    with pytest.raises(_Reached):
        draft(target_positions=optimistic, target_hidden_states=torch.zeros(len(optimistic), 1),
              last_token_indices=torch.arange(batch) * Q + K,
              attn_metadata={layer.cache_name: meta for layer in draft.layers},
              raw_sampled_token_ids=_sampled(valid, K),
              prev_sampled_token_ids=_sampled(Q - rejected, K),
              prev_num_draft_tokens=torch.full((batch,), K, dtype=torch.int32),
              req_indices_per_token=torch.arange(batch).repeat_interleave(Q),
              cur_num_draft_tokens=torch.full((batch,), K, dtype=torch.int32))
    return seen


def _host(table: np.ndarray, positions: np.ndarray, requests: np.ndarray, block: int, dcp: int,
          rank: int) -> np.ndarray:
    """The runner's host rule at ``positions``, with an interleave of one block."""
    out = np.zeros(len(positions), dtype=np.int64)
    _compute_slot_mapping_cpu(table, out, positions, requests, block, dcp, rank, block)
    return out


def _today(positions, table, requests, block: int) -> torch.Tensor:
    """The correction's slot rule on one rank: table column position // block, -1 for no block."""
    positions = positions.to(torch.int64)
    ids = table.view(-1)[requests * table.shape[1] + positions // block]
    slot = ids.to(torch.int64) * block + positions % block
    return torch.where(ids < 0, torch.full_like(slot, -1), slot)


def _step(runner, model, draft, rank: int, dcp: int, starts, rejected, valid) -> dict:
    """One verify step through the target's prologue, then the drafter's slots."""
    starts, rejected, valid = (np.array(v + (0,)) for v in (starts, rejected, valid))
    optimistic = ((starts + rejected)[:, None] + np.arange(Q)).reshape(-1)
    corrected = (starts[:, None] + np.arange(Q)).reshape(-1)
    requests = np.repeat(np.arange(ROWS), Q)
    meta = _metadata(runner, optimistic[: SEQS * Q : Q], optimistic, requests)
    seen = _target_slots(model, rank, optimistic, meta,
                         prev_sampled_token_ids=_sampled(Q - rejected, K),
                         prev_num_draft_tokens=torch.full((ROWS,), K, dtype=torch.int32),
                         req_indices_per_token=torch.from_numpy(requests))
    tables = {name: layer.get("full_block_table_tensor", layer["block_table_tensor"])
              for name, layer in meta.items()}
    today = {name: _today(torch.from_numpy(corrected), table, torch.from_numpy(requests),
                          meta[name]["block_size"]) for name, table in tables.items()}
    # The real rows and their sequences, then the drafter's block rows and theirs.
    blocks = ((starts + valid)[:SEQS, None] + np.arange(K)).reshape(-1)
    rows = [(corrected[: SEQS * Q], np.repeat(np.arange(SEQS), Q)),
            (blocks, np.repeat(np.arange(SEQS), K))]
    group = runner.input_batch.block_table[_latent_group(runner)]
    raw, table = runner._raw_block_table(group.block_table), tables[LATENT][:SEQS]
    host = [_host(raw, pos, seq, BLOCK, dcp, rank) for pos, seq in rows]
    draft_today = [_paged_slots(torch.from_numpy(pos), torch.from_numpy(seq), table, BLOCK,
                                table.shape[1]) for pos, seq in rows]
    drafted = _draft_slots(draft, table, starts[:SEQS], rejected[:SEQS], valid[:SEQS])
    return {"corrected": corrected, "seen": seen, "today": today, "rows": [p for p, _ in rows],
            "host": host, "draft": drafted, "draft_today": draft_today}


def _latent_group(runner) -> int:
    """The index of the KV cache group that holds the MLA layer."""
    groups = runner.kv_cache_config.kv_cache_groups
    return next(i for i, group in enumerate(groups) if LATENT in group.layer_names)


def _metadata(runner, computed, positions, requests) -> dict:
    """The runner's metadata for one verify step, its slots at ``positions``."""
    runner.input_batch.num_computed_tokens_cpu[:SEQS] = computed
    runner._compute_slot_mappings(positions, requests)
    runner._save_block_table_to_device()
    runner._save_slot_mapping_to_device(len(positions))
    cached = int(computed.max())
    return runner._build_attention_metadata(
        padded_num_reqs=ROWS, total_num_scheduled_tokens=len(positions), max_query_len=Q,
        max_num_draft_tokens=K, cached_seq_len=cached, max_decode_ctx_len=cached + Q)


def _rank(rank: int, root: Path, model_dir: Path, dcp: int) -> None:
    """One gloo rank at TP 2 and ``dcp``: every step, then one step with async scheduling off."""
    from vllm_neuron.parallel.neuron_parallel_state import (
        initialize_neuron_parallel_state, stamp_spmd_rank_tensors)

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{root}/rdzv-{dcp}", rank=rank,
                            world_size=TP)
    initialize_neuron_parallel_state(tp_global_ranks=list(range(TP)), local_rank=rank,
                                     dcp_size=dcp)
    stamped = stamp_spmd_rank_tensors("cpu")
    model = _small_model()
    runner = _runner(_engine_config(model_dir, dcp), model, dcp)
    per_seq = MAX_LEN // (BLOCK * dcp)
    for i in range(SEQS):
        first = 1 + SEQS + i * per_seq
        _add(runner, f"seq{i}", [0], list(range(first, first + per_seq)), 1 + i)
    with torch.device("meta"):
        draft = dspark.K3DSparkForCausalLM.from_configs(
            json.loads((DRAFT / "config.json").read_text()), start_layer_idx=4)
    draft.num_speculative_tokens, draft.mask_token_id = K, 1
    draft.embed_tokens = lambda ids, scatter_tokens, rank: torch.zeros(len(ids), 1)
    out = {name: _step(runner, model, draft, rank, dcp, *step) for name, step in STEPS.items()}
    positions = (np.array(STEPS["none"][0] + (0,))[:, None] + np.arange(Q)).reshape(-1)
    meta = _metadata(runner, positions[:SEQS * Q : Q], positions, np.repeat(np.arange(ROWS), Q))
    host = {name: layer["slot_mapping"].clone() for name, layer in meta.items()}
    out["async_off"] = (host, _target_slots(model, rank, positions, meta)["slots"])
    torch.save(out, root / f"dcp{dcp}-rank{rank}.pt")
    del stamped
    dist.destroy_process_group()


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> dict[int, list[dict]]:
    """Each world's records by rank, at DCP 2 and at DCP 1."""
    root = tmp_path_factory.mktemp("async-dcp")
    model_dir = _small_config_dir(root / "model")
    for dcp in (DCP, 1):
        mp.spawn(_rank, args=(root, model_dir, dcp), nprocs=TP, join=True)
    return {dcp: [torch.load(root / f"dcp{dcp}-rank{r}.pt", weights_only=False) for r in range(TP)]
            for dcp in (DCP, 1)}


def _wrong(slots, host, positions) -> tuple[int, int]:
    """Rows read, and rows not real on exactly the owner or whose owner slot is not the host's."""
    wrong = 0
    for i, pos in enumerate(positions):
        owner = int(pos) // BLOCK % DCP
        real = [r for r in range(DCP) if int(slots[r][i]) >= BLOCK]
        wrong += real != [owner] or int(slots[owner][i]) != int(host[owner][i])
    return len(positions), wrong


def _target_wrong(runs, names) -> dict:
    """Rows read and rows wrong of the MLA layer's verify slots at each named step, at DCP 2."""
    out = {}
    for name in names:
        steps = [run[name] for run in runs[DCP]]
        assert torch.equal(steps[0]["seen"]["positions"], torch.from_numpy(steps[0]["corrected"]))
        out[name] = _wrong([s["seen"]["slots"][LATENT][: SEQS * Q] for s in steps],
                           [s["host"][0] for s in steps], steps[0]["rows"][0])
    return out


def _crossing(name: str) -> list:
    """Each sequence's window rows whose block differs from the row before."""
    starts = STEPS[name][0]
    return [[j for j in range(1, Q) if (s + j) // BLOCK != (s + j - 1) // BLOCK] for s in starts]


def test_verify_rows_keep_their_owner_slot_through_the_prologue(runs):
    """After a step with rejected drafts and one with none: each row real on its owner only."""
    counts = _target_wrong(runs, ("rejected", "none"))
    assert any(_crossing("rejected")) and any(_crossing("none"))
    assert all(wrong == 0 for _, wrong in counts.values()), counts


def test_dcp_1_keeps_todays_slots_and_the_host_path(runs):
    """DCP 1 async on: today's slots; async off: the host path's; DCP 1 layers at DCP 2: today's."""
    differ = {}
    for dcp, layers in ((1, None), (DCP, ("layers.0.self_attn", "layers.1.self_attn",
                                          "layers.2.self_attn"))):
        for rank, run in enumerate(runs[dcp]):
            for name in STEPS:
                for layer in layers or run[name]["today"]:
                    got, ref = run[name]["seen"]["slots"][layer], run[name]["today"][layer]
                    differ[dcp, rank, name, layer] = int((got != ref).sum())
            if dcp == 1:
                host, seen = run["async_off"]
                for layer in host:
                    differ[dcp, rank, "async_off", layer] = int((host[layer] != seen[layer]).sum())
    assert len(differ) and not any(differ.values()), {k: v for k, v in differ.items() if v}


def test_window_across_a_block_boundary_with_rejections_at_both_ends(runs):
    """A block boundary after the window's first row and before its last, drafts rejected."""
    assert [1 in rows for rows in _crossing("first_row")] == [True] * SEQS
    assert [K in rows for rows in _crossing("last_row")] == [True] * SEQS
    assert all(min(STEPS[name][1]) > 0 for name in ("first_row", "last_row"))
    counts = _target_wrong(runs, ("first_row", "last_row"))
    assert all(wrong == 0 for _, wrong in counts.values()), counts


def test_draft_rows_keep_their_owner_slot(runs):
    """The drafter's context and block rows at DCP 2 real on their owner only; DCP 1 is today's."""
    counts, differ = {}, {}
    for name in STEPS:
        steps = [run[name] for run in runs[DCP]]
        for kind, key in enumerate(("context", "block")):
            counts[name, key] = _wrong([s["draft"][key] for s in steps],
                                       [s["host"][kind] for s in steps], steps[0]["rows"][kind])
            for rank, run in enumerate(runs[1]):
                got, ref = run[name]["draft"][key], run[name]["draft_today"][kind]
                differ[name, rank, key] = int((got != ref).sum())
    assert all(wrong == 0 for _, wrong in counts.values()), counts
    assert not any(differ.values()), differ
