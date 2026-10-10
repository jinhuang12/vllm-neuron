# SPDX-License-Identifier: Apache-2.0
"""The draft layer's bank-form pooled-store writes, as the device runs a verify step.

A decode step of two or more requests hands every sparse layer its whole indexer banks
beside one slot per request (the bank form). The MTP draft layer runs the indexer more
than once in one speculative verify graph: ``populate`` at the ``T = 1 + k`` verify rows,
then each draft iteration that runs the indexer chain. Each run writes its pooled rows
into the layer's ``pool_cache`` bank and the next run's score kernel reads that bank.

The backend's ``InPlaceToOutOfPlacePass`` moves only the later uses of a write's own
first argument onto the write's result, and the aliasing pass stores only the last
write of each graph input. A write through a view made for that write
(``pool_bank.view(-1, dim).index_copy_``) is therefore seen by nothing after it: only
the last run's rows reach the bank, and every later run reads the bank as it was before
the graph. The trunk's layers write once per graph, so they kept their rows.

Every test here runs the step the way the device does (as
``test_glm5next_slot_residue``): the root's forward is traced with Dynamo as one
graph, the backend's default pass list rewrites it, the rewritten graph runs on CPU and
each aliased output (``io_map``) is copied onto its input.

1. Every pooled row a two- or four-request verify step writes persists, on every
   sparse layer, and every other bank matches the eager step.
2. Every read of a pooled store by a score kernel sees the rows written before it in
   the same graph: the bank the kernel is handed and the scores it returns equal the
   eager step's, call by call, and the draft iterations do read rows the step wrote.
3. The runner hands the store flat, the form whose writes the backend chains: the
   eager step on it selects and writes as on the ``[slots, rows, dim]`` bank, bit for
   bit, and a flat store that does not split over the rings is refused by name.

    NKI_SIMULATOR=1 VLLM_NEURON_CPU_MODE=1 python -m pytest \\
        test/vllm_neuron/worker/test_glm5next_pool_bank_writes.py
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import vllm_neuron.model.glm5_next.model_fp8 as model_fp8
from vllm_neuron.functional.dsa import decode_trow
from vllm_neuron.functional.kda.fused_decode import FUSED_DECODE_ENV
from vllm_neuron.vllm.worker import glm5next_state_banks
from vllm_neuron.vllm.worker.neuron_model_runner import NeuronModelRunner

from test.vllm_neuron.functional.dsa import dsa_batch_case as case
from test.vllm_neuron.functional.dsa.dsa_decode_case import build_attention, decode_config
from test.vllm_neuron.model.glm5_next import test_mtp_e2e_spec as verify
from test.vllm_neuron.model.glm5_next import test_shadow_draft_e2e as shadow
from test.vllm_neuron.model.glm5_next.tiny import test_tiny_glm5next_batch_decode as batch
from test.vllm_neuron.worker import test_glm5next_slot_residue as residue

pytestmark = [pytest.mark.forked]

#: The served draft count.
K = 3
#: Each request's last sampled id and its ``K`` drafts, request by request.
FIRST_IDS = (3, 9, 12, 4)
DRAFTS = ((11, 23, 37), (5, 6, 7), (41, 2, 19), (8, 8, 8))
#: Prompt lengths of the requests in one verify step. The two-request pairs put a pool
#: boundary at a different verify row of each request (the tiny indexer pools every 4
#: positions); the four-request case is one more bank-form bucket.
CASES = {"bs2-4-6": (4, 6), "bs2-5-7": (5, 7), "bs4-4-5-6-7": (4, 5, 6, 7)}


# ── the step ────────────────────────────────────────────────────────────────


def _bank_form_world(monkeypatch, lengths: tuple[int, ...], *, share: bool = True):
    """An mtp server's tiny root at ``k = K`` with ``len(lengths)`` prefilled requests and
    the draft head's weights; ``share`` is ``index_share_for_mtp_iteration``."""
    monkeypatch.delenv(glm5next_state_banks.STATE_BANKS_ENV, raising=False)
    monkeypatch.delenv(FUSED_DECODE_ENV, raising=False)
    # The speculative server's deeper indexer ring, from the first side-cache allocation.
    monkeypatch.setattr(NeuronModelRunner, "_glm5next_speculative_tokens", lambda self: K)
    world = verify._verify_world(monkeypatch, list(lengths), k=K)
    shadow._materialise_head(world.root, shadow.SEED_HEAD)
    monkeypatch.setattr(world.root.mtp.text_config, "index_share_for_mtp_iteration", share)
    return world


def _verify_kwargs(world, lengths: tuple[int, ...]) -> dict:
    """The root's kwargs for one verify step of every request. Converted once: the
    conversion moves the runner's ring cursors, so the eager and the served step share
    these kwargs."""
    requests = len(lengths)
    width = 1 + K
    drafts = [list(DRAFTS[r]) for r in range(requests)]
    inputs = [token for r in range(requests) for token in (FIRST_IDS[r], *drafts[r])]
    runner = world.runner
    runner.input_batch.req_ids = list(world.req_ids)
    runner._glm5next_request_tokens = np.array([width] * requests, np.int32)
    metadata = batch._metadata(
        world, list(range(requests)), cached=list(lengths), tokens=len(inputs)
    )
    for entry in metadata.values():
        entry["max_query_len"] = width
        entry["decode_token_threshold"] = width
    kwargs = runner._glm5next_model_kwargs({
        "input_ids": torch.tensor(inputs, dtype=torch.int32),
        "positions": None,
        "attn_metadata": metadata,
        "sampling_positions": torch.arange(len(inputs), dtype=torch.long),
        "sampling_params": torch.tensor([shadow.GREEDY_ROW] * len(inputs), dtype=torch.float32),
        "spec_decode_metadata": verify._metadata_for(drafts),
        "rank": None,
        "logit_mask": None,
    })
    draft = kwargs["layer_carriers"][len(world.root.model.layers)]
    assert torch.is_tensor(draft.get("state_slots")), (
        f"a {requests}-request decode step must hand the draft layer the bank form "
        f"(state_slots beside the whole banks); its carrier holds {sorted(draft)}"
    )
    return kwargs


def _recorded_reads(monkeypatch) -> list[dict]:
    """Record every ``T``-row score kernel call: the pooled-store bank it is handed, the
    slots and positions it reads at, and the scores it returns.

    A traced call records the served graph's values (Dynamo replays the list appends
    with the run's tensors); the clone is the bank's value at the call, which is what
    the kernel reads."""
    calls: list[dict] = []
    kernel = decode_trow.dsa_decode_scores_rows

    def recording(query, weights, pool_bank, slots, position, pooled, **kw):
        scores = kernel(query, weights, pool_bank, slots, position, pooled, **kw)
        calls.append({
            "bank": pool_bank.clone(), "slots": slots.clone(), "position": position.clone(),
            "rows": torch.full((1,), int(query.shape[0]) // int(slots.shape[0])),
            "scores": scores.clone(),
        })
        return scores

    monkeypatch.setattr(decode_trow, "dsa_decode_scores_rows", recording)
    return calls


def _eager_and_served(world, kwargs: dict) -> tuple[dict, dict, dict]:
    """``(before, eager, served)``: every floating-point bank the step can write, before
    the step, after the eager step, and after the served step (from the same before)."""
    state = residue._state(world, kwargs)
    before = {name: bank.clone() for name, bank in state.items()}
    world.root.forward(**dict(kwargs))
    eager = {name: bank.clone() for name, bank in state.items()}
    for name, bank in state.items():
        bank.copy_(before[name])
    residue._served_forward(world, kwargs)
    served = {name: bank.clone() for name, bank in state.items()}
    return before, eager, served


def _changed_rows(after: torch.Tensor, before: torch.Tensor) -> list[int]:
    return (residue._rows(after) != residue._rows(before)).any(dim=1).nonzero().flatten().tolist()


# ── 1. every pooled row persists ────────────────────────────────────────────


@pytest.mark.parametrize("lengths", CASES.values(), ids=CASES.keys())
def test_every_pooled_row_a_bank_form_verify_step_writes_persists(monkeypatch, lengths):
    world = _bank_form_world(monkeypatch, lengths)
    kwargs = _verify_kwargs(world, lengths)
    before, eager, served = _eager_and_served(world, kwargs)
    draft = len(world.root.model.layers)
    draft_pool = f"layer_carriers[{draft}]['pool_cache']"
    assert _changed_rows(eager[draft_pool], before[draft_pool]), (
        f"{draft_pool}: the eager step writes no pooled row, so this case tests nothing"
    )
    lost = []
    for name, bank in served.items():
        if torch.equal(bank, eager[name]):
            continue
        wrote = _changed_rows(eager[name], before[name])
        kept = [row for row in wrote
                if torch.equal(residue._rows(bank)[row], residue._rows(eager[name])[row])]
        lost.append(f"{name}: the eager step writes flat row(s) {wrote}; the served "
                    f"step keeps {kept}")
    assert not lost, (
        f"prompts {lengths}, k={K}: the served step leaves {len(lost)} bank(s) unlike "
        f"the eager step (the draft layer's pooled store is {draft_pool}); "
        + "; ".join(lost)
    )


# ── 2. every read sees the rows written before it ───────────────────────────


def _consumed_rows(call: dict, store_rows: int, pool: int) -> list[int]:
    """The flat bank rows a score call reads: each request's slot, at the candidate pools
    its rows' causal lengths reach (a pool is read once it is closed)."""
    rows = int(call["rows"][0])
    out = set()
    for slot, start in zip(call["slots"].tolist(), call["position"].tolist()):
        closed = (int(start) + rows) // pool
        out.update(int(slot) * store_rows + c for c in range(closed))
    return sorted(out)


@pytest.mark.parametrize(
    "lengths, share",
    [((4, 6), True), ((4, 6), False), ((5, 7), False)],
    ids=["bs2-4-6-shared", "bs2-4-6-unshared", "bs2-5-7-unshared"],
)
def test_every_score_kernel_reads_the_pooled_rows_written_before_it(monkeypatch, lengths, share):
    world = _bank_form_world(monkeypatch, lengths, share=share)
    kwargs = _verify_kwargs(world, lengths)
    calls = _recorded_reads(monkeypatch)
    state = residue._state(world, kwargs)
    before = {name: bank.clone() for name, bank in state.items()}
    world.root.forward(**dict(kwargs))
    eager = list(calls)
    calls.clear()
    for name, bank in state.items():
        bank.copy_(before[name])
    residue._through_the_backend(lambda **kw: world.root.forward(**kw), dict(kwargs))
    served = list(calls)
    trunk = len(world.root.model.layers)
    assert len(served) == len(eager) > trunk, (len(served), len(eager), trunk)
    stale = []
    for index, (got, want) in enumerate(zip(served, eager)):
        who = f"call {index} ({'trunk layer ' + str(index) if index < trunk else 'draft layer'})"
        if not torch.equal(got["bank"], want["bank"]):
            stale.append(f"{who}: the bank it reads differs at flat row(s) "
                         f"{_changed_rows(got['bank'], want['bank'])}")
        elif not torch.equal(got["scores"], want["scores"]):
            stale.append(f"{who}: its scores differ")
    assert not stale, (
        f"prompts {lengths}, k={K}, index share {share}: a score kernel of the served "
        f"step reads a pooled store unlike the eager step's; " + "; ".join(stale)
    )
    # The draft layer's later calls do read rows the step wrote before them: the check
    # above compares fresh reads, not untouched ones.
    indexer = world.root.mtp.block.self_attn.indexer
    pool = int(indexer.index_kpool)
    draft_before = residue._rows(before[f"layer_carriers[{trunk}]['pool_cache']"])
    store_rows = int(eager[trunk]["bank"].shape[1])
    fresh = [
        index for index, call in enumerate(eager[trunk + 1:], start=trunk + 1)
        if any(not torch.equal(residue._rows(call["bank"])[row], draft_before[row])
               for row in _consumed_rows(call, store_rows, pool))
    ]
    if not share:
        assert fresh, (
            f"prompts {lengths}: no draft iteration reads a pooled row the step wrote "
            f"before it, so this case tests nothing"
        )


# ── 3. the flat store is the same store ─────────────────────────────────────


def _sparse_layer():
    """The functional DSA cases' sparse layer and its config."""
    config = decode_config(hidden_size=512, q_lora_rank=256)
    return build_attention(model_fp8, config), config


#: Request lengths that straddle the selection bound, pool completions and page edges.
FLAT_CASES = {"b2": (2100, 4096), "b4": (2100, 3000, 4096, 8000),
              "b8": (2100, 3000, 4096, 8000, 2052, 5003, 7999, 6144)}
FLAT_MAX_SEQ_LEN = 8192


@pytest.mark.parametrize("lengths", FLAT_CASES.values(), ids=FLAT_CASES.keys())
def test_the_flat_store_selects_and_writes_as_the_bank(lengths):
    """The runner hands the pooled store flat; the eager step is the 3-D bank's, bit for bit."""
    module, config = _sparse_layer()
    indexer = module.indexer
    ops = case.batch_operands(config, list(lengths), max_seq_len=FLAT_MAX_SEQ_LEN, seed=11)
    q_latent = module.project_query_latent(ops["hidden"])
    projected = indexer.project_stage(ops["hidden"], q_latent)
    by_bank, by_flat = case.cloned(ops), case.cloned(ops)
    results = []
    for operands, store in ((by_bank, by_bank["pool_bank"]),
                            (by_flat, by_flat["pool_bank"].view(-1, int(indexer.index_head_dim)))):
        results.append(indexer.forward_requests(
            ops["hidden"], q_latent, store, operands["tail_bank"], operands["state_slots"],
            ops["seq_lens"], ops["position"], max_seq_len=FLAT_MAX_SEQ_LEN,
            projected=projected,
        ))
    assert torch.equal(results[1], results[0])
    assert torch.equal(by_flat["pool_bank"], by_bank["pool_bank"])
    assert torch.equal(by_flat["tail_bank"], by_bank["tail_bank"])
    # Something was written, so the equality is not two untouched copies.
    assert not torch.equal(by_flat["pool_bank"], ops["pool_bank"])


def test_a_flat_store_that_does_not_split_over_the_rings_is_refused_by_name():
    module, config = _sparse_layer()
    indexer = module.indexer
    ops = case.batch_operands(config, list(FLAT_CASES["b2"]), max_seq_len=FLAT_MAX_SEQ_LEN)
    flat = ops["pool_bank"].reshape(-1, int(indexer.index_head_dim))
    q_latent = module.project_query_latent(ops["hidden"])
    with pytest.raises(model_fp8.Glm5NextDSAIndexerError, match="a flat pool_bank must be"):
        indexer.forward_requests(
            ops["hidden"], q_latent, flat[:-1], ops["tail_bank"], ops["state_slots"],
            ops["seq_lens"], ops["position"], max_seq_len=FLAT_MAX_SEQ_LEN,
        )
