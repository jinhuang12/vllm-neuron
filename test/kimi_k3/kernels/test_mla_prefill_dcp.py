# SPDX-License-Identifier: Apache-2.0
"""The paged MLA prefill of one rank of a DCP group, byte for byte against the copy path.

The rank reads the cached prefix from a window of the sequence's latent blocks gathered over
the group, rank by rank, and stores each chunk row at the runner's slot in its own page buffer:
a real slot where the rank holds the row, the null block elsewhere. The copy path, run here on
copies of the same inputs as the oracle, is the paged prefill on the window with each token's
window position as its slot, then a torch copy of those rows into the cache. The page buffer
must equal the copy path's in every byte but the null row, block 0's first latent row, which
both paths write with no defined winner; the attention output must equal the paged prefill's
bit for bit. Each shape runs clean, then with one NaN ``qkv_a`` row at a real slot.

On a Neuron device the wrappers run under ``torch.compile`` on core 0 of
``NEURON_RT_VISIBLE_CORES``; with ``VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1`` they run in the NKI
simulator.
"""

import math

import numpy as np
import pytest
import torch
import test_mla_paged as paged

from vllm_neuron.functional.attention.mla_paged import (
    mla_paged_prefill,
    mla_paged_prefill_dcp,
    prepare_mla_weights,
)
from vllm_neuron.vllm.worker.neuron_slot_mapping import compute_slot_mapping_cpu

BLOCK, BLOCKS, FIRST_OWN = 256, 220, 128  # the rank's own blocks lie in 128 to 219
LATENT, ROW_BYTES = paged.LATENT, paged.ROW_BYTES
PAGE_BYTES = BLOCK * ROW_BYTES  # one latent block per page, from byte 0: slot s is row s


def _case(dcp, chunk, weights, gen):
    """A chunk over a one-chunk prefix on one rank of ``dcp``: the window and its table, the
    rank's page buffer, the runner's slots and one real token for the NaN run."""
    prefix = chunk
    per_rank = math.ceil(2 * chunk / (BLOCK * dcp))
    blocks = dcp * per_rank
    # Sequence block j is rank j % dcp's (j // dcp)-th block, in that rank's part of the window.
    table = torch.tensor([j % dcp * per_rank + j // dcp for j in range(blocks)],
                         dtype=torch.int32)
    rows = torch.randn(2 * chunk, paged.QKV, generator=gen)
    cached = paged._latent(rows[:prefix], weights[1]).bfloat16()
    window = torch.full((blocks, BLOCK, LATENT), float("nan"), dtype=torch.bfloat16)
    for j in range(math.ceil(prefix / BLOCK)):
        part = cached[j * BLOCK:(j + 1) * BLOCK]
        window[int(table[j]), :part.shape[0]] = part
    rank = (prefix // BLOCK + 1) % dcp
    own = FIRST_OWN + torch.randperm(BLOCKS - FIRST_OWN, generator=gen)[:per_rank]
    slots = np.zeros(chunk, dtype=np.int64)
    compute_slot_mapping_cpu(own.numpy()[None].astype(np.int32), slots,
                             np.arange(prefix, prefix + chunk), np.zeros(chunk, dtype=np.int64),
                             BLOCK, dcp, rank, BLOCK)
    slots = torch.from_numpy(slots)
    slots[-2:] = -1  # two padding rows
    real = (slots > 0).nonzero().flatten()
    slots[real[-1]] = BLOCKS * BLOCK  # a slot past the buffer
    return {"rows": rows, "prefix": prefix, "window": window, "table": table, "slots": slots,
            "pages": torch.randint(-128, 128, (BLOCKS, PAGE_BYTES), generator=gen,
                                   dtype=torch.int8),
            "gate": torch.randn(chunk, paged.HEADS * paged.V_DIM, generator=gen).bfloat16(),
            "nan_token": int(real[77])}


def _inputs(case, weights, qkv_a):
    q_norm, kv_norm, q_b, kv_b = weights
    return (qkv_a.float(), case["gate"], q_norm, kv_norm, *prepare_mla_weights(q_b, kv_b))


def _copy_path(case, weights, qkv_a):
    """The paged prefill on the window at each token's window position, then the rows' copy to
    their slots of the cache, on copies: the output and the page buffer."""
    window, pages = paged._as_pages(case["window"].clone()), case["pages"].clone()
    table, chunk = case["table"], qkv_a.shape[0]
    pos = (case["prefix"] + torch.arange(chunk)).clamp(max=table.shape[0] * BLOCK - 1)
    local = (table.long()[pos // BLOCK] * BLOCK + pos % BLOCK).to(torch.int32)
    out, window = paged._run(mla_paged_prefill, *_inputs(case, weights, qkv_a), window, 0, BLOCK,
                             table, torch.tensor([case["prefix"]], dtype=torch.int32), local,
                             paged.EPS)
    rows = window.view(torch.bfloat16).view(-1, LATENT).index_select(0, local.long())
    s = case["slots"].long()
    s = torch.where((s < 0) | (s >= BLOCKS * BLOCK), 0, s)
    target = s // BLOCK * (PAGE_BYTES // ROW_BYTES) + s % BLOCK
    pages.view(torch.bfloat16).view(-1, LATENT).index_copy_(0, target, rows)
    return out, pages


def _dcp(case, weights, qkv_a):
    """The DCP prefill entry on copies: the output and the rank's page buffer."""
    return paged._run(mla_paged_prefill_dcp, *_inputs(case, weights, qkv_a), case["pages"].clone(),
                      0, BLOCK, paged._as_pages(case["window"].clone()), case["table"],
                      torch.tensor([case["prefix"]], dtype=torch.int32), case["slots"], paged.EPS)


def _bits(x):
    return x.view(torch.int16)


def _check_against_copy_path(got, want):
    (out, pages), (ref_out, ref_pages) = got, want
    kept = torch.ones(BLOCKS, PAGE_BYTES, dtype=torch.bool)
    kept[0, :ROW_BYTES] = False  # the null row
    differ = int((pages != ref_pages)[kept].sum())
    assert differ == 0, f"{differ} bytes of the page buffer differ from the copy path's"
    assert torch.equal(_bits(out), _bits(ref_out)), "the output differs from the paged prefill's"


@pytest.mark.parametrize("dcp, chunk", [(2, 1024), (2, 512), (4, 1024), (4, 512)],
                         ids=["dcp2-chunk1024", "dcp2-chunk512", "dcp4-chunk1024",
                              "dcp4-chunk512"])
def test_rows_stored_at_the_rank_slots_as_the_copy_path(dcp, chunk):
    """Clean, then with one NaN qkv_a row at a real slot: the copy path's bytes and output."""
    gen = torch.Generator().manual_seed(20261029 + 10 * dcp + chunk)
    weights = paged._weights(gen)
    case = _case(dcp, chunk, weights, gen)
    qkv_a = case["rows"][case["prefix"]:]
    clean = _dcp(case, weights, qkv_a)
    _check_against_copy_path(clean, _copy_path(case, weights, qkv_a))
    assert torch.isfinite(clean[0][case["slots"] != -1].float()).all()

    t = case["nan_token"]
    poisoned = qkv_a.clone()
    poisoned[t] = float("nan")
    out, pages = _dcp(case, weights, poisoned)
    _check_against_copy_path((out, pages), _copy_path(case, weights, poisoned))
    assert torch.isnan(pages.view(torch.bfloat16).view(-1, LATENT)[int(case["slots"][t])]).all()
    assert torch.equal(_bits(out[:t]), _bits(clean[0][:t]))
