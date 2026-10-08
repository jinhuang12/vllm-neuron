# SPDX-License-Identifier: Apache-2.0
"""The decode score kernel past 16,384 candidates: the candidate axis in chunks.

``dsa_decode_scores`` served at most ``128 * 128 = 16,384`` candidates (65,536 tokens of
context at pool 4) because the output transpose put every candidate tile on the
partitions at once and every key of a request sat in SBUF at once. The kernel now walks
the candidate axis in chunks of at most 128 tiles. Two claims are checked here:

* Up to 16,384 candidates the kernel is the unchunked kernel: one chunk, the same
  instructions, so the scores are equal bit for bit. The reference is the last unchunked
  commit's own file, loaded from git into a module of its own (never by editing the
  tree).
* Past 16,384 candidates the scores match the torch oracle (the bf16 products summed in
  fp32 in another order: ``SCORE_RTOL``), every bounded column holds ``BOUND_FILL``
  exactly, and this step's pool stands in at its own column in any chunk.
"""

from __future__ import annotations

import pytest
import torch

from vllm_neuron.functional.dsa import decode_batch as DB
from vllm_neuron.functional.dsa.causal_bound import BOUND_FILL
from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig
from vllm_neuron.utils.neuron_utils import can_run_kernel

from test.vllm_neuron.functional.dsa.test_decode_batch import SCORE_ATOL, SCORE_RTOL
from test.vllm_neuron.functional.reference_at_commit import load_reference, needs_reference

_CONFIG = Glm5NextTextConfig()
#: GLM-5.3-Flash's indexer: tokens per candidate pool, key width and query heads.
POOL = _CONFIG.index_kpool
HEAD_DIM = _CONFIG.index_head_dim
HEADS = _CONFIG.index_n_heads
#: The last commit whose kernel has no chunks: the chunked one must reproduce it bit for
#: bit.
UNCHUNKED_COMMIT = "b17526a"
#: The unchunked kernel's widest candidate axis, which is one chunk now
#: (``test_one_chunk_is_the_old_widest_axis`` checks it against that kernel).
OLD_MAX = DB.CHUNK_CANDIDATES


#: The reference file at :data:`UNCHUNKED_COMMIT`.
UNCHUNKED_PATH = "vllm_neuron/functional/dsa/decode_batch.py"
#: The bit-equality tests need the unchunked kernel from git; without it they skip, and say why.
needs_unchunked = needs_reference(UNCHUNKED_COMMIT, UNCHUNKED_PATH)


@pytest.fixture(scope="module")
def base():
    return load_reference(UNCHUNKED_COMMIT, UNCHUNKED_PATH, "decode_batch_unchunked")


@pytest.fixture(autouse=True)
def _kernels_on():
    assert can_run_kernel(), "run with VLLM_NEURON_CPU_MODE=1 and NKI_SIMULATOR=1"
    DB.reset_decode_batch_dispatch_counters()


def _bf16(gen, *shape, scale=1.0):
    return (torch.randn(shape, generator=gen) * scale).to(torch.bfloat16)


def _case(batch: int, seed: int, *, candidates: int, lengths, slots_total: int = 5):
    gen = torch.Generator().manual_seed(seed)
    bank = _bf16(gen, slots_total, candidates + 1, HEAD_DIM, scale=0.5)
    slots = torch.randperm(slots_total, generator=gen)[:batch].to(torch.int32)
    seq_lens = torch.tensor(lengths[:batch], dtype=torch.int32)
    position = seq_lens - 1
    query = _bf16(gen, batch, HEADS, HEAD_DIM)
    weights = torch.randn(batch, HEADS, generator=gen) * HEADS ** -0.5
    pooled = _bf16(gen, batch, HEAD_DIM, scale=0.5)
    return query, weights, bank, slots, seq_lens, position, pooled


def _lengths(candidates: int) -> list[int]:
    """Four lengths: the whole axis, a pool closed mid-axis, a short one, a ragged tail.

    Lengths that are a multiple of the pool close a pool this step (``position % 4 ==
    3``), so the stand-in lands at column ``length / 4 - 1``; the others leave it out.
    """
    full = candidates * POOL
    mid = (candidates // 2 + 7) * POOL
    return [full, mid, 9, full - 2]


def _served_programs(monkeypatch, lnc):
    """LNC2 (the served setting) splits ``B >= 2`` over two programs; unset runs one."""
    if lnc is None:
        monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    else:
        monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", lnc)


@needs_unchunked
def test_one_chunk_is_the_old_widest_axis(base):
    assert OLD_MAX == base.MAX_CANDIDATES == DB.CHUNK_TILES * DB.PARTITIONS


@needs_unchunked
@pytest.mark.parametrize("lnc", [None, "2"])
@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("candidates", [512, 2048, OLD_MAX])
def test_chunked_kernel_equals_the_unchunked_kernel_bit_for_bit(base, monkeypatch, batch, candidates,
                                                    lnc):
    _served_programs(monkeypatch, lnc)
    case = _case(batch, seed=candidates + batch, candidates=candidates,
                 lengths=_lengths(candidates))
    want = base.dsa_decode_scores(*case, candidates=candidates, pool_size=POOL)
    got = DB.dsa_decode_scores(*case, candidates=candidates, pool_size=POOL)
    assert DB.decode_batch_dispatch_counters() == (1, 0)
    two = 1 if (lnc == "2" and batch >= 2) else 0
    assert DB.decode_batch_route_counts()[2] == two
    assert got.shape == want.shape == (batch, candidates)
    assert torch.equal(got, want)


@pytest.mark.parametrize("lnc", [None, "2"])
@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("candidates", [OLD_MAX + 1, 2 * OLD_MAX, 4 * OLD_MAX])
def test_chunked_kernel_matches_the_oracle_past_the_old_cap(monkeypatch, batch,
                                                             candidates, lnc):
    _served_programs(monkeypatch, lnc)
    case = _case(batch, seed=candidates + 3 * batch, candidates=candidates,
                 lengths=_lengths(candidates))
    got = DB.dsa_decode_scores(*case, candidates=candidates, pool_size=POOL)
    assert DB.decode_batch_dispatch_counters() == (1, 0)
    two = 1 if (lnc == "2" and batch >= 2) else 0
    assert DB.decode_batch_route_counts()[2] == two
    want = DB.dsa_decode_scores_torch_oracle(*case, candidates=candidates,
                                             pool_size=POOL)
    assert got.shape == want.shape == (batch, candidates)
    filled = want == BOUND_FILL
    assert torch.equal(got == BOUND_FILL, filled)
    torch.testing.assert_close(got[~filled], want[~filled], rtol=SCORE_RTOL,
                               atol=SCORE_ATOL)
    # The stand-in: a request whose length is a multiple of the pool scores ``pooled``
    # at its last column, which is past the first chunk for the long requests.
    query, weights, bank, slots, seq_lens, position, pooled = case
    for b in range(batch):
        if int(seq_lens[b]) % POOL:
            continue
        column = int(seq_lens[b]) // POOL - 1
        own = (query[b].float() @ pooled[b].float()).clamp(min=0.0)
        expect = (own * weights[b].float()).sum()
        torch.testing.assert_close(got[b, column], expect, rtol=SCORE_RTOL,
                                   atol=SCORE_ATOL)


@needs_unchunked
def test_the_old_kernel_refused_what_the_chunked_one_serves(base):
    candidates = OLD_MAX + 1
    case = _case(1, seed=2, candidates=candidates, lengths=[candidates * POOL])
    with pytest.raises(base.DecodeBatchError, match="not served"):
        base.dsa_decode_scores(*case, candidates=candidates, pool_size=POOL)
    got = DB.dsa_decode_scores(*case, candidates=candidates, pool_size=POOL)
    assert got.shape == (1, candidates)
