# SPDX-License-Identifier: Apache-2.0
"""The decode score kernel past 16,384 candidates: the candidate axis in chunks.

``dsa_decode_scores`` served at most ``128 * 128 = 16,384`` candidates (65,536 tokens of
context at pool 4) because the output transpose put every candidate tile on the
partitions at once and every key of a request sat in SBUF at once. The kernel now walks
the candidate axis in chunks of at most 128 tiles. Two claims are checked here:

* Up to 16,384 candidates the kernel is the b17526a kernel: one chunk, the same
  instructions, so the scores are equal bit for bit. The reference is b17526a's own
  file, loaded from git into a module of its own (never by editing the tree).
* Past 16,384 candidates the scores match the torch oracle (the bf16 products summed in
  fp32 in another order: ``SCORE_RTOL``), every bounded column holds ``BOUND_FILL``
  exactly, and this step's pool stands in at its own column in any chunk.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from vllm_neuron.functional.dsa import decode_batch as DB
from vllm_neuron.functional.dsa.causal_bound import BOUND_FILL
from vllm_neuron.utils.neuron_utils import can_run_kernel

POOL = 4
HEAD_DIM = 128
HEADS = 32
#: The commit whose kernel the chunked one must reproduce bit for bit.
BASE_COMMIT = "b17526a"
#: See test_decode_batch.py: fp32 rounding of a 128-term bf16 dot product and a 32-term
#: head sum, in two orders.
SCORE_RTOL = 1e-5
SCORE_ATOL = 1e-5
#: The old kernel's widest candidate axis.
OLD_MAX = 128 * 128


def _load_base_module():
    """b17526a's decode_batch.py as a module of its own, from ``git show``."""
    root = Path(DB.__file__).resolve().parents[3]
    source = subprocess.run(
        ["git", "-C", str(root), "show",
         f"{BASE_COMMIT}:vllm_neuron/functional/dsa/decode_batch.py"],
        check=True, capture_output=True).stdout
    tmp = Path(__import__("tempfile").mkdtemp(prefix="decode_batch_base_"))
    path = tmp / "decode_batch_b17526a.py"
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location("decode_batch_b17526a", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def base():
    return _load_base_module()


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


@pytest.mark.parametrize("lnc", [None, "2"])
@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("candidates", [512, 2048, OLD_MAX])
def test_chunked_kernel_equals_b17526a_bit_for_bit(base, monkeypatch, batch, candidates,
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


def test_the_old_kernel_refused_what_the_chunked_one_serves(base):
    candidates = OLD_MAX + 1
    case = _case(1, seed=2, candidates=candidates, lengths=[candidates * POOL])
    with pytest.raises(base.DecodeBatchError, match="not served"):
        base.dsa_decode_scores(*case, candidates=candidates, pool_size=POOL)
    got = DB.dsa_decode_scores(*case, candidates=candidates, pool_size=POOL)
    assert got.shape == (1, candidates)
