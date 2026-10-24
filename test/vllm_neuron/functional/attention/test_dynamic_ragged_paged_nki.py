# SPDX-License-Identifier: Apache-2.0
"""Reference and regression tests for dynamic-q_len NKI paged attention.

CPU tests do not import the Neuron SDK and run in ordinary CI. Set
NKI_RPA_SIMULATE=1 with the Neuron SDK installed to run the NKI simulator test.
No test silently claims that the Neuron backend compiled successfully.
"""
from __future__ import annotations

import ast
import math
import os
from pathlib import Path

import pytest
import torch

PAGE = 128
DIM = 128
SCALE = 1.0 / math.sqrt(DIM)
KERNEL = Path(__file__).resolve().parents[4] / (
    "vllm_neuron/functional/attention/dynamic_ragged_paged_nki.py"
)


def _make_case(q_lens, old_lens, num_q_heads=4, num_kv_heads=2):
    """Give every request noncontiguous physical KV pages.

    Cache includes *both* prior K/V and the current-step K/V. No cross-request
    page sharing. The logical-page -> physical-page mapping is shuffled.
    """
    assert len(q_lens) == len(old_lens)
    assert num_q_heads % num_kv_heads == 0
    reqs = len(q_lens)
    kv_lens = [old + q for old, q in zip(old_lens, q_lens)]
    pages_per_req = max(1, max((length + PAGE - 1) // PAGE for length in kv_lens))
    total_pages = sum((length + PAGE - 1) // PAGE for length in kv_lens)
    total_pages += 5  # unused physical pages and nontrivial page IDs
    generator = torch.Generator().manual_seed(20261009)
    key_pages = (torch.randn(total_pages, num_kv_heads, DIM, PAGE,
                             generator=generator) * 0.35).bfloat16()
    value_pages = (torch.randn(total_pages, num_kv_heads, PAGE, DIM,
                               generator=generator) * 0.35).bfloat16()
    physical = torch.randperm(total_pages, generator=generator)
    table = torch.full((reqs, pages_per_req), -1, dtype=torch.int32)
    cursor = 0
    for r, length in enumerate(kv_lens):
        n = (length + PAGE - 1) // PAGE
        if n:
            table[r, :n] = physical[cursor:cursor + n].to(torch.int32)
            cursor += n
    cuq = [0]
    for n in q_lens:
        cuq.append(cuq[-1] + n)
    max_tokens = max(1, cuq[-1])  # kernel's static capacity can exceed active T
    q = (torch.randn(num_q_heads, DIM, max_tokens,
                     generator=generator) * 0.35).bfloat16()
    return (q, key_pages, value_pages, table,
            torch.tensor(cuq, dtype=torch.int32),
            torch.tensor(kv_lens, dtype=torch.int32))


def _logical_cache(key_pages, value_pages, table, req, length):
    blocks = (length + PAGE - 1) // PAGE
    k = []
    v = []
    for idx in table[req, :blocks].tolist():
        assert idx >= 0
        k.append(key_pages[idx].permute(0, 2, 1))   # [head, page, dim]
        v.append(value_pages[idx])
    if not k:
        return None, None
    return (torch.cat(k, dim=1)[:, :length],
            torch.cat(v, dim=1)[:, :length])


def _dense_reference(q, kp, vp, table, cuq, kv_lens):
    """Independent FP32 attention, with exact per-request causal prefix."""
    heads, _, tokens = q.shape
    out = torch.zeros_like(q)
    kvheads = kp.shape[1]
    assert heads % kvheads == 0
    for req in range(len(kv_lens)):
        begin, end = cuq[req:req + 2].tolist()
        length = int(kv_lens[req])
        old = length - (end - begin)
        if end == begin:
            continue
        k, v = _logical_cache(kp, vp, table, req, length)
        for head in range(heads):
            kvhead = head // (heads // kvheads)
            for idx in range(begin, end):
                visible = old + (idx - begin) + 1
                query = q[head, :, idx].float()
                keys = k[kvhead, :visible].float()
                vals = v[kvhead, :visible].float()
                prob = (query @ keys.T * SCALE).softmax(dim=0)
                out[head, :, idx] = (prob @ vals).bfloat16()
    return out


def _page_online_reference(q, kp, vp, table, cuq, kv_lens):
    """CPU reconstruction of NKI's 1-query/page-by-page FlashAttention."""
    heads, _, tokens = q.shape
    out = torch.zeros_like(q)
    kvheads = kp.shape[1]
    for req in range(len(kv_lens)):
        begin, end = cuq[req:req + 2].tolist()
        length = int(kv_lens[req])
        old = length - (end - begin)
        for head in range(heads):
            kvhead = head // (heads // kvheads)
            for idx in range(begin, end):
                limit = old + idx - begin + 1
                query = q[head, :, idx].float()
                m = -1.0e30
                l = torch.zeros((), dtype=torch.float32)
                acc = torch.zeros((DIM,), dtype=torch.float32)
                for page in range((limit + PAGE - 1) // PAGE):
                    physical = int(table[req, page])
                    k = kp[physical, kvhead].float()  # [dim, page]
                    v = vp[physical, kvhead].float()  # [page, dim]
                    score = query @ k * SCALE
                    valid = min(PAGE, limit - page * PAGE)
                    score[valid:] = -1.0e30
                    new_m = max(m, float(score.max()))
                    alpha = math.exp(m - new_m)
                    probs = (score - new_m).exp()
                    l = alpha * l + probs.sum()
                    # The NKI TensorE second GEMM consumes rounded BF16 probs.
                    pv = probs.to(torch.bfloat16).float() @ v
                    acc = alpha * acc + pv
                    m = new_m
                out[head, :, idx] = (acc / l).bfloat16()
    return out


@pytest.mark.parametrize("q_lens,old_lens", [
    ((1, 3, 2), (127, 5, 0)),     # mixed decode/prefill, short tail
    ((2, 129), (127, 1)),         # crosses both 128-token boundaries
    ((1, 1, 1), (0, 255, 509)),  # decode-only
    ((0, 1, 0), (0, 0, 300)),    # zero scheduled queries
    ((4, 7, 1), (255, 127, 128)),
])
def test_paged_online_softmax_matches_dense(q_lens, old_lens):
    case = _make_case(q_lens, old_lens)
    dense = _dense_reference(*case)
    online = _page_online_reference(*case)
    # BF16 probability weights in the NKI PV GEMM add quantization error.
    torch.testing.assert_close(online.float(), dense.float(),
                               atol=0.022, rtol=0.022)
    # Inactive padded tokens must never be compared or consumed.
    assert dense.shape == case[0].shape


def test_kernel_has_actual_runtime_q_and_kv_loops():
    """Reject accidental regressions to static maximum-Q/KV iteration."""
    code = KERNEL.read_text()
    ast.parse(code)  # no SDK needed for Python syntax validation
    assert "nl.fori_loop(q_start_reg, q_end_reg, query_body)" in code
    assert "nl.fori_loop(0, pages_reg, kv_page_body)" in code
    assert "scalar_offset=q_index, indirect_dim=2" in code
    assert "scalar_offset=page_id, indirect_dim=0" in code
    assert "nisa.tensor_copy_predicated" in code
    assert "nisa.nc_matmul" in code


@pytest.mark.skipif(os.environ.get("NKI_RPA_SIMULATE") != "1",
                    reason="set NKI_RPA_SIMULATE=1 with NKI SDK installed")
def test_nki_simulator_matches_dense_mixed_batch():
    nki = pytest.importorskip("nki")
    from vllm_neuron.functional.attention.dynamic_ragged_paged_nki import (
        dynamic_ragged_paged_attention_nki,
    )

    case = _make_case((1, 3, 2), (127, 5, 0))
    expected = _dense_reference(*case)
    # Simulate the (request, head) grid, which shares one output buffer.
    actual = nki.simulate(
        dynamic_ragged_paged_attention_nki[(3, case[0].shape[0])]
    )(*case)
    if not isinstance(actual, torch.Tensor):
        actual = torch.from_numpy(actual)
    torch.testing.assert_close(actual.float(), expected.float(),
                               atol=0.035, rtol=0.035)
