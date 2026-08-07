# SPDX-License-Identifier: Apache-2.0
"""Increment 2 (P-EAGLE mask-input construction) CPU-only unit tests.

Golden-value tests for ``vllm_neuron/functional/parallel_draft_inputs.py``. Each
vectorized function is checked against a slow python reference loop that
reimplements upstream v0.21.0's parallel-drafting expansion semantics
(``vllm/v1/spec_decode/utils.py`` ``copy_and_expand_eagle_inputs_kernel`` and
``vllm/v1/spec_decode/llm_base_proposer.py:748-753``) for the decode
parallel-draft slot layout: per request, slot 0 is the real seed token, slots
``1..K-1`` are the mask (``ptd``) token / mask-hidden.

Runs without Neuron hardware (pure torch on CPU).
"""
import pytest
import torch

from vllm_neuron.functional.parallel_draft_inputs import (
    build_parallel_draft_hidden_mask,
    build_parallel_draft_input_ids,
    build_parallel_draft_positions,
    substitute_mask_hidden,
)

PTD = 201020


# --- slow python reference implementations (mirror upstream semantics) ------


def _ref_input_ids(seed_token_ids, K, ptd_token_id):
    """Reference: slot 0 = seed (bonus region), slots 1..K-1 = ptd token."""
    batch = len(seed_token_ids)
    out = [[0] * K for _ in range(batch)]
    for i in range(batch):
        for k in range(K):
            # is_bonus_region: k == 0 -> seed; is_parallel_draft_region: k>0.
            out[i][k] = int(seed_token_ids[i]) if k == 0 else ptd_token_id
    return out


def _ref_positions(base_positions, K):
    """Reference: positions = start_pos + j (utils.py:416)."""
    batch = len(base_positions)
    return [[int(base_positions[i]) + k for k in range(K)] for i in range(batch)]


def _ref_mask(batch, K):
    """Reference: is_masked_out True on parallel-draft region (k >= 1)."""
    return [[k >= 1 for k in range(K)] for _ in range(batch)]


def _ref_substitute(hidden, mask, mask_hidden):
    """Reference: torch.where(mask, mask_hidden, hidden) over last dim."""
    hidden = hidden.clone()
    for idx in range(hidden.shape[0]):
        for k in range(hidden.shape[1]):
            if mask[idx][k]:
                hidden[idx, k] = mask_hidden.to(hidden.dtype)
    return hidden


# --- input id expansion ------------------------------------------------------


@pytest.mark.parametrize("K", [2, 4])
@pytest.mark.parametrize("batch", [1, 3])
def test_input_ids_match_reference(K, batch):
    seeds = torch.arange(10, 10 + batch, dtype=torch.int32)
    got = build_parallel_draft_input_ids(seeds, K, PTD)
    ref = torch.tensor(_ref_input_ids(seeds, K, PTD), dtype=torch.int32)
    assert got.shape == (batch, K)
    assert torch.equal(got, ref)


def test_input_ids_mixed_seed_tokens():
    """Mixed per-request seed tokens land in slot 0 only; rest are ptd."""
    seeds = torch.tensor([7, 100, 3], dtype=torch.int32)
    K = 4
    got = build_parallel_draft_input_ids(seeds, K, PTD)
    assert got[:, 0].tolist() == [7, 100, 3]
    assert torch.all(got[:, 1:] == PTD)


def test_input_ids_dtype_preserved():
    for dtype in (torch.int32, torch.int64):
        seeds = torch.tensor([1, 2], dtype=dtype)
        got = build_parallel_draft_input_ids(seeds, 3, PTD)
        assert got.dtype == dtype


def test_input_ids_k1_no_mask_slots():
    """K=1: only the seed slot, no ptd tokens."""
    seeds = torch.tensor([5, 6], dtype=torch.int32)
    got = build_parallel_draft_input_ids(seeds, 1, PTD)
    assert got.shape == (2, 1)
    assert got[:, 0].tolist() == [5, 6]


def test_input_ids_invalid_k_raises():
    seeds = torch.tensor([1], dtype=torch.int32)
    with pytest.raises(ValueError):
        build_parallel_draft_input_ids(seeds, 0, PTD)


# --- positions ---------------------------------------------------------------


@pytest.mark.parametrize("K", [2, 4])
@pytest.mark.parametrize("batch", [1, 3])
def test_positions_match_reference(K, batch):
    base = torch.tensor([10 * (i + 1) for i in range(batch)], dtype=torch.int32)
    got = build_parallel_draft_positions(base, K)
    ref = torch.tensor(_ref_positions(base, K), dtype=torch.int32)
    assert got.shape == (batch, K)
    assert torch.equal(got, ref)


def test_positions_mixed_base_and_dtype():
    base = torch.tensor([0, 15, 999], dtype=torch.int64)
    got = build_parallel_draft_positions(base, 3)
    assert got.dtype == torch.int64
    assert got.tolist() == [[0, 1, 2], [15, 16, 17], [999, 1000, 1001]]


# --- hidden mask -------------------------------------------------------------


@pytest.mark.parametrize("K", [2, 4])
@pytest.mark.parametrize("batch", [1, 3])
def test_hidden_mask_match_reference(K, batch):
    got = build_parallel_draft_hidden_mask(batch, K)
    ref = torch.tensor(_ref_mask(batch, K), dtype=torch.bool)
    assert got.shape == (batch, K)
    assert got.dtype == torch.bool
    assert torch.equal(got, ref)


def test_hidden_mask_positions_exact():
    m = build_parallel_draft_hidden_mask(2, 4)
    # Column 0 all False (seed), columns 1..K-1 all True (masked).
    assert torch.all(~m[:, 0])
    assert torch.all(m[:, 1:])


# --- mask hidden substitution ------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("K", [2, 4])
def test_substitute_mask_hidden_match_reference(K, dtype):
    batch, H = 3, 8
    torch.manual_seed(0)
    hidden = torch.randn(batch, K, H, dtype=dtype)
    mask = build_parallel_draft_hidden_mask(batch, K)
    mask_hidden = torch.randn(H, dtype=torch.float32)
    got = substitute_mask_hidden(hidden, mask, mask_hidden)
    ref = _ref_substitute(hidden, _ref_mask(batch, K), mask_hidden)
    assert got.shape == hidden.shape
    assert got.dtype == dtype
    assert torch.equal(got, ref)


def test_substitute_seed_slot_unchanged():
    """Slot 0 must retain its original hidden state exactly."""
    batch, K, H = 2, 4, 5
    hidden = torch.arange(batch * K * H, dtype=torch.float32).reshape(batch, K, H)
    mask = build_parallel_draft_hidden_mask(batch, K)
    mask_hidden = torch.full((H,), -1.0)
    got = substitute_mask_hidden(hidden, mask, mask_hidden)
    assert torch.equal(got[:, 0], hidden[:, 0])
    # Every masked slot equals mask_hidden.
    assert torch.all(got[:, 1:] == -1.0)


def test_substitute_dtype_cast_of_mask_hidden():
    """mask_hidden provided as fp32 is cast to hidden's dtype (bf16)."""
    batch, K, H = 1, 2, 4
    hidden = torch.zeros(batch, K, H, dtype=torch.bfloat16)
    mask = build_parallel_draft_hidden_mask(batch, K)
    mask_hidden = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
    got = substitute_mask_hidden(hidden, mask, mask_hidden)
    assert got.dtype == torch.bfloat16
    assert torch.equal(got[0, 1], mask_hidden.to(torch.bfloat16))


def test_substitute_flattened_layout():
    """Works with a flattened [batch*K, H] hidden and [batch*K] mask."""
    batch, K, H = 3, 2, 6
    hidden = torch.randn(batch * K, H)
    mask = build_parallel_draft_hidden_mask(batch, K).reshape(-1)
    mask_hidden = torch.randn(H)
    got = substitute_mask_hidden(hidden, mask, mask_hidden)
    assert got.shape == (batch * K, H)
    # Rows where mask is True equal mask_hidden.
    for i in range(batch * K):
        if mask[i]:
            assert torch.equal(got[i], mask_hidden)
        else:
            assert torch.equal(got[i], hidden[i])


# --- integration: full expansion coherence -----------------------------------


def test_full_expansion_coherence():
    """input_ids, positions, and mask agree on which slots are masked."""
    seeds = torch.tensor([5, 9, 2], dtype=torch.int32)
    base_pos = torch.tensor([10, 20, 30], dtype=torch.int32)
    K = 4
    ids = build_parallel_draft_input_ids(seeds, K, PTD)
    pos = build_parallel_draft_positions(base_pos, K)
    mask = build_parallel_draft_hidden_mask(3, K)
    # Masked slots (mask True) are exactly the ptd-token slots.
    assert torch.equal(ids == PTD, mask)
    # Positions strictly increment per request regardless of masking.
    assert pos.tolist() == [[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]]
