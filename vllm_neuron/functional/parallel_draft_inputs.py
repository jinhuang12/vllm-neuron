# SPDX-License-Identifier: Apache-2.0
"""NEFF-compilable parallel-draft (P-EAGLE) input construction.

Pure-tensor reimplementation of upstream vLLM v0.21.0's parallel-drafting
input expansion, without the Triton kernel or any custom C++ op so it traces
cleanly into a Neuron NEFF. All shapes are static functions of
``(batch, num_speculative_tokens)``; there is no data-dependent control flow.

Upstream reference (all at tag ``v0.21.0``):

* ``vllm/v1/spec_decode/utils.py`` ``copy_and_expand_eagle_inputs_kernel``
  (:308-455). Region layout per request, for the EAGLE parallel-drafting case
  (``shift_input_ids`` / ``pass_hidden_states_to_model`` True):

  - a "bonus" slot that receives the real seed token id
    (``is_bonus_region``, :380, filled from ``next_token_ids`` at :406);
  - ``num_padding_slots_per_request`` "parallel draft" slots that receive
    ``parallel_drafting_token_id`` (``is_parallel_draft_region``, :381-383,
    filled at :407-409) and are flagged in ``out_is_masked_token_mask``
    (``is_masked_out``, :422);
  - positions increment from the request's start position:
    ``positions = start_pos + j`` (:416).

* ``vllm/v1/spec_decode/llm_base_proposer.py`` ``set_inputs_first_pass``
  (:743-753): after the kernel writes the drafting buffers, masked hidden
  slots are substituted with the learned mask-hidden vector via
  ``torch.where(is_masked_token_mask, parallel_drafting_hidden_state_tensor,
  hidden_states)`` — the DtoH-sync-free substitution reimplemented here as
  :func:`substitute_mask_hidden`.

Scope: this module covers the decode parallel-draft slot layout the Neuron
drafter graph consumes — a fixed ``num_speculative_tokens`` slots per request,
slot 0 the real seed token and slots ``1..K-1`` masked. Upstream's kernel also
handles variable per-request query lengths, prefill copy, and rejected-token
padding; those are dynamic-shape concerns handled outside the compiled drafter
forward and are intentionally not reproduced here (they would break the
static-shape requirement).
"""

from __future__ import annotations

import torch


def build_parallel_draft_input_ids(
    seed_token_ids: torch.Tensor,
    num_speculative_tokens: int,
    ptd_token_id: int,
) -> torch.Tensor:
    """Expand per-request seed tokens into the K-slot parallel-draft layout.

    Each request gets ``num_speculative_tokens`` (K) contiguous slots: slot 0
    holds the request's real seed token (upstream's "bonus" slot), and slots
    ``1..K-1`` hold ``ptd_token_id`` (upstream's parallel-draft slots). This is
    the single-pass input the drafter runs over instead of a K-step loop.

    Args:
        seed_token_ids: ``[batch]`` integer tensor, the first real token to
            draft from for each request (the accepted/bonus token).
        num_speculative_tokens: K, the number of drafter slots per request.
            Must be >= 1.
        ptd_token_id: The drafter checkpoint's mask token id written into the
            trailing ``K-1`` slots.

    Returns:
        ``[batch, K]`` integer tensor with the same dtype and device as
        ``seed_token_ids``. Column 0 is the seed token; columns ``1..K-1`` are
        ``ptd_token_id``.

    Example:
        >>> seeds = torch.tensor([5, 9], dtype=torch.int32)
        >>> build_parallel_draft_input_ids(seeds, 3, 42)
        tensor([[ 5, 42, 42],
                [ 9, 42, 42]], dtype=torch.int32)
    """
    if num_speculative_tokens < 1:
        raise ValueError(
            f"num_speculative_tokens must be >= 1, got {num_speculative_tokens}"
        )
    seed_2d = seed_token_ids.unsqueeze(1).expand(-1, num_speculative_tokens)
    ptd_filled = torch.full_like(seed_2d, ptd_token_id)
    is_masked = build_parallel_draft_hidden_mask(
        seed_token_ids.shape[0], num_speculative_tokens, device=seed_token_ids.device
    )
    # Slot 0 keeps the real seed; masked slots take the ptd token id.
    return torch.where(is_masked, ptd_filled, seed_2d)


def build_parallel_draft_positions(
    base_positions: torch.Tensor,
    num_speculative_tokens: int,
) -> torch.Tensor:
    """Build the per-slot positions for the parallel-draft layout.

    Positions increment from each request's base position, mirroring upstream
    ``positions = start_pos + j`` (``utils.py``:416). Slot ``k`` of request
    ``i`` gets ``base_positions[i] + k``.

    Args:
        base_positions: ``[batch]`` integer tensor, the position of each
            request's seed token (slot 0).
        num_speculative_tokens: K, the number of drafter slots per request.

    Returns:
        ``[batch, K]`` integer tensor with the same dtype and device as
        ``base_positions``.

    Example:
        >>> pos = torch.tensor([10, 4], dtype=torch.int32)
        >>> build_parallel_draft_positions(pos, 3)
        tensor([[10, 11, 12],
                [ 4,  5,  6]], dtype=torch.int32)
    """
    if num_speculative_tokens < 1:
        raise ValueError(
            f"num_speculative_tokens must be >= 1, got {num_speculative_tokens}"
        )
    offsets = torch.arange(
        num_speculative_tokens,
        dtype=base_positions.dtype,
        device=base_positions.device,
    )
    return base_positions.unsqueeze(1) + offsets.unsqueeze(0)


def build_parallel_draft_hidden_mask(
    batch: int,
    num_speculative_tokens: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build the boolean masked-slot mask for the parallel-draft layout.

    ``True`` marks the parallel-draft (masked) slots whose hidden state must be
    replaced with the learned mask-hidden vector; ``False`` marks slot 0, the
    real seed token. Matches upstream ``is_masked_out`` (``utils.py``:422),
    which is ``True`` exactly on the parallel-draft region (slots ``1..K-1``).

    Args:
        batch: Number of requests.
        num_speculative_tokens: K, the number of drafter slots per request.
        device: Device for the mask. Defaults to CPU.

    Returns:
        ``[batch, K]`` bool tensor; column 0 is ``False``, columns ``1..K-1``
        are ``True``.

    Example:
        >>> build_parallel_draft_hidden_mask(2, 3)
        tensor([[False,  True,  True],
                [False,  True,  True]])
    """
    if num_speculative_tokens < 1:
        raise ValueError(
            f"num_speculative_tokens must be >= 1, got {num_speculative_tokens}"
        )
    slot_idx = torch.arange(num_speculative_tokens, device=device)
    row = slot_idx >= 1  # [K]
    return row.unsqueeze(0).expand(batch, num_speculative_tokens)


def substitute_mask_hidden(
    hidden_states: torch.Tensor,
    mask: torch.Tensor,
    mask_hidden: torch.Tensor,
) -> torch.Tensor:
    """Substitute the learned mask-hidden vector into masked slots.

    Reimplements upstream's DtoH-sync-free hidden substitution
    (``llm_base_proposer.py``:748-753):
    ``torch.where(mask, mask_hidden, hidden_states)``. Masked slots take
    ``mask_hidden``; all other slots keep their original hidden state.

    Args:
        hidden_states: ``[..., H]`` hidden states (e.g. ``[batch, K, H]`` or a
            flattened ``[batch * K, H]``).
        mask: Boolean tensor broadcastable to ``hidden_states`` without its
            last dim (e.g. ``[batch, K]`` or ``[batch * K]``). ``True`` slots
            receive ``mask_hidden``. Typically produced by
            :func:`build_parallel_draft_hidden_mask`.
        mask_hidden: The learned mask-hidden vector, shape ``[H]`` (or any
            shape broadcastable to ``hidden_states``).

    Returns:
        Tensor with the same shape and dtype as ``hidden_states``.

    Example:
        >>> h = torch.zeros(2, 3, 4)
        >>> m = build_parallel_draft_hidden_mask(2, 3)
        >>> mh = torch.arange(4, dtype=torch.float32)
        >>> out = substitute_mask_hidden(h, m, mh)
        >>> out[0, 0].tolist()  # slot 0 unchanged
        [0.0, 0.0, 0.0, 0.0]
        >>> out[0, 1].tolist()  # masked slot -> mask_hidden
        [0.0, 1.0, 2.0, 3.0]
    """
    mask = mask.unsqueeze(-1)  # [..., 1] to broadcast over hidden dim
    mask_hidden = mask_hidden.to(hidden_states.dtype)
    return torch.where(mask, mask_hidden, hidden_states)
