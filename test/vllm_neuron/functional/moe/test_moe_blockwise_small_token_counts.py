# SPDX-License-Identifier: Apache-2.0
"""The blockwise MoE mapping builder at small token counts.

A decode step carries one token, which leaves the indexed-flatten kernel's
``f_len`` at 0, so the routing predicate has to answer instead of dividing by it
and the builder has to take its torch construction. Every comparison here is
exact -- integer mappings compared with ``torch.equal``, boolean predicate
answers -- so no tolerance appears anywhere.
"""

from __future__ import annotations

import os

import pytest
import torch

from vllm_neuron.functional.moe import moe_blockwise
from vllm_neuron.utils.neuron_utils import can_run_kernel

#: One decode step's routing shape: 16 local experts, two experts per token.
E_LOCAL = 16
TOP_K = 2
BLOCK_SIZE = 128

#: Token counts on both sides of the 16-token floor, the floor itself, and one
#: past it.
TOKEN_COUNTS = [1, 2, 8, 15, 16, 17, 32, 126, 128]

#: What ``_can_use_indexed_flatten_kernel`` must answer at each of those counts.
#: ``build_blockwise_mapping`` sizes the kernel with
#: ``f_len = min(128, total_tokens // 16)``, and the kernel needs both
#: ``T % f_len == 0`` and ``(T // f_len) % 16 == 0``. Below 16 tokens ``f_len`` is
#: 0, so the predicate answers False instead of dividing by it.
EXPECTED_INDEXED_FLATTEN = {
    1: False,
    2: False,
    8: False,
    15: False,
    16: True,
    17: False,
    32: True,
    126: False,
    128: True,
}


class MissingSimulatorEnvError(AssertionError):
    """The environment cannot run NKI kernels, so a comparison would prove nothing."""


def _require_modes() -> None:
    """Both variables must be in the process environment, not set from a fixture.

    ``can_run_kernel`` reads them at call time: in CPU mode it answers True only
    when ``NKI_SIMULATOR=1``. Without that the predicate returns False on its
    first line and never reaches the arithmetic these tests exercise.
    """
    for name in ("VLLM_NEURON_CPU_MODE", "NKI_SIMULATOR"):
        if os.environ.get(name) != "1":
            raise MissingSimulatorEnvError(
                f"{name} must be 1 in the process environment; without it "
                f"can_run_kernel answers False and nothing here is exercised"
            )


def _caller_f_len(tokens: int) -> int:
    """The sizing expression ``build_blockwise_mapping`` passes to the predicate."""
    return min(128, tokens // 16)


def _affinities(tokens: int) -> torch.Tensor:
    """``[T, E_LOCAL]`` affinities with exactly ``TOP_K`` non-zero entries per row.

    Built by arithmetic rather than from a seed: token ``t`` selects experts ``t``
    and ``t + 5`` modulo ``E_LOCAL``, and the two weights differ so a mapping that
    confused them would show. A seeded fixture would be harder to reproduce from a
    failure alone.
    """
    affinities = torch.zeros(tokens, E_LOCAL, dtype=torch.float32)
    for position in range(tokens):
        affinities[position, position % E_LOCAL] = 0.75
        affinities[position, (position + 5) % E_LOCAL] = 0.25
    return affinities


def _torch_route(tokens: int, expert_mask: torch.Tensor) -> tuple:
    """The torch route's own mapping, and the ``conditions`` derived from it.

    The caller's derivation is reproduced here from its own two lines so the
    comparison covers everything the public builder returns, not only the two
    mappings.
    """
    token_position_to_id, block_to_expert, num_blocks = (
        moe_blockwise._build_blockwise_mapping_torch(
            expert_mask=expert_mask,
            num_local_experts=E_LOCAL,
            num_experts_per_token=TOP_K,
            block_size=BLOCK_SIZE,
            total_tokens=tokens,
            tp_degree=1,
            moe_group=None,
        )
    )
    blocks = token_position_to_id.view(num_blocks, BLOCK_SIZE)
    conditions = torch.any(blocks != -1, dim=1).to(torch.int32)
    return (
        token_position_to_id.to(torch.int32),
        block_to_expert.to(torch.int32),
        conditions,
    )


@pytest.mark.parametrize("tokens", TOKEN_COUNTS)
def test_the_mapping_builder_matches_the_torch_route_at_every_token_count(
    tokens: int,
) -> None:
    """The public builder returns, and its mapping is the torch route's bit for bit."""
    _require_modes()
    affinities = _affinities(tokens)
    expert_mask = (affinities != 0).to(torch.float32)
    if not can_run_kernel(expert_mask):
        raise MissingSimulatorEnvError(
            "can_run_kernel answered False, so the predicate returns on its first "
            "line and this token count never reaches the arithmetic it exercises"
        )

    masked, token_position_to_id, block_to_expert, conditions = (
        moe_blockwise.build_blockwise_mapping(
            expert_affinities=affinities,
            num_local_experts=E_LOCAL,
            num_experts_per_token=TOP_K,
            block_size=BLOCK_SIZE,
            moe_group=None,
            tp_degree=1,
        )
    )
    expected_tpid, expected_bte, expected_conditions = _torch_route(
        tokens, expert_mask
    )
    assert token_position_to_id.dtype == expected_tpid.dtype
    assert block_to_expert.dtype == expected_bte.dtype
    assert conditions.dtype == expected_conditions.dtype
    #: Exact equality, not a tolerance: these are integer mappings.
    assert torch.equal(token_position_to_id, expected_tpid)
    assert torch.equal(block_to_expert, expected_bte)
    assert torch.equal(conditions, expected_conditions)
    assert torch.equal(
        masked, moe_blockwise._apply_padding_mask(affinities, None).view(-1, 1)
    )


@pytest.mark.parametrize("tokens", TOKEN_COUNTS)
def test_the_indexed_flatten_predicate_answers_at_every_token_count(
    tokens: int,
) -> None:
    """The predicate answers, and its answer is the one the kernel's rule gives."""
    _require_modes()
    expert_mask = (_affinities(tokens) != 0).to(torch.float32)
    if not can_run_kernel(expert_mask):
        raise MissingSimulatorEnvError(
            "can_run_kernel answered False, so the predicate returns on its first "
            "line and this token count compares nothing"
        )
    f_len = _caller_f_len(tokens)
    answer = moe_blockwise._can_use_indexed_flatten_kernel(
        T=tokens, tensor=expert_mask, f_len=f_len
    )
    assert answer is EXPECTED_INDEXED_FLATTEN[tokens], (
        f"tokens={tokens} f_len={f_len}: the predicate answered {answer}, "
        f"expected {EXPECTED_INDEXED_FLATTEN[tokens]}"
    )
