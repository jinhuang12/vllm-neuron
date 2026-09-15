# SPDX-License-Identifier: Apache-2.0
"""Tier N acceptance for `inc-glm53f-054g` -- the indexed-flatten predicate answers below 16 tokens.

Acceptance command (plan block, ``#### inc-glm53f-054g``)::

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
    pytest test/vllm_neuron/functional/moe/test_moe_blockwise_small_t_054g.py -s -rA

WHAT WENT WRONG, AND WHERE
--------------------------
``build_blockwise_mapping`` sizes the indexed-flatten kernel with
``f_len = min(128, total_tokens // 16)`` (``moe_blockwise.py:110``). A decode step carries ONE token,
so that expression is ``0``, and the routing predicate
``_can_use_indexed_flatten_kernel`` then evaluates ``T % f_len`` (``:536`` before this increment) and
raises ``ZeroDivisionError`` instead of answering. It was measured on a leased host, not guessed:
`increments/launch-054b-r5-trn2-1-20260909T230709Z.out`, where two items of the `-054b` acceptance
die on that line inside a decode step's MoE layer.

The sibling predicate already answers at that size -- ``_can_use_find_nonzero_indices_kernel``
returns False because ``chunk_size % 128 != 0`` -- and the caller already has a torch route for every
case the kernel cannot take (``_build_blockwise_mapping_torch``, ``:146``). So the fix is that the
predicate ANSWERS: one guard, ``if f_len < 1: return False``, ahead of the division.

WHY THIS IS NOT A KERNEL-CLASS CHANGE (P13)
-------------------------------------------
No new functionality is added on either route. The kernel route is untouched, the torch route is the
upstream design for small token counts, and the guard only decides which of two EXISTING routes runs.
Nothing here implements maths in torch that the NKI library would otherwise do.

HOW EACH CLAIM IS SETTLED
-------------------------
1. THE PRODUCT ANSWERS. At every token count in :data:`TOKEN_COUNTS` the public mapping builder
   returns rather than raising, and its four returned tensors equal the torch route's own -- the same
   private function called directly on the same inputs, compared with ``torch.equal``.
2. THE GUARD CHANGES NOTHING ABOVE THE FLOOR. For ``T >= 16`` the product predicate's answer must
   equal the PARENT predicate's answer, bit for bit, at the same inputs.
3. THE CONTROL IS ARMED. :func:`_parent_predicate` is the parent's own body, copied verbatim from
   ``b7b3d80e:vllm_neuron/functional/moe/moe_blockwise.py:527-545``. It must RAISE
   ``ZeroDivisionError`` at ``f_len == 0`` on the same call where the product returns ``False``. If
   the copy stopped raising, the control would pass while measuring nothing, so the raise is asserted
   rather than assumed.

NO TOLERANCE ANYWHERE. Every comparison in this file is exact: the mappings are integer tensors
compared with ``torch.equal``, and the predicate answers are booleans compared with ``is``. There is
no ``assert_close``, no ``rtol`` and no ``atol``, because nothing here is a numeric approximation.
"""

from __future__ import annotations

import os

import pytest
import torch

from vllm_neuron.functional.moe import moe_blockwise
from vllm_neuron.utils.neuron_utils import can_run_kernel

#: The shape the failing decode step actually had: 16 local experts, two per token.
E_LOCAL = 16
TOP_K = 2
BLOCK_SIZE = 128

#: Token counts on both sides of the floor, including the exact boundary and one past it.
TOKEN_COUNTS = [1, 2, 8, 15, 16, 17, 32, 126, 128]

#: The token count below which the caller's own expression makes ``f_len`` zero.
F_LEN_FLOOR = 16


class VacuousControlError(AssertionError):
    """A case that cannot measure what it claims. Raised rather than skipped, so it is read."""


def _require_modes() -> None:
    """Both variables must be in the PROCESS environment, not set from a fixture.

    ``can_run_kernel`` reads them at call time (``neuron_utils.py:18-23``): in CPU mode it answers
    True only when ``NKI_SIMULATOR=1``. Without that, the predicate returns False on its FIRST line
    and never reaches the division -- so every claim in this file would pass while measuring nothing.
    """
    for name in ("VLLM_NEURON_CPU_MODE", "NKI_SIMULATOR"):
        if os.environ.get(name) != "1":
            raise VacuousControlError(
                f"{name} must be 1 in the process environment; without it can_run_kernel answers "
                f"False and this file's claims are vacuous"
            )


def _caller_f_len(tokens: int) -> int:
    """The caller's own sizing expression, written once here and cited, not re-invented.

    ``moe_blockwise.py:110``: ``f_len = min(128, total_tokens // 16)``.
    """
    return min(128, tokens // 16)


def _affinities(tokens: int) -> torch.Tensor:
    """``[T, E_LOCAL]`` affinities with exactly ``TOP_K`` non-zero entries per row.

    DETERMINISTIC BY ARITHMETIC, not by a seed: token ``t`` selects experts ``t`` and ``t + 5``
    modulo ``E_LOCAL``, and the two weights are distinct so a mapping that confused them would show.
    A random fixture would make a failure hard to reproduce from the transcript alone.
    """
    affinities = torch.zeros(tokens, E_LOCAL, dtype=torch.float32)
    for position in range(tokens):
        affinities[position, position % E_LOCAL] = 0.75
        affinities[position, (position + 5) % E_LOCAL] = 0.25
    return affinities


def _parent_predicate(T: int, tensor: torch.Tensor, f_len: int) -> bool:
    """The predicate as it stood BEFORE this increment, copied verbatim.

    Source: ``b7b3d80e:vllm_neuron/functional/moe/moe_blockwise.py:527-545``. The body below is that
    function's body character for character, comments included, so the control measures the parent's
    behaviour rather than a paraphrase of it.
    """
    if not can_run_kernel(tensor):
        return False

    # T must be divisible by f_len
    if T % f_len != 0:
        return False

    # (T // f_len) must be divisible by 16
    if (T // f_len) % 16 != 0:
        return False

    return True


def _route_reading(tokens: int, expert_mask: torch.Tensor) -> dict:
    """Which route the builder will take, recomputed from the same two predicates it consults.

    Printed as a reading per case so a green run says WHICH path produced the mapping. Nothing is
    patched: these are the module's own predicates at the module's own inputs.
    """
    f_len = _caller_f_len(tokens)
    find_nonzero = moe_blockwise._can_use_find_nonzero_indices_kernel(
        T=tokens, E=E_LOCAL, chunk_size=min(tokens, 16384), tensor=expert_mask
    )
    indexed_flatten = moe_blockwise._can_use_indexed_flatten_kernel(
        T=tokens, tensor=expert_mask, f_len=f_len
    )
    return {
        "f_len": f_len,
        "find_nonzero": bool(find_nonzero),
        "indexed_flatten": bool(indexed_flatten),
        "kernel_flow": bool(find_nonzero and indexed_flatten),
    }


def _torch_route(tokens: int, expert_mask: torch.Tensor) -> tuple:
    """The torch route's own answer, and the ``conditions`` the caller derives from it.

    The caller's derivation is reproduced here from its own two lines (``moe_blockwise.py:157-159``)
    so the comparison covers everything the public builder returns, not only the two mappings.
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
    return (token_position_to_id.to(torch.int32), block_to_expert.to(torch.int32), conditions)


@pytest.mark.parametrize("tokens", TOKEN_COUNTS)
def test_the_mapping_builder_answers_at_every_token_count(tokens: int) -> None:
    """The public builder returns, and its mapping is the torch route's mapping bit for bit."""
    _require_modes()
    affinities = _affinities(tokens)
    expert_mask = (affinities != 0).to(torch.float32)
    if not can_run_kernel(expert_mask):
        raise VacuousControlError(
            "can_run_kernel answered False, so the predicate returns on its first line and this "
            "case never reaches the arithmetic it exists to measure"
        )
    route = _route_reading(tokens, expert_mask)
    print(
        f"MOE054G|route|tokens={tokens}|f_len={route['f_len']}"
        f"|find_nonzero={route['find_nonzero']}|indexed_flatten={route['indexed_flatten']}"
        f"|kernel_flow={route['kernel_flow']}"
    )
    if tokens < F_LEN_FLOOR and route["f_len"] != 0:
        raise VacuousControlError(
            f"tokens={tokens} was chosen because the caller's f_len is 0 there, but the expression "
            f"gives {route['f_len']}; this case no longer covers the defect"
        )

    #: The whole point: no exception. A raise here fails the item at this line.
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
    want_tpid, want_bte, want_conditions = _torch_route(tokens, expert_mask)
    print(
        f"MOE054G|mapping|tokens={tokens}|token_position_to_id={tuple(token_position_to_id.shape)}"
        f":{token_position_to_id.dtype}|block_to_expert={tuple(block_to_expert.shape)}"
        f":{block_to_expert.dtype}|conditions={tuple(conditions.shape)}:{conditions.dtype}"
        f"|masked={tuple(masked.shape)}:{masked.dtype}"
    )
    assert token_position_to_id.dtype == want_tpid.dtype
    assert block_to_expert.dtype == want_bte.dtype
    assert conditions.dtype == want_conditions.dtype
    #: Exact equality, not a tolerance: these are integer mappings.
    assert torch.equal(token_position_to_id, want_tpid)
    assert torch.equal(block_to_expert, want_bte)
    assert torch.equal(conditions, want_conditions)
    assert torch.equal(masked, moe_blockwise._apply_padding_mask(affinities, None).view(-1, 1))


@pytest.mark.parametrize("tokens", TOKEN_COUNTS)
def test_the_guard_answers_below_the_floor_and_changes_nothing_above_it(tokens: int) -> None:
    """False below 16 tokens; identical to the parent's answer at or above it."""
    _require_modes()
    expert_mask = (_affinities(tokens) != 0).to(torch.float32)
    if not can_run_kernel(expert_mask):
        raise VacuousControlError(
            "can_run_kernel answered False, so both predicates return on their first line and this "
            "case compares nothing"
        )
    f_len = _caller_f_len(tokens)
    answer = moe_blockwise._can_use_indexed_flatten_kernel(
        T=tokens, tensor=expert_mask, f_len=f_len
    )
    if tokens < F_LEN_FLOOR:
        print(f"MOE054G|predicate|tokens={tokens}|f_len={f_len}|product={answer}|parent=raises")
        assert answer is False
        with pytest.raises(ZeroDivisionError):
            _parent_predicate(T=tokens, tensor=expert_mask, f_len=f_len)
    else:
        parent = _parent_predicate(T=tokens, tensor=expert_mask, f_len=f_len)
        print(f"MOE054G|predicate|tokens={tokens}|f_len={f_len}|product={answer}|parent={parent}")
        assert answer is parent


def test_the_parent_predicate_raises_where_the_product_answers() -> None:
    """The must-fail control, on the one input that defines the defect: a single-token decode."""
    _require_modes()
    expert_mask = (_affinities(1) != 0).to(torch.float32)
    if not can_run_kernel(expert_mask):
        raise VacuousControlError(
            "can_run_kernel answered False, so the parent returns False instead of raising and this "
            "control cannot separate the two versions"
        )
    f_len = _caller_f_len(1)
    assert f_len == 0
    with pytest.raises(ZeroDivisionError) as raised:
        _parent_predicate(T=1, tensor=expert_mask, f_len=f_len)
    print(f"MOE054G|control|parent_raised={type(raised.value).__name__}|message={raised.value}")
    product = moe_blockwise._can_use_indexed_flatten_kernel(
        T=1, tensor=expert_mask, f_len=f_len
    )
    print(f"MOE054G|control|product_answered={product}|f_len={f_len}")
    assert product is False


def test_the_callers_own_sizing_expression_is_zero_below_the_floor() -> None:
    """The premise of every case above, read off the caller's expression rather than assumed."""
    readings = {tokens: _caller_f_len(tokens) for tokens in TOKEN_COUNTS}
    print(f"MOE054G|f_len_by_token_count|{readings}")
    assert all(value == 0 for tokens, value in readings.items() if tokens < F_LEN_FLOOR)
    assert all(value >= 1 for tokens, value in readings.items() if tokens >= F_LEN_FLOOR)
