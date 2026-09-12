# SPDX-License-Identifier: Apache-2.0
"""Acceptance -- the KDA attention output reduction across a row-parallel shard.

THE DECLARED ACCEPTANCE, the Tier N harness this package uses:

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \\
      NEURON_PLATFORM_TARGET_OVERRIDE=trn2 python -m pytest \\
      test/vllm_neuron/model/glm5_next/test_kda_reduction_038c.py -q -s \\
      --timeout 300 -p no:cacheprovider

``o_proj_weight`` is row-parallel, so each rank's projection is a partial sum and
the whole sum needs a collective. Four items, one test each:

* A01 -- two ranks holding different heads reduce to the unsharded reference.
* A02 -- the must-fail arm: neither rank's own partial equals that reference, so
  A01 cannot pass with a reduction that does nothing.
* A03 -- the collective is entered exactly once per forward, on an fp32 tensor of
  the caller's shape, and BEFORE the cast to the caller's dtype: the carrier is
  bf16 there, so the closing cast is not the identity and a reduction placed after
  it would record the narrow dtype. The same forward with no group returns
  different bytes, which is what makes A04's identity a fact about the rank count.
* A04 -- one rank changes nothing, with NOTHING replaced: the product's own
  resolver runs, answers ``None`` at this process's real world size, and two
  forwards return bit-identical tensors.
* A05 -- the declared shard extents of the three KDA per-head families are the
  widths the forward reads: the gate bias per channel, the decay and the beta
  projection per head.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "config.json"

#: KDA geometry, from ``config.py``'s ``linear_attn_config`` and checked below
#: against it rather than trusted.
DECLARED_KDA_NUM_HEADS = 64
DECLARED_KDA_HEAD_SIZE = 128
DECLARED_KDA_CONV_KERNEL_SIZE = 4

#: Two ranks, because two is what shows a cross-rank sum; the registered degree is
#: 64 and the arithmetic under test does not depend on which of them it is.
WORLD = 2

#: The tolerance ``test_kda_layer.py:142-143`` already uses for this comparison.
DECLARED_RTOL = 5e-3
DECLARED_ATOL = 1e-5

#: Enough tokens to cross a chunk boundary in the prefill route.
DECLARED_TOKENS = 17

SEED = 20260910


def _impl():
    """Import the implementation module INSIDE a test body, never at import."""
    from vllm_neuron.model.glm5_next import model_fp8

    return model_fp8


class _CountedTwoRankGroup:
    """The injected coordinator: it counts, and it really sums.

    Re-authored here rather than imported from ``test_ffn_reduction_054d.py``,
    whose own docstring gives the reason for the same choice: a shared object
    would make that file a dependency of this one and take away its single
    writer. On the FIRST pass it records the partial and leaves the tensor alone;
    on the SECOND it adds the recorded partial back in place, so the second
    caller's tensor is the fully reduced one.
    """

    def __init__(self, world_size: int = WORLD) -> None:
        self.world_size = world_size
        self.calls = 0
        self.shapes: list[tuple[int, ...]] = []
        self.dtypes: list[torch.dtype] = []
        self._recorded: list[torch.Tensor] = []

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.shapes.append(tuple(tensor.shape))
        self.dtypes.append(tensor.dtype)
        if not self._recorded:
            self._recorded.append(tensor.detach().clone())
        else:
            tensor.add_(self._recorded.pop())
        return tensor


class _DoublingGroup(_CountedTwoRankGroup):
    """A coordinator whose reduction is visible in one forward.

    ``_CountedTwoRankGroup`` leaves the FIRST caller's tensor alone -- it needs a
    second pass to add the recorded partial back -- so a single forward through it
    returns the same bytes as a forward with no group, and a contrast taken from it
    would read equal and mean nothing. This one sums this rank's partial with an
    identical peer's, which is the value two equal ranks would reduce to.
    """

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.shapes.append(tuple(tensor.shape))
        self.dtypes.append(tensor.dtype)
        tensor.add_(tensor)
        return tensor


def _full_weights(hidden: int, heads: int, head_dim: int, kernel: int) -> dict:
    """Random weights at unit-ish scale for the WHOLE head count."""
    width = heads * head_dim

    def rnd(*shape: int, scale: float) -> torch.Tensor:
        return (torch.randn(*shape) * scale).to(torch.float32)

    return {
        "q_proj_weight": rnd(width, hidden, scale=hidden**-0.5),
        "k_proj_weight": rnd(width, hidden, scale=hidden**-0.5),
        "v_proj_weight": rnd(width, hidden, scale=hidden**-0.5),
        "b_proj_weight": rnd(heads, hidden, scale=hidden**-0.5),
        "f_a_proj_weight": rnd(head_dim, hidden, scale=hidden**-0.5),
        "f_b_proj_weight": rnd(width, head_dim, scale=head_dim**-0.5),
        "g_a_proj_weight": rnd(head_dim, hidden, scale=hidden**-0.5),
        "g_b_proj_weight": rnd(width, head_dim, scale=head_dim**-0.5),
        "q_conv1d_weight": rnd(width, 1, kernel, scale=0.5),
        "k_conv1d_weight": rnd(width, 1, kernel, scale=0.5),
        "v_conv1d_weight": rnd(width, 1, kernel, scale=0.5),
        "o_norm_weight": 1.0 + rnd(head_dim, scale=0.05),
        "o_proj_weight": rnd(hidden, width, scale=width**-0.5),
        "A_log": rnd(heads, scale=0.3),
        "dt_bias": rnd(width, scale=0.3),
    }


#: ``dt_bias`` is read per CHANNEL by the forward -- ``model_fp8.py:4280`` reshapes
#: it flat and ``:4285`` takes ``[h * head_dim : (h + 1) * head_dim]`` per head --
#: and the landed builder at ``test_kda_layer.py:367`` allocates it at the head
#: width to match. The shard table declares that same width, and A05 is the item
#: that reads the table and says so. This case still takes its slice from the width
#: the forward indexes rather than from the table, so the reduction under test
#: stays measurable on a tree whose table disagrees with the forward.
_BY_HEAD_WIDTH = ("dt_bias",)
_REPLICATED = ("f_a_proj_weight", "g_a_proj_weight", "o_norm_weight")


def _rank_slice(module: nn.Module, leaf: str, tensor: torch.Tensor, rank: int):
    """This rank's shard of a full-width tensor, taken from the declared geometry."""
    if leaf in _REPLICATED:
        return tensor.clone()
    heads = int(module.num_kv_heads_per_rank)
    if leaf in _BY_HEAD_WIDTH:
        size, dim = heads * DECLARED_KDA_HEAD_SIZE, 0
    else:
        geometry = _impl()._shard_geometry_for(module, leaf, WORLD)
        assert geometry is not None, f"{leaf} resolved replicated at world {WORLD}"
        size, dim = int(geometry.shard_size), int(geometry.shard_dim)
    return tensor.narrow(dim, rank * size, size).contiguous().clone()


def _attach(module: nn.Module, weights: dict, rank: int | None) -> None:
    """Attach whole weights when ``rank`` is None, else this rank's shards."""
    for name, tensor in weights.items():
        value = tensor.clone() if rank is None else _rank_slice(module, name, tensor, rank)
        setattr(module, name, nn.Parameter(value, requires_grad=False))


def _zero_state(module: nn.Module) -> dict:
    """The two carriers this module declares, allocated from its own shapes."""
    return {
        "conv_state": torch.zeros(
            module.kda_conv_state_shape, dtype=module.kda_conv_state_dtype
        ),
        "recurrent_state": torch.zeros(
            module.kda_recurrent_state_shape, dtype=module.kda_recurrent_state_dtype
        ),
    }


@pytest.fixture(scope="module")
def case() -> SimpleNamespace:
    """Drive the unsharded reference once, then each of the two ranks once."""
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    text_config = Glm5NextTextConfig()
    linear = text_config.linear_attn_config
    assert linear["num_heads"] == DECLARED_KDA_NUM_HEADS
    assert linear["head_dim"] == DECLARED_KDA_HEAD_SIZE
    assert linear["short_conv_kernel_size"] == DECLARED_KDA_CONV_KERNEL_SIZE
    assert FIXTURE_PATH.is_file()

    hidden = int(text_config.hidden_size)
    torch.manual_seed(SEED)
    weights = _full_weights(
        hidden,
        DECLARED_KDA_NUM_HEADS,
        DECLARED_KDA_HEAD_SIZE,
        DECLARED_KDA_CONV_KERNEL_SIZE,
    )
    torch.manual_seed(SEED + 1)
    tokens = torch.randn(DECLARED_TOKENS, hidden, dtype=torch.float32)

    reference_module = _impl().Glm5NextKDAAttention(text_config, 1)
    _attach(reference_module, weights, None)
    reference = reference_module.forward(
        tokens.clone(), **_zero_state(reference_module), is_prefill=True
    )

    group = _CountedTwoRankGroup()
    partials: list[torch.Tensor] = []
    returned: list[torch.Tensor] = []
    calls_per_forward: list[int] = []
    # The resolver is restored in the finally, so nothing this fixture injects
    # outlives it -- a leaked group would silently reduce someone else's case.
    saved = _impl()._resolve_tp_group
    try:
        for rank in range(WORLD):
            for answer, sink in ((group, returned), (None, partials)):
                _impl()._resolve_tp_group = lambda a=answer: a
                module = _impl().Glm5NextKDAAttention(text_config, WORLD)
                _attach(module, weights, rank)
                before = group.calls
                sink.append(
                    module.forward(
                        tokens.clone(), **_zero_state(module), is_prefill=True
                    )
                )
                if answer is group:
                    calls_per_forward.append(group.calls - before)

        # THE NARROW-CARRIER DRIVE, one rank, for the placement claim. fp32 in
        # makes the closing cast the identity, so a reduction on either side of it
        # records the same dtype; bf16 in makes the two sides distinguishable.
        narrow_group = _DoublingGroup()
        narrow_tokens = tokens.to(torch.bfloat16)
        _impl()._resolve_tp_group = lambda: narrow_group
        narrow_module = _impl().Glm5NextKDAAttention(text_config, WORLD)
        _attach(narrow_module, weights, 0)
        narrow_returned = narrow_module.forward(
            narrow_tokens.clone(), **_zero_state(narrow_module), is_prefill=True
        )
        # The same rank with no group at all: the bytes must differ, or "nothing
        # moved" would be true of a reduction that never happened.
        _impl()._resolve_tp_group = lambda: None
        bare_module = _impl().Glm5NextKDAAttention(text_config, WORLD)
        _attach(bare_module, weights, 0)
        narrow_bare = bare_module.forward(
            narrow_tokens.clone(), **_zero_state(bare_module), is_prefill=True
        )
    finally:
        _impl()._resolve_tp_group = saved

    return SimpleNamespace(
        reference=reference,
        returned=returned,
        partials=partials,
        group=group,
        calls_per_forward=calls_per_forward,
        heads_per_rank=DECLARED_KDA_NUM_HEADS // WORLD,
        narrow_carrier=narrow_tokens.dtype,
        narrow_group=narrow_group,
        narrow_returned=narrow_returned,
        narrow_bare=narrow_bare,
    )




def test_a01_two_ranks_reduce_to_the_unsharded_reference(case) -> None:
    """The reduced output equals the whole-head reference."""
    print(f"A01|heads_per_rank={case.heads_per_rank}|shape={tuple(case.reference.shape)}")
    torch.testing.assert_close(
        case.returned[-1],
        case.reference,
        rtol=DECLARED_RTOL,
        atol=DECLARED_ATOL,
    )


def test_a02_no_rank_partial_equals_the_reference(case) -> None:
    """The must-fail arm: a reduction that did nothing would fail A01."""
    for rank, partial in enumerate(case.partials):
        gap = (partial - case.reference).abs().max().item()
        print(f"A02|rank={rank}|max_abs_gap_to_reference={gap:.6f}")
        assert gap > DECLARED_ATOL, (
            f"rank {rank}'s own partial already equals the reference, so A01 "
            "would pass without any reduction"
        )


def test_a03_the_collective_runs_once_per_forward_and_before_the_cast(case) -> None:
    """One entry per forward, on an fp32 tensor of the caller's shape, before the cast."""
    print(f"A03|calls_per_forward={case.calls_per_forward}|dtypes={case.group.dtypes}")
    assert case.calls_per_forward == [1] * WORLD
    assert case.group.dtypes == [torch.float32] * WORLD
    assert case.group.shapes == [tuple(case.reference.shape)] * WORLD
    # PLACEMENT, on a carrier the closing cast really narrows: the entry is fp32
    # and the returned tensor is bf16, so a reduction after the cast would have
    # recorded bf16 here and this reading would fail.
    print(f"A03|narrow_carrier={case.narrow_carrier}|entry_dtypes={case.narrow_group.dtypes}"
          f"|returned={case.narrow_returned.dtype}|calls={case.narrow_group.calls}")
    assert case.narrow_carrier is torch.bfloat16
    assert case.narrow_group.dtypes == [torch.float32]
    assert case.narrow_group.shapes == [tuple(case.narrow_returned.shape)]
    assert case.narrow_group.calls == 1
    assert case.narrow_returned.dtype is torch.bfloat16
    # THE CONTRAST A04's identity rests on: with no group the same rank returns
    # different bytes, so "nothing moved" is a statement about the rank count.
    moved = not torch.equal(case.narrow_returned, case.narrow_bare)
    print(f"A03|group_vs_no_group_differ={moved}")
    assert moved


def test_a04_one_rank_changes_nothing_with_nothing_replaced() -> None:
    """The product's own resolver runs, answers None here, and the bytes do not move.

    NOTHING IS PATCHED IN THIS ITEM, deliberately. Replacing the resolver with a
    stub that returns ``None`` would make the claim new code against itself; the
    resolver reads this process's own world size, which is one under this harness,
    and that is the condition being measured. That the identity below is a fact
    about the rank count rather than about the test is A03's contrast limb, which
    shows the same rank returning different bytes when a group IS present.
    """
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    impl = _impl()
    print(f"A04|world_size={impl._resolve_world_size()}|resolver={impl._resolve_tp_group()}")
    assert impl._resolve_world_size() == 1
    assert impl._resolve_tp_group() is None

    text_config = Glm5NextTextConfig()
    hidden = int(text_config.hidden_size)
    torch.manual_seed(SEED)
    weights = _full_weights(
        hidden,
        DECLARED_KDA_NUM_HEADS,
        DECLARED_KDA_HEAD_SIZE,
        DECLARED_KDA_CONV_KERNEL_SIZE,
    )
    torch.manual_seed(SEED + 1)
    tokens = torch.randn(DECLARED_TOKENS, hidden, dtype=torch.float32)

    outputs = []
    for _run in range(2):
        module = impl.Glm5NextKDAAttention(text_config, 1)
        _attach(module, weights, None)
        outputs.append(
            module.forward(tokens.clone(), **_zero_state(module), is_prefill=True)
        )
    print(f"A04|equal_bytes={torch.equal(outputs[0], outputs[1])}|dtype={outputs[0].dtype}")
    assert torch.equal(outputs[0], outputs[1])


def test_a05_the_declared_kda_extents_are_the_widths_the_forward_reads(monkeypatch) -> None:
    """The gate bias is declared per channel; the decay and beta rows stay per head."""
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    impl = _impl()
    module = impl.Glm5NextKDAAttention(Glm5NextTextConfig(), WORLD)
    heads = int(module.num_kv_heads_per_rank)
    channels = heads * DECLARED_KDA_HEAD_SIZE
    declared = {
        leaf: impl._shard_geometry_for(module, leaf, WORLD)
        for leaf in ("dt_bias", "A_log", "b_proj_weight")
    }
    extents = {leaf: None if g is None else int(g.shard_size) for leaf, g in declared.items()}
    print(f"A05|heads_per_rank={heads}|channels_per_rank={channels}|declared={extents}")
    assert extents["dt_bias"] == channels
    assert extents["A_log"] == heads
    assert extents["b_proj_weight"] == heads

    # The slice the forward actually reads: each rank must get the channels of its
    # own heads, and the value at every position says which head it came from.
    full = torch.arange(
        DECLARED_KDA_NUM_HEADS * DECLARED_KDA_HEAD_SIZE, dtype=torch.float32
    )
    size = extents["dt_bias"]
    for rank in range(WORLD):
        shard = full.narrow(0, rank * size, size)
        first = rank * heads
        expected = full.narrow(
            0, first * DECLARED_KDA_HEAD_SIZE, heads * DECLARED_KDA_HEAD_SIZE
        )
        print(f"A05|rank={rank}|shard={tuple(shard.shape)}|first_head={first}")
        assert torch.equal(shard, expected)

    # THE MUST-FAIL ARM: put the pin's own row back and read the table again. It
    # returns the head COUNT, and the narrow it authorises hands one rank a
    # fraction of a single head's channels -- so the reading above is a statement
    # about this row and not about the reader.
    monkeypatch.setitem(
        impl._SHARD_GEOMETRY["Glm5NextKDAAttention"],
        "dt_bias",
        impl._DeclaredShard(0, impl._kda_head_count, "the pin's declaration"),
    )
    stale = impl._shard_geometry_for(module, "dt_bias", WORLD)
    stale_size = int(stale.shard_size)
    stale_shard = full.narrow(0, 0, stale_size)
    print(f"A05|stale_extent={stale_size}|stale_shard={tuple(stale_shard.shape)}")
    assert stale_size == heads
    assert stale_shard.numel() < DECLARED_KDA_HEAD_SIZE
