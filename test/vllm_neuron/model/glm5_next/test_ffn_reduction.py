# SPDX-License-Identifier: Apache-2.0
"""One row-parallel reduction at the feed-forward site.

The routed bank, the shared expert and the dense MLP combine in a single collective,
so the reduced sharded half has to equal the unsharded output.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from vllm_neuron.functional.blockwise_fp8_mm import SCALE_BLOCK_SIZE

DENSE_BLOCK = SCALE_BLOCK_SIZE
TILE_SIZE = 128

#: ``hidden_size % 256 == 0`` and, at the dense block, two scale-grid rows: the
#: smallest admissible contraction extent for the two parallel projections.
HIDDEN_SIZE = 256
#: four ``DENSE_BLOCK`` columns, so each of two ranks holds two whole scale blocks
#: after the shard -- it was two columns and one block per rank while the grids were
#: built at 256. One column could not be split at all, and doubling the width again
#: would pay for the same reading twice.
INTERMEDIATE_SIZE = 512
#: A whole number of ``TILE_SIZE`` rows. The dense seam does not pad, so a
#: non-multiple would exercise its refusal instead of these numerics -- the gap
#: the dense pad owns, and not this test's subject.
TOKENS = 128
WORLD = 2
SHARD_INTERMEDIATE = INTERMEDIATE_SIZE // WORLD

RTOL = 3e-2
ATOL = 1e-5

SEED_HIDDEN, SEED_GATE, SEED_UP, SEED_DOWN, SEED_GAIN = 5401, 5402, 5403, 5404, 5405

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "config.json"

class VacuousControlError(AssertionError):
    """A control that cannot fail, which makes the test it guards meaningless."""


def _impl():
    """Import the modeling module inside a test body, this package's convention."""
    from vllm_neuron.model.glm5_next import model_fp8

    return model_fp8


def _pinned_raw_config() -> dict:
    """The fixture config, so nothing here is hand-fed."""
    return json.loads(FIXTURE_PATH.read_bytes().decode())


def _quant_config():
    from vllm_neuron.model.glm5_next.config import Glm5NextConfig

    return _impl().Glm5NextQuantConfig.from_model_config(
        Glm5NextConfig.from_configs(_pinned_raw_config())
    )


def _text_config():
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    return Glm5NextTextConfig(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_key_value_heads=2,
    )


def _fp8_grid_values(seed: int, *shape: int) -> torch.Tensor:
    """Values already on the fp8-e4m3 grid: multiples of ``1/8`` in ``[1/8, 7/8]``. """
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(1, 8, shape, generator=generator).to(torch.float32) / 8.0


def _pow2_scales(exponents: tuple[int, ...], rows: int) -> torch.Tensor:
    """A ``[rows, len(exponents)]`` fp32 grid of exact powers of two, per column. """
    row = torch.tensor([float(2.0**e) for e in exponents], dtype=torch.float32)
    return row.repeat(rows, 1)


#: How many dense blocks one declared scale regime covers, derived rather than typed.
#: The regimes are written one per rank's share of the intermediate width, which is
#: what this test's two ranks divide.
_BLOCKS_PER_REGIME = SHARD_INTERMEDIATE // DENSE_BLOCK


def _per_regime(regimes: tuple[int, ...]) -> tuple[int, ...]:
    """Each declared regime repeated over the dense blocks it covers. """
    return tuple(e for e in regimes for _ in range(_BLOCKS_PER_REGIME))


def _scale_grid_attribute(leaf: str) -> str:
    """The grid's attribute name, asked of its one definition rather than retyped."""
    return _impl().Glm5NextForConditionalGeneration._sibling_scale_grid_name(leaf)


def _attach(module, leaf: str, weight: torch.Tensor, grid: torch.Tensor) -> None:
    """Bind one declared weight and the plain-attribute grid beside it."""
    setattr(module, leaf, torch.nn.Parameter(weight.clone(), requires_grad=False))
    setattr(module, _scale_grid_attribute(leaf), grid.clone())


def _whole_operands() -> dict:
    """The unsharded dense MLP's three weights and three public block-scale grids."""
    hidden = _fp8_grid_values(SEED_HIDDEN, TOKENS, HIDDEN_SIZE) * float(2.0**-3)
    k_parallel = HIDDEN_SIZE // DENSE_BLOCK
    k_down = INTERMEDIATE_SIZE // DENSE_BLOCK
    return {
        "hidden": hidden.to(torch.bfloat16),
        "gain": _fp8_grid_values(SEED_GAIN, HIDDEN_SIZE).to(torch.bfloat16),
        "gate_proj_weight": (
            _fp8_grid_values(SEED_GATE, HIDDEN_SIZE, INTERMEDIATE_SIZE),
            _pow2_scales(_per_regime((-1, 1)), k_parallel),
        ),
        "up_proj_weight": (
            _fp8_grid_values(SEED_UP, HIDDEN_SIZE, INTERMEDIATE_SIZE),
            _pow2_scales(_per_regime((1, -1)), k_parallel),
        ),
        "down_proj_weight": (
            _fp8_grid_values(SEED_DOWN, INTERMEDIATE_SIZE, HIDDEN_SIZE),
            _pow2_scales((-1,) * (HIDDEN_SIZE // DENSE_BLOCK), k_down),
        ),
    }


def _shard(operands: dict, rank: int) -> dict:
    """This rank's slice of the three FFN weights, on their declared dimensions. """
    lo, hi = rank * SHARD_INTERMEDIATE, (rank + 1) * SHARD_INTERMEDIATE
    glo, ghi = lo // DENSE_BLOCK, hi // DENSE_BLOCK
    out = {}
    for leaf in ("gate_proj_weight", "up_proj_weight"):
        weight, grid = operands[leaf]
        out[leaf] = (weight[:, lo:hi].clone(), grid[:, glo:ghi].clone())
    weight, grid = operands["down_proj_weight"]
    out["down_proj_weight"] = (weight[lo:hi, :].clone(), grid[glo:ghi, :].clone())
    return out


def _dense_module(text_config, operands: dict):
    """A production ``Glm5NextDenseMLP`` carrying the given three pairs."""
    module = _impl().Glm5NextDenseMLP(text_config)
    for leaf in ("gate_proj_weight", "up_proj_weight", "down_proj_weight"):
        _attach(module, leaf, *operands[leaf])
    return module


def _carrier(text_config):
    """A ``Glm5NextModel`` stand-in that carries only what ``_ffn_half`` reads. """
    import types

    model_fp8 = _impl()
    carrier = object.__new__(model_fp8.Glm5NextModel)
    object.__setattr__(carrier, "text_config", text_config)
    object.__setattr__(
        carrier, "_rms_norm", types.MethodType(model_fp8.Glm5NextModel._rms_norm, carrier)
    )
    return carrier


def _layer(mlp, gain: torch.Tensor):
    """A layer stand-in holding the two attributes ``_ffn_half`` reads off one."""
    layer = torch.nn.Module()
    layer.mlp = mlp
    layer.post_attention_layernorm_weight = torch.nn.Parameter(
        gain.clone(), requires_grad=False
    )
    layer.layer_idx = 0
    return layer


def _run_half(carrier, layer, hidden: torch.Tensor) -> torch.Tensor:
    """The production ``_ffn_half``, unbound, on the stand-in carrier."""
    return _impl().Glm5NextModel._ffn_half(
        carrier,
        layer,
        hidden,
        quant_config=_quant_config(),
        block_size=None,
        moe_group=None,
        tp_degree=WORLD,
        expert_parallel_rank=0,
    )


class _CountedTwoRankGroup:
    """The injected coordinator: it counts, and it really sums. """

    def __init__(self, world_size: int = WORLD) -> None:
        self.world_size = world_size
        self.calls = 0
        self.shapes: list[tuple[int, ...]] = []
        self.dtypes: list[torch.dtype] = []
        self.recording = True
        self._recorded: list[torch.Tensor] = []
        self._replayed = 0

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.shapes.append(tuple(tensor.shape))
        self.dtypes.append(tensor.dtype)
        if self.recording:
            self._recorded.append(tensor.detach().clone())
        else:
            tensor.add_(self._recorded[self._replayed])
            self._replayed += 1
        return tensor


def _worst_relative_error(got: torch.Tensor, ref: torch.Tensor) -> tuple[float, int]:
    """``(worst |got-ref| / (atol + rtol*|ref|), elements outside the pair)``. """
    a = got.to(torch.float32)
    b = ref.to(torch.float32)
    allowed = ATOL + RTOL * b.abs()
    ratio = (a - b).abs() / allowed
    return float(ratio.max()), int((ratio > 1.0).sum())


def _run_two_ranks(monkeypatch, text_config, operands, group, *, reduce: bool = True):
    """Both ranks in order, through the production ``_ffn_half``. Returns rank 1's."""
    model_fp8 = _impl()
    monkeypatch.setattr(model_fp8, "_resolve_world_size", lambda: WORLD)
    monkeypatch.setattr(
        model_fp8, "_resolve_tp_group", (lambda: group) if reduce else (lambda: None)
    )
    carrier = _carrier(text_config)
    last = None
    for rank in range(WORLD):
        group.recording = rank == 0
        module = _dense_module(text_config, _shard(operands, rank))
        layer = _layer(module, operands["gain"])
        last = _run_half(carrier, layer, operands["hidden"])
    return last


# --------------------------------------------------------------------------- #
# check 1 -- the numeric test.                                                  #
# --------------------------------------------------------------------------- #
def test_the_reduced_sharded_ffn_half_equals_the_unsharded_output(
    monkeypatch,
) -> None:
    """Two ranks' reduced FFN half equals the unsharded one, and reduces once per rank."""
    model_fp8 = _impl()
    text_config = _text_config()
    operands = _whole_operands()

    # -- arm 1: the unsharded reference, at world size 1, with the real guard read
    #    live. At world 1 the production line cannot reduce and imports no vllm
    #    symbol, which is what makes check-2 below a control rather than a hope.
    assert model_fp8._resolve_tp_group() is None
    whole = _dense_module(text_config, operands)
    reference = _run_half(_carrier(text_config), _layer(whole, operands["gain"]), operands["hidden"])
    if float(reference.abs().max()) == 0.0:
        raise VacuousControlError(
            "the unsharded reference is all zeros; every arm below would agree with "
            "every other and the test could not fail"
        )

    # -- arm 2: two ranks, each on its own intermediate shard, reduced at the site.
    group = _CountedTwoRankGroup()
    reduced = _run_two_ranks(monkeypatch, text_config, operands, group)
    assert group.calls == WORLD, (
        f"the FFN site reduced {group.calls} times across {WORLD} ranks; one "
        f"reduction per rank per layer is the criterion, and a reduction inside a "
        f"per-block loop or a second one in a helper would not read {WORLD}"
    )
    assert set(group.shapes) == {(TOKENS, HIDDEN_SIZE)}
    assert set(group.dtypes) == {torch.float32}, (
        f"the reduction ran on {set(group.dtypes)}; it must sum the seam's own "
        f"float32 BEFORE the cast, or each rank's fraction is rounded first"
    )

    worst, outside = _worst_relative_error(reduced, reference)
    assert outside == 0, (
        f"{outside} of {reference.numel()} elements fall outside "
        f"(rtol={RTOL}, atol={ATOL}); worst ratio {worst:.6f}"
    )

    bare = _CountedTwoRankGroup()
    unreduced = _run_two_ranks(monkeypatch, text_config, operands, bare, reduce=False)
    _c_worst, c_outside = _worst_relative_error(unreduced, reference)
    assert bare.calls == 0
    if c_outside == 0:
        raise VacuousControlError(
            "the rank-local result already satisfies the criterion, so this test "
            "cannot tell a reduction from its absence"
        )


# --------------------------------------------------------------------------- #
# check 2 -- the vacuity test.                                                  #
# --------------------------------------------------------------------------- #
