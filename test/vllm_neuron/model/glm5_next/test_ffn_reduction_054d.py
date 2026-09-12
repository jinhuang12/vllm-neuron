# SPDX-License-Identifier: Apache-2.0
"""Tier T acceptance for ``inc-glm53f-054d`` -- ONE row-parallel reduction at the FFN site.

Acceptance command (plan block ``#### inc-glm53f-054d``, CPU mode)::

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \
    python -m pytest test/vllm_neuron/model/glm5_next/test_ffn_reduction_054d.py \
      -q -s --timeout 1800 -p no:cacheprovider

WHAT THE INCREMENT FIXES. Every FFN weight family that is declared row-parallel is
declared so on its INTERMEDIATE width, so above world size 1 each rank returns a
PARTIAL SUM at the full output width -- and at the pin nothing summed them. The
file's only collective was the MLA ``o_proj`` reduction, so the dense MLP's result,
the shared expert's and the routed bank's all reached the residual add rank-local.

THE TWO ITEMS, AND WHY THERE ARE TWO
1. **the numeric item** -- two ranks, each holding its own intermediate-width shard
   of ``gate_proj``, ``up_proj`` and ``down_proj``, run through the PRODUCTION
   ``_ffn_half`` and reduced through the production site; rank 1's value compared
   to the unsharded output; the injected group's ``all_reduce`` count read as a
   number; and a REDUCTION-REMOVED control whose gap is printed.
2. **the vacuity item** -- a ``world_size == 1`` run is shown UNABLE to satisfy the
   criterion. The tiny suite's ``world_size != 1`` guard
   (``tiny/test_tiny_glm5next_forward.py:5178``) raises rather than measures, which
   is exactly why nothing detected the gap, so this file states in a counted way
   that a single-rank pass is not evidence for this increment.

THE TOLERANCE IS THE CAMPAIGN'S REGISTERED fp8 PAIR AND IS NOT RE-REGISTERED:
``rtol=3e-2, atol=1e-5``, order named inline per design law D3, the pair
``inc-glm53f-005`` registered and ``-025`` and ``-071`` already compare at. The
predicate is spelled out here rather than delegated, because
``_DEFAULT_DTYPE_TOLERANCE`` has NO fp8 entry and an omitted pair silently inherits
the bf16 one (PIT-13). The worst relative error is REPORTED as a number either way.

THE GROUP CONVENTION IS ``inc-glm53f-100``'s, cited and re-authored rather than
imported: ``test_mla_decode.py:1086``'s ``_CountedTwoRankGroup`` records rank 0's
partial and adds it back on rank 1, so two ranks meeting in one process reproduce
what the collective does on hardware. That file is another block's surface and is
not edited by this one.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from vllm_neuron.functional.blockwise_fp8_mm import SCALE_BLOCK_SIZE

#: THE DENSE CONSUMER'S BLOCK, IMPORTED (``inc-glm53f-112`` round 2, ruling 2). This
#: file used to type ``BLOCK_QUANT_SIZE = 256`` and build its grids at that number,
#: which the dense seam refuses since `-112`: the kernel indexes the checkpoint's own
#: tiles. It is imported rather than re-typed as 128 so this fixture follows the
#: kernel the next time that number moves, instead of going stale beside it.
DENSE_BLOCK = SCALE_BLOCK_SIZE
TILE_SIZE = 128

#: ``hidden_size % 256 == 0`` and, at the dense block, TWO scale-grid rows: the
#: smallest admissible contraction extent for the two parallel projections.
HIDDEN_SIZE = 256
#: FOUR ``DENSE_BLOCK`` columns, so each of two ranks holds two whole scale blocks
#: after the shard -- it was two columns and one block per rank while the grids were
#: built at 256. One column could not be split at all, and doubling the width again
#: would pay for the same reading twice.
INTERMEDIATE_SIZE = 512
#: A whole number of ``TILE_SIZE`` rows. The dense seam does not pad, so a
#: non-multiple would exercise its refusal instead of these numerics -- the gap
#: ``inc-glm53f-026b`` owns, and not this item's subject.
TOKENS = 128
WORLD = 2
SHARD_INTERMEDIATE = INTERMEDIATE_SIZE // WORLD

RTOL = 3e-2
ATOL = 1e-5

SEED_HIDDEN, SEED_GATE, SEED_UP, SEED_DOWN, SEED_GAIN = 5401, 5402, 5403, 5404, 5405

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "config.json"
FIXTURE_SHA256 = "5ed24d23a3e14a038352e1bdc21fd25fc90ff2291d3f6a310acf5d4036665a1d"


class VacuousControlError(AssertionError):
    """A control that cannot fail, which makes the item it guards meaningless."""


def say(tag: str, *values) -> None:
    """One printed line per read value. ``-s`` is in the acceptance command."""
    print("FFNRED|" + tag + "|" + "|".join(str(v) for v in values))


def _impl():
    """Import the modeling module INSIDE a test body, this package's convention."""
    from vllm_neuron.model.glm5_next import model_fp8

    return model_fp8


def _pinned_raw_config() -> dict:
    """The pinned fixture config, digest-checked, so nothing here is hand-fed."""
    raw = FIXTURE_PATH.read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != FIXTURE_SHA256:
        raise VacuousControlError(
            f"{FIXTURE_PATH} digests to {got}, not the campaign's registered "
            f"{FIXTURE_SHA256}; the quant config under test would not be the pin's"
        )
    return json.loads(raw.decode())


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
    """Values already on the fp8-e4m3 grid: multiples of ``1/8`` in ``[1/8, 7/8]``.

    Unsigned and bf16-exact, the tiny suite's primitive, so the seam's cast of the
    activations introduces nothing this item would then have to explain away.
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(1, 8, shape, generator=generator).to(torch.float32) / 8.0


def _pow2_scales(exponents: tuple[int, ...], rows: int) -> torch.Tensor:
    """A ``[rows, len(exponents)]`` fp32 grid of exact powers of two, per column.

    DISTINCT per column on purpose: a permuted or transposed scale layout cannot
    survive a comparison whose blocks differ by factors of four.
    """
    row = torch.tensor([float(2.0**e) for e in exponents], dtype=torch.float32)
    return row.repeat(rows, 1)


#: How many DENSE blocks one declared scale regime covers, DERIVED rather than typed.
#: The regimes are written one per RANK's share of the intermediate width, which is
#: what this item's two ranks divide.
_BLOCKS_PER_REGIME = SHARD_INTERMEDIATE // DENSE_BLOCK


def _per_regime(regimes: tuple[int, ...]) -> tuple[int, ...]:
    """Each declared regime repeated over the dense blocks it covers.

    WHY THE REGIMES REPEAT RATHER THAN MULTIPLY (``inc-glm53f-112`` round 2). Moving
    the grids from the producer's 256 to the dense kernel's 128 doubles the number of
    scale entries. Giving each new entry its own exponent would change the effective
    matrix and every reference number in this file with it; repeating the regime the
    entry sits inside leaves the dequantised weight BIT-IDENTICAL, so the bands and
    the comparators registered for these two items are untouched.

    WHAT IT GIVES UP, disclosed: :func:`_pow2_scales` says its exponents are distinct
    per column, and adjacent pairs are now equal, so a permutation WITHIN one former
    256 block would not be caught. A permutation across regimes still is, and so is a
    transpose, because the grid is not square.
    """
    return tuple(e for e in regimes for _ in range(_BLOCKS_PER_REGIME))


def _scale_grid_attribute(leaf: str) -> str:
    """The grid's attribute name, ASKED of its one definition rather than retyped."""
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
    """This rank's slice of the three FFN weights, on their DECLARED dimensions.

    ``gate`` and ``up`` are column-parallel on the intermediate width and ``down``
    is row-parallel on it, exactly as ``_SHARD_GEOMETRY`` declares for
    ``Glm5NextDenseMLP``. A scale grid answers with its weight's geometry, so each
    grid is sliced on the axis its weight was sliced on and at grid granularity.
    """
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
    """A ``Glm5NextModel`` stand-in that carries ONLY what ``_ffn_half`` reads.

    ``_ffn_half`` reads ``self._rms_norm`` and ``self.text_config`` and nothing
    else -- measured, not assumed -- so this binds the PRODUCTION ``_rms_norm`` to
    a bare instance and sets the config. The method under test and the norm it
    calls are both the shipped ones; only the construction is stood in for, which
    is what keeps this item about the reduction's PLACE in the chain.
    """
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
    """The PRODUCTION ``_ffn_half``, unbound, on the stand-in carrier."""
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
    """The injected coordinator: it COUNTS, and it really sums.

    ``inc-glm53f-100``'s object, re-authored here rather than imported from
    ``test_mla_decode.py``, so that file keeps its single writer. On the FIRST pass
    it records each partial and leaves the tensor alone; on the SECOND it adds the
    recorded partial back IN PLACE. Rank 1's returned value is therefore the fully
    reduced one, which is what rank 1 returns on hardware. ``world_size`` is a
    plain attribute because that is all the production code reads off the group
    besides ``all_reduce``.
    """

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
    """``(worst |got-ref| / (atol + rtol*|ref|), elements outside the pair)``.

    The predicate is spelled out rather than delegated because
    ``_DEFAULT_DTYPE_TOLERANCE`` has no fp8 entry and an omitted pair silently
    inherits the bf16 one. Order named inline: ``(rtol, atol) = (3e-2, 1e-5)``.
    """
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
# ITEM 1 -- the numeric item.                                                  #
# --------------------------------------------------------------------------- #
def test_054d_the_reduced_sharded_ffn_half_equals_the_unsharded_output(
    monkeypatch,
) -> None:
    """Two ranks' reduced FFN half equals the unsharded one, and reduces ONCE per rank."""
    model_fp8 = _impl()
    text_config = _text_config()
    operands = _whole_operands()

    # -- ARM 1: the unsharded reference, at world size 1, with the REAL guard read
    #    live. At world 1 the production line cannot reduce and imports no vllm
    #    symbol, which is what makes item 2 below a control rather than a hope.
    say("GUARD_AT_WORLD_1", model_fp8._resolve_tp_group())
    assert model_fp8._resolve_tp_group() is None
    whole = _dense_module(text_config, operands)
    reference = _run_half(_carrier(text_config), _layer(whole, operands["gain"]), operands["hidden"])
    say("REFERENCE", tuple(reference.shape), reference.dtype,
        float(reference.abs().max()), float(reference.abs().mean()))
    if float(reference.abs().max()) == 0.0:
        raise VacuousControlError(
            "the unsharded reference is all zeros; every arm below would agree with "
            "every other and the item could not fail"
        )

    # -- ARM 2: two ranks, each on its own intermediate shard, reduced at the site.
    group = _CountedTwoRankGroup()
    reduced = _run_two_ranks(monkeypatch, text_config, operands, group)
    say("REDUCE_CALLS", group.calls, "expected", WORLD,
        "shapes", set(group.shapes), "dtypes", set(group.dtypes))
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
    say("AGREEMENT", "worst_ratio", f"{worst:.6f}", "elements_outside", outside,
        "of", int(reference.numel()), "rtol", RTOL, "atol", ATOL)
    assert outside == 0, (
        f"{outside} of {reference.numel()} elements fall outside "
        f"(rtol={RTOL}, atol={ATOL}); worst ratio {worst:.6f}"
    )

    # -- THE REDUCTION-REMOVED CONTROL. The same two ranks with the group absent,
    #    which is the pin's behaviour, so the gap this increment closes is measured
    #    rather than asserted.
    bare = _CountedTwoRankGroup()
    unreduced = _run_two_ranks(monkeypatch, text_config, operands, bare, reduce=False)
    c_worst, c_outside = _worst_relative_error(unreduced, reference)
    gap = float((unreduced.to(torch.float32) - reference.to(torch.float32)).abs().max())
    say("CONTROL_REDUCTION_REMOVED", "reduce_calls", bare.calls, "worst_ratio",
        f"{c_worst:.6f}", "elements_outside", c_outside, "max_abs_gap", gap)
    assert bare.calls == 0
    if c_outside == 0:
        raise VacuousControlError(
            "the rank-local result already satisfies the criterion, so this item "
            "cannot tell a reduction from its absence"
        )


# --------------------------------------------------------------------------- #
# ITEM 2 -- the vacuity item.                                                  #
# --------------------------------------------------------------------------- #
def test_054d_a_single_rank_run_cannot_satisfy_this_criterion(monkeypatch) -> None:
    """A ``world_size == 1`` run is UNABLE to measure the reduction. Counted.

    The tiny suite refuses at ``world_size != 1`` rather than measuring, so a green
    single-rank suite is not evidence for this increment. Two counted readings say
    so: at world 1 the resolver hands back no group and the site cannot reduce, and
    a sharded module run at world 1 disagrees with the unsharded reference.
    """
    model_fp8 = _impl()
    text_config = _text_config()
    operands = _whole_operands()

    assert model_fp8._resolve_world_size() == 1
    assert model_fp8._resolve_tp_group() is None
    say("WORLD_1_HAS_NO_GROUP", model_fp8._resolve_world_size(),
        model_fp8._resolve_tp_group())

    whole = _dense_module(text_config, operands)
    reference = _run_half(_carrier(text_config), _layer(whole, operands["gain"]), operands["hidden"])

    # NOTHING IS INJECTED HERE ON PURPOSE. This arm is what a single-rank suite
    # actually runs: the real resolver, no group, one rank's shard. It disagrees
    # with the whole, so no single-rank pass can stand in for item 1.
    shard = _dense_module(text_config, _shard(operands, 0))
    rank_local = _run_half(_carrier(text_config), _layer(shard, operands["gain"]), operands["hidden"])
    worst, outside = _worst_relative_error(rank_local, reference)
    say("WORLD_1_SHARD_DISAGREES", "reduce_calls", 0, "worst_ratio",
        f"{worst:.6f}", "elements_outside", outside, "of", int(reference.numel()))
    if outside == 0:
        raise VacuousControlError(
            "one rank's shard already agrees with the whole, so this fixture's "
            "shard is not a shard and item 1 would pass without any reduction"
        )
