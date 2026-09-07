# SPDX-License-Identifier: Apache-2.0
"""Acceptance for `inc-glm53f-054a` -- the seven forwards of the 45-layer text model.

**SEVEN ITEMS, ONE PER REPLACED ``forward``, and no ``parametrize`` decorator in this
file** (campaign rule D1.2). Each item runs one forward once on the tiny config and
compares its output against a torch reference built from the same weights, at
``assert_close(rtol=1e-2, atol=1e-5)``. Each item names the component whose behaviour
it certifies (D1.4).

THE TOLERANCE AND THE CONSTRAINT SET ARE ADOPTED, NOT MINTED HERE (P9). ``rtol=1e-2``,
``atol=1e-5`` and the constraint set below are the end-to-end criterion's own numbers,
carried unchanged.

THE READINGS ARE PIPE-DELIMITED AND PREFIXED ``TINYFWD|``. That prefix is this file's
own: `-088`'s readings are bracketed, `-089`'s carry ``PRODUCTION|`` and `-090`'s carry
``SCALEPREP|``, so an extractor written for any of those reads zero here and a round
that predicted zero would pass while proving nothing.

WHY THE REFERENCE IS NOT THE FORK'S OWN ORACLE. Each item dequantises the fp8 weights
with its own ``repeat_interleave`` broadcast and contracts in one fp32 matmul, rather
than calling ``blockwise_fp8_mm_torch_oracle``. The two would agree by construction on
a shared misreading of the block-to-scale assignment; an independent broadcast cannot.
This is ``test_moe_path.py:677-699``'s reason, adopted.

WHY THE FIXTURE VALUES ARE CONDITIONED THE WAY THEY ARE. The weights are UNSIGNED
multiples of ``1/8``, which are exact in fp8-e4m3 and in bf16, and the block scales are
exact powers of two. With signed weights every dot product over the contraction is a
near-cancelling sum, so reference elements land arbitrarily close to zero while the
terms that built them are large, and a pointwise RELATIVE tolerance is then dominated
by cancellation rather than by kernel error -- ``inc-glm53f-025`` measured exactly that.
The one deliberate exception is documented at :func:`_dense_operands`: a single weight
block is negated so the SwiGLU's lower clamp is exercised, and it is a WHOLE block, so
every term in the affected columns shares a sign and nothing cancels.

EVERY CLAMP THIS INCREMENT ADDS CARRIES ITS OWN DISCRIMINATION CONTROL, and the control
is MEASURED rather than argued. For each clamp branch the reference declares, the item
recomputes the whole reference with that branch removed and asserts the result falls
OUTSIDE the same tolerance band the item passes inside. So a fixture that grew too
weak to separate the clamped function from the unclamped one fails as a control instead
of passing as an item. Reasoning about magnitudes was tried first and rejected: the
first draft of this file put the negated block on a small output scale, where deleting
the lower clamp moved the result by about 0.9% against a 1% tolerance -- it would have
passed with the branch missing.

EVERY ITEM ALSO CARRIES THE REGISTERED ROUTE PREDICATE, D13 form R-3, and it is the
reason a green item means anything. Each forward completes, and matches a torch
reference, whether or not one NKI kernel ran -- so each item reads the seam counters
around its own forward call and asserts that its own seam dispatched the number of times
it declares, that the torch-fallback total across every seam this campaign owns is 0, and
that the set of seams which fired is non-empty. An item that ran torch end to end fails
twice over: 0 dispatches on its own seam and a non-zero fallback.

THE FIRED SET IS ASSERTED PER ITEM RATHER THAN ACCUMULATED ACROSS THE SEVEN, because
``pytestmark`` below carries ``pytest.mark.forked``: every item runs in its own forked
process, so module-level state cannot survive from one item to the next and accumulating
would assert nothing. The union over the seven is read from this run's transcript, which
is what the registered wording's "reported" asks for.

Run::

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 \\
    python -m pytest test/vllm_neuron/model/glm5_next/tiny/test_tiny_glm5next_forward.py -q

Expected: exactly 7 collected, 7 passed, 0 failed, exit 0.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest
import torch

from vllm_neuron.functional.moe.blockwise_fp8_retile import BLOCK_QUANT_SIZE, TILE_SIZE

pytestmark = [pytest.mark.fast, pytest.mark.forked]

_FP8 = torch.float8_e4m3fn

# --------------------------------------------------------------------------- #
# The pinned checkpoint fixture, and its registered digest. Checked before the  #
# file is parsed, on ``test_experts.py:1266-1288``'s landed idiom: the          #
# quantisation policy these call sites route on must be the campaign's          #
# registered one and not a value this test invented.                            #
# --------------------------------------------------------------------------- #
FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "config.json"
FIXTURE_SHA256 = "5ed24d23a3e14a038352e1bdc21fd25fc90ff2291d3f6a310acf5d4036665a1d"

# --------------------------------------------------------------------------- #
# THE TINY-CONFIG CONSTRAINT SET, adopted from the end-to-end criterion.        #
# --------------------------------------------------------------------------- #
#: ``hidden_size % 256 == 0``. It is also the contraction extent of the two
#: parallel projections, and 256 is exactly ONE ``BLOCK_QUANT_SIZE`` row of scale
#: grid -- the smallest admissible value rather than a round number.
HIDDEN_SIZE = 256
#: ``intermediate_size >= 512``. FOUR ``BLOCK_QUANT_SIZE`` columns rather than the
#: minimum two, because the clamp controls need blocks above the bound, inside it
#: and below its negation, and two blocks cannot carry three regimes.
INTERMEDIATE_SIZE = 1024
#: ``num_key_value_heads = 2`` and ``head_dim <= 128``, carried for the attention
#: items further down this file; the MLP items do not read them.
NUM_KEY_VALUE_HEADS = 2
MAX_HEAD_DIM = 128
#: < 10 M parameters. Asserted per item against the module actually built, so the
#: bound is measured rather than declared.
MAX_PARAMETERS = 10_000_000

#: A whole number of ``TILE_SIZE`` rows -- the dense seam tiles ``M`` over the PSUM
#: partition axis and does not pad (``blockwise_fp8_mm.py:239-245``), so padding is
#: the caller's, and a tiny case that ignored it would exercise the refusal rather
#: than the numerics.
TOKENS = 128

RTOL = 1e-2
ATOL = 1e-5

#: Distinct per operand, so a swapped operand cannot pass on a shared seed.
SEED_HIDDEN = 5401
SEED_GATE = 5402
SEED_UP = 5403
SEED_DOWN = 5404

#: The hidden states are scaled DOWN by this power of two so the two projections
#: land either side of the checkpoint's SwiGLU bound rather than all above it.
#: Every value stays an exact binary fraction, so nothing is lost to the cast.
HIDDEN_SCALE_EXPONENT = -3


class VacuousControlError(AssertionError):
    """A precondition this file's own fixture must satisfy did not hold.

    Raised rather than asserted plainly so a fixture that stopped exercising the
    behaviour under test says so by name, instead of passing while proving
    nothing.
    """


class ReferenceShapeError(AssertionError):
    """A reference was handed operands it cannot combine."""


def _impl():
    """Import the modeling module INSIDE a test body, never at import time.

    This package's uniform convention (``test_experts.py:110-120``,
    ``test_shared_expert_scale_prep.py:71-84``). The functional imports at the top
    of this file are NOT this module and stay there.
    """
    from vllm_neuron.model.glm5_next import model_fp8

    return model_fp8


# --------------------------------------------------------------------------- #
# THE REGISTERED ROUTE PREDICATE, D13 form R-3, registered at plan §4b.2.       #
# --------------------------------------------------------------------------- #
#: The seams this campaign owns that these seven forwards reach, keyed by the
#: kernel entry point a reader would grep for, and valued by the FULL DOTTED PATH
#: of the module that owns the accessors. The path is spelled out because
#: :func:`_seam_modules` resolves it with :func:`importlib.import_module`, and the
#: docstring there is where the reason lives.
_SEAM_MODULE_PATHS = {
    "blockwise_fp8_mm": "vllm_neuron.functional.blockwise_fp8_mm",
    "blockwise_fp8_moe": "vllm_neuron.functional.moe.moe_blockwise_fp8",
}
#: Derived rather than written a second time: a seam listed in one and missing
#: from the other would make the route predicate iterate a name nothing resolves.
_SEAMS = tuple(_SEAM_MODULE_PATHS)


def _seam_modules() -> dict:
    """``{seam: module}`` for the seams this campaign owns.

    Resolved inside a call, like :func:`_impl`, so this file's import time does
    not depend on the plugin being importable.

    WHY ``import_module`` AND NOT A FROM-IMPORT -- load-bearing, not stylistic
    (``inc-glm53f-054a``, review finding M1). ``functional/__init__.py:8`` reads
    ``from .blockwise_fp8_mm import blockwise_fp8_mm``: it re-exports the seam
    FUNCTION under the name of the submodule that defines it. The import
    machinery sets that submodule as an attribute of the package first, and the
    statement then overwrites it. So ``from vllm_neuron.functional import
    blockwise_fp8_mm`` binds a function, and every accessor read below raises
    ``AttributeError`` on the first line of :func:`_reset_seam_counters` -- which
    is the first line of every item's route predicate, before any forward runs.

    ``import x.y as z`` is not a fix. That form performs the same attribute
    lookup on the parent package. Only ``importlib.import_module``, or a
    ``sys.modules`` read, hands back the module itself.

    THE MOE SEAM IS RESOLVED THE SAME WAY THOUGH IT DOES NOT COLLIDE TODAY. Its
    submodule is ``moe_blockwise_fp8`` and its export ``blockwise_fp8_moe``, two
    different words, so that attribute survives. But that is an accident of
    naming which a rename would take away in silence, and six of the parent
    package's twenty-one submodules are already shadowed this way -- the form is
    the package's convention rather than one slip to route around.

    The previous form's ALIASING was not the mistake, and its reason still holds:
    both seam modules spell their accessors identically, so a bare ``from ...
    import dispatch_counters`` would resolve to whichever module was imported
    last. Naming the module is what avoids that collision. An alias simply was
    not enough to make the bound object a module.
    """
    return {name: importlib.import_module(path)
            for name, path in _SEAM_MODULE_PATHS.items()}


def _read_seam_counters() -> dict:
    """``{seam: (nki_dispatch, torch_fallback)}`` as each seam reports itself."""
    return {name: tuple(mod.dispatch_counters())
            for name, mod in _seam_modules().items()}


def _reset_seam_counters() -> None:
    """Zero every seam this campaign owns, so a reading is this item's own."""
    for mod in _seam_modules().values():
        mod.reset_dispatch_counters()


def _assert_route_predicate(item: str, expected: dict, before: dict, after: dict) -> None:
    """The registered predicate, over one item's own forward call.

    ``expected`` is ``{seam: dispatches}`` and names ONLY the seams this item's
    forward should reach, so a forward that quietly took a different seam fails
    on the comparison rather than passing on a total.

    Three conjuncts, each from the registered wording:

    1. ``can_run_kernel()`` is True. Under ``VLLM_NEURON_CPU_MODE=1`` this reads
       the ``NKI_SIMULATOR`` flag (``utils/neuron_utils.py:20-21``), so a run
       launched without the simulator is refused here instead of passing on the
       torch oracle.
    2. the torch-fallback total across every seam reads exactly 0.
    3. the set of seams that fired is non-empty, and equals ``expected``.

    Raises:
        VacuousControlError: on any conjunct, named so the transcript says which.
    """
    from vllm_neuron.utils.neuron_utils import can_run_kernel

    gate = bool(can_run_kernel(torch.zeros(1)))
    fired = {}
    fallbacks = 0
    for name in _SEAMS:
        dispatched = after[name][0] - before[name][0]
        fell_back = after[name][1] - before[name][1]
        fallbacks += fell_back
        if dispatched:
            fired[name] = dispatched
    print(
        f"TINYFWD|route|item={item}|can_run_kernel={gate}"
        f"|fired={sorted(fired.items())}|expected={sorted(expected.items())}"
        f"|torch_fallback={fallbacks}"
    )
    if not gate:
        raise VacuousControlError(
            f"item {item}: can_run_kernel() is False, so this forward ran on the "
            f"torch oracle rather than the NKI route. The comparison would pass "
            f"either way, which is what this predicate exists to refuse. Run with "
            f"VLLM_NEURON_CPU_MODE=1 and NKI_SIMULATOR=1"
        )
    if fallbacks != 0:
        raise VacuousControlError(
            f"item {item}: the torch-fallback counters total {fallbacks} across "
            f"{list(_SEAMS)}; the registered predicate declares exactly 0"
        )
    if not fired:
        raise VacuousControlError(
            f"item {item}: no seam this campaign owns dispatched at all, so this "
            f"item certifies torch composed end to end and not the kernel"
        )
    if fired != expected:
        raise VacuousControlError(
            f"item {item}: the seams that fired were {sorted(fired.items())} and "
            f"this item declares {sorted(expected.items())}"
        )


def _pinned_raw_config() -> dict:
    """The pinned checkpoint config, digest-verified before it is parsed."""
    digest = hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest()
    if digest != FIXTURE_SHA256:
        raise VacuousControlError(
            f"pinned fixture digest moved: {digest} != {FIXTURE_SHA256}. The "
            f"quantisation policy these call sites route on would no longer be "
            f"the campaign's registered one."
        )
    return json.loads(FIXTURE_PATH.read_text())


def _quant_config():
    """``Glm5NextQuantConfig`` for the pinned checkpoint -- nothing hand-fed."""
    from vllm_neuron.model.glm5_next.config import Glm5NextConfig

    return _impl().Glm5NextQuantConfig.from_model_config(
        Glm5NextConfig.from_configs(_pinned_raw_config())
    )


def _tiny_text_config():
    """The tiny text config, every field inside the adopted constraint set."""
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    return Glm5NextTextConfig(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_key_value_heads=NUM_KEY_VALUE_HEADS,
    )


# --------------------------------------------------------------------------- #
# The fixture primitives.                                                      #
# --------------------------------------------------------------------------- #
def _fp8_grid_values(seed: int, *shape: int) -> torch.Tensor:
    """Values already on the fp8-e4m3 grid: multiples of ``1/8`` in ``[1/8, 7/8]``.

    UNSIGNED, for the conditioning reason this module's docstring records. Every
    value is also exact in bf16, so the seam's cast of the activations introduces
    nothing of its own.
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(1, 8, shape, generator=generator).to(torch.float32) / 8.0


def _pow2_scales(exponents: tuple[int, ...], rows: int) -> torch.Tensor:
    """A ``[rows, len(exponents)]`` fp32 grid of exact powers of two.

    EXPLICIT rather than random, and that is the point: each item states which
    blocks its projections put above and below the SwiGLU bound, and a random draw
    could put them all on one side while still passing its own seed check. The
    exponents are DISTINCT per column so a permuted or transposed scale layout
    cannot survive the comparison.
    """
    row = torch.tensor([float(2.0**e) for e in exponents], dtype=torch.float32)
    return row.repeat(rows, 1)


def _dequantise(weight_fp8: torch.Tensor, block_scale: torch.Tensor) -> torch.Tensor:
    """``weight[k, n] * scale[k // 256, n // 256]``, expanded, in fp32.

    The block scale is broadcast by ``repeat_interleave`` on both axes rather than
    by an index computation, so this repeats none of the bridge's arithmetic and
    cannot share an off-by-one with it (``test_moe_path.py:677-699``'s reason).
    """
    if weight_fp8.dim() != 2 or block_scale.dim() != 2:
        raise ReferenceShapeError(
            f"expected 2-D weight and 2-D block scale, got "
            f"{tuple(weight_fp8.shape)} and {tuple(block_scale.shape)}"
        )
    rows, cols = weight_fp8.shape
    want = (rows // BLOCK_QUANT_SIZE, cols // BLOCK_QUANT_SIZE)
    if tuple(block_scale.shape) != want:
        raise ReferenceShapeError(
            f"block scale {tuple(block_scale.shape)} does not tile a "
            f"{rows}x{cols} weight at granularity {BLOCK_QUANT_SIZE}"
        )
    expanded = block_scale.repeat_interleave(
        BLOCK_QUANT_SIZE, dim=0
    ).repeat_interleave(BLOCK_QUANT_SIZE, dim=1)
    return weight_fp8.to(torch.float32) * expanded


def _scale_grid_attribute(leaf: str) -> str:
    """The attribute a weight leaf's scale grid arrives under -- ASKED, not retyped.

    The rule strips ``_weight`` before it appends, so ``gate_proj_weight`` names
    ``gate_proj_weight_scale_inv``. It has ONE definition,
    :meth:`Glm5NextForConditionalGeneration._sibling_scale_grid_name`, and this
    asks that definition -- ``test_load_weights.py:2470``'s repair, adopted, for
    the same reason it gives: a second copy of the naming rule is the drift.
    """
    return _impl().Glm5NextForConditionalGeneration._sibling_scale_grid_name(leaf)


def _attach(module, leaf: str, weight: torch.Tensor, grid: torch.Tensor) -> None:
    """Bind one declared weight and the plain-attribute grid beside it.

    ``nn.Parameter(..., requires_grad=False)`` because the leaf was declared with
    ``register_parameter(name, None)`` and torch refuses a plain tensor there
    (``test_kda_layer.py:412``'s landed form). The grid is a PLAIN attribute and
    not a parameter, which is the arrangement ``_scale_prep_leaves`` documents.
    """
    setattr(module, leaf, torch.nn.Parameter(weight.clone(), requires_grad=False))
    setattr(module, _scale_grid_attribute(leaf), grid.clone())


def _dense_operands() -> dict:
    """The dense MLP's three weights and three public block-scale grids.

    WHAT THE SCALE EXPONENTS ARE FOR. The two parallel projections must straddle
    the checkpoint's SwiGLU bound, or the clamp this increment adds is invisible.
    With unsigned ``1/8``-grid weights and hidden states scaled by ``2**-3``, one
    column block sums to about ``8 * scale`` before the clamp -- a tight
    distribution, because a 256-term sum of bounded independent products has a
    standard deviation of only a few percent of its mean. Against a bound of
    ``10`` the exponents put the four column blocks near:

      gate  ``2**-3, 2**-1, 2**1, 2**3``  ->  1, 4, 16, 64   (two below, two above)
      up    ``2**1, 2**3, 2**-3, 2**1``   ->  16, 64, 1, -16 (see the negation)

    THE NEGATED BLOCK. ``up``'s last column block has its weights negated, so that
    block sums to about ``-16`` and is clamped by the reference's LOWER bound.
    Without it every value on this fixture is positive, the lower bound never
    binds, and a one-sided clamp on ``up`` would be indistinguishable from the
    reference's two-sided one. It is a WHOLE block, so every term in those columns
    shares a sign and the sum does not cancel.

      down  ``2**-3, 2**-3, 2**-3, 2**3``  ->  the LAST row block dominates the
            output by 64x, and it is the block the lower clamp acts on. The first
            draft of this fixture dominated with the first block instead, which
            left the lower clamp contributing under 1% of the output -- inside the
            adopted tolerance, so the item would have passed with that branch
            deleted. The controls in the item measure this rather than trusting
            the arrangement.
    """
    blocks = INTERMEDIATE_SIZE // BLOCK_QUANT_SIZE
    if blocks != 4:
        raise VacuousControlError(
            f"this fixture's exponent choices are written for 4 column blocks; "
            f"INTERMEDIATE_SIZE={INTERMEDIATE_SIZE} gives {blocks}"
        )
    if TOKENS % TILE_SIZE:
        raise VacuousControlError(
            f"TOKENS={TOKENS} is not a whole number of TILE_SIZE={TILE_SIZE} rows; "
            f"the dense seam would refuse this geometry before any numerics ran"
        )

    hidden = _fp8_grid_values(SEED_HIDDEN, TOKENS, HIDDEN_SIZE) * float(
        2.0**HIDDEN_SCALE_EXPONENT
    )

    gate_w = _fp8_grid_values(SEED_GATE, HIDDEN_SIZE, INTERMEDIATE_SIZE)
    up_w = _fp8_grid_values(SEED_UP, HIDDEN_SIZE, INTERMEDIATE_SIZE)
    up_w[:, -BLOCK_QUANT_SIZE:] = -up_w[:, -BLOCK_QUANT_SIZE:]
    down_w = _fp8_grid_values(SEED_DOWN, INTERMEDIATE_SIZE, HIDDEN_SIZE)

    k_blocks_parallel = HIDDEN_SIZE // BLOCK_QUANT_SIZE
    k_blocks_down = INTERMEDIATE_SIZE // BLOCK_QUANT_SIZE
    gate_s = _pow2_scales((-3, -1, 1, 3), k_blocks_parallel)
    up_s = _pow2_scales((1, 3, -3, 1), k_blocks_parallel)
    # ``down`` is ``[I, H]``, so its grid is ``[I//256, H//256]`` -- FOUR rows and
    # ONE column, which is why it is built column-wise and then filled per row.
    down_s = _pow2_scales((-3,), k_blocks_down)
    down_s[-1, :] = float(2.0**3)

    return {
        "hidden": hidden.to(torch.bfloat16),
        "gate_proj_weight": (gate_w.to(_FP8), gate_s),
        "up_proj_weight": (up_w.to(_FP8), up_s),
        "down_proj_weight": (down_w.to(_FP8), down_s),
    }


def _dense_output(
    operands: dict,
    gate_max: float | None,
    up_min: float | None,
    up_max: float | None,
) -> dict:
    """The dense MLP in torch, in fp32, with the clamp branches switchable.

    ``down(silu(clamp(gate)) * clamp(up))``. The reference's own clamps are
    ``modeling_glm5_next.py:102`` (gate, upper bound only) and ``:103`` (up, both
    bounds), applied to the PROJECTION OUTPUTS and before the product -- which is
    why a caller holding only the final output could not apply them.

    The bounds are ARGUMENTS so the item can rebuild this with one branch removed
    and prove the branch matters. Passing ``None`` for a bound omits it, which is
    also how ``:102`` expresses its missing lower bound.
    """
    from torch.nn.functional import silu

    x = operands["hidden"].to(torch.float32)
    gate = x @ _dequantise(*operands["gate_proj_weight"])
    up = x @ _dequantise(*operands["up_proj_weight"])

    gate_c = gate.clamp(min=None, max=gate_max)
    up_c = up.clamp(min=up_min, max=up_max)
    activated = silu(gate_c) * up_c

    # The module casts the activation back to the caller's dtype before the down
    # projection, because the seam's declared input dtype is bf16
    # (``blockwise_fp8_mm.py:446``). The reference casts identically -- skipping it
    # would compare against a function the shipped path does not compute.
    down = activated.to(torch.bfloat16).to(torch.float32) @ _dequantise(
        *operands["down_proj_weight"]
    )
    return {"out": down, "gate": gate, "up": up}


# --------------------------------------------------------------------------- #
# ITEM 1 of 7 -- ``Glm5NextDenseMLP.forward``.                                 #
# Certifying component: ``model_fp8.Glm5NextDenseMLP.forward``.                 #
# --------------------------------------------------------------------------- #
def test_tiny_dense_mlp_forward_matches_the_reference() -> None:
    """The dense MLP on one layer, against the reference's clamped SwiGLU.

    ``inc-glm53f-054a`` item 1 of 7. THREE dispatches: gate, up, down.
    """
    model_fp8 = _impl()
    text_config = _tiny_text_config()
    module = model_fp8.Glm5NextDenseMLP(text_config)

    limit = float(module.swiglu_limit)
    if limit != float(text_config.swiglu_limit):
        raise VacuousControlError(
            f"the module resolved swiglu_limit={limit} but the config declares "
            f"{text_config.swiglu_limit}; the bound under test would not be the "
            f"checkpoint's"
        )

    operands = _dense_operands()
    for leaf in ("gate_proj_weight", "up_proj_weight", "down_proj_weight"):
        _attach(module, leaf, *operands[leaf])

    parameters = sum(int(p.numel()) for p in module.parameters() if p is not None)
    if parameters >= MAX_PARAMETERS:
        raise VacuousControlError(
            f"the tiny dense MLP holds {parameters} parameters, at or above the "
            f"adopted bound of {MAX_PARAMETERS}"
        )

    reference = _dense_output(operands, limit, -limit, limit)

    # ---- PRECONDITION 1: every clamp branch has elements to act on. Counted and
    # reported, so a fixture that drifted says which regime it lost.
    above_gate = int((reference["gate"] > limit).sum())
    below_gate = int((reference["gate"] <= limit).sum())
    above_up = int((reference["up"] > limit).sum())
    within_up = int(((reference["up"] >= -limit) & (reference["up"] <= limit)).sum())
    below_up = int((reference["up"] < -limit).sum())
    print(
        f"TINYFWD|dense|limit={limit}|gate_above={above_gate}|gate_below={below_gate}"
        f"|up_above={above_up}|up_within={within_up}|up_below={below_up}"
        f"|params={parameters}"
    )
    for name, count in (
        ("gate above the bound", above_gate),
        ("gate at or below the bound", below_gate),
        ("up above the bound", above_up),
        ("up inside the bound", within_up),
        ("up below the negated bound", below_up),
    ):
        if count == 0:
            raise VacuousControlError(
                f"no element has {name}, so this item cannot tell the reference's "
                f"clamp from its absence"
            )

    # ---- PRECONDITION 2: and each branch MOVES THE OUTPUT further than the
    # tolerance this item passes inside. Elements to act on are not enough -- the
    # first draft of this fixture had all five regimes populated and still let a
    # missing lower clamp through at 0.9% against a 1% tolerance. This asks the
    # question the item actually needs answered.
    for name, variant in (
        ("gate upper clamp", _dense_output(operands, None, -limit, limit)),
        ("up upper clamp", _dense_output(operands, limit, -limit, None)),
        ("up lower clamp", _dense_output(operands, limit, None, limit)),
    ):
        moved = not torch.allclose(
            variant["out"], reference["out"], rtol=RTOL, atol=ATOL
        )
        gap = float(
            (variant["out"] - reference["out"]).abs().max()
            / reference["out"].abs().max()
        )
        print(f"TINYFWD|dense_control|branch={name}|outside_tolerance={moved}|gap={gap:.4f}")
        if not moved:
            raise VacuousControlError(
                f"removing the {name} leaves the result inside rtol={RTOL}, "
                f"atol={ATOL}; this item would pass with that branch deleted from "
                f"the module"
            )

    # ---- THE REGISTERED ROUTE PREDICATE, around this item's own call. The
    # reset sits immediately before the forward and the read immediately after,
    # so nothing the reference computed above is inside the window.
    _reset_seam_counters()
    before = _read_seam_counters()
    got = module.forward(operands["hidden"], quant_config=_quant_config())
    after = _read_seam_counters()
    # THREE dispatches on the dense seam and nothing on the routed one: the
    # forward's gate, up and down projections. The dense MLP reaches no MoE
    # kernel at all, so naming only its own seam here is what makes a forward
    # that took the wrong route fail rather than pass on a total.
    _assert_route_predicate("1 dense MLP", {"blockwise_fp8_mm": 3}, before, after)

    if tuple(got.shape) != (TOKENS, HIDDEN_SIZE):
        raise ReferenceShapeError(
            f"the forward returned {tuple(got.shape)}, expected "
            f"{(TOKENS, HIDDEN_SIZE)}"
        )
    torch.testing.assert_close(
        got.float(), reference["out"].float(), rtol=RTOL, atol=ATOL
    )
