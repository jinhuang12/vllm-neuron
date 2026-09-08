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

THE FIRED SET IS ASSERTED PER ITEM RATHER THAN ACCUMULATED, and what makes that sound is
in this file rather than in a plugin. Each item calls :func:`_reset_seam_counters` and
then reads the counters immediately before and immediately after its own forward, and
:func:`_assert_route_predicate` compares the DIFFERENCE against the seams that item
declares. A difference taken across one call cannot be reached by anything an earlier
item did, so module-level state surviving between items changes no assertion here.

THIS FILE THEREFORE NEEDS NO PROCESS ISOLATION, and it no longer asks for any. An earlier
draft of this paragraph said the per-item reading was possible *because* ``pytestmark``
carried ``pytest.mark.forked``, which was never the reason the code worked: the reset and
the difference were already doing that job, and they do it more strictly than a fresh
process would, because they compare against an exact expected set rather than merely
finding the set non-empty. The marker asked the run for a ``pytest-forked`` plugin that
is not installed in the campaign venv and that no lease authorises installing, so it
could only ever have turned a sound run red. The union over the items is read from this
run's transcript, which is what the registered wording's "reported" asks for.

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

#: ``fast`` only. ``forked`` is deliberately absent -- see the paragraph on process
#: isolation in this module's docstring: the reset-and-difference around each forward
#: already gives every item a clean reading, and the marker would have required a
#: ``pytest-forked`` plugin the campaign venv does not have.
pytestmark = [pytest.mark.fast]

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
#: ONE ROW PER COUNTER FAMILY, and a family is what a counter counts rather than
#: what a module holds: ``mla_sparse`` holds THREE families and each gets its own
#: row, because two of them are DECLARED ZEROS at this geometry and a zero that is
#: read is worth more than a zero that is not looked at.
#:
#: THE READ ACCESSOR IS NAMED AND THE RESET IS DERIVED as ``"reset_" + read``,
#: which is the convention every seam module in this repository follows -- so the
#: pair has one declaration and cannot drift apart. Naming the reader rather than
#: discovering it is deliberate: ``test_dsa_layer.py:380`` discovers the pair by
#: scanning for the ``_dispatch_counters`` suffix and asserts exactly one pair per
#: module, which is true of every module below EXCEPT ``mla_sparse``. The coverage
#: control at :func:`_assert_every_counter_family_is_registered` keeps the naming
#: honest by refusing a module that grows a family no row claims.
_SEAM_REGISTRY = {
    "blockwise_fp8_mm": (
        "vllm_neuron.functional.blockwise_fp8_mm", "dispatch_counters"),
    "blockwise_fp8_moe": (
        "vllm_neuron.functional.moe.moe_blockwise_fp8", "dispatch_counters"),
    "mla_projection": (
        "vllm_neuron.functional.attention.mla_projections",
        "mla_projection_dispatch_counters"),
    "mla_absorb": (
        "vllm_neuron.functional.attention.mla_absorb",
        "mla_absorb_dispatch_counters"),
    "mla_sparse": (
        "vllm_neuron.functional.attention.mla_sparse",
        "mla_sparse_dispatch_counters"),
    "mla_sparse_tiled": (
        "vllm_neuron.functional.attention.mla_sparse",
        "mla_sparse_tiled_dispatch_counters"),
    "mla_sparse_row_tiled": (
        "vllm_neuron.functional.attention.mla_sparse",
        "mla_sparse_row_tiled_dispatch_counters"),
    "dsa_kpool_hadamard": (
        "vllm_neuron.functional.dsa.kpool_hadamard",
        "kpool_hadamard_dispatch_counters"),
    "dsa_paged_gather": (
        "vllm_neuron.functional.dsa.paged_gather",
        "paged_gather_dispatch_counters"),
    "dsa_score_gemm": (
        "vllm_neuron.functional.dsa.score_gemm", "score_gemm_dispatch_counters"),
    "dsa_topk_select": (
        "vllm_neuron.functional.dsa.topk_select", "topk_select_dispatch_counters"),
    "dsa_index_expand": (
        "vllm_neuron.functional.dsa.index_expand",
        "index_expand_dispatch_counters"),
    "dsa_decode_tail_update": (
        "vllm_neuron.functional.dsa.decode_tail_update",
        "decode_tail_dispatch_counters"),
    "dsa_ragged_pack": (
        "vllm_neuron.functional.dsa.ragged_pack", "ragged_pack_dispatch_counters"),
}
#: Derived rather than written a second time: a seam listed in one and missing
#: from the other would make the route predicate iterate a name nothing resolves.
_SEAMS = tuple(_SEAM_REGISTRY)

#: The suffix every counter accessor in this repository carries. Read by the
#: coverage control below, so the rule it enforces has one spelling.
#:
#: NO LEADING UNDERSCORE, and that is measured rather than assumed
#: (``probe-054a-counter-names-r1``, twelve modules read): the two blockwise seam
#: modules spell their accessors plainly ``dispatch_counters`` /
#: ``reset_dispatch_counters``, with no family prefix at all, so a suffix of
#: ``"_dispatch_counters"`` matches nothing in them and would make the control
#: below fire on every item -- the four already landed included.
_COUNTER_SUFFIX = "dispatch_counters"


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
    the two blockwise seam modules spell their accessors identically, so a bare
    ``from ... import dispatch_counters`` would resolve to whichever module was
    imported last. Naming the module is what avoids that collision. An alias
    simply was not enough to make the bound object a module.
    """
    return {name: importlib.import_module(path)
            for name, (path, _reader) in _SEAM_REGISTRY.items()}


def _seam_counter_api(name: str) -> tuple:
    """``(read, reset)`` for one registered family. The reset name is DERIVED."""
    path, reader = _SEAM_REGISTRY[name]
    module = importlib.import_module(path)
    return getattr(module, reader), getattr(module, f"reset_{reader}")


def _read_seam_counters() -> dict:
    """``{seam: (nki_dispatch, torch_fallback)}`` as each family reports itself."""
    return {name: tuple(int(v) for v in _seam_counter_api(name)[0]())
            for name in _SEAMS}


def _reset_seam_counters() -> None:
    """Zero every family this campaign owns, so a reading is this item's own."""
    for name in _SEAMS:
        _seam_counter_api(name)[1]()


def _assert_every_counter_family_is_registered() -> None:
    """Every counter family in every registered module is claimed by a row.

    The registry names its readers, so a module that grows a SECOND family would
    be read by nobody and its dispatches would leave the torch-fallback total
    silently incomplete. This walks each registered module for the accessor suffix
    and requires the set it finds to equal the set the registry claims for that
    module -- which is the property naming buys and discovery does not.

    ``dir()`` shows imported names too, so a module that imported another's
    accessor would read as owning a family it does not define. Measured, not
    assumed: across all twelve registered modules no name ending in the suffix is
    imported or assigned, only defined (``probe-054a-counter-names-r1``).

    Raises:
        VacuousControlError: naming which module and which family is unclaimed.
    """
    claimed: dict[str, set] = {}
    for name in _SEAMS:
        path, reader = _SEAM_REGISTRY[name]
        claimed.setdefault(path, set()).add(reader)
    for path, readers in claimed.items():
        module = importlib.import_module(path)
        found = {
            attribute for attribute in dir(module)
            if attribute.endswith(_COUNTER_SUFFIX)
            and not attribute.startswith("reset_")
        }
        print(f"TINYFWD|counter_families|module={path}"
              f"|found={sorted(found)}|claimed={sorted(readers)}")
        if found != readers:
            raise VacuousControlError(
                f"{path} reports the counter families {sorted(found)} and this "
                f"file's registry claims {sorted(readers)}. An unclaimed family "
                f"is a seam whose dispatches and whose torch fallbacks no route "
                f"predicate reads"
            )


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

    THE POPULATION IS CHECKED FIRST, because conjuncts 2 and 3 are statements
    about a set and an unclaimed counter family would quietly shrink it.

    Raises:
        VacuousControlError: on any conjunct, named so the transcript says which.
    """
    from vllm_neuron.utils.neuron_utils import can_run_kernel

    _assert_every_counter_family_is_registered()
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


# --------------------------------------------------------------------------- #
# ITEM 2's OWN GEOMETRY. It does not reuse item 1's, and the reason is measured. #
# --------------------------------------------------------------------------- #
#: The routed bank chooses its own hidden size, and 256 would hide a real defect.
#: ``Glm5NextRoutedExperts.__init__`` reads no hidden size at all -- only the three
#: expert fields -- so the routed fixture is free to pick one. It picks 512 because
#: the down retile's H and I roles are INDISTINGUISHABLE at 256: with one 256-block
#: along H the correct flat index ``i_block * h_256 + h_block`` and the swapped
#: ``h_block * i_256 + i_block`` both simply enumerate the I axis in order, so both
#: frames agree on all four blocks and an item built at 256 would pass with the
#: mapping wrong. At 512 they disagree on 6 of 8 blocks. Measured, not reasoned:
#: ``increments/probe-054a-down-role-geometry.out``, under the campaign artifacts
#: root -- where every probe this file cites lives. Item 1's ``HIDDEN_SIZE`` is
#: untouched.
ROUTED_HIDDEN_SIZE = 512
#: Four ``BLOCK_QUANT_SIZE`` blocks of I, one per clamp regime below.
ROUTED_INTERMEDIATE_SIZE = 1024
#: Four experts and top-2. Four divides any expert-parallel degree this run uses,
#: which is what ``require_uniform_expert_partition`` refuses on, and the bank costs
#: ``E * 3 * I * H`` = 6.3 M parameters against the adopted 10 M cap.
ROUTED_EXPERTS = 4
ROUTED_EXPERTS_PER_TOKEN = 2

#: The two router weights each token's top-2 carries. They sum to the pinned config's
#: ``routed_scaling_factor`` of 2.5, which is what ``norm_topk_prob=True`` with that
#: factor produces, and both are exact binary fractions so no cast loses anything.
#:
#: NEITHER IS 1.0, AND THAT IS THE POINT. ``PRE_SCALE`` and ``POST_SCALE`` agree
#: EXACTLY at affinity 1.0 -- the router weight is then the identity wherever it is
#: applied -- so a fixture whose affinities sat at 1 would carry a mode control that
#: could not fail. Measured across four candidate pairs in section F of
#: ``increments/probe-054a-item2-clamp-feasibility-r5d.out``: ``(1.5, 1.0)`` separates
#: the two modes by 0.68%, inside the tolerance this item passes at, while this pair
#: separates them by 63.9%. The stronger-looking ``(2.25, 0.25)`` was REJECTED: it
#: sends the PRE_SCALE result NEGATIVE against a positive reference, a 775% "gap"
#: that is a different answer rather than a measured sensitivity.
ROUTED_AFFINITIES = (2.0, 0.5)

#: One scale exponent per 256-column block of I, per projection, and the block whose
#: weights are negated. THE VALUES ARE ON A LATTICE, not chosen freely: with weights
#: on the fp8 ``1/8`` grid, hidden states scaled by ``2**-3`` and every block scale an
#: exact power of two, a block's pre-activation is ``H * 2**-3/2 * 1/2 * 2**e``, which
#: at H=512 is exactly ``16 * 2**e``. So the reachable magnitudes are 1, 2, 4, 8, 16,
#: 32 and nothing between them. The four regimes are
#:
#:   block 0  gate  16.0   up    4.0   the gate UPPER clamp binds
#:   block 1  gate   4.0   up   16.0   the up UPPER clamp binds
#:   block 2  gate   2.0   up  -32.0   the up LOWER clamp binds, at a different
#:                                     magnitude from block 1 so the two do not cancel
#:   block 3  gate -16.0   up    8.0   gate below -L, where the reference has NO clamp
#:
#: and each projection has a block strictly INSIDE the bound, so a kernel clamping at
#: the WRONG limit fails on either projection. Chosen and checked in sections A and
#: C of ``increments/probe-054a-item2-clamp-feasibility-r5d.out``.
ROUTED_GATE_EXPONENTS = (0, -2, -3, 0)
ROUTED_GATE_NEGATED_BLOCK = 3
ROUTED_UP_EXPONENTS = (-2, 0, 1, -1)
ROUTED_UP_NEGATED_BLOCK = 2

#: ``down``'s exponent on the row block that I-block 3 feeds, and on the other rows.
#: Block 3 is where gate falls below ``-L`` and the reference applies no lower clamp,
#: so the gate-lower control's whole signal lives in that block and this row scale is
#: what makes it visible. 2**10 was chosen from a sweep of every exponent from -2 to
#: 17 under four criteria carried from the earlier round: the weakest control clears
#: the tolerance by at least 3x, NO control moves the output by more than 100% (a
#: variant that moves it further is a different answer, not a sensitivity), the
#: condition number stays at or under 4, and both neighbouring exponents are
#: admissible too, so a later edit to any pre-activation cannot walk the fixture into
#: a pathological cell. Eleven exponents are admissible and four are neighbour-robust;
#: 2**10 is the robust one with the strongest weakest control, at 38.3%.
ROUTED_DOWN_EXPONENT_BLOCK3 = 10
ROUTED_DOWN_EXPONENT_OTHER = -3

#: The fixture's own tripwires, carried from the same probe. The regimes above are
#: CHOSEN rather than emergent, so a later edit could walk them into near-cancellation
#: -- where the output is a small residue of large opposing terms, every control
#: percentage is inflated by a vanishing denominator, and the item still passes. These
#: two bounds are what a drifted fixture trips on.
#:
#: THEY ARE ASSERTED IN THE MAX-NORM the controls themselves use, which is a
#: deliberate translation of the probe's per-element scalar. Each control is measured
#: as ``(variant - reference).abs().max() / reference.abs().max()``, so the quantity
#: that could deflate a control's denominator is the max-norm of the output, not any
#: single element's value. A per-element ratio would instead blow up on whatever
#: element of a real draw happens to sit nearest zero, which is a property of the draw
#: and not of the fixture's conditioning.
ROUTED_MAX_CONDITION = 4.0
ROUTED_MAX_BLOCK_SHARE = 1.5

#: Distinct from item 1's four, for item 1's reason: a swapped operand must not be
#: able to pass on a shared seed. Distinct from each OTHER across experts too, because
#: ``_fp8_grid_values`` draws the whole ``[E, ., .]`` bank in one call, so every expert
#: gets different weights and a bank that put one expert's weights behind another's
#: router column fails the comparison.
SEED_ROUTED_HIDDEN = 5411
SEED_ROUTED_GATE = 5412
SEED_ROUTED_UP = 5413
SEED_ROUTED_DOWN = 5414

#: The two scaling points, as this file's own labels. Local strings rather than the
#: kernel's enum, so the reference does not import the thing it exists to check; the
#: item ties these labels to the enum's own member names in one precondition.
_POST_SCALE = "POST_SCALE"
_PRE_SCALE = "PRE_SCALE"


def _coarsen_to_256(grid: torch.Tensor) -> torch.Tensor:
    """A ``TILE_SIZE`` scale grid read back at ``BLOCK_QUANT_SIZE``.

    The checkpoint quantises at ``TILE_SIZE`` and the load-time retile widens to
    ``BLOCK_QUANT_SIZE``, so this fixture supplies the 128 grid a loader would and the
    reference needs the 256 view of the same values. DERIVED rather than written a
    second time: two grids stating one fact is the drift :func:`_scale_grid_attribute`
    exists to avoid.

    THE UNIFORMITY REFUSAL IS LOAD-BEARING, not defensive. If the four ``TILE_SIZE``
    entries inside one ``BLOCK_QUANT_SIZE`` block disagreed, the widening would have to
    pick one of them and the producer would count an ``inexact_rescale`` -- so this is
    where "every scale survives the widening exactly" is measured rather than assumed.

    Raises:
        ReferenceShapeError: if the grid does not tile at ``BLOCK_QUANT_SIZE``.
        VacuousControlError: if any 256-block is not uniform.
    """
    step = BLOCK_QUANT_SIZE // TILE_SIZE
    if grid.dim() != 2 or grid.shape[0] % step or grid.shape[1] % step:
        raise ReferenceShapeError(
            f"a {TILE_SIZE} grid of shape {tuple(grid.shape)} does not tile at "
            f"{BLOCK_QUANT_SIZE}"
        )
    coarse = grid[::step, ::step].clone()
    rebuilt = coarse.repeat_interleave(step, dim=0).repeat_interleave(step, dim=1)
    if not torch.equal(rebuilt, grid):
        raise VacuousControlError(
            f"the {TILE_SIZE} scale grid is not uniform inside its "
            f"{BLOCK_QUANT_SIZE} blocks, so the load-time retile would have to "
            f"rescale inexactly and this fixture's exact-power-of-two premise is "
            f"gone"
        )
    return coarse


def _routed_tile_grid(exponents: tuple[int, ...], *, i_first: bool) -> torch.Tensor:
    """``[E, I/128, H/128]`` powers of two, one exponent per 256-block of I.

    ``i_first=False`` returns the ``[E, H/128, I/128]`` transpose, which is the
    orientation the checkpoint stores ``down``'s grid in. Uniform along H and across
    both ``TILE_SIZE`` halves of each 256-block, which is what :func:`_coarsen_to_256`
    then measures rather than trusts.
    """
    step = BLOCK_QUANT_SIZE // TILE_SIZE
    i_tiles = ROUTED_INTERMEDIATE_SIZE // TILE_SIZE
    h_tiles = ROUTED_HIDDEN_SIZE // TILE_SIZE
    per_tile = [float(2.0 ** exponents[tile // step]) for tile in range(i_tiles)]
    grid = torch.tensor(per_tile, dtype=torch.float32).reshape(i_tiles, 1)
    grid = grid.repeat(1, h_tiles)
    if not i_first:
        grid = grid.t().contiguous()
    return grid.unsqueeze(0).repeat(ROUTED_EXPERTS, 1, 1).contiguous()


def _routed_affinities() -> torch.Tensor:
    """``[T, E]`` scattered top-2 router scores -- what ``route_tokens`` returns.

    The gate weight at each selected expert's column and zero elsewhere, at the GLOBAL
    router width, which is the form ``block_quant_expert_mm`` declares and refuses
    anything else.

    THE TWO ROLES ROTATE ACROSS TOKENS so every expert carries the high weight on some
    token and the low weight on another. A bank that put one expert's weights behind
    another's router column would otherwise be able to agree on a fixture where each
    expert always had the same weight.
    """
    high, low = ROUTED_AFFINITIES
    affinities = torch.zeros(TOKENS, ROUTED_EXPERTS, dtype=torch.float32)
    for token in range(TOKENS):
        affinities[token, token % ROUTED_EXPERTS] = high
        affinities[token, (token + 1) % ROUTED_EXPERTS] = low
    selected = int((affinities != 0).sum(dim=1).min())
    if selected != ROUTED_EXPERTS_PER_TOKEN:
        raise VacuousControlError(
            f"a token carries {selected} nonzero router columns and this fixture "
            f"declares top-{ROUTED_EXPERTS_PER_TOKEN}; the block mapping is built "
            f"from that count and would disagree with the mask"
        )
    return affinities


def _routed_operands() -> dict:
    """The bank's three weights, three ``TILE_SIZE`` grids, hidden states, affinities.

    The weights are the CHECKPOINT's orientations, which is what
    ``prepare_scale_operands`` declares it takes: gate and up ``[E, I, H]``, down
    ``[E, H, I]``. It transposes on the way in, and getting that wrong is the defect
    the 512 hidden size exists to expose.
    """
    blocks = ROUTED_INTERMEDIATE_SIZE // BLOCK_QUANT_SIZE
    if blocks != len(ROUTED_GATE_EXPONENTS) or blocks != len(ROUTED_UP_EXPONENTS):
        raise VacuousControlError(
            f"this fixture declares {len(ROUTED_GATE_EXPONENTS)} gate and "
            f"{len(ROUTED_UP_EXPONENTS)} up regimes for {blocks} blocks of I"
        )
    if ROUTED_HIDDEN_SIZE % BLOCK_QUANT_SIZE or ROUTED_HIDDEN_SIZE // BLOCK_QUANT_SIZE < 2:
        raise VacuousControlError(
            f"ROUTED_HIDDEN_SIZE={ROUTED_HIDDEN_SIZE} must be a multiple of "
            f"{BLOCK_QUANT_SIZE} and give at least TWO blocks along H, or the down "
            f"retile's H and I roles are indistinguishable and this item would pass "
            f"with that mapping wrong"
        )

    hidden = _fp8_grid_values(
        SEED_ROUTED_HIDDEN, TOKENS, ROUTED_HIDDEN_SIZE
    ) * float(2.0**HIDDEN_SCALE_EXPONENT)

    gate_w = _fp8_grid_values(
        SEED_ROUTED_GATE, ROUTED_EXPERTS, ROUTED_INTERMEDIATE_SIZE, ROUTED_HIDDEN_SIZE
    )
    up_w = _fp8_grid_values(
        SEED_ROUTED_UP, ROUTED_EXPERTS, ROUTED_INTERMEDIATE_SIZE, ROUTED_HIDDEN_SIZE
    )
    down_w = _fp8_grid_values(
        SEED_ROUTED_DOWN, ROUTED_EXPERTS, ROUTED_HIDDEN_SIZE, ROUTED_INTERMEDIATE_SIZE
    )
    # A WHOLE 256-row block is negated on each of the two projections that need a
    # sign, so all 256 terms of the affected columns share it and the sum does not
    # cancel -- item 1's device, for item 1's reason. The row axis of gate and up IS
    # the I axis in the checkpoint's orientation.
    for negated, weight in ((ROUTED_GATE_NEGATED_BLOCK, gate_w),
                            (ROUTED_UP_NEGATED_BLOCK, up_w)):
        rows = slice(negated * BLOCK_QUANT_SIZE, (negated + 1) * BLOCK_QUANT_SIZE)
        weight[:, rows, :] = -weight[:, rows, :]

    down_exponents = tuple(
        ROUTED_DOWN_EXPONENT_BLOCK3 if block == ROUTED_GATE_NEGATED_BLOCK
        else ROUTED_DOWN_EXPONENT_OTHER
        for block in range(blocks)
    )
    return {
        "hidden": hidden.to(torch.bfloat16),
        "expert_affinities": _routed_affinities(),
        "gate_proj_weight": (gate_w.to(_FP8),
                             _routed_tile_grid(ROUTED_GATE_EXPONENTS, i_first=True)),
        "up_proj_weight": (up_w.to(_FP8),
                           _routed_tile_grid(ROUTED_UP_EXPONENTS, i_first=True)),
        "down_proj_weight": (down_w.to(_FP8),
                             _routed_tile_grid(down_exponents, i_first=False)),
    }


def _routed_output(
    operands: dict,
    *,
    mode: str,
    gate_max: float | None,
    gate_min: float | None,
    up_max: float | None,
    up_min: float | None,
) -> dict:
    """The whole routed bank in torch, in fp32, with the scaling point AND the four
    clamp branches switchable.

    ``sum_e a_e * down(silu(clamp(gate)) * clamp(up))`` at ``POST_SCALE``, which is the
    checkpoint reference's function: ``modeling_glm5_next.py:132-133`` projects
    ``hidden_states[token_idx]`` UNSCALED and then multiplies ``top_k_weights`` into
    the down projection's result. At ``PRE_SCALE`` the weight multiplies the hidden
    states before both projections instead, which is a DIFFERENT function rather than a
    rearranged one -- the repository's own torch implementation says so at
    ``vllm_neuron/functional/moe/moe_cte.py:490-492`` (the DIRECTORY matters: the
    vendor kernel library ships a ``moe_cte.py`` of its own, whose lines there are
    different): "this is NOT mathematically equivalent to POST_SCALE because the
    nonlinear activation breaks the linearity: act(a * x) != a * act(x)".

    ``mode`` IS AN ARGUMENT for the same reason the clamp bounds are: the item
    recomputes this reference at ``PRE_SCALE`` and requires the answer OUTSIDE the
    tolerance it passes inside, so the scaling point is measured rather than declared.
    Until this increment the call site passed no mode at all and inherited the shim's
    ``PRE_SCALE`` default, so this control is the one that would have caught it.

    Returns the output, the per-block contributions the conditioning tripwire reads,
    and the two pre-activation stacks the regime preconditions count.
    """
    from torch.nn.functional import silu

    if mode not in (_POST_SCALE, _PRE_SCALE):
        raise ReferenceShapeError(f"unknown scaling point {mode!r}")
    hidden = operands["hidden"].to(torch.float32)
    affinities = operands["expert_affinities"]
    gate_w, gate_grid = operands["gate_proj_weight"]
    up_w, up_grid = operands["up_proj_weight"]
    down_w, down_grid = operands["down_proj_weight"]
    blocks = ROUTED_INTERMEDIATE_SIZE // BLOCK_QUANT_SIZE

    terms = [torch.zeros(TOKENS, ROUTED_HIDDEN_SIZE, dtype=torch.float32)
             for _ in range(blocks)]
    gates, ups = [], []
    for expert in range(ROUTED_EXPERTS):
        weight = affinities[:, expert:expert + 1]
        activations = hidden * weight if mode == _PRE_SCALE else hidden
        # ``[I, H]`` dequantised, so the projection is ``x @ W.t()``.
        gate = activations @ _dequantise(
            gate_w[expert], _coarsen_to_256(gate_grid[expert])
        ).t()
        up = activations @ _dequantise(
            up_w[expert], _coarsen_to_256(up_grid[expert])
        ).t()
        gates.append(gate)
        ups.append(up)

        clamped = silu(gate.clamp(min=gate_min, max=gate_max)) * up.clamp(
            min=up_min, max=up_max
        )
        # The kernel's declared activation dtype is bf16, so the reference casts too.
        # Skipping it would compare against a function the shipped path never computes.
        clamped = clamped.to(torch.bfloat16).to(torch.float32)
        down = _dequantise(down_w[expert], _coarsen_to_256(down_grid[expert]))
        for block in range(blocks):
            columns = slice(block * BLOCK_QUANT_SIZE, (block + 1) * BLOCK_QUANT_SIZE)
            term = clamped[:, columns] @ down[:, columns].t()
            terms[block] += term * weight if mode == _POST_SCALE else term

    stack = torch.stack(terms)
    out = stack.sum(dim=0)
    scale = float(out.abs().max())
    return {
        "out": out,
        "blocks": stack,
        "gate": torch.stack(gates),
        "up": torch.stack(ups),
        "condition": float(stack.abs().sum(dim=0).max()) / scale if scale else float("inf"),
        "share": float(stack.abs().max()) / scale if scale else float("inf"),
    }


def _routed_text_config():
    """The routed item's text config -- its own extents, the shared constraint set.

    Two keyword arguments item 1's helper does not pass, and no change to
    ``config.py``: ``n_routed_experts`` and ``num_experts_per_tok`` already exist there
    with production defaults, so the fixture overrides them here and the module under
    test is never edited to accommodate a test.
    """
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    return Glm5NextTextConfig(
        hidden_size=ROUTED_HIDDEN_SIZE,
        intermediate_size=ROUTED_INTERMEDIATE_SIZE,
        num_key_value_heads=NUM_KEY_VALUE_HEADS,
        n_routed_experts=ROUTED_EXPERTS,
        num_experts_per_tok=ROUTED_EXPERTS_PER_TOKEN,
    )


# --------------------------------------------------------------------------- #
# ITEM 2 of 7 -- ``Glm5NextRoutedExperts.forward``.                            #
# Certifying component: ``model_fp8.Glm5NextRoutedExperts.forward``.            #
# --------------------------------------------------------------------------- #
def test_tiny_routed_experts_forward_matches_the_reference() -> None:
    """The routed bank on one MoE layer, against the checkpoint's POST_SCALE reference.

    ``inc-glm53f-054a`` item 2 of 7. ONE dispatch on the MoE seam and nothing on the
    dense one.

    FIVE CONTROLS, not item 1's three. The four clamp branches, plus the SCALING POINT
    -- where the router weight multiplies. That fifth one is the increment's material
    finding: the call site used to pass no mode and inherit ``PRE_SCALE``, a different
    function from the checkpoint's, with no shape moving and nothing raising.
    """
    model_fp8 = _impl()
    from vllm_neuron.functional.moe.moe_blockwise_fp8 import ExpertAffinityScaleMode

    text_config = _routed_text_config()
    module = model_fp8.Glm5NextRoutedExperts(text_config)

    limit = float(module.swiglu_limit)
    if limit != float(text_config.swiglu_limit):
        raise VacuousControlError(
            f"the bank resolved swiglu_limit={limit} but the config declares "
            f"{text_config.swiglu_limit}; the bound under test would not be the "
            f"checkpoint's"
        )
    # THE LABELS THIS FILE REASONS WITH ARE THE KERNEL'S OWN MEMBER NAMES. The
    # reference above takes a local string so it does not import the thing it exists to
    # check; this ties that string to the enum the call site passes, so a renamed or
    # re-spelled member cannot leave the reference certifying a mode nothing selects.
    for label in (_POST_SCALE, _PRE_SCALE):
        if getattr(ExpertAffinityScaleMode, label).name != label:
            raise VacuousControlError(
                f"this file reasons about a scaling point it calls {label!r} and the "
                f"seam's enum spells that member "
                f"{getattr(ExpertAffinityScaleMode, label).name!r}"
            )

    operands = _routed_operands()
    for leaf in ("gate_proj_weight", "up_proj_weight", "down_proj_weight"):
        _attach(module, leaf, *operands[leaf])

    parameters = sum(int(p.numel()) for p in module.parameters() if p is not None)
    if parameters >= MAX_PARAMETERS:
        raise VacuousControlError(
            f"the tiny routed bank holds {parameters} parameters, at or above the "
            f"adopted bound of {MAX_PARAMETERS}"
        )

    # ---- THE LOAD-TIME PREP, once, as production runs it. The bank's forward reads
    # its four kernel operands through ``_prepared_kernel_operand``, which REFUSES if
    # this has not run -- that refusal is what makes "retiled once at load time, never
    # per forward step" checkable, so the item runs the prep rather than reaching past
    # it.
    built = module.prepare_scale_operands(
        gate_proj_weight=operands["gate_proj_weight"][0],
        up_proj_weight=operands["up_proj_weight"][0],
        down_proj_weight=operands["down_proj_weight"][0],
        gate_proj_scale=operands["gate_proj_weight"][1],
        up_proj_scale=operands["up_proj_weight"][1],
        down_proj_scale=operands["down_proj_weight"][1],
    )
    health = getattr(module, module.RETILE_HEALTH_ATTR)
    print(
        f"TINYFWD|routed_prep|operands={built}|params={parameters}"
        + "".join(f"|{bank}={counts}" for bank, counts in sorted(health.items()))
    )
    if built != 4:
        raise VacuousControlError(
            f"prepare_scale_operands built {built} operands and the bank's forward "
            f"looks up 4"
        )
    # ONLY THE THIRD COUNT IS ASSERTED, and the other two are reported. The producer
    # writes ONE fusion half per call and leaves the other unwritten, so a nonzero
    # ``emitted_unsupplied`` on gate and up is the arrangement working rather than
    # failing. ``inexact_rescales`` is the one this fixture's exact-power-of-two premise
    # predicts to be zero, and it is the reading that says the widening from TILE_SIZE
    # to BLOCK_QUANT_SIZE lost nothing.
    inexact = {bank: counts[2] for bank, counts in health.items() if counts[2]}
    if inexact:
        raise VacuousControlError(
            f"the retile reports inexact rescales {inexact}; this fixture's scales are "
            f"exact powers of two uniform inside every 256-block, so a nonzero count "
            f"means the widening is not the one this reference models"
        )

    reference = _routed_output(
        operands, mode=_POST_SCALE, gate_max=limit, gate_min=None,
        up_max=limit, up_min=-limit,
    )

    # ---- PRECONDITION 1: every regime the four controls need has elements in it.
    gate, up = reference["gate"], reference["up"]
    counts = {
        "gate above the bound": int((gate > limit).sum()),
        "gate inside the bound": int(((gate >= -limit) & (gate <= limit)).sum()),
        "gate below the negated bound": int((gate < -limit).sum()),
        "up above the bound": int((up > limit).sum()),
        "up inside the bound": int(((up >= -limit) & (up <= limit)).sum()),
        "up below the negated bound": int((up < -limit).sum()),
    }
    print(
        f"TINYFWD|routed|limit={limit}|condition={reference['condition']:.4f}"
        f"|worst_block_share={reference['share']:.4f}"
        + "".join(f"|{name.replace(' ', '_')}={count}" for name, count in counts.items())
    )
    for name, count in counts.items():
        if count == 0:
            raise VacuousControlError(
                f"no element has {name}, so this item cannot tell the reference's "
                f"clamp from its absence"
            )

    # ---- PRECONDITION 2: the fixture is still WELL CONDITIONED. The regimes above are
    # chosen rather than emergent, so a later edit could walk them into a
    # near-cancellation where the output is a small residue of large opposing terms and
    # every control percentage below is inflated by a vanishing denominator. This is the
    # fixture's own tripwire, and it fires before any control is believed.
    if reference["condition"] > ROUTED_MAX_CONDITION:
        raise VacuousControlError(
            f"the block contributions sum to {reference['condition']:.2f} times the "
            f"output they produce, above the declared bound of "
            f"{ROUTED_MAX_CONDITION}: the fixture has drifted into near-cancellation "
            f"and every control gap it reports is inflated"
        )
    if reference["share"] > ROUTED_MAX_BLOCK_SHARE:
        raise VacuousControlError(
            f"one 256-block carries {reference['share'] * 100:.0f}% of the output, "
            f"above the declared bound of {ROUTED_MAX_BLOCK_SHARE * 100:.0f}%"
        )

    # ---- PRECONDITION 3: and each of the five branches MOVES THE OUTPUT further than
    # the tolerance this item passes inside. Elements to act on are not enough: item 1's
    # first fixture had every regime populated and still let a missing lower clamp
    # through at 0.9% against a 1% tolerance.
    variants = {
        "gate upper clamp removed": dict(
            mode=_POST_SCALE, gate_max=None, gate_min=None, up_max=limit, up_min=-limit),
        "up upper clamp removed": dict(
            mode=_POST_SCALE, gate_max=limit, gate_min=None, up_max=None, up_min=-limit),
        "up lower clamp removed": dict(
            mode=_POST_SCALE, gate_max=limit, gate_min=None, up_max=limit, up_min=None),
        # THE BRANCH THE REFERENCE DOES NOT HAVE. ``gate`` is bounded from ABOVE only
        # (``modeling_glm5_next.py:139`` passes ``min=None``) and ``up`` on both sides
        # (``:140``). Making gate two-sided would be a second wrong function rather
        # than a tidier one, and this control is what turns that asymmetry from an
        # assumption into a measurement.
        "gate lower clamp wrongly added": dict(
            mode=_POST_SCALE, gate_max=limit, gate_min=-limit, up_max=limit,
            up_min=-limit),
        # THE SCALING POINT. Everything else about the reference is held fixed.
        "scaling point moved to PRE_SCALE": dict(
            mode=_PRE_SCALE, gate_max=limit, gate_min=None, up_max=limit, up_min=-limit),
    }
    for name, keywords in variants.items():
        variant = _routed_output(operands, **keywords)
        moved = not torch.allclose(
            variant["out"], reference["out"], rtol=RTOL, atol=ATOL
        )
        gap = float(
            (variant["out"] - reference["out"]).abs().max()
            / reference["out"].abs().max()
        )
        print(
            f"TINYFWD|routed_control|branch={name}|outside_tolerance={moved}"
            f"|gap={gap:.4f}"
        )
        if not moved:
            raise VacuousControlError(
                f"changing the reference so that its {name} leaves the result inside "
                f"rtol={RTOL}, atol={ATOL}; this item would pass with that branch "
                f"wrong in the module"
            )

    # ---- THE REGISTERED ROUTE PREDICATE, around this item's own call.
    _reset_seam_counters()
    before = _read_seam_counters()
    got = module.forward(
        operands["hidden"],
        operands["expert_affinities"],
        _quant_config(),
    )
    after = _read_seam_counters()
    # ONE dispatch on the MoE seam and nothing on the dense one. The bank reaches
    # ``blockwise_fp8_moe`` exactly once per forward and no dense projection at all, so
    # naming only that seam is what makes a forward which took the wrong route fail
    # instead of passing on a total.
    _assert_route_predicate("2 routed experts", {"blockwise_fp8_moe": 1}, before, after)

    if tuple(got.shape) != (TOKENS, ROUTED_HIDDEN_SIZE):
        raise ReferenceShapeError(
            f"the forward returned {tuple(got.shape)}, expected "
            f"{(TOKENS, ROUTED_HIDDEN_SIZE)} -- the padding-token row is the callee's "
            f"to slice off"
        )
    torch.testing.assert_close(
        got.float(), reference["out"].float(), rtol=RTOL, atol=ATOL
    )


# --------------------------------------------------------------------------- #
# ITEM 3 of 7 -- ``Glm5NextSharedExperts.forward``.                            #
# Certifying component: ``model_fp8.Glm5NextSharedExperts.forward``.           #
#                                                                              #
# IT REUSES ITEM 1's FIXTURE AND ITS REFERENCE, and that is a reading rather    #
# than a shortcut. The shared expert and the dense MLP compute the SAME         #
# function -- three dense-seam projections and the checkpoint's clamped SwiGLU  #
# -- from the same reference class (``modeling_glm5_next.py:86``'s one          #
# ``Glm5NextTextMLP``, built at ``:196`` as ``shared_experts`` and at ``:1271`` #
# as the dense ``mlp``). Two references for one function would be two things to #
# keep in step. What THIS item adds is everything the two paths do differently: #
# the load-time scale-operand prep, the forward's lookup of the three grids off #
# the module, and the refusals that stand where the dense path has none.        #
# --------------------------------------------------------------------------- #
def test_tiny_shared_experts_forward_matches_the_reference() -> None:
    """The always-on shared expert on one MoE layer, against the same reference.

    ``inc-glm53f-054a`` item 3 of 7. THREE dispatches: gate, up, down.
    """
    model_fp8 = _impl()
    text_config = _tiny_text_config()
    module = model_fp8.Glm5NextSharedExperts(text_config)

    limit = float(module.swiglu_limit)
    if limit != float(text_config.swiglu_limit):
        raise VacuousControlError(
            f"the module resolved swiglu_limit={limit} but the config declares "
            f"{text_config.swiglu_limit}; the bound under test would not be the "
            f"checkpoint's"
        )

    operands = _dense_operands()
    leaves = ("gate_proj_weight", "up_proj_weight", "down_proj_weight")
    for leaf in leaves:
        _attach(module, leaf, *operands[leaf])

    parameters = sum(int(p.numel()) for p in module.parameters() if p is not None)
    if parameters >= MAX_PARAMETERS:
        raise VacuousControlError(
            f"the tiny shared expert holds {parameters} parameters, at or above "
            f"the adopted bound of {MAX_PARAMETERS}"
        )

    # ---- CONTROL A: THE PREP IS REQUIRED, and the forward says so by name.
    # ``shared_expert_mm`` reads three operands ``prepare_scale_operands`` builds
    # at load time and refuses to build them per forward step. Calling the forward
    # first is how this item proves the shipped path really reads those operands
    # rather than quietly rebuilding them -- a lazy rebuild would make this call
    # succeed and would put the per-call scatter back with nothing reporting it.
    with pytest.raises(model_fp8.Glm5NextSharedExpertRouteError) as unprepared:
        module.forward(operands["hidden"], quant_config=_quant_config())
    print(f"TINYFWD|shared_control|unprepared={str(unprepared.value)[:60]!r}")

    # ---- THE LOAD-TIME PREP, run the way ``_run_load_time_preps`` runs it: off
    # this module's own attributes, in the declaration order the two methods
    # share. The count is the method's own return, not a length this item counts.
    built = module.prepare_scale_operands(
        *(getattr(module, leaf) for leaf in leaves),
        *(getattr(module, _scale_grid_attribute(leaf)) for leaf in leaves),
    )
    if built != 3:
        raise VacuousControlError(
            f"the load-time prep reported {built} operands, not the 3 the shared "
            f"expert's three projections need"
        )

    reference = _dense_output(operands, limit, -limit, limit)

    # ---- PRECONDITION 1: every clamp branch has elements to act on.
    above_gate = int((reference["gate"] > limit).sum())
    below_gate = int((reference["gate"] <= limit).sum())
    above_up = int((reference["up"] > limit).sum())
    within_up = int(((reference["up"] >= -limit) & (reference["up"] <= limit)).sum())
    below_up = int((reference["up"] < -limit).sum())
    print(
        f"TINYFWD|shared|limit={limit}|prepared={built}"
        f"|shared_experts={int(module.num_shared_experts)}"
        f"|gate_above={above_gate}|gate_below={below_gate}|up_above={above_up}"
        f"|up_within={within_up}|up_below={below_up}|params={parameters}"
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
    # tolerance this item passes inside. Item 1's reason, and item 1's measurement:
    # its first fixture had every regime populated and still let a missing lower
    # clamp through at 0.9% against a 1% tolerance.
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
        print(
            f"TINYFWD|shared_control|branch={name}|outside_tolerance={moved}"
            f"|gap={gap:.4f}"
        )
        if not moved:
            raise VacuousControlError(
                f"removing the {name} leaves the result inside rtol={RTOL}, "
                f"atol={ATOL}; this item would pass with that branch deleted from "
                f"the module"
            )

    # ---- THE REGISTERED ROUTE PREDICATE, around this item's own call.
    _reset_seam_counters()
    before = _read_seam_counters()
    got = module.forward(operands["hidden"], quant_config=_quant_config())
    after = _read_seam_counters()
    # THREE dispatches on the dense seam and nothing on the MoE one. The shared
    # expert is the DENSE blockwise route at its own width (DECISIONS §77), so a
    # forward that reached the expert kernel fails here rather than passing on a
    # total.
    _assert_route_predicate("3 shared experts", {"blockwise_fp8_mm": 3}, before, after)

    if tuple(got.shape) != (TOKENS, HIDDEN_SIZE):
        raise ReferenceShapeError(
            f"the forward returned {tuple(got.shape)}, expected "
            f"{(TOKENS, HIDDEN_SIZE)}"
        )
    torch.testing.assert_close(
        got.float(), reference["out"].float(), rtol=RTOL, atol=ATOL
    )

    # ---- CONTROL B: THE GRID LOOKUP IS THE FORWARD'S OWN, and a missing grid is
    # refused by name rather than reaching an unscaled matmul. Taken AFTER the
    # comparison so the item's own reading is never taken on a mutated module:
    # ``down_proj``'s grid is removed from a module that has already been read.
    removed = _scale_grid_attribute("down_proj_weight")
    delattr(module, removed)
    with pytest.raises(model_fp8.Glm5NextSharedExpertRouteError) as missing:
        module.forward(operands["hidden"], quant_config=_quant_config())
    if removed not in str(missing.value):
        raise VacuousControlError(
            f"the refusal for a missing grid does not name {removed}: "
            f"{missing.value}"
        )
    print(f"TINYFWD|shared_control|missing_grid={removed}|named=True")


# --------------------------------------------------------------------------- #
# ITEM 4's OWN ADDITIONS. It runs the routed bank and a shared expert TOGETHER,  #
# so it reuses item 2's bank fixture whole and needs a shared expert at the      #
# bank's own hidden size, which item 1's 256-wide one is not.                    #
# --------------------------------------------------------------------------- #
#: The shared expert's scale exponents at ``ROUTED_HIDDEN_SIZE``, one per 256-block
#: of I. The magnitudes follow item 2's lattice reading, which is the same
#: arithmetic on the same axis: with weights on the fp8 ``1/8`` grid and hidden
#: states scaled by ``2**HIDDEN_SCALE_EXPONENT``, a block's pre-activation at
#: H=512 is exactly ``16 * 2**e``. Against the checkpoint's bound of 10 that puts
#:
#:   gate  ``2**0, 2**-3, 2**0, 2**-3``   ->  16,  2, 16,  2   above and inside
#:   up    ``2**-3, 2**0, 2**-3, 2**0``   ->   2, 16,  2, -16  the last one negated
#:
#: so both projections have a block the upper clamp binds on and one strictly
#: inside the bound, and ``up``'s negated block is below the negated bound. The
#: clamp DISCRIMINATION controls stay in items 1 and 3, which own that reading;
#: this item needs the regimes populated only so its shared half is not a
#: degenerate one.
SHARED_AT_ROUTED_GATE_EXPONENTS = (0, -3, 0, -3)
SHARED_AT_ROUTED_UP_EXPONENTS = (-3, 0, -3, 0)
SHARED_AT_ROUTED_UP_NEGATED_BLOCK = 3
#: One exponent per 256-column of H, uniform down the rows, so no single row block
#: dominates the output and the sum stays well conditioned.
SHARED_AT_ROUTED_DOWN_EXPONENT = -3

SEED_SHARED_GATE = 5421
SEED_SHARED_UP = 5422
SEED_SHARED_DOWN = 5423
SEED_MOE_HIDDEN = 5424
SEED_MOE_ROUTER = 5425

#: The FFN norm's gain, and it is deliberately NOT ones. RMSNorm is not
#: idempotent, but a gain of ones on hidden states whose RMS is already 1 comes
#: close enough to it that applying the norm twice would move almost nothing --
#: and "the pre-norm tensor reaches the router, the normed one reaches the
#: experts" is exactly what this item has to be able to see. These four values
#: repeat along H, are exact in bf16, and give the normed states an RMS near 1.4,
#: so a second application is a measurable change rather than a rounding one.
MOE_GAMMA_VALUES = (1.0, 1.25, 1.5, 1.75)

#: The router's own scale, following the landed fixture this item copies
#: (``test_moe_path.py:1899-1905``): a small normal draw for the weight and a
#: smaller one for the correction bias.
MOE_ROUTER_WEIGHT_SCALE = 0.1
MOE_ROUTER_BIAS_SCALE = 0.05


def _shared_at_routed_operands() -> dict:
    """A shared expert's three weights and PUBLIC grids at the bank's hidden size.

    The shared route consumes the 256-granularity grid directly -- that is what
    ``prepare_scale_operands`` takes and what the load-path retile publishes -- so
    unlike the bank's fixture this one supplies no ``TILE_SIZE`` grid and nothing
    is coarsened. Item 1's builder is untouched: it is 256 wide by its own
    declared reading and this item needs 512.
    """
    h = ROUTED_HIDDEN_SIZE
    i = ROUTED_INTERMEDIATE_SIZE
    blocks = i // BLOCK_QUANT_SIZE
    if blocks != len(SHARED_AT_ROUTED_GATE_EXPONENTS):
        raise VacuousControlError(
            f"this fixture declares {len(SHARED_AT_ROUTED_GATE_EXPONENTS)} gate "
            f"regimes for {blocks} blocks of I"
        )

    gate_w = _fp8_grid_values(SEED_SHARED_GATE, h, i)
    up_w = _fp8_grid_values(SEED_SHARED_UP, h, i)
    columns = slice(
        SHARED_AT_ROUTED_UP_NEGATED_BLOCK * BLOCK_QUANT_SIZE,
        (SHARED_AT_ROUTED_UP_NEGATED_BLOCK + 1) * BLOCK_QUANT_SIZE,
    )
    up_w[:, columns] = -up_w[:, columns]
    down_w = _fp8_grid_values(SEED_SHARED_DOWN, i, h)

    h_blocks = h // BLOCK_QUANT_SIZE
    i_blocks = i // BLOCK_QUANT_SIZE
    return {
        "gate_proj_weight": (
            gate_w.to(_FP8),
            _pow2_scales(SHARED_AT_ROUTED_GATE_EXPONENTS, h_blocks),
        ),
        "up_proj_weight": (
            up_w.to(_FP8),
            _pow2_scales(SHARED_AT_ROUTED_UP_EXPONENTS, h_blocks),
        ),
        "down_proj_weight": (
            down_w.to(_FP8),
            _pow2_scales((SHARED_AT_ROUTED_DOWN_EXPONENT,) * h_blocks, i_blocks),
        ),
    }


def _ffn_gamma() -> torch.Tensor:
    """``[H]`` the FFN norm's gain, repeating :data:`MOE_GAMMA_VALUES` along H."""
    row = torch.tensor(MOE_GAMMA_VALUES, dtype=torch.float32)
    return row.repeat(ROUTED_HIDDEN_SIZE // len(MOE_GAMMA_VALUES))


def _ffn_norm(hidden: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    """The layer's FFN RMSNorm, the body the two landed layers already carry.

    ``x / sqrt(mean(x**2) + eps) * gain``, computed in fp32 and cast back, which
    is ``Glm5NextDSALayer._input_norm``'s own arithmetic. It lives here because
    the MoE block's forward takes the normalised tensor as an argument: the norm
    is the LAYER's, and the layer forward is a later item. This item is the caller
    and so it normalises.
    """
    x = hidden.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    normed = x * torch.rsqrt(variance + eps)
    normed = normed * gamma.to(torch.float32)
    return normed.to(hidden.dtype)


# --------------------------------------------------------------------------- #
# ITEM 4 of 7 -- ``Glm5NextMoEBlock.forward``.                                 #
# Certifying component: ``model_fp8.Glm5NextMoEBlock.forward``.                #
#                                                                              #
# WHAT IT CERTIFIES, AND WHAT IT DELIBERATELY LEAVES TO ITS NEIGHBOURS. This   #
# forward is a composition: the fused router, then this rank's routed bank,     #
# then the one add of the shared contribution. So the item measures the         #
# COMPOSITION -- that the shared half is added exactly once, that the router    #
# reads the PRE-norm activations while the experts read the normalised ones,    #
# and that a block with no shared expert returns the routed half untouched.     #
# The clamp discrimination belongs to items 1, 2 and 3, which own those paths.  #
#                                                                              #
# THE ROUTER IS EXECUTED RATHER THAN IMITATED, which is the landed convention   #
# for this path (``test_moe_path.py:1890-1895``: "the router is executed rather #
# than imitated, so the form the call site consumes is the form the producer    #
# actually emits"). Its affinities are an INPUT to the reference. Re-deriving   #
# them here would put this item in the business of certifying the router, which #
# ``inc-glm53f-032``'s own acceptance owns, and would make the comparison       #
# hostage to a near-tie flipping one token's expert set.                        #
#                                                                              #
# THE ROUTER'S OWN SEAM CARRIES NO DISPATCH COUNTERS, disclosed rather than     #
# papered over: ``functional/moe/router.py`` defines none, so the route         #
# predicate below reads the two seams that do. A router fallback on this        #
# fixture cannot pass quietly even so, and that is measured rather than hoped:  #
# its torch oracle selects ``NOAUX_TC_K`` = 8 columns regardless of the         #
# caller's ``top_k`` (``router.py:1736-1738``), and this fixture has 4 experts, #
# so a fallback raises out of ``torch.topk`` instead of returning a plausible   #
# answer.                                                                       #
# --------------------------------------------------------------------------- #
def test_tiny_moe_block_forward_matches_the_reference() -> None:
    """One sparse layer's MLP: route, run the experts, add the shared expert once.

    ``inc-glm53f-054a`` item 4 of 7. ONE dispatch on the MoE seam for the bank and
    THREE on the dense seam for the shared expert.
    """
    model_fp8 = _impl()
    text_config = _routed_text_config()
    quant_config = _quant_config()

    if int(text_config.n_shared_experts) < 1:
        raise VacuousControlError(
            f"this item needs a shared expert and the config declares "
            f"n_shared_experts={text_config.n_shared_experts}"
        )

    # ---- THE GROUP STAGE OF THE REFERENCE'S ROUTER IS AN IDENTITY ON THIS
    # CHECKPOINT, read from the pinned config rather than assumed. The reference
    # masks all but the top ``topk_group`` of ``n_group`` expert groups
    # (``modeling_glm5_next.py:163-176``) and this fork's seam takes no group
    # arguments at all. With one group of which one is kept, the mask is every
    # column, so the two agree. A checkpoint with more groups would make this
    # item's reference wrong, which is why it is checked here and not believed.
    raw = _pinned_raw_config()
    raw_text = raw.get("text_config", raw)
    groups = int(raw_text.get("n_group", 1))
    kept = int(raw_text.get("topk_group", 1))
    print(f"TINYFWD|moe_groups|n_group={groups}|topk_group={kept}")
    if groups != kept:
        raise VacuousControlError(
            f"the pinned checkpoint declares n_group={groups} and "
            f"topk_group={kept}, so the reference's group mask is not an identity "
            f"and this fork's router, which takes no group arguments, computes a "
            f"different selection"
        )

    block = model_fp8.Glm5NextMoEBlock(text_config, world_size=1, ep_degree=1)
    if getattr(block, "shared_experts", None) is None:
        raise VacuousControlError(
            "the block built no shared expert, so this item's one add is not on "
            "its route at all"
        )

    # ---- THE BANK, item 2's fixture whole, and its load-time prep.
    routed = _routed_operands()
    for leaf in ("gate_proj_weight", "up_proj_weight", "down_proj_weight"):
        _attach(block.experts, leaf, *routed[leaf])
    bank_built = block.experts.prepare_scale_operands(
        gate_proj_weight=routed["gate_proj_weight"][0],
        up_proj_weight=routed["up_proj_weight"][0],
        down_proj_weight=routed["down_proj_weight"][0],
        gate_proj_scale=routed["gate_proj_weight"][1],
        up_proj_scale=routed["up_proj_weight"][1],
        down_proj_scale=routed["down_proj_weight"][1],
    )

    # ---- THE SHARED EXPERT at the bank's hidden size, and its own prep.
    shared = _shared_at_routed_operands()
    for leaf in ("gate_proj_weight", "up_proj_weight", "down_proj_weight"):
        _attach(block.shared_experts, leaf, *shared[leaf])
    shared_built = block.shared_experts.prepare_scale_operands(
        *block.shared_experts.scale_route_operands()
    )

    # ---- THE ROUTER's two parameters, in the orientation the seam consumes:
    # ``[H, E]`` for the weight, which ``noaux_tc_rmsnorm_router_topk`` reads its
    # expert count off (``router.py:1615``), and one bias per expert. The landed
    # fixture this copies is ``test_moe_path.py:1899-1905``.
    generator = torch.Generator().manual_seed(SEED_MOE_ROUTER)
    block.experts.router_weight = torch.nn.Parameter(
        (
            torch.randn(
                ROUTED_HIDDEN_SIZE, ROUTED_EXPERTS, generator=generator
            )
            * MOE_ROUTER_WEIGHT_SCALE
        ).to(torch.bfloat16),
        requires_grad=False,
    )
    block.experts.router_bias = torch.nn.Parameter(
        (
            torch.randn(ROUTED_EXPERTS, generator=generator)
            * MOE_ROUTER_BIAS_SCALE
        ).to(torch.bfloat16),
        requires_grad=False,
    )

    parameters = sum(int(p.numel()) for p in block.parameters() if p is not None)
    print(
        f"TINYFWD|moe_prep|bank_operands={bank_built}|shared_operands={shared_built}"
        f"|params={parameters}"
    )
    if bank_built != 4 or shared_built != 3:
        raise VacuousControlError(
            f"the load-time preps built {bank_built} bank and {shared_built} "
            f"shared operands; the two forwards look up 4 and 3"
        )
    if parameters >= MAX_PARAMETERS:
        raise VacuousControlError(
            f"the tiny MoE block holds {parameters} parameters, at or above the "
            f"adopted bound of {MAX_PARAMETERS}"
        )

    # ---- THE TWO ACTIVATION TENSORS. The block takes both because the router's
    # RMSNorm is fused inside the kernel while the experts consume the normalised
    # states; the section note on the forward is where that reading lives.
    pre_norm = (
        _fp8_grid_values(SEED_MOE_HIDDEN, TOKENS, ROUTED_HIDDEN_SIZE)
        * float(2.0**HIDDEN_SCALE_EXPONENT)
    ).to(torch.bfloat16)
    gamma = _ffn_gamma()
    eps = float(text_config.rms_norm_eps)
    normed = _ffn_norm(pre_norm, gamma, eps)

    # ---- THE ROUTER, EXECUTED, and its output checked for reproducibility before
    # it is used as a reference input. The forward calls it a third time on the
    # same operands, so a router that did not return the same affinities twice
    # would make every comparison below meaningless.
    _logits_a, _index_a, affinities = block.experts.route_tokens(
        pre_norm.unsqueeze(0), gamma, text_config
    )
    _logits_b, _index_b, again = block.experts.route_tokens(
        pre_norm.unsqueeze(0), gamma, text_config
    )
    if not torch.equal(affinities, again):
        raise VacuousControlError(
            "route_tokens returned different affinities for the same operands, so "
            "the reference cannot be built from one call and compared against "
            "another"
        )
    selected = int((affinities != 0).sum(dim=1).min())
    print(
        f"TINYFWD|moe_router|affinities={tuple(affinities.shape)}"
        f"|min_selected={selected}|top_k={int(text_config.num_experts_per_tok)}"
    )
    if tuple(affinities.shape) != (TOKENS, ROUTED_EXPERTS):
        raise ReferenceShapeError(
            f"route_tokens returned {tuple(affinities.shape)}, expected "
            f"{(TOKENS, ROUTED_EXPERTS)}"
        )
    if selected != int(text_config.num_experts_per_tok):
        raise VacuousControlError(
            f"a token carries {selected} nonzero router columns and the config "
            f"declares top-{int(text_config.num_experts_per_tok)}"
        )

    # ---- THE REFERENCE, the two halves separately so the add can be measured.
    limit = float(text_config.swiglu_limit)
    routed_reference = _routed_output(
        {**routed, "hidden": normed, "expert_affinities": affinities},
        mode=_POST_SCALE,
        gate_max=limit,
        gate_min=None,
        up_max=limit,
        up_min=-limit,
    )
    shared_reference = _dense_output(
        {**shared, "hidden": normed}, limit, -limit, limit
    )
    expected = routed_reference["out"] + shared_reference["out"]

    # ---- PRECONDITION: THE SHARED HALF IS NOT NEGLIGIBLE. If it were, "added
    # exactly once" would be inside the tolerance and controls A and B below would
    # both pass with the add missing.
    share = float(shared_reference["out"].abs().max() / expected.abs().max())
    print(
        f"TINYFWD|moe|limit={limit}|shared_share={share:.4f}"
        f"|routed_max={float(routed_reference['out'].abs().max()):.4f}"
        f"|shared_max={float(shared_reference['out'].abs().max()):.4f}"
    )

    # ---- CONTROLS A and B: the shared contribution enters ONCE. Omitting it and
    # doubling it must both fall outside the tolerance this item passes inside, or
    # the item cannot tell one add from none or from two.
    for name, variant in (
        ("shared half omitted", routed_reference["out"]),
        ("shared half added twice", expected + shared_reference["out"]),
    ):
        moved = not torch.allclose(variant, expected, rtol=RTOL, atol=ATOL)
        gap = float((variant - expected).abs().max() / expected.abs().max())
        print(
            f"TINYFWD|moe_control|branch={name}|outside_tolerance={moved}"
            f"|gap={gap:.4f}"
        )
        if not moved:
            raise VacuousControlError(
                f"with the {name} the result is still inside rtol={RTOL}, "
                f"atol={ATOL}; this item cannot tell the one add from the wrong "
                f"count"
            )

    # ---- THE REGISTERED ROUTE PREDICATE, around this item's own call.
    _reset_seam_counters()
    before = _read_seam_counters()
    got = block.forward(
        pre_norm,
        normed,
        router_gamma=gamma,
        text_config=text_config,
        quant_config=quant_config,
    )
    after = _read_seam_counters()
    # ONE MoE dispatch for the bank and THREE dense ones for the shared expert.
    # Both seams by name, so a block that ran the shared half through the expert
    # kernel, or the bank through three dense calls, fails instead of passing on a
    # total.
    _assert_route_predicate(
        "4 MoE block", {"blockwise_fp8_moe": 1, "blockwise_fp8_mm": 3}, before, after
    )

    if tuple(got.shape) != (TOKENS, ROUTED_HIDDEN_SIZE):
        raise ReferenceShapeError(
            f"the forward returned {tuple(got.shape)}, expected "
            f"{(TOKENS, ROUTED_HIDDEN_SIZE)}"
        )
    torch.testing.assert_close(got.float(), expected.float(), rtol=RTOL, atol=ATOL)

    # ---- CONTROL C: THE ROUTER READS THE PRE-NORM TENSOR. Handing the normalised
    # states in both positions normalises twice inside the fused kernel, which is
    # a different router and so a different set of affinities. It must move the
    # answer outside the tolerance, or this item would pass with the two arguments
    # swapped and the defect would be invisible.
    doubled = block.forward(
        normed,
        normed,
        router_gamma=gamma,
        text_config=text_config,
        quant_config=quant_config,
    )
    moved = not torch.allclose(doubled.float(), expected.float(), rtol=RTOL, atol=ATOL)
    gap = float((doubled.float() - expected.float()).abs().max() / expected.abs().max())
    print(
        f"TINYFWD|moe_control|branch=router fed the normalised tensor"
        f"|outside_tolerance={moved}|gap={gap:.4f}"
    )
    if not moved:
        raise VacuousControlError(
            f"normalising twice leaves the result inside rtol={RTOL}, atol={ATOL}; "
            f"this item cannot tell the pre-norm argument from the normalised one"
        )

    # ---- CONTROL D: A BLOCK WITH NO SHARED EXPERT returns the routed half and
    # reaches no dense projection at all. The landed add refuses such a call by
    # name, so the branch that avoids it is what this measures -- against the
    # reference control A already computed.
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    # BUILT with the keyword rather than assigned afterwards, so the config's own
    # validator sees the value this control depends on.
    dense_free_config = Glm5NextTextConfig(
        hidden_size=ROUTED_HIDDEN_SIZE,
        intermediate_size=ROUTED_INTERMEDIATE_SIZE,
        num_key_value_heads=NUM_KEY_VALUE_HEADS,
        n_routed_experts=ROUTED_EXPERTS,
        num_experts_per_tok=ROUTED_EXPERTS_PER_TOKEN,
        n_shared_experts=0,
    )
    bare = model_fp8.Glm5NextMoEBlock(dense_free_config, world_size=1, ep_degree=1)
    if getattr(bare, "shared_experts", None) is not None:
        raise VacuousControlError(
            "the block still built a shared expert at n_shared_experts=0, so this "
            "control does not exercise the branch it names"
        )
    for leaf in ("gate_proj_weight", "up_proj_weight", "down_proj_weight"):
        _attach(bare.experts, leaf, *routed[leaf])
    bare.experts.prepare_scale_operands(
        gate_proj_weight=routed["gate_proj_weight"][0],
        up_proj_weight=routed["up_proj_weight"][0],
        down_proj_weight=routed["down_proj_weight"][0],
        gate_proj_scale=routed["gate_proj_weight"][1],
        up_proj_scale=routed["up_proj_weight"][1],
        down_proj_scale=routed["down_proj_weight"][1],
    )
    bare.experts.router_weight = block.experts.router_weight
    bare.experts.router_bias = block.experts.router_bias

    _reset_seam_counters()
    before = _read_seam_counters()
    bare_got = bare.forward(
        pre_norm,
        normed,
        router_gamma=gamma,
        text_config=dense_free_config,
        quant_config=quant_config,
    )
    after = _read_seam_counters()
    _assert_route_predicate(
        "4 MoE block, no shared expert", {"blockwise_fp8_moe": 1}, before, after
    )
    print("TINYFWD|moe_control|branch=no shared expert|dense_dispatches=0")
    torch.testing.assert_close(
        bare_got.float(), routed_reference["out"].float(), rtol=RTOL, atol=ATOL
    )


# --------------------------------------------------------------------------- #
# ITEM 5's OWN GEOMETRY. The four MLP items above run at the MLP widths and read #
# no attention field; this one needs a whole attention geometry, and every width #
# below is ADOPTED from the landed DSA fixture (``test_dsa_layer.py:231-242``)   #
# together with the reason that fixture records for it. Adopted rather than      #
# re-chosen because those values are already measured against each seam's own    #
# validator, and re-choosing them would put this item in the business of         #
# re-deriving admissibility it does not own.                                     #
#                                                                              #
# THE THREE THAT MAY NOT MOVE, in that fixture's own words: ``index_head_dim``  #
# is PINNED at 128 because three seams refuse any other width (the Hadamard     #
# path is a 128-point transform); ``index_kpool`` stays a power of two, which    #
# ``can_run_dsa_index_expand`` requires; and ``qk_rope_head_dim`` is 0, the      #
# checkpoint's own value, which ``mla_sparse.py:1287`` admits by name.           #
# --------------------------------------------------------------------------- #
MLA_HIDDEN_SIZE = 256
MLA_HEADS = 4
MLA_Q_LORA_RANK = 128
#: An exact fit for the sparse seam's 128-wide partition tile, which selects the
#: UNTILED body (``mla_sparse.py:1420``) -- the same body production's 512 takes.
#: A ragged latent would move the tiled counters this item declares at zero.
MLA_KV_LORA_RANK = 128
MLA_QK_NOPE_HEAD_DIM = 64
MLA_QK_ROPE_HEAD_DIM = 0
MLA_V_HEAD_DIM = 64
MLA_INDEX_N_HEADS = 4
MLA_INDEX_HEAD_DIM = 128
MLA_INDEX_KPOOL = 4
#: Pools selected per query. ``can_run_dsa_topk_select`` needs ``0 < k < width``
#: STRICTLY, so two selected pools against eight candidates leaves room on both
#: sides of that bound.
MLA_TOPK_POOLS = 2
MLA_PAGE_SIZE = 4
MLA_PAGES = 8

#: ``35 = 8 * 4 + 3``: eight complete pools plus a FULL tail, which is the landed
#: fixture's own choice and its reason is arithmetic rather than taste --
#: ``35 % 4 == 3`` keeps every tail column a real token index, and the prefill
#: candidate width lands on 8, the granularity ``nisa.max8`` emits
#: (``test_dsa_layer.py:135-166`` records the whole derivation and the risk it
#: bounds). This item runs the PREFILL leg once; the decode leg is
#: ``inc-glm53f-042``'s and ``inc-glm53f-051``'s and both are landed.
MLA_TOKENS = 35

#: One generator, drawn in ``projection_widths()`` order. Successive draws from one
#: stream share no values, so the per-operand seeds items 1 to 4 use to keep two
#: separately-built operands apart are not needed here -- and the landed attention
#: fixtures both use exactly this form (``test_mla_decode.py:136``,
#: ``test_dsa_layer.py:1893``).
SEED_MLA = 5431

#: THE REGISTERED SOFTMAX SCALE, and it is the plan's value rather than this
#: file's: ``(qk_nope_head_dim + qk_rope_head_dim) ** -0.5``, registered at the
#: increment plan's ``inc-glm53f-054a`` block on the reference implementation's own
#: derivation (``modeling_glm5_next.py:1128`` with ``:1087``, applied once at
#: ``:1052``). It is ``0.125`` at this geometry and ``0.0625`` at the checkpoint's.
#:
#: THE NEAREST LANDED CONSTANT IS NOT ADOPTED, deliberately: two landed test files
#: compute this scale as ``kv_lora_rank ** -0.5``, which is the same number times
#: ``sqrt(2)`` on both fixture geometries, and ``inc-glm53f-109`` repairs them. A
#: control below recomputes this item's reference at that retired derivation and
#: requires it to fall OUTSIDE the tolerance, so this item is measurably sensitive
#: to which of the two it uses.
MLA_SOFTMAX_SCALE = float(
    (MLA_QK_NOPE_HEAD_DIM + MLA_QK_ROPE_HEAD_DIM) ** -0.5
)
MLA_RETIRED_SOFTMAX_SCALE = float(MLA_KV_LORA_RANK ** -0.5)


def _mla_text_config():
    """The checkpoint's config narrowed to item 5's geometry.

    ``dataclasses.replace`` on a default construction, which is the landed
    attention fixture's idiom (``test_dsa_layer.py:1002-1009``): every field this
    item does not name keeps the checkpoint's value, so a config drift reaches
    this item instead of being overwritten by it.

    ``select_k()`` is ``index_topk // index_kpool``, so the dial set here is
    ``index_topk`` -- the one upstream itself expresses in TOKENS -- and the pool
    granularity stays derived.
    """
    from dataclasses import replace

    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    return replace(
        Glm5NextTextConfig(),
        hidden_size=MLA_HIDDEN_SIZE,
        num_key_value_heads=NUM_KEY_VALUE_HEADS,
        num_attention_heads=MLA_HEADS,
        q_lora_rank=MLA_Q_LORA_RANK,
        kv_lora_rank=MLA_KV_LORA_RANK,
        qk_nope_head_dim=MLA_QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=MLA_QK_ROPE_HEAD_DIM,
        v_head_dim=MLA_V_HEAD_DIM,
        index_n_heads=MLA_INDEX_N_HEADS,
        index_head_dim=MLA_INDEX_HEAD_DIM,
        index_kpool=MLA_INDEX_KPOOL,
        index_topk=MLA_TOPK_POOLS * MLA_INDEX_KPOOL,
    )


def _mla_contract(x: torch.Tensor, weight_out_in: torch.Tensor) -> torch.Tensor:
    """One projection, from the CHECKPOINT-shaped ``[out, in]`` leaf, in fp32.

    The transpose happens here rather than being read out of
    ``_prepared_weight``, so the reference never consumes the implementation's own
    prepared cache -- the same rule items 1 to 4 follow when they dequantise their
    own operands instead of calling the seam's oracle.
    """
    return x.to(torch.float32) @ weight_out_in.to(torch.float32).t()


def _mla_latent_norm(
    x: torch.Tensor, gain: torch.Tensor, eps: float
) -> torch.Tensor:
    """RMSNorm on a latent: ``x / sqrt(mean(x**2) + eps) * gain``, fp32, no cast.

    ``eps`` is resolved from the config by the caller, never written here, which is
    the acceptance's own requirement: a wrong call-site epsilon must redden a
    comparison rather than cancel out on both sides.
    """
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + float(eps)) * gain.to(torch.float32)


def _mla_attention_fixture(*, seed: int = SEED_MLA):
    """One MLA attention module, every leaf materialised and both preps run.

    Returns ``(attention, raw, gains, config)`` where ``raw`` holds the five
    projection leaves in the checkpoint's ``[out, in]`` orientation and ``gains``
    the two latent-norm gains -- the operands the reference reads.

    THE WEIGHTS ARE ``randn * in_features ** -0.5``, the scaling both landed
    attention fixtures use (``test_mla_decode.py:139-143``), so activations stay
    order one instead of growing until the comparison measures overflow. The fp8
    ``1/8``-grid conditioning items 1 to 4 need is NOT used here and its absence is
    deliberate: those items compare a blockwise-quantised matmul, where a signed
    draw makes every dot product a near-cancelling sum, while this path is fp32
    end to end and has no block scales to line up.
    """
    model_fp8 = _impl()
    cfg = _mla_text_config()
    attention = model_fp8.Glm5NextMLAAttention(cfg)
    gen = torch.Generator().manual_seed(int(seed))

    raw: dict = {}
    for name, in_features, out_features in attention.projection_widths():
        weight = torch.randn(
            out_features, in_features, generator=gen, dtype=torch.float32
        ) * (in_features ** -0.5)
        raw[name] = weight
        setattr(attention, f"{name}_weight", torch.nn.Parameter(weight))

    gains: dict = {}
    for gain_name, width in (
        ("q_a_layernorm_weight", int(cfg.q_lora_rank)),
        ("kv_a_layernorm_weight", int(cfg.kv_lora_rank)),
    ):
        gain = 1.0 + torch.randn(width, generator=gen, dtype=torch.float32) * 0.05
        gains[gain_name] = gain
        setattr(attention, gain_name, torch.nn.Parameter(gain))

    prepared = attention.prepare_projection_weights()
    absorbed = attention.prepare_absorb_weights()
    if (prepared, absorbed) != (5, 2):
        raise VacuousControlError(
            f"the load-time preps built {prepared} projection(s) and {absorbed} "
            f"absorb operand(s); this geometry declares five and two"
        )
    _materialise_mla_indexer(attention.indexer, gen)

    parameters = sum(int(p.numel()) for p in attention.parameters())
    print(f"TINYFWD|mla_fixture|parameters={parameters}|bound={MAX_PARAMETERS}"
          f"|heads={MLA_HEADS}|latent={MLA_KV_LORA_RANK}|tokens={MLA_TOKENS}")
    if parameters >= MAX_PARAMETERS:
        raise VacuousControlError(
            f"the attention module carries {parameters} parameters, at or past "
            f"the adopted tiny bound of {MAX_PARAMETERS}"
        )
    # The constraint set's head-width bound, MEASURED on this geometry's own head
    # widths rather than declared: the four MLP items above read no head field, so
    # this is the first item where the bound means anything.
    widths = {
        "qk_nope_head_dim": int(cfg.qk_nope_head_dim),
        "qk_rope_head_dim": int(cfg.qk_rope_head_dim),
        "v_head_dim": int(cfg.v_head_dim),
        "index_head_dim": int(cfg.index_head_dim),
    }
    print(f"TINYFWD|mla_fixture|head_widths={sorted(widths.items())}"
          f"|bound={MAX_HEAD_DIM}")
    over = {name: width for name, width in widths.items() if width > MAX_HEAD_DIM}
    if over:
        raise VacuousControlError(
            f"{sorted(over.items())} exceeds the adopted tiny bound "
            f"head_dim <= {MAX_HEAD_DIM}"
        )
    return attention, raw, gains, cfg


def _materialise_mla_indexer(indexer, gen: torch.Generator) -> None:
    """The indexer's seven leaves, then its own load-time prep.

    The attribute name comes from the indexer's OWN ``PROJECTION_PARAMETERS`` map
    rather than from a suffix rule, because that map is not uniform -- three sites
    carry a ``_weight`` suffix and ``index_kpool_compress_gate`` does not, so
    guessing would materialise three of four (``test_dsa_layer.py:1591-1593``).
    """
    for name, in_features, out_features in indexer.projection_widths():
        weight = torch.randn(
            out_features, in_features, generator=gen, dtype=torch.float32
        ) * (in_features ** -0.5)
        setattr(
            indexer, indexer.PROJECTION_PARAMETERS[name], torch.nn.Parameter(weight)
        )
    head_dim = int(indexer.index_head_dim)
    pool = int(indexer.index_kpool)
    indexer.k_norm_weight = torch.nn.Parameter(
        1.0 + torch.randn(head_dim, generator=gen, dtype=torch.float32) * 0.05
    )
    indexer.k_norm_bias = torch.nn.Parameter(
        torch.randn(head_dim, generator=gen, dtype=torch.float32) * 0.02
    )
    # A bf16 checkpoint leaf, stored in the checkpoint's dtype: the consumer casts
    # it per call, so pre-casting here would let the fixture agree with a cast the
    # implementation still has to do.
    indexer.index_kpool_compress_ape = torch.nn.Parameter(
        (torch.randn(pool, head_dim, generator=gen, dtype=torch.float32) * 0.1).to(
            torch.bfloat16
        ),
        requires_grad=False,
    )
    prepared = indexer.prepare_projection_weights()
    if prepared != 4:
        raise VacuousControlError(
            f"the indexer prep built {prepared} projections, not four"
        )


def _mla_selection_operands() -> dict:
    """The operands the selection stage needs, each derived from its own rule.

    ``slot_mapping`` is pool-granular: a position carries its pool's id where a
    pool COMPLETES and ``-1`` where it does not, which is how a position says "my
    window is not a whole pool" and is steered to the trash row rather than
    dropped. ``seq_lens`` carries one length per score ROW and the rows are
    tokens, so each token's own context length is its own -- which makes the tail
    each token's own incomplete pool.
    """
    slots = torch.full((MLA_TOKENS,), -1, dtype=torch.int32)
    for position in range(MLA_TOKENS):
        if (position + 1) % MLA_INDEX_KPOOL == 0:
            slots[position] = position // MLA_INDEX_KPOOL
    candidates = MLA_TOKENS // MLA_INDEX_KPOOL
    rows = MLA_PAGES * MLA_PAGE_SIZE
    if candidates <= MLA_TOPK_POOLS:
        raise VacuousControlError(
            f"{candidates} candidate pool(s) is not more than the "
            f"{MLA_TOPK_POOLS} selected, and the indexer serves the bypass "
            f"regime there instead of selecting"
        )
    if candidates > rows - 1:
        raise VacuousControlError(
            f"{candidates} candidates leaves no trash row above them in {rows} "
            f"pooled-key rows"
        )
    return {
        "slot_mapping": slots,
        "seq_lens": torch.arange(1, MLA_TOKENS + 1, dtype=torch.int32),
        "candidates": candidates,
        "pool_rows": rows,
    }


def _mla_pool_cache() -> torch.Tensor:
    """The pooled-key store, ``[rows, index_head_dim]`` bf16, written in place."""
    return torch.zeros(
        MLA_PAGES * MLA_PAGE_SIZE, MLA_INDEX_HEAD_DIM, dtype=torch.bfloat16
    )


def _mla_latent_cache(attention) -> torch.Tensor:
    """The latent cache at the layer's OWN declared spec, one latent per token.

    ``head_size`` is read off the module rather than typed, because the module
    derives it from two config fields and a cache built from a literal would stop
    tracking that derivation.

    FLOAT32 RATHER THAN THE SPEC'S BF16, and the choice is the caller's: the shape
    check is the only thing ``attend()`` asserts about this tensor, and both the
    path and the reference write the same latents through the same dtype, so bf16
    would round both sides identically and prove nothing extra while spending a
    tenth of the tolerance. The landed DSA fixture makes the same choice
    (``test_dsa_layer.py:1984-1986``); the bf16 spec dtype is exercised by
    ``inc-glm53f-042``'s own items.
    """
    return torch.zeros(
        MLA_TOKENS, attention.NUM_LATENT_KV_HEADS, int(attention.head_size),
        dtype=torch.float32,
    )


def _mla_dense_reference(
    attention,
    raw: dict,
    gains: dict,
    normed: torch.Tensor,
    latent_cache: torch.Tensor,
    topk_indices: torch.Tensor,
    *,
    softmax_scale: float,
    honour_sentinels: bool = True,
) -> torch.Tensor:
    """Dense MLA attention in torch, from the same weights. ``[tokens, hidden]``.

    THE REFERENCE IS THE DENSE FORM, WHICH IS NOT THE FORM UNDER TEST, and that is
    the whole value of it. The reference implementation this campaign compares
    against never absorbs: it materialises full-width keys and values through
    ``kv_b_proj`` and contracts them against a query at the head width
    (``modeling_glm5_next.py:1145-1152``, ``:1164``). The fork's path absorbs
    ``kv_b_proj``'s two halves into the query and into the output instead. The two
    are the same dot product rewritten, so a dense reference certifies the absorb
    algebra as well as the composition -- and it reads neither ``W_UK`` nor
    ``W_UV``, so a permutation error in the split cannot cancel out.

    THE SENTINEL SEMANTICS ARE UPSTREAM'S DESIGN, named explicitly because they
    change the function: a selected-row column holding ``-1`` carries NO token, so
    it takes no probability mass, and a row that is wholly sentinel produces
    zeros. Upstream initialises its whole index buffer to ``-1`` and writes the
    ``-1``-bearing expansion straight into it
    (``sparse_attn_indexer_kpool.py:435``, ``:606``); the fork's seam masks the
    value rather than reading it and admits ``-1`` alone below zero
    (``mla_sparse.py:1394-1405``). ``honour_sentinels=False`` is the FAILING
    CONTROL for that reading: it points every sentinel column at cache row 0, the
    in-range filler upstream can only afford because it also clamps the row's
    length and zeroes the output afterwards.

    The gather-then-softmax structure is mirrored rather than replaced by a mask
    over the whole cache: softmax normalises over the columns it is GIVEN, so a
    column that appeared twice would take twice the mass, and only the gathered
    form reproduces that.

    MUTATES ``latent_cache``, exactly as ``attend()`` does and in the same order --
    the write lands before the read, so a token attends to its own latent. Each
    caller therefore hands this function its own cache. The start position is 0
    for every caller in this item, so the written slots ARE the whole read prefix
    and no start argument is threaded here; ``attend()``'s own items cover a
    non-zero start.
    """
    tokens = int(normed.shape[0])
    heads = int(attention.num_attention_heads)
    nope = int(attention.qk_nope_head_dim)
    vdim = int(attention.v_head_dim)
    eps = float(attention.rms_norm_eps)
    x = normed.to(torch.float32)

    q_latent = _mla_latent_norm(
        _mla_contract(x, raw["q_a_proj"]), gains["q_a_layernorm_weight"], eps
    )
    query = _mla_contract(q_latent, raw["q_b_proj"]).reshape(tokens, heads, nope)
    kv_latent = _mla_latent_norm(
        _mla_contract(x, raw["kv_a_proj_with_mqa"]),
        gains["kv_a_layernorm_weight"],
        eps,
    )

    latent_cache[0:tokens, 0, :] = kv_latent.to(latent_cache.dtype)
    c_kv = latent_cache[:tokens, 0, :].to(torch.float32)

    # THE DENSE EXPANSION. One weight, both halves, split on the boundary the
    # reference implementation splits on.
    key_value = _mla_contract(c_kv, raw["kv_b_proj"]).reshape(
        tokens, heads, nope + vdim
    )
    key_nope = key_value[..., :nope]
    value = key_value[..., nope:]

    index = topk_indices.to(torch.int64)
    keep = index >= 0
    rows = index.clamp(min=0)
    out = torch.empty(tokens, heads, vdim, dtype=torch.float32)
    for token in range(tokens):
        gathered_key = key_nope[rows[token]]        # [K, H, nope]
        gathered_value = value[rows[token]]         # [K, H, v]
        scores = torch.einsum("hd,khd->hk", query[token], gathered_key)
        if honour_sentinels:
            scores = scores.masked_fill(~keep[token], float("-inf"))
        # ``nan_to_num`` because softmax of an all -inf row is NaN rather than the
        # zeros the kernels write for a wholly-sentinel row.
        weights = torch.nan_to_num(
            torch.softmax(scores * float(softmax_scale), dim=-1)
        )
        out[token] = torch.einsum("hk,khv->hv", weights, gathered_value)

    flat = out.reshape(tokens, heads * vdim)
    return _mla_contract(flat, raw["o_proj"]).to(normed.dtype)


def _mla_outside_tolerance(label: str, moved: torch.Tensor, base: torch.Tensor) -> None:
    """One clamp-style control: ``moved`` must fall OUTSIDE this item's band.

    The file's declared control form -- recompute the whole reference with one
    branch changed and require the result to leave the tolerance the item passes
    inside, so a fixture that stopped discriminating fails as a control instead of
    passing as an item.
    """
    outside = not torch.allclose(moved.float(), base.float(), rtol=RTOL, atol=ATOL)
    gap = float((moved.float() - base.float()).abs().max() / base.abs().max())
    print(f"TINYFWD|mla_control|branch={label}|outside_tolerance={outside}"
          f"|gap={gap:.6f}")
    if not outside:
        raise VacuousControlError(
            f"{label}: the change leaves the reference inside rtol={RTOL}, "
            f"atol={ATOL}, so this item cannot tell the two apart"
        )


# --------------------------------------------------------------------------- #
# ITEM 5 of 7 -- ``Glm5NextMLAAttention.forward``.                             #
# Certifying component: ``model_fp8.Glm5NextMLAAttention.forward``.            #
#                                                                              #
# WHAT IT CERTIFIES. This forward is the attention half's whole composition:    #
# the query latent the indexer contracts, the indexer that selects rows, and    #
# ``attend()``, which ends in the output projection. So the item runs it once   #
# and compares against a DENSE MLA reference built from the same raw weights --  #
# the form the reference implementation itself computes, which the absorbed     #
# path is an exact rewrite of. Nothing here re-certifies a callee: the absorb    #
# split is ``inc-glm53f-042``'s, the selection chain ``inc-glm53f-051``'s.       #
#                                                                              #
# THE INDEXER IS EXECUTED RATHER THAN IMITATED, item 4's convention on the      #
# router applied to the other selector: its indices are an INPUT to the         #
# reference. Re-deriving them would put this item in the business of certifying  #
# the indexer -- eight seams and a Hadamard transform -- and would make the      #
# comparison hostage to a near-tie flipping one query's pool set. The item does  #
# check that two calls on the same operands return the same indices, because the #
# forward makes a THIRD call and the reference stands on the first.              #
#                                                                              #
# THE PREFILL LEG ONLY, once, which is what the acceptance asks for: one        #
# forward, one call. The decode leg's tail ring is ``inc-glm53f-042``'s and      #
# ``inc-glm53f-051``'s and both are landed with their own items.                 #
#                                                                              #
# CAUSALITY IS NOT MEASURED HERE, disclosed rather than implied: a query's       #
# selected pools may sit past its own position, because bounding them is the     #
# indexer's business and ``inc-glm53f-051``'s acceptance owns it. Both sides of  #
# this comparison read the same indices, so the composition is what is measured. #
# --------------------------------------------------------------------------- #
def test_tiny_mla_attention_forward_matches_the_reference() -> None:
    """The MLA attention forward equals dense MLA attention on the same weights.

    D1.4 certifying component: ``Glm5NextMLAAttention.forward`` -- the three-call
    composition, the registered softmax scale it is handed, and the sentinel
    semantics of the indices it passes through untouched.
    """
    attention, raw, gains, cfg = _mla_attention_fixture()
    operands = _mla_selection_operands()
    normed = (
        torch.randn(
            MLA_TOKENS, MLA_HIDDEN_SIZE,
            generator=torch.Generator().manual_seed(SEED_MLA + 1),
            dtype=torch.float32,
        )
        * 0.5
    )

    # ---- THE INDEXER, EXECUTED. Twice, on two of its OWN pooled-key stores, so
    # the reference's indices are measured to be the ones a third call will
    # produce rather than assumed to be.
    q_latent = attention.project_query_latent(normed)
    selections = []
    for _ in range(2):
        selections.append(
            attention.indexer(
                normed,
                q_latent,
                _mla_pool_cache(),
                operands["seq_lens"],
                max_seq_len=MLA_TOKENS,
                page_size=MLA_PAGE_SIZE,
                slot_mapping=operands["slot_mapping"],
            )
        )
    topk_indices, repeat = selections
    sentinels = int((topk_indices < 0).sum())
    print(f"TINYFWD|mla_selection|shape={tuple(topk_indices.shape)}"
          f"|sentinel_columns={sentinels}"
          f"|max_row={int(topk_indices.max())}|tokens={MLA_TOKENS}")
    if not torch.equal(topk_indices, repeat):
        raise VacuousControlError(
            "two indexer calls on identical operands returned different "
            "selections, so the reference cannot stand on the first while the "
            "forward makes a third"
        )
    if sentinels == 0:
        raise VacuousControlError(
            "no selected-row column carries the -1 sentinel, so the sentinel "
            "control below would measure nothing. The expansion pads its raw "
            f"width up to a multiple of the sparse seam's key chunk, and this "
            f"geometry emits {tuple(topk_indices.shape)}"
        )

    # ---- THE QUERY LATENT AGREES WITH THE REFERENCE'S OWN DERIVATION. Free, and
    # it settles that the value the indexer selected on is the value the reference
    # would have computed -- so the two chains part at the selection and nowhere
    # earlier.
    reference_latent = _mla_latent_norm(
        _mla_contract(normed, raw["q_a_proj"]),
        gains["q_a_layernorm_weight"],
        float(cfg.rms_norm_eps),
    )
    torch.testing.assert_close(
        q_latent.float(), reference_latent.float(), rtol=RTOL, atol=ATOL
    )

    expected = _mla_dense_reference(
        attention, raw, gains, normed, _mla_latent_cache(attention), topk_indices,
        softmax_scale=MLA_SOFTMAX_SCALE,
    )

    # ---- THE REGISTERED ROUTE PREDICATE, around this item's own call.
    latent_cache = _mla_latent_cache(attention)
    _reset_seam_counters()
    before = _read_seam_counters()
    got = attention.forward(
        normed,
        latent_cache=latent_cache,
        pool_cache=_mla_pool_cache(),
        seq_lens=operands["seq_lens"],
        start_position=0,
        softmax_scale=MLA_SOFTMAX_SCALE,
        max_seq_len=MLA_TOKENS,
        page_size=MLA_PAGE_SIZE,
        slot_mapping=operands["slot_mapping"],
    )
    after = _read_seam_counters()
    # EVERY FIGURE IS READ OFF A LANDED, GREEN TABLE FOR THIS SAME COMPOSITION
    # rather than counted by eye. Nine projections per layer per phase is
    # ``test_dsa_layer.py:479-506``'s closed form, term by term: one for the query
    # latent, four inside the indexer's projection stage, three inside
    # ``project_query_and_latent`` (its first statement is the nested
    # ``project_query_latent``) and one for the output. The DSA figures are that
    # file's prefill column (``:297-307``): the pooling and the query rotation
    # share one module and so read two, and the tail-ring update is decode-only
    # and reads zero. Two absorbs and one sparse call are ``attend()``'s own.
    # The two tiled sparse counters are DECLARED ZEROS at this geometry -- the
    # latent is an exact 128 fit and 128 selected rows is inside one moving tile --
    # so they are registered and read rather than left out of the population.
    _assert_route_predicate(
        "5 MLA attention",
        {
            "mla_projection": 9,
            "mla_absorb": 2,
            "mla_sparse": 1,
            "dsa_kpool_hadamard": 2,
            "dsa_paged_gather": 1,
            "dsa_score_gemm": 1,
            "dsa_topk_select": 1,
            "dsa_index_expand": 1,
        },
        before,
        after,
    )

    if tuple(got.shape) != (MLA_TOKENS, MLA_HIDDEN_SIZE):
        raise ReferenceShapeError(
            f"the forward returned {tuple(got.shape)}, expected "
            f"{(MLA_TOKENS, MLA_HIDDEN_SIZE)}"
        )
    print(f"TINYFWD|mla_compare|max_abs_diff="
          f"{float((got.float() - expected.float()).abs().max()):.10g}"
          f"|peak_reference={float(expected.abs().max()):.10g}")
    torch.testing.assert_close(got.float(), expected.float(), rtol=RTOL, atol=ATOL)

    # ---- THE CACHE WAS WRITTEN, and by the path rather than only by the
    # reference. Every token's latent lands in its own slot, so a forward that
    # skipped the write would attend to zeros and this reading is where that shows
    # before the comparison above is trusted.
    written = latent_cache[:MLA_TOKENS, 0, :]
    if int((written.abs().sum(dim=-1) == 0).sum()) != 0:
        raise VacuousControlError(
            "the forward left at least one cache slot all zero, so the latents "
            "it attended to are not the ones it computed"
        )

    # ---- CONTROL A: THE SENTINEL COLUMNS CARRY NO MASS. Pointing them at a real
    # cache row is upstream's other option and it is a DIFFERENT function; the
    # reference recomputed that way must leave the band.
    _mla_outside_tolerance(
        "sentinel columns filled with cache row 0",
        _mla_dense_reference(
            attention, raw, gains, normed, _mla_latent_cache(attention),
            topk_indices, softmax_scale=MLA_SOFTMAX_SCALE,
            honour_sentinels=False,
        ),
        expected,
    )

    # ---- CONTROL B: THE REGISTERED SCALE IS THE ONE THAT MATTERS. Recomputed at
    # the retired ``kv_lora_rank ** -0.5`` derivation two landed files still carry,
    # which is this item's scale times sqrt(2).
    _mla_outside_tolerance(
        f"softmax scale {MLA_RETIRED_SOFTMAX_SCALE:.7f} rather than the registered "
        f"{MLA_SOFTMAX_SCALE:.7f}",
        _mla_dense_reference(
            attention, raw, gains, normed, _mla_latent_cache(attention),
            topk_indices, softmax_scale=MLA_RETIRED_SOFTMAX_SCALE,
        ),
        expected,
    )

    # ---- CONTROL C: THE FORWARD READS THE PREPARED WEIGHTS AND DOES NOT REBUILD
    # THEM. Called before the load-time prep it must refuse by name, and no seam
    # may move -- which is what tells a refusal from a silent per-call rebuild.
    model_fp8 = _impl()
    bare = model_fp8.Glm5NextMLAAttention(cfg)
    for name in raw:
        setattr(bare, f"{name}_weight", torch.nn.Parameter(raw[name]))
    for gain_name, gain in gains.items():
        setattr(bare, gain_name, torch.nn.Parameter(gain))
    _reset_seam_counters()
    unprepared_before = _read_seam_counters()
    with pytest.raises(ValueError, match="prepare_projection_weights"):
        bare.forward(
            normed,
            latent_cache=_mla_latent_cache(attention),
            pool_cache=_mla_pool_cache(),
            seq_lens=operands["seq_lens"],
            start_position=0,
            softmax_scale=MLA_SOFTMAX_SCALE,
            max_seq_len=MLA_TOKENS,
            page_size=MLA_PAGE_SIZE,
            slot_mapping=operands["slot_mapping"],
        )
    unprepared_after = _read_seam_counters()
    moved = {
        seam: unprepared_after[seam][0] - unprepared_before[seam][0]
        for seam in _SEAMS
        if unprepared_after[seam][0] != unprepared_before[seam][0]
    }
    print(f"TINYFWD|mla_control|branch=forward before the projection prep"
          f"|refused=True|seams_that_moved={sorted(moved.items())}")
    if moved:
        raise VacuousControlError(
            f"the refusal ran after {sorted(moved.items())} dispatched, so it is "
            f"not the first thing the forward does with an unprepared weight"
        )

    # ---- CONTROL D: THE ABSORB OPERANDS ARE READ, NOT REBUILT. Same reading one
    # prep later: projections prepared, absorb split not run, refusal by name.
    half = model_fp8.Glm5NextMLAAttention(cfg)
    for name in raw:
        setattr(half, f"{name}_weight", torch.nn.Parameter(raw[name]))
    for gain_name, gain in gains.items():
        setattr(half, gain_name, torch.nn.Parameter(gain))
    half.prepare_projection_weights()
    _materialise_mla_indexer(half.indexer, torch.Generator().manual_seed(SEED_MLA))
    with pytest.raises(
        model_fp8.Glm5NextMLADecodeError, match="prepare_absorb_weights"
    ):
        half.forward(
            normed,
            latent_cache=_mla_latent_cache(attention),
            pool_cache=_mla_pool_cache(),
            seq_lens=operands["seq_lens"],
            start_position=0,
            softmax_scale=MLA_SOFTMAX_SCALE,
            max_seq_len=MLA_TOKENS,
            page_size=MLA_PAGE_SIZE,
            slot_mapping=operands["slot_mapping"],
        )
    print("TINYFWD|mla_control|branch=forward before the absorb split|refused=True")
