# SPDX-License-Identifier: Apache-2.0
"""Acceptance for `inc-glm53f-054a` -- the seven forwards of the 45-layer text model.

**SEVEN ITEMS, ONE PER REPLACED ``forward``, and no ``parametrize`` decorator in this
file** (campaign rule D1.2). Each item runs one forward once on the tiny config and
compares its output against a torch reference built from the same weights. Each item
names the component whose behaviour it certifies (D1.4).

TWO BANDS, EACH ADOPTED AND NEITHER MINTED HERE (P9). ``rtol=1e-2, atol=1e-5`` is the
end-to-end criterion's own pair and governs every comparison whose value crossed no
eight-expert sum. ``rtol=3e-2, atol=1e-5`` -- ``MOE_RTOL``/``MOE_ATOL`` below -- governs
the MoE-seam comparisons, and is the pair ``inc-glm53f-027`` already landed on this same
layer (``test_moe_path.py:181-182``). Each site's band is named at the site, and the
constraint set below is the end-to-end criterion's, carried unchanged.

A CONTROL CARRIES ITS OWN ITEM'S BAND. A control must drive the output OUTSIDE the band
its item passes inside, so a control left on the tighter band under a wider acceptance
would certify less than the acceptance requires.

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
import os
from pathlib import Path

import pytest
import torch

from vllm_neuron.functional.moe.blockwise_fp8_retile import BLOCK_QUANT_SIZE, TILE_SIZE
from vllm_neuron.model.glm5_next.weight_loaders_fp8 import (
    compensate_block_scales,
    downscale_fp8_weight_bytes,
)

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
#: < 32 M parameters. Asserted per item against the module actually built, so the
#: bound is measured rather than declared.
#:
#: THE BOUND MOVED FROM 10 M, and it is the only member of the adopted constraint set
#: that moved. The production router refuses any top-k that is not 8 and any expert
#: count below 8 (``router.py:1037-1051``), so the bank needs at least 8 experts. At the
#: expert width this file's clamp controls are built from, 8 experts cost 12.6 M
#: parameters and 16 cost 25.2 M, so NO layout satisfies both the router and 10 M. The
#: alternative was to narrow the expert intermediate extent, which deletes the very
#: blocks the controls live in. Counted in
#: ``increments/probe-054a-topk-feasibility-mac-r1-20260908T213254Z.out``, ruled in
#: ``approvals/LEAD-LOG.md`` §738 against the constraint set at
#: ``design/increment-plan.md:1147``.
MAX_PARAMETERS = 32_000_000

#: A whole number of ``TILE_SIZE`` rows -- the dense seam tiles ``M`` over the PSUM
#: partition axis and does not pad (``blockwise_fp8_mm.py:239-245``), so padding is
#: the caller's, and a tiny case that ignored it would exercise the refusal rather
#: than the numerics.
TOKENS = 128

RTOL = 1e-2
ATOL = 1e-5

#: The MoE-seam pair, for the comparisons whose value carries an EIGHT-EXPERT SUM.
#: A LANDED PRECEDENT, NOT A NUMBER MINTED HERE: ``inc-glm53f-027`` compares this same
#: MoE layer against a pure-torch reference at exactly this pair -- see
#: ``test/vllm_neuron/model/glm5_next/test_moe_path.py:181-182`` and the plan's ``:1553``.
#: Order named inline, per design law D3: the pin holds two tolerance maps in OPPOSITE
#: orders, so a bare pair is ambiguous. This one is ``(rtol, atol)``.
#:
#: THE PRECEDENT IS THE WHOLE GROUND HERE. A bf16 unit-roundoff argument agrees with this
#: pair, but its premise -- which dtype the seam accumulates in -- is OWED and unmeasured,
#: so it is not offered as a reason. No hardware run has passed these sites at any band.
MOE_RTOL = 3e-2
MOE_ATOL = 1e-5

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
#: EVERY ACCESSOR IS NAMED -- READ AND RESET BOTH -- AND NOTHING IS DERIVED, and
#: that changed on a counterexample rather than on taste (``inc-glm53f-054a``,
#: repair R; ``probe-054a-counter-population-r1``, 21 modules and 24 families read
#: from the package's own source). ``inc-glm53f-054e`` c6 re-read the same source
#: with the control's own regex and finds 22 modules and 26 families: the extra two
#: are ``inc-glm53f-103``'s causal bound and causal sentinel, and grant 181's run
#: names them itself (``TINYFWD|counter_population|found=26|claimed=24``). The
#: convention IS ``"reset_" + read`` in twenty-one of the twenty-two modules, and
#: ``functional/moe/router.py`` breaks it: its
#: reader is ``noaux_tc_dispatch_counters`` and its reset is
#: ``reset_noaux_tc_counters``, so a derived name is an ``AttributeError`` on the
#: first line of every item's route predicate. Naming both costs one string per row
#: and check 5 of the static gate asserts each name exists.
#:
#: THE POPULATION IS THE WHOLE ``vllm_neuron.functional`` TREE, not the subset these
#: seven forwards were expected to reach, because the registered predicate says
#: "every seam this campaign owns" and every module below is named in this
#: campaign's plan. Ten families were unregistered before repair R and one of them
#: is dispatched by a forward this file already tested: ``route_tokens`` enters
#: ``functional/moe/router.py`` (``model_fp8.py:1489-1494``), whose family
#: increments once per call (``router.py:1664``). So item 4's fallback aggregate was
#: reading thirteen of the fourteen seams its own forward could reach, and the
#: earlier disclosure that "the router's own seam defines no dispatch counters" was
#: false of the tree.
#:
#: Naming the reader rather than discovering it stays deliberate:
#: ``test_dsa_layer.py:380`` discovers the pair by scanning for the
#: ``_dispatch_counters`` suffix and asserts exactly one pair per module, which is
#: true of every module below EXCEPT ``mla_sparse`` (three families),
#: ``kda/chunked_recurrence`` (two) and, since ``inc-glm53f-054e`` c6 registered it,
#: ``dsa/causal_bound`` (two). That file already knows it: its own
#: ``_causal_bound_apis`` asserts two readers and two resets there and asserts the
#: single-pair helper REFUSES the module (``test_dsa_layer.py:3665-3692``). The two
#: coverage controls below keep the naming
#: honest: one refuses a registered module that grows a family no row claims, the
#: other refuses a family anywhere in the package that no row claims at all.
_SEAM_REGISTRY = {
    "blockwise_fp8_mm": (
        "vllm_neuron.functional.blockwise_fp8_mm",
        "dispatch_counters", "reset_dispatch_counters"),
    "blockwise_fp8_moe": (
        "vllm_neuron.functional.moe.moe_blockwise_fp8",
        "dispatch_counters", "reset_dispatch_counters"),
    "noaux_tc_router": (
        "vllm_neuron.functional.moe.router",
        "noaux_tc_dispatch_counters", "reset_noaux_tc_counters"),
    "mla_projection": (
        "vllm_neuron.functional.attention.mla_projections",
        "mla_projection_dispatch_counters",
        "reset_mla_projection_dispatch_counters"),
    "mla_absorb": (
        "vllm_neuron.functional.attention.mla_absorb",
        "mla_absorb_dispatch_counters", "reset_mla_absorb_dispatch_counters"),
    "mla_sparse": (
        "vllm_neuron.functional.attention.mla_sparse",
        "mla_sparse_dispatch_counters", "reset_mla_sparse_dispatch_counters"),
    "mla_sparse_tiled": (
        "vllm_neuron.functional.attention.mla_sparse",
        "mla_sparse_tiled_dispatch_counters",
        "reset_mla_sparse_tiled_dispatch_counters"),
    "mla_sparse_row_tiled": (
        "vllm_neuron.functional.attention.mla_sparse",
        "mla_sparse_row_tiled_dispatch_counters",
        "reset_mla_sparse_row_tiled_dispatch_counters"),
    "dsa_kpool_hadamard": (
        "vllm_neuron.functional.dsa.kpool_hadamard",
        "kpool_hadamard_dispatch_counters",
        "reset_kpool_hadamard_dispatch_counters"),
    "dsa_paged_gather": (
        "vllm_neuron.functional.dsa.paged_gather",
        "paged_gather_dispatch_counters", "reset_paged_gather_dispatch_counters"),
    "dsa_score_gemm": (
        "vllm_neuron.functional.dsa.score_gemm",
        "score_gemm_dispatch_counters", "reset_score_gemm_dispatch_counters"),
    "dsa_topk_select": (
        "vllm_neuron.functional.dsa.topk_select",
        "topk_select_dispatch_counters", "reset_topk_select_dispatch_counters"),
    "dsa_index_expand": (
        "vllm_neuron.functional.dsa.index_expand",
        "index_expand_dispatch_counters", "reset_index_expand_dispatch_counters"),
    "dsa_decode_tail_update": (
        "vllm_neuron.functional.dsa.decode_tail_update",
        "decode_tail_dispatch_counters", "reset_decode_tail_dispatch_counters"),
    "dsa_ragged_pack": (
        "vllm_neuron.functional.dsa.ragged_pack",
        "ragged_pack_dispatch_counters", "reset_ragged_pack_dispatch_counters"),
    "dsa_causal_fill": (
        "vllm_neuron.functional.dsa.causal_fill",
        "causal_fill_dispatch_counters", "reset_causal_fill_dispatch_counters"),
    # ``inc-glm53f-054e`` c6: the two families ``inc-glm53f-103`` added, which no row
    # claimed. Grant 181's run read them as the gap itself --
    # ``TINYFWD|counter_population|found=26|claimed=24`` -- and refused all seven
    # items before any forward ran. One row per family, because this module holds
    # two entry points with an instance each (``causal_bound.py:187-208``).
    "dsa_causal_bound": (
        "vllm_neuron.functional.dsa.causal_bound",
        "causal_bound_dispatch_counters", "reset_causal_bound_dispatch_counters"),
    "dsa_causal_sentinel": (
        "vllm_neuron.functional.dsa.causal_bound",
        "causal_sentinel_dispatch_counters",
        "reset_causal_sentinel_dispatch_counters"),
    "kda_chunked_recurrence": (
        "vllm_neuron.functional.kda.chunked_recurrence",
        "dispatch_counters", "reset_dispatch_counters"),
    "kda_chunked_recurrence_inter": (
        "vllm_neuron.functional.kda.chunked_recurrence",
        "inter_dispatch_counters", "reset_inter_dispatch_counters"),
    "kda_decode_state": (
        "vllm_neuron.functional.kda.decode_state",
        "decode_dispatch_counters", "reset_decode_dispatch_counters"),
    "kda_depthwise_conv1d": (
        "vllm_neuron.functional.kda.depthwise_conv1d",
        "dispatch_counters", "reset_dispatch_counters"),
    "kda_gate_clamp": (
        "vllm_neuron.functional.kda.gate_clamp",
        "gate_clamp_dispatch_counters", "reset_gate_clamp_dispatch_counters"),
    "mhc_hyper_connection": (
        "vllm_neuron.functional.mhc.hyper_connection",
        "dispatch_counters", "reset_dispatch_counters"),
    "mhc_sinkhorn": (
        "vllm_neuron.functional.mhc.sinkhorn",
        "dispatch_counters", "reset_dispatch_counters"),
    "vision_patch_embed": (
        "vllm_neuron.functional.vision.patch_embed",
        "dispatch_counters", "reset_dispatch_counters"),
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
            for name, (path, _reader, _reset) in _SEAM_REGISTRY.items()}


def _seam_counter_api(name: str) -> tuple:
    """``(read, reset)`` for one registered family. BOTH names come from the row."""
    path, reader, reset = _SEAM_REGISTRY[name]
    module = importlib.import_module(path)
    return getattr(module, reader), getattr(module, reset)


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
    assumed: across all twenty-two modules that define a family, no name ending in
    the suffix is imported or assigned, only defined
    (``probe-054a-counter-names-r1``, ``probe-054a-counter-population-r1``, and for
    the twenty-second ``inc-glm53f-054e`` c6 re-read ``dsa/causal_bound.py``, which
    defines four such names at column 0 and imports none).

    THIS CONTROL CANNOT SEE A MODULE NO ROW NAMES, which is what let ten families
    sit unregistered until repair R; :func:`_assert_no_unregistered_counter_family`
    is the other half and reads the package rather than the registry.

    Raises:
        VacuousControlError: naming which module and which family is unclaimed.
    """
    claimed: dict[str, set] = {}
    for name in _SEAMS:
        path, reader, _reset = _SEAM_REGISTRY[name]
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


def _assert_no_unregistered_counter_family() -> None:
    """No counter family anywhere in ``vllm_neuron.functional`` is unregistered.

    The registered predicate's second conjunct is a statement about EVERY seam this
    campaign owns, so a family in a module no row names shrinks that statement
    without saying so. This reads the package's own source tree -- one text scan of
    every ``.py`` under ``functional/``, no per-module import -- and requires the
    ``(module, reader)`` pairs it finds to be exactly the pairs the registry claims.

    WHY SOURCE AND NOT IMPORTS. Importing every module under ``functional/`` to call
    ``dir()`` on it would pull in every kernel in the package on the way to counting
    accessors, on every item. A regex over ``def`` lines needs no import and cannot
    be defeated by an import-time failure in a module this campaign never calls.

    WHY IT FIRES ON A NEW SEAM RATHER THAN IGNORING IT. A campaign-owned seam that
    nothing reads is the failure mode this control exists for; a seam this campaign
    does NOT own would be an exclusion with a reason, and there is none today --
    every one of the twenty-two modules is named in this campaign's plan, the
    twenty-second being ``inc-glm53f-103``'s ``dsa/causal_bound``. This control did
    its job once for real: it is what refused grant 181's run rather than letting
    seven forwards read two of this campaign's own seams as if they did not exist.

    Raises:
        VacuousControlError: naming the module and the family that no row claims.
    """
    import re

    package = importlib.import_module("vllm_neuron.functional")
    root = Path(package.__file__).resolve().parent
    pattern = re.compile(r"^def (\w*%s)\(" % _COUNTER_SUFFIX, re.M)
    found: set[tuple[str, str]] = set()
    for file in sorted(root.rglob("*.py")):
        dotted = "vllm_neuron.functional." + ".".join(
            file.relative_to(root).with_suffix("").parts
        )
        for reader in pattern.findall(file.read_text(errors="replace")):
            if reader.startswith("reset_"):
                continue
            found.add((dotted, reader))
    claimed = {(path, reader) for path, reader, _reset in _SEAM_REGISTRY.values()}
    print(f"TINYFWD|counter_population|found={len(found)}|claimed={len(claimed)}")
    if found != claimed:
        unclaimed = sorted(found - claimed)
        phantom = sorted(claimed - found)
        raise VacuousControlError(
            f"the package defines {len(found)} counter families and this file's "
            f"registry claims {len(claimed)}. Families no row claims: {unclaimed}. "
            f"Rows the package does not define: {phantom}. An unclaimed family is "
            f"a seam this campaign owns whose torch fallbacks no predicate totals"
        )


def _declare_bound_and_sentinel(expected: dict) -> None:
    """Declare ``-103``'s two causal families at the SELECTOR's count, not at a number.

    ``Glm5NextDSAIndexer.select_bounded_pools`` composes the three seams in one
    straight-line method with no branch between them (``model_fp8.py:5003-5009``):
    ``dsa_causal_bound``, then ``dsa_topk_select``, then ``dsa_causal_sentinel``. So
    whatever an item declares for the selector is arithmetically what these two owe,
    per item and per layer, and taking it FROM the selector's own entry is what stops
    a later change to one of the three from leaving the other two stale.

    ``test_dsa_layer.py`` measured this pair through the indexer at THIS file's dials
    -- ``select_k == TOPK_POOLS == 2`` and ``pool == POOL_SIZE == 4`` there against
    ``MLA_TOPK_POOLS = 2`` and ``MLA_INDEX_KPOOL = 4`` here -- and read ``(1, 0)``
    for both entry points (``:3907-3922``, repeated at ``:4251-4259``). The zero half
    of that reading is the one conjunct 2 already aggregates, and it stays zero
    because ``can_run_kernel()`` is True in this file's launch mode
    (``VLLM_NEURON_CPU_MODE=1`` with ``NKI_SIMULATOR=1``, ``neuron_utils.py:16-23``),
    so both entry points take their NKI branch rather than a torch oracle -- which is
    P13's requirement, not a preference.

    An item whose forward reaches no indexer declares no selector count, and this
    helper then declares nothing either: items 1 to 4 are exactly that case, and
    their zeros stay READ rather than becoming expectations.
    """
    selector = expected.get("dsa_topk_select")
    if selector is None:
        return
    expected["dsa_causal_bound"] = selector
    expected["dsa_causal_sentinel"] = selector


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

    THE POPULATION IS CHECKED FIRST, BOTH WAYS, because conjuncts 2 and 3 are
    statements about a set and an unclaimed counter family would quietly shrink it:
    the registry must claim every family in each module it names, and the package
    must define no family the registry does not name.

    Raises:
        VacuousControlError: on any conjunct, named so the transcript says which.
    """
    from vllm_neuron.utils.neuron_utils import can_run_kernel

    _assert_every_counter_family_is_registered()
    _assert_no_unregistered_counter_family()
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

    IT APPLIES THE trn2 PAIR ONCE, ASKED RATHER THAN RETYPED. On a 240-clamp
    platform the store is a matched pair -- the bytes are squeezed and the grid is
    compensated by the exact inverse -- so a reference that applied neither would
    disagree with a correct product by that factor, and one that applied only the
    squeeze would disagree by the same factor the other way. Both halves come from
    the loader's own functions, never a constant copied into this file, so a change
    to the factor moves this reference with it: ``inc-glm53f-054e`` moved it from
    ``240/448`` to an exact ``1/2`` and not a line here changed. On a platform where the
    clamp is 448 both calls are no-ops and this is the arithmetic it always was.

    WHAT IT DELIBERATELY DOES NOT MEASURE. The squeeze re-quantises through fp8,
    so this states the CHECKPOINT-to-stored invariant and not the fidelity of the
    stored weight to the raw HF value. That fidelity is ``inc-glm53f-054e``'s
    acceptance, not this file's.
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
    expanded = compensate_block_scales(block_scale).scale_inv.repeat_interleave(
        BLOCK_QUANT_SIZE, dim=0
    ).repeat_interleave(BLOCK_QUANT_SIZE, dim=1)
    return downscale_fp8_weight_bytes(weight_fp8).to(torch.float32) * expanded


def _scale_grid_attribute(leaf: str) -> str:
    """The attribute a weight leaf's scale grid arrives under -- ASKED, not retyped.

    The rule strips ``_weight`` before it appends, so ``gate_proj_weight`` names
    ``gate_proj_weight_scale_inv``. It has ONE definition,
    :meth:`Glm5NextForConditionalGeneration._sibling_scale_grid_name`, and this
    asks that definition -- ``test_load_weights.py:2470``'s repair, adopted, for
    the same reason it gives: a second copy of the naming rule is the drift.
    """
    return _impl().Glm5NextForConditionalGeneration._sibling_scale_grid_name(leaf)


def _attach(
    module,
    leaf: str,
    weight: torch.Tensor,
    grid: torch.Tensor,
    *,
    prep_will_compensate: bool = False,
) -> None:
    """Bind one declared weight and the plain-attribute grid beside it.

    ``nn.Parameter(..., requires_grad=False)`` because the leaf was declared with
    ``register_parameter(name, None)`` and torch refuses a plain tensor there
    (``test_kda_layer.py:412``'s landed form). The grid is a PLAIN attribute and
    not a parameter, which is the arrangement ``_scale_prep_leaves`` documents.

    IT BINDS WHAT THE LOADER DELIVERS, WHICH IS A PAIR. On a 240-clamp platform a
    real load squeezes the bytes and compensates the grid by the exact inverse, so
    binding the checkpoint's own bytes beside the checkpoint's own grid is not a load
    at all -- it is half of one, and a forward built on it disagrees with a correct
    product by that factor. The bytes are therefore always squeezed here. The factor
    itself is never named in this file, which is why ``inc-glm53f-054e`` moving it to
    an exact ``1/2`` left this argument untouched.

    ``prep_will_compensate`` IS ABOUT WHERE THE OTHER HALF COMES FROM, NOT WHETHER.
    A module whose test then calls ``retile_checkpoint_scale_grids`` gets its grid
    compensated by the load-path prep, so this binds that grid RAW and the pair is
    completed exactly once. Every other module in this file gets no prep at all,
    so the compensation has to arrive here or it never arrives. Ten binds in this
    file, and only the two objects the prep runs on pass the keyword.
    """
    setattr(
        module,
        leaf,
        torch.nn.Parameter(
            downscale_fp8_weight_bytes(weight).clone(), requires_grad=False
        ),
    )
    delivered = grid if prep_will_compensate else compensate_block_scales(grid).scale_inv
    setattr(module, _scale_grid_attribute(leaf), delivered.clone())


def _prep_operands_from_the_module(module, leaves, fixture: dict) -> tuple:
    """The six arguments ``prepare_scale_operands`` takes, read off the MODULE.

    THE PREP IS HANDED WHAT THE LOAD BOUND, NOT WHAT THE FIXTURE HELD.
    ``_run_load_time_preps`` (``model_fp8.py:7887-7895``) passes this module's own
    attributes, and a bank's forward multiplies only what this call built. A site that
    hands the fixture's dict straight through therefore builds operands the load never
    touched: ``_attach``'s squeeze and compensation are skipped, and on a 240-clamp
    platform the item measures the checkpoint instead of the store. This file's
    shared-expert item already reads them off the module in exactly this form, so this
    is that form asked once rather than written at four more sites.

    THE REFUSAL IS BY IDENTITY, NOT EQUALITY. On a platform where the pair is a no-op
    a copy of the fixture's tensor compares equal to the module's, so equality could
    not tell the two apart; ``is`` can.

    Raises:
        VacuousControlError: if a tensor the fixture holds reached the prep, which
            means the bind between the fixture and the prep stopped transforming.
    """
    weights = tuple(getattr(module, leaf) for leaf in leaves)
    grids = tuple(getattr(module, _scale_grid_attribute(leaf)) for leaf in leaves)
    for leaf, weight, grid in zip(leaves, weights, grids):
        if weight is fixture[leaf][0] or grid is fixture[leaf][1]:
            raise VacuousControlError(
                f"{leaf} reached the prep as the fixture's own tensor, so the load's "
                f"squeeze and compensation were skipped and this item would measure "
                f"the checkpoint rather than what a load stores"
            )
    return weights + grids


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


def _clamped(
    tensor: torch.Tensor, low: float | None, high: float | None
) -> torch.Tensor:
    """``tensor`` with the bounds that were GIVEN, and no clamp at all when neither was.

    WHY THIS EXISTS, and it is the first hardware run's own finding. These references
    take each bound as an argument so a control arm can remove one and show the branch
    matters, and the convention is that ``None`` omits a bound -- the way the reference
    model's own gate clamp omits its lower one (``modeling_glm5_next.py:102``). But
    ``Tensor.clamp(min=None, max=None)`` does not return the tensor: it raises
    ``RuntimeError: torch.clamp: At least one of 'min' or 'max' must not be None``. So
    the arm that removes BOTH bounds -- the one that asks what the clamp is worth at all
    -- could not be expressed, and three of the seven acceptance items died in their
    reference rather than in the code under test
    (``increments/fetch-054a-r6-step1-host-20260908T205349Z.out``, this file's ``:720``
    and ``:1271`` at the time).

    It is a pass-through and not a tolerance: with a bound present the clamp is exactly
    the one torch would have applied, and with none present there is nothing to apply.
    """
    if low is None and high is None:
        return tensor
    return tensor.clamp(min=low, max=high)


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
    also how ``:102`` expresses its missing lower bound. Omitting BOTH bounds is a
    control arm asking for no clamp at all, and :func:`_clamped` is what makes that
    expressible -- ``Tensor.clamp(min=None, max=None)`` raises rather than returning
    the tensor, which is the defect the first hardware run of this file found.
    """
    from torch.nn.functional import silu

    x = operands["hidden"].to(torch.float32)
    gate = x @ _dequantise(*operands["gate_proj_weight"])
    up = x @ _dequantise(*operands["up_proj_weight"])

    gate_c = _clamped(gate, None, gate_max)
    up_c = _clamped(up, up_min, up_max)
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

    # ---- THE LOADER-FRAME CONJUNCT. Repair round 1, and the reading the seven
    # items could not make before it.
    #
    # WHAT WENT WRONG, so the next reader knows what this is guarding. Everything
    # above binds weights in the frame the KERNEL multiplies in, with grids already
    # at the public 256 granularity -- and that is not what a checkpoint load
    # delivers. The loader delivers the checkpoint's own layout, gate and up
    # ``[I, H]`` and down ``[H, I]`` (the shard table shards gate and up on dim 0,
    # "the intermediate width", ``model_fp8.py:503-513``), at the checkpoint's 128
    # granularity. So the fixture above was the loader's job done by hand, in the
    # opposite frame, and it passed 7 of 7 while a real load into this class refused
    # at the first of GLM-5.3-Flash's three dense layers. The review that found it
    # is ``bless-054a-code-ebcff0ce-findings.md`` finding 1.
    #
    # WHAT THIS CONJUNCT DOES. It binds the SAME numbers in the loader's frame at
    # the loader's granularity, runs the load-path prep that the real
    # ``load_weights`` runs (``_run_load_time_preps`` reaches this very method by
    # ``hasattr``), and then runs the forward and compares it to the SAME reference
    # the item computed above. Nothing is transposed by hand on the way in: if the
    # prep does not publish the compute frame, the forward refuses or the numbers
    # move, and either way this conjunct fails.
    #
    # WHY THE 128 GRID IS BUILT BY REPEATING THE PUBLIC ONE. Each 256 block then
    # carries ONE scale across its four 128 tiles, so the coarsening is lossless and
    # the published weight is item 1's weight again -- which is what lets this
    # conjunct compare against the reference already computed, at the tolerance
    # already registered, instead of needing a second reference that would have to
    # model the requantisation. The losslessness is REPORTED below rather than
    # assumed, and the comparison that decides the item is the forward's output.
    tiles_per_block = BLOCK_QUANT_SIZE // TILE_SIZE
    loaded = model_fp8.Glm5NextDenseMLP(text_config)
    for leaf in ("gate_proj_weight", "up_proj_weight", "down_proj_weight"):
        compute_weight, public_grid = operands[leaf]
        checkpoint_grid = public_grid.repeat_interleave(
            tiles_per_block, dim=0
        ).repeat_interleave(tiles_per_block, dim=1)
        # ``.t()`` INTO the loader frame, which is the only hand transpose in this
        # conjunct and is on the INPUT side: it manufactures what the loader would
        # have delivered. The output side is never transposed by this test.
        _attach(
            loaded,
            leaf,
            compute_weight.t().contiguous(),
            checkpoint_grid.t().contiguous(),
            prep_will_compensate=True,
        )

    # ---- CONTROL: the loader's frame ALONE is refused, by name. This is the
    # defect's own signature, so if a later edit made the forward tolerant of
    # either frame -- by transposing on the fly, say -- this control fails and the
    # conjunct below stops proving that the PREP is what fixed it.
    with pytest.raises(
        model_fp8.Glm5NextDenseMLPRouteError, match=r"gate_proj_weight must be \[H="
    ):
        loaded.forward(operands["hidden"], quant_config=_quant_config())

    # ---- THE PREP, and what it published.
    retiled = loaded.retile_checkpoint_scale_grids()
    if retiled != 3:
        raise VacuousControlError(
            f"the load-path prep retiled {retiled} projections, not 3; at "
            f"[{HIDDEN_SIZE},{INTERMEDIATE_SIZE}] every extent is a whole "
            f"{BLOCK_QUANT_SIZE} block, so a skip means it could not read them"
        )
    health = getattr(loaded, loaded.DENSE_RETILE_HEALTH_ATTR)
    for leaf in ("gate_proj_weight", "up_proj_weight", "down_proj_weight"):
        compute_weight, public_grid = operands[leaf]
        record = health[leaf]
        published = getattr(loaded, leaf)
        published_grid = getattr(loaded, _scale_grid_attribute(leaf))
        bytes_identical = bool(torch.equal(published.data, compute_weight))
        print(
            f"TINYFWD|dense_loader_frame|{leaf}"
            f"|loader={tuple(record['loader_frame'])}"
            f"|compute={tuple(record['compute_frame'])}"
            f"|grid={tuple(record['compute_grid'])}"
            f"|inexact_rescales={record.get('inexact_rescales')}"
            f"|weight_bytes_identical={bytes_identical}"
        )
        if tuple(published.shape) != tuple(compute_weight.shape):
            raise ReferenceShapeError(
                f"after the prep {leaf} is {tuple(published.shape)}; the frame the "
                f"seam multiplies in is {tuple(compute_weight.shape)}"
            )
        if tuple(published_grid.shape) != tuple(public_grid.shape):
            raise ReferenceShapeError(
                f"after the prep {leaf}'s grid is {tuple(published_grid.shape)}; "
                f"the public grid for this weight is {tuple(public_grid.shape)}"
            )

    # ---- THE FORWARD, from a loader-frame bind, against the SAME reference.
    _reset_seam_counters()
    before_loaded = _read_seam_counters()
    got_loaded = loaded.forward(operands["hidden"], quant_config=_quant_config())
    after_loaded = _read_seam_counters()
    _assert_route_predicate(
        "1 dense MLP from the loader frame",
        {"blockwise_fp8_mm": 3},
        before_loaded,
        after_loaded,
    )
    torch.testing.assert_close(
        got_loaded.float(), reference["out"].float(), rtol=RTOL, atol=ATOL
    )

    # ---- CONTROL: THE PAIR, WRONG IN EITHER DIRECTION, IS REFUSED BY NAME.
    # ``inc-glm53f-054c`` makes the load-path prep compensate the grid, so from here
    # there are exactly two ways to hold the pair wrong: one compensation too few,
    # which is un-squeezed bytes under a compensated grid, and one too many, which
    # is a grid this file compensated and the prep compensated again. The conjunct
    # above cannot see either, because it would pass unchanged if the reference and
    # the product moved together -- which is the fault ``-054c`` found in
    # ``test_load_weights.py``'s own reference. So both are built and refused here.
    #
    # THE READINGS ARE ON THE PUBLISHED GRID, NOT THE FORWARD. The SwiGLU clamp
    # saturates, so an output ratio is compressed below the factor and would be a
    # misleading number to print; the grid ratio is uniform and exact. The refusal
    # is what the control asserts, and no tolerance is introduced to state it.
    for label in ("one_compensation_too_few", "one_compensation_too_many"):
        wrong = model_fp8.Glm5NextDenseMLP(text_config)
        for leaf in ("gate_proj_weight", "up_proj_weight", "down_proj_weight"):
            compute_weight, public_grid = operands[leaf]
            checkpoint_grid = public_grid.repeat_interleave(
                tiles_per_block, dim=0
            ).repeat_interleave(tiles_per_block, dim=1)
            loader_weight = compute_weight.t().contiguous()
            loader_grid = checkpoint_grid.t().contiguous()
            if label == "one_compensation_too_few":
                # BY HAND, bypassing the helper on purpose: the control has to state
                # the defect itself rather than ask the helper that fixes it.
                setattr(
                    wrong,
                    leaf,
                    torch.nn.Parameter(loader_weight.clone(), requires_grad=False),
                )
                setattr(wrong, _scale_grid_attribute(leaf), loader_grid.clone())
            else:
                # The keyword is FORGOTTEN -- the one edit that produces the double.
                _attach(wrong, leaf, loader_weight, loader_grid)
        if wrong.retile_checkpoint_scale_grids() != 3:
            raise VacuousControlError(
                f"the {label} control retiled fewer than 3 projections, so it is "
                f"not the arrangement this control means to refuse"
            )
        for leaf in ("gate_proj_weight", "up_proj_weight", "down_proj_weight"):
            attribute = _scale_grid_attribute(leaf)
            right = getattr(loaded, attribute).to(torch.float32)
            ratio = (getattr(wrong, attribute).to(torch.float32) / right).unique()
            print(
                f"TINYFWD|dense_pair_control|{label}|{leaf}"
                f"|published_grid_ratio={[round(float(v), 7) for v in ratio]}"
                f"|bytes_identical_to_the_correct_bind="
                f"{bool(torch.equal(getattr(wrong, leaf).data, getattr(loaded, leaf).data))}"
            )
        got_wrong = wrong.forward(operands["hidden"], quant_config=_quant_config())
        with pytest.raises(AssertionError):
            torch.testing.assert_close(
                got_wrong.float(), reference["out"].float(), rtol=RTOL, atol=ATOL
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
#: Sixteen experts and top-8, which is what the production router accepts and nothing
#: less: it refuses a top-k that is not exactly 8 and an expert count below 8
#: (``router.py:1037-1051``), because ``nisa.max8`` emits 8 values per partition and
#: ``nisa.nc_find_index8`` consumes exactly 8.
#:
#: SIXTEEN RATHER THAN EIGHT, so the selection stays REAL. Top-8 of 8 selects every
#: expert for every token, and a mask that selects everything cannot show a wrong expert
#: mapping -- the defect the rotating roles below exist to catch. Sixteen also divides
#: any expert-parallel degree this run uses, which is what
#: ``require_uniform_expert_partition`` refuses on, and it stays inside the router's own
#: ceiling of 512 (``router.py:1049``).
#:
#: The bank costs ``E * 3 * I * H`` = 25.2 M parameters, and the MoE block that adds a
#: shared expert and a router reaches 26.7 M, the worst item in this file, against the
#: 32 M cap. Counted in
#: ``increments/probe-054a-e16k8-invariants-mac-r1-20260908T213943Z.out``.
ROUTED_EXPERTS = 16
ROUTED_EXPERTS_PER_TOKEN = 8

#: The eight router weights each token's top-8 carries. They sum to the pinned config's
#: ``routed_scaling_factor`` of 2.5, which is what ``norm_topk_prob=True`` with that
#: factor produces, and every one is an exact binary fraction that survives the
#: ``bfloat16`` cast the router weights take, so no cast loses anything.
#:
#: NONE IS 1.0, AND THAT IS THE POINT. ``PRE_SCALE`` and ``POST_SCALE`` agree EXACTLY at
#: affinity 1.0 -- the router weight is then the identity wherever it is applied -- so a
#: fixture whose affinities sat at 1 would carry a mode control that could not fail.
#: That was MEASURED for the two-value set these replace, across four candidate pairs in
#: section F of ``increments/probe-054a-item2-clamp-feasibility-r5d.out``: ``(1.5, 1.0)``
#: separated the two modes by 0.68%, inside the tolerance this item passes at, while
#: ``(2.0, 0.5)`` separated them by 63.9%, and ``(2.25, 0.25)`` was REJECTED for sending
#: the PRE_SCALE result negative against a positive reference -- a different answer
#: rather than a measured sensitivity.
#:
#: THESE EIGHT CARRY EVERY CHECKABLE PROPERTY ACROSS, AND THEIR MODE SEPARATION IS NOT
#: YET MEASURED. Stated rather than implied: the spread is wider than the pair they
#: replace, 24x against 4x, so the separation should be stronger, but no probe has run at
#: eight values because it needs torch on the bank. A weak separation REDDENS this item
#: instead of passing it, since the mode control asserts a difference OUTSIDE the
#: tolerance, so the run adjudicates this choice rather than inheriting it. The four
#: properties that ARE checkable -- eight values, exact sum of 2.5, all exact in
#: ``bfloat16``, none equal to 1.0 -- are gated in
#: ``increments/probe-054a-e16k8-invariants-mac-r1-20260908T213943Z.out``.
ROUTED_AFFINITIES = (0.75, 0.5, 0.375, 0.3125, 0.25, 0.1875, 0.09375, 0.03125)

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


def _routed_tile_grid(
    exponents: tuple[int, ...],
    *,
    i_first: bool,
    intermediate: int = ROUTED_INTERMEDIATE_SIZE,
    hidden: int = ROUTED_HIDDEN_SIZE,
    experts: int = ROUTED_EXPERTS,
) -> torch.Tensor:
    """``[E, I/128, H/128]`` powers of two, one exponent per 256-block of I.

    ``i_first=False`` returns the ``[E, H/128, I/128]`` transpose, which is the
    orientation the checkpoint stores ``down``'s grid in. Uniform along H and across
    both ``TILE_SIZE`` halves of each 256-block, which is what :func:`_coarsen_to_256`
    then measures rather than trusts.

    THE THREE EXTENTS DEFAULT TO ITEM 2's, so items 2 and 4 call this exactly as they
    did. Item 6 runs a bank at a narrower ``I`` and passes its own.
    """
    step = BLOCK_QUANT_SIZE // TILE_SIZE
    i_tiles = intermediate // TILE_SIZE
    h_tiles = hidden // TILE_SIZE
    per_tile = [float(2.0 ** exponents[tile // step]) for tile in range(i_tiles)]
    grid = torch.tensor(per_tile, dtype=torch.float32).reshape(i_tiles, 1)
    grid = grid.repeat(1, h_tiles)
    if not i_first:
        grid = grid.t().contiguous()
    return grid.unsqueeze(0).repeat(experts, 1, 1).contiguous()


def _routed_affinities() -> torch.Tensor:
    """``[T, E]`` scattered top-8 router scores -- what ``route_tokens`` returns.

    The gate weight at each selected expert's column and zero elsewhere, at the GLOBAL
    router width, which is the form ``block_quant_expert_mm`` declares and refuses
    anything else.

    THE EIGHT ROLES ROTATE ACROSS TOKENS so every expert carries every one of the eight
    weights on some token. A bank that put one expert's weights behind another's router
    column would otherwise be able to agree on a fixture where each expert always had
    the same weight. With 128 tokens over 16 experts the rotation closes eight whole
    cycles, so that coverage is exact rather than nearly so, and the count below is what
    refuses a weight list that has drifted out of step with the declared top-k.
    """
    affinities = torch.zeros(TOKENS, ROUTED_EXPERTS, dtype=torch.float32)
    for token in range(TOKENS):
        for role, weight in enumerate(ROUTED_AFFINITIES):
            affinities[token, (token + role) % ROUTED_EXPERTS] = weight
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
    # EVERY EXTENT IS READ OFF THE OPERANDS, not off this section's constants, and
    # that changed in ``inc-glm53f-054a`` item 6 rather than at item 2. The four
    # values are the same numbers at items 2 and 4 -- their operands are built from
    # those constants -- so no landed reading moves. Item 6 runs a bank at a
    # narrower ``I`` inside a residual stack, and one reference read by three items
    # is what stops a second copy of this arithmetic from drifting from it.
    tokens, hidden_size = hidden.shape[0], hidden.shape[1]
    experts = int(affinities.shape[1])
    blocks = down_w.shape[2] // BLOCK_QUANT_SIZE

    terms = [torch.zeros(tokens, hidden_size, dtype=torch.float32)
             for _ in range(blocks)]
    gates, ups = [], []
    for expert in range(experts):
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

        clamped = silu(_clamped(gate, gate_min, gate_max)) * _clamped(
            up, up_min, up_max
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
        *_prep_operands_from_the_module(module, ("gate_proj_weight", "up_proj_weight", "down_proj_weight"), operands)
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
            variant["out"], reference["out"], rtol=MOE_RTOL, atol=MOE_ATOL
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
                f"rtol={MOE_RTOL}, atol={MOE_ATOL}; this item would pass with that "
                f"branch "
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
    # THE MoE BAND: this output is a per-token sum of EIGHT fp8-quantised expert products.
    torch.testing.assert_close(
        got.float(), reference["out"].float(), rtol=MOE_RTOL, atol=MOE_ATOL
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

    # ---- THE LOADER-FRAME CONJUNCT, and on this route it carries the reason the
    # repair had to be a load-time prep at all. Item 1 records what the defect was
    # and why the fixture above could not see it; this is the same conjunct on the
    # route where the frame is LOCKED.
    #
    # WHY THIS ROUTE COULD NOT TRANSPOSE AT COMPUTE TIME, which is what the
    # package's own recorded rule says a consumer should do
    # (``weight_loaders_fp8.py:1764``). ``prepare_scale_operands`` builds the kernel
    # scale operand ONCE from the STORED weight's extents and keeps it on the
    # module. A forward that transposed the weight would multiply it against an
    # operand built in the other frame: the shapes agree and the numbers are wrong,
    # which is the one failure class nothing in this tree would catch. So the
    # transpose has to happen BEFORE this prep reads the module, and the chain below
    # is run in the order ``_run_load_time_preps`` runs it -- retile and republish,
    # THEN prepare, then forward.
    tiles_per_block = BLOCK_QUANT_SIZE // TILE_SIZE
    loaded = model_fp8.Glm5NextSharedExperts(text_config)
    for leaf in leaves:
        compute_weight, public_grid = operands[leaf]
        checkpoint_grid = public_grid.repeat_interleave(
            tiles_per_block, dim=0
        ).repeat_interleave(tiles_per_block, dim=1)
        _attach(
            loaded,
            leaf,
            compute_weight.t().contiguous(),
            checkpoint_grid.t().contiguous(),
            prep_will_compensate=True,
        )

    republished = loaded.retile_checkpoint_scale_grids()
    if republished != 3:
        raise VacuousControlError(
            f"the load-path prep retiled {republished} projections, not 3; every "
            f"extent here is a whole {BLOCK_QUANT_SIZE} block"
        )
    health = getattr(loaded, loaded.SHARED_RETILE_HEALTH_ATTR)
    for leaf in leaves:
        compute_weight, public_grid = operands[leaf]
        record = health[leaf]
        print(
            f"TINYFWD|shared_loader_frame|{leaf}"
            f"|loader={tuple(record['loader_frame'])}"
            f"|compute={tuple(record['compute_frame'])}"
            f"|grid={tuple(record['compute_grid'])}"
            f"|inexact_rescales={record.get('inexact_rescales')}"
        )
        if tuple(getattr(loaded, leaf).shape) != tuple(compute_weight.shape):
            raise ReferenceShapeError(
                f"after the prep {leaf} is {tuple(getattr(loaded, leaf).shape)}, "
                f"not the seam's frame {tuple(compute_weight.shape)}"
            )
        if tuple(getattr(loaded, _scale_grid_attribute(leaf)).shape) != tuple(
            public_grid.shape
        ):
            raise ReferenceShapeError(
                f"after the prep {leaf}'s grid is not the public grid "
                f"{tuple(public_grid.shape)}"
            )

    # ---- CONTROL B: the prep operands are built from what the republish left, not
    # from what the loader left. Building them from the loader frame is the silent
    # failure this whole repair exists to close, so the operand's own extents are
    # read back against the compute frame.
    built_after = loaded.prepare_scale_operands(
        *(getattr(loaded, leaf) for leaf in leaves),
        *(getattr(loaded, _scale_grid_attribute(leaf)) for leaf in leaves),
    )
    if built_after != 3:
        raise VacuousControlError(
            f"the scale prep reported {built_after} operands after the republish, "
            f"not 3"
        )

    _reset_seam_counters()
    before_loaded = _read_seam_counters()
    got_loaded = loaded.forward(operands["hidden"], quant_config=_quant_config())
    after_loaded = _read_seam_counters()
    _assert_route_predicate(
        "3 shared experts from the loader frame",
        {"blockwise_fp8_mm": 3},
        before_loaded,
        after_loaded,
    )
    torch.testing.assert_close(
        got_loaded.float(), reference["out"].float(), rtol=RTOL, atol=ATOL
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


def _shared_at_routed_operands(
    *,
    seed_offset: int = 0,
    gate_exponents: tuple = SHARED_AT_ROUTED_GATE_EXPONENTS,
    up_exponents: tuple = SHARED_AT_ROUTED_UP_EXPONENTS,
    down_exponent: int = SHARED_AT_ROUTED_DOWN_EXPONENT,
) -> dict:
    """A shared expert's three weights and PUBLIC grids at the bank's hidden size.

    The shared route consumes the 256-granularity grid directly -- that is what
    ``prepare_scale_operands`` takes and what the load-path retile publishes -- so
    unlike the bank's fixture this one supplies no ``TILE_SIZE`` grid and nothing
    is coarsened. Item 1's builder is untouched: it is 256 wide by its own
    declared reading and this item needs 512.

    ``seed_offset`` DEFAULTS TO ZERO and the three EXPONENT arguments default to
    this module's own, so item 4's call draws exactly what it drew and scales it
    exactly as it scaled it. Item 6 builds two DENSE MLPs from this same shape at two
    offsets, because two layers holding identical weights cannot show that each layer
    read its own.

    THE EXPONENTS ARE ARGUMENTS AND ITEM 6 MOVES THEM. The sentence that used to
    stand here said the offset moves only the draw and never the scale regimes. It
    did say that, and at ITEM 6's magnitudes it was wrong: the declared regime is
    16-above and 2-inside against a limit of 10, and item 6's operands put both
    blocks at 165 and 20, so both clamped and the item's whole stack collapsed into
    one row. :data:`STACK_DENSE_GATE_EXPONENTS` and its two siblings carry the shift
    and the reading it rests on. What is untouched is ITEM 4's call at these
    DEFAULTS, which is the one the conditioning arguments above rest on.
    """
    h = ROUTED_HIDDEN_SIZE
    i = ROUTED_INTERMEDIATE_SIZE
    blocks = i // BLOCK_QUANT_SIZE
    if blocks != len(gate_exponents) or blocks != len(up_exponents):
        raise VacuousControlError(
            f"this call declares {len(gate_exponents)} gate and "
            f"{len(up_exponents)} up regimes for {blocks} blocks of I"
        )

    gate_w = _fp8_grid_values(SEED_SHARED_GATE + seed_offset, h, i)
    up_w = _fp8_grid_values(SEED_SHARED_UP + seed_offset, h, i)
    columns = slice(
        SHARED_AT_ROUTED_UP_NEGATED_BLOCK * BLOCK_QUANT_SIZE,
        (SHARED_AT_ROUTED_UP_NEGATED_BLOCK + 1) * BLOCK_QUANT_SIZE,
    )
    up_w[:, columns] = -up_w[:, columns]
    down_w = _fp8_grid_values(SEED_SHARED_DOWN + seed_offset, i, h)

    h_blocks = h // BLOCK_QUANT_SIZE
    i_blocks = i // BLOCK_QUANT_SIZE
    return {
        "gate_proj_weight": (
            gate_w.to(_FP8),
            _pow2_scales(gate_exponents, h_blocks),
        ),
        "up_proj_weight": (
            up_w.to(_FP8),
            _pow2_scales(up_exponents, h_blocks),
        ),
        "down_proj_weight": (
            down_w.to(_FP8),
            _pow2_scales((down_exponent,) * h_blocks, i_blocks),
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
# THE ROUTER'S OWN SEAM IS COUNTED, and this paragraph used to say the opposite. #
# ``functional/moe/router.py`` defines ``noaux_tc_dispatch_counters``           #
# (``router.py:1017``) and increments it once per call (``:1664``), and         #
# ``route_tokens`` is what this forward enters -- so the earlier claim that the #
# seam "carries no dispatch counters" was false of the tree and this item's     #
# fallback aggregate omitted the one seam its own control D argues about.       #
# Repaired in ``inc-glm53f-054a`` repair R; the predicate below now reads it.   #
# A router fallback on this fixture could not pass quietly even before that,    #
# and that is measured rather than hoped: its torch oracle selects              #
# ``NOAUX_TC_K`` = 8 columns regardless of the caller's ``top_k``               #
# (``router.py:1736-1738``), and this fixture has 4 experts, so a fallback      #
# raises out of ``torch.topk`` instead of returning a plausible answer.         #
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
        *_prep_operands_from_the_module(block.experts, ("gate_proj_weight", "up_proj_weight", "down_proj_weight"), routed)
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
        moved = not torch.allclose(variant, expected, rtol=MOE_RTOL, atol=MOE_ATOL)
        gap = float((variant - expected).abs().max() / expected.abs().max())
        print(
            f"TINYFWD|moe_control|branch={name}|outside_tolerance={moved}"
            f"|gap={gap:.4f}"
        )
        if not moved:
            raise VacuousControlError(
                f"with the {name} the result is still inside rtol={MOE_RTOL}, "
                f"atol={MOE_ATOL}; this item cannot tell the one add from the wrong "
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
    # ONE MoE dispatch for the bank, THREE dense ones for the shared expert, and ONE
    # for the router. Every seam by name, so a block that ran the shared half through
    # the expert kernel, or the bank through three dense calls, fails instead of
    # passing on a total.
    #
    # THE ROUTER'S DISPATCH IS COUNTED SINCE REPAIR R, and the line this replaces
    # said the router seam defines no counters. It defines
    # ``noaux_tc_dispatch_counters`` (``router.py:1017``) and increments it once per
    # call (``:1664``), and ``route_tokens`` is what this forward enters
    # (``model_fp8.py:1489-1494``) -- so the fallback aggregate over this forward used
    # to omit the one seam whose fallback the item's own control D discusses.
    _assert_route_predicate(
        "4 MoE block",
        {"blockwise_fp8_moe": 1, "blockwise_fp8_mm": 3, "noaux_tc_router": 1},
        before,
        after,
    )

    if tuple(got.shape) != (TOKENS, ROUTED_HIDDEN_SIZE):
        raise ReferenceShapeError(
            f"the forward returned {tuple(got.shape)}, expected "
            f"{(TOKENS, ROUTED_HIDDEN_SIZE)}"
        )
    # THE MoE BAND: the block's output carries the routed half's eight-expert sum.
    torch.testing.assert_close(got.float(), expected.float(),
                               rtol=MOE_RTOL, atol=MOE_ATOL)

    # ---- CONTROL C: THE ROUTER READS THE CHANNEL ORDER IT WAS HANDED.
    #
    # WHAT USED TO STAND HERE AND WHY IT IS GONE. The old arm fed the NORMALISED
    # states in both positions, so the fused kernel normalised twice. The world says
    # that arm cannot discriminate: it moved the answer by a whole-tensor gap of
    # 0.0200 (``increments/run054a-diag2-trn2-1-at-88321db0-20260909T052440Z.out``),
    # which is INSIDE ``MOE_RTOL = 3e-2``, so it raised `VacuousControlError` on a
    # block that was behaving correctly. Two absorbers explain it, and
    # ``increments/proposal-054a-controls-r2.md`` section 7.1 names them from the
    # router's own source: the selected SET moves only when the top-8-of-16 boundary
    # flips, and the selected weights are NORMALISED over that set, so a shift that
    # moves the selected sigmoids together largely cancels in the ratio.
    #
    # THE REPAIRED CONTROL, section 7.3. Swap two CHANNELS of ONE token's router
    # input. ``mean(x**2)`` is invariant under a permutation of a row, so the fused
    # RMSNorm cannot absorb the swap, while ``router_weight``'s columns differ, so
    # the gate projection is not permutation-invariant. What the arm certifies is
    # that this forward routes on the argument it was told to route on: the block
    # routes on argument 1 and hands argument 2 to the experts, so only argument 1 is
    # perturbed here and argument 2 is left exactly as the main compare had it.
    #
    # THE PERTURBATION IS PROVED THROUGH THE FORWARD, AND THAT PROOF IS THE GATE
    # (R2', LEAD-LOG 858). What stood here proved the swap changed the SELECTED
    # EXPERT SET when THIS TEST called ``block.experts.route_tokens`` itself, and
    # gated nothing on the set the FORWARD's own router call returned, so a forward
    # that routed on argument 2 would have passed the arm whose whole purpose is to
    # refuse it (round 3, finding F1:
    # ``reviews/glm-5.3-flash-port/design-and-code-054a-controls-r3-cb12f3d9-findings.md``).
    # The reading is now taken from INSIDE the forward. The block's own
    # ``route_tokens`` -- the method ``Glm5NextMoEBlock.forward`` routes with
    # (``model_fp8.py:3334-3336``) -- is wrapped on the instance for the duration of
    # each ``block.forward`` call, every call it makes is recorded, and the wrapper
    # returns the router's own tuple unchanged. It is removed in a ``finally``, so no
    # later reading in this file sees it.
    #
    # THE GATE IS DISCRETE AND IT IS THE FORWARD'S: the set recorded inside the
    # PERTURBED forward must differ from the set recorded inside the REFERENCE
    # forward at the swapped token. A forward that routed on argument 2, or on a
    # tensor it rebuilt, cannot produce that difference, because only argument 1
    # carries the swap.
    #
    # THE DIRECT CALL STAYS AS A SEARCH, NEVER AS A GATE. Finding which of the eight
    # channel pairs flips a set costs up to 128 router calls; spending 128 forwards
    # on the same question would cost the whole item. So the cheap oracle picks the
    # candidates and the forward decides. That order is also why the forward gate
    # cannot be vacuous on correct code: the oracle has already proved a flip exists
    # for the pair the forward is then asked about.
    #
    # AND THE ARGUMENT-2 ARM, which is what F1's falsifier needs stated positively.
    # The same swap is placed in argument 2 with argument 1 exactly as the main
    # compare had it, and the recorded set must NOT move. The two arms together say
    # the forward routes on the argument this block declares: the swap is visible from
    # argument 1 and invisible from argument 2.
    #
    # WHY row_gap IS NO LONGER THE GATE. The counted run proved the numeric arm
    # vacuous: the swap that flipped the set moved that token's row by 0.017289 while
    # ``MOE_RTOL`` is 0.03, so the comparison could not see a real routing change
    # (``increments/launch-054a-r16-driver-20260909T080034Z.out:208``). Two absorbers
    # named in ``increments/proposal-054a-controls-r2.md`` section 7.1 keep it that
    # way: the selected weights are renormalised over the selected set, and the shared
    # half carries about 0.45 of the row. No tolerance this file may touch would make
    # the row gap the deciding reading, so it stays PRINTED and is not compared.
    #
    # THE STRONGEST CANDIDATE IS KEPT, NOT THE FIRST. One candidate per channel pair,
    # the first token whose set flips for that pair, then the kept swap is the one
    # with the LARGEST row gap of those whose set moved INSIDE THE FORWARD. The counts
    # are printed so a reader knows what the maximum was taken over.
    _real_route_tokens = block.experts.route_tokens
    _had_own_route_tokens = "route_tokens" in vars(block.experts)
    _recorded_sets = []

    def _recording_route_tokens(*args, **kwargs):
        """The block's own router, with the set it returns recorded."""
        _logits, _index, _affinities = _real_route_tokens(*args, **kwargs)
        _selected = (_affinities != 0).clone()
        if _selected.dim() == 3 and int(_selected.shape[0]) == 1:
            _selected = _selected[0]
        _recorded_sets.append(_selected)
        return _logits, _index, _affinities

    def _forward_recording_the_router(_first, _second):
        """``block.forward(_first, _second)``, and every router call it made."""
        del _recorded_sets[:]
        block.experts.route_tokens = _recording_route_tokens
        try:
            _out = block.forward(
                _first,
                _second,
                router_gamma=gamma,
                text_config=text_config,
                quant_config=quant_config,
            )
        finally:
            if _had_own_route_tokens:
                block.experts.route_tokens = _real_route_tokens
            else:
                del block.experts.route_tokens
        return _out, list(_recorded_sets)

    def _routed_or_raise(_sets, _which):
        """The first recorded set, or the raise that names the router path."""
        if not _sets:
            raise VacuousControlError(
                f"the {_which} forward made no call to block.experts.route_tokens, "
                f"the method this control wrapped and the one "
                f"Glm5NextMoEBlock.forward routes with: nothing this control "
                f"perturbs can be read from the router's output, so it cannot speak"
            )
        return _sets[0]

    # ---- THE REFERENCE FORWARD, RE-RUN WITH THE ROUTER RECORDED. The item's own
    # compare above already ran this forward and its route predicate is closed
    # (control D resets the counters before its own), so this call is counted by
    # nothing that is registered.
    _ref_out, _ref_sets = _forward_recording_the_router(pre_norm, normed)
    _ref_set = _routed_or_raise(_ref_sets, "reference")
    print(
        f"TINYFWD|moe_router_recorded|forward=reference"
        f"|router_calls={len(_ref_sets)}|set_shape={tuple(_ref_set.shape)}"
        f"|output_repeated_the_item_s_own_call={bool(torch.equal(_ref_out, got))}"
        f"|note=the calls are counted INSIDE the forward; the repeat of the output"
        f" is a reading and is gated by nothing"
    )
    _base_set = affinities != 0
    _probes = 0
    _candidates = []
    for _i, _j in MOE_SWAP_CHANNEL_CANDIDATES:
        for _t in range(min(int(pre_norm.shape[0]), MOE_SWAP_TOKEN_SCAN)):
            _candidate = pre_norm.clone()
            _candidate[_t, [_i, _j]] = _candidate[_t, [_j, _i]]
            if torch.equal(_candidate[_t], pre_norm[_t]):
                continue
            _probes += 1
            _, _, _probe_affinities = block.experts.route_tokens(
                _candidate.unsqueeze(0), gamma, text_config
            )
            if not torch.equal((_probe_affinities != 0)[_t], _base_set[_t]):
                _candidates.append((_t, _i, _j, _candidate))
                break
    if not _candidates:
        raise VacuousControlError(
            f"no swap among {len(MOE_SWAP_CHANNEL_CANDIDATES)} channel pairs on any "
            f"of the first {MOE_SWAP_TOKEN_SCAN} tokens changed the SELECTED EXPERT "
            f"SET in {_probes} probes of this block's own router, so no perturbation "
            f"this control can make would be visible in its output and there is "
            f"nothing for the forward to be asked about"
        )
    _evaluated = []
    for _t, _i, _j, _candidate in _candidates:
        _probe_out, _probe_sets = _forward_recording_the_router(_candidate, normed)
        _probe_set = _routed_or_raise(_probe_sets, "perturbed")
        _through = not torch.equal(_probe_set[_t], _ref_set[_t])
        _probe_gap = float(
            (_probe_out[_t].float() - expected[_t].float()).abs().max()
            / expected[_t].abs().max()
        )
        print(
            f"TINYFWD|moe_swap_forward|token={_t}|channels={_i} and {_j}"
            f"|router_calls={len(_probe_sets)}"
            f"|selected_set_changed_inside_the_forward={_through}"
            f"|selected_inside_reference={int(_ref_set[_t].sum())}"
            f"|selected_inside_perturbed={int(_probe_set[_t].sum())}"
            f"|row_gap={_probe_gap:.6f}"
        )
        if _through:
            _evaluated.append((_probe_gap, _t, _i, _j, _probe_out))
    if not _evaluated:
        raise VacuousControlError(
            f"{len(_candidates)} channel swaps changed the selected expert set when "
            f"this control called block.experts.route_tokens itself, and not one of "
            f"them changed the set THE SAME ROUTER returned INSIDE block.forward: "
            f"this forward does not route on the argument it was handed the swap in, "
            f"so its output cannot be read as a consequence of the routing this "
            f"control perturbed"
        )
    _best = max(_evaluated, key=lambda item: item[0])
    _row_gap, _swap_token, _swap_i, _swap_j, swapped = _best
    # ---- THE ARGUMENT-2 ARM. The kept swap, placed in argument 2 while argument 1
    # stays the main compare's own tensor. The recorded set must not move, because
    # the router reads argument 1; if it moves, this forward routed on the states it
    # hands the experts and the arm above proved nothing about its routing.
    _arm_second = normed.clone()
    _arm_second[_swap_token, [_swap_i, _swap_j]] = _arm_second[
        _swap_token, [_swap_j, _swap_i]
    ]
    _arm_out, _arm_sets = _forward_recording_the_router(pre_norm, _arm_second)
    _arm_set = _routed_or_raise(_arm_sets, "argument-2 arm")
    _arm_changed = not torch.equal(_arm_set[_swap_token], _ref_set[_swap_token])
    print(
        f"TINYFWD|moe_swap_arm|branch=the kept swap placed in argument 2"
        f"|token={_swap_token}|router_calls={len(_arm_sets)}"
        f"|selected_set_changed_inside_the_forward={_arm_changed}"
        f"|note=argument 1 is the main compare's own tensor, so the set must not"
        f" move: the swap is visible from argument 1 and invisible from argument 2"
    )
    if _arm_changed:
        raise VacuousControlError(
            f"the swap placed in ARGUMENT 2 changed the selected expert set inside "
            f"the forward at token {_swap_token} while argument 1 was the main "
            f"compare's own tensor, so this forward routes on the states it hands "
            f"the experts and the set change measured above followed the wrong "
            f"argument"
        )
    # ---- AND THE RETIRED ARM, MEASURED RATHER THAN GATED. The arm this control
    # replaced fed the normalised states in both positions. It is run here because
    # R2' asks for it, and it is a READING: nothing in the router's arithmetic
    # promises that a different input leaves the top-8-of-16 boundary where it was,
    # so a gate here could redden a correct block. What it is worth is the number --
    # whether the retired arm was visible to the router at all.
    _double_out, _double_sets = _forward_recording_the_router(normed, normed)
    _double_changed = (
        bool(not torch.equal(_double_sets[0][_swap_token], _ref_set[_swap_token]))
        if _double_sets
        else None
    )
    print(
        f"TINYFWD|moe_swap_arm|branch=the retired arm, normalised states in both"
        f" positions|token={_swap_token}|router_calls={len(_double_sets)}"
        f"|selected_set_changed_inside_the_forward={_double_changed}"
        f"|note=a reading, gated by nothing: a different router input may move the"
        f" top-k boundary on correct code"
    )
    print(
        f"TINYFWD|moe_swap_probe|token={_swap_token}|channels={_swap_i} and "
        f"{_swap_j}|probes={_probes}"
        f"|selected_before={int(_base_set[_swap_token].sum())}"
        f"|set_changed_pairs={len(_candidates)} of "
        f"{len(MOE_SWAP_CHANNEL_CANDIDATES)}|evaluated={len(_evaluated)}"
        f"|token_scan={MOE_SWAP_TOKEN_SCAN}"
        f"|note=one candidate per channel pair, the kept swap has the largest row"
        f" gap of those whose set changed inside the forward"
    )
    _row_identical = bool(torch.equal(swapped[_swap_token], expected[_swap_token]))
    moved = not torch.allclose(swapped[_swap_token].float(),
                               expected[_swap_token].float(),
                               rtol=MOE_RTOL, atol=MOE_ATOL)
    gap = float((swapped.float() - expected.float()).abs().max() / expected.abs().max())
    print(
        f"TINYFWD|moe_control|branch=router input channels {_swap_i} and {_swap_j} "
        f"swapped at token {_swap_token}"
        f"|selected_set_changed_inside_the_forward=True"
        f"|row_bitwise_identical={_row_identical}"
        f"|outside_tolerance={moved}|row_gap={_row_gap:.6f}"
        f"|whole_tensor_gap={gap:.6f}"
        f"|note=the proved change of the set the FORWARD's own router returned is"
        f" the gate; every number here is a reading and none is compared against"
        f" MOE_RTOL"
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
        *_prep_operands_from_the_module(bare.experts, ("gate_proj_weight", "up_proj_weight", "down_proj_weight"), routed)
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
        "4 MoE block, no shared expert",
        {"blockwise_fp8_moe": 1, "noaux_tc_router": 1},
        before,
        after,
    )
    print("TINYFWD|moe_control|branch=no shared expert|dense_dispatches=0")
    # THE MoE BAND: the bare block returns the routed half, eight-expert sum and all.
    torch.testing.assert_close(
        bare_got.float(), routed_reference["out"].float(),
        rtol=MOE_RTOL, atol=MOE_ATOL
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


def _mla_text_config(**overrides):
    """The checkpoint's config narrowed to item 5's geometry.

    ``**overrides`` WINS OVER THE DIALS BELOW and exists so item 6 has ONE authority
    for the attention geometry rather than a second copy of the dial list. Item 5
    calls this with no arguments and gets exactly what it got before; item 6 adds the
    stack's own fields -- its layer schedule, its dense/sparse split, its expert
    count, its vocabulary -- and widens ``hidden_size``, which the MoE seam forces
    (:data:`STACK_HIDDEN_SIZE` records that measurement).

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

    dials = {
        "hidden_size": MLA_HIDDEN_SIZE,
        "num_key_value_heads": NUM_KEY_VALUE_HEADS,
        "num_attention_heads": MLA_HEADS,
        "q_lora_rank": MLA_Q_LORA_RANK,
        "kv_lora_rank": MLA_KV_LORA_RANK,
        "qk_nope_head_dim": MLA_QK_NOPE_HEAD_DIM,
        "qk_rope_head_dim": MLA_QK_ROPE_HEAD_DIM,
        "v_head_dim": MLA_V_HEAD_DIM,
        "index_n_heads": MLA_INDEX_N_HEADS,
        "index_head_dim": MLA_INDEX_HEAD_DIM,
        "index_kpool": MLA_INDEX_KPOOL,
        "index_topk": MLA_TOPK_POOLS * MLA_INDEX_KPOOL,
    }
    dials.update(overrides)
    return replace(Glm5NextTextConfig(), **dials)


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


def _materialise_mla_attention(
    attention, cfg, *, seed: int, output_damping: float = 1.0
) -> tuple[dict, dict]:
    """Every leaf of ONE ALREADY-BUILT attention module, both preps run, indexer too.

    Returns ``(raw, gains)`` -- the five projection leaves in the checkpoint's
    ``[out, in]`` orientation and the two latent-norm gains, which are the operands
    the reference reads.

    FACTORED OUT OF :func:`_mla_attention_fixture` BY ITEM 6 for one reason: item 6's
    three attention modules are built by ``Glm5NextModel``'s own constructor, inside
    its layers, so the fixture cannot be the thing that constructs them. Everything
    here is that function's landed body, unmoved; what stays behind is the pair of
    tiny-bound assertions, which are a statement about a STANDALONE module and would
    be the wrong statement about one layer of a stack.

    One generator, drawn in ``projection_widths()`` order, then the two gains, then
    the indexer -- so a caller that passes two different seeds gets two modules that
    share no value.

    THE WEIGHTS ARE ``randn * in_features ** -0.5``, the scaling both landed
    attention fixtures use (``test_mla_decode.py:139-143``), so activations stay
    order one instead of growing until the comparison measures overflow.

    ``output_damping`` SCALES ``o_proj`` AND DEFAULTS TO 1.0, so item 5 draws exactly
    what it drew. Item 6 passes an exact power of two, and its reason is in
    :data:`STACK_ATTENTION_DAMPING`: it needs the attention half to be a measurable
    but MINORITY share of a residual stream whose sign it must not flip. The scaling
    happens before the load-time prep, because the prep caches a transposed copy and
    a leaf edited afterwards would leave the two disagreeing.
    """
    gen = torch.Generator().manual_seed(int(seed))

    raw: dict = {}
    for name, in_features, out_features in attention.projection_widths():
        weight = torch.randn(
            out_features, in_features, generator=gen, dtype=torch.float32
        ) * (in_features ** -0.5)
        if name == "o_proj":
            weight = weight * float(output_damping)
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
    return raw, gains


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
    raw, gains = _materialise_mla_attention(attention, cfg, seed=seed)

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


def _mla_selection_operands(
    *, tokens: int = MLA_TOKENS, pages: int = MLA_PAGES
) -> dict:
    """The operands the selection stage needs, each derived from its own rule.

    BOTH EXTENTS DEFAULT TO ITEM 5's, so item 5's call is unchanged. Item 6 runs the
    same rules at its own token count and its own page count, and the two
    preconditions below are what make that safe rather than assumed: they are
    inequalities in these two numbers, so a token count that outgrew the pooled-key
    store fails as a control instead of selecting a row that is not there.

    ``slot_mapping`` is pool-granular: a position carries its pool's id where a
    pool COMPLETES and ``-1`` where it does not, which is how a position says "my
    window is not a whole pool" and is steered to the trash row rather than
    dropped. ``seq_lens`` carries one length per score ROW and the rows are
    tokens, so each token's own context length is its own -- which makes the tail
    each token's own incomplete pool.
    """
    slots = torch.full((tokens,), -1, dtype=torch.int32)
    for position in range(tokens):
        if (position + 1) % MLA_INDEX_KPOOL == 0:
            slots[position] = position // MLA_INDEX_KPOOL
    candidates = tokens // MLA_INDEX_KPOOL
    rows = pages * MLA_PAGE_SIZE
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
        "seq_lens": torch.arange(1, tokens + 1, dtype=torch.int32),
        "candidates": candidates,
        "pool_rows": rows,
    }


def _mla_pool_cache(*, pages: int = MLA_PAGES) -> torch.Tensor:
    """The pooled-key store, ``[rows, index_head_dim]`` bf16, written in place."""
    return torch.zeros(
        pages * MLA_PAGE_SIZE, MLA_INDEX_HEAD_DIM, dtype=torch.bfloat16
    )


def _mla_latent_cache(attention, *, tokens: int = MLA_TOKENS) -> torch.Tensor:
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
        tokens, attention.NUM_LATENT_KV_HEADS, int(attention.head_size),
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
    #
    # -054e c6: the causal bound and the causal sentinel are NOT zeros here. This
    # item's fixture refuses the bypass regime by name (see the candidate-count
    # guard above), so the indexer selects, and the selecting path dispatches all
    # three of bound, selector and sentinel once each. Their counts are therefore
    # taken from the selector's own entry rather than written twice more --
    # :func:`_declare_bound_and_sentinel` carries the call chain and the landed
    # reading it rests on.
    route_expected = {
        "mla_projection": 9,
        "mla_absorb": 2,
        "mla_sparse": 1,
        "dsa_kpool_hadamard": 2,
        "dsa_paged_gather": 1,
        "dsa_score_gemm": 1,
        "dsa_topk_select": 1,
        "dsa_index_expand": 1,
    }
    _declare_bound_and_sentinel(route_expected)
    _assert_route_predicate("5 MLA attention", route_expected, before, after)

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


# --------------------------------------------------------------------------- #
# ITEM 6's OWN GEOMETRY -- the decoder STACK. The five items above each run ONE  #
# module; this one runs a whole tree, so it is the first item that has to choose #
# a geometry every seam on the stack admits at once. Each number below is        #
# forced by a seam's own refusal or by the adopted parameter bound, and says     #
# which.                                                                        #
# --------------------------------------------------------------------------- #
#: 512, NOT item 5's 256, and the MoE seam is what forces it. ``_require_blocked``
#: refuses any hidden size below ``MIN_HIDDEN`` and any that is not a multiple of
#: ``PSUM_SIZE``, and both constants are 512 on this image
#: (``moe_blockwise_fp8.py:109``, ``:115``, checked against the seam's own module by
#: :func:`_stack_geometry_preconditions` rather than trusted here). The refusal is an
#: ERROR and not a fallback by that function's own design, so a 256-wide stack with
#: one MoE layer would raise before any numerics ran. Item 2 already runs its bank at
#: 512 and this is the same number for the same reason.
STACK_HIDDEN_SIZE = ROUTED_HIDDEN_SIZE

#: Three layers, all ``deepseek_sparse_attention``. THREE is the smallest count that
#: measures an ORDER: with two, a stack that ran its layers backwards is the same
#: multiset, and with one there is no order at all. The checkpoint's 45-layer 3:1
#: hybrid schedule is NOT reproduced here, and that is a declared exclusion with a
#: cost: the linear-attention family's own state is ``inc-glm53f-038``'s, its fixture
#: declares fifteen parameters and two vLLM state calculators, and none of that is
#: this item's to certify. What this item certifies is that the loop is FAMILY-BLIND
#: -- it hands each layer its own mapping and holds no per-family branch -- and a
#: stack of one family measures that property exactly as well as a hybrid does.
STACK_LAYERS = 3

#: Two dense MLP layers then one sparse. ``_build_mlp`` is monotone -- dense strictly
#: below ``first_k_dense_replace``, sparse at and above -- so a split of 2 in a stack
#: of 3 is the only arrangement that reaches BOTH branches of the FFN half AND gives
#: two layers on one branch, which is what makes per-layer weight binding measurable.
#: The parameter bound is the reason the majority branch is the dense one: a bank
#: costs ``E * 3 * I * H`` and two banks do not fit under 10 M at this hidden size.
STACK_FIRST_K_DENSE = 2
STACK_DENSE_LAYERS = STACK_FIRST_K_DENSE
STACK_MOE_LAYERS = STACK_LAYERS - STACK_FIRST_K_DENSE

#: 128 tokens, and the dense seam forces it rather than the fixture choosing it:
#: ``_require_blocked`` refuses ``M`` that is not a positive multiple of
#: ``TILE_SIZE`` = 128 and says why -- "the kernel tiles M over the PSUM partition
#: axis and does not pad" (``blockwise_fp8_mm.py:238-243``). 128 is therefore the
#: SMALLEST admissible token count, and item 5's 35 is inadmissible the moment a
#: dense MLP is on the same stack.
STACK_TOKENS = TOKENS

#: A tiny vocabulary, because the embedding table is ``vocab x hidden`` and the
#: checkpoint's 154,880 rows alone would be 79 M parameters at this hidden size. The
#: item embeds by INDEXING that table, so a small row count changes nothing about the
#: lookup being certified.
STACK_VOCAB_SIZE = 256

#: 16 pages of :data:`MLA_PAGE_SIZE`, so 64 pooled-key rows for the 32 candidate
#: pools 128 tokens produce. :func:`_mla_selection_operands` asserts the inequality
#: this has to satisfy (a trash row above every candidate), so the slack is measured
#: there rather than argued here. Pages cost no parameters -- the pool cache is a
#: carrier, not a weight -- so the headroom is free.
STACK_PAGES = 16

#: The dense MLP's intermediate extent, item 4's shared-expert shape reused whole:
#: 512 wide by 1024, which is four ``BLOCK_QUANT_SIZE`` column blocks and exactly what
#: :func:`_shared_at_routed_operands` builds. Reused rather than re-derived because
#: ``Glm5NextSharedExperts`` and ``Glm5NextDenseMLP`` consume the SAME three leaves in
#: the same orientation at the same 256-granularity grid.
STACK_DENSE_INTERMEDIATE_SIZE = ROUTED_INTERMEDIATE_SIZE

#: The bank's intermediate extent, and 512 is the SMALLEST the MoE seam admits: it
#: refuses any ``I_TP`` that is not a multiple of ``BLOCK_QUANT_SIZE * NUM_SHARDS``
#: (``moe_blockwise_fp8.py:249-254``), which is 512 on this image. It is also the
#: constraint set's own ``intermediate_size >= 512`` floor, met exactly. Item 2's 1024
#: is not reused here for one measured reason: at this hidden size and sixteen experts a
#: 1024-wide bank costs 25.2 M parameters on its own (:data:`ROUTED_EXPERTS`' note
#: records that figure), and this stack carries three attention layers and two dense
#: MLPs beside it. At 512 the stack reaches 17.6 M against the 32 M cap.
STACK_MOE_INTERMEDIATE_SIZE = 512

STACK_EXPERTS = ROUTED_EXPERTS
STACK_EXPERTS_PER_TOKEN = ROUTED_EXPERTS_PER_TOKEN

#: ``o_proj`` is scaled by this before the load-time prep, so the attention half is a
#: MINORITY share of the residual stream. It is an exact power of two, so the scaling
#: itself rounds nothing away, and the reference reads the scaled leaf -- this is a
#: conditioning choice about the fixture, not a change to the function under test.
#:
#: TWO REASONS, both about this being the first item that composes the two halves.
#: (a) THE SIGN OF THE RESIDUAL. The embedding table is drawn on the unsigned fp8
#: grid, so the residual stream starts positive, and every MLP weight on this stack is
#: unsigned too; that is the premise this whole file's conditioning rests on (see the
#: module docstring: signed operands make each dot product a near-cancelling sum and a
#: relative tolerance then measures cancellation). The attention weights are SIGNED --
#: item 5's draw, kept, because the absorb algebra is what item 5 measures -- so an
#: undamped attention half would drive the FFN's input through zero and put every
#: downstream fp8 comparison back into the cancelling regime.
#: (b) PRECISION. This stack runs in bf16, which is the checkpoint's activation dtype
#: and the dtype every seam on it declares, and ``attend()`` rounds its latent-space
#: result to that dtype mid-chain (``model_fp8.py``, ``attend``'s
#: ``attended.to(hidden_states.dtype)``) while the dense reference stays in fp32.
#: Item 5 never sees that rounding because item 5 hands the forward fp32. One bf16
#: rounding is 2**-8 relative, which is 39% of this item's ``rtol``; damping the half
#: that carries it to an eighth of the stream leaves it under 5% of the budget.
STACK_ATTENTION_DAMPING = 0.125

#: One exponent per 256-block of I for the bank's three projections. ALL AT 2**-3 and
#: uniform, which is a deliberate difference from item 2's four straddling regimes:
#: item 2 owns the clamp discrimination and needs its pre-activations either side of
#: the SwiGLU bound, while this item owns the COMPOSITION and needs a bank whose
#: output is well conditioned against a residual stream it did not choose. The item
#: RECORDS which clamps bind rather than declaring none does -- both sides clamp with
#: the same bound from the same config field, so a binding clamp is exercised
#: identically and is not this item's to discriminate.
STACK_BANK_GATE_EXPONENTS = (-3, -3)
STACK_BANK_UP_EXPONENTS = (-3, -3)
STACK_BANK_DOWN_EXPONENTS = (-3, -3)

#: The bank's conditioning tripwire, in the max-norm item 2's own bounds use. Looser
#: than item 2's 4.0 and deliberately so: item 2 feeds its bank a fixture it
#: conditioned itself, while this one feeds it whatever the two layers above produced.
#: The bound's job is to catch near-cancellation, where the output is a small residue
#: of large opposing terms and every reading is inflated by a vanishing denominator.
#: IT IS A TRIPWIRE AND NOT A MEASURED TARGET: no run of this file exists yet, so the
#: first counted run is what says where the reading actually sits.
STACK_MAX_CONDITION = 32.0

#: Distinct from every seed above, and distinct per site, so no two tensors on this
#: stack can pass on a shared draw. The attention seed is a BASE and each layer adds
#: its index, which is what makes "each layer read its own weights" measurable.
SEED_STACK_EMBED = 5441
SEED_STACK_IDS = 5442
SEED_STACK_ATTENTION = 5443
SEED_STACK_BANK_GATE = 5461
SEED_STACK_BANK_UP = 5462
SEED_STACK_BANK_DOWN = 5463
SEED_STACK_ROUTER = 5464

#: Added to :func:`_shared_at_routed_operands`' three seeds, one offset per dense
#: layer. 100 and 200 rather than 0 and 1 so neither offset can land on another
#: fixture's seed.
STACK_DENSE_SEED_OFFSETS = (100, 200)

#: THE DENSE LAYERS' SCALE REGIME, SHIFTED OFF ITEM 4'S AND STATED IN ITEM 4'S OWN
#: TERMS. :data:`SHARED_AT_ROUTED_GATE_EXPONENTS` puts one block of each projection
#: ABOVE the SwiGLU bound and one STRICTLY INSIDE it, and the comment above those
#: exponents states the magnitudes it means: 16 and 2 against a limit of 10. At ITEM
#: 4's operand scale that holds. AT THIS STACK'S SCALE IT DOES NOT. Grant 127 read
#: layer 0's FFN norm peaking at 5.4375, which drives the gate to 165.42 and the up
#: to 163.31, so BOTH blocks clamp, 100% of both projections sit at the limit, every
#: activated row becomes the same constant, the FFN half reaches 3440 against a
#: hidden peak of 0.4766, and the residual add then absorbs the hidden term in EVERY
#: element -- after which every row below layer 0 is identical. That is the row
#: collapse, in one chain, from
#: ``increments/run054a-diag2-trn2-1-at-88321db0-20260909T052440Z.out``.
#:
#: THE SHIFT RESTORES ITEM 4'S DECLARED 16-AND-2 REGIME AT THIS SCALE. Dropping gate
#: and up by ``2**-3`` puts the ``2**0`` block at a predicted 20.7 and the ``2**-3``
#: block at 2.59, which is item 4's own 16 and 2 to within 1.3x: one block still
#: binds and one is strictly inside. Dropping down by ``2**-6`` is a STACK-ONLY
#: requirement item 4 never had -- item 4 compares one block's output against itself
#: and lives happily at a peak of 7739, while this item compares a RESIDUAL ADD, and
#: a half that outweighs the stream it is added to erases the other half instead of
#: being certified beside it.
#:
#: BOTH SHIFTS ARE EXACT POWERS OF TWO on an already exact power-of-two grid, so no
#: stored weight is re-rounded and the fp8 draw is untouched: the seeds, the shapes
#: and the negated block stay the ones item 4 declares.
STACK_DENSE_GATE_UP_SHIFT = -3
STACK_DENSE_DOWN_SHIFT = -6
STACK_DENSE_GATE_EXPONENTS = tuple(
    e + STACK_DENSE_GATE_UP_SHIFT for e in SHARED_AT_ROUTED_GATE_EXPONENTS
)
STACK_DENSE_UP_EXPONENTS = tuple(
    e + STACK_DENSE_GATE_UP_SHIFT for e in SHARED_AT_ROUTED_UP_EXPONENTS
)
STACK_DENSE_DOWN_EXPONENT = SHARED_AT_ROUTED_DOWN_EXPONENT + STACK_DENSE_DOWN_SHIFT

#: A PRINTED READING'S BOUND AND NOT A GATE. The reading is the fraction of elements
#: of a dense layer's residual add where the sum equals the FFN term BIT FOR BIT, so
#: the hidden term contributed nothing at all; grant 127 read 1.000000 at the old
#: scale -- every element. Half is where the useful-fixture line probably sits, but
#: NO RUN HAS MEASURED THIS FRACTION AT THE NEW SCALE, and half a bf16 ULP at a small
#: FFN peak already absorbs the smallest hidden elements, so a half bound could fail
#: a healthy fixture on an unmeasured prediction. GUARD (ii) IS STRUCTURAL INSTEAD --
#: absorption strictly below 1 and a nonzero row spread on the sum, which is exactly
#: the collapse being absent -- and this bound is printed beside the reading so the
#: first counted run says where the fraction really sits.
STACK_ABSORPTION_CEILING = 0.5

#: THE PEAK-SCALED BAND for this item's recomputed comparisons, and the CEILING that
#: stops a reference from buying its own budget. The reasoning is in
#: :func:`_stack_peak_band`; the arithmetic is
#: ``increments/proposal-054a-controls-r2.md`` sections 4.1 to 4.4.
STACK_RECOMPUTE_RTOL = 2.0**-8
STACK_RECOMPUTE_ATOL_FACTOR = 2.0**-8
STACK_RECOMPUTE_PEAK_CEILING = 256.0

#: CANDIDATE CHANNEL PAIRS for item 4's swap control, spread across all four
#: 256-column blocks of H so no pair lands inside one block of ``router_weight``'s
#: columns. The control SEARCHES these and keeps the first pair that changes a
#: token's selected expert set; it never assumes one does.
MOE_SWAP_CHANNEL_CANDIDATES = (
    (0, 1),
    (0, 256),
    (1, 257),
    (7, 263),
    (13, 400),
    (64, 320),
    (128, 384),
    (0, 511),
)

#: HOW FAR THE SWAP SEARCH SCANS, per channel pair. The counted run found its flip at
#: token 2 (``increments/launch-054a-r16-driver-20260909T080034Z.out:207``), so a scan
#: of sixteen tokens keeps the whole search at most 128 router calls -- the same order
#: as the fifteen the search this replaces spent -- while giving every one of the eight
#: pairs its own chance to flip a set. It is a search bound, not a tolerance: nothing
#: is compared against it, and the control's raise names it so a future run that flips
#: nothing says how far it looked.
MOE_SWAP_TOKEN_SCAN = 16

#: The norm gains, as a rotating pattern of exact eighths near 1. EIGHT values for
#: SEVEN sites -- two per layer plus the stack's final norm -- so every site gets a
#: DIFFERENT rotation and a forward that applied one layer's gain at another layer's
#: norm fails the comparison. Exact in bf16, so the stack's own dtype rounds none of
#: them, and near 1 so the RMSNorm they follow keeps the stream order one. NOT ONES:
#: a gain of ones on states whose RMS is already 1 makes the norm nearly an identity,
#: and "this norm ran with this gain" is exactly what has to be visible
#: (:data:`MOE_GAMMA_VALUES` records the same reasoning for item 4).
_STACK_GAIN_VALUES = (1.0, 1.25, 1.5, 1.75, 0.75, 1.125, 0.875, 1.375)


def _stack_gain(site: int) -> torch.Tensor:
    """``[H]`` the norm gain for one site, its own rotation of the eight values."""
    row = torch.tensor(_STACK_GAIN_VALUES, dtype=torch.float32).roll(int(site))
    return row.repeat(STACK_HIDDEN_SIZE // len(_STACK_GAIN_VALUES))


def _stack_text_config(**overrides):
    """The tiny stack's config -- item 5's attention dials plus the tree's own fields.

    ``overrides`` reaches :func:`_mla_text_config` last, so item 7 can turn one
    field -- ``tie_word_embeddings`` -- without a second copy of this dial set.

    Built through :func:`_mla_text_config`, so the attention geometry has ONE
    authority in this file and a dial that moves there moves here too. The overrides
    are the fields a TREE has and a single attention module does not.

    ``num_hidden_layers`` AND ``layer_types`` ARE BOTH PASSED, and that is required
    rather than belt-and-braces: ``dataclasses.replace`` copies the source instance's
    already-defaulted 45-entry schedule, and ``__post_init__`` then refuses a schedule
    whose length disagrees with the layer count (``config.py:337-341``).
    """
    from vllm_neuron.model.glm5_next.config import DSA_LAYER_TYPE

    return _mla_text_config(
        hidden_size=STACK_HIDDEN_SIZE,
        intermediate_size=STACK_DENSE_INTERMEDIATE_SIZE,
        moe_intermediate_size=STACK_MOE_INTERMEDIATE_SIZE,
        num_hidden_layers=STACK_LAYERS,
        layer_types=[DSA_LAYER_TYPE] * STACK_LAYERS,
        first_k_dense_replace=STACK_FIRST_K_DENSE,
        n_routed_experts=STACK_EXPERTS,
        num_experts_per_tok=STACK_EXPERTS_PER_TOKEN,
        n_shared_experts=0,
        vocab_size=STACK_VOCAB_SIZE,
        **overrides,
    )


def _stack_geometry_preconditions() -> None:
    """Every extent this stack chose, checked against the seam that constrains it.

    THE NUMBERS COME FROM THE SEAMS' OWN MODULES, not from this file. The three
    constants that forced :data:`STACK_HIDDEN_SIZE` and
    :data:`STACK_MOE_INTERMEDIATE_SIZE` are read out of ``moe_blockwise_fp8`` and the
    token rule out of the ``TILE_SIZE`` this file already imports, so an image whose
    seam moved a bound fails here by name instead of raising from inside a kernel.

    Resolved inside the call, like :func:`_impl`, so this file's import time does not
    depend on the MoE kernel being importable.
    """
    from vllm_neuron.functional.moe.moe_blockwise_fp8 import (
        MAX_HIDDEN,
        MIN_HIDDEN,
        NUM_SHARDS,
        PSUM_SIZE,
    )

    print(f"TINYFWD|stack_geometry|hidden={STACK_HIDDEN_SIZE}"
          f"|min_hidden={MIN_HIDDEN}|max_hidden={MAX_HIDDEN}|psum={PSUM_SIZE}"
          f"|moe_intermediate={STACK_MOE_INTERMEDIATE_SIZE}"
          f"|i_tp_step={BLOCK_QUANT_SIZE * NUM_SHARDS}"
          f"|tokens={STACK_TOKENS}|tile={TILE_SIZE}")
    problems = []
    if not MIN_HIDDEN <= STACK_HIDDEN_SIZE <= MAX_HIDDEN:
        problems.append(
            f"hidden {STACK_HIDDEN_SIZE} outside the MoE seam's "
            f"[{MIN_HIDDEN}, {MAX_HIDDEN}]"
        )
    if STACK_HIDDEN_SIZE % PSUM_SIZE or STACK_HIDDEN_SIZE % BLOCK_QUANT_SIZE:
        problems.append(
            f"hidden {STACK_HIDDEN_SIZE} is not a multiple of PSUM_SIZE={PSUM_SIZE} "
            f"and of BLOCK_QUANT_SIZE={BLOCK_QUANT_SIZE}"
        )
    if STACK_MOE_INTERMEDIATE_SIZE % (BLOCK_QUANT_SIZE * NUM_SHARDS):
        problems.append(
            f"bank I_TP {STACK_MOE_INTERMEDIATE_SIZE} is not a multiple of "
            f"BLOCK_QUANT_SIZE * NUM_SHARDS = {BLOCK_QUANT_SIZE * NUM_SHARDS}"
        )
    if STACK_TOKENS % TILE_SIZE:
        problems.append(
            f"tokens {STACK_TOKENS} is not a whole number of TILE_SIZE={TILE_SIZE} "
            f"rows, which the dense seam refuses and does not pad"
        )
    if STACK_HIDDEN_SIZE % len(_STACK_GAIN_VALUES):
        problems.append(
            f"hidden {STACK_HIDDEN_SIZE} is not a multiple of "
            f"{len(_STACK_GAIN_VALUES)}, so the gain pattern does not tile it"
        )
    if problems:
        raise VacuousControlError(
            "this stack's geometry is inadmissible: " + "; ".join(problems)
        )


def _stack_bank_operands() -> dict:
    """The bank's three weights and three ``TILE_SIZE`` grids at the stack's widths.

    The checkpoint's orientations, which is what ``prepare_scale_operands`` declares
    it takes: gate and up ``[E, I, H]``, down ``[E, H, I]``. Item 2's builder is not
    reused because its exponent regimes are written for four column blocks of I and
    this bank has two -- and its 1024-wide bank does not fit this stack's parameter
    budget. Nothing here is negated: the clamp discrimination is items 1 to 3's, and
    an unsigned draw is what keeps the sums out of the cancelling regime.
    """
    h = STACK_HIDDEN_SIZE
    i = STACK_MOE_INTERMEDIATE_SIZE
    e = STACK_EXPERTS
    blocks = i // BLOCK_QUANT_SIZE
    for label, exponents in (
        ("gate", STACK_BANK_GATE_EXPONENTS),
        ("up", STACK_BANK_UP_EXPONENTS),
        ("down", STACK_BANK_DOWN_EXPONENTS),
    ):
        if len(exponents) != blocks:
            raise VacuousControlError(
                f"the bank's {label} projection declares {len(exponents)} scale "
                f"regimes for {blocks} blocks of I={i}"
            )
    grid_extents = dict(intermediate=i, hidden=h, experts=e)
    return {
        "gate_proj_weight": (
            _fp8_grid_values(SEED_STACK_BANK_GATE, e, i, h).to(_FP8),
            _routed_tile_grid(
                STACK_BANK_GATE_EXPONENTS, i_first=True, **grid_extents),
        ),
        "up_proj_weight": (
            _fp8_grid_values(SEED_STACK_BANK_UP, e, i, h).to(_FP8),
            _routed_tile_grid(
                STACK_BANK_UP_EXPONENTS, i_first=True, **grid_extents),
        ),
        "down_proj_weight": (
            _fp8_grid_values(SEED_STACK_BANK_DOWN, e, h, i).to(_FP8),
            _routed_tile_grid(
                STACK_BANK_DOWN_EXPONENTS, i_first=False, **grid_extents),
        ),
    }


def _stack_fixture(model=None) -> dict:
    """The whole tiny stack, every mapped tensor bound and every load-time prep run.

    Returns the model, its config, the per-layer attention operands, the per-layer
    MLP operands, the embedding table and the final gain -- everything the reference
    reads. Nothing here is read back out of the implementation: each operand is the
    tensor this function drew and then bound.

    THE MLP PARTITION IS MEASURED OFF THE BUILT TREE, not declared. ``_build_mlp`` is
    the single authority for which layers carry experts, and an item that assumed the
    split would still pass if the split moved -- while every reference below would be
    computing the wrong branch.

    ``model`` LETS ITEM 7 BIND THE SAME WEIGHTS ONTO THE STACK THE ROOT BUILT, rather
    than onto one this function builds. Item 7 cannot hand its root a stack -- the
    root's ``__init__`` builds its own -- so the direction is inverted here instead of
    duplicating 130 lines of materialisation. Passed ``None``, this function builds the
    stack exactly as it did for item 6, and its config comes from
    :func:`_stack_text_config` either way; a caller supplying a model supplies one
    built from that same config, which the layer-count check below enforces.
    """
    model_fp8 = _impl()
    from vllm_neuron.model.glm5_next.config import Glm5NextConfig

    _stack_geometry_preconditions()
    cfg = _stack_text_config()
    if model is None:
        model = model_fp8.Glm5NextModel(Glm5NextConfig(text_config=cfg), 1)
    layers = list(model.layers)
    if len(layers) != STACK_LAYERS:
        raise VacuousControlError(
            f"the config declares {STACK_LAYERS} layers and the tree built "
            f"{len(layers)}"
        )

    dense_at = [
        index for index, layer in enumerate(layers)
        if isinstance(layer.mlp, model_fp8.Glm5NextDenseMLP)
    ]
    moe_at = [
        index for index, layer in enumerate(layers)
        if isinstance(layer.mlp, model_fp8.Glm5NextMoEBlock)
    ]
    print(f"TINYFWD|stack_partition|dense_layers={dense_at}|moe_layers={moe_at}"
          f"|first_k_dense_replace={int(cfg.first_k_dense_replace)}")
    if dense_at != list(range(STACK_DENSE_LAYERS)) or moe_at != list(
        range(STACK_DENSE_LAYERS, STACK_LAYERS)
    ):
        raise VacuousControlError(
            f"_build_mlp put dense MLPs at {dense_at} and expert blocks at "
            f"{moe_at}; this item's references are written for the first "
            f"{STACK_DENSE_LAYERS} dense and the rest sparse"
        )
    if len(dense_at) != len(STACK_DENSE_SEED_OFFSETS):
        raise VacuousControlError(
            f"{len(dense_at)} dense layers and {len(STACK_DENSE_SEED_OFFSETS)} "
            f"seed offsets; two layers sharing a draw cannot show that each read "
            f"its own weights"
        )

    # ---- THE TWO MAPPED ROOT TENSORS. The table is drawn on the unsigned fp8 grid
    # and scaled down by a power of two, so the residual stream starts POSITIVE and
    # every value is exact in bf16 -- the stack's own dtype rounds nothing on entry.
    table = (
        _fp8_grid_values(SEED_STACK_EMBED, STACK_VOCAB_SIZE, STACK_HIDDEN_SIZE)
        * float(2.0**HIDDEN_SCALE_EXPONENT)
    ).to(torch.bfloat16)
    model.embed_tokens_weight = torch.nn.Parameter(table, requires_grad=False)
    final_gain = _stack_gain(2 * STACK_LAYERS)
    model.norm_weight = torch.nn.Parameter(final_gain, requires_grad=False)

    attention_operands = []
    mlp_operands: dict = {}
    for index, layer in enumerate(layers):
        layer.input_layernorm_weight = torch.nn.Parameter(
            _stack_gain(2 * index), requires_grad=False
        )
        layer.post_attention_layernorm_weight = torch.nn.Parameter(
            _stack_gain(2 * index + 1), requires_grad=False
        )
        attention_operands.append(
            _materialise_mla_attention(
                layer.self_attn,
                cfg,
                seed=SEED_STACK_ATTENTION + index,
                output_damping=STACK_ATTENTION_DAMPING,
            )
        )
        if index in dense_at:
            operands = _shared_at_routed_operands(
                seed_offset=STACK_DENSE_SEED_OFFSETS[dense_at.index(index)],
                gate_exponents=STACK_DENSE_GATE_EXPONENTS,
                up_exponents=STACK_DENSE_UP_EXPONENTS,
                down_exponent=STACK_DENSE_DOWN_EXPONENT,
            )
            for leaf in ("gate_proj_weight", "up_proj_weight", "down_proj_weight"):
                _attach(layer.mlp, leaf, *operands[leaf])
            mlp_operands[index] = operands
            continue

        # ---- THE EXPERT BLOCK. Item 4's recipe with the shared half absent: the
        # config declares ``n_shared_experts=0``, so no shared expert is built and
        # the block's forward returns the routed half. That add is item 4's own
        # certification and re-running it here would cost 1.5 M parameters.
        operands = _stack_bank_operands()
        for leaf in ("gate_proj_weight", "up_proj_weight", "down_proj_weight"):
            _attach(layer.mlp.experts, leaf, *operands[leaf])
        built = layer.mlp.experts.prepare_scale_operands(
            *_prep_operands_from_the_module(
                layer.mlp.experts, ("gate_proj_weight", "up_proj_weight", "down_proj_weight"), operands
            )
        )
        if built != 4:
            raise VacuousControlError(
                f"the bank's load-time prep built {built} operands; its forward "
                f"looks up 4"
            )
        if getattr(layer.mlp, "shared_experts", None) is not None:
            raise VacuousControlError(
                "the block built a shared expert at n_shared_experts=0, so this "
                "item's reference for the sparse layer is missing a term"
            )
        generator = torch.Generator().manual_seed(SEED_STACK_ROUTER)
        layer.mlp.experts.router_weight = torch.nn.Parameter(
            (
                torch.randn(
                    STACK_HIDDEN_SIZE, STACK_EXPERTS, generator=generator
                )
                * MOE_ROUTER_WEIGHT_SCALE
            ).to(torch.bfloat16),
            requires_grad=False,
        )
        layer.mlp.experts.router_bias = torch.nn.Parameter(
            (
                torch.randn(STACK_EXPERTS, generator=generator)
                * MOE_ROUTER_BIAS_SCALE
            ).to(torch.bfloat16),
            requires_grad=False,
        )
        mlp_operands[index] = operands

    parameters = sum(int(p.numel()) for p in model.parameters() if p is not None)
    widths = {
        "qk_nope_head_dim": int(cfg.qk_nope_head_dim),
        "qk_rope_head_dim": int(cfg.qk_rope_head_dim),
        "v_head_dim": int(cfg.v_head_dim),
        "index_head_dim": int(cfg.index_head_dim),
    }
    print(f"TINYFWD|stack_fixture|parameters={parameters}|bound={MAX_PARAMETERS}"
          f"|layers={STACK_LAYERS}|hidden={STACK_HIDDEN_SIZE}"
          f"|tokens={STACK_TOKENS}|head_widths={sorted(widths.items())}"
          f"|head_bound={MAX_HEAD_DIM}")
    if parameters >= MAX_PARAMETERS:
        raise VacuousControlError(
            f"the tiny stack holds {parameters} parameters, at or past the adopted "
            f"bound of {MAX_PARAMETERS}"
        )
    over = {name: width for name, width in widths.items() if width > MAX_HEAD_DIM}
    if over:
        raise VacuousControlError(
            f"{sorted(over.items())} exceeds the adopted tiny bound "
            f"head_dim <= {MAX_HEAD_DIM}"
        )
    return {
        "model": model,
        "cfg": cfg,
        "layers": layers,
        "table": table,
        "final_gain": final_gain,
        "attention_operands": attention_operands,
        "mlp_operands": mlp_operands,
        "dense_at": dense_at,
        "moe_at": moe_at,
    }


def _stack_carriers(layers, selection: dict) -> list:
    """One carrier mapping per layer, in stack order, each with its OWN caches.

    Both caches are written in place by the path, so two layers sharing one would
    have the second reading the first's latents. The eight keys are exactly
    ``Glm5NextDSALayer.forward``'s required keyword set plus ``slot_mapping``;
    ``tail`` and ``position`` are the decode leg's and are left at their defaults,
    which is what makes this the PREFILL leg.
    """
    return [
        {
            "latent_cache": _mla_latent_cache(
                layer.self_attn, tokens=STACK_TOKENS
            ),
            "pool_cache": _mla_pool_cache(pages=STACK_PAGES),
            "seq_lens": selection["seq_lens"],
            "start_position": 0,
            "softmax_scale": MLA_SOFTMAX_SCALE,
            "max_seq_len": STACK_TOKENS,
            "page_size": MLA_PAGE_SIZE,
            "slot_mapping": selection["slot_mapping"],
        }
        for layer in layers
    ]


def _stack_attention_half(layer, raw, gains, hidden, cfg, selection):
    """One layer's attention half in torch, from the tensor that layer RECEIVED.

    Returns ``(normed, topk_indices, attended)`` where ``attended`` is fp32 and
    carries no residual -- the add is the caller's, exactly as it is in the layer.

    THE INDEXER IS EXECUTED, item 4's and item 5's convention on selectors, and here
    it also removes this item's only real fragility. A pool selection is a
    DISCONTINUOUS function of its input: two candidate scores within float noise of
    each other can swap, and then both sides compute a different -- individually
    correct -- attention. Executing the indexer on the very tensor the layer was
    handed makes the reference's selection the layer's own selection by construction,
    so this item measures the composition instead of tie-breaking.

    ``normed.float()`` REACHES THE DENSE REFERENCE while the bf16 ``normed`` reaches
    the indexer. Same values either way -- bf16 to fp32 is exact -- and it keeps the
    reference in fp32 so the comparison absorbs exactly ONE bf16 rounding, the
    forward's own, rather than two.
    """
    normed = _ffn_norm(
        hidden, layer.input_layernorm_weight, float(cfg.rms_norm_eps)
    )
    attention = layer.self_attn
    q_latent = attention.project_query_latent(normed)
    topk_indices = attention.indexer(
        normed,
        q_latent,
        _mla_pool_cache(pages=STACK_PAGES),
        selection["seq_lens"],
        max_seq_len=STACK_TOKENS,
        page_size=MLA_PAGE_SIZE,
        slot_mapping=selection["slot_mapping"],
    )
    attended = _mla_dense_reference(
        attention,
        raw,
        gains,
        normed.float(),
        _mla_latent_cache(attention, tokens=STACK_TOKENS),
        topk_indices,
        softmax_scale=MLA_SOFTMAX_SCALE,
    )
    return normed, topk_indices, attended


def _stack_ffn_half(layer, hidden, cfg, operands, *, routed: bool, gain=None,
                    router_input=None, expert_input=None) -> dict:
    """One layer's FFN half in torch, from the tensor that layer's MLP RECEIVED.

    Returns whichever of :func:`_dense_output`'s or :func:`_routed_output`'s mappings
    applies; both carry ``out``, ``gate`` and ``up``, so the caller reads one shape.
    The residual add is the caller's.

    ``gain`` DEFAULTS TO THE LAYER'S POST-ATTENTION GAIN, which is the gain the FFN
    norm uses, and is an argument only so a control can recompute this half with the
    layer's INPUT gain and require the answer to leave the band.

    ``router_input`` DEFAULTS TO THE PRE-NORM TENSOR, which is what the fused router
    consumes: it applies the FFN RMSNorm itself, inside the kernel, so it takes the
    un-normalised states together with the norm's gain. It is an argument for the same
    reason -- a control feeds it the normalised tensor, which normalises twice and is
    a different router.

    ``expert_input`` DEFAULTS TO THE NORMALISED TENSOR, which is what the block's expert
    side consumes. It is an argument for one reason -- a reading below hands the experts
    the pre-norm states while the router gets the normalised tensor -- and it changes
    nothing at all when it is not passed.
    """
    limit = float(cfg.swiglu_limit)
    if gain is None:
        gain = layer.post_attention_layernorm_weight
    normed = _ffn_norm(hidden, gain, float(cfg.rms_norm_eps))
    if expert_input is not None:
        normed = expert_input
    if not routed:
        return _dense_output({**operands, "hidden": normed}, limit, -limit, limit)
    if router_input is None:
        router_input = hidden
    _logits, _index, affinities = layer.mlp.experts.route_tokens(
        router_input.unsqueeze(0), gain, cfg
    )
    return _routed_output(
        {**operands, "hidden": normed, "expert_affinities": affinities},
        mode=_POST_SCALE,
        gate_max=limit,
        gate_min=None,
        up_max=limit,
        up_min=-limit,
    )


def _stack_peak_band(expected: torch.Tensor, label: str) -> tuple:
    """``(rtol, atol)`` for one recompute comparison, scaled by the reference's peak.

    WHY A PEAK-SCALED ATOL AND NOT THIS FILE'S ``1e-5``. The two tensors compared at
    these sites are a bf16 forward against a float32 TORCH RECOMPUTE of the same
    arithmetic, and the recompute's error is set by the LARGEST INTERMEDIATE it
    passes through, not by the magnitude of the cell it lands in. Grant 117 read the
    worst cell at 0.002199 absolute where that cell's own value is at least 0.2189 --
    4.5 bf16 ULPs at the cell and 1.126 ULPs at the tensor's peak. So a per-cell
    relative band cannot hold at the near-zero cells, and a flat ``1e-5`` floor sits
    two orders below the error floor the recompute actually has.
    ``increments/proposal-054a-controls-r2.md`` sections 4.1 to 4.4 carry the
    arithmetic, and section 4.3 states plainly what this pair is: a LOOSENING below
    ``|expected| = 0.3035`` and a TIGHTENING above it, not the dtype's own bound.

    THE PEAK IS READ HERE, AT RUNTIME, from the reference itself, because the dense
    rescale moves every peak in this item and no constant frozen before the run would
    still be the peak after it.

    THAT OPENS ONE HOLE AND THIS CLOSES IT: a reference that BLEW UP would buy itself
    a wider budget. So the peak is refused above
    :data:`STACK_RECOMPUTE_PEAK_CEILING`, and a peak of zero is refused too, since a
    zero-peak band is a bare equality test that no real error could fail. The ceiling
    is 256: every peak this item predicts is at or below about 30, so it leaves
    roughly 8x of headroom, and it still refuses this fixture's own collapse peak of
    3440 by more than 13x. At the ceiling the absolute budget is 1.0, which is
    ``2**-8`` of the tensor it bounds -- so even at the bound the promise stays
    relative.
    """
    peak = float(expected.abs().max())
    if not peak > 0.0:
        raise VacuousControlError(
            f"{label}: the reference's peak is {peak}, so a peak-scaled band would "
            f"be a bare equality test and this comparison could not fail on any "
            f"error the recompute makes"
        )
    if peak > STACK_RECOMPUTE_PEAK_CEILING:
        raise VacuousControlError(
            f"{label}: the reference peaks at {peak:.6g}, above the declared ceiling "
            f"of {STACK_RECOMPUTE_PEAK_CEILING}. A reference that grew this far may "
            f"not widen its own tolerance; this fixture's own collapse produced a "
            f"peak of 3440"
        )
    return STACK_RECOMPUTE_RTOL, peak * STACK_RECOMPUTE_ATOL_FACTOR


def _stack_band_controls(produced: torch.Tensor, expected: torch.Tensor,
                         rtol: float, atol: float, label: str) -> None:
    """Controls A and B for one peak-scaled comparison, run AFTER it has passed.

    ORDERING IS LOAD-BEARING (``proposal-054a-controls-r2.md`` section 6.1). Each
    probe is built from PRODUCED, so it keeps the product's own error in every other
    cell; if the real comparison were still failing, a probe's failure would say
    nothing about the band. The planted cell is the argmin of ``|expected|``, where a
    peak-scaled atol dominates the relative term and so where the loosening lives,
    and its value is OVERWRITTEN rather than added to, so the planted delta is
    exactly the number named.

    EACH ARM IS A MULTIPLE OF THAT CELL'S OWN ALLOWANCE, and that is the R1 repair
    (LEAD-LOG 850). ``torch.allclose`` accepts a cell whose error is at most
    ``atol + rtol * |expected|`` THERE, so an arm planted as a multiple of ``atol``
    alone is a multiple of only part of the allowance. At conjunct 3 layer 1 the
    argmin cell held ``|expected|`` 8.67 against a peak of 12.17, the relative term
    added 0.0339 to the 0.0475 atol, and the 1.5 x atol arm landed INSIDE a 0.0814
    allowance -- so the arm could not fail and the fixture reddened an item that was
    behaving (``increments/launch-054a-r16-driver-20260909T080034Z.out:260``).
    Planting ``factor * (atol + rtol * |expected|)`` restores the arithmetic at every
    cell of every tensor, whatever its spread:

    * ``A`` plants 2.0 allowances and MUST fail, because 2 > 1.
    * ``B_under`` plants 0.5 allowances and MUST pass, because 0.5 < 1.
    * ``B_over`` plants 1.5 allowances and MUST fail, because 1.5 > 1.

    NO ARM HAS A PRECONDITION LEFT. The two ``discriminates_iff`` fields stay in the
    header row as READINGS -- they say where the old atol-only arms would have been
    blind -- and neither is consulted to decide whether an arm may raise. A raise
    happens only when an arm's measured verdict contradicts that arm's own
    requirement, which is a statement about the band and never about the data.
    """
    peak = float(expected.abs().max())
    cols = int(expected.shape[-1])
    row, col = divmod(int(expected.abs().reshape(-1).argmin()), cols)
    a_min = float(expected.abs()[row, col])
    allowance = atol + rtol * a_min
    print(f"TINYFWD|stack_band_control|site={label}|row={row}|col={col}"
          f"|expected_at_cell={float(expected[row, col]):.10g}"
          f"|abs_at_cell={a_min:.10g}|peak={peak:.10g}"
          f"|rtol={rtol:.10g}|atol={atol:.10g}"
          f"|allowance_at_cell={allowance:.10g}"
          f"|a_discriminates_iff_peak_gt_cell={peak > a_min}"
          f"|b_over_discriminates_iff_cell_lt_half_peak={a_min < peak / 2.0}"
          f"|note=both iff fields are readings about the retired atol-only arms; the"
          f" arms below are multiples of allowance_at_cell and have no precondition")
    for arm, factor, must_fail in (("A", 2.0, True),
                                   ("B_under", 0.5, False),
                                   ("B_over", 1.5, True)):
        probe = produced.clone()
        probe[row, col] = expected[row, col] + factor * allowance
        inside = bool(torch.allclose(probe, expected, rtol=rtol, atol=atol))
        print(f"TINYFWD|stack_band_control|site={label}|arm={arm}"
              f"|planted_multiple_of_allowance={factor}"
              f"|delta={factor * allowance:.10g}"
              f"|inside_band={inside}|must_fail={must_fail}")
        if must_fail and inside:
            raise VacuousControlError(
                f"{label}: control {arm} planted {factor} x the argmin cell's own "
                f"allowance = {factor * allowance:.6g}, which is atol {atol:.6g} "
                f"plus rtol {rtol:.6g} x |expected| {a_min:.6g}, and the band still "
                f"accepted it -- so this band cannot see an error of that size"
            )
        if not must_fail and not inside:
            raise VacuousControlError(
                f"{label}: control {arm} planted {factor} x the argmin cell's own "
                f"allowance = {factor * allowance:.6g} and the band REFUSED it, so "
                f"the band is tighter than the pair it reports"
            )


def _diag_mode() -> bool:
    """Diagnostic mode, and it is OFF unless the launcher sets it.

    ``TINY_054A_DIAG=1`` makes item 6's refusal sites after conjunct 5 PRINT what they
    would have raised and CONTINUE, so ONE run enumerates every one of them instead of
    one run per site. With the variable unset -- which is every run this fixture has
    ever had -- each site raises exactly as it does at ``0f5182a``.
    """
    return os.environ.get("TINY_054A_DIAG", "") == "1"


def _diag_or_raise(site: str, message: str, *, gap: str = "n/a",
                   band: str = "n/a") -> None:
    """Raise ``VacuousControlError(message)``; in DIAG mode print the row and continue."""
    if _diag_mode():
        print(f"TINYFWD|diag_control|site={site}|gap={gap}|band={band}"
              f"|would_raise=True")
        return
    raise VacuousControlError(message)


def _stack_outside_tolerance(label: str, moved: torch.Tensor,
                             base: torch.Tensor) -> None:
    """One control: ``moved`` must fall OUTSIDE this item's band.

    This file's declared control form -- recompute the reference with one branch
    changed and require the result to leave the tolerance the item passes inside, so
    a fixture that stopped discriminating fails as a control instead of passing as an
    item.
    """
    outside = not torch.allclose(moved.float(), base.float(), rtol=RTOL, atol=ATOL)
    gap = float((moved.float() - base.float()).abs().max() / base.abs().max())
    print(f"TINYFWD|stack_control|branch={label}|outside_tolerance={outside}"
          f"|gap={gap:.6f}")
    if _diag_mode():
        print(f"TINYFWD|diag_control|site={label}|gap={gap:.6f}"
              f"|band=rtol {RTOL} atol {ATOL}|would_raise={not outside}")
        return
    if not outside:
        raise VacuousControlError(
            f"{label}: the change leaves the reference inside rtol={RTOL}, "
            f"atol={ATOL}, so this item cannot tell the two apart"
        )


# --------------------------------------------------------------------------- #
# ITEM 6 of 7 -- ``Glm5NextModel.forward``.                                    #
# Certifying component: ``model_fp8.Glm5NextModel.forward`` together with its   #
# two private helpers, ``_ffn_half`` and ``_rms_norm``.                         #
#                                                                              #
# WHAT IT CERTIFIES. Four things, and nothing a callee already owns: the        #
# embedding is an INDEX into the mapped table; every layer runs once, in config #
# order, on the previous layer's output; each layer receives EXACTLY its own    #
# carrier mapping; each layer's FFN half normalises with that layer's own       #
# post-attention gain and takes the MLP branch ``_build_mlp`` gave it; and the  #
# stack's final norm closes the chain. The attention numerics are item 5's, the #
# expert bank's are items 2 and 4's, the dense MLP's are item 1's.              #
#                                                                              #
# THE COMPARISON IS MADE AT THE LAYER BOUNDARIES THE FORWARD ITSELF PRODUCED,   #
# and that is what makes this item sound rather than merely convenient. Two     #
# torch forward hooks record what each layer was handed and what it returned;   #
# the reference for each half is then built from the recorded tensor and         #
# compared against the next recorded tensor. An end-to-end comparison against a #
# reference that carried its OWN hidden states would be hostage to a pool       #
# selection flipping between the two chains -- the selection is a discontinuous #
# function of its input and the two chains differ by kernel noise -- so it      #
# would measure tie-breaking rather than composition. The end-to-end equality   #
# FOLLOWS from the chain below and is stated rather than separately measured.   #
#                                                                              #
# THE HOOKS NEED ``with_kwargs``, torch 2.0's. An older torch raises            #
# ``TypeError`` at registration, which is a red run and not a quiet pass, and   #
# both hooks are asserted to have fired once per layer before anything is read  #
# out of them.                                                                  #
#                                                                              #
# THE PRECISION BUDGET, STATED SO IT CAN BE CHECKED RATHER THAN TRUSTED. This   #
# stack runs in bf16 -- the checkpoint's activation dtype and the dtype every    #
# seam on it declares -- while every reference below stays in fp32. So each      #
# comparison carries the forward's OWN roundings and none of its own. The        #
# attention boundary carries one: half an ulp, 2**-9 = 0.20% of a value, 20% of  #
# this item's 1% ``rtol``. The FFN boundary carries two, the sublayer output and  #
# the residual sum, so at most 0.39% or 39% of the budget. ``attend()`` also     #
# rounds its latent-space result to the stream dtype mid-chain, which item 5     #
# never sees because item 5 hands the forward fp32; damping ``o_proj`` by an     #
# eighth (:data:`STACK_ATTENTION_DAMPING`) is what keeps that term under 5%.     #
# NONE OF THESE FIGURES IS MEASURED YET: no run of this file exists, so the      #
# transcript of the first counted run is what adjudicates the arithmetic above.  #
#                                                                              #
# THE STACK IS ONE FAMILY AND ONE PHASE, disclosed rather than implied. Every   #
# layer is ``deepseek_sparse_attention`` and every carrier is a prefill         #
# carrier; the linear-attention family is ``inc-glm53f-038``'s and the decode   #
# leg is ``inc-glm53f-042``'s and ``inc-glm53f-051``'s. What this item measures #
# about the loop -- that it holds no per-family branch and hands each layer its #
# own mapping -- is measured exactly as well by one family as by two.           #
#                                                                              #
# THE ONE-STREAM EXCLUSION THAT STOOD HERE IS RETIRED, AND THIS ITEM IS BEING  #
# RE-POINTED. It used to say the inter-layer carrier was a ``[T, H]`` add and   #
# that the checkpoint's 4-stream mHC carrier was ``inc-glm53f-030b``'s.         #
# ``inc-glm53f-030d`` part (a) built that carrier in                            #
# ``Glm5NextModel.forward``: the embedding is expanded across the stream axis   #
# (``modeling_glm5_next.py:1477``), both per-layer sites mix                    #
# (``:1316-1318``, ``:1325-1327``) and an unweighted mean collapses the streams #
# before the final norm (``:1493``, ``:302``).                                  #
#                                                                              #
# SO THE PER-LAYER ADD THIS ITEM READS THROUGH ITS HOOKS IS FALSE BY            #
# CONSTRUCTION, and saying so is the honest state of this file rather than a    #
# claim that the body already changed. The stack passes streams unconditionally #
# and ``_mhc_site`` REFUSES a streams call on a layer carrying none of the six  #
# mHC leaves (``model_fp8.py:7578-7582``), which this fixture does not load, so #
# this item cannot run against the carrier until its fixture loads those        #
# leaves, binds the sites, and its conjuncts compare the mixes instead of the   #
# adds. That is the re-point ruled at §943 Q5 and it is the NEXT commit's, not  #
# this one's; the plan carries it as this block's own work rather than a        #
# deletion.                                                                     #
# --------------------------------------------------------------------------- #
def _row_spread_stats(rows: "torch.Tensor") -> tuple:
    """Relative L2 spread over every pair of rows: ``(rows, max, min, median)``.

    READING SUPPORT ONLY. Nothing in here gates and nothing in here stops a caller. The metric is the
    pairwise L2 distance divided by the mean of the two row norms, so it is scale free and comparable
    across stages whose magnitudes differ by orders of magnitude. Every pair is measured rather than
    sampled: for ``n`` rows that is ``n * (n - 1) / 2`` values, and the upper triangle is taken so no
    pair is counted twice and no row is compared against itself.
    """
    flat = rows.reshape(rows.shape[0], -1).float()
    dist = (flat.unsqueeze(1) - flat.unsqueeze(0)).norm(dim=2)
    norms = flat.norm(dim=1)
    mean = (norms.unsqueeze(1) + norms.unsqueeze(0)) / 2.0
    spread = dist / mean.clamp_min(1e-12)
    upper = torch.triu(torch.ones_like(spread), diagonal=1) > 0
    vals = spread[upper]
    if vals.numel() == 0:
        return (int(flat.shape[0]), 0.0, 0.0, 0.0)
    return (int(flat.shape[0]), float(vals.max()), float(vals.min()),
            float(vals.median()))


def test_tiny_model_forward_matches_the_reference() -> None:
    """The decoder stack equals its torch composition, layer boundary by boundary.

    ``inc-glm53f-054a`` item 6 of 7. D1.4 certifying component:
    ``Glm5NextModel.forward`` with ``_ffn_half`` and ``_rms_norm`` -- the embedding
    index, the per-layer loop and its two residual adds, the FFN branch and its gain,
    and the final norm.
    """
    fixture = _stack_fixture()
    model, cfg, layers = fixture["model"], fixture["cfg"], fixture["layers"]
    quant_config = _quant_config()
    selection = _mla_selection_operands(tokens=STACK_TOKENS, pages=STACK_PAGES)
    carriers = _stack_carriers(layers, selection)

    input_ids = torch.randint(
        0, STACK_VOCAB_SIZE, (STACK_TOKENS,),
        generator=torch.Generator().manual_seed(SEED_STACK_IDS),
        dtype=torch.int64,
    )
    if int(input_ids.unique().numel()) < STACK_LAYERS:
        raise VacuousControlError(
            f"the token ids take only {int(input_ids.unique().numel())} distinct "
            f"values; a near-constant embedding makes every row of the residual "
            f"stream alike and the comparison stops discriminating"
        )

    # ---- THE HOOKS. One pre-hook and one post-hook per layer, recording the tensor
    # and the mapping each layer was handed and the tensor it returned.
    recorded_in: list = []
    recorded_out: list = []
    recorded_mlp: list = []

    def _record_input(module, args, kwargs):
        recorded_in.append((module, args, kwargs))

    def _record_output(module, args, kwargs, output):
        recorded_out.append((module, output))

    def _record_mlp(module, args, kwargs, output):
        recorded_mlp.append((module, args, kwargs, output))

    handles = []
    for layer in layers:
        handles.append(
            layer.register_forward_pre_hook(_record_input, with_kwargs=True)
        )
        handles.append(
            layer.register_forward_hook(_record_output, with_kwargs=True)
        )
    # THE LAST LAYER'S MLP TOO, FOR A READING ONLY. Conjunct 5 is the only place the
    # routed bank's output reaches a comparison, and its own agreement with the torch
    # recompute is not measured anywhere at this band. This hook lets the reading
    # below report that term directly instead of leaving it to another round. It
    # rides the same handle list, so the finally below removes it.
    handles.append(
        layers[-1].mlp.register_forward_hook(_record_mlp, with_kwargs=True)
    )

    try:
        _reset_seam_counters()
        before = _read_seam_counters()
        got = model.forward(
            input_ids,
            layer_carriers=carriers,
            quant_config=quant_config,
        )
        after = _read_seam_counters()
    finally:
        for handle in handles:
            handle.remove()

    # ---- THE REGISTERED ROUTE PREDICATE, around this item's own forward. EVERY
    # FIGURE IS ITEM 5's PER-LAYER SET TIMES THE LAYER COUNT, plus item 1's three
    # dense dispatches per dense layer and item 4's two per sparse layer. The
    # per-layer figures are not re-derived here: item 5 reads them off
    # ``test_dsa_layer.py``'s landed closed form, and that table is per layer per
    # PHASE (``DECLARED_PER_LAYER``), so it does not move with the token count.
    route_expected = {
        "mla_projection": 9 * STACK_LAYERS,
        "mla_absorb": 2 * STACK_LAYERS,
        "mla_sparse": 1 * STACK_LAYERS,
        "dsa_kpool_hadamard": 2 * STACK_LAYERS,
        "dsa_paged_gather": 1 * STACK_LAYERS,
        "dsa_score_gemm": 1 * STACK_LAYERS,
        "dsa_topk_select": 1 * STACK_LAYERS,
        "dsa_index_expand": 1 * STACK_LAYERS,
        "blockwise_fp8_mm": 3 * STACK_DENSE_LAYERS,
        "blockwise_fp8_moe": 1 * STACK_MOE_LAYERS,
        "noaux_tc_router": 1 * STACK_MOE_LAYERS,
    }
    _declare_bound_and_sentinel(route_expected)
    _assert_route_predicate("6 the decoder stack", route_expected, before, after)

    # ---- THE HOOKS FIRED ONCE PER LAYER, IN STACK ORDER. Read before anything is
    # taken out of them: a loop that skipped a layer, ran one twice or ran them out
    # of order is a different list here, and every comparison below stands on this.
    print(f"TINYFWD|stack_trace|recorded_in={len(recorded_in)}"
          f"|recorded_out={len(recorded_out)}|layers={len(layers)}")
    if len(recorded_in) != len(layers) or len(recorded_out) != len(layers):
        raise VacuousControlError(
            f"the hooks recorded {len(recorded_in)} entries and "
            f"{len(recorded_out)} exits for {len(layers)} layers, so the forward "
            f"did not run each layer exactly once"
        )
    for index, layer in enumerate(layers):
        if recorded_in[index][0] is not layer or recorded_out[index][0] is not layer:
            raise VacuousControlError(
                f"position {index} of the recorded order is not layer {index} of "
                f"the stack, so the loop did not run the layers in config order"
            )

    # ---- READINGS ONLY, BLOCK A: is each row of the hidden state distinct, stage by stage.
    # Placed HERE, above every comparison in this item, so a per-layer redness further down cannot
    # suppress the rows that would explain it. Grant 117 measured the root item's 128 rows as bit
    # identical -- spread exactly 0 over all 8128 pairs -- and these rows say whether they are
    # already equal at the embedding or become equal at a layer, and at which one. Nothing gates.
    _sp = _row_spread_stats(recorded_in[0][1][0])
    print(f"TINYFWD|rowspread_stage|stage=embedding"
          f"|dtype={recorded_in[0][1][0].dtype}|rows={_sp[0]}"
          f"|max_spread={_sp[1]:.6g}|min_spread={_sp[2]:.6g}|median_spread={_sp[3]:.6g}")
    for _stage in range(len(layers)):
        _sin = recorded_in[_stage][1][0]
        _sout = recorded_out[_stage][1]
        _spi = _row_spread_stats(_sin)
        _spo = _row_spread_stats(_sout)
        print(f"TINYFWD|rowspread_stage|stage=layer{_stage}_in"
              f"|dtype={_sin.dtype}|rows={_spi[0]}|max_spread={_spi[1]:.6g}"
              f"|min_spread={_spi[2]:.6g}|median_spread={_spi[3]:.6g}")
        print(f"TINYFWD|rowspread_stage|stage=layer{_stage}_out"
              f"|dtype={_sout.dtype}|rows={_spo[0]}|max_spread={_spo[1]:.6g}"
              f"|min_spread={_spo[2]:.6g}|median_spread={_spo[3]:.6g}")
    # ---- end of the BLOCK A readings.
    # ---- BLOCK C: THE SEAM READINGS. `investigation-054a-seam-r1.md` section 7, taken in BLOCK A
    # POSITION -- above every comparison -- so a red conjunct cannot suppress them. Every row re-runs
    # `model_fp8.py:6714`'s own pieces on the RECORDED layer-0 output, which is the object the stack
    # loop handed forward. READINGS ONLY: nothing here gates and nothing here raises. The whole block
    # is wrapped so a defect in THESE lines cannot change what the item decides -- the except prints a
    # named row instead of killing the item.
    try:
        import math as _math

        from torch.nn.functional import silu as _silu

        from vllm_neuron.functional.blockwise_fp8_mm import blockwise_fp8_mm as _bmm

        _L0 = 0
        _hidden = recorded_out[_L0][1]
        _layer0 = layers[_L0]
        _gain0 = _layer0.post_attention_layernorm_weight
        _mlp0 = _layer0.mlp
        _normed0 = model._rms_norm(_hidden, _gain0)
        _out0 = _mlp0(_normed0, quant_config=quant_config)
        _out0c = _out0.to(_hidden.dtype)
        _sum0 = _hidden + _out0c

        # The three public scale grids, by the name the product's own lookup builds
        # (`model_fp8.py:3536-3538`). Read as a dict comprehension rather than a helper, because a
        # `return` anywhere in this item's body could skip a comparison and the checker below bans one.
        _grids = {
            _leaf: getattr(_mlp0, f"{_leaf[: -len('_weight')]}_weight_scale_inv")
            for _leaf in ("gate_proj_weight", "up_proj_weight", "down_proj_weight")
        }

        # ---- C2, TAKEN FIRST. Absorption is measured DIRECTLY -- the fraction of elements where the
        # sum equals the FFN term bit for bit -- so the reading does not depend on any ULP argument.
        # `sum_max_spread` is the faithfulness check: grant 121 read `layer1_in` as 0, and this row
        # recomputes the same tensor, so a 0 here means this block reproduces the real seam.
        _hp = float(_hidden.float().abs().max())
        _fp = float(_out0c.float().abs().max())
        _ulp = 2.0 ** (int(_math.floor(_math.log2(_fp))) - 7) if _fp > 0.0 else 0.0
        _absorbed = float((_sum0.float() == _out0c.float()).float().mean())
        print(f"TINYFWD|seam_add|layer={_L0}|hidden_peak={_hp:.10g}|ffn_peak={_fp:.10g}"
              f"|ratio={_fp / max(_hp, 1e-30):.6g}|one_ulp_at_ffn_peak={_ulp:.10g}"
              f"|elements_absorbed_frac={_absorbed:.6f}"
              f"|sum_max_spread={_row_spread_stats(_sum0)[1]:.6g}"
              f"|note=absorbed counts elements where hidden plus out equals out bit for bit")

        # ---- C4: the three matmul seams, read BEFORE the clamp. Measured here and not after, because
        # after the clamp a saturating clamp and a broadcasting kernel are indistinguishable.
        _gate = _bmm(_normed0, _mlp0.gate_proj_weight, _grids["gate_proj_weight"])
        _up = _bmm(_normed0, _mlp0.up_proj_weight, _grids["up_proj_weight"])
        _gs = _row_spread_stats(_gate)
        _us = _row_spread_stats(_up)
        print(f"TINYFWD|seam_kernel|layer={_L0}|gate_max_spread={_gs[1]:.6g}"
              f"|up_max_spread={_us[1]:.6g}|gate_min_spread={_gs[2]:.6g}|up_min_spread={_us[2]:.6g}"
              f"|gate_peak={float(_gate.float().abs().max()):.10g}"
              f"|up_peak={float(_up.float().abs().max()):.10g}|measured=before the clamp")

        # ---- C1: the SwiGLU clamps, and the faithfulness check for C4's operands.
        _lim = float(_mlp0.swiglu_limit)
        _gc = _gate.clamp(min=None, max=_lim)
        _uc = _up.clamp(min=-_lim, max=_lim)
        _act = _silu(_gc) * _uc
        _re = _bmm(_act.to(_hidden.dtype), _mlp0.down_proj_weight, _grids["down_proj_weight"])
        print(f"TINYFWD|seam_clamp|layer={_L0}|limit={_lim:.10g}"
              f"|gate_at_limit_frac={float((_gate.float() >= _lim).float().mean()):.6f}"
              f"|up_at_limit_frac={float((_up.float().abs() >= _lim).float().mean()):.6f}"
              f"|activated_max_spread={_row_spread_stats(_act)[1]:.6g}"
              f"|recompute_matches_real_out={float((_re.float() - _out0.float()).abs().max()):.6g}"
              f"|note=recompute near zero means the gate and up above are the real pre-clamp ones")

        # ---- C3: direction versus scale. `_row_spread_stats` divides by the mean row norm but does
        # NOT remove each row's own scale, so it cannot tell parallel rows of different length from
        # rows pointing different ways. `unit_max_spread` removes the scale first and settles it.
        _flat0 = _hidden.reshape(_hidden.shape[0], -1).float()
        _norms0 = _flat0.norm(dim=1)
        _unit0 = _flat0 / _norms0.unsqueeze(1).clamp_min(1e-30)
        print(f"TINYFWD|seam_direction|stage=layer0_out"
              f"|raw_max_spread={_row_spread_stats(_hidden)[1]:.6g}"
              f"|unit_max_spread={_row_spread_stats(_unit0)[1]:.6g}"
              f"|normed_max_spread={_row_spread_stats(_normed0)[1]:.6g}"
              f"|row_norm_min={float(_norms0.min()):.10g}|row_norm_max={float(_norms0.max()):.10g}"
              f"|note=unit removes each row's scale before the spread, raw does not")

        # ---- THE FIXTURE'S OWN DRAW, so an order-100 FFN output can be traced to the random draw or
        # ruled out as its cause. The weights are fp8 values times a per-block power-of-two grid, so
        # both halves are printed: the stored values and the grid that scales them.
        print(f"TINYFWD|seam_weights|layer={_L0}"
              f"|gate_abs_max={float(_mlp0.gate_proj_weight.to(torch.float32).abs().max()):.10g}"
              f"|up_abs_max={float(_mlp0.up_proj_weight.to(torch.float32).abs().max()):.10g}"
              f"|down_abs_max={float(_mlp0.down_proj_weight.to(torch.float32).abs().max()):.10g}"
              f"|gate_grid_max={float(_grids['gate_proj_weight'].to(torch.float32).max()):.10g}"
              f"|up_grid_max={float(_grids['up_proj_weight'].to(torch.float32).max()):.10g}"
              f"|down_grid_max={float(_grids['down_proj_weight'].to(torch.float32).max()):.10g}"
              f"|gate_exponents={SHARED_AT_ROUTED_GATE_EXPONENTS}"
              f"|up_exponents={SHARED_AT_ROUTED_UP_EXPONENTS}"
              f"|down_exponent={SHARED_AT_ROUTED_DOWN_EXPONENT}"
              f"|dense_seed_offset={STACK_DENSE_SEED_OFFSETS[0]}"
              f"|normed_peak={float(_normed0.float().abs().max()):.10g}")
    except Exception as _seam_exc:
        print(f"TINYFWD|seam_error|stage=block_c|exception={type(_seam_exc).__name__}: {_seam_exc}")
    # ---- end of the BLOCK C readings.

    # ---- BLOCK D: THE THREE SCALE GUARDS, in BLOCK A position -- above every
    # comparison -- so a red conjunct cannot suppress their readings. Each guard
    # carries THE OLD SCALE as its EXECUTED failing control.
    #
    # WHY THEY EXIST. Grant 127 read layer 0 saturating both SwiGLU clamps at 100%,
    # which made every activated row the same constant, drove the FFN half to 3440
    # against a hidden peak of 0.4766, and let the residual add absorb the hidden
    # term in every element -- after which every row below layer 0 was identical and
    # three of this file's items could not see a per-token defect at all. These
    # guards refuse that fixture, and a guard that only ever fired on the scale it
    # was written for would be untestable, so each one is re-run on the OLD grids and
    # required to FIRE there.
    #
    # LAYER 0'S CONTROL IS EXACT. The FFN half is added OUTSIDE the layer call
    # (``model_fp8.py:6712-6714``), so ``recorded_out[0][1]`` is the ATTENTION half
    # and the dense rescale cannot move it. The old-scale recompute at layer 0
    # therefore runs on the very tensor grant 127 ran on.
    #
    # NOTHING HERE TOUCHES A DISPATCH COUNTER. ``_stack_ffn_half`` on a dense layer
    # reaches ``_ffn_norm`` and ``_dense_output`` only, both pure torch, and the
    # route predicate's window closed above in any case.
    from torch.nn.functional import silu as _dsilu

    _old_dense_ops = {}
    for _di, _dense_index in enumerate(fixture["dense_at"]):
        _dh = recorded_out[_dense_index][1]
        _dlimit = float(cfg.swiglu_limit)
        _old_dense_ops[_dense_index] = _shared_at_routed_operands(
            seed_offset=STACK_DENSE_SEED_OFFSETS[_di]
        )
        _halves = (
            ("new", _stack_ffn_half(
                layers[_dense_index], _dh, cfg,
                fixture["mlp_operands"][_dense_index], routed=False)),
            ("old", _stack_ffn_half(
                layers[_dense_index], _dh, cfg,
                _old_dense_ops[_dense_index], routed=False)),
        )
        _readings = {}
        for _tag, _half in _halves:
            _g, _u = _half["gate"], _half["up"]
            _act = _dsilu(_clamped(_g, None, _dlimit)) * _clamped(
                _u, -_dlimit, _dlimit
            )
            _o = _half["out"].to(_dh.dtype)
            _sum = _dh + _o
            _reading = (
                float((_g >= _dlimit).float().mean()),
                float((_u.abs() >= _dlimit).float().mean()),
                _row_spread_stats(_act)[1],
                float((_sum.float() == _o.float()).float().mean()),
                _row_spread_stats(_sum)[1],
            )
            _readings[_tag] = _reading
            _hpeak = float(_dh.float().abs().max())
            _fpeak = float(_half["out"].abs().max())
            print(f"TINYFWD|stack_scale_guard|layer={_dense_index}|scale={_tag}"
                  f"|gate_peak={float(_g.abs().max()):.10g}"
                  f"|up_peak={float(_u.abs().max()):.10g}|limit={_dlimit:.10g}"
                  f"|gate_at_limit_frac={_reading[0]:.6f}"
                  f"|up_at_limit_frac={_reading[1]:.6f}"
                  f"|activated_max_spread={_reading[2]:.6g}"
                  f"|ffn_peak={_fpeak:.10g}|hidden_peak={_hpeak:.10g}"
                  f"|ffn_share={_fpeak / max(_hpeak, 1e-30):.6g}"
                  f"|elements_absorbed_frac={_reading[3]:.6f}"
                  f"|sum_max_spread={_reading[4]:.6g}"
                  f"|absorption_reading_bound={STACK_ABSORPTION_CEILING}")

        _gf, _uf, _as, _absorbed, _sumspread = _readings["new"]
        # GUARD (i): THE CLAMP MUST BIND, AND MUST NOT BIND EVERYWHERE. Both
        # fractions above zero says the SwiGLU bound is live on both projections;
        # a nonzero activated row spread says it did not saturate every row into one
        # constant. The fractions themselves are READINGS -- no threshold on them is
        # ruled yet, and this refuses only the two degenerate ends.
        if not (_gf > 0.0 and _uf > 0.0 and _as > 0.0):
            raise VacuousControlError(
                f"layer {_dense_index}: gate at-limit fraction {_gf:.6f}, up "
                f"at-limit fraction {_uf:.6f}, activated row spread {_as:.6g}. This "
                f"item needs the SwiGLU clamp live on BOTH projections and needs the "
                f"activated rows to still differ: a zero fraction means the clamp is "
                f"dead at this scale, and a zero spread means it saturated every row "
                f"into the same constant"
            )
        # GUARD (ii): THE RESIDUAL ADD MUST NOT ABSORB THE HIDDEN TERM. STRUCTURAL,
        # for the same reason guard (i) is: absorption strictly below every element,
        # and a sum whose rows still differ. That pair IS the collapse being absent,
        # and it needs no threshold nobody has measured yet -- the fraction itself is
        # printed above beside :data:`STACK_ABSORPTION_CEILING` as a reading.
        if not (_absorbed < 1.0 and _sumspread > 0.0):
            raise VacuousControlError(
                f"layer {_dense_index}: the residual add absorbs the hidden term in "
                f"{_absorbed:.6f} of elements and the sum's row spread is "
                f"{_sumspread:.6g}. The FFN half peaks at "
                f"{float(_halves[0][1]['out'].abs().max()):.6g} against a hidden "
                f"peak of {float(_dh.float().abs().max()):.6g}, so the sum carries "
                f"the FFN half alone and every comparison below it is blind to the "
                f"other one"
            )
        # THE FAILING CONTROLS, EXECUTED ONLY WHERE THE CONTROL TENSOR IS EXACT.
        #
        # The OLD grids must fire BOTH guards, or the guards are statements no scale
        # in this file's history could have violated. But this control is a
        # REPRODUCTION of grant 127, and it reproduces only where it runs on the
        # tensor grant 127 ran on. Layer 0 is that place: the FFN half is added
        # OUTSIDE the layer call, so ``recorded_out[0][1]`` is the attention half and
        # this commit cannot move it. Layer 1 is not: its input carries layer 0's FFN
        # half AT THE NEW SCALE, which no run has measured.
        #
        # WHY REQUIRING IT THERE WOULD BE A FALSE RED. At the old scale a layer-1 row
        # sums about 3200 per column, so one bf16 ULP is 16 and half a ULP is 8; a
        # single layer-1 element reaching 8 leaves ``_oabs`` below 1.0 and ``_osum``
        # above 0, and the guard (ii) control raises on a healthy fixture. And
        # ``_oas == 0`` needs EVERY gate element and EVERY |up| element at or past
        # the limit in EVERY row, which grant 127 proved for layer 0 only.
        #
        # SO THE RAISES RUN WHERE THE CONTROL IS EXACT, and layer 1's old-scale
        # numbers stay the READINGS they always were: the ``scale=old`` row above
        # prints them, and the legend row below says which layer's control gates and
        # which only reads. Nothing else about the guards moves.
        _ogf, _ouf, _oas, _oabs, _osum = _readings["old"]
        _control_is_exact = not any(
            _below < _dense_index for _below in fixture["dense_at"]
        )
        _control_class = "gated" if _control_is_exact else "reading_only"
        print(f"TINYFWD|stack_scale_guard_legend|layer={_dense_index}"
              f"|old_scale_control={_control_class}"
              f"|old_gate_at_limit_frac={_ogf:.6f}"
              f"|old_up_at_limit_frac={_ouf:.6f}"
              f"|old_activated_max_spread={_oas:.6g}"
              f"|old_elements_absorbed_frac={_oabs:.6f}"
              f"|old_sum_max_spread={_osum:.6g}"
              f"|gates_iff_no_dense_layer_sits_below_this_one={_control_is_exact}"
              f"|note=only there is the old-scale recompute the tensor grant 127 read")
        if _control_is_exact:
            if _ogf > 0.0 and _ouf > 0.0 and _oas > 0.0:
                raise VacuousControlError(
                    f"layer {_dense_index}: guard (i)'s control did not "
                    f"fire. The OLD exponents give gate at-limit {_ogf:.6f}, "
                    f"up at-limit {_ouf:.6f}, activated spread {_oas:.6g}, all "
                    f"of which the guard accepts -- so the guard is not what "
                    f"tells the two scales apart"
                )
            if _oabs < 1.0 and _osum > 0.0:
                raise VacuousControlError(
                    f"layer {_dense_index}: guard (ii)'s control did not "
                    f"fire. The OLD exponents absorb the hidden term in "
                    f"{_oabs:.6f} of elements and leave the sum's row spread "
                    f"at {_osum:.6g}, both of which the guard accepts -- so "
                    f"the guard is not what tells the two scales apart"
                )

    # ---- GUARD (iii): EVERY STAGE BELOW LAYER 0 STILL CARRIES DISTINCT ROWS. Grant
    # 127 read `layer1_in`, `layer1_out`, `layer2_in` and `layer2_out` at a row
    # spread of EXACTLY ZERO, which is the collapse in one number per stage.
    for _si in range(1, len(layers)):
        for _stage, _t in ((f"layer{_si}_in", recorded_in[_si][1][0]),
                           (f"layer{_si}_out", recorded_out[_si][1])):
            _rows, _mx, _mn, _md = _row_spread_stats(_t)
            print(f"TINYFWD|stack_row_guard|stage={_stage}|rows={_rows}"
                  f"|max_spread={_mx:.6g}|min_spread={_mn:.6g}"
                  f"|median_spread={_md:.6g}")
            if not _mx > 0.0:
                raise VacuousControlError(
                    f"{_stage} carries {_rows} rows whose pairwise spread is exactly "
                    f"zero, so every token below layer 0 holds the same vector and "
                    f"no comparison here can see a per-token defect"
                )
    # ITS FAILING CONTROL, at the seam the collapse ran through: layer 1's input
    # rebuilt with the OLD grids. One dense recompute, not a second stack run.
    _c0 = fixture["dense_at"][0]
    _ch = recorded_out[_c0][1]
    _cold = _stack_ffn_half(
        layers[_c0], _ch, cfg, _old_dense_ops[_c0], routed=False
    )["out"].to(_ch.dtype)
    _cspread = _row_spread_stats(_ch + _cold)[1]
    print(f"TINYFWD|stack_row_guard|stage=layer{_c0 + 1}_in_at_the_old_scale"
          f"|max_spread={_cspread:.6g}|bound=0"
          f"|note=this is the seam grant 127 read at zero")
    if _cspread > 0.0:
        raise VacuousControlError(
            f"guard (iii)'s control did not fire. Layer {_c0 + 1}'s input rebuilt "
            f"with the OLD exponents has row spread {_cspread:.6g}, which the guard "
            f"accepts -- so the guard is not what tells the two scales apart"
        )
    # ---- end of the BLOCK D guards.

    # ---- CONJUNCT 1: THE EMBEDDING IS AN INDEX. Exact equality, not a tolerance:
    # the lookup copies rows and computes nothing, so a difference of any size is a
    # different function.
    first_input = recorded_in[0][1][0]
    if not torch.equal(first_input, fixture["table"][input_ids]):
        raise VacuousControlError(
            "the first layer was handed a tensor that is not "
            "embed_tokens_weight[input_ids], so the embedding is not the index "
            "this forward's docstring declares"
        )

    # ---- CONJUNCT 2: EACH LAYER RECEIVED EXACTLY ITS OWN CARRIER. Identity, not
    # equality: ``**mapping`` hands the callee the very objects the caller put in the
    # mapping, so ``is`` is the sharp test and a carrier built for another layer --
    # another layer's latent cache above all -- fails it.
    for index, carrier in enumerate(carriers):
        got_kwargs = recorded_in[index][2]
        if set(got_kwargs) != set(carrier):
            raise VacuousControlError(
                f"layer {index} was handed the keywords {sorted(got_kwargs)} and "
                f"its carrier declares {sorted(carrier)}"
            )
        wrong = [key for key in carrier if got_kwargs[key] is not carrier[key]]
        if wrong:
            raise VacuousControlError(
                f"layer {index} received {wrong} from some other object than its "
                f"own carrier mapping, so the per-layer state is not bound to the "
                f"layer that owns it"
            )
    print(f"TINYFWD|stack_carriers|layers={len(carriers)}"
          f"|keys={sorted(carriers[0])}|bound_by_identity=True")

    # ---- CONJUNCT 3: THE ATTENTION HALF AND ITS RESIDUAL ADD, per layer. The
    # reference reads the recorded input, so its selection is that layer's own.
    for index, layer in enumerate(layers):
        raw, gains = fixture["attention_operands"][index]
        hidden = recorded_in[index][1][0]
        _normed, topk_indices, attended = _stack_attention_half(
            layer, raw, gains, hidden, cfg, selection
        )
        expected = hidden.float() + attended.float()
        produced = recorded_out[index][1].float()
        print(f"TINYFWD|stack_attention|layer={index}"
              f"|selected={tuple(topk_indices.shape)}"
              f"|sentinels={int((topk_indices < 0).sum())}"
              f"|attention_share="
              f"{float(attended.abs().max() / expected.abs().max()):.6f}"
              f"|max_abs_diff={float((produced - expected).abs().max()):.10g}"
              f"|peak_reference={float(expected.abs().max()):.10g}")
        # ---- THE PEAK-SCALED BAND, and the readings that let it be checked. The
        # frozen pair here was `RTOL`/`ATOL`, and this site is one of the two the
        # lead's option (b) ruling moves; the module pair itself does not move and
        # the other seventeen sites that share it are untouched.
        _rtol, _atol = _stack_peak_band(expected, f"conjunct 3 layer {index}")
        _ae = (produced - expected).abs()
        _rel = _ae / expected.abs().clamp_min(1e-30)
        print(f"TINYFWD|stack_attention_band|layer={index}"
              f"|peak={float(expected.abs().max()):.10g}"
              f"|worst_abs={float(_ae.max()):.10g}"
              f"|worst_rel_against_its_own_cell={float(_rel.max()):.10g}"
              f"|rtol={_rtol:.10g}|atol={_atol:.10g}"
              f"|cells_outside_this_pair="
              f"{int((_ae > (_atol + _rtol * expected.abs())).sum())}"
              f"|cells_outside_the_old_pair="
              f"{int((_ae > (ATOL + RTOL * expected.abs())).sum())}"
              f"|old_pair=rtol {RTOL} atol {ATOL}"
              f"|note=worst_rel divides each cell's error by that cell's own value")
        torch.testing.assert_close(produced, expected, rtol=_rtol, atol=_atol)
        _stack_band_controls(produced, expected, _rtol, _atol,
                             f"conjunct 3 layer {index}")

    # ---- CONJUNCT 4: THE FFN HALF, ITS GAIN, ITS BRANCH AND ITS RESIDUAL ADD. For
    # every layer but the last the next recorded input is the answer; for the last
    # one the stack's final norm is, which is conjunct 5.
    ffn = []
    for index, layer in enumerate(layers):
        hidden = recorded_out[index][1]
        routed = index in fixture["moe_at"]
        half = _stack_ffn_half(
            layer, hidden, cfg, fixture["mlp_operands"][index], routed=routed
        )
        ffn.append(half)
        limit = float(cfg.swiglu_limit)
        print(f"TINYFWD|stack_ffn|layer={index}"
              f"|branch={'routed' if routed else 'dense'}|limit={limit}"
              f"|gate_peak={float(half['gate'].abs().max()):.4f}"
              f"|up_peak={float(half['up'].abs().max()):.4f}"
              f"|clamp_binds={bool(float(half['gate'].abs().max()) > limit or float(half['up'].abs().max()) > limit)}"
              f"|ffn_share={float(half['out'].abs().max() / hidden.abs().max()):.6f}")
        if routed:
            print(f"TINYFWD|stack_conditioning|layer={index}"
                  f"|condition={half['condition']:.4f}"
                  f"|block_share={half['share']:.4f}"
                  f"|bound={STACK_MAX_CONDITION}")
            if half["condition"] > STACK_MAX_CONDITION:
                raise VacuousControlError(
                    f"layer {index}'s bank output has condition "
                    f"{half['condition']:.4f} against a bound of "
                    f"{STACK_MAX_CONDITION}: the output is a small residue of "
                    f"large opposing terms, so every reading here is inflated by "
                    f"a vanishing denominator"
                )
        expected = hidden.float() + half["out"].float()
        if index + 1 < len(layers):
            # THIS COMPARISON NEVER SEES THE ROUTED LAYER. The guard above runs it
            # for every layer except the last, and this fixture's only routed layer IS
            # the last (``STACK_LAYERS = 3``, ``STACK_FIRST_K_DENSE = 2``), so it
            # compares DENSE outputs alone. The eight-expert sum reaches a comparison
            # only through conjunct 5's final norm below.
            #
            # DO NOT SWITCH THIS TO THE MoE BAND. That warning stands and it is about
            # ``MOE_RTOL``/``MOE_ATOL``: the routed LAYER does run, but this comparison
            # never sees it, so a per-branch MoE selector here would widen nothing today
            # and would arm an unruled widening the moment the layer count or the dense
            # split moves.
            #
            # THE TWO SENTENCES THAT USED TO SAY "so it stays on the bf16 pair" AND
            # "conjunct 5's final norm below, which is frozen at ``RTOL``/``ATOL``" ARE
            # GONE, because both moved. This site and conjunct 5's take the peak-scaled
            # recompute band, which is not a per-branch selector at all: it is ONE band
            # at every recompute-against-recorded site in this item, it is TIGHTER than
            # ``RTOL`` on the large cells, and it is refused outright above a declared
            # peak ceiling. :func:`_stack_peak_band` carries the reasoning.
            # ---- THE THIRD SITE THE OPTION (b) RULING MOVES. This comparison
            # has conjunct 3's exact shape: `expected` is a float32 torch recompute
            # (`_dense_output`, plain matmuls on dequantised weights) against a bf16
            # forward, so its error floor is set by the largest intermediate it passes
            # through and not by each cell's own value, which is what `ATOL = 1e-5`
            # cannot express. IT HAS NEVER BEEN REACHED -- layer 0's conjunct 3 raised
            # first on every run so far -- so no world reading exists for it, and both
            # mismatch counts are printed to say what each pair would have decided.
            _nx = recorded_in[index + 1][1][0].float()
            _nrtol, _natol = _stack_peak_band(expected, f"conjunct 4 layer {index}")
            _nae = (_nx - expected).abs()
            print(f"TINYFWD|stack_ffn_band|layer={index}"
                  f"|peak={float(expected.abs().max()):.10g}"
                  f"|worst_abs={float(_nae.max()):.10g}"
                  f"|rtol={_nrtol:.10g}|atol={_natol:.10g}"
                  f"|cells_outside_this_pair="
                  f"{int((_nae > (_natol + _nrtol * expected.abs())).sum())}"
                  f"|cells_outside_the_old_pair="
                  f"{int((_nae > (ATOL + RTOL * expected.abs())).sum())}"
                  f"|old_pair=rtol {RTOL} atol {ATOL}")
            torch.testing.assert_close(_nx, expected, rtol=_nrtol, atol=_natol)
            _stack_band_controls(_nx, expected, _nrtol, _natol,
                                 f"conjunct 4 layer {index}")

    # ---- CONJUNCT 5: THE FINAL NORM CLOSES THE CHAIN, on the last layer's own
    # output rather than on any earlier one. TWO COMPARISONS, EACH AT THE BAND THIS
    # FILE ALREADY REGISTERS FOR ITS PATH (lead ruling, LEAD-LOG section 880):
    #   5a THE LAYER-2 ROUTED HALF, at ``MOE_RTOL``/``MOE_ATOL`` -- the pair item 4
    #      registers for this same expert bank. Layer 2 is the one layer whose FFN
    #      half reached no bank comparison at all before, because conjunct 4 stops
    #      one layer short of the routed one by construction.
    #   5b THE FINAL ADD AND THE FINAL NORM, at the peak-scaled band, over the
    #      PRODUCT's own routed half rather than the fixture's recompute of it. So
    #      each half certifies the arithmetic it is named for and nothing else.
    #
    # WHY THE SPLIT, MEASURED RATHER THAN ARGUED. One comparison over both paths is
    # structurally unpassable, and two counted runs proved it. Round 20 read 179 of
    # 65536 cells outside while the reference summed in float32. Round 21 mirrored
    # the model's cast points -- ``model_fp8.py:6616`` casts the half to the
    # residual's dtype, ``:6714`` adds in that dtype, ``:6532`` casts the norm's
    # output -- and still read 751 cells outside, because the reading added with that
    # fix measured the term left over: ``routed_bank_term`` put the bank's own
    # recompute at 0.95% of its peak, 1.22 bf16 steps, with 31193 of 65536 cells
    # landing on a different bf16 value after the cast. A 1.22-step term cannot fit a
    # band whose whole allowance is one step, and this file already says where that
    # term belongs. So the band does NOT move and never did; the comparison is split
    # along the paths the two bands were registered for.
    # ``increments/investigation-054a-stack-r20.md`` carries the round-20 arithmetic.
    last = recorded_out[-1][1]
    # THE HOOK MUST HAVE FIRED, and on a tensor of the shape the add needs. A gated
    # path never skips silently: an empty recording means the product never called
    # the routed MLP at all, and a shape disagreement means the half it returned
    # cannot be the one the residual add consumes.
    if not recorded_mlp:
        raise VacuousControlError(
            "the last layer's MLP hook recorded nothing, so the product never "
            "called the routed MLP and conjunct 5 has no product half either to "
            "compare or to add; a skip here would hide a stack that ran no experts"
        )
    _half_product = recorded_mlp[-1][3]
    _half_ref = ffn[-1]["out"]
    if tuple(_half_product.shape) != tuple(_half_ref.shape):
        raise ReferenceShapeError(
            f"the routed MLP returned {tuple(_half_product.shape)} and the "
            f"reference computed {tuple(_half_ref.shape)}; conjunct 5 compares them "
            f"cell by cell and adds one of them to the residual, so a shape "
            f"disagreement is a refusal and not a skip"
        )
    final_input = last + _half_product.to(last.dtype)
    expected = _ffn_norm(final_input, fixture["final_gain"], float(cfg.rms_norm_eps))
    if tuple(got.shape) != (STACK_TOKENS, STACK_HIDDEN_SIZE):
        raise ReferenceShapeError(
            f"the forward returned {tuple(got.shape)}, expected "
            f"{(STACK_TOKENS, STACK_HIDDEN_SIZE)}"
        )
    # ---- READINGS ONLY, BLOCK B: what dtype does to the spread, and where this item's worst
    # element actually is. The float32 sum is the one the model does NOT compute, kept here as an
    # explicit term so this reading still says what storage does to the rows: if the float32 spread
    # is nonzero while the bf16 spread is zero then storage is what erased the difference; if both
    # are zero the rows were already equal before any rounding. Nothing gates.
    _sum_f32 = last.float() + ffn[-1]["out"].float()
    _f32 = _row_spread_stats(_sum_f32)
    _bf = _row_spread_stats(_sum_f32.to(torch.bfloat16))
    print(f"TINYFWD|rowspread_dtype|stage=final_input"
          f"|float32_max_spread={_f32[1]:.6g}|bf16_max_spread={_bf[1]:.6g}"
          f"|float32_median_spread={_f32[3]:.6g}|bf16_median_spread={_bf[3]:.6g}"
          f"|note=float32 is the sum the model does not compute; the compared final_input is the bf16 sum")
    # ---- READINGS ONLY, BLOCK B2: the cast points 5b mirrors, and the term 5a owns.
    # `residual_share` below one bf16 step is what reddened round 20, and `routed_bank_term`
    # is the bank's own recompute difference, which round 21 measured at 1.22 bf16 steps and
    # 5a compares at its registered pair. Both rows are unconditional: the refusals above
    # already proved the operands, so a guard here would be dead. Nothing gates.
    _resid_share = float(last.abs().max() / _half_ref.abs().max())
    print(f"TINYFWD|final_add_castpoints|half_reference_dtype={_half_ref.dtype}"
          f"|residual_dtype={last.dtype}|sum_dtype={final_input.dtype}"
          f"|residual_share={_resid_share:.6g}|one_bf16_step={2.0 ** -7:.6g}"
          f"|residual_below_one_step={bool(_resid_share < 2.0 ** -7)}"
          f"|note=the model casts the half at model_fp8.py:6616 and adds in that dtype at :6714")
    _bank_ae = (_half_product.float() - _half_ref.float()).abs()
    _bank_peak = float(_half_ref.abs().max())
    print(f"TINYFWD|routed_bank_term|product_dtype={_half_product.dtype}"
          f"|reference_dtype={_half_ref.dtype}"
          f"|worst_abs={float(_bank_ae.max()):.10g}|peak={_bank_peak:.10g}"
          f"|worst_against_the_peak="
          f"{(float(_bank_ae.max()) / _bank_peak if _bank_peak else float('inf')):.6g}"
          f"|one_bf16_step={2.0 ** -7:.6g}"
          f"|cells_apart_after_the_cast="
          f"{int((_half_product.to(last.dtype) != _half_ref.to(last.dtype)).sum())}"
          f"|cells={int(_half_ref.numel())}"
          f"|note=5a compares this at MOE_RTOL, the pair item 4 registers for this bank")
    _got_f, _exp_f = got.float(), expected.float()
    _abs_err = (_got_f - _exp_f).abs()
    _rel_err = _abs_err / _exp_f.abs().clamp_min(1e-30)
    _cols = int(_abs_err.shape[1])
    _ar, _ac = divmod(int(_abs_err.argmax()), _cols)
    _rr, _rc = divmod(int(_rel_err.argmax()), _cols)
    print(f"TINYFWD|model_forward_worst|kind=worst_abs|row={_ar}|col={_ac}"
          f"|expected={float(_exp_f[_ar, _ac]):.10g}|got={float(_got_f[_ar, _ac]):.10g}"
          f"|abs={float(_abs_err[_ar, _ac]):.6g}|rel={float(_rel_err[_ar, _ac]):.6g}")
    print(f"TINYFWD|model_forward_worst|kind=worst_rel|row={_rr}|col={_rc}"
          f"|expected={float(_exp_f[_rr, _rc]):.10g}|got={float(_got_f[_rr, _rc]):.10g}"
          f"|abs={float(_abs_err[_rr, _rc]):.6g}|rel={float(_rel_err[_rr, _rc]):.6g}"
          f"|expected_abs_max={float(_exp_f.abs().max()):.10g}")
    print(f"TINYFWD|model_forward_worst|kind=grant_117_reported|row=55|col=6"
          f"|expected={float(_exp_f[55, 6]):.10g}|got={float(_got_f[55, 6]):.10g}"
          f"|abs={float(_abs_err[55, 6]):.6g}|rel={float(_rel_err[55, 6]):.6g}"
          f"|one_bf16_ulp_at_1={float(2.0 ** -8):.6g}")
    # ---- end of the BLOCK B readings.
    # ---- 5a THE LAYER-2 ROUTED HALF, at the pair item 4 registers for this bank. This
    # is the only comparison in this file that reaches layer 2's expert bank at all:
    # conjunct 4 stops one layer short of the routed layer, so before this line the
    # eight-expert sum was certified at every layer except the one the stack ends on.
    # Round 21 measured the term at 0.95% of the peak -- inside this pair and outside a
    # one-step band, which is the whole reason the comparison is split here.
    torch.testing.assert_close(_half_product.float(), _half_ref,
                               rtol=MOE_RTOL, atol=MOE_ATOL)
    # ---- 5b THE FINAL ADD AND THE FINAL NORM, at the peak-scaled band the option (b)
    # ruling put here, unchanged. `expected` is now the fixture's own norm over the
    # PRODUCT's own tensors -- its last layer output plus its own routed half, cast and
    # added the way the model does it -- so it is bf16 rather than a float32 recompute
    # of a whole path, and the error this band must hold is the norm's own rounding.
    _rtol, _atol = _stack_peak_band(expected.float(), "conjunct 5 the final norm")
    _ae = (got.float() - expected.float()).abs()
    print(f"TINYFWD|stack_compare|max_abs_diff={float(_ae.max()):.10g}"
          f"|peak_reference={float(expected.abs().max()):.10g}"
          f"|rtol={_rtol:.10g}|atol={_atol:.10g}"
          f"|cells_outside_this_pair="
          f"{int((_ae > (_atol + _rtol * expected.float().abs())).sum())}"
          f"|cells_outside_the_old_pair="
          f"{int((_ae > (ATOL + RTOL * expected.float().abs())).sum())}"
          f"|old_pair=rtol {RTOL} atol {ATOL}")
    torch.testing.assert_close(got.float(), expected.float(),
                               rtol=_rtol, atol=_atol)
    _stack_band_controls(got.float(), expected.float(), _rtol, _atol,
                         "conjunct 5 the final norm")

    # ---- CONTROL A: EACH LAYER READ ITS OWN MLP WEIGHTS. The two dense layers'
    # references are swapped and the answer must leave the band, or two layers
    # holding different weights would be indistinguishable from two sharing one set.
    first_dense, second_dense = fixture["dense_at"]
    _stack_outside_tolerance(
        f"layer {second_dense}'s FFN recomputed with layer {first_dense}'s weights",
        _stack_ffn_half(
            layers[second_dense], recorded_out[second_dense][1], cfg,
            fixture["mlp_operands"][first_dense], routed=False,
        )["out"],
        ffn[second_dense]["out"],
    )

    # ---- CONTROL B: THE FFN NORM USES THE POST-ATTENTION GAIN. Recomputed with the
    # same layer's INPUT gain, which is the neighbouring mapped tensor and the one a
    # transposed read would reach.
    _stack_outside_tolerance(
        f"layer {first_dense}'s FFN normalised with its INPUT gain",
        _stack_ffn_half(
            layers[first_dense], recorded_out[first_dense][1], cfg,
            fixture["mlp_operands"][first_dense], routed=False,
            gain=layers[first_dense].input_layernorm_weight,
        )["out"],
        ffn[first_dense]["out"],
    )

    # ---- CONTROL C: THE ORDER THE PRODUCT HANDED ITS TWO ACTIVATION TENSORS IN,
    # CERTIFIED BY STRUCTURE AND NOT BY A BAND. ``_ffn_half`` passes the PRE-NORM states
    # first, the normalised tensor second and the norm's gain as ``router_gamma``
    # (``model_fp8.py:6595-6598``), and that order is this forward's decision; item 4 owns
    # what the block does with them. THIS FIXTURE CANNOT SEE THE ORDER THROUGH THE BANK:
    # layer 2's routed half is clamp-saturated here, so every expert's SwiGLU term is
    # constant and the half moves with the affinities alone -- the two readings at the end
    # of this control measure exactly that, and both come back at the same number (ruling
    # LEAD-LOG section 889, debt D-054A-L2-BANK-SATURATED; the bank's arithmetic off the
    # clamp is items 2 and 4's). So the control reads the ARGUMENTS the product passed,
    # recorded by the same MLP hook conjunct 5 already installs, and certifies the order
    # itself. The arm below plants the swap in the reading and requires it to fail, which
    # is what stops this certificate from holding no matter what the product did.
    moe_layer = fixture["moe_at"][0]
    moe_input = recorded_out[moe_layer][1]
    _c_gain = layers[-1].post_attention_layernorm_weight
    _c_eps = float(cfg.rms_norm_eps)
    _c_args = recorded_mlp[-1][1]
    _c_kwargs = recorded_mlp[-1][2]
    if len(_c_args) < 2:
        raise ReferenceShapeError(
            f"the routed MLP was called with {len(_c_args)} positional tensors; this "
            f"control reads the two activation tensors the block is handed, so a "
            f"shorter call is a refusal and not a skip"
        )
    _c_normed = _ffn_norm(_c_args[0], _c_gain, _c_eps)
    _c1 = torch.equal(_c_args[1], _c_normed)
    _c1_max = float((_c_args[1].float() - _c_normed.float()).abs().max())
    _c_router_gamma = _c_kwargs.get("router_gamma")
    _c2 = _c_router_gamma is _c_gain or (
        _c_router_gamma is not None and torch.equal(_c_router_gamma, _c_gain)
    )
    print(f"TINYFWD|stack_control|branch=layer {moe_layer}'s block handed its two "
          f"activation tensors in order|c1={_c1}|c2={_c2}|max_abs_c1={_c1_max:.6g}")
    if not (_c1 and _c2):
        raise AssertionError(
            f"layer {moe_layer}'s block was handed its arguments in the wrong order: "
            f"the second positional tensor is the FFN norm of the first at c1={_c1} "
            f"(worst cell {_c1_max:.6g}) and the router's gain is this layer's "
            f"post-attention gain at c2={_c2}; the fused router applies the norm itself, "
            f"so the pre-norm states go first and the normalised tensor second"
        )
    # ---- THE ARM. Reading the same two tensors SWAPPED must not also hold, or the
    # certificate above would pass on any pair of tensors the product handed over.
    _c_arm_failed = not torch.equal(_c_args[0], _ffn_norm(_c_args[1], _c_gain, _c_eps))
    print(f"TINYFWD|stack_control|branch=layer {moe_layer}'s block handed its two "
          f"activation tensors in order|arm=swapped|must_fail=True"
          f"|failed={_c_arm_failed}")
    if not _c_arm_failed:
        _diag_or_raise(
            f"item 6 control C's arm: the swapped reading held too",
            f"layer {moe_layer}'s order certificate cannot tell the two apart: reading "
            f"the tensors swapped ALSO holds, so it certifies nothing",
        )
    # ---- READINGS ONLY, GATED ON NOTHING: the two plants that used to BE control C.
    # Their numbers are worth keeping -- they are the measurement that says the bank
    # cannot see its own expert input at this layer -- and a plant that cannot
    # discriminate is a reading, never a gate.
    _moe_normed = _ffn_norm(
        moe_input,
        layers[moe_layer].post_attention_layernorm_weight,
        _c_eps,
    )
    _ro_base = ffn[moe_layer]["out"]
    for _r_label, _r_kwargs, _r_note in (
        ("block handed its two activation tensors in the swapped order",
         {"router_input": _moe_normed, "expert_input": moe_input},
         "the bank is clamp-saturated at this layer, so its output cannot see this"),
        ("router alone fed the FFN-normalised tensor",
         {"router_input": _moe_normed},
         "this plant reaches the affinities only"),
    ):
        _r_moved = _stack_ffn_half(
            layers[moe_layer], moe_input, cfg,
            fixture["mlp_operands"][moe_layer], routed=True, **_r_kwargs,
        )["out"]
        _r_outside = not torch.allclose(_r_moved.float(), _ro_base.float(),
                                        rtol=RTOL, atol=ATOL)
        _r_gap = float((_r_moved.float() - _ro_base.float()).abs().max()
                       / _ro_base.abs().max())
        print(f"TINYFWD|stack_control_reading|branch=layer {moe_layer}'s {_r_label}"
              f"|outside_tolerance={_r_outside}|gap={_r_gap:.6f}|note={_r_note}")
    # ---- READING: HOW FAR APART THE TWO TENSORS ACTUALLY ARE. If this were zero the
    # pre-norm states and the normalised tensor would be the same input and every plant
    # on the order would be a no-op by arithmetic. It is not zero -- the router-only
    # reading above already moves the answer -- and this row says so directly.
    _pn_peak = float(moe_input.float().abs().max())
    _pn_rel = float((moe_input.float() - _moe_normed.float()).abs().max()
                    / max(_pn_peak, 1e-30))
    print(f"TINYFWD|stack_control_reading|branch=layer {moe_layer} pre-norm vs normalised"
          f"|rel_max_abs={_pn_rel:.6f}|pre_norm_peak={_pn_peak:.10g}"
          f"|note=zero here would mean the two tensors are one input")
    # ---- READING: THE ROUTED BANK'S OWN CLAMP CENSUS, in the same form the dense guard
    # rows print, over the cells this bank actually clamps -- the gate and up activations
    # of the experts the router SELECTED for each token. Saturation is why nothing handed
    # to the experts can be seen through this half (ruling LEAD-LOG section 889).
    _bank_gate = ffn[moe_layer]["gate"].float()
    _bank_up = ffn[moe_layer]["up"].float()
    _bank_limit = float(cfg.swiglu_limit)
    _bank_logits, _bank_index, _bank_aff = layers[moe_layer].mlp.experts.route_tokens(
        moe_input.unsqueeze(0),
        layers[moe_layer].post_attention_layernorm_weight,
        cfg,
    )
    _sel = (_bank_aff.t() != 0).unsqueeze(-1).expand_as(_bank_gate)
    _sel_any = bool(_sel.any())
    _sel_cells = int(_sel.sum())
    _bg_peak = float(_bank_gate[_sel].abs().max()) if _sel_any else 0.0
    _bu_peak = float(_bank_up[_sel].abs().max()) if _sel_any else 0.0
    _bg_frac = float((_bank_gate >= _bank_limit)[_sel].float().mean()) if _sel_any else 0.0
    _bu_frac = float((_bank_up.abs() >= _bank_limit)[_sel].float().mean()) if _sel_any else 0.0
    print(f"TINYFWD|stack_scale_guard|layer={moe_layer}|scale=new|bank=routed"
          f"|gate_peak={_bg_peak:.10g}|up_peak={_bu_peak:.10g}"
          f"|limit={_bank_limit:.10g}"
          f"|gate_at_limit_frac={_bg_frac:.6f}|up_at_limit_frac={_bu_frac:.6f}"
          f"|selected_cells={_sel_cells}|experts_selected_per_token="
          f"{int(( _bank_aff != 0).sum(dim=1).max())}"
          f"|note=a reading and not a gate: the fractions cover the selected experts only")

    # ---- CONTROL D: A CARRIER LIST THAT DOES NOT MATCH THE STACK REFUSES BY NAME,
    # and it refuses before any layer runs. A short sequence would otherwise run a
    # prefix of the stack and return a plausible tensor.
    with pytest.raises(ValueError, match="one mapping per layer"):
        model.forward(
            input_ids,
            layer_carriers=carriers[:-1],
            quant_config=quant_config,
        )
    print(f"TINYFWD|stack_control|branch={len(carriers) - 1} carriers for "
          f"{len(layers)} layers|refused=True")

    # ---- CONTROL E: A MAPPED TENSOR THAT WAS NEVER LOADED REFUSES BY NAME AND BY
    # MAP LINE, and no seam moves -- which is what tells a named refusal from a
    # forward that ran and produced zeros.
    model.embed_tokens_weight = None
    _reset_seam_counters()
    unloaded_before = _read_seam_counters()
    with pytest.raises(ValueError, match="weight_loaders_fp8.py:378"):
        model.forward(
            input_ids,
            layer_carriers=carriers,
            quant_config=quant_config,
        )
    unloaded_after = _read_seam_counters()
    moved = {
        seam: unloaded_after[seam][0] - unloaded_before[seam][0]
        for seam in _SEAMS
        if unloaded_after[seam][0] != unloaded_before[seam][0]
    }
    print(f"TINYFWD|stack_control|branch=forward with no embedding table"
          f"|refused=True|seams_that_moved={sorted(moved.items())}")
    if moved:
        _diag_or_raise(
            "item 6 control E: seams dispatched before the named refusal",
            f"the refusal ran after {sorted(moved.items())} dispatched, so it is "
            f"not reached before the stack starts",
        )


# --------------------------------------------------------------------------- #
# Item 7's own fixture geometry. The root adds exactly two things to item 6's   #
# stack -- a head tensor and a row selection -- so everything else below is     #
# item 6's, reused rather than redrawn.                                         #
# --------------------------------------------------------------------------- #
#: The head weight's power-of-two scale. Chosen so the logits land near 1.0:
#: each logit sums ``STACK_HIDDEN_SIZE`` = 512 products of a post-norm activation
#: (order 1) with a head value on the unsigned fp8 grid (mean 1/2), so the sum is
#: about ``512 * 0.5 * 2**e``, which is about 1 at ``e = -8``. A number near 1 is
#: what keeps ``ATOL`` and ``RTOL`` both meaningful in the same comparison.
ROOT_HEAD_SCALE_EXPONENT = -8

SEED_ROOT_HEAD = 5481

#: The rows this item asks for, and every property of this tuple is load-bearing.
#: It is OUT OF ORDER (the last token first), so a slice or a sort cannot produce
#: it; it REPEATS one row, so a de-duplicating implementation returns the wrong
#: shape; and it is SHORTER than the token count, so a forward that projected
#: every row would fail on shape. The repeat is not contrived: the runner pads
#: its own ``logits_indices`` by repeating the last real index
#: (``neuron_model_runner.py:3896-3900``), and builds them ``dtype=torch.long``
#: (``:2941``), which is the dtype this item passes.
ROOT_SAMPLING_POSITIONS = (STACK_TOKENS - 1, 0, 7, 7)

#: The same conditioning bound item 6 uses on the bank, applied to the head's
#: dot products. All-positive terms give exactly 1.0; a cancelling sum inflates
#: every reading taken from it.
ROOT_MAX_CONDITION = STACK_MAX_CONDITION

#: HOW MANY OF ITS OWN ALLOWANCES control E plants on the hidden state (R3, LEAD-LOG
#: 850). Four, so the planted move clears the widest allowance any cell of that slot
#: has by a factor of four and no row spread, homogenised or not, can absorb it. It is
#: a control's plant size and not a tolerance: it widens nothing this item compares.
ROOT_CONTROL_E_MULTIPLE = 4.0


def _root_config(**overrides):
    """The root's ``Glm5NextConfig``: the tiny stack plus the PINNED quantisation fields.

    THE QUANTISATION FIELDS ARE THE CHECKPOINT'S, not this dataclass's defaults, and
    that matters because the root resolves the policy ITSELF from this object -- there
    is no ``quant_config`` argument to hand it. ``Glm5NextConfig``'s defaults happen to
    agree with the pinned checkpoint on ``quant_method``, ``activation_scheme`` and
    ``weight_block_size`` (``config.py:437-441`` against the fixture's
    ``quantization_config``), so an item that took the defaults would pass today and
    stop measuring the campaign's registered policy the moment either side moved.
    Lifting them from the digest-verified fixture is what ties this item to it.

    ONE FIELD IS ABSENT FROM THE PINNED CHECKPOINT, and it is read the way the shipped
    loader reads it. The fixture's ``quantization_config`` holds exactly four keys --
    ``activation_scheme``, ``fmt``, ``quant_method``, ``weight_block_size`` -- and no
    ``modules_to_not_convert``. ``config.py`` declares that field optional
    (``:462``, ``list[str] | None = None``) and lifts it with ``.get`` (``:524``), so a
    bracket read here demanded a key the checkpoint does not carry and the item died
    before it reached the model
    (``increments/fetch-054a-r6-step1-host-20260908T205349Z.out``). The absence is now
    REPORTED rather than defaulted silently: a checkpoint that starts carrying the field
    changes this reading, and the reading is what the item ties itself to.
    """
    from vllm_neuron.model.glm5_next.config import Glm5NextConfig

    raw = _pinned_raw_config()["quantization_config"]
    text = _stack_text_config(**overrides)
    exempt = raw.get("modules_to_not_convert")
    print(
        f"TINYFWD|root_quant_policy|keys={sorted(raw)}"
        f"|modules_to_not_convert={exempt}"
    )
    return Glm5NextConfig(
        text_config=text,
        tie_word_embeddings=bool(text.tie_word_embeddings),
        quant_method=raw["quant_method"],
        activation_scheme=raw["activation_scheme"],
        weight_block_size=list(raw["weight_block_size"]),
        modules_to_not_convert=None if exempt is None else list(exempt),
        fmt=raw["fmt"],
    )


def _root_head_weight() -> torch.Tensor:
    """``[vocab, hidden]`` head weight, on the unsigned fp8 grid and exact in bf16.

    UNSIGNED for the reason this module's docstring gives and this item measures: a
    logit is a 512-term dot product, and a signed draw against a positive residual
    stream would make each one a small residue of large opposing terms, so the
    conditioning check below would report a large number and every tolerance reading
    would be inflated by a vanishing denominator.
    """
    return (
        _fp8_grid_values(SEED_ROOT_HEAD, STACK_VOCAB_SIZE, STACK_HIDDEN_SIZE)
        * float(2.0**ROOT_HEAD_SCALE_EXPONENT)
    ).to(torch.bfloat16)


def _root_fixture(**overrides) -> dict:
    """The root module with item 6's stack bound inside it, plus its head tensor.

    The root builds its OWN ``Glm5NextModel`` in ``__init__``, so the weights are bound
    onto that stack rather than onto one this file built -- :func:`_stack_fixture` takes
    the model for exactly this reason. Everything item 6's fixture returns is returned
    here too, so the references below are item 6's references.

    ``world_size`` IS ASSERTED, not assumed. The root resolves it from the process
    group (``model_fp8.py:179-182``) and every per-rank width in this fixture is
    written for one rank, so a distributed session would shard the tree while this
    file's references stayed whole.
    """
    model_fp8 = _impl()
    config = _root_config(**overrides)
    root = model_fp8.Glm5NextForConditionalGeneration(config)
    if int(root.world_size) != 1:
        raise VacuousControlError(
            f"the root resolved world_size={root.world_size}; every width in this "
            f"fixture is written for a single rank"
        )
    fixture = _stack_fixture(model=root.model)
    cfg = fixture["cfg"]
    mismatched = {
        name: (getattr(root.text_config, name), getattr(cfg, name))
        for name in ("hidden_size", "num_hidden_layers", "vocab_size",
                     "first_k_dense_replace")
        if getattr(root.text_config, name) != getattr(cfg, name)
    }
    if mismatched:
        raise VacuousControlError(
            f"the root's text config and this fixture's disagree on "
            f"{sorted(mismatched.items())}, so the references are written for a "
            f"different tree than the one that ran"
        )

    tied = bool(root.text_config.tie_word_embeddings)
    declared = root.declared_parameter_names()
    if tied:
        head = fixture["table"]
    else:
        if "lm_head_weight" not in declared:
            raise VacuousControlError(
                "the untied root declares no lm_head_weight, so this item has no "
                "head tensor to bind and the arm it means to measure is absent"
            )
        head = _root_head_weight()
        root.lm_head_weight = torch.nn.Parameter(head, requires_grad=False)

    parameters = sum(int(p.numel()) for p in root.parameters() if p is not None)
    print(f"TINYFWD|root_fixture|tied={tied}|head={tuple(head.shape)}"
          f"|head_peak={float(head.abs().max()):.6f}"
          f"|parameters={parameters}|bound={MAX_PARAMETERS}"
          f"|declares_lm_head={'lm_head_weight' in declared}"
          f"|world_size={int(root.world_size)}")
    if parameters >= MAX_PARAMETERS:
        raise VacuousControlError(
            f"the root holds {parameters} parameters with its head bound, at or "
            f"past the adopted bound of {MAX_PARAMETERS}"
        )
    fixture.update(root=root, config=config, head=head, tied=tied)
    return fixture


def _root_reference(hidden: torch.Tensor, head: torch.Tensor,
                    positions) -> torch.Tensor:
    """``[rows, vocab]`` logits in fp32: gather the named rows, then project.

    THE GATHER IS A PYTHON LOOP OVER THE DECLARED POSITIONS, deliberately not
    ``torch.index_select``. ``index_select`` is the operation under test, and a
    reference built from it would agree with a forward that selected the wrong rows in
    the same wrong way.
    """
    rows = torch.stack([hidden[int(index)].float() for index in positions])
    return rows @ head.float().t()


# --------------------------------------------------------------------------- #
# ITEM 7 of 7 -- ``Glm5NextForConditionalGeneration.forward``.                  #
# Certifying component: ``model_fp8.Glm5NextForConditionalGeneration.forward``   #
# together with its one private helper, ``_head_weight``.                       #
#                                                                              #
# WHAT IT CERTIFIES, and nothing a callee already owns. The root resolves the    #
# quantisation policy from its OWN config and threads it down; it runs the stack #
# exactly once on the arguments it was given; it SELECTS the caller's rows out   #
# of the stack's output BEFORE projecting them; it projects them through the     #
# head tensor its tied/untied arm chooses; and it refuses an unloaded head       #
# BEFORE spending the stack. The stack's numerics are item 6's, the attention's  #
# item 5's, the experts' items 2 and 4's, the dense MLP's item 1's.              #
#                                                                              #
# THE COMPARISON IS MADE AGAINST THE HIDDEN STATES THE ROOT'S OWN STACK          #
# RETURNED, captured with one forward hook on ``root.model``, for item 6's       #
# reason: a reference that re-ran the stack would be hostage to a pool selection #
# flipping between two chains that differ by kernel noise, and would measure     #
# tie-breaking rather than the projection. Item 6 certifies that those hidden    #
# states are the stack's correct output; this item certifies what the root does  #
# with them.                                                                    #
#                                                                              #
# THE HEAD IS A ``torch`` PROJECTION AND THAT IS THE CHECKPOINT'S OWN            #
# DECLARATION, NOT A FALLBACK (P13). ``lm_head`` is one of the nine bare entries #
# in this checkpoint's 1,509-entry ``modules_to_not_convert`` list, so the head  #
# ships BF16 and no block-FP8 kernel applies to it -- read off the same          #
# digest-verified fixture this item's quantisation fields come from, not         #
# recalled. So this item's dispatch figures are item 6's, UNCHANGED: the root    #
# adds no seam. The predicate's fallback aggregate is a statement about seams    #
# and the head is not one, which is why the checkpoint's skip list, and not a    #
# counter, is what makes the projection legitimate.                              #
#                                                                              #
# THE TIED ARM IS CERTIFIED AT ``_head_weight``, NOT WITH A SECOND FORWARD. The  #
# two arms differ in exactly one thing -- which tensor ``_head_weight`` returns  #
# -- and everything after it is shared code that the untied arm already runs. So #
# the tied arm is measured by object IDENTITY against the embedding table, which #
# is the sharp test for "reads the table itself rather than a copy", and a second #
# 3-layer forward is not spent to re-measure shared code.                        #
#                                                                              #
# THE ONE-STREAM EXCLUSION THAT STOOD HERE IS RETIRED, AND THIS ITEM IS BEING    #
# RE-POINTED, on the same terms as the model-forward item above. It used to say   #
# the logits came from a one-stream carrier and that the four-stream carrier was  #
# ``inc-glm53f-030b``'s; ``inc-glm53f-030d`` part (a) built that carrier in       #
# ``Glm5NextModel.forward``, which now collapses the streams with an unweighted   #
# mean before the final norm (``modeling_glm5_next.py:1493``, ``:302``). The      #
# ``[T, H]`` the root projects is therefore the reference's own post-collapse     #
# tensor and the exclusion has nothing left to exclude.                           #
#                                                                              #
# THIS ITEM STILL CANNOT RUN AGAINST THAT CARRIER until its fixture loads the six #
# mHC leaves per layer and binds the sites, because the stack passes streams      #
# unconditionally and ``_mhc_site`` refuses a streams call on a layer that        #
# carries none of them (``model_fp8.py:7578-7582``). That re-point is the NEXT    #
# commit's (ruled §943 Q5, never a deletion); this comment states the true        #
# state of the file rather than a change that has not happened yet.               #
#                                                                              #
# WHAT IT DOES NOT TOUCH. On-device sampling: ``sampling_params``,               #
# ``logit_mask`` and ``spec_decode_metadata`` are runner keys this tree          #
# implements nowhere, and control D measures that the forward REFUSES them by    #
# name rather than swallowing them. Threading the runner's own dicts into this    #
# signature is ``inc-glm53f-054b``'s work.                                       #
#                                                                              #
# NOTHING BELOW HAS BEEN RUN. Every expected count and every tolerance claim in  #
# this section is a prediction the first counted run adjudicates.                #
# --------------------------------------------------------------------------- #
def test_tiny_root_forward_matches_the_reference() -> None:
    """The root equals the head projection of the rows it was asked to sample.

    ``inc-glm53f-054a`` item 7 of 7. D1.4 certifying component:
    ``Glm5NextForConditionalGeneration.forward`` with ``_head_weight`` -- the policy
    resolution, the single stack call, the row selection before the projection, and
    the tied/untied head arm.
    """
    fixture = _root_fixture()
    root, cfg, layers = fixture["root"], fixture["cfg"], fixture["layers"]
    head = fixture["head"]
    selection = _mla_selection_operands(tokens=STACK_TOKENS, pages=STACK_PAGES)
    carriers = _stack_carriers(layers, selection)
    positions = torch.tensor(ROOT_SAMPLING_POSITIONS, dtype=torch.long)

    input_ids = torch.randint(
        0, STACK_VOCAB_SIZE, (STACK_TOKENS,),
        generator=torch.Generator().manual_seed(SEED_STACK_IDS),
        dtype=torch.int64,
    )

    # ---- ONE HOOK ON THE STACK. It records what the root handed the stack and what
    # the stack returned; every conjunct below reads it rather than re-running.
    recorded: list = []

    def _record(module, args, kwargs, output):
        recorded.append((args, kwargs, output))

    handle = root.model.register_forward_hook(_record, with_kwargs=True)
    try:
        _reset_seam_counters()
        before = _read_seam_counters()
        got = root.forward(
            input_ids,
            layer_carriers=carriers,
            sampling_positions=positions,
        )
        after = _read_seam_counters()
    finally:
        handle.remove()

    # ---- THE REGISTERED ROUTE PREDICATE. Item 6's figures, declared again here
    # rather than shared, so a root that smuggled in one extra dispatch fails this
    # item on its own declaration.
    route_expected = {
        "mla_projection": 9 * STACK_LAYERS,
        "mla_absorb": 2 * STACK_LAYERS,
        "mla_sparse": 1 * STACK_LAYERS,
        "dsa_kpool_hadamard": 2 * STACK_LAYERS,
        "dsa_paged_gather": 1 * STACK_LAYERS,
        "dsa_score_gemm": 1 * STACK_LAYERS,
        "dsa_topk_select": 1 * STACK_LAYERS,
        "dsa_index_expand": 1 * STACK_LAYERS,
        "blockwise_fp8_mm": 3 * STACK_DENSE_LAYERS,
        "blockwise_fp8_moe": 1 * STACK_MOE_LAYERS,
        "noaux_tc_router": 1 * STACK_MOE_LAYERS,
    }
    _declare_bound_and_sentinel(route_expected)
    _assert_route_predicate("7 the root", route_expected, before, after)

    # ---- CONJUNCT 1: THE STACK RAN ONCE, ON THE ROOT'S OWN ARGUMENTS. Read before
    # anything is taken out of the recording, and by identity where identity is the
    # claim: the ids and the carrier sequence are forwarded, not rebuilt.
    print(f"TINYFWD|root_stack_calls|calls={len(recorded)}")
    if len(recorded) != 1:
        raise VacuousControlError(
            f"the root called its stack {len(recorded)} times; the forward it "
            f"certifies calls it exactly once"
        )
    args, kwargs, hidden = recorded[0]
    expected_keys = {
        "layer_carriers", "quant_config", "block_size", "moe_group", "tp_degree",
        "expert_parallel_rank",
    }
    print(f"TINYFWD|root_stack_call|positional={len(args)}"
          f"|keywords={sorted(kwargs)}"
          f"|input_ids_forwarded={args[0] is input_ids if args else False}"
          f"|carriers_forwarded={kwargs.get('layer_carriers') is carriers}")
    if len(args) != 1 or args[0] is not input_ids:
        raise VacuousControlError(
            f"the stack was handed {len(args)} positional arguments and the first "
            f"is not the caller's input_ids, so the ids were rebuilt on the way "
            f"down"
        )
    if set(kwargs) != expected_keys:
        raise VacuousControlError(
            f"the stack was handed the keywords {sorted(kwargs)} and this item "
            f"declares {sorted(expected_keys)}"
        )
    if kwargs["layer_carriers"] is not carriers:
        raise VacuousControlError(
            "the stack received some other object than the caller's carrier "
            "sequence, so the per-layer state is not the caller's"
        )
    defaults = {"block_size": None, "moe_group": None, "tp_degree": 1,
                "expert_parallel_rank": 0}
    wrong = {
        name: kwargs[name] for name, value in defaults.items()
        if kwargs[name] != value
    }
    if wrong:
        raise VacuousControlError(
            f"the root forwarded {sorted(wrong.items())} where its own declared "
            f"defaults are {sorted(defaults.items())}"
        )

    # ---- CONJUNCT 2: THE POLICY IS THE ROOT'S OWN RESOLUTION, and it is the pinned
    # checkpoint's. The root takes no quant_config argument, so this is the only
    # place the resolution can be read; it is compared against the same fixture's
    # policy on the two fields every call site in this file reads.
    resolved = kwargs["quant_config"]
    pinned = _quant_config()
    print(f"TINYFWD|root_policy|type={type(resolved).__name__}"
          f"|is_block_quantized={resolved.is_block_quantized}"
          f"|block_shape={resolved.block_shape}"
          f"|pinned_block_shape={pinned.block_shape}"
          f"|method={type(resolved.method).__name__}")
    if not isinstance(resolved, _impl().Glm5NextQuantConfig):
        raise VacuousControlError(
            f"the root threaded down a {type(resolved).__name__}, not a "
            f"Glm5NextQuantConfig, so the MLPs' route selector is not the "
            f"resolved policy"
        )
    if not resolved.is_block_quantized or resolved.block_shape != pinned.block_shape:
        raise VacuousControlError(
            f"the root resolved is_block_quantized={resolved.is_block_quantized}, "
            f"block_shape={resolved.block_shape}; the pinned checkpoint's policy "
            f"is block-quantised at {pinned.block_shape}"
        )

    # ---- CONJUNCT 3: THE HEAD ARM. The untied root projects with its own mapped
    # tensor, by identity.
    if fixture["tied"] or root._head_weight() is not root.lm_head_weight:
        raise VacuousControlError(
            "the untied root's head tensor is not lm_head_weight, so this item is "
            "not measuring the arm it declares"
        )

    # ---- CONJUNCT 4: THE LOGITS ARE THE PROJECTION OF THE SELECTED ROWS. Shape
    # first -- a forward that projected every token, or de-duplicated the repeated
    # row, is a different shape and not a small numeric difference.
    rows = len(ROOT_SAMPLING_POSITIONS)
    if tuple(got.shape) != (rows, STACK_VOCAB_SIZE):
        raise ReferenceShapeError(
            f"the root returned {tuple(got.shape)}, expected "
            f"{(rows, STACK_VOCAB_SIZE)}: {rows} requested rows by the "
            f"{STACK_VOCAB_SIZE}-wide vocabulary"
        )
    expected = _root_reference(hidden, head, ROOT_SAMPLING_POSITIONS)

    # ---- THE READING CLASS C TURNS ON, PRINTED AND GATED BY NOTHING. Control E below
    # no longer depends on any of it -- R3 replaced the rolled-positions recompute with
    # a perturbation planted on the hidden state, which is sized from the band and
    # cannot be absorbed by row spread -- and these rows STAY, because they are what
    # the round-3 review asked for and what proved the old control could not work: the
    # bar on the relative row spread is about 0.13, a max over the 1024 compared
    # elements rather than the one-sigma size an earlier proposal of mine used, and the
    # rows arrive ALREADY HOMOGENISED by the stack, whose unsigned-weight MLPs make
    # each output row nearly a function of its input row's mean. The rows are also
    # post-RMS-norm, so no magnitude difference between positions survives to the head
    # and only shape differences reach it. How far apart the 128 positions really sit
    # is therefore a measurement, and this is that measurement.
    #
    # SPREAD IS ``||x_a - x_b||_2 / mean(||x_a||_2, ||x_b||_2)``, so it is symmetric and
    # dimensionless. NOTHING BELOW RAISES, nothing is compared against a bound, and no
    # constant, comparator, position or fixture value is touched: these are readings.
    if hidden.dim() != 2:
        print(f"TINYFWD|root_row_spread|unavailable|hidden_dim={hidden.dim()}")
    else:
        _rows = hidden.float()
        _norms = _rows.norm(dim=1)
        # The difference is broadcast explicitly rather than taken from ``torch.cdist``,
        # whose default compute mode switches to a matrix-multiply formula above 25 rows.
        # At 128 x 512 the explicit form costs about 33 MB and is exact.
        _dist = (_rows.unsqueeze(1) - _rows.unsqueeze(0)).norm(dim=2)
        _norm_mean = (_norms.unsqueeze(1) + _norms.unsqueeze(0)) / 2.0
        _spread = _dist / _norm_mean.clamp_min(1e-12)
        # The three pairs THIS ITEM SAMPLES, derived from the frozen tuple rather than
        # written out, so they follow the tuple if it ever moves under a ruling.
        for _a, _b in (
            (ROOT_SAMPLING_POSITIONS[1], ROOT_SAMPLING_POSITIONS[0]),
            (ROOT_SAMPLING_POSITIONS[1], ROOT_SAMPLING_POSITIONS[2]),
            (ROOT_SAMPLING_POSITIONS[2], ROOT_SAMPLING_POSITIONS[0]),
        ):
            print(f"TINYFWD|root_row_spread|pair=({_a}, {_b})"
                  f"|l2_diff={float(_dist[_a, _b]):.6g}"
                  f"|l2_row_mean={float(_norm_mean[_a, _b]):.6g}"
                  f"|spread={float(_spread[_a, _b]):.6f}")
        # And the whole matrix, so a ruling does not need another run to learn whether a
        # better pair exists. Unordered pairs only: the statistic is symmetric. The argmax
        # is divided back into a row and a column, which cannot mislabel the pair.
        _upper = torch.triu(torch.ones_like(_spread), diagonal=1) > 0
        _flat = _spread[_upper]
        _masked = _spread.masked_fill(~_upper, -1.0)
        _best = int(_masked.argmax())
        _ba, _bb = divmod(_best, _spread.shape[1])
        print(f"TINYFWD|root_row_spread_max|rows={_spread.shape[0]}"
              f"|unordered_pairs={int(_flat.numel())}"
              f"|argmax=({_ba}, {_bb})"
              f"|spread={float(_spread[_ba, _bb]):.6f}"
              f"|min_spread={float(_flat.min()):.6f}"
              f"|median_spread={float(_flat.median()):.6f}")
        print("TINYFWD|root_row_spread_population"
              + "".join(f"|above_{_t:g}={int((_flat > _t).sum())}"
                        for _t in (0.05, 0.10, 0.13, 0.20))
              + "|note=0.13 is the round-3 review's derived bar for this control,"
              + " not a fixture constant and not gated on here")

    terms = torch.stack(
        [hidden[int(index)].abs().float() for index in ROOT_SAMPLING_POSITIONS]
    ) @ head.abs().float().t()
    condition = float((terms / expected.abs().clamp_min(1e-12)).max())
    print(f"TINYFWD|root_logits|dtype={got.dtype}|rows={rows}"
          f"|positions={list(ROOT_SAMPLING_POSITIONS)}"
          f"|hidden={tuple(hidden.shape)}"
          f"|max_abs_diff={float((got.float() - expected).abs().max()):.10g}"
          f"|peak_reference={float(expected.abs().max()):.10g}"
          f"|condition={condition:.4f}|bound={ROOT_MAX_CONDITION}")
    if condition > ROOT_MAX_CONDITION:
        raise VacuousControlError(
            f"the logits' dot products have condition {condition:.4f} against a "
            f"bound of {ROOT_MAX_CONDITION}: each logit is a small residue of "
            f"large opposing terms, so every reading here is inflated by a "
            f"vanishing denominator"
        )
    torch.testing.assert_close(got.float(), expected, rtol=RTOL, atol=ATOL)

    # ---- CONJUNCT 5: THE REPEATED ROW IS PRESERVED EXACTLY. Two requests for the
    # same row must return the same bytes, which no tolerance is needed to state and
    # which a de-duplicate-then-scatter implementation cannot fake.
    first, second = (
        index for index, value in enumerate(ROOT_SAMPLING_POSITIONS)
        if value == ROOT_SAMPLING_POSITIONS[-1]
    )
    print(f"TINYFWD|root_repeat|rows=({first}, {second})"
          f"|identical={bool(torch.equal(got[first], got[second]))}")
    if not torch.equal(got[first], got[second]):
        raise VacuousControlError(
            f"rows {first} and {second} ask for the same position and came back "
            f"different, so the selection is not the index the caller gave"
        )

    # ---- CONTROL A: THE TIED ARM READS THE EMBEDDING TABLE ITSELF. A separate tiny
    # root, built with the flag turned on: it must declare NO head parameter and its
    # head tensor must BE the table object, not a tensor equal to it.
    tied_root = _impl().Glm5NextForConditionalGeneration(
        _root_config(tie_word_embeddings=True)
    )
    table = _fp8_grid_values(
        SEED_STACK_EMBED, STACK_VOCAB_SIZE, STACK_HIDDEN_SIZE
    ).to(torch.bfloat16)
    tied_root.model.embed_tokens_weight = torch.nn.Parameter(
        table, requires_grad=False
    )
    tied_declared = tied_root.declared_parameter_names()
    tied_head = tied_root._head_weight()
    print(f"TINYFWD|root_control|branch=tied head"
          f"|declares_lm_head={'lm_head_weight' in tied_declared}"
          f"|is_embedding_table="
          f"{tied_head is tied_root.model.embed_tokens_weight}")
    if "lm_head_weight" in tied_declared:
        raise VacuousControlError(
            "the tied root declares lm_head_weight; the weight map adds no "
            "lm_head.weight entry in that case (weight_loaders_fp8.py:382-383), "
            "so there is no checkpoint tensor to fill it"
        )
    if tied_head is not tied_root.model.embed_tokens_weight:
        raise VacuousControlError(
            "the tied root's head tensor is not the embedding table object, so "
            "the two can drift apart"
        )

    # ---- CONTROL B: AN UNLOADED HEAD REFUSES BY NAME AND BY MAP LINE, BEFORE THE
    # STACK RUNS. No seam may move -- that is what tells a named refusal from a
    # forward that ran a whole stack and then discovered it had no head.
    #
    # THE HEAD IS BORROWED, NOT SPENT, and putting it back is this control's own
    # business. Unsetting the parameter is how this control makes the product refuse;
    # leaving it unset hands every later line of this item a root the product will not
    # run. ``Glm5NextForConditionalGeneration.forward`` resolves the head in its FIRST
    # statement (``head = self._head_weight()``, ``model_fp8.py:7907``) and the untied
    # arm raises there when it is None (``model_fp8.py:7795``), so control E's product
    # call raised THIS control's ValueError, outside any ``pytest.raises``, instead of
    # measuring which rows the root selected: item 7 failed on correct code (round 4,
    # finding F1). The restore runs in a ``finally``, so a control that raises still
    # leaves the root usable for the controls after it.
    _saved_head = root.lm_head_weight
    try:
        root.lm_head_weight = None
        _reset_seam_counters()
        unloaded_before = _read_seam_counters()
        with pytest.raises(ValueError, match="weight_loaders_fp8.py:383"):
            root.forward(
                input_ids,
                layer_carriers=carriers,
                sampling_positions=positions,
            )
        unloaded_after = _read_seam_counters()
        moved = {
            seam: unloaded_after[seam][0] - unloaded_before[seam][0]
            for seam in _SEAMS
            if unloaded_after[seam][0] != unloaded_before[seam][0]
        }
        print(f"TINYFWD|root_control|branch=forward with no head tensor"
              f"|refused=True|seams_that_moved={sorted(moved.items())}")
        if moved:
            raise VacuousControlError(
                f"the refusal ran after {sorted(moved.items())} dispatched, so the "
                f"head is resolved after the stack instead of before it"
            )
    finally:
        root.lm_head_weight = _saved_head
    # AND THE PUT-BACK IS GATED, not assumed: the parameter is the object this control
    # borrowed, and the product's own resolver hands that same object back.
    if root.lm_head_weight is not _saved_head:
        raise VacuousControlError(
            "control B did not put root.lm_head_weight back, so every control after "
            "it runs against a root whose head the product refuses by name "
            "(model_fp8.py:7795)"
        )
    if root._head_weight() is not root.lm_head_weight:
        raise VacuousControlError(
            "root._head_weight() no longer returns root.lm_head_weight after control "
            "B, so the head the product projects with is not the tensor this item "
            "loaded"
        )
    print(f"TINYFWD|root_control|branch=head put back after the unloaded arm"
          f"|is_the_borrowed_parameter={root.lm_head_weight is _saved_head}"
          f"|product_resolver_agrees="
          f"{root._head_weight() is root.lm_head_weight}")

    # ---- CONTROL C: THE ROW SELECTION IS REQUIRED. No default, so a caller that
    # forgets it gets a TypeError at the call rather than a whole-prefill projection.
    with pytest.raises(TypeError, match="sampling_positions"):
        root.forward(input_ids, layer_carriers=carriers)

    # ---- CONTROL D: THERE IS NO ``**kwargs`` SINK. ``sampling_params`` is a real
    # runner key (``neuron_model_runner.py:7036``) that this tree implements nowhere;
    # it must be refused at the call, not accepted and dropped.
    with pytest.raises(TypeError, match="sampling_params"):
        root.forward(
            input_ids,
            layer_carriers=carriers,
            sampling_positions=positions,
            sampling_params=None,
        )
    print("TINYFWD|root_control|branch=unnamed runner key|refused=True")

    # ---- CONTROL E: THE ROWS ARE THE CALLER'S, AND THE PRODUCT SAYS SO UNDER A
    # PLANTED PERTURBATION (R3', LEAD-LOG 858).
    #
    # WHAT STOOD HERE AND WHY IT WAS NOT ENOUGH. R3 replaced a rolled-positions
    # recompute that could not discriminate -- rolling ``(127, 0, 7, 7)`` by one maps
    # position 7 to position 7 and the stack hands the head rows it has already
    # homogenised, so the recompute landed inside this item's band at a gap of
    # 0.000313 (``increments/launch-054a-r16-driver-20260909T080034Z.out:330``) -- with
    # a perturbation planted on the hidden state, which is sized from the band and
    # cannot be absorbed by row spread. That much is kept. But both tensors it compared
    # came from ``_root_reference``, so it proved the TEST's reference selects rows by
    # position and proved nothing about the product's
    # ``torch.index_select(hidden_states, dim=0, index=sampling_positions)``
    # (``model_fp8.py:7918``): a forward reading ``positions - 1``, or one constant
    # position, passed it (round 3, finding F2).
    #
    # HOW THE PLANT REACHES THE PRODUCT. ``root.forward`` takes ``input_ids`` and
    # builds the hidden state itself, so there is no argument to plant. The one route
    # its own API allows is the one this control takes: a forward hook on
    # ``root.model`` RETURNS the planted tensor in place of the stack's output, and
    # everything after that line -- the index_select this conjunct certifies and the
    # projection -- is the product's own code, run on a state whose one perturbed row
    # this control knows. The hook is removed in a ``finally``.
    #
    # THE PLANT IS SIZED FROM THE BAND ITSELF. The hidden state at the position slot 0
    # asks for gains a delta along ONE head row, scaled so the logit that row projects
    # moves by :data:`ROOT_CONTROL_E_MULTIPLE` times the widest allowance any cell of
    # that slot has, ``ATOL + RTOL * peak``. It is planted in the stack output's own
    # dtype, because that tensor is handed back to the product and the product's rows
    # are the bytes it selects. Only that one position is touched, so three things a
    # wrong selection cannot satisfy together are stated of the PRODUCT's logits:
    #
    #   * slot 0 MUST leave the band, because the planted move is four allowances wide;
    #   * the slots for positions 0 and 7 MUST come back BIT-IDENTICAL, so a forward
    #     reading ``positions - 1``, one constant position, or any other row moves a
    #     row this plant never touched;
    #   * and the product's logits on the planted state MUST still match the
    #     reference's, in this item's own compare form -- the gate a wrong row cannot
    #     pass even when it moves, because the reference knows which row was planted.
    #
    # AND IT CARRIES ITS OWN VACUITY GUARD, which names the head: if the delta maps to
    # no change at slot 0 of the REFERENCE, the head row is zero or masked, and then
    # the control proves nothing and says that instead of passing.
    #
    # IT ALSO REFUSES TO PLANT INTO A ROOT WHOSE HEAD IS NOT THERE. Every gate below
    # calls the product, and the product resolves the head in its first statement
    # (``model_fp8.py:7907``), so a control above this one that left ``lm_head_weight``
    # unset would make this control raise a ValueError about the WEIGHT MAP where the
    # reader is looking for a row-selection failure -- which is what round 4 found.
    # Control B borrows the parameter and puts it back in a ``finally``; this is a
    # guard on that, not a second repair of it, and it is here so this file can never
    # be red for that reason again without saying so.
    if root.lm_head_weight is None:
        raise VacuousControlError(
            "root.lm_head_weight is None before control E plants anything: control B "
            "unsets it to make the product refuse and must put it back in its "
            "finally. The product would refuse by name here, because "
            "model_fp8.py:7907 resolves the head before the stack runs, instead of "
            "selecting the rows this control asks about"
        )
    if root._head_weight() is not root.lm_head_weight:
        raise VacuousControlError(
            "root._head_weight() does not return root.lm_head_weight before control "
            "E plants anything, so the head the product projects with is not the "
            "tensor this item loaded and this control cannot say which rows the root "
            "selected"
        )
    _slot = 0
    _plant_position = int(ROOT_SAMPLING_POSITIONS[_slot])
    _slot_peak = float(expected[_slot].abs().max())
    _slot_band = ATOL + RTOL * _slot_peak
    _head_rows = head.float()
    _head_row = int(_head_rows.norm(dim=1).argmax())
    _head_norm2 = float(_head_rows[_head_row].pow(2).sum())
    if _head_norm2 <= 0.0:
        raise VacuousControlError(
            f"head row {_head_row} is the widest of {int(_head_rows.shape[0])} and "
            f"still has zero norm, so no perturbation of the hidden state can move "
            f"the logit it projects and this control cannot say which rows the root "
            f"selected"
        )
    _plant_scale = ROOT_CONTROL_E_MULTIPLE * _slot_band
    _probe_hidden = hidden.clone()
    _probe_hidden[_plant_position] = (
        hidden[_plant_position].float()
        + (_plant_scale / _head_norm2) * _head_rows[_head_row]
    ).to(hidden.dtype)
    _moved = _root_reference(_probe_hidden, head, ROOT_SAMPLING_POSITIONS)
    _slot_diff = float((_moved[_slot] - expected[_slot]).abs().max())
    print(f"TINYFWD|root_control_e|slot={_slot}|position={_plant_position}"
          f"|source=reference|head_row={_head_row}"
          f"|head_rows={int(_head_rows.shape[0])}"
          f"|slot_peak={_slot_peak:.10g}|allowance={_slot_band:.10g}"
          f"|multiple={ROOT_CONTROL_E_MULTIPLE}"
          f"|planted_logit_delta={_plant_scale:.10g}"
          f"|planted_dtype={_probe_hidden.dtype}"
          f"|measured_slot_max_abs_diff={_slot_diff:.10g}")
    if _slot_diff <= 0.0:
        raise VacuousControlError(
            f"the plant of {_plant_scale:.6g} along head row {_head_row} moved slot "
            f"{_slot} of the reference by exactly nothing, so this head masks it and "
            f"the control cannot say which rows the root selected"
        )
    for _other in range(1, len(ROOT_SAMPLING_POSITIONS)):
        print(f"TINYFWD|root_control_e_untouched|slot={_other}"
              f"|position={int(ROOT_SAMPLING_POSITIONS[_other])}|source=reference"
              f"|max_abs_diff="
              f"{float((_moved[_other] - expected[_other]).abs().max()):.10g}"
              f"|note=a reading; the gate on this slot is the product's own row"
              f" below")
    _planted_stack_calls = []

    def _plant_the_stack_output(_module, _args, _kwargs, _output):
        """Hand the root's own tail the planted state instead of the stack's."""
        _planted_stack_calls.append(_output)
        return _probe_hidden

    _replay_carriers = _stack_carriers(layers, selection)
    _plant_handle = root.model.register_forward_hook(
        _plant_the_stack_output, with_kwargs=True
    )
    try:
        got_planted = root.forward(
            input_ids,
            layer_carriers=_replay_carriers,
            sampling_positions=positions,
        )
    finally:
        _plant_handle.remove()
    _stack_repeat = (bool(torch.equal(_planted_stack_calls[0], hidden))
                     if _planted_stack_calls else None)
    print(f"TINYFWD|root_control_e_product|stack_calls={len(_planted_stack_calls)}"
          f"|hidden_dtype={hidden.dtype}|logits_dtype={got_planted.dtype}"
          f"|shape={tuple(got_planted.shape)}"
          f"|stack_repeated_its_first_output={_stack_repeat}"
          f"|note=the hook replaced the stack's output, so the selection and the"
          f" projection read below are the product's own; the repeat is a reading")
    if len(_planted_stack_calls) != 1:
        raise VacuousControlError(
            f"the planting hook on root.model fired {len(_planted_stack_calls)} "
            f"times where this control plants once, so the logits it compares are "
            f"not the product's reading of the planted state and it cannot say "
            f"which rows the root selected"
        )
    if tuple(got_planted.shape) != tuple(got.shape):
        raise ReferenceShapeError(
            f"the root returned {tuple(got_planted.shape)} on the planted state and "
            f"{tuple(got.shape)} on the stack's own, so the two cannot be compared "
            f"slot by slot"
        )
    _p_slot_diff = float(
        (got_planted[_slot].float() - got[_slot].float()).abs().max()
    )
    _p_outside = not torch.allclose(got_planted[_slot].float(),
                                    got[_slot].float(),
                                    rtol=RTOL, atol=ATOL)
    print(f"TINYFWD|root_control_e|slot={_slot}|position={_plant_position}"
          f"|source=product|planted_logit_delta={_plant_scale:.10g}"
          f"|allowance={_slot_band:.10g}|max_abs_diff={_p_slot_diff:.10g}"
          f"|outside_tolerance={_p_outside}")
    if not _p_outside:
        raise VacuousControlError(
            f"the root's own logits for slot {_slot} stayed inside rtol={RTOL}, "
            f"atol={ATOL} after {_plant_scale:.6g} was planted on position "
            f"{_plant_position}, the row that slot asks for, so this forward does "
            f"not read that row"
        )
    for _other in range(1, len(ROOT_SAMPLING_POSITIONS)):
        _o_diff = float(
            (got_planted[_other].float() - got[_other].float()).abs().max()
        )
        _o_identical = bool(torch.equal(got_planted[_other], got[_other]))
        print(f"TINYFWD|root_control_e_untouched|slot={_other}"
              f"|position={int(ROOT_SAMPLING_POSITIONS[_other])}|source=product"
              f"|max_abs_diff={_o_diff:.10g}|bitwise_identical={_o_identical}")
        if not _o_identical:
            raise VacuousControlError(
                f"slot {_other} asks for position "
                f"{int(ROOT_SAMPLING_POSITIONS[_other])}, which this control planted "
                f"nothing on, and the root's own logits for it moved by "
                f"{_o_diff:.6g}: this forward is reading a row the caller did not ask "
                f"for, or its projection mixes rows"
            )
    print(f"TINYFWD|root_control_e_product_vs_reference"
          f"|max_abs_diff={float((got_planted.float() - _moved).abs().max()):.10g}"
          f"|peak_reference={float(_moved.abs().max()):.10g}"
          f"|rtol={RTOL}|atol={ATOL}"
          f"|note=the item's own compare form, on the planted state")
    torch.testing.assert_close(got_planted.float(), _moved, rtol=RTOL, atol=ATOL)
    _stack_outside_tolerance(
        f"the root's own logits with {_plant_scale:.6g} planted on the hidden state "
        f"at position {_plant_position}, the row slot {_slot} asks for",
        got_planted,
        got,
    )
