# SPDX-License-Identifier: Apache-2.0
"""Tier N acceptance for ``inc-glm53f-027`` -- WP6, the block-quant MoE call site.

The declared acceptance (increment plan revision 32,
``bfb7199ef72039a66dca7bfefdf47c116cc733ac44aabe1e74ea38dea4d9c1e4``, L884),
verbatim on both of the things it declares:

    "the MoE layer's output matches a pure-torch reference MoE built from the
    same weights and router scores with ``assert_close(rtol=3e-2, atol=1e-5)``,
    **1/1** tiny-config case; and the test asserts by inspection that dispatch
    reached the **block-quant** route in **1/1** calls (counter incremented in
    the seam) -- a silent fallback to ``QuantizationType.NONE`` cannot pass."

Tier N, so the command is D1's Tier N template and every env var in it is
load-bearing::

    VLLM_NEURON_CPU_MODE=1 NKI_SIMULATOR=1 NKI_PRECISE_FP=1 \\
    NEURON_PLATFORM_TARGET_OVERRIDE=trn2 \\
    python -m pytest test/vllm_neuron/model/glm5_next/test_moe_path.py \\
      --timeout 60 -v -s

THE ROUTE PREDICATE (D13 form R-2, plan L885). This increment owns NO seam: it
is glue that selects and feeds ``inc-glm53f-025``'s kernel. So the predicate is
the R-2 form -- simulator dispatches counted on the F1 chain (``wrap_nki`` ->
``NKIHOPCaller`` -> HOP -> ``DispatchKey.CPU`` -> ``nki.simulator.simulate_kernel``)
THROUGH A SEAM THIS INCREMENT DOES NOT OWN, namely ``-025``'s
``functional/moe/moe_blockwise_fp8.py``. Four instruments per declared case,
each reported as a number:

1. ``-025``'s seam dispatch counter -- ``nki_dispatch == 1``;
2. the same module's torch-fallback counter -- exactly ``0``;
3. ``can_run_kernel()`` -- ``True``;
4. real ``nki.simulator.simulate_kernel`` entries -- ``1``, and ATTRIBUTED: the
   Python frame chain at each entry is walked for ``-025``'s seam module, so the
   reading distinguishes "a kernel ran" from "*that* kernel ran". Instrument 4
   is the vendor's own entry point, so a bug in instruments 1-3, all of which
   are this campaign's bookkeeping, cannot fake it.

Instrument 4's counted value IS the acceptance bullet's counter clause; it is
cited here, not restated as a second criterion.

WHY THE ATTRIBUTION LEG EXISTS AND WHAT IT IS NOT. ``-026``'s dense seam
(``functional/blockwise_fp8_mm.py``) is NOT on this call site's path -- it is the
dense half, counted downstream by ``inc-glm53f-033`` (plan L933) -- and this
test adds no dense-half dispatch conjunct. The attribution leg is here because
the call site also calls ``build_blockwise_mapping``, which HAS a NKI flow, and
a total-only count could not say which component produced the dispatches.
The mapping's OWN share is therefore MEASURED, not assumed, in the same run --
:func:`_measure_mapping_count` runs the mapping alone under the same instrument
and :func:`_assert_route` then requires the identity
``total == through_seam + mapping_count`` -- and the attribution instrument's
discrimination is armed against a REAL foreign dispatch in
:func:`test_moe_path_attribution_control_reads_a_foreign_seam_as_elsewhere`.

F1: WHY THE NUMERIC ARM ALONE IS A FALSE GREEN WAITING TO HAPPEN. ``-025``'s
seam falls back to the VENDOR'S OWN TORCH REFERENCE when ``can_run_kernel()`` is
false. That reference computes the same function, so the numeric comparison
below passes on the fallback path too --
:func:`test_moe_path_f1_numeric_arm_alone_cannot_discriminate` MEASURES that it
does. The dispatch-count clause is therefore not belt-and-braces; it is the only
thing standing between this file and a green run that exercised no kernel. This
is the plan's own L886 warning, and it is the reason instrument 4 is an
acceptance conjunct rather than a diagnostic print.

WHAT THE COMPARATOR IS. A pure-torch reference MoE authored in this file
(:func:`torch_reference_moe`), built from the SAME fp8 weights, the SAME block
scales and the SAME router scores the call site is handed. Its provenance is not
taken on trust: :func:`test_moe_path_reference_agrees_with_vendor_torch_oracle`
runs the vendor's own reference over the same case and requires agreement AT THE
SAME DECLARED TOLERANCES. So a shared misreading of the scale layout would have
to be shared with ``nkilib``'s reference too.

FIXTURE CONDITIONING (the plan's carry #7, and ``-025``'s lesson at the layer
below). ``-025`` attempt 1 died to catastrophic cancellation in a SIGNED random
fixture at this same ``rtol=3e-2``: over an ``H=512`` contraction, signed terms
of magnitude ~1e4 sum to elements ~1e-1, and no correct bf16-accumulating kernel
can satisfy a pointwise relative tolerance dominated by cancellation. This
fixture is therefore UNSIGNED on the fp8-e4m3 grid, and the conditioning is
asserted rather than hoped for
(:func:`test_moe_path_fixture_conditioning_is_measured_not_assumed`). Every
fixture value is exact in both fp8-e4m3 and bf16 -- weights and hidden states
are multiples of ``1/8`` in ``[0.125, 0.875]``, affinities are multiples of
``1/8``, and scales are exact powers of two -- so no cast anywhere in the
fixture introduces error the tolerance would then have to absorb. The declared
tolerance is UNCHANGED; only the world it is measured in is conditioned.

D1.2: the item count reported for this file is pytest ITEMS from a dedicated
``--collect-only -q`` run, not a count declared here.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest
import torch

import nki
import nki.simulator

from vllm_neuron.functional.moe.blockwise_fp8_retile import (
    BLOCK_QUANT_SIZE,
    DOWN,
    GATE_UP,
    TILE_SIZE,
    consumer_scale_shape,
    is_pow2_exact,
    retile_block_scales,
)
from vllm_neuron.functional.moe.moe_blockwise_fp8 import (
    blockwise_fp8_moe,
    blockwise_fp8_moe_torch_oracle,
    dispatch_counters,
    kernel_identity,
    kernel_scale_shape,
    reset_dispatch_counters,
)
from vllm_neuron.functional.moe.moe_blockwise_fp8 import (
    to_kernel_scale_layout as moe_to_kernel_scale_layout,
)
from vllm_neuron.utils.neuron_utils import can_run_kernel

#: ``-025``'s seam module. The attribution target of instrument 4, resolved as a
#: FILE PATH off the imported module rather than spelled as a string, so a moved
#: module fails the import instead of silently making the attribution unmatchable.
import vllm_neuron.functional.moe.moe_blockwise_fp8 as _seam_module

_SEAM_FILE = _seam_module.__file__

# ---------------------------------------------------------------------------
# The tiny config. Every extent is forced, and by what is cited next to it.
# ---------------------------------------------------------------------------

#: ``H`` and ``I_TP`` are the SMALLEST pair ``-025``'s five admission gates
#: accept (``moe_blockwise_fp8.py:_require_blocked``): ``H % 256 == 0``,
#: ``512 <= H <= 8192``, ``H % PSUM_SIZE(512) == 0``, ``I_TP % 256 == 0`` and
#: ``I_TP % (256 * NUM_SHARDS) == 0``. Chosen to ADMIT the kernel, which is
#: ``-025``'s landed carry: a geometry it refuses raises rather than falling
#: back, so an inadmissible tiny config would measure the refusal path.
H = 512
I_TP = 512

#: This rank's local expert count. ``4`` rather than ``2`` so that
#: ``block_to_expert`` is not the identity and a permuted block-to-expert
#: mapping is observable, and so ``num_blocks`` exceeds the number of OCCUPIED
#: blocks -- the empty-block case the vendor kernel handles by indexing the
#: padding slot. Both facts are measured in
#: :func:`test_moe_path_mapping_shape_is_the_one_the_seam_consumes`.
E = 4

#: Experts per token. ``2`` keeps every expert's occupancy at ``T * K / E = 128``
#: tokens, comfortably inside one ``block_size`` so the mapping needs exactly one
#: block per expert.
K = 2

#: Real tokens the call site is handed. ``256`` is the extent
#: ``build_blockwise_mapping`` sees, because the call site builds the mapping over
#: the real token count and appends the kernel's padding slot AFTERWARDS. Both of
#: the mapping's kernel gates turn on that extent being even, so the order is
#: load-bearing; see
#: :func:`test_moe_path_call_site_maps_before_padding_and_dispatches_nki`, which
#: measures it, and
#: :func:`test_moe_path_mapping_order_moves_no_numbers`, which measures that the
#: two orders produce the same tensors.
T = 256

#: Tokens per block. ``256`` is the vendor kernel's ``B % 256 == 0`` assert
#: (``bwmm_shard_on_I.py:667``) at its minimum.
B = 256

H_256 = H // BLOCK_QUANT_SIZE
I_256 = I_TP // BLOCK_QUANT_SIZE

#: The plan's declared tolerance, order named inline per D3. Never widened here:
#: widening a declared value is the user's election through the lead, not this
#: file's.
RTOL = 3e-2
ATOL = 1e-5

#: The plan's declared case count for the numeric arm: ``1/1``.
DECLARED_CASES = 1

#: The plan's declared dispatch count for the route arm: ``1/1`` calls.
#:
#: WHAT THE PLAN ACTUALLY DECLARES, AND WHAT IT DOES NOT. The plan declares the
#: SEAM reading and only the seam reading: ``-025``'s ``seam_nki_dispatch == 1``,
#: ``seam_torch_fallback == 0``, and the attributed ``through_seam == 1``, in
#: ``1/1`` calls. It declares NO total simulator-entry count and NO mapping
#: dispatch count anywhere. So the total clause inside :func:`_assert_route` is
#: THIS FILE's own instrument against the F1 false green, not a plan-declared
#: value: it requires at least one entry and then ATTRIBUTES every entry to a
#: measured source. Repair round 2 of ``inc-glm53f-027`` corrected this comment,
#: which had read the total as plan-backed, and replaced the equality with the
#: attribution identity.
DECLARED_DISPATCHES = 1

_FP8 = torch.float8_e4m3fn

#: The campaign's pinned checkpoint config, and its digest. The quantisation
#: policy this call site routes on is read through the REAL recognition path
#: from this file, not hand-fed: ``config.json`` -> ``Glm5NextConfig`` ->
#: ``QuantizationSpec`` -> ``BlockFp8QuantMethod``. ``inc-glm53f-023``'s test
#: pins the same digest; a divergence here means one of the two is stale.
FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "config.json"
FIXTURE_SHA256 = "5ed24d23a3e14a038352e1bdc21fd25fc90ff2291d3f6a310acf5d4036665a1d"

#: Fixture-only constants. The plan declares the tolerances, the case count and
#: the tiny-config shape family; it declares nothing about how the fixture is
#: built, so these are labelled as this file's own choices.
WEIGHT_SEED_GATE = 271
WEIGHT_SEED_UP = 272
WEIGHT_SEED_DOWN = 273
HIDDEN_SEED = 274
SCALE_SEED_GATE = 275
SCALE_SEED_UP = 276
SCALE_SEED_DOWN = 277

#: Dyadic affinity values -- multiples of ``1/8``, hence exact in bf16, which is
#: the dtype the call site casts the masked affinities to before the kernel
#: multiplies them into the hidden states. FIVE values, not four: with a table
#: length equal to ``E`` the value index would collapse onto the expert index
#: and every one of an expert's 128 tokens would carry the same affinity, so a
#: token-permuted affinity assignment would be invisible. Five is coprime to
#: ``E = 4``, which decouples the two.
AFFINITY_VALUES = (0.5, 0.25, 0.375, 0.75, 0.625)

#: Tokens each expert must receive, by construction. Asserted, not assumed.
TOKENS_PER_EXPERT = T * K // E


# ---------------------------------------------------------------------------
# Named errors. A failure must say WHICH instrument disagreed; a bare
# ``AssertionError`` from three different causes is one message for three bugs.
# ---------------------------------------------------------------------------
class RouteInstrumentError(AssertionError):
    """A route reading that is not what the plan declares."""


class VacuousControlError(AssertionError):
    """A control whose input could not have made it fail.

    Raised when a control's stream is empty or its two arms are identical: a
    zero over vacuous input measures nothing, so the control refuses to report a
    pass it did not earn (D1.5).
    """


class ReferenceShapeError(AssertionError):
    """The comparator was handed an operand of a shape it cannot mean."""


class ExportSurfaceError(AssertionError):
    """The export hub does not resolve, or resolves a colliding name."""


class F1PreconditionError(AssertionError):
    """The pow2 losslessness precondition did not hold on this case's scales."""


class FixtureConditioningError(AssertionError):
    """The fixture is not conditioned the way the tolerance is measured against."""


# ---------------------------------------------------------------------------
# Instrument 4, with attribution.
# ---------------------------------------------------------------------------
class _AttributedSimulatorCounter:
    """Counts ``nki.simulator.simulate_kernel`` entries and attributes each one.

    ``total`` is the raw count -- the vendor's entry point, independent of every
    counter this campaign wrote. ``through_seam`` is the subset whose Python
    frame chain contains a frame executing in ``seam_file``; ``elsewhere`` is
    the remainder.

    The frame walk is what makes the R-2 predicate say "through ``-025``'s
    seam" rather than merely "something dispatched". It works across the
    ``wrap_nki`` -> HOP -> ``DispatchKey.CPU`` hop because CPython links each new
    Python frame to the interpreter's current top frame regardless of
    intervening C++ frames, so the calling Python frame stays reachable through
    ``f_back``. That this is true is not argued here -- it is measured, because
    :func:`test_moe_path_attribution_control_reads_a_foreign_seam_as_elsewhere`
    requires ``through_seam == 0`` on a dispatch through a DIFFERENT seam while
    ``total == 1``, and the acceptance requires ``through_seam == 1`` on this
    one. One instrument, two opposite readings on real input.
    """

    def __init__(self, seam_file: str = _SEAM_FILE) -> None:
        self.total = 0
        self.through_seam = 0
        self.elsewhere = 0
        self._seam_file = os.path.realpath(seam_file)
        self._real = None

    def __enter__(self) -> "_AttributedSimulatorCounter":
        self._real = nki.simulator.simulate_kernel
        real = self._real
        seam_file = self._seam_file

        def counting(*args, **kwargs):
            self.total += 1
            frame = sys._getframe(1)
            attributed = False
            while frame is not None:
                if os.path.realpath(frame.f_code.co_filename) == seam_file:
                    attributed = True
                    break
                frame = frame.f_back
            if attributed:
                self.through_seam += 1
            else:
                self.elsewhere += 1
            return real(*args, **kwargs)

        nki.simulator.simulate_kernel = counting
        return self

    def __exit__(self, *exc_info) -> None:
        nki.simulator.simulate_kernel = self._real


def _assert_route(
    sim: _AttributedSimulatorCounter,
    expected: int,
    label: str,
    *,
    mapping_count: int,
) -> str:
    """Read all four route instruments and return the reading for the transcript.

    The certifying component of each conjunct is named in the message (D1.4):
    conjuncts 1 and 2 are ``-025``'s module-level counters, conjunct 3 is
    ``vllm_neuron.utils.neuron_utils.can_run_kernel``, conjuncts 4 and 5 are
    ``nki.simulator.simulate_kernel`` itself, conjunct 6 is the frame walk.

    ``expected`` is the PLAN-DECLARED seam count (``1/1``); it governs conjuncts
    1 and 6. ``mapping_count`` is a READING, never a literal: the caller measures
    the token-block mapping alone with :func:`_measure_mapping_count`, under the
    same instrument and in the same run, and passes what the instrument said.

    Conjuncts 4 and 5 are this file's F1 guard, not a plan-declared count (see
    :data:`DECLARED_DISPATCHES`). Conjunct 4 refuses a numeric pass with no
    simulator entry at all. Conjunct 5 then requires every entry to be
    ATTRIBUTED: the total must equal the seam's attributed share plus the
    mapping's measured share, with nothing left over. That identity is what
    catches an unattributed third dispatcher AND -- because the mapping's share
    is nonzero only when the call site builds the mapping over the real token
    count -- a call site that pads before it maps.
    """
    nki_dispatch, torch_fallback = dispatch_counters()
    gate = can_run_kernel(torch.zeros(1))
    reading = (
        f"[{label}] seam_nki_dispatch={nki_dispatch} "
        f"seam_torch_fallback={torch_fallback} can_run_kernel={gate} "
        f"simulate_kernel_total={sim.total} "
        f"simulate_kernel_through_025_seam={sim.through_seam} "
        f"simulate_kernel_elsewhere={sim.elsewhere} "
        f"measured_mapping_count={mapping_count} "
        f"attribution_sum={sim.through_seam + mapping_count}"
    )
    print(reading)
    if nki_dispatch != expected:
        raise RouteInstrumentError(
            f"{label}: -025's seam dispatch counter read {nki_dispatch}, "
            f"declared {expected}. {reading}"
        )
    if torch_fallback != 0:
        raise RouteInstrumentError(
            f"{label}: -025's torch-fallback counter read {torch_fallback}, "
            f"declared exactly 0. A fallback pass compares the vendor's torch "
            f"reference against a torch reference and measures no kernel. "
            f"{reading}"
        )
    if gate is not True:
        raise RouteInstrumentError(
            f"{label}: can_run_kernel() read {gate!r}, declared True. {reading}"
        )
    if sim.total < 1:
        raise RouteInstrumentError(
            f"{label}: nki.simulator.simulate_kernel ran {sim.total} times, so "
            f"no kernel was simulated at all. A numeric pass without a simulator "
            f"entry is the F1 false green. {reading}"
        )
    if sim.total != sim.through_seam + mapping_count:
        raise RouteInstrumentError(
            f"{label}: {sim.total} simulator entries do not add up. "
            f"{sim.through_seam} attributed to -025's seam plus "
            f"{mapping_count} measured for the token-block mapping alone is "
            f"{sim.through_seam + mapping_count}. Either some component nobody "
            f"measured dispatched, or the call site handed the mapping an extent "
            f"whose NKI gates refuse -- which is what padding before mapping "
            f"does, finding B21-027. {reading}"
        )
    if sim.through_seam != expected:
        raise RouteInstrumentError(
            f"{label}: {sim.through_seam} of {sim.total} simulator entries were "
            f"attributed to -025's seam ({_SEAM_FILE}), declared {expected}. "
            f"An unattributed dispatch means some OTHER component produced it, "
            f"which is not what R-2 counts. {reading}"
        )
    return reading


def _run_mapping(expert_affinities: torch.Tensor, label: str) -> dict:
    """Run the token-block mapping ALONE under the file's own counter.

    Returns the mapping's four outputs and its measured simulator entry count.
    The call site calls exactly two components that can dispatch -- ``-025``'s
    seam and ``build_blockwise_mapping`` -- so the mapping's own share, measured
    on its own, is what makes :func:`_assert_route`'s attribution identity
    complete rather than a subtraction.

    The mapping's entries must land in ``elsewhere``, never in ``through_seam``:
    the mapping lives in ``functional/moe/moe_blockwise.py`` and ``-025``'s seam
    is a different file. That is asserted here, so adding ``mapping_count`` to
    ``through_seam`` provably double-counts nothing.
    """
    from vllm_neuron.functional import build_blockwise_mapping

    with _AttributedSimulatorCounter() as mapping_sim:
        masked, token_position_to_id, block_to_expert, conditions = (
            build_blockwise_mapping(
                expert_affinities=expert_affinities,
                num_local_experts=E,
                num_experts_per_token=K,
                block_size=B,
                moe_group=None,
                tp_degree=1,
            )
        )
    print(
        f"[mapping-{label}] total_tokens={expert_affinities.shape[0]} "
        f"simulate_kernel_total={mapping_sim.total} "
        f"through_025_seam={mapping_sim.through_seam} "
        f"elsewhere={mapping_sim.elsewhere} "
        f"masked_shape={tuple(masked.shape)} "
        f"num_blocks={block_to_expert.numel()} "
        f"conditions={conditions.tolist()} "
        f"occupied_positions={int((token_position_to_id >= 0).sum())} "
        f"masked_nonzeros={int((masked != 0).sum())}"
    )
    if mapping_sim.through_seam != 0:
        raise RouteInstrumentError(
            f"mapping-{label}: {mapping_sim.through_seam} of "
            f"{mapping_sim.total} entries produced by the token-block mapping "
            f"were attributed to -025's seam ({_SEAM_FILE}). The mapping is a "
            f"different file, so this reading would make the attribution "
            f"identity double-count."
        )
    return {
        "masked": masked,
        "token_position_to_id": token_position_to_id,
        "block_to_expert": block_to_expert,
        "conditions": conditions,
        "total": mapping_sim.total,
    }


def _measure_mapping_count(expert_affinities: torch.Tensor, label: str) -> int:
    """The mapping's measured simulator entry count, for the attribution identity."""
    return _run_mapping(expert_affinities, label)["total"]


def _envelope_affinities(tokens: int) -> torch.Tensor:
    """This file's fixture scatter pattern, at an arbitrary token count.

    ``K`` experts per token, rotating through the experts and through
    :data:`AFFINITY_VALUES`, which is exactly what :func:`_build_case` does at
    ``T``. It exists so the envelope arm can read the mapping's route at token
    counts the fixture does not build. The plan declares nothing about extents
    above ``T``, so this pattern is labelled as this file's own choice.
    """
    out = torch.zeros(tokens, E, dtype=torch.float32)
    for token in range(tokens):
        for slot in range(K):
            out[token, (token + slot) % E] = AFFINITY_VALUES[
                (token + slot) % len(AFFINITY_VALUES)
            ]
    return out


# ---------------------------------------------------------------------------
# The quantisation policy, read through the real recognition path.
# ---------------------------------------------------------------------------
def _pinned_raw_config() -> dict:
    """The pinned checkpoint config, digest-verified before it is parsed."""
    digest = hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest()
    if digest != FIXTURE_SHA256:
        raise VacuousControlError(
            f"pinned fixture digest moved: {digest} != {FIXTURE_SHA256}. The "
            f"quantisation policy this call site routes on would no longer be "
            f"the campaign's registered one."
        )
    return json.loads(FIXTURE_PATH.read_text())


def _block_quant_config():
    """``Glm5NextQuantConfig`` for the pinned checkpoint -- nothing hand-fed.

    The import is FUNCTION-LOCAL. ``test_factory.py``'s C03 asserts
    ``"vllm_neuron.model.glm5_next.model_fp8" not in sys.modules``, which is
    what certifies ``factory.py``'s lazy import; a module-level import here
    would populate ``sys.modules`` for the whole directory run. This is
    ``inc-glm53f-023``'s and ``-032``'s landed idiom in this directory, adopted
    for the same reason.
    """
    from vllm_neuron.model.glm5_next.config import Glm5NextConfig
    from vllm_neuron.model.glm5_next.model_fp8 import Glm5NextQuantConfig

    return Glm5NextQuantConfig.from_model_config(
        Glm5NextConfig.from_configs(_pinned_raw_config())
    )


def _build_bank():
    """The routed-expert bank at this rank's tiny partition, plus its config."""
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig
    from vllm_neuron.model.glm5_next.model_fp8 import Glm5NextRoutedExperts

    text_config = Glm5NextTextConfig(
        hidden_size=H,
        moe_intermediate_size=I_TP,
        n_routed_experts=E,
        num_experts_per_tok=K,
    )
    bank = Glm5NextRoutedExperts(text_config, world_size=1)
    if int(bank.num_local_experts) != E:
        raise VacuousControlError(
            f"the bank reports num_local_experts={bank.num_local_experts}, but "
            f"the fixture is built for {E}; the call site's own extent check "
            f"would fire before any numerics ran"
        )
    return bank, text_config


# ---------------------------------------------------------------------------
# The fixture. Built THROUGH ``inc-glm53f-024``'s retile producer, so the
# producer is exercised rather than mimicked, and the scales the comparator
# reads are the ones the kernel is handed.
# ---------------------------------------------------------------------------
def _pow2_checkpoint_scales(seed: int, rows: int, cols: int) -> torch.Tensor:
    """``(E, rows//128, cols//128)`` fp32 scales, every entry an exact power of 2.

    Exact powers of two, hence mutually pow2-related within any ``256``-block and
    each retained block scale itself a power of two: the COMPLETE losslessness
    condition ``inc-glm53f-024`` part 5 declares. Satisfied by construction here
    and RE-ASSERTED from the producer's emitted records in
    :func:`test_moe_path_f1_precondition_complete_condition_n_over_n`.

    Exponents vary per tile so the scales are DISTINCT: a comparison run on
    uniform scales cannot see a permuted scale layout.
    """
    generator = torch.Generator().manual_seed(seed)
    exponents = torch.randint(
        -3, 4, (E, rows // TILE_SIZE, cols // TILE_SIZE), generator=generator
    )
    return torch.ldexp(torch.ones_like(exponents, dtype=torch.float32), exponents)


def _fp8_grid_values(seed: int, *shape: int) -> torch.Tensor:
    """Values already on the fp8-e4m3 grid: multiples of ``1/8`` in ``[1/8, 7/8]``.

    UNSIGNED, and that is the conditioning choice the plan's carry #7 requires.
    With signed values every dot product over the ``H=512`` contraction is a
    near-cancelling sum, so reference elements land arbitrarily close to zero
    while the terms that built them are ~1e4; a pointwise RELATIVE tolerance is
    then dominated by cancellation rather than by kernel error, and no correct
    bf16-accumulating kernel can satisfy it. ``-025`` measured exactly that one
    layer down. Every value here is also exact in bf16, so the call site's cast
    of the affinities and the hidden states introduces nothing.
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(1, 8, shape, generator=generator).to(torch.float32) / 8.0


def _build_case() -> dict:
    """The tiny config in the form the CALL SITE takes -- not the seam's form.

    The call site's contract differs from the seam's in three ways this builder
    honours, because honouring them is what makes the test exercise the glue:

    * no padding-token row -- the call site appends it (``[T, H]``, not
      ``[T+1, H]``);
    * ``[T, E]`` scattered router scores -- the call site flattens and masks
      them through ``build_blockwise_mapping``, not ``[(T+1)*E, 1]``;
    * FLAT consumer scales as ``inc-glm53f-024``'s producer emits them -- the
      call site bridges them to the kernel's logical view.
    """
    # --- gate/up: the producer runs per fusion half, on (E, H, I_TP). ------- #
    # The two halves are merged in the FLAT domain, which is legitimate because
    # ``moe_to_kernel_scale_layout`` is a documented C-order reshape
    # (``moe_blockwise_fp8.py:170-206``): a ``view`` onto the logical shape
    # addresses exactly the elements that reshape would.
    gup_flat = torch.full(
        consumer_scale_shape(E, H, I_TP, GATE_UP), float("nan"), dtype=torch.float32
    )
    gup_logical_view = gup_flat.view(*kernel_scale_shape(E, H, I_TP, GATE_UP))
    gup_weight = torch.empty((E, H, 2, I_TP), dtype=torch.float32)
    gup_results = []
    for gate_or_up, (weight_seed, scale_seed) in enumerate(
        ((WEIGHT_SEED_GATE, SCALE_SEED_GATE), (WEIGHT_SEED_UP, SCALE_SEED_UP))
    ):
        checkpoint = _pow2_checkpoint_scales(scale_seed, H, I_TP)
        weights = _fp8_grid_values(weight_seed, E, H, I_TP)
        result = retile_block_scales(
            weights.to(_FP8), checkpoint, projection=GATE_UP, gate_or_up=gate_or_up
        )
        gup_results.append(result)
        gup_weight[:, :, gate_or_up, :] = result.retiled_weights.to(torch.float32)
        bridged = moe_to_kernel_scale_layout(
            result.consumer_scales, E, H, I_TP, projection=GATE_UP
        )
        # The producer writes only THIS half's slots and leaves the other half
        # NaN on purpose, so take this half's slice. Merging the two is the
        # weight loader's step in production; here it is the fixture's.
        gup_logical_view[:, :, gate_or_up, :, :] = bridged[:, :, gate_or_up, :, :]

    # NaN survives arithmetic, so an unwritten slot would poison the output
    # rather than pass quietly -- but a poisoned output fails the numeric arm
    # for a fixture reason. Refuse here, where the cause is legible.
    if not bool(torch.isfinite(gup_flat).all()):
        raise VacuousControlError(
            f"{int((~torch.isfinite(gup_flat)).sum())} gate/up scale slots are "
            f"still NaN after both fusion halves were written; the fixture, not "
            f"the call site, is wrong"
        )

    # --- down: the producer's ``rows`` is the H axis and ``cols`` the I axis, --
    # --- so the physically-[E, I_TP, H] weight is retiled in (E, H, I_TP) view.
    down_checkpoint = _pow2_checkpoint_scales(SCALE_SEED_DOWN, H, I_TP)
    down_weight_hi = _fp8_grid_values(WEIGHT_SEED_DOWN, E, H, I_TP)
    down_result = retile_block_scales(
        down_weight_hi.to(_FP8), down_checkpoint, projection=DOWN
    )
    down_flat = down_result.consumer_scales
    if not bool(torch.isfinite(down_flat).all()):
        raise VacuousControlError(
            "down-projection scale slots contain NaN; a single producer call "
            "covers every DOWN slot, so this is a producer contract change"
        )
    # Back to the kernel's physical [E, I_TP, H].
    down_weight = (
        down_result.retiled_weights.to(torch.float32).transpose(1, 2).contiguous()
    )

    # --- activations and router scores, in the call site's own form. -------- #
    hidden = _fp8_grid_values(HIDDEN_SEED, T, H).to(torch.bfloat16)
    affinities = torch.zeros(T, E, dtype=torch.float32)
    for token in range(T):
        for slot in range(K):
            expert = (token + slot) % E
            affinities[token, expert] = AFFINITY_VALUES[(token + slot) % len(AFFINITY_VALUES)]

    return {
        "call_site_inputs": dict(
            hidden_states=hidden,
            expert_affinities=affinities,
            gate_up_proj_weight=gup_weight.to(_FP8),
            down_proj_weight=down_weight.to(_FP8),
            gate_up_consumer_scales=gup_flat,
            down_consumer_scales=down_flat,
        ),
        "gup_logical": gup_logical_view.clone(),
        "down_logical": moe_to_kernel_scale_layout(
            down_flat, E, H, I_TP, projection=DOWN
        ),
        "gup_results": gup_results,
        "down_result": down_result,
    }


# ---------------------------------------------------------------------------
# The comparator: a pure-torch reference MoE over the SAME weights and scores.
# ---------------------------------------------------------------------------
def _dequantise(weight_fp8: torch.Tensor, block_scale: torch.Tensor) -> torch.Tensor:
    """``weight[k, n] * scale[k // 256, n // 256]``, expanded, in fp32.

    The block scale is broadcast by ``repeat_interleave`` on both axes rather
    than by an index computation, so this function repeats none of the
    producer's or the bridge's arithmetic and cannot share an off-by-one with
    them.
    """
    if weight_fp8.dim() != 2 or block_scale.dim() != 2:
        raise ReferenceShapeError(
            f"expected 2-D weight and 2-D block scale, got "
            f"{tuple(weight_fp8.shape)} and {tuple(block_scale.shape)}"
        )
    rows, cols = weight_fp8.shape
    if tuple(block_scale.shape) != (rows // BLOCK_QUANT_SIZE, cols // BLOCK_QUANT_SIZE):
        raise ReferenceShapeError(
            f"block scale {tuple(block_scale.shape)} does not tile a "
            f"{rows}x{cols} weight at granularity {BLOCK_QUANT_SIZE}"
        )
    expanded = block_scale.repeat_interleave(BLOCK_QUANT_SIZE, dim=0).repeat_interleave(
        BLOCK_QUANT_SIZE, dim=1
    )
    return weight_fp8.to(torch.float32) * expanded


def torch_reference_moe(
    hidden_states: torch.Tensor,
    expert_affinities: torch.Tensor,
    gate_up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    gate_up_logical_scale: torch.Tensor,
    down_logical_scale: torch.Tensor,
) -> torch.Tensor:
    """A pure-torch block-quant MoE. No NKI, no mapping, no vendor code.

    Semantics transcribed from the vendor's own reference
    (``nkilib/core/moe/moe_cte/moe_cte_torch.py``) at the two points where a
    plausible alternative would give different numbers:

    * ``PRE_SCALE`` -- the expert affinity multiplies the HIDDEN STATES before
      the gate/up matmuls (``:193``), not the expert output after the down
      matmul. The two differ because SiLU is nonlinear, so this is not a
      refactor of the same expression. ``PRE_SCALE`` is the kernel's own default
      (``bwmm_shard_on_I.py:122``) and ``-025``'s seam forwards it.
    * the affinity is applied IN THE ACTIVATION DTYPE (bf16), matching the
      call site's ``.to(hidden_states.dtype)`` cast, and the per-expert
      contribution is rounded to bf16 before accumulation, matching the
      vendor's ``scaled.to(bfloat16).to(float32)``.

    The trailing ``TILE_SIZE`` axis of both logical scale tensors is the
    partition broadcast -- 128 copies of one scalar -- so index ``0`` is read.
    That this is a broadcast and not data is ``-025``'s settled finding, not an
    assumption of this file.

    Returns:
        ``[T, H]`` fp32. No padding row: this reference never had one.
    """
    if hidden_states.dim() != 2 or expert_affinities.dim() != 2:
        raise ReferenceShapeError(
            f"expected [T, H] hidden and [T, E] affinities, got "
            f"{tuple(hidden_states.shape)} and {tuple(expert_affinities.shape)}"
        )
    tokens, hidden = hidden_states.shape
    num_experts = expert_affinities.shape[1]
    if expert_affinities.shape[0] != tokens:
        raise ReferenceShapeError(
            f"affinities cover {expert_affinities.shape[0]} tokens, hidden "
            f"states cover {tokens}"
        )
    output = torch.zeros(tokens, hidden, dtype=torch.float32)
    for expert in range(num_experts):
        gate_weight = _dequantise(
            gate_up_proj_weight[expert, :, 0, :],
            gate_up_logical_scale[expert, :, 0, :, 0],
        )
        up_weight = _dequantise(
            gate_up_proj_weight[expert, :, 1, :],
            gate_up_logical_scale[expert, :, 1, :, 0],
        )
        down_weight = _dequantise(
            down_proj_weight[expert], down_logical_scale[expert, :, :, 0]
        )
        rows = torch.nonzero(expert_affinities[:, expert], as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        scale = expert_affinities[rows, expert].to(hidden_states.dtype).unsqueeze(1)
        local = (scale * hidden_states[rows]).to(torch.float32)
        gate_act = local @ gate_weight
        up_act = local @ up_weight
        intermediate = torch.nn.functional.silu(gate_act) * up_act
        output[rows] += (intermediate @ down_weight).to(torch.bfloat16).to(
            torch.float32
        )
    return output


def _max_rel_error(got: torch.Tensor, want: torch.Tensor) -> float:
    """``max |got - want| / (|want| + ATOL)`` -- a number, not a verdict."""
    return float(((got - want).abs() / (want.abs() + ATOL)).max())


def _nonempty_or_raise(reference: torch.Tensor, label: str) -> int:
    """Refuse a comparison whose reference is all zeros (D1.5)."""
    nonzero_rows = int((reference.abs().sum(-1) > 0).sum())
    if nonzero_rows == 0:
        raise VacuousControlError(
            f"{label}: every reference row is zero, so assert_close would pass "
            f"on a function that returns zeros"
        )
    return nonzero_rows


# ===========================================================================
# THE DECLARED ACCEPTANCE CASE. Both conjuncts, one call, 1/1.
# ===========================================================================
def test_moe_path_output_matches_pure_torch_reference() -> None:
    """The MoE call site's output vs a pure-torch reference, and the route.

    The plan's declared Expected, both halves:
    ``assert_close(rtol=3e-2, atol=1e-5)`` on ``1/1`` tiny-config case, AND the
    dispatch reached the block-quant route in ``1/1`` calls.
    """
    bank, _text_config = _build_bank()
    quant_config = _block_quant_config()
    case = _build_case()

    reset_dispatch_counters()
    with _AttributedSimulatorCounter() as sim:
        got = bank.block_quant_expert_mm(
            quant_config=quant_config, block_size=B, **case["call_site_inputs"]
        )
    # The mapping's own share, measured on the SAME inputs the call site hands it
    # (the real token count -- the padding slot is appended after the mapping),
    # under the same instrument, in this same run. A reading, never a literal.
    mapping_count = _measure_mapping_count(
        case["call_site_inputs"]["expert_affinities"], "acceptance"
    )
    reading = _assert_route(
        sim, DECLARED_DISPATCHES, "acceptance", mapping_count=mapping_count
    )

    if tuple(got.shape) != (T, H):
        raise ReferenceShapeError(
            f"the call site returned {tuple(got.shape)}, declared ({T}, {H}) -- "
            f"the padding-token row must be sliced off"
        )

    want = torch_reference_moe(
        hidden_states=case["call_site_inputs"]["hidden_states"],
        expert_affinities=case["call_site_inputs"]["expert_affinities"],
        gate_up_proj_weight=case["call_site_inputs"]["gate_up_proj_weight"],
        down_proj_weight=case["call_site_inputs"]["down_proj_weight"],
        gate_up_logical_scale=case["gup_logical"],
        down_logical_scale=case["down_logical"],
    )
    got_f32 = got.to(torch.float32)
    nonzero_rows = _nonempty_or_raise(want, "acceptance")

    print(
        f"[acceptance] cases={DECLARED_CASES}/{DECLARED_CASES} "
        f"nonzero_reference_rows={nonzero_rows}/{T} "
        f"max_rel_error={_max_rel_error(got_f32, want):.6e} "
        f"max_abs_error={float((got_f32 - want).abs().max()):.6e} "
        f"rtol={RTOL} atol={ATOL} reference_absmax={float(want.abs().max()):.6e} "
        f"|| {reading}"
    )
    torch.testing.assert_close(got_f32, want, rtol=RTOL, atol=ATOL)


# ===========================================================================
# The comparator's own provenance, at the SAME declared tolerances.
# ===========================================================================
def test_moe_path_reference_agrees_with_vendor_torch_oracle() -> None:
    """This file's reference vs ``nkilib``'s own, on the same case.

    Without this arm the acceptance compares the call site against arithmetic
    authored in the same repository under the same reading of the scale layout,
    and a SHARED misreading would be invisible. The vendor's reference is
    independent of both. Compared at ``rtol=3e-2, atol=1e-5`` -- the declared
    tolerances, unchanged: this arm is a provenance check, not a looser one.

    The vendor reference consumes the seam's operand form, so the mapping is
    rebuilt here with the same inputs the call site uses. Rebuilding it is
    legitimate for a SUPPLEMENTARY arm; the declared arm above never sees it.
    """
    from vllm_neuron.functional import build_blockwise_mapping

    case = _build_case()
    inputs = case["call_site_inputs"]
    hidden = inputs["hidden_states"]
    affinities = inputs["expert_affinities"]

    padded_hidden = torch.cat(
        [hidden, torch.zeros(1, H, dtype=hidden.dtype)], dim=0
    )
    padded_affinities = torch.cat(
        [affinities, torch.zeros(1, E, dtype=affinities.dtype)], dim=0
    )
    masked, token_position_to_id, block_to_expert, _conditions = (
        build_blockwise_mapping(
            expert_affinities=padded_affinities,
            num_local_experts=E,
            num_experts_per_token=K,
            block_size=B,
            moe_group=None,
            tp_degree=1,
        )
    )
    oracle = blockwise_fp8_moe_torch_oracle(
        hidden_states=padded_hidden,
        expert_affinities_masked=masked.to(hidden.dtype),
        gate_up_proj_weight=inputs["gate_up_proj_weight"],
        down_proj_weight=inputs["down_proj_weight"],
        block_size=B,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert.reshape(-1, 1),
        gate_up_proj_scale=case["gup_logical"],
        down_proj_scale=case["down_logical"],
    )
    vendor = oracle.to(torch.float32)[:T]
    mine = torch_reference_moe(
        hidden_states=hidden,
        expert_affinities=affinities,
        gate_up_proj_weight=inputs["gate_up_proj_weight"],
        down_proj_weight=inputs["down_proj_weight"],
        gate_up_logical_scale=case["gup_logical"],
        down_logical_scale=case["down_logical"],
    )
    _nonempty_or_raise(vendor, "reference-provenance")
    print(
        f"[reference-provenance] max_rel_error="
        f"{_max_rel_error(mine, vendor):.6e} "
        f"max_abs_error={float((mine - vendor).abs().max()):.6e} "
        f"rtol={RTOL} atol={ATOL}"
    )
    torch.testing.assert_close(mine, vendor, rtol=RTOL, atol=ATOL)


# ===========================================================================
# F1. The two arms that say why the counter clause is load-bearing.
# ===========================================================================
def test_moe_path_f1_numeric_arm_alone_cannot_discriminate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MEASURED: the numeric comparison passes on the FALLBACK path too.

    ``-025``'s seam falls back to the vendor's own torch reference, which
    computes the same function, so a silently-unrouted call site produces
    numbers this file's tolerance accepts. This arm runs that world and shows
    the numeric arm going green while ZERO kernels ran -- which is exactly why
    the dispatch count is an acceptance conjunct and not a diagnostic.

    It asserts the numbers PASS and the counters read fallback. It is not a
    tolerance claim about the kernel and it is not the declared arm.
    """
    bank, _text_config = _build_bank()
    quant_config = _block_quant_config()
    case = _build_case()

    monkeypatch.setitem(os.environ, "NKI_SIMULATOR", "0")
    if can_run_kernel(torch.zeros(1)) is not False:
        raise VacuousControlError(
            "the gate did not flip with NKI_SIMULATOR=0, so this control is "
            "unarmed and its reading would say nothing"
        )

    reset_dispatch_counters()
    with _AttributedSimulatorCounter() as sim:
        got = bank.block_quant_expert_mm(
            quant_config=quant_config, block_size=B, **case["call_site_inputs"]
        )
    nki_dispatch, torch_fallback = dispatch_counters()
    want = torch_reference_moe(
        hidden_states=case["call_site_inputs"]["hidden_states"],
        expert_affinities=case["call_site_inputs"]["expert_affinities"],
        gate_up_proj_weight=case["call_site_inputs"]["gate_up_proj_weight"],
        down_proj_weight=case["call_site_inputs"]["down_proj_weight"],
        gate_up_logical_scale=case["gup_logical"],
        down_logical_scale=case["down_logical"],
    )
    got_f32 = got.to(torch.float32)
    error = _max_rel_error(got_f32, want)
    print(
        f"[f1-hazard] fallback_path seam_nki_dispatch={nki_dispatch} "
        f"seam_torch_fallback={torch_fallback} simulate_kernel_total={sim.total} "
        f"max_rel_error={error:.6e} rtol={RTOL} atol={ATOL}"
    )
    assert (nki_dispatch, torch_fallback) == (0, 1), (
        f"expected the fallback reading (0, 1), got "
        f"({nki_dispatch}, {torch_fallback}); this control cannot demonstrate "
        f"the hazard if the route did not actually change"
    )
    assert sim.total == 0, (
        f"the simulator ran {sim.total} times on the fallback path, so the "
        f"instrument is not measuring the route"
    )
    # THE POINT: the numbers are fine. Only the counters know the difference.
    torch.testing.assert_close(got_f32, want, rtol=RTOL, atol=ATOL)


def test_moe_path_route_control_fallback_counter_discriminates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The declared ``(1, 0)`` reading is a measurement, not an unwired counter.

    Same instruments, opposite world. ``_assert_route`` must REFUSE the fallback
    reading, so a passing acceptance cannot be a counter that always reads 1.
    """
    bank, _text_config = _build_bank()
    quant_config = _block_quant_config()
    case = _build_case()

    monkeypatch.setitem(os.environ, "NKI_SIMULATOR", "0")
    reset_dispatch_counters()
    with _AttributedSimulatorCounter() as sim:
        bank.block_quant_expert_mm(
            quant_config=quant_config, block_size=B, **case["call_site_inputs"]
        )
    print(
        f"[route-control] seam_counters={dispatch_counters()} "
        f"simulate_kernel_total={sim.total} "
        f"through_025_seam={sim.through_seam}"
    )
    # Measured in the SAME monkeypatched world, so the mapping's share is read
    # off the fallback path too rather than carried over from the real one.
    mapping_count = _measure_mapping_count(
        case["call_site_inputs"]["expert_affinities"], "route-control"
    )
    with pytest.raises(RouteInstrumentError):
        _assert_route(
            sim, DECLARED_DISPATCHES, "route-control", mapping_count=mapping_count
        )


def test_moe_path_attribution_control_reads_a_foreign_seam_as_elsewhere() -> None:
    """INSTRUMENT CONTROL for the attribution leg. NOT an acceptance conjunct.

    A dispatch through ``inc-glm53f-026``'s DENSE seam
    (``functional/blockwise_fp8_mm.py``) must read ``total == 1`` and
    ``through_seam == 0`` on a counter attributed to ``-025``'s seam. Without
    this arm, ``through_seam == total`` in the acceptance could be a frame walk
    that matches everything.

    This is emphatically NOT a dense-half dispatch conjunct: ``-026``'s seam is
    not on this call site's path, and ``inc-glm53f-033`` counts it downstream
    (plan L933). It appears here only as a known-foreign dispatch to arm the
    instrument.
    """
    from vllm_neuron.functional import blockwise_fp8_mm

    dense_m, dense_k, dense_n = 128, 256, 256
    activations = _fp8_grid_values(281, dense_m, dense_k).to(torch.bfloat16)
    weight = _fp8_grid_values(282, dense_k, dense_n).to(_FP8)
    weight_scale = torch.ones(
        dense_k // BLOCK_QUANT_SIZE, dense_n // BLOCK_QUANT_SIZE, dtype=torch.float32
    )

    with _AttributedSimulatorCounter() as sim:
        blockwise_fp8_mm(activations, weight, weight_scale)
    print(
        f"[attribution-control] simulate_kernel_total={sim.total} "
        f"through_025_seam={sim.through_seam} elsewhere={sim.elsewhere} "
        f"attributed_to={_SEAM_FILE}"
    )
    if sim.total == 0:
        raise VacuousControlError(
            "the foreign seam produced no simulator entry, so this control "
            "cannot show that the attribution discriminates"
        )
    assert sim.through_seam == 0, (
        f"{sim.through_seam} of {sim.total} entries produced by -026's dense "
        f"seam were attributed to -025's seam; the frame walk is matching "
        f"frames it should not, which would make the acceptance's "
        f"through_seam reading meaningless"
    )
    assert sim.elsewhere == sim.total


def test_moe_path_call_site_maps_before_padding_and_dispatches_nki() -> None:
    """MEASURED THROUGH THE CALL SITE: the mapping runs its NKI flow, not torch.

    This is finding ``B21-027``'s repair-round-2 demonstration. Two readings,
    both taken inside ONE real call-site call:

    1. The tensor the call site hands ``build_blockwise_mapping`` has ``T`` rows,
       not ``T + 1``. The padding slot is appended AFTER the mapping.
    2. During that mapping call the simulator was entered a NONZERO number of
       times, and none of those entries were attributed to ``-025``'s seam.

    Reading 2 is the point. Both of the mapping's kernel gates turn on the extent
    being even -- ``chunk_size % 128 == 0`` (``moe_blockwise.py:520``) and
    ``T % f_len == 0`` with ``f_len = min(128, T // 16)`` (``:536``) -- so with
    the padding row appended FIRST the extent was ``257``, both gates refused,
    and the mapping fell to ``_build_blockwise_mapping_torch`` on every call at
    every declared shape. That is a silent torch fallback for per-token device
    work the fork already ships NKI subkernels for (``moe_blockwise.py:10-11``),
    which P13 forbids. The reading below is the count, not the argument.

    The nested counter also completes the attribution
    :func:`_assert_route` requires: the outer total must equal the seam's
    attributed share plus the mapping's nested share, with nothing left over.

    D1.5: nothing here is a zero. The seam dispatches, the mapping dispatches,
    and both counts are printed, so no reading is a silent unwired instrument.
    """
    bank, _text_config = _build_bank()
    quant_config = _block_quant_config()
    case = _build_case()

    seen: list[torch.Tensor] = []
    nested: list[_AttributedSimulatorCounter] = []

    import vllm_neuron.functional as NF

    real_mapping = NF.build_blockwise_mapping

    def spy(expert_affinities, *args, **kwargs):
        seen.append(expert_affinities.detach().clone())
        # A NESTED counter, so the mapping's own share is read inside the real
        # call-site call rather than reconstructed from a separate run. Nesting
        # composes: this counter's "real" is the outer counter's wrapper, so the
        # outer total still sees every entry.
        with _AttributedSimulatorCounter() as inner:
            result = real_mapping(expert_affinities, *args, **kwargs)
        nested.append(inner)
        return result

    reset_dispatch_counters()
    NF.build_blockwise_mapping = spy
    try:
        with _AttributedSimulatorCounter() as outer:
            out = bank.block_quant_expert_mm(
                quant_config=quant_config, block_size=B, **case["call_site_inputs"]
            )
    finally:
        NF.build_blockwise_mapping = real_mapping

    if len(seen) != 1 or len(nested) != 1:
        raise VacuousControlError(
            f"the call site entered the mapping {len(seen)} times, not once; "
            f"every reading here assumes exactly one entry"
        )
    handed, inner = seen[0], nested[0]
    counters = dispatch_counters()
    print(
        f"[call-site-mapping] handed_shape={tuple(handed.shape)} "
        f"real_tokens={T} padded_tokens={T + 1} "
        f"mapping_simulate_kernel_total={inner.total} "
        f"mapping_through_025_seam={inner.through_seam} "
        f"mapping_elsewhere={inner.elsewhere} "
        f"outer_total={outer.total} outer_through_025_seam={outer.through_seam} "
        f"attribution_sum={outer.through_seam + inner.total} "
        f"seam_counters={counters} out_shape={tuple(out.shape)} "
        f"out_absmax={float(out.abs().max()):.6e}"
    )

    assert tuple(handed.shape) == (T, E), (
        f"the call site handed the mapping {tuple(handed.shape)}; it must hand "
        f"the REAL token count ({T}, {E}) and append the kernel's padding slot "
        f"afterwards. ({T + 1}, {E}) is the pad-first order finding B21-027 "
        f"reports, and it turns both of the mapping's NKI gates off."
    )
    assert inner.total > 0, (
        f"the mapping dispatched {inner.total} kernels inside a real call-site "
        f"call, so it took the torch fallback for per-token device work the "
        f"fork ships NKI subkernels for -- finding B21-027, P13"
    )
    assert inner.through_seam == 0, (
        f"{inner.through_seam} of the mapping's {inner.total} entries were "
        f"attributed to -025's seam, so the attribution sum double-counts"
    )
    assert outer.through_seam == DECLARED_DISPATCHES, (
        f"the seam was entered {outer.through_seam} times, declared "
        f"{DECLARED_DISPATCHES}"
    )
    assert outer.total == outer.through_seam + inner.total, (
        f"{outer.total} entries in the whole call do not add up: "
        f"{outer.through_seam} through the seam plus {inner.total} in the "
        f"mapping is {outer.through_seam + inner.total}. Some component nobody "
        f"measured dispatched."
    )
    assert counters == (1, 0), (
        f"seam counters {counters}: the mapping's route must not change how many "
        f"times the block-quant kernel is entered, nor take a fallback"
    )
    assert tuple(out.shape) == (T, H)
    assert bool(torch.isfinite(out).all())
    assert float(out.abs().max()) > 0.0


def test_moe_path_mapping_order_moves_no_numbers() -> None:
    """The two pad orders give the SAME tensors, so no declared value moves.

    Turning the mapping's NKI flow on would be a real contradiction if it changed
    what the seam consumes -- the numeric arm's declared
    ``assert_close(rtol=3e-2, atol=1e-5)`` is frozen and this increment has no
    authority over it. So the equality is MEASURED here, four readings:

    * ``token_position_to_id``, ``block_to_expert`` and ``conditions`` are
      element-wise equal between the pad-first (``T + 1``) and pad-after (``T``)
      orders.
    * appending ``E`` zero entries to the FLAT masked tensor the pad-after order
      returns is byte-identical to the pad-first order's masked tensor. The flat
      layout is token-major -- a ``view(-1, 1)`` of ``[T, E]``
      (``moe_blockwise.py:103-105``) -- so appending at the end of the flat form
      is the same tensor as appending a row before the view. Measured, not
      assumed.
    * the pad-first order dispatches ZERO kernels and the pad-after order
      dispatches a nonzero count. That contrast is the mechanism behind finding
      ``B21-027``, and it arms the zero: the same instrument reads nonzero in the
      same test.
    """
    case = _build_case()
    affinities = case["call_site_inputs"]["expert_affinities"]
    padded = torch.cat(
        [affinities, torch.zeros(1, E, dtype=affinities.dtype)], dim=0
    )

    pad_first = _run_mapping(padded, "pad-first-257")
    pad_after = _run_mapping(affinities, "pad-after-256")

    assert pad_first["total"] == 0, (
        f"the pad-first order dispatched {pad_first['total']} kernels; finding "
        f"B21-027's mechanism is that both NKI gates refuse at T + 1 = {T + 1}"
    )
    assert pad_after["total"] > 0, (
        f"the pad-after order dispatched {pad_after['total']} kernels, so the "
        f"repair did not turn the mapping's NKI flow on"
    )

    pad_flat = torch.zeros(E, 1, dtype=pad_after["masked"].dtype)
    repaired_masked = torch.cat([pad_after["masked"], pad_flat], dim=0)
    print(
        f"[mapping-order] flat_append_shape={tuple(repaired_masked.shape)} "
        f"row_append_shape={tuple(pad_first['masked'].shape)} "
        f"masked_byte_identical="
        f"{bool(torch.equal(repaired_masked, pad_first['masked']))}"
    )
    assert tuple(repaired_masked.shape) == tuple(pad_first["masked"].shape)
    assert torch.equal(repaired_masked, pad_first["masked"]), (
        "appending E zero entries to the flat masked tensor is not the same "
        "tensor as appending a zero row before the view, so the pad-order "
        "change would move what the seam consumes"
    )

    for key in ("token_position_to_id", "block_to_expert", "conditions"):
        left, right = pad_first[key], pad_after[key]
        equal = tuple(left.shape) == tuple(right.shape) and bool(
            torch.equal(left, right)
        )
        print(
            f"[mapping-order-{key}] pad_first_shape={tuple(left.shape)} "
            f"pad_after_shape={tuple(right.shape)} element_wise_equal={equal}"
        )
        assert equal, (
            f"{key} differs between the two pad orders, so turning the NKI flow "
            f"on changes what the seam consumes and the declared tolerance is "
            f"no longer the same measurement"
        )


@pytest.mark.parametrize("tokens", [T, 512, 1024, 2048])
def test_moe_path_mapping_dispatches_nki_across_the_token_envelope(
    tokens: int,
) -> None:
    """The mapping's NKI flow is on at every token count in the review's bar.

    ``T`` is the fixture's own count; ``512``, ``1024`` and ``2048`` are the bar
    the implementation review set, so a nonzero reading at one convenient extent
    cannot stand in for the envelope. Each count is read twice -- the real count
    and the same count plus the padding row -- so the reading is a CONTRAST and
    not one number in isolation.

    The affinity pattern is this file's own fixture scatter at a different token
    count, not a plan-declared shape; the plan declares nothing about extents
    above ``T``.
    """
    real = _envelope_affinities(tokens)
    padded = torch.cat([real, torch.zeros(1, E, dtype=real.dtype)], dim=0)

    after = _run_mapping(real, f"envelope-{tokens}-real")
    first = _run_mapping(padded, f"envelope-{tokens}-padded")

    assert after["total"] > 0, (
        f"at {tokens} real tokens the mapping dispatched {after['total']} "
        f"kernels, so its NKI flow is off at this extent"
    )
    assert first["total"] == 0, (
        f"at {tokens} + 1 tokens the mapping dispatched {first['total']} "
        f"kernels; the contrast that makes the reading above meaningful is that "
        f"the pad-first extent refuses both gates"
    )


# ===========================================================================
# The route selector. B.6: no enum member, a named refusal instead.
# ===========================================================================
def test_moe_path_unquantised_config_raises_by_name() -> None:
    """An unquantised config must RAISE, not reach the substrate's NONE default.

    ``functional/mlp.py:81`` and ``:249`` default ``QuantizationType.NONE`` by
    OMISSION, so a call site that forgot to route gets it with no error at all
    and computes a different function while every shape check passes. The
    acceptance's counter clause DETECTS that; this refusal makes it impossible.
    """
    from vllm_neuron.model.glm5_next.model_fp8 import (
        Glm5NextBlockQuantRouteError,
        Glm5NextQuantConfig,
    )

    bank, _text_config = _build_bank()
    case = _build_case()
    unquantised = Glm5NextQuantConfig(None)
    if unquantised.is_block_quantized:
        raise VacuousControlError(
            "Glm5NextQuantConfig(None) reports is_block_quantized=True, so this "
            "control's input is not the unquantised world it claims to be"
        )
    reset_dispatch_counters()
    with pytest.raises(Glm5NextBlockQuantRouteError, match="block-quant route"):
        bank.block_quant_expert_mm(
            quant_config=unquantised, block_size=B, **case["call_site_inputs"]
        )
    assert dispatch_counters() == (0, 0), (
        f"the refusal must fire BEFORE any dispatch, read "
        f"{dispatch_counters()}"
    )


def test_moe_path_foreign_checkpoint_block_shape_raises_by_name() -> None:
    """A checkpoint block shape the retile does not bridge must RAISE.

    The scales this route consumes are retiled FROM ``(128, 128)`` checkpoint
    blocks. Any other declared shape means the fixture the kernel would receive
    was never built, and continuing would feed the kernel scales that tile a
    different weight.
    """
    from vllm_neuron.model.glm5_next.model_fp8 import Glm5NextBlockQuantRouteError

    bank, _text_config = _build_bank()
    case = _build_case()
    real = _block_quant_config()
    if tuple(real.block_shape) != (TILE_SIZE, TILE_SIZE):
        raise VacuousControlError(
            f"the pinned checkpoint declares block_shape={real.block_shape}, "
            f"not ({TILE_SIZE}, {TILE_SIZE}); this control's premise is stale"
        )

    class _ForeignShapeMethod:
        block_shape = (64, 64)

    class _ForeignShapeConfig:
        is_block_quantized = True
        method = _ForeignShapeMethod()
        block_shape = (64, 64)

    reset_dispatch_counters()
    with pytest.raises(Glm5NextBlockQuantRouteError, match="weight_block_size"):
        bank.block_quant_expert_mm(
            quant_config=_ForeignShapeConfig(),
            block_size=B,
            **case["call_site_inputs"],
        )
    assert dispatch_counters() == (0, 0)


def _imported_names(source: str, filename: str) -> set[str]:
    """Every name an ``import`` statement BINDS in ``source``, by AST.

    An AST walk rather than a text scan, because the property B.6 constrains is
    *does this package take a dependency on the substrate's quantisation enum*,
    and the three places ``model_fp8.py`` mentions the enum by name are a
    comment, an error message and a docstring -- all of them DOCUMENTING the
    negative. A text scan would flag the documentation of the rule as a
    violation of it.
    """
    import ast

    bound: set[str] = set()
    tree = ast.parse(source, filename=filename)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add(alias.asname or alias.name.split(".")[0])
    return bound


def test_moe_path_package_binds_no_quantisation_enum(tmp_path: Path) -> None:
    """B.6, mechanically: no module of this arch package imports ``QuantizationType``.

    The route is selected by WHICH FUNCTION IS CALLED, never by a new enum
    member. The enum lives in the substrate, so adding a member means importing
    it; a package that never binds the name cannot have extended it. Every
    ``.py`` in the arch package is walked, and the item count is reported.
    """
    from vllm_neuron.model.glm5_next import model_fp8

    package_dir = Path(model_fp8.__file__).resolve().parent
    modules = sorted(package_dir.glob("*.py"))
    if not modules:
        raise VacuousControlError(
            f"no modules found under {package_dir}, so this scan read nothing"
        )
    offenders = {}
    for module_path in modules:
        bound = _imported_names(
            module_path.read_text(), filename=str(module_path)
        )
        if "QuantizationType" in bound:
            offenders[module_path.name] = sorted(bound & {"QuantizationType"})
    print(f"[b6] modules_scanned={len(modules)} offenders={offenders}")
    assert offenders == {}, (
        f"a module of this arch package binds the substrate's QuantizationType, "
        f"which is the precondition for adding a member to it: {offenders}"
    )

    # FIRING CONTROL. The same collector, over source that DOES bind the name.
    control = tmp_path / "control_binds_the_enum.py"
    control.write_text(
        "from vllm_neuron.functional.quantization import QuantizationType\n"
    )
    control_bound = _imported_names(control.read_text(), filename=str(control))
    if "QuantizationType" not in control_bound:
        raise VacuousControlError(
            f"the collector did not find QuantizationType in source that "
            f"imports it (found {sorted(control_bound)}), so the empty offender "
            f"set above is not a measurement"
        )


# ===========================================================================
# The export hub. This increment is the plan's only declared writer of both
# ``functional/__init__.py`` and ``functional/moe/__init__.py``.
# ===========================================================================
def test_moe_path_exports_resolve_to_their_own_modules() -> None:
    """Both WP6 seams resolve through the hub, and are the module's own objects."""
    import vllm_neuron.functional as functional
    import vllm_neuron.functional.moe as functional_moe
    from vllm_neuron.functional.blockwise_fp8_mm import (
        blockwise_fp8_mm as dense_seam,
    )
    from vllm_neuron.functional.moe.blockwise_fp8_retile import (
        retile_block_scales as retile_producer,
    )

    if functional.blockwise_fp8_moe is not blockwise_fp8_moe:
        raise ExportSurfaceError(
            "vllm_neuron.functional.blockwise_fp8_moe is not -025's seam object"
        )
    if functional.blockwise_fp8_mm is not dense_seam:
        raise ExportSurfaceError(
            "vllm_neuron.functional.blockwise_fp8_mm is not -026's seam object"
        )
    if functional_moe.blockwise_fp8_moe is not blockwise_fp8_moe:
        raise ExportSurfaceError(
            "vllm_neuron.functional.moe.blockwise_fp8_moe is not -025's seam"
        )
    if functional_moe.retile_block_scales is not retile_producer:
        raise ExportSurfaceError(
            "vllm_neuron.functional.moe.retile_block_scales is not -024's producer"
        )
    for name in ("blockwise_fp8_mm", "blockwise_fp8_moe"):
        if name not in functional.__all__:
            raise ExportSurfaceError(f"{name!r} missing from functional.__all__")
    print(
        f"[exports] functional.__all__={len(functional.__all__)} entries; "
        f"functional.moe.__all__={sorted(functional_moe.__all__)}"
    )
    assert functional.__all__ == sorted(functional.__all__), (
        "functional.__all__ is no longer alphabetical, which is the convention "
        "the file itself declares"
    )
    assert functional_moe.__all__ == sorted(functional_moe.__all__)


def test_moe_path_colliding_helper_names_are_not_flat_exported() -> None:
    """The name collisions stay QUALIFIED, and the collision is real.

    Three WP6 modules define same-named helpers at DIFFERENT signatures. A flat
    re-export would resolve to whichever module imported last and the arity
    mismatch would surface as a shape error far from its cause. The hub
    deliberately exports neither, and this arm measures that the collision it
    guards against actually exists.
    """
    import vllm_neuron.functional as functional
    import vllm_neuron.functional.moe as functional_moe
    from vllm_neuron.functional.blockwise_fp8_mm import (
        to_kernel_scale_layout as dense_bridge,
    )

    # The collision is REAL: same name, different arity, both live.
    dense_arity = dense_bridge.__code__.co_argcount
    moe_arity = moe_to_kernel_scale_layout.__code__.co_argcount
    print(
        f"[collision] to_kernel_scale_layout dense_argcount={dense_arity} "
        f"moe_argcount={moe_arity}"
    )
    if dense_bridge is moe_to_kernel_scale_layout or dense_arity == moe_arity:
        raise VacuousControlError(
            f"the two to_kernel_scale_layout helpers are the same object or "
            f"share an arity ({dense_arity} vs {moe_arity}), so there is no "
            f"collision for the hub to guard against and this arm asserts a "
            f"property of nothing"
        )
    for name in (
        "to_kernel_scale_layout",
        "flat_scale_index",
        "kernel_scale_shape",
        "dispatch_counters",
        "reset_dispatch_counters",
        "kernel_identity",
        "BLOCK_QUANT_SIZE",
        "TILE_SIZE",
    ):
        for hub, label in (
            (functional, "vllm_neuron.functional"),
            (functional_moe, "vllm_neuron.functional.moe"),
        ):
            if name in getattr(hub, "__all__", ()):
                raise ExportSurfaceError(
                    f"{label}.__all__ flat-exports the colliding name {name!r}"
                )
            if hasattr(hub, name):
                raise ExportSurfaceError(
                    f"{label} has attribute {name!r}, which resolves one of two "
                    f"different-arity definitions by import order"
                )


# ===========================================================================
# Preconditions and fixture conditioning.
# ===========================================================================
def test_moe_path_f1_precondition_complete_condition_n_over_n() -> None:
    """Both losslessness conjuncts hold on THIS case's scales, N/N blocks.

    Conjunct 1: the constituent ``[128, 128]`` scales are mutually
    power-of-two-related. Conjunct 2: the retained ``256``-block scale is itself
    a power of two. Both by the BIT-PATTERN predicate ``is_pow2_exact``, never
    ``log2``. Recomputed from the producer's emitted records rather than read
    off its own ``lossless`` flag, so this is an independent instrument.

    The declared tolerance certifies KERNEL error only when both conjuncts hold;
    if they did not, the comparison would be measuring the retile's remapping
    loss and the tolerance would be answering the wrong question.
    """
    case = _build_case()
    banks = [
        (f"gate_up[g={index}]", result)
        for index, result in enumerate(case["gup_results"])
    ] + [("down", case["down_result"])]

    total = 0
    lossless = 0
    for label, result in banks:
        if not result.records:
            raise VacuousControlError(
                f"{label}: the producer emitted 0 block records, so an N/N "
                f"reading would be 0/0 -- vacuous"
            )
        for record in result.records:
            total += 1
            conjunct1 = all(is_pow2_exact(ratio) for ratio in record.ratios)
            conjunct2 = is_pow2_exact(record.block_scale)
            if conjunct1 and conjunct2:
                lossless += 1
            else:
                raise F1PreconditionError(
                    f"{label} block {record.key}: conjunct1={conjunct1} "
                    f"ratios={record.ratios} conjunct2={conjunct2} "
                    f"block_scale={record.block_scale!r}"
                )
        assert result.inexact_rescales == 0, (
            f"{label}: inexact_rescales={result.inexact_rescales}, declared 0"
        )
        assert result.input_scales_dropped == 0, (
            f"{label}: input_scales_dropped={result.input_scales_dropped}, "
            f"declared 0"
        )
    print(f"[f1] complete_condition_blocks={lossless}/{total} (N/N required)")
    assert total > 0
    assert lossless == total


def test_moe_path_fixture_conditioning_is_measured_not_assumed() -> None:
    """The conditioning the declared tolerance is measured against, asserted.

    Four properties, each a number in the transcript: the weights and hidden
    states are unsigned and on the fp8 grid; every fixture value round-trips
    through fp8-e4m3 and bf16 EXACTLY; the affinities are dyadic and bf16-exact;
    and the fixture is not degenerate -- the scales are distinct and every
    expert is occupied.
    """
    case = _build_case()
    inputs = case["call_site_inputs"]
    hidden = inputs["hidden_states"]
    affinities = inputs["expert_affinities"]
    gup = inputs["gate_up_proj_weight"]
    down = inputs["down_proj_weight"]

    hidden_min = float(hidden.to(torch.float32).min())
    gup_min = float(gup.to(torch.float32).min())
    down_min = float(down.to(torch.float32).min())
    affinity_nonzeros = int((affinities != 0).sum())
    per_expert = (affinities != 0).sum(dim=0).tolist()
    distinct_gup_scales = int(torch.unique(case["gup_logical"]).numel())
    distinct_down_scales = int(torch.unique(case["down_logical"]).numel())
    affinity_roundtrip = float(
        (
            affinities - affinities.to(torch.bfloat16).to(torch.float32)
        ).abs().max()
    )
    hidden_roundtrip = float(
        (
            hidden.to(torch.float32)
            - hidden.to(torch.float32).to(_FP8).to(torch.float32)
        ).abs().max()
    )
    print(
        f"[conditioning] hidden_min={hidden_min} gate_up_min={gup_min} "
        f"down_min={down_min} affinity_nonzeros={affinity_nonzeros} "
        f"tokens_per_expert={per_expert} "
        f"distinct_gate_up_scales={distinct_gup_scales} "
        f"distinct_down_scales={distinct_down_scales} "
        f"affinity_bf16_roundtrip_error={affinity_roundtrip} "
        f"hidden_fp8_roundtrip_error={hidden_roundtrip}"
    )
    if hidden_min <= 0.0 or gup_min <= 0.0 or down_min <= 0.0:
        raise FixtureConditioningError(
            f"the fixture contains non-positive values (hidden_min={hidden_min}, "
            f"gate_up_min={gup_min}, down_min={down_min}); the declared "
            f"pointwise rtol={RTOL} is then dominated by catastrophic "
            f"cancellation over the H={H} contraction rather than by kernel "
            f"error, which is how -025 attempt 1 failed"
        )
    if affinity_roundtrip != 0.0 or hidden_roundtrip != 0.0:
        raise FixtureConditioningError(
            f"a fixture value does not round-trip exactly "
            f"(affinity={affinity_roundtrip}, hidden={hidden_roundtrip}); the "
            f"tolerance would be absorbing fixture cast error"
        )
    if per_expert != [TOKENS_PER_EXPERT] * E:
        raise FixtureConditioningError(
            f"expert occupancy is {per_expert}, declared "
            f"{[TOKENS_PER_EXPERT] * E}; an empty expert would make its share "
            f"of the comparison vacuous"
        )
    if distinct_gup_scales < 2 or distinct_down_scales < 2:
        raise FixtureConditioningError(
            f"the scales are effectively uniform (gate_up distinct="
            f"{distinct_gup_scales}, down distinct={distinct_down_scales}); a "
            f"comparison on uniform scales cannot observe a permuted layout"
        )


def test_moe_path_mapping_shape_is_the_one_the_seam_consumes() -> None:
    """The mapping the call site builds has the extents the seam declares.

    Read as numbers rather than derived in prose: ``N`` blocks of ``B`` tokens,
    ``block_to_expert`` covering every local expert, and at least one EMPTY
    block -- the case where the kernel indexes the padding slot, which this
    tiny config reaches on purpose.
    """
    from vllm_neuron.functional import build_blockwise_mapping

    case = _build_case()
    padded_affinities = torch.cat(
        [
            case["call_site_inputs"]["expert_affinities"],
            torch.zeros(1, E, dtype=torch.float32),
        ],
        dim=0,
    )
    masked, token_position_to_id, block_to_expert, conditions = (
        build_blockwise_mapping(
            expert_affinities=padded_affinities,
            num_local_experts=E,
            num_experts_per_token=K,
            block_size=B,
            moe_group=None,
            tp_degree=1,
        )
    )
    num_blocks = int(block_to_expert.numel())
    empty_blocks = int((conditions == 0).sum())
    covered = sorted({int(value) for value in block_to_expert.tolist()})
    print(
        f"[mapping-shape] num_blocks={num_blocks} block_size={B} "
        f"token_position_to_id={tuple(token_position_to_id.shape)} "
        f"masked={tuple(masked.shape)} experts_covered={covered} "
        f"empty_blocks={empty_blocks}"
    )
    assert tuple(masked.shape) == ((T + 1) * E, 1)
    assert tuple(token_position_to_id.shape) == (num_blocks * B,)
    assert token_position_to_id.dtype is torch.int32
    assert block_to_expert.dtype is torch.int32
    assert covered == list(range(E)), (
        f"block_to_expert covers experts {covered}, not every local expert; "
        f"an uncovered expert's weights would never be read"
    )
    assert empty_blocks >= 1, (
        "no empty block in this configuration, so the padding-slot path the "
        "call site's appended row exists for is not exercised"
    )


def test_moe_path_kernel_identity_is_the_d5b_inner_kernel() -> None:
    """D5(b), read off the object: the INNER kernel, not the ``moe_cte`` dispatcher.

    The plan's D5(b) decision is to bypass the public dispatcher because it will
    not forward block scales. That the seam this call site enters targets the
    inner member is read from ``kernel_identity()`` rather than argued, so a
    substitution shows up as a changed reading instead of as silence.
    """
    module, qualname = kernel_identity()
    print(f"[kernel-identity] module={module!r} qualname={qualname!r}")
    assert "bwmm_shard_on_I" in module, (
        f"the seam's kernel lives in {module!r}, not the D5(b) inner-kernel "
        f"module"
    )
    assert qualname == "blockwise_mm_baseline_shard_intermediate", (
        f"the seam dispatches to {qualname!r}, not the declared inner kernel"
    )
    assert "moe_cte_dispatch" not in qualname


def test_moe_path_block_size_must_be_block_quant_granular() -> None:
    """A ``block_size`` the vendor kernel's ``B % 256 == 0`` assert refuses."""
    from vllm_neuron.model.glm5_next.model_fp8 import Glm5NextBlockQuantRouteError

    bank, _text_config = _build_bank()
    quant_config = _block_quant_config()
    case = _build_case()
    reset_dispatch_counters()
    with pytest.raises(Glm5NextBlockQuantRouteError, match="BLOCK_QUANT_SIZE"):
        bank.block_quant_expert_mm(
            quant_config=quant_config,
            block_size=BLOCK_QUANT_SIZE + 1,
            **case["call_site_inputs"],
        )
    assert dispatch_counters() == (0, 0)


def test_moe_path_expert_count_disagreement_raises_by_name() -> None:
    """A weight bank that disagrees with the partition must RAISE.

    The bank's ``num_local_experts`` and the weight tensor's leading extent are
    two independent facts, and a mismatch means this rank would compute other
    ranks' experts. Shape checks downstream would not catch it: the kernel reads
    ``E`` off the weight, so it would run happily on the wrong partition.
    """
    from vllm_neuron.model.glm5_next.model_fp8 import Glm5NextBlockQuantRouteError

    bank, _text_config = _build_bank()
    quant_config = _block_quant_config()
    case = _build_case()
    inputs = dict(case["call_site_inputs"])
    inputs["gate_up_proj_weight"] = inputs["gate_up_proj_weight"][: E - 1]
    reset_dispatch_counters()
    with pytest.raises(Glm5NextBlockQuantRouteError, match="experts"):
        bank.block_quant_expert_mm(
            quant_config=quant_config, block_size=B, **inputs
        )
    assert dispatch_counters() == (0, 0)


def test_moe_path_landed_sections_are_untouched() -> None:
    """``-031``'s, ``-032``'s and ``-013``'s members still resolve, unchanged.

    This increment's D14 section is a PURE INSERTION into a coordinated merge
    point. The three landed members whose acceptances are already recorded must
    still be present and still behave: a refactor that "tidied" them would
    invalidate evidence this increment has no authority over.
    """
    from vllm_neuron.model.glm5_next.model_fp8 import Glm5NextRoutedExperts

    bank, text_config = _build_bank()
    assert callable(Glm5NextRoutedExperts.route_tokens)
    assert callable(Glm5NextRoutedExperts.local_expert_indices)
    assert bank.local_expert_indices(0) == tuple(range(E))
    assert int(bank.num_routed_experts) == E
    assert int(bank.num_experts_per_tok) == K
    assert int(text_config.n_routed_experts) == E
    with pytest.raises(NotImplementedError, match="inc-glm53f-013"):
        bank.forward()
    print(
        f"[landed] num_routed_experts={bank.num_routed_experts} "
        f"num_local_experts={bank.num_local_experts} "
        f"local_expert_indices(0)={bank.local_expert_indices(0)}"
    )


# ===========================================================================
# The DISPATCH step: global router columns -> this rank's slice.
# Repair batch R5, finding ``B21-027-router-affinities-are-global-not-local``.
# ===========================================================================
#
# WHY THIS SECTION EXISTS. Every arm above builds the bank through
# :func:`_build_bank`, which passes ``world_size=1``. At degree 1 the global and
# the local expert counts COINCIDE at ``E``, so that is the one configuration in
# which the call site's affinity contract cannot be wrong -- and no arm above can
# see whether the site consumes the router's global ``[T, E]`` or this rank's
# local ``[T, E_local]``. These two arms run at a degree ABOVE 1, where the two
# shapes differ and the question has an answer.
#
# NOTHING HERE IS HAND-CHOSEN. The global expert width and the router's ``top_k``
# are the PINNED CHECKPOINT's own values, read off the same digest-verified
# fixture the rest of this file routes on, and the degree is derived from them.


#: This rank's local expert extent, at the degree below. ``E`` rather than a new
#: number, because ``E`` is the extent :func:`_build_case` already builds weights
#: and scales for, and the dispatch is a SHAPE question that must not be answered
#: by rebuilding the fixture around it.
EP_LOCAL_EXPERTS = E

#: Fixture-only seed for this section's router inputs. This file's own choice.
EP_ROUTER_SEED = 9027


def _ep_partition() -> tuple[int, int, int]:
    """``(global experts, expert-parallel degree, router top_k)``, all measured.

    The global width and ``top_k`` are READ off the pinned config rather than
    written here, so this arm always runs at the router the campaign registered.
    The pinned values are 288 experts at ``top_k=8``; the degree is then derived
    as the one that makes this rank own exactly :data:`EP_LOCAL_EXPERTS`.

    The fixture's own ``K=2`` cannot drive this section at all. The router seam
    admits exactly one ``top_k`` and refuses every other by name::

        NoauxTcRouterError: top_k must be exactly 8: `nisa.max8` emits 8 values
        per partition and `nisa.nc_find_index8` consumes exactly 8, and nkilib
        refuses k > 8 (router_topk.py:582-583). got top_k=2

    That reading is why ``top_k`` is taken from the checkpoint here instead of
    from ``K``, and the checkpoint's value happens to be the seam's only
    admissible one.
    """
    text = _pinned_raw_config()["text_config"]
    global_experts = int(text["n_routed_experts"])
    top_k = int(text["num_experts_per_tok"])
    if global_experts % EP_LOCAL_EXPERTS:
        raise VacuousControlError(
            f"the pinned config's n_routed_experts={global_experts} is not "
            f"divisible by the fixture's local extent {EP_LOCAL_EXPERTS}, so no "
            f"uniform degree gives this rank the weights _build_case builds"
        )
    degree = global_experts // EP_LOCAL_EXPERTS
    if degree <= 1:
        raise VacuousControlError(
            f"the derived expert-parallel degree is {degree}; this section's "
            f"whole subject is a degree ABOVE 1 and cannot be tested at 1"
        )
    return global_experts, degree, top_k


def _build_ep_bank():
    """A bank at a degree above 1 whose LOCAL extent is the fixture's ``E``.

    Returns ``(bank, text_config, global_experts, degree)``. The campaign's
    registered TP freeze of 64 is deliberately not used: 288 experts do not
    divide 64 ways, so that bank raises during construction. That refusal is
    campaign gap G4 and the lead's to dispose; this arm needs a degree the
    partition admits, and derives one.
    """
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig
    from vllm_neuron.model.glm5_next.model_fp8 import Glm5NextRoutedExperts

    global_experts, degree, top_k = _ep_partition()
    text_config = Glm5NextTextConfig(
        hidden_size=H,
        moe_intermediate_size=I_TP,
        n_routed_experts=global_experts,
        num_experts_per_tok=top_k,
    )
    bank = Glm5NextRoutedExperts(text_config, world_size=degree)
    if int(bank.num_local_experts) != EP_LOCAL_EXPERTS:
        raise VacuousControlError(
            f"the bank reports num_local_experts={bank.num_local_experts} at "
            f"degree {degree}, but this section's fixture is built for "
            f"{EP_LOCAL_EXPERTS}; the weight extent check would fire before any "
            f"affinity was looked at"
        )
    if int(bank.num_routed_experts) == int(bank.num_local_experts):
        raise VacuousControlError(
            "the global and local expert counts coincide, so this bank is the "
            "degree-1 case again and the dispatch step would be a no-op"
        )
    # ``-031``'s partition, checked as a PARTITION before this arm indexes with
    # it. If it did not cover every global column exactly once, the per-rank
    # column comparisons below would be against the wrong reference.
    seen: list[int] = []
    for rank in range(degree):
        seen.extend(bank.local_expert_indices(rank))
    if sorted(seen) != list(range(global_experts)):
        raise VacuousControlError(
            f"local_expert_indices over {degree} ranks does not cover "
            f"0..{global_experts - 1} exactly once, so the columns this arm "
            f"compares against are not a partition"
        )
    return bank, text_config, global_experts, degree


def _ep_route(bank, global_experts: int, text_config):
    """Run ``route_tokens`` FOR REAL and return its own output tensors.

    The affinity tensor this returns is handed to the call site unchanged; that
    is what finding ``B21-027`` means by "no slicing done by the caller". The
    router is executed rather than imitated, so the form the call site consumes
    is the form the producer actually emits and not this file's idea of it.
    """
    gen = torch.Generator().manual_seed(EP_ROUTER_SEED)
    hidden_bsh = (torch.randn(1, T, H, generator=gen) * 0.5).to(torch.bfloat16)
    gamma = torch.ones(1, H, dtype=torch.bfloat16)
    bank.router_weight = torch.nn.Parameter(
        (torch.randn(H, global_experts, generator=gen) * 0.1).to(torch.bfloat16),
        requires_grad=False,
    )
    bank.router_bias = torch.nn.Parameter(
        (torch.randn(global_experts, generator=gen) * 0.05).to(torch.bfloat16),
        requires_grad=False,
    )
    _logits, expert_index, affinities = bank.route_tokens(
        hidden_bsh, gamma, text_config
    )
    if tuple(affinities.shape) != (T, global_experts):
        raise VacuousControlError(
            f"route_tokens returned {tuple(affinities.shape)}; this section's "
            f"whole subject is its GLOBAL [T={T}, E={global_experts}] form"
        )
    if int((affinities != 0).sum()) == 0:
        raise VacuousControlError(
            "route_tokens returned an all-zero affinity tensor, so every "
            "reading below would hold for the wrong reason"
        )
    return affinities, expert_index


class _MappingAffinitySpy:
    """Captures the affinity tensor ``build_blockwise_mapping`` is handed.

    The call site imports the mapping FUNCTION-LOCALLY from
    ``vllm_neuron.functional`` (``model_fp8.py:1060``), so replacing the
    attribute on that module is what a call actually resolves. The real mapping
    still runs and its result is still used, so the kernel below is measured on
    the real path rather than on a stub.
    """

    def __init__(self) -> None:
        self.seen: list[torch.Tensor] = []
        self._module = None
        self._real = None

    def __enter__(self) -> "_MappingAffinitySpy":
        import vllm_neuron.functional as NF

        self._module = NF
        self._real = NF.build_blockwise_mapping
        real = self._real

        def spy(expert_affinities, *args, **kwargs):
            self.seen.append(expert_affinities.detach().clone())
            return real(expert_affinities, *args, **kwargs)

        NF.build_blockwise_mapping = spy
        return self

    def __exit__(self, *_exc) -> bool:
        self._module.build_blockwise_mapping = self._real
        return False

    @property
    def only(self) -> torch.Tensor:
        if len(self.seen) != 1:
            raise VacuousControlError(
                f"the mapping was entered {len(self.seen)} times, not once; "
                f"every per-rank reading here assumes exactly one entry"
            )
        return self.seen[0]


def test_moe_path_dispatch_maps_global_router_output_at_degree_above_one() -> None:
    """``route_tokens``' GLOBAL output reaches the kernel as ``[T, E_local]``.

    This is the demonstration finding ``B21-027`` requires: one case at an
    expert-parallel degree above 1 in which ``route_tokens``' output reaches the
    kernel as ``[T, E_local]`` with no slicing done by the caller. What the
    caller hands in carries the GLOBAL width; what the mapping seam receives is
    read out of the seam itself rather than inferred.

    Three ranks are probed rather than one, chosen by MEASURED occupancy, so a
    gather that ignored its rank argument would have to produce the same slice
    three times over disjoint column sets to survive.
    """
    bank, text_config, global_experts, degree = _build_ep_bank()
    quant_config = _block_quant_config()
    case = _build_case()
    affinities, expert_index = _ep_route(bank, global_experts, text_config)
    print(
        f"[dispatch] degree={degree} global_experts={global_experts} "
        f"local_experts={bank.num_local_experts} "
        f"top_k={bank.num_experts_per_tok} "
        f"route_affinities_shape={tuple(affinities.shape)} "
        f"route_affinities_dtype={affinities.dtype} "
        f"expert_index_shape={tuple(expert_index.shape)} "
        f"global_nonzeros={int((affinities != 0).sum())}"
    )

    # Rank choice is MEASURED: the three ranks carrying the most routing mass,
    # so no probed slice is all zero and no two are trivially equal.
    occupancy = {}
    for rank in range(degree):
        cols = torch.tensor(bank.local_expert_indices(rank), dtype=torch.int64)
        occupancy[rank] = int((affinities[:, cols] != 0).sum())
    probed = sorted(occupancy, key=lambda r: (-occupancy[r], r))[:3]
    zero_ranks = sum(1 for value in occupancy.values() if value == 0)
    print(
        f"[dispatch] probed_ranks={probed} "
        f"probed_occupancies={[occupancy[r] for r in probed]} "
        f"zero_occupancy_ranks={zero_ranks}/{degree}"
    )
    assert len(probed) == 3, (
        f"only {len(probed)} ranks exist at degree {degree}; the rank-liveness "
        f"reading below needs three distinct column sets"
    )
    assert min(occupancy[r] for r in probed) > 0, (
        "a probed rank owns no routed tokens at all, so its slice is all zero "
        "and the gather reading below would hold for the wrong reason"
    )

    gathered: dict[int, torch.Tensor] = {}
    for rank in probed:
        inputs = dict(case["call_site_inputs"])
        inputs["expert_affinities"] = affinities
        # The no-slicing claim, stated as a reading rather than an assumption:
        # what goes in is the GLOBAL width, which is not this rank's extent.
        assert tuple(inputs["expert_affinities"].shape) == (T, global_experts)
        assert global_experts != EP_LOCAL_EXPERTS
        reset_dispatch_counters()
        with _MappingAffinitySpy() as spy:
            out = bank.block_quant_expert_mm(
                quant_config=quant_config,
                block_size=B,
                expert_parallel_rank=rank,
                **inputs,
            )
        seam_affinities = spy.only
        counters = dispatch_counters()
        columns = torch.tensor(
            bank.local_expert_indices(rank), dtype=torch.int64
        )
        # The expectation is built by plain advanced indexing, which is a
        # DIFFERENT mechanism from the ``torch.gather`` the call site uses, so
        # this is a comparison and not a restatement.
        want = affinities[:, columns]
        got = seam_affinities
        print(
            f"[dispatch-rank] rank={rank} columns={tuple(columns.tolist())} "
            f"seam_affinity_shape={tuple(seam_affinities.shape)} "
            f"gathered_equals_columns={bool(torch.equal(got, want))} "
            f"gathered_nonzeros={int((got != 0).sum())} "
            f"out_shape={tuple(out.shape)} "
            f"out_absmax={float(out.abs().max()):.6e} "
            f"seam_counters={counters}"
        )
        # ``[T, E_local]`` with NO padding row: repair round 2 moved the padding
        # slot to AFTER the mapping, because both of the mapping's NKI gates
        # refuse an odd extent (finding ``B21-027``). So this reading is also the
        # pad-order reading, at degree ``{degree}`` rather than at degree 1 --
        # revert the order in the call site and this assertion reads ``T + 1``.
        assert tuple(seam_affinities.shape) == (T, EP_LOCAL_EXPERTS), (
            f"the mapping was handed {tuple(seam_affinities.shape)}; the whole "
            f"point of the dispatch step is that it sees "
            f"[T={T}, E_local={EP_LOCAL_EXPERTS}], and the padding slot is "
            f"appended after the mapping, not before it"
        )
        assert torch.equal(got, want), (
            f"rank {rank}'s gathered slice is not its own global columns "
            f"{tuple(columns.tolist())}"
        )
        assert tuple(out.shape) == (T, H)
        assert bool(torch.isfinite(out).all())
        assert float(out.abs().max()) > 0.0, (
            "the kernel returned an all-zero output, so nothing downstream of "
            "the dispatch actually computed"
        )
        assert counters == (1, 0), (
            f"seam counters {counters}: the dispatch must not change how many "
            f"times the block-quant kernel is entered, nor take a fallback"
        )
        gathered[rank] = got

    # The rank argument is LIVE: disjoint column sets give different slices.
    for index, left in enumerate(probed):
        for right in probed[index + 1 :]:
            assert not torch.equal(gathered[left], gathered[right]), (
                f"ranks {left} and {right} received the same slice over "
                f"disjoint columns, so the gather is not reading its rank"
            )

    # The default rank is documented as 0, and it is the same call as an
    # explicit 0. Read rather than asserted from the signature.
    inputs = dict(case["call_site_inputs"])
    inputs["expert_affinities"] = affinities
    with _MappingAffinitySpy() as spy_default:
        bank.block_quant_expert_mm(
            quant_config=quant_config, block_size=B, **inputs
        )
    with _MappingAffinitySpy() as spy_zero:
        bank.block_quant_expert_mm(
            quant_config=quant_config,
            block_size=B,
            expert_parallel_rank=0,
            **inputs,
        )
    same = bool(torch.equal(spy_default.only, spy_zero.only))
    print(f"[dispatch-default] default_rank_equals_explicit_zero={same}")
    assert same, "the default expert_parallel_rank is documented as rank 0"


def test_moe_path_dispatch_refuses_a_caller_sliced_local_form_above_degree_one() -> None:
    """Above degree 1 the call site REFUSES a caller-sliced ``[T, E_local]``.

    "With no slicing done by the caller" is a two-sided claim and this is the
    other side. A caller that does the slice itself is the outcome
    ``inc-glm53f-032``'s landed note exists to prevent -- two owners for one
    behaviour -- and it now fails by name instead of quietly computing on a
    slice this site did not make.

    At degree 1 the local and global widths coincide, so this refusal cannot
    fire there and no landed arm is affected by it.
    """
    from vllm_neuron.model.glm5_next.model_fp8 import Glm5NextBlockQuantRouteError

    bank, text_config, global_experts, degree = _build_ep_bank()
    quant_config = _block_quant_config()
    case = _build_case()
    affinities, _expert_index = _ep_route(bank, global_experts, text_config)
    columns = torch.tensor(bank.local_expert_indices(1), dtype=torch.int64)
    pre_sliced = affinities[:, columns]
    assert tuple(pre_sliced.shape) == (T, EP_LOCAL_EXPERTS), (
        "the control must hand in exactly the local form a slicing caller "
        "would produce"
    )
    inputs = dict(case["call_site_inputs"])
    inputs["expert_affinities"] = pre_sliced
    reset_dispatch_counters()
    with pytest.raises(Glm5NextBlockQuantRouteError, match="GLOBAL router width"):
        bank.block_quant_expert_mm(
            quant_config=quant_config,
            block_size=B,
            expert_parallel_rank=1,
            **inputs,
        )
    counters = dispatch_counters()
    print(
        f"[dispatch-refusal] degree={degree} "
        f"passed_shape={tuple(pre_sliced.shape)} "
        f"global_width={global_experts} seam_counters={counters}"
    )
    assert counters == (0, 0), (
        "the refusal must happen before the block-quant kernel is entered"
    )
