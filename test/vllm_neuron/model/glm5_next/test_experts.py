# SPDX-License-Identifier: Apache-2.0
"""Acceptance test for ``inc-glm53f-031`` -- WP7: MoE-288 expert plumbing and
the TP freeze.

The declared acceptance (increment plan revision 29, L898), verbatim:

    "288 experts over the declared TP degree partitions with **0** experts
    dropped and **0** duplicated (counts summed and compared exactly); the
    288/64 ragged case is asserted to raise a **named** error rather than
    silently pad -- the blocker is the plugin's ``factory.py`` TP=64 freeze, not
    the substrate."

Both halves are measured AT the declared degree. That is not a coincidence and
not a reinterpretation: ``partition_experts`` covers all 288 experts exactly
once at TP=64 (nothing dropped, nothing duplicated) *and* the uniformity gate
refuses that same split, because covering every expert and giving every rank the
same count are two different properties and only the first one holds at 64.

THIS FILE IS CO-AUTHORED, AND THE PARTITION IS A RULE, NOT AN ETIQUETTE.
``inc-glm53f-031`` owns every item ``-k sharding`` collects; ``inc-glm53f-033``
(unlanded) owns the items ``-k shared`` will collect when it extends this file
(plan L912/L915). The two acceptance commands therefore cannot collect each
other's items, so neither counted predicate can be satisfied or broken by the
other increment's tests. **Every item below carries ``sharding`` in its name and
none carries the other selector's token**, which is what makes the partition
mechanical rather than a promise. The reciprocal declaration lives on
``-033``'s plan block; ``-031``'s block does not carry it, and that narrow plan
defect is on the lead's revision list rather than this seat's to repair.

TP=64 IS CITED, NEVER RE-DERIVED. The registration is
``approvals/DECISIONS.md`` section 6 -- *"Preconditions (registered): TP=64,
bf16 KV cache"* -- corroborated by the read-only site
``vllm_neuron/functional/process_groups.py:111``. Nothing here recomputes it and
nothing here makes it configurable.

WHAT MAKES THE COUNTED ZEROS NON-VACUOUS (D1.5). Two behaviours that drop
experts at TP=64 are reachable in this repo today, both measured at the
unmodified parent ``031535b`` before a line was authored:
``gpt_oss/model_bf16.py:1072``'s floor division (the fork's only landed
expert-partition formula) and ``model_fp8.py``'s own ``_per_rank`` helper. Both
report 4 experts per rank, covering 256 of 288 and dropping **32** in silence.
Every counted zero below is controlled against that reachable non-zero.
"""

import ast
import inspect
from pathlib import Path

import pytest

from vllm_neuron.model.glm5_next import config as cfgmod
from vllm_neuron.model.glm5_next import factory as fmod

# ---------------------------------------------------------------------------
# Declared values. Every one is either the checkpoint's own or the registered
# freeze; none is chosen here.
# ---------------------------------------------------------------------------

TOTAL_ROUTED_EXPERTS = 288
DECLARED_TP_DEGREE = 64
EXPERTS_PER_TOK = 8
RAGGED_REMAINDER = TOTAL_ROUTED_EXPERTS % DECLARED_TP_DEGREE  # 32

# The two silent repairs the named raise must refuse, as numbers.
PAD_TARGET = 320  # 5 x 64 -- invents 32 experts
FLOOR_TARGET = 256  # 4 x 64 -- drops 32 experts

# The parent's landed floor-division precedent, restated as a function so the
# control computes it rather than quoting it.
GPT_OSS_PRECEDENT = "vllm_neuron/model/gpt_oss/model_bf16.py:1072"

FACTORY_SIGNATURES_AT_PARENT = {
    "__init__": (
        "(self, hf_config: transformers.configuration_utils.PreTrainedConfig, "
        "text_neuron_config: vllm_neuron.model.neuron_config.NeuronConfig | "
        "None = None, vision_neuron_config: "
        "vllm_neuron.model.neuron_config.VisionNeuronConfig | None = None, "
        "**kwargs) -> None"
    ),
    "from_configs": (
        "(hf_config: transformers.configuration_utils.PreTrainedConfig, "
        "text_neuron_config: vllm_neuron.model.neuron_config.NeuronConfig | "
        "None = None, vision_neuron_config: "
        "vllm_neuron.model.neuron_config.VisionNeuronConfig | None = None) -> "
        "torch.nn.modules.module.Module"
    ),
    "_select_implementation": (
        "(hf_config: transformers.configuration_utils.PreTrainedConfig, "
        "text_neuron_config: vllm_neuron.model.neuron_config.NeuronConfig | "
        "None, vision_neuron_config: "
        "vllm_neuron.model.neuron_config.VisionNeuronConfig | None) -> "
        "torch.nn.modules.module.Module"
    ),
}

# The enum name is assembled rather than written, so the B.6 guard's own source
# is not a hit against itself.
FORBIDDEN_ENUM = "Quantization" + "Type"

_READINGS: dict[str, object] = {}


def _record(**readings: object) -> None:
    """Collect a reading so the reporting item can print all of them."""
    _READINGS.update(readings)
    for key, value in readings.items():
        print(f"{key}={value}")


def _impl():
    """Import the modeling module INSIDE a test body, never at import time.

    ``test_kv_spec.py:157-161``'s idiom, kept for the same reason
    ``test_block_quant_recognition.py`` keeps it: this file sorts before
    ``test_factory.py`` (``e`` < ``f``), so a module-level import here would
    populate ``sys.modules`` for every later item in the package. That is
    legitimate behaviour and ``test_factory.py``'s C03 no longer measures the
    session (``inc-glm53f-031`` repaired it to a subprocess), but keeping the
    footprint minimal costs four lines and keeps this package's convention
    uniform.
    """
    from vllm_neuron.model.glm5_next import model_fp8

    return model_fp8


def _floor_division_precedent(num_experts: int, tp_degree: int) -> dict[str, int]:
    """The fork's landed formula, recomputed -- the D1.5 control's engine.

    ``gpt_oss/model_bf16.py:1072`` is ``num_local_experts // ep_degree`` with no
    raggedness gate, and ``:1148-1151`` builds each rank's index range from it.
    Reproduced here so the control MEASURES the drop instead of asserting it.
    """
    per_rank = num_experts // tp_degree
    covered = set()
    for rank in range(tp_degree):
        covered.update(range(rank * per_rank, (rank + 1) * per_rank))
    return {
        "per_rank": per_rank,
        "assigned": per_rank * tp_degree,
        "covered": len(covered),
        "dropped": num_experts - len(covered),
        "duplicated": per_rank * tp_degree - len(covered),
    }


def _text_config(**overrides: object):
    return cfgmod.Glm5NextTextConfig(**overrides)


# ---------------------------------------------------------------------------
# S01 -- 288 experts over the declared TP degree: 0 dropped, 0 duplicated
# ---------------------------------------------------------------------------


def test_sharding_of_288_experts_at_the_declared_tp_degree_drops_zero_and_duplicates_zero():
    """The coverage half of the declared expected result, counts summed exactly.

    D1.4 certifying component:
    ``vllm_neuron/model/glm5_next/factory.py::partition_experts`` returning
    ``factory.py::ExpertPartition``, whose ``counts`` / ``offsets`` /
    ``local_expert_indices`` are what the two zeros are computed from.
    """
    part = fmod.partition_experts(TOTAL_ROUTED_EXPERTS, DECLARED_TP_DEGREE)

    # "counts summed and compared exactly" -- the sum, not a sample.
    assert sum(part.counts) == TOTAL_ROUTED_EXPERTS
    assert len(part.counts) == DECLARED_TP_DEGREE
    assert len(part.offsets) == DECLARED_TP_DEGREE

    # The union is built from the same index sets a rank would actually own.
    union: set[int] = set()
    for rank in range(DECLARED_TP_DEGREE):
        union.update(part.local_expert_indices(rank))
    assert union == set(range(TOTAL_ROUTED_EXPERTS))

    assert part.covered == TOTAL_ROUTED_EXPERTS
    assert part.dropped == 0
    assert part.duplicated == 0
    assert part.assigned == TOTAL_ROUTED_EXPERTS

    # Nothing was invented and nothing was truncated, stated as the two numbers
    # the named raise refuses.
    assert part.assigned != PAD_TARGET
    assert part.covered != FLOOR_TARGET

    # D1.5 CONTROL 1 -- dropped MOVES. The fork's own landed formula, on the
    # same two numbers, through the same predicate.
    control = _floor_division_precedent(TOTAL_ROUTED_EXPERTS, DECLARED_TP_DEGREE)
    assert control["dropped"] == 32
    assert control["covered"] == FLOOR_TARGET

    # D1.5 CONTROL 2 -- duplicated MOVES. Shift every offset down by one and the
    # same duplication predicate reports a non-zero.
    overlapped = fmod.ExpertPartition(
        num_experts=TOTAL_ROUTED_EXPERTS,
        tp_degree=DECLARED_TP_DEGREE,
        counts=part.counts,
        offsets=tuple(max(0, o - 1) for o in part.offsets),
    )
    assert overlapped.duplicated > 0
    assert overlapped.dropped > 0

    _record(
        s01_tp_degree=DECLARED_TP_DEGREE,
        s01_num_experts=TOTAL_ROUTED_EXPERTS,
        s01_counts_sum=sum(part.counts),
        s01_covered=part.covered,
        s01_dropped=part.dropped,
        s01_duplicated=part.duplicated,
        s01_is_uniform=part.is_uniform,
        s01_remainder=part.remainder,
        s01_distinct_counts=sorted(set(part.counts)),
        s01_control_precedent=GPT_OSS_PRECEDENT,
        s01_control_dropped=control["dropped"],
        s01_control_MOVES=f"dropped 0 -> {control['dropped']}",
        s01_control2_duplicated=overlapped.duplicated,
        s01_control2_MOVES=f"duplicated 0 -> {overlapped.duplicated}",
    )


# ---------------------------------------------------------------------------
# S02 -- the 288/64 ragged case raises a NAMED error rather than padding
# ---------------------------------------------------------------------------


def test_sharding_at_288_over_64_raises_a_named_error_rather_than_padding():
    """The raise half of the declared expected result, 1/1, at the freeze.

    D1.4 certifying component:
    ``factory.py::require_uniform_expert_partition`` raising
    ``factory.py::RaggedExpertPartitionError``, reached through the arch member
    ``factory.py::Glm5NextForConditionalGeneration.expert_sharding_plan`` at its
    default degree.
    """
    raised = 0
    with pytest.raises(fmod.RaggedExpertPartitionError) as gate:
        fmod.require_uniform_expert_partition(
            TOTAL_ROUTED_EXPERTS, DECLARED_TP_DEGREE
        )
    raised += 1
    assert raised == 1

    # The error is NAMED, not a bare ValueError -- and a caller can tell it from
    # the out-of-range class.
    assert type(gate.value) is fmod.RaggedExpertPartitionError
    assert issubclass(fmod.RaggedExpertPartitionError, ValueError)
    assert not issubclass(
        fmod.RaggedExpertPartitionError, cfgmod.Glm5NextExpertConfigError
    )

    message = str(gate.value)
    for token in (
        str(TOTAL_ROUTED_EXPERTS),
        str(DECLARED_TP_DEGREE),
        str(RAGGED_REMAINDER),
        str(PAD_TARGET),
        str(FLOOR_TARGET),
        "G4",
    ):
        assert token in message, f"{token!r} missing from the named raise"

    # The same raise reaches the arch-level member at its DEFAULT degree, which
    # is the freeze. 1/1.
    arch_raised = 0
    with pytest.raises(fmod.RaggedExpertPartitionError):
        fmod.Glm5NextForConditionalGeneration.expert_sharding_plan(_text_config())
    arch_raised += 1
    assert arch_raised == 1

    # NOTHING WAS PADDED: no partition this module builds ever assigns the pad
    # target, at the freeze or anywhere near it.
    for degree in (DECLARED_TP_DEGREE, DECLARED_TP_DEGREE // 2):
        assert (
            fmod.partition_experts(TOTAL_ROUTED_EXPERTS, degree).assigned
            == TOTAL_ROUTED_EXPERTS
        )

    # D1.5 CONTROL -- the raise MOVES. A degree that divides 288 raises nothing
    # and returns a uniform plan through the same gate.
    control_raised = 0
    try:
        control = fmod.require_uniform_expert_partition(TOTAL_ROUTED_EXPERTS, 32)
    except fmod.RaggedExpertPartitionError:  # pragma: no cover - control arm
        control_raised += 1
    assert control_raised == 0
    assert control.is_uniform
    assert set(control.counts) == {9}
    assert control.dropped == 0 and control.duplicated == 0

    _record(
        s02_raised=f"{raised}/1",
        s02_error_type=type(gate.value).__name__,
        s02_error=message,
        s02_arch_member_raised=f"{arch_raised}/1",
        s02_arch_member_default_degree=inspect.signature(
            fmod.Glm5NextForConditionalGeneration.expert_sharding_plan
        )
        .parameters["tp_degree"]
        .default,
        s02_pad_target_never_assigned=True,
        s02_control_raised=control_raised,
        s02_control_counts=sorted(set(control.counts)),
        s02_control_MOVES=f"raised {raised} -> {control_raised} at tp=32",
    )


# ---------------------------------------------------------------------------
# S03 -- the gate IS the raggedness predicate, over a censused domain
# ---------------------------------------------------------------------------


def test_sharding_uniformity_gate_tracks_raggedness_over_every_degree_up_to_64():
    """The raise is a RULE over the whole degree domain, not a special case.

    Without this item, S02's raise could be a hard-coded refusal of the single
    pair (288, 64) and S01's zeros could hold at one degree only.

    D1.4 certifying component: ``factory.py::require_uniform_expert_partition``
    (the raise decision) over ``factory.py::partition_experts`` (the coverage
    arithmetic), evaluated at every degree in ``range(1, 65)``.
    """
    domain = list(range(1, DECLARED_TP_DEGREE + 1))
    uniform, ragged = [], []
    dropped_total = duplicated_total = 0

    for degree in domain:
        part = fmod.partition_experts(TOTAL_ROUTED_EXPERTS, degree)
        # Coverage holds at EVERY degree, ragged or not.
        dropped_total += part.dropped
        duplicated_total += part.duplicated
        assert sum(part.counts) == TOTAL_ROUTED_EXPERTS

        try:
            fmod.require_uniform_expert_partition(TOTAL_ROUTED_EXPERTS, degree)
        except fmod.RaggedExpertPartitionError:
            ragged.append(degree)
        else:
            uniform.append(degree)

    # The gate's decision is exactly the divisibility predicate -- both
    # directions, so neither an over- nor an under-refusal can hide.
    assert uniform == [d for d in domain if TOTAL_ROUTED_EXPERTS % d == 0]
    assert ragged == [d for d in domain if TOTAL_ROUTED_EXPERTS % d != 0]
    assert len(uniform) + len(ragged) == len(domain) == DECLARED_TP_DEGREE
    assert DECLARED_TP_DEGREE in ragged

    # The two counted zeros, over the whole censused population.
    assert dropped_total == 0
    assert duplicated_total == 0

    # D1.5 CONTROL -- the population-wide zero MOVES. The landed floor-division
    # precedent drops on exactly the ragged degrees, and only there.
    control_dropped = {
        degree: _floor_division_precedent(TOTAL_ROUTED_EXPERTS, degree)["dropped"]
        for degree in domain
    }
    assert sum(control_dropped.values()) > 0
    assert [d for d, n in control_dropped.items() if n > 0] == ragged

    _record(
        s03_domain=f"1..{DECLARED_TP_DEGREE}",
        s03_uniform_count=len(uniform),
        s03_ragged_count=len(ragged),
        s03_uniform_degrees=uniform,
        s03_declared_degree_is_ragged=DECLARED_TP_DEGREE in ragged,
        s03_dropped_total=dropped_total,
        s03_duplicated_total=duplicated_total,
        s03_control_dropped_total=sum(control_dropped.values()),
        s03_control_MOVES=(
            f"dropped_total 0 -> {sum(control_dropped.values())} over "
            f"{len(ragged)} ragged degrees"
        ),
    )


# ---------------------------------------------------------------------------
# S04 -- the model-level bank consumes the partition
# ---------------------------------------------------------------------------


def test_sharding_plan_is_consumed_by_the_routed_expert_bank_in_model_fp8():
    """The ``model_fp8.py`` half of the declared surface.

    D1.4 certifying component:
    ``model_fp8.py::Glm5NextRoutedExperts.__init__`` setting
    ``expert_partition`` / ``num_local_experts`` / ``tp_degree`` and
    ``model_fp8.py::Glm5NextRoutedExperts.local_expert_indices``, reached
    through ``model_fp8.py::Glm5NextMoEBlock.__init__``.
    """
    impl = _impl()
    text_config = _text_config()

    bank = impl.Glm5NextRoutedExperts(text_config, world_size=32)
    assert bank.tp_degree == 32
    assert bank.num_routed_experts == TOTAL_ROUTED_EXPERTS
    assert bank.num_local_experts == 9
    assert bank.expert_partition.dropped == 0
    assert bank.expert_partition.duplicated == 0
    assert bank.local_expert_indices(0) == tuple(range(9))
    assert bank.local_expert_indices(31) == tuple(range(279, 288))

    # Every rank's slice, summed -- the model-level restatement of S01.
    union: set[int] = set()
    for rank in range(32):
        union.update(bank.local_expert_indices(rank))
    assert union == set(range(TOTAL_ROUTED_EXPERTS))

    block = impl.Glm5NextMoEBlock(text_config, world_size=32)
    assert block.experts.num_local_experts == 9

    # The default resolves the process group, which is 1 undistributed -- so the
    # landed tree still builds and no neighbour's item moves.
    default_bank = impl.Glm5NextRoutedExperts(text_config)
    assert default_bank.tp_degree == impl._resolve_world_size() == 1
    assert default_bank.num_local_experts == TOTAL_ROUTED_EXPERTS

    # THE FREEZE'S CONSEQUENCE IS VISIBLE WHERE THE MODEL IS BUILT, not only at
    # the arithmetic. This is G4 reaching the model level.
    with pytest.raises(fmod.RaggedExpertPartitionError):
        impl.Glm5NextRoutedExperts(text_config, world_size=DECLARED_TP_DEGREE)
    with pytest.raises(fmod.RaggedExpertPartitionError):
        impl.Glm5NextMoEBlock(text_config, world_size=DECLARED_TP_DEGREE)

    # D1.5 CONTROL -- the drop-32 formula is still reachable in this very
    # module, so the bank's 0 dropped is a distinction between two live
    # behaviours rather than the only thing the file can do.
    assert impl._per_rank(TOTAL_ROUTED_EXPERTS, DECLARED_TP_DEGREE) == 4
    assert 4 * DECLARED_TP_DEGREE == FLOOR_TARGET

    # ``_build_mlp``'s signature is UNCHANGED -- the world_size addition is
    # trailing and optional, so no call site outside this increment moved.
    assert str(inspect.signature(impl._build_mlp)) == (
        "(text_config: 'Glm5NextTextConfig', layer_idx: 'int') -> 'nn.Module'"
    )

    _record(
        s04_bank_tp_degree=bank.tp_degree,
        s04_bank_num_local_experts=bank.num_local_experts,
        s04_bank_dropped=bank.expert_partition.dropped,
        s04_bank_duplicated=bank.expert_partition.duplicated,
        s04_bank_rank0_indices=f"{bank.local_expert_indices(0)[:3]}...",
        s04_moeblock_num_local_experts=block.experts.num_local_experts,
        s04_default_tp_degree=default_bank.tp_degree,
        s04_default_num_local_experts=default_bank.num_local_experts,
        s04_raises_at_the_freeze=True,
        s04_control_per_rank_helper=impl._per_rank(
            TOTAL_ROUTED_EXPERTS, DECLARED_TP_DEGREE
        ),
        s04_control_MOVES=(
            f"in-module _per_rank still covers {FLOOR_TARGET}/"
            f"{TOTAL_ROUTED_EXPERTS} at tp={DECLARED_TP_DEGREE}"
        ),
        s04_build_mlp_signature_unchanged=True,
    )


# ---------------------------------------------------------------------------
# S05 -- config-side expert-count validation
# ---------------------------------------------------------------------------


def test_sharding_config_validation_rejects_out_of_range_expert_counts():
    """The ``config.py`` half of the declared surface, plus its named boundary.

    FIVE expert-count refusals are measured, across TWO named gates, because the
    two questions live at different layers and that split was measured rather
    than assumed:

    * FOUR **per-field, well-formedness** refusals at construction time --
      ``config.py::Glm5NextTextConfig._validate_expert_counts`` raising
      ``config.py::Glm5NextExpertConfigError`` from ``__post_init__``. This is
      the layer ``_validate_layer_types`` already validates at.
    * ONE **cross-field, router** refusal on the sharding path --
      ``factory.py::require_routable_expert_counts`` raising the same error
      class, reached through
      ``factory.py::Glm5NextForConditionalGeneration.expert_sharding_plan``.

    WHY THE SPLIT IS REAL AND NOT A CONVENIENCE. Asking the router question at
    construction time rejected ``inc-glm53f-011``'s landed ``mini_config``
    fixture (``test_weight_loaders.py:282``): a 4-expert bank inheriting the
    checkpoint's top-8 default, i.e. a structural key-mapping fixture that never
    routes a token. That is a latent incoherence in a landed fixture, routed to
    the lead rather than repaired here -- ``test_weight_loaders.py`` is outside
    this increment's declared surface. The refusal is KEPT, at the layer that
    routes. The last arm below is the regression guard for that boundary.
    """
    # -- gate 1: per-field, at construction --------------------------------
    rejected_at_construction = [
        {"n_routed_experts": 0},
        {"n_routed_experts": -8},
        {"num_experts_per_tok": 0},
        {"n_shared_experts": -1},
    ]
    construction_raised = 0
    messages = {}
    for overrides in rejected_at_construction:
        with pytest.raises(cfgmod.Glm5NextExpertConfigError) as err:
            _text_config(**overrides)
        construction_raised += 1
        field, value = next(iter(overrides.items()))
        messages[f"{field}={value}"] = str(err.value)
    assert construction_raised == len(rejected_at_construction) == 4

    # -- gate 2: cross-field, on the sharding path -------------------------
    plan_raised = 0
    with pytest.raises(cfgmod.Glm5NextExpertConfigError) as router_err:
        fmod.require_routable_expert_counts(TOTAL_ROUTED_EXPERTS, 999)
    plan_raised += 1
    messages["num_experts_per_tok=999"] = str(router_err.value)

    # and it is reached through the arch member, not only callable directly.
    with pytest.raises(cfgmod.Glm5NextExpertConfigError):
        fmod.Glm5NextForConditionalGeneration.expert_sharding_plan(
            _text_config(n_routed_experts=4, num_experts_per_tok=8), tp_degree=4
        )
    assert plan_raised == 1
    assert construction_raised + plan_raised == 5

    # D1.5 CONTROL -- the refusal MOVES. All five were ACCEPTED at the unmodified
    # parent, and the valid neighbours are still accepted here, so the validators
    # discriminate rather than refusing everything.
    accepted = 0
    for overrides in (
        {},
        {"n_routed_experts": 1, "num_experts_per_tok": 1},
        {"n_routed_experts": TOTAL_ROUTED_EXPERTS},
        {"num_experts_per_tok": TOTAL_ROUTED_EXPERTS},
        {"n_shared_experts": 0},
        {"n_shared_experts": 1},
    ):
        text_config = _text_config(**overrides)
        accepted += 1
        assert text_config.n_routed_experts >= 1
    assert accepted == 6
    assert fmod.require_routable_expert_counts(TOTAL_ROUTED_EXPERTS, 8) is None

    # The checkpoint's own values pass, 1/1 -- this validator must not reject
    # the model it is for.
    checkpoint = _text_config()
    assert checkpoint.n_routed_experts == TOTAL_ROUTED_EXPERTS
    assert checkpoint.num_experts_per_tok == EXPERTS_PER_TOK

    # BOUNDARY REGRESSION GUARD: ``inc-glm53f-011``'s landed fixture shape must
    # still CONSTRUCT. If a later hand moves the router question back into
    # ``__post_init__``, this arm goes red before that increment's acceptance
    # does.
    mini = _text_config(
        num_hidden_layers=4,
        n_routed_experts=4,
        n_shared_experts=1,
        first_k_dense_replace=3,
        tie_word_embeddings=False,
    )
    assert mini.n_routed_experts == 4
    assert mini.num_experts_per_tok == EXPERTS_PER_TOK  # the inherited default
    assert mini.num_experts_per_tok > mini.n_routed_experts  # the incoherence

    # The two error classes are distinct, so a caller can route on them.
    assert not issubclass(
        cfgmod.Glm5NextExpertConfigError, fmod.RaggedExpertPartitionError
    )
    assert not issubclass(
        fmod.RaggedExpertPartitionError, cfgmod.Glm5NextExpertConfigError
    )

    _record(
        s05_rejected_at_construction=f"{construction_raised}/4",
        s05_rejected_on_the_shard_path=f"{plan_raised}/1",
        s05_rejected_total=construction_raised + plan_raised,
        s05_accepted=f"{accepted}/6",
        s05_error_type=cfgmod.Glm5NextExpertConfigError.__name__,
        s05_messages=messages,
        s05_checkpoint_values=(
            checkpoint.n_routed_experts,
            checkpoint.num_experts_per_tok,
            checkpoint.n_shared_experts,
        ),
        s05_inc011_fixture_shape_still_constructs=True,
        s05_control_MOVES="parent ACCEPTED all 5; here rejected 5/5, accepted 6/6",
    )


# ---------------------------------------------------------------------------
# S06 -- no landed co-author's member signature moved (routing fact 2)
# ---------------------------------------------------------------------------


def test_sharding_members_are_a_pure_addition_to_the_co_authored_factory():
    """``factory.py`` is co-authored; this increment adds BESIDE, never edits.

    ``inc-glm53f-009`` owns the class plus ``from_configs`` /
    ``_select_implementation``; ``inc-glm53f-074`` owns the trailing
    ``**kwargs`` on ``__init__`` plus ``embed_input_ids`` / ``compute_logits``.
    Modifying any of their signatures would be ``evidence_contradicts_design``,
    so this item measures that none moved -- the expected strings are the ones
    the parent probe read at ``031535b``.

    D1.4 certifying component: ``inspect.signature`` over
    ``factory.py::Glm5NextForConditionalGeneration``'s landed members, plus an
    ``ast`` walk over ``factory.py`` for the two boundary methods' bodies.
    """
    cls = fmod.Glm5NextForConditionalGeneration

    for member, expected in FACTORY_SIGNATURES_AT_PARENT.items():
        observed = str(inspect.signature(getattr(cls, member)))
        assert observed == expected, f"{member} signature moved:\n{observed}"

    # ``-074``'s two boundary members still raise, unchanged in contract.
    for member in ("embed_input_ids", "compute_logits"):
        assert callable(getattr(cls, member))
    with pytest.raises(NotImplementedError):
        cls.embed_input_ids(cls, None)
    with pytest.raises(NotImplementedError):
        cls.compute_logits(cls, None)

    # This increment's own members are PRESENT -- otherwise "nothing moved"
    # would be satisfied by an empty increment.
    for name in ("expert_sharding_plan",):
        assert hasattr(cls, name)
    for name in (
        "TP_DEGREE_FREEZE",
        "RaggedExpertPartitionError",
        "ExpertPartition",
        "partition_experts",
        "require_uniform_expert_partition",
    ):
        assert hasattr(fmod, name)

    # The lazy implementation import is still the only route to ``model_fp8``
    # from this module: no module-level import of it was added.
    source = Path(fmod.__file__).read_text()
    tree = ast.parse(source)
    module_level_imports = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    for node in module_level_imports:
        rendered = ast.dump(node)
        assert "model_fp8" not in rendered

    # D1.5 CONTROL -- the comparison MOVES. A deliberately wrong expected string
    # fails the same predicate, so the equalities above are not vacuous.
    control_mismatch = 0
    if str(inspect.signature(cls.from_configs)) != "(self) -> None":
        control_mismatch += 1
    assert control_mismatch == 1

    _record(
        s06_signatures_checked=len(FACTORY_SIGNATURES_AT_PARENT),
        s06_signatures_moved=0,
        s06_boundary_members_still_raise=2,
        s06_new_members_present=6,
        s06_module_level_imports=len(module_level_imports),
        s06_module_level_model_fp8_imports=0,
        s06_control_MOVES=f"mismatch 0 -> {control_mismatch} on a wrong expectation",
    )


# ---------------------------------------------------------------------------
# S07 -- constraint B.6: no vendor quantisation enum reference
# ---------------------------------------------------------------------------


def test_sharding_adds_no_vendor_quantisation_enum_reference():
    """Plan section 11 constraint B.6, kept at the 0 ``inc-glm53f-023`` landed.

    D1.4 certifying component: an ``ast`` walk over every ``.py`` file in
    ``vllm_neuron/model/glm5_next/``, counting ``Name`` / ``Attribute`` /
    ``ImportFrom`` nodes only -- so neither a prose mention nor a string literal
    can register as a code reference, and a code reference cannot hide in a
    comment.
    """

    def enum_refs(source: str) -> int:
        hits = 0
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Name) and node.id == FORBIDDEN_ENUM:
                hits += 1
            elif isinstance(node, ast.Attribute) and node.attr == FORBIDDEN_ENUM:
                hits += 1
            elif isinstance(node, ast.ImportFrom):
                hits += sum(1 for a in node.names if a.name == FORBIDDEN_ENUM)
        return hits

    package = Path(fmod.__file__).parent
    per_file = {
        path.name: enum_refs(path.read_text()) for path in sorted(package.glob("*.py"))
    }
    assert sum(per_file.values()) == 0, per_file
    assert len(per_file) >= 6

    # D1.5 CONTROL -- the count MOVES on a synthetic source that really does
    # reference the enum, through the same walk.
    control_source = (
        f"from nkilib.core.utils.common_types import {FORBIDDEN_ENUM}\n"
        f"x = {FORBIDDEN_ENUM}.NONE\n"
    )
    control = enum_refs(control_source)
    assert control == 2

    _record(
        s07_files_scanned=len(per_file),
        s07_per_file=per_file,
        s07_total_code_refs=sum(per_file.values()),
        s07_control_code_refs=control,
        s07_control_MOVES=f"0 -> {control}",
    )


# ---------------------------------------------------------------------------
# S08 -- the TP freeze is cited, not derived, and not configurable here
# ---------------------------------------------------------------------------


def test_sharding_tp_degree_freeze_is_the_registered_value_and_not_configurable():
    """Routing fact 4: cite, never re-derive, never make it configurable here.

    D1.4 certifying component:
    ``factory.py::TP_DEGREE_FREEZE`` and the ``tp_degree`` default of
    ``factory.py::Glm5NextForConditionalGeneration.expert_sharding_plan``, plus
    an ``ast`` walk over ``factory.py`` for environment reads.
    """
    assert fmod.TP_DEGREE_FREEZE == DECLARED_TP_DEGREE == 64

    default = (
        inspect.signature(fmod.Glm5NextForConditionalGeneration.expert_sharding_plan)
        .parameters["tp_degree"]
        .default
    )
    assert default == fmod.TP_DEGREE_FREEZE

    # NOT CONFIGURABLE HERE: no environment read and no config field feeds it.
    source = Path(fmod.__file__).read_text()
    tree = ast.parse(source)
    env_reads = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in {"environ", "getenv"}:
            env_reads += 1
        if isinstance(node, ast.Name) and node.id in {"getenv", "environ"}:
            env_reads += 1
    assert env_reads == 0
    assert "os.environ" not in source and "getenv" not in source

    # The value is a module-level literal, not computed from anything.
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "TP_DEGREE_FREEZE"
            for t in node.targets
        )
    ]
    assert len(assignments) == 1
    assert isinstance(assignments[0].value, ast.Constant)
    assert assignments[0].value.value == 64

    # The corroborating read-only site is real, at the line the plan cites.
    repo_root = Path(fmod.__file__).resolve().parents[3]
    pg_line = (
        (repo_root / "vllm_neuron" / "functional" / "process_groups.py")
        .read_text()
        .splitlines()[110]
    )
    assert "group_size == 64" in pg_line

    # D1.5 CONTROL -- the default is what is USED, not merely declared: passing
    # an explicit degree changes the outcome through the same member.
    control = fmod.Glm5NextForConditionalGeneration.expert_sharding_plan(
        _text_config(), tp_degree=32
    )
    assert control.tp_degree == 32 and control.is_uniform
    control_raised = 0
    try:
        fmod.Glm5NextForConditionalGeneration.expert_sharding_plan(_text_config())
    except fmod.RaggedExpertPartitionError:
        control_raised += 1
    assert control_raised == 1

    _record(
        s08_TP_DEGREE_FREEZE=fmod.TP_DEGREE_FREEZE,
        s08_member_default=default,
        s08_env_reads=env_reads,
        s08_literal_assignments=len(assignments),
        s08_process_groups_L111=pg_line.strip(),
        s08_control_explicit_degree=control.tp_degree,
        s08_control_MOVES=(
            f"default degree raises ({control_raised}), explicit 32 does not"
        ),
    )


# ---------------------------------------------------------------------------
# Reporting -- the readings the evidence record quotes
# ---------------------------------------------------------------------------


def test_sharding_report_the_measured_readings(capsys):
    """Prints every reading the evidence record quotes, in one place.

    Depends on the items above having run (pytest's declaration order), which is
    why it is last and why it asserts a floor on the reading count rather than a
    value it computes itself.
    """
    with capsys.disabled():
        print()
        print(f"[freeze] TP_DEGREE_FREEZE = {fmod.TP_DEGREE_FREEZE} (registered)")
        print(
            f"[S01] 288 over {DECLARED_TP_DEGREE}: dropped="
            f"{_READINGS.get('s01_dropped')} duplicated="
            f"{_READINGS.get('s01_duplicated')} counts_sum="
            f"{_READINGS.get('s01_counts_sum')}"
        )
        print(f"[S02] named raise: {_READINGS.get('s02_error_type')}")
        print(
            f"[S03] domain 1..64: uniform={_READINGS.get('s03_uniform_count')} "
            f"ragged={_READINGS.get('s03_ragged_count')} "
            f"dropped_total={_READINGS.get('s03_dropped_total')}"
        )
        print(
            f"[S04] bank at tp=32: num_local_experts="
            f"{_READINGS.get('s04_bank_num_local_experts')}"
        )
        print(
            f"[S05] expert-count refusals: "
            f"construction={_READINGS.get('s05_rejected_at_construction')} "
            f"shard-path={_READINGS.get('s05_rejected_on_the_shard_path')} "
            f"total={_READINGS.get('s05_rejected_total')}"
        )
        print(f"[S06] signatures moved={_READINGS.get('s06_signatures_moved')}")
        print(f"[B.6] enum code refs={_READINGS.get('s07_total_code_refs')}")
        print("--- all readings ---")
        for key in sorted(_READINGS):
            print(f"{key}={_READINGS[key]}")

    assert len(_READINGS) >= 40
    assert _READINGS["s01_dropped"] == 0
    assert _READINGS["s01_duplicated"] == 0
    assert _READINGS["s03_dropped_total"] == 0
    assert _READINGS["s07_total_code_refs"] == 0


# ===========================================================================
# ``inc-glm53f-033`` -- WP7: the shared expert. THE OTHER HALF OF THIS
# CO-AUTHORED FILE.
#
# THE PARTITION IS A RULE, NOT AN ETIQUETTE (plan L940). ``-031`` owns every
# item ``-k sharding`` collects and this increment owns every item ``-k shared``
# collects. Every item below carries ``shared`` in its name and NONE carries the
# token ``sharding``, so the two acceptance commands cannot collect each other's
# items and neither counted predicate can be satisfied or broken by the other
# increment's tests. Both counts are MEASURED by dedicated ``--collect-only -q``
# runs and recorded in ``increments/evidence-033.md``, never declared here.
#
# EVERYTHING BELOW IS ADDED, NOTHING ABOVE IS TOUCHED. ``-031``'s items are
# LANDED and its recorded acceptance asserts them, so a modification of any line
# above this banner is ``evidence_contradicts_design`` rather than a repair
# (plan L940). This section therefore appends, keeps its own readings dict rather
# than writing ``-031``'s ``_READINGS``, and takes every import inside a function
# body -- the file's own ``_impl()`` idiom, kept so no line of the landed import
# block moves either.
#
# THE DECLARED ACCEPTANCE (plan L933), verbatim:
#
#     "the layer output equals routed-plus-shared from a torch reference at
#     ``assert_close(rtol=3e-2, atol=1e-5)``; **and** a zeroed-shared-expert
#     case reproduces the routed-only output at **atol 1e-5**, proving the shared
#     contribution is added exactly once rather than twice."
#
# THE DECLARED ROUTE PREDICATE (plan L934-935, revision 33): ``-026``'s dispatch
# counter reads EXACTLY 3 -- one per projection site -- per shared-expert call,
# and its torch-fallback counter reads 0. The count was re-derived at revision 33
# from this seat's attempt-1 measurement (``increments/evidence-033.md``); the
# earlier ``1`` was structurally unreachable. No tolerance, count or comparator
# below is invented, widened or narrowed.
# ===========================================================================


# --------------------------------------------------------------------------- #
# Declared values. The tolerances and the count are the plan's; the tiny-config  #
# extents are chosen to ADMIT ``-026``'s kernel, which is a fixture decision.    #
# --------------------------------------------------------------------------- #

#: The plan's tolerance pair for conjunct 1, and its atol for conjunct 2.
SHARED_RTOL = 3e-2
SHARED_ATOL = 1e-5

#: The plan's route-predicate count (L934-935): one seam entry per projection.
SHARED_DECLARED_SEAM_ENTRIES = 3

#: Tiny config. Every extent is forced by ``-026``'s own admission gates
#: (``blockwise_fp8_mm.py::_require_blocked``), read off that source rather than
#: guessed, and chosen to ADMIT because a geometry the kernel REFUSES raises --
#: it does not fall back -- and a refused shape would leave the counter at 0.
#:   T % TILE_SIZE == 0        (``:239`` -- M tiles over the PSUM partition axis)
#:   H % BLOCK_QUANT_SIZE == 0 (``:246`` -- K needs a whole number of block scales)
#:   I % BLOCK_QUANT_SIZE == 0 (``:252`` -- N likewise)
#: H and I are 2 blocks rather than 1 deliberately: a ``[1, 1]`` scale grid cannot
#: distinguish a transposed flat index, and this fixture's block scales are
#: distinct and asymmetric so that a mis-mapping is numerically visible.
SHARED_T = 128
SHARED_H = 512
SHARED_I = 512

#: Per-block exponents for the three projections' scale grids. DETERMINISTIC, not
#: sampled: three properties are load-bearing and a draw satisfies them only by
#: luck -- every entry an exact power of two (F1, below), the four entries
#: DISTINCT so a transposed flat index moves the numbers, and the matrix
#: ASYMMETRIC so a transpose is not the identity.
SHARED_GATE_EXPONENTS = ((-2, 1), (0, -1))
SHARED_UP_EXPONENTS = ((1, -1), (-2, 0))
SHARED_DOWN_EXPONENTS = ((0, 2), (-1, 1))

# --------------------------------------------------------------------------- #
# `B22-M1-shared-expert-swiglu-clamp-omitted`, repair round 1.                  #
#                                                                              #
# THE FINDING, in one sentence: this section computed `silu(gate) * up` where    #
# the checkpoint's shared expert CLAMPS both projections first, and the oracle   #
# below repeated the same omission, so no bar in this file could see it.         #
#                                                                              #
# The reference, read first-hand: transformers v5.16.1                          #
# `models/glm5_next/modeling_glm5_next.py:98-104` (`Glm5NextTextMLP.forward`),   #
# whose two clamp lines are `:102` and `:103`, and `:196-197`, where            #
# `Glm5NextTextMoE.__init__` builds `shared_experts` as that same MLP.           #
#                                                                              #
# NOT TOUCHED HERE: no tolerance, extent, seed, exponent or comparator moves.    #
# `SHARED_RTOL`, `SHARED_ATOL`, `SHARED_T`, `SHARED_H`, `SHARED_I`, the three    #
# exponent matrices and `SHARED_DECLARED_SEAM_ENTRIES` are byte-identical to     #
# their landed values, and every new arm is an addition.                         #
# --------------------------------------------------------------------------- #

#: `-078` lands `fixtures/hf-config.json` as a byte-identical copy of the
#: published GLM-5.3-Flash config and pins it by this digest in its own conjunct;
#: `-080`'s landed `test_config.py` reads the same file the same way. This
#: section READS it and never writes it. Reading the bound from here rather than
#: typing `10.0` is what makes the clamp the CHECKPOINT'S bound: the finding asks
#: for the value to be sourced from the checkpoint, and a literal in this file
#: would fail that half of it just as a literal in the shipped path would.
#: R01's provenance probe value, and the ONE reason it is not ``10.0``. The
#: checkpoint declares ``swiglu_limit = 10.0`` and ``Glm5NextTextConfig``'s field
#: defaults to the same number on purpose (``config.py``, following
#: ``rms_norm_eps``), so a reading of ``10.0`` on a built object cannot tell a
#: config read apart from a default. ``7.5`` is a value the checkpoint does NOT
#: carry, which is what makes the read path falsifiable. It is not a comparator
#: and nothing is measured against it: it is an input pushed through the adapter.
#: The same device ``inc-glm53f-080`` uses for its epsilon
#: (``test_config.py``'s ``C080_NON_DEFAULT_RMS_NORM_EPS = 3e-05``).
R01_NON_DEFAULT_BOUND = 7.5

SHARED_VENDOR_CONFIG_SHA256 = (
    "bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f"
)

#: The power-of-two divisor that puts the gate operand ASTRIDE the clamp.
#: MEASURED, not chosen: at the landed fixture every element of both operands is
#: outside `[-10, 10]`, so `silu(clamp(gate))` is the constant `silu(10)` and the
#: landed arms stop responding to the gate projection altogether -- halving the
#: gate weights moves the clamped output by `0.000000e+00` where it moves the
#: unclamped output by `5.000000e-01`. Dividing the hidden states by 8 leaves
#: 49,474 of 65,536 gate elements above the bound and 16,062 below it, restores a
#: `3.667817e-01` response to that same perturbation, and still reads
#: `0.000000e+00` between kernel and oracle. Both readings and the full scan over
#: divisors 1 to 32 are in `increments/probe-R7-straddle-and-sensitivity.out`.
#: A POWER OF TWO, so every bf16 hidden value stays exactly representable and
#: `-026`'s F1 losslessness precondition is untouched -- the weights, the block
#: scales and every extent stay the landed fixture's own.
SHARED_STRADDLE_DIVISOR = 8


class SharedRouteInstrumentError(AssertionError):
    """A route reading that is not what the plan declares."""


class SharedF1PreconditionError(AssertionError):
    """The pow2 losslessness precondition did not hold on this case's scales."""


class SharedVacuousControlError(AssertionError):
    """A control whose input could not have made it fail.

    A zero or a pass over vacuous input measures nothing, so the control refuses
    to report a result it did not earn.
    """


class SharedSectionOwnershipError(AssertionError):
    """A structural claim about this increment's own section did not hold."""


_SHARED_READINGS: dict[str, object] = {}


def _shared_record(**readings: object) -> None:
    """Collect a reading for the reporting item.

    Writes this increment's OWN dict. ``-031``'s ``_READINGS`` and its
    ``len(_READINGS) >= 40`` floor are landed and are not touched, so neither
    increment's reporting item can be moved by the other's readings.
    """
    _SHARED_READINGS.update(readings)
    for key, value in readings.items():
        print(f"{key}={value}")


# --------------------------------------------------------------------------- #
# Route instrumentation. Counts the VENDOR entry point as well as this          #
# campaign's own counters, so a bug in ours cannot fake the reading.            #
# --------------------------------------------------------------------------- #
class _SharedSimulatorCounter:
    """Counts real ``nki.simulator.simulate_kernel`` calls for the duration.

    Structure carried verbatim from ``-026``'s landed
    ``test_blockwise_fp8_mm.py::_SimulatorCounter``; the import is function-local
    for this file's reasons rather than that file's.
    """

    def __init__(self) -> None:
        self.calls = 0
        self._nki = None
        self._real = None

    def __enter__(self) -> "_SharedSimulatorCounter":
        # BOTH imports are required and the second is not redundant:
        # ``nki.simulator`` is a SUBMODULE, so ``import nki`` alone leaves
        # ``nki.simulator`` unbound and attribute access raises
        # ``AttributeError: module 'nki' has no attribute 'simulator'``. This is
        # ``-026``'s landed pair (``test_blockwise_fp8_mm.py`` imports ``nki``
        # and ``nki.simulator`` on consecutive lines) and it is repeated here for
        # the same reason rather than trusted to import order elsewhere.
        import nki
        import nki.simulator  # noqa: F401  -- binds nki.simulator

        self._nki = nki
        self._real = nki.simulator.simulate_kernel
        real = self._real

        def counting(*args, **kwargs):
            self.calls += 1
            return real(*args, **kwargs)

        nki.simulator.simulate_kernel = counting
        return self

    def __exit__(self, *exc_info) -> None:
        self._nki.simulator.simulate_kernel = self._real


def _shared_seam():
    """``-026``'s module, re-acquired through ``importlib``.

    THIS IS THE R-2 FORM AND THE MECHANISM IS THE POINT. The counted seam is
    ``-026``'s, not this increment's, so this module resets and reads counters it
    does not own, across a module boundary, exactly as ``-026``'s own
    ``test_dispatch_counters_are_module_level_state_reachable_from_elsewhere``
    proved was possible. Re-acquiring by ``importlib`` rather than binding the
    functions once makes the module-level state visible as shared state.
    """
    import importlib

    return importlib.import_module("vllm_neuron.functional.blockwise_fp8_mm")


def _assert_shared_route(
    sim: _SharedSimulatorCounter, expected_entries: int, label: str
) -> str:
    """Read all four route instruments; return the reading for the transcript.

    Four instruments, and instrument 4 is the vendor's own entry point, so a
    defect in this campaign's three counters cannot fake a green route reading.
    """
    from vllm_neuron.utils.neuron_utils import can_run_kernel

    import torch

    seam = _shared_seam()
    nki_dispatch, torch_fallback = seam.dispatch_counters()
    gate = can_run_kernel(torch.zeros(1))
    reading = (
        f"[{label}] nki_dispatch={nki_dispatch} torch_fallback={torch_fallback} "
        f"can_run_kernel={gate} simulate_kernel_calls={sim.calls} "
        f"declared={expected_entries}"
    )
    print(reading)
    if nki_dispatch != expected_entries:
        raise SharedRouteInstrumentError(
            f"{label}: -026's seam dispatch counter read {nki_dispatch}, declared "
            f"{expected_entries} (one per projection site: gate, up, down). A "
            f"bypassed projection reads fewer and is exactly what this counts. "
            f"{reading}"
        )
    if torch_fallback != 0:
        raise SharedRouteInstrumentError(
            f"{label}: -026's torch-fallback counter read {torch_fallback}, "
            f"declared exactly 0 -- a fallback pass would compare torch against "
            f"torch and is this campaign's F1 false green. {reading}"
        )
    if gate is not True:
        raise SharedRouteInstrumentError(
            f"{label}: can_run_kernel() read {gate!r}, declared True. {reading}"
        )
    if sim.calls != expected_entries:
        raise SharedRouteInstrumentError(
            f"{label}: nki.simulator.simulate_kernel ran {sim.calls} times, "
            f"declared {expected_entries}. A numeric pass without a simulator "
            f"call is the F1 false green. {reading}"
        )
    return reading


# --------------------------------------------------------------------------- #
# Fixture construction.                                                        #
# --------------------------------------------------------------------------- #
def _shared_pow2_scales(exponents, rows: int, cols: int):
    """The PUBLIC ``[rows//256, cols//256]`` block-scale grid, every entry pow2.

    F1, AND WHY IT IS HERE RATHER THAN INHERITED. ``-026``'s kernel accumulates
    the two ``128``-wide contraction tiles of one ``256`` block in PSUM and
    applies the block scale AFTER that accumulation
    (``blockwise_fp8_mm.py:203-220``, and its module docstring states the order
    is deliberate). ``increments/evidence-071.md`` F1 measured **720 fp32 ulp**
    of remapping error under a non-pow2 block scale against **0** under a pow2
    one. So a non-pow2 fixture would make ``rtol=3e-2`` certify remapping error
    on top of this increment's plumbing, and the tolerance would no longer mean
    what it says. The tolerance is UNCHANGED; only the world it is measured in is
    narrowed -- exactly what the plan's own F1 clause does for ``-025``/``-026``.
    """
    import torch

    from vllm_neuron.functional.blockwise_fp8_mm import scale_grid_shape

    want = scale_grid_shape(rows, cols)
    grid = torch.zeros(want, dtype=torch.int64)
    for k_block in range(want[0]):
        for n_block in range(want[1]):
            grid[k_block, n_block] = exponents[k_block % 2][n_block % 2]
    return torch.ldexp(torch.ones(want, dtype=torch.float32), grid)


def _shared_fp8_grid(seed: int, *shape: int, signed: bool = False):
    """Values already on the fp8-e4m3 grid, so every cast in the fixture is exact.

    ``signed=False`` IS A CONDITIONING CHOICE CARRIED FROM ``-025`` ATTEMPT 1,
    which read ``max_rel_error=8.32e+01`` against this same ``rtol=3e-2`` from
    catastrophic cancellation in a SIGNED fixture over a 512-wide contraction --
    not from a kernel defect. With signed values the reference lands arbitrarily
    close to zero while the terms that built it are ~1e2, so a pointwise RELATIVE
    tolerance is dominated by cancellation and no correct implementation can
    satisfy it. The hazard is sharper here because this increment chains THREE
    contractions, so cancellation compounds. Signed coverage is kept as its own
    arm at the SAME declared tolerances
    (:func:`test_shared_expert_signed_fixture_agrees_in_norm_under_cancellation`).
    """
    import torch

    generator = torch.Generator().manual_seed(seed)
    low = -7 if signed else 1
    return torch.randint(low, 8, shape, generator=generator).to(torch.float32) / 8.0


def _shared_build_case(zero_shared: bool = False, signed: bool = False) -> dict:
    """The tiny config: three fp8 projections, three pow2 public scale grids.

    ``zero_shared=True`` zeroes the three projection weights, which is the plan's
    second declared conjunct. The SCALES are left alone and the extents are
    unchanged, so the geometry the kernel admits does not move and all three seam
    entries still happen -- the count is a statement about the route, not about
    whether the numbers are nonzero.
    """
    import torch

    fp8 = torch.float8_e4m3fn

    gate_w = _shared_fp8_grid(11, SHARED_H, SHARED_I, signed=signed)
    up_w = _shared_fp8_grid(12, SHARED_H, SHARED_I, signed=signed)
    down_w = _shared_fp8_grid(13, SHARED_I, SHARED_H, signed=signed)
    if zero_shared:
        gate_w = torch.zeros_like(gate_w)
        up_w = torch.zeros_like(up_w)
        down_w = torch.zeros_like(down_w)

    hidden_states = _shared_fp8_grid(31, SHARED_T, SHARED_H, signed=signed).to(
        torch.bfloat16
    )

    return {
        "hidden_states": hidden_states,
        "gate_w": gate_w.to(fp8),
        "up_w": up_w.to(fp8),
        "down_w": down_w.to(fp8),
        "gate_s": _shared_pow2_scales(SHARED_GATE_EXPONENTS, SHARED_H, SHARED_I),
        "up_s": _shared_pow2_scales(SHARED_UP_EXPONENTS, SHARED_H, SHARED_I),
        "down_s": _shared_pow2_scales(SHARED_DOWN_EXPONENTS, SHARED_I, SHARED_H),
    }


def _shared_block_quant_config():
    """``Glm5NextQuantConfig`` for the pinned checkpoint -- nothing hand-fed.

    The pinned fixture is digest-verified before it is parsed, so the
    quantisation policy this call site routes on is the campaign's registered one
    and not a value this test invented. Idiom and digest carried from ``-027``'s
    landed ``test_moe_path.py:363-390``.
    """
    import hashlib
    import json
    from pathlib import Path

    fixture = Path(__file__).resolve().parent / "fixtures" / "config.json"
    expected = "5ed24d23a3e14a038352e1bdc21fd25fc90ff2291d3f6a310acf5d4036665a1d"
    digest = hashlib.sha256(fixture.read_bytes()).hexdigest()
    if digest != expected:
        raise SharedVacuousControlError(
            f"pinned fixture digest moved: {digest} != {expected}. The "
            f"quantisation policy this call site routes on would no longer be "
            f"the campaign's registered one."
        )

    from vllm_neuron.model.glm5_next.config import Glm5NextConfig

    model_fp8 = _impl()
    return model_fp8.Glm5NextQuantConfig.from_model_config(
        Glm5NextConfig.from_configs(json.loads(fixture.read_text()))
    )


def _shared_build_block():
    """A ``Glm5NextMoEBlock`` whose shared expert exists, at the tiny config.

    ``world_size=1`` keeps ``-031``'s uniformity gate satisfied at this tiny
    expert count; the routed bank is built but never driven here, because the
    routed path is ``-027``'s landed and separately-accepted surface.

    THE SWIGLU BOUND IS PASSED EXPLICITLY, since ``B22-M1`` repair round 2. The
    shared expert now reads ``text_config.swiglu_limit`` at construction, and the
    dataclass default happens to be the checkpoint's own ``10.0`` -- so a config
    that said nothing would still produce the right number and no arm below could
    tell a config read from a default. Passing the digest-checked vendor value
    removes that accidental agreement: the bound every numeric arm drives the
    shipped path with is the checkpoint's, from the file, by construction.
    """
    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    model_fp8 = _impl()
    text_config = Glm5NextTextConfig(
        hidden_size=SHARED_H,
        moe_intermediate_size=SHARED_I,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        swiglu_limit=_shared_swiglu_limit(),
    )
    return model_fp8.Glm5NextMoEBlock(text_config, world_size=1)


def _shared_swiglu_limit() -> float:
    """The checkpoint's ``text_config.swiglu_limit``, digest-checked first.

    NOT A LITERAL, and that is the point of the function existing at all. The
    finding asks for the bound to come from the checkpoint; a `10.0` typed into
    this file would satisfy the arithmetic and fail the requirement. The digest
    is checked before the file is parsed, so a moved fixture raises instead of
    quietly supplying a different bound.
    """
    import hashlib
    import json
    from pathlib import Path

    vendor = Path(__file__).resolve().parent / "fixtures" / "hf-config.json"
    digest = hashlib.sha256(vendor.read_bytes()).hexdigest()
    if digest != SHARED_VENDOR_CONFIG_SHA256:
        raise SharedVacuousControlError(
            f"the vendor config moved: sha256={digest} != "
            f"{SHARED_VENDOR_CONFIG_SHA256}. The clamp bound every arm below "
            f"uses would no longer be the checkpoint's."
        )
    return float(json.loads(vendor.read_text())["text_config"]["swiglu_limit"])


def _shared_apply_reference_clamps(gate, up, limit: float):
    """The reference's two clamps, transliterated (``:102`` and ``:103``).

    ONE-SIDED ON THE GATE and two-sided on the up operand, because that is what
    the reference does. ``min=None`` is written out rather than dropped so the
    asymmetry is visible at the call site and cannot be "tidied" into a
    symmetric clamp, which would be a different function.
    """
    return gate.clamp(min=None, max=limit), up.clamp(min=-limit, max=limit)


def _shared_swiglu_formula(case: dict, gate, up, *, clamp: bool, limit: float):
    """``down(silu(gate) * up)`` with the clamps switched IN or OUT.

    Both formulas in one function, selected by an argument, so the two arms that
    need to tell them apart cannot drift into comparing two different things by
    accident. The projections are handed in rather than computed here, which is
    what lets one caller pass the ORACLE's projections and another pass the
    KERNEL's.
    """
    from torch.nn.functional import silu

    from vllm_neuron.functional.blockwise_fp8_mm import blockwise_fp8_mm_torch_oracle

    if clamp:
        gate, up = _shared_apply_reference_clamps(gate, up, limit)
    activated = silu(gate) * up
    return blockwise_fp8_mm_torch_oracle(
        activated.to(case["hidden_states"].dtype), case["down_w"], case["down_s"]
    )


def _shared_oracle_projections(case: dict):
    """``gate`` and ``up`` as ``-026``'s torch oracle computes them, pre-clamp."""
    from vllm_neuron.functional.blockwise_fp8_mm import blockwise_fp8_mm_torch_oracle

    hidden = case["hidden_states"]
    return (
        blockwise_fp8_mm_torch_oracle(hidden, case["gate_w"], case["gate_s"]),
        blockwise_fp8_mm_torch_oracle(hidden, case["up_w"], case["up_s"]),
    )


def _shared_straddling_case(base: dict) -> dict:
    """The landed case with its hidden states divided by a power of two.

    See ``SHARED_STRADDLE_DIVISOR`` for why this fixture exists and why the
    divisor is the one it is. Nothing else about the case moves.
    """
    case = dict(base)
    case["hidden_states"] = base["hidden_states"] / SHARED_STRADDLE_DIVISOR
    return case


def _shared_expert_torch_reference(case: dict):
    """The independent torch formulation of ``down(silu(gate(x)) * up(x))``.

    WHY THIS IS A REAL COMPARISON AND NOT A RESTATEMENT. Each projection goes
    through ``-026``'s ``blockwise_fp8_mm_torch_oracle``, which dequantises the
    whole weight FIRST and contracts in one fp32 matmul, where the kernel
    contracts per ``256`` block and applies the block scale BETWEEN blocks and
    never consults ``flat_scale_index`` on this side. The two therefore disagree
    in arithmetic ORDER while agreeing in value, so a transposed block-to-scale
    assignment in the seam shows up as a numeric disagreement here.

    THE DTYPE FLOW IS MIRRORED AT THE SAME POINTS, deliberately. The reference
    applies the same ``.to(bfloat16)`` before the down projection that the
    implementation applies, because that cast is a real precision step and a
    reference that skipped it would make this comparison measure a dtype
    mismatch this test invented rather than the plumbing under acceptance.

    AND IT CLAMPS, as of `B22-M1` repair round 1. This oracle used to compute the
    unclamped product, which is exactly why the landed acceptance read
    ``max_rel_error=0.000000e+00`` while the shipped path computed a different
    function from the checkpoint's: both sides shared one omission. The bound is
    the checkpoint's own, read through :func:`_shared_swiglu_limit`.
    """
    gate, up = _shared_oracle_projections(case)
    return _shared_swiglu_formula(
        case, gate, up, clamp=True, limit=_shared_swiglu_limit()
    )


def _shared_routed_stand_in(shared_reference):
    """A conditioned ``[T, H]`` routed contribution, at the shared half's scale.

    WHY THE ROUTED HALF IS A FIXTURE AND NOT A CALL. The routed path is
    ``-027``'s ``block_quant_expert_mm``, LANDED and separately accepted against
    its own criteria. Driving it here would (i) import ``-025``'s five admission
    gates into this increment's acceptance, so a refusal there would read as a
    failure here, and (ii) put another increment's seam inside this increment's
    counter window. This increment's surface is the shared-expert path and the
    residual add, and ``combine_routed_and_shared`` takes the routed output as an
    argument precisely so the two halves compose without either owning the other.

    THE MAGNITUDE IS MEASURED, NOT HOPED. It is rescaled to the shared half's own
    absmax, which is what makes the double-add control below able to fail: if the
    routed half dominated by orders of magnitude, doubling the shared
    contribution would move the sum by less than ``rtol`` and the control would
    report a pass it had not earned.
    """
    import torch

    routed = _shared_fp8_grid(41, SHARED_T, SHARED_H).to(torch.float32)
    scale = shared_reference.abs().max() / routed.abs().max().clamp_min(1e-12)
    return routed * scale


def _shared_max_rel_error(got, want) -> float:
    """``max |got - want| / (|want| + atol)`` -- a number, not a verdict."""
    return float(((got - want).abs() / (want.abs() + SHARED_ATOL)).max())


def _shared_call_layer(block, case: dict, quant_config, routed):
    """Drive the increment's own call site once, under all four instruments.

    THE BOUND IS NO LONGER AN ARGUMENT, since ``B22-M1`` repair round 2. It is
    read from ``text_config.swiglu_limit`` when the block is built, so the value
    reaching the clamp is the one :func:`_shared_build_block` resolved from the
    checkpoint -- see that helper, which passes the digest-checked vendor value
    rather than letting a dataclass default stand in for it.
    """
    seam = _shared_seam()
    seam.reset_dispatch_counters()
    with _SharedSimulatorCounter() as sim:
        got = block.combine_routed_and_shared(
            routed,
            case["hidden_states"],
            case["gate_w"],
            case["up_w"],
            case["down_w"],
            case["gate_s"],
            case["up_s"],
            case["down_s"],
            quant_config,
        )
    return got, sim


def _shared_source_method(class_name: str, method_name: str):
    """The AST of one method of ``model_fp8.py``, located by name.

    Used by the structural items below. Reads the file rather than
    ``inspect.getsource`` on a bound method so the reading is of the shipped
    bytes at HEAD and not of anything a test import may have rebound.
    """
    source = Path(_impl().__file__).read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for member in node.body:
                if isinstance(member, ast.FunctionDef) and member.name == method_name:
                    return member
    raise SharedSectionOwnershipError(
        f"{class_name}.{method_name} not found in model_fp8.py -- this increment's "
        f"own section is missing, so every structural reading below would be "
        f"vacuous"
    )


def _shared_count_calls(fn, callee: str) -> int:
    """How many times ``fn`` calls the function or method named ``callee``."""
    total = 0
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name == callee:
            total += 1
    return total


# ---------------------------------------------------------------------------
# H01 -- DECLARED CONJUNCT 1: layer output == routed + shared, rtol 3e-2 / atol 1e-5
# ---------------------------------------------------------------------------


def test_shared_expert_layer_output_equals_routed_plus_shared():
    """The plan's first declared conjunct, at the plan's declared tolerances.

    The comparison is against an independent torch formulation (see
    :func:`_shared_expert_torch_reference`), and the route is read in the SAME
    window as the numbers, because under F1 a numeric pass alone cannot prove the
    substrate ran.
    """
    import torch

    case = _shared_build_case()
    block = _shared_build_block()
    quant_config = _shared_block_quant_config()

    shared_reference = _shared_expert_torch_reference(case)
    routed = _shared_routed_stand_in(shared_reference)
    want = routed + shared_reference

    got, sim = _shared_call_layer(block, case, quant_config, routed)
    reading = _assert_shared_route(sim, SHARED_DECLARED_SEAM_ENTRIES, "acceptance")

    # NON-VACUITY GATE. An all-zero reference would make assert_close pass on a
    # function that returns zeros, so the comparison refuses to run over one.
    nonzero_rows = int((want.abs().sum(dim=1) > 0).sum())
    if nonzero_rows != SHARED_T:
        raise SharedVacuousControlError(
            f"only {nonzero_rows}/{SHARED_T} reference rows are nonzero; a "
            f"tolerance over a vacuous reference measures nothing"
        )

    max_rel = _shared_max_rel_error(got, want)
    max_abs = float((got - want).abs().max())
    print(
        f"[acceptance] cases=1/1 nonzero_reference_rows={nonzero_rows}/{SHARED_T} "
        f"max_rel_error={max_rel:.6e} max_abs_error={max_abs:.6e} "
        f"rtol={SHARED_RTOL} atol={SHARED_ATOL} "
        f"reference_absmax={float(want.abs().max()):.6e}"
    )
    torch.testing.assert_close(got, want, rtol=SHARED_RTOL, atol=SHARED_ATOL)

    assert tuple(got.shape) == (SHARED_T, SHARED_H)
    _shared_record(
        h01_max_rel_error=f"{max_rel:.6e}",
        h01_max_abs_error=f"{max_abs:.6e}",
        h01_rtol=SHARED_RTOL,
        h01_atol=SHARED_ATOL,
        h01_nonzero_reference_rows=f"{nonzero_rows}/{SHARED_T}",
        h01_route_reading=reading,
        h01_cases="1/1",
    )


# ---------------------------------------------------------------------------
# H02 -- DECLARED CONJUNCT 2: zeroed shared reproduces routed-only at atol 1e-5
# ---------------------------------------------------------------------------


def test_shared_expert_zeroed_case_reproduces_routed_only():
    """The plan's second declared conjunct: the shared half is added exactly once.

    ``rtol`` is pinned to ``0`` because the plan declares ONLY ``atol 1e-5`` for
    this conjunct; ``0`` is the strictest reading of that declaration and the one
    that cannot be mistaken for a widening. The shared contribution is separately
    shown to be EXACTLY zero, so "reproduces" is an exact statement here and the
    tolerance is headroom rather than the thing being used.

    The route is read in the same window: all three seam entries still happen on
    zeroed weights, which is what makes the count a statement about the ROUTE and
    not a proxy for the output being nonzero.
    """
    import torch

    case = _shared_build_case(zero_shared=True)
    block = _shared_build_block()
    quant_config = _shared_block_quant_config()

    shared_reference = _shared_expert_torch_reference(case)
    shared_absmax = float(shared_reference.abs().max())

    # The routed-only output. Built from the NONZERO case's shared scale so this
    # arm's routed operand has the same magnitude as H01's -- otherwise "equals
    # routed-only" could be a comparison of two tiny tensors.
    routed = _shared_routed_stand_in(_shared_expert_torch_reference(_shared_build_case()))

    # NON-VACUITY GATE on the routed half: if it were zero, this arm would be
    # comparing zero against zero and would pass for the wrong reason.
    nonzero_rows = int((routed.abs().sum(dim=1) > 0).sum())
    if nonzero_rows != SHARED_T:
        raise SharedVacuousControlError(
            f"only {nonzero_rows}/{SHARED_T} routed rows are nonzero; "
            f"'reproduces the routed-only output' would be vacuous"
        )

    got, sim = _shared_call_layer(block, case, quant_config, routed)
    reading = _assert_shared_route(sim, SHARED_DECLARED_SEAM_ENTRIES, "zeroed-shared")

    max_abs = float((got - routed).abs().max())
    print(
        f"[zeroed-shared] shared_absmax={shared_absmax:.6e} "
        f"nonzero_routed_rows={nonzero_rows}/{SHARED_T} "
        f"max_abs_error_vs_routed_only={max_abs:.6e} atol={SHARED_ATOL} rtol=0"
    )
    assert shared_absmax == 0.0, (
        f"the zeroed shared expert produced a nonzero contribution "
        f"({shared_absmax}); the zeroing did not take effect"
    )
    torch.testing.assert_close(got, routed, rtol=0, atol=SHARED_ATOL)

    _shared_record(
        h02_shared_absmax=f"{shared_absmax:.6e}",
        h02_max_abs_error_vs_routed_only=f"{max_abs:.6e}",
        h02_atol=SHARED_ATOL,
        h02_rtol=0,
        h02_nonzero_routed_rows=f"{nonzero_rows}/{SHARED_T}",
        h02_route_reading=reading,
    )


# ---------------------------------------------------------------------------
# H03 -- D1.5: the declared tolerance can DETECT a double add
# ---------------------------------------------------------------------------


def test_shared_expert_double_add_is_refused_by_the_declared_tolerance():
    """Add the shared half twice; the DECLARED comparison must raise.

    This is the arm that makes "added exactly once" a measurement rather than a
    claim. Without it, H01's pass would be indistinguishable from a comparison
    that cannot fail. No tolerance is changed: the control uses the declared pair,
    and the magnitude of the perturbation is measured and reported so the margin
    is visible rather than assumed.
    """
    import torch

    case = _shared_build_case()
    shared_reference = _shared_expert_torch_reference(case)
    routed = _shared_routed_stand_in(shared_reference)

    once = routed + shared_reference
    twice = routed + 2.0 * shared_reference

    perturbation = _shared_max_rel_error(twice, once)
    print(
        f"[double-add-control] max_rel_error_of_double_add={perturbation:.6e} "
        f"rtol={SHARED_RTOL} atol={SHARED_ATOL}"
    )
    if perturbation <= SHARED_RTOL:
        raise SharedVacuousControlError(
            f"doubling the shared contribution moved the sum by only "
            f"{perturbation:.6e}, which is inside rtol={SHARED_RTOL}. This "
            f"control could not have failed, so it certifies nothing about the "
            f"once-versus-twice property."
        )

    with pytest.raises(AssertionError):
        torch.testing.assert_close(twice, once, rtol=SHARED_RTOL, atol=SHARED_ATOL)

    _shared_record(
        h03_double_add_max_rel_error=f"{perturbation:.6e}",
        h03_declared_comparison_refused_double_add=True,
    )


# ---------------------------------------------------------------------------
# H04 -- THE ROUTE PREDICATE: 3 per call, one per projection site (plan L934-935)
# ---------------------------------------------------------------------------


def test_shared_expert_seam_entries_are_one_per_projection_site():
    """``-026``'s counter reads 3 per shared-expert call, and 3 is per-call.

    BOTH READINGS ARE RECORDED, which is the reviewer's round-26 item N2: the
    PER-CALL value (3) and the PER-CASE total with the case's call multiplicity
    stated, so the case-level number is recorded rather than silently chosen.
    Two calls inside one reset window must read 6 -- that is what shows 3 is a
    per-call delta and not a constant the counter happens to sit at.
    """
    case = _shared_build_case()
    block = _shared_build_block()
    quant_config = _shared_block_quant_config()
    routed = _shared_routed_stand_in(_shared_expert_torch_reference(case))
    seam = _shared_seam()

    seam.reset_dispatch_counters()
    assert seam.dispatch_counters() == (0, 0), (
        "the reset did not zero -026's counters, so every reading below would be "
        "cumulative and none of them would mean what it says"
    )

    readings = []
    with _SharedSimulatorCounter() as sim:
        for call_index in range(2):
            block.combine_routed_and_shared(
                routed,
                case["hidden_states"],
                case["gate_w"],
                case["up_w"],
                case["down_w"],
                case["gate_s"],
                case["up_s"],
                case["down_s"],
                quant_config,
            )
            readings.append(seam.dispatch_counters())
            print(
                f"[route-arity] after_call={call_index + 1} "
                f"counters={readings[-1]} simulate_kernel_calls={sim.calls}"
            )

    per_call = [readings[0][0], readings[1][0] - readings[0][0]]
    if readings[0] != (SHARED_DECLARED_SEAM_ENTRIES, 0):
        raise SharedRouteInstrumentError(
            f"after one shared-expert call -026's counters read {readings[0]}, "
            f"declared ({SHARED_DECLARED_SEAM_ENTRIES}, 0)"
        )
    if readings[1] != (2 * SHARED_DECLARED_SEAM_ENTRIES, 0):
        raise SharedRouteInstrumentError(
            f"after two calls -026's counters read {readings[1]}, expected "
            f"({2 * SHARED_DECLARED_SEAM_ENTRIES}, 0). The declared 3 is a "
            f"PER-CALL delta; a counter that cannot advance is not an instrument."
        )
    assert per_call == [SHARED_DECLARED_SEAM_ENTRIES, SHARED_DECLARED_SEAM_ENTRIES]
    assert sim.calls == 2 * SHARED_DECLARED_SEAM_ENTRIES

    # The structural reading behind the number: three call sites in the source.
    method = _shared_source_method("Glm5NextSharedExperts", "shared_expert_mm")
    source_entries = _shared_count_calls(method, "blockwise_fp8_mm")
    assert source_entries == SHARED_DECLARED_SEAM_ENTRIES, (
        f"shared_expert_mm contains {source_entries} calls to blockwise_fp8_mm, "
        f"declared {SHARED_DECLARED_SEAM_ENTRIES} (gate, up, down)"
    )

    _shared_record(
        h04_per_call_deltas=per_call,
        h04_per_call_declared=SHARED_DECLARED_SEAM_ENTRIES,
        h04_case_call_multiplicity=2,
        h04_per_case_total=readings[1][0],
        h04_declared_case_multiplicity_for_h01=1,
        h04_declared_case_total_for_h01=SHARED_DECLARED_SEAM_ENTRIES,
        h04_torch_fallback=readings[1][1],
        h04_simulate_kernel_calls=sim.calls,
        h04_source_call_sites=source_entries,
    )


# ---------------------------------------------------------------------------
# H05 -- D1.5: the (3, 0) reading is a measurement, not an always-3 counter
# ---------------------------------------------------------------------------


def test_shared_expert_route_control_fallback_counter_discriminates(monkeypatch):
    """With the simulator off, the same call reads ``(0, 3)`` and the route FAILS.

    This is what makes ``torch_fallback == 0`` and ``nki_dispatch == 3`` above
    meaningful: the same instruments are shown reading the opposite values
    through the real gate rather than a mock, and ``_assert_shared_route`` is
    shown REFUSING that reading. It is also the measured form of the plan's claim
    that a pure-torch shared expert cannot pass.
    """
    import os

    import torch

    from vllm_neuron.utils.neuron_utils import can_run_kernel

    case = _shared_build_case()
    block = _shared_build_block()
    quant_config = _shared_block_quant_config()
    routed = _shared_routed_stand_in(_shared_expert_torch_reference(case))

    monkeypatch.setitem(os.environ, "NKI_SIMULATOR", "0")
    assert can_run_kernel(torch.zeros(1)) is False, (
        "the gate did not flip with NKI_SIMULATOR=0, so this control is unarmed"
    )

    got, sim = _shared_call_layer(block, case, quant_config, routed)
    counters = _shared_seam().dispatch_counters()
    print(
        f"[route-control] counters={counters} simulate_kernel_calls={sim.calls} "
        f"can_run_kernel=False"
    )

    assert counters == (0, SHARED_DECLARED_SEAM_ENTRIES), (
        f"expected (0, {SHARED_DECLARED_SEAM_ENTRIES}) on the fallback path, got "
        f"{counters}"
    )
    assert sim.calls == 0, f"the simulator ran {sim.calls} times with it disabled"
    assert tuple(got.shape) == (SHARED_T, SHARED_H)

    # The route assertion must REFUSE this reading. Without this leg the control
    # would show the counters moving but not that the acceptance notices.
    with pytest.raises(SharedRouteInstrumentError):
        _assert_shared_route(sim, SHARED_DECLARED_SEAM_ENTRIES, "route-control")

    _shared_record(
        h05_fallback_counters=counters,
        h05_fallback_simulate_kernel_calls=sim.calls,
        h05_route_assertion_refused_the_fallback=True,
    )


# ---------------------------------------------------------------------------
# H06 -- "exactly once" is STRUCTURAL, not only numeric
# ---------------------------------------------------------------------------


def test_shared_expert_add_is_structurally_exactly_once():
    """One call to the shared path and one ``+`` in ``combine_routed_and_shared``.

    The numeric conjuncts measure the once-versus-twice property on one fixture.
    This item settles it over the SOURCE, so the property does not depend on the
    fixture having been chosen well: if a second add or a second shared call ever
    appears, this reading moves even when the numbers happen to still agree.
    """
    method = _shared_source_method("Glm5NextMoEBlock", "combine_routed_and_shared")

    shared_calls = _shared_count_calls(method, "shared_expert_mm")
    adds = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add)
    ]
    aug_adds = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Add)
    ]
    print(
        f"[structure] shared_expert_mm_calls={shared_calls} add_binops={len(adds)} "
        f"augmented_adds={len(aug_adds)}"
    )

    if shared_calls != 1:
        raise SharedSectionOwnershipError(
            f"combine_routed_and_shared calls shared_expert_mm {shared_calls} "
            f"times, declared exactly 1"
        )
    if len(adds) != 1 or aug_adds:
        raise SharedSectionOwnershipError(
            f"combine_routed_and_shared contains {len(adds)} '+' expressions and "
            f"{len(aug_adds)} '+=' statements, declared exactly one '+' and no "
            f"'+='. Two adds is the defect the plan's second conjunct exists to "
            f"exclude."
        )

    _shared_record(
        h06_shared_calls_in_combine=shared_calls,
        h06_add_binops_in_combine=len(adds),
        h06_augmented_adds_in_combine=len(aug_adds),
    )


# ---------------------------------------------------------------------------
# H07 -- the two same-named scale helpers: neither is imported by this section
# ---------------------------------------------------------------------------


def test_shared_expert_section_imports_neither_scale_layout_helper():
    """This increment's section imports no ``to_kernel_scale_layout`` at all.

    The campaign carries two helpers of that name at different arities
    (``functional/blockwise_fp8_mm.py:309`` takes ``(weight_scale, rows, cols)``;
    ``functional/moe/moe_blockwise_fp8.py:170`` takes ``(consumer_scales,
    num_experts, rows, cols, projection)``), and they are deliberately not
    flat-exported. This section removes the hazard instead of navigating it: it
    passes the PUBLIC scale grid and lets ``blockwise_fp8_mm`` apply the dense
    helper itself. Asserted mechanically so a later edit cannot quietly
    reintroduce the ambiguity -- and, in particular, cannot reach the 5-arg MoE
    helper from the dense path, where an arity failure reads as a shape bug.
    """
    names: list[str] = []
    for class_name, method_name in (
        ("Glm5NextSharedExperts", "shared_expert_mm"),
        ("Glm5NextMoEBlock", "combine_routed_and_shared"),
    ):
        method = _shared_source_method(class_name, method_name)
        for node in ast.walk(method):
            if isinstance(node, ast.ImportFrom):
                names += [f"{node.module}.{alias.name}" for alias in node.names]
            elif isinstance(node, ast.Import):
                names += [alias.name for alias in node.names]

    print(f"[imports] this_section_imports={names}")
    offenders = [name for name in names if "to_kernel_scale_layout" in name]
    moe_offenders = [name for name in names if "moe_blockwise_fp8" in name]

    # Non-vacuity: the scan must have read a real, non-empty import list.
    if not names:
        raise SharedVacuousControlError(
            "no imports were found in this increment's section, so the scan read "
            "nothing and its zero certifies nothing"
        )
    assert offenders == [], (
        f"this section imports a to_kernel_scale_layout helper: {offenders}. "
        f"blockwise_fp8_mm applies the dense one itself at :436."
    )
    assert moe_offenders == [], (
        f"this section imports from the MoE scale module: {moe_offenders}. The "
        f"dense path must not reach the 5-arg MoE helper."
    )

    _shared_record(
        h07_section_import_count=len(names),
        h07_scale_layout_imports=len(offenders),
        h07_moe_scale_module_imports=len(moe_offenders),
    )


# ---------------------------------------------------------------------------
# H08 -- F1: every block scale this acceptance runs on is an exact power of two
# ---------------------------------------------------------------------------


def test_shared_expert_f1_precondition_block_scales_are_pow2():
    """All three projections' block scales are exact pow2, over N/N blocks.

    Why this belongs in this file: ``-026``'s kernel applies the block scale AFTER
    accumulating the two contraction tiles of a block, and
    ``increments/evidence-071.md`` F1 measured 720 fp32 ulp of remapping error
    under a non-pow2 block scale against 0 under a pow2 one. Asserted HERE rather
    than inherited from ``-026``'s test, because it is this file's fixture that
    the declared tolerance is measured on.
    """
    from vllm_neuron.functional.moe.blockwise_fp8_retile import is_pow2_exact

    case = _shared_build_case()
    checked = 0
    distinct: set[float] = set()
    for label in ("gate_s", "up_s", "down_s"):
        grid = case[label]
        for value in grid.reshape(-1).tolist():
            checked += 1
            distinct.add(value)
            if not is_pow2_exact(value):
                raise SharedF1PreconditionError(
                    f"{label} carries {value!r}, which is not an exact power of "
                    f"two; the declared rtol would then certify remapping error "
                    f"on top of this increment's plumbing"
                )

    print(
        f"[f1] pow2_block_scales={checked}/{checked} distinct_values={len(distinct)}"
    )
    if checked == 0:
        raise SharedVacuousControlError("no block scales were checked")
    # The detector must be able to say no -- otherwise the N/N above is a
    # tautology about is_pow2_exact rather than about this fixture.
    assert not is_pow2_exact(3.0), "the pow2 detector accepts a non-pow2 value"
    assert len(distinct) > 1, (
        "every block scale is the same value, so a transposed block-to-scale "
        "assignment would be numerically invisible in this fixture"
    )

    _shared_record(
        h08_pow2_block_scales=f"{checked}/{checked}",
        h08_distinct_block_scales=len(distinct),
        h08_detector_rejects_non_pow2=True,
    )


# ---------------------------------------------------------------------------
# H09 -- the named refusal: no silent path to QuantizationType.NONE (B.6)
# ---------------------------------------------------------------------------


def test_shared_expert_refuses_a_non_block_quant_config_by_name():
    """An unquantised ``quant_config`` raises by name rather than falling through.

    The failure this closes is a call site that reaches the substrate's
    ``QuantizationType.NONE`` default by OMISSION (``functional/mlp.py:81``,
    ``:249``) and computes a different function while every shape check passes.
    The route-counter clause DETECTS that; this raise makes it impossible.

    No quantisation enum member is named or added anywhere in this increment
    (constraint B.6) -- the route is selected by which function is called.
    """
    case = _shared_build_case()
    block = _shared_build_block()
    model_fp8 = _impl()

    unquantised = model_fp8.Glm5NextQuantConfig(None)
    if unquantised.is_block_quantized:
        raise SharedVacuousControlError(
            "Glm5NextQuantConfig(None) reports is_block_quantized=True, so this "
            "control could not fire"
        )

    seam = _shared_seam()
    seam.reset_dispatch_counters()
    with pytest.raises(model_fp8.Glm5NextSharedExpertRouteError, match="block-quant"):
        block.combine_routed_and_shared(
            _shared_routed_stand_in(_shared_expert_torch_reference(case)),
            case["hidden_states"],
            case["gate_w"],
            case["up_w"],
            case["down_w"],
            case["gate_s"],
            case["up_s"],
            case["down_s"],
            unquantised,
        )
    refused_counters = seam.dispatch_counters()
    print(f"[refusal] counters_after_refusal={refused_counters}")
    assert refused_counters == (0, 0), (
        f"the refusal ran {refused_counters} seam entries; it must refuse BEFORE "
        f"touching the seam"
    )

    # B.6, over this increment's own two methods. The walk counts ``Name`` /
    # ``Attribute`` / ``ImportFrom`` nodes ONLY -- ``-031``'s landed
    # ``test_sharding_adds_no_vendor_quantisation_enum_reference`` is the
    # authority for that scoping and the reason is load-bearing here: this
    # section NAMES the enum in prose, in the refusal message that explains which
    # default-by-omission it exists to block (``-027``'s landed raise carries the
    # same sentence). A raw-text count would score that prose as a violation and
    # would pressure the message to be made less clear to satisfy the scan, which
    # inverts what B.6 is for. A CODE reference is what B.6 forbids.
    def enum_refs(node_tree) -> int:
        hits = 0
        for node in ast.walk(node_tree):
            if isinstance(node, ast.Name) and node.id == FORBIDDEN_ENUM:
                hits += 1
            elif isinstance(node, ast.Attribute) and node.attr == FORBIDDEN_ENUM:
                hits += 1
            elif isinstance(node, ast.ImportFrom):
                hits += sum(1 for a in node.names if a.name == FORBIDDEN_ENUM)
        return hits

    enum_hits = 0
    prose_mentions = 0
    for class_name, method_name in (
        ("Glm5NextSharedExperts", "shared_expert_mm"),
        ("Glm5NextMoEBlock", "combine_routed_and_shared"),
    ):
        method = _shared_source_method(class_name, method_name)
        enum_hits += enum_refs(method)
        prose_mentions += ast.unparse(method).count(FORBIDDEN_ENUM)

    # The control must FIRE, over real non-empty input, or the zero above is a
    # statement about the walk rather than about this section.
    control = enum_refs(
        ast.parse(
            f"from nkilib.core.utils.common_types import {FORBIDDEN_ENUM}\n"
            f"x = {FORBIDDEN_ENUM}.NONE\n"
        )
    )
    print(
        f"[b6] enum_code_references_in_this_section={enum_hits} "
        f"prose_mentions={prose_mentions} control={control}"
    )
    assert enum_hits == 0
    assert control == 2, (
        f"the B.6 walk scored {control} on a source that really does reference "
        f"the enum twice; the zero above would certify nothing"
    )
    if prose_mentions == 0:
        raise SharedVacuousControlError(
            "this section mentions the enum in no prose at all, so the "
            "code-versus-prose distinction this scan draws is untested here"
        )

    _shared_record(
        h09_refusal_error="Glm5NextSharedExpertRouteError",
        h09_counters_after_refusal=refused_counters,
        h09_enum_code_references_in_this_section=enum_hits,
        h09_enum_prose_mentions=prose_mentions,
        h09_b6_control=control,
    )


# ---------------------------------------------------------------------------
# H10 -- signed coverage, at the SAME declared tolerances
# ---------------------------------------------------------------------------


def test_shared_expert_signed_fixture_agrees_in_norm_under_cancellation():
    """A signed fixture, compared in a cancellation-robust norm at the same tolerances.

    The declared arms run on a positive fixture, and
    :func:`_shared_fp8_grid` records why: ``-025`` attempt 1 died to catastrophic
    cancellation at this same ``rtol``, and this increment chains THREE
    contractions so cancellation compounds. Dropping signed coverage entirely
    would leave the sign path unexercised, so it is kept here and compared in a
    RELATIVE FROBENIUS NORM -- which is dominated by the bulk of the tensor rather
    than by whichever single element happened to land near zero.

    THE TOLERANCES ARE THE PLAN'S, UNCHANGED. Only the norm the residual is
    measured in differs from H01's pointwise form, and that difference is the
    point of this arm rather than a relaxation of it.
    """
    case = _shared_build_case(signed=True)
    block = _shared_build_block()
    quant_config = _shared_block_quant_config()

    shared_reference = _shared_expert_torch_reference(case)
    routed = _shared_routed_stand_in(shared_reference)
    want = routed + shared_reference

    got, sim = _shared_call_layer(block, case, quant_config, routed)
    reading = _assert_shared_route(sim, SHARED_DECLARED_SEAM_ENTRIES, "signed")

    reference_norm = float(want.norm())
    if reference_norm == 0.0:
        raise SharedVacuousControlError("the signed reference is identically zero")
    relative_norm = float((got - want).norm()) / reference_norm
    pointwise = _shared_max_rel_error(got, want)
    print(
        f"[signed] relative_frobenius={relative_norm:.6e} rtol={SHARED_RTOL} "
        f"pointwise_max_rel_error={pointwise:.6e} (recorded, not asserted) "
        f"reference_norm={reference_norm:.6e}"
    )
    assert relative_norm <= SHARED_RTOL, (
        f"signed fixture disagrees in norm: {relative_norm:.6e} > "
        f"rtol={SHARED_RTOL}"
    )

    _shared_record(
        h10_signed_relative_frobenius=f"{relative_norm:.6e}",
        h10_signed_pointwise_max_rel_error=f"{pointwise:.6e}",
        h10_signed_route_reading=reading,
    )


# ---------------------------------------------------------------------------
# `B22-M1` REPAIR -- four added arms. Rounds 1 and 2.
#
# ROUND 2 REWROTE R01 AND LEFT THE OTHER THREE ALONE. Round 1 passed the SwiGLU
# bound into the two shipped methods as a required argument, because
# `Glm5NextTextConfig` did not model `swiglu_limit`; round 2 lifted the field and
# moved the read to `Glm5NextSharedExperts.__init__`, so R01 now asks whether the
# value on the object came from the config rather than whether an argument has a
# default. R02, R03 and R04 are numeric arms over the shipped path and do not
# name the parameter, so none of their readings moves.
#
# R01 the bound is READ FROM THE CONFIG and no literal governs it, and the
#     shipped clamps keep the reference's asymmetry.
# R02 the shipped path matches the CLAMPED formula and NOT the unclamped one --
#     the arm the finding asks for, and the one that goes red if either clamp is
#     dropped from the shipped path.
# R03 a straddling fixture keeps the gate projection visible, which is the
#     coverage the clamp removes from the landed fixture.
# R04 the two-sided lower bound is reached by the landed signed fixture, and
#     dropping it changes the answer.
#
# Every one is an addition. No landed item, tolerance, extent, seed, exponent or
# comparator moves.
# ---------------------------------------------------------------------------


def test_shared_expert_swiglu_bound_is_the_checkpoints_and_no_literal_governs_it():
    """R01. The bound is READ FROM THE CONFIG, and the clamps stay asymmetric.

    REWRITTEN AT REPAIR ROUND 2, because the thing it checks changed. Round 1
    passed the bound in as a required argument, because `Glm5NextTextConfig` did
    not model `swiglu_limit` at all; round 2 added the field and moved the read to
    `Glm5NextSharedExperts.__init__`. So this arm no longer asks whether an
    argument has a default -- there is no argument -- it asks whether the value on
    the object came from the config.

    FOUR READINGS, because "sourced from the checkpoint" has four ways to fail and
    a value check alone catches only one of them.

    Reading 1 -- the value in the published config is `10.0`, read from the
    digest-checked vendor copy, and the block the section builds resolves that
    same number. Equality here is necessary but weak on its own, which is what
    reading 2 exists for.

    Reading 2 -- THE PROVENANCE READING, on a value the checkpoint does not carry.
    `10.0` is also any natural default, so a `10.0` on the object proves nothing
    about where it came from. This reading pushes a NON-DEFAULT `7.5` through the
    real adapter (`Glm5NextTextConfig.from_hf_config`, the same path production
    uses) and asserts the built block resolved `7.5` -- and that the shipped
    clamp then computes a different answer, so the field governs the arithmetic
    and is not just an attribute nobody reads.

    Reading 3 -- neither shipped method takes `swiglu_limit` as a parameter any
    more, and no numeric literal equal to the bound appears in either. Together
    those say the bound cannot enter the clamp from anywhere except the config.

    Reading 4 -- the shipped clamps keep the reference's ASYMMETRY. The gate is
    bounded above only (`min=None`) and the up operand on both sides, read off
    the shipped AST rather than off a docstring. A symmetric gate clamp computes
    a different function and no numeric arm at this fixture would notice, because
    every pre-activation here is positive.
    """
    import ast
    import copy
    import hashlib
    import inspect
    import json
    from pathlib import Path

    import torch

    from vllm_neuron.model.glm5_next.config import Glm5NextTextConfig

    model_fp8 = _impl()
    limit = _shared_swiglu_limit()
    print(f"\n[swiglu-provenance] checkpoint_swiglu_limit={limit}")

    # Reading 1: what the block this section builds actually resolved.
    block = _shared_build_block()
    resolved = block.shared_experts.swiglu_limit
    print(f"[swiglu-provenance] block_resolved_swiglu_limit={resolved}")

    # Reading 2: the read path, on a value the checkpoint does not carry. The
    # vendor dict is digest-checked by `_shared_swiglu_limit` above; it is read
    # again here because this reading needs the whole sub-config, not one value.
    vendor_path = Path(__file__).resolve().parent / "fixtures" / "hf-config.json"
    assert (
        hashlib.sha256(vendor_path.read_bytes()).hexdigest()
        == SHARED_VENDOR_CONFIG_SHA256
    )
    vendor_text = json.loads(vendor_path.read_text())["text_config"]
    probe_bound = R01_NON_DEFAULT_BOUND
    mutated = copy.deepcopy(vendor_text)
    mutated["swiglu_limit"] = probe_bound
    # The tiny shape this section builds at, so the block stays buildable; only
    # the bound differs between the two configs below.
    tiny = {
        "hidden_size": SHARED_H,
        "moe_intermediate_size": SHARED_I,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "n_shared_experts": 1,
    }
    mutated.update(tiny)
    at_checkpoint = copy.deepcopy(vendor_text)
    at_checkpoint.update(tiny)
    non_default_cfg = Glm5NextTextConfig.from_hf_config(mutated)
    checkpoint_cfg = Glm5NextTextConfig.from_hf_config(at_checkpoint)
    non_default_block = model_fp8.Glm5NextMoEBlock(non_default_cfg, world_size=1)
    checkpoint_block = model_fp8.Glm5NextMoEBlock(checkpoint_cfg, world_size=1)
    read_through_adapter = non_default_block.shared_experts.swiglu_limit
    print(
        f"[swiglu-provenance] dataclass_default={Glm5NextTextConfig().swiglu_limit} "
        f"probe_bound={probe_bound} read_through_the_adapter={read_through_adapter}"
    )

    # ... and the bound governs the arithmetic, not just the attribute. The same
    # case is run through both blocks, through this section's own call site, and
    # the two answers are compared. Both runs go through all four route
    # instruments, so neither reading can come from a torch fallback.
    case = _shared_build_case()
    quant_config = _shared_block_quant_config()
    routed = torch.zeros(
        case["hidden_states"].shape[0], SHARED_H, dtype=torch.float32
    )
    at_probe, sim_probe = _shared_call_layer(
        non_default_block, case, quant_config, routed
    )
    reading_probe = _assert_shared_route(
        sim_probe, SHARED_DECLARED_SEAM_ENTRIES, "swiglu-provenance-probe"
    )
    at_ten, sim_ten = _shared_call_layer(
        checkpoint_block, case, quant_config, routed
    )
    reading_ten = _assert_shared_route(
        sim_ten, SHARED_DECLARED_SEAM_ENTRIES, "swiglu-provenance-checkpoint"
    )
    bound_response = _shared_max_rel_error(at_ten, at_probe)
    print(
        f"[swiglu-provenance] output_response_to_the_bound={bound_response:.6e} "
        f"rtol={SHARED_RTOL}"
    )

    # Reading 3: the parameter is gone from both methods.
    signatures = {}
    for owner, method in (
        (model_fp8.Glm5NextSharedExperts, "shared_expert_mm"),
        (model_fp8.Glm5NextMoEBlock, "combine_routed_and_shared"),
    ):
        parameter = inspect.signature(getattr(owner, method)).parameters.get(
            "swiglu_limit"
        )
        signatures[method] = parameter
        print(
            f"[swiglu-provenance] {method}: swiglu_limit_parameter="
            f"{'ABSENT' if parameter is None else parameter!r}"
        )

    # Reading 4: the clamp calls, read off the shipped source.
    method_ast = _shared_source_method("Glm5NextSharedExperts", "shared_expert_mm")
    clamps = {}
    for node in ast.walk(method_ast):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "clamp"):
            continue
        target = getattr(node.func.value, "id", "?")
        clamps[target] = {kw.arg: ast.unparse(kw.value) for kw in node.keywords}
    print(f"[swiglu-provenance] shipped_clamp_calls={clamps}")

    # A literal equal to the bound anywhere in the shipped method would mean the
    # config read is decoration. Counted over BOTH methods, and each count is
    # printed beside its population so a zero is not read as an empty search.
    literals = {}
    for class_name, method_name in (
        ("Glm5NextSharedExperts", "shared_expert_mm"),
        ("Glm5NextMoEBlock", "combine_routed_and_shared"),
    ):
        node_ast = _shared_source_method(class_name, method_name)
        constants = [
            node.value
            for node in ast.walk(node_ast)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        ]
        literals[method_name] = constants
        print(
            f"[swiglu-provenance] numeric_constants_in_{method_name}={constants} "
            f"equal_to_the_bound="
            f"{[c for c in constants if float(c) in (limit, probe_bound)]}"
        )

    assert limit == 10.0, (
        f"the published checkpoint carries swiglu_limit={limit}; every reading "
        f"in this section is taken against the checkpoint's own value"
    )
    # Reading 1.
    assert resolved == limit, (
        f"the block this section builds resolved swiglu_limit={resolved}, but the "
        f"checkpoint declares {limit}"
    )
    # Reading 2 -- the one that proves provenance rather than agreement.
    assert probe_bound != Glm5NextTextConfig().swiglu_limit, (
        "the probe bound equals the dataclass default, so it could not tell a "
        "config read apart from a default and this reading would be vacuous"
    )
    assert read_through_adapter == probe_bound, (
        f"a config carrying swiglu_limit={probe_bound} built a shared expert "
        f"holding {read_through_adapter}; the value is not coming from the config"
    )
    assert bound_response > SHARED_RTOL, (
        f"changing the checkpoint's bound from {limit} to {probe_bound} moved the "
        f"shipped output by {bound_response:.6e}, which is inside the declared "
        f"rtol={SHARED_RTOL}. The field would then be an attribute nobody reads."
    )
    assert torch.isfinite(at_probe).all() and torch.isfinite(at_ten).all()
    # Reading 3.
    for method, parameter in signatures.items():
        assert parameter is None, (
            f"{method} still declares a swiglu_limit parameter ({parameter!r}). "
            f"Round 2 retires it: the bound is read from the config at "
            f"construction, so a caller cannot supply a different one."
        )
    # Reading 4.
    assert clamps.get("gate") == {"min": "None", "max": "self.swiglu_limit"}, (
        f"the shipped gate clamp reads {clamps.get('gate')}; the reference bounds "
        f"the gate ABOVE ONLY (modeling_glm5_next.py:102) with the config's bound"
    )
    assert clamps.get("up") == {
        "min": "-self.swiglu_limit",
        "max": "self.swiglu_limit",
    }, (
        f"the shipped up clamp reads {clamps.get('up')}; the reference bounds the "
        f"up operand on BOTH sides (modeling_glm5_next.py:103)"
    )
    for method_name, constants in literals.items():
        assert [c for c in constants if float(c) in (limit, probe_bound)] == [], (
            f"{method_name} carries a numeric literal equal to the bound, so the "
            f"config read may be decoration"
        )
    assert literals["shared_expert_mm"], (
        "no numeric constant at all was found in the shipped method, so the "
        "search above proves nothing -- the AST read is broken, not the code"
    )

    _shared_record(
        r01_checkpoint_swiglu_limit=limit,
        r01_block_resolved_swiglu_limit=resolved,
        r01_dataclass_default=Glm5NextTextConfig().swiglu_limit,
        r01_probe_bound=probe_bound,
        r01_read_through_the_adapter=read_through_adapter,
        r01_output_response_to_the_bound=f"{bound_response:.6e}",
        r01_route_reading_at_the_probe_bound=reading_probe,
        r01_route_reading_at_the_checkpoint_bound=reading_ten,
        r01_shared_expert_mm_parameter="ABSENT",
        r01_combine_parameter="ABSENT",
        r01_gate_clamp=str(clamps.get("gate")),
        r01_up_clamp=str(clamps.get("up")),
        r01_literals_equal_to_the_bound=0,
        r01_numeric_constants_examined=len(literals["shared_expert_mm"]),
    )


def test_shared_expert_matches_the_clamped_formula_and_not_the_unclamped_one():
    """R02. The arm the finding asks for: red on the unclamped form, green on the
    clamped one.

    WHY THIS IS NOT A RESTATEMENT OF H01. H01 compares the shipped path against
    ``_shared_expert_torch_reference``. Before this repair BOTH of those computed
    the unclamped product, so H01 read an exact zero and said nothing -- that is
    the finding. This arm builds BOTH formulas itself, through
    :func:`_shared_swiglu_formula`, and asserts the shipped path agrees with one
    and disagrees with the other. Deleting the clamp from the oracle does not
    reach this arm's reference, so the pre-repair state cannot make it green.

    THE NON-VACUITY READING COMES FIRST. If the two formulas agreed at this
    fixture, the second assertion would be unfalsifiable. They differ here by
    more than four thousand times the declared ``rtol``, and the pre-activation
    ranges printed beside it are the reason: every element of both operands sits
    outside the checkpoint's box.
    """
    import torch

    limit = _shared_swiglu_limit()
    case = _shared_build_case()
    gate, up = _shared_oracle_projections(case)

    above = int((gate > limit).sum())
    outside = int(((up > limit) | (up < -limit)).sum())
    ranges = (
        f"gate=[{float(gate.min()):.4f}, {float(gate.max()):.4f}] "
        f"up=[{float(up.min()):.4f}, {float(up.max()):.4f}]"
    )
    print(f"\n[clamped-vs-unclamped] pre_activation_ranges {ranges}")
    print(
        f"[clamped-vs-unclamped] gate_above_limit={above}/{gate.numel()} "
        f"up_outside_limit={outside}/{up.numel()} limit={limit}"
    )

    clamped = _shared_swiglu_formula(case, gate, up, clamp=True, limit=limit)
    unclamped = _shared_swiglu_formula(case, gate, up, clamp=False, limit=limit)
    separation = _shared_max_rel_error(unclamped, clamped)
    print(
        f"[clamped-vs-unclamped] the_two_formulas_differ_by={separation:.6e} "
        f"rtol={SHARED_RTOL} clamped_absmax={float(clamped.abs().max()):.6e} "
        f"unclamped_absmax={float(unclamped.abs().max()):.6e}"
    )
    if separation <= SHARED_RTOL:
        raise SharedVacuousControlError(
            f"the clamped and unclamped formulas differ by only {separation:.6e} "
            f"at this fixture, inside rtol={SHARED_RTOL}, so this arm could not "
            f"tell them apart and would report a pass it had not earned"
        )

    block = _shared_build_block()
    quant_config = _shared_block_quant_config()
    routed = torch.zeros_like(clamped)
    got, sim = _shared_call_layer(block, case, quant_config, routed)
    reading = _assert_shared_route(sim, SHARED_DECLARED_SEAM_ENTRIES, "clamped-path")

    against_clamped = _shared_max_rel_error(got, clamped)
    against_unclamped = _shared_max_rel_error(got, unclamped)
    print(
        f"[clamped-vs-unclamped] shipped_vs_clamped={against_clamped:.6e} "
        f"shipped_vs_unclamped={against_unclamped:.6e} "
        f"rtol={SHARED_RTOL} atol={SHARED_ATOL}"
    )

    torch.testing.assert_close(got, clamped, rtol=SHARED_RTOL, atol=SHARED_ATOL)
    assert against_unclamped > SHARED_RTOL, (
        f"the shipped path is within rtol={SHARED_RTOL} of the UNCLAMPED formula "
        f"({against_unclamped:.6e}), so a clamp is missing from it. This is the "
        f"reading B22-M1 is about."
    )

    _shared_record(
        r02_pre_activation_ranges=ranges,
        r02_gate_above_limit=f"{above}/{gate.numel()}",
        r02_up_outside_limit=f"{outside}/{up.numel()}",
        r02_formula_separation=f"{separation:.6e}",
        r02_shipped_vs_clamped=f"{against_clamped:.6e}",
        r02_shipped_vs_unclamped=f"{against_unclamped:.6e}",
        r02_route_reading=reading,
    )


def _shared_landed_fixture_gate_response(base, limit, block, quant_config) -> float:
    """The landed fixture's response to the same gate perturbation R03 applies.

    Recorded beside R03's own reading so the disclosure is a measurement in the
    transcript rather than a sentence in a record. Not a bar: R03 asserts on its
    OWN fixture and only prints this one.
    """
    import torch

    gate, up = _shared_oracle_projections(base)
    reference = _shared_swiglu_formula(base, gate, up, clamp=True, limit=limit)
    routed = torch.zeros_like(reference)
    got, _ = _shared_call_layer(block, base, quant_config, routed)
    perturbed = dict(base)
    perturbed["gate_w"] = (base["gate_w"].to(torch.float32) * 0.5).to(
        base["gate_w"].dtype
    )
    perturbed_got, _ = _shared_call_layer(block, perturbed, quant_config, routed)
    return _shared_max_rel_error(perturbed_got, got)


def test_shared_expert_straddling_fixture_keeps_the_gate_projection_visible():
    """R03. A fixture astride the clamp, so the gate projection stays measured.

    WHY THIS ARM EXISTS, and it is a consequence of the repair rather than a
    complaint about it. At the landed fixture every pre-activation is outside the
    checkpoint's box, so ``silu(clamp(gate))`` is the constant ``silu(10)`` and
    the landed numeric arms no longer respond to the gate projection at all:
    halving the gate weights moves the clamped output by ``0.000000e+00`` where
    it moves the unclamped output by ``5.000000e-01``
    (``increments/probe-R7-straddle-and-sensitivity.out``). The clamp is the
    checkpoint's and does not move; what moves is where this file measures it.

    THREE CONJUNCTS. Both clamp regimes present in the gate operand; the shipped
    path still agrees with the clamped reference at the declared tolerances; and
    the shipped output RESPONDS to a change in the gate weights, which is the
    falsifiable form of "the gate projection is still measured".
    """
    import torch

    limit = _shared_swiglu_limit()
    base = _shared_build_case()
    case = _shared_straddling_case(base)
    gate, up = _shared_oracle_projections(case)

    above = int((gate > limit).sum())
    below = int((gate <= limit).sum())
    print(
        f"\n[straddle] divisor={SHARED_STRADDLE_DIVISOR} limit={limit} "
        f"gate=[{float(gate.min()):.4f}, {float(gate.max()):.4f}] "
        f"gate_above={above}/{gate.numel()} gate_at_or_below={below}/{gate.numel()}"
    )
    if not (above > 0 and below > 0):
        raise SharedVacuousControlError(
            f"the straddling fixture puts {above} gate elements above the bound "
            f"and {below} at or below it; this arm needs BOTH regimes or it is "
            f"just another saturated case"
        )

    clamped = _shared_swiglu_formula(case, gate, up, clamp=True, limit=limit)
    block = _shared_build_block()
    quant_config = _shared_block_quant_config()
    routed = torch.zeros_like(clamped)
    got, sim = _shared_call_layer(block, case, quant_config, routed)
    reading = _assert_shared_route(sim, SHARED_DECLARED_SEAM_ENTRIES, "straddle")
    agreement = _shared_max_rel_error(got, clamped)

    # The sensitivity conjunct, measured on the SHIPPED path both times.
    perturbed = dict(case)
    perturbed["gate_w"] = (case["gate_w"].to(torch.float32) * 0.5).to(
        case["gate_w"].dtype
    )
    perturbed_got, perturbed_sim = _shared_call_layer(
        block, perturbed, quant_config, routed
    )
    _assert_shared_route(
        perturbed_sim, SHARED_DECLARED_SEAM_ENTRIES, "straddle-perturbed"
    )
    response = _shared_max_rel_error(perturbed_got, got)

    saturated_response = _shared_landed_fixture_gate_response(base, limit, block,
                                                              quant_config)
    print(
        f"[straddle] shipped_vs_clamped_reference={agreement:.6e} "
        f"rtol={SHARED_RTOL} atol={SHARED_ATOL}"
    )
    print(
        f"[straddle] gate_halved_response_here={response:.6e} "
        f"gate_halved_response_at_the_landed_fixture={saturated_response:.6e}"
    )

    torch.testing.assert_close(got, clamped, rtol=SHARED_RTOL, atol=SHARED_ATOL)
    assert response > SHARED_RTOL, (
        f"halving the gate weights moved the shipped output by only "
        f"{response:.6e}, inside rtol={SHARED_RTOL}, so this fixture does not "
        f"measure the gate projection either and the arm earns nothing"
    )

    _shared_record(
        r03_straddle_divisor=SHARED_STRADDLE_DIVISOR,
        r03_gate_above_limit=f"{above}/{gate.numel()}",
        r03_gate_at_or_below_limit=f"{below}/{gate.numel()}",
        r03_shipped_vs_clamped=f"{agreement:.6e}",
        r03_gate_halved_response=f"{response:.6e}",
        r03_gate_halved_response_at_the_landed_fixture=f"{saturated_response:.6e}",
        r03_route_reading=reading,
    )


def test_shared_expert_two_sided_lower_bound_is_reached_and_changes_the_answer():
    """R04. ``min=-swiglu_limit`` is exercised, and dropping it is visible.

    The unsigned fixtures are entirely positive, so the up operand's LOWER bound
    never fires on any of them: an arm that only ran those would leave half of
    the reference's two-sided clamp untested. The landed SIGNED fixture does
    reach it -- 5,753 of 65,536 up elements sit below ``-10``
    (``increments/probe-R7-straddle-and-sensitivity.out``) -- so this arm uses
    that landed case and adds no fixture of its own.

    The conjunct is not "the branch ran". It is that a ONE-SIDED up clamp
    computes a different answer, which is what makes the lower bound load-bearing
    rather than decorative.
    """
    limit = _shared_swiglu_limit()
    signed = _shared_build_case(signed=True)
    gate, up = _shared_oracle_projections(signed)

    up_below = int((up < -limit).sum())
    gate_below = int((gate < -limit).sum())
    print(
        f"\n[lower-bound] limit={limit} "
        f"up=[{float(up.min()):.4f}, {float(up.max()):.4f}] "
        f"up_below_negative_limit={up_below}/{up.numel()} "
        f"gate_below_negative_limit={gate_below}/{gate.numel()}"
    )
    if up_below == 0:
        raise SharedVacuousControlError(
            f"no up element of the landed signed fixture is below -{limit}, so "
            f"this arm cannot say anything about the lower bound"
        )

    two_sided = _shared_swiglu_formula(signed, gate, up, clamp=True, limit=limit)
    upper_only = _shared_swiglu_formula(
        signed,
        gate.clamp(min=None, max=limit),
        up.clamp(min=None, max=limit),
        clamp=False,
        limit=limit,
    )
    difference = _shared_max_rel_error(upper_only, two_sided)
    frobenius = float(
        (upper_only - two_sided).norm() / two_sided.norm().clamp_min(1e-12)
    )
    print(
        f"[lower-bound] upper_only_vs_two_sided max_rel_error={difference:.6e} "
        f"relative_frobenius={frobenius:.6e} rtol={SHARED_RTOL}"
    )

    assert difference > SHARED_RTOL, (
        f"dropping the lower bound changed the answer by only {difference:.6e}, "
        f"inside rtol={SHARED_RTOL}, so this arm does not show the lower bound "
        f"matters"
    )

    _shared_record(
        r04_signed_up_below_negative_limit=f"{up_below}/{up.numel()}",
        r04_signed_gate_below_negative_limit=f"{gate_below}/{gate.numel()}",
        r04_upper_only_vs_two_sided=f"{difference:.6e}",
        r04_upper_only_relative_frobenius=f"{frobenius:.6e}",
    )


# ---------------------------------------------------------------------------
# Reporting -- the readings the evidence record quotes
# ---------------------------------------------------------------------------


def test_shared_expert_report_the_measured_readings(capsys):
    """Prints every reading this increment's evidence record quotes.

    Mirrors ``-031``'s landed reporting convention and reads this increment's OWN
    dict, so neither increment's reporting item depends on the other's items
    having run. Last by declaration order, for the same reason ``-031``'s is.
    """
    with capsys.disabled():
        print()
        print(
            f"[H01] layer output == routed+shared: "
            f"max_rel_error={_SHARED_READINGS.get('h01_max_rel_error')} vs "
            f"rtol={_SHARED_READINGS.get('h01_rtol')} "
            f"atol={_SHARED_READINGS.get('h01_atol')} "
            f"cases={_SHARED_READINGS.get('h01_cases')}"
        )
        print(
            f"[H02] zeroed shared: shared_absmax="
            f"{_SHARED_READINGS.get('h02_shared_absmax')} "
            f"max_abs_vs_routed_only="
            f"{_SHARED_READINGS.get('h02_max_abs_error_vs_routed_only')} "
            f"atol={_SHARED_READINGS.get('h02_atol')}"
        )
        print(
            f"[H03] double-add refused: "
            f"{_SHARED_READINGS.get('h03_declared_comparison_refused_double_add')} "
            f"at max_rel_error={_SHARED_READINGS.get('h03_double_add_max_rel_error')}"
        )
        print(
            f"[H04] R-2 per-call={_SHARED_READINGS.get('h04_per_call_deltas')} "
            f"declared={_SHARED_READINGS.get('h04_per_call_declared')} "
            f"per-case total={_SHARED_READINGS.get('h04_per_case_total')} "
            f"at multiplicity={_SHARED_READINGS.get('h04_case_call_multiplicity')} "
            f"| H01 case: multiplicity="
            f"{_SHARED_READINGS.get('h04_declared_case_multiplicity_for_h01')} "
            f"total={_SHARED_READINGS.get('h04_declared_case_total_for_h01')} "
            f"fallback={_SHARED_READINGS.get('h04_torch_fallback')}"
        )
        print(
            f"[H05] fallback control: {_SHARED_READINGS.get('h05_fallback_counters')} "
            f"refused={_SHARED_READINGS.get('h05_route_assertion_refused_the_fallback')}"
        )
        print(
            f"[H06] structure: shared_calls="
            f"{_SHARED_READINGS.get('h06_shared_calls_in_combine')} adds="
            f"{_SHARED_READINGS.get('h06_add_binops_in_combine')}"
        )
        print(
            f"[H08] F1 pow2: {_SHARED_READINGS.get('h08_pow2_block_scales')} "
            f"distinct={_SHARED_READINGS.get('h08_distinct_block_scales')}"
        )
        print(
            f"[H09] refusal={_SHARED_READINGS.get('h09_refusal_error')} "
            f"enum_code_refs={_SHARED_READINGS.get('h09_enum_code_references_in_this_section')} "
            f"prose={_SHARED_READINGS.get('h09_enum_prose_mentions')}"
        )
        print(
            f"[H10] signed norm="
            f"{_SHARED_READINGS.get('h10_signed_relative_frobenius')}"
        )
        print("--- all readings ---")
        for key in sorted(_SHARED_READINGS):
            print(f"{key}={_SHARED_READINGS[key]}")

    # The two declared conjuncts and the route predicate must all have reported.
    assert _SHARED_READINGS["h04_per_call_deltas"] == [
        SHARED_DECLARED_SEAM_ENTRIES,
        SHARED_DECLARED_SEAM_ENTRIES,
    ]
    assert _SHARED_READINGS["h04_torch_fallback"] == 0
    assert _SHARED_READINGS["h02_shared_absmax"] == f"{0.0:.6e}"
    assert _SHARED_READINGS["h03_declared_comparison_refused_double_add"] is True
    assert _SHARED_READINGS["h06_add_binops_in_combine"] == 1
    assert _SHARED_READINGS["h09_enum_code_references_in_this_section"] == 0
    assert len(_SHARED_READINGS) >= 30
