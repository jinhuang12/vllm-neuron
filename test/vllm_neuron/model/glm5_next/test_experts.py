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
