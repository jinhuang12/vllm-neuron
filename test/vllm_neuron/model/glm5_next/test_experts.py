# SPDX-License-Identifier: Apache-2.0
"""Expert plumbing: the 288-expert bank, its partition and the shared expert.

Covers the routed bank's shard geometry, the scale-operand prep both dense routes
share, and the refusals a ragged partition raises.
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

# The enum name is assembled rather than written, so the guard's own source
# is not a hit against itself.
FORBIDDEN_ENUM = "Quantization" + "Type"


def _impl():
    """Import the modeling module inside a test body, never at import time. """
    from vllm_neuron.model.glm5_next import model_fp8

    return model_fp8


def _floor_division_precedent(num_experts: int, tp_degree: int) -> dict[str, int]:
    """The fork's formula, recomputed -- the control's engine. """
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
# 288 experts over the declared TP degree: 0 dropped, 0 duplicated
# ---------------------------------------------------------------------------


def test_sharding_of_288_experts_at_the_declared_tp_degree_drops_zero_and_duplicates_zero():
    """The coverage half of the declared expected result, counts summed exactly. """
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

    # Control -- dropped moves. The fork's own formula, on the
    # same two numbers, through the same predicate.
    control = _floor_division_precedent(TOTAL_ROUTED_EXPERTS, DECLARED_TP_DEGREE)
    assert control["dropped"] == 32
    assert control["covered"] == FLOOR_TARGET

    # Control -- duplicated moves. Shift every offset down by one and the
    # same duplication predicate reports a non-zero.
    overlapped = fmod.ExpertPartition(
        num_experts=TOTAL_ROUTED_EXPERTS,
        tp_degree=DECLARED_TP_DEGREE,
        counts=part.counts,
        offsets=tuple(max(0, o - 1) for o in part.offsets),
    )
    assert overlapped.duplicated > 0
    assert overlapped.dropped > 0


# ---------------------------------------------------------------------------
# The 288/64 ragged case raises a named error rather than padding
# ---------------------------------------------------------------------------


def test_sharding_at_288_over_64_raises_a_named_error_rather_than_padding():
    """The raise half of the declared expected result, 1/1, at the freeze. """
    raised = 0
    with pytest.raises(fmod.RaggedExpertPartitionError) as gate:
        fmod.require_uniform_expert_partition(
            TOTAL_ROUTED_EXPERTS, DECLARED_TP_DEGREE
        )
    raised += 1
    assert raised == 1

    # The error is named, not a bare ValueError -- and a caller can tell it from
    # the out-of-range class.
    assert type(gate.value) is fmod.RaggedExpertPartitionError
    assert issubclass(fmod.RaggedExpertPartitionError, ValueError)
    assert not issubclass(
        fmod.RaggedExpertPartitionError, cfgmod.Glm5NextExpertConfigError
    )

    message = str(gate.value)
    # The five numeric tokens are functions of (288, degree) and this call passes 64
    # outright. The message names the expert-parallel degree, not the tensor-parallel one.
    for token in (
        str(TOTAL_ROUTED_EXPERTS),
        str(DECLARED_TP_DEGREE),
        str(RAGGED_REMAINDER),
        str(PAD_TARGET),
        str(FLOOR_TARGET),
        "expert-parallel degree",
    ):
        assert token in message, f"{token!r} missing from the named raise"

    # And the two things it must no longer say, asserted rather than assumed.
    for token in ("tensor-parallel degree",):
        assert token not in message, f"{token!r} must not appear in the raise"

    # The member's default degree resolves the expert-parallel degree, which is 1 when
    # expert parallelism was never initialised, so the default builds a uniform plan. A
    # ragged degree has to be passed explicitly to be refused.
    arch_default = fmod.Glm5NextForConditionalGeneration.expert_sharding_plan(
        _text_config()
    )
    assert arch_default.tp_degree == 1
    assert arch_default.counts == (TOTAL_ROUTED_EXPERTS,)
    assert arch_default.is_uniform

    arch_raised = 0
    with pytest.raises(fmod.RaggedExpertPartitionError):
        fmod.Glm5NextForConditionalGeneration.expert_sharding_plan(
            _text_config(), ep_degree=DECLARED_TP_DEGREE
        )
    arch_raised += 1
    assert arch_raised == 1

    # Nothing was padded: no partition this module builds ever assigns the pad
    # target, at the freeze or anywhere near it.
    for degree in (DECLARED_TP_DEGREE, DECLARED_TP_DEGREE // 2):
        assert (
            fmod.partition_experts(TOTAL_ROUTED_EXPERTS, degree).assigned
            == TOTAL_ROUTED_EXPERTS
        )

    # Control -- the raise moves. A degree that divides 288 raises nothing
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


# ---------------------------------------------------------------------------
# The gate is the raggedness predicate, over a censused domain
# ---------------------------------------------------------------------------


def test_sharding_uniformity_gate_tracks_raggedness_over_every_degree_up_to_64():
    """The raise is a rule over the whole degree domain, not a special case. """
    domain = list(range(1, DECLARED_TP_DEGREE + 1))
    uniform, ragged = [], []
    dropped_total = duplicated_total = 0

    for degree in domain:
        part = fmod.partition_experts(TOTAL_ROUTED_EXPERTS, degree)
        # Coverage holds at every degree, ragged or not.
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

    # Control -- the population-wide zero moves. The floor-division
    # precedent drops on exactly the ragged degrees, and only there.
    control_dropped = {
        degree: _floor_division_precedent(TOTAL_ROUTED_EXPERTS, degree)["dropped"]
        for degree in domain
    }
    assert sum(control_dropped.values()) > 0
    assert [d for d, n in control_dropped.items() if n > 0] == ragged


# ---------------------------------------------------------------------------
# The model-level bank consumes the partition
# ---------------------------------------------------------------------------


def test_sharding_plan_is_consumed_by_the_routed_expert_bank_in_model_fp8():
    """The ``model_fp8.py`` half of the declared surface. """
    impl = _impl()
    text_config = _text_config()

    bank = impl.Glm5NextRoutedExperts(text_config, world_size=32, ep_degree=32)
    assert bank.tp_degree == 32
    assert bank.num_routed_experts == TOTAL_ROUTED_EXPERTS
    assert bank.num_local_experts == 9
    assert bank.expert_partition.dropped == 0
    assert bank.expert_partition.duplicated == 0
    assert bank.local_expert_indices(0) == tuple(range(9))
    assert bank.local_expert_indices(31) == tuple(range(279, 288))

    # Every rank's slice, summed -- the model-level restatement of the first check.
    union: set[int] = set()
    for rank in range(32):
        union.update(bank.local_expert_indices(rank))
    assert union == set(range(TOTAL_ROUTED_EXPERTS))

    block = impl.Glm5NextMoEBlock(text_config, world_size=32, ep_degree=32)
    assert block.experts.num_local_experts == 9

    # The default resolves the process group, which is 1 undistributed -- so the
    # tree still builds and no neighbour's test moves.
    default_bank = impl.Glm5NextRoutedExperts(text_config)
    assert default_bank.tp_degree == impl._resolve_world_size() == 1
    assert default_bank.num_local_experts == TOTAL_ROUTED_EXPERTS

    # The tensor-parallel degree is not the expert divisor, so both constructions build
    # rather than refuse: the expert-parallel degree resolves to 1 and all 288 experts are
    # local on every rank.
    for built in (
        impl.Glm5NextRoutedExperts(text_config, world_size=DECLARED_TP_DEGREE),
        impl.Glm5NextMoEBlock(text_config, world_size=DECLARED_TP_DEGREE).experts,
    ):
        assert built.tp_degree == DECLARED_TP_DEGREE
        assert built.ep_degree == 1
        assert built.num_local_experts == TOTAL_ROUTED_EXPERTS

    # The refusal moves to its true subject, at both constructors, 2/2. 64 does
    # not divide 288, so an explicit ragged expert-parallel degree still raises.
    model_level_raised = 0
    for construct in (impl.Glm5NextRoutedExperts, impl.Glm5NextMoEBlock):
        with pytest.raises(fmod.RaggedExpertPartitionError):
            construct(text_config, ep_degree=DECLARED_TP_DEGREE)
        model_level_raised += 1
    assert model_level_raised == 2

    # Control -- the drop-32 formula is still reachable in this very
    # module, so the bank's 0 dropped is a distinction between two live
    # behaviours rather than the only thing the file can do.
    assert impl._per_rank(TOTAL_ROUTED_EXPERTS, DECLARED_TP_DEGREE) == 4
    assert 4 * DECLARED_TP_DEGREE == FLOOR_TARGET

    assert str(inspect.signature(impl._build_mlp)) == (
        "(text_config: 'Glm5NextTextConfig', layer_idx: 'int') -> 'nn.Module'"
    )


# ---------------------------------------------------------------------------
# Config-side expert-count validation
# ---------------------------------------------------------------------------


def test_sharding_config_validation_rejects_out_of_range_expert_counts():
    """The ``config.py`` half of the declared surface, plus its named boundary. """
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
            _text_config(n_routed_experts=4, num_experts_per_tok=8), ep_degree=4
        )
    assert plan_raised == 1
    assert construction_raised + plan_raised == 5

    # Control -- the refusal moves. All five were accepted at the unmodified
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


# ---------------------------------------------------------------------------
# No existing member signature moved
# ---------------------------------------------------------------------------


def test_sharding_members_are_a_pure_addition_to_the_co_authored_factory():
    """The factory's boundary members refuse by name, and no signature moved."""
    cls = fmod.Glm5NextForConditionalGeneration

    for member, expected in FACTORY_SIGNATURES_AT_PARENT.items():
        observed = str(inspect.signature(getattr(cls, member)))
        assert observed == expected, f"{member} signature moved:\n{observed}"

    # The registration's two boundary members still raise, unchanged in contract.
    for member in ("embed_input_ids", "compute_logits"):
        assert callable(getattr(cls, member))
    with pytest.raises(NotImplementedError):
        cls.embed_input_ids(cls, None)
    with pytest.raises(NotImplementedError):
        cls.compute_logits(cls, None)

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

    # Control -- the comparison moves. A deliberately wrong expected string
    # fails the same predicate, so the equalities above are not vacuous.
    control_mismatch = 0
    if str(inspect.signature(cls.from_configs)) != "(self) -> None":
        control_mismatch += 1
    assert control_mismatch == 1


# ---------------------------------------------------------------------------
# No vendor quantisation enum reference
# ---------------------------------------------------------------------------


def test_sharding_adds_no_vendor_quantisation_enum_reference():
    """The sharding members reference no vendor quantisation enum."""

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

    # Control -- the count moves on a synthetic source that really does
    # reference the enum, through the same walk.
    control_source = (
        f"from nkilib.core.utils.common_types import {FORBIDDEN_ENUM}\n"
        f"x = {FORBIDDEN_ENUM}.NONE\n"
    )
    control = enum_refs(control_source)
    assert control == 2


# ---------------------------------------------------------------------------
# The TP freeze is cited, not derived, and not configurable here
# ---------------------------------------------------------------------------


def test_sharding_tp_degree_freeze_is_the_registered_value_and_not_configurable():
    """The tensor-parallel degree is cited here, never re-derived, and not configurable."""
    assert fmod.TP_DEGREE_FREEZE == DECLARED_TP_DEGREE == 64

    # The member's degree parameter defaults to ``None``, meaning "resolve the live
    # expert-parallel degree", rather than to the tensor-parallel value asserted above,
    # which stays a non-configurable module literal.
    default = (
        inspect.signature(fmod.Glm5NextForConditionalGeneration.expert_sharding_plan)
        .parameters["ep_degree"]
        .default
    )
    assert default is None

    # Not configurable here: no environment read and no config field feeds it.
    source = Path(fmod.__file__).read_text()
    tree = ast.parse(source)
    env_reads = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in {"environ", "getenv"}:
            env_reads += 1
        if isinstance(node, ast.Name) and node.id in {"getenv", "environ"}:
            env_reads += 1
    assert env_reads == 0, (
        f"the factory module reads the environment at {env_reads} site(s), so this "
        f"degree could be set outside this file"
    )
    # The walk above is the whole reading. It already covers both names, and it covers
    # them as code: the substring check that used to sit here read the file's text, so
    # a comment or a docstring naming either name would have failed a module that never
    # reads the environment.

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

    # The corroborating read-only site is real, at the line this file cites.
    repo_root = Path(fmod.__file__).resolve().parents[3]
    pg_line = (
        (repo_root / "vllm_neuron" / "functional" / "process_groups.py")
        .read_text()
        .splitlines()[110]
    )
    assert "group_size == 64" in pg_line

    # Control -- replaced here. The control
    # discriminated by the default raising, and the default no longer raises, so
    # that shape would have become vacuous. This one discriminates on three arms
    # through the same member and reads 1 / 0 / 0: an explicit ragged degree
    # raises, an explicit exact degree does not, and the resolved default does not.
    explicit_ragged_raised = 0
    try:
        fmod.Glm5NextForConditionalGeneration.expert_sharding_plan(
            _text_config(), ep_degree=DECLARED_TP_DEGREE
        )
    except fmod.RaggedExpertPartitionError:
        explicit_ragged_raised += 1
    assert explicit_ragged_raised == 1

    explicit_exact_raised = 0
    try:
        control = fmod.Glm5NextForConditionalGeneration.expert_sharding_plan(
            _text_config(), ep_degree=32
        )
    except fmod.RaggedExpertPartitionError:  # pragma: no cover - control arm
        explicit_exact_raised += 1
    assert explicit_exact_raised == 0
    assert control.tp_degree == 32 and control.is_uniform

    resolved_default_raised = 0
    try:
        resolved = fmod.Glm5NextForConditionalGeneration.expert_sharding_plan(
            _text_config()
        )
    except fmod.RaggedExpertPartitionError:  # pragma: no cover - control arm
        resolved_default_raised += 1
    assert resolved_default_raised == 0
    assert resolved.tp_degree == 1 and resolved.is_uniform


# --------------------------------------------------------------------------- #
# declared values. The tolerances and the count are this file's; the tiny-config  #
# extents are chosen to admit the dense half's kernel, a fixture decision.       #
# --------------------------------------------------------------------------- #

#: The declared tolerance pair for check 1, and its atol for check 2.
SHARED_RTOL = 3e-2
SHARED_ATOL = 1e-5

#: One seam entry per projection.
SHARED_DECLARED_SEAM_ENTRIES = 3

#: Tiny config. Every extent is forced by the dense half's own admission gates
#: (``blockwise_fp8_mm.py::_require_blocked``), read off that source rather than
#: guessed, and chosen to admit because a geometry the kernel refuses raises --
#: it does not fall back -- and a refused shape would leave the counter at 0.
#:   T % TILE_SIZE == 0         (M tiles over the PSUM partition axis)
#:   H % SCALE_BLOCK_SIZE == 0  (K needs a whole number of block scales)
#:   I % SCALE_BLOCK_SIZE == 0  (N likewise)
#: The gate reads the dense consumer's own ``SCALE_BLOCK_SIZE``, which is 128, not the
#: routed bank's 256. H and I are 4 blocks at that constant, and more than 1
#: deliberately: a ``[1, 1]`` scale grid cannot distinguish a transposed flat index, so
#: this fixture's block scales are distinct and asymmetric and a mis-mapping is
#: numerically visible.
SHARED_T = 128
SHARED_H = 512
SHARED_I = 512

#: Per-block exponents for the three projections' scale grids. Deterministic, not
#: sampled: three properties are load-bearing and a draw satisfies them only by
#: luck -- every entry an exact power of two (checked below), the four entries
#: distinct so a transposed flat index moves the numbers, and the matrix
#: asymmetric so a transpose is not the identity.
SHARED_GATE_EXPONENTS = ((-2, 1), (0, -1))
SHARED_UP_EXPONENTS = ((1, -1), (-2, 0))
SHARED_DOWN_EXPONENTS = ((0, 2), (-1, 1))


#: One block lands `fixtures/hf-config.json` as a byte-identical copy of the
#: published glm-5.3-Flash config and pins it by this digest in its own check;
#: `test_config.py` reads the same file the same way. This
#: section reads it and never writes it. Reading the bound from here rather than
#: typing `10.0` is what makes the clamp the checkpoint's bound: the finding asks
#: for the value to be sourced from the checkpoint, and a literal in this file
#: would fail that half of it just as a literal in the shipped path would.
#: The provenance probe value, and the one reason it is not ``10.0``. The
#: checkpoint declares ``swiglu_limit = 10.0`` and ``Glm5NextTextConfig``'s field
#: defaults to the same number on purpose (``config.py``, following
#: ``rms_norm_eps``), so a reading of ``10.0`` on a built object cannot tell a
#: config read apart from a default. ``7.5`` is a value the checkpoint does not
#: carry, which is what makes the read path falsifiable. It is not a comparator
#: and nothing is measured against it: it is an input pushed through the adapter.
#: The same device used for the epsilon
#: (``test_config.py``'s ``C080_NON_DEFAULT_RMS_NORM_EPS = 3e-05``).
R01_NON_DEFAULT_BOUND = 7.5

SHARED_VENDOR_CONFIG_SHA256 = (
    "bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f"
)

#: The power-of-two divisor that puts the gate operand astride the clamp.
SHARED_STRADDLE_DIVISOR = 8


class SharedRouteInstrumentError(AssertionError):
    """A route reading that is not what this file declares."""


class SharedF1PreconditionError(AssertionError):
    """The pow2 losslessness precondition did not hold on this case's scales."""


class SharedVacuousControlError(AssertionError):
    """A control whose input could not have made it fail. """


class SharedSectionOwnershipError(AssertionError):
    """A structural claim about the shared-expert section did not hold."""


class _SharedSimulatorCounter:
    """Counts real ``nki.simulator.simulate_kernel`` calls for the duration. """

    def __init__(self) -> None:
        self.calls = 0
        self._nki = None
        self._real = None

    def __enter__(self) -> "_SharedSimulatorCounter":
        # Both imports are required and the second is not redundant:
        # ``nki.simulator`` is a submodule, so ``import nki`` alone leaves
        # ``nki.simulator`` unbound and attribute access raises
        # ``AttributeError: module 'nki' has no attribute 'simulator'``. This is
        # the dense half's pair (``test_blockwise_fp8_mm.py`` imports ``nki``
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
    """The dense half's module, re-acquired through ``importlib``. """
    import importlib

    return importlib.import_module("vllm_neuron.functional.blockwise_fp8_mm")


def _assert_shared_route(
    sim: _SharedSimulatorCounter, expected_entries: int, label: str
) -> str:
    """Read all four route instruments and return the reading."""
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
    if nki_dispatch != expected_entries:
        raise SharedRouteInstrumentError(
            f"{label}: the seam dispatch counter read {nki_dispatch}, declared "
            f"{expected_entries} (one per projection site: gate, up, down). A "
            f"bypassed projection reads fewer and is exactly what this counts. "
            f"{reading}"
        )
    if torch_fallback != 0:
        raise SharedRouteInstrumentError(
            f"{label}: the torch-fallback counter read {torch_fallback}, "
            f"declared exactly 0 -- a fallback pass would compare torch against "
            f"torch, so the comparison would be torch against torch. {reading}"
        )
    if gate is not True:
        raise SharedRouteInstrumentError(
            f"{label}: can_run_kernel() read {gate!r}, declared True. {reading}"
        )
    if sim.calls != expected_entries:
        raise SharedRouteInstrumentError(
            f"{label}: nki.simulator.simulate_kernel ran {sim.calls} times, "
            f"declared {expected_entries}. A numeric pass without a simulator "
            f"call is the vacuous pass this arm screens for. {reading}"
        )
    return reading


# --------------------------------------------------------------------------- #
# fixture construction.                                                        #
# --------------------------------------------------------------------------- #
def _shared_pow2_scales(exponents, rows: int, cols: int):
    """The public ``[rows//256, cols//256]`` block-scale grid, every entry pow2. """
    import torch

    from vllm_neuron.functional.blockwise_fp8_mm import scale_grid_shape

    want = scale_grid_shape(rows, cols)
    grid = torch.zeros(want, dtype=torch.int64)
    for k_block in range(want[0]):
        for n_block in range(want[1]):
            grid[k_block, n_block] = exponents[k_block % 2][n_block % 2]
    return torch.ldexp(torch.ones(want, dtype=torch.float32), grid)


def _shared_fp8_grid(seed: int, *shape: int, signed: bool = False):
    """Values already on the fp8-e4m3 grid, so every cast in the fixture is exact. """
    import torch

    generator = torch.Generator().manual_seed(seed)
    low = -7 if signed else 1
    return torch.randint(low, 8, shape, generator=generator).to(torch.float32) / 8.0


def _shared_build_case(zero_shared: bool = False, signed: bool = False) -> dict:
    """The tiny config: three fp8 projections, three pow2 public scale grids. """
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
    """``Glm5NextQuantConfig`` for the pinned checkpoint -- nothing hand-fed. """
    import json
    from pathlib import Path

    fixture = Path(__file__).resolve().parent / "fixtures" / "config.json"

    from vllm_neuron.model.glm5_next.config import Glm5NextConfig

    model_fp8 = _impl()
    return model_fp8.Glm5NextQuantConfig.from_model_config(
        Glm5NextConfig.from_configs(json.loads(fixture.read_text()))
    )


def _shared_build_block():
    """A ``Glm5NextMoEBlock`` whose shared expert exists, at the tiny config. """
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
    """The checkpoint's ``text_config.swiglu_limit``, digest-checked first. """
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
    """The reference's two clamps, transliterated."""
    return gate.clamp(min=None, max=limit), up.clamp(min=-limit, max=limit)


def _shared_swiglu_formula(case: dict, gate, up, *, clamp: bool, limit: float):
    """``down(silu(gate) * up)`` with the clamps switched in or out. """
    from torch.nn.functional import silu

    from vllm_neuron.functional.blockwise_fp8_mm import blockwise_fp8_mm_torch_oracle

    if clamp:
        gate, up = _shared_apply_reference_clamps(gate, up, limit)
    activated = silu(gate) * up
    return blockwise_fp8_mm_torch_oracle(
        activated.to(case["hidden_states"].dtype), case["down_w"], case["down_s"]
    )


def _shared_oracle_projections(case: dict):
    """``gate`` and ``up`` as the dense half's torch oracle computes them, pre-clamp."""
    from vllm_neuron.functional.blockwise_fp8_mm import blockwise_fp8_mm_torch_oracle

    hidden = case["hidden_states"]
    return (
        blockwise_fp8_mm_torch_oracle(hidden, case["gate_w"], case["gate_s"]),
        blockwise_fp8_mm_torch_oracle(hidden, case["up_w"], case["up_s"]),
    )


def _shared_straddling_case(base: dict) -> dict:
    """The case with its hidden states divided by a power of two. """
    case = dict(base)
    case["hidden_states"] = base["hidden_states"] / SHARED_STRADDLE_DIVISOR
    return case


def _shared_expert_torch_reference(case: dict):
    """The independent torch formulation of ``down(silu(gate(x)) * up(x))``. """
    gate, up = _shared_oracle_projections(case)
    return _shared_swiglu_formula(
        case, gate, up, clamp=True, limit=_shared_swiglu_limit()
    )


def _shared_routed_stand_in(shared_reference):
    """A conditioned ``[T, H]`` routed contribution, at the shared half's scale. """
    import torch

    routed = _shared_fp8_grid(41, SHARED_T, SHARED_H).to(torch.float32)
    scale = shared_reference.abs().max() / routed.abs().max().clamp_min(1e-12)
    return routed * scale


def _shared_max_rel_error(got, want) -> float:
    """``max |got - want| / (|want| + atol)`` -- a number, not a verdict."""
    return float(((got - want).abs() / (want.abs() + SHARED_ATOL)).max())


def _shared_call_layer(block, case: dict, quant_config, routed):
    """Drive the shared expert's call site once, under all four instruments."""
    seam = _shared_seam()
    block.shared_experts.prepare_scale_operands(
        case["gate_w"],
        case["up_w"],
        case["down_w"],
        case["gate_s"],
        case["up_s"],
        case["down_s"],
    )
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
    """The AST of one method of ``model_fp8.py``, located by name. """
    source = Path(_impl().__file__).read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for member in node.body:
                if isinstance(member, ast.FunctionDef) and member.name == method_name:
                    return member
    raise SharedSectionOwnershipError(
        f"{class_name}.{method_name} not found in model_fp8.py -- the "
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
# Declared check 1: layer output == routed + shared, rtol 3e-2 / atol 1e-5
# ---------------------------------------------------------------------------


def test_shared_expert_layer_output_equals_routed_plus_shared():
    """The declared first declared check, at the declared tolerances. """
    import torch

    case = _shared_build_case()
    block = _shared_build_block()
    quant_config = _shared_block_quant_config()

    shared_reference = _shared_expert_torch_reference(case)
    routed = _shared_routed_stand_in(shared_reference)
    want = routed + shared_reference

    got, sim = _shared_call_layer(block, case, quant_config, routed)
    _assert_shared_route(sim, SHARED_DECLARED_SEAM_ENTRIES, "acceptance")

    # Non-vacuity gate. An all-zero reference would make assert_close pass on a
    # function that returns zeros, so the comparison refuses to run over one.
    nonzero_rows = int((want.abs().sum(dim=1) > 0).sum())
    if nonzero_rows != SHARED_T:
        raise SharedVacuousControlError(
            f"only {nonzero_rows}/{SHARED_T} reference rows are nonzero; a "
            f"tolerance over a vacuous reference measures nothing"
        )

    torch.testing.assert_close(got, want, rtol=SHARED_RTOL, atol=SHARED_ATOL)

    assert tuple(got.shape) == (SHARED_T, SHARED_H)


# ---------------------------------------------------------------------------
# Declared check 2: zeroed shared reproduces routed-only at atol 1e-5
# ---------------------------------------------------------------------------


def test_shared_expert_zeroed_case_reproduces_routed_only():
    """The declared second declared check: the shared half is added exactly once. """
    import torch

    case = _shared_build_case(zero_shared=True)
    block = _shared_build_block()
    quant_config = _shared_block_quant_config()

    shared_reference = _shared_expert_torch_reference(case)
    shared_absmax = float(shared_reference.abs().max())

    # The routed-only output. Built from the nonzero case's shared scale so this
    # arm's routed operand has the same magnitude as the first check's -- otherwise "equals
    # routed-only" could be a comparison of two tiny tensors.
    routed = _shared_routed_stand_in(_shared_expert_torch_reference(_shared_build_case()))

    # Non-vacuity gate on the routed half: if it were zero, this arm would be
    # comparing zero against zero and would pass for the wrong reason.
    nonzero_rows = int((routed.abs().sum(dim=1) > 0).sum())
    if nonzero_rows != SHARED_T:
        raise SharedVacuousControlError(
            f"only {nonzero_rows}/{SHARED_T} routed rows are nonzero; "
            f"'reproduces the routed-only output' would be vacuous"
        )

    got, sim = _shared_call_layer(block, case, quant_config, routed)
    _assert_shared_route(sim, SHARED_DECLARED_SEAM_ENTRIES, "zeroed-shared")

    assert shared_absmax == 0.0, (
        f"the zeroed shared expert produced a nonzero contribution "
        f"({shared_absmax}); the zeroing did not take effect"
    )
    torch.testing.assert_close(got, routed, rtol=0, atol=SHARED_ATOL)


# ---------------------------------------------------------------------------
# The control: the declared tolerance can detect a double add
# ---------------------------------------------------------------------------


def test_shared_expert_double_add_is_refused_by_the_declared_tolerance():
    """Add the shared half twice; the declared comparison must raise. """
    import torch

    case = _shared_build_case()
    shared_reference = _shared_expert_torch_reference(case)
    routed = _shared_routed_stand_in(shared_reference)

    once = routed + shared_reference
    twice = routed + 2.0 * shared_reference

    perturbation = _shared_max_rel_error(twice, once)
    if perturbation <= SHARED_RTOL:
        raise SharedVacuousControlError(
            f"doubling the shared contribution moved the sum by only "
            f"{perturbation:.6e}, which is inside rtol={SHARED_RTOL}. This "
            f"control could not have failed, so it certifies nothing about the "
            f"once-versus-twice property."
        )

    with pytest.raises(AssertionError):
        torch.testing.assert_close(twice, once, rtol=SHARED_RTOL, atol=SHARED_ATOL)


# ---------------------------------------------------------------------------
# The route predicate: 3 per call, one per projection site
# ---------------------------------------------------------------------------


def test_shared_expert_seam_entries_are_one_per_projection_site():
    """The dense half's counter reads 3 per shared-expert call, and 3 is per-call. """
    case = _shared_build_case()
    block = _shared_build_block()
    quant_config = _shared_block_quant_config()
    routed = _shared_routed_stand_in(_shared_expert_torch_reference(case))
    seam = _shared_seam()

    block.shared_experts.prepare_scale_operands(
        case["gate_w"],
        case["up_w"],
        case["down_w"],
        case["gate_s"],
        case["up_s"],
        case["down_s"],
    )
    seam.reset_dispatch_counters()
    assert seam.dispatch_counters() == (0, 0), (
        "the reset did not zero the seam's counters, so every reading below would be "
        "cumulative and none of them would mean what it says"
    )

    readings = []
    with _SharedSimulatorCounter() as sim:
        for _call_index in range(2):
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

    per_call = [readings[0][0], readings[1][0] - readings[0][0]]
    if readings[0] != (SHARED_DECLARED_SEAM_ENTRIES, 0):
        raise SharedRouteInstrumentError(
            f"after one shared-expert call the seam's counters read {readings[0]}, "
            f"declared ({SHARED_DECLARED_SEAM_ENTRIES}, 0)"
        )
    if readings[1] != (2 * SHARED_DECLARED_SEAM_ENTRIES, 0):
        raise SharedRouteInstrumentError(
            f"after two calls the seam's counters read {readings[1]}, expected "
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


# ---------------------------------------------------------------------------
# The control: the (3, 0) reading is a measurement, not an always-3 counter
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# "exactly once" is structural, not only numeric
# ---------------------------------------------------------------------------


def test_shared_expert_add_is_structurally_exactly_once():
    """One call to the shared path and one ``+`` in ``combine_routed_and_shared``. """
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

    if shared_calls != 1:
        raise SharedSectionOwnershipError(
            f"combine_routed_and_shared calls shared_expert_mm {shared_calls} "
            f"times, declared exactly 1"
        )
    if len(adds) != 1 or aug_adds:
        raise SharedSectionOwnershipError(
            f"combine_routed_and_shared contains {len(adds)} '+' expressions and "
            f"{len(aug_adds)} '+=' statements, declared exactly one '+' and no "
            f"'+='. Two adds is the defect the declared second check exists to "
            f"exclude."
        )


# ---------------------------------------------------------------------------
# The two same-named scale helpers: neither is imported by this section
# ---------------------------------------------------------------------------


def test_shared_expert_section_imports_neither_scale_layout_helper():
    """The shared-expert section imports no ``to_kernel_scale_layout`` at all."""
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

    offenders = [name for name in names if "to_kernel_scale_layout" in name]
    moe_offenders = [name for name in names if "moe_blockwise_fp8" in name]

    # Non-vacuity: the scan must have read a real, non-empty import list.
    if not names:
        raise SharedVacuousControlError(
            "no imports were found in the shared-expert section, so the scan read "
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


# ---------------------------------------------------------------------------
# Every block scale this arm runs on is an exact power of two
# ---------------------------------------------------------------------------


def test_shared_expert_f1_precondition_block_scales_are_pow2():
    """All three projections' block scales are exact pow2, over N/N blocks. """
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
                    f"on top of the shared expert's plumbing"
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


# ---------------------------------------------------------------------------
# The named refusal: no silent path to QuantizationType.none
# ---------------------------------------------------------------------------


def test_shared_expert_refuses_a_non_block_quant_config_by_name():
    """An unquantised ``quant_config`` raises by name rather than falling through. """
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
    assert refused_counters == (0, 0), (
        f"the refusal ran {refused_counters} seam entries; it must refuse BEFORE "
        f"touching the seam"
    )

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

    # The control must fire, over real non-empty input, or the zero above is a
    # statement about the walk rather than about this section.
    control = enum_refs(
        ast.parse(
            f"from nkilib.core.utils.common_types import {FORBIDDEN_ENUM}\n"
            f"x = {FORBIDDEN_ENUM}.NONE\n"
        )
    )
    assert enum_hits == 0
    assert control == 2, (
        f"the enum walk scored {control} on a source that really does reference "
        f"the enum twice; the zero above would certify nothing"
    )
    if prose_mentions == 0:
        raise SharedVacuousControlError(
            "this section mentions the enum in no prose at all, so the "
            "code-versus-prose distinction this scan draws is untested here"
        )


# ---------------------------------------------------------------------------
# Signed coverage, at the same declared tolerances
# ---------------------------------------------------------------------------


def test_shared_expert_signed_fixture_agrees_in_norm_under_cancellation():
    """A signed fixture, compared in a cancellation-robust norm at the same tolerances.
    """
    case = _shared_build_case(signed=True)
    block = _shared_build_block()
    quant_config = _shared_block_quant_config()

    shared_reference = _shared_expert_torch_reference(case)
    routed = _shared_routed_stand_in(shared_reference)
    want = routed + shared_reference

    got, sim = _shared_call_layer(block, case, quant_config, routed)
    _assert_shared_route(sim, SHARED_DECLARED_SEAM_ENTRIES, "signed")

    reference_norm = float(want.norm())
    if reference_norm == 0.0:
        raise SharedVacuousControlError("the signed reference is identically zero")
    relative_norm = float((got - want).norm()) / reference_norm
    assert relative_norm <= SHARED_RTOL, (
        f"signed fixture disagrees in norm: {relative_norm:.6e} > "
        f"rtol={SHARED_RTOL}"
    )


def test_shared_expert_swiglu_bound_is_the_checkpoints_and_no_literal_governs_it():
    """The bound is read from the config, and the clamps stay asymmetric."""
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

    # Reading 1: what the block this section builds actually resolved.
    block = _shared_build_block()
    resolved = block.shared_experts.swiglu_limit

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

    # ... And the bound governs the arithmetic, not just the attribute. The same
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
    _assert_shared_route(
        sim_probe, SHARED_DECLARED_SEAM_ENTRIES, "swiglu-provenance-probe"
    )
    at_ten, sim_ten = _shared_call_layer(
        checkpoint_block, case, quant_config, routed
    )
    _assert_shared_route(
        sim_ten, SHARED_DECLARED_SEAM_ENTRIES, "swiglu-provenance-checkpoint"
    )
    bound_response = _shared_max_rel_error(at_ten, at_probe)

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

    # A literal equal to the bound anywhere in the shipped method would mean the
    # config read is decoration. Counted over both methods, and each count is
    # read beside its population so a zero is not read as an empty search.
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
            f"the bound is read from the config at "
            f"construction, so a caller cannot supply a different one."
        )
    # Reading 4.
    assert clamps.get("gate") == {"min": "None", "max": "self.swiglu_limit"}, (
        f"the shipped gate clamp reads {clamps.get('gate')}; the reference bounds "
        f"the gate ABOVE ONLY (modeling_glm5_next.py) with the config's bound"
    )
    assert clamps.get("up") == {
        "min": "-self.swiglu_limit",
        "max": "self.swiglu_limit",
    }, (
        f"the shipped up clamp reads {clamps.get('up')}; the reference bounds the "
        f"up operand on BOTH sides (modeling_glm5_next.py)"
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


def test_shared_expert_matches_the_clamped_formula_and_not_the_unclamped_one():
    """The clamped formula is what the shared expert computes; the unclamped one fails."""
    import torch

    limit = _shared_swiglu_limit()
    case = _shared_build_case()
    gate, up = _shared_oracle_projections(case)


    clamped = _shared_swiglu_formula(case, gate, up, clamp=True, limit=limit)
    unclamped = _shared_swiglu_formula(case, gate, up, clamp=False, limit=limit)
    separation = _shared_max_rel_error(unclamped, clamped)
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
    _assert_shared_route(sim, SHARED_DECLARED_SEAM_ENTRIES, "clamped-path")

    against_unclamped = _shared_max_rel_error(got, unclamped)

    torch.testing.assert_close(got, clamped, rtol=SHARED_RTOL, atol=SHARED_ATOL)
    assert against_unclamped > SHARED_RTOL, (
        f"the shipped path is within rtol={SHARED_RTOL} of the UNCLAMPED formula "
        f"({against_unclamped:.6e}), so a clamp is missing from it. This is the "
        f"the SwiGLU clamp is about."
    )


def _shared_fixture_gate_response(base, limit, block, quant_config) -> float:
    """The fixture's response to the same gate perturbation the astride arm applies."""
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
    """A fixture astride the clamp, so the gate projection stays measured."""
    import torch

    limit = _shared_swiglu_limit()
    base = _shared_build_case()
    case = _shared_straddling_case(base)
    gate, up = _shared_oracle_projections(case)

    above = int((gate > limit).sum())
    below = int((gate <= limit).sum())
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
    _assert_shared_route(sim, SHARED_DECLARED_SEAM_ENTRIES, "straddle")

    # The sensitivity check, measured on the shipped path both times.
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


    torch.testing.assert_close(got, clamped, rtol=SHARED_RTOL, atol=SHARED_ATOL)
    assert response > SHARED_RTOL, (
        f"halving the gate weights moved the shipped output by only "
        f"{response:.6e}, inside rtol={SHARED_RTOL}, so this fixture does not "
        f"measure the gate projection either and the arm earns nothing"
    )


def test_shared_expert_two_sided_lower_bound_is_reached_and_changes_the_answer():
    """``min=-swiglu_limit`` is exercised, and dropping it is visible."""
    limit = _shared_swiglu_limit()
    signed = _shared_build_case(signed=True)
    gate, up = _shared_oracle_projections(signed)

    up_below = int((up < -limit).sum())
    if up_below == 0:
        raise SharedVacuousControlError(
            f"no up element of the signed fixture is below -{limit}, so "
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

    assert difference > SHARED_RTOL, (
        f"dropping the lower bound changed the answer by only {difference:.6e}, "
        f"inside rtol={SHARED_RTOL}, so this arm does not show the lower bound "
        f"matters"
    )


