# SPDX-License-Identifier: Apache-2.0
"""Acceptance test for ``inc-glm53f-087`` -- WP7 REPAIR: the expert partition is
keyed on the expert-parallel degree, not on world size.

WHAT WAS WRONG. Landed ``inc-glm53f-031`` divided this checkpoint's 288 routed
experts by the tensor-parallel world size and raised
``RaggedExpertPartitionError`` when the division was inexact. At the campaign's
registered TP = 64, ``divmod(288, 64) == (4, 32)``, so THE MODEL REFUSED TO BUILD
-- for a constraint the fork itself does not impose.

THE RULE THE FORK ACTUALLY USES, read from its one landed EP-aware expert bank:
the divisor is the EXPERT-PARALLEL degree. With expert parallelism off the degree
is 1, every expert is local on every rank, and TP shards the INTERMEDIATE
dimension instead (``gpt_oss/model_bf16.py:986-988``); the division itself is
``num_local_experts // self.ep_degree`` (``:1072``). The degree comes from
``get_neuron_ep_degree()``, which returns 1 when expert parallelism was never
initialised (``neuron_parallel_state.py:1195-1198``).

THE ROUTE PARTITION. Every conjunct here reads the RESOLVED degree -- ``ep_degree``
is left ``None`` and the getter answers, which is the production route. The
EXPLICIT-degree readings belong to ``-031``'s re-pinned items in
``test_experts.py`` and are deliberately not re-taken here, so nothing is
certified twice.

Tier T: a config-time counted predicate, no kernel reached.
Substrate: NON-KERNEL-CLASS.
"""

import ast
import inspect
from pathlib import Path

import pytest

from vllm_neuron.model.glm5_next import factory as fmod
from vllm_neuron.parallel import neuron_parallel_state as npsmod

# Declared values: each is the checkpoint's own or the registered freeze.
TOTAL_ROUTED_EXPERTS = 288
DECLARED_TP_DEGREE = 64
EXACT_EP_DEGREE = 32  # divides 288: 288 // 32 == 9
RAGGED_REMAINDER = TOTAL_ROUTED_EXPERTS % DECLARED_TP_DEGREE  # 32
PAD_TARGET = 320  # 5 x 64 -- would invent 32 experts
FLOOR_TARGET = 256  # 4 x 64 -- would drop 32 experts
GPT_OSS_PRECEDENT = "gpt_oss/model_bf16.py:1072"

_READINGS: dict[str, object] = {}


def _record(**readings: object) -> None:
    """Collect a reading and print it, so every number is in the transcript."""
    _READINGS.update(readings)
    for key, value in readings.items():
        print(f"{key}={value}")


def _impl():
    """Import the modeling module inside a test body, never at import time.

    ``test_experts.py``'s idiom, kept for its reason: this file sorts before
    ``test_factory.py``, so a module-level import would populate ``sys.modules``
    for every later item in the package.
    """
    from vllm_neuron.model.glm5_next import model_fp8

    return model_fp8


def _text_config(**overrides: object):
    from vllm_neuron.model.glm5_next import config as cfgmod

    return cfgmod.Glm5NextTextConfig(**overrides)


# ---------------------------------------------------------------------------
# C1 -- EP off at the frozen TP degree: the bank builds whole
# ---------------------------------------------------------------------------


def test_ep_off_at_the_frozen_tp_degree_builds_all_288_experts_and_raises_nothing():
    """The repair itself: TP = 64 no longer refuses, because TP is not the divisor.

    D1.4 certifying component:
    ``model_fp8.py::Glm5NextRoutedExperts.__init__`` (degree resolution and gate
    call) over ``neuron_parallel_state.py::get_neuron_ep_degree``'s uninitialised
    branch (``:1195-1198``, ``if _NEURON_EP is None: return 1``).
    """
    impl = _impl()
    text_config = _text_config()

    # The uninitialised branch is REACHED, not assumed: no group exists here.
    assert npsmod._NEURON_EP is None
    assert npsmod.get_neuron_ep_degree() == 1

    bank = impl.Glm5NextRoutedExperts(text_config, world_size=DECLARED_TP_DEGREE)

    # THE HEADLINE: at the frozen degree nothing raises and nothing is lost.
    assert bank.ep_degree == 1
    assert bank.num_local_experts == TOTAL_ROUTED_EXPERTS
    assert bank.num_routed_experts == TOTAL_ROUTED_EXPERTS

    # ``tp_degree`` keeps its name and meaning; it is simply not a divisor.
    assert bank.tp_degree == DECLARED_TP_DEGREE
    assert bank.tp_degree != bank.ep_degree

    assert bank.expert_partition.dropped == 0
    assert bank.expert_partition.duplicated == 0
    assert bank.local_expert_indices(0) == tuple(range(TOTAL_ROUTED_EXPERTS))

    # The block-level pass-through resolves the same way, so the repair reaches
    # the model and not only the bank.
    block = impl.Glm5NextMoEBlock(text_config, world_size=DECLARED_TP_DEGREE)
    assert block.experts.ep_degree == 1
    assert block.experts.num_local_experts == TOTAL_ROUTED_EXPERTS

    _record(
        c1_getter_uninitialised_degree=npsmod.get_neuron_ep_degree(),
        c1_raised=0,
        c1_bank_ep_degree=bank.ep_degree,
        c1_bank_tp_degree=bank.tp_degree,
        c1_bank_num_local_experts=bank.num_local_experts,
        c1_block_num_local_experts=block.experts.num_local_experts,
        c1_dropped=bank.expert_partition.dropped,
        c1_duplicated=bank.expert_partition.duplicated,
    )


# ---------------------------------------------------------------------------
# C2 -- EP on at a degree that divides 288, through the resolved route
# ---------------------------------------------------------------------------


def test_ep_on_at_a_degree_that_divides_288_yields_nine_local_experts():
    """EP on still divides, by the degree the getter reports.

    The getter is PATCHED rather than a degree passed, so this reads the
    resolution seam. The explicit-degree reading at this same 32 is ``-031``'s
    re-pinned S04 item and is not re-taken here.

    D1.4 certifying component: ``factory.py::_resolve_ep_degree`` reading
    ``neuron_parallel_state.py::get_neuron_ep_degree`` at call time, feeding
    ``factory.py::require_uniform_expert_partition``.
    """
    impl = _impl()
    text_config = _text_config()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(npsmod, "get_neuron_ep_degree", lambda: EXACT_EP_DEGREE)
        assert npsmod.get_neuron_ep_degree() == EXACT_EP_DEGREE

        bank = impl.Glm5NextRoutedExperts(text_config)
        assert bank.ep_degree == EXACT_EP_DEGREE
        assert bank.num_local_experts == TOTAL_ROUTED_EXPERTS // EXACT_EP_DEGREE == 9
        assert bank.expert_partition.dropped == 0
        assert bank.expert_partition.duplicated == 0

        # Every rank's slice, summed: the partition is over the EP degree.
        union: set[int] = set()
        for rank in range(EXACT_EP_DEGREE):
            union.update(bank.local_expert_indices(rank))
        assert union == set(range(TOTAL_ROUTED_EXPERTS))

    # The patch is what moved the reading, and it was scoped: outside the context
    # the real getter answers 1 again. That is why this is the resolved route.
    assert npsmod.get_neuron_ep_degree() == 1

    _record(
        c2_patched_degree=EXACT_EP_DEGREE,
        c2_raised=0,
        c2_bank_ep_degree=bank.ep_degree,
        c2_bank_num_local_experts=bank.num_local_experts,
        c2_union_covers_all=len(union) == TOTAL_ROUTED_EXPERTS,
        c2_getter_restored_after_context=npsmod.get_neuron_ep_degree(),
    )


# ---------------------------------------------------------------------------
# C3 -- EP on at a degree that does not divide 288: the named raise
# ---------------------------------------------------------------------------


def test_ep_on_at_a_ragged_degree_raises_naming_the_expert_parallel_degree():
    """The raggedness gate survives the repair, with its subject corrected.

    Reached through the SAME patched getter as C2, so this is the arithmetic
    ``-031`` already refused at, re-subjected rather than re-derived.

    D1.4 certifying component:
    ``factory.py::require_uniform_expert_partition`` raising
    ``factory.py::RaggedExpertPartitionError`` through the resolution seam.
    """
    impl = _impl()
    text_config = _text_config()

    # 288 = 2**5 * 3**2, so 64 = 2**6 divides it in no reading.
    assert TOTAL_ROUTED_EXPERTS % DECLARED_TP_DEGREE == RAGGED_REMAINDER == 32

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(npsmod, "get_neuron_ep_degree", lambda: DECLARED_TP_DEGREE)
        raised = 0
        with pytest.raises(fmod.RaggedExpertPartitionError) as gate:
            impl.Glm5NextRoutedExperts(text_config)
        raised += 1

    assert raised == 1
    assert type(gate.value) is fmod.RaggedExpertPartitionError
    message = str(gate.value)

    # THE REQUIRED SET, declared so this conjunct and the re-pinned landed token
    # list in ``test_experts.py`` cannot drift apart.
    required = (
        str(TOTAL_ROUTED_EXPERTS),
        str(DECLARED_TP_DEGREE),
        str(RAGGED_REMAINDER),
        str(PAD_TARGET),
        str(FLOOR_TARGET),
        GPT_OSS_PRECEDENT,
        "expert-parallel degree",
    )
    for token in required:
        assert token in message, f"{token!r} missing from the named raise"

    # THE TWO PROHIBITIONS: the message must stop being evidence of a gap it is
    # not, and must stop naming the wrong degree.
    for token in ("G4", "tensor-parallel degree"):
        assert token not in message, f"{token!r} must not appear in the raise"

    _record(
        c3_raised=f"{raised}/1",
        c3_error_type=type(gate.value).__name__,
        c3_error=message,
        c3_required_tokens_present=len(required),
        c3_forbidden_tokens_present=0,
        c3_remainder=RAGGED_REMAINDER,
    )


# ---------------------------------------------------------------------------
# C4 -- the counted zero, with a control that MOVES it (D1.5)
# ---------------------------------------------------------------------------


def test_the_expert_count_divisor_reads_the_world_size_in_exactly_zero_places():
    """The defect class is gone from the class, counted rather than described.

    D1.4 certifying component: an ``ast`` walk over
    ``model_fp8.py::Glm5NextRoutedExperts`` for arguments to
    ``require_uniform_expert_partition`` derived from the world size.
    """
    impl = _impl()
    source = Path(inspect.getsourcefile(impl)).read_text()

    def world_size_divisors(text: str) -> tuple[int, int]:
        """``(gate calls found, gate calls whose divisor reads the world size)``."""
        tree = ast.parse(text)
        klass = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == "Glm5NextRoutedExperts"
        )
        calls = divisors = 0
        for node in ast.walk(klass):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name != "require_uniform_expert_partition":
                continue
            calls += 1
            for arg in node.args[1:]:
                rendered = ast.unparse(arg)
                if "world_size" in rendered or "tp_degree" in rendered:
                    divisors += 1
        return calls, divisors

    gate_calls, world_size_reads = world_size_divisors(source)

    # The zero is not the zero of an absent call: the gate is still called once.
    assert gate_calls == 1
    assert world_size_reads == 0

    # D1.5 CONTROL -- the predicate MOVES. The same census over a counter-example
    # that passes the world size counts 1, so the 0 above distinguishes two
    # readable shapes rather than being all this predicate can say.
    counter_example = (
        "class Glm5NextRoutedExperts:\n"
        "    def __init__(self, text_config, world_size=None):\n"
        "        self.tp_degree = world_size\n"
        "        self.expert_partition = require_uniform_expert_partition(\n"
        "            self.num_routed_experts, self.tp_degree\n"
        "        )\n"
    )
    control_calls, control_reads = world_size_divisors(counter_example)
    assert control_calls == 1
    assert control_reads == 1

    _record(
        c4_gate_calls_in_class=gate_calls,
        c4_world_size_divisor_reads=world_size_reads,
        c4_control_gate_calls=control_calls,
        c4_control_world_size_divisor_reads=control_reads,
        c4_control_MOVES=f"{world_size_reads} -> {control_reads} on the counter-example",
    )


# ---------------------------------------------------------------------------
# C5 -- the TP freeze is unmoved and still shards the intermediate dimension
# ---------------------------------------------------------------------------


def test_the_tp_freeze_is_unmoved_and_still_shards_the_intermediate_dimension():
    """The repair removes a divisor, not the freeze.

    D1.4 certifying component: ``factory.py::TP_DEGREE_FREEZE`` and
    ``config.py::Glm5NextTextConfig.moe_intermediate_size``, read together as the
    per-rank intermediate width.
    """
    text_config = _text_config()

    assert fmod.TP_DEGREE_FREEZE == DECLARED_TP_DEGREE == 64

    # TP's job is the INTERMEDIATE dimension. The width is READ from the config
    # and recorded as a number, not written here.
    moe_intermediate = int(text_config.moe_intermediate_size)
    per_rank_intermediate = moe_intermediate // fmod.TP_DEGREE_FREEZE
    assert moe_intermediate % fmod.TP_DEGREE_FREEZE == 0
    assert per_rank_intermediate == 32

    # The two facts side by side ARE the repair: at this degree the expert count
    # does not divide, the intermediate width does, and C1 builds regardless.
    assert TOTAL_ROUTED_EXPERTS % fmod.TP_DEGREE_FREEZE == RAGGED_REMAINDER

    _record(
        c5_TP_DEGREE_FREEZE=fmod.TP_DEGREE_FREEZE,
        c5_moe_intermediate_size=moe_intermediate,
        c5_per_rank_intermediate_width=per_rank_intermediate,
        c5_experts_remainder_at_the_freeze=RAGGED_REMAINDER,
        c5_intermediate_remainder_at_the_freeze=moe_intermediate
        % fmod.TP_DEGREE_FREEZE,
        c5_all_readings=len(_READINGS),
    )
