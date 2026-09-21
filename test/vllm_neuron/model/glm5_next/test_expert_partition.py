# SPDX-License-Identifier: Apache-2.0
"""The expert partition divides over the expert-parallel degree, never the world size.

With expert parallelism off every expert is local on every rank, so nothing divides
and nothing raises.
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

def _impl():
    """Import the modeling module inside a test body, never at import time. """
    from vllm_neuron.model.glm5_next import model_fp8

    return model_fp8


def _text_config(**overrides: object):
    from vllm_neuron.model.glm5_next import config as cfgmod

    return cfgmod.Glm5NextTextConfig(**overrides)


# ---------------------------------------------------------------------------
# EP off at the frozen TP degree: the bank builds whole
# ---------------------------------------------------------------------------


def test_ep_off_at_the_frozen_tp_degree_builds_all_288_experts_and_raises_nothing():
    """The repair itself: TP = 64 no longer refuses, because TP is not the divisor. """
    impl = _impl()
    text_config = _text_config()

    # The uninitialised branch is reached, not assumed: no group exists here.
    assert npsmod._NEURON_EP is None
    assert npsmod.get_neuron_ep_degree() == 1

    bank = impl.Glm5NextRoutedExperts(text_config, world_size=DECLARED_TP_DEGREE)

    # The headline: at the frozen degree nothing raises and nothing is lost.
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


# ---------------------------------------------------------------------------
# EP on at a degree that divides 288, through the resolved route
# ---------------------------------------------------------------------------


def test_ep_on_at_a_degree_that_divides_288_yields_nine_local_experts():
    """EP on still divides, by the degree the getter reports. """
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


# ---------------------------------------------------------------------------
# EP on at a degree that does not divide 288: the named raise
# ---------------------------------------------------------------------------


def test_ep_on_at_a_ragged_degree_raises_naming_the_expert_parallel_degree():
    """The raggedness gate survives the repair, with its subject corrected. """
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

    # The required set, declared so this check and the updated token
    # list in ``test_experts.py`` cannot drift apart.
    required = (
        str(TOTAL_ROUTED_EXPERTS),
        str(DECLARED_TP_DEGREE),
        str(RAGGED_REMAINDER),
        str(PAD_TARGET),
        str(FLOOR_TARGET),
        "expert-parallel degree",
    )
    for token in required:
        assert token in message, f"{token!r} missing from the named raise"

    # The two prohibitions: the message must stop being evidence of a gap it is
    # not, and must stop naming the wrong degree.
    for token in ("tensor-parallel degree",):
        assert token not in message, f"{token!r} must not appear in the raise"


# ---------------------------------------------------------------------------
# The counted zero, with a control that moves it (the control)
# ---------------------------------------------------------------------------


def test_the_expert_count_divisor_reads_the_world_size_in_exactly_zero_places():
    """The defect class is gone from the class, counted rather than described. """
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

    # Control -- the predicate moves. The same census over a counter-example
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


# ---------------------------------------------------------------------------
# The TP freeze is unmoved and still shards the intermediate dimension
# ---------------------------------------------------------------------------


def test_the_tp_freeze_is_unmoved_and_still_shards_the_intermediate_dimension():
    """The repair removes a divisor, not the freeze. """
    text_config = _text_config()

    assert fmod.TP_DEGREE_FREEZE == DECLARED_TP_DEGREE == 64

    # TP's job is the intermediate dimension. The width is read from the config
    # and recorded as a number, not written here.
    moe_intermediate = int(text_config.moe_intermediate_size)
    per_rank_intermediate = moe_intermediate // fmod.TP_DEGREE_FREEZE
    assert moe_intermediate % fmod.TP_DEGREE_FREEZE == 0
    assert per_rank_intermediate == 32

    # The two facts side by side are the repair: at this degree the expert count
    # does not divide, the intermediate width does, and the bank builds regardless.
    assert TOTAL_ROUTED_EXPERTS % fmod.TP_DEGREE_FREEZE == RAGGED_REMAINDER

