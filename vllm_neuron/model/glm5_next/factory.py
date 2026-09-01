# SPDX-License-Identifier: Apache-2.0
"""Factory for GLM-5.3-Flash (``glm5_next``) implementation selection.

Follows the package convention every other arch in this tree uses
(``qwen3/factory.py``, ``qwen3_vl/factory.py``): this module *defines* the
arch-named class that the registry registers, the class extends ``nn.Module``
so vLLM's ``ModelRegistry`` accepts it, and the concrete implementation module
is imported lazily inside the selection classmethod.

The lazy import is load-bearing, not stylistic: importing this module -- and
looking its class up through ``vllm_neuron.model.registry`` -- must never pull
in model code or allocate weights.
"""

from dataclasses import dataclass

import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig, VisionNeuronConfig

from .config import Glm5NextExpertConfigError

# ---------------------------------------------------------------------------
# Expert-sharding members -- ``inc-glm53f-031`` (WP7, M2 Lane B position 2).
#
# These are a PURE ADDITION BY MEMBER to a co-authored file: ``inc-glm53f-009``
# owns the class plus ``from_configs`` / ``_select_implementation``, and
# ``inc-glm53f-074`` owns the trailing ``**kwargs`` on ``__init__`` plus
# ``embed_input_ids`` / ``compute_logits``. No member of either is written
# here, and none of their signatures is modified -- that partition is the plan's
# section 11.A.1 register, and modifying a landed co-author's signature would be
# a recorded contradiction rather than a build-time edit.
#
# THE ARITHMETIC LIVES HERE, NOT IN THE MODELING MODULE, so that nothing on
# this path imports ``model_fp8``: the factory's lazy implementation import is
# the property ``test_factory.py``'s C03 certifies, and it is what lets the arch
# class be looked up without allocating a 45-layer stack.
# ---------------------------------------------------------------------------

#: The campaign's tensor-parallel degree. **REGISTERED, CITED, NEVER RE-DERIVED
#: AND NOT CONFIGURABLE HERE.** The registration is
#: ``approvals/DECISIONS.md`` section 6 -- *"Preconditions (registered): TP=64,
#: bf16 KV cache"* -- and the corroborating read-only site in this repo is
#: ``vllm_neuron/functional/process_groups.py:111``, whose TRN2 8x8 mesh branch
#: is gated on ``group_size == 64``. No environment variable and no config field
#: overrides this value; a different degree is passed explicitly by a caller
#: that has one, which is what makes the freeze's consequence measurable.
TP_DEGREE_FREEZE = 64


class RaggedExpertPartitionError(ValueError):
    """The routed experts do not partition uniformly over the given TP degree.

    This is the campaign's **G4** gap surfaced as a NAMED RAISE. The two silent
    repairs a partitioner could reach for are both refused here: padding invents
    experts the checkpoint does not contain, and flooring drops experts it does.
    The blocker is the plugin's TP degree freeze, not the substrate.
    """


@dataclass(frozen=True)
class ExpertPartition:
    """One exact-coverage assignment of routed experts to tensor-parallel ranks.

    ``counts[rank]`` experts live on ``rank``, starting at global expert index
    ``offsets[rank]``. The assignment is **exact by construction**: every expert
    is placed on exactly one rank, so :attr:`dropped` and :attr:`duplicated` are
    both ``0`` for any partition this module builds. Those two are exposed as
    measurable properties rather than assumed, because the fork's own landed
    expert-partition precedent (``gpt_oss/model_bf16.py:1072``, floor division
    with no raggedness gate) drops **32** of this checkpoint's **288** experts at
    ``TP_DEGREE_FREEZE`` -- so "0 dropped" is a real distinction between two
    reachable behaviours, not a tautology.
    """

    num_experts: int
    tp_degree: int
    counts: tuple[int, ...]
    offsets: tuple[int, ...]

    @property
    def remainder(self) -> int:
        """``num_experts % tp_degree`` -- the raggedness, as a number."""
        return self.num_experts % self.tp_degree

    @property
    def is_uniform(self) -> bool:
        """True when every rank carries the same expert count."""
        return len(set(self.counts)) == 1

    @property
    def assigned(self) -> int:
        """Total expert slots assigned, counting a duplicate twice."""
        return sum(self.counts)

    @property
    def covered(self) -> int:
        """Distinct global expert indices assigned to some rank."""
        covered: set[int] = set()
        for rank in range(self.tp_degree):
            covered.update(self.local_expert_indices(rank))
        return len(covered)

    @property
    def dropped(self) -> int:
        """Experts the assignment never places. Exactly ``0`` here."""
        return self.num_experts - self.covered

    @property
    def duplicated(self) -> int:
        """Expert slots placed more than once. Exactly ``0`` here."""
        return self.assigned - self.covered

    def local_expert_indices(self, rank: int) -> tuple[int, ...]:
        """The global expert indices ``rank`` owns, in ascending order."""
        if not 0 <= rank < self.tp_degree:
            raise ValueError(
                f"rank {rank} is outside the partition's {self.tp_degree} ranks"
            )
        start = self.offsets[rank]
        return tuple(range(start, start + self.counts[rank]))


def partition_experts(num_experts: int, tp_degree: int) -> ExpertPartition:
    """Assign ``num_experts`` routed experts across ``tp_degree`` ranks, exactly.

    Ragged-aware and lossless: the first ``num_experts % tp_degree`` ranks take
    one extra expert, so the assignment covers every expert exactly once whether
    or not the division is even. **Nothing is padded and nothing is floored** --
    that is the whole point of this function existing beside the gate below.

    It does not decide whether the result is *usable*; that is
    :func:`require_uniform_expert_partition`'s job.
    """
    num_experts = int(num_experts)
    tp_degree = int(tp_degree)
    if num_experts < 1:
        raise ValueError(f"num_experts must be >= 1, got {num_experts}")
    if tp_degree < 1:
        raise ValueError(f"tp_degree must be >= 1, got {tp_degree}")

    base, remainder = divmod(num_experts, tp_degree)
    counts = tuple(
        base + 1 if rank < remainder else base for rank in range(tp_degree)
    )
    offsets: list[int] = []
    running = 0
    for count in counts:
        offsets.append(running)
        running += count
    return ExpertPartition(
        num_experts=num_experts,
        tp_degree=tp_degree,
        counts=counts,
        offsets=tuple(offsets),
    )


def require_routable_expert_counts(num_experts: int, experts_per_tok: int) -> None:
    """The ROUTER precondition: top-k cannot select more experts than exist.

    Lives on the sharding path rather than in ``config.py``'s ``__post_init__``
    because it is a **cross-field** question about routing, not
    well-formedness of one field -- and because a config dataclass that asks it
    at construction time refuses a structural key-mapping fixture that never
    routes a token (``inc-glm53f-011``'s ``mini_config``: a 4-expert bank
    inheriting the checkpoint's top-8 default). ``config.py``'s
    ``_validate_expert_counts`` docstring is the single authority for that
    boundary; this function is the half it names.

    Raises :class:`~vllm_neuron.model.glm5_next.config.Glm5NextExpertConfigError`
    -- an expert-COUNT error, deliberately not
    :class:`RaggedExpertPartitionError`, so a caller can tell "the counts are
    incoherent" from "the counts are fine but do not shard uniformly".
    """
    num_experts = int(num_experts)
    experts_per_tok = int(experts_per_tok)
    if experts_per_tok > num_experts:
        raise Glm5NextExpertConfigError(
            f"num_experts_per_tok={experts_per_tok} exceeds "
            f"n_routed_experts={num_experts}; top-k cannot select more experts "
            "than the bank contains"
        )


def require_uniform_expert_partition(
    num_experts: int, tp_degree: int
) -> ExpertPartition:
    """The gate: return the partition, or raise :class:`RaggedExpertPartitionError`.

    A uniform per-rank expert count is required because the routed-expert bank's
    parameter shapes carry it -- one tensor per projection covering every local
    expert. A ragged split has no single shape, and the two ways to manufacture
    one both change the model: padding invents experts, flooring drops them.
    """
    partition = partition_experts(num_experts, tp_degree)
    if partition.is_uniform:
        return partition

    remainder = partition.remainder
    pad_to = num_experts + (tp_degree - remainder)
    floor_to = num_experts - remainder
    raise RaggedExpertPartitionError(
        f"{num_experts} routed experts do not partition uniformly over "
        f"tensor-parallel degree {tp_degree}: {num_experts} % {tp_degree} == "
        f"{remainder}. A uniform per-rank expert count is required because the "
        f"expert-bank parameter shapes carry it. Neither silent repair is taken: "
        f"padding to {pad_to} would invent {pad_to - num_experts} experts the "
        f"checkpoint does not contain, and flooring to {floor_to} would drop "
        f"{num_experts - floor_to} experts it does contain -- the shape the "
        f"fork's own landed floor-division precedent produces "
        f"(gpt_oss/model_bf16.py:1072). The blocker is this plugin's registered "
        f"TP degree freeze of {TP_DEGREE_FREEZE}, not the substrate; it is "
        f"campaign gap G4 and its disposition is the lead's."
    )


class Glm5NextForConditionalGeneration(nn.Module):
    """Factory that selects the GLM-5.3-Flash implementation.

    Extends nn.Module to satisfy vLLM's ModelRegistry requirements.

    The model runner passes ``text_neuron_config`` and ``vision_neuron_config``
    separately because the text decoder and the vision encoder carry their own
    parallelism and compilation settings -- the same split that
    ``Glm5NextConfig.from_configs`` already models.
    """

    def __init__(
        self,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None = None,
        vision_neuron_config: VisionNeuronConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self._model = self._select_implementation(
            hf_config, text_neuron_config, vision_neuron_config
        )

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def embed_input_ids(self, input_ids):
        """Boundary member: present so config-time interface validation passes.

        This class is a selection seam, never a compute path -- the selected
        implementation owns embedding. No call site for this method exists in
        ``vllm_neuron``, so the raise is the permanent contract.
        """
        raise NotImplementedError(
            "Glm5NextForConditionalGeneration is a selection factory; "
            "embed_input_ids belongs to the selected implementation."
        )

    def compute_logits(self, hidden_states):
        """Boundary member: present so config-time interface validation passes.

        Same contract as ``embed_input_ids`` -- the selected implementation
        owns logits, and no call site for this method exists in
        ``vllm_neuron``.
        """
        raise NotImplementedError(
            "Glm5NextForConditionalGeneration is a selection factory; "
            "compute_logits belongs to the selected implementation."
        )

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None = None,
        vision_neuron_config: VisionNeuronConfig | None = None,
    ) -> nn.Module:
        return cls._select_implementation(
            hf_config, text_neuron_config, vision_neuron_config
        )

    @classmethod
    def _select_implementation(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig | None,
        vision_neuron_config: VisionNeuronConfig | None,
    ) -> nn.Module:
        # Blockwise-FP8 is the only weight format in scope for this checkpoint,
        # so there is a single implementation module and no format branch here.
        # The import stays local so that registration and arch lookup work
        # without importing model code or allocating weights.
        from .model_fp8 import Glm5NextForConditionalGeneration as Model

        return Model.from_configs(
            hf_config,
            text_neuron_config=text_neuron_config,
            vision_neuron_config=vision_neuron_config,
        )

    # ── expert sharding (``inc-glm53f-031``) ─────────────────────────────
    # Appended after the landed members rather than woven between them, so the
    # co-authorship partition is readable in the diff: no line of ``-009``'s or
    # ``-074``'s members moves.

    @classmethod
    def expert_sharding_plan(
        cls,
        text_config: object,
        tp_degree: int = TP_DEGREE_FREEZE,
    ) -> ExpertPartition:
        """The routed-expert shard plan for this arch, at the frozen TP degree.

        ``text_config`` is read by attribute (``n_routed_experts``) rather than
        typed, so this member adds no import to a co-authored file and cannot
        create an import cycle with the modeling module.

        The default ``tp_degree`` **is** the registered freeze
        (:data:`TP_DEGREE_FREEZE`); on this checkpoint's 288 routed experts that
        default RAISES :class:`RaggedExpertPartitionError`, which is the
        designed, visible consequence of the freeze rather than a defect in this
        member. An explicit argument is how a caller that has a different degree
        supplies one; it is not a configuration knob for the freeze.
        """
        num_experts = getattr(text_config, "n_routed_experts", None)
        if num_experts is None:
            raise ValueError(
                "text_config carries no n_routed_experts; the expert shard plan "
                f"cannot be resolved from {type(text_config).__name__}"
            )
        experts_per_tok = getattr(text_config, "num_experts_per_tok", None)
        if experts_per_tok is not None:
            require_routable_expert_counts(num_experts, experts_per_tok)
        return require_uniform_expert_partition(int(num_experts), int(tp_degree))
