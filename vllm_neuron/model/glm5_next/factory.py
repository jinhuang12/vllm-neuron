# SPDX-License-Identifier: Apache-2.0
"""Factory for glm-5.3-Flash (``glm5_next``) implementation selection.

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
# Expert sharding.
#
# The arithmetic lives here rather than in the modeling module so that nothing on
# this path imports ``model_fp8``: the factory's lazy implementation import is what
# lets the arch class be looked up without allocating a 45-layer stack.
# ---------------------------------------------------------------------------

#: The tensor-parallel degree this port targets. Not configurable here: no
#: environment variable and no config field overrides it, and a different degree is
#: passed explicitly by a caller that has one. The corroborating site in this repo
#: is ``vllm_neuron/functional/process_groups.py``, whose TRN2 8x8 mesh branch is
#: gated on ``group_size == 64``.
TP_DEGREE_FREEZE = 64


class RaggedExpertPartitionError(ValueError):
    """The routed experts do not divide evenly over the expert-parallel degree.

    The two silent repairs a partitioner could reach for are both refused here:
    padding invents experts the checkpoint does not contain, and flooring drops
    experts it does. The fork's own precedent floor-divides with no gate at all
    (``gpt_oss/model_bf16.py``), which is the behaviour this named raise refuses.

    The subject is the expert-parallel degree, not the tensor-parallel one: it is not
    the tensor-parallel degree that divides an expert bank. With expert parallelism
    off the degree is ``1``, every expert is local on every rank and no division
    happens, so this error is unreachable at that setting whatever the
    tensor-parallel degree is. It is reachable only when a caller partitions over a
    degree that does not divide the bank.
    """


@dataclass(frozen=True)
class ExpertPartition:
    """One exact-coverage assignment of routed experts to ranks.

    ``counts[rank]`` experts live on ``rank``, starting at global expert index
    ``offsets[rank]``. The assignment is exact by construction: every expert is
    placed on exactly one rank, so :attr:`dropped` and :attr:`duplicated` are both
    ``0`` for any partition this module builds. Those two are exposed as measurable
    properties rather than assumed, because the fork's own expert-partition
    precedent (``gpt_oss/model_bf16.py``, floor division with no raggedness gate)
    drops 32 of this checkpoint's 288 experts at a degree of 64 -- so "0 dropped" is
    a real distinction between two reachable behaviours, not a tautology.

    ``tp_degree`` holds whatever degree the caller partitions over, which is the
    expert-parallel degree on every path in this module. The field keeps that name
    because renaming a frozen-dataclass field cascades through every reader of the
    returned partition and buys nothing; read it as "ranks partitioned over".
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

    Ragged-aware and lossless: the first ``num_experts % tp_degree`` ranks take one
    extra expert, so the assignment covers every expert exactly once whether or not
    the division is even. Nothing is padded and nothing is floored.

    It does not decide whether the result is usable; that is
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
    """The router precondition: top-k cannot select more experts than exist.

    Lives on the sharding path rather than in ``config.py``'s ``__post_init__``
    because it is a cross-field question about routing rather than the
    well-formedness of one field -- and because a config dataclass that asks it at
    construction time refuses a structural key-mapping fixture that never routes a
    token (a 4-expert bank inheriting the checkpoint's top-8 default).
    ``config.py``'s ``_validate_expert_counts`` docstring is the single authority for
    that boundary; this function is the half it names.

    Raises :class:`~vllm_neuron.model.glm5_next.config.Glm5NextExpertConfigError` --
    an expert-count error, deliberately not :class:`RaggedExpertPartitionError`, so a
    caller can tell "the counts are incoherent" from "the counts are fine but do not
    shard uniformly".
    """
    num_experts = int(num_experts)
    experts_per_tok = int(experts_per_tok)
    if experts_per_tok > num_experts:
        raise Glm5NextExpertConfigError(
            f"num_experts_per_tok={experts_per_tok} exceeds "
            f"n_routed_experts={num_experts}; top-k cannot select more experts "
            "than the bank contains"
        )


def _resolve_ep_degree(ep_degree: int | None) -> int:
    """The expert-parallel degree to divide the bank by.

    ``None`` means "ask the process group", which answers ``1`` when expert
    parallelism was never initialised -- the route under which the bank builds whole.
    An explicit value is how a caller that has a degree supplies one.

    The getter is reached through the module rather than through a name bound at
    import time, for two reasons that are the same reason: a module attribute is what
    a test can patch, and a module attribute is what reflects a process group
    initialised after this module was imported. The import is function-local so that
    importing this factory still pulls in no parallel state.
    """
    if ep_degree is not None:
        return int(ep_degree)
    from vllm_neuron.parallel import neuron_parallel_state

    return int(neuron_parallel_state.get_neuron_ep_degree())


def require_uniform_expert_partition(
    num_experts: int, ep_degree: int
) -> ExpertPartition:
    """The gate: return the partition, or raise :class:`RaggedExpertPartitionError`.

    A uniform per-rank expert count is required because the routed-expert bank's
    parameter shapes carry it -- one tensor per projection covering every local
    expert. A ragged split has no single shape, and the two ways to manufacture one
    both change the model: padding invents experts, flooring drops them.

    ``ep_degree`` is the expert-parallel degree. A parameter named for the
    tensor-parallel degree is the premise that made this gate refuse a bank the fork
    itself admits, which is why it carries this name.
    """
    partition = partition_experts(num_experts, ep_degree)
    if partition.is_uniform:
        return partition

    remainder = partition.remainder
    pad_to = num_experts + (ep_degree - remainder)
    floor_to = num_experts - remainder
    raise RaggedExpertPartitionError(
        f"{num_experts} routed experts do not divide evenly over "
        f"expert-parallel degree {ep_degree}: {num_experts} % {ep_degree} == "
        f"{remainder}. A uniform per-rank expert count is required because the "
        f"expert-bank parameter shapes carry it. Neither silent repair is taken: "
        f"padding to {pad_to} would invent {pad_to - num_experts} experts the "
        f"checkpoint does not contain, and flooring to {floor_to} would drop "
        f"{num_experts - floor_to} experts it does contain, which is the shape "
        f"this fork's own floor-division precedent produces. What is refused "
        f"here is the expert-parallel degree: with expert parallelism off that "
        f"degree is 1, "
        f"every expert is local on every rank and no division happens at all, so "
        f"a caller reaching this message is dividing {num_experts} experts by a "
        f"degree that does not divide them."
    )


class Glm5NextForConditionalGeneration(nn.Module):
    """Factory that selects the glm-5.3-Flash implementation.

    Extends nn.Module to satisfy vLLM's ModelRegistry requirements.

    The model runner passes ``text_neuron_config`` and ``vision_neuron_config``
    separately because the text decoder and the vision encoder carry their own
    parallelism and compilation settings -- the same split that
    ``Glm5NextConfig.from_configs`` already models.
    """

    #: The forward returns logits; the host's vLLM Sampler turns them into tokens.
    supports_on_device_sampling = False
    #: glm builds query and cached-KV carriers with independent lengths.
    supports_independent_prefill_buckets = True

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
        """Present so config-time interface validation passes.

        This class selects an implementation and is never a compute path -- the
        selected implementation owns embedding. No call site for this method exists
        in ``vllm_neuron``, so the raise is the permanent contract.
        """
        raise NotImplementedError(
            "Glm5NextForConditionalGeneration is a selection factory; "
            "embed_input_ids belongs to the selected implementation."
        )

    def compute_logits(self, hidden_states):
        """Present so config-time interface validation passes.

        Same contract as ``embed_input_ids``: the selected implementation owns
        logits, and no call site for this method exists in ``vllm_neuron``.
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

    # ── expert sharding ──────────────────────────────────────────────────

    @classmethod
    def expert_sharding_plan(
        cls,
        text_config: object,
        ep_degree: int | None = None,
    ) -> ExpertPartition:
        """The routed-expert shard plan for this arch, at the resolved EP degree.

        ``text_config`` is read by attribute (``n_routed_experts``) rather than
        typed, so this member adds no import and cannot create an import cycle with
        the modeling module.

        ``ep_degree`` defaults to ``None``, resolved by :func:`_resolve_ep_degree` to
        the live expert-parallel degree, which is ``1`` when expert parallelism was
        never initialised. At that degree every expert is local on every rank,
        nothing divides and nothing raises. Defaulting it to
        :data:`TP_DEGREE_FREEZE` instead would raise on this checkpoint's 288 routed
        experts -- one member of this module refusing a partition the expert bank
        beside it builds.

        An explicit argument is how a caller that has a degree supplies one.
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
        return require_uniform_expert_partition(
            int(num_experts), _resolve_ep_degree(ep_degree)
        )
