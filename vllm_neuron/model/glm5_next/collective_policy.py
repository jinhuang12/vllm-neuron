# SPDX-License-Identifier: Apache-2.0
"""The one entry point for GLM-5.3-Flash's tensor-parallel row-parallel all-reduce.

Each decoder layer reduces two row-parallel partials: one at its attention output
projection (KDA or MLA, by layer type) and one at its feed-forward half. The layer code
hands each partial to :func:`reduce_row_parallel` with the :class:`RowParallelSite` it
comes from and the :class:`StepRows` it was built with, and does nothing else: the wire
dtype, the compiler's tiling and any future overlap are decided here, keyed by site and
step, so a branch that composes these layers (a fused glue path, an extra MTP layer
built from ``_ffn_half``) inherits the policy unchanged.

Three switches, all registered in :mod:`vllm_neuron.envs` and read when a site runs --
which on the device is trace time:

``VLLM_NEURON_TP_ALLREDUCE_DTYPE`` = ``bf16`` (default) | ``fp32``
    The dtype a partial crosses the wire in, at the steps
    ``VLLM_NEURON_TP_ALLREDUCE_DTYPE_PHASES`` names. ``bf16`` rounds this rank's partial
    to bfloat16 and reduces that copy, which halves the bytes every rank moves. ``fp32``
    is the full-width path, which every other step also takes: the coordinator reduces
    the partial itself, in place, in the dtype it was computed in. Every site casts the
    reduced value to its carrier dtype right after; in this model the carrier between
    layers is the bfloat16 residual stream, so the ``fp32`` path already rounds the
    reduced sum to bfloat16 once, and ``bf16`` adds the rounding of each rank's partial
    and of the reduction's intermediate sums.

    Derived, not yet measured in the served model: on isolated 64-rank all-reduces one
    8 MiB bf16 operation takes 156.57 us against 261.58 us for the fused 16 MiB fp32
    one, so the 90 sites of a 1024-token chunk save 9.45 ms, about 2.42 s at 262,127
    tokens; against the compiler's tiled fp32 form, the served model's per-piece
    transfer times give 3.61 s. The cast is not counted. The user approved the
    numerics tolerance on 2026-10-14.

``VLLM_NEURON_TP_ALLREDUCE_DTYPE_PHASES`` = ``prefill`` (default) | ``all`` | a comma list
    The steps that reduce in the wire dtype: ``all``, or any of ``prefill``, ``decode``
    and ``verify`` (``functional.glue.PHASES``). A site tells its step from its
    partial's row count (:class:`StepRows`). The default narrows the prefill chunks
    only: a decode or verify step takes the ``fp32`` path, so its graph is the
    ``fp32`` wire's. A prefill chunk's sums still reach the later steps, through the
    caches the chunk writes.

``VLLM_NEURON_TP_ALLREDUCE_FUSE`` = ``1`` (default) | ``0``
    The model issues one collective per site. Left alone, neuronx-cc's
    ``SimpleAllReduceTiling`` pass splits an all-reduce larger than 8 MiB into up to
    four tiles, two per fp32 site at a 1024-token chunk. That is a compiler decision
    for the whole graph, not one a call can make, so this switch only names the
    compiler argument that turns the pass off (:func:`fuse_compiler_args`), and
    ``NeuronModelRunner.load_model`` appends it to its ``compiler_args``. One collective
    per site moves the same bytes in fewer operations; its sums are not bitwise
    those of the tiled form. ``0`` keeps the compiler's tiles.

    Measured on the lever-A build (3417e57c) with the fp32 wire at 262,127 tokens: TTFT
    -1,199.8 ms against an arm-0 restart spread of 27.5 ms (the collectives -4.52 ms
    per chunk); GSM8K strict-match 199 against 198 / 198 at n = 200, one discordant
    question, McNemar p = 1. The user approved the tolerance on 2026-10-14.

At world size 1 there is no coordinator and nothing happens in any setting: no
collective and no cast, so a single-rank run is bitwise unchanged.

The sequence-parallel prefill route uses two more entry points, called only when the
model's sequence-parallel switch is on (the model code reads it; this module does not):
:func:`scatter_row_parallel` replaces a site's all-reduce with a reduce-scatter over the
token rows, so each rank keeps the summed rows it owns, and :func:`gather_sequence`
all-gathers those row blocks back to every token before a sub-block that needs them.
The reduce-scatter follows the same wire-dtype rule as the all-reduce: the step its
partial's row count names (:class:`StepRows`), then the two dtype switches above. With
the switch off neither is called, and the program is the all-reduce form above.
"""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from vllm_neuron import envs
from vllm_neuron.functional.glue import PHASES, phase_of_rows

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator

    from vllm_neuron.model.neuron_config import NeuronConfig

__all__ = [
    "DTYPE_ENV",
    "DTYPE_PHASES_ENV",
    "FUSE_COMPILER_ARG",
    "FUSE_ENV",
    "CollectivePolicyError",
    "RowParallelSite",
    "StepRows",
    "allreduce_dtype_phases",
    "allreduce_fuse_enabled",
    "allreduce_wire_dtype",
    "fuse_compiler_args",
    "gather_sequence",
    "reduce_row_parallel",
    "scatter_row_parallel",
]

#: The wire-dtype switch (``vllm_neuron.envs``).
DTYPE_ENV = "VLLM_NEURON_TP_ALLREDUCE_DTYPE"

#: The switch naming the steps that reduce in the wire dtype (``vllm_neuron.envs``).
DTYPE_PHASES_ENV = "VLLM_NEURON_TP_ALLREDUCE_DTYPE_PHASES"

#: The fusion switch (``vllm_neuron.envs``).
FUSE_ENV = "VLLM_NEURON_TP_ALLREDUCE_FUSE"

#: The neuronx-cc argument that turns ``SimpleAllReduceTiling`` off; the Tensorizer's
#: other options are kept. On neuronx-cc 2.27 a 16 MiB fp32 all-reduce lowers to one
#: collective with it and to two without it.
FUSE_COMPILER_ARG = "--tensorizer-options=--disable-tiling-allreduce"

#: The values ``VLLM_NEURON_TP_ALLREDUCE_DTYPE`` accepts, and the dtype each names.
_WIRE_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}

#: The ``VLLM_NEURON_TP_ALLREDUCE_DTYPE_PHASES`` value that names every step.
_EVERY_PHASE = "all"

#: The partial dtypes a site may hand in: fp32 at every site on the device; bf16 where a
#: CPU route returns its sum already rounded to bfloat16 (the routed experts).
_PARTIAL_DTYPES = (torch.float32, torch.bfloat16)


class CollectivePolicyError(ValueError):
    """A switch value or a partial this policy does not define, refused by name."""


class RowParallelSite(str, enum.Enum):
    """Where a row-parallel partial comes from.

    Every site hands :func:`reduce_row_parallel` (or :func:`scatter_row_parallel`) this
    rank's partial sum at the full
    output width, ``[tokens, hidden_size]``, freshly computed (never a view of a weight
    or of the caller's residual), in fp32 on the device. Every site follows the same
    policy today; the key lets a per-site policy change one site without touching the
    layer code.
    """

    #: ``Glm5NextKDAAttention._gated_output``: the gated-norm output times this rank's
    #: rows of ``o_proj_weight``.
    KDA_O_PROJ = "kda_o_proj"
    #: ``Glm5NextMLAAttention.project_output``: this rank's heads through
    #: ``o_proj_weight``.
    MLA_O_PROJ = "mla_o_proj"
    #: ``Glm5NextModel._ffn_half``: the dense MLP's down projection, or the routed and
    #: shared experts' sum, of this rank's shard.
    FFN = "ffn"


@dataclasses.dataclass(frozen=True)
class StepRows:
    """How a row-parallel site tells which step a call serves: by its row count.

    A site sees only ``[tokens, hidden_size]``, so it reads the step as the mHC glue
    sites do (``functional.glue.phase_of_rows``): a verify-step row count is ``verify``,
    a count up to the largest decode batch is ``decode``, and a larger one is a
    ``prefill`` chunk. A prefill chunk whose row count a decode batch or the verify step
    can also have reads as that step, and takes that step's wire.

    Each module that reduces builds one with itself (:meth:`of_server`): the verify
    step's row counts come from the speculative config, which is current only while the
    model is built.

    Attributes:
        max_decode_rows: the largest decode batch, ``max(num_seqs_buckets)``. ``None``
            when the module is built without the runner's buckets; then no call's step
            is known.
        verify_rows: the verify step's row counts (``functional.glue.verify_rows``);
            empty without speculation.
    """

    max_decode_rows: int | None
    verify_rows: frozenset[int] = frozenset()

    @classmethod
    def of_server(cls, neuron_config: NeuronConfig | None) -> StepRows:
        """The rule for the server ``neuron_config`` describes.

        Reads the runner's decode buckets (``num_seqs_buckets``) and, through
        ``mtp.shadow_draft_k``, the speculative config's draft count, so it is called
        where the model is built.
        """
        decode_buckets = getattr(neuron_config, "num_seqs_buckets", None)
        if not decode_buckets:
            return cls(None)
        from vllm_neuron.functional.glue import verify_draft_k, verify_rows

        from .mtp import shadow_draft_k

        return cls(
            max(decode_buckets),
            verify_rows(decode_buckets, verify_draft_k(shadow_draft_k())),
        )

    def phase(self, rows: int) -> str | None:
        """The step a call of ``rows`` rows serves, one of ``functional.glue.PHASES``;
        ``None`` when the module was built without the runner's buckets."""
        if self.max_decode_rows is None:
            return None
        return phase_of_rows(rows, self.max_decode_rows, self.verify_rows)


def allreduce_dtype_phases() -> frozenset[str]:
    """The steps whose partials cross the wire in the wire dtype.

    Read from ``VLLM_NEURON_TP_ALLREDUCE_DTYPE_PHASES`` through :mod:`vllm_neuron.envs`:
    ``all``, or a comma list of names in ``functional.glue.PHASES``.

    Raises:
        CollectivePolicyError: the switch names no step, or a name that is not a step.
    """
    value = envs.VLLM_NEURON_TP_ALLREDUCE_DTYPE_PHASES
    if value == _EVERY_PHASE:
        return frozenset(PHASES)
    names = [name.strip() for name in value.split(",")]
    unknown = [name for name in names if name not in PHASES]
    if unknown:
        raise CollectivePolicyError(
            f"{DTYPE_PHASES_ENV}={value!r} names {unknown}, which is not a step; use "
            f"{_EVERY_PHASE} or a comma list of {', '.join(PHASES)}, or set "
            f"{DTYPE_ENV}=fp32 for the full-width reduction at every step"
        )
    return frozenset(names)


def allreduce_wire_dtype(phase: str | None) -> torch.dtype:
    """The dtype a partial of a step in ``phase`` crosses the wire in.

    ``VLLM_NEURON_TP_ALLREDUCE_DTYPE``'s dtype when
    ``VLLM_NEURON_TP_ALLREDUCE_DTYPE_PHASES`` names ``phase``, else fp32. A step the
    site cannot tell (``phase`` ``None``, see :meth:`StepRows.phase`) takes the wire
    dtype only when the switch names every step.

    Raises:
        CollectivePolicyError: either switch holds a value this module does not define.
    """
    key = envs.VLLM_NEURON_TP_ALLREDUCE_DTYPE
    if key not in _WIRE_DTYPES:
        raise CollectivePolicyError(
            f"{DTYPE_ENV}={key!r} is not one of {sorted(_WIRE_DTYPES)}; unset it "
            f"for the bf16 default, or set fp32 for the full-width reduction"
        )
    phases = allreduce_dtype_phases()
    if phase is None:
        narrowed = all(step in phases for step in PHASES)
    else:
        narrowed = phase in phases
    return _WIRE_DTYPES[key] if narrowed else torch.float32


def allreduce_fuse_enabled() -> bool:
    """Whether the compiler should keep each all-reduce one collective.

    Read from ``VLLM_NEURON_TP_ALLREDUCE_FUSE`` through :mod:`vllm_neuron.envs`.

    Raises:
        CollectivePolicyError: the switch is not an integer, the form every boolean
            switch in :mod:`vllm_neuron.envs` takes.
    """
    try:
        return envs.VLLM_NEURON_TP_ALLREDUCE_FUSE
    except ValueError as error:
        raise CollectivePolicyError(
            f"{FUSE_ENV} must be 0 or 1; set it to 0 to keep the compiler's "
            f"all-reduce tiling"
        ) from error


def fuse_compiler_args() -> list[str]:
    """The neuronx-cc arguments the fusion switch asks for: empty unless it is on."""
    return [FUSE_COMPILER_ARG] if allreduce_fuse_enabled() else []


def reduce_row_parallel(
    partial: torch.Tensor,
    *,
    site: RowParallelSite,
    group: GroupCoordinator | None,
    steps: StepRows,
) -> torch.Tensor:
    """Sum this rank's row-parallel ``partial`` across the tensor-parallel ``group``.

    Args:
        partial: this rank's partial sum, ``[tokens, hidden_size]`` (any shape reduces;
            the sum is elementwise, and the leading extent is the row count ``steps``
            reads), fp32 or bf16, fresh as :class:`RowParallelSite` states: the fp32
            path reduces it in place.
        site: the :class:`RowParallelSite` the partial comes from.
        group: the tensor-parallel coordinator, or ``None`` at world size 1. Its
            ``all_reduce`` is called in statement form, the result read from the tensor
            handed in, as vLLM's device communicator reduces in place.
        steps: the calling module's :class:`StepRows`, which tells this call's step
            from ``partial``'s row count.

    Returns:
        ``partial`` itself when ``group`` is ``None``. On the fp32 path, ``partial``
        itself, reduced in place in its own dtype: the operations the sites ran before
        this entry point. In the bf16 wire dtype, a bfloat16 tensor holding the reduced
        sum (``partial`` reduced in place if it already is bfloat16, else a new tensor,
        with ``partial`` left holding this rank's own share). The caller casts the
        result to its carrier.

    Raises:
        CollectivePolicyError: ``site`` is not a :class:`RowParallelSite` member,
            ``partial`` is neither fp32 nor bf16, or a switch holds a value this module
            does not define.
    """
    _check_partial(partial, site)
    if group is None:
        return partial
    wire_dtype = allreduce_wire_dtype(steps.phase(int(partial.shape[0])))
    if wire_dtype is torch.float32 or partial.dtype is wire_dtype:
        group.all_reduce(partial)
        return partial
    wire = partial.to(wire_dtype)
    group.all_reduce(wire)
    return wire


def scatter_row_parallel(
    partial: torch.Tensor,
    *,
    site: RowParallelSite,
    group: GroupCoordinator | None,
    steps: StepRows,
) -> torch.Tensor:
    """Sum ``partial`` across ``group`` and keep only this rank's block of token rows.

    The sequence-parallel form of :func:`reduce_row_parallel`: one reduce-scatter on
    dim 0 in place of the all-reduce, so the sum crosses the wire once and each rank
    receives ``rows / world_size`` rows of it. Rank ``r`` of ``W`` gets rows
    ``[r * rows / W, (r + 1) * rows / W)`` of the summed ``[rows, ...]`` tensor, the
    order :func:`gather_sequence` restores.

    Args:
        partial: this rank's partial sum, ``[tokens, hidden_size]``, fp32 or bf16,
            fresh as :class:`RowParallelSite` states. Dim 0 is the token rows; its
            size must divide by the group's world size.
        site: the :class:`RowParallelSite` the partial comes from.
        group: the tensor-parallel coordinator, or ``None`` at world size 1.
        steps: the calling module's :class:`StepRows`, which tells this call's step
            from ``partial``'s row count, the whole chunk's.

    Returns:
        ``partial`` itself when ``group`` is ``None``. Otherwise a new tensor, this
        rank's ``[tokens / world_size, ...]`` block of the sum, in the wire dtype that
        :func:`reduce_row_parallel` would use for this step: ``partial``'s own dtype on
        the fp32 path, bfloat16 in the bf16 wire dtype. Unlike
        :func:`reduce_row_parallel` the result is never ``partial``, so the caller reads
        the return value. The caller casts it to its carrier.

    Raises:
        CollectivePolicyError: ``site`` is not a :class:`RowParallelSite` member,
            ``partial`` is neither fp32 nor bf16, its rows do not divide by the world
            size, or a switch holds a value this module does not define.
    """
    _check_partial(partial, site)
    if group is None:
        return partial
    if partial.dim() == 0 or partial.shape[0] % group.world_size != 0:
        raise CollectivePolicyError(
            f"the {site.value} site handed a partial of shape {tuple(partial.shape)}; "
            f"a reduce-scatter over {group.world_size} ranks needs dim 0 (the token "
            f"rows) to divide by {group.world_size}"
        )
    wire_dtype = allreduce_wire_dtype(steps.phase(int(partial.shape[0])))
    wire = partial if wire_dtype is torch.float32 else partial.to(wire_dtype)
    return group.reduce_scatter(wire, dim=0)


def gather_sequence(
    tensors: Sequence[torch.Tensor], *, group: GroupCoordinator | None
) -> tuple[torch.Tensor, ...]:
    """All-gather each rank's block of token rows back to every row, in rank order.

    The inverse of :func:`scatter_row_parallel`'s row split: rank ``r``'s rows land at
    ``[r * rows, (r + 1) * rows)`` of each result. Several tensors cross the wire as
    one collective: they are stacked on a new dim 1, gathered on dim 0 and split back.
    No dtype changes.

    Args:
        tensors: one or more tensors of one dtype and one shape, ``[rows, ...]``, dim 0
            this rank's token rows.
        group: the tensor-parallel coordinator, or ``None`` at world size 1.

    Returns:
        One tensor per input, in input order, each ``[world_size * rows, ...]``. With
        ``group`` ``None`` the inputs themselves. With two or more inputs the results
        are views of one gathered buffer (each strided on dim 0 by the input count),
        so a consumer that needs contiguous rows makes them contiguous itself.

    Raises:
        CollectivePolicyError: ``tensors`` is empty, or its tensors differ in dtype or
            shape.
    """
    if len(tensors) == 0:
        raise CollectivePolicyError("gather_sequence needs at least one tensor")
    first = tensors[0]
    for other in tensors[1:]:
        if other.dtype != first.dtype or other.shape != first.shape:
            raise CollectivePolicyError(
                f"gather_sequence gathers tensors of one dtype and shape; got "
                f"{first.dtype} {tuple(first.shape)} and {other.dtype} "
                f"{tuple(other.shape)}"
            )
    if group is None:
        return tuple(tensors)
    if len(tensors) == 1:
        # ``all_gather_into_tensor`` refuses a non-contiguous input; this is a no-op
        # on the contiguous block a reduce-scatter returns.
        return (group.all_gather(first.contiguous(), dim=0),)
    gathered = group.all_gather(torch.stack(tuple(tensors), dim=1), dim=0)
    return tuple(gathered.unbind(dim=1))


def _check_partial(partial: torch.Tensor, site: RowParallelSite) -> None:
    """Refuse a ``site`` that is not a member and a ``partial`` of another dtype."""
    # ``isinstance``, not ``RowParallelSite(site)``: the sites run inside a
    # ``fullgraph=True`` capture, and dynamo cannot trace the enum's lookup by value.
    if not isinstance(site, RowParallelSite):
        raise CollectivePolicyError(f"{site!r} is not a RowParallelSite member")
    if partial.dtype not in _PARTIAL_DTYPES:
        raise CollectivePolicyError(
            f"the {site.value} site handed a {partial.dtype} partial; a row-parallel "
            f"partial is one of {[str(d) for d in _PARTIAL_DTYPES]}"
        )
