# SPDX-License-Identifier: Apache-2.0
"""The one entry point for GLM-5.3-Flash's tensor-parallel row-parallel all-reduce.

Each decoder layer reduces two row-parallel partials: one at its attention output
projection (KDA or MLA, by layer type) and one at its feed-forward half. The layer code
hands each partial to :func:`reduce_row_parallel` with the :class:`RowParallelSite` it
comes from, and does nothing else: the wire dtype, the compiler's tiling and any future
overlap are decided here, keyed by site, so a branch that composes these layers (a fused
glue path, an extra MTP layer built from ``_ffn_half``) inherits the policy unchanged.

Two switches, both registered in :mod:`vllm_neuron.envs` and read when a site runs --
which on the device is trace time:

``VLLM_NEURON_TP_ALLREDUCE_DTYPE`` = ``fp32`` (default) | ``bf16``
    ``fp32`` is the as-built path: the coordinator reduces the partial itself, in place,
    in the dtype it was computed in. ``bf16`` rounds this rank's partial to bfloat16 and
    reduces that copy, which halves the bytes every rank moves. Every site casts the
    reduced value to its carrier dtype right after; in this model the carrier between
    layers is the bfloat16 residual stream, so the as-built path already rounds the
    reduced sum to bfloat16 once, and ``bf16`` adds the rounding of each rank's partial
    and of the reduction's intermediate sums.

``VLLM_NEURON_TP_ALLREDUCE_FUSE`` = ``0`` (default) | ``1``
    The model issues one collective per site. A device profile shows two per site at a
    1024-token chunk because neuronx-cc's ``SimpleAllReduceTiling`` pass splits an
    all-reduce larger than 8 MiB into up to four tiles. That is a compiler decision for
    the whole graph, not one a call can make, so this switch only names the compiler
    argument that turns the pass off (:func:`fuse_compiler_args`), and
    ``NeuronModelRunner.load_model`` appends it to its ``compiler_args``.

At world size 1 there is no coordinator and nothing happens in either setting: no
collective and no cast, so a single-rank run is bitwise unchanged.
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING

import torch

from vllm_neuron import envs

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator

__all__ = [
    "DTYPE_ENV",
    "FUSE_COMPILER_ARG",
    "FUSE_ENV",
    "CollectivePolicyError",
    "RowParallelSite",
    "allreduce_fuse_enabled",
    "allreduce_wire_dtype",
    "fuse_compiler_args",
    "reduce_row_parallel",
]

#: The wire-dtype switch (``vllm_neuron.envs``).
DTYPE_ENV = "VLLM_NEURON_TP_ALLREDUCE_DTYPE"

#: The fusion switch (``vllm_neuron.envs``).
FUSE_ENV = "VLLM_NEURON_TP_ALLREDUCE_FUSE"

#: The neuronx-cc argument that turns ``SimpleAllReduceTiling`` off; the Tensorizer's
#: other options are kept. On neuronx-cc 2.27 a 16 MiB fp32 all-reduce lowers to one
#: collective with it and to two without it.
FUSE_COMPILER_ARG = "--tensorizer-options=--disable-tiling-allreduce"

#: The values ``VLLM_NEURON_TP_ALLREDUCE_DTYPE`` accepts, and the dtype each names.
_WIRE_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}

#: The partial dtypes a site may hand in: fp32 at every site on the device; bf16 where a
#: CPU route returns its sum already rounded to bfloat16 (the routed experts).
_PARTIAL_DTYPES = (torch.float32, torch.bfloat16)


class CollectivePolicyError(ValueError):
    """A switch value or a partial this policy does not define, refused by name."""


class RowParallelSite(str, enum.Enum):
    """Where a row-parallel partial comes from.

    Every site hands :func:`reduce_row_parallel` this rank's partial sum at the full
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


def allreduce_wire_dtype() -> torch.dtype:
    """The dtype a row-parallel partial crosses the wire in.

    Read from ``VLLM_NEURON_TP_ALLREDUCE_DTYPE`` through :mod:`vllm_neuron.envs`.

    Raises:
        CollectivePolicyError: the switch holds a value other than ``fp32`` or
            ``bf16``.
    """
    key = envs.VLLM_NEURON_TP_ALLREDUCE_DTYPE
    if key not in _WIRE_DTYPES:
        raise CollectivePolicyError(
            f"{DTYPE_ENV}={key!r} is not one of {sorted(_WIRE_DTYPES)}; unset it "
            f"for the as-built fp32 reduction"
        )
    return _WIRE_DTYPES[key]


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
            f"{FUSE_ENV} must be 0 or 1; unset it to keep the compiler's "
            f"all-reduce tiling"
        ) from error


def fuse_compiler_args() -> list[str]:
    """The neuronx-cc arguments the fusion switch asks for: empty unless it is on."""
    return [FUSE_COMPILER_ARG] if allreduce_fuse_enabled() else []


def reduce_row_parallel(
    partial: torch.Tensor, *, site: RowParallelSite, group: GroupCoordinator | None
) -> torch.Tensor:
    """Sum this rank's row-parallel ``partial`` across the tensor-parallel ``group``.

    Args:
        partial: this rank's partial sum, ``[tokens, hidden_size]`` (any shape reduces;
            the sum is elementwise), fp32 or bf16, fresh as :class:`RowParallelSite`
            states: the fp32 path reduces it in place.
        site: the :class:`RowParallelSite` the partial comes from.
        group: the tensor-parallel coordinator, or ``None`` at world size 1. Its
            ``all_reduce`` is called in statement form, the result read from the tensor
            handed in, as vLLM's device communicator reduces in place.

    Returns:
        ``partial`` itself when ``group`` is ``None``. Under ``fp32``, ``partial``
        itself, reduced in place in its own dtype: the operations the sites ran before
        this entry point. Under ``bf16``, a bfloat16 tensor holding the reduced sum
        (``partial`` reduced in place if it already is bfloat16, else a new tensor,
        with ``partial`` left holding this rank's own share). The caller casts the
        result to its carrier.

    Raises:
        CollectivePolicyError: ``site`` is not a :class:`RowParallelSite` member,
            ``partial`` is neither fp32 nor bf16, or a switch holds a value this module
            does not define.
    """
    # ``isinstance``, not ``RowParallelSite(site)``: the sites run inside a
    # ``fullgraph=True`` capture, and dynamo cannot trace the enum's lookup by value.
    if not isinstance(site, RowParallelSite):
        raise CollectivePolicyError(f"{site!r} is not a RowParallelSite member")
    if partial.dtype not in _PARTIAL_DTYPES:
        raise CollectivePolicyError(
            f"the {site.value} site handed a {partial.dtype} partial; a row-parallel "
            f"partial is one of {[str(d) for d in _PARTIAL_DTYPES]}"
        )
    if group is None:
        return partial
    wire_dtype = allreduce_wire_dtype()
    if wire_dtype is torch.float32 or partial.dtype is wire_dtype:
        group.all_reduce(partial)
        return partial
    wire = partial.to(wire_dtype)
    group.all_reduce(wire)
    return wire
